"""Shared context, persistence, and turn loop for API and file-based execution."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PREFIX_ORDER = ["author.txt", "direction.md", "canon.md"]
SNAPSHOT_FILES = {*PREFIX_ORDER, "revision.txt", "player.json"}


class NarrativeError(RuntimeError):
    """A session cannot safely advance."""


class ResponseError(NarrativeError):
    """An attempted response cannot be published."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, data: bytes) -> None:
    """Replace a complete file, then make its directory entry durable."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


@contextmanager
def locked(session: Path):
    # A failed lookup must not initialize a session as a side effect.
    if not (session / "state.json").is_file():
        raise NarrativeError("Session is missing or initialization did not finish")
    with (session / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def init_session(
    session: Path,
    story: Path,
    *,
    prompts: Path = ROOT / "prompts",
    player_file: Path | None = None,
    transport: str = "proxy",
    model: str = "gpt-5.6-terra",
    reasoning: str = "max",
    max_output_tokens: int = 12000,
) -> dict:
    if transport not in {"api", "proxy"} or not model.strip() or not reasoning.strip():
        raise ValueError("Specify a transport, model, and reasoning effort")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    sources = {
        "author.txt": prompts / "author.txt",
        "revision.txt": prompts / "revision.txt",
        "direction.md": story / "direction.md",
        "canon.md": story / "canon.md",
        "player.json": player_file or story / "player.json",
    }
    contents = {name: path.read_bytes() for name, path in sources.items()}
    if any(not data.decode("utf-8").strip() for data in contents.values()):
        raise ValueError("Story and prompt sources must contain text")
    player = json.loads(contents["player.json"])
    if (
        not isinstance(player, dict)
        or set(player) != {"name", "description"}
        or any(not isinstance(value, str) or not value.strip() for value in player.values())
    ):
        raise ValueError("player.json needs nonempty name and description strings")
    session.mkdir(parents=True, exist_ok=False)
    (session / "snapshot").mkdir()
    (session / "attempts").mkdir()
    for name, data in contents.items():
        atomic_write(session / "snapshot" / name, data)
    manifest = {
        "version": 1,
        "created_at": now(),
        "story": story.name,
        "transport": transport,
        "model": model,
        "reasoning_effort": reasoning,
        "max_output_tokens": max_output_tokens,
        "prefix_order": PREFIX_ORDER.copy(),
        "source_sha256": {name: digest(data) for name, data in contents.items()},
    }
    write_json(session / "manifest.json", manifest)
    write_json(
        session / "state.json",
        {
            "manifest_sha256": digest((session / "manifest.json").read_bytes()),
            "turns": [],
            "pending": None,
        },
    )
    return manifest


def load_session(session: Path) -> tuple[dict, dict]:
    manifest = read_json(session / "manifest.json")
    state = read_json(session / "state.json")
    if digest((session / "manifest.json").read_bytes()) != state["manifest_sha256"]:
        raise NarrativeError("Frozen session settings changed; initialize a new session")
    if manifest.get("version") != 1 or manifest.get("prefix_order") != PREFIX_ORDER:
        raise NarrativeError("Unsupported session format; initialize a new session")
    if set(manifest["source_sha256"]) != SNAPSHOT_FILES:
        raise NarrativeError("Incomplete source snapshot")
    for name, expected in manifest["source_sha256"].items():
        if digest((session / "snapshot" / name).read_bytes()) != expected:
            raise NarrativeError(f"Frozen source changed: {name}")
    for index, turn in enumerate(state["turns"]):
        if turn["turn"] != index or not turn["input"].strip() or not turn["output"].strip():
            raise NarrativeError("Invalid published history")
    return manifest, state


def response_text(response: dict) -> str:
    if response["transport"] == "proxy":
        text = response["raw"]
    else:
        raw = response["raw"]
        if raw.get("status") != "completed":
            raise ResponseError(f"Response was {raw.get('status')}; raw output preserved")
        parts = []
        for item in raw.get("output", []):
            if item.get("type") != "message":
                continue
            for block in item.get("content", []):
                if block.get("type") == "refusal":
                    raise ResponseError("Model refused; raw output preserved")
                if block.get("type") == "output_text":
                    parts.append(block["text"])
        text = "".join(parts)
    if not isinstance(text, str) or not text.strip():
        raise ResponseError("Response contained no prose; raw output preserved")
    return text


def attempt_dir(session: Path, request_id: str) -> Path:
    if len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id):
        raise NarrativeError("Invalid request id")
    return session / "attempts" / request_id


def make_request(session: Path, manifest: dict, state: dict) -> dict:
    pending = state["pending"]
    if pending is None:
        raise NarrativeError("No pending submission")
    snapshot = session / "snapshot"
    player = read_json(snapshot / "player.json")
    messages = [
        {"role": "user", "content": (f"My character: {player['name']}. {player['description']}")}
    ]
    for turn in state["turns"]:
        messages.extend(
            [
                {"role": "user", "content": turn["input"]},
                {"role": "assistant", "content": turn["output"]},
            ]
        )
    messages.append({"role": "user", "content": pending["input"]})
    if pending["author"]:
        draft = response_text(read_json(attempt_dir(session, pending["author"]) / "response.json"))
        messages.extend(
            [
                {"role": "assistant", "content": draft},
                {
                    "role": "user",
                    "content": (snapshot / "revision.txt").read_text(encoding="utf-8").strip(),
                },
            ]
        )
    return {
        "model": manifest["model"],
        "reasoning": {"effort": manifest["reasoning_effort"]},
        "instructions": "\n\n".join(
            (snapshot / name).read_text(encoding="utf-8").strip()
            for name in manifest["prefix_order"]
        ),
        "input": messages,
        "max_output_tokens": manifest["max_output_tokens"],
        "store": False,
        "truncation": "disabled",
    }


def render_request(request: dict) -> str:
    """A readable projection of exactly the text messages sent by either transport."""
    sections = ["<system>\n" + request["instructions"] + "\n</system>"]
    for message in request["input"]:
        role = message["role"]
        sections.append(f"<{role}>\n{message['content']}\n</{role}>")
    return "\n\n".join(sections) + "\n"


def _prepare(session: Path, manifest: dict, state: dict) -> Path:
    pending = state["pending"]
    request_id = uuid.uuid4().hex
    path = attempt_dir(session, request_id)
    path.mkdir()
    request = make_request(session, manifest, state)
    write_json(path / "request.json", request)
    atomic_write(path / "request.txt", render_request(request).encode())
    write_json(
        path / "meta.json",
        {
            "request_id": request_id,
            "turn": len(state["turns"]),
            "stage": "editor" if pending["author"] else "author",
            "transport": manifest["transport"],
            "prepared_at": now(),
        },
    )
    pending["request_id"] = request_id
    write_json(session / "state.json", state)
    return path


def _check_request(session: Path, manifest: dict, state: dict, path: Path) -> dict:
    request = read_json(path / "request.json")
    if request != make_request(session, manifest, state):
        raise NarrativeError("Pending request no longer matches its frozen context")
    if (path / "request.txt").read_text(encoding="utf-8") != render_request(request):
        raise NarrativeError("Pending text request was changed")
    return request


def _call_api(path: Path, request: dict, client: Any) -> None:
    if client is None:
        raise NarrativeError("API execution requires a client")
    write_json(path / "started.json", {"started_at": now()})
    started = time.monotonic()
    try:
        response = client.responses.create(**request)
        write_json(
            path / "response.json",
            {
                "transport": "api",
                "received_at": now(),
                "duration_s": time.monotonic() - started,
                "raw": response.model_dump(mode="json"),
            },
        )
    except Exception as exc:
        # Do not persist exception strings that might echo credentials or headers.
        write_json(
            path / "error.json",
            {
                "type": type(exc).__name__,
                "failed_at": now(),
                "duration_s": time.monotonic() - started,
            },
        )
        raise NarrativeError(
            f"API attempt failed ({type(exc).__name__}); use resume to retry explicitly"
        ) from exc


def _drive(session: Path, manifest: dict, state: dict, client: Any, *, retry: bool) -> dict:
    while state["pending"] is not None:
        pending = state["pending"]
        path = (
            attempt_dir(session, pending["request_id"])
            if pending["request_id"]
            else _prepare(session, manifest, state)
        )
        request = _check_request(session, manifest, state, path)
        if (path / "response.json").exists():
            try:
                text = response_text(read_json(path / "response.json"))
            except ResponseError:
                if not retry:
                    raise
                pending["request_id"] = None
                write_json(session / "state.json", state)
                retry = False
                continue
            request_id = pending["request_id"]
            if pending["author"] is None:
                pending["author"] = request_id
                pending["request_id"] = None
                write_json(session / "state.json", state)
                continue
            turn = {
                "turn": len(state["turns"]),
                "input": pending["input"],
                "output": text,
                "author": pending["author"],
                "editor": request_id,
                "published_at": now(),
            }
            state["turns"].append(turn)
            state["pending"] = None
            write_json(session / "state.json", state)
            _export(session, manifest, state)
            return {"status": "published", **turn}
        if manifest["transport"] == "proxy":
            return {
                "status": "pending",
                "stage": "editor" if pending["author"] else "author",
                "request_id": pending["request_id"],
                "request_json": str((path / "request.json").resolve()),
                "request_text": str((path / "request.txt").resolve()),
            }
        if (path / "started.json").exists():
            if not retry:
                raise NarrativeError("Interrupted API attempt; use resume to retry explicitly")
            pending["request_id"] = None
            write_json(session / "state.json", state)
            retry = False
            continue
        retry = False
        _call_api(path, request, client)
    _export(session, manifest, state)
    return {"status": "idle", "published_turns": len(state["turns"])}


def submit(
    session: Path, text: str, client: Any = None, *, expected_turns: int | None = None
) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A player submission cannot be empty")
    with locked(session):
        manifest, state = load_session(session)
        if expected_turns is not None and expected_turns != len(state["turns"]):
            raise NarrativeError("The story changed; refresh before submitting this turn")
        if state["pending"] is not None:
            raise NarrativeError("A turn is pending; accept its response or resume it first")
        state["pending"] = {"input": text, "author": None, "request_id": None}
        write_json(session / "state.json", state)
        return _drive(session, manifest, state, client, retry=False)


def accept(session: Path, request_id: str, text: str) -> dict:
    with locked(session):
        manifest, state = load_session(session)
        pending = state["pending"]
        if manifest["transport"] != "proxy":
            raise NarrativeError("accept is only available for proxy sessions")
        if pending is None or pending["request_id"] != request_id:
            raise NarrativeError("Stale or duplicate response; request id is not pending")
        path = attempt_dir(session, request_id)
        _check_request(session, manifest, state, path)
        if (path / "response.json").exists():
            raise NarrativeError("Response already recorded; use resume")
        write_json(
            path / "response.json",
            {
                "transport": "proxy",
                "received_at": now(),
                "duration_s": None,
                "raw": text,
            },
        )
        return _drive(session, manifest, state, None, retry=False)


def resume(session: Path, client: Any = None) -> dict:
    with locked(session):
        manifest, state = load_session(session)
        return _drive(session, manifest, state, client, retry=True)


def _export(session: Path, manifest: dict, state: dict) -> dict:
    lines = [f"# {manifest['story']}: published story", ""]
    for turn in state["turns"]:
        lines.extend(
            [
                f"## Turn {turn['turn'] + 1}",
                "",
                "**Player**",
                "",
                turn["input"],
                "",
                "**Story**",
                "",
                turn["output"],
                "",
            ]
        )
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    prepared = received = api_started = usage_reported = 0
    seconds = 0.0
    for path in (session / "attempts").iterdir():
        if not path.is_dir() or not (path / "meta.json").exists():
            continue
        prepared += 1
        api_started += (path / "started.json").exists()
        response_path = path / "response.json"
        response = read_json(response_path) if response_path.exists() else None
        if response is not None:
            received += 1
            seconds += response.get("duration_s") or 0
            usage = response["raw"].get("usage") if response["transport"] == "api" else None
            if usage is not None:
                usage_reported += 1
                totals["input_tokens"] += usage.get("input_tokens", 0)
                totals["output_tokens"] += usage.get("output_tokens", 0)
                totals["cached_tokens"] += (usage.get("input_tokens_details") or {}).get(
                    "cached_tokens", 0
                )
                totals["reasoning_tokens"] += (usage.get("output_tokens_details") or {}).get(
                    "reasoning_tokens", 0
                )
        elif (path / "error.json").exists():
            seconds += read_json(path / "error.json").get("duration_s") or 0
    summary = {
        "published_turns": len(state["turns"]),
        "pending_stage": ("editor" if state["pending"]["author"] else "author")
        if state["pending"]
        else None,
        "requests_prepared": prepared,
        "api_calls_started": api_started,
        "responses_received": received,
        "responses_with_usage": usage_reported,
        "reported_usage": totals if usage_reported else None,
        "recorded_api_seconds": round(seconds, 3) if api_started else None,
        "visible_words": sum(len(turn["output"].split()) for turn in state["turns"]),
    }
    atomic_write(session / "transcript.md", "\n".join(lines).encode())
    write_json(session / "summary.json", summary)
    return summary


def export(session: Path) -> dict:
    with locked(session):
        manifest, state = load_session(session)
        return _export(session, manifest, state)
