"""Shared context, persistence, and turn loop for API and file-based execution."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PREFIX_ORDER = ["author.txt", "direction.md", "canon.md"]
SNAPSHOT_FILES = {*PREFIX_ORDER, "regenerate.txt", "checkup.txt", "player.json"}


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


def normalize_player_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Choose a name for your protagonist")
    name = " ".join(value.split())
    if not name or len(name) > 80 or any(unicodedata.category(c) in {"Cc", "Cs"} for c in name):
        raise ValueError("Use a protagonist name of 1–80 characters without control characters")
    return name


def player_identity(session: Path, state: dict) -> dict:
    player = read_json(session / "snapshot" / "player.json")
    names = state.get("player_names", [player["name"]])
    return {
        "name": names[-1],
        "description": player["description"],
        "previous_names": list(dict.fromkeys(name for name in names if name != names[-1])),
    }


def state_version(state: dict) -> str:
    """An optimistic concurrency token derived from the single publication record."""
    return digest(json.dumps(state, sort_keys=True, ensure_ascii=False).encode())


def check_version(state: dict, expected: str | None) -> None:
    if expected is not None and expected != state_version(state):
        raise NarrativeError("The story changed; refresh before trying again")


@contextmanager
def locked(session: Path):
    # A failed lookup must not initialize a session as a side effect.
    if not (session / "state.json").is_file():
        raise NarrativeError("Session is missing or initialization did not finish")
    with (session / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def is_busy(session: Path) -> bool:
    """Read the existing lock without waiting or creating files."""
    try:
        handle = (session / ".lock").open("r")
    except FileNotFoundError:
        return False
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def init_session(
    session: Path,
    story: Path,
    *,
    prompts: Path = ROOT / "prompts",
    player_file: Path | None = None,
    player_name: str | None = None,
    transport: str = "proxy",
    model: str = "gpt-5.6-terra",
    reasoning: str = "max",
    max_output_tokens: int = 12000,
    checkup_every: int = 5,
) -> dict:
    if transport not in {"api", "proxy"} or not model.strip() or not reasoning.strip():
        raise ValueError("Specify a transport, model, and reasoning effort")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if type(checkup_every) is not int or checkup_every < 0:
        raise ValueError("checkup_every must be a nonnegative integer (0 disables checkups)")
    sources = {
        "author.txt": prompts / "author.txt",
        "regenerate.txt": prompts / "regenerate.txt",
        "checkup.txt": prompts / "checkup.txt",
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
    chosen_name = normalize_player_name(player["name"] if player_name is None else player_name)
    session.mkdir(parents=True, exist_ok=False)
    (session / "snapshot").mkdir()
    (session / "attempts").mkdir()
    for name, data in contents.items():
        atomic_write(session / "snapshot" / name, data)
    manifest = {
        "version": 3,
        "created_at": now(),
        "story": story.name,
        "transport": transport,
        "model": model,
        "reasoning_effort": reasoning,
        "max_output_tokens": max_output_tokens,
        "checkup_every": checkup_every,
        "prefix_order": PREFIX_ORDER.copy(),
        "source_sha256": {name: digest(data) for name, data in contents.items()},
    }
    write_json(session / "manifest.json", manifest)
    write_json(
        session / "state.json",
        {
            "manifest_sha256": digest((session / "manifest.json").read_bytes()),
            "player_names": [chosen_name],
            "turns": [],
            "checkups": [],
            "pending": None,
        },
    )
    return manifest


def load_session(session: Path) -> tuple[dict, dict]:
    manifest = read_json(session / "manifest.json")
    state = read_json(session / "state.json")
    if digest((session / "manifest.json").read_bytes()) != state["manifest_sha256"]:
        raise NarrativeError("Frozen session settings changed; initialize a new session")
    if manifest.get("version") != 3 or manifest.get("prefix_order") != PREFIX_ORDER:
        raise NarrativeError("Unsupported session format; initialize a new session")
    if type(manifest["checkup_every"]) is not int or manifest["checkup_every"] < 0:
        raise NarrativeError("Invalid checkup interval")
    if set(manifest["source_sha256"]) != SNAPSHOT_FILES:
        raise NarrativeError("Incomplete source snapshot")
    for name, expected in manifest["source_sha256"].items():
        if digest((session / "snapshot" / name).read_bytes()) != expected:
            raise NarrativeError(f"Frozen source changed: {name}")
    for index, turn in enumerate(state["turns"]):
        if turn["turn"] != index or not turn["input"].strip() or not turn["output"].strip():
            raise NarrativeError("Invalid published history")
        if not isinstance(turn["responses"], list) or not turn["responses"]:
            raise NarrativeError("Missing response history")
        for request_id in turn["responses"]:
            attempt_dir(session, request_id)
    pending = state["pending"]
    if pending is not None:
        if pending["kind"] not in {"turn", "regenerate"} or not pending["input"].strip():
            raise NarrativeError("Invalid pending submission")
        if pending["kind"] == "regenerate" and not state["turns"]:
            raise NarrativeError("No passage to regenerate")
        if pending["stage"] not in {"author", "checkup"} or (
            pending["stage"] == "checkup" and pending["kind"] != "turn"
        ):
            raise NarrativeError("Invalid pending stage")
    previous_turn = -1
    for checkup in state["checkups"]:
        before_turn = checkup["before_turn"]
        if type(before_turn) is not int or not previous_turn < before_turn <= len(state["turns"]):
            raise NarrativeError("Invalid checkup history")
        if before_turn == len(state["turns"]) and (
            not pending or pending["kind"] != "turn" or pending["stage"] != "author"
        ):
            raise NarrativeError("Checkup is ahead of the active story")
        attempt_dir(session, checkup["request_id"])
        previous_turn = before_turn
    if "player_names" in state:
        names = state["player_names"]
        if not isinstance(names, list) or not names:
            raise NarrativeError("Invalid protagonist name history")
        if any(normalize_player_name(name) != name for name in names):
            raise NarrativeError("Invalid protagonist name history")
    return manifest, state


def response_text(response: dict) -> str:
    if response["transport"] == "proxy":
        if response.get("status", "completed") != "completed":
            raise ResponseError(
                f"Coding-agent response {response['status']} (exit {response.get('exit_code')}); "
                "raw output preserved. Check codex login status and resume to retry."
            )
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


def response_summary(response: dict) -> str:
    """Project only exposed summary text for inspection, never narrative context."""
    if response["transport"] == "proxy":
        parts = response.get("reasoning_summaries", [])
    else:
        parts = [
            block["text"]
            for item in response["raw"].get("output", [])
            if item.get("type") == "reasoning"
            for block in item.get("summary", [])
            if block.get("type") == "summary_text"
        ]
    return "\n\n".join(part for part in parts if isinstance(part, str) and part.strip())


def attempt_dir(session: Path, request_id: str) -> Path:
    if len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id):
        raise NarrativeError("Invalid request id")
    return session / "attempts" / request_id


def make_request(session: Path, manifest: dict, state: dict) -> dict:
    pending = state["pending"]
    if pending is None:
        raise NarrativeError("No pending submission")
    snapshot = session / "snapshot"
    player = player_identity(session, state)
    identity = f"My character: {player['name']}. {player['description']}"
    if player["previous_names"]:
        identity += (
            "\nEarlier names for this protagonist: "
            + json.dumps(player["previous_names"], ensure_ascii=False)
            + ". This is a name correction, not an event in the story; "
            "their background and relationships stay the same."
        )
    messages = [{"role": "user", "content": identity}]
    for turn in state["turns"]:
        messages.extend(
            [
                {"role": "user", "content": turn["input"]},
                {"role": "assistant", "content": turn["output"]},
            ]
        )
    if state["checkups"]:
        path = attempt_dir(session, state["checkups"][-1]["request_id"])
        guidance = response_text(read_json(path / "response.json"))
        messages.append(
            {"role": "user", "content": f"<backstage_guidance>\n{guidance}\n</backstage_guidance>"}
        )
    instruction = pending["input"]
    if pending["kind"] == "regenerate":
        instruction = (
            (snapshot / "regenerate.txt").read_text(encoding="utf-8").strip() + "\n\n" + instruction
        )
    messages.append({"role": "user", "content": instruction})
    if pending["stage"] == "checkup":
        messages.append(
            {
                "role": "user",
                "content": (snapshot / "checkup.txt").read_text(encoding="utf-8").strip(),
            }
        )
    return {
        "model": manifest["model"],
        "reasoning": {"effort": manifest["reasoning_effort"], "summary": "detailed"},
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
            "turn": len(state["turns"]) - (pending["kind"] == "regenerate"),
            "kind": pending["kind"],
            "stage": pending["stage"],
            "submission_id": pending["submission_id"],
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
    if (path / "request.txt").read_bytes().decode("utf-8") != render_request(request):
        raise NarrativeError("Pending text request was changed")
    return request


def _call_model(path: Path, request: dict, client: Any, transport: str) -> None:
    if client is None:
        raise NarrativeError("Automatic execution requires a client")
    write_json(path / "started.json", {"started_at": now()})
    started = time.monotonic()
    try:
        response = (
            {"raw": client.responses.create(**request).model_dump(mode="json")}
            if transport == "api"
            else client(request)
        )
        write_json(
            path / "response.json",
            {
                **response,
                "transport": transport,
                "received_at": now(),
                "duration_s": time.monotonic() - started,
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
        if isinstance(exc, NarrativeError):
            raise
        raise NarrativeError(
            f"Automatic response failed ({type(exc).__name__}); use resume to retry explicitly"
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
            if pending["stage"] == "checkup":
                state["checkups"].append(
                    {"before_turn": len(state["turns"]), "request_id": request_id}
                )
                pending.update(stage="author", request_id=None)
                write_json(session / "state.json", state)
                retry = False
                continue
            previous = state["turns"][-1] if pending["kind"] == "regenerate" else None
            turn = {
                "turn": previous["turn"] if previous else len(state["turns"]),
                "input": previous["input"] if previous else pending["input"],
                "output": text,
                "responses": [*(previous["responses"] if previous else []), request_id],
                "submission_id": pending["submission_id"],
                "published_at": now(),
            }
            if previous:
                state["turns"][-1] = turn
            else:
                state["turns"].append(turn)
            state["pending"] = None
            write_json(session / "state.json", state)
            _export(session, manifest, state)
            return {"status": "published", **turn}
        if manifest["transport"] == "proxy" and client is None:
            return {
                "status": "pending",
                "kind": pending["kind"],
                "stage": pending["stage"],
                "request_id": pending["request_id"],
                "request_json": str((path / "request.json").resolve()),
                "request_text": str((path / "request.txt").resolve()),
            }
        if (path / "started.json").exists():
            if not retry:
                raise NarrativeError("Interrupted model attempt; use resume to retry explicitly")
            pending["request_id"] = None
            write_json(session / "state.json", state)
            retry = False
            continue
        retry = False
        _call_model(path, request, client, manifest["transport"])
    _export(session, manifest, state)
    return {"status": "idle", "published_turns": len(state["turns"])}


def submit(
    session: Path,
    text: str,
    client: Any = None,
    *,
    expected_version: str | None = None,
    submission_id: str | None = None,
) -> dict:
    return _submit(session, text, client, "turn", expected_version, submission_id)


def regenerate(
    session: Path,
    text: str,
    client: Any = None,
    *,
    expected_version: str | None = None,
    submission_id: str | None = None,
) -> dict:
    return _submit(session, text, client, "regenerate", expected_version, submission_id)


def _submit(
    session: Path,
    text: str,
    client: Any,
    kind: str,
    expected_version: str | None,
    submission_id: str | None,
) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A player submission cannot be empty")
    try:
        identifier = uuid.uuid4().hex if submission_id is None else uuid.UUID(submission_id).hex
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Invalid submission id") from exc
    with locked(session):
        manifest, state = load_session(session)
        check_version(state, expected_version)
        if state["pending"] is not None:
            raise NarrativeError("A turn is pending; accept its response or resume it first")
        if kind == "regenerate" and not state["turns"]:
            raise NarrativeError("There is no passage to regenerate yet")
        state["pending"] = {
            "kind": kind,
            "stage": "checkup"
            if kind == "turn"
            and manifest["checkup_every"]
            and (len(state["turns"]) + 1) % manifest["checkup_every"] == 0
            else "author",
            "input": text,
            "submission_id": identifier,
            "request_id": None,
        }
        write_json(session / "state.json", state)
        return _drive(session, manifest, state, client, retry=False)


def rename_player(session: Path, name: str, *, expected_version: str | None = None) -> dict:
    chosen_name = normalize_player_name(name)
    with locked(session):
        _, state = load_session(session)
        check_version(state, expected_version)
        if state["pending"] is not None:
            raise NarrativeError("Finish the pending response before renaming your protagonist")
        player = player_identity(session, state)
        if chosen_name != player["name"]:
            state["player_names"] = [
                *state.get("player_names", [player["name"]]),
                chosen_name,
            ]
            write_json(session / "state.json", state)
        return player_identity(session, state)


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
    prepared = received = automatic_started = usage_reported = 0
    seconds = 0.0
    for path in (session / "attempts").iterdir():
        if not path.is_dir() or not (path / "meta.json").exists():
            continue
        prepared += 1
        automatic_started += (path / "started.json").exists()
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
    api_started = automatic_started if manifest["transport"] == "api" else 0
    proxy_started = automatic_started if manifest["transport"] == "proxy" else 0
    summary = {
        "published_turns": len(state["turns"]),
        "completed_checkups": len(state["checkups"]),
        "pending_kind": state["pending"]["kind"] if state["pending"] else None,
        "requests_prepared": prepared,
        "api_calls_started": api_started,
        "proxy_calls_started": proxy_started,
        "responses_received": received,
        "responses_with_usage": usage_reported,
        "reported_usage": totals if usage_reported else None,
        "recorded_api_seconds": round(seconds, 3) if api_started else None,
        "recorded_proxy_seconds": round(seconds, 3) if proxy_started else None,
        "visible_words": sum(len(turn["output"].split()) for turn in state["turns"]),
    }
    atomic_write(session / "transcript.md", "\n".join(lines).encode())
    write_json(session / "summary.json", summary)
    return summary


def export(session: Path) -> dict:
    with locked(session):
        manifest, state = load_session(session)
        return _export(session, manifest, state)
