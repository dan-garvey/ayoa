"""Durable, bounded, matched long playtests; run again only to recover saved work."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTIVE = Path("/home/dan/ayoa-worktrees/covenant-single-llm")


def hashes(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != ".lock" and "__pycache__" not in p.parts
    }


if not (HERE / "provenance.json").exists():
    (HERE / "runtime/narrative").mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "core.py", "codex.py"):
        shutil.copyfile(ACTIVE / "narrative" / name, HERE / "runtime/narrative" / name)
    shutil.copytree(ACTIVE / "prompts", HERE / "sources/prompts", dirs_exist_ok=True)
    for story in ("covenant", "breakwater"):
        shutil.copytree(
            ACTIVE / "stories" / story, HERE / "sources" / story, dirs_exist_ok=True
        )
    initial = {
        "runtime_commit": "661b41c",
        "runtime": hashes(HERE / "runtime"),
        "sources": hashes(HERE / "sources"),
        "inputs_sha256": hashlib.sha256(
            (HERE / "inputs.json").read_bytes()
        ).hexdigest(),
        "live_session_files_before": hashes(ACTIVE / "sessions"),
    }
    (HERE / "provenance.json").write_text(json.dumps(initial, indent=2) + "\n")

sys.path.insert(0, str(HERE / "runtime"))
from narrative import core
from narrative.codex import CodexProxy

provenance = core.read_json(HERE / "provenance.json")
assert hashes(HERE / "runtime") == provenance["runtime"]
assert hashes(HERE / "sources") == provenance["sources"]
assert core.digest((HERE / "inputs.json").read_bytes()) == provenance["inputs_sha256"]
inputs = core.read_json(HERE / "inputs.json")
assert set(inputs) == {"covenant", "breakwater"}
assert all(len(values) == 24 for values in inputs.values())
if not (HERE / "conditions.json").exists():
    conditions = {}
    for story in inputs:
        enabled = secrets.choice(("A", "B"))
        for label in ("A", "B"):
            conditions[f"{story}_{label}"] = {
                "story": story,
                "checkup_every": 5 if label == enabled else 0,
            }
    core.write_json(HERE / "conditions.json", conditions)
conditions = core.read_json(HERE / "conditions.json")
guard = threading.Lock()
slots = threading.Semaphore(3)


def event(**value):
    with guard, (HERE / "events.jsonl").open("a") as handle:
        handle.write(json.dumps({"at": core.now(), **value}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def status(lane, **value):
    with guard:
        path = HERE / "progress.json"
        progress = core.read_json(path) if path.exists() else {}
        progress[lane] = {"at": core.now(), **value}
        core.write_json(path, progress)


class LoggedProxy:
    def __init__(self, session):
        self.session = session
        self.proxy = CodexProxy()

    def __call__(self, request):
        with slots:
            state = core.read_json(self.session / "state.json")
            pending = state["pending"]
            event(
                event="dispatch",
                lane=self.session.name,
                stage=pending["stage"],
                turn=len(state["turns"]) + 1,
                request_id=pending["request_id"],
            )
            result = self.proxy(request)
            event(
                event="received",
                lane=self.session.name,
                stage=pending["stage"],
                turn=len(state["turns"]) + 1,
                request_id=pending["request_id"],
                status=result["status"],
            )
            return result


def play(session, messages):
    lane = session.name
    proxy = LoggedProxy(session)
    try:
        while True:
            _manifest, state = core.load_session(session)
            count = len(state["turns"])
            if count >= len(messages):
                status(lane, completed=count, status="complete")
                return True
            status(lane, completed=count, status="running")
            if state["pending"]:
                pending = state["pending"]
                assert pending["input"] == messages[count]
                if pending["request_id"]:
                    path = core.attempt_dir(session, pending["request_id"])
                    if (path / "response.json").exists():
                        core.response_text(core.read_json(path / "response.json"))
                    elif (path / "started.json").exists():
                        raise RuntimeError(
                            "Unknown outcome requires explicit operator recovery"
                        )
                event(event="resume_saved", lane=lane, turn=count + 1)
                result = core.resume(session, proxy)
            else:
                result = core.submit(session, messages[count], proxy)
            assert result["status"] == "published"
            event(
                event="published",
                lane=lane,
                turn=count + 1,
                request_id=result["responses"][-1],
            )
            print(f"{lane}: {count + 1}/{len(messages)} published", flush=True)
    except Exception as exc:  # noqa: BLE001 - retain failure evidence and stop this lane
        state = core.read_json(session / "state.json")
        status(
            lane,
            completed=len(state["turns"]),
            status="stopped",
            error_type=type(exc).__name__,
        )
        event(event="stopped", lane=lane, error_type=type(exc).__name__)
        print(
            f"{lane}: stopped ({type(exc).__name__}); saved state retained", flush=True
        )
        return False


seeds = {}
for story in inputs:
    path = HERE / "sessions" / f"{story}_seed"
    if not path.exists():
        core.init_session(
            path,
            HERE / "sources" / story,
            prompts=HERE / "sources/prompts",
            checkup_every=0,
        )
    seeds[story] = path
with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(
        pool.map(lambda story: play(seeds[story], inputs[story][:4]), inputs)
    )
if not all(results):
    raise SystemExit(1)

for lane, condition in conditions.items():
    session = HERE / "sessions" / lane
    if not session.exists():
        shutil.copytree(
            seeds[condition["story"]], session, ignore=shutil.ignore_patterns(".lock")
        )
        manifest, state = core.load_session(session)
        manifest["checkup_every"] = condition["checkup_every"]
        core.write_json(session / "manifest.json", manifest)
        state["manifest_sha256"] = core.digest((session / "manifest.json").read_bytes())
        core.write_json(session / "state.json", state)
    else:
        manifest, _ = core.load_session(session)
        assert manifest["checkup_every"] == condition["checkup_every"]

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(
        pool.map(
            lambda lane: play(
                HERE / "sessions" / lane, inputs[conditions[lane]["story"]]
            ),
            conditions,
        )
    )
core.write_json(HERE / "live_session_files_after.json", hashes(ACTIVE / "sessions"))
if not all(results):
    raise SystemExit(1)
print("All four 24-turn lanes complete. Every first result retained.", flush=True)
