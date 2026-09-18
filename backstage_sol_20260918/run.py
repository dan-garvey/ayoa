"""Durable matched playtests using one persistent server and continued threads."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rpc import AppServer

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "backstage_long_20260918"
ACTIVE = Path("/home/dan/ayoa-worktrees/covenant-single-llm")


def hashes(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != ".lock" and "__pycache__" not in p.parts
    }


if not (HERE / "provenance.json").exists():
    shutil.copytree(
        PREVIOUS / "runtime",
        HERE / "runtime",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(PREVIOUS / "sources", HERE / "sources", dirs_exist_ok=True)
    shutil.copyfile(PREVIOUS / "inputs.json", HERE / "inputs.json")
    shutil.copyfile(PREVIOUS / "read_prose.py", HERE / "read_prose.py")
    (HERE / "provenance.json").write_text(
        json.dumps(
            {
                "reference_archive_commit": "84de4738e8e71c04e11b7a6a60caf661befac590",
                "runtime": hashes(HERE / "runtime"),
                "sources": hashes(HERE / "sources"),
                "inputs_sha256": hashlib.sha256(
                    (HERE / "inputs.json").read_bytes()
                ).hexdigest(),
                "live_session_files_before": hashes(ACTIVE / "sessions"),
                "model": "gpt-5.6-sol",
                "effort": "max",
                "summary": "detailed",
            },
            indent=2,
        )
        + "\n"
    )

sys.path.insert(0, str(HERE / "runtime"))
from narrative import core

provenance = core.read_json(HERE / "provenance.json")
assert hashes(HERE / "runtime") == provenance["runtime"]
assert hashes(HERE / "sources") == provenance["sources"]
assert core.digest((HERE / "inputs.json").read_bytes()) == provenance["inputs_sha256"]
inputs = core.read_json(HERE / "inputs.json")
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
        values = core.read_json(path) if path.exists() else {}
        values[lane] = {"at": core.now(), **value}
        core.write_json(path, values)


class ContinuedProxy:
    def __init__(self, session, server):
        self.session, self.server = session, server

    def __call__(self, request):
        with slots:
            _, state = core.load_session(self.session)
            pending = state["pending"]
            binding = core.read_json(self.session / "agent.json")
            attempt = core.attempt_dir(self.session, pending["request_id"])
            if not state["turns"] and not state["checkups"]:
                sent = core.render_request(request)
            elif pending["stage"] == "checkup":
                sent = (
                    f"<user>\n{pending['input']}\n</user>\n\n<user>\n"
                    + (self.session / "snapshot/checkup.txt").read_text().strip()
                    + "\n</user>\n"
                )
            else:
                sent = (
                    "<user>\nReturn the next passage of fiction.\n\n"
                    + pending["input"]
                    + "\n</user>\n"
                )
            params = {
                "threadId": binding["thread_id"],
                "model": request["model"],
                "effort": "max",
                "summary": "detailed",
                "input": [{"type": "text", "text": sent}],
            }
            core.write_json(attempt / "wire_request.json", params)
            core.atomic_write(attempt / "wire_request.txt", sent.encode())
            event(
                event="dispatch",
                lane=self.session.name,
                stage=pending["stage"],
                turn=len(state["turns"]) + 1,
                request_id=pending["request_id"],
                thread_id=binding["thread_id"],
                server_pid=self.server.process.pid,
            )

            def save(public_event):
                with (attempt / "public_events.jsonl").open("a") as handle:
                    handle.write(json.dumps(public_event, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())

            result = self.server.turn(params, save)
            core.write_json(attempt / "agent_result.json", result)
            event(
                event="received",
                lane=self.session.name,
                stage=pending["stage"],
                turn=len(state["turns"]) + 1,
                request_id=pending["request_id"],
                thread_id=binding["thread_id"],
                turn_id=result["turn_id"],
                status=result["status"],
            )
            return result


def play(session, messages, server):
    proxy = ContinuedProxy(session, server)
    try:
        while True:
            _, state = core.load_session(session)
            count = len(state["turns"])
            if count >= len(messages):
                status(session.name, completed=count, status="complete")
                return True
            status(session.name, completed=count, status="running")
            if state["pending"]:
                pending = state["pending"]
                assert pending["input"] == messages[count]
                if pending["request_id"]:
                    attempt = core.attempt_dir(session, pending["request_id"])
                    if (attempt / "response.json").exists():
                        core.response_text(core.read_json(attempt / "response.json"))
                    elif (attempt / "started.json").exists():
                        raise RuntimeError(
                            "Unknown outcome: inspect retained thread before recovery"
                        )
                result = core.resume(session, proxy)
            else:
                result = core.submit(session, messages[count], proxy)
            assert result["status"] == "published"
            event(
                event="published",
                lane=session.name,
                turn=count + 1,
                request_id=result["responses"][-1],
            )
            print(f"{session.name}: {count + 1}/{len(messages)} published", flush=True)
    except Exception as exc:  # noqa: BLE001 - stop the lane and preserve every attempt
        state = core.read_json(session / "state.json")
        status(
            session.name,
            completed=len(state["turns"]),
            status="stopped",
            error_type=type(exc).__name__,
        )
        event(event="stopped", lane=session.name, error_type=type(exc).__name__)
        print(
            f"{session.name}: stopped ({type(exc).__name__}); evidence retained",
            flush=True,
        )
        return False


def bind(session, server, workspace, parent=None):
    path = session / "agent.json"
    if path.exists():
        binding = core.read_json(path)
        response = server.rpc(
            "thread/resume",
            {
                "threadId": binding["thread_id"],
                "model": "gpt-5.6-sol",
                "cwd": workspace,
                "approvalPolicy": "never",
                "sandbox": "read-only",
            },
        )
        assert response["thread"]["id"] == binding["thread_id"]
        event(event="thread_resumed", lane=session.name, thread_id=binding["thread_id"])
        return
    params = {
        "model": "gpt-5.6-sol",
        "cwd": workspace,
        "approvalPolicy": "never",
        "sandbox": "read-only",
    }
    method = "thread/start"
    if parent:
        method = "thread/fork"
        parent_binding = core.read_json(parent / "agent.json")
        parent_state = core.read_json(parent / "state.json")
        last = core.read_json(
            core.attempt_dir(parent, parent_state["turns"][-1]["responses"][-1])
            / "response.json"
        )
        params.update(threadId=parent_binding["thread_id"], lastTurnId=last["turn_id"])
    core.write_json(
        session / "thread_request.json", {"method": method, "params": params}
    )
    response = server.rpc(method, params)
    assert response["model"] == "gpt-5.6-sol"
    assert not response.get("instructionSources", [])
    binding = {
        "thread_id": response["thread"]["id"],
        "model": response["model"],
        "parent_thread_id": params.get("threadId"),
        "through_turn_id": params.get("lastTurnId"),
        "instruction_sources": response.get("instructionSources", []),
    }
    core.write_json(path, binding)
    core.write_json(session / "thread_response.json", response)
    event(event="thread_bound", lane=session.name, **binding)


with tempfile.TemporaryDirectory(prefix="ayoa-sol-eval-") as workspace:
    server = AppServer(workspace)
    event(event="server_started", pid=server.process.pid, command=server.command)
    try:
        seeds = {}
        for story in inputs:
            path = HERE / "sessions" / f"{story}_seed"
            if not path.exists():
                core.init_session(
                    path,
                    HERE / "sources" / story,
                    prompts=HERE / "sources/prompts",
                    model="gpt-5.6-sol",
                    checkup_every=0,
                )
            bind(path, server, workspace)
            seeds[story] = path
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda story: play(seeds[story], inputs[story][:4], server), inputs
                )
            )
        if not all(results):
            raise SystemExit(1)
        for lane, condition in conditions.items():
            session = HERE / "sessions" / lane
            if not session.exists():
                shutil.copytree(
                    seeds[condition["story"]],
                    session,
                    ignore=shutil.ignore_patterns(
                        ".lock", "agent.json", "thread_*.json"
                    ),
                )
                manifest, state = core.load_session(session)
                manifest["checkup_every"] = condition["checkup_every"]
                core.write_json(session / "manifest.json", manifest)
                state["manifest_sha256"] = core.digest(
                    (session / "manifest.json").read_bytes()
                )
                core.write_json(session / "state.json", state)
            bind(session, server, workspace, parent=seeds[condition["story"]])
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(
                    lambda lane: play(
                        HERE / "sessions" / lane,
                        inputs[conditions[lane]["story"]],
                        server,
                    ),
                    conditions,
                )
            )
        core.write_json(
            HERE / "live_session_files_after.json", hashes(ACTIVE / "sessions")
        )
        if not all(results):
            raise SystemExit(1)
        print(
            "All four 24-turn lanes complete. All first outputs retained.", flush=True
        )
    finally:
        server.close()
        event(
            event="server_stopped",
            pid=server.process.pid,
            exit_code=server.process.returncode,
        )
