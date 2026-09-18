"""Frozen, first-result continuations from one native story checkpoint."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "backstage_sol_20260918"
ACTIVE = Path("/home/dan/ayoa-worktrees/covenant-single-llm")
MODEL = "gpt-5.6-sol"
LOCK = threading.Lock()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def hashes(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def now():
    return datetime.now(timezone.utc).isoformat()


def event(**data):
    data = {"at": now(), **data}
    with LOCK:
        with (HERE / "events.jsonl").open("a") as output:
            output.write(json.dumps(data) + "\n")
        print(json.dumps(data), flush=True)


def wire_text(player_input, note=""):
    guidance = (
        f"<backstage_guidance>\n{note.strip()}\n</backstage_guidance>\n\n"
        if note
        else ""
    )
    return f"<user>\n{guidance}Return the next passage of fiction.\n\n{player_input}\n</user>\n"


def verify_fork(response, checkpoint):
    assert response["model"] == MODEL
    assert response["reasoningEffort"] == "max"
    assert response.get("instructionSources", []) == []
    turns = response["thread"]["turns"]
    expected = checkpoint["native_turns"]
    assert len(turns) == len(expected)
    for native, saved in zip(turns, expected):
        assert native["id"] == saved["turn_id"]
        final = "\n\n".join(
            item["text"]
            for item in native["items"]
            if item["type"] == "agentMessage"
            and item.get("phase") in {None, "final_answer"}
        )
        assert final.strip() == saved["output"].strip()
        user = [item for item in native["items"] if item["type"] == "userMessage"]
        actual = [
            part["text"]
            for item in user
            for part in item["content"]
            if part["type"] == "text"
        ]
        assert actual == [saved["wire_text"]]


def prepare():
    if (HERE / "provenance.json").exists():
        raise RuntimeError("Already prepared; keep the frozen notes and mapping")
    shutil.copyfile(PREVIOUS / "rpc.py", HERE / "rpc.py")
    shutil.copytree(PREVIOUS / "sources/breakwater", HERE / "sources/breakwater")
    shutil.copytree(PREVIOUS / "sources/prompts", HERE / "sources/prompts")
    notes = HERE / "notes"
    (notes / "control.txt").write_text("")
    (notes / "combined.txt").write_text(
        (notes / "correction.txt").read_text().strip()
        + "\n\n"
        + (notes / "development.txt").read_text().strip()
        + "\n"
    )
    parent = PREVIOUS / "sessions/breakwater_A"
    state = read(parent / "state.json")
    native = []
    for index, turn in enumerate(state["turns"][:18]):
        ids = [c["request_id"] for c in state["checkups"] if c["before_turn"] == index]
        ids.append(turn["responses"][-1])
        for request_id in ids:
            attempt = parent / "attempts" / request_id
            response = read(attempt / "response.json")
            request = read(attempt / "wire_request.json")
            native.append(
                {
                    "turn_id": response["turn_id"],
                    "request_id": request_id,
                    "kind": "author" if request_id == ids[-1] else "checkup",
                    "passage": index + 1,
                    "output": response["raw"],
                    "wire_text": request["input"][0]["text"],
                }
            )
    checkpoint = {
        "archive_commit": "78caed4e7fe50523097452dc95a30d6ed7c82d6a",
        "parent_session": "backstage_sol_20260918/sessions/breakwater_A",
        "thread_id": read(parent / "agent.json")["thread_id"],
        "last_turn_id": native[-1]["turn_id"],
        "native_turns": native,
        "published_turns": state["turns"][:18],
    }
    write(HERE / "checkpoint.json", checkpoint)
    write(HERE / "inputs.json", read(PREVIOUS / "inputs.json")["breakwater"][18:24])
    conditions = {}
    for replicate in (1, 2):
        names = ["control", "correction", "development", "combined", "reframe"]
        random.SystemRandom().shuffle(names)
        for offset, name in enumerate(names, start=1):
            label = f"sample-{(replicate - 1) * 5 + offset:02d}"
            conditions[label] = {"condition": name, "replicate": replicate}
    write(HERE / "conditions.json", conditions)
    write(HERE / "live_before.json", hashes(ACTIVE / "sessions"))
    source_files = [
        "PROTOCOL.md",
        "run.py",
        "rpc.py",
        "checkpoint.json",
        "inputs.json",
        "conditions.json",
    ]
    source_files += [
        str(p.relative_to(HERE))
        for root in (notes, HERE / "sources")
        for p in root.rglob("*")
        if p.is_file()
    ]
    write(
        HERE / "provenance.json",
        {
            "prepared_at": now(),
            "model": MODEL,
            "effort": "max",
            "summary": "detailed",
            "concurrency": 3,
            "source_sha256": {
                name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                for name in sorted(source_files)
            },
        },
    )
    print("Prepared 10 masked continuations, 6 new passages each; no model calls.")


def bind(server, workspace, label, checkpoint):
    path = HERE / "samples" / label
    binding_path = path / "binding.json"
    if binding_path.exists():
        binding = read(binding_path)
        response = server.rpc(
            "thread/resume",
            {
                "threadId": binding["thread_id"],
                "model": MODEL,
                "cwd": workspace,
                "approvalPolicy": "never",
                "sandbox": "read-only",
            },
        )
        assert response["thread"]["id"] == binding["thread_id"]
        assert response["model"] == MODEL and response["reasoningEffort"] == "max"
        assert not response.get("instructionSources", [])
        return binding
    params = {
        "threadId": checkpoint["thread_id"],
        "lastTurnId": checkpoint["last_turn_id"],
        "model": MODEL,
        "cwd": workspace,
        "approvalPolicy": "never",
        "sandbox": "read-only",
    }
    write(path / "fork_request.json", params)
    response = server.rpc("thread/fork", params)
    write(path / "fork_response.json", response)
    verify_fork(response, checkpoint)
    binding = {
        "thread_id": response["thread"]["id"],
        "parent_thread_id": checkpoint["thread_id"],
        "through_turn_id": checkpoint["last_turn_id"],
    }
    write(binding_path, binding)
    event(event="bound", sample=label, **binding)
    return binding


def generate(server, label, binding, index, player_input, note):
    path = HERE / "samples" / label / f"turn-{index + 19:02d}"
    if (path / "response.json").exists():
        response = read(path / "response.json")
        assert response["status"] == "completed" and response["raw"].strip()
        return
    if (path / "started.json").exists():
        raise RuntimeError(
            f"Unknown outcome for {label}/{path.name}; inspect native thread, do not resample"
        )
    params = {
        "threadId": binding["thread_id"],
        "model": MODEL,
        "effort": "max",
        "summary": "detailed",
        "input": [
            {
                "type": "text",
                "text": wire_text(player_input, note if index == 0 else ""),
            }
        ],
    }
    write(path / "request.json", params)
    (path / "request.txt").write_text(params["input"][0]["text"])
    write(path / "started.json", {"at": now(), "thread_id": binding["thread_id"]})
    event(event="started", sample=label, passage=index + 19)

    def save(item):
        with (path / "public_events.jsonl").open("a") as output:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")

    try:
        response = server.turn(params, save)
        write(path / "response.json", response)
        assert response["status"] == "completed" and response["raw"].strip()
        assert set(response["item_types"]) <= {
            "userMessage",
            "reasoning",
            "agentMessage",
        }
        (path / "output.md").write_text(response["raw"])
        event(
            event="completed",
            sample=label,
            passage=index + 19,
            turn_id=response["turn_id"],
            words=len(response["raw"].split()),
        )
    except Exception as error:
        write(path / "error.json", {"at": now(), "error": repr(error)})
        raise


def run():
    from rpc import AppServer

    provenance = read(HERE / "provenance.json")
    for name, digest in provenance["source_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    checkpoint = read(HERE / "checkpoint.json")
    conditions = read(HERE / "conditions.json")
    inputs = read(HERE / "inputs.json")
    with tempfile.TemporaryDirectory(prefix="ayoa-authored-checkups-") as workspace:
        server = AppServer(workspace)
        event(event="server_started", pid=server.process.pid, command=server.command)
        try:
            bindings = {
                label: bind(server, workspace, label, checkpoint)
                for label in conditions
            }
            with ThreadPoolExecutor(max_workers=3) as pool:
                for index, player_input in enumerate(inputs):
                    futures = [
                        pool.submit(
                            generate,
                            server,
                            label,
                            bindings[label],
                            index,
                            player_input,
                            (
                                HERE / "notes" / (condition["condition"] + ".txt")
                            ).read_text(),
                        )
                        for label, condition in conditions.items()
                    ]
                    for future in as_completed(futures):
                        future.result()
                    event(event="round_completed", passage=index + 19)
        finally:
            server.close()
            write(HERE / "live_after.json", hashes(ACTIVE / "sessions"))
            event(event="server_stopped")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "run"}:
        raise SystemExit("Usage: run.py prepare|run")
    {"prepare": prepare, "run": run}[sys.argv[1]]()
