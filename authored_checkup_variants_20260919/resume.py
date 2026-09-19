"""Resume preserved native conversations after confirmed pre-output capacity errors."""

import json
import shutil
import time

import run


def capacity_failure(path):
    if not (path / "response.json").exists():
        return False
    response = run.read(path / "response.json")
    events = [
        json.loads(line)
        for line in (path / "public_events.jsonl").read_text().splitlines()
    ]
    return (
        response["status"] == "failed"
        and response["raw"] == ""
        and set(response["item_types"]) <= {"userMessage"}
        and any(
            e["method"] == "error"
            and e["params"]["error"].get("codexErrorInfo") == "serverOverloaded"
            for e in events
        )
    )


def recover(server, label, binding, path):
    assert capacity_failure(path)
    response = run.read(path / "response.json")
    before = server.rpc(
        "thread/read", {"threadId": binding["thread_id"], "includeTurns": True}
    )
    run.write(path / "recovery_before.json", before)
    turns = before["thread"]["turns"]
    expected = run.read(run.HERE / "checkpoint.json")["native_turns"]
    preceding = sorted(
        p for p in path.parent.glob("turn-*/response.json") if p.parent.name < path.name
    )
    expected_ids = [t["turn_id"] for t in expected] + [
        run.read(p)["turn_id"] for p in preceding
    ]
    assert [t["id"] for t in turns] == expected_ids + [response["turn_id"]]
    assert not any(
        i["type"] in {"reasoning", "agentMessage"} for i in turns[-1]["items"]
    )
    params = {
        "threadId": binding["thread_id"],
        "lastTurnId": expected_ids[-1],
        "model": run.MODEL,
        "cwd": before["thread"]["cwd"],
        "approvalPolicy": "never",
        "sandbox": "read-only",
    }
    run.write(path / "recovery_fork_request.json", params)
    after = server.rpc("thread/fork", params)
    run.write(path / "recovery_after.json", after)
    assert [t["id"] for t in after["thread"]["turns"]] == expected_ids
    assert after["model"] == run.MODEL and after["reasoningEffort"] == "max"
    assert not after.get("instructionSources", [])
    for previous, inherited in zip(turns[:-1], after["thread"]["turns"]):
        assert previous["items"] == inherited["items"]
    run.write(path / "prior_binding.json", binding)
    binding["thread_id"] = after["thread"]["id"]
    run.write(path.parent / "active_binding.json", binding)
    archive = path.parent / "failed_attempts" / path.name
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / f"capacity-{len(list(archive.iterdir())) + 1:02d}"
    shutil.move(str(path), target)
    run.event(
        event="capacity_recovered",
        sample=label,
        passage=int(path.name[-2:]),
        failed_turn=response["turn_id"],
        evidence=str(target.relative_to(run.HERE)),
    )


def main():
    original = run.generate
    original_bind = run.bind

    def bind(server, workspace, label, checkpoint):
        path = run.HERE / "samples" / label / "active_binding.json"
        if not path.exists():
            return original_bind(server, workspace, label, checkpoint)
        binding = run.read(path)
        response = server.rpc(
            "thread/resume",
            {
                "threadId": binding["thread_id"],
                "model": run.MODEL,
                "cwd": workspace,
                "approvalPolicy": "never",
                "sandbox": "read-only",
            },
        )
        assert response["thread"]["id"] == binding["thread_id"]
        assert response["model"] == run.MODEL and response["reasoningEffort"] == "max"
        assert not response.get("instructionSources", [])
        return binding

    def retry(server, label, binding, index, player_input, note):
        path = run.HERE / "samples" / label / f"turn-{index + 19:02d}"
        if capacity_failure(path):
            recover(server, label, binding, path)
        for attempt in range(4):
            try:
                return original(server, label, binding, index, player_input, note)
            except AssertionError:
                if not capacity_failure(path) or attempt == 3:
                    raise
                recover(server, label, binding, path)
                time.sleep(20)

    run.generate = retry
    run.bind = bind
    run.run()


if __name__ == "__main__":
    main()
