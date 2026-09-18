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
    after = server.rpc(
        "thread/rollback", {"threadId": binding["thread_id"], "numTurns": 1}
    )
    run.write(path / "recovery_after.json", after)
    assert [t["id"] for t in after["thread"]["turns"]] == expected_ids
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
    run.run()


if __name__ == "__main__":
    main()
