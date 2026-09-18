import json
import subprocess
import sys

from rpc import AppServer, public_result


def test_nested_reasoning_content_is_never_retained():
    value = {
        "turn": {
            "items": [
                {
                    "type": "reasoning",
                    "id": "r",
                    "summary": ["Exposed summary"],
                    "content": ["PRIVATE"],
                    "encrypted_content": "OPAQUE",
                }
            ]
        }
    }
    safe = public_result(value)
    assert safe["turn"]["items"][0]["summary"] == ["Exposed summary"]
    assert "PRIVATE" not in json.dumps(safe) and "OPAQUE" not in json.dumps(safe)


def test_two_turns_share_one_process_and_thread(monkeypatch, tmp_path):
    script = tmp_path / "fake_codex.py"
    script.write_text("""import json, sys
count = 0
for line in sys.stdin:
    q = json.loads(line)
    if "id" not in q:
        continue
    if q["method"] == "initialize":
        print(json.dumps({"id": q["id"], "result": {}}), flush=True)
        continue
    assert q["method"] == "turn/start"
    assert q["params"]["threadId"] == "same-thread"
    count += 1
    tid = "turn-" + str(count)
    print(json.dumps({"id": q["id"], "result": {"turn": {"id": tid}}}), flush=True)
    for item in [
        {"type": "reasoning", "id": "r" + tid, "summary": ["Exposed"], "content": ["PRIVATE"]},
        {"type": "agentMessage", "id": "a" + tid, "phase": "final_answer", "text": str(count)}
    ]:
        print(json.dumps({"method": "item/completed", "params": {
            "threadId": "same-thread", "turnId": tid, "item": item}}), flush=True)
    print(json.dumps({"method": "turn/completed", "params": {
        "threadId": "same-thread", "turn": {"id": tid, "status": "completed", "items": []}
    }}), flush=True)
""")
    popen = subprocess.Popen
    spawned = []

    def spawn(command, **kwargs):
        spawned.append(command)
        return popen([sys.executable, str(script)], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    server = AppServer(tmp_path)
    events = []
    try:
        first = server.turn({"threadId": "same-thread", "input": []}, events.append)
        second = server.turn({"threadId": "same-thread", "input": []}, events.append)
    finally:
        server.close()
    assert len(spawned) == 1
    assert first["raw"] == "1" and second["raw"] == "2"
    assert first["thread_id"] == second["thread_id"] == "same-thread"
    assert second["reasoning_summaries"] == ["Exposed"]
    assert "PRIVATE" not in json.dumps(events)
