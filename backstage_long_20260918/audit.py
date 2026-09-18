"""Offline reconstruction, condition unblinding and mechanical evidence summary."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime"))
from narrative import core


def hashes(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != ".lock" and "__pycache__" not in p.parts
    }


provenance = core.read_json(HERE / "provenance.json")
assert hashes(HERE / "runtime") == provenance["runtime"]
assert hashes(HERE / "sources") == provenance["sources"]
assert core.digest((HERE / "inputs.json").read_bytes()) == provenance["inputs_sha256"]
conditions = core.read_json(HERE / "conditions.json")
inputs = core.read_json(HERE / "inputs.json")
unique_calls = {}
report = {}
all_prose = ["# Longer evaluation: full prose transcripts", ""]
all_notes = ["# Longer evaluation: all checkup outputs", ""]
for lane in [*(f"{story}_seed" for story in inputs), *conditions]:
    session = HERE / "sessions" / lane
    manifest, state = core.load_session(session)
    assert state["pending"] is None
    story = manifest["story"]
    expected_count = 4 if lane.endswith("_seed") else 24
    assert len(state["turns"]) == expected_count
    expected_due = [4, 9, 14, 19] if manifest["checkup_every"] else []
    assert [item["before_turn"] for item in state["checkups"]] == expected_due
    assert [turn["input"] for turn in state["turns"]] == inputs[story][:expected_count]
    rows = []
    all_prose.extend([f"## {lane}", ""])
    for turn in state["turns"]:
        number = turn["turn"]
        simulated = copy.deepcopy(state)
        simulated["turns"] = state["turns"][:number]
        current_checkup = next(
            (c for c in state["checkups"] if c["before_turn"] == number), None
        )
        stages = [("author", turn["responses"][-1])]
        if current_checkup:
            stages.insert(0, ("checkup", current_checkup["request_id"]))
        for stage, request_id in stages:
            path = core.attempt_dir(session, request_id)
            meta = core.read_json(path / "meta.json")
            assert meta["stage"] == stage and meta["turn"] == number
            simulated["checkups"] = [
                c
                for c in state["checkups"]
                if c["before_turn"] < number
                or (stage == "author" and c["before_turn"] == number)
            ]
            simulated["pending"] = {
                "kind": "turn",
                "stage": stage,
                "input": turn["input"],
                "submission_id": meta["submission_id"],
                "request_id": request_id,
            }
            request = core.make_request(session, manifest, simulated)
            assert request == core.read_json(path / "request.json")
            assert core.render_request(request) == (path / "request.txt").read_text()
            assert request["model"] == "gpt-5.6-terra"
            assert request["reasoning"] == {"effort": "max", "summary": "detailed"}
            raw = core.read_json(path / "response.json")
            assert raw["transport"] == "proxy" and raw["executor"] == "codex"
            assert raw["status"] == "completed" and raw["exit_code"] == 0
            output = core.response_text(raw)
            response_hash = core.digest((path / "response.json").read_bytes())
            if request_id in unique_calls:
                assert unique_calls[request_id]["response_sha256"] == response_hash
            else:
                unique_calls[request_id] = {
                    "lane": lane,
                    "stage": stage,
                    "turn": number + 1,
                    "duration_s": raw["duration_s"],
                    "words": len(output.split()),
                    "summary_characters": len(core.response_summary(raw)),
                    "response_sha256": response_hash,
                }
            if stage == "author":
                assert turn["output"] == output
            else:
                assert output not in (session / "transcript.md").read_text()
                all_notes.extend(
                    [f"## {lane}: before turn {number + 1}", "", output, ""]
                )
        rows.append(
            {
                "turn": number + 1,
                "words": len(turn["output"].split()),
                "question_marks": turn["output"].count("?"),
                "ends_with_question": bool(re.search(r"\?[\s\"”’*]*$", turn["output"])),
                "private_marker": "blue glass bird" in turn["output"].lower()
                or "red umbrella" in turn["output"].lower(),
            }
        )
        all_prose.extend(
            [
                f"### Turn {number + 1}",
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
    report[lane] = {
        "story": story,
        "interval": manifest["checkup_every"],
        "turns": rows,
        "request_reconstruction": True,
        "publication_exact": True,
        "checkups_excluded_from_transcript": True,
    }
    if not lane.endswith("_seed"):
        seed = core.load_session(HERE / "sessions" / f"{story}_seed")[1]
        assert state["turns"][:4] == seed["turns"]
        assert len(list((session / "attempts").iterdir())) == len(state["turns"]) + len(
            state["checkups"]
        )

assert len(unique_calls) == 96
assert sum(c["stage"] == "checkup" for c in unique_calls.values()) == 8
events = [json.loads(line) for line in (HERE / "events.jsonl").read_text().splitlines()]
dispatched = {}
received = {}
published = set()
in_flight = set()
maximum_concurrent = 0
for event in events:
    request_id = event["request_id"]
    if event["event"] == "dispatch":
        assert request_id not in dispatched
        dispatched[request_id] = event
        in_flight.add(request_id)
        maximum_concurrent = max(maximum_concurrent, len(in_flight))
    elif event["event"] == "received":
        assert event["status"] == "completed"
        assert request_id in in_flight and request_id not in received
        received[request_id] = event
        in_flight.remove(request_id)
    else:
        assert event["event"] == "published"
        assert request_id in received and request_id not in published
        published.add(request_id)
assert not in_flight and maximum_concurrent <= 3
assert set(dispatched) == set(received) == set(unique_calls)
assert published == {k for k, v in unique_calls.items() if v["stage"] == "author"}
for request_id, call in unique_calls.items():
    first, last = dispatched[request_id], received[request_id]
    assert (first["lane"], first["stage"], first["turn"]) == (
        call["lane"],
        call["stage"],
        call["turn"],
    )
    call["proxy_elapsed_s_excluding_queue"] = (
        datetime.fromisoformat(last["at"]) - datetime.fromisoformat(first["at"])
    ).total_seconds()
review_freeze = core.read_json(HERE / "review_before_unblind.json")
assert (
    core.digest((HERE / review_freeze["file"]).read_bytes()) == review_freeze["sha256"]
)
before = provenance["live_session_files_before"]
after = core.read_json(HERE / "live_session_files_after.json")
report["validation"] = {
    "unique_calls": len(unique_calls),
    "checkups": 8,
    "author_calls": 88,
    "existing_live_sessions_unchanged": before == after,
    "all_requests_exact": True,
    "all_first_outputs_kept": True,
    "frozen_runtime_sources_and_inputs_unchanged": True,
    "frozen_pre_note_review_unchanged": True,
    "dispatch_receive_publication_log_exact": True,
    "maximum_concurrent_calls": maximum_concurrent,
    "conditions": conditions,
    "calls": unique_calls,
}
core.write_json(HERE / "validation.json", report)
core.atomic_write(HERE / "TRANSCRIPTS.md", "\n".join(all_prose).encode())
core.atomic_write(HERE / "CHECKUPS.md", "\n".join(all_notes).encode())
metrics = {}
for lane in conditions:
    turns = report[lane]["turns"][4:]
    metrics[lane] = {
        "interval": report[lane]["interval"],
        "continuation_turns": len(turns),
        "words_mean": round(statistics.mean(r["words"] for r in turns), 1),
        "words_median": statistics.median(r["words"] for r in turns),
        "words_total": sum(r["words"] for r in turns),
        "question_ending_passages": sum(r["ends_with_question"] for r in turns),
        "passages_with_question_marks": sum(r["question_marks"] > 0 for r in turns),
    }
for stage in ("author", "checkup"):
    rows = [c for c in unique_calls.values() if c["stage"] == stage]
    elapsed = [r["proxy_elapsed_s_excluding_queue"] for r in rows]
    metrics[stage] = {
        "count": len(rows),
        "proxy_elapsed_s_median_excluding_queue": round(statistics.median(elapsed), 1),
        "proxy_elapsed_s_min_excluding_queue": round(min(elapsed), 1),
        "proxy_elapsed_s_max_excluding_queue": round(max(elapsed), 1),
        "call_elapsed_s_median_including_queue": round(
            statistics.median(r["duration_s"] for r in rows), 1
        ),
        "words_median": statistics.median(r["words"] for r in rows),
        "words_min": min(r["words"] for r in rows),
        "words_max": max(r["words"] for r in rows),
        "summary_nonempty": sum(r["summary_characters"] > 0 for r in rows),
        "summary_characters_min": min(r["summary_characters"] for r in rows),
        "summary_characters_max": max(r["summary_characters"] for r in rows),
    }
core.write_json(HERE / "descriptive_metrics.json", metrics)
print(
    json.dumps(
        {k: v for k, v in report["validation"].items() if k != "calls"}, indent=2
    )
)
