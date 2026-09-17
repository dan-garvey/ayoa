"""Reconstruct every new request and check private-note projections without calls."""

import copy
import json
import sys
from pathlib import Path

ROOT = Path("/home/dan/ayoa-worktrees/covenant-single-llm")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from narrative import core
from narrative.chat import ChatApp

app = ChatApp(HERE)
report = {}
transcript = ["# Periodic checkup smoke test: all first outputs", ""]
for name, prefix_count in [
    ("covenant", 19),
    ("breakwater", 0),
    ("breakwater_verified", 1),
]:
    session = HERE / name
    manifest, final = core.load_session(session)
    assert final["pending"] is None
    state = copy.deepcopy(final)
    state.update(turns=state["turns"][:prefix_count], checkups=[])
    attempts = []
    for path in (session / "attempts").iterdir():
        meta = core.read_json(path / "meta.json")
        if "stage" in meta and meta["turn"] >= prefix_count:
            attempts.append((meta["prepared_at"], path, meta))
    attempts.sort(key=lambda item: item[0])
    rows = []
    for _, path, meta in attempts:
        turn = final["turns"][meta["turn"]]
        state["pending"] = {
            "kind": "turn",
            "stage": meta["stage"],
            "input": turn["input"],
            "submission_id": meta["submission_id"],
            "request_id": meta["request_id"],
        }
        expected = core.make_request(session, manifest, state)
        assert expected == core.read_json(path / "request.json")
        assert core.render_request(expected) == (path / "request.txt").read_text()
        assert expected["model"] == "gpt-5.6-terra"
        assert expected["reasoning"] == {"effort": "max", "summary": "detailed"}
        response = core.read_json(path / "response.json")
        assert response["transport"] == "proxy" and response["executor"] == "codex"
        assert response["status"] == "completed" and response["exit_code"] == 0
        text = core.response_text(response)
        summary = core.response_summary(response)
        assert not summary or summary not in core.render_request(expected)
        if meta["stage"] == "checkup":
            state["checkups"].append(
                {"before_turn": len(state["turns"]), "request_id": meta["request_id"]}
            )
        else:
            assert (
                turn["output"] == text and turn["responses"][-1] == meta["request_id"]
            )
            state["turns"].append(turn)
        transcript.extend(
            [
                f"## {name}: {meta['stage']} before/for passage {meta['turn'] + 1}",
                "",
                "**Player submission**",
                "",
                turn["input"],
                "",
                "**Complete output**",
                "",
                text,
                "",
            ]
        )
        rows.append(
            {
                "stage": meta["stage"],
                "passage": meta["turn"] + 1,
                "request_id": meta["request_id"],
                "duration_s": response["duration_s"],
                "words": len(text.split()),
                "summary_characters": len(core.response_summary(response)),
            }
        )
    normal = app.view(name)
    inspected = app.view(name, compare=True)
    assert "checkups" not in normal and "checkup_every" not in normal
    assert len(inspected["checkups"]) == len(final["checkups"])
    for item in final["checkups"]:
        notes = core.response_text(
            core.read_json(
                core.attempt_dir(session, item["request_id"]) / "response.json"
            )
        )
        assert notes not in (session / "transcript.md").read_text()
        assert notes not in json.dumps(normal, ensure_ascii=False)
    report[name] = {
        "calls": rows,
        "request_reconstruction": True,
        "publication_exact": True,
        "notes_only_in_inspection": True,
        "no_pending": True,
    }
core.atomic_write(HERE / "TRANSCRIPTS.md", "\n".join(transcript).encode())
core.write_json(HERE / "validation.json", report)
print(json.dumps(report, indent=2))
