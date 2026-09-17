"""Check transport, exact context changes and final/summary exports without calls."""

import copy
import json
import sys
from pathlib import Path

WORKTREE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(WORKTREE))
from narrative import core  # noqa: E402

ROOT = Path(__file__).resolve().parent
rows = []
transcript = [
    "# Closure and rhyme comparisons\n",
    "Independent continuations, not a sequential story.\n",
]
for phase, base in (("primary", ROOT), ("rhyme_refinement", ROOT / "rhyme_refinement")):
    manifest = core.read_json(base / "manifest.json")
    for name, expected in manifest["source_hashes"].items():
        assert core.digest((base / name).read_bytes()) == expected, name
    for cell in manifest["cells"]:
        path = base / "cells" / cell["id"]
        request = core.read_json(path / "request.json")
        assert core.digest((path / "request.json").read_bytes()) == cell["request_sha256"]
        assert request["model"] == "gpt-5.6-terra"
        assert request["reasoning"] == {"effort": "max", "summary": "detailed"}
        if phase == "primary":
            original_case = "t21_reading" if cell["case"] == "t21_engage" else cell["case"]
            original = core.read_json(ROOT / "originals" / original_case / "request.json")
            expected = copy.deepcopy(original)
            expected["reasoning"] = request["reasoning"]
            expected["instructions"] = "\n\n".join(
                (ROOT / "snapshots" / cell["condition"] / name).read_text().strip()
                for name in core.PREFIX_ORDER
            )
            if cell["case"] == "t21_engage":
                expected["input"][-1]["content"] = (
                    'I set my book aside and turn toward Seraphel. "What are you reading?"'
                )
        else:
            expected = core.read_json(ROOT / "cells" / cell["case"] / "closure/1/request.json")
            if cell["condition"] == "paired_rhyme":
                old = (base / "current_canon.md").read_text().strip()
                new = (base / "paired_canon.md").read_text().strip()
                assert expected["instructions"].count(old) == 1
                expected["instructions"] = expected["instructions"].replace(old, new)
        assert request == expected, cell["id"]
        assert (path / "request.txt").read_text() == core.render_request(request)
        response = core.read_json(path / "response.json")
        assert response["transport"] == "proxy" and response["executor"] == "codex"
        result = core.read_json(path / "result.json")
        assert result["status"] == "completed"
        assert core.response_text(response) == (path / "output.md").read_text()
        assert core.response_summary(response) == (path / "summary.txt").read_text()
        rows.append({"phase": phase, **cell, **result})
        transcript.extend(
            ["\n## " + phase + "/" + cell["id"] + "\n", (path / "output.md").read_text()]
        )
session = WORKTREE / "sessions/covenant-9780e3d01fcb"
source_state = ROOT / "live_update/before/state.json"
if not source_state.exists():
    source_state = session / "state.json"
assert (
    core.digest(source_state.read_bytes())
    == core.read_json(ROOT / "manifest.json")["source_state_sha256"]
)
core.write_json(
    ROOT / "validation.json",
    {
        "validated_at": core.now(),
        "all_completed": True,
        "coding_agent_calls": len(rows),
        "direct_api_calls": 0,
        "all_context_differences_match_protocol": True,
        "all_exports_match_raw_responses": True,
        "evaluated_source_state_matches_frozen_before_correction": True,
        "cells": rows,
    },
)
(ROOT / "TRANSCRIPTS.md").write_text("\n".join(transcript) + "\n")
print(
    json.dumps({"coding_agent_calls": len(rows), "direct_api_calls": 0, "all_checks_passed": True})
)
