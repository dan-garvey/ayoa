"""Prepare a separately recorded follow-up after an observed unrhymed response."""

import copy
import sys
from pathlib import Path

WORKTREE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(WORKTREE))
from narrative import core  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rhyme_refinement"
OUT.mkdir()
old = (ROOT / "snapshots/closure/canon.md").read_text()
old_rule = """All her spoken dialogue, including brief replies and questions, must rhyme.
Use genuine end rhymes and natural word order without padding or distorting the
meaning to complete a rhyme."""
new_rule = """Every line of her spoken dialogue has an audible end-rhyme partner within
that same speech, including brief replies and questions. Keep the meaning and
word order natural, without padding or distorting a thought to complete a rhyme."""
assert old.count(old_rule) == 1
new = old.replace(old_rule, new_rule)
(OUT / "current_canon.md").write_text(old)
(OUT / "paired_canon.md").write_text(new)
cells = []
for repeat in (1, 2):
    for case in ("t21_reading", "t21_engage"):
        source_path = ROOT / "cells" / case / "closure/1/request.json"
        source = core.read_json(source_path)
        for condition in ("current_rhyme", "paired_rhyme"):
            request = copy.deepcopy(source)
            if condition == "paired_rhyme":
                assert request["instructions"].count(old.strip()) == 1
                request["instructions"] = request["instructions"].replace(old.strip(), new.strip())
            cell_id = f"{case}/{condition}/{repeat}"
            path = OUT / "cells" / cell_id
            path.mkdir(parents=True)
            core.write_json(path / "request.json", request)
            core.atomic_write(path / "request.txt", core.render_request(request).encode())
            cells.append(
                {
                    "id": cell_id,
                    "case": case,
                    "condition": condition,
                    "repeat": repeat,
                    "request_sha256": core.digest((path / "request.json").read_bytes()),
                }
            )
core.write_json(
    OUT / "manifest.json",
    {
        "prepared_at": core.now(),
        "transport": "codex_proxy_only",
        "model": "gpt-5.6-terra",
        "reasoning": "max",
        "summary": "detailed",
        "trigger": "../cells/t21_reading/closure/2/output.md",
        "change": "Require an audible end-rhyme partner for every spoken line within the same speech; no fixed meter, rhyme scheme or example dialogue.",
        "source_hashes": {
            name: core.digest((OUT / name).read_bytes())
            for name in ("current_canon.md", "paired_canon.md")
        },
        "cells": cells,
    },
)
print("Prepared 8 fresh coding-agent rhyme comparison calls; no inference yet")
