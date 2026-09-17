"""Replay the same Breakwater prefix after adding temporal-evidence verification."""

import shutil
import sys
from pathlib import Path

ROOT = Path("/home/dan/ayoa-worktrees/covenant-single-llm")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from narrative import core
from narrative.codex import CodexProxy

source = HERE / "breakwater"
session = HERE / "breakwater_verified"
shutil.copytree(source, session, ignore=shutil.ignore_patterns(".lock"))
manifest, state = core.load_session(session)
player_input = state["turns"][1]["input"]
state.update(turns=state["turns"][:1], checkups=[], pending=None)
kept = set(state["turns"][0]["responses"])
for path in (session / "attempts").iterdir():
    if path.name not in kept:
        shutil.rmtree(path)
data = (ROOT / "prompts/checkup.txt").read_bytes()
core.atomic_write(session / "snapshot/checkup.txt", data)
manifest["source_sha256"]["checkup.txt"] = core.digest(data)
core.write_json(session / "manifest.json", manifest)
state["manifest_sha256"] = core.digest((session / "manifest.json").read_bytes())
core.write_json(session / "state.json", state)
core.export(session)
print(
    "Replaying the same player message and opening; only checkup.txt changed.",
    flush=True,
)
result = core.submit(session, player_input, CodexProxy())
assert result["status"] == "published"
print("Both first calls saved.", flush=True)
