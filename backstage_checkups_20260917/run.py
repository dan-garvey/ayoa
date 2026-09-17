"""First-result smoke test of periodic checkups, using fresh coding proxies only."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/home/dan/ayoa-worktrees/covenant-single-llm")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from narrative import core
from narrative.codex import CodexProxy


def hashes(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != ".lock"
    }


source = ROOT / "sessions/covenant-9780e3d01fcb"
before = hashes(source)
original = core.read_json(source / "state.json")
assert not original["pending"] and not core.is_busy(source)
assert len(original["turns"]) == 22
covenant = HERE / "covenant"
shutil.copytree(source, covenant, ignore=shutil.ignore_patterns(".lock"))
manifest = core.read_json(covenant / "manifest.json")
state = core.read_json(covenant / "state.json")
manifest.update(version=3, checkup_every=5)
for name in ("author.txt", "checkup.txt"):
    data = (ROOT / "prompts" / name).read_bytes()
    core.atomic_write(covenant / "snapshot" / name, data)
    manifest["source_sha256"][name] = core.digest(data)
core.write_json(covenant / "manifest.json", manifest)
state.update(
    manifest_sha256=core.digest((covenant / "manifest.json").read_bytes()),
    turns=state["turns"][:19],
    checkups=[],
    pending=None,
)
core.write_json(covenant / "state.json", state)
# Keep only attempts belonging to this replay's active prefix and earlier versions.
kept = {rid for turn in state["turns"] for rid in turn["responses"]}
for attempt in (covenant / "attempts").iterdir():
    if attempt.name not in kept:
        shutil.rmtree(attempt)
core.export(covenant)

breakwater = HERE / "breakwater"
core.init_session(breakwater, ROOT / "stories/breakwater", checkup_every=2)
inputs = {
    "covenant": [
        original["turns"][19]["input"],
        original["turns"][20]["input"],
        "I keep reading until the end of the chapter.",
    ],
    "breakwater": [
        "Begin the story.",
        "I introduce myself and ask what is most urgent before the screening.",
    ],
}
core.write_json(
    HERE / "protocol.json",
    {
        "model": "gpt-5.6-terra",
        "reasoning": "max",
        "summary": "detailed",
        "transport": "fresh coding proxies",
        "inputs": inputs,
        "source_session_hashes": before,
        "source_turns_retained": 19,
        "limitations": "Single samples, no baseline; workflow smoke test, not quality proof.",
        "runtime_hashes": hashes(ROOT / "narrative"),
    },
)
runtime_copy = HERE / "runtime_source"
runtime_copy.mkdir()
for name in ("core.py", "codex.py"):
    shutil.copyfile(ROOT / "narrative" / name, runtime_copy / name)

runner = CodexProxy()
for session in (covenant, breakwater):
    for text in inputs[session.name]:
        print(f"{session.name}: submitting {text!r}", flush=True)
        result = core.submit(session, text, runner)
        print(
            f"{session.name}: {result['status']}, passage {result['turn'] + 1}",
            flush=True,
        )
    print(json.dumps(core.export(session)), flush=True)
assert hashes(source) == before
core.write_json(HERE / "source_unchanged.json", {"unchanged": True, "hashes": before})
print("All first results saved; original session unchanged.", flush=True)
