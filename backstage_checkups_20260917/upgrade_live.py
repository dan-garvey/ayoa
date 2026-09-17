"""Explicit local format-3 conversion after stopping the idle chat server."""

import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

ROOT = Path("/home/dan/ayoa-worktrees/covenant-single-llm")
HERE = Path(__file__).resolve().parent
BACKUP = Path("/home/dan/ayoa-evaluations/backstage_checkups_20260917/live_before")
sys.path.insert(0, str(ROOT))
from narrative import core


def files(path):
    return {
        str(p.relative_to(path)): core.digest(p.read_bytes())
        for p in path.rglob("*")
        if p.is_file() and p.name != ".lock"
    }


sessions = sorted(path.parent for path in (ROOT / "sessions").rglob("state.json"))
assert len(sessions) == 7
before = {}
with ExitStack() as stack:
    for session in sessions:
        assert not core.is_busy(session), session
        stack.enter_context(core.locked(session))
        name = session.relative_to(ROOT / "sessions").as_posix()
        backup = BACKUP / name
        if not backup.exists():
            shutil.copytree(session, backup)
        manifest = core.read_json(backup / "manifest.json")
        state = core.read_json(backup / "state.json")
        assert manifest["version"] == 2 and state["pending"] is None
        assert state["manifest_sha256"] == core.digest(
            (backup / "manifest.json").read_bytes()
        )
        for source_name, digest in manifest["source_sha256"].items():
            assert (
                core.digest((backup / "snapshot" / source_name).read_bytes()) == digest
            )
        assert (backup / "snapshot/author.txt").read_text().count(
            "Return the next passage of fiction."
        ) == 1
        current = core.read_json(session / "state.json")
        assert current["pending"] is None and current["turns"] == state["turns"]
        before[name] = files(backup)
        live = files(session)
        for file, digest in before[name].items():
            if file not in {"state.json", "manifest.json", "snapshot/author.txt"}:
                assert live[file] == digest
    report = {}
    for session in sessions:
        name = session.relative_to(ROOT / "sessions").as_posix()
        manifest = core.read_json(BACKUP / name / "manifest.json")
        state = core.read_json(BACKUP / name / "state.json")
        author = (BACKUP / name / "snapshot/author.txt").read_text()
        new_clause = (
            (ROOT / "prompts/author.txt")
            .read_text()
            .split("character. ", 1)[1]
            .split("\n\n", 1)[0]
        )
        author = author.replace("Return the next passage of fiction.", new_clause)
        for source_name, data in {
            "author.txt": author.encode(),
            "checkup.txt": (ROOT / "prompts/checkup.txt").read_bytes(),
        }.items():
            core.atomic_write(session / "snapshot" / source_name, data)
            manifest["source_sha256"][source_name] = core.digest(data)
        manifest.update(version=3, checkup_every=5)
        core.write_json(session / "manifest.json", manifest)
        state.update(
            checkups=[],
            manifest_sha256=core.digest((session / "manifest.json").read_bytes()),
        )
        core.write_json(session / "state.json", state)
        loaded_manifest, loaded_state = core.load_session(session)
        old_state = core.read_json(BACKUP / name / "state.json")
        assert loaded_state["turns"] == old_state["turns"]
        assert loaded_state.get("player_names") == old_state.get("player_names")
        after = files(session)
        changed = {
            file for file, digest in before[name].items() if after[file] != digest
        }
        assert changed == {"state.json", "manifest.json", "snapshot/author.txt"}
        assert set(after) - set(before[name]) == {"snapshot/checkup.txt"}
        report[name] = {
            "backup": str(BACKUP / name),
            "before": before[name],
            "after": after,
            "published_turns_preserved": len(loaded_state["turns"]),
            "changed_files": sorted(changed),
            "added_files": ["snapshot/checkup.txt"],
            "checkup_every": 5,
            "next_checkup_message": (len(state["turns"]) // 5 + 1) * 5,
        }
    core.write_json(HERE / "live_upgrade.json", report)
print(
    "Backed up and converted all seven sessions; all passages and attempts unchanged."
)
