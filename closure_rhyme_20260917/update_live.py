"""Apply this user's approved prompt correction with a full local backup."""

import copy
import os
import shutil
import signal
import sys
from pathlib import Path

WORKTREE = Path(sys.argv[1]).resolve()
SERVER_PID = int(sys.argv[2])
BACKUP = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(WORKTREE))
from narrative import core  # noqa: E402

ROOT = Path(__file__).resolve().parent
session = WORKTREE / "sessions/covenant-9780e3d01fcb"
for state_file in (WORKTREE / "sessions").glob("*/state.json"):
    assert not core.is_busy(state_file.parent), "An existing story is generating; defer restart"
    assert core.read_json(state_file).get("pending") is None, "An existing story has a pending turn"
command = Path(f"/proc/{SERVER_PID}/cmdline").read_bytes().split(b"\0")
assert b"narrative" in command and b"chat" in command
assert not BACKUP.exists(), "Backup exists; inspect recovery state instead of applying again"
out = ROOT / "live_update"
out.mkdir()
with core.locked(session):
    manifest, state = core.load_session(session)
    assert state["pending"] is None
    assert (session / "snapshot/author.txt").read_bytes() == (
        ROOT / "snapshots/author.txt"
    ).read_bytes()
    assert (session / "snapshot/canon.md").read_bytes() == (
        ROOT / "snapshots/canon.md"
    ).read_bytes()
    assert (WORKTREE / "prompts/author.txt").read_bytes() == (
        ROOT / "snapshots/closure/author.txt"
    ).read_bytes()
    assert (WORKTREE / "stories/covenant/canon.md").read_bytes() == (
        ROOT / "snapshots/closure/canon.md"
    ).read_bytes()
    shutil.copytree(session, BACKUP)
    before = {
        str(p.relative_to(session)): core.digest(p.read_bytes())
        for p in session.rglob("*")
        if p.is_file()
    }
    os.kill(SERVER_PID, signal.SIGINT)
    changed = ("snapshot/author.txt", "snapshot/canon.md", "manifest.json", "state.json")
    try:
        for name, source in (
            ("author.txt", WORKTREE / "prompts/author.txt"),
            ("canon.md", WORKTREE / "stories/covenant/canon.md"),
        ):
            core.atomic_write(session / "snapshot" / name, source.read_bytes())
            manifest["source_sha256"][name] = core.digest(source.read_bytes())
        core.write_json(session / "manifest.json", manifest)
        state["manifest_sha256"] = core.digest((session / "manifest.json").read_bytes())
        core.write_json(session / "state.json", state)
        new_manifest, new_state = core.load_session(session)
        old_state = core.read_json(BACKUP / "state.json")
        assert {k: v for k, v in new_state.items() if k != "manifest_sha256"} == {
            k: v for k, v in old_state.items() if k != "manifest_sha256"
        }
        after = {
            str(p.relative_to(session)): core.digest(p.read_bytes())
            for p in session.rglob("*")
            if p.is_file()
        }
        assert set(before) == set(after)
        assert {name for name in before if before[name] != after[name]} == set(changed)
        future = copy.deepcopy(new_state)
        future["pending"] = {"kind": "turn", "input": "VALIDATION_ONLY_NOT_SUBMITTED"}
        request = core.make_request(session, new_manifest, future)
        assert request["reasoning"] == {"effort": "max", "summary": "detailed"}
        assert request["input"][-1]["content"] == "VALIDATION_ONLY_NOT_SUBMITTED"
        assert request["instructions"] == "\n\n".join(
            (session / "snapshot" / name).read_text().strip() for name in core.PREFIX_ORDER
        )
    except Exception:
        for name in changed:
            core.atomic_write(session / name, (BACKUP / name).read_bytes())
        raise
    for side, folder in (("before", BACKUP), ("after", session)):
        for name in changed:
            target = out / side / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / name, target)
    core.write_json(
        out / "audit.json",
        {
            "updated_at": core.now(),
            "session": session.name,
            "backup": str(BACKUP),
            "changed_files": list(changed),
            "before_sha256": before,
            "after_sha256": after,
            "published_turns_unchanged": len(new_state["turns"]),
            "all_attempts_and_transcript_unchanged": True,
            "future_request_uses_updated_sources_and_detailed_summaries": True,
            "generation_calls": 0,
        },
    )
print(
    "Updated current playtest prompts; all published prose and attempts preserved. Restart chat now."
)
