"""Frozen-prefix closure comparison using only fresh coding-agent proxies."""

from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "forced_engagement_20260917"
OLD_DIRECTION = """Let the player's conduct set the degree of direction. Pursue an interest they
act on; when they watch or hesitate, let other people's purposes develop into
something that happens. A declined cause can continue without them. Honor quiet
and the endpoint of a requested activity or time skip. Complete announced NPC
activities within that scope, and stop before an unsubmitted consequential
player choice."""
NEW_DIRECTION = """Follow the activity and degree of interaction the player chooses. People can
finish conversations and pursue purposes that do not involve the protagonist;
a natural stopping point is a valid end to a passage. When the player returns
to an activity, let it proceed, with interruptions arising from the situation
and other people's purposes rather than a need to offer another interaction.
Honor the endpoint of a requested activity or time skip. Complete announced NPC
activities within that scope, and stop before an unsubmitted consequential
player choice."""
OLD_HANDOFF = """When someone addresses the protagonist, end the passage early to allow the
player to react. Favor back-and-forth dialogue over giving the player several
things to respond to at once."""
NEW_HANDOFF = """When an exchange calls for the player's response, leave room for it before
introducing another demand on their attention."""


def prepare(worktree, session, core):
    if (ROOT / "manifest.json").exists():
        raise RuntimeError("Already prepared; refusing to overwrite frozen inputs")
    snapshots = ROOT / "snapshots"
    snapshots.mkdir()
    for condition in ("control", "closure"):
        (snapshots / condition).mkdir()
    for name in ("author.txt", "direction.md", "canon.md"):
        shutil.copyfile(PREVIOUS / "originals" / "snapshot" / name, snapshots / name)
    old_author = (snapshots / "author.txt").read_text()
    assert old_author.count(OLD_DIRECTION) == old_author.count(OLD_HANDOFF) == 1
    candidate = old_author.replace(OLD_DIRECTION, NEW_DIRECTION).replace(OLD_HANDOFF, NEW_HANDOFF)
    for condition, author in (("control", old_author), ("closure", candidate)):
        (snapshots / condition / "author.txt").write_text(author)
        shutil.copyfile(snapshots / "direction.md", snapshots / condition / "direction.md")
        shutil.copyfile(worktree / "stories/covenant/canon.md", snapshots / condition / "canon.md")
    originals = ROOT / "originals"
    originals.mkdir()
    paths = {
        "t13_observe": session / "attempts/aec57874692f46ed8fc9b72c0bc2b0a4",
        "t16_departure": PREVIOUS / "originals/t16_departure",
        "t20_library": PREVIOUS / "originals/t20_library",
        "t21_reading": PREVIOUS / "originals/t21_reading",
    }
    for case, source in paths.items():
        shutil.copytree(source, originals / case)
    paths["t21_engage"] = paths["t21_reading"]
    cells = []
    old_prefix = "\n\n".join((snapshots / name).read_text().strip() for name in core.PREFIX_ORDER)
    for repeat in (1, 2):
        for case, source in paths.items():
            original = core.read_json(source / "request.json")
            assert original["instructions"] == old_prefix
            assert original["model"] == "gpt-5.6-terra"
            for condition in ("control", "closure"):
                request = copy.deepcopy(original)
                request["reasoning"] = {"effort": "max", "summary": "detailed"}
                request["instructions"] = "\n\n".join(
                    (snapshots / condition / name).read_text().strip() for name in core.PREFIX_ORDER
                )
                if case == "t21_engage":
                    request["input"][-1]["content"] = (
                        'I set my book aside and turn toward Seraphel. "What are you reading?"'
                    )
                cell_id = f"{case}/{condition}/{repeat}"
                path = ROOT / "cells" / cell_id
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
    implementation = ROOT / "implementation"
    implementation.mkdir()
    for name in ("core.py", "codex.py"):
        shutil.copyfile(worktree / "narrative" / name, implementation / name)
    files = [
        p for base in (snapshots, originals, implementation) for p in base.rglob("*") if p.is_file()
    ]
    core.write_json(
        ROOT / "manifest.json",
        {
            "prepared_at": core.now(),
            "model": "gpt-5.6-terra",
            "reasoning": "max",
            "summary": "detailed",
            "transport": "codex_proxy_only",
            "cells": cells,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
            ).strip(),
            "source_session": session.name,
            "source_state_sha256": core.digest((session / "state.json").read_bytes()),
            "codex_version": subprocess.check_output(["codex", "--version"], text=True).strip(),
            "source_hashes": {
                str(p.relative_to(ROOT)): core.digest(p.read_bytes()) for p in sorted(files)
            },
        },
    )
    print(f"Prepared {len(cells)} coding-agent calls; no inference yet", flush=True)


def execute(cell, core):
    from narrative.codex import CodexProxy

    path = ROOT / "cells" / cell["id"]
    if (path / "started.json").exists():
        raise RuntimeError("Refusing to repeat a dispatched call")
    assert core.digest((path / "request.json").read_bytes()) == cell["request_sha256"]
    request = core.read_json(path / "request.json")
    print("Starting " + cell["id"], flush=True)
    try:
        core._call_model(path, request, CodexProxy(), "proxy")
        response = core.read_json(path / "response.json")
        output, summary = core.response_text(response), core.response_summary(response)
        core.atomic_write(path / "output.md", output.encode())
        core.atomic_write(path / "summary.txt", summary.encode())
        core.write_json(
            path / "result.json",
            {
                "status": "completed",
                "word_count": len(output.split()),
                "summary_characters": len(summary),
                "duration_s": response["duration_s"],
            },
        )
        print("Completed " + cell["id"], flush=True)
    except Exception as exc:
        core.write_json(path / "result.json", {"status": "failed", "type": type(exc).__name__})
        print("Failed " + cell["id"] + ": " + type(exc).__name__, flush=True)


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--prepare", type=Path, metavar="SESSION")
    parser.add_argument("--pending", action="store_true")
    args = parser.parse_args()
    if args.evidence_root:
        ROOT = args.evidence_root.resolve()
    sys.path.insert(0, str(args.worktree))
    from narrative import core

    if args.prepare:
        prepare(args.worktree, args.prepare, core)
    elif args.pending:
        manifest = core.read_json(ROOT / "manifest.json")
        for name, expected in manifest["source_hashes"].items():
            assert core.digest((ROOT / name).read_bytes()) == expected, name
        cells = [
            c for c in manifest["cells"] if not (ROOT / "cells" / c["id"] / "started.json").exists()
        ]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(execute, cell, core) for cell in cells]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
