"""Round-trip on-disk checkpoint JSON through `CheckpointFile` (current Pydantic
models) so legacy fields are migrated or dropped in a well-defined way.

Pydantic validators and defaults apply on load; `model_dump_json(indent=2)` is
written back only when the serialized form differs (so unchanged files are not
touched on disk except by timestamp if you re-save without diff — this script
skips the write when bytes match).

Typical use after a schema bump (e.g. `is_player`→`is_playable`, removal of
`prompt_versions`, new optional fields with defaults like `player_primer`):

    .venv/bin/python scripts/migrate_checkpoints.py --dry-run

    .venv/bin/python scripts/migrate_checkpoints.py --skip sengoku_koryu_s1

Paths default to `app/storage/stories` and `app/storage/sessions` under the
repo root. Skip patterns match **directory or file name segments** (each path
component) with exact string or `fnmatch` glob (e.g. `*_old`).

This script is intentionally self-contained: re-run after future schema
bumps. It does not call the LLM; it does not fabricate `player_primer` text.

Before validate, a **migration-only** pass removes
`canonical_events[*].canonical_event.scene_delta.new_scene_id` if present
(old shape; the live `SceneDelta` model no longer has that key). Everything
else is straight Pydantic load/save.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas.checkpoint import CheckpointFile


def _strip_legacy_scene_delta_keys(data: Any) -> None:
    """Remove keys that older checkpoints still carry but `SceneDelta` no longer
    allows (`extra="forbid"`). Relocation history lives in
    `EventRouterOutput.roster_moves` now, not on `scene_delta.new_scene_id`.

    Migration script only — keeps production models strict while old saves
    can still round-trip.
    """
    if not isinstance(data, dict):
        return
    cev = data.get("canonical_events")
    if not isinstance(cev, list):
        return
    for item in cev:
        if not isinstance(item, dict):
            continue
        ce = item.get("canonical_event")
        if not isinstance(ce, dict):
            continue
        sd = ce.get("scene_delta")
        if isinstance(sd, dict) and "new_scene_id" in sd:
            sd.pop("new_scene_id", None)


def _iter_checkpoint_paths(stories_dir: Path, sessions_dir: Path) -> list[Path]:
    out: list[Path] = []
    for base in (stories_dir, sessions_dir):
        if not base.is_dir():
            logging.warning("not a directory or missing, skipping: %s", base)
            continue
        for p in sorted(base.rglob("ckpt_*.json")):
            if p.is_file():
                out.append(p)
    return out


def _parse_skip_args(skip_raw: list[str] | None) -> list[str]:
    """Split comma-separated tokens and individual patterns into a flat list."""
    if not skip_raw:
        return []
    out: list[str] = []
    for chunk in skip_raw:
        for part in chunk.split(","):
            t = part.strip()
            if t:
                out.append(t)
    return out


def _path_matches_skip(absolute: Path, skip_patterns: list[str]) -> bool:
    for part in absolute.parts:
        for pat in skip_patterns:
            if part == pat or fnmatch.fnmatch(part, pat):
                return True
    return False


def _dropped_top_level_keys(raw: object, model_keys: set[str]) -> list[str]:
    if not isinstance(raw, dict):
        return []
    return sorted(k for k in raw if k not in model_keys)


def _normalization_summary(old_text: str, ckpt: CheckpointFile) -> str:
    """Human-readable list of what was normalized away (best-effort)."""
    try:
        raw = json.loads(old_text)
    except json.JSONDecodeError:
        return "normalized (could not re-parse for drop summary)"
    # Keys present in input dict but not in model serialization (e.g. unknown/extra
    # top-level fields dropped by Pydantic).
    dump = ckpt.model_dump(mode="json", exclude_none=False)
    if not isinstance(raw, dict):
        return "normalized"
    model_keys = set(dump.keys())
    dropped = _dropped_top_level_keys(raw, model_keys)
    notes: list[str] = []
    if dropped:
        notes.append(", ".join(dropped) + " (top-level)")
    if '"is_player"' in old_text:
        notes.append("is_player → is_playable (character records)")
    if '"new_scene_id"' in old_text:
        notes.append("scene_delta.new_scene_id (legacy, dropped)")
    if not notes:
        notes.append("reordered keys/whitespace/typed values")
    return "normalized (dropped: " + "; ".join(notes) + ")"


def _on_disk_text_matches_dump(old_text: str, new_text: str) -> bool:
    """True if the only difference is a single trailing newline after `}`."""
    return new_text == old_text or new_text == old_text.rstrip("\n")


def _migrate_file(path: Path, *, dry_run: bool) -> tuple[str, str | None]:
    """Returns (one-line report, error message or None)."""
    try:
        old_text = path.read_text(encoding="utf-8")
    except OSError as e:
        return (f"{path}: error: read failed: {e}", str(e))

    try:
        raw: Any = json.loads(old_text)
    except json.JSONDecodeError as e:
        return (f"{path}: error: JSON decode: {e!r}", repr(e))

    if isinstance(raw, dict):
        _strip_legacy_scene_delta_keys(raw)

    try:
        ckpt = CheckpointFile.model_validate(raw)
    except Exception as e:  # noqa: BLE001 — surface any validation error per file
        return (f"{path}: error: {e!r}", repr(e))

    new_text = ckpt.model_dump_json(indent=2)
    if _on_disk_text_matches_dump(old_text, new_text):
        return (f"{path}: unchanged", None)

    summary = _normalization_summary(old_text, ckpt)
    if not dry_run:
        try:
            path.write_text(new_text, encoding="utf-8")
        except OSError as e:
            return (f"{path}: error: write failed: {e!r}", repr(e))
    return (f"{path}: {summary}", None)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__.split("Typical use")[0].strip(),
    )
    parser.add_argument(
        "--stories-dir",
        type=Path,
        default=repo / "app/storage/stories",
        help="Root for story-import checkpoints (default: app/storage/stories)",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=repo / "app/storage/sessions",
        help="Root for session checkpoints (default: app/storage/sessions)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate every file; do not write",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Directory/file name glob or exact segment; comma-separate in one arg. "
        "May be passed multiple times. Matches any path component.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    skip_patterns = _parse_skip_args(args.skip)
    all_paths = _iter_checkpoint_paths(args.stories_dir, args.sessions_dir)
    selected: list[Path] = [p for p in all_paths if not _path_matches_skip(p, skip_patterns)]

    counts: Counter[str] = Counter()
    error_paths: list[str] = []

    for path in selected:
        line, err = _migrate_file(path, dry_run=args.dry_run)
        print(line, flush=True)
        if err is not None:
            counts["error"] += 1
            error_paths.append(str(path))
        elif line.rsplit(": ", 1)[-1] == "unchanged":
            counts["unchanged"] += 1
        else:
            counts["normalized"] += 1

    skipped = len(all_paths) - len(selected)
    if skipped:
        logging.info("skipped %d file(s) matching --skip patterns", skipped)

    print(
        f"\nSummary: total={len(selected)} "
        f"unchanged={counts['unchanged']} "
        f"normalized={counts['normalized']} "
        f"errors={counts['error']}",
        flush=True,
    )
    if error_paths:
        print("Error files:", flush=True)
        for ep in error_paths:
            print(f"  {ep}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
