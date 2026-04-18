"""One-shot migration of legacy (schema 1.0) checkpoints to the rolling-conversation schema (2.0).

Changes applied:
- Adds session_conversation, narrator_conversation, character_conversations.
- Removes characters[].memory (episodic/summaries/observation_queue).
- Adds characters[].pending_observations = [].
- Updates prompt_versions keys: narrator/discriminator/agent → event_router/agent/narrator_phase2.
- Bumps schema_version to "2.0".

Existing files are backed up to `<file>.pre_conv_migration` before overwriting.

Usage:
    .venv/bin/python scripts/migrate_saves.py [saves_dir]

Default saves_dir: app/storage/saves
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def migrate_checkpoint(data: dict) -> tuple[dict, bool]:
    """Convert a legacy checkpoint dict to the new schema.

    Returns (migrated_data, changed). If `changed` is False, the file was
    already on 2.0+ and nothing needs writing.
    """
    version = data.get("schema_version", "1.0")
    if version == "2.0":
        return data, False

    data["schema_version"] = "2.0"

    data.setdefault("session_conversation", [])
    data.setdefault("narrator_conversation", [])
    data.setdefault("character_conversations", {})

    for char in data.get("characters", []):
        char.pop("memory", None)
        char.setdefault("pending_observations", [])

    data["prompt_versions"] = {
        "event_router": "v3",
        "agent": "v5",
        "narrator_phase2": "v4",
    }

    return data, True


def migrate_file(path: Path, dry_run: bool = False) -> str:
    """Migrate a single checkpoint file. Returns a status string."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return f"SKIP (not json): {path}: {e}"

    migrated, changed = migrate_checkpoint(data)
    if not changed:
        return f"SKIP (already v2.0): {path}"

    if dry_run:
        return f"WOULD MIGRATE: {path}"

    backup = path.with_suffix(path.suffix + ".pre_conv_migration")
    shutil.copy2(path, backup)
    path.write_text(json.dumps(migrated, indent=2))
    return f"MIGRATED:      {path}  (backup: {backup.name})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "saves_dir",
        nargs="?",
        default="app/storage/saves",
        help="Directory containing per-session save folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be migrated without writing.",
    )
    args = parser.parse_args()

    root = Path(args.saves_dir)
    if not root.exists():
        print(f"ERROR: saves dir does not exist: {root}", file=sys.stderr)
        return 1

    ckpts = sorted(root.glob("*/ckpt_*.json"))
    if not ckpts:
        print(f"No checkpoints found under {root}.")
        return 0

    print(f"Found {len(ckpts)} checkpoint file(s) under {root}")
    migrated = skipped = 0
    for path in ckpts:
        msg = migrate_file(path, dry_run=args.dry_run)
        print(f"  {msg}")
        if msg.startswith("MIGRATED") or msg.startswith("WOULD MIGRATE"):
            migrated += 1
        else:
            skipped += 1

    print()
    verb = "would migrate" if args.dry_run else "migrated"
    print(f"Summary: {verb} {migrated}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
