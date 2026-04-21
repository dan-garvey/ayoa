"""Migrate a legacy flat `app/storage/saves/` layout into the split
stories/ + sessions/ layout.

All three paths are explicit arguments — no defaults that could
accidentally target production state. Pass --dry-run to see the
classification without moving anything.

Usage:
    .venv/bin/python scripts/migrate_storage.py \\
        --legacy-dir app/storage/saves \\
        --stories-dir app/storage/stories \\
        --sessions-dir app/storage/sessions

    # Preview first:
    .venv/bin/python scripts/migrate_storage.py \\
        --legacy-dir app/storage/saves \\
        --stories-dir app/storage/stories \\
        --sessions-dir app/storage/sessions \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bot.engine_bridge import migrate_legacy_saves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--legacy-dir", required=True, help="Path to legacy saves/ directory")
    parser.add_argument("--stories-dir", required=True, help="Target dir for story imports")
    parser.add_argument("--sessions-dir", required=True, help="Target dir for player sessions")
    parser.add_argument("--dry-run", action="store_true", help="Classify without moving")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr,
    )
    try:
        s, n = migrate_legacy_saves(
            legacy_dir=Path(args.legacy_dir),
            stories_dir=Path(args.stories_dir),
            sessions_dir=Path(args.sessions_dir),
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    action = "Would move" if args.dry_run else "Moved"
    print(f"{action} {s} story dir(s) and {n} session dir(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
