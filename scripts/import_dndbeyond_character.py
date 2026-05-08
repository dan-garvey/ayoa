#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.dnd_character_import import (
    load_dndbeyond_export,
    mechanics_from_snapshot,
    normalize_dndbeyond_export,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a user-exported D&D Beyond character JSON file."
    )
    parser.add_argument("input", help="Path to the D&D Beyond browser export JSON")
    parser.add_argument(
        "--out",
        help="Optional path for the canonical Ayoa D&D character snapshot JSON.",
    )
    parser.add_argument(
        "--mechanics-out",
        help="Optional path for the compact CharacterRecord.mechanics JSON.",
    )
    parser.add_argument(
        "--no-raw-source",
        action="store_true",
        help="Do not preserve the raw DDB payload inside the snapshot.",
    )
    args = parser.parse_args()

    export = load_dndbeyond_export(args.input)
    snapshot = normalize_dndbeyond_export(
        export,
        include_raw_source=not args.no_raw_source,
    )
    mechanics = mechanics_from_snapshot(snapshot)

    if args.out:
        Path(args.out).write_text(
            json.dumps(snapshot, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.mechanics_out:
        Path(args.mechanics_out).write_text(
            json.dumps(mechanics, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    statblock = snapshot["statblock"]
    defenses = statblock["defenses"]
    hp = defenses["hit_points"]
    print(f"name: {snapshot['identity']['name']}")
    print(f"ruleset: {snapshot['ruleset_id']}")
    print(f"level: {snapshot['identity'].get('total_level', 0)}")
    print(f"proficiency_bonus: {statblock['proficiency_bonus']}")
    print(f"armor_class: {defenses['armor_class']['value']}")
    print(f"hit_points: {hp['current']}/{hp['max']} (+{hp.get('temporary', 0)} temp)")
    print(f"skills: {len(statblock['skills'])}")
    print(f"actions: {len(statblock.get('actions', []))}")
    print(f"spells: {len((statblock.get('spellcasting') or {}).get('spells', []))}")
    print(f"resources: {len(statblock.get('resources', []))}")
    if args.out:
        print(f"snapshot_out: {args.out}")
    if args.mechanics_out:
        print(f"mechanics_out: {args.mechanics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
