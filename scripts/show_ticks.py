#!/usr/bin/env python3
"""Extract off-stage tick-agent outputs from a checkpoint.

Ticks are interleaved into each character's rolling conversation alongside
regular responses. They're identified by a marker string from the
agent_tick prompt template.

Usage:
    .venv/bin/python scripts/show_ticks.py <path_to_ckpt.json>
    .venv/bin/python scripts/show_ticks.py app/storage/saves/New-0/ckpt_0007.json
    .venv/bin/python scripts/show_ticks.py <ckpt.json> --character ashara_vel_kothren
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TICK_MARKER = "## What You Do This Tick"


def _content_text(msg: dict) -> str:
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c:
        return c[0].get("text", "") or ""
    return ""


def _tick_pairs(msgs: list[dict]) -> list[tuple[str, str]]:
    """Walk user/assistant pairs and return only those whose user message
    carries the tick marker."""
    out: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(msgs):
        u, a = msgs[i], msgs[i + 1]
        if u.get("role") == "user" and a.get("role") == "assistant":
            utxt = _content_text(u)
            if TICK_MARKER in utxt:
                out.append((utxt, _content_text(a)))
        i += 2
    return out


def _split_parenthetical(text: str) -> tuple[str, str]:
    """Same last-trailing-paren split as the engine. Returns (prose, intent)."""
    stripped = (text or "").rstrip()
    if not stripped or not stripped.endswith(")"):
        return text or "", ""
    depth = 0
    open_idx = -1
    for i in range(len(stripped) - 1, -1, -1):
        ch = stripped[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                open_idx = i
                break
    if open_idx == -1:
        return text or "", ""
    return stripped[:open_idx].rstrip(), stripped[open_idx + 1 : -1].strip()


def _print_tick(idx: int, assistant_text: str) -> None:
    """v11+: agent outputs are prose + trailing parenthetical, not JSON.

    Pre-rework ticks were structured JSON with `private_updates` fields;
    those checkpoints fall through to the JSON branch for back-compat.
    """
    if assistant_text.lstrip().startswith("{"):
        try:
            obj = json.loads(assistant_text)
        except json.JSONDecodeError:
            print(f"  tick {idx} (legacy json): (unparseable)")
            return
        pu = obj.get("private_updates", {})
        print(f"  tick {idx} (legacy json):")
        for o in pu.get("current_objectives", []) or []:
            print(f"    - {o}")
        for d in pu.get("directives_sent", []) or []:
            print(f"    → {d.get('to')}: {d.get('content', '')}")
        if pu.get("moved_to"):
            print(f"    moved_to: {pu['moved_to']}")
        for s in pu.get("scenes_created", []) or []:
            print(f"    scenes_created: {s.get('scene_id')} — {s.get('name', '')}")
        return

    prose, intent = _split_parenthetical(assistant_text)
    print(f"  tick {idx}:")
    if prose.strip():
        print(f"    prose: {prose.strip()}")
    else:
        print(f"    prose: (silent)")
    if intent:
        print(f"    intent: {intent}")
    else:
        print(f"    intent: (none — missing trailing parenthetical)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Show tick-agent outputs from a checkpoint.")
    ap.add_argument("checkpoint", help="Path to ckpt_NNNN.json")
    ap.add_argument(
        "--character", "-c",
        help="Filter to a single character_id.",
    )
    args = ap.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    with path.open() as f:
        d = json.load(f)

    cc = d.get("character_conversations", {})
    if args.character:
        cc = {args.character: cc.get(args.character, [])}

    any_printed = False
    for cid, msgs in cc.items():
        pairs = _tick_pairs(msgs)
        if not pairs:
            continue
        any_printed = True
        print(f"\n===== {cid}  ({len(pairs)} tick(s)) =====")
        for k, (_, atxt) in enumerate(pairs, 1):
            _print_tick(k, atxt)

    if not any_printed:
        print("(no tick outputs found in this checkpoint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
