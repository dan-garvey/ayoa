#!/usr/bin/env python3
"""Dump the full agent prompt for a single character at a checkpoint.

Renders the system prompt the same way `CharacterAgent._run_beat` does
(template + character-derived variables), then prints the conversation
history exactly as it was at the time of the checkpoint. The LATEST user
message in the conversation is the user-message body that the LLM saw on
that turn (mode header + mode body + pending observations +
acting-character framing).

This is a debug tool: load the checkpoint AFTER the turn you want to
inspect (e.g. ckpt_0027 to see what Ashara was sent during turn 27).
The history goes up to AND INCLUDES that turn's exchange.

Usage:
    .venv/bin/python scripts/dump_agent_prompt.py <ckpt.json> <character_id>
    .venv/bin/python scripts/dump_agent_prompt.py app/storage/sessions/t8/ckpt_0027.json ashara_vel_kothren
    .venv/bin/python scripts/dump_agent_prompt.py <ckpt.json> <id> --last-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_checkpoint(path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.schemas.checkpoint import CheckpointFile
    return CheckpointFile.model_validate_json(path.read_text())


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return first.get("text", "") or ""
    return ""


def _render_system_and_user_template(checkpoint, character) -> tuple[str, str]:
    """Render the agent template the way CharacterAgent does, then split
    it on `<<<USER>>>` to separate the cached system prefix from the
    per-turn user-message template.

    The system text is byte-identical regardless of mode_block — only
    the user message changes per turn. We inject placeholders for
    `mode_header` and `mode_block` so the per-turn-only fields are
    obvious in the user template dump (the LATEST user message in
    history shows the real values for the most recent turn).
    """
    from app.engine.context_builder import (
        build_character_packet,
        build_character_state,
        build_world_context,
        format_elapsed_agent_turn_block,
        format_pending_observations_block,
    )
    from app.engine.prompt_manager import PromptManager

    pm = PromptManager()
    char_identity = build_character_packet(character)
    char_state = build_character_state(character)
    pending_block = (
        format_elapsed_agent_turn_block(character, checkpoint)
        + format_pending_observations_block(character)
    )
    ruleset_id = str(
        getattr(
            getattr(checkpoint.session.config, "settings", None),
            "ruleset_id",
            "",
        ) or ""
    )
    ruleset_addon = (
        pm.render("agent_ruleset_dnd5e").strip()
        if ruleset_id == "dnd5e_basic"
        else ""
    )

    msgs = pm.render_messages(
        "agent",
        agent_ruleset_system_addon=ruleset_addon,
        **char_identity,
        **char_state,
        world_context=build_world_context(character, checkpoint),
        pending_observations_block=pending_block,
        mode_header="<<MODE_HEADER_PLACEHOLDER>>",
        mode_block="<<MODE_BLOCK_PLACEHOLDER>>",
    )
    return msgs[0]["content"], msgs[1]["content"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="Path to ckpt_XXXX.json")
    p.add_argument("character_id", help="Character id (e.g. ashara_vel_kothren)")
    p.add_argument(
        "--last-only", action="store_true",
        help="Print only the most recent (user, assistant) exchange.",
    )
    p.add_argument(
        "--no-system", action="store_true",
        help="Skip the system prompt render (useful when you only want history).",
    )
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    ckpt = _load_checkpoint(args.checkpoint)
    char = next(
        (c for c in ckpt.characters if c.character_id == args.character_id),
        None,
    )
    if char is None:
        print(
            f"Character {args.character_id!r} not found in checkpoint. "
            f"Available: {[c.character_id for c in ckpt.characters]}",
            file=sys.stderr,
        )
        return 2

    history = ckpt.character_conversations.get(args.character_id, [])
    msgs = [
        {"role": m.role, "content": _content_text(m.content)}
        for m in history
    ]

    print("=" * 80)
    print("AGENT PROMPT DUMP")
    print(f"  checkpoint:  {args.checkpoint}")
    print(f"  character:   {char.name} ({args.character_id})")
    print(f"  location:    {char.location}")
    print(f"  history:     {len(msgs)} messages")
    print(f"  pending obs: {len(char.pending_observations)} entries")
    print("=" * 80)

    if not args.no_system:
        sys_text, user_template = _render_system_and_user_template(ckpt, char)
        print()
        print("#" * 80)
        print(f"# SYSTEM PROMPT  ({len(sys_text)} chars, ~{len(sys_text)//4} tokens)")
        print("# Sent as `system=` to the API. Cached on every call (byte-identical")
        print("# between turns for this character).")
        print("#" * 80)
        print(sys_text)
        print()
        print("#" * 80)
        print(
            f"# PER-TURN USER-MESSAGE TEMPLATE  "
            f"({len(user_template)} chars, ~{len(user_template)//4} tokens, with placeholders)"
        )
        print("# Re-rendered every turn with fresh mode_header/mode_block + state.")
        print("# The LATEST exchange below shows the actual filled-in version.")
        print("#" * 80)
        print(user_template)

    print()
    print("#" * 80)
    if args.last_only:
        print("# LATEST USER + ASSISTANT (turn that produced this checkpoint)")
    else:
        print("# FULL CONVERSATION HISTORY")
    print("#" * 80)

    pairs = list(_pair_messages(msgs))
    if args.last_only:
        pairs = pairs[-1:]

    for i, pair in enumerate(pairs, start=1):
        print()
        print(f"---- exchange {i}/{len(pairs)} " + "-" * 60)
        for msg in pair:
            role = msg["role"].upper()
            print(f"\n>>> {role}")
            print(msg["content"])

    return 0


def _pair_messages(msgs):
    """Yield (user, assistant) pairs from a flat alternating list. If the
    list is malformed (orphan user or starts with assistant), yield
    singletons so the caller still sees the message."""
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "user" and i + 1 < len(msgs) and msgs[i+1]["role"] == "assistant":
            yield [m, msgs[i+1]]
            i += 2
        else:
            yield [m]
            i += 1


if __name__ == "__main__":
    raise SystemExit(main())
