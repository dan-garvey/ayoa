#!/usr/bin/env python3
"""Dump a character prompt and its saved conversation at a checkpoint.

Renders the generic system prompt and a representative current user packet,
then prints the conversation history exactly as it was at the checkpoint. The
latest saved user message is the actual packet from that turn, apart from the
disposable presentation catalog that is intentionally omitted from history.

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
    """Render the generic system prefix and one representative user packet."""
    from app.engine.context_builder import (
        build_character_turn_request_packet,
        build_dnd_character_identity_sentence,
        build_dnd_player_identities_block,
        format_elapsed_agent_turn_block,
        format_pending_observations_block,
    )
    from app.engine.prompt_manager import PromptManager
    from app.engine.turn_loop_contracts import format_character_moment

    pm = PromptManager()
    current_moment = "\n\n".join(
        block.strip()
        for block in (
            build_dnd_character_identity_sentence(checkpoint, character),
            build_dnd_player_identities_block(checkpoint),
            format_elapsed_agent_turn_block(character, checkpoint),
            format_pending_observations_block(character),
            format_character_moment(
                frame="foreground",
                local_context="<<IMMEDIATE_CIRCUMSTANCE_PLACEHOLDER>>",
            ),
            "<<ACTIVE_RULESET_STATE_WHEN_APPLICABLE>>",
            "<<PRESENTATION_CATALOG_WHEN_AVAILABLE>>",
        )
        if block.strip()
    )
    request_packet = build_character_turn_request_packet(
        character,
        checkpoint,
        current_moment,
    )
    ruleset_id = str(
        getattr(
            getattr(checkpoint.session.config, "settings", None),
            "ruleset_id",
            "",
        ) or ""
    )
    if ruleset_id == "dnd5e_basic":
        ruleset_guidance = pm.render("agent_ruleset_dnd5e").strip()
    elif ruleset_id == "one_star_ascension":
        ruleset_guidance = pm.render("agent_ruleset_one_star").strip()
    else:
        ruleset_guidance = ""

    msgs = pm.render_messages(
        "agent_turn",
        ruleset_guidance=ruleset_guidance,
        request_packet=request_packet,
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
        print("# Sent as `system=` to the API. Cached for characters using the")
        print("# same active ruleset guidance.")
        print("#" * 80)
        print(sys_text)
        print()
        print("#" * 80)
        print(
            f"# CURRENT USER PACKET  "
            f"({len(user_template)} chars, ~{len(user_template)//4} tokens, with placeholders)"
        )
        print("# Re-rendered every turn with current self and circumstance.")
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
