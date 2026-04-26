"""Replay test for Option B (drop resolved_outcome from narrator input).

Reconstructs narrator calls from a saved session checkpoint with the
`resolved_outcome:` lines stripped out of each canonical event block (in
both the current turn's user message AND in the prior rolling history),
runs the narrator, and prints the original render alongside the new one
for side-by-side comparison.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.client import LLMClient  # noqa: E402
from app.llm.config import LLMConfig  # noqa: E402
from app.schemas.narrator import NarratorFinalOutput  # noqa: E402
from app.engine.prompt_manager import PromptManager  # noqa: E402

CHECKPOINT = Path("app/storage/sessions/t8/ckpt_0028.json")
TURNS = [7, 17, 18]


def strip_resolved_outcome(text: str) -> str:
    """Remove every `resolved_outcome: ...` line from a narrator user message.

    The canonical event block formats one event per `## Event N:` section,
    each with attempted_action / resolved_outcome / observable_facts lines.
    We snip the resolved_outcome line and leave everything else intact.
    """
    out_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("resolved_outcome:"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def parse_assistant_text(content) -> str:
    if isinstance(content, list):
        for blk in content:
            if blk.get("type") == "text":
                t = blk["text"]
                try:
                    return json.loads(t).get("final_text", "")
                except Exception:
                    return t
    if isinstance(content, str):
        return content
    return ""


async def replay_one_turn(
    client: LLMClient,
    pm: PromptManager,
    hist: list[dict],
    turn_idx: int,
) -> tuple[str, str]:
    """Returns (original_render, new_render) for the given 1-indexed turn."""
    user_idx = 2 * (turn_idx - 1)
    asst_idx = user_idx + 1
    if asst_idx >= len(hist):
        raise ValueError(f"turn {turn_idx} out of range")

    original_render = parse_assistant_text(hist[asst_idx]["content"])
    original_user = hist[user_idx]["content"]
    if not isinstance(original_user, str):
        raise ValueError("user message is not a plain string")

    scrubbed_user = strip_resolved_outcome(original_user)

    scrubbed_history: list[dict] = []
    for i in range(0, user_idx, 2):
        prev_user = hist[i]["content"]
        prev_asst = hist[i + 1]["content"]
        if isinstance(prev_user, str):
            prev_user = strip_resolved_outcome(prev_user)
        scrubbed_history.append({"role": "user", "content": prev_user})
        scrubbed_history.append({"role": "assistant", "content": prev_asst})

    full_msgs = pm.render_messages(
        "narrator_phase2",
        setting_summary="",
        narrative_rules="",
        canonical_event="",
        agent_outputs="",
        scene_context="",
        user_input="",
        acting_character_name="",
        player_characters_block="",
    )
    system_msg = full_msgs[0]

    messages: list[dict] = [system_msg]
    messages.extend(scrubbed_history)
    messages.append({"role": "user", "content": scrubbed_user})

    response = await client.complete(
        role="narrator",
        messages=messages,
        response_model=NarratorFinalOutput,
        temperature=0.5,
        max_tokens=4000,
        cache=False,
        compact=False,
    )
    new_render = response.parsed.final_text if response.parsed else ""
    return original_render, new_render


async def main():
    ck = json.loads(CHECKPOINT.read_text())
    hist = ck["narrator_conversations"]["player_garvey"]
    print(f"loaded checkpoint with {len(hist)} narrator history messages")

    client = LLMClient(config=LLMConfig.from_env())
    pm = PromptManager()

    for turn in TURNS:
        print()
        print("#" * 78)
        print(f"# TURN {turn}")
        print("#" * 78)
        try:
            original, new = await replay_one_turn(client, pm, hist, turn)
        except Exception as e:
            print(f"ERROR on turn {turn}: {e}")
            continue
        print()
        print("--- ORIGINAL (resolved_outcome present) ---")
        print(original)
        print()
        print("--- NEW (resolved_outcome stripped) ---")
        print(new)
        print()


if __name__ == "__main__":
    asyncio.run(main())
