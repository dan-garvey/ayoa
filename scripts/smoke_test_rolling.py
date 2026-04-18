"""End-to-end smoke test for the rolling-conversation refactor.

Loads a migrated checkpoint, runs one real turn through the full pipeline
(EventRouter → Agents → Narrator), and verifies that the rolling conversations
on the checkpoint got populated and the checkpoint round-trips.

Usage:
    .venv/bin/python scripts/smoke_test_rolling.py [session_id]
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.checkpoint_manager import CheckpointManager
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile
from app.schemas.requests import TurnRequest


async def main(session_id: str) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    # Clone the session to a temp dir so the smoke test doesn't mutate real saves.
    real_save = Path("app/storage/saves") / session_id
    if not real_save.exists():
        print(f"ERROR: {real_save} does not exist", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir) / "saves"
        tmp_save = tmp_root / session_id
        shutil.copytree(real_save, tmp_save)

        # Load the latest pre-turn checkpoint for comparison — that's what the
        # orchestrator will resume from.
        before_path = sorted(tmp_save.glob("ckpt_*.json"))[-1]
        before_data = json.loads(before_path.read_text())
        before = CheckpointFile.model_validate(before_data)
        before_session_id = before.session.session_id
        print(f"Loaded {before_path.name} (session_id={before_session_id})")
        print(f"  session_conversation:    {len(before.session_conversation)} msgs")
        print(f"  narrator_conversation:   {len(before.narrator_conversation)} msgs")
        print(f"  character_conversations: {len(before.character_conversations)} chars")
        print()

        client = LLMClient(config=LLMConfig.from_env())
        ckpt_mgr = CheckpointManager(save_dir=str(tmp_root))
        prompt_mgr = PromptManager(prompts_dir="app/prompts")
        orch = Orchestrator(client, ckpt_mgr, prompt_mgr)

        inputs = [
            "I look around and take in my surroundings.",
            "I step further into the room and examine what I see.",
        ]

        try:
            for i, user_input in enumerate(inputs, 1):
                print(f"=== Turn {i}: {user_input!r} ===")
                response = await orch.process_turn(TurnRequest(
                    session_id=before_session_id,
                    user_input=user_input,
                    debug=True,
                ))
                print(f"  turn_index: {response.turn_index}  "
                      f"output: {len(response.output_text)} chars")

                if response.debug:
                    print(f"  {'phase':20s}{'ms':>8s}{'in':>8s}{'out':>8s}"
                          f"{'cache_read':>12s}{'cache_write':>14s}")
                    for lat in response.debug.latencies:
                        print(
                            f"  {lat.phase:20s}"
                            f"{lat.duration_ms:>8.0f}"
                            f"{lat.input_tokens:>8d}"
                            f"{lat.output_tokens:>8d}"
                            f"{lat.cache_read_input_tokens:>12d}"
                            f"{lat.cache_creation_input_tokens:>14d}"
                        )
                print()

            # After turn 2, cache_read should be nonzero in at least one phase
            # (the system prompt at minimum; ideally the prior conversation too).
            final_debug = response.debug
            cache_read_total = sum(
                lat.cache_read_input_tokens for lat in final_debug.latencies
            )
            cache_write_total = sum(
                lat.cache_creation_input_tokens for lat in final_debug.latencies
            )
            print(f"Turn 2 totals: cache_read={cache_read_total}, "
                  f"cache_write={cache_write_total}")
        finally:
            await client.close()

        # Final checkpoint sanity
        after_path = sorted(tmp_save.glob("ckpt_*.json"))[-1]
        after = CheckpointFile.model_validate_json(after_path.read_text())
        print()
        print(f"Final checkpoint: {after_path.name}")
        print(f"  session_conversation:    {len(after.session_conversation)} msgs")
        print(f"  narrator_conversation:   {len(after.narrator_conversation)} msgs")
        print(f"  character_conversations: {len(after.character_conversations)} chars")
        for cid, conv in after.character_conversations.items():
            print(f"    {cid}: {len(conv)} msgs")

        errors = []
        # Each of 2 turns appends a user+assistant pair.
        if len(after.session_conversation) != 4:
            errors.append(
                f"session_conversation should have 4 msgs after 2 turns, got "
                f"{len(after.session_conversation)}"
            )
        if after.session.turn_index != before.session.turn_index + 2:
            errors.append(
                f"turn_index should be +2, got {after.session.turn_index}"
            )
        if cache_read_total == 0:
            errors.append(
                "turn 2 cache_read_input_tokens was 0 across all phases — "
                "caching didn't hit. Check the dual-breakpoint placement."
            )

        if errors:
            print()
            print("FAIL:")
            for e in errors:
                print(f"  - {e}")
            return 2

        print()
        print("PASS: end-to-end 2-turn flow with cache hits on turn 2.")
        return 0


if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else "foreign_diplomat"
    raise SystemExit(asyncio.run(main(session_id)))
