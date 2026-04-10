"""Interactive test for Narrator Phase 1.

Loads a checkpoint and lets you type player actions to see NP1 adjudication.

Usage:
    .venv/bin/python scripts/test_narrator_interactive.py [checkpoint_path]

If no checkpoint is given, uses the Covenant of Thrones import.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from app.engine.narrator import Narrator
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_CHECKPOINT = "app/storage/saves/covenant_of_thrones/ckpt_0000.json"


async def run_interactive(checkpoint_path: str):
    """Run an interactive NP1 testing session."""
    with open(checkpoint_path) as f:
        checkpoint = CheckpointFile(**json.load(f))

    logger.info(f"Loaded checkpoint: {checkpoint.session.session_id}")
    logger.info(f"  Setting: {checkpoint.world_state.setting.genre} / {checkpoint.world_state.setting.tone}")
    logger.info(f"  Scene: {checkpoint.world_state.locations.current_scene_id}")
    logger.info(f"  Characters: {len(checkpoint.characters)}")
    logger.info("")

    config = LLMConfig.from_env()
    client = LLMClient(config=config)
    prompt_mgr = PromptManager("app/prompts")
    narrator = Narrator(client, prompt_mgr)

    try:
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            try:
                event = await narrator.phase_1(user_input, checkpoint)

                print(f"\n{'='*60}")
                print(f"Intent:    {event.user_intent}")
                print(f"Feasible:  {event.world_adjudication.feasible}")
                print(f"Action:    {event.world_adjudication.attempted_action}")
                print(f"Outcome:   {event.world_adjudication.resolved_outcome}")
                print(f"Time:      {event.scene_delta.time_advanced_seconds}s")
                print(f"\nObservable facts:")
                for i, fact in enumerate(event.observable_facts, 1):
                    print(f"  {i}. {fact}")
                print(f"{'='*60}")

            except Exception as e:
                logger.error(f"NP1 failed: {e}")

    finally:
        await client.close()


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Run the story importer first: .venv/bin/python scripts/import_story.py stories/covenant_of_thrones.txt")
        sys.exit(1)

    asyncio.run(run_interactive(checkpoint_path))


if __name__ == "__main__":
    main()
