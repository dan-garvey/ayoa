"""Non-interactive batch test for Narrator Phase 1 against live LLM.

Tests a set of actions with varying feasibility levels.

Usage:
    .venv/bin/python scripts/test_narrator_batch.py [checkpoint_path]
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
from scripts.checkpoint_utils import resolve_checkpoint_path

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TEST_ACTIONS = [
    # Feasible: simple social action
    "I look around the room and take stock of my surroundings.",
    # Infeasible: impossible physical feat
    "I punch through the stone wall with my bare fist.",
    # Partially feasible: social action with consequences
    "I publicly announce that Article Nineteen should be repealed.",
]


async def run_batch(checkpoint_path: str):
    with open(checkpoint_path) as f:
        checkpoint = CheckpointFile(**json.load(f))

    config = LLMConfig.from_env()
    client = LLMClient(config=config)
    prompt_mgr = PromptManager("app/prompts")
    narrator = Narrator(client, prompt_mgr)

    try:
        for i, action in enumerate(TEST_ACTIONS):
            print(f"\n{'='*70}")
            print(f"TEST {i+1}: {action}")
            print(f"{'='*70}")

            event = await narrator.phase_1(action, checkpoint)

            print(f"Intent:    {event.user_intent}")
            print(f"Feasible:  {event.world_adjudication.feasible}")
            print(f"Action:    {event.world_adjudication.attempted_action}")
            print(f"Outcome:   {event.world_adjudication.resolved_outcome}")
            print(f"Time:      {event.scene_delta.time_advanced_seconds}s")
            print(f"Facts:     {len(event.observable_facts)}")
            for j, fact in enumerate(event.observable_facts, 1):
                print(f"  {j}. {fact}")

    finally:
        await client.close()


def main():
    checkpoint_path = resolve_checkpoint_path(
        sys.argv[1] if len(sys.argv) > 1 else None
    )
    asyncio.run(run_batch(checkpoint_path))


if __name__ == "__main__":
    main()
