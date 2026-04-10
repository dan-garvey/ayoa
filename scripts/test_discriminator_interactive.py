"""Test script that chains NP1 -> Discriminator against the live LLM.

Usage:
    .venv/bin/python scripts/test_discriminator_interactive.py [checkpoint_path]
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
from app.engine.discriminator import Discriminator
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_CHECKPOINT = "app/storage/saves/covenant_of_thrones/ckpt_0000.json"

TEST_ACTIONS = [
    # Quiet action in private room — only characters in same/adjacent location should observe
    "I quietly open the locked drawer under my desk.",
    # Loud action — adjacent characters might hear
    "I shout at the top of my lungs: 'I challenge Rashid vel Amara to a duel!'",
]


async def run_test(checkpoint_path: str):
    with open(checkpoint_path) as f:
        checkpoint = CheckpointFile(**json.load(f))

    config = LLMConfig.from_env()
    client = LLMClient(config=config)
    prompt_mgr = PromptManager("app/prompts")
    narrator = Narrator(client, prompt_mgr)
    discriminator = Discriminator(client, prompt_mgr)

    try:
        for i, action in enumerate(TEST_ACTIONS):
            print(f"\n{'='*70}")
            print(f"TEST {i+1}: {action}")
            print(f"{'='*70}")

            # Phase 1: Adjudicate
            event = await narrator.phase_1(action, checkpoint)
            print(f"\n--- NP1 Result ---")
            print(f"Feasible: {event.world_adjudication.feasible}")
            print(f"Outcome: {event.world_adjudication.resolved_outcome}")
            print(f"Observable facts:")
            for j, fact in enumerate(event.observable_facts, 1):
                print(f"  {j}. {fact}")

            # Discriminator: Who observes?
            disc_output = await discriminator.run(event, checkpoint)
            print(f"\n--- Discriminator Result ---")
            print(f"Observers: {len(disc_output.observers)}")
            for obs in disc_output.observers:
                char_name = next(
                    (c.name for c in checkpoint.characters if c.character_id == obs.character_id),
                    obs.character_id,
                )
                print(f"\n  {char_name} ({obs.character_id}):")
                print(f"    Level: {obs.observation_level}")
                print(f"    Should respond: {obs.should_respond}")
                print(f"    Perceived facts:")
                for k, fact in enumerate(obs.facts, 1):
                    print(f"      {k}. {fact}")

            if disc_output.spawn:
                print(f"\n  Spawn requests:")
                for sp in disc_output.spawn:
                    print(f"    - {sp.character_id}: {sp.seed}")
            if disc_output.dormant:
                print(f"\n  Dormant: {disc_output.dormant}")

    finally:
        await client.close()


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    asyncio.run(run_test(checkpoint_path))


if __name__ == "__main__":
    main()
