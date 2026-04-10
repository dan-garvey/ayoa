"""Test script that chains NP1 -> Discriminator -> Character Agents against live LLM.

Uses a modified checkpoint where characters are placed in the same room
to ensure the discriminator selects them as observers.

Usage:
    .venv/bin/python scripts/test_agent_interactive.py [checkpoint_path]
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
from app.engine.character_agent import CharacterAgent
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_CHECKPOINT = "app/storage/saves/covenant_of_thrones/ckpt_0000.json"


async def run_test(checkpoint_path: str):
    with open(checkpoint_path) as f:
        checkpoint = CheckpointFile(**json.load(f))

    # Move Ashara and Rashid into the player's scene for testing
    scene_id = checkpoint.world_state.locations.current_scene_id
    chars_moved = []
    for char in checkpoint.characters:
        if char.character_id in ("ashara_vel_kothren", "rashid_vel_amara"):
            char.location = scene_id
            chars_moved.append(char.name)

    print(f"Moved {', '.join(chars_moved)} to {scene_id} for testing")

    config = LLMConfig.from_env()
    client = LLMClient(config=config)
    prompt_mgr = PromptManager("app/prompts")
    narrator = Narrator(client, prompt_mgr)
    discriminator = Discriminator(client, prompt_mgr)
    agent = CharacterAgent(client, prompt_mgr)

    action = "I introduce myself to the room: 'I am the heir of House Garvey. I know some of you have opinions about that.'"

    try:
        print(f"\n{'='*70}")
        print(f"ACTION: {action}")
        print(f"{'='*70}")

        # NP1
        event = await narrator.phase_1(action, checkpoint)
        print(f"\n--- NP1 ---")
        print(f"Feasible: {event.world_adjudication.feasible}")
        print(f"Outcome: {event.world_adjudication.resolved_outcome}")

        # Discriminator
        disc_output = await discriminator.run(event, checkpoint)
        print(f"\n--- Discriminator ---")
        responding = [o for o in disc_output.observers if o.should_respond]
        print(f"Observers: {len(disc_output.observers)}, Responding: {len(responding)}")

        # Run responding agents in parallel
        if responding:
            tasks = []
            for obs in responding:
                char = next(
                    (c for c in checkpoint.characters if c.character_id == obs.character_id),
                    None,
                )
                if char:
                    tasks.append(agent.respond(char, obs.facts, checkpoint))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            print(f"\n--- Agent Responses ---")
            for result in results:
                if isinstance(result, Exception):
                    print(f"  ERROR: {result}")
                    continue
                char = next(
                    (c for c in checkpoint.characters if c.character_id == result.character_id),
                    None,
                )
                name = char.name if char else result.character_id
                print(f"\n  {name}:")
                if result.public_response.actions:
                    print(f"    Actions: {result.public_response.actions}")
                if result.public_response.dialogue:
                    for line in result.public_response.dialogue:
                        print(f'    Says: "{line}"')
                if result.public_response.expression:
                    print(f"    Expression: {result.public_response.expression}")
                if result.private_updates.attitude_delta:
                    print(f"    Attitude shifts: {result.private_updates.attitude_delta}")
                if result.memory_writes:
                    print(f"    Remembers: {result.memory_writes}")
        else:
            print("  No characters responding.")

    finally:
        await client.close()


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    asyncio.run(run_test(checkpoint_path))


if __name__ == "__main__":
    main()
