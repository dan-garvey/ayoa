"""Full pipeline test: NP1 -> Discriminator -> Agents -> NP2.

This is the first end-to-end test. It chains all four LLM components
with a real checkpoint and prints the final narrative.

Usage:
    .venv/bin/python scripts/test_full_pipeline_manual.py [checkpoint_path]
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

TEST_ACTION = (
    "I walk into the common area and say: "
    "'Good evening. I understand we're all here because the Covenant thinks "
    "we matter. I'd like to think it's right.'"
)


async def run_pipeline(checkpoint_path: str):
    with open(checkpoint_path) as f:
        checkpoint = CheckpointFile(**json.load(f))

    # Move key characters to common area for a richer scene
    common_scene = "garvey_house_common"
    checkpoint.world_state.locations.current_scene_id = common_scene
    moved = []
    for char in checkpoint.characters:
        if char.character_id in (
            "ashara_vel_kothren", "rashid_vel_amara", "ysolde_thornmantle",
            "seraphel_dawnquill",
        ):
            char.location = common_scene
            moved.append(char.name)

    print(f"Scene: {common_scene}")
    print(f"Characters present: {', '.join(moved)}")
    print(f"Action: {TEST_ACTION}")

    config = LLMConfig.from_env()
    client = LLMClient(config=config)
    prompt_mgr = PromptManager("app/prompts")
    narrator = Narrator(client, prompt_mgr)
    discriminator = Discriminator(client, prompt_mgr)
    agent_engine = CharacterAgent(client, prompt_mgr)

    try:
        # === NP1: Adjudicate ===
        print(f"\n{'='*70}")
        print("PHASE 1: ADJUDICATION")
        print(f"{'='*70}")
        event = await narrator.phase_1(TEST_ACTION, checkpoint)
        print(f"Feasible: {event.world_adjudication.feasible}")
        print(f"Outcome: {event.world_adjudication.resolved_outcome}")
        print(f"Facts: {len(event.observable_facts)}")

        # === Discriminator ===
        print(f"\n{'='*70}")
        print("DISCRIMINATOR")
        print(f"{'='*70}")
        disc_output = await discriminator.run(event, checkpoint)
        responding = [o for o in disc_output.observers if o.should_respond]
        print(f"Observers: {len(disc_output.observers)}, Responding: {len(responding)}")
        for obs in disc_output.observers:
            name = next(
                (c.name for c in checkpoint.characters if c.character_id == obs.character_id),
                obs.character_id,
            )
            print(f"  {name}: {obs.observation_level}, respond={obs.should_respond}, facts={len(obs.facts)}")

        # === Agents (parallel) ===
        print(f"\n{'='*70}")
        print("CHARACTER AGENTS")
        print(f"{'='*70}")
        agent_tasks = []
        for obs in responding:
            char = next(
                (c for c in checkpoint.characters if c.character_id == obs.character_id),
                None,
            )
            if char:
                agent_tasks.append(agent_engine.respond(char, obs.facts, checkpoint))

        agent_outputs = []
        if agent_tasks:
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    print(f"  AGENT ERROR: {result}")
                else:
                    agent_outputs.append(result)
                    name = next(
                        (c.name for c in checkpoint.characters if c.character_id == result.character_id),
                        result.character_id,
                    )
                    print(f"\n  {name}:")
                    if result.public_response.dialogue:
                        for line in result.public_response.dialogue:
                            print(f'    Says: "{line}"')
                    if result.public_response.actions:
                        print(f"    Actions: {result.public_response.actions}")
                    if result.public_response.expression:
                        print(f"    Expression: {result.public_response.expression}")
        else:
            print("  No characters responding.")

        # === NP2: Compose ===
        print(f"\n{'='*70}")
        print("PHASE 2: FINAL COMPOSITION")
        print(f"{'='*70}")
        final = await narrator.phase_2(TEST_ACTION, event, agent_outputs, checkpoint)

        print(f"\n{final.final_text}")
        print(f"\n--- Turn Summary ---")
        print(f"{final.turn_summary}")

    finally:
        await client.close()


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    asyncio.run(run_pipeline(checkpoint_path))


if __name__ == "__main__":
    main()
