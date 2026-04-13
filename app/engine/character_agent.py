"""Character agent engine — generates in-character responses.

Takes a character record, observed facts, and scene context,
returns a CharacterAgentOutput with dialogue, actions, and private updates.
"""

from __future__ import annotations

import logging

from app.engine.context_builder import (
    build_character_packet,
    build_scene_context,
    build_world_context,
    build_recent_transcript,
    format_observed_facts,
    build_characters_present,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile

logger = logging.getLogger(__name__)


class CharacterAgent:
    """Generates in-character responses for a single character."""

    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager

    async def respond(
        self,
        character: CharacterRecord,
        observed_facts: list[str],
        checkpoint: CheckpointFile,
    ) -> CharacterAgentOutput:
        """Generate an in-character response based on observed facts.

        Args:
            character: The character record for this agent.
            observed_facts: Facts this character perceived (from discriminator).
            checkpoint: Current game state for scene/world context.

        Returns:
            CharacterAgentOutput with public response and private updates.
        """
        # Build prompt context
        char_vars = build_character_packet(character)
        scene = build_scene_context(checkpoint)
        world = build_world_context(checkpoint)
        transcript = build_recent_transcript(checkpoint)
        facts_str = format_observed_facts(observed_facts)
        characters_present = build_characters_present(character, checkpoint)

        prompt = self.prompt_manager.render(
            "agent",
            **char_vars,
            scene_context=scene,
            world_context=world,
            recent_transcript=transcript,
            observed_facts=facts_str,
            characters_present=characters_present,
        )

        logger.info(
            "Agent %s (%s): generating response to %d facts",
            character.name,
            character.character_id,
            len(observed_facts),
        )

        response = await self.client.complete(
            role="agent",
            messages=[{"role": "user", "content": prompt}],
            response_model=CharacterAgentOutput,
            temperature=0.6,
            max_tokens=3000,
        )
        result: CharacterAgentOutput = response.parsed

        # Ensure character_id matches
        result.character_id = character.character_id

        logger.info(
            "Agent %s: %d actions, %d dialogue lines, %d memory writes",
            character.name,
            len(result.public_response.actions),
            len(result.public_response.dialogue),
            len(result.memory_writes),
        )

        return result
