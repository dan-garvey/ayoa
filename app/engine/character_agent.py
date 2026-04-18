"""Character agent engine — generates in-character responses.

Each character carries a rolling conversation on the checkpoint
(`checkpoint.character_conversations[character_id]`). Every time the character
responds, we send the full prior conversation plus a fresh user message
describing this turn; the API sees verbatim history of everything the
character has said across the session.
"""

from __future__ import annotations

import logging

from app.engine.context_builder import (
    build_character_packet,
    build_character_state,
    build_characters_present,
    build_scene_context,
    build_world_context,
    format_observed_facts,
    format_pending_observations_block,
    format_prior_responses,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, serialize_assistant_content
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage

logger = logging.getLogger(__name__)


class CharacterAgent:
    """Generates in-character responses over a per-character rolling conversation."""

    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager
        # Usage from the most recent respond() call.
        self.last_usage: dict[str, int] = {}

    async def respond(
        self,
        character: CharacterRecord,
        observed_facts: list[str],
        checkpoint: CheckpointFile,
        prior_responses: list[CharacterAgentOutput] | None = None,
    ) -> CharacterAgentOutput:
        """Generate an in-character response and append it to the rolling conversation.

        Flushes `character.pending_observations` into the user message (and clears
        it), then appends both the user message and the assistant response to
        `checkpoint.character_conversations[character.character_id]`.
        """
        history = checkpoint.character_conversations.get(character.character_id, [])

        pending_block = format_pending_observations_block(character)
        # Clear pending now that we're flushing into this turn's user message.
        character.pending_observations = []

        char_identity = build_character_packet(character)
        char_state = build_character_state(character)

        messages = self.prompt_manager.render_conversation(
            "agent",
            history=history,
            **char_identity,
            **char_state,
            world_context=build_world_context(checkpoint),
            scene_context=build_scene_context(checkpoint),
            characters_present=build_characters_present(character, checkpoint),
            observed_facts=format_observed_facts(observed_facts),
            prior_character_responses=format_prior_responses(
                prior_responses or [], checkpoint
            ),
            pending_observations_block=pending_block,
        )

        # Capture the plain-text user content before LLMClient wraps it for caching.
        user_content = messages[-1]["content"]

        logger.info(
            "Agent %s (%s): generating response to %d facts (history=%d msgs)",
            character.name,
            character.character_id,
            len(observed_facts),
            len(history),
        )

        response = await self.client.complete(
            role="agent",
            messages=messages,
            response_model=CharacterAgentOutput,
            temperature=0.6,
            max_tokens=3000,
            cache=True,
            compact=True,
        )
        result: CharacterAgentOutput = response.parsed
        result.character_id = character.character_id
        self.last_usage = response.usage

        assistant_content = serialize_assistant_content(response.raw_response.content)
        new_history = list(history)
        new_history.append(ConversationMessage(role="user", content=user_content))
        new_history.append(
            ConversationMessage(role="assistant", content=assistant_content)
        )
        checkpoint.character_conversations[character.character_id] = new_history

        logger.info(
            "Agent %s: %d actions, %d dialogue lines",
            character.name,
            len(result.public_response.actions),
            len(result.public_response.dialogue),
        )

        return result
