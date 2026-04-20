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
    append_turn_to_conversation,
    build_character_packet,
    build_character_state,
    build_characters_present,
    build_player_characters_block,
    build_scene_context,
    build_world_context,
    clear_character_inbox,
    format_observed_facts,
    format_pending_observations_block,
    format_prior_responses,
    resolve_acting_character,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile

logger = logging.getLogger(__name__)


_SCENE_CREATION_BLOCK_ENABLED = """\
6. **Create a new scene if the fiction genuinely needs one.** You may populate `private_updates.scenes_created` with spaces the scene graph doesn't yet contain — a private study you retreat to, a courier route between courts, a room in your household. Each entry needs a snake_case `scene_id`, a `name`, a short sensory `description`, and `connected_to` listing scene_ids this new space is reachable from (existing scenes or other newly-created ones). Reverse edges are added automatically. If you create a scene you want to move into, set `moved_to` to its id on the same tick. Use this sparingly — only when your off-stage action genuinely requires a space that doesn't exist yet. Empty `scenes_created` is the common case."""

_SCENE_CREATION_BLOCK_DISABLED = """\
6. **Do not invent new scenes.** `scenes_created` must be empty. If your planning requires a space that isn't in the graph, adjust your plan to use existing scenes; the router (not you) is responsible for growing world topology. If you genuinely need a new space, the narrator will pick it up from your objectives and route it through the router on a later turn."""


def _build_scene_creation_block(checkpoint: CheckpointFile) -> str:
    """Render the scene-creation instruction block for an agent tick
    prompt. Gated on the session's `agents_can_create_scenes` setting.
    Default is disabled (the router owns world topology in the baseline
    pipeline); operators flip the flag via /settings to experiment with
    more emergent world-building."""
    if checkpoint.session.config.settings.agents_can_create_scenes:
        return _SCENE_CREATION_BLOCK_ENABLED
    return _SCENE_CREATION_BLOCK_DISABLED


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
        acting_character_id: str = "",
    ) -> CharacterAgentOutput:
        """Generate an in-character response and append it to the rolling conversation.

        Flushes `character.pending_observations` into the user message (and clears
        it), then appends both the user message and the assistant response to
        `checkpoint.character_conversations[character.character_id]`.
        """
        history = checkpoint.character_conversations.get(character.character_id, [])

        pending_block = format_pending_observations_block(character)
        # Clear the pending queues now that we're flushing into this turn's
        # user message. Both observations and directives are one-shot — once
        # delivered to the agent, they belong to its rolling conversation.
        clear_character_inbox(character)

        char_identity = build_character_packet(character)
        char_state = build_character_state(character)

        acting_id, _, acting_name = resolve_acting_character(
            checkpoint, acting_character_id,
        )

        messages = self.prompt_manager.render_conversation(
            "agent",
            history=history,
            **char_identity,
            **char_state,
            world_context=build_world_context(character, checkpoint),
            scene_context=build_scene_context(checkpoint),
            characters_present=build_characters_present(character, checkpoint),
            observed_facts=format_observed_facts(observed_facts),
            prior_character_responses=format_prior_responses(
                prior_responses or [], checkpoint
            ),
            pending_observations_block=pending_block,
            acting_character_name=acting_name,
            player_characters_block=build_player_characters_block(
                checkpoint, acting_id
            ),
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

        conv = checkpoint.character_conversations.setdefault(
            character.character_id, [],
        )
        append_turn_to_conversation(conv, user_content, response)

        logger.info(
            "Agent %s: %d actions, %d dialogue lines",
            character.name,
            len(result.public_response.actions),
            len(result.public_response.dialogue),
        )

        return result

    async def tick(
        self,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
        acting_character_id: str = "",
    ) -> CharacterAgentOutput:
        """Off-stage tick: character advances their objectives without being in
        a scene with the player. Appended to the same rolling conversation as
        regular responses so continuity holds across ticks and responses.

        Uses the `agent_tick` prompt variant — same character identity and
        same Player Characters block (so caching still lines up), but the
        user message tells the agent they're off-stage and to produce only
        private updates (objectives / directives / movement), no public
        response prose.
        """
        history = checkpoint.character_conversations.get(character.character_id, [])

        pending_block = format_pending_observations_block(character)
        clear_character_inbox(character)

        char_identity = build_character_packet(character)
        char_state = build_character_state(character)

        acting_id, _, acting_name = resolve_acting_character(
            checkpoint, acting_character_id,
        )

        # Scene context for the tick is the character's OWN location, not
        # the active player scene — the character is off-stage, reasoning
        # from wherever they actually are.
        own_scene_id = character.location or ""
        own_scene = checkpoint.world_state.locations.scene_graph.get(own_scene_id, {})
        if own_scene_id and isinstance(own_scene, dict):
            scene_ctx = (
                f"Location: {own_scene.get('name', own_scene_id)} (id: {own_scene_id})\n"
                f"{own_scene.get('description', '') or ''}"
            ).strip()
        else:
            scene_ctx = "Off-screen / unspecified location."

        # Gate agent-side scene creation on a per-session setting. The
        # block is rendered either as encouragement + mechanics when the
        # flag is on, or an explicit ban when off — keeps the prompt
        # clear about what the agent is or isn't allowed to do.
        scene_creation_block = _build_scene_creation_block(checkpoint)

        messages = self.prompt_manager.render_conversation(
            "agent_tick",
            history=history,
            **char_identity,
            **char_state,
            world_context=build_world_context(character, checkpoint),
            scene_context=scene_ctx,
            pending_observations_block=pending_block,
            acting_character_name=acting_name,
            player_characters_block=build_player_characters_block(
                checkpoint, acting_id,
            ),
            scene_creation_block=scene_creation_block,
        )

        user_content = messages[-1]["content"]

        logger.info(
            "Agent %s (%s): off-stage tick (history=%d msgs)",
            character.name, character.character_id, len(history),
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

        conv = checkpoint.character_conversations.setdefault(
            character.character_id, [],
        )
        append_turn_to_conversation(conv, user_content, response)

        logger.info(
            "Agent %s tick: %d objectives, %d directives, moved_to=%r",
            character.name,
            len(result.private_updates.current_objectives),
            len(result.private_updates.directives_sent),
            result.private_updates.moved_to,
        )

        return result
