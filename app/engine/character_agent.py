"""Character agent engine — generates in-character responses.

Each character carries a rolling conversation on the checkpoint
(`checkpoint.character_conversations[character_id]`). Every response and
tick is appended verbatim — including the trailing parenthetical — so
the agent's own future self sees its prior interior. Cross-agent /
narrator chokepoints strip the parenthetical (see `_extract_parenthetical`
and `format_prior_responses` in context_builder).
"""

from __future__ import annotations

import logging
import time

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


def _extract_parenthetical(text: str) -> tuple[str, str]:
    """Split agent prose output into `(public_text, intent)`.

    The agent prompt instructs the model to end every response with a
    single trailing parenthetical containing internal intent. We extract
    the LAST balanced parenthetical group at the end of the text, after
    trimming trailing whitespace.

    On missing or malformed trailing paren, returns `(text, "")` and
    logs a warning. The empty intent then short-circuits the engine's
    `last_intent` writeback (Commit 3); routing still works because
    `public_text` is just the original prose.

    Mid-prose parentheticals (stage directions like "she pauses (just
    long enough to be noticed)") are preserved in `public_text` —
    only the FINAL group at the very end of the trimmed text counts
    as intent.
    """
    if not text:
        return "", ""
    stripped = text.rstrip()
    if not stripped or not stripped.endswith(")"):
        logger.warning(
            "Agent output missing trailing parenthetical — last 80 chars: %r",
            stripped[-80:],
        )
        return text, ""

    depth = 0
    open_idx = -1
    for i in range(len(stripped) - 1, -1, -1):
        ch = stripped[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                open_idx = i
                break

    if open_idx == -1:
        logger.warning(
            "Agent output ends with ')' but parens are unbalanced — "
            "last 80 chars: %r",
            stripped[-80:],
        )
        return text, ""

    public_text = stripped[:open_idx].rstrip()
    intent = stripped[open_idx + 1 : -1].strip()
    return public_text, intent


class CharacterAgent:
    """Generates in-character responses over a per-character rolling conversation."""

    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager
        # Usage from the most recent respond() / tick() call.
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
        `checkpoint.character_conversations[character.character_id]`. The full
        assistant text (prose + trailing parenthetical) is what's persisted —
        the parenthetical strip happens at engine consumption sites only.
        """
        history = checkpoint.character_conversations.get(character.character_id, [])

        pending_block = format_pending_observations_block(character)
        clear_character_inbox(character)

        char_identity = build_character_packet(character)
        char_state = build_character_state(character)

        acting_id, _, acting_name = resolve_acting_character(
            checkpoint, acting_character_id,
        )

        render_t0 = time.monotonic()
        messages = self.prompt_manager.render_conversation(
            "agent",
            history=history,
            **char_identity,
            **char_state,
            world_context=build_world_context(character, checkpoint),
            scene_context=build_scene_context(checkpoint, character.character_id),
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
        render_ms = (time.monotonic() - render_t0) * 1000

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
            temperature=0.6,
            max_tokens=2000,
            cache=True,
            compact=True,
        )
        public_text, intent = _extract_parenthetical(response.content)
        result = CharacterAgentOutput(
            character_id=character.character_id,
            public_text=public_text,
            intent=intent,
        )
        self.last_usage = {**response.usage, "prompt_render_ms": render_ms}

        conv = checkpoint.character_conversations.setdefault(
            character.character_id, [],
        )
        append_turn_to_conversation(conv, user_content, response)

        if intent:
            character.last_intent = intent
            character.last_intent_turn = checkpoint.session.turn_index

        logger.info(
            "Agent %s: %d chars public, %d chars intent",
            character.name,
            len(result.public_text),
            len(result.intent),
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
        user message tells the agent they're off-stage and to produce one
        tight beat ending with a parenthetical of intent.
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

        render_t0 = time.monotonic()
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
        )
        render_ms = (time.monotonic() - render_t0) * 1000

        user_content = messages[-1]["content"]

        logger.info(
            "Agent %s (%s): off-stage tick (history=%d msgs)",
            character.name, character.character_id, len(history),
        )

        response = await self.client.complete(
            role="agent",
            messages=messages,
            temperature=0.6,
            max_tokens=2000,
            cache=True,
            compact=True,
        )
        public_text, intent = _extract_parenthetical(response.content)
        result = CharacterAgentOutput(
            character_id=character.character_id,
            public_text=public_text,
            intent=intent,
        )
        self.last_usage = {**response.usage, "prompt_render_ms": render_ms}

        conv = checkpoint.character_conversations.setdefault(
            character.character_id, [],
        )
        append_turn_to_conversation(conv, user_content, response)

        if intent:
            character.last_intent = intent
            character.last_intent_turn = checkpoint.session.turn_index

        logger.info(
            "Agent %s tick: %d chars public, %d chars intent",
            character.name, len(result.public_text), len(result.intent),
        )

        return result
