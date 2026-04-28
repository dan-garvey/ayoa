"""Character agent engine — generates in-character responses.

Each character carries a rolling conversation on the checkpoint
(`checkpoint.character_conversations[character_id]`). Every response and
tick is appended verbatim — including the trailing parenthetical — so
the agent's own future self sees its prior interior. Cross-agent /
narrator chokepoints strip the parenthetical via `_extract_parenthetical`
before its public_text is forwarded.

Cache lineage (v11): on-stage and off-stage calls share a SINGLE
unified system prompt (`agent_v*.txt`) and a single rolling history
per character. The mode distinction is signaled by a first-token
header in the user message — `## ON-STAGE` vs `## TICK`, defined as
`AGENT_ON_STAGE_HEADER` / `AGENT_TICK_HEADER` in
`turn_loop_contracts`. Switching between respond and tick within the
same character does NOT invalidate the system-prompt cache; cache
hits compound across both modes. The mode-specific user-message body
is assembled by the matching `format_agent_*_body` helper.
"""

from __future__ import annotations

import logging
import time

from app.engine.context_builder import (
    append_turn_to_conversation,
    build_character_packet,
    build_character_state,
    build_world_context,
    clear_character_inbox,
    format_pending_observations_block,
)
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import (
    AGENT_ON_STAGE_HEADER,
    AGENT_PERCEPTION_HEADER,
    AGENT_TICK_HEADER,
    format_agent_on_stage_body,
    format_agent_perception_body,
    format_agent_tick_body,
)
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
    logs a warning. Routing still works because `public_text` is just
    the original prose; the parenthetical is preserved verbatim in the
    rolling conversation history (the agent's own future-self memory),
    so a missed parse only loses the stripped-vs-prose distinction
    for one downstream hop.

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
        checkpoint: CheckpointFile,
        acting_character_id: str = "",
    ) -> CharacterAgentOutput:
        """On-stage agent beat — character is contextually present for the beat.

        The on-stage body (`format_agent_on_stage_body()`) is empty in
        v11-r10: the agent reads everything they need this turn through
        their `pending_observations` inbox (the block rendered above the
        body). The cascade dispatch path used to also pipe in
        `observed_facts` and `prior_responses` here, but production
        always passed `[]` / `None` because perception and cascade fan-out
        both land on the inbox via `broadcast_event` (which pushes each
        observer's visible observable_facts onto their inbox when local
        or explicitly named by a mediated fact), so the parameters were dead
        weight. They've been removed from the signature.

        See `_run_beat` for the shared plumbing.
        """
        return await self._run_beat(
            character=character,
            checkpoint=checkpoint,
            acting_character_id=acting_character_id,
            mode_header=AGENT_ON_STAGE_HEADER,
            mode_block=format_agent_on_stage_body(),
            log_label="respond",
            log_extra="on-stage",
        )

    async def perceive(
        self,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
        acting_character_id: str = "",
    ) -> str:
        """Observer-agnostic perception beat — return this character's
        current visual loadout as plain prose (1-3 sentences).

        Used by the observation-harvest fork in `turn_loop.run_beat`
        when a player's action is purely observational
        (`ends_beat_reason="observation_harvest"`), and reachable
        later from `/query` for "what does X look like?" questions.

        Distinct from `respond` and `tick` in three load-bearing ways:

        1. **No history append.** The exchange is NOT persisted onto
           `checkpoint.character_conversations[character_id]`. Perception
           is meta — folding it in would pollute the agent's continuity-
           of-self with non-fictional self-description ("the world asked
           me what I look like; I told them"). The on-stage / tick path
           reads its own past responses to maintain voice consistency;
           perception responses being interleaved would surface as
           confusing cross-talk on the next on-stage turn. Cache lineage
           is preserved because the system prompt is byte-identical with
           respond/tick — only the user message differs, and Anthropic's
           cache hashes the prefix, so perception calls cache-hit on
           the system prompt without ever appending to the message
           prefix the next on-stage call sees.

        2. **No parenthetical parse.** Perception mode in `agent.txt`
           tells the model not to emit a trailing parenthetical;
           output is pure prose. Returning `response.content.strip()`
           directly skips the `_extract_parenthetical` round-trip
           (which would otherwise log a spurious "missing trailing
           parenthetical" warning on every call).

        3. **Lower max_tokens.** Cap is 3 sentences (~150 tokens of
           prose, leave headroom for the model's own pacing). 600
           leaves slack but stays cheap.

        `acting_character_id` is currently vestigial — the agent
        prompt no longer surfaces who is acting this beat (that
        framing primed the model toward sycophantic
        protagonist-deference). The parameter is kept on the
        signature for callers (perception harvest, /query) but
        nothing inside this method consumes it. Pass empty string
        when called outside a beat.
        """
        history = checkpoint.character_conversations.get(character.character_id, [])

        # Deliberately DO NOT call `clear_character_inbox`: perception is
        # a side query, not an on-stage beat. The pending_observations
        # queue holds events the character hasn't yet reacted to in
        # fiction; the next `respond`/`tick` is what consumes them.
        # Draining here would silently swallow off-location or mediated
        # perceptions the next on-stage turn needs to acknowledge. We
        # also pass an EMPTY
        # pending-observations block to the render — a self-presentation
        # query shouldn't be primed by "react to these incoming events."
        # The character's freshest in-fiction interior is already in
        # their rolling history (`history` above), which is what should
        # color their visual loadout.
        char_identity = build_character_packet(character)
        char_state = build_character_state(character)

        render_t0 = time.monotonic()
        messages = self.prompt_manager.render_conversation(
            "agent",
            history=history,
            **char_identity,
            **char_state,
            world_context=build_world_context(character, checkpoint),
            pending_observations_block="",
            mode_header=AGENT_PERCEPTION_HEADER,
            mode_block=format_agent_perception_body(),
        )
        render_ms = (time.monotonic() - render_t0) * 1000

        logger.info(
            "Agent %s (%s) perceive: history=%d msgs",
            character.name, character.character_id, len(history),
        )

        response = await self.client.complete(
            role="agent",
            messages=messages,
            temperature=0.5,
            max_tokens=600,
            cache=True,
            compact=True,
        )
        text = (response.content or "").strip()
        self.last_usage = {**response.usage, "prompt_render_ms": render_ms}

        logger.info(
            "Agent %s perceive: %d chars",
            character.name, len(text),
        )
        return text

    async def tick(
        self,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
        acting_character_id: str = "",
    ) -> CharacterAgentOutput:
        """Off-stage tick — character is not contextually present with the player.

        They get one short beat in their own location to advance an
        objective. Same unified system prompt as `respond`; the
        `## TICK` first-token header in the user message flips the
        agent into Tick Mode. Appended to the same rolling
        conversation as on-stage responses so continuity holds across
        ticks and responses (one history per character, one cache
        lineage per character).
        """
        location_context = (
            f"Location: {character.location}"
            if character.location else "Off-screen / unspecified location."
        )

        return await self._run_beat(
            character=character,
            checkpoint=checkpoint,
            acting_character_id=acting_character_id,
            mode_header=AGENT_TICK_HEADER,
            mode_block=format_agent_tick_body(location_context=location_context),
            log_label="tick",
            log_extra="off-stage",
        )

    async def _run_beat(
        self,
        *,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
        acting_character_id: str,
        mode_header: str,
        mode_block: str,
        log_label: str,
        log_extra: str,
    ) -> CharacterAgentOutput:
        """Shared agent-beat plumbing for both `respond` and `tick`.

        Renders the unified `agent` template, calls the LLM, parses
        the trailing parenthetical, and persists the user/assistant
        pair onto the rolling conversation. Both modes flow through
        here so any future tweak (model swap, retry policy,
        compaction, telemetry) lands once.

        The system prefix is byte-identical between modes when called
        with the same character + checkpoint — same template, same
        character-derived variables. Only the user message changes:
        `mode_header` (`## ON-STAGE` or `## TICK`) is the first token,
        and `mode_block` is the mode-specific body. This shared
        prefix is what lets one cache lineage cover both modes for
        the same character.

        `acting_character_id` is currently vestigial — same reasoning
        as `perceive` above. Kept on the signature so callers don't
        have to special-case the boundary; nothing here consumes it.
        """
        history = checkpoint.character_conversations.get(character.character_id, [])

        pending_block = format_pending_observations_block(character)
        clear_character_inbox(character)

        char_identity = build_character_packet(character)
        char_state = build_character_state(character)

        render_t0 = time.monotonic()
        messages = self.prompt_manager.render_conversation(
            "agent",
            history=history,
            **char_identity,
            **char_state,
            world_context=build_world_context(character, checkpoint),
            pending_observations_block=pending_block,
            mode_header=mode_header,
            mode_block=mode_block,
        )
        render_ms = (time.monotonic() - render_t0) * 1000

        user_content = messages[-1]["content"]

        logger.info(
            "Agent %s (%s) %s [%s]: history=%d msgs",
            character.name, character.character_id,
            log_label, log_extra, len(history),
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

        logger.info(
            "Agent %s %s: %d chars public, %d chars intent",
            character.name, log_label,
            len(result.public_text), len(result.intent),
        )

        return result
