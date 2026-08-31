"""Character agent engine -- generates in-character responses.

Each character carries a rolling conversation on the checkpoint
(`checkpoint.character_conversations[character_id]`). Every committed
response keeps only its observable prose (or the explicit `<silence/>` marker)
in that character's own assistant history. Presentation metadata is stripped
before the response is persisted or forwarded to any other role.

Foreground and private/background calls share the turn contract while their
frame and witnessed input remain in the user tail. Character identity and
sparse actor-owned facts are supplied in that current packet. Each character
keeps its own rolling history; character calls do not request provider-side
context compaction.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from app.engine import dnd_spatial
from app.engine.dnd_combat_access import (
    checkpoint_active_combat as _active_combat,
    combatant_character_id as _combatant_character_id,
    combatant_for_character as _combatant_for_character,
    combatants as _combatants,
    current_combatant as _current_combatant,
    obj_get as _obj_get,
)
from app.engine.context_builder import (
    build_character_perception_request_packet,
    build_character_turn_request_packet,
    build_dnd_character_identity_sentence,
    build_dnd_player_identities_block,
    clear_character_inbox,
    format_character_location_for_agent,
    format_elapsed_agent_turn_block,
    format_pending_observations_block,
)
from app.engine.character_presentation import (
    apply_character_presentation_choice,
    format_character_presentation_catalog,
    parse_character_presentation_footer,
)
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import (
    format_character_moment,
)
from app.llm.client import LLMClient
from app.schemas.agents import (
    CharacterAgentOutput,
    CharacterPerceptionOutput,
    CharacterPresentationChoice,
)
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    is_non_social_hazard,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage

logger = logging.getLogger(__name__)


DND5E_BASIC_RULESET_ID = "dnd5e_basic"
ONE_STAR_ASCENSION_RULESET_ID = "one_star_ascension"
PLOT_AGENT_ROLE = "agent"
STANDARD_AGENT_ROLE = "agent_standard"
CONVENIENCE_AGENT_ROLE = "agent_convenience"


@dataclass(frozen=True)
class CharacterAgentTurnDraft:
    """A completed agent call whose history effects are not committed yet."""

    output: CharacterAgentOutput
    user_message: ConversationMessage
    assistant_message: ConversationMessage


_SILENCE_MARKER = "<silence/>"
_SILENCE_TOKEN_RE = re.compile(
    r"<\s*/?\s*silence\b",
    re.IGNORECASE,
)


class CharacterAgentOutputError(ValueError):
    """A character response violated the observable turn contract."""


def _session_ruleset_id(checkpoint: CheckpointFile) -> str:
    settings = getattr(
        getattr(checkpoint.session, "config", None),
        "settings",
        None,
    )
    return str(getattr(settings, "ruleset_id", "") or "")


def model_role_for_character(character: CharacterRecord) -> str:
    if character.agent_tier == CharacterAgentTier.standard:
        return STANDARD_AGENT_ROLE
    if character.agent_tier == CharacterAgentTier.utility:
        return CONVENIENCE_AGENT_ROLE
    return PLOT_AGENT_ROLE


def _combat_state_line(combatant: Any) -> str:
    ac = _obj_get(combatant, "armor_class", 10)
    hp_current = _obj_get(combatant, "hit_points_current", 0)
    hp_max = _obj_get(combatant, "hit_points_max", 0)
    hp_temp = _obj_get(combatant, "hit_points_temporary", 0)
    defeat_state = str(_obj_get(combatant, "defeat_state", "") or "active")
    conditions = [
        str(condition)
        for condition in (_obj_get(combatant, "conditions", []) or [])
        if str(condition).strip()
    ]
    hp = f"{hp_current}/{hp_max}"
    if hp_temp:
        hp = f"{hp} (+{hp_temp} temp)"
    condition_text = ", ".join(conditions) if conditions else "none"
    effect_names = []
    for effect in _obj_get(combatant, "active_effects", []) or []:
        name = str(
            _obj_get(effect, "name", "")
            or _obj_get(effect, "slug", "")
        ).strip()
        if name:
            effect_names.append(name)
    effects_text = ", ".join(effect_names) if effect_names else "none"
    parts = [f"AC {ac}", f"HP {hp}"]
    if defeat_state != "active":
        parts.append(f"state {defeat_state}")
    parts.append(f"conditions: {condition_text}")
    parts.append(f"effects: {effects_text}")
    return f"Your combat state: {'; '.join(parts)}."


def _combat_action_lines(character: CharacterRecord) -> list[str]:
    mechanics_state = getattr(character, "mechanics", None) or {}
    statblock = (
        (mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {}
    )
    actions = statblock.get("actions") or []
    lines: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_id = str(action.get("id") or "").strip()
        name = str(action.get("name") or action_id or "Unnamed action").strip()
        attack = action.get("attack") or {}
        if not isinstance(attack, dict):
            attack = {}
        parts = [name]
        if action_id:
            parts.append(f"id {action_id}")
        bonus = attack.get("bonus", "")
        if bonus != "":
            parts.append(f"attack +{bonus}")
        damage = str(attack.get("damage") or action.get("damage") or "").strip()
        if damage:
            parts.append(f"damage {damage}")
        range_text = str(
            attack.get("range") or action.get("range") or ""
        ).strip()
        if range_text:
            parts.append(f"range {range_text}")
        notes = str(action.get("notes") or "").strip()
        if notes:
            parts.append(notes)
        lines.append("- " + "; ".join(parts))
    return lines


def _join_prompt_blocks(*blocks: str) -> str:
    return "\n\n".join(block.strip() for block in blocks if block.strip())


def _reject_retired_private_carry(text: str) -> None:
    """Reject the retired private marker before any response is committed."""
    if re.search(r"<\s*/?\s*private_carry\b", text or "", re.IGNORECASE):
        raise CharacterAgentOutputError(
            "retired <private_carry> marker is not supported"
        )


def _assistant_history_message(response: Any, text: str) -> ConversationMessage:
    """Preserve provider blocks while replacing all assistant text safely.

    Provider responses may contain non-text blocks (notably compaction
    blocks) that must round-trip in the actor's history. Text blocks are
    collapsed into the parsed public history text.
    """

    from app.engine.context_builder import assistant_message_from_response

    original = assistant_message_from_response(response)
    if not isinstance(original.content, list):
        return ConversationMessage(role="assistant", content=text)
    blocks: list[dict[str, Any]] = []
    inserted = False
    for raw_block in original.content:
        block = dict(raw_block)
        if block.get("type") == "text":
            if not inserted:
                blocks.append({"type": "text", "text": text})
                inserted = True
            continue
        blocks.append(block)
    if not inserted:
        blocks.append({"type": "text", "text": text})
    return ConversationMessage(role="assistant", content=blocks)


def _parse_agent_turn_response(
    text: str,
) -> tuple[str, bool, CharacterPresentationChoice]:
    """Parse one free-form turn into public prose, silence, and display.

    Empty and presentation-only responses are rejected because a turn cannot
    consume observations or advance actor time without observable prose or an
    explicit silence. The retired private marker is rejected before any of
    those values can cross a role boundary.
    """
    raw = text or ""
    _reject_retired_private_carry(raw)
    public_text, presentation = parse_character_presentation_footer(
        raw
    )
    public_text = public_text.strip()
    if public_text == _SILENCE_MARKER:
        return "", True, presentation
    if _SILENCE_TOKEN_RE.search(public_text):
        raise CharacterAgentOutputError(
            "<silence/> must be the exact observable turn body"
        )
    if not public_text:
        raise CharacterAgentOutputError(
            "character turn requires observable prose or explicit <silence/>"
        )
    return public_text, False, presentation


def _parse_agent_perception_response(
    text: str,
) -> tuple[str, CharacterPresentationChoice]:
    """Parse exterior prose without accepting any turn-only control marker."""
    raw = text or ""
    _reject_retired_private_carry(raw)
    if _SILENCE_TOKEN_RE.search(raw):
        raise CharacterAgentOutputError(
            "perception output cannot contain silence markers"
        )
    public_text, presentation = parse_character_presentation_footer(raw)
    public_text = public_text.strip()
    if not public_text:
        raise CharacterAgentOutputError(
            "perception output requires visible exterior prose"
        )
    return public_text, presentation


def sanitize_character_public_text(text: str) -> str:
    """Return only observable character prose for a public-facing synthesis."""
    public_text, is_silence, _presentation = _parse_agent_turn_response(text)
    return "" if is_silence else public_text


def _usage_with_prompt_render(
    *usage_dicts: dict[str, int],
    prompt_render_ms: float,
) -> dict[str, int | float]:
    """Combine provider usage for one logical agent call."""
    combined: dict[str, int | float] = {}
    for usage in usage_dicts:
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                combined[key] = combined.get(key, 0) + value
            else:
                # The current providers expose integer counters, but retain a
                # future non-numeric field rather than silently dropping it.
                combined[key] = value
    combined["prompt_render_ms"] = prompt_render_ms
    return combined


class CharacterAgent:
    """Generates in-character responses over a per-character rolling conversation."""

    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager
        # Usage from the most recent agent turn/perception call.
        self.last_usage: dict[str, int | float] = {}

    async def turn(
        self,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
        frame: str = "foreground",
        local_context: str = "",
    ) -> CharacterAgentOutput:
        """Committed agent turn in a router-selected frame."""
        draft = await self.draft_turn(
            character=character,
            checkpoint=checkpoint,
            frame=frame,
            local_context=local_context,
        )
        self._commit_draft(character, checkpoint, draft)
        return draft.output

    def _ruleset_guidance(self, checkpoint: CheckpointFile) -> str:
        ruleset_id = _session_ruleset_id(checkpoint)
        if ruleset_id == DND5E_BASIC_RULESET_ID:
            return self.prompt_manager.render("agent_ruleset_dnd5e").strip()
        if ruleset_id == ONE_STAR_ASCENSION_RULESET_ID:
            return self.prompt_manager.render(
                "agent_ruleset_one_star"
            ).strip()
        return ""

    @staticmethod
    def _dnd_character_identity_block(
        character: CharacterRecord,
        checkpoint: CheckpointFile,
    ) -> str:
        """Keep active D&D identity in the current adapter-tail packet."""

        if _session_ruleset_id(checkpoint) != DND5E_BASIC_RULESET_ID:
            return ""
        return build_dnd_character_identity_sentence(checkpoint, character).strip()

    @staticmethod
    def _one_star_agent_state_block(
        character: CharacterRecord,
        checkpoint: CheckpointFile,
    ) -> str:
        if _session_ruleset_id(checkpoint) != ONE_STAR_ASCENSION_RULESET_ID:
            return ""
        from app.engine.one_star_projection import one_star_agent_state_block

        return one_star_agent_state_block(checkpoint, character)

    @staticmethod
    def _one_star_perception_block(character: CharacterRecord) -> str:
        from app.engine.one_star_projection import (
            visible_equipped_item_description,
        )

        description = visible_equipped_item_description(character)
        if not description:
            return ""
        return "Visible equipped items:\n" + description

    def _dnd_combat_context(
        self,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
    ) -> str:
        if _session_ruleset_id(checkpoint) != DND5E_BASIC_RULESET_ID:
            return ""
        combat = _active_combat(checkpoint)
        if combat is None or not _combatants(combat):
            return ""
        current = _current_combatant(combat)
        current_id = _combatant_character_id(current) if current else ""
        own = _combatant_for_character(combat, character.character_id)

        lines = [
            "## D&D Combat",
            "Active D&D 5e initiative is running.",
            f"Round: {_obj_get(combat, 'round_number', 1)}.",
            f"Current turn: {current_id or '(unknown)'}.",
        ]
        if own is None:
            lines.append("You are not listed as a combatant.")
        else:
            if _combatant_character_id(own) == current_id:
                lines.append("It is your initiative turn.")
            else:
                lines.append("It is not your initiative turn.")
            lines.append(_combat_state_line(own))
            pending_action = str(
                _obj_get(own, "pending_initiating_action", "") or ""
            ).strip()
            if pending_action and _combatant_character_id(own) == current_id:
                lines.append(
                    "Before initiative, you declared this pending intent: "
                    f"{pending_action}"
                )
            action_lines = _combat_action_lines(character)
            if action_lines:
                lines.append("Available combat actions:")
                lines.extend(action_lines)
        map_lines = dnd_spatial.render_battle_map_summary(
            combat,
            actor_id=character.character_id,
        )
        if map_lines:
            lines.append("## Tactical Map")
            lines.extend(map_lines)
        return "\n".join(lines)

    async def perceive(
        self,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
    ) -> CharacterPerceptionOutput:
        """Observer-agnostic perception beat — return this character's
        current visual loadout as plain prose (1-3 sentences).

        Used by the observation-harvest fork in `turn_loop.run_beat`
        when a player's action is purely observational
        (`event_kind="observation_harvest"`), and reachable
        later from `/query` for "what does X look like?" questions.

        Distinct from committed agent turns in two load-bearing ways:

        1. **No turn-marker parse.** Perception mode in `agent_perception.txt`
           tells the model to emit only exterior prose plus its presentation
           footer. Returning `response.content.strip()` directly avoids
           interpreting perception output as a character turn.

        2. **Lower max_tokens.** Cap is 3 sentences (~150 tokens of
           prose, leave headroom for the model's own pacing). 600
           leaves slack but stays cheap.

        Perception DOES append to the character's rolling conversation.
        A character should remember how they chose to present themself in
        the current scene, especially after repeated look/query harvests.
        Unlike normal agent turns, this still does not drain pending_observations
        and accepts only exterior prose plus its presentation footer.

        """
        if is_non_social_hazard(character):
            raise ValueError(
                "Non-social hazards have no character-agent perception turn: "
                f"{character.character_id}"
            )

        history = checkpoint.character_conversations.get(character.character_id, [])

        # Deliberately DO NOT call `clear_character_inbox`: perception is
        # a side query, not an on-stage beat. The pending_observations
        # queue holds events the character hasn't yet reacted to in
        # fiction; the next normal agent turn is what consumes them.
        # Draining here would silently swallow off-location or mediated
        # perceptions the next on-stage turn needs to acknowledge. We
        # do not add those observations to this request — a self-presentation
        # query should not be primed by incoming events.
        # The character's freshest in-fiction interior is already in
        # their rolling history (`history` above), which is what should
        # color their visual loadout. The resulting loadout is appended
        # after the call so future beats remember what was established.
        one_star_perception = (
            self._one_star_perception_block(character)
            if _session_ruleset_id(checkpoint)
            == ONE_STAR_ASCENSION_RULESET_ID
            else ""
        )
        presentation_catalog = format_character_presentation_catalog(
            checkpoint,
            character,
        )
        request_packet = build_character_perception_request_packet(
            character,
            checkpoint,
            _join_prompt_blocks(
                one_star_perception,
                presentation_catalog,
            ),
        )

        render_t0 = time.monotonic()
        messages = self.prompt_manager.render_conversation(
            "agent_perception",
            history=history,
            request_packet=request_packet,
        )
        render_ms = (time.monotonic() - render_t0) * 1000
        user_content = messages[-1]["content"]

        logger.info(
            "Agent %s (%s) perceive: history=%d msgs",
            character.name, character.character_id, len(history),
        )

        response = await self.client.complete(
            role=model_role_for_character(character),
            messages=messages,
            temperature=0.5,
            max_tokens=600,
            cache=True,
            compact=False,
        )
        text, presentation = _parse_agent_perception_response(
            response.content or ""
        )
        result = CharacterPerceptionOutput(
            public_text=text,
            presentation=presentation,
        )
        self.last_usage = _usage_with_prompt_render(
            response.usage,
            prompt_render_ms=render_ms,
        )

        conv = checkpoint.character_conversations.setdefault(
            character.character_id, [],
        )
        conversation_user_content = user_content
        if presentation_catalog:
            conversation_user_content = conversation_user_content.replace(
                presentation_catalog,
                "",
                1,
            )
        conv.extend((
            ConversationMessage(
                role="user",
                content=conversation_user_content,
            ),
            _assistant_history_message(response, text),
        ))
        apply_character_presentation_choice(
            checkpoint,
            character,
            result.presentation,
        )

        logger.info(
            "Agent %s perceive: %d chars",
            character.name, len(text),
        )
        return result

    async def draft_turn(
        self,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
        frame: str = "foreground",
        local_context: str = "",
    ) -> CharacterAgentTurnDraft:
        """Prepare an agent turn without mutating agent memory."""
        if is_non_social_hazard(character):
            raise ValueError(
                "Non-social hazards have no character-agent intention turn: "
                f"{character.character_id}"
            )
        frame = (frame or "foreground").strip().lower()
        if frame not in {"foreground", "private", "background"}:
            frame = "foreground"
        location_label = format_character_location_for_agent(
            character.location,
            checkpoint,
        )
        foreground_block = (
            self._dnd_combat_context(character, checkpoint)
            if frame == "foreground"
            else ""
        )
        presentation_catalog = format_character_presentation_catalog(
            checkpoint,
            character,
        )

        return await self._draft_beat(
            character=character,
            checkpoint=checkpoint,
            current_moment=_join_prompt_blocks(
                format_character_moment(
                    frame=frame,
                    location=location_label,
                    local_context=local_context,
                ),
                foreground_block,
                presentation_catalog,
            ),
            presentation_catalog=presentation_catalog,
            log_label="turn",
            log_extra=frame,
        )

    def _commit_draft(
        self,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
        draft: CharacterAgentTurnDraft,
    ) -> None:
        clear_character_inbox(character)
        current_at_s = max(
            int(checkpoint.session.leading_at_s),
            int(character.clock_at_s),
        )
        previous = character.last_agent_turn_at_s
        character.last_agent_turn_at_s = max(
            previous if previous is not None else 0,
            current_at_s,
        )
        conv = checkpoint.character_conversations.setdefault(
            character.character_id, [],
        )
        conv.extend([draft.user_message, draft.assistant_message])
        apply_character_presentation_choice(
            checkpoint,
            character,
            draft.output.presentation,
        )

    def commit_draft(
        self,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
        draft: CharacterAgentTurnDraft,
    ) -> None:
        """Commit a previously completed draft after its event is accepted."""

        self._commit_draft(character, checkpoint, draft)

    async def _draft_beat(
        self,
        *,
        character: CharacterRecord,
        checkpoint: CheckpointFile,
        current_moment: str,
        presentation_catalog: str,
        log_label: str,
        log_extra: str,
    ) -> CharacterAgentTurnDraft:
        """Shared agent-beat plumbing for all committed/drafted turns.

        Renders the `agent_turn` template, calls the LLM, parses the
        observable response, and prepares the user/assistant history pair.
        All ordinary turn frames flow through here so future call behavior
        stays in one place.

        The generic system contract is shared across characters. This
        character's identity and current fictional circumstance live in the
        user packet, which is retained verbatim in history except for a
        disposable presentation catalog.

        """
        history = checkpoint.character_conversations.get(character.character_id, [])

        elapsed_time_block = format_elapsed_agent_turn_block(
            character, checkpoint,
        )
        one_star_state = self._one_star_agent_state_block(
            character, checkpoint
        )
        request_packet = build_character_turn_request_packet(
            character,
            checkpoint,
            _join_prompt_blocks(
                self._dnd_character_identity_block(character, checkpoint),
                build_dnd_player_identities_block(checkpoint),
                elapsed_time_block,
                format_pending_observations_block(character),
                current_moment,
                one_star_state,
            ),
        )

        render_t0 = time.monotonic()
        messages = self.prompt_manager.render_conversation(
            "agent_turn",
            history=history,
            ruleset_guidance=self._ruleset_guidance(checkpoint),
            request_packet=request_packet,
        )
        render_ms = (time.monotonic() - render_t0) * 1000

        user_content = messages[-1]["content"]

        logger.info(
            "Agent %s (%s) %s [%s]: history=%d msgs",
            character.name, character.character_id,
            log_label, log_extra, len(history),
        )

        role = model_role_for_character(character)
        response = await self.client.complete(
            role=role,
            messages=messages,
            temperature=0.6,
            max_tokens=2000,
            cache=True,
            compact=False,
        )
        final_response = response
        public_text, is_silence, presentation = (
            _parse_agent_turn_response(final_response.content)
        )
        result = CharacterAgentOutput(
            character_id=character.character_id,
            public_text=public_text,
            is_silence=is_silence,
            presentation=presentation,
        )
        self.last_usage = _usage_with_prompt_render(
            final_response.usage,
            prompt_render_ms=render_ms,
        )
        conversation_user_content = user_content
        if presentation_catalog:
            conversation_user_content = conversation_user_content.replace(
                presentation_catalog,
                "",
                1,
            )
        user_message = ConversationMessage(
            role="user",
            content=conversation_user_content,
        )
        assistant_history_text = _SILENCE_MARKER if is_silence else public_text
        assistant_message = _assistant_history_message(
            final_response,
            assistant_history_text,
        )

        logger.info(
            "Agent %s %s: %d chars public, silence=%s",
            character.name, log_label,
            len(result.public_text), is_silence,
        )

        return CharacterAgentTurnDraft(
            output=result,
            user_message=user_message,
            assistant_message=assistant_message,
        )
