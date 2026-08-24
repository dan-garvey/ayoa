"""Character agent engine -- generates in-character responses.

Each character carries a rolling conversation on the checkpoint
(`checkpoint.character_conversations[character_id]`). Every committed
response is appended verbatim -- including the trailing
parenthetical — so the agent's own future self sees its prior interior.
Cross-agent / narrator chokepoints strip the parenthetical via
`_extract_parenthetical` before its public_text is forwarded.

Cache lineage (v11): foreground and private/background calls share a
single unified system prompt (`agent.txt`). Character identity/current
state and the turn frame live in the user tail so characters on the same
model role can share the cached system prefix. Each character still
keeps its own rolling history.
"""

from __future__ import annotations

import logging
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
    append_turn_to_conversation,
    build_character_packet,
    build_dnd_player_identities_block,
    build_character_state,
    build_world_context,
    clear_character_inbox,
    conversation_turn_messages,
    format_character_location_for_agent,
    format_elapsed_agent_turn_block,
    format_pending_observations_block,
)
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import (
    AGENT_TURN_HEADER,
    AGENT_PERCEPTION_HEADER,
    format_agent_perception_body,
    format_agent_turn_body,
)
from app.llm.client import LLMClient
from app.schemas.agents import CharacterAgentOutput
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


def _conversation_safe_user_content(text: str) -> str:
    lines: list[str] = []
    skipping_tactical_map = False
    skipping_local_context = False
    for line in text.splitlines():
        if line.strip() == "## Tactical Map":
            skipping_tactical_map = True
            continue
        if line.strip() == "## Local Context":
            skipping_local_context = True
            continue
        if skipping_tactical_map and (
            line.startswith("## ") or line.strip() == "</input>"
        ):
            skipping_tactical_map = False
        if skipping_local_context and (
            line.startswith("## ") or line.strip() == "</input>"
        ):
            skipping_local_context = False
        if not skipping_tactical_map:
            if skipping_local_context:
                continue
            lines.append(line)
    return "\n".join(lines)


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


def _join_mode_blocks(*blocks: str) -> str:
    return "\n\n".join(block.strip() for block in blocks if block.strip())


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
        # Usage from the most recent agent turn/perception call.
        self.last_usage: dict[str, int] = {}

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

    def _agent_ruleset_system_addon(self, checkpoint: CheckpointFile) -> str:
        ruleset_id = _session_ruleset_id(checkpoint)
        if ruleset_id == DND5E_BASIC_RULESET_ID:
            return self.prompt_manager.render("agent_ruleset_dnd5e").strip()
        if ruleset_id == ONE_STAR_ASCENSION_RULESET_ID:
            return self.prompt_manager.render(
                "agent_ruleset_one_star"
            ).strip()
        return ""

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
        return "## Visible Equipped Items\n" + description

    def _dnd_combat_mode_block(
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
    ) -> str:
        """Observer-agnostic perception beat — return this character's
        current visual loadout as plain prose (1-3 sentences).

        Used by the observation-harvest fork in `turn_loop.run_beat`
        when a player's action is purely observational
        (`event_kind="observation_harvest"`), and reachable
        later from `/query` for "what does X look like?" questions.

        Distinct from committed agent turns in two load-bearing ways:

        1. **No parenthetical parse.** Perception mode in `agent.txt`
           tells the model not to emit a trailing parenthetical;
           output is pure prose. Returning `response.content.strip()`
           directly skips the `_extract_parenthetical` round-trip
           (which would otherwise log a spurious "missing trailing
           parenthetical" warning on every call).

        2. **Lower max_tokens.** Cap is 3 sentences (~150 tokens of
           prose, leave headroom for the model's own pacing). 600
           leaves slack but stays cheap.

        Perception DOES append to the character's rolling conversation.
        A character should remember how they chose to present themself in
        the current scene, especially after repeated look/query harvests.
        Unlike normal agent turns, this still does not drain pending_observations
        and does not produce private intent.

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
        # also pass an EMPTY
        # pending-observations block to the render — a self-presentation
        # query shouldn't be primed by "react to these incoming events."
        # The character's freshest in-fiction interior is already in
        # their rolling history (`history` above), which is what should
        # color their visual loadout. The resulting loadout is appended
        # after the call so future beats remember what was established.
        char_identity = build_character_packet(character, checkpoint)
        char_state = build_character_state(character, checkpoint)
        one_star_perception = (
            self._one_star_perception_block(character)
            if _session_ruleset_id(checkpoint)
            == ONE_STAR_ASCENSION_RULESET_ID
            else ""
        )

        render_t0 = time.monotonic()
        messages = self.prompt_manager.render_conversation(
            "agent",
            history=history,
            agent_ruleset_system_addon=self._agent_ruleset_system_addon(
                checkpoint
            ),
            **char_identity,
            **char_state,
            world_context=build_world_context(character, checkpoint),
            pending_observations_block="",
            mode_header=AGENT_PERCEPTION_HEADER,
            mode_block=_join_mode_blocks(
                format_agent_perception_body(),
                one_star_perception,
            ),
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
            compact=True,
        )
        text = (response.content or "").strip()
        self.last_usage = {**response.usage, "prompt_render_ms": render_ms}

        conv = checkpoint.character_conversations.setdefault(
            character.character_id, [],
        )
        conversation_user_content = user_content
        if one_star_perception:
            conversation_user_content = conversation_user_content.replace(
                one_star_perception, "", 1
            )
        append_turn_to_conversation(
            conv,
            _conversation_safe_user_content(conversation_user_content),
            response,
        )

        logger.info(
            "Agent %s perceive: %d chars",
            character.name, len(text),
        )
        return text

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
        location_context = (
            f"Location: {location_label}"
            if location_label else "Location: Off-screen / unspecified location."
        )
        foreground_block = (
            self._dnd_combat_mode_block(character, checkpoint)
            if frame == "foreground"
            else ""
        )

        return await self._draft_beat(
            character=character,
            checkpoint=checkpoint,
            mode_header=AGENT_TURN_HEADER,
            mode_block=_join_mode_blocks(
                format_agent_turn_body(
                    frame=frame,
                    location_context=location_context,
                    local_context=local_context,
                ),
                foreground_block,
            ),
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
        mode_header: str,
        mode_block: str,
        log_label: str,
        log_extra: str,
    ) -> CharacterAgentTurnDraft:
        """Shared agent-beat plumbing for all committed/drafted turns.

        Renders the unified `agent` template, calls the LLM, parses
        the trailing parenthetical, and prepares the user/assistant
        history pair. Both modes flow through
        here so any future tweak (model swap, retry policy,
        compaction, telemetry) lands once.

        The system prefix is byte-identical between modes and across
        characters for the same ruleset. Character-derived variables, the
        mode header (`## AGENT-TURN`), and the turn frame live in the user
        message. This shared prefix is what lets one cache lineage cover
        multiple characters on the same model role.

        """
        history = checkpoint.character_conversations.get(character.character_id, [])

        elapsed_time_block = format_elapsed_agent_turn_block(
            character, checkpoint,
        )
        pending_block = (
            build_dnd_player_identities_block(checkpoint)
            + elapsed_time_block
            + format_pending_observations_block(character)
        )
        one_star_state = self._one_star_agent_state_block(
            character, checkpoint
        )

        char_identity = build_character_packet(character, checkpoint)
        char_state = build_character_state(character, checkpoint)

        render_t0 = time.monotonic()
        messages = self.prompt_manager.render_conversation(
            "agent",
            history=history,
            agent_ruleset_system_addon=self._agent_ruleset_system_addon(
                checkpoint
            ),
            **char_identity,
            **char_state,
            world_context=build_world_context(character, checkpoint),
            pending_observations_block=pending_block,
            mode_header=mode_header,
            mode_block=_join_mode_blocks(mode_block, one_star_state),
        )
        render_ms = (time.monotonic() - render_t0) * 1000

        user_content = messages[-1]["content"]

        logger.info(
            "Agent %s (%s) %s [%s]: history=%d msgs",
            character.name, character.character_id,
            log_label, log_extra, len(history),
        )

        response = await self.client.complete(
            role=model_role_for_character(character),
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
        conversation_user_content = user_content
        if one_star_state:
            conversation_user_content = conversation_user_content.replace(
                one_star_state, "", 1
            )
        user_message, assistant_message = conversation_turn_messages(
            _conversation_safe_user_content(conversation_user_content),
            response,
        )

        logger.info(
            "Agent %s %s: %d chars public, %d chars intent",
            character.name, log_label,
            len(result.public_text), len(result.intent),
        )

        return CharacterAgentTurnDraft(
            output=result,
            user_message=user_message,
            assistant_message=assistant_message,
        )
