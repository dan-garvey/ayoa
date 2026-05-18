"""v11-A3a: LLMDispatcher — concrete Dispatcher binding turn_loop to the
real router / agent / narrator modules.

`turn_loop.run_beat` talks to the world exclusively through the
`Dispatcher` Protocol. This module provides the production binding:
the three async methods call through to the `event_router` prompt
template, CharacterAgent.respond(), and
narrator.compose_pov_render() (per-POV entry point), constructing
their user-message blocks through the shared helpers in
`turn_loop_contracts` so prompt-code contracts stay in lockstep.

The legacy `EventRouter` engine class that wrapped this same prompt
template was murdered in v11-r7j; this dispatcher is the only
production caller of the `event_router` template now.

Tests should prefer passing fakes into `run_beat` directly; this
class is what the orchestrator constructs at wire-up time.
"""

from __future__ import annotations

import asyncio
import logging

from app.engine import narrator as narrator_module
from app.engine.character_agent import CharacterAgent
from app.engine.context_builder import (
    build_player_characters_block,
    build_setting_summary,
    resolve_acting_character,
)
from app.engine.prompt_manager import PromptManager
from app.engine.dnd_cat_ii import (
    DND5E_BASIC_RULESET_ID,
    DndCatIIResolver,
    dnd_cat_ii_router_enabled,
    dnd_combat_manager_enabled,
)
from app.engine.dnd_combat_resolution import DndCombatResolver
from app.engine.turn_loop_contracts import (
    format_cat_ii_resolution_block,
    format_agent_output_entry,
    format_human_initiator_intention,
    format_npc_cascade_intention,
    format_router_continuation_block,
)
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import DndEventRouterOutput, EventRouterOutput
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import OpenCatIIEvent, RenderBufferEntry

logger = logging.getLogger(__name__)


def _session_ruleset_id(ckpt: CheckpointFile) -> str:
    return str(getattr(ckpt.session.config.settings, "ruleset_id", "") or "")


def _dnd_fresh_router_enabled(
    ckpt: CheckpointFile,
    cat_ii_event: OpenCatIIEvent | None,
) -> bool:
    return (
        cat_ii_event is None
        and _session_ruleset_id(ckpt) == DND5E_BASIC_RULESET_ID
    )


def _router_ruleset_template_vars(
    prompt_mgr: PromptManager,
    *,
    dnd_fresh: bool,
) -> dict[str, str]:
    if dnd_fresh:
        return {
            "ruleset_router_addon": prompt_mgr.render(
                "event_router_ruleset_dnd5e",
            ).strip(),
            "ruleset_output_schema_fields": prompt_mgr.render(
                "event_router_ruleset_dnd5e_output_fields",
            ),
        }
    return {
        "ruleset_router_addon": "",
        "ruleset_output_schema_fields": "",
    }


# v11-r7j note: a mirror copy of the legacy `EventRouter` engine
# class's private helpers used to live in app/engine/event_router.py
# and these were straight-ported during the v11 transition. The
# legacy class was murdered in v11-r7j; this dispatcher is now the
# only home for these helpers and the duplication concern is gone.


# `build_setting_summary` is now in `app/engine/context_builder.py`
# and imported at the top of this module — pre-v11-r7j three near-
# identical copies of the same helper lived in this module, narrator,
# and engine_bridge.


def _build_world_rules(checkpoint: CheckpointFile) -> str:
    physics = checkpoint.world_state.physics_ruleset
    parts = [f"Strength limits: {physics.strength_limits}"]
    parts.append(f"Magic: {'enabled' if physics.magic_enabled else 'disabled'}")
    return "\n".join(parts)


def _build_router_world_lore(checkpoint: CheckpointFile) -> str:
    """Stable common-knowledge world context for the router system prefix."""
    parts: list[str] = []
    facts = [fact for fact in (checkpoint.world_state.facts or []) if fact]
    if facts:
        parts.append(
            "Key world facts:\n"
            + "\n".join(f"- {fact}" for fact in facts)
        )
    if checkpoint.world_state.lore:
        parts.append(checkpoint.world_state.lore)
    if not parts:
        return "No detailed lore available."
    return "\n\n".join(parts)


def _build_initial_roster_block(checkpoint: CheckpointFile) -> str:
    """Render the full NPC roster + interior on turn 1 only.

    Pre-Commit-3 the per-turn user message carried `character_registry`
    EVERY turn — name + role + status + location for every non-player
    character (~80 tokens × N NPCs, ~1000 tokens/turn for hollowstone).
    The router has its own session conversation history, so re-feeding
    identity every turn was duplication: turn-1's inject plus spawn
    outcomes are already in history.

    This block lands ONCE — on turn 1 only — and carries richer signal
    than the dropped registry: name, id, role, location, goals,
    current_objectives. The router uses it to seed "who's in this
    world and what are they trying to do" so picking decisions on turn
    1 aren't blind. Returns "" on every turn after the first.

    NOTE: this block does NOT carry per-NPC interior beyond
    importer-seeded goals/objectives. An agent's freshest interior
    (the trailing parenthetical from its last committed turn)
    lives in that agent's own rolling history and is deliberately
    NOT mirrored to the router — the router decides who acts based on
    public signals (cascade intentions, prior canonical events) +
    seeded objectives, not stolen interior.
    """
    if checkpoint.session_conversation:
        return ""
    if not checkpoint.characters:
        return ""

    from app.engine.context_builder import collect_player_ids
    player_ids = collect_player_ids(checkpoint)
    entries: list[str] = []
    for char in checkpoint.characters:
        if char.character_id in player_ids:
            continue
        if char.status.value != "active":
            continue

        role = char.public_sheet.role or "unknown role"

        parts = [
            f"- {char.character_id}",
            f"  Role: {role}",
        ]
        goals = [g for g in (char.private_state.goals or []) if g]
        if goals:
            parts.append(
                "  Goals (long-term): " + "; ".join(goals)
            )
        objs = [o for o in (char.private_state.current_objectives or []) if o]
        if objs:
            parts.append(
                "  Current objectives (active pursuits): " + "; ".join(objs)
            )
        entries.append("\n".join(parts))

    if not entries:
        return ""

    header = (
        "## Initial Roster\n"
        "Every active NPC in this world, with their long-term goals "
        "and active pursuits.\n\n"
    )
    return header + "\n\n".join(entries) + "\n"


def _build_engine_state_updates_block(checkpoint: CheckpointFile) -> str:
    """Drain non-router engine mutations for the next fresh router call.

    Exhaustive current producers:
    - player loot claim: character inventory changed from a D&D loot offer
    - player loot currency split: party inventories changed from a D&D loot offer
    - D&D sheet attached: character mechanics/ruleset state changed externally
    - player binding changed: /join bound a human to a character
    - player binding removed: /leave returned a character to AI control
    - custom player character created by frontend takeover/create flow
    - existing character replaced/overwritten by player character creation

    Explicitly excluded:
    - router-authored spawn/dormant/cull/location/commitment changes
    - spawned-character summaries; spawn materialization failure is fatal
    - combat deaths/effect expirations when they are already emitted through
      compact history or post-combat continuity updates
    - D&D Cat II resolver outputs, which are compact router history
    - routine D&D combat-manager outputs while initiative remains active
    - clocks/session leading time derived from canonical event timing
    - XP awards unless they become intentionally visible to fiction
    """
    queued = checkpoint.session.pending_engine_state_updates or []
    if not queued:
        return ""
    checkpoint.session.pending_engine_state_updates = []
    body = "\n".join(f"- {entry}" for entry in queued)
    return (
        "## Engine State Updates\n"
        "Durable state changed outside router-authored canonical events. "
        "Fold these updates into the next adjudication without replaying "
        "them as new visible action unless the current intention makes "
        "that visibility relevant.\n\n"
        f"{body}\n"
    )


def _build_hidden_facts(checkpoint: CheckpointFile) -> str:
    facts = checkpoint.world_state.hidden_facts
    if not facts:
        return "None."
    return "\n".join(f"- {fact}" for fact in facts)


def _build_router_input_block(*blocks: str) -> str:
    return "\n\n".join(
        block.strip()
        for block in blocks
        if block and block.strip()
    )


def _compact_router_history_text(text: str) -> str:
    return " ".join((text or "").split())


def _compact_id_list(values: list[str]) -> str:
    return ",".join(value for value in values if value) or "-"


def _defer_history_user_prompt(intention: str) -> str:
    """Return the compact user-history entry for defer, if applicable.

    Router history usually stores only deterministic assistant-side
    `prior_event` records. `(defer)` is the exception: repeated defers
    are pacing feedback, so the next router call needs to see that the
    player explicitly deferred rather than merely infer it from whatever
    event the previous router authored.
    """
    if (intention or "").strip().lower() == "(defer)":
        return "(defer)"
    return ""


def _router_history_record(
    *,
    acting_character_id: str,
    result: EventRouterOutput,
    mode: str = "intention",
) -> str:
    """Compact assistant-side memory of a prior router output.

    Router history exists to carry canonical continuity, not to replay the
    full structured-output envelope. Store a deterministic event memory and
    omit the raw user message, `decision_rationale`, feasibility boilerplate,
    empty schema fields, and JSON punctuation.
    """
    header = (
        f"prior_event {result.event_id} @{result.effective_at_s}"
        f"+{result.duration_s} source={acting_character_id or '-'} mode={mode}"
    )
    if result.ends_beat_reason:
        header += f" end={result.ends_beat_reason}"
    elif result.ends_beat:
        header += " end=true"
    if result.requires_responders:
        header += f" requires={_compact_id_list(result.required_responders)}"
    if result.next_output_character_ids:
        header += f" next={_compact_id_list(result.next_output_character_ids)}"
    if result.perception_enrichment_character_ids:
        header += (
            " enrich="
            f"{_compact_id_list(result.perception_enrichment_character_ids)}"
        )

    lines = [header]

    for fact in result.canonical_event.observable_facts:
        text = _compact_router_history_text(fact.text)
        if not text:
            continue
        audience = (
            "all"
            if fact.audience == "all_observers"
            else f"only[{_compact_id_list(fact.visible_to)}]"
        )
        lines.append(
            f"fact {audience} @{fact.at_offset_s}+{fact.duration_s}: {text}"
        )

    if result.observers:
        observer_bits = [
            f"{observer.character_id}:{observer.observation_level}:"
            f"{observer.routing_role}"
            for observer in result.observers
            if observer.character_id
        ]
        if observer_bits:
            lines.append("obs " + " ".join(observer_bits))

    if result.spawn:
        for spawn in result.spawn:
            objectives = "; ".join(
                _compact_router_history_text(objective)
                for objective in spawn.seed.objectives
                if objective
            )
            seed_bits = [
                f"role={_compact_router_history_text(spawn.seed.role)}",
                f"reason={_compact_router_history_text(spawn.seed.reason)}",
                f"loc={_compact_router_history_text(spawn.seed.location)}",
            ]
            if objectives:
                seed_bits.append(f"objectives={objectives}")
            lines.append(f"spawn {spawn.character_id} " + " ".join(seed_bits))
    if result.dormant:
        lines.append(f"dormant {_compact_id_list(result.dormant)}")
    if result.cull:
        lines.append(f"cull {_compact_id_list(result.cull)}")

    commitment_open = result.commitment_open
    if commitment_open.present:
        lines.append(
            "commit_open "
            f"actors={_compact_id_list(commitment_open.actor_ids)} "
            f"expected={commitment_open.expected_duration_s} "
            f"max={commitment_open.max_duration_s} "
            f"loc={_compact_router_history_text(commitment_open.location_label)} "
            f"desc={_compact_router_history_text(commitment_open.description)}"
        )
    for signal in result.commitment_resolutions:
        lines.append(
            "commit_resolve "
            f"id={signal.commitment_id or '-'} "
            f"actors={_compact_id_list(signal.actor_ids)} "
            f"at={signal.resolved_at_offset_s} "
            f"reason={_compact_router_history_text(signal.reason)}"
        )
    for signal in result.commitment_interrupts:
        lines.append(
            "commit_interrupt "
            f"id={signal.commitment_id or '-'} "
            f"actors={_compact_id_list(signal.actor_ids)} "
            f"at={signal.observed_at_offset_s} "
            f"reason={_compact_router_history_text(signal.reason)}"
        )
    for update in result.location_updates:
        lines.append(
            "loc "
            f"{update.character_id}={_compact_router_history_text(update.location_label)}"
        )

    interaction_mode = getattr(result, "interaction_mode", "")
    if interaction_mode:
        lines.append(f"dnd_mode {interaction_mode}")
    combatant_ids = getattr(result, "combatant_ids", [])
    if combatant_ids:
        lines.append(f"combatants {_compact_id_list(combatant_ids)}")

    return "\n".join(lines)


def _append_router_history_record(
    conversation: list[ConversationMessage],
    *,
    acting_character_id: str,
    result: EventRouterOutput,
    mode: str = "intention",
    user_prompt: str = "",
) -> None:
    if user_prompt:
        conversation.append(ConversationMessage(
            role="user",
            content=user_prompt,
        ))
    conversation.append(ConversationMessage(
        role="assistant",
        content=_router_history_record(
            acting_character_id=acting_character_id,
            result=result,
            mode=mode,
        ),
    ))


def refresh_router_history_record(
    conversation: list[ConversationMessage],
    *,
    acting_character_id: str,
    result: EventRouterOutput,
    mode: str = "intention",
) -> None:
    """Replace the compact memory for a router event after mutation.

    Harvest paths append authoritative perception facts after the router
    output has already been normalized and stored. Keep the durable
    router ledger aligned with the final canonical event object.
    """
    content = _router_history_record(
        acting_character_id=acting_character_id,
        result=result,
        mode=mode,
    )
    prefix = f"prior_event {result.event_id} "
    for index in range(len(conversation) - 1, -1, -1):
        message = conversation[index]
        if (
            message.role == "assistant"
            and isinstance(message.content, str)
            and message.content.startswith(prefix)
        ):
            conversation[index] = ConversationMessage(
                role="assistant",
                content=content,
            )
            return
    logger.warning(
        "Router history refresh found no prior record for event %s",
        result.event_id,
    )


def _normalize_router_result_for_history(
    ckpt: CheckpointFile,
    *,
    result: EventRouterOutput,
    clock_anchor_character_id: str = "",
    cat_ii_event: OpenCatIIEvent | None = None,
    minimum_effective_at_s: int | None = None,
) -> None:
    if clock_anchor_character_id:
        actor = next(
            (
                c for c in ckpt.characters
                if c.character_id == clock_anchor_character_id
            ),
            None,
        )
        if actor is not None and cat_ii_event is None:
            result.effective_at_s = max(result.effective_at_s, actor.clock_at_s)
    if cat_ii_event is not None and cat_ii_event.opening_event_id:
        opening = next(
            (
                event for event in ckpt.canonical_events
                if event.event_id == cat_ii_event.opening_event_id
            ),
            None,
        )
        if opening is not None:
            result.effective_at_s = opening.effective_at_s
    if minimum_effective_at_s is not None:
        result.effective_at_s = max(
            result.effective_at_s,
            minimum_effective_at_s,
        )


def _roll_transaction_actor_id(ckpt: CheckpointFile, event_id: str) -> str:
    for transaction in ckpt.session.cat_ii_roll_transactions:
        if transaction.event_id == event_id:
            return transaction.actor_id
    return ""


def _build_router_context(
    ckpt: CheckpointFile,
    acting_character_id: str,
    *,
    resolve_actor_fallback: bool = True,
    include_engine_state_updates: bool = True,
) -> dict[str, str]:
    """Collect every context variable the event_router template needs
    aside from the current router input block.

    Returns a dict ready to splat into `prompt_mgr.render_messages`
    after merging in `{router_input_block}`.
    """
    if resolve_actor_fallback:
        acting_id, _acting_char, _acting_name = resolve_acting_character(
            ckpt, acting_character_id,
        )
    else:
        acting_id = acting_character_id

    return {
        "setting_summary": build_setting_summary(ckpt),
        "world_lore": _build_router_world_lore(ckpt),
        "world_rules": _build_world_rules(ckpt),
        "hidden_lore": ckpt.world_state.hidden_lore or "None.",
        "hidden_facts": _build_hidden_facts(ckpt),
        "acting_character_id": acting_id,
        "player_characters_block": build_player_characters_block(
            ckpt, acting_id,
        ),
        "initial_roster_block": _build_initial_roster_block(ckpt),
        "engine_state_updates_block": (
            _build_engine_state_updates_block(ckpt)
            if include_engine_state_updates else ""
        ),
    }


class LLMDispatcher:
    """Production Dispatcher implementation — binds `turn_loop.run_beat`
    to the real router / agent / narrator modules."""

    def __init__(self, client: LLMClient, prompt_mgr: PromptManager):
        self.client = client
        self.prompt_mgr = prompt_mgr
        # Character agent is stateless aside from `last_usage`; reusing one
        # instance avoids per-call allocation.
        self._agent = CharacterAgent(client, prompt_mgr)
        self._dnd_cat_ii = DndCatIIResolver(client, prompt_mgr)
        self._dnd_combat = DndCombatResolver(client, prompt_mgr)

    # ------------------------------------------------------------------
    # route_intention
    # ------------------------------------------------------------------

    async def route_intention(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        intention: str,
        cat_ii_event: OpenCatIIEvent | None = None,
    ) -> EventRouterOutput:
        """Classify + adjudicate one intention through event_router."""

        if cat_ii_event is not None and dnd_cat_ii_router_enabled(ckpt):
            logger.info(
                "LLMDispatcher.route_intention: actor=%s cat_ii=%s "
                "using dnd_cat_ii_router",
                actor_id, cat_ii_event.event_id,
            )
            result = await self._dnd_cat_ii.resolve_cat_ii(
                ckpt=ckpt,
                cat_ii_event=cat_ii_event,
            )
            _normalize_router_result_for_history(
                ckpt,
                result=result,
                clock_anchor_character_id=actor_id,
                cat_ii_event=cat_ii_event,
            )
            _append_router_history_record(
                ckpt.session_conversation,
                acting_character_id=actor_id,
                result=result,
                mode="cat_ii_resolution",
            )
            return result

        saved_engine_updates = list(
            ckpt.session.pending_engine_state_updates
        )
        try:
            ctx = _build_router_context(
                ckpt,
                actor_id,
            )
            dnd_fresh = _dnd_fresh_router_enabled(ckpt, cat_ii_event)

            if cat_ii_event is None:
                bindings = ckpt.session.character_bindings or {}
                if actor_id in bindings:
                    intention_block = format_human_initiator_intention(
                        actor_id, intention,
                    )
                else:
                    intention_block = format_npc_cascade_intention(
                        actor_id, intention,
                    )
                cat_ii_resolution_block = ""
            else:
                evt = cat_ii_event
                responders: list[tuple[str, str]] = [
                    (rid, evt.collected_intentions[rid])
                    for rid in evt.required_responders
                    if rid in evt.collected_intentions
                ]
                cat_ii_resolution_block = format_cat_ii_resolution_block(
                    initiator_id=evt.initiator_id,
                    initiator_intention=evt.initiator_intention,
                    responders=responders,
                    swept_responders=list(evt.swept_responders),
                )
                intention_block = ""

            router_input_block = _build_router_input_block(
                ctx.pop("initial_roster_block", ""),
                ctx.pop("engine_state_updates_block", ""),
                cat_ii_resolution_block,
                intention_block,
            )
            template_vars = {
                **ctx,
                **_router_ruleset_template_vars(
                    self.prompt_mgr,
                    dnd_fresh=dnd_fresh,
                ),
                "router_input_block": router_input_block,
            }

            # Use render_conversation so the rolling router ledger rides
            # along, and append this turn's exchange after the call so
            # continuity compounds across turns. The caller
            # (Orchestrator) persists the checkpoint after run_beat returns.
            messages = self.prompt_mgr.render_conversation(
                "event_router",
                history=ckpt.session_conversation,
                **template_vars,
            )

            logger.info(
                "LLMDispatcher.route_intention: actor=%s cat_ii=%s",
                actor_id, cat_ii_event.event_id if cat_ii_event else None,
            )

            response = await self.client.complete(
                role="event_router",
                messages=messages,
                response_model=(
                    DndEventRouterOutput if dnd_fresh else EventRouterOutput
                ),
                temperature=0.35,
                max_tokens=5000,
                cache=True,
                compact=True,
            )
        except Exception:
            ckpt.session.pending_engine_state_updates = saved_engine_updates
            raise

        result: EventRouterOutput = response.parsed
        _normalize_router_result_for_history(
            ckpt,
            result=result,
            clock_anchor_character_id=actor_id,
            cat_ii_event=cat_ii_event,
        )
        # Persist only compact canonical memory. The current turn's raw
        # input already shaped this result; replaying it beside the result
        # duplicates context and delays cache benefit.
        _append_router_history_record(
            ckpt.session_conversation,
            acting_character_id=actor_id,
            result=result,
            mode="cat_ii_resolution" if cat_ii_event else "intention",
            user_prompt=(
                _defer_history_user_prompt(intention)
                if cat_ii_event is None else ""
            ),
        )
        return result

    async def route_combat_action(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        intention: str,
    ) -> EventRouterOutput:
        """Resolve one active D&D combat turn through the ruleset adapter."""
        if not dnd_combat_manager_enabled(ckpt):
            raise RuntimeError("D&D combat routing requested outside active D&D combat.")
        logger.info(
            "LLMDispatcher.route_combat_action: actor=%s using dnd_combat_manager",
            actor_id,
        )
        result = await self._dnd_combat.resolve_combat_action(
            ckpt=ckpt,
            actor_id=actor_id,
            intention=intention,
        )
        _normalize_router_result_for_history(
            ckpt,
            result=result,
            clock_anchor_character_id=actor_id,
        )
        if ckpt.session.active_combat is None:
            _append_router_history_record(
                ckpt.session_conversation,
                acting_character_id=actor_id,
                result=result,
                mode="dnd_combat_end",
            )
        return result

    async def continue_combat_transaction(
        self,
        *,
        ckpt: CheckpointFile,
        event_id: str,
    ) -> EventRouterOutput:
        if not dnd_combat_manager_enabled(ckpt):
            raise RuntimeError(
                "D&D combat roll continuation requested outside active D&D combat."
            )
        logger.info(
            "LLMDispatcher.continue_combat_transaction: event=%s", event_id,
        )
        result = await self._dnd_combat.continue_combat_transaction(
            ckpt=ckpt,
            event_id=event_id,
        )
        actor_id = _roll_transaction_actor_id(ckpt, event_id)
        _normalize_router_result_for_history(
            ckpt,
            result=result,
            clock_anchor_character_id=actor_id,
        )
        if ckpt.session.active_combat is None:
            _append_router_history_record(
                ckpt.session_conversation,
                acting_character_id=actor_id,
                result=result,
                mode="dnd_combat_end",
            )
        return result

    # ------------------------------------------------------------------
    # route_continuation
    # ------------------------------------------------------------------

    async def route_continuation(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        prior_result: EventRouterOutput,
    ) -> EventRouterOutput:
        """Ask the router to advance an open beat with no next-output target."""

        saved_engine_updates = list(
            ckpt.session.pending_engine_state_updates
        )
        try:
            ctx = _build_router_context(
                ckpt,
                actor_id,
            )
            continuation_block = format_router_continuation_block(
                prior_rationale=prior_result.decision_rationale,
            )

            router_input_block = _build_router_input_block(
                ctx.pop("initial_roster_block", ""),
                ctx.pop("engine_state_updates_block", ""),
                continuation_block,
            )
            template_vars = {
                **ctx,
                **_router_ruleset_template_vars(
                    self.prompt_mgr,
                    dnd_fresh=False,
                ),
                "router_input_block": router_input_block,
            }

            messages = self.prompt_mgr.render_conversation(
                "event_router",
                history=ckpt.session_conversation,
                **template_vars,
            )

            logger.info(
                "LLMDispatcher.route_continuation: actor=%s", actor_id,
            )

            response = await self.client.complete(
                role="event_router",
                messages=messages,
                response_model=EventRouterOutput,
                temperature=0.35,
                max_tokens=5000,
                cache=True,
                compact=True,
            )
        except Exception:
            ckpt.session.pending_engine_state_updates = saved_engine_updates
            raise

        result: EventRouterOutput = response.parsed
        _normalize_router_result_for_history(
            ckpt,
            result=result,
            clock_anchor_character_id=actor_id,
        )
        _append_router_history_record(
            ckpt.session_conversation,
            acting_character_id=actor_id,
            result=result,
            mode="continuation",
        )
        return result

    # ------------------------------------------------------------------
    # route_agent_output
    # ------------------------------------------------------------------

    async def route_agent_output(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        public_text: str,
    ) -> EventRouterOutput:
        """Route one returned character output into one canonical event.

        Agent parentheticals are already stripped by the agent adapter. The
        router receives only public result text and chooses the next response
        or player-facing boundary through its normal output fields.
        """
        ctx = _build_router_context(
            ckpt,
            character_id,
            resolve_actor_fallback=False,
            include_engine_state_updates=False,
        )

        agent_output = format_agent_output_entry(character_id, public_text)

        template_vars = {
            **ctx,
            **_router_ruleset_template_vars(
                self.prompt_mgr,
                dnd_fresh=False,
            ),
            "router_input_block": agent_output,
        }

        base_messages = self.prompt_mgr.render_messages(
            "event_router",
            **template_vars,
        )
        messages = [base_messages[0]]
        for item in ckpt.session_conversation:
            messages.append({"role": item.role, "content": item.content})
        messages.append({
            "role": "user",
            "content": agent_output,
        })

        logger.info(
            "LLMDispatcher.route_agent_output: character=%s",
            character_id,
        )

        response = await self.client.complete(
            role="event_router",
            messages=messages,
            response_model=EventRouterOutput,
            temperature=0.35,
            max_tokens=5000,
            cache=True,
            compact=True,
        )

        result: EventRouterOutput = response.parsed
        _normalize_router_result_for_history(
            ckpt,
            result=result,
            minimum_effective_at_s=ckpt.session.leading_at_s,
        )
        _append_router_history_record(
            ckpt.session_conversation,
            acting_character_id=character_id,
            result=result,
            mode="agent_output",
        )
        return result

    # ------------------------------------------------------------------
    # agent_intend
    # ------------------------------------------------------------------

    async def agent_intend(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        frame: str = "foreground",
        local_context: str = "",
    ) -> str:
        """Invoke the character agent and return its prose for the router.

        The agent's prose IS the intention — no serialization layer
        needed. The trailing parenthetical (private intent) is stripped
        at parse time; what we return here is the public surface only,
        which is what the router reads as the acting character's intention.

        Three result shapes the caller (run_beat) must distinguish:
          - **non-empty prose** → real intention, route normally.
          - **`"(remains silent)"`** → the agent had a non-empty intent
            parenthetical but emitted no public prose (the agent
            prompt's "Sparse is valid" shared rule — paren-only
            output is in-character). The cascade MUST treat this as
            a real beat and route it; otherwise the prompt's promise
            that silence is a valid in-character choice gets quietly
            broken by `_is_agent_refusal` collapsing it to
            `cascade_exhausted`.
          - **`""`** → true refusal: no public prose AND no intent (or
            the parser logged a "missing trailing parenthetical"
            warning and we have nothing to route). The cascade ends.
        """
        character = next(
            (c for c in ckpt.characters if c.character_id == character_id),
            None,
        )
        if character is None:
            logger.warning(
                "agent_intend: unknown character_id %s", character_id,
            )
            return ""

        output = await self._agent.turn(
            character=character,
            checkpoint=ckpt,
            acting_character_id=character_id,
            frame=frame,
            local_context=local_context,
        )
        public = output.public_text.strip()
        if public:
            return public
        if output.intent.strip():
            # Agent chose deliberate silence (paren-only output). Surface
            # a fixed sentinel so the router can adjudicate a "watches
            # without speaking" beat instead of the cascade dying. The
            # sentinel is intentionally short, parenthesized, and
            # identical every time so the router can recognize it.
            logger.info(
                "Agent %s emitted silent beat (intent=%d chars); "
                "routing via sentinel.",
                character.name, len(output.intent),
            )
            return "(remains silent)"
        return ""

    # ------------------------------------------------------------------
    # harvest_perceptions  (v11-r8a: observation_harvest fork)
    # ------------------------------------------------------------------

    async def harvest_perceptions(
        self,
        *,
        ckpt: CheckpointFile,
        character_ids: list[str],
        acting_character_id: str,
    ) -> list[str]:
        """Fan out CharacterAgent.perceive() across `character_ids` in
        parallel.

        Returns one string per id in input order — empty string for any
        character whose perception call failed (unknown id, LLM error,
        empty output). The harvest fork in `run_beat` filters empties
        and appends the non-empty fragments to the canonical event's
        `observable_facts` block.

        Per-character exceptions are absorbed locally rather than
        bubbled. The harvest is a UX enrichment, not the beat's
        critical path; one failed perception out of three should
        leave the other two on the player's screen instead of taking
        the whole render down. The caller logs dropped fragments at
        WARN so test playthroughs still surface the failure.

        Cache lineage: every `perceive` call shares the SAME system
        prompt as normal agent turns under the same ruleset (single
        unified `agent` template). Character identity lives in the
        per-call user message, so parallel fan-out compounds well with
        this — a 3-character harvest bills three Haiku calls in roughly
        one round-trip wall time, all hitting the cached system prefix.
        """
        if not character_ids:
            return []

        by_id = {c.character_id: c for c in ckpt.characters}

        async def _one(cid: str) -> str:
            character = by_id.get(cid)
            if character is None:
                logger.warning(
                    "harvest_perceptions: unknown character_id %s", cid,
                )
                return ""
            try:
                return await self._agent.perceive(
                    character=character,
                    checkpoint=ckpt,
                    acting_character_id=acting_character_id,
                )
            except Exception as exc:  # noqa: BLE001 — see docstring
                logger.warning(
                    "harvest_perceptions: perceive() failed for %s: %s",
                    cid, exc,
                )
                return ""

        logger.info(
            "harvest_perceptions: firing %d parallel perceive calls",
            len(character_ids),
        )
        return list(await asyncio.gather(*(_one(c) for c in character_ids)))

    # ------------------------------------------------------------------
    # narrator_compose
    # ------------------------------------------------------------------

    async def narrator_compose(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        buffered_events: list[RenderBufferEntry],
        partial_mode_override: bool | None = None,
        user_input: str = "",
    ) -> tuple[NarratorFinalOutput, "TranscriptEntry"]:
        """Render per-POV prose via narrator.compose_pov_render.

        Returns `(NarratorFinalOutput, TranscriptEntry)` so run_beat
        can populate ckpt.transcript via the parallel
        BeatResult.transcript_entries map. The transcript entry is
        constructed engine-side from `user_input` (the real player
        utterance for the acting POV; "" for incidental POVs in a
        multi-human beat) and the rendered prose. Pre-r7j the LLM
        owned the transcript entry and emitted `"{name} — "` for the
        user field every time.

        `partial_mode` defaults to True iff this character is currently
        pinned as a Cat II responder in the current beat — the narrator renders
        a partial view because the beat still has outstanding resolution
        work. `partial_mode_override`, when not None, wins over the slot
        scan (v11-r6a: Cat II-open render path sets this True for the
        initiator + pinned humans so they see the mid-attempt cliffhanger
        even though the initiator isn't pinned themselves).
        """
        if partial_mode_override is not None:
            partial_mode = partial_mode_override
        else:
            partial_mode = _is_pinned_as_cat_ii_responder(ckpt, character_id)

        envelope, entry = await narrator_module.compose_pov_render(
            client=self.client,
            prompt_mgr=self.prompt_mgr,
            ckpt=ckpt,
            pov_character_id=character_id,
            buffered_events=buffered_events,
            partial_mode=partial_mode,
            user_input=user_input,
        )
        return envelope, entry


def _is_pinned_as_cat_ii_responder(
    ckpt: CheckpointFile, character_id: str,
) -> bool:
    """True iff `character_id` is pinned as a Cat II responder this beat."""
    entry = ckpt.session.active_act_slots.get(character_id)
    return bool(entry is not None and entry.reason == "cat_ii_responder")
