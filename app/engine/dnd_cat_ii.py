from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone

from app.engine import dice, dnd_combat, mechanics
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication
from app.schemas.dnd_cat_ii import (
    CombatStateDelta,
    PlannedRoll,
    RollPlan,
    RulesAdjudication,
)
from app.schemas.state import (
    CatIIRollDamageRecord,
    CatIIRollRecord,
    CatIIRollTransaction,
    OpenCatIIEvent,
    SlotEntry,
)

logger = logging.getLogger(__name__)
DND5E_BASIC_RULESET_ID = "dnd5e_basic"


class DndCatIIRollsPending(RuntimeError):
    """Raised when D&D Cat II resolution must pause for human dice UI."""

    def __init__(self, transaction: CatIIRollTransaction):
        self.transaction = transaction
        super().__init__(
            f"Cat II event {transaction.event_id} is awaiting player rolls."
        )


def dnd_cat_ii_router_enabled(ckpt: CheckpointFile) -> bool:
    return ckpt.session.config.settings.ruleset_id == DND5E_BASIC_RULESET_ID


def dnd_combat_router_enabled(ckpt: CheckpointFile) -> bool:
    combat = getattr(ckpt.session, "active_combat", None)
    return (
        ckpt.session.config.settings.ruleset_id == DND5E_BASIC_RULESET_ID
        and combat is not None
        and getattr(combat, "status", "active") == "active"
    )


def _combatant_defeat_state(combatant: object) -> str:
    state = str(getattr(combatant, "defeat_state", "") or "")
    if bool(getattr(combatant, "defeated", False)) and state in {"", "active"}:
        return "defeated"
    if state:
        return state
    return "defeated" if bool(getattr(combatant, "defeated", False)) else "active"


class DndCatIIResolver:
    """Router-owned D&D Cat II resolver.

    This uses the event_router model role for roll planning and final
    adjudication, but does not append the roll-planning/finalize calls to the
    router's rolling conversation. Durable roll details live in checkpoint
    transactions; normal prompt history receives only canonical outcome facts.
    """

    def __init__(self, client: LLMClient, prompt_mgr: PromptManager):
        self.client = client
        self.prompt_mgr = prompt_mgr

    async def resolve_cat_ii(
        self,
        *,
        ckpt: CheckpointFile,
        cat_ii_event: OpenCatIIEvent,
    ) -> EventRouterOutput:
        packet = _build_contested_packet(ckpt, cat_ii_event)
        transaction = _find_transaction(ckpt, cat_ii_event.event_id)

        if transaction is None:
            plan = await self._plan_rolls(packet)
            transaction = _create_transaction(ckpt, cat_ii_event, plan)
            _execute_available_rolls(ckpt, transaction)
        elif transaction.status == "awaiting_player_rolls":
            if _pending_player_rolls(transaction):
                _pin_pending_player_rolls(ckpt, transaction)
                raise DndCatIIRollsPending(transaction)
            transaction.status = "ready_to_finalize"
            transaction.updated_at = _utcnow_iso()
        elif transaction.status in {"planning", "planned"}:
            _execute_available_rolls(ckpt, transaction)

        if _pending_player_rolls(transaction):
            transaction.status = "awaiting_player_rolls"
            transaction.updated_at = _utcnow_iso()
            _pin_pending_player_rolls(ckpt, transaction)
            raise DndCatIIRollsPending(transaction)

        adjudication = await self._finalize(packet, transaction.ledger_lines)
        result = _compile_event_router_output(
            ckpt, cat_ii_event, adjudication
        )
        transaction.status = "finalized"
        transaction.final_event_id = result.event_id
        transaction.updated_at = _utcnow_iso()
        _queue_router_state_change(
            ckpt, result, label="D&D Cat II resolved",
        )
        return result

    async def _plan_rolls(self, packet: str) -> RollPlan:
        messages = self.prompt_mgr.render_messages(
            "dnd_cat_ii_router",
            phase="PLAN_ROLLS",
            contested_action_packet=packet,
            roll_ledger_block="No rolls have been made yet.",
        )
        response = await self.client.complete(
            role="event_router",
            messages=messages,
            response_model=RollPlan,
            temperature=0.2,
            max_tokens=2500,
            cache=True,
            compact=False,
        )
        return response.parsed

    async def _finalize(
        self,
        packet: str,
        ledger_lines: list[str],
    ) -> RulesAdjudication:
        messages = self.prompt_mgr.render_messages(
            "dnd_cat_ii_router",
            phase="FINALIZE_OUTCOME",
            contested_action_packet=packet,
            roll_ledger_block="\n".join(ledger_lines) or "No rolls were made.",
        )
        response = await self.client.complete(
            role="event_router",
            messages=messages,
            response_model=RulesAdjudication,
            temperature=0.2,
            max_tokens=3000,
            cache=True,
            compact=False,
        )
        return response.parsed


class DndCombatResolver:
    """Router-owned D&D combat turn resolver.

    Combat uses the same event-router model role as Cat II D&D resolution,
    but the durable transaction source is `combat` and the engine applies
    code-owned dice and HP changes before the final visible event is emitted.
    """

    def __init__(self, client: LLMClient, prompt_mgr: PromptManager):
        self.client = client
        self.prompt_mgr = prompt_mgr

    async def resolve_combat_action(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        intention: str,
    ) -> EventRouterOutput:
        packet = _build_combat_packet(ckpt, actor_id, intention)
        event_id = f"cmb_{uuid.uuid4().hex[:12]}"
        plan = await self._plan_rolls(packet)
        transaction = _create_combat_transaction(
            ckpt=ckpt,
            event_id=event_id,
            actor_id=actor_id,
            intention=intention,
            packet=json.loads(packet),
            plan=plan,
        )
        _execute_available_rolls(ckpt, transaction)
        return await self._resolve_transaction(ckpt, transaction)

    async def continue_combat_transaction(
        self,
        *,
        ckpt: CheckpointFile,
        event_id: str,
    ) -> EventRouterOutput:
        transaction = _find_transaction(ckpt, event_id)
        if transaction is None or transaction.source != "combat":
            raise ValueError(f"No D&D combat roll transaction for {event_id}.")
        if transaction.status == "awaiting_player_rolls":
            if _pending_player_rolls(transaction):
                _pin_pending_player_rolls(ckpt, transaction)
                raise DndCatIIRollsPending(transaction)
            transaction.status = "ready_to_finalize"
            transaction.updated_at = _utcnow_iso()
        elif transaction.status in {"planning", "planned"}:
            _execute_available_rolls(ckpt, transaction)
        return await self._resolve_transaction(ckpt, transaction)

    async def _resolve_transaction(
        self,
        ckpt: CheckpointFile,
        transaction: CatIIRollTransaction,
    ) -> EventRouterOutput:
        if _pending_player_rolls(transaction):
            transaction.status = "awaiting_player_rolls"
            transaction.updated_at = _utcnow_iso()
            _pin_pending_player_rolls(ckpt, transaction)
            raise DndCatIIRollsPending(transaction)

        _execute_combat_damage_rolls(ckpt, transaction)
        packet = json.dumps(transaction.context, indent=2, sort_keys=True)
        adjudication = await self._finalize(packet, transaction.ledger_lines)
        _apply_combat_damage_records(ckpt, transaction)
        _apply_combat_state_deltas(ckpt, adjudication.combat_state_deltas)
        result = _compile_combat_router_output(
            ckpt=ckpt,
            transaction=transaction,
            adjudication=adjudication,
        )
        if adjudication.combat_status == "ended":
            _end_combat_after_adjudication(ckpt, result)
        transaction.status = "finalized"
        transaction.final_event_id = result.event_id
        transaction.updated_at = _utcnow_iso()
        _queue_router_state_change(
            ckpt, result, label="D&D combat resolved",
        )
        return result

    async def _plan_rolls(self, packet: str) -> RollPlan:
        messages = self.prompt_mgr.render_messages(
            "dnd_combat_router",
            phase="PLAN_ROLLS",
            combat_action_packet=packet,
            roll_ledger_block="No rolls have been made yet.",
        )
        response = await self.client.complete(
            role="event_router",
            messages=messages,
            response_model=RollPlan,
            temperature=0.2,
            max_tokens=2500,
            cache=True,
            compact=False,
        )
        return response.parsed

    async def _finalize(
        self,
        packet: str,
        ledger_lines: list[str],
    ) -> RulesAdjudication:
        messages = self.prompt_mgr.render_messages(
            "dnd_combat_router",
            phase="FINALIZE_OUTCOME",
            combat_action_packet=packet,
            roll_ledger_block="\n".join(ledger_lines) or "No rolls were made.",
        )
        response = await self.client.complete(
            role="event_router",
            messages=messages,
            response_model=RulesAdjudication,
            temperature=0.2,
            max_tokens=3000,
            cache=True,
            compact=False,
        )
        return response.parsed


def _end_combat_after_adjudication(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
) -> None:
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is None:
        return
    dnd_combat.append_audit_line(
        combat,
        f"Combat ended from D&D combat adjudication: {result.event_id}.",
    )
    dnd_combat.end_combat(ckpt.session)
    result.canonical_event.observable_facts.append(ObservableFact.all(
        "D&D combat ends; the scene is no longer in initiative order."
    ))


def complete_pending_player_roll(
    ckpt: CheckpointFile,
    *,
    event_id: str,
    roll_id: str,
    completed_by_user_id: str = "",
) -> CatIIRollRecord:
    """Execute one stored player roll request from Discord/UI input."""
    transaction = _find_transaction(ckpt, event_id)
    if transaction is None:
        raise ValueError(f"No D&D roll transaction for Cat II event {event_id}.")

    record = next((r for r in transaction.rolls if r.roll_id == roll_id), None)
    if record is None:
        raise ValueError(f"No pending roll {roll_id} for Cat II event {event_id}.")
    if record.actor_control != "player":
        raise ValueError(f"Roll {roll_id} is not a player-controlled roll.")
    if record.status != "pending":
        raise ValueError(f"Roll {roll_id} has already been completed.")

    _execute_roll_record(
        transaction,
        record,
        completed_by_user_id=completed_by_user_id,
    )
    if not _pending_player_rolls_for_actor(transaction, record.actor_id):
        _release_roll_slot(ckpt, transaction.event_id, record.actor_id)
    if not _pending_player_rolls(transaction):
        transaction.status = "ready_to_finalize"
    transaction.updated_at = _utcnow_iso()
    return record


def pending_player_rolls(
    ckpt: CheckpointFile,
    *,
    event_id: str = "",
    actor_id: str = "",
) -> list[CatIIRollRecord]:
    out: list[CatIIRollRecord] = []
    for transaction in ckpt.session.cat_ii_roll_transactions:
        if event_id and transaction.event_id != event_id:
            continue
        for record in _pending_player_rolls(transaction):
            if actor_id and record.actor_id != actor_id:
                continue
            out.append(record)
    return out


def roll_transaction_source(ckpt: CheckpointFile, event_id: str) -> str:
    transaction = _find_transaction(ckpt, event_id)
    return transaction.source if transaction is not None else ""


def _create_transaction(
    ckpt: CheckpointFile,
    cat_ii_event: OpenCatIIEvent,
    plan: RollPlan,
) -> CatIIRollTransaction:
    bindings = ckpt.session.character_bindings or {}
    by_id = {c.character_id: c for c in ckpt.characters}
    now = _utcnow_iso()
    rolls: list[CatIIRollRecord] = []
    for request in plan.roll_requests:
        actor_control = "player" if request.actor_id in bindings else "agent"
        modifier = mechanics.roll_modifier(by_id.get(request.actor_id), request)
        rolls.append(
            CatIIRollRecord(
                roll_id=request.roll_id,
                actor_id=request.actor_id,
                actor_control=actor_control,
                request=request.model_dump(),
                modifier=modifier,
                label=_roll_label(request),
                reason=request.reason,
            )
        )

    transaction = CatIIRollTransaction(
        transaction_id=f"rolltxn_{uuid.uuid4().hex[:12]}",
        event_id=cat_ii_event.event_id,
        ruleset_id=ckpt.session.config.settings.ruleset_id,
        status="planned" if plan.needs_rolls else "ready_to_finalize",
        plan=plan.model_dump(),
        no_roll_reason=plan.no_roll_reason,
        rolls=rolls,
        ledger_lines=[] if plan.needs_rolls else _no_roll_ledger(plan),
        created_at=now,
        updated_at=now,
    )
    ckpt.session.cat_ii_roll_transactions.append(transaction)
    cat_ii_event.roll_transaction_id = transaction.transaction_id
    return transaction


def _create_combat_transaction(
    *,
    ckpt: CheckpointFile,
    event_id: str,
    actor_id: str,
    intention: str,
    packet: dict,
    plan: RollPlan,
) -> CatIIRollTransaction:
    bindings = ckpt.session.character_bindings or {}
    by_id = {c.character_id: c for c in ckpt.characters}
    now = _utcnow_iso()
    rolls: list[CatIIRollRecord] = []
    for request in plan.roll_requests:
        actor_control = "player" if request.actor_id in bindings else "agent"
        modifier = mechanics.roll_modifier(by_id.get(request.actor_id), request)
        rolls.append(
            CatIIRollRecord(
                roll_id=request.roll_id,
                actor_id=request.actor_id,
                actor_control=actor_control,
                request=request.model_dump(),
                modifier=modifier,
                label=_roll_label(request),
                reason=request.reason,
            )
        )

    transaction = CatIIRollTransaction(
        transaction_id=f"rolltxn_{uuid.uuid4().hex[:12]}",
        event_id=event_id,
        source="combat",
        actor_id=actor_id,
        intention=intention,
        ruleset_id=ckpt.session.config.settings.ruleset_id,
        status="planned" if plan.needs_rolls else "ready_to_finalize",
        plan=plan.model_dump(),
        context=packet,
        no_roll_reason=plan.no_roll_reason,
        rolls=rolls,
        ledger_lines=[] if plan.needs_rolls else _no_roll_ledger(plan),
        created_at=now,
        updated_at=now,
    )
    ckpt.session.cat_ii_roll_transactions.append(transaction)
    return transaction


def _execute_available_rolls(
    ckpt: CheckpointFile,
    transaction: CatIIRollTransaction,
) -> None:
    player_roll_mode = ckpt.session.config.settings.player_roll_mode
    for record in transaction.rolls:
        if record.status != "pending":
            continue
        if record.actor_control == "player" and player_roll_mode == "interactive":
            continue
        _execute_roll_record(transaction, record, completed_by_user_id="engine")

    transaction.status = (
        "awaiting_player_rolls"
        if _pending_player_rolls(transaction)
        else "ready_to_finalize"
    )
    transaction.updated_at = _utcnow_iso()


def _execute_roll_record(
    transaction: CatIIRollTransaction,
    record: CatIIRollRecord,
    *,
    completed_by_user_id: str,
) -> None:
    request = PlannedRoll.model_validate(record.request)
    result = dice.roll_d20_check(
        roll_id=record.roll_id,
        modifier=record.modifier,
        actor_id=record.actor_id,
        reason=record.reason,
        advantage_state=request.advantage_state,
    )
    record.status = "completed"
    record.result = result.model_dump()
    record.completed_by_user_id = completed_by_user_id
    record.completed_at = _utcnow_iso()
    transaction.ledger_lines.append(_format_ledger_line(request, result))


def _execute_combat_damage_rolls(
    ckpt: CheckpointFile,
    transaction: CatIIRollTransaction,
) -> None:
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is None:
        return
    for record in transaction.rolls:
        if record.status != "completed":
            continue
        request = PlannedRoll.model_validate(record.request)
        if request.kind != "attack_roll" or not request.target_id:
            continue
        marker = f"damage_for={record.roll_id}"
        if any(
            damage.roll_id == record.roll_id
            for damage in transaction.damage_records
        ):
            continue
        result = record.result or {}
        hit = _attack_hits(combat, request, result)
        target_ac = _target_ac(combat, request.target_id)
        transaction.ledger_lines.append(
            f"{record.roll_id}: attack total {result.get('total', 0)} "
            f"vs AC {target_ac} -> {'hit' if hit else 'miss'}"
        )
        if not hit:
            continue
        expression = _damage_expression_for_action(ckpt, request)
        if not expression:
            transaction.ledger_lines.append(
                f"{marker}: no code-readable damage expression for "
                f"{request.actor_id} action {request.action_id or request.skill}"
            )
            continue
        if result.get("crit") == "crit":
            expression = _crit_damage_expression(expression)
        damage = dice.roll_expression(
            dice.RollRequest(
                roll_id=f"damage_{record.roll_id}",
                expression=expression,
                actor_id=request.actor_id,
                reason=f"Damage for {record.reason}",
            )
        )
        transaction.ledger_lines.append(
            f"{marker}: {request.actor_id} deals {damage.detail} = "
            f"{damage.total} damage to {request.target_id}"
        )
        transaction.damage_records.append(
            CatIIRollDamageRecord(
                roll_id=record.roll_id,
                target_id=request.target_id,
                amount=damage.total,
                expression=damage.expression,
                detail=damage.detail,
                applied=False,
            )
        )


def _apply_combat_damage_records(
    ckpt: CheckpointFile,
    transaction: CatIIRollTransaction,
) -> None:
    for damage in transaction.damage_records:
        if damage.applied:
            continue
        if damage.amount <= 0:
            damage.applied = True
            continue
        dnd_combat.apply_damage(
            ckpt.session, damage.target_id, damage.amount
        )
        damage.applied = True


def _attack_hits(
    combat: object,
    request: PlannedRoll,
    result: dict[str, object],
) -> bool:
    crit = str(result.get("crit") or "none")
    if crit == "crit":
        return True
    if crit == "fail":
        return False
    try:
        total = int(result.get("total") or 0)
    except (TypeError, ValueError):
        total = 0
    dc = request.dc or _target_ac(combat, request.target_id)
    return total >= dc


def _target_ac(combat: object, target_id: str) -> int:
    for combatant in list(getattr(combat, "combatants", []) or []):
        ids = {
            str(getattr(combatant, "combatant_id", "") or ""),
            str(getattr(combatant, "character_id", "") or ""),
        }
        if target_id in ids:
            return int(getattr(combatant, "armor_class", 10) or 10)
    return 10


def _damage_expression_for_action(
    ckpt: CheckpointFile,
    request: PlannedRoll,
) -> str:
    action_key = request.action_id or request.skill
    character = next(
        (c for c in ckpt.characters if c.character_id == request.actor_id),
        None,
    )
    if character is None:
        return ""
    mechanics_state = character.mechanics or {}
    statblock = (
        (mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {}
    )
    action = _find_action(statblock, action_key, reason=request.reason)
    if action is None:
        return ""
    return _action_damage_expression(action)


def _find_action(
    statblock: dict[str, object],
    action_key: str,
    *,
    reason: str = "",
) -> dict[str, object] | None:
    wanted = _normalize_action_text(action_key)
    reason_text = _normalize_action_text(reason)
    first_damaging_action: dict[str, object] | None = None
    damaging_action_count = 0
    for action in statblock.get("actions") or []:
        if not isinstance(action, dict):
            continue
        if _action_damage_expression(action):
            damaging_action_count += 1
            if first_damaging_action is None:
                first_damaging_action = action
        names = _action_names(action)
        if wanted and wanted in names:
            return action
        if reason_text and any(
            _contains_action_name(reason_text, name) for name in names
        ):
            return action
    if not wanted and damaging_action_count == 1:
        return first_damaging_action
    return None


def _action_damage_expression(action: dict[str, object]) -> str:
    attack = action.get("attack") or {}
    if not isinstance(attack, dict):
        return ""
    raw = str(attack.get("damage") or "")
    return _clean_damage_expression(raw)


def _action_names(action: dict[str, object]) -> set[str]:
    names = {
        _normalize_action_text(action.get("id") or ""),
        _normalize_action_text(action.get("name") or ""),
    }
    return {name for name in names if name}


def _normalize_action_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _contains_action_name(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return re.search(rf"(^|\s){re.escape(needle)}($|\s)", haystack) is not None


def _clean_damage_expression(raw: str) -> str:
    match = re.search(r"\d+d\d+(?:\s*[+-]\s*\d+)?", raw.strip().lower())
    if match is None:
        return ""
    return re.sub(r"\s+", "", match.group(0))


def _crit_damage_expression(expression: str) -> str:
    # Simple D&D crit support: double the first damage dice group, keep
    # flat modifiers unchanged. "1d8+4" -> "2d8+4".
    def repl(match: re.Match[str]) -> str:
        return f"{int(match.group(1)) * 2}d{match.group(2)}"

    return re.sub(r"(\d+)d(\d+)", repl, expression, count=1)


def _build_contested_packet(
    ckpt: CheckpointFile,
    cat_ii_event: OpenCatIIEvent,
) -> str:
    by_id = {c.character_id: c for c in ckpt.characters}
    bindings = ckpt.session.character_bindings or {}
    participant_ids = _participant_ids(cat_ii_event)
    participants = []
    for cid in participant_ids:
        char = by_id.get(cid)
        if char is None:
            participants.append({"character_id": cid, "missing": True})
            continue
        participants.append({
            "character_id": cid,
            "name": char.name,
            "role": char.public_sheet.role,
            "location": char.location,
            "player_controlled": cid in bindings,
            "mechanics": mechanics.mechanics_summary(char),
        })

    payload = {
        "ruleset_id": ckpt.session.config.settings.ruleset_id,
        "player_roll_mode": ckpt.session.config.settings.player_roll_mode,
        "initiator_id": cat_ii_event.initiator_id,
        "initiator_intention": cat_ii_event.initiator_intention,
        "required_responders": cat_ii_event.required_responders,
        "collected_intentions": cat_ii_event.collected_intentions,
        "swept_responders": cat_ii_event.swept_responders,
        "opening_observable_facts": cat_ii_event.opening_observable_facts,
        "participants": participants,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _build_combat_packet(
    ckpt: CheckpointFile,
    actor_id: str,
    intention: str,
) -> str:
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is None:
        raise ValueError("D&D combat resolver requires active combat.")
    by_id = {c.character_id: c for c in ckpt.characters}
    bindings = ckpt.session.character_bindings or {}
    current = _current_combatant(combat)
    participants = []
    for combatant in list(getattr(combat, "combatants", []) or []):
        cid = str(
            getattr(combatant, "character_id", "")
            or getattr(combatant, "combatant_id", "")
            or ""
        )
        char = by_id.get(cid)
        participants.append({
            "combatant_id": str(getattr(combatant, "combatant_id", "") or cid),
            "character_id": cid,
            "name": str(getattr(combatant, "name", "") or cid),
            "player_controlled": cid in bindings,
            "current": combatant is current,
            "armor_class": int(getattr(combatant, "armor_class", 10) or 10),
            "hit_points": {
                "current": int(
                    getattr(combatant, "hit_points_current", 0) or 0
                ),
                "max": int(getattr(combatant, "hit_points_max", 0) or 0),
                "temporary": int(
                    getattr(combatant, "hit_points_temporary", 0) or 0
                ),
            },
            "conditions": list(getattr(combatant, "conditions", []) or []),
            "defeat_state": _combatant_defeat_state(combatant),
            "death_saves": {
                "successes": int(
                    getattr(combatant, "death_save_successes", 0) or 0
                ),
                "failures": int(
                    getattr(combatant, "death_save_failures", 0) or 0
                ),
            },
            "mechanics": (
                mechanics.mechanics_summary(char) if char is not None else {}
            ),
            "actions": _combat_action_summaries(char),
            "spellcasting": (
                _combat_spellcasting_summary(char) if cid == actor_id else {}
            ),
            "spells": _combat_spell_summaries(char) if cid == actor_id else [],
        })

    payload = {
        "ruleset_id": ckpt.session.config.settings.ruleset_id,
        "player_roll_mode": ckpt.session.config.settings.player_roll_mode,
        "round_number": int(getattr(combat, "round_number", 1) or 1),
        "current_turn": {
            "actor_id": actor_id,
            "name": _character_name(ckpt, actor_id),
        },
        "intention": intention,
        "house_rules": [
            "Opportunity attacks are automatic for players and NPCs.",
            "Player opportunity attacks do not consume optional reaction prompts.",
            "Open optional player reaction prompts only for meaningful choices.",
        ],
        "combatants": participants,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _combat_action_summaries(character: object | None) -> list[dict[str, object]]:
    if character is None:
        return []
    mechanics_state = getattr(character, "mechanics", None) or {}
    statblock = (
        (mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {}
    )
    actions: list[dict[str, object]] = []
    for action in statblock.get("actions") or []:
        if not isinstance(action, dict):
            continue
        attack = action.get("attack") or {}
        if not isinstance(attack, dict):
            attack = {}
        actions.append({
            "id": str(action.get("id") or ""),
            "name": str(action.get("name") or ""),
            "attack_bonus": attack.get("bonus", ""),
            "damage": str(attack.get("damage") or ""),
        })
    return actions


def _combat_spellcasting_summary(character: object | None) -> dict[str, object]:
    if character is None:
        return {}
    mechanics_state = getattr(character, "mechanics", None) or {}
    statblock = (
        (mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {}
    )
    spellcasting = statblock.get("spellcasting") or {}
    if not isinstance(spellcasting, dict):
        return {}
    profiles: list[dict[str, object]] = []
    for profile in spellcasting.get("profiles") or []:
        if not isinstance(profile, dict):
            continue
        profiles.append({
            "id": str(profile.get("id") or ""),
            "name": str(profile.get("name") or ""),
            "ability": str(profile.get("ability") or ""),
            "spell_attack_bonus": profile.get("spell_attack_bonus", ""),
            "spell_save_dc": profile.get("spell_save_dc", ""),
        })
    return {
        "profiles": profiles,
        "slots": spellcasting.get("slots") or {},
        "pact_slots": spellcasting.get("pact_slots") or {},
    }


def _combat_spell_summaries(character: object | None) -> list[dict[str, object]]:
    if character is None:
        return []
    mechanics_state = getattr(character, "mechanics", None) or {}
    statblock = (
        (mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {}
    )
    spellcasting = statblock.get("spellcasting") or {}
    if not isinstance(spellcasting, dict):
        return []

    spells: list[dict[str, object]] = []
    for spell in spellcasting.get("spells") or []:
        if not isinstance(spell, dict):
            continue
        attack = spell.get("attack") or {}
        if not isinstance(attack, dict):
            attack = {}
        save = spell.get("save") or {}
        if not isinstance(save, dict):
            save = {}
        spells.append({
            "id": str(spell.get("id") or ""),
            "name": str(spell.get("name") or ""),
            "level": spell.get("level", 0),
            "prepared": bool(spell.get("prepared")),
            "always_prepared": bool(spell.get("always_prepared")),
            "concentration": bool(spell.get("concentration")),
            "attack": {
                "ability": str(attack.get("ability") or ""),
                "bonus": attack.get("bonus", ""),
            },
            "save": {
                "ability": str(save.get("ability") or ""),
                "dc": save.get("dc", ""),
            },
            "damage": _formula_summaries(spell.get("damage")),
            "healing": _formula_summaries(spell.get("healing")),
            "consumes": _resource_summaries(spell.get("consumes")),
        })
    return spells


def _formula_summaries(value: object) -> list[str]:
    out: list[str] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if not isinstance(item, dict):
            continue
        formula = str(item.get("formula") or "").strip()
        if formula:
            out.append(formula)
    return out


def _resource_summaries(value: object) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if not isinstance(item, dict):
            continue
        resource_id = str(item.get("resource_id") or "").strip()
        if not resource_id:
            continue
        out.append({
            "resource_id": resource_id,
            "amount": item.get("amount", 1),
        })
    return out


def _current_combatant(combat: object) -> object | None:
    combatants = list(getattr(combat, "combatants", []) or [])
    if not combatants:
        return None
    try:
        idx = int(getattr(combat, "turn_index", 0) or 0) % len(combatants)
    except (TypeError, ValueError):
        idx = 0
    return combatants[idx]


def _character_name(ckpt: CheckpointFile, character_id: str) -> str:
    return next(
        (c.name for c in ckpt.characters if c.character_id == character_id),
        character_id,
    )


def _compile_event_router_output(
    ckpt: CheckpointFile,
    cat_ii_event: OpenCatIIEvent,
    adjudication: RulesAdjudication,
) -> EventRouterOutput:
    observer_ids = _observer_ids(cat_ii_event)
    notes = "; ".join(adjudication.rules_notes)
    rationale_parts = [
        part for part in (
            adjudication.mechanical_summary,
            f"Rules notes: {notes}" if notes else "",
            f"Fallback: {adjudication.fallback_reason}"
            if adjudication.fallback_reason else "",
        ) if part
    ]
    return EventRouterOutput(
        event_id="",
        decision_rationale=" ".join(rationale_parts) or "D&D Cat II adjudication.",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                feasible=adjudication.feasible
            ),
            observable_facts=[
                ObservableFact.all(fact)
                for fact in adjudication.visible_outcome_facts
            ],
        ),
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="cat_ii_resolution",
        observers=[
            ObserverEntry(
                character_id=cid,
                observation_level="d",
                response_priority=5 if cid in _participant_ids(cat_ii_event) else 3,
            )
            for cid in observer_ids
            if _character_exists(ckpt, cid)
        ],
        spawn=[],
        dormant=[],
        cull=[],
    )


def _compile_combat_router_output(
    *,
    ckpt: CheckpointFile,
    transaction: CatIIRollTransaction,
    adjudication: RulesAdjudication,
) -> EventRouterOutput:
    combat = getattr(ckpt.session, "active_combat", None)
    observer_ids = []
    if combat is not None:
        for combatant in list(getattr(combat, "combatants", []) or []):
            if (
                _combatant_defeat_state(combatant) != "active"
                or bool(getattr(combatant, "removed", False))
            ):
                continue
            cid = str(
                getattr(combatant, "character_id", "")
                or getattr(combatant, "combatant_id", "")
                or ""
            )
            if cid:
                observer_ids.append(cid)
    observer_ids = _dedupe(observer_ids or [transaction.actor_id])
    affected_ids = _combat_affected_ids(transaction, adjudication)
    notes = "; ".join(adjudication.rules_notes)
    rationale_parts = [
        part for part in (
            adjudication.mechanical_summary,
            f"Rules notes: {notes}" if notes else "",
            f"Fallback: {adjudication.fallback_reason}"
            if adjudication.fallback_reason else "",
        ) if part
    ]
    return EventRouterOutput(
        event_id="",
        decision_rationale=" ".join(rationale_parts) or "D&D combat adjudication.",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                feasible=adjudication.feasible
            ),
            observable_facts=[
                ObservableFact.all(fact)
                for fact in adjudication.visible_outcome_facts
            ],
        ),
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="ruleset_resolution",
        observers=[
            ObserverEntry(
                character_id=cid,
                observation_level="d",
                response_priority=5 if cid in affected_ids else 3,
            )
            for cid in observer_ids
            if _character_exists(ckpt, cid)
        ],
        spawn=[],
        dormant=[],
        cull=[],
    )


def _combat_affected_ids(
    transaction: CatIIRollTransaction,
    adjudication: RulesAdjudication,
) -> set[str]:
    affected = {transaction.actor_id} if transaction.actor_id else set()
    for record in transaction.rolls:
        if record.actor_id:
            affected.add(record.actor_id)
        request = PlannedRoll.model_validate(record.request)
        if request.target_id:
            affected.add(request.target_id)
    for damage in transaction.damage_records:
        if damage.target_id:
            affected.add(damage.target_id)
    for delta in adjudication.combat_state_deltas:
        if delta.target_id:
            affected.add(delta.target_id)
    return affected


def _apply_combat_state_deltas(
    ckpt: CheckpointFile,
    deltas: list[CombatStateDelta],
) -> None:
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is None:
        return
    for delta in deltas:
        if delta.kind == "healing":
            if delta.amount:
                dnd_combat.apply_healing(ckpt.session, delta.target_id, delta.amount)
        elif delta.kind in {"condition_add", "condition_remove"}:
            _apply_condition_delta(ckpt, delta)


def _apply_condition_delta(
    ckpt: CheckpointFile,
    delta: CombatStateDelta,
) -> None:
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is None or not delta.condition:
        return
    for combatant in list(getattr(combat, "combatants", []) or []):
        ids = {
            str(getattr(combatant, "combatant_id", "") or ""),
            str(getattr(combatant, "character_id", "") or ""),
        }
        if delta.target_id not in ids:
            continue
        conditions = list(getattr(combatant, "conditions", []) or [])
        if delta.kind == "condition_add":
            if delta.condition not in conditions:
                conditions.append(delta.condition)
        else:
            conditions = [c for c in conditions if c != delta.condition]
        setattr(combatant, "conditions", conditions)
        return


def _queue_router_state_change(
    ckpt: CheckpointFile,
    routed: EventRouterOutput,
    *,
    label: str,
) -> None:
    facts = [
        fact.text for fact in routed.canonical_event.observable_facts
        if fact.text
    ]
    if facts:
        ckpt.session.pending_router_state_changes.append(
            f"{label}: " + " / ".join(facts)
        )


def _find_transaction(
    ckpt: CheckpointFile,
    event_id: str,
) -> CatIIRollTransaction | None:
    return next(
        (
            transaction
            for transaction in ckpt.session.cat_ii_roll_transactions
            if transaction.event_id == event_id
        ),
        None,
    )


def _pending_player_rolls(
    transaction: CatIIRollTransaction,
) -> list[CatIIRollRecord]:
    return [
        record for record in transaction.rolls
        if record.actor_control == "player" and record.status == "pending"
    ]


def _pending_player_rolls_for_actor(
    transaction: CatIIRollTransaction,
    actor_id: str,
) -> list[CatIIRollRecord]:
    return [
        record for record in _pending_player_rolls(transaction)
        if record.actor_id == actor_id
    ]


def _pin_pending_player_rolls(
    ckpt: CheckpointFile,
    transaction: CatIIRollTransaction,
) -> None:
    now = _utcnow_iso()
    for record in _pending_player_rolls(transaction):
        ckpt.session.active_act_slots[record.actor_id] = SlotEntry(
            reason="cat_ii_roll",
            cat_ii_event_id=transaction.event_id,
            claimed_at=now,
        )


def _release_roll_slot(
    ckpt: CheckpointFile,
    event_id: str,
    actor_id: str,
) -> None:
    entry = ckpt.session.active_act_slots.get(actor_id)
    if (
        entry is not None
        and entry.reason == "cat_ii_roll"
        and entry.cat_ii_event_id == event_id
    ):
        ckpt.session.active_act_slots.pop(actor_id, None)


def _format_ledger_line(
    request: PlannedRoll,
    result: dice.RollResult,
) -> str:
    dc_part = f", DC {request.dc}" if request.dc else ""
    opposed_part = (
        f", opposed by {request.opposed_by}" if request.opposed_by else ""
    )
    return (
        f"{result.roll_id}: {request.actor_id} {request.kind} "
        f"({request.ability}"
        f"{', ' + request.skill if request.skill else ''}) "
        f"rolled {result.detail} = {result.total}"
        f"{dc_part}{opposed_part}; reason: {request.reason}"
    )


def _no_roll_ledger(plan: RollPlan) -> list[str]:
    reason = plan.no_roll_reason.strip()
    return [f"No rolls: {reason}"] if reason else []


def _roll_label(request: PlannedRoll) -> str:
    if request.kind == "attack_roll" and request.action_id:
        return f"Attack ({request.action_id.replace('_', ' ').title()})"
    if request.kind == "skill_check" and request.skill:
        return request.skill.title()
    if request.kind == "saving_throw":
        return f"{request.ability.upper()} Save"
    if request.kind == "attack_roll":
        return "Attack"
    return f"{request.ability.upper()} Check"


def _participant_ids(cat_ii_event: OpenCatIIEvent) -> list[str]:
    ids = [
        cat_ii_event.initiator_id,
        *cat_ii_event.required_responders,
        *cat_ii_event.collected_intentions.keys(),
    ]
    return _dedupe(ids)


def _observer_ids(cat_ii_event: OpenCatIIEvent) -> list[str]:
    return _dedupe(
        [
            *cat_ii_event.opening_observer_ids,
            *cat_ii_event.required_responders,
            cat_ii_event.initiator_id,
        ]
    )


def _dedupe(ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for cid in ids:
        cid = cid.strip()
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _character_exists(ckpt: CheckpointFile, character_id: str) -> bool:
    return any(c.character_id == character_id for c in ckpt.characters)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
