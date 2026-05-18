from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.engine import dice, dnd_combat, dnd_equipment, dnd_spatial, mechanics
from app.engine.dnd_combat_access import (
    combatant_defeat_state as _combatant_defeat_state,
    combatants as _combatants,
    current_combatant as _current_combatant,
    target_armor_class as _target_ac,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import (
    DndEventRouterOutput,
    DndObserverEntry,
    EventRouterOutput,
    ObserverEntry,
)
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication
from app.schemas.dnd_cat_ii import (
    CombatStateDelta,
    EffectDelta,
    PlannedRoll,
    RollPlan,
    RulesAdjudication,
)
from app.schemas.state import (
    CatIIRollDamageAdjustmentRecord,
    CatIIRollDamageComponentRecord,
    CatIIRollDamageRecord,
    CatIIRollRecord,
    CatIIRollTransaction,
    DndRuntimeEffect,
    OpenCatIIEvent,
    SlotEntry,
)

logger = logging.getLogger(__name__)
DND5E_BASIC_RULESET_ID = "dnd5e_basic"
_DAMAGE_TYPES = {
    "acid",
    "bludgeoning",
    "cold",
    "fire",
    "force",
    "lightning",
    "necrotic",
    "piercing",
    "poison",
    "psychic",
    "radiant",
    "slashing",
    "thunder",
}


@dataclass(frozen=True)
class _DamageComponent:
    expression: str = ""
    damage_type: str = ""


@dataclass(frozen=True)
class _DamageProfile:
    components: tuple[_DamageComponent, ...] = ()

    @property
    def expression(self) -> str:
        return " + ".join(component.expression for component in self.components)

    @property
    def damage_type(self) -> str:
        return _join_damage_types(
            component.damage_type for component in self.components
        )


@dataclass(frozen=True)
class _DamageAdjustmentCandidate:
    source: str
    kind: str
    damage_type: str
    reason: str
    scope: str = "component"


class DndCatIIRollsPending(RuntimeError):
    """Raised when D&D Cat II resolution must pause for human dice UI."""

    def __init__(self, transaction: CatIIRollTransaction):
        self.transaction = transaction
        super().__init__(
            f"Cat II event {transaction.event_id} is awaiting player rolls."
        )


def dnd_cat_ii_router_enabled(ckpt: CheckpointFile) -> bool:
    return ckpt.session.config.settings.ruleset_id == DND5E_BASIC_RULESET_ID


def dnd_combat_manager_enabled(ckpt: CheckpointFile) -> bool:
    combat = getattr(ckpt.session, "active_combat", None)
    return (
        ckpt.session.config.settings.ruleset_id == DND5E_BASIC_RULESET_ID
        and combat is not None
        and getattr(combat, "status", "active") == "active"
    )


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


def _end_combat_after_adjudication(
    ckpt: CheckpointFile,
    result: EventRouterOutput,
) -> None:
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is None:
        return
    for fact in dnd_combat.drain_pending_visible_facts(combat):
        result.canonical_event.observable_facts.append(ObservableFact.all(fact))
    dnd_combat.append_audit_line(
        combat,
        f"Combat ended from D&D combat adjudication: {result.event_id}.",
    )
    dnd_combat.queue_router_observed_fact_updates(ckpt.session, combat)
    dnd_combat.end_combat(ckpt.session, characters=ckpt.characters)
    result.canonical_event.observable_facts.append(ObservableFact.all(
        "D&D combat ends."
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
        if transaction.status == "cancelled":
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
        modifier = _roll_modifier_for_request(by_id.get(request.actor_id), request)
        rolls.append(
            CatIIRollRecord(
                roll_id=request.roll_id,
                actor_id=request.actor_id,
                actor_control=actor_control,
                request=request.model_dump(),
                modifier=modifier,
                label=_roll_label(request, by_id.get(request.actor_id)),
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
        modifier = _roll_modifier_for_request(by_id.get(request.actor_id), request)
        rolls.append(
            CatIIRollRecord(
                roll_id=request.roll_id,
                actor_id=request.actor_id,
                actor_control=actor_control,
                request=request.model_dump(),
                modifier=modifier,
                label=_roll_label(request, by_id.get(request.actor_id)),
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
        defense = (
            f"effective DC {request.dc} (base AC {target_ac})"
            if request.dc and request.dc != target_ac
            else f"AC {target_ac}"
        )
        _append_ledger_line_once(
            transaction,
            f"{record.roll_id}: attack total {result.get('total', 0)} "
            f"vs {defense} -> {'hit' if hit else 'miss'}",
        )
        if not hit:
            continue
        damage_profile = _damage_profile_for_action(ckpt, request)
        if not damage_profile.components:
            _append_ledger_line_once(
                transaction,
                f"{marker}: no code-readable damage expression for "
                f"{request.actor_id} action {request.action_id or request.skill}",
            )
            continue
        components = _roll_damage_components(
            ckpt,
            request,
            record=record,
            damage_profile=damage_profile,
            crit=result.get("crit") == "crit",
        )
        raw_amount = sum(component.raw_amount for component in components)
        component_total = sum(component.amount for component in components)
        component_adjustments = [
            adjustment
            for component in components
            for adjustment in component.adjustments
        ]
        final_amount, attack_adjustments = _adjust_attack_total_amount(
            request,
            raw_amount=component_total,
            components=components,
        )
        adjustments = [*component_adjustments, *attack_adjustments]
        _append_ledger_line_once(
            transaction,
            _damage_ledger_line(
                marker,
                actor_id=request.actor_id,
                target_id=request.target_id,
                components=components,
                raw_total=raw_amount,
                final_total=final_amount,
                attack_adjustments=attack_adjustments,
            ),
        )
        transaction.damage_records.append(
            CatIIRollDamageRecord(
                roll_id=record.roll_id,
                target_id=request.target_id,
                raw_amount=raw_amount,
                amount=final_amount,
                damage_type=damage_profile.damage_type,
                adjustments=adjustments,
                components=components,
                expression=" + ".join(
                    component.expression for component in components
                ),
                detail="; ".join(component.detail for component in components),
                applied=False,
            )
        )


def _append_ledger_line_once(
    transaction: CatIIRollTransaction,
    line: str,
) -> None:
    if line not in transaction.ledger_lines:
        transaction.ledger_lines.append(line)


def _roll_damage_components(
    ckpt: CheckpointFile,
    request: PlannedRoll,
    *,
    record: CatIIRollRecord,
    damage_profile: _DamageProfile,
    crit: bool,
) -> list[CatIIRollDamageComponentRecord]:
    components: list[CatIIRollDamageComponentRecord] = []
    for index, component in enumerate(damage_profile.components, start=1):
        expression = (
            _crit_damage_expression(component.expression)
            if crit else component.expression
        )
        damage = dice.roll_expression(
            dice.RollRequest(
                roll_id=f"damage_{record.roll_id}_{index}",
                expression=expression,
                actor_id=request.actor_id,
                reason=f"Damage for {record.reason}",
            )
        )
        final_amount, adjustments = _adjust_damage_amount(
            ckpt,
            request,
            raw_amount=damage.total,
            damage_type=component.damage_type,
        )
        components.append(
            CatIIRollDamageComponentRecord(
                expression=damage.expression,
                detail=damage.detail,
                damage_type=component.damage_type,
                raw_amount=damage.total,
                amount=final_amount,
                adjustments=adjustments,
            )
        )
    return components


def _damage_ledger_line(
    marker: str,
    *,
    actor_id: str,
    target_id: str,
    components: list[CatIIRollDamageComponentRecord],
    raw_total: int,
    final_total: int,
    attack_adjustments: list[CatIIRollDamageAdjustmentRecord],
) -> str:
    parts = []
    for component in components:
        type_text = f" {component.damage_type}" if component.damage_type else ""
        parts.append(
            f"{component.detail} = {component.raw_amount}{type_text}"
            f"{_damage_adjustment_ledger_text(component.adjustments)}"
        )
    attack_adjustment_text = _damage_adjustment_ledger_text(attack_adjustments)
    total_text = (
        f"; total {raw_total}->{final_total}{attack_adjustment_text}"
        if raw_total != final_total else f"; total {final_total}"
    )
    return (
        f"{marker}: {actor_id} deals "
        f"{'; '.join(parts)} damage to {target_id}{total_text}"
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
            ckpt.session,
            damage.target_id,
            damage.amount,
            characters=ckpt.characters,
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


def _combat_actions_for_character(character: object | None) -> list[dict[str, object]]:
    if character is None:
        return []
    mechanics_state = getattr(character, "mechanics", None) or {}
    if not isinstance(mechanics_state, dict):
        return []
    statblock = (
        (mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {}
    )
    actions = [
        action for action in statblock.get("actions") or []
        if isinstance(action, dict)
    ]
    actions.extend(dnd_equipment.runtime_weapon_actions(character))
    return actions


def _roll_modifier_for_request(
    character: object | None,
    request: PlannedRoll,
) -> int:
    if request.kind == "attack_roll" and character is not None:
        mechanics_state = getattr(character, "mechanics", None) or {}
        statblock = (
            (mechanics_state.get("dnd5e_sheet") or {}).get("statblock") or {}
        )
        if isinstance(statblock, dict):
            actions = _combat_actions_for_character(character)
            action = _find_action(
                {"actions": actions},
                request.action_id or request.skill,
                reason=request.reason,
            )
            if action is not None:
                bonus = _action_attack_bonus(action)
                if bonus is not None:
                    return bonus
            if request.action_id or _attack_action_count({"actions": actions}) > 1:
                strict_request = request.model_copy(
                    update={"reason": "", "skill": ""}
                )
                return mechanics.roll_modifier(character, strict_request)
    return mechanics.roll_modifier(character, request)


def _damage_profile_for_action(
    ckpt: CheckpointFile,
    request: PlannedRoll,
) -> _DamageProfile:
    action_key = request.action_id or request.skill
    character = next(
        (c for c in ckpt.characters if c.character_id == request.actor_id),
        None,
    )
    if character is None:
        return _DamageProfile()
    action = _find_action(
        {"actions": _combat_actions_for_character(character)},
        action_key,
        reason=request.reason,
    )
    if action is None:
        return _DamageProfile()
    return _action_damage_profile(action)


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
    reason_matches: list[dict[str, object]] = []
    for action in statblock.get("actions") or []:
        if not isinstance(action, dict):
            continue
        if _action_damage_profile(action).expression:
            damaging_action_count += 1
            if first_damaging_action is None:
                first_damaging_action = action
        names = _action_names(action)
        if wanted and wanted in names:
            return action
        if reason_text and any(
            _contains_action_name(reason_text, name) for name in names
        ):
            reason_matches.append(action)
    if wanted:
        return None
    if len(reason_matches) == 1:
        return reason_matches[0]
    if not wanted and damaging_action_count == 1:
        return first_damaging_action
    return None


def _action_damage_profile(action: dict[str, object]) -> _DamageProfile:
    attack = action.get("attack") or {}
    if not isinstance(attack, dict):
        attack = {}
    raw = str(attack.get("damage") or "")
    components = _damage_components_from_text(raw)
    component_profile = _damage_component_profile(action.get("damage"))
    if components:
        return _DamageProfile(components=tuple(components))
    return component_profile


def _action_attack_bonus(action: dict[str, object]) -> int | None:
    attack = action.get("attack") or {}
    if not isinstance(attack, dict):
        return None
    bonus = attack.get("bonus")
    if bonus is None:
        return None
    try:
        return int(bonus)
    except (TypeError, ValueError):
        return None


def _attack_action_count(statblock: dict[str, object]) -> int:
    count = 0
    for action in statblock.get("actions") or []:
        if isinstance(action, dict) and _action_attack_bonus(action) is not None:
            count += 1
    return count


def _action_damage_summary(action: dict[str, object]) -> str:
    attack = action.get("attack") or {}
    if isinstance(attack, dict):
        raw = str(attack.get("damage") or "").strip()
        if raw:
            return raw
    profile = _damage_component_profile(action.get("damage"))
    if not profile.expression:
        return ""
    return _damage_profile_summary(profile)


def _damage_component_profile(value: object) -> _DamageProfile:
    if not isinstance(value, list):
        return _DamageProfile()
    components: list[_DamageComponent] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        formula = str(item.get("formula") or "").strip()
        expression = _clean_damage_expression(formula)
        if not expression:
            continue
        damage_type = _extract_damage_type(
            item.get("damage_type") or item.get("type") or formula
        )
        components.append(
            _DamageComponent(expression=expression, damage_type=damage_type)
        )
    return _DamageProfile(components=tuple(_merge_damage_components(components)))


def _action_names(action: dict[str, object]) -> set[str]:
    names = {
        _normalize_action_text(action.get("id") or ""),
        _normalize_action_text(action.get("name") or ""),
    }
    return {name for name in names if name}


def _normalize_action_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _contains_action_name(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return re.search(rf"(^|\s){re.escape(needle)}($|\s)", haystack) is not None


def _clean_damage_expression(raw: str) -> str:
    terms = re.findall(
        r"[+-]?\s*(?:\d+d\d+|\d+)",
        raw.strip().lower(),
    )
    if not terms:
        return ""
    expression = "".join(re.sub(r"\s+", "", term) for term in terms)
    return expression.lstrip("+")


def _damage_components_from_text(raw: object) -> list[_DamageComponent]:
    text = _primary_damage_text(raw)
    if not text:
        return []
    matches = list(_damage_type_spans(text))
    components: list[_DamageComponent] = []
    previous_type_end = 0
    for damage_type, start, end in matches:
        expression = _clean_damage_expression(text[previous_type_end:start])
        previous_type_end = end
        if not expression:
            continue
        components.append(
            _DamageComponent(expression=expression, damage_type=damage_type)
        )
    if components:
        return _merge_damage_components(components)
    expression = _clean_damage_expression(text)
    if not expression:
        return []
    return [_DamageComponent(expression=expression, damage_type="")]


def _primary_damage_text(raw: object) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\bplus\b", "+", text)
    return re.split(r"\bor\b", text, maxsplit=1)[0]


def _damage_type_spans(text: str):
    pattern = "|".join(re.escape(damage_type) for damage_type in sorted(_DAMAGE_TYPES))
    for match in re.finditer(
        rf"(?<![a-z0-9])({pattern})(?![a-z0-9])",
        text,
        flags=re.IGNORECASE,
    ):
        yield match.group(1).lower(), match.start(), match.end()


def _merge_damage_components(
    components: list[_DamageComponent],
) -> list[_DamageComponent]:
    merged: list[_DamageComponent] = []
    index_by_type: dict[str, int] = {}
    for component in components:
        if not component.expression:
            continue
        key = component.damage_type
        if key not in index_by_type:
            index_by_type[key] = len(merged)
            merged.append(component)
            continue
        idx = index_by_type[key]
        existing = merged[idx]
        merged[idx] = _DamageComponent(
            expression=f"{existing.expression}+{component.expression}",
            damage_type=existing.damage_type,
        )
    return merged


def _damage_profile_summary(profile: _DamageProfile) -> str:
    parts = []
    for component in profile.components:
        type_text = f" {component.damage_type}" if component.damage_type else ""
        parts.append(f"{component.expression}{type_text}")
    return " + ".join(parts)


def _extract_damage_type(raw: object) -> str:
    text = _normalize_damage_text(raw)
    for damage_type in sorted(_DAMAGE_TYPES):
        if _contains_action_name(text, damage_type):
            return damage_type
    return ""


def _normalize_damage_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _adjust_damage_amount(
    ckpt: CheckpointFile,
    request: PlannedRoll,
    *,
    raw_amount: int,
    damage_type: str,
) -> tuple[int, list[CatIIRollDamageAdjustmentRecord]]:
    amount = max(0, int(raw_amount or 0))
    damage_type = _extract_damage_type(damage_type) or _normalize_damage_text(
        damage_type
    )
    candidates = _damage_adjustment_candidates(
        ckpt,
        request,
        damage_type,
        scope="component",
    )
    return _apply_damage_adjustments(amount, candidates, damage_type=damage_type)


def _adjust_attack_total_amount(
    request: PlannedRoll,
    *,
    raw_amount: int,
    components: list[CatIIRollDamageComponentRecord],
) -> tuple[int, list[CatIIRollDamageAdjustmentRecord]]:
    amount = max(0, int(raw_amount or 0))
    candidates: list[_DamageAdjustmentCandidate] = []
    component_types = {
        component.damage_type
        for component in components
        if component.damage_type
    }
    for adjustment in request.damage_adjustments:
        if adjustment.scope != "attack_total":
            continue
        item_type = _normalize_damage_text(adjustment.damage_type)
        if not _attack_total_adjustment_matches(item_type, component_types):
            continue
        candidates.append(_DamageAdjustmentCandidate(
            source="router",
            kind=adjustment.kind,
            damage_type=item_type,
            reason=adjustment.reason or "router-authored adjustment",
            scope="attack_total",
        ))
    damage_type = _join_damage_types(
        component.damage_type for component in components
    ) or "damage"
    return _apply_damage_adjustments(amount, candidates, damage_type=damage_type)


def _attack_total_adjustment_matches(
    candidate_type: str,
    component_types: set[str],
) -> bool:
    candidate_type = _normalize_damage_text(candidate_type)
    if candidate_type in {"all", "any", "damage"}:
        return True
    return bool(candidate_type and component_types == {candidate_type})


def _apply_damage_adjustments(
    amount: int,
    candidates: list[_DamageAdjustmentCandidate],
    *,
    damage_type: str,
) -> tuple[int, list[CatIIRollDamageAdjustmentRecord]]:
    adjustments: list[CatIIRollDamageAdjustmentRecord] = []

    immunity = [c for c in candidates if c.kind == "immunity"]
    if immunity:
        return (
            0,
            [
                _damage_adjustment_record(
                    immunity,
                    kind="immunity",
                    damage_type=damage_type,
                    before=amount,
                    after=0,
                )
            ],
        )

    double = [c for c in candidates if c.kind == "double"]
    if double:
        before = amount
        amount *= 2
        adjustments.append(_damage_adjustment_record(
            double,
            kind="double",
            damage_type=damage_type,
            before=before,
            after=amount,
        ))

    halve = [c for c in candidates if c.kind == "halve"]
    if halve:
        before = amount
        amount //= 2
        adjustments.append(_damage_adjustment_record(
            halve,
            kind="halve",
            damage_type=damage_type,
            before=before,
            after=amount,
        ))

    resistance = [c for c in candidates if c.kind == "resistance"]
    if resistance:
        before = amount
        amount //= 2
        adjustments.append(_damage_adjustment_record(
            resistance,
            kind="resistance",
            damage_type=damage_type,
            before=before,
            after=amount,
        ))

    vulnerability = [c for c in candidates if c.kind == "vulnerability"]
    if vulnerability:
        before = amount
        amount *= 2
        adjustments.append(_damage_adjustment_record(
            vulnerability,
            kind="vulnerability",
            damage_type=damage_type,
            before=before,
            after=amount,
        ))

    return amount, adjustments


def _damage_adjustment_candidates(
    ckpt: CheckpointFile,
    request: PlannedRoll,
    damage_type: str,
    *,
    scope: str,
) -> list[_DamageAdjustmentCandidate]:
    candidates: list[_DamageAdjustmentCandidate] = []
    for kind, key in (
        ("resistance", "damage_resistances"),
        ("immunity", "damage_immunities"),
        ("vulnerability", "damage_vulnerabilities"),
    ):
        for item in _target_damage_defense_entries(ckpt, request.target_id, key):
            item_type = _damage_defense_type(item)
            if not _damage_type_matches(item_type, damage_type, source="sheet"):
                continue
            name = str(item.get("name") or item.get("id") or item_type).strip()
            condition = str(item.get("condition") or "").strip()
            reason = f"{name} ({condition})" if condition else name
            candidates.append(_DamageAdjustmentCandidate(
                source="sheet",
                kind=kind,
                damage_type=item_type,
                reason=reason,
                scope="component",
            ))

    for adjustment in request.damage_adjustments:
        if adjustment.scope != scope:
            continue
        item_type = _normalize_damage_text(adjustment.damage_type)
        if not _damage_type_matches(item_type, damage_type, source="router"):
            continue
        candidates.append(_DamageAdjustmentCandidate(
            source="router",
            kind=adjustment.kind,
            damage_type=item_type or damage_type,
            reason=adjustment.reason or "router-authored adjustment",
            scope=adjustment.scope,
        ))
    return candidates


def _target_damage_defense_entries(
    ckpt: CheckpointFile,
    target_id: str,
    key: str,
) -> list[dict[str, object]]:
    character = _character_for_combat_target(ckpt, target_id)
    if character is None:
        return []
    mechanics_state = character.mechanics or {}
    defenses = mechanics_state.get("defenses")
    if not isinstance(defenses, dict):
        defenses = (
            (mechanics_state.get("dnd5e_sheet") or {})
            .get("statblock", {})
            .get("defenses", {})
        )
    if not isinstance(defenses, dict):
        return []
    entries = defenses.get(key) or []
    if not isinstance(entries, list):
        return []
    normalized = []
    for entry in entries:
        if isinstance(entry, dict):
            normalized.append(entry)
        elif isinstance(entry, str) and entry.strip():
            normalized.append({"id": entry.strip(), "name": entry.strip()})
    return normalized


def _character_for_combat_target(
    ckpt: CheckpointFile,
    target_id: str,
) -> object | None:
    combat = getattr(ckpt.session, "active_combat", None)
    character_id = target_id
    if combat is not None:
        for combatant in _combatants(combat):
            ids = {
                str(getattr(combatant, "combatant_id", "") or ""),
                str(getattr(combatant, "character_id", "") or ""),
            }
            if target_id not in ids:
                continue
            character_id = (
                str(getattr(combatant, "character_id", "") or "")
                or str(getattr(combatant, "combatant_id", "") or "")
            )
            break
    return next(
        (c for c in ckpt.characters if c.character_id == character_id),
        None,
    )


def _damage_defense_type(item: dict[str, object]) -> str:
    condition = _normalize_damage_text(item.get("condition") or "")
    if condition:
        return ""
    text = _normalize_damage_text(item.get("id") or item.get("name") or "")
    if text in {"all", "any", "damage"} or text in _DAMAGE_TYPES:
        return text
    for damage_type in sorted(_DAMAGE_TYPES):
        if not _contains_action_name(text, damage_type):
            continue
        extra_tokens = set(text.split()) - {damage_type, "damage"}
        if extra_tokens:
            return ""
        return damage_type
    return text


def _damage_type_matches(
    candidate_type: str,
    damage_type: str,
    *,
    source: str,
) -> bool:
    candidate_type = _normalize_damage_text(candidate_type)
    damage_type = _normalize_damage_text(damage_type)
    if candidate_type in {"all", "any", "damage"}:
        return True
    if not candidate_type or not damage_type:
        return False
    if candidate_type == damage_type:
        return True
    return _contains_action_name(candidate_type, damage_type)


def _damage_adjustment_record(
    candidates: list[_DamageAdjustmentCandidate],
    *,
    kind: str,
    damage_type: str,
    before: int,
    after: int,
) -> CatIIRollDamageAdjustmentRecord:
    return CatIIRollDamageAdjustmentRecord(
        source=", ".join(_unique_text(c.source for c in candidates)),
        kind=kind,
        damage_type=damage_type,
        amount_before=before,
        amount_after=after,
        reason="; ".join(_unique_text(c.reason for c in candidates)),
    )


def _damage_adjustment_ledger_text(
    adjustments: list[CatIIRollDamageAdjustmentRecord],
) -> str:
    if not adjustments:
        return ""
    parts = [
        (
            f"{adjustment.kind} {adjustment.amount_before}"
            f"->{adjustment.amount_after}"
            f" ({adjustment.source}: {adjustment.reason})"
        )
        for adjustment in adjustments
    ]
    return "; adjusted by " + "; ".join(parts)


def _unique_text(values) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _join_damage_types(values) -> str:
    return ", ".join(_unique_text(values))


def _crit_damage_expression(expression: str) -> str:
    # D&D crit support: double each damage dice group, keep flat modifiers
    # unchanged. "1d8+4" -> "2d8+4"; "1d8+1d6" -> "2d8+2d6".
    def repl(match: re.Match[str]) -> str:
        return f"{int(match.group(1)) * 2}d{match.group(2)}"

    return re.sub(r"(\d+)d(\d+)", repl, expression)


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
            "role": char.public_sheet.role,
            "location": char.location,
            "player_controlled": cid in bindings,
            "mechanics": mechanics.mechanics_summary(
                char,
                include_inventory_resources=(
                    cid == cat_ii_event.initiator_id
                ),
            ),
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
    for combatant in _combatants(combat):
        cid = str(
            getattr(combatant, "character_id", "")
            or getattr(combatant, "combatant_id", "")
            or ""
        )
        char = by_id.get(cid)
        participants.append({
            "combatant_id": str(getattr(combatant, "combatant_id", "") or cid),
            "character_id": cid,
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
            "active_effects": _combat_effect_summaries(combatant),
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
                mechanics.mechanics_summary(
                    char,
                    include_inventory_resources=(cid == actor_id),
                )
                if char is not None else {}
            ),
            "actions": _combat_action_summaries(char),
            "spellcasting": (
                _combat_spellcasting_summary(char) if cid == actor_id else {}
            ),
            "spells": _combat_spell_summaries(char) if cid == actor_id else [],
            "pending_initiating_action": (
                str(getattr(combatant, "pending_initiating_action", "") or "")
                if cid == actor_id else ""
            ),
        })

    spatial_context = dnd_spatial.combat_packet_context(combat, actor_id)
    payload = {
        "ruleset_id": ckpt.session.config.settings.ruleset_id,
        "player_roll_mode": ckpt.session.config.settings.player_roll_mode,
        "round_number": int(getattr(combat, "round_number", 1) or 1),
        "current_turn": {
            "actor_id": actor_id,
        },
        "intention": intention,
        "house_rules": [
            "Opportunity attacks are automatic for players and NPCs.",
            "Player opportunity attacks do not consume optional reaction prompts.",
            "Open optional player reaction prompts only for meaningful choices.",
            "Agents do not need roll details; expose dice only through player UI.",
        ],
        "combatants": participants,
        **spatial_context,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _combat_effect_summaries(combatant: object) -> list[dict[str, object]]:
    effects: list[dict[str, object]] = []
    for effect in list(getattr(combatant, "active_effects", []) or []):
        save = getattr(effect, "recurring_save", None)
        effects.append({
            "effect_id": str(getattr(effect, "effect_id", "") or ""),
            "name": str(getattr(effect, "name", "") or ""),
            "slug": str(getattr(effect, "slug", "") or ""),
            "originator_id": str(getattr(effect, "originator_id", "") or ""),
            "target_id": str(getattr(effect, "target_id", "") or ""),
            "conditions": list(getattr(effect, "conditions", []) or []),
            "concentration": bool(getattr(effect, "concentration", False)),
            "remaining_rounds": int(getattr(effect, "remaining_rounds", 0) or 0),
            "duration_text": str(getattr(effect, "duration_text", "") or ""),
            "break_triggers": list(getattr(effect, "break_triggers", []) or []),
            "recurring_save": (
                {
                    "ability": str(getattr(save, "ability", "") or ""),
                    "dc": int(getattr(save, "dc", 0) or 0),
                    "timing": str(getattr(save, "timing", "") or ""),
                    "ends_on": str(getattr(save, "ends_on", "") or ""),
                }
                if save is not None else None
            ),
        })
    return effects


def _combat_action_summaries(character: object | None) -> list[dict[str, object]]:
    if character is None:
        return []
    actions: list[dict[str, object]] = []
    for action in _combat_actions_for_character(character):
        if not isinstance(action, dict):
            continue
        attack = action.get("attack") or {}
        if not isinstance(attack, dict):
            attack = {}
        damage_profile = _action_damage_profile(action)
        actions.append({
            "id": str(action.get("id") or ""),
            "name": str(action.get("name") or ""),
            "attack_bonus": attack.get("bonus", ""),
            "damage": _action_damage_summary(action),
            "damage_type": damage_profile.damage_type,
            "range": str(attack.get("range") or action.get("range") or ""),
            "notes": str(action.get("notes") or ""),
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
            "duration": spell.get("duration") or {},
            "range": spell.get("range") or {},
            "target": spell.get("target") or {},
            "components": spell.get("components") or {},
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
        ends_beat=True,
        ends_beat_reason="cat_ii_resolution",
        observers=[
            ObserverEntry(
                character_id=cid,
                observation_level="d",
                routing_role="observe_only",
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
) -> DndEventRouterOutput:
    combat = getattr(ckpt.session, "active_combat", None)
    affected_ids = _combat_affected_ids(transaction, adjudication)
    observer_ids = []
    if combat is not None:
        for combatant in _combatants(combat):
            if bool(getattr(combatant, "removed", False)):
                continue
            cid = str(
                getattr(combatant, "character_id", "")
                or getattr(combatant, "combatant_id", "")
                or ""
            )
            combatant_id = str(getattr(combatant, "combatant_id", "") or "")
            if not cid:
                continue
            if (
                _combatant_defeat_state(combatant) == "active"
                or cid in affected_ids
                or combatant_id in affected_ids
            ):
                observer_ids.append(cid)
    observer_ids = _dedupe(observer_ids or [transaction.actor_id])
    visible_facts = [
        *adjudication.visible_outcome_facts,
        *dnd_combat.drain_pending_visible_facts(combat),
    ]
    notes = "; ".join(adjudication.rules_notes)
    rationale_parts = [
        part for part in (
            adjudication.mechanical_summary,
            f"Rules notes: {notes}" if notes else "",
            f"Fallback: {adjudication.fallback_reason}"
            if adjudication.fallback_reason else "",
        ) if part
    ]
    return DndEventRouterOutput(
        event_id="",
        decision_rationale=" ".join(rationale_parts) or "D&D combat adjudication.",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                feasible=adjudication.feasible
            ),
            observable_facts=[
                ObservableFact.all(fact)
                for fact in visible_facts
            ],
        ),
        requires_responders=False,
        required_responders=[],
        ends_beat=True,
        ends_beat_reason="ruleset_resolution",
        observers=[
            DndObserverEntry(
                character_id=cid,
                observation_level="d",
                routing_role=(
                    "dnd_reaction"
                    if cid in affected_ids
                    else "observe_only"
                ),
            )
            for cid in observer_ids
            if _character_exists(ckpt, cid)
        ],
        spawn=[],
        dormant=[],
        cull=[],
        interaction_mode="cat_i",
        combatant_ids=[],
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
    for delta in adjudication.effect_deltas:
        if delta.target_id:
            affected.add(delta.target_id)
    for delta in adjudication.spatial_deltas:
        if delta.target_id:
            affected.add(delta.target_id)
        if delta.character_id:
            affected.add(delta.character_id)
    return affected


def _apply_combat_spatial_deltas(
    ckpt: CheckpointFile,
    deltas: list[Any],
) -> list[str]:
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is None or not deltas:
        return []
    notes = dnd_spatial.apply_spatial_deltas(combat, deltas)
    for note in notes:
        dnd_combat.append_audit_line(combat, note)
    return notes


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


def _apply_combat_effect_deltas(
    ckpt: CheckpointFile,
    deltas: list[EffectDelta],
    *,
    default_originator_id: str,
) -> list[str]:
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is None:
        return []
    notes: list[str] = []
    for delta in deltas:
        if delta.operation == "start":
            effect = _runtime_effect_from_delta(
                delta,
                default_originator_id=default_originator_id,
            )
            dnd_combat.start_effect(ckpt.session, effect)
            if not _combat_effect_instance_exists(combat, effect):
                notes.append(
                    "Effect start skipped for "
                    f"{_effect_delta_display_name(delta)}: "
                    f"{_effect_delta_selector(delta)}."
                )
        elif delta.operation == "end":
            ended = dnd_combat.end_effect(
                ckpt.session,
                effect_id=delta.effect_id,
                target_id=delta.target_id,
                slug=delta.slug,
                originator_id=delta.originator_id,
                reason=delta.reason,
            )
            if not ended:
                notes.append(
                    "Effect end skipped for "
                    f"{_effect_delta_display_name(delta)}: "
                    f"{_effect_delta_selector(delta)}."
                )
        elif delta.operation == "update":
            updated = dnd_combat.update_effect(
                ckpt.session,
                effect_id=delta.effect_id,
                target_id=delta.target_id,
                slug=delta.slug,
                originator_id=delta.originator_id,
                name=delta.name or None,
                conditions=list(delta.conditions) if delta.conditions else None,
                concentration=delta.concentration if delta.concentration else None,
                duration_kind=(
                    delta.duration_kind
                    if delta.duration_kind != "until_removed" else None
                ),
                duration_amount=delta.duration_amount
                if delta.duration_amount else None,
                remaining_rounds=delta.remaining_rounds
                if delta.remaining_rounds else None,
                duration_text=delta.duration_text or None,
                break_triggers=list(delta.break_triggers)
                if delta.break_triggers else None,
                recurring_save=delta.recurring_save,
                reason=delta.reason,
            )
            if updated is None:
                notes.append(
                    "Effect update skipped for "
                    f"{_effect_delta_display_name(delta)}: "
                    f"{_effect_delta_selector(delta)}."
                )
    return notes


def _combat_effect_instance_exists(
    combat: object,
    effect: DndRuntimeEffect,
) -> bool:
    for combatant in _combatants(combat):
        if any(existing is effect for existing in combatant.active_effects):
            return True
    return False


def _effect_delta_display_name(delta: EffectDelta) -> str:
    return (
        delta.name.strip()
        or delta.slug.strip()
        or delta.effect_id.strip()
        or "effect"
    )


def _effect_delta_selector(delta: EffectDelta) -> str:
    parts = []
    for label, value in (
        ("effect_id", delta.effect_id),
        ("target_id", delta.target_id),
        ("slug", delta.slug),
        ("originator_id", delta.originator_id),
    ):
        text = str(value or "").strip()
        if text:
            parts.append(f"{label}={text!r}")
    return ", ".join(parts) if parts else "no selector"


def _runtime_effect_from_delta(
    delta: EffectDelta,
    *,
    default_originator_id: str,
) -> DndRuntimeEffect:
    return DndRuntimeEffect(
        effect_id=delta.effect_id,
        name=delta.name or delta.slug,
        slug=delta.slug or _slug(delta.name),
        source_type=delta.source_type,
        source_id=delta.source_id,
        originator_id=delta.originator_id or default_originator_id,
        target_id=delta.target_id,
        conditions=list(delta.conditions),
        concentration=delta.concentration,
        duration_kind=delta.duration_kind,
        duration_amount=delta.duration_amount,
        remaining_rounds=delta.remaining_rounds,
        duration_text=delta.duration_text,
        break_triggers=list(delta.break_triggers),
        recurring_save=delta.recurring_save,
        metadata={"reason": delta.reason} if delta.reason else {},
    )


def _sync_combat_effects(ckpt: CheckpointFile) -> None:
    dnd_combat.sync_combat_effects_to_characters(
        getattr(ckpt.session, "active_combat", None),
        ckpt.characters,
    )


def _apply_condition_delta(
    ckpt: CheckpointFile,
    delta: CombatStateDelta,
) -> None:
    combat = getattr(ckpt.session, "active_combat", None)
    if combat is None or not delta.condition:
        return
    for combatant in _combatants(combat):
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


def _roll_label(
    request: PlannedRoll,
    character: object | None = None,
) -> str:
    if request.kind == "attack_roll" and request.action_id:
        action = _find_action(
            {"actions": _combat_actions_for_character(character)},
            request.action_id,
            reason=request.reason,
        )
        if action is not None:
            name = str(action.get("name") or "").strip()
            if name:
                return f"Attack ({name})"
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
