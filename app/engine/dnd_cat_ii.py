from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from app.engine import dice, mechanics
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication
from app.schemas.dnd_cat_ii import PlannedRoll, RollPlan, RulesAdjudication
from app.schemas.state import (
    CatIIRollRecord,
    CatIIRollTransaction,
    OpenCatIIEvent,
    SlotEntry,
)

logger = logging.getLogger(__name__)


class DndCatIIRollsPending(RuntimeError):
    """Raised when D&D Cat II resolution must pause for human dice UI."""

    def __init__(self, transaction: CatIIRollTransaction):
        self.transaction = transaction
        super().__init__(
            f"Cat II event {transaction.event_id} is awaiting player rolls."
        )


def dnd_cat_ii_router_enabled(ckpt: CheckpointFile) -> bool:
    settings = ckpt.session.config.settings
    return settings.cat_ii_resolution_mode in {
        "dnd5e_router",
        # Back-compat for saves/settings created by the first D&D slice.
        "rules_arbitrator",
    }


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
        _queue_router_state_change(ckpt, result)
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


def _queue_router_state_change(
    ckpt: CheckpointFile,
    routed: EventRouterOutput,
) -> None:
    facts = [
        fact.text for fact in routed.canonical_event.observable_facts
        if fact.text
    ]
    if facts:
        ckpt.session.pending_router_state_changes.append(
            "D&D Cat II resolved: " + " / ".join(facts)
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
