from __future__ import annotations

import json
import logging

from app.engine import mechanics
from app.engine.dice import roll_d20_check
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication
from app.schemas.rules_arbitrator import RollPlan, RulesAdjudication
from app.schemas.state import OpenCatIIEvent

logger = logging.getLogger(__name__)


def cat_ii_rules_arbitrator_enabled(ckpt: CheckpointFile) -> bool:
    settings = ckpt.session.config.settings
    return settings.cat_ii_resolution_mode == "rules_arbitrator"


class RulesArbitrator:
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
        plan = await self._plan_rolls(ckpt, packet)
        ledger_lines = _execute_roll_plan(ckpt, plan)
        adjudication = await self._finalize(ckpt, packet, ledger_lines)
        result = _compile_event_router_output(
            ckpt, cat_ii_event, adjudication, ledger_lines
        )
        _queue_router_state_change(ckpt, result)
        return result

    async def _plan_rolls(
        self,
        ckpt: CheckpointFile,
        packet: str,
    ) -> RollPlan:
        messages = self.prompt_mgr.render_messages(
            "rules_arbitrator",
            phase="PLAN_ROLLS",
            contested_action_packet=packet,
            roll_ledger_block="No rolls have been made yet.",
        )
        response = await self.client.complete(
            role="rules_arbitrator",
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
        ckpt: CheckpointFile,
        packet: str,
        ledger_lines: list[str],
    ) -> RulesAdjudication:
        messages = self.prompt_mgr.render_messages(
            "rules_arbitrator",
            phase="FINALIZE_OUTCOME",
            contested_action_packet=packet,
            roll_ledger_block="\n".join(ledger_lines) or "No rolls were made.",
        )
        response = await self.client.complete(
            role="rules_arbitrator",
            messages=messages,
            response_model=RulesAdjudication,
            temperature=0.2,
            max_tokens=3000,
            cache=True,
            compact=False,
        )
        return response.parsed


def _build_contested_packet(
    ckpt: CheckpointFile,
    cat_ii_event: OpenCatIIEvent,
) -> str:
    by_id = {c.character_id: c for c in ckpt.characters}
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
            "mechanics": mechanics.mechanics_summary(char),
        })

    payload = {
        "ruleset_id": ckpt.session.config.settings.ruleset_id,
        "initiator_id": cat_ii_event.initiator_id,
        "initiator_intention": cat_ii_event.initiator_intention,
        "required_responders": cat_ii_event.required_responders,
        "collected_intentions": cat_ii_event.collected_intentions,
        "swept_responders": cat_ii_event.swept_responders,
        "opening_observable_facts": cat_ii_event.opening_observable_facts,
        "participants": participants,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _execute_roll_plan(ckpt: CheckpointFile, plan: RollPlan) -> list[str]:
    if not plan.needs_rolls:
        reason = plan.no_roll_reason.strip()
        return [f"No rolls: {reason}"] if reason else []
    by_id = {c.character_id: c for c in ckpt.characters}
    ledger_lines: list[str] = []
    for request in plan.roll_requests:
        character = by_id.get(request.actor_id)
        modifier = mechanics.roll_modifier(character, request)
        result = roll_d20_check(
            roll_id=request.roll_id,
            modifier=modifier,
            actor_id=request.actor_id,
            reason=request.reason,
            advantage_state=request.advantage_state,
        )
        dc_part = f", DC {request.dc}" if request.dc else ""
        opposed_part = (
            f", opposed by {request.opposed_by}" if request.opposed_by else ""
        )
        ledger_lines.append(
            f"{result.roll_id}: {request.actor_id} {request.kind} "
            f"({request.ability}"
            f"{', ' + request.skill if request.skill else ''}) "
            f"rolled {result.detail} = {result.total}"
            f"{dc_part}{opposed_part}; reason: {request.reason}"
        )
    return ledger_lines


def _compile_event_router_output(
    ckpt: CheckpointFile,
    cat_ii_event: OpenCatIIEvent,
    adjudication: RulesAdjudication,
    ledger_lines: list[str],
) -> EventRouterOutput:
    observer_ids = _observer_ids(cat_ii_event)
    notes = "; ".join(adjudication.rules_notes)
    ledger = " | ".join(ledger_lines)
    rationale_parts = [
        part for part in (
            adjudication.mechanical_summary,
            f"Rolls: {ledger}" if ledger else "",
            f"Rules notes: {notes}" if notes else "",
            f"Fallback: {adjudication.fallback_reason}"
            if adjudication.fallback_reason else "",
        ) if part
    ]
    return EventRouterOutput(
        event_id="",
        decision_rationale=" ".join(rationale_parts) or "Rules adjudication.",
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
            "Rules adjudication resolved Cat II: " + " / ".join(facts)
        )


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
