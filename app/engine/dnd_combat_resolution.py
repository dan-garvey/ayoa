from __future__ import annotations

import json
import uuid

from app.engine import dnd_cat_ii as cat
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_cat_ii import RollPlan, RulesAdjudication
from app.schemas.event_router import EventRouterOutput
from app.schemas.state import CatIIRollTransaction


class DndCombatResolver:
    """Router-owned D&D combat turn resolver.

    Combat shares Cat II's dice transaction ledger, but this module owns the
    active-combat flow so generic Cat II resolution stays smaller.
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
        packet = cat._build_combat_packet(ckpt, actor_id, intention)
        event_id = f"cmb_{uuid.uuid4().hex[:12]}"
        plan = await self._plan_rolls(packet)
        transaction = cat._create_combat_transaction(
            ckpt=ckpt,
            event_id=event_id,
            actor_id=actor_id,
            intention=intention,
            packet=json.loads(packet),
            plan=plan,
        )
        cat._execute_available_rolls(ckpt, transaction)
        return await self._resolve_transaction(ckpt, transaction)

    async def continue_combat_transaction(
        self,
        *,
        ckpt: CheckpointFile,
        event_id: str,
    ) -> EventRouterOutput:
        transaction = cat._find_transaction(ckpt, event_id)
        if transaction is None or transaction.source != "combat":
            raise ValueError(f"No D&D combat roll transaction for {event_id}.")
        if transaction.status == "finalized":
            raise ValueError(f"D&D combat roll transaction {event_id} is finalized.")
        if transaction.status == "awaiting_player_rolls":
            if cat._pending_player_rolls(transaction):
                cat._pin_pending_player_rolls(ckpt, transaction)
                raise cat.DndCatIIRollsPending(transaction)
            transaction.status = "ready_to_finalize"
            transaction.updated_at = cat._utcnow_iso()
        elif transaction.status in {"planning", "planned"}:
            cat._execute_available_rolls(ckpt, transaction)
        return await self._resolve_transaction(ckpt, transaction)

    async def _resolve_transaction(
        self,
        ckpt: CheckpointFile,
        transaction: CatIIRollTransaction,
    ) -> EventRouterOutput:
        if cat._pending_player_rolls(transaction):
            transaction.status = "awaiting_player_rolls"
            transaction.updated_at = cat._utcnow_iso()
            cat._pin_pending_player_rolls(ckpt, transaction)
            raise cat.DndCatIIRollsPending(transaction)

        cat._execute_combat_damage_rolls(ckpt, transaction)
        packet = json.dumps(transaction.context, indent=2, sort_keys=True)
        adjudication = await self._finalize(packet, transaction.ledger_lines)
        cat._apply_combat_damage_records(ckpt, transaction)
        cat._apply_combat_state_deltas(ckpt, adjudication.combat_state_deltas)
        effect_notes = cat._apply_combat_effect_deltas(
            ckpt,
            adjudication.effect_deltas,
            default_originator_id=transaction.actor_id,
        )
        spatial_notes = cat._apply_combat_spatial_deltas(
            ckpt,
            adjudication.spatial_deltas,
        )
        adjudication.rules_notes.extend(effect_notes)
        adjudication.rules_notes.extend(spatial_notes)
        cat._sync_combat_effects(ckpt)
        result = cat._compile_combat_router_output(
            ckpt=ckpt,
            transaction=transaction,
            adjudication=adjudication,
        )
        if adjudication.combat_status == "ended":
            cat._end_combat_after_adjudication(ckpt, result)
        transaction.status = "finalized"
        transaction.final_event_id = result.event_id
        transaction.updated_at = cat._utcnow_iso()
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
