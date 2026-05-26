from __future__ import annotations

import json
import re
import uuid

from app.engine import dnd_cat_ii as cat
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_cat_ii import (
    DndCombatManagerAdjudication,
    DndCombatTurnPlan,
)
from app.schemas.event_router import EventRouterOutput
from app.schemas.state import CatIIRollTransaction


COMBAT_MANAGER_PLAN_MAX_TOKENS = 20000
COMBAT_MANAGER_FINALIZE_MAX_TOKENS = 20000


class DndCombatResolver:
    """D&D initiative-scene turn resolver.

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
        content_context_records: list[str] | None = None,
    ) -> EventRouterOutput:
        packet = cat._build_combat_packet(
            ckpt,
            actor_id,
            intention,
            content_context_records=content_context_records,
        )
        event_id = f"cmb_{uuid.uuid4().hex[:12]}"
        plan = await self._plan_turn(packet)
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
        content_context_records: list[str] | None = None,
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
        if content_context_records:
            _merge_content_context_records(
                transaction.context,
                content_context_records,
            )
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
        cat._apply_combat_resource_spends(ckpt, transaction)
        packet = cat.build_combat_finalization_packet(transaction)
        adjudication = await self._finalize(
            packet,
            transaction.ledger_lines,
            cat.planned_actions_block(transaction),
        )
        cat._validate_and_apply_readied_releases(ckpt, transaction, adjudication)
        cat._apply_combat_damage_records(ckpt, transaction)
        cat._apply_combat_state_deltas(
            ckpt,
            adjudication.combat_state_deltas,
            transaction=transaction,
        )
        effect_notes = cat._apply_combat_effect_deltas(
            ckpt,
            adjudication.effect_deltas,
            default_originator_id=transaction.actor_id,
            transaction=transaction,
        )
        spatial_notes = cat._apply_combat_spatial_deltas(
            ckpt,
            adjudication.spatial_deltas,
        )
        cat.dnd_combat.record_router_observed_facts(
            ckpt.session.active_combat,
            getattr(adjudication, "router_observed_facts", []),
        )
        adjudication.rules_notes.extend(effect_notes)
        adjudication.rules_notes.extend(spatial_notes)
        cat._sync_combat_effects(ckpt)
        cat._auto_end_if_spawned_hostiles_defeated(ckpt, adjudication)
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

    async def _plan_turn(self, packet: str) -> DndCombatTurnPlan:
        messages = self.prompt_mgr.render_messages(
            "dnd_combat_manager",
            phase="PLAN_TURN",
            combat_action_packet=packet,
            roll_ledger_block="No rolls have been made yet.",
            planned_actions_block="No planned actions yet.",
        )
        response = await self.client.complete(
            role="dnd_combat_manager",
            messages=messages,
            response_model=DndCombatTurnPlan,
            temperature=0.2,
            max_tokens=COMBAT_MANAGER_PLAN_MAX_TOKENS,
            cache=True,
            compact=False,
        )
        return response.parsed

    async def _finalize(
        self,
        packet: str,
        ledger_lines: list[str],
        planned_actions_block: str,
    ) -> DndCombatManagerAdjudication:
        messages = self.prompt_mgr.render_messages(
            "dnd_combat_manager",
            phase="FINALIZE_OUTCOME",
            combat_action_packet=packet,
            roll_ledger_block="\n".join(ledger_lines) or "No rolls were made.",
            planned_actions_block=planned_actions_block,
        )
        response = await self.client.complete(
            role="dnd_combat_manager",
            messages=messages,
            response_model=DndCombatManagerAdjudication,
            temperature=0.2,
            max_tokens=COMBAT_MANAGER_FINALIZE_MAX_TOKENS,
            cache=True,
            compact=False,
        )
        adjudication = response.parsed
        _scrub_visible_bookkeeping(adjudication)
        _scrub_private_outcome_leaks(adjudication)
        return adjudication


def _merge_content_context_records(
    context: dict[str, object],
    records: list[str],
) -> None:
    existing = context.get("content_context")
    merged = cat._safe_content_context_records([
        str(record) for record in (existing if isinstance(existing, list) else [])
    ])
    seen = set(merged)
    for text in cat._safe_content_context_records(records):
        if text and text not in seen:
            merged.append(text)
            seen.add(text)
    if merged:
        context["content_context"] = merged


_VISIBLE_BOOKKEEPING_TERMS = (
    "saving throw",
    "strength save",
    "dexterity save",
    "constitution save",
    "intelligence save",
    "wisdom save",
    "charisma save",
    "opportunity attack",
    "hit points",
    "spell slot",
    "concentrating",
    "dash action",
    "roll ledger",
    "no attacks",
    "no attack was made",
    "no rolls",
    "no saving throw",
    "no damage",
    "no damage was",
)

_PRIVATE_OUTCOME_LEAK_WORDS = {
    "illusion",
    "illusory",
    "phantasm",
    "phantasmal",
    "hallucination",
    "imagined",
    "imaginary",
    "fake",
}

_VISIBLE_SAVE_RESULT_RE = re.compile(
    r"\b(?:fails|failed|succeeds on|succeeds|succeeded on)\s+"
    r"(?:the |a |an )?[^.;,]*?\s+"
    r"(?:strength|dexterity|constitution|intelligence|wisdom|charisma)\s+"
    r"save(?:\s+and\s+)?",
    re.IGNORECASE,
)


def _scrub_visible_bookkeeping(
    adjudication: DndCombatManagerAdjudication,
) -> None:
    rewritten = [
        _rewrite_visible_bookkeeping(fact)
        for fact in adjudication.visible_outcome_facts
    ]
    adjudication.visible_outcome_facts = [
        fact for fact in rewritten
        if not _text_has_visible_bookkeeping(fact)
    ]
    if not adjudication.visible_outcome_facts:
        adjudication.visible_outcome_facts = ["The action resolves."]


def _rewrite_visible_bookkeeping(text: str) -> str:
    cleaned = _VISIBLE_SAVE_RESULT_RE.sub("", str(text or ""))
    cleaned = re.sub(r"\b[Dd]ashes\b", "rushes", cleaned)
    cleaned = re.sub(
        r"\buses?\s+(?:the\s+)?Dash action\b",
        "rushes",
        cleaned,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split())


def _text_has_visible_bookkeeping(text: str) -> bool:
    lower = " ".join(str(text or "").lower().split())
    return any(term in lower for term in _VISIBLE_BOOKKEEPING_TERMS)


def _scrub_private_outcome_leaks(
    adjudication: DndCombatManagerAdjudication,
) -> None:
    private_terms = _private_outcome_terms(adjudication)
    if not private_terms:
        return
    adjudication.visible_outcome_facts = [
        fact for fact in adjudication.visible_outcome_facts
        if not _text_leaks_private_outcome(fact, private_terms)
    ]
    adjudication.router_observed_facts = [
        fact for fact in adjudication.router_observed_facts
        if not _text_leaks_private_outcome(
            f"{fact.fact} {fact.reason}",
            private_terms,
        )
    ]
    if not adjudication.visible_outcome_facts:
        adjudication.visible_outcome_facts = ["The spell takes effect."]


def _private_outcome_terms(adjudication: DndCombatManagerAdjudication) -> set[str]:
    terms: set[str] = set()
    for fact in adjudication.private_outcome_facts:
        terms.update(
            token for token in re.findall(r"[a-z0-9]+", fact.text.lower())
            if len(token) >= 5
            and token not in {
                "about",
                "above",
                "across",
                "behind",
                "below",
                "completely",
                "north",
                "south",
                "east",
                "west",
                "feels",
                "looks",
                "sound",
                "sounds",
                "there",
            }
        )
    return terms


def _text_leaks_private_outcome(text: str, private_terms: set[str]) -> bool:
    lower = " ".join(str(text or "").lower().split())
    tokens = set(re.findall(r"[a-z0-9]+", lower))
    if tokens.intersection(_PRIVATE_OUTCOME_LEAK_WORDS):
        return True
    if "not real" in lower:
        return True
    if private_terms.intersection(tokens):
        return True
    if "as if" in lower:
        return True
    return False
