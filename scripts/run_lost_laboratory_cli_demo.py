#!/usr/bin/env python3
"""Live CLI demo for the Lost Laboratory imported-content startup slice.

The script builds a temporary story from the reviewed private import artifacts,
drives the same CLIState command handlers used by scripts/play.py, and writes a
succinct report under app/storage/playtest_reports. It uses real LLM calls.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import asyncio
import io
import json
import logging
import os
import shutil
import sqlite3
import sys
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine import dnd_combat
from app.engine.content_pack_compiler import (
    SCHEMA_VERSION as CONTENT_PACK_SCHEMA_VERSION,
)
from app.bot.engine_bridge import EngineBridge
from app.llm.config import LIVE_PLAY_REQUIRED_ROLES, LLMConfig
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterStatus,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content import (
    ContentFrontState,
    ContentKnowledgeEntityState,
    ContentPackState,
    PendingContentSignal,
)
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.state import (
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)
from scripts.play import CLIState


PACK_ARTIFACT_DIR = REPO_ROOT / "private_extractions/lost_laboratory_of_kwalish"
CATALOG_PATH = PACK_ARTIFACT_DIR / "manual_semantic_catalog_startup_reviewed_v1.json"
SEED_INPUTS_PATH = PACK_ARTIFACT_DIR / "startup_checkpoint_seed_inputs_v1.json"
COMPILED_PACK_PATH = (
    REPO_ROOT
    / "private_extractions/compiled/lost_laboratory_kwalish_startup_reviewed_v1.sqlite"
)
REPORT_ROOT = REPO_ROOT / "app/storage/playtest_reports"
STORY_ID = "lost_laboratory_kwalish_cli_demo"
PLAYER_ID = "pc_expedition_leader"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = REPORT_ROOT / f"lost_laboratory_cli_demo_{TS}"
JSON_PATH = RUN_DIR / "report.json"
MD_PATH = RUN_DIR / "report.md"
LOG_PATH = RUN_DIR / "run.log"
ISSUES_FIXED = [
    (
        "Demo runner imported the content-pack schema constant from the wrong "
        "module.",
        "Changed the import to use app.engine.content_pack_compiler, which is "
        "where the compiler/runtime schema version is defined.",
    ),
    (
        "Content-manager live calls were truncated before JSON parsing with "
        "OpenAI incomplete max_output_tokens errors.",
        "Raised the content-manager completion cap to 8000 tokens and set the "
        "content-manager OpenAI reasoning default to low so lookup/ref "
        "selection does not consume router-grade hidden reasoning budget.",
    ),
    (
        "Deep playtest combat failed because the manual demo checkpoint gave "
        "characters HP/AC but no listed attack sources for the combat manager "
        "to select.",
        "Added minimal reviewed-demo combat scaffolding to the manual "
        "checkpoint: the player has a longsword action and the two expedition "
        "companions have simple field-ready actions with HP/AC.",
    ),
    (
        "Content-manager candidate hints could fail the whole turn when a "
        "non-authoritative related_content_refs token had a typoed pack id.",
        "Kept validation strict for required router knowledge, knowledge-map "
        "updates, and agent broadcasts, but now strips invalid related refs "
        "from router turn-candidate hints.",
    ),
    (
        "Content-manager turn candidates included dormant imported NPCs; the "
        "router then tried to spawn those existing ids when it wanted to bring "
        "them into the scene.",
        "Restricted router turn-candidate hints to active roster characters. "
        "Dormant/off-stage entities still remain in the knowledge map, but "
        "they are not advertised as immediate router turn candidates yet.",
    ),
    (
        "Route playtest latency was paying for content-manager calls on every "
        "router pass, including cycles where deterministic pending content or "
        "recently known refs were already enough.",
        "Added a session-level content-manager refresh interval so route "
        "cycles still drain pending/deterministic content but only call the "
        "manager every few eligible cycles.",
    ),
    (
        "Extended route exploration could fail when the OpenAI event-router "
        "call hit the 5000-token output cap before returning structured JSON.",
        "Raised the shared event-router output cap to 8000 tokens so long "
        "medium-reasoning router calls have room for hidden reasoning plus "
        "the visible schema response.",
    ),
    (
        "A D&D combat-start router output could include the same creature in "
        "both combatant_spawns and the generic spawn list, causing the "
        "orchestrator to materialize the combatant and then try to spawn the "
        "same id again.",
        "After combatant_spawns are materialized for D&D combat, duplicate "
        "generic spawn requests for those ids are stripped before normal "
        "roster side effects run.",
    ),
]
SCENARIO_COMMANDS = {
    "startup": [
        f"/story start {STORY_ID}",
        "/characters",
        f"/join {PLAYER_ID}",
        "/begin",
        (
            "I ask the Cartophile what terms he requires before he "
            "releases the maps, and which custodian he recommends."
        ),
        (
            "I turn to Garret and ask him to make his case for joining "
            "as custodian, including what risks he thinks the route hides."
        ),
    ],
    "deep": [
        f"/story start {STORY_ID}",
        "/characters",
        f"/join {PLAYER_ID}",
        "/begin",
        (
            "I ask the Cartophile what terms he requires before he "
            "releases the maps, and which custodian he recommends."
        ),
        (
            "I accept the return-documentation terms, appoint Garret as "
            "custodian of the maps, and ask Gearbox to accompany us as a "
            "technical specialist. Then I ask the Cartophile to release "
            "the route materials for the wooded foothills and cave approach."
        ),
        (
            "I confirm the return-documentation terms are fair. We leave at "
            "first light and follow the maps toward the Barrier Peaks route, "
            "comparing ridgelines and watching for the cave approach."
        ),
        (
            "At the wooded foothills, I scout ahead carefully with the map "
            "open, checking loose rock, tracks, and sightlines before we move "
            "toward the cave approach."
        ),
        (
            "A hostile mountain predator bursts from the rocks and charges "
            "Garret; I draw my longsword and intercept it."
        ),
        "If the predator is still fighting, I attack it with my longsword.",
    ],
    "field": [
        f"/story start {STORY_ID}",
        "/characters",
        f"/join {PLAYER_ID}",
        "/begin",
        (
            "At the wooded foothills, I scout ahead carefully with the map "
            "open, checking loose rock, tracks, and sightlines before we move "
            "toward the cave approach."
        ),
        (
            "A hostile mountain predator bursts from the rocks and charges "
            "Garret; I draw my longsword and intercept it."
        ),
        "If the predator is still fighting, I attack it with my longsword.",
    ],
    "soak": [
        f"/story start {STORY_ID}",
        "/characters",
        f"/join {PLAYER_ID}",
        "/begin",
        (
            "At the wooded foothills, I scout ahead carefully with the map "
            "open, checking loose rock, tracks, and sightlines before we move "
            "toward the cave approach."
        ),
        (
            "A hostile mountain predator bursts from the rocks and charges "
            "Garret; I draw my longsword and intercept it."
        ),
        "If the predator is still fighting, I attack it with my longsword.",
    ],
}
SOAK_COMBAT_COMMANDS = (
    (
        "If the predator is still standing, I press the attack with my "
        "longsword; otherwise I check Garret and Gearbox for injuries and "
        "secure the route folios."
    ),
    (
        "I keep myself between the predator and the expedition, striking it "
        "with my longsword if it is still a threat."
    ),
    (
        "I look for an opening, call the target clearly for Garret and "
        "Gearbox, and attack the predator again."
    ),
    (
        "I hold the line at the rocky approach and make another measured "
        "longsword attack if the beast remains in the fight."
    ),
)
SOAK_EXPLORATION_COMMANDS = (
    (
        "I take a minute to check Garret and Gearbox for injuries, then "
        "recover and sort the route folios."
    ),
    (
        "I search the cleft and nearby stones for tracks, scat, claw marks, "
        "or signs that the predator had a den nearby."
    ),
    (
        "I compare the foothill ridges against the Cartophile's notes and "
        "mark the safest path forward."
    ),
    (
        "I ask Garret what dangers on the route match this ambush, and ask "
        "Gearbox what equipment we should recheck before pressing on."
    ),
    (
        "I advance to the best nearby overlook and study the cave approach "
        "before committing the group."
    ),
    (
        "I organize our marching order, with Gearbox protected in the middle "
        "and Garret watching the rear."
    ),
    (
        "I inspect the predator's approach path to decide whether this was a "
        "random attack or a creature guarding something."
    ),
    (
        "I listen for water, machinery, wind movement, or voices from deeper "
        "in the rocks."
    ),
    (
        "I pause to update our route notes for the Cartophile's required "
        "return report."
    ),
    (
        "We continue cautiously toward the cave approach, stopping whenever "
        "the map details no longer match the terrain."
    ),
)


@dataclass(frozen=True)
class RefCard:
    ref: str
    content_hash: str
    kind: str
    summary: str


def _weapon_action(
    action_id: str,
    name: str,
    *,
    attack_bonus: int,
    damage: str,
    damage_type: str,
    notes: str = "",
    attack_range: str = "5 ft",
) -> dict[str, Any]:
    return {
        "id": action_id,
        "name": name,
        "kind": "attack",
        "attack": {
            "bonus": attack_bonus,
            "damage": f"{damage} {damage_type}",
            "range": attack_range,
        },
        "damage": [{"formula": damage, "damage_type": damage_type}],
        "notes": notes,
    }


def _dnd_mechanics(
    *,
    level: int,
    proficiency_bonus: int,
    ability_scores: dict[str, int],
    armor_class: int,
    hp: int,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "ruleset_id": "dnd5e_basic",
        "level": level,
        "proficiency_bonus": proficiency_bonus,
        "ability_scores": {
            "str": ability_scores.get("str", 10),
            "dex": ability_scores.get("dex", 10),
            "con": ability_scores.get("con", 10),
            "int": ability_scores.get("int", 10),
            "wis": ability_scores.get("wis", 10),
            "cha": ability_scores.get("cha", 10),
        },
        "skill_proficiencies": [],
        "saving_throw_proficiencies": [],
        "armor_class": armor_class,
        "hit_points": {"current": hp, "max": hp, "temporary": 0},
        "conditions": [],
        "resources": {},
        "dnd5e_sheet": {
            "statblock": {
                "actions": actions,
                "spellcasting": {},
                "defenses": {},
            },
        },
        "raw": {},
    }


def _npc_demo_mechanics(character_id: str) -> dict[str, Any]:
    if character_id == "npc_garret":
        return _dnd_mechanics(
            level=3,
            proficiency_bonus=2,
            ability_scores={
                "str": 12,
                "dex": 14,
                "con": 12,
                "int": 11,
                "wis": 14,
                "cha": 10,
            },
            armor_class=13,
            hp=20,
            actions=[
                _weapon_action(
                    "short_blade",
                    "Short Blade",
                    attack_bonus=4,
                    damage="1d6+2",
                    damage_type="piercing",
                    notes=(
                        "Demo combat scaffold for the reviewed startup slice; "
                        "represents Garret's practical field knife."
                    ),
                ),
            ],
        )
    if character_id == "npc_gearbox":
        return _dnd_mechanics(
            level=3,
            proficiency_bonus=2,
            ability_scores={
                "str": 13,
                "dex": 12,
                "con": 14,
                "int": 15,
                "wis": 10,
                "cha": 9,
            },
            armor_class=12,
            hp=22,
            actions=[
                _weapon_action(
                    "weighted_wrench",
                    "Weighted Wrench",
                    attack_bonus=3,
                    damage="1d6+1",
                    damage_type="bludgeoning",
                    notes=(
                        "Demo combat scaffold for the reviewed startup slice; "
                        "represents Gearbox's tool used defensively."
                    ),
                ),
            ],
        )
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cards() -> dict[str, RefCard]:
    if not COMPILED_PACK_PATH.exists():
        raise FileNotFoundError(f"compiled pack not found: {COMPILED_PACK_PATH}")
    with sqlite3.connect(COMPILED_PACK_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT ref, content_hash, kind, summary
            FROM content_cards
            WHERE gate_status = 'runtime_ready'
            ORDER BY ref
            """
        ).fetchall()
    return {
        row["ref"]: RefCard(
            ref=row["ref"],
            content_hash=row["content_hash"],
            kind=row["kind"],
            summary=row["summary"],
        )
        for row in rows
    }


def _compact_ref(pack_id: str, ref: str, cards: dict[str, RefCard]) -> str:
    card = cards[ref]
    return f"{pack_id}:{ref}@{card.content_hash}"


def _refs_for_actor(
    actor: dict[str, Any],
    context_slice: dict[str, Any],
) -> list[str]:
    refs = [
        actor["ref"],
        actor.get("agent_context_slice_ref", ""),
        *actor.get("home_location_refs", []),
        *actor.get("front_refs", []),
        *actor.get("knowledge_channel_refs", []),
        *context_slice.get("local_context_refs", []),
        *context_slice.get("graph_edge_refs", []),
    ]
    return [ref for ref in dict.fromkeys(refs) if ref]


def _npc_character(
    *,
    actor: dict[str, Any],
    context_slice: dict[str, Any],
    pack_id: str,
    cards: dict[str, RefCard],
) -> CharacterRecord:
    character_id = actor.get("character_id_hint") or actor["ref"].replace(".", "_")
    display_name = {
        "npc_cartophile": "The Cartophile",
        "npc_ctenmiir": "Ctenmiir",
        "npc_garret": "Garret Levistusson",
        "npc_gearbox": "Gearbox",
        "npc_mary": "Mary Greymalkin",
    }.get(character_id, character_id.replace("_", " ").title())
    active = character_id in {"npc_cartophile", "npc_garret", "npc_gearbox"}
    known_refs = [
        _compact_ref(pack_id, ref, cards)
        for ref in _refs_for_actor(actor, context_slice)
        if ref in cards
    ]
    beliefs = "; ".join(context_slice.get("beliefs", []))
    uncertainties = "; ".join(context_slice.get("uncertainties", []))
    boundaries = "; ".join(context_slice.get("hard_boundaries", []))
    known_context = " ".join(
        part
        for part in (
            context_slice.get("known_context", ""),
            f"Beliefs: {beliefs}." if beliefs else "",
            f"Uncertainties: {uncertainties}." if uncertainties else "",
            f"Boundaries: {boundaries}." if boundaries else "",
            f"Reviewed content refs known to this agent: {', '.join(known_refs)}.",
        )
        if part
    )
    secrets = []
    private_text = context_slice.get("private_state", "")
    if private_text:
        secrets.append(private_text)
    constraints = actor.get("constraints", [])
    if constraints:
        secrets.append("Constraints: " + "; ".join(constraints))
    return CharacterRecord(
        character_id=character_id,
        name=display_name,
        status=CharacterStatus.active if active else CharacterStatus.dormant,
        location="loc.cartophile_collection",
        is_playable=False,
        agent_tier=CharacterAgentTier.standard,
        public_sheet=PublicSheet(
            role=actor.get("actor_kind", "npc").replace("_", " "),
            appearance="A reviewed startup NPC from the expedition launch slice.",
            faction="Lost Laboratory expedition",
        ),
        private_state=PrivateState(
            goals=actor.get("goals", []),
            current_objectives=context_slice.get("current_agenda", []),
            secrets=secrets,
            intentions_enabled=character_id in {"npc_cartophile", "npc_garret"},
            tick_cues=actor.get("initiative_triggers", []),
        ),
        backstory=actor.get("summary", ""),
        personality=(
            "Play the reviewed startup role concretely. Keep pressure social "
            "and contractual unless the router frames a contested escalation."
        ),
        known_context=known_context,
        mechanics=_npc_demo_mechanics(character_id),
    )


def _player_character() -> CharacterRecord:
    return CharacterRecord(
        character_id=PLAYER_ID,
        name="Demo Expedition Leader",
        status=CharacterStatus.active,
        location="loc.cartophile_collection",
        is_playable=True,
        agent_tier=CharacterAgentTier.standard,
        public_sheet=PublicSheet(
            role="player character expedition lead",
            appearance="Travel-ready adventurer with notebook, packs, and a cautious eye.",
            faction="Lost Laboratory expedition",
        ),
        private_state=PrivateState(
            goals=["Assemble a viable expedition and keep the party alive."],
            current_objectives=[
                "Understand the patron's terms.",
                "Choose a trustworthy custodian or negotiate an alternative.",
            ],
            secrets=[],
            intentions_enabled=False,
        ),
        known_context=(
            "You are at the expedition sponsor's map collection. You know the "
            "expedition needs route material, terms, and a decision about who "
            "will safeguard the loaned documents."
        ),
        mechanics=_dnd_mechanics(
            level=5,
            proficiency_bonus=3,
            ability_scores={
                "str": 14,
                "dex": 14,
                "con": 14,
                "int": 13,
                "wis": 12,
                "cha": 10,
            },
            armor_class=15,
            hp=38,
            actions=[
                _weapon_action(
                    "longsword",
                    "Longsword",
                    attack_bonus=5,
                    damage="1d8+2",
                    damage_type="slashing",
                    notes=(
                        "Demo combat scaffold for the reviewed startup slice; "
                        "the expedition leader is visibly armed with this "
                        "weapon when they intercept the predator."
                    ),
                ),
            ],
        ),
    )


def _content_state(
    *,
    seed_inputs: dict[str, Any],
    catalog: dict[str, Any],
    cards: dict[str, RefCard],
) -> dict[str, ContentPackState]:
    pack_id = seed_inputs["pack_id"]
    pending: dict[str, PendingContentSignal] = {}
    for index, ref in enumerate(seed_inputs.get("initial_router_lookup_refs", []), start=1):
        if ref not in cards:
            continue
        card = cards[ref]
        pending[f"startup_{index:02d}"] = PendingContentSignal(
            signal_id=f"startup_{index:02d}",
            pack_id=pack_id,
            ref_id=ref,
            content_hash=card.content_hash,
            reason="startup router context",
            priority=10,
            requested_fields=["summary"],
            metadata={
                "kind": card.kind,
                "visibility": "router_hidden",
                "summary": card.summary,
            },
        )

    active_fronts: dict[str, ContentFrontState] = {}
    for front_ref in seed_inputs.get("active_front_refs", []):
        if front_ref in cards:
            active_fronts[front_ref] = ContentFrontState(
                front_id=front_ref,
                label=front_ref,
                status="active",
                notes=cards[front_ref].summary,
            )

    contexts = {
        item["actor_ref"]: item
        for item in catalog.get("agent_context_slices", [])
    }
    knowledge_map: dict[str, ContentKnowledgeEntityState] = {}
    for actor in catalog.get("actor_dossiers", []):
        context_slice = contexts.get(actor["ref"], {})
        character_id = actor.get("character_id_hint")
        if not character_id:
            continue
        known_refs = [
            _compact_ref(pack_id, ref, cards)
            for ref in _refs_for_actor(actor, context_slice)
            if ref in cards
        ]
        knowledge_map[character_id] = ContentKnowledgeEntityState(
            entity_id=character_id,
            known_refs=known_refs,
            notes="Reviewed startup knowledge map seed.",
        )

    return {
        pack_id: ContentPackState(
            pack_id=pack_id,
            pending_signals=pending,
            fronts=active_fronts,
            knowledge_map=knowledge_map,
            metadata={
                "db_path": str(COMPILED_PACK_PATH.relative_to(REPO_ROOT)),
                "pack_version": seed_inputs["pack_version"],
                "source_fingerprint": seed_inputs["source_fingerprint"],
                "schema_version": CONTENT_PACK_SCHEMA_VERSION,
                "active_front_refs": seed_inputs.get("active_front_refs", []),
                "catalog": [
                    {"ref": ref, "aliases": [ref.replace(".", " ")]}
                    for ref in seed_inputs.get("initial_router_lookup_refs", [])
                    if ref in cards
                ],
            },
        )
    }


def _field_start_checkpoint(
    ckpt: CheckpointFile,
    *,
    seed_inputs: dict[str, Any],
    cards: dict[str, RefCard],
) -> None:
    pack_id = seed_inputs["pack_id"]
    route_ref = "loc.barrier_peaks_route"
    route_card = cards.get(route_ref)
    if route_card is not None:
        pack_state = ckpt.session.content_state.get(pack_id)
        if pack_state is not None:
            pack_state.pending_signals["field_route"] = PendingContentSignal(
                signal_id="field_route",
                pack_id=pack_id,
                ref_id=route_ref,
                content_hash=route_card.content_hash,
                reason="field-start router context",
                priority=20,
                requested_fields=["summary"],
                metadata={
                    "kind": route_card.kind,
                    "visibility": "router_hidden",
                    "summary": route_card.summary,
                },
            )
            for entity_id in ("npc_garret", "npc_gearbox"):
                state = pack_state.knowledge_map.get(entity_id)
                if state is None:
                    state = ContentKnowledgeEntityState(entity_id=entity_id)
                known_refs = list(state.known_refs)
                for ref in (route_ref, "handout.cartophile_maps"):
                    if ref in cards:
                        compact = _compact_ref(pack_id, ref, cards)
                        if compact not in known_refs:
                            known_refs.append(compact)
                pack_state.knowledge_map[entity_id] = state.model_copy(
                    update={
                        "known_refs": known_refs,
                        "notes": (
                            "Field-start seed: expedition has route folios "
                            "and has reached the wooded foothills."
                        ),
                    }
                )

    for char in ckpt.characters:
        if char.character_id in {PLAYER_ID, "npc_garret", "npc_gearbox"}:
            char.status = CharacterStatus.active
            char.location = route_ref
        else:
            char.status = CharacterStatus.dormant
        if char.character_id == PLAYER_ID:
            char.known_context = (
                "You accepted the Cartophile's return-documentation terms. "
                "Garret carries the route folios as custodian, Gearbox travels "
                "as technical support, and the party has reached the wooded "
                "foothills on the Barrier Peaks route."
            )
            char.private_state.current_objectives = [
                "Scout the wooded foothills without losing the route.",
                "Protect the expedition from terrain and wildlife hazards.",
            ]
        elif char.character_id == "npc_garret":
            char.private_state.current_objectives = [
                "Keep the route folios dry, intact, and useful.",
                "Warn the expedition when terrain does not match the page.",
            ]
        elif char.character_id == "npc_gearbox":
            char.private_state.current_objectives = [
                "Support the expedition as a technical specialist.",
                "Watch for unstable terrain and mechanical anomalies.",
            ]

    ckpt.player_primer = (
        "You are already in the wooded foothills of the Barrier Peaks route "
        "with Garret as map custodian and Gearbox as technical support."
    )
    ckpt.world_state.facts = [
        "The expedition accepted the Cartophile's terms before departure.",
        "Garret is custodian of the route folios.",
        "Gearbox accompanies the expedition as technical support.",
        "The party is scouting the wooded foothills before the cave approach.",
    ]
    ckpt.session.config.narrative_rules = (
        "Start in the field at loc.barrier_peaks_route. Keep play grounded in "
        "route scouting, terrain choices, and immediate expedition hazards. "
        "Do not reopen the Cartophile negotiation unless the player sends a "
        "message back."
    )


def _story_checkpoint(*, start_mode: str = "startup") -> CheckpointFile:
    catalog = _read_json(CATALOG_PATH)
    seed_inputs = _read_json(SEED_INPUTS_PATH)
    cards = _load_cards()
    context_by_actor = {
        item["actor_ref"]: item
        for item in catalog.get("agent_context_slices", [])
    }
    npcs = [
        _npc_character(
            actor=actor,
            context_slice=context_by_actor.get(actor["ref"], {}),
            pack_id=seed_inputs["pack_id"],
            cards=cards,
        )
        for actor in catalog.get("actor_dossiers", [])
    ]
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=STORY_ID,
            story_id=STORY_ID,
            config=SessionConfig(),
            content_state=_content_state(
                seed_inputs=seed_inputs,
                catalog=catalog,
                cards=cards,
            ),
        ),
        player_primer=(
            "You are opening a Lost Laboratory expedition from the patron's "
            "map collection. The immediate job is to secure terms, understand "
            "the loaned route material, and decide who will accompany or "
            "safeguard the expedition records."
        ),
        world_state=WorldState(
            facts=[
                "The expedition is still in its startup negotiation phase.",
                "The patron can provide route material but expects recovered documentation.",
                "Candidate custodians and specialists may be negotiated before travel.",
            ],
            physics_ruleset=PhysicsRuleset(
                strength_limits="dnd5e_basic",
                magic_enabled=True,
            ),
            setting=StorySetting(
                genre="D&D expedition adventure",
                era="fantasy with strange lost technology",
                tone="concrete, table-play practical, lightly ominous",
                premise=(
                    "A party prepares to follow incomplete route lore toward "
                    "a lost inventor's laboratory."
                ),
            ),
            lore=(
                "Use only reviewed runtime-ready Lost Laboratory startup "
                "content. The downstream monastery, city, final laboratory, "
                "tactical maps, statblocks, and appendices are not available "
                "in this demo slice."
            ),
        ),
        characters=[_player_character(), *npcs],
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    ckpt.session.config.settings.player_roll_mode = "auto"
    ckpt.session.config.narrative_rules = (
        "Keep the startup scene grounded in negotiation, route preparation, "
        "and choosing expedition support. Do not invent downstream locations "
        "or encounters beyond reviewed runtime-ready content."
    )
    if start_mode == "field":
        _field_start_checkpoint(ckpt, seed_inputs=seed_inputs, cards=cards)
    return ckpt


def _write_story(stories_dir: Path, *, start_mode: str = "startup") -> Path:
    story_dir = stories_dir / STORY_ID
    story_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _story_checkpoint(start_mode=start_mode)
    path = story_dir / "ckpt_0000.json"
    path.write_text(
        ckpt.model_dump_json(
            indent=2,
            context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
        ),
        encoding="utf-8",
    )
    CheckpointFile.model_validate_json(path.read_text(encoding="utf-8"))
    return path


async def _run_cli_line(state: CLIState, line: str) -> str:
    output = io.StringIO()
    with redirect_stdout(output):
        await state.handle_line(line)
    return output.getvalue()


def _message_text(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
    return "\n".join(chunks)


def _safe_excerpt(text: str, *, limit: int = 1200) -> str:
    cleaned = "\n".join(line.rstrip() for line in text.strip().splitlines())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "\n...[truncated]"


def _role_label(config: LLMConfig, role: str) -> str:
    return f"{config.provider_for_role(role)}:{config.model_for_role(role)}"


def _tag_block(text: str, tag: str) -> str:
    start_marker = f"<{tag}"
    end_marker = f"</{tag}>"
    start = text.find(start_marker)
    if start < 0:
        return ""
    start = text.find(">", start)
    if start < 0:
        return ""
    end = text.find(end_marker, start)
    if end < 0:
        return ""
    return text[start + 1:end].strip()


def _tag_lines(text: str, tag: str, *, limit: int = 24) -> list[str]:
    block = _tag_block(text, tag)
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    return lines[:limit]


def _content_catalog_refs(text: str, *, limit: int = 80) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for line in _tag_lines(text, "available_catalog", limit=300):
        item: dict[str, str] = {}
        for key in ("pack", "ref", "kind", "visibility"):
            marker = f"{key}="
            value = ""
            for part in line.split():
                if part.startswith(marker):
                    value = part.removeprefix(marker).strip()
                    break
            if value:
                item[key] = value
        if item:
            refs.append(item)
        if len(refs) >= limit:
            break
    return refs


def _content_manager_prompt_audit(prompt_text: str) -> dict[str, Any]:
    return {
        "recent_facts": _tag_lines(prompt_text, "recent_facts"),
        "knowledge_map": _tag_lines(prompt_text, "engine_knowledge_map"),
        "known_router_refs": _tag_lines(prompt_text, "known_router_refs"),
        "candidate_characters": _tag_lines(prompt_text, "candidate_characters"),
        "available_catalog_refs": _content_catalog_refs(prompt_text),
    }


def _content_manager_output_audit(parsed: Any) -> dict[str, Any]:
    if not hasattr(parsed, "model_dump"):
        return {}
    data = parsed.model_dump(mode="json")
    return {
        "knowledge_updates": data.get("knowledge_updates", []),
        "router_required_knowledge": data.get("router_required_knowledge", []),
        "router_turn_candidates": data.get("router_turn_candidates", []),
        "agent_context_broadcasts": data.get("agent_context_broadcasts", []),
        "no_update_reason": data.get("no_update_reason", ""),
    }


def _router_output_audit(parsed: Any) -> dict[str, Any]:
    if not hasattr(parsed, "model_dump"):
        return {}
    observers = []
    for observer in getattr(parsed, "observers", []) or []:
        observers.append({
            "character_id": getattr(observer, "character_id", ""),
            "routing_role": getattr(observer, "routing_role", ""),
            "observation_level": getattr(observer, "observation_level", ""),
        })
    return {
        "event_kind": getattr(parsed, "event_kind", ""),
        "interaction_mode": getattr(parsed, "interaction_mode", ""),
        "combatant_ids": list(getattr(parsed, "combatant_ids", []) or []),
        "combatant_spawns": [
            {
                "character_id": getattr(spawn, "character_id", ""),
                "monster_key": getattr(spawn, "monster_key", ""),
                "statblock_ref": getattr(spawn, "statblock_ref", ""),
                "name": getattr(spawn, "name", ""),
            }
            for spawn in getattr(parsed, "combatant_spawns", []) or []
        ],
        "observers": observers,
        "decision_rationale": getattr(parsed, "decision_rationale", ""),
    }


def _combat_state_summary(checkpoint: CheckpointFile | None) -> dict[str, Any]:
    if checkpoint is None:
        return {"active": False}
    combat = getattr(checkpoint.session, "active_combat", None)
    if combat is None:
        return {"active": False}
    public = dnd_combat.public_status(checkpoint.session)
    return {
        "active": True,
        "status": getattr(combat, "status", ""),
        "round_number": getattr(combat, "round_number", 0),
        "turn_index": getattr(combat, "turn_index", 0),
        "public_status": public,
        "combatants": [
            {
                "character_id": getattr(combatant, "character_id", ""),
                "name": getattr(combatant, "name", ""),
                "hp": {
                    "current": getattr(combatant, "hit_points_current", 0),
                    "max": getattr(combatant, "hit_points_max", 0),
                    "temporary": getattr(combatant, "hit_points_temporary", 0),
                },
                "initiative_total": getattr(combatant, "initiative_total", 0),
                "defeat_state": getattr(combatant, "defeat_state", ""),
                "pending_initiating_action": getattr(
                    combatant,
                    "pending_initiating_action",
                    "",
                ),
            }
            for combatant in getattr(combat, "combatants", []) or []
        ],
        "audit_tail": list(getattr(combat, "audit_lines", []) or [])[-12:],
    }


def _checkpoint_turn_index(checkpoint: CheckpointFile | None) -> int:
    if checkpoint is None:
        return 0
    return int(getattr(checkpoint.session, "turn_index", 0) or 0)


def _next_soak_command(checkpoint: CheckpointFile | None, index: int) -> str:
    combat = (
        getattr(checkpoint.session, "active_combat", None)
        if checkpoint is not None else None
    )
    if combat is not None:
        return SOAK_COMBAT_COMMANDS[index % len(SOAK_COMBAT_COMMANDS)]
    return SOAK_EXPLORATION_COMMANDS[index % len(SOAK_EXPLORATION_COMMANDS)]


def _soak_defeated_spawned_hostiles_active(
    checkpoint: CheckpointFile | None,
) -> bool:
    combat = (
        getattr(checkpoint.session, "active_combat", None)
        if checkpoint is not None else None
    )
    if combat is None:
        return False
    spawned_hostiles = []
    for combatant in getattr(combat, "combatants", []) or []:
        character_id = str(getattr(combatant, "character_id", "") or "")
        combatant_id = str(getattr(combatant, "combatant_id", "") or "")
        if character_id.startswith("mon_") or combatant_id.startswith("mon_"):
            spawned_hostiles.append(combatant)
    if not spawned_hostiles:
        return False
    return all(
        str(getattr(combatant, "defeat_state", "") or "") != "active"
        or int(getattr(combatant, "hit_points_current", 0) or 0) <= 0
        or bool(getattr(combatant, "removed", False))
        for combatant in spawned_hostiles
    )


async def _run_demo(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    os.environ.setdefault("LLM_MODEL_AGENT", "claude-haiku-4-5")
    os.environ.setdefault("LLM_MODEL_CHARACTER_GEN", "claude-haiku-4-5")
    os.environ.setdefault("LLM_MODEL_AGENT_CONVENIENCE", "claude-haiku-4-5")

    config = LLMConfig.from_env()
    required = set(LIVE_PLAY_REQUIRED_ROLES) | {"content_manager"}
    missing = config.missing_credentials(required)
    if missing:
        formatted = ", ".join(
            f"{item.role} ({item.provider}; {', '.join(item.env_names)})"
            for item in missing
        )
        raise RuntimeError(f"Missing live credentials: {formatted}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    stories_dir = RUN_DIR / "stories"
    sessions_dir = RUN_DIR / "sessions"
    story_path = _write_story(
        stories_dir,
        start_mode="field" if args.scenario in {"field", "soak"} else "startup",
    )
    if args.install_story:
        installed_dir = REPO_ROOT / "app/storage/stories" / STORY_ID
        if installed_dir.exists():
            shutil.rmtree(installed_dir)
        shutil.copytree(story_path.parent, installed_dir)

    engine = EngineBridge(
        stories_dir=str(stories_dir),
        sessions_dir=str(sessions_dir),
        prompts_dir=str(REPO_ROOT / "app/prompts"),
        llm_config=config,
    )
    role_calls: list[dict[str, Any]] = []
    real_complete = engine.client.complete

    async def _recording_complete(*call_args, **kwargs):
        role = str(kwargs.get("role") or (call_args[0] if call_args else ""))
        response_model = kwargs.get("response_model")
        messages = kwargs.get("messages") or []
        prompt_text = _message_text(messages)
        record: dict[str, Any] = {
            "role": role,
            "response_model": response_model.__name__ if response_model else "",
            "message_count": len(messages),
            "contains_engine_knowledge_map": "engine_knowledge_map" in prompt_text,
            "contains_knowledge_entity_rows": " entity=npc_" in prompt_text,
            "contains_runtime_content_card": any(
                marker in prompt_text
                for marker in (
                    "location_card ref=loc.cartophile_collection",
                    "front_signal ref=front.expedition_obligation",
                    "content_known ref=handout.cartophile_maps",
                )
            ),
            "contains_turn_hint": "turn_hint " in prompt_text,
            "contains_private_path": "private_extractions/" in prompt_text,
        }
        if role == "content_manager":
            record["content_manager_input"] = _content_manager_prompt_audit(
                prompt_text
            )
        try:
            response = await real_complete(*call_args, **kwargs)
        except Exception:
            record["error"] = traceback.format_exc()
            role_calls.append(record)
            raise
        record["model"] = response.model
        if role == "content_manager":
            record["content_manager_output"] = _content_manager_output_audit(
                response.parsed
            )
        elif role == "event_router":
            record["router_output"] = _router_output_audit(response.parsed)
        role_calls.append(record)
        return response

    engine.client.complete = _recording_complete  # type: ignore[method-assign]

    session_id = args.session or f"{STORY_ID}_{TS.lower()}"
    transcript: list[dict[str, str]] = []
    try:
        engine.create_empty_session(session_id)
        state = CLIState(engine, session_id, "")
        lines = SCENARIO_COMMANDS[args.scenario]
        for line in lines:
            output = await _run_cli_line(state, line)
            transcript.append({"input": line, "output": output})
            if "error:" in output.lower():
                break
        if args.scenario == "soak" and (
            not transcript or "error:" not in transcript[-1]["output"].lower()
        ):
            soak_index = 0
            while len(transcript) < args.max_soak_commands:
                current_ckpt = engine.load_latest(session_id)
                if _checkpoint_turn_index(current_ckpt) >= args.target_turns:
                    break
                pending_rolls = state._joined_pending_roll_prompts()
                if _soak_defeated_spawned_hostiles_active(current_ckpt):
                    line = "/combat end"
                elif pending_rolls:
                    line = "/roll all"
                else:
                    line = _next_soak_command(current_ckpt, soak_index)
                    soak_index += 1
                output = await _run_cli_line(state, line)
                transcript.append({"input": line, "output": output})
                if "error:" in output.lower():
                    break
        ckpt = engine.load_latest(session_id)
        return _build_report(
            args=args,
            config=config,
            session_id=session_id,
            story_path=story_path,
            transcript=transcript,
            role_calls=role_calls,
            checkpoint=ckpt,
            error="",
        )
    except Exception:
        return _build_report(
            args=args,
            config=config,
            session_id=session_id,
            story_path=story_path,
            transcript=transcript,
            role_calls=role_calls,
            checkpoint=None,
            error=traceback.format_exc(),
        )
    finally:
        await engine.close()


def _build_report(
    *,
    args: argparse.Namespace,
    config: LLMConfig,
    session_id: str,
    story_path: Path,
    transcript: list[dict[str, str]],
    role_calls: list[dict[str, Any]],
    checkpoint: CheckpointFile | None,
    error: str,
) -> dict[str, Any]:
    introduced_refs: list[str] = []
    knowledge_entities: list[str] = []
    canonical_fact_count = 0
    turn_index = 0
    if checkpoint is not None:
        turn_index = checkpoint.session.turn_index
        canonical_fact_count = sum(
            len(event.canonical_event.observable_facts)
            for event in checkpoint.canonical_events
        )
        for pack_state in checkpoint.session.content_state.values():
            introduced_refs.extend(sorted(pack_state.introduced_refs))
            knowledge_entities.extend(sorted(pack_state.knowledge_map))

    cli_errors = [
        item
        for item in transcript
        if "error:" in item["output"].lower()
    ]
    router_calls = [call for call in role_calls if call["role"] == "event_router"]
    content_calls = [
        call for call in role_calls if call["role"] == "content_manager"
    ]
    content_call_count = len(content_calls)
    router_call_count = len(router_calls)
    router_modes = [
        call.get("router_output", {}).get("interaction_mode", "")
        for call in router_calls
    ]
    content_requested_refs = [
        item.get("ref", "")
        for call in content_calls
        for item in call.get("content_manager_output", {}).get(
            "router_required_knowledge",
            [],
        )
    ]
    agent_calls = [
        call
        for call in role_calls
        if call["role"] in {"agent", "agent_standard", "agent_convenience"}
    ]
    checks = [
        _check("story_seed_validated", story_path.exists(), str(story_path)),
        _check("cli_completed_without_error_output", not cli_errors, cli_errors),
        _check("content_manager_called", bool(content_calls), content_calls),
        _check(
            "router_never_received_full_knowledge_map",
            all(
                not call["contains_engine_knowledge_map"]
                and not call["contains_knowledge_entity_rows"]
                for call in router_calls
            ),
            router_calls,
        ),
        _check(
            "router_received_projected_runtime_content",
            any(call["contains_runtime_content_card"] for call in router_calls),
            router_calls,
        ),
        _check(
            "character_agents_used_haiku_role",
            bool(agent_calls)
            and all(call["role"] != "agent" for call in agent_calls)
            and all("haiku" in str(call.get("model", "")).lower() for call in agent_calls),
            agent_calls,
        ),
        _check(
            "no_private_pack_paths_in_model_prompts",
            all(not call["contains_private_path"] for call in role_calls),
            role_calls,
        ),
        _check("canonical_events_created", canonical_fact_count > 0, canonical_fact_count),
        _check(
            "startup_content_introduced",
            any("loc.cartophile_collection" in ref for ref in introduced_refs)
            and any("front.expedition_obligation" in ref for ref in introduced_refs),
            introduced_refs,
        ),
        _check(
            "knowledge_map_seed_present",
            "npc_cartophile" in knowledge_entities,
            knowledge_entities,
        ),
        _check(
            "content_manager_outputs_recorded",
            bool(content_calls)
            and all("content_manager_output" in call for call in content_calls),
            content_calls,
        ),
    ]
    if args.scenario in {"deep", "field", "soak"}:
        checks.extend([
            _check(
                "content_manager_throttled_below_router_calls",
                content_call_count < router_call_count,
                {
                    "content_manager_calls": content_call_count,
                    "event_router_calls": router_call_count,
                },
            ),
            _check(
                "route_exploration_content_introduced",
                any("loc.barrier_peaks_route" in ref for ref in introduced_refs),
                introduced_refs,
            ),
        ])
    if args.scenario == "soak":
        checks.extend([
            _check(
                "soak_target_turns_reached",
                turn_index >= args.target_turns,
                {
                    "turn_index": turn_index,
                    "target_turns": args.target_turns,
                    "commands": len(transcript),
                    "max_soak_commands": args.max_soak_commands,
                },
            ),
            _check(
                "soak_not_left_in_defeated_combat_loop",
                not _soak_defeated_spawned_hostiles_active(checkpoint),
                _combat_state_summary(checkpoint),
            ),
        ])
    if args.scenario == "deep":
        checks.extend([
            _check(
                "content_manager_requested_route_context",
                "loc.barrier_peaks_route" in content_requested_refs,
                content_requested_refs,
            ),
        ])
    if args.scenario in {"deep", "field", "soak"}:
        checks.extend([
            _check(
                "combat_path_exercised",
                any(mode == "dnd_combat_start" for mode in router_modes),
                router_modes,
            ),
        ])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario": args.scenario,
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "session_id": session_id,
        "story_id": STORY_ID,
        "story_path": str(story_path.relative_to(REPO_ROOT)),
        "installed_story": bool(args.install_story),
        "target_turns": args.target_turns if args.scenario == "soak" else None,
        "max_soak_commands": (
            args.max_soak_commands if args.scenario == "soak" else None
        ),
        "command_count": len(transcript),
        "manual_combat_end_used": any(
            item["input"].strip().lower() == "/combat end"
            for item in transcript
        ),
        "roles": {
            role: _role_label(config, role)
            for role in (
                "content_manager",
                "event_router",
                "narrator",
                "agent",
                "agent_standard",
                "agent_convenience",
            )
        },
        "turn_index": turn_index,
        "canonical_fact_count": canonical_fact_count,
        "introduced_refs": introduced_refs,
        "knowledge_entities": knowledge_entities,
        "combat": _combat_state_summary(checkpoint),
        "call_counts": {
            "content_manager": content_call_count,
            "event_router": router_call_count,
            "role_calls_total": len(role_calls),
        },
        "transcript": transcript,
        "role_calls": role_calls,
        "issues_fixed": [
            {"issue": issue, "fix": fix}
            for issue, fix in ISSUES_FIXED
        ],
        "checks": checks,
        "error": error,
    }


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _markdown(report: dict[str, Any]) -> str:
    passed = sum(1 for check in report["checks"] if check["passed"])
    total = len(report["checks"])
    lines = [
        "# Lost Laboratory CLI Demo Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Scenario: `{report['scenario']}`",
        f"Run directory: `{report['run_dir']}`",
        f"Story: `{report['story_id']}`",
        f"Session: `{report['session_id']}`",
    ]
    if report.get("target_turns"):
        lines.append(f"Target turns: `{report['target_turns']}`")
    lines.extend([
        f"Checks: `{passed}/{total}`",
        f"Turn index: `{report.get('turn_index', 0)}`",
        f"CLI commands: `{report.get('command_count', 0)}`",
        f"Manual combat end used: `{report.get('manual_combat_end_used', False)}`",
        "",
        "## Result",
        "",
    ])
    if report.get("error"):
        lines.extend(["The run ended with an exception:", "", "```text"])
        lines.append(report["error"].strip())
        lines.extend(["```", ""])
    else:
        lines.append("The CLI demo completed and produced canonical events.")
        lines.append("")

    lines.extend(["## Checks", ""])
    for check in report["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {mark}: `{check['name']}`")
    lines.append("")

    lines.extend(["## Issues Hit And Fixed", ""])
    for item in report.get("issues_fixed", []):
        lines.append(f"- Issue: {item['issue']}")
        lines.append(f"  Fix: {item['fix']}")
    lines.append("")

    lines.extend(["## Role Calls", ""])
    call_counts = report.get("call_counts", {})
    if call_counts:
        lines.append(
            "Call counts: "
            f"content_manager={call_counts.get('content_manager', 0)}, "
            f"event_router={call_counts.get('event_router', 0)}, "
            f"total={call_counts.get('role_calls_total', 0)}."
        )
        lines.append("")
    for index, call in enumerate(report["role_calls"], start=1):
        lines.append(
            "- "
            f"{index}. role=`{call['role']}` model=`{call.get('model', '-')}` "
            f"schema=`{call['response_model']}` "
            f"knowledge_map={call['contains_engine_knowledge_map']} "
            f"runtime_content={call['contains_runtime_content_card']} "
            f"turn_hint={call['contains_turn_hint']}"
        )
    lines.append("")

    lines.extend(["## Content Manager Audit", ""])
    for index, call in enumerate(
        [
            item for item in report["role_calls"]
            if item["role"] == "content_manager"
        ],
        start=1,
    ):
        output = call.get("content_manager_output", {})
        input_summary = call.get("content_manager_input", {})
        requested = [
            item.get("ref", "")
            for item in output.get("router_required_knowledge", [])
            if item.get("ref", "")
        ]
        updates = [
            item.get("ref", "")
            for item in output.get("knowledge_updates", [])
            if item.get("ref", "")
        ]
        candidates = [
            item.get("character_id", "")
            for item in output.get("router_turn_candidates", [])
            if item.get("character_id", "")
        ]
        lines.append(
            "- "
            f"{index}. recent_facts={len(input_summary.get('recent_facts', []))} "
            f"known_entities={len(input_summary.get('knowledge_map', []))} "
            f"known_router_refs={len(input_summary.get('known_router_refs', []))} "
            f"catalog_refs={len(input_summary.get('available_catalog_refs', []))} "
            f"required_refs={requested or '-'} "
            f"knowledge_updates={updates or '-'} "
            f"turn_candidates={candidates or '-'} "
            f"no_update={output.get('no_update_reason') or '-'}"
        )
    lines.append("")

    lines.extend(["## Combat State", ""])
    combat = report.get("combat", {})
    if not combat.get("active"):
        lines.append("No active D&D combat at final checkpoint.")
    else:
        lines.append(
            f"Active combat: status=`{combat.get('status', '')}` "
            f"round=`{combat.get('round_number', 0)}`."
        )
        for combatant in combat.get("combatants", []):
            hp = combatant.get("hp", {})
            lines.append(
                "- "
                f"`{combatant.get('character_id', '')}` "
                f"hp={hp.get('current', 0)}/{hp.get('max', 0)} "
                f"init={combatant.get('initiative_total', 0)} "
                f"state={combatant.get('defeat_state', '')}"
            )
        if combat.get("audit_tail"):
            lines.append("Audit tail:")
            for audit in combat["audit_tail"]:
                lines.append(f"- {audit}")
    lines.append("")

    lines.extend(["## CLI Transcript", ""])
    for item in report["transcript"]:
        lines.extend(["```text", f"> {item['input']}"])
        lines.append(_safe_excerpt(item["output"].rstrip(), limit=1800))
        lines.extend(["```", ""])

    lines.extend(["## Content State", ""])
    lines.append(f"Introduced refs: `{len(report['introduced_refs'])}`")
    for ref in report["introduced_refs"]:
        lines.append(f"- `{ref}`")
    lines.append("")
    lines.append("Knowledge-map entities:")
    for entity in report["knowledge_entities"]:
        lines.append(f"- `{entity}`")
    lines.append("")

    lines.extend(["## Notes", ""])
    lines.append(
        "- Character-agent calls are expected to use `agent_standard` with a "
        "Haiku model. The script also sets `LLM_MODEL_AGENT` and "
        "`LLM_MODEL_AGENT_CONVENIENCE` to Haiku unless the environment already "
        "overrides them."
    )
    lines.append(
        "- The reviewed startup slice intentionally excludes downstream sites, "
        "combat statblocks, tactical maps, and player-safe image derivatives."
    )
    lines.append(
        "- The report intentionally records only compact runtime refs and CLI "
        "output, not raw OCR, source PDF text, or private source paths."
    )
    return "\n".join(lines)


async def main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_COMMANDS),
        default="startup",
        help="Which scripted CLI playtest sequence to run.",
    )
    parser.add_argument(
        "--session",
        default="",
        help="Session id for the demo run. Defaults to a timestamped id.",
    )
    parser.add_argument(
        "--install-story",
        action="store_true",
        help=(
            "Also copy the generated story seed into app/storage/stories for "
            "manual scripts/play.py replay."
        ),
    )
    parser.add_argument(
        "--target-turns",
        type=int,
        default=35,
        help=(
            "For --scenario soak, continue scripted play until the checkpoint "
            "turn index reaches this value."
        ),
    )
    parser.add_argument(
        "--max-soak-commands",
        type=int,
        default=60,
        help=(
            "For --scenario soak, stop after this many total CLI commands even "
            "if the target turn index has not been reached."
        ),
    )
    args = parser.parse_args()

    report = await _run_demo(args)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(_markdown(report), encoding="utf-8")
    print(JSON_PATH.relative_to(REPO_ROOT))
    print(MD_PATH.relative_to(REPO_ROOT))
    failed = bool(report.get("error")) or any(
        not check["passed"] for check in report["checks"]
    )
    for check in report["checks"]:
        print(("PASS" if check["passed"] else "FAIL") + f" {check['name']}")
    return 1 if failed else 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
