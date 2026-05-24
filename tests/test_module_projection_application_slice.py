from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from app.engine import dnd_inventory
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.content_assets import safe_asset_reveals_for_viewer
from app.engine.content_fronts import FRONT_RUNTIME_METADATA_KEY
from app.engine.content_lookup import (
    EventRouterContentLookupOutput,
    append_router_content_lookup_records_with_llm,
)
from app.engine.content_pack_compiler import (
    CompiledContentPackWriter,
)
from app.engine.content_pack_projections import (
    build_content_pack_projection_artifact,
    content_pack_state_from_projection,
)
from app.engine.content_resolver import append_pending_router_content_records
from app.engine.imported_encounters import (
    apply_resolved_encounter_to_router_output,
    resolve_combat_start_from_content_state,
)
from app.engine.imported_statblocks import resolve_spawn_character_from_content_state
from app.engine.imported_trap_hazards import catalog_from_content_state
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop import broadcast_event
from app.llm.client import LLMClient
from app.schemas.content import (
    ContentCombatMapOverlayState,
    ContentOverlayState,
    ContentPovRevealState,
    ContentTrapOverlayState,
    ContentTreasureOverlayState,
)
from app.schemas.content_pack import (
    CompiledContentCard,
    ContentPackDomainCatalog,
)
from app.schemas.dnd_monsters import DndCombatantSpawn
from app.schemas.events import ObservableFact
from tests.support.factories import (
    character_record,
    checkpoint,
    dnd_router_output,
    llm_response,
    router_output,
)


PACK_ID = "synthetic-pack"
PACK_VERSION = "1.0.0"
SOURCE_FINGERPRINT = "sha256:synthetic-source"


def test_module_projection_application_vertical_slice(tmp_path):
    db_path = _compiled_pack(tmp_path)
    ckpt = checkpoint(
        session_id="slice",
        bindings={"alice": "u1"},
        player_character_id="alice",
        characters=[
            character_record(
                "alice",
                name="Alice",
                role="player",
                is_playable=True,
                location="loc.entry",
            )
        ],
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    ckpt.session.content_state = _content_state(db_path)

    manager = CheckpointManager(str(tmp_path / "sessions"))
    ckpt.session.turn_index = 1
    manager.save(ckpt)

    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=llm_response(
        EventRouterContentLookupOutput(
            requests=[{"pack_id": PACK_ID, "ref": "loc.secret"}]
        )
    ))
    lookup_records = asyncio.run(
        append_router_content_lookup_records_with_llm(
            ckpt,
            actor_id="alice",
            current_input="I study the draft along the western stones.",
            client=client,
            prompt_mgr=PromptManager("app/prompts"),
        )
    )
    assert lookup_records == [
        (
            "location_card ref=loc.entry visibility=router_hidden "
            "hash=hash-loc-entry pack=synthetic-pack "
            'summary="A reviewed entry location record."'
        ),
        (
            "location_card ref=loc.secret visibility=router_hidden "
            "hash=hash-loc-secret pack=synthetic-pack "
            'summary="A reviewed hidden alcove record."'
        )
    ]

    asset_payloads = safe_asset_reveals_for_viewer(
        _asset_records(),
        _asset_reveals(),
        character_id="alice",
        event_observer_ids=["alice"],
    )
    assert [payload.asset_id for payload in asset_payloads] == [
        "asset.map.entry"
    ]
    assert asset_payloads[0].delivery_ref == "asset://synthetic-pack/asset.map.entry"

    trap_catalog = catalog_from_content_state(ckpt.session.content_state)
    assert trap_catalog is not None
    trap_record = trap_catalog.router_context_records(["trap.floor"])[0]
    assert "trap.floor" in trap_record
    trap_payload = json.loads(trap_record.split("payload=", 1)[1])
    assert trap_payload["mechanics"]["detection_dc"] == 13

    dnd_inventory.apply_loot_offers_from_events(ckpt, [])
    offer = ckpt.session.dnd_inventory_offers[0]
    assert offer.source_ref == "treasure.cache"
    assert offer.items[0].item_id == "item.synthetic_key"
    assert offer.currency.gp == 10

    resolved_encounter = resolve_combat_start_from_content_state(
        ckpt.session.content_state,
        location_ref="loc.entry",
    )
    assert resolved_encounter is not None
    assert resolved_encounter.encounter_ref == "enc.entry"
    combat_result = dnd_router_output(
        interaction_mode="dnd_combat_start",
        combatant_ids=["alice"],
        combatant_spawns=[],
    )
    apply_resolved_encounter_to_router_output(combat_result, resolved_encounter)
    assert combat_result.battle_map_seed.present is True
    assert combat_result.combatant_spawns[0].statblock_ref == "stat.guardian"

    guardian = resolve_spawn_character_from_content_state(
        DndCombatantSpawn(
            character_id="guardian_1",
            statblock_ref="stat.guardian",
        ),
        content_state=ckpt.session.content_state,
        default_location="loc.entry",
    )
    assert guardian is not None
    assert guardian.mechanics["armor_class"] == 15
    assert guardian.mechanics["hit_points"]["max"] == 33

    public_event = router_output(
        event_id="evt_public_shelter",
        event_kind="public_fact",
        observer_ids=["alice"],
        facts=[ObservableFact.all("The entry bell rings loud enough for spies.")],
        effective_at_s=20,
    )
    broadcast_event(ckpt, public_event, actor_id="alice")
    front_records = append_pending_router_content_records(ckpt)
    assert len(front_records) == 1
    assert front_records[0].startswith(
        "front_signal ref=front.watchers actor=npc.overseer"
    )
    front_runtime = ckpt.session.content_state[PACK_ID].metadata[
        FRONT_RUNTIME_METADATA_KEY
    ]["fronts"]["front.watchers"]
    assert front_runtime["known_facts"] == [
        "The entry bell rings loud enough for spies."
    ]

    ckpt.session.turn_index = 2
    manager.save(ckpt)
    restored = manager.load("slice", "ckpt_0001")
    restored_pack = restored.session.content_state[PACK_ID]
    assert restored_pack.introduced_refs == {}
    assert FRONT_RUNTIME_METADATA_KEY not in restored_pack.metadata
    assert restored_pack.overlay.pov_reveals["alice"].revealed_asset_ids == [
        "asset.map.entry"
    ]
    assert (
        restored_pack.overlay.pov_reveals["alice"]
        .map_overlays["map.entry::hash-map-entry"]
        .fogged_area_ref_ids
        == ["area.secret"]
    )
    assert restored_pack.overlay.treasures[
        "treasure.cache::hash-treasure-cache"
    ].remaining_ref_ids == []

    default_dump = json.dumps(ckpt.model_dump(mode="json"), sort_keys=True)
    for forbidden in (
        "/private",
        "raw_ocr",
        "raw_text",
        "source_path",
        "PROTECTED_SOURCE_EXCERPT",
        "asset://private",
    ):
        assert forbidden not in default_dump


def _domain_catalog() -> ContentPackDomainCatalog:
    return ContentPackDomainCatalog(
        pack_id=PACK_ID,
        pack_version=PACK_VERSION,
        source_fingerprint=SOURCE_FINGERPRINT,
        locations=[
            {
                "pack_id": PACK_ID,
                "ref": "loc.entry",
                "content_hash": "hash-loc-entry",
                "record_kind": "location",
                "visibility": "router_hidden",
                "title": "Entry",
                "summary": "A reviewed entry location record.",
                "review_status": "approved",
                "gate_status": "runtime_ready",
                "confidence": 0.95,
            },
            {
                "pack_id": PACK_ID,
                "ref": "loc.secret",
                "content_hash": "hash-loc-secret",
                "record_kind": "location",
                "visibility": "router_hidden",
                "title": "Secret Alcove",
                "summary": "A reviewed hidden alcove record.",
                "review_status": "approved",
                "gate_status": "runtime_ready",
                "confidence": 0.95,
            },
        ],
        front_dossiers=[_front_dossier()],
        encounter_templates=[_encounter()],
        statblocks=[_statblock()],
        tactical_map_templates=[_map_template()],
        trap_hazards=[_trap_hazard()],
        treasures=[_treasure()],
    )


def _compiled_cards(catalog: ContentPackDomainCatalog) -> list[CompiledContentCard]:
    cards: list[CompiledContentCard] = []
    for record in catalog._domain_records():
        card_kind = (
            "location_card"
            if record.record_kind == "location"
            else "front_signal"
            if record.record_kind == "front_dossier"
            else record.record_kind
        )
        cards.append(
            CompiledContentCard(
                pack_id=PACK_ID,
                ref=record.ref,
                content_hash=record.content_hash,
                card_kind=card_kind,
                visibility=record.visibility,
                title=record.title,
                summary=record.summary,
                review_status="approved",
                gate_status="runtime_ready",
                confidence=record.confidence,
            )
        )
    return cards


def _compiled_pack(tmp_path):
    db_path = tmp_path / "synthetic_pack.sqlite"
    catalog = _domain_catalog()
    writer = CompiledContentPackWriter(
        db_path,
        pack_id=PACK_ID,
        pack_version=PACK_VERSION,
        source_fingerprint=SOURCE_FINGERPRINT,
    )
    writer.write_pack(
        pages=[],
        cards=_compiled_cards(catalog),
    )
    return db_path


def _content_state(db_path):
    treasure_overlay = ContentTreasureOverlayState(
        treasure_id="treasure.cache",
        content_hash="hash-treasure-cache",
        revealed=True,
    )
    trap_overlay = ContentTrapOverlayState(
        trap_id="trap.floor",
        content_hash="hash-trap-floor",
    )
    projection = build_content_pack_projection_artifact(
        _domain_catalog(),
        runtime_cards=_compiled_cards(_domain_catalog()),
        initial_router_lookup_refs=["loc.entry"],
        active_front_refs=["front.watchers"],
    )
    state = content_pack_state_from_projection(
        projection,
        db_path=str(db_path),
    )[PACK_ID]
    state.overlay = ContentOverlayState(
        traps={trap_overlay.overlay_key(): trap_overlay},
        treasures={treasure_overlay.overlay_key(): treasure_overlay},
        pov_reveals={
            "alice": ContentPovRevealState(
                viewer_id="alice",
                revealed_asset_ids=["asset.map.entry"],
                map_overlays={
                    "map.entry::hash-map-entry": ContentCombatMapOverlayState(
                        map_id="map.entry",
                        content_hash="hash-map-entry",
                        revealed_area_ref_ids=["area.entry"],
                        fogged_area_ref_ids=["area.secret"],
                    )
                },
            )
        },
    )
    return {
        PACK_ID: state,
    }


def _asset_records() -> list[dict]:
    return [
        {
            "pack_id": PACK_ID,
            "asset_id": "asset.map.entry",
            "kind": "player_safe_map",
            "title": "Entry map",
            "mime_type": "image/png",
            "width": 640,
            "height": 480,
            "sha256": "hash-map-entry",
            "source_ref": "src-map-entry",
            "review_status": "approved",
            "spoiler_class": "low",
            "player_safe_alt_text": "A safe crop of the entry room.",
            "player_safe_caption": "A sketched entry room map.",
            "delivery_ref": "asset://synthetic-pack/asset.map.entry",
            "safe_for_players": True,
        }
    ]


def _asset_reveals() -> list[dict]:
    return [
        {
            "pack_id": PACK_ID,
            "asset_id": "asset.map.entry",
            "audience": "all_observers",
            "presentation": "map_overlay",
        }
    ]


def _encounter() -> dict:
    return {
        "pack_id": PACK_ID,
        "ref": "enc.entry",
        "content_hash": "hash-enc-entry",
        "summary": "Reviewed entry encounter.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "trigger": "The party enters the watched threshold.",
        "location_refs": ["loc.entry"],
        "participants": [
            {
                "participant_id": "guardian",
                "statblock_ref": "stat.guardian",
                "count": 1,
                "role": "sentinel",
                "starting_anchor_ref": "spawn.enemies",
            }
        ],
        "map_template_refs": ["map.entry"],
        "trap_refs": ["trap.floor"],
        "treasure_refs": ["treasure.cache"],
    }


def _statblock() -> dict:
    return {
        "pack_id": PACK_ID,
        "ref": "stat.guardian",
        "content_hash": "hash-stat-guardian",
        "title": "Synthetic Guardian",
        "summary": "A combat-ready synthetic guardian.",
        "confidence": 0.98,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "automation_scope": "combat",
        "size": "Medium",
        "creature_type": "Construct",
        "alignment": "Unaligned",
        "armor_class": 15,
        "hit_points": 33,
        "hit_dice": "6d8+6",
        "speed_ft_by_mode": {"walk": 30},
        "ability_scores": {
            "strength": 16,
            "dexterity": 14,
            "constitution": 13,
            "intelligence": 8,
            "wisdom": 12,
            "charisma": 7,
        },
        "proficiency_bonus": 2,
        "passive_perception": 15,
        "challenge_rating": "2",
        "actions": [
            {
                "feature_id": "slam",
                "name": "Slam",
                "economy": "action",
                "attack_bonus": 5,
                "reach_ft": 5,
                "target": "one target",
                "damage": [
                    {"expression": "1d8+3", "damage_type": "bludgeoning"}
                ],
            }
        ],
    }


def _map_template() -> dict:
    return {
        "pack_id": PACK_ID,
        "ref": "map.entry",
        "content_hash": "hash-map-entry",
        "summary": "Reviewed entry map.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "derived_from_map_asset_id": "asset.map.entry",
        "grid_width": 8,
        "grid_height": 6,
        "spawn_anchors": [
            {
                "anchor_id": "spawn.enemies",
                "anchor_kind": "enemies",
                "cells": [{"x": 5, "y": 3}],
            }
        ],
    }


def _trap_hazard() -> dict:
    return {
        "pack_id": PACK_ID,
        "ref": "trap.floor",
        "content_hash": "hash-trap-floor",
        "summary": "A reviewed floor trap.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "trigger": "A creature crosses the marked stones.",
        "detection": "A seam crosses the floor.",
        "countermeasures": ["Jam the floor plate"],
        "linked_location_refs": ["loc.entry"],
        "placements": [
            {
                "placement_id": "place.floor",
                "location_ref": "loc.entry",
                "map_template_ref": "map.entry",
                "map_feature_ref": "feature.floor",
                "bounds": {"x": 2, "y": 2, "width": 1, "height": 1},
            }
        ],
        "mechanics": {
            "detection_dc": 13,
            "disarm_dc": 14,
            "save_dc": 12,
            "save_ability": "dexterity",
            "damage": [{"expression": "2d6", "damage_type": "piercing"}],
            "depletion_ref": "depleted.trap.floor",
        },
    }


def _treasure() -> dict:
    return {
        "pack_id": PACK_ID,
        "ref": "treasure.cache",
        "content_hash": "hash-treasure-cache",
        "summary": "A reviewed cache.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "treasure_kind": "container",
        "container_ref": "container.cache",
        "depletion_ref": "depleted.treasure.cache",
        "currency": [{"denomination": "gp", "amount": 10}],
        "items": [
            {
                "item_ref": "item.synthetic_key",
                "name": "Synthetic key",
                "quantity": 1,
                "item_type": "key",
                "value_gp": 0,
            }
        ],
    }


def _front_dossier() -> dict:
    return {
        "pack_id": PACK_ID,
        "ref": "front.watchers",
        "content_hash": "hash-front-watchers",
        "title": "Watcher Front",
        "summary": "Reviewed pressure dossier.",
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "villain_refs": ["npc.overseer"],
        "initial_knowledge": ["The entry alarm matters."],
        "action_palette": [
            {
                "action_id": "send_spies",
                "action_kind": "spy",
                "priority": 5,
                "trigger": "An entry alarm becomes public.",
                "summary": "Send spies to watch the entry.",
            }
        ],
    }
