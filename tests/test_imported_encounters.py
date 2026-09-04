from __future__ import annotations

import pytest

from app.engine.imported_encounters import (
    ImportedEncounterValidationError,
    apply_resolved_encounter_to_router_output,
    resolve_combat_start_from_content_state,
)
from app.engine.imported_statblocks import ImportedStatBlockNotFoundError
from app.schemas.content import ContentPackState
from tests.support.factories import dnd_router_event_draft


def test_resolves_location_encounter_to_spawns_map_and_refs() -> None:
    result = dnd_router_event_draft(
        interaction_mode="dnd_combat_start",
        combatant_ids=["alice"],
        combatant_spawns=[],
    )

    resolved = resolve_combat_start_from_content_state(
        _content_state(),
        location_ref="loc.entry",
    )
    apply_resolved_encounter_to_router_output(result, resolved)

    assert resolved is not None
    assert resolved.encounter_ref == "enc.entry"
    assert [spawn.character_id for spawn in resolved.combatant_spawns] == [
        "guardian_1",
        "guardian_2",
    ]
    assert [spawn.statblock_ref for spawn in resolved.combatant_spawns] == [
        "stat.guardian",
        "stat.guardian",
    ]
    assert resolved.battle_map is not None
    assert resolved.battle_map.source_template_ref == "map.entry"
    assert result.combatant_ids == ["alice"]
    assert result.combatant_spawns[0].statblock_ref == "stat.guardian"
    assert result.battle_map_seed.present is True


def test_imported_encounter_replaces_router_authored_monster_spawns() -> None:
    result = dnd_router_event_draft(
        interaction_mode="dnd_combat_start",
        combatant_ids=["alice"],
        combatant_spawns=[
            {
                "character_id": "rat_1",
                "monster_key": "rat",
                "statblock_ref": "stat.rat",
                "name": "Rat",
            }
        ],
    )

    resolved = resolve_combat_start_from_content_state(
        _content_state(),
        location_ref="loc.entry",
    )
    apply_resolved_encounter_to_router_output(result, resolved)

    assert [spawn.character_id for spawn in result.combatant_spawns] == [
        "guardian_1",
        "guardian_2",
    ]
    assert result.combatant_ids == ["alice"]


def test_missing_statblock_ref_blocks_encounter_resolution() -> None:
    content_state = _content_state(
        encounter={
            **_encounter(),
            "participants": [
                {
                    "participant_id": "missing",
                    "statblock_ref": "stat.missing",
                    "count": 1,
                }
            ],
        }
    )

    with pytest.raises(ImportedStatBlockNotFoundError, match="stat.missing"):
        resolve_combat_start_from_content_state(
            content_state,
            location_ref="loc.entry",
        )


def test_missing_map_ref_blocks_encounter_resolution() -> None:
    content_state = _content_state(
        encounter={**_encounter(), "map_template_refs": ["map.missing"]},
    )

    with pytest.raises(ImportedEncounterValidationError, match="map.missing"):
        resolve_combat_start_from_content_state(
            content_state,
            location_ref="loc.entry",
        )


def _content_state(
    *,
    encounter: dict | None = None,
) -> dict[str, ContentPackState]:
    return {
        "synthetic-pack": ContentPackState(
            pack_id="synthetic-pack",
            metadata={
                "locations": [{"ref": "loc.entry"}],
                "encounter_templates": [encounter or _encounter()],
                "statblocks": [_statblock()],
                "tactical_map_templates": [_map_template()],
                "trap_hazards": [_trap_hazard()],
                "treasures": [_treasure()],
            },
        )
    }


def _encounter() -> dict:
    return {
        "pack_id": "synthetic-pack",
        "ref": "enc.entry",
        "content_hash": "sha256:enc-entry",
        "title": "Entry Encounter",
        "summary": "Reviewed encounter seed.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "trigger": "The party enters the watched area.",
        "location_refs": ["loc.entry"],
        "participants": [
            {
                "participant_id": "guardian",
                "statblock_ref": "stat.guardian",
                "count": 2,
                "role": "sentinel",
                "starting_anchor_ref": "spawn.enemies",
                "tactics": "Hold the threshold.",
            }
        ],
        "map_template_refs": ["map.entry"],
        "trap_refs": ["trap.floor"],
        "treasure_refs": ["treasure.cache"],
    }


def _statblock() -> dict:
    return {
        "pack_id": "synthetic-pack",
        "ref": "stat.guardian",
        "content_hash": "sha256:stat-guardian",
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
        "senses": ["darkvision 60 ft."],
        "passive_perception": 15,
        "challenge_rating": "2",
        "xp": 450,
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
                "description": "Melee Weapon Attack.",
            }
        ],
    }


def _map_template() -> dict:
    return {
        "pack_id": "synthetic-pack",
        "ref": "map.entry",
        "content_hash": "sha256:map-entry",
        "title": "Entry Map",
        "summary": "Reviewed entry map.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "derived_from_map_asset_id": "asset.map.entry.player",
        "grid_width": 8,
        "grid_height": 6,
        "spawn_anchors": [
            {
                "anchor_id": "spawn.enemies",
                "anchor_kind": "enemies",
                "cells": [{"x": 5, "y": 3}],
                "label": "Enemy start",
            }
        ],
    }


def _trap_hazard() -> dict:
    return {
        "pack_id": "synthetic-pack",
        "ref": "trap.floor",
        "content_hash": "sha256:trap-floor",
        "title": "Entry Floor Trap",
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
        "pack_id": "synthetic-pack",
        "ref": "treasure.cache",
        "content_hash": "sha256:treasure-cache",
        "title": "Entry Cache",
        "summary": "A reviewed cache.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "treasure_kind": "container",
        "container_ref": "container.cache",
        "currency": [{"denomination": "gp", "amount": 10}],
    }
