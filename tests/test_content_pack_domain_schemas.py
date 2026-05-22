from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.content_pack import (
    ContentCrossReference,
    ContentPackDomainCatalog,
    DndStatBlockRecord,
    EncounterTemplateRecord,
    FrontDossierRecord,
    GridRect,
    TacticalMapFeature,
    TacticalMapTemplateRecord,
    TrapHazardRecord,
    TreasureRecord,
)


def test_domain_catalog_accepts_prioritized_module_records():
    catalog = ContentPackDomainCatalog(
        pack_id="synthetic-pack",
        pack_version="0.1.0",
        source_fingerprint="sha256:synthetic-redacted",
        build_hash="sha256:synthetic-build",
        tactical_map_templates=[_map_template()],
        statblocks=[_statblock()],
        trap_hazards=[_trap()],
        treasures=[_treasure()],
        front_dossiers=[_front()],
        encounter_templates=[_encounter()],
        cross_refs=[
            _cross_ref(
                ref="xref.encounter.map",
                record_ref="enc.entry",
                target_ref="map.entry",
                relation="uses",
                target_kind="tactical_map_template",
            ),
            _cross_ref(
                ref="xref.encounter.statblock",
                record_ref="enc.entry",
                target_ref="stat.guardian",
                relation="uses",
                target_kind="dnd_statblock",
            ),
        ],
    )

    assert catalog.schema_version == "content-pack-domain-v1"
    assert catalog.tactical_map_templates[0].pack_id == "synthetic-pack"
    assert catalog.statblocks[0].automation_scope == "combat"
    assert catalog.trap_hazards[0].mechanics is not None
    assert catalog.treasures[0].items[0].name == "Synthetic key"
    assert catalog.encounter_templates[0].participants[0].statblock_ref == (
        "stat.guardian"
    )


def test_domain_records_forbid_surplus_source_or_runtime_fields():
    with pytest.raises(ValidationError):
        TacticalMapTemplateRecord(
            **_map_template().model_dump(),
            source_path="synthetic-source-path-sentinel",
        )

    with pytest.raises(ValidationError):
        DndStatBlockRecord(
            **_statblock().model_dump(),
            prompt_payload={"raw_ocr": "forbidden"},
        )


def test_runtime_ready_domain_records_require_hash_review_and_reveal_trigger():
    with pytest.raises(ValidationError, match="content_hash"):
        TreasureRecord(
            **{
                **_treasure().model_dump(),
                "content_hash": "",
            }
        )

    with pytest.raises(ValidationError, match="reviewed or approved"):
        TreasureRecord(
            **{
                **_treasure().model_dump(),
                "review_status": "needs_review",
            }
        )

    with pytest.raises(ValidationError, match="reveal_trigger"):
        TreasureRecord(
            **{
                **_treasure().model_dump(),
                "spoiler_class": "high",
                "reveal_trigger": "",
            }
        )


def test_catalog_rejects_duplicate_refs_and_missing_required_cross_ref_targets():
    with pytest.raises(ValidationError, match="duplicate content pack ref"):
        ContentPackDomainCatalog(
            pack_id="synthetic-pack",
            statblocks=[_statblock()],
            treasures=[
                TreasureRecord(
                    **{
                        **_treasure().model_dump(),
                        "ref": "stat.guardian",
                    }
                )
            ],
        )

    with pytest.raises(ValidationError, match="required cross-reference target"):
        ContentPackDomainCatalog(
            pack_id="synthetic-pack",
            encounter_templates=[_encounter()],
            cross_refs=[
                _cross_ref(
                    ref="xref.missing",
                    record_ref="enc.entry",
                    target_ref="stat.missing",
                    relation="uses",
                    target_kind="dnd_statblock",
                )
            ],
        )

    catalog = ContentPackDomainCatalog(
        pack_id="synthetic-pack",
        encounter_templates=[_encounter()],
        cross_refs=[
            _cross_ref(
                ref="xref.external",
                record_ref="enc.entry",
                target_ref="appendix.external",
                relation="mentions",
                external=True,
                target_kind="appendix",
            )
        ],
    )

    assert catalog.cross_refs[0].external is True


def test_adapter_schema_guards_combat_and_tactical_map_invariants():
    with pytest.raises(ValidationError, match="at least one action"):
        DndStatBlockRecord(
            **{
                **_statblock().model_dump(),
                "actions": [],
            }
        )

    with pytest.raises(ValidationError, match="reveal_trigger"):
        TacticalMapFeature(
            feature_id="feature.secret",
            feature_kind="secret_feature",
            secret=True,
        )

    with pytest.raises(ValidationError, match="width must be positive"):
        GridRect(x=0, y=0, width=0, height=2)


def _map_template() -> TacticalMapTemplateRecord:
    return TacticalMapTemplateRecord(
        ref="map.entry",
        content_hash="hash-map-entry",
        title="Synthetic Entry Map",
        summary="Reviewed tactical geometry for a synthetic entry area.",
        confidence=0.96,
        review_status="approved",
        gate_status="runtime_ready",
        derived_from_map_asset_id="asset.map.entry.player",
        grid_width=12,
        grid_height=8,
        square_size_ft=5,
        spawn_anchors=[
            {
                "anchor_id": "spawn.players",
                "anchor_kind": "players",
                "cells": [{"x": 1, "y": 2}, {"x": 1, "y": 3}],
                "label": "Player start",
            }
        ],
        terrain_features=[
            {
                "feature_id": "wall.north",
                "feature_kind": "wall",
                "bounds": {"x": 0, "y": 0, "width": 12, "height": 1},
                "blocks_movement": True,
                "blocks_line_of_sight": True,
            },
            {
                "feature_id": "secret.panel",
                "feature_kind": "secret_feature",
                "cells": [{"x": 9, "y": 2}],
                "secret": True,
                "reveal_trigger": "A character searches the east wall.",
                "linked_refs": ["trap.floor"],
            },
        ],
        area_links=[
            {
                "area_id": "area.entry",
                "location_ref": "loc.entry",
                "bounds": {"x": 0, "y": 0, "width": 12, "height": 8},
            }
        ],
    )


def _statblock() -> DndStatBlockRecord:
    return DndStatBlockRecord(
        ref="stat.guardian",
        content_hash="hash-stat-guardian",
        title="Synthetic Guardian",
        summary="A combat-ready synthetic statblock.",
        confidence=0.93,
        review_status="approved",
        gate_status="runtime_ready",
        automation_scope="combat",
        size="Medium",
        creature_type="construct",
        alignment="unaligned",
        armor_class=13,
        hit_points=22,
        hit_dice="4d8+4",
        speed_ft_by_mode={"walk": 30},
        ability_scores={
            "strength": 14,
            "dexterity": 12,
            "constitution": 12,
            "intelligence": 6,
            "wisdom": 10,
            "charisma": 5,
        },
        proficiency_bonus=2,
        senses=["darkvision 60 ft."],
        passive_perception=10,
        challenge_rating="1/2",
        xp=100,
        actions=[
            {
                "feature_id": "slam",
                "name": "Slam",
                "economy": "action",
                "attack_bonus": 4,
                "reach_ft": 5,
                "target": "one target",
                "damage": [{"expression": "1d6+2", "damage_type": "bludgeoning"}],
            }
        ],
    )


def _trap() -> TrapHazardRecord:
    return TrapHazardRecord(
        ref="trap.floor",
        content_hash="hash-trap-floor",
        title="Synthetic Floor Hazard",
        summary="A reviewed synthetic floor hazard.",
        confidence=0.91,
        review_status="approved",
        gate_status="runtime_ready",
        trigger="A creature crosses the marked floor.",
        detection="Careful inspection can reveal disturbed stonework.",
        countermeasures=["Jam the mechanism", "Avoid the marked cells"],
        linked_location_refs=["loc.entry"],
        mechanics={
            "detection_dc": 13,
            "disarm_dc": 14,
            "save_dc": 12,
            "save_ability": "dexterity",
            "damage": [{"expression": "2d6", "damage_type": "piercing"}],
            "depletion_ref": "depleted.trap.floor",
        },
    )


def _treasure() -> TreasureRecord:
    return TreasureRecord(
        ref="treasure.cache",
        content_hash="hash-treasure-cache",
        title="Synthetic Cache",
        summary="A reviewed synthetic cache with one key and currency.",
        confidence=0.95,
        review_status="approved",
        gate_status="runtime_ready",
        container_ref="loc.entry.cache",
        depletion_ref="depleted.treasure.cache",
        currency=[{"denomination": "gp", "amount": 12}],
        items=[
            {
                "item_ref": "item.synthetic_key",
                "name": "Synthetic key",
                "quantity": 1,
                "item_type": "key",
                "value_gp": 0,
            }
        ],
    )


def _front() -> FrontDossierRecord:
    return FrontDossierRecord(
        ref="front.clock",
        content_hash="hash-front-clock",
        title="Synthetic Front",
        summary="A reviewed front dossier for synthetic pressure.",
        confidence=0.9,
        review_status="approved",
        gate_status="runtime_ready",
        villain_refs=["npc.synthetic_villain"],
        goals=["Recover the synthetic key"],
        initial_knowledge=["The entry cache exists"],
        clocks=[
            {
                "clock_id": "clock.search",
                "label": "Search progress",
                "current": 0,
                "maximum": 4,
                "thresholds": {"2": "scouts arrive", "4": "villain arrives"},
            }
        ],
        action_palette=[
            {
                "action_id": "send_scouts",
                "action_kind": "spy",
                "trigger": "The cache is disturbed.",
                "summary": "Send scouts to inspect the entry.",
                "encounter_template_refs": ["enc.entry"],
                "statblock_refs": ["stat.guardian"],
            }
        ],
    )


def _encounter() -> EncounterTemplateRecord:
    return EncounterTemplateRecord(
        ref="enc.entry",
        content_hash="hash-enc-entry",
        title="Synthetic Entry Encounter",
        summary="A reviewed encounter seed for the entry area.",
        confidence=0.92,
        review_status="approved",
        gate_status="runtime_ready",
        difficulty="easy",
        trigger="The party enters the watched area.",
        location_refs=["loc.entry"],
        participants=[
            {
                "participant_id": "guardian",
                "statblock_ref": "stat.guardian",
                "count": 1,
                "role": "sentinel",
                "starting_anchor_ref": "spawn.enemies",
            }
        ],
        map_template_refs=["map.entry"],
        trap_refs=["trap.floor"],
        treasure_refs=["treasure.cache"],
        front_refs=["front.clock"],
        noncombat_resolution="The guardian can stand down if the key is returned.",
        xp_policy="award once if defeated or bypassed through play.",
    )


def _cross_ref(**overrides) -> ContentCrossReference:
    values = {
        "ref": "xref.synthetic",
        "content_hash": "hash-xref-synthetic",
        "title": "Synthetic cross ref",
        "summary": "Synthetic reference edge.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "record_ref": "enc.entry",
        "target_ref": "stat.guardian",
        "relation": "uses",
        "target_kind": "dnd_statblock",
    }
    values.update(overrides)
    return ContentCrossReference(**values)
