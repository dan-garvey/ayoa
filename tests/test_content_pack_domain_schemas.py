from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.content_pack import (
    ActorDossierRecord,
    AdventureTableRecord,
    AgentContextSliceRecord,
    ContentCrossReference,
    ContentPackDomainCatalog,
    ContentSectionRecord,
    ContentSpanRecord,
    DndStatBlockRecord,
    EncounterTemplateRecord,
    FrontDossierRecord,
    GridRect,
    HandoutRecord,
    KeyedAreaRecord,
    KnowledgeGraphEdgeRecord,
    LocationExit,
    LocationRecord,
    RevealGraphEdge,
    TacticalMapFeature,
    TacticalMapRevealRegion,
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
        sections=[_section()],
        spans=[_span()],
        locations=[_location()],
        keyed_areas=[_keyed_area()],
        reveal_edges=[_reveal_edge()],
        handouts=[_handout()],
        tables=[_table()],
        tactical_map_templates=[_map_template()],
        statblocks=[_statblock()],
        trap_hazards=[_trap()],
        treasures=[_treasure()],
        front_dossiers=[_front()],
        actor_dossiers=[_actor()],
        agent_context_slices=[_agent_context_slice()],
        knowledge_graph_edges=[_knowledge_edge()],
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
    assert catalog.sections[0].span_refs == ["span.entry.body"]
    assert catalog.spans[0].section_ref == "sec.entry"
    assert catalog.locations[0].exits[0].to_ref == "area.entry"
    assert catalog.keyed_areas[0].keyed_label == "A1"
    assert catalog.reveal_edges[0].to_ref == "handout.entry"
    assert catalog.handouts[0].partial_reveals[0].safe_asset_ids == [
        "asset.handout.entry"
    ]
    assert catalog.tables[0].rows[0].range_start == 1
    assert catalog.tactical_map_templates[0].pack_id == "synthetic-pack"
    assert catalog.tactical_map_templates[0].floors[0].floor_id == "floor.ground"
    assert catalog.tactical_map_templates[0].fog_masks[0].mask_id == "fog.entry"
    assert catalog.tactical_map_templates[0].reveal_regions[0].reveal_id == (
        "reveal.entry"
    )
    assert catalog.statblocks[0].automation_scope == "combat"
    assert catalog.trap_hazards[0].mechanics is not None
    assert catalog.trap_hazards[0].placements[0].bounds is not None
    assert catalog.trap_hazards[0].runtime_consequences == [
        "The entry alarm draws a patrol."
    ]
    assert catalog.treasures[0].items[0].name == "Synthetic key"
    assert catalog.actor_dossiers[0].agent_context_slice_ref == (
        "agent_context.villain.startup"
    )
    assert catalog.agent_context_slices[0].actor_ref == "actor.villain"
    assert catalog.knowledge_graph_edges[0].relation == "can_dispatch"
    assert catalog.treasures[0].field_provenance["summary"][0].span_id == (
        "span.treasure.summary"
    )
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

    with pytest.raises(ValidationError):
        HandoutRecord(
            **_handout().model_dump(),
            unsafe_delivery_ref="synthetic-unsafe-ref",
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


def test_catalog_validates_location_and_reveal_graph_targets():
    with pytest.raises(ValidationError, match="location exit target"):
        ContentPackDomainCatalog(
            pack_id="synthetic-pack",
            locations=[
                LocationRecord(
                    **{
                        **_location().model_dump(),
                        "exits": [
                            {
                                "exit_id": "exit.missing",
                                "to_ref": "area.missing",
                            }
                        ],
                    }
                )
            ],
        )

    with pytest.raises(ValidationError, match="reveal edge target"):
        ContentPackDomainCatalog(
            pack_id="synthetic-pack",
            keyed_areas=[_keyed_area()],
            reveal_edges=[
                RevealGraphEdge(
                    **{
                        **_reveal_edge().model_dump(),
                        "from_ref": "area.entry",
                        "to_ref": "handout.missing",
                    }
                )
            ],
        )


def test_catalog_validates_actor_context_and_knowledge_graph_targets():
    with pytest.raises(ValidationError, match="knowledge graph edge target"):
        ContentPackDomainCatalog(
            pack_id="synthetic-pack",
            actor_dossiers=[_actor()],
            agent_context_slices=[_agent_context_slice()],
            knowledge_graph_edges=[
                KnowledgeGraphEdgeRecord(
                    **{
                        **_knowledge_edge().model_dump(),
                        "to_ref": "loc.missing",
                    }
                )
            ],
        )

    with pytest.raises(ValidationError, match="actor context slice"):
        ContentPackDomainCatalog(
            pack_id="synthetic-pack",
            actor_dossiers=[
                ActorDossierRecord(
                    **{
                        **_actor().model_dump(),
                        "agent_context_slice_ref": "agent_context.missing",
                    }
                )
            ],
        )

    with pytest.raises(ValidationError, match="agent context actor"):
        ContentPackDomainCatalog(
            pack_id="synthetic-pack",
            agent_context_slices=[_agent_context_slice()],
        )


def test_added_domain_schema_invariants_are_strict():
    with pytest.raises(ValidationError, match="keyed_label"):
        KeyedAreaRecord(
            **{
                **_keyed_area().model_dump(),
                "keyed_label": " ",
            }
        )

    with pytest.raises(ValidationError, match="reveal_trigger"):
        LocationExit(
            exit_id="exit.secret",
            to_ref="area.entry",
            secret=True,
        )

    with pytest.raises(ValidationError, match="range_end"):
        AdventureTableRecord(
            **{
                **_table().model_dump(),
                "rows": [
                    {
                        "row_id": "row.bad",
                        "range_start": 6,
                        "range_end": 1,
                        "result_summary": "Bad synthetic range.",
                    }
                ],
            }
        )

    with pytest.raises(ValidationError, match="to_ref or object_summary"):
        KnowledgeGraphEdgeRecord(
            **{
                **_knowledge_edge().model_dump(),
                "to_ref": "",
                "object_summary": "",
            }
        )

    with pytest.raises(ValidationError, match="actor_ref"):
        AgentContextSliceRecord(
            **{
                **_agent_context_slice().model_dump(),
                "actor_ref": " ",
            }
        )


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

    with pytest.raises(ValidationError, match="reveal_trigger"):
        TacticalMapRevealRegion(
            reveal_id="reveal.bad",
            bounds=GridRect(x=0, y=0, width=1, height=1),
        )

    with pytest.raises(ValidationError, match="width must be positive"):
        GridRect(x=0, y=0, width=0, height=2)


def _section() -> ContentSectionRecord:
    return ContentSectionRecord(
        ref="sec.entry",
        content_hash="hash-sec-entry",
        title="Synthetic Entry Section",
        summary="Reviewed section boundary for synthetic entry records.",
        confidence=0.98,
        review_status="approved",
        gate_status="runtime_ready",
        section_kind="section",
        page_refs=["page.001", "page.001"],
        span_refs=["span.entry.body"],
    )


def _span() -> ContentSpanRecord:
    return ContentSpanRecord(
        ref="span.entry.body",
        content_hash="hash-span-entry-body",
        title="Synthetic Entry Span",
        summary="Reviewed span pointer with redacted summary.",
        confidence=0.94,
        review_status="approved",
        gate_status="runtime_ready",
        section_ref="sec.entry",
        page_id="page.001",
        source_span_id="span.private.001",
        span_role="body",
        ordinal=1,
        bbox=[0.1, 0.2, 0.8, 0.4],
        redacted_summary="Synthetic redacted span summary.",
    )


def _location() -> LocationRecord:
    return LocationRecord(
        ref="loc.entry",
        content_hash="hash-loc-entry",
        title="Synthetic Entry Location",
        summary="A reviewed synthetic approach location.",
        confidence=0.96,
        review_status="approved",
        gate_status="runtime_ready",
        location_kind="site",
        section_ref="sec.entry",
        player_arrival_summary="A safe paraphrased arrival summary.",
        exits=[
            {
                "exit_id": "exit.to_area",
                "to_ref": "area.entry",
                "label": "entry door",
                "requirements": ["door accessible", "door accessible"],
            }
        ],
        handout_refs=["handout.entry"],
        table_refs=["table.entry"],
        map_template_refs=["map.entry"],
        front_refs=["front.clock"],
    )


def _keyed_area() -> KeyedAreaRecord:
    return KeyedAreaRecord(
        ref="area.entry",
        content_hash="hash-area-entry",
        title="Synthetic Keyed Area",
        summary="A reviewed synthetic keyed area.",
        confidence=0.97,
        review_status="approved",
        gate_status="runtime_ready",
        keyed_label="A1",
        parent_location_ref="loc.entry",
        section_ref="sec.entry",
        trap_refs=["trap.floor"],
        treasure_refs=["treasure.cache"],
        encounter_template_refs=["enc.entry"],
        exits=[
            {
                "exit_id": "exit.back",
                "to_ref": "loc.entry",
                "label": "back to approach",
            }
        ],
    )


def _reveal_edge() -> RevealGraphEdge:
    return RevealGraphEdge(
        ref="reveal.entry.handout",
        content_hash="hash-reveal-entry-handout",
        title="Synthetic reveal edge",
        summary="A synthetic reveal edge from area to handout.",
        confidence=0.95,
        review_status="approved",
        gate_status="runtime_ready",
        from_ref="area.entry",
        to_ref="handout.entry",
        relation="reveals",
        trigger="A character searches the synthetic cache.",
        audience="players",
    )


def _handout() -> HandoutRecord:
    return HandoutRecord(
        ref="handout.entry",
        content_hash="hash-handout-entry",
        title="Synthetic Handout",
        summary="A reviewed synthetic handout.",
        confidence=0.94,
        review_status="approved",
        gate_status="runtime_ready",
        handout_kind="document",
        safe_asset_ids=["asset.handout.entry", "asset.handout.entry"],
        player_safe_text="Player-safe synthetic handout text.",
        player_safe_caption="A safe synthetic caption.",
        player_safe_alt_text="A safe synthetic alt text.",
        possession_ref="treasure.cache",
        reading_constraints=["must hold the handout", "must hold the handout"],
        partial_reveals=[
            {
                "reveal_id": "reveal.safe.asset",
                "trigger": "The handout is opened.",
                "safe_text": "A safe partial reveal.",
                "safe_asset_ids": ["asset.handout.entry"],
            }
        ],
    )


def _table() -> AdventureTableRecord:
    return AdventureTableRecord(
        ref="table.entry",
        content_hash="hash-table-entry",
        title="Synthetic Entry Table",
        summary="A reviewed synthetic lookup table.",
        confidence=0.93,
        review_status="approved",
        gate_status="runtime_ready",
        table_kind="random_encounter",
        roll_formula="1d6",
        rows=[
            {
                "row_id": "row.1",
                "range_start": 1,
                "range_end": 3,
                "result_ref": "enc.entry",
                "result_summary": "Use the synthetic entry encounter.",
            },
            {
                "row_id": "row.2",
                "range_start": 4,
                "range_end": 6,
                "result_summary": "No synthetic encounter.",
            },
        ],
    )


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
        floors=[
            {
                "floor_id": "floor.ground",
                "label": "Ground floor",
                "grid_width": 12,
                "grid_height": 8,
                "area_refs": ["loc.entry"],
            }
        ],
        spawn_anchors=[
            {
                "anchor_id": "spawn.players",
                "anchor_kind": "players",
                "floor_id": "floor.ground",
                "cells": [{"x": 1, "y": 2}, {"x": 1, "y": 3}],
                "label": "Player start",
            }
        ],
        terrain_features=[
            {
                "feature_id": "wall.north",
                "feature_kind": "wall",
                "floor_id": "floor.ground",
                "bounds": {"x": 0, "y": 0, "width": 12, "height": 1},
                "blocks_movement": True,
                "blocks_line_of_sight": True,
            },
            {
                "feature_id": "secret.panel",
                "feature_kind": "secret_feature",
                "floor_id": "floor.ground",
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
                "floor_id": "floor.ground",
                "bounds": {"x": 0, "y": 0, "width": 12, "height": 8},
            }
        ],
        fog_masks=[
            {
                "mask_id": "fog.entry",
                "floor_id": "floor.ground",
                "bounds": {"x": 8, "y": 0, "width": 4, "height": 3},
                "area_refs": ["area.entry"],
                "revealed_by_region_refs": ["reveal.entry"],
            }
        ],
        reveal_regions=[
            {
                "reveal_id": "reveal.entry",
                "floor_id": "floor.ground",
                "cells": [{"x": 8, "y": 2}],
                "reveal_trigger": "A character searches the east wall.",
                "pov_area_refs": ["loc.entry"],
                "revealed_area_refs": ["area.entry"],
                "fog_mask_refs": ["fog.entry"],
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
        placements=[
            {
                "placement_id": "place.trap.floor",
                "location_ref": "loc.entry",
                "map_template_ref": "map.entry",
                "map_feature_ref": "feature.trap.floor",
                "area_ref": "area.entry",
                "floor_id": "floor.ground",
                "bounds": {"x": 1, "y": 1, "width": 2, "height": 2},
                "label": "Marked floor stones",
                "reveal_trigger": "The floor is inspected or triggered.",
            }
        ],
        runtime_consequences=["The entry alarm draws a patrol."],
        mechanics={
            "target": "creatures on the marked floor",
            "detection_dc": 13,
            "disarm_dc": 14,
            "save_dc": 12,
            "save_ability": "dexterity",
            "save_success": "The creature clears the floor stones.",
            "save_failure": "The floor spikes catch the creature.",
            "damage": [{"expression": "2d6", "damage_type": "piercing"}],
            "effects": ["the floor locks open"],
            "reset_policy": "manual reset from the service niche",
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
        field_provenance={
            "summary": [
                {
                    "source_asset_id": "asset.page.001",
                    "page_id": "page.001",
                    "span_id": "span.treasure.summary",
                    "method": "human-review",
                    "confidence": 0.97,
                    "human_review_status": "approved",
                }
            ]
        },
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


def _actor() -> ActorDossierRecord:
    return ActorDossierRecord(
        ref="actor.villain",
        content_hash="hash-actor-villain",
        title="Synthetic Villain Actor",
        summary="A reviewed actor dossier for a story-driving antagonist.",
        confidence=0.92,
        review_status="approved",
        gate_status="runtime_ready",
        actor_kind="villain",
        character_id_hint="villain",
        front_refs=["front.clock"],
        home_location_refs=["loc.entry"],
        statblock_ref="stat.guardian",
        agent_context_slice_ref="agent_context.villain.startup",
        goals=["Recover the synthetic key"],
        constraints=["Do not reveal the key's purpose too early"],
        resources=["scouts", "informants"],
        knowledge_channel_refs=["kg.villain.can_dispatch_scouts"],
        relationship_edges=[
            {
                "target_ref": "actor.scout",
                "stance": "commands",
                "summary": "Uses scouts to gather reports.",
            }
        ],
        initiative_triggers=["The cache is disturbed"],
        escalation_limits=["Avoid direct lethal pressure at campaign start"],
        secrets_known_refs=["treasure.cache"],
    )


def _agent_context_slice() -> AgentContextSliceRecord:
    return AgentContextSliceRecord(
        ref="agent_context.villain.startup",
        content_hash="hash-agent-context-villain-startup",
        title="Synthetic Villain Startup Context",
        summary="Reviewed startup context for a synthetic antagonist agent.",
        confidence=0.91,
        review_status="approved",
        gate_status="runtime_ready",
        actor_ref="actor.villain",
        slice_kind="strategic",
        known_context="The antagonist knows the entry cache is important.",
        private_state="The antagonist wants reports before acting directly.",
        current_agenda=["Watch the entry", "Recover the key"],
        beliefs=["The entry has been quiet recently"],
        uncertainties=["Who disturbed the cache"],
        hard_boundaries=["Do not act on unauthored rooms"],
        local_context_refs=["loc.entry"],
        graph_edge_refs=["kg.villain.can_dispatch_scouts"],
        refresh_triggers=["A scout reports new visitors"],
    )


def _knowledge_edge() -> KnowledgeGraphEdgeRecord:
    return KnowledgeGraphEdgeRecord(
        ref="kg.villain.can_dispatch_scouts",
        content_hash="hash-kg-villain-dispatch",
        title="Synthetic Villain Dispatch Edge",
        summary="The synthetic villain can dispatch scouts to the entry.",
        confidence=0.9,
        review_status="approved",
        gate_status="runtime_ready",
        from_ref="actor.villain",
        relation="can_dispatch",
        to_ref="loc.entry",
        object_kind="location",
        channel="scouts",
        condition="The entry cache is disturbed.",
        reliability=0.8,
        latency="next background beat",
        mutable=True,
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
