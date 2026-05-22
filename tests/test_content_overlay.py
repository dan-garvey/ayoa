from __future__ import annotations

import json
from copy import deepcopy

from app.schemas.checkpoint import CheckpointFile
from app.schemas.content import (
    ContentActivePlanState,
    ContentCombatMapOverlayState,
    ContentDoorOverlayState,
    ContentFrontKnowledgeState,
    ContentLocationOverlayState,
    ContentModuleOverrideState,
    ContentOverlayState,
    ContentPackState,
    ContentSpawnOverlayState,
    ContentTrapOverlayState,
    ContentTreasureOverlayState,
)
from tests.support.factories import checkpoint


def test_content_overlay_round_trips_through_checkpoint_dump() -> None:
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            overlay=ContentOverlayState(
                locations={
                    "stale": ContentLocationOverlayState(
                        location_id=" location/chapel ",
                        content_hash=" sha256:loc ",
                        revealed=True,
                        visited=True,
                    )
                },
                doors={
                    "door": ContentDoorOverlayState(
                        door_id="door/north",
                        content_hash="sha256:door",
                        opened=True,
                    )
                },
                traps={
                    "trap": ContentTrapOverlayState(
                        trap_id="trap/pit",
                        content_hash="sha256:trap",
                        revealed=True,
                        sprung=True,
                    )
                },
                treasures={
                    "treasure": ContentTreasureOverlayState(
                        treasure_id="treasure/altar",
                        content_hash="sha256:loot",
                        looted=True,
                        depleted=True,
                        claimed_ref_ids=["item/gem", "item/gem", "coin/cache"],
                    )
                },
                spawn_refs={
                    "spawn": ContentSpawnOverlayState(
                        spawn_ref_id="spawn/wight-1",
                        content_hash="sha256:spawn",
                        source_ref_id="encounter/crypt",
                        current_location_ref_id="location/crypt",
                        current_location_hash="sha256:crypt",
                        status="moved",
                    )
                },
                combat_maps={
                    "map": ContentCombatMapOverlayState(
                        map_id="map/crypt",
                        content_hash="sha256:map",
                        revealed_area_ref_ids=["area/entry", "area/entry"],
                        fogged_area_ref_ids=["area/altar"],
                        visible_spawn_ref_ids=["spawn/wight-1"],
                    )
                },
                front_knowledge={
                    "front": ContentFrontKnowledgeState(
                        front_id="front/curse",
                        content_hash="sha256:front",
                        status="known",
                        known_ref_ids=["npc/abbot"],
                        revealed_clue_ref_ids=["clue/sigil"],
                    )
                },
                active_plans={
                    "plan": ContentActivePlanState(
                        plan_id="plan/midnight-rite",
                        content_hash="sha256:plan",
                        front_id="front/curse",
                        owner_ref_id="npc/abbot",
                        current_step_ref_id="step/gather-relics",
                        target_ref_ids=["location/chapel"],
                        completed_step_ref_ids=["step/find-site"],
                        status="active",
                    )
                },
                module_overrides={
                    "override": ContentModuleOverrideState(
                        override_id="override/chapel-door",
                        target_ref_id="door/north",
                        content_hash="sha256:override",
                        kind="state",
                        flags={"opened": True},
                        ref_overrides={"door/north": "door/north-open"},
                    )
                },
            ),
        )
    }

    payload = ckpt.model_dump(mode="json")
    rebuilt = CheckpointFile.model_validate(payload)

    assert rebuilt.model_dump(mode="json") == payload
    overlay = rebuilt.session.content_state["pack"].overlay
    assert sorted(overlay.locations) == ["location/chapel::sha256:loc"]
    assert overlay.doors["door/north::sha256:door"].opened is True
    assert overlay.traps["trap/pit::sha256:trap"].sprung is True
    assert overlay.treasures["treasure/altar::sha256:loot"].claimed_ref_ids == [
        "item/gem",
        "coin/cache",
    ]
    assert overlay.spawn_refs["spawn/wight-1::sha256:spawn"].status == "moved"
    assert overlay.combat_maps["map/crypt::sha256:map"].fogged_area_ref_ids == [
        "area/altar"
    ]
    assert overlay.front_knowledge["front/curse::sha256:front"].status == "known"
    plan = overlay.active_plans["plan/midnight-rite::sha256:plan"]
    assert plan.status == "active"
    assert overlay.module_overrides[
        "override/chapel-door::sha256:override"
    ].ref_overrides == {"door/north": "door/north-open"}


def test_content_overlay_default_dump_omits_private_module_material() -> None:
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            overlay=ContentOverlayState(
                locations={
                    "unsafe": ContentLocationOverlayState(
                        location_id="/private/table/source-map.png",
                        content_hash="sha256:loc",
                        visited=True,
                    )
                },
                combat_maps={
                    "map": ContentCombatMapOverlayState(
                        map_id="map/crypt",
                        content_hash="sha256:map",
                        revealed_area_ref_ids=[
                            "area/entry",
                            "private_extractions/page-1",
                        ],
                    )
                },
                module_overrides={
                    "unsafe": ContentModuleOverrideState(
                        override_id="override/main",
                        target_ref_id="asset://secret/hidden-map",
                        content_hash="sha256:override",
                        flags={
                            "opened": True,
                            "protected_excerpt": True,
                            "source_path": True,
                            "raw text": True,
                        },
                        ref_overrides={
                            "door/north": "door/north-open",
                            "raw_text": "/private/source.pdf",
                        },
                    )
                },
            ),
        )
    }

    dumped = json.dumps(ckpt.model_dump(mode="json"), sort_keys=True)

    for sentinel in (
        "/private/table/source-map.png",
        "private_extractions/page-1",
        "asset://secret/hidden-map",
        "protected_excerpt",
        "source_path",
        "raw text",
        "/private/source.pdf",
        "raw_text",
    ):
        assert sentinel not in dumped
    assert "door/north-open" in dumped
    assert '"opened": true' in dumped


def test_content_overlay_restores_from_deepcopy_snapshot() -> None:
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            overlay=ContentOverlayState(
                locations={
                    "location/chapel": ContentLocationOverlayState(
                        location_id="location/chapel",
                        content_hash="sha256:loc",
                        visited=True,
                    )
                },
                doors={
                    "door/north": ContentDoorOverlayState(
                        door_id="door/north",
                        content_hash="sha256:door",
                        opened=False,
                    )
                },
            ),
        )
    }
    rewind_source = ckpt.model_copy(deep=True)
    content_snapshot = deepcopy(ckpt.session.content_state)

    overlay = ckpt.session.content_state["pack"].overlay
    overlay.locations["location/chapel::sha256:loc"].visited = False
    overlay.doors["door/north::sha256:door"].opened = True
    overlay.treasures["treasure/altar"] = ContentTreasureOverlayState(
        treasure_id="treasure/altar",
        content_hash="sha256:loot",
        depleted=True,
    )

    assert (
        rewind_source.session.content_state["pack"]
        .overlay.locations["location/chapel::sha256:loc"]
        .visited
        is True
    )
    ckpt.session.content_state = deepcopy(content_snapshot)

    restored = ckpt.session.content_state["pack"].overlay
    assert restored.locations["location/chapel::sha256:loc"].visited is True
    assert restored.doors["door/north::sha256:door"].opened is False
    assert restored.treasures == {}
