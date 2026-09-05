from __future__ import annotations

import json

import pytest

from app.engine.imported_trap_hazards import (
    ImportedTrapHazardCatalog,
    ImportedTrapHazardValidationError,
    catalog_from_content_state,
    trap_overlay_from_content_state,
)
from app.schemas.content import (
    ContentOverlayState,
    ContentPackState,
    ContentTrapOverlayState,
    content_overlay_key,
)
from app.schemas.content_pack import ContentPackDomainCatalog


def test_reviewed_trap_hazard_context_reaches_router_and_combat_only():
    domain_catalog = ContentPackDomainCatalog(
        pack_id="synthetic-pack",
        trap_hazards=[_trap_hazard()],
    )
    catalog = ImportedTrapHazardCatalog.from_domain_catalog(domain_catalog)
    overlay = ContentTrapOverlayState(
        trap_id="trap.needle",
        content_hash="sha256:trap-needle",
    )

    router_record = catalog.router_context_records(
        ["trap.needle"],
        overlays={overlay.overlay_key(): overlay},
    )[0]
    combat_record = catalog.combat_context_records(
        ["trap.needle"],
        overlays={overlay.overlay_key(): overlay},
    )[0]
    payload = _payload(router_record)

    assert router_record.startswith("trap_hazard_context audience=router")
    assert combat_record.startswith(
        "trap_hazard_context audience=dnd_combat_resolver"
    )
    assert payload["ref"] == "trap.needle"
    assert "content_hash" not in payload
    assert payload["kind"] == "trap"
    assert payload["state"]["status"] == "hidden"
    assert payload["trigger"] == "A creature opens the reliquary latch."
    assert payload["detection"] == "Inspection reveals a thin seam in the latch."
    assert payload["countermeasures"] == ["Jam the needle port"]
    assert payload["mechanics"]["detection_dc"] == 15
    assert payload["mechanics"]["disarm_dc"] == 13
    assert payload["mechanics"]["save"] == {
        "ability": "dexterity",
        "dc": 14,
        "failure": "The needle catches the opener.",
        "success": "The opener twists away before the needle lands.",
    }
    assert payload["mechanics"]["damage"] == [
        {"damage_type": "poison", "expression": "2d10"}
    ]
    assert payload["mechanics"]["effects"] == ["alarm bell rings"]
    assert payload["runtime_consequences"] == [
        "The reliquary alarm alerts the next room."
    ]
    assert payload["placements"] == [
        {
            "area_ref": "area.reliquary",
            "bounds": {"height": 1, "width": 1, "x": 4, "y": 2},
            "floor_id": "floor.ground",
            "hidden": True,
            "label": "Reliquary latch",
            "location_ref": "loc.reliquary",
            "map_feature_ref": "feature.latch",
            "map_template_ref": "map.reliquary",
            "placement_id": "place.latch",
            "reveal_trigger": "The latch is inspected or triggered.",
        }
    ]
    assert "/private/source.pdf" not in router_record
    assert "PROTECTED_SOURCE_EXCERPT" not in router_record


def test_hidden_trap_hazard_does_not_render_to_players_or_logs_before_reveal():
    catalog = ImportedTrapHazardCatalog([_trap_hazard()])
    hidden_overlay = ContentTrapOverlayState(
        trap_id="trap.needle",
        content_hash="sha256:trap-needle",
    )

    assert catalog.player_safe_records(
        ["trap.needle"],
        overlays={hidden_overlay.overlay_key(): hidden_overlay},
    ) == []

    log_record = catalog.ordinary_log_records(
        ["trap.needle"],
        overlays={hidden_overlay.overlay_key(): hidden_overlay},
    )[0]

    assert "trap_hazard_state" in log_record
    assert "state=hidden" in log_record
    for hidden_text in (
        "opens the reliquary latch",
        "thin seam",
        "Jam the needle port",
        "2d10",
        "alarm alerts",
    ):
        assert hidden_text not in log_record

    revealed_overlay = ContentTrapOverlayState(
        trap_id="trap.needle",
        content_hash="sha256:trap-needle",
        revealed=True,
    )
    player_records = catalog.player_safe_records(
        ["trap.needle"],
        overlays={revealed_overlay.overlay_key(): revealed_overlay},
    )

    assert len(player_records) == 1
    visible_payload = _payload(player_records[0])
    assert visible_payload == {
        "content_hash": "sha256:trap-needle",
        "countermeasures": ["Jam the needle port"],
        "detection": "Inspection reveals a thin seam in the latch.",
        "kind": "trap",
        "pack_id": "synthetic-pack",
        "ref": "trap.needle",
        "state": "revealed",
        "summary": "A reviewed reliquary latch trap.",
        "title": "Reliquary Needle",
    }


def test_validation_blocks_unreviewed_missing_mechanics_and_unsafe_text():
    missing_mechanics = _trap_hazard()
    missing_mechanics.pop("mechanics")
    missing_catalog = ImportedTrapHazardCatalog([missing_mechanics])

    with pytest.raises(ImportedTrapHazardValidationError, match="mechanics"):
        missing_catalog.router_context_records()

    unreviewed = _trap_hazard()
    unreviewed["review_status"] = "needs_review"
    unreviewed["gate_status"] = "flagged"
    unreviewed_catalog = ImportedTrapHazardCatalog([unreviewed])

    with pytest.raises(ImportedTrapHazardValidationError, match="review_status"):
        unreviewed_catalog.combat_context_records()

    unsafe = _trap_hazard()
    unsafe["trigger"] = "A creature opens /private/source.pdf."
    unsafe_catalog = ImportedTrapHazardCatalog([unsafe])

    with pytest.raises(ImportedTrapHazardValidationError, match="unsafe:trigger"):
        unsafe_catalog.router_context_records()


def test_catalog_from_content_state_reads_records_and_overlay_state():
    trap = _trap_hazard()
    overlay = ContentTrapOverlayState(
        trap_id=trap["ref"],
        content_hash=trap["content_hash"],
        sprung=True,
        depleted=True,
    )
    content_state = {
        "synthetic-pack": ContentPackState(
            pack_id="synthetic-pack",
            metadata={"trap_hazards": [trap]},
            overlay=ContentOverlayState(
                traps={content_overlay_key(trap["ref"], trap["content_hash"]): overlay}
            ),
        )
    }

    catalog = catalog_from_content_state(content_state)

    assert catalog is not None
    assert catalog.refs == ("trap.needle",)
    resolved = catalog.resolve("trap.needle")
    resolved_overlay = trap_overlay_from_content_state(content_state, resolved)
    assert resolved_overlay is not None
    assert resolved_overlay.depleted is True

    payload = _payload(
        catalog.router_context_records(
            ["trap.needle"],
            overlays=content_state["synthetic-pack"],
        )[0]
    )
    assert payload["state"]["status"] == "depleted"
    assert payload["state"]["depletion_ref"] == "depleted.trap.needle"


def _payload(record: str) -> dict:
    return json.loads(record.split("payload=", 1)[1])


def _trap_hazard() -> dict:
    return {
        "pack_id": "synthetic-pack",
        "ref": "trap.needle",
        "content_hash": "sha256:trap-needle",
        "title": "Reliquary Needle",
        "summary": "A reviewed reliquary latch trap.",
        "body": "/private/source.pdf raw_ocr=PROTECTED_SOURCE_EXCERPT",
        "confidence": 0.96,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "trap_hazard_kind": "trap",
        "visibility": "router_hidden",
        "trigger": "A creature opens the reliquary latch.",
        "detection": "Inspection reveals a thin seam in the latch.",
        "countermeasures": ["Jam the needle port"],
        "linked_location_refs": ["loc.reliquary"],
        "linked_map_feature_refs": ["feature.latch"],
        "placements": [
            {
                "placement_id": "place.latch",
                "location_ref": "loc.reliquary",
                "map_template_ref": "map.reliquary",
                "map_feature_ref": "feature.latch",
                "area_ref": "area.reliquary",
                "floor_id": "floor.ground",
                "bounds": {"x": 4, "y": 2, "width": 1, "height": 1},
                "label": "Reliquary latch",
                "hidden": True,
                "reveal_trigger": "The latch is inspected or triggered.",
            }
        ],
        "runtime_consequences": [
            "The reliquary alarm alerts the next room."
        ],
        "mechanics": {
            "target": "the creature opening the latch",
            "detection_dc": 15,
            "disarm_dc": 13,
            "save_dc": 14,
            "save_ability": "dexterity",
            "save_success": "The opener twists away before the needle lands.",
            "save_failure": "The needle catches the opener.",
            "damage": [{"expression": "2d10", "damage_type": "poison"}],
            "effects": ["alarm bell rings"],
            "reset_policy": "manual reset after the needle is replaced",
            "depletion_ref": "depleted.trap.needle",
        },
    }
