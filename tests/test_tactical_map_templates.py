from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.engine import dnd_spatial
from app.engine.tactical_map_templates import (
    TacticalMapTemplateCompileError,
    compile_tactical_map_template,
    compile_tactical_map_template_battle_map_state,
)
from app.schemas.content_pack import ContentImageAsset, TacticalMapTemplateRecord


RAW_SOURCE_PATH = "/private/source/maps/entry-map.png"


def test_compiles_runtime_ready_template_to_immutable_geometry():
    compiled = compile_tactical_map_template(
        _template(),
        map_assets=[_asset()],
        authored_refs={"loc.entry", "area.upper", "trap.panel"},
        required_layers=[
            "map_ref",
            "spawn_anchors",
            "terrain",
            "areas",
            "secrets",
            "vertical_links",
        ],
    )

    assert compiled.ref == "map.entry"
    assert compiled.source_map_asset_id == "asset.map.entry.player"
    assert compiled.grid_width == 12
    assert compiled.spawn_anchors[0].cells[0].x == 1
    assert compiled.terrain_features[0].bounds is not None
    assert compiled.secret_features[0].feature_id == "secret.panel"
    assert compiled.vertical_links[0].feature_id == "stairs.upper"
    assert compiled.area_links[0].location_ref == "loc.entry"

    with pytest.raises(FrozenInstanceError):
        compiled.ref = "map.changed"  # type: ignore[misc]

    compiled_repr = repr(compiled)
    assert RAW_SOURCE_PATH not in compiled_repr
    assert "asset:///private" not in compiled_repr


def test_rejects_unready_or_missing_map_assets():
    with pytest.raises(TacticalMapTemplateCompileError, match="missing"):
        compile_tactical_map_template(_template(), map_assets=[])

    with pytest.raises(TacticalMapTemplateCompileError, match="needs_review"):
        compile_tactical_map_template(
            _template(),
            map_assets=[_asset(review_status="needs_review")],
        )


def test_rejects_path_like_map_asset_refs():
    with pytest.raises(TacticalMapTemplateCompileError, match="logical asset id"):
        compile_tactical_map_template(
            _template(derived_from_map_asset_id="/tmp/map.png"),
            map_assets=None,
        )


def test_rejects_templates_that_are_not_runtime_ready():
    with pytest.raises(TacticalMapTemplateCompileError, match="not runtime_ready"):
        compile_tactical_map_template(
            _template(review_status="approved", gate_status="flagged"),
            map_assets=[_asset()],
        )


def test_rejects_out_of_bounds_geometry_and_missing_refs():
    with pytest.raises(TacticalMapTemplateCompileError) as exc:
        compile_tactical_map_template(
            _template(
                spawn_anchors=[
                    {
                        "anchor_id": "spawn.players",
                        "anchor_kind": "players",
                        "cells": [{"x": 12, "y": 1}],
                    }
                ],
            ),
            map_assets=[_asset()],
            authored_refs={"loc.entry"},
        )

    message = str(exc.value)
    assert "outside map bounds" in message
    assert "missing content ref 'trap.panel'" in message
    assert "missing content ref 'area.upper'" in message


def test_required_floor_and_reveal_layers_fail_loudly_when_missing():
    with pytest.raises(TacticalMapTemplateCompileError) as exc:
        compile_tactical_map_template(
            _template(),
            map_assets=[_asset()],
            required_layers=["fog_reveal_regions", "floors_submaps"],
        )

    message = str(exc.value)
    assert "required fog-mask layer is missing" in message
    assert "required reveal-region layer is missing" in message
    assert "required floors/submaps layer is missing" in message


def test_compiles_multi_floor_fog_and_reveal_regions_to_runtime_state():
    compiled = compile_tactical_map_template(
        _template(
            floors=[
                {
                    "floor_id": "floor.ground",
                    "label": "Ground floor",
                    "grid_width": 12,
                    "grid_height": 8,
                    "area_refs": ["loc.entry"],
                },
                {
                    "floor_id": "floor.upper",
                    "floor_kind": "submap",
                    "label": "Upper landing",
                    "grid_width": 6,
                    "grid_height": 4,
                    "level_index": 1,
                    "elevation_ft": 10,
                    "parent_floor_id": "floor.ground",
                    "parent_bounds": {"x": 8, "y": 5, "width": 3, "height": 2},
                    "map_asset_id": "asset.map.upper.player",
                    "area_refs": ["area.upper"],
                },
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
                    "feature_id": "stairs.upper",
                    "feature_kind": "stairs",
                    "floor_id": "floor.ground",
                    "cells": [{"x": 10, "y": 6}],
                    "label": "Upper stairs",
                    "linked_refs": ["area.upper"],
                },
                {
                    "feature_id": "balcony.rail",
                    "feature_kind": "cover",
                    "floor_id": "floor.upper",
                    "bounds": {"x": 2, "y": 1, "width": 2, "height": 1},
                    "label": "Balcony rail",
                    "cover": "half",
                },
            ],
            area_links=[
                {
                    "area_id": "area.entry",
                    "location_ref": "loc.entry",
                    "floor_id": "floor.ground",
                    "bounds": {"x": 0, "y": 0, "width": 12, "height": 8},
                },
                {
                    "area_id": "area.upper",
                    "location_ref": "area.upper",
                    "floor_id": "floor.upper",
                    "bounds": {"x": 0, "y": 0, "width": 6, "height": 4},
                },
            ],
            fog_masks=[
                {
                    "mask_id": "fog.upper",
                    "floor_id": "floor.upper",
                    "bounds": {"x": 0, "y": 0, "width": 6, "height": 4},
                    "label": "Upper landing fog",
                    "area_refs": ["area.upper"],
                    "revealed_by_region_refs": ["reveal.upper"],
                }
            ],
            reveal_regions=[
                {
                    "reveal_id": "reveal.upper",
                    "floor_id": "floor.ground",
                    "cells": [{"x": 10, "y": 6}],
                    "label": "View up the stairs",
                    "reveal_trigger": "A character reaches the stair landing.",
                    "pov_character_ids": ["pc.rogue", "pc.rogue"],
                    "pov_area_refs": ["loc.entry"],
                    "revealed_area_refs": ["area.upper"],
                    "fog_mask_refs": ["fog.upper"],
                }
            ],
        ),
        map_assets=[_asset(), _asset(asset_id="asset.map.upper.player")],
        authored_refs={"loc.entry", "area.upper"},
        required_layers=[
            "map_ref",
            "floors_submaps",
            "fog_reveal_regions",
            "vertical_links",
        ],
    )

    assert [floor.floor_id for floor in compiled.floors] == [
        "floor.ground",
        "floor.upper",
    ]
    assert compiled.floors[1].parent_bounds is not None
    assert compiled.floors[1].parent_bounds.x == 8
    assert compiled.fog_masks[0].revealed_by_region_refs == ("reveal.upper",)
    assert compiled.reveal_regions[0].pov_character_ids == ("pc.rogue",)
    assert compiled.terrain_features[1].floor_id == "floor.upper"

    battle_map = compiled.to_battle_map_state()

    assert battle_map.floors[1].parent_floor_id == "floor.ground"
    assert battle_map.fog_masks[0].mask_id == "fog.upper"
    assert battle_map.reveal_regions[0].revealed_area_refs == ["area.upper"]
    assert battle_map.features[1].floor_id == "floor.upper"

    public_text = repr(dnd_spatial.battle_map_status(battle_map))
    assert "A character reaches the stair landing" not in public_text
    assert "fog.upper" not in public_text
    assert "asset.map.upper.player" not in public_text
    assert RAW_SOURCE_PATH not in battle_map.model_dump_json()


def test_rejects_invalid_floor_and_reveal_refs_or_unsafe_shapes():
    with pytest.raises(TacticalMapTemplateCompileError) as exc:
        compile_tactical_map_template(
            _template(
                floors=[
                    {
                        "floor_id": "floor.ground",
                        "label": "Ground floor",
                        "grid_width": 12,
                        "grid_height": 8,
                    },
                    {
                        "floor_id": "floor.upper",
                        "floor_kind": "submap",
                        "grid_width": 6,
                        "grid_height": 4,
                        "parent_floor_id": "floor.missing",
                        "parent_bounds": {"x": 99, "y": 0, "width": 1, "height": 1},
                    },
                ],
                fog_masks=[
                    {
                        "mask_id": "fog.upper",
                        "floor_id": "floor.upper",
                        "cells": [{"x": 0, "y": 0}],
                        "bounds": {"x": 0, "y": 0, "width": 1, "height": 1},
                        "revealed_by_region_refs": ["reveal.missing"],
                    }
                ],
                reveal_regions=[
                    {
                        "reveal_id": "reveal.upper",
                        "floor_id": "floor.missing",
                        "bounds": {"x": 0, "y": 0, "width": 1, "height": 1},
                        "reveal_trigger": f"Open {RAW_SOURCE_PATH}",
                        "revealed_area_refs": ["area.missing"],
                        "fog_mask_refs": ["fog.missing"],
                    }
                ],
            ),
            map_assets=[_asset()],
            authored_refs={"loc.entry", "area.upper", "trap.panel"},
            required_layers=["floors_submaps", "fog_reveal_regions"],
        )

    message = str(exc.value)
    assert "references missing parent floor 'floor.missing'" in message
    assert "parent_bounds bounds origin is outside map bounds" in message
    assert "must use cells or bounds, not both" in message
    assert "references missing map floor 'floor.missing'" in message
    assert "must not contain raw source paths" in message
    assert "references missing content ref 'area.missing'" in message
    assert "references missing reveal region 'reveal.missing'" in message
    assert "references missing fog mask 'fog.missing'" in message


def test_compiled_runtime_map_state_preserves_imported_template_semantics():
    compiled = compile_tactical_map_template(
        _template(),
        map_assets=[_asset()],
        authored_refs={"loc.entry", "area.upper", "trap.panel"},
    )

    battle_map = compiled.to_battle_map_state()

    assert battle_map.source_template_ref == "map.entry"
    assert battle_map.spawn_anchors[0].anchor_id == "spawn.players"
    assert battle_map.area_links[0].location_ref == "loc.entry"
    secret = next(
        feature for feature in battle_map.features
        if feature.feature_id == "secret.panel"
    )
    assert secret.secret is True
    assert secret.reveal_trigger == "A character searches the east wall."
    difficult = next(
        feature for feature in battle_map.features
        if feature.feature_id == "rubble.slope"
    )
    assert difficult.difficult_terrain is True
    stairs = next(
        feature for feature in battle_map.features
        if feature.feature_id == "stairs.upper"
    )
    assert stairs.linked_refs == ["area.upper"]

    player_payload = dnd_spatial.battle_map_status(battle_map)
    player_text = repr(player_payload)
    assert "secret.panel" not in player_text
    assert "A character searches" not in player_text
    assert "spawn.players" not in player_text
    assert "loc.entry" not in player_text
    assert "area.upper" not in player_text
    assert "hash-map-entry" not in player_text
    assert "rubble.slope" in player_text


def test_simple_visible_template_projects_to_battle_map_state():
    battle_map = compile_tactical_map_template_battle_map_state(
        _template(
            spawn_anchors=[],
            area_links=[],
            terrain_features=[
                {
                    "feature_id": "crate",
                    "feature_kind": "cover",
                    "bounds": {"x": 3, "y": 2, "width": 1, "height": 1},
                    "label": "Crate",
                    "blocks_movement": True,
                    "cover": "half",
                }
            ],
        ),
        map_assets=[_asset()],
        required_layers=["map_ref", "terrain"],
    )

    assert battle_map.present is True
    assert battle_map.map_name == "Entry Map"
    assert battle_map.width == 12
    assert battle_map.terrain[0].zone_id == "crate"
    assert battle_map.terrain[0].cover == "half"
    assert battle_map.features[0].feature_id == "crate"
    assert battle_map.tokens == []
    assert battle_map.areas == []
    assert "asset.map.entry.player" not in battle_map.model_dump_json()


def _asset(**overrides) -> ContentImageAsset:
    data = {
        "pack_id": "synthetic-pack",
        "asset_id": "asset.map.entry.player",
        "kind": "player_safe_map",
        "title": "Entry map display artifact",
        "mime_type": "image/png",
        "width": 800,
        "height": 600,
        "sha256": "hash-map-asset",
        "source_ref": "source.map.entry",
        "review_status": "approved",
        "safe_for_players": True,
        "delivery_ref": "asset://synthetic-pack/asset.map.entry.player",
        "metadata": {"source_path": RAW_SOURCE_PATH},
    }
    data.update(overrides)
    return ContentImageAsset(**data)


def _template(**overrides) -> TacticalMapTemplateRecord:
    data = {
        "pack_id": "synthetic-pack",
        "ref": "map.entry",
        "content_hash": "hash-map-entry",
        "title": "Entry Map",
        "summary": "Reviewed synthetic tactical geometry.",
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "derived_from_map_asset_id": "asset.map.entry.player",
        "grid_width": 12,
        "grid_height": 8,
        "square_size_ft": 5,
        "spawn_anchors": [
            {
                "anchor_id": "spawn.players",
                "anchor_kind": "players",
                "cells": [{"x": 1, "y": 2}, {"x": 1, "y": 3}],
                "label": "Player start",
            }
        ],
        "terrain_features": [
            {
                "feature_id": "wall.north",
                "feature_kind": "wall",
                "bounds": {"x": 0, "y": 0, "width": 12, "height": 1},
                "blocks_movement": True,
                "blocks_line_of_sight": True,
            },
            {
                "feature_id": "rubble.slope",
                "feature_kind": "difficult_ground",
                "bounds": {"x": 4, "y": 4, "width": 2, "height": 2},
                "label": "Loose rubble",
                "difficult_terrain": True,
            },
            {
                "feature_id": "secret.panel",
                "feature_kind": "secret_feature",
                "cells": [{"x": 9, "y": 2}],
                "secret": True,
                "reveal_trigger": "A character searches the east wall.",
                "linked_refs": ["trap.panel"],
            },
            {
                "feature_id": "stairs.upper",
                "feature_kind": "stairs",
                "cells": [{"x": 10, "y": 6}],
                "label": "Upper stairs",
                "linked_refs": ["area.upper"],
            },
        ],
        "area_links": [
            {
                "area_id": "area.entry",
                "location_ref": "loc.entry",
                "bounds": {"x": 0, "y": 0, "width": 12, "height": 8},
            }
        ],
    }
    data.update(overrides)
    return TacticalMapTemplateRecord(**data)
