from __future__ import annotations

from app.engine.content_assets import (
    load_asset_catalog,
    safe_asset_reveals_for_viewer,
    write_asset_catalog,
)


RAW_SOURCE_PATH = "/private/table/maps/dm-map.png"
PROTECTED_EXCERPT = "PROTECTED MAP LABEL"


def test_asset_reveals_filter_per_pov_and_payloads_hide_private_fields():
    assets = [
        {
            "pack_id": "synthetic",
            "asset_id": "map-public",
            "kind": "player_safe_map",
            "title": "Entry map",
            "mime_type": "image/png",
            "width": 640,
            "height": 480,
            "sha256": "hash-map-public",
            "source_ref": "src-dm-map-001",
            "review_status": "approved",
            "spoiler_class": "low",
            "player_safe_alt_text": "A small room with one visible exit.",
            "player_safe_caption": "A sketched room map.",
            "delivery_ref": "asset://synthetic/map-public",
            "safe_for_players": True,
            "safe_for_llm": False,
            "metadata": {
                "source_path": RAW_SOURCE_PATH,
                "dm_notes": PROTECTED_EXCERPT,
                "raw_bytes": "not real bytes",
            },
        },
        {
            "pack_id": "synthetic",
            "asset_id": "handout-rogue",
            "kind": "handout",
            "title": "Folded note",
            "mime_type": "image/png",
            "width": 320,
            "height": 180,
            "sha256": "hash-handout",
            "source_ref": "src-handout-001",
            "review_status": "approved",
            "spoiler_class": "moderate",
            "player_safe_alt_text": "A folded note with a wax mark.",
            "player_safe_caption": "A note visible only to the holder.",
            "delivery_ref": "asset://synthetic/handout-rogue",
            "safe_for_players": True,
        },
    ]
    reveals = [
        {
            "pack_id": "synthetic",
            "asset_id": "map-public",
            "audience": "all_observers",
            "presentation": "map_overlay",
            "caption": "The visible part of the room.",
        },
        {
            "pack_id": "synthetic",
            "asset_id": "handout-rogue",
            "audience": "only",
            "visible_to_character_ids": ["rogue"],
            "presentation": "attachment",
        },
    ]

    rogue_payloads = safe_asset_reveals_for_viewer(
        assets,
        reveals,
        character_id="rogue",
        event_observer_ids=["rogue", "cleric"],
    )
    cleric_payloads = safe_asset_reveals_for_viewer(
        assets,
        reveals,
        character_id="cleric",
        event_observer_ids=["rogue", "cleric"],
    )

    assert [payload.asset_id for payload in rogue_payloads] == [
        "map-public",
        "handout-rogue",
    ]
    assert [payload.asset_id for payload in cleric_payloads] == ["map-public"]
    assert rogue_payloads[0].presentation == "map_overlay"
    assert rogue_payloads[1].presentation == "attachment"

    flattened = "\n".join(
        payload.model_dump_json() for payload in rogue_payloads + cleric_payloads
    )
    assert RAW_SOURCE_PATH not in flattened
    assert PROTECTED_EXCERPT not in flattened
    assert "dm_notes" not in flattened
    assert "raw_bytes" not in flattened
    assert "source_ref" not in flattened


def test_asset_catalog_persists_safe_refs_without_private_paths_or_notes(tmp_path):
    db_path = tmp_path / "assets.sqlite"
    write_asset_catalog(
        db_path,
        [
            {
                "pack_id": "synthetic",
                "asset_id": "map-safe",
                "kind": "player_safe_map",
                "title": "Safe crop",
                "mime_type": "image/png",
                "width": 200,
                "height": 160,
                "sha256": "hash-safe",
                "source_ref": "src-map-001",
                "review_status": "approved",
                "spoiler_class": "low",
                "player_safe_alt_text": "A safe crop.",
                "player_safe_caption": "A safe crop of the room.",
                "delivery_ref": "asset://synthetic/map-safe",
                "safe_for_players": True,
                "safe_for_llm": True,
                "metadata": {
                    "safe_tag": "fixture",
                    "source_path": RAW_SOURCE_PATH,
                    "dm_notes": PROTECTED_EXCERPT,
                    "nested": {"raw_text": PROTECTED_EXCERPT},
                },
            }
        ],
        protected_terms=[PROTECTED_EXCERPT],
    )

    raw_db = db_path.read_bytes()
    assert RAW_SOURCE_PATH.encode() not in raw_db
    assert PROTECTED_EXCERPT.encode() not in raw_db

    catalog = load_asset_catalog(db_path, pack_id="synthetic")
    asset = catalog["synthetic::map-safe"]
    assert asset.delivery_ref == "asset://synthetic/map-safe"
    assert asset.source_ref == "src-map-001"
    assert asset.metadata == {"safe_tag": "fixture", "nested": {}}


def test_unsafe_or_unreviewed_assets_are_not_revealed():
    assets = [
        {
            "pack_id": "synthetic",
            "asset_id": "dm-map",
            "kind": "dm_map",
            "title": "DM map",
            "delivery_ref": "asset://synthetic/dm-map",
            "review_status": "approved",
            "safe_for_players": False,
        },
        {
            "pack_id": "synthetic",
            "asset_id": "unreviewed-handout",
            "kind": "handout",
            "title": "Unreviewed handout",
            "delivery_ref": "asset://synthetic/unreviewed-handout",
            "review_status": "needs_review",
            "safe_for_players": True,
        },
        {
            "pack_id": "synthetic",
            "asset_id": "path-ref",
            "kind": "handout",
            "title": "Path backed handout",
            "delivery_ref": RAW_SOURCE_PATH,
            "review_status": "approved",
            "safe_for_players": True,
        },
    ]
    reveals = [
        {
            "pack_id": "synthetic",
            "asset_id": "dm-map",
            "audience": "all_observers",
        },
        {
            "pack_id": "synthetic",
            "asset_id": "unreviewed-handout",
            "audience": "all_observers",
        },
        {
            "pack_id": "synthetic",
            "asset_id": "path-ref",
            "audience": "all_observers",
        },
    ]

    assert (
        safe_asset_reveals_for_viewer(
            assets,
            reveals,
            character_id="cleric",
            event_observer_ids=["cleric"],
        )
        == []
    )
