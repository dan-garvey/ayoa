from __future__ import annotations

import hashlib

import pytest

from app.engine.content_asset_bytes import (
    AssetByteResolutionError,
    resolve_asset_bytes,
)
from app.schemas.content_pack import ContentImageAsset, SafeAssetRevealPayload


PACK_ID = "synthetic"
ASSET_ID = "map-safe"
MEDIA_BYTES = b"synthetic player-safe image bytes"


def test_valid_asset_resolves_from_content_addressed_file_and_verifies_hash(
    tmp_path,
):
    asset = _asset(review_status="reviewed")
    media_root = tmp_path / "media"
    _write_media(media_root, asset.sha256, MEDIA_BYTES)

    resolved = resolve_asset_bytes(
        _payload(asset),
        asset,
        media_roots={PACK_ID: media_root},
    )

    assert resolved.data == MEDIA_BYTES
    assert resolved.sha256 == asset.sha256
    assert resolved.byte_count == len(MEDIA_BYTES)
    assert resolved.mime_type == "image/png"
    assert resolved.width == 640
    assert resolved.height == 480
    assert resolved.pack_id == PACK_ID
    assert resolved.asset_id == ASSET_ID
    assert resolved.delivery_ref == f"asset://{PACK_ID}/{ASSET_ID}"


def test_safe_filename_does_not_expose_source_filenames_paths_or_captions(tmp_path):
    asset = _asset(
        title="private-source-map.png",
        player_safe_caption="Caption with original-map.png",
        metadata={"safe_label": "source-map.png"},
    )
    media_root = tmp_path / "media"
    _write_media(media_root, asset.sha256, MEDIA_BYTES)
    payload = _payload(asset, caption="/private/table/original-map.png")

    resolved = resolve_asset_bytes(
        payload,
        asset,
        media_roots={PACK_ID: media_root},
    )

    assert resolved.filename == f"asset-{asset.sha256[:16]}.png"
    assert PACK_ID not in resolved.filename
    assert ASSET_ID not in resolved.filename
    assert "private-source-map" not in resolved.filename
    assert "source-map" not in resolved.filename
    assert "original-map" not in resolved.filename
    assert "/" not in resolved.filename


def test_cache_root_can_back_asset_when_media_root_does_not_have_it(tmp_path):
    asset = _asset()
    cache_root = tmp_path / "cache"
    _write_media(cache_root, asset.sha256, MEDIA_BYTES)

    resolved = resolve_asset_bytes(
        _payload(asset),
        asset,
        media_roots={PACK_ID: tmp_path / "media"},
        cache_roots={PACK_ID: cache_root},
    )

    assert resolved.data == MEDIA_BYTES


@pytest.mark.parametrize(
    ("setup", "expected_code"),
    [
        ("missing_media", "missing_media"),
        ("hash_mismatch", "hash_mismatch"),
        ("oversized", "asset_too_large"),
        ("unsafe_mime", "unsafe_mime_type"),
        ("unreviewed", "asset_not_approved"),
        ("blocked", "asset_not_approved"),
        ("not_player_safe", "asset_not_player_safe"),
        ("delivery_ref_mismatch", "delivery_ref_mismatch"),
        ("unsafe_payload_ref", "unsafe_delivery_ref"),
        ("unsafe_asset_ref", "unsafe_delivery_ref"),
        ("unsafe_metadata", "unsafe_asset_metadata"),
    ],
)
def test_invalid_assets_fail_loudly_with_typed_non_spoiling_errors(
    tmp_path,
    setup,
    expected_code,
):
    asset = _asset()
    payload_override = {}
    max_bytes = 1024
    write_bytes = MEDIA_BYTES
    write_file = setup not in {"missing_media", "unsafe_mime"}

    if setup == "hash_mismatch":
        write_bytes = b"different bytes"
    elif setup == "oversized":
        max_bytes = len(MEDIA_BYTES) - 1
    elif setup == "unsafe_mime":
        asset = _asset(mime_type="image/svg+xml")
    elif setup == "unreviewed":
        asset = _asset(review_status="unreviewed")
    elif setup == "blocked":
        asset = _asset(review_status="blocked")
    elif setup == "not_player_safe":
        asset = _asset(safe_for_players=False)
    elif setup == "delivery_ref_mismatch":
        payload_override["delivery_ref"] = f"asset://{PACK_ID}/other-map"
    elif setup == "unsafe_payload_ref":
        payload_override["delivery_ref"] = "https://cdn.example.invalid/map.png"
    elif setup == "unsafe_asset_ref":
        asset = _asset(delivery_ref="/private/table/maps/source-map.png")
    elif setup == "unsafe_metadata":
        asset = _asset(metadata={"source_path": "/private/table/maps/source-map.png"})

    media_root = tmp_path / "media"
    if write_file:
        _write_media(media_root, asset.sha256, write_bytes)

    payload = _payload(asset, **payload_override)
    with pytest.raises(AssetByteResolutionError) as exc_info:
        resolve_asset_bytes(
            payload,
            asset,
            media_roots={PACK_ID: media_root},
            max_bytes=max_bytes,
        )

    error = exc_info.value
    assert error.code == expected_code
    assert error.pack_id == PACK_ID
    assert error.asset_id == ASSET_ID
    assert PACK_ID not in str(error)
    assert ASSET_ID not in str(error)
    assert "source-map" not in str(error)
    assert "/private/table" not in str(error)


@pytest.mark.parametrize(
    "delivery_ref",
    [
        "file:///private/table/maps/source-map.png",
        "/private/table/maps/source-map.png",
        "maps/source-map.png",
        "../maps/source-map.png",
        "s3://bucket/source-map.png",
        "gs://bucket/source-map.png",
        "data:image/png;base64,AAAA",
        "javascript:alert(1)",
        "ftp://example.invalid/source-map.png",
        "https://example.invalid/source-map.png",
    ],
)
def test_unsafe_delivery_schemes_and_paths_fail_without_reading_files(
    tmp_path,
    delivery_ref,
):
    asset = _asset()
    media_root = tmp_path / "media"
    _write_media(media_root, asset.sha256, MEDIA_BYTES)
    payload = _payload(asset, delivery_ref=delivery_ref)

    with pytest.raises(AssetByteResolutionError) as exc_info:
        resolve_asset_bytes(payload, asset, media_roots={PACK_ID: media_root})

    assert exc_info.value.code == "unsafe_delivery_ref"


def test_payload_and_catalog_identity_must_match(tmp_path):
    asset = _asset()
    media_root = tmp_path / "media"
    _write_media(media_root, asset.sha256, MEDIA_BYTES)
    payload = _payload(asset, pack_id="other-pack")

    with pytest.raises(AssetByteResolutionError) as exc_info:
        resolve_asset_bytes(payload, asset, media_roots={PACK_ID: media_root})

    assert exc_info.value.code == "pack_mismatch"


def _asset(**overrides) -> ContentImageAsset:
    values = {
        "pack_id": PACK_ID,
        "asset_id": ASSET_ID,
        "kind": "player_safe_map",
        "title": "Safe map",
        "mime_type": "image/png",
        "width": 640,
        "height": 480,
        "sha256": _sha256(MEDIA_BYTES),
        "source_ref": "private-source-ref",
        "review_status": "approved",
        "spoiler_class": "low",
        "player_safe_alt_text": "A safe map crop.",
        "player_safe_caption": "A safe map.",
        "delivery_ref": f"asset://{PACK_ID}/{ASSET_ID}",
        "safe_for_players": True,
        "safe_for_llm": False,
        "metadata": {"safe_tag": "fixture"},
    }
    values.update(overrides)
    return ContentImageAsset(**values)


def _payload(asset: ContentImageAsset, **overrides) -> SafeAssetRevealPayload:
    values = {
        "pack_id": asset.pack_id,
        "asset_id": asset.asset_id,
        "kind": asset.kind,
        "title": asset.title,
        "mime_type": asset.mime_type,
        "width": asset.width,
        "height": asset.height,
        "sha256": asset.sha256,
        "delivery_ref": asset.delivery_ref,
        "presentation": "attachment",
        "caption": asset.player_safe_caption,
        "alt_text": asset.player_safe_alt_text,
    }
    values.update(overrides)
    return SafeAssetRevealPayload(**values)


def _write_media(root, sha256: str, data: bytes) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{sha256}.png").write_bytes(data)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
