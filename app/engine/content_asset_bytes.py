from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.engine.content_assets import (
    APPROVED_ASSET_REVIEW_STATUSES,
)
from app.schemas.content_pack import ContentImageAsset, SafeAssetRevealPayload


DEFAULT_MAX_ASSET_BYTES = 8 * 1024 * 1024
SAFE_IMAGE_MIME_EXTENSIONS: Mapping[str, str] = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^/\s]+/)+[^/\s]+")
_UNSAFE_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ASSET_DELIVERY_REF_RE = re.compile(
    r"^asset://(?P<pack_id>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<asset_id>[A-Za-z0-9][A-Za-z0-9_.-]*)$"
)
_FORBIDDEN_METADATA_KEYS = {
    "dm_notes",
    "file_path",
    "hidden_labels",
    "local_path",
    "path",
    "raw_bytes",
    "raw_ocr",
    "raw_source_path",
    "raw_text",
    "source_path",
}


@dataclass(frozen=True)
class ResolvedAssetBytes:
    pack_id: str
    asset_id: str
    delivery_ref: str
    filename: str
    mime_type: str
    data: bytes
    sha256: str
    byte_count: int
    width: int = 0
    height: int = 0


class AssetByteResolutionError(RuntimeError):
    """Non-spoiling asset byte failure for private caller notices."""

    def __init__(self, code: str, *, pack_id: str = "", asset_id: str = "") -> None:
        self.code = code
        self.pack_id = pack_id
        self.asset_id = asset_id
        detail = f" pack={pack_id or '-'} asset={asset_id or '-'}"
        super().__init__(f"Asset byte resolution failed ({code}).{detail}")


def resolve_asset_bytes(
    payload: SafeAssetRevealPayload,
    asset: ContentImageAsset,
    *,
    media_roots: Mapping[str, str | Path | Sequence[str | Path]],
    cache_roots: Mapping[str, str | Path | Sequence[str | Path]] | None = None,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    allowed_mime_types: Iterable[str] = SAFE_IMAGE_MIME_EXTENSIONS,
) -> ResolvedAssetBytes:
    """Resolve a player-safe asset reveal into verified local media bytes."""

    pack_id = payload.pack_id.strip()
    asset_id = payload.asset_id.strip()
    _validate_payload_matches_asset(payload, asset)
    delivery_ref = _validated_delivery_ref(payload, asset)
    mime_type = asset.mime_type.strip().lower()
    extension = _safe_extension(
        mime_type,
        allowed_mime_types={value.strip().lower() for value in allowed_mime_types},
        pack_id=pack_id,
        asset_id=asset_id,
    )
    expected_sha256 = _validated_sha256(
        asset.sha256, pack_id=pack_id, asset_id=asset_id
    )
    if payload.sha256 and _normalized_sha256(payload.sha256) != expected_sha256:
        raise AssetByteResolutionError(
            "payload_hash_mismatch",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if _metadata_has_unsafe_source(asset.metadata):
        raise AssetByteResolutionError(
            "unsafe_asset_metadata",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if max_bytes < 1:
        raise AssetByteResolutionError(
            "invalid_byte_limit",
            pack_id=pack_id,
            asset_id=asset_id,
        )

    roots = [
        *_roots_for_pack(media_roots, pack_id),
        *_roots_for_pack(cache_roots or {}, pack_id),
    ]
    if not roots:
        raise AssetByteResolutionError(
            "missing_pack_media_root",
            pack_id=pack_id,
            asset_id=asset_id,
        )

    media_path = _find_content_addressed_file(
        roots,
        sha256=expected_sha256,
        extension=extension,
        pack_id=pack_id,
        asset_id=asset_id,
    )
    data = _read_limited_bytes(
        media_path,
        max_bytes=max_bytes,
        pack_id=pack_id,
        asset_id=asset_id,
    )
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        raise AssetByteResolutionError(
            "hash_mismatch",
            pack_id=pack_id,
            asset_id=asset_id,
        )

    return ResolvedAssetBytes(
        pack_id=pack_id,
        asset_id=asset_id,
        delivery_ref=delivery_ref,
        filename=f"asset-{expected_sha256[:16]}{extension}",
        mime_type=mime_type,
        data=data,
        sha256=expected_sha256,
        byte_count=len(data),
        width=max(0, asset.width),
        height=max(0, asset.height),
    )


def _validate_payload_matches_asset(
    payload: SafeAssetRevealPayload,
    asset: ContentImageAsset,
) -> None:
    pack_id = payload.pack_id.strip()
    asset_id = payload.asset_id.strip()
    if not pack_id or not asset_id:
        raise AssetByteResolutionError(
            "missing_asset_identity",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if asset.pack_id.strip() != pack_id:
        raise AssetByteResolutionError(
            "pack_mismatch",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if asset.asset_id.strip() != asset_id:
        raise AssetByteResolutionError(
            "asset_mismatch",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if not asset.safe_for_players:
        raise AssetByteResolutionError(
            "asset_not_player_safe",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if asset.review_status not in APPROVED_ASSET_REVIEW_STATUSES:
        raise AssetByteResolutionError(
            "asset_not_approved",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if payload.kind.strip().lower() != asset.kind.strip().lower():
        raise AssetByteResolutionError(
            "payload_kind_mismatch",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if (
        payload.mime_type
        and payload.mime_type.strip().lower() != asset.mime_type.strip().lower()
    ):
        raise AssetByteResolutionError(
            "payload_mime_mismatch",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if payload.width > 0 and payload.width != asset.width:
        raise AssetByteResolutionError(
            "payload_dimensions_mismatch",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if payload.height > 0 and payload.height != asset.height:
        raise AssetByteResolutionError(
            "payload_dimensions_mismatch",
            pack_id=pack_id,
            asset_id=asset_id,
        )


def _validated_delivery_ref(
    payload: SafeAssetRevealPayload,
    asset: ContentImageAsset,
) -> str:
    pack_id = payload.pack_id.strip()
    asset_id = payload.asset_id.strip()
    payload_ref = _safe_delivery_ref(
        payload.delivery_ref,
        pack_id=pack_id,
        asset_id=asset_id,
    )
    asset_ref = _safe_delivery_ref(
        asset.delivery_ref,
        pack_id=pack_id,
        asset_id=asset_id,
    )
    if payload_ref is None or asset_ref is None:
        raise AssetByteResolutionError(
            "unsafe_delivery_ref",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    if payload_ref != asset_ref:
        raise AssetByteResolutionError(
            "delivery_ref_mismatch",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    return payload_ref


def _safe_delivery_ref(value: str, *, pack_id: str, asset_id: str) -> str | None:
    delivery_ref = value.strip()
    match = _ASSET_DELIVERY_REF_RE.fullmatch(delivery_ref)
    if match is None:
        return None
    if match.group("pack_id") != pack_id or match.group("asset_id") != asset_id:
        raise AssetByteResolutionError(
            "delivery_ref_mismatch",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    return delivery_ref


def _safe_extension(
    mime_type: str,
    *,
    allowed_mime_types: set[str],
    pack_id: str,
    asset_id: str,
) -> str:
    if mime_type not in allowed_mime_types:
        raise AssetByteResolutionError(
            "unsafe_mime_type",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    extension = SAFE_IMAGE_MIME_EXTENSIONS.get(mime_type)
    if not extension:
        raise AssetByteResolutionError(
            "unsupported_mime_type",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    return extension


def _validated_sha256(value: str, *, pack_id: str, asset_id: str) -> str:
    sha256 = _normalized_sha256(value)
    if not _SHA256_RE.fullmatch(sha256):
        raise AssetByteResolutionError(
            "invalid_sha256",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    return sha256


def _normalized_sha256(value: str) -> str:
    return value.strip().lower()


def _roots_for_pack(
    roots_by_pack: Mapping[str, str | Path | Sequence[str | Path]],
    pack_id: str,
) -> list[Path]:
    raw_roots = roots_by_pack.get(pack_id)
    if raw_roots is None:
        return []
    if isinstance(raw_roots, str | Path):
        values = [raw_roots]
    else:
        values = list(raw_roots)
    return [Path(value).resolve() for value in values]


def _find_content_addressed_file(
    roots: Sequence[Path],
    *,
    sha256: str,
    extension: str,
    pack_id: str,
    asset_id: str,
) -> Path:
    relative_candidates = (
        Path(f"{sha256}{extension}"),
        Path(sha256),
        Path(sha256[:2]) / f"{sha256}{extension}",
        Path(sha256[:2]) / sha256,
    )
    for root in roots:
        for relative in relative_candidates:
            candidate = root / relative
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise AssetByteResolutionError(
                    "media_path_outside_root",
                    pack_id=pack_id,
                    asset_id=asset_id,
                ) from exc
            return resolved
    raise AssetByteResolutionError(
        "missing_media",
        pack_id=pack_id,
        asset_id=asset_id,
    )


def _read_limited_bytes(
    path: Path,
    *,
    max_bytes: int,
    pack_id: str,
    asset_id: str,
) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise AssetByteResolutionError(
            "asset_too_large",
            pack_id=pack_id,
            asset_id=asset_id,
        )
    return data


def _metadata_has_unsafe_source(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).strip().lower()
            if key_text in _FORBIDDEN_METADATA_KEYS:
                return True
            if _metadata_has_unsafe_source(item):
                return True
        return False
    if isinstance(value, list):
        return any(_metadata_has_unsafe_source(item) for item in value)
    if isinstance(value, str):
        text = value.strip()
        if _ABSOLUTE_PATH_RE.search(text):
            return True
        if text.startswith(("../", "./")):
            return True
        if _UNSAFE_URI_RE.match(text) and not text.startswith("asset://"):
            return True
    return False
