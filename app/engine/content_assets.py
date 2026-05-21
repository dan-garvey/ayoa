from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.schemas.content_pack import (
    AssetReveal,
    ContentImageAsset,
    SafeAssetRevealPayload,
)


APPROVED_ASSET_REVIEW_STATUSES = {"reviewed", "approved"}
_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^/\s]+/)+[^/\s]+")
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


def write_asset_catalog(
    db_path: str | Path,
    assets: Iterable[ContentImageAsset | Mapping[str, Any]],
    *,
    protected_terms: Sequence[str] = (),
) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [_coerce_asset(asset) for asset in assets]
    with sqlite3.connect(path) as conn:
        _create_asset_schema(conn)
        for asset in records:
            metadata = _sanitize_metadata(asset.metadata, protected_terms)
            conn.execute(
                """
                INSERT OR REPLACE INTO content_assets (
                    pack_id, asset_id, kind, title, mime_type, width, height,
                    sha256, source_ref, review_status, spoiler_class,
                    player_safe_alt_text, player_safe_caption, delivery_ref,
                    safe_for_players, safe_for_llm, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset.pack_id,
                    asset.asset_id,
                    asset.kind,
                    _redact_unsafe_text(asset.title),
                    asset.mime_type,
                    asset.width,
                    asset.height,
                    asset.sha256,
                    asset.source_ref,
                    asset.review_status,
                    asset.spoiler_class,
                    _redact_unsafe_text(asset.player_safe_alt_text),
                    _redact_unsafe_text(asset.player_safe_caption),
                    _safe_delivery_ref(
                        asset.delivery_ref,
                        pack_id=asset.pack_id,
                        asset_id=asset.asset_id,
                    ),
                    int(asset.safe_for_players),
                    int(asset.safe_for_llm),
                    json.dumps(metadata, sort_keys=True),
                ),
            )
        conn.commit()


def load_asset_catalog(
    db_path: str | Path,
    *,
    pack_id: str | None = None,
) -> dict[str, ContentImageAsset]:
    path = Path(db_path)
    if not path.exists():
        return {}
    where: list[str] = []
    params: list[Any] = []
    if pack_id:
        where.append("pack_id = ?")
        params.append(pack_id)
    sql = "SELECT * FROM content_assets"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY pack_id, asset_id"
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        if not _table_exists(conn, "content_assets"):
            return {}
        rows = conn.execute(sql, params).fetchall()
    return {_asset_key(row["pack_id"], row["asset_id"]): _asset_from_row(row) for row in rows}


def safe_asset_reveals_for_viewer(
    assets: Mapping[str, ContentImageAsset | Mapping[str, Any]]
    | Iterable[ContentImageAsset | Mapping[str, Any]],
    reveals: Iterable[AssetReveal | Mapping[str, Any]],
    *,
    character_id: str,
    user_id: str = "",
    event_observer_ids: Iterable[str] = (),
) -> list[SafeAssetRevealPayload]:
    asset_lookup = _asset_lookup(assets)
    observer_set = {value.strip() for value in event_observer_ids if value.strip()}
    viewer_character = character_id.strip()
    viewer_user = user_id.strip()
    payloads: list[SafeAssetRevealPayload] = []

    for reveal_value in reveals:
        reveal = _coerce_reveal(reveal_value)
        if not _viewer_can_see_reveal(
            reveal,
            character_id=viewer_character,
            user_id=viewer_user,
            event_observer_ids=observer_set,
        ):
            continue
        asset = asset_lookup.get(_asset_key(reveal.pack_id, reveal.asset_id))
        if asset is None and not reveal.pack_id:
            asset = asset_lookup.get(reveal.asset_id)
        if asset is None or not _asset_is_player_safe(asset):
            continue
        delivery_ref = _safe_delivery_ref(
            asset.delivery_ref,
            pack_id=asset.pack_id,
            asset_id=asset.asset_id,
        )
        caption = _redact_unsafe_text(reveal.caption or asset.player_safe_caption)
        payloads.append(
            SafeAssetRevealPayload(
                pack_id=asset.pack_id,
                asset_id=asset.asset_id,
                kind=asset.kind,
                title=_redact_unsafe_text(asset.title),
                mime_type=asset.mime_type,
                width=asset.width,
                height=asset.height,
                sha256=asset.sha256,
                delivery_ref=delivery_ref,
                presentation=reveal.presentation,
                caption=caption,
                alt_text=_redact_unsafe_text(asset.player_safe_alt_text),
            )
        )

    return payloads


def _create_asset_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS content_assets (
            pack_id TEXT NOT NULL,
            asset_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            review_status TEXT NOT NULL,
            spoiler_class TEXT NOT NULL,
            player_safe_alt_text TEXT NOT NULL,
            player_safe_caption TEXT NOT NULL,
            delivery_ref TEXT NOT NULL,
            safe_for_players INTEGER NOT NULL,
            safe_for_llm INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (pack_id, asset_id)
        )
        """
    )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _coerce_asset(asset: ContentImageAsset | Mapping[str, Any]) -> ContentImageAsset:
    if isinstance(asset, ContentImageAsset):
        return asset
    return ContentImageAsset(**dict(asset))


def _coerce_reveal(reveal: AssetReveal | Mapping[str, Any]) -> AssetReveal:
    if isinstance(reveal, AssetReveal):
        return reveal
    return AssetReveal(**dict(reveal))


def _asset_lookup(
    assets: Mapping[str, ContentImageAsset | Mapping[str, Any]]
    | Iterable[ContentImageAsset | Mapping[str, Any]],
) -> dict[str, ContentImageAsset]:
    if isinstance(assets, Mapping):
        values = assets.values()
    else:
        values = assets
    lookup: dict[str, ContentImageAsset] = {}
    for value in values:
        asset = _coerce_asset(value)
        lookup[_asset_key(asset.pack_id, asset.asset_id)] = asset
        lookup.setdefault(asset.asset_id, asset)
    return lookup


def _asset_key(pack_id: str, asset_id: str) -> str:
    pack = pack_id.strip()
    asset = asset_id.strip()
    return f"{pack}::{asset}" if pack else asset


def _viewer_can_see_reveal(
    reveal: AssetReveal,
    *,
    character_id: str,
    user_id: str,
    event_observer_ids: set[str],
) -> bool:
    viewer_is_observer = bool(character_id and character_id in event_observer_ids)
    if reveal.audience == "all_observers":
        return viewer_is_observer
    if not viewer_is_observer:
        return False
    if not character_id or character_id not in reveal.visible_to_character_ids:
        return False
    if reveal.visible_to_user_ids:
        return bool(user_id and user_id in reveal.visible_to_user_ids)
    return True


def _asset_is_player_safe(asset: ContentImageAsset) -> bool:
    return (
        asset.safe_for_players
        and asset.review_status in APPROVED_ASSET_REVIEW_STATUSES
        and bool(
            _safe_delivery_ref(
                asset.delivery_ref,
                pack_id=asset.pack_id,
                asset_id=asset.asset_id,
            )
        )
    )


def _asset_from_row(row: sqlite3.Row) -> ContentImageAsset:
    return ContentImageAsset(
        pack_id=row["pack_id"],
        asset_id=row["asset_id"],
        kind=row["kind"],
        title=row["title"],
        mime_type=row["mime_type"],
        width=row["width"],
        height=row["height"],
        sha256=row["sha256"],
        source_ref=row["source_ref"],
        review_status=row["review_status"],
        spoiler_class=row["spoiler_class"],
        player_safe_alt_text=row["player_safe_alt_text"],
        player_safe_caption=row["player_safe_caption"],
        delivery_ref=row["delivery_ref"],
        safe_for_players=bool(row["safe_for_players"]),
        safe_for_llm=bool(row["safe_for_llm"]),
        metadata=_json_mapping(row["metadata_json"]),
    )


def _sanitize_metadata(value: Any, protected_terms: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.strip().lower() in _FORBIDDEN_METADATA_KEYS:
                continue
            sanitized_value = _sanitize_metadata(item, protected_terms)
            if sanitized_value is not None:
                sanitized[key_text] = sanitized_value
        return sanitized
    if isinstance(value, list):
        return [
            sanitized
            for item in value
            if (sanitized := _sanitize_metadata(item, protected_terms)) is not None
        ]
    if isinstance(value, str):
        if _ABSOLUTE_PATH_RE.search(value):
            return None
        if any(term in value for term in protected_terms):
            return None
    return value


def _safe_delivery_ref(
    value: str,
    *,
    pack_id: str = "",
    asset_id: str = "",
) -> str:
    delivery_ref = value.strip()
    match = _ASSET_DELIVERY_REF_RE.fullmatch(delivery_ref)
    if match is None:
        return ""
    if pack_id and match.group("pack_id") != pack_id.strip():
        return ""
    if asset_id and match.group("asset_id") != asset_id.strip():
        return ""
    return delivery_ref


def _redact_unsafe_text(value: str) -> str:
    return "" if _ABSOLUTE_PATH_RE.search(value) else value.strip()


def _json_mapping(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}
