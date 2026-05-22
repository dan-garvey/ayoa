from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import MutableMapping, MutableSequence
from typing import Any, Iterable, Mapping, Sequence

from app.schemas.conversation import ConversationMessage

try:  # The content schemas may land after this scaffold.
    from app.schemas.content import IntroducedContentRef as _IntroducedContentRef
except Exception:  # pragma: no cover - defensive import boundary.
    _IntroducedContentRef = None


_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")
_INTRODUCED_ATTRS = (
    "introduced_refs",
    "router_content_memory",
    "introduced_content_refs",
    "introduced_content",
)
_PENDING_ATTRS = (
    "pending_signals",
    "pending_content_signals",
    "content_signals",
)


@dataclass(frozen=True)
class ContentCard:
    pack_id: str
    ref: str
    content_hash: str = ""
    kind: str = ""
    visibility: str = ""
    summary: str = ""
    title: str = ""
    body: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


def format_content_known_record(
    ref_or_record: Any,
    *,
    pack_id: str | None = None,
    content_hash: str | None = None,
    kind: str | None = None,
    visibility: str | None = None,
    summary: str | None = None,
    scope: str = "router",
) -> str:
    fields = _base_fields(
        ref_or_record,
        pack_id=pack_id,
        content_hash=content_hash,
        kind=kind,
        visibility=visibility,
        summary=summary,
    )
    fields["scope"] = scope
    return _format_record(
        "content_known",
        fields,
        ("ref", "scope", "visibility", "hash", "kind", "pack", "summary"),
    )


def format_front_signal_record(
    signal: Any,
    *,
    pack_id: str | None = None,
    content_hash: str | None = None,
) -> str:
    fields = _base_fields(signal, pack_id=pack_id, content_hash=content_hash)
    for source, target in (
        ("actor", "actor"),
        ("knows", "knows"),
        ("pressure", "pressure"),
        ("summary", "summary"),
        ("villains", "villains"),
        ("goals", "goals"),
        ("constraints", "constraints"),
        ("knowledge_channels", "knowledge_channels"),
        ("resources", "resources"),
        ("minions", "minions"),
        ("minion_refs", "minions"),
        ("escalation_thresholds", "escalation_thresholds"),
        ("cooldowns", "cooldowns"),
        ("restraints", "restraints"),
        ("actions", "actions"),
    ):
        value = _value(signal, source)
        if value not in (None, "", [], ()):
            fields[target] = value
    return _format_record(
        "front_signal",
        fields,
        (
            "ref",
            "actor",
            "villains",
            "knows",
            "pressure",
            "visibility",
            "hash",
            "pack",
            "summary",
            "goals",
            "constraints",
            "knowledge_channels",
            "resources",
            "minions",
            "escalation_thresholds",
            "cooldowns",
            "restraints",
            "actions",
        ),
    )


def format_location_card_record(
    card: Any,
    *,
    pack_id: str | None = None,
    content_hash: str | None = None,
) -> str:
    fields = _base_fields(card, pack_id=pack_id, content_hash=content_hash)
    for key in ("exits", "hazards", "clues", "summary"):
        value = _value(card, key)
        if value not in (None, "", [], ()):
            fields[key] = value
    return _format_record(
        "location_card",
        fields,
        (
            "ref",
            "exits",
            "hazards",
            "clues",
            "visibility",
            "hash",
            "pack",
            "summary",
        ),
    )


def content_ref_needs_introduction(
    state: Any,
    ref_or_record: Any,
    *,
    pack_id: str | None = None,
    content_hash: str | None = None,
) -> bool:
    ref = _normalized_ref(ref_or_record)
    if not ref:
        return False
    pack = _resolved_pack_id(state, ref_or_record, pack_id)
    new_hash = _normalized_text(
        content_hash if content_hash is not None else _value(ref_or_record, "content_hash")
    )

    for introduced in _introduced_refs(state):
        if _normalized_ref(introduced) != ref:
            continue
        introduced_pack = _resolved_pack_id(state, introduced, None)
        if pack and introduced_pack and introduced_pack != pack:
            continue
        old_hash = _normalized_text(_value(introduced, "content_hash"))
        return bool(new_hash and new_hash != old_hash)
    return True


def mark_ref_introduced(
    state: Any,
    ref_or_record: Any,
    *,
    pack_id: str | None = None,
    content_hash: str | None = None,
    kind: str | None = None,
    visibility: str | None = None,
    summary: str | None = None,
) -> bool:
    ref = _normalized_ref(ref_or_record)
    if not ref:
        return False
    pack = _resolved_pack_id(state, ref_or_record, pack_id)
    fields = _base_fields(
        ref_or_record,
        pack_id=pack,
        content_hash=content_hash,
        kind=kind,
        visibility=visibility,
        summary=summary,
    )
    fields["ref"] = ref

    introduced_container = _introduced_container(state)
    for introduced_key, introduced in _introduced_entries(introduced_container):
        if _normalized_ref(introduced) != ref:
            continue
        introduced_pack = _resolved_pack_id(state, introduced, None)
        if pack and introduced_pack and introduced_pack != pack:
            continue
        if not content_ref_needs_introduction(
            state,
            ref_or_record,
            pack_id=pack,
            content_hash=fields.get("content_hash") or fields.get("hash"),
        ):
            return False
        if isinstance(introduced_container, MutableMapping):
            new_ref = _make_introduced_ref(fields)
            new_key = _introduced_key(new_ref)
            if introduced_key != new_key:
                introduced_container.pop(introduced_key, None)
            introduced_container[new_key] = new_ref
        else:
            _update_known_fields(introduced, fields)
        return True

    introduced_ref = _make_introduced_ref(fields)
    if isinstance(introduced_container, MutableMapping):
        introduced_container[_introduced_key(introduced_ref)] = introduced_ref
    else:
        introduced_container.append(introduced_ref)
    return True


def drain_pending_content_signals(state: Any) -> list[str]:
    pending = _pending_signal_items(state)
    if not pending:
        return []

    records: list[str] = []
    for signal in pending:
        if _content_signal_status(signal) != "pending":
            continue
        if not content_ref_needs_introduction(state, signal):
            continue
        records.append(
            format_compact_record(
                signal,
                pack_id=_resolved_pack_id(state, signal, None),
            )
        )
        mark_ref_introduced(state, signal)

    # Draining consumes this one-shot queue: terminal resolved/dismissed
    # entries are pruned without formatting or introduction.
    _clear_pending_signals(state)
    return records


def append_pending_router_content_records(ckpt: Any) -> list[str]:
    """Drain pending content deltas into compact router history records.

    These records are assistant-side memory for the router, not current-turn
    user input and not narrator/agent context. The dispatcher snapshots and
    restores checkpoint state around this call when a router call fails.
    """
    session = getattr(ckpt, "session", None)
    if session is None:
        return []
    content_state = getattr(session, "content_state", None) or {}
    if not isinstance(content_state, Mapping):
        return []
    conversation = getattr(ckpt, "session_conversation", None)
    if conversation is None:
        return []

    records: list[str] = []
    for pack_id, pack_state in content_state.items():
        if hasattr(pack_state, "pack_id") and not getattr(pack_state, "pack_id", ""):
            try:
                pack_state.pack_id = str(pack_id)
            except Exception:
                pass
        for record in drain_pending_content_signals(pack_state):
            if record:
                records.append(record)

    for record in records:
        conversation.append(ConversationMessage(role="assistant", content=record))
    return records


def format_compact_record(
    record: Any,
    *,
    pack_id: str | None = None,
    content_hash: str | None = None,
) -> str:
    record_kind = _normalized_text(_value(record, "record_kind") or _value(record, "kind"))
    if record_kind == "front_signal" or any(
        _value(record, key) not in (None, "", [], ())
        for key in ("actor", "knows", "pressure")
    ):
        return format_front_signal_record(
            record,
            pack_id=pack_id,
            content_hash=content_hash,
        )
    if record_kind == "location_card" or any(
        _value(record, key) not in (None, "", [], ())
        for key in ("exits", "hazards", "clues")
    ):
        return format_location_card_record(
            record,
            pack_id=pack_id,
            content_hash=content_hash,
        )
    return format_content_known_record(
        record,
        pack_id=pack_id,
        content_hash=content_hash,
    )


def load_content_cards(
    db_path: str | Path,
    *,
    refs: Iterable[str] | None = None,
    pack_id: str | None = None,
    limit: int | None = None,
) -> list[ContentCard]:
    path = Path(db_path)
    if not path.exists():
        return []

    wanted_refs = [_normalized_text(ref) for ref in refs or () if _normalized_text(ref)]
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        if not _table_exists(conn, "content_cards"):
            return []
        columns = _table_columns(conn, "content_cards")
        select_columns = [
            col
            for col in (
                "pack_id",
                "ref",
                "content_hash",
                "kind",
                "visibility",
                "summary",
                "title",
                "body",
                "metadata",
                "metadata_json",
            )
            if col in columns
        ]
        if "ref" not in select_columns:
            return []

        where: list[str] = []
        params: list[Any] = []
        if pack_id and "pack_id" in columns:
            where.append("pack_id = ?")
            params.append(pack_id)
        if wanted_refs:
            where.append(f"ref IN ({','.join('?' for _ in wanted_refs)})")
            params.extend(wanted_refs)

        sql = f"SELECT {', '.join(select_columns)} FROM content_cards"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ref"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))

        rows = conn.execute(sql, params).fetchall()
    return [_card_from_row(row) for row in rows]


def _base_fields(
    record: Any,
    *,
    pack_id: str | None = None,
    content_hash: str | None = None,
    kind: str | None = None,
    visibility: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    ref = _normalized_ref(record)
    fields: dict[str, Any] = {"ref": ref, "ref_id": ref}
    pack = _normalized_text(pack_id if pack_id is not None else _value(record, "pack_id"))
    if pack:
        fields["pack"] = pack
        fields["pack_id"] = pack
    resolved_hash = _normalized_text(
        content_hash if content_hash is not None else _value(record, "content_hash")
    )
    if resolved_hash:
        fields["hash"] = resolved_hash
        fields["content_hash"] = resolved_hash
    resolved_kind = _normalized_text(kind if kind is not None else _value(record, "kind"))
    if resolved_kind:
        fields["kind"] = resolved_kind
    resolved_visibility = _normalized_text(
        visibility if visibility is not None else _value(record, "visibility")
    )
    if resolved_visibility:
        fields["visibility"] = resolved_visibility
    resolved_summary = _normalized_text(
        summary if summary is not None else _value(record, "summary")
    )
    if resolved_summary:
        fields["summary"] = resolved_summary
    return fields


def _format_record(
    prefix: str,
    fields: Mapping[str, Any],
    ordered_keys: Sequence[str],
) -> str:
    parts = [prefix]
    for key in ordered_keys:
        value = fields.get(key)
        if value in (None, "", [], ()):
            continue
        parts.append(f"{key}={_format_value(value)}")
    return " ".join(parts)


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set, frozenset)):
        if any(isinstance(item, Mapping) for item in value):
            json_ready = [
                _json_ready(item)
                for item in value
                if _json_ready(item) not in (None, "", [], {})
            ]
            return json.dumps(json_ready, ensure_ascii=True, separators=(",", ":"))
        values = [_normalized_text(item) for item in value if _normalized_text(item)]
        if not values:
            return "[]"
        if all(_TOKEN_RE.match(item) for item in values):
            return ",".join(values)
        return json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    if isinstance(value, Mapping):
        return json.dumps(_json_ready(value), ensure_ascii=True, separators=(",", ":"))
    text = _normalized_text(value)
    if _TOKEN_RE.match(text):
        return text
    return json.dumps(text, ensure_ascii=True, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            _normalized_text(key): _json_ready(item)
            for key, item in value.items()
            if _normalized_text(key)
            and _json_ready(item) not in (None, "", [], {})
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            ready
            for item in value
            if (ready := _json_ready(item)) not in (None, "", [], {})
        ]
    return _normalized_text(value)


def _value(record: Any, key: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        if key in record:
            return record.get(key, default)
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping) and key in metadata:
            return metadata.get(key, default)
        return default
    if hasattr(record, key):
        return getattr(record, key)
    metadata = getattr(record, "metadata", None)
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata.get(key, default)
    if hasattr(record, "model_dump"):
        dumped = record.model_dump()
        if isinstance(dumped, Mapping):
            if key in dumped:
                return dumped.get(key, default)
            metadata = dumped.get("metadata")
            if isinstance(metadata, Mapping) and key in metadata:
                return metadata.get(key, default)
    return default


def _set_value(record: Any, key: str, value: Any) -> None:
    if isinstance(record, dict):
        record[key] = value
        return
    if hasattr(record, key):
        setattr(record, key, value)


def _normalized_ref(ref_or_record: Any) -> str:
    if isinstance(ref_or_record, str):
        return _normalized_text(ref_or_record)
    return _normalized_text(
        _value(ref_or_record, "ref")
        or _value(ref_or_record, "ref_id")
        or _value(ref_or_record, "content_ref")
    )


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _resolved_pack_id(
    state: Any,
    ref_or_record: Any,
    explicit_pack_id: str | None,
) -> str:
    return _normalized_text(
        explicit_pack_id
        if explicit_pack_id is not None
        else _value(ref_or_record, "pack_id") or _value(state, "pack_id")
    )


def _introduced_refs(state: Any) -> list[Any]:
    for attr in _INTRODUCED_ATTRS:
        values = _value(state, attr)
        if values is not None:
            if isinstance(values, Mapping):
                return list(values.values())
            return values
    return []


def _introduced_entries(container: Any) -> list[tuple[Any, Any]]:
    if isinstance(container, Mapping):
        return list(container.items())
    return list(enumerate(container or []))


def _introduced_container(state: Any) -> Any:
    for attr in _INTRODUCED_ATTRS:
        values = _value(state, attr)
        if values is not None:
            return values
    if isinstance(state, dict):
        state["introduced_refs"] = []
        return state["introduced_refs"]
    setattr(state, "introduced_refs", [])
    return getattr(state, "introduced_refs")


def _introduced_key(record: Any) -> str:
    return "::".join(
        (
            _normalized_text(_value(record, "pack_id")),
            _normalized_ref(record),
            _normalized_text(_value(record, "content_hash")),
        )
    )


def _pending_signal_items(state: Any) -> list[Any]:
    for attr in _PENDING_ATTRS:
        values = _value(state, attr)
        if values is not None:
            if isinstance(values, Mapping):
                return list(values.values())
            return values
    return []


def _content_signal_status(signal: Any) -> str:
    return _normalized_text(_value(signal, "status")).lower()


def _clear_pending_signals(state: Any) -> None:
    for attr in _PENDING_ATTRS:
        values = _value(state, attr)
        if values is None:
            continue
        if isinstance(values, (MutableMapping, MutableSequence)):
            values.clear()
        else:
            _set_value(state, attr, [])
        return


def _make_introduced_ref(fields: Mapping[str, Any]) -> Any:
    ref = fields.get("ref") or fields.get("ref_id") or ""
    summary = fields.get("summary") or ""
    payload = {
        "pack_id": fields.get("pack_id") or fields.get("pack") or "",
        "ref": ref,
        "ref_id": ref,
        "content_hash": fields.get("content_hash") or fields.get("hash") or "",
        "kind": fields.get("kind") or "",
        "visibility": fields.get("visibility") or "",
        "summary": summary,
        "label": summary,
    }
    if _IntroducedContentRef is not None:
        try:
            schema_fields = getattr(_IntroducedContentRef, "model_fields", {})
            schema_payload = {
                key: value
                for key, value in payload.items()
                if not schema_fields or key in schema_fields
            }
            return _IntroducedContentRef(**schema_payload)
        except Exception:
            pass
    return payload


def _update_known_fields(record: Any, fields: Mapping[str, Any]) -> None:
    ref = fields.get("ref") or fields.get("ref_id") or ""
    updates = {
        "pack_id": fields.get("pack_id") or fields.get("pack") or "",
        "ref": ref,
        "ref_id": ref,
        "content_hash": fields.get("content_hash") or fields.get("hash") or "",
        "kind": fields.get("kind") or "",
        "visibility": fields.get("visibility") or "",
        "summary": fields.get("summary") or "",
    }
    for key, value in updates.items():
        if value:
            _set_value(record, key, value)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _card_from_row(row: sqlite3.Row) -> ContentCard:
    metadata_raw = row["metadata"] if "metadata" in row.keys() else None
    if metadata_raw is None and "metadata_json" in row.keys():
        metadata_raw = row["metadata_json"]
    metadata: Mapping[str, Any] = {}
    if metadata_raw:
        try:
            parsed = json.loads(metadata_raw)
            if isinstance(parsed, Mapping):
                metadata = parsed
        except (TypeError, json.JSONDecodeError):
            metadata = {}
    return ContentCard(
        pack_id=str(row["pack_id"] or "") if "pack_id" in row.keys() else "",
        ref=str(row["ref"] or ""),
        content_hash=str(row["content_hash"] or "")
        if "content_hash" in row.keys()
        else "",
        kind=str(row["kind"] or "") if "kind" in row.keys() else "",
        visibility=str(row["visibility"] or "") if "visibility" in row.keys() else "",
        summary=str(row["summary"] or "") if "summary" in row.keys() else "",
        title=str(row["title"] or "") if "title" in row.keys() else "",
        body=str(row["body"] or "") if "body" in row.keys() else "",
        metadata=metadata,
    )
