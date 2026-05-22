from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.engine.content_resolver import (
    append_pending_router_content_records,
    content_ref_needs_introduction,
    format_compact_record,
    load_content_cards,
    mark_ref_introduced,
)
from app.engine.content_pack_compiler import (
    CompiledContentPackMismatchError,
    CompiledContentPackReader,
    SCHEMA_VERSION as CONTENT_PACK_SCHEMA_VERSION,
)
from app.schemas.conversation import ConversationMessage


_REF_RE = re.compile(r"\bref=([A-Za-z0-9_.:/@+-]+)")
_APPROVED_REVIEW_STATUSES = {"approved", "reviewed"}
_MIN_ALIAS_CONFIDENCE = 0.70


@dataclass(frozen=True)
class ContentLookupRequest:
    pack_id: str
    ref: str
    reason: str = ""
    alias: str = ""
    urgency: str = "required"
    spoiler_boundary: str = "router_hidden"
    already_known: bool = False


class MissingContentError(RuntimeError):
    """Raised when a bounded router preflight needs content the pack lacks."""

    def __init__(self, requests: Iterable[ContentLookupRequest]):
        self.requests = list(requests)
        details = "; ".join(
            (
                f"pack={request.pack_id or '-'} ref={request.ref or '-'}"
                f" alias={request.alias or '-'} reason={request.reason or '-'}"
            )
            for request in self.requests
        )
        super().__init__(
            "Router content lookup preflight could not resolve required content"
            + (f": {details}" if details else ".")
        )


def append_router_content_lookup_records(
    ckpt: Any,
    *,
    actor_id: str,
    current_input: str,
    max_lookup_passes: int = 1,
) -> list[str]:
    """Append one-shot router content records before the normal router call.

    This is intentionally the smallest deterministic preflight slice: queued
    pending signals are drained first, then pack-local aliases/catalog entries
    are matched against current actor input and location. A future LLM lookup
    prompt can plug into `plan_llm_router_content_lookup_requests` without
    changing the ordinary EventRouterOutput schema.
    """
    records = append_pending_router_content_records(ckpt)

    if max_lookup_passes < 1:
        return records

    session = getattr(ckpt, "session", None)
    content_state = getattr(session, "content_state", None) if session else None
    conversation = getattr(ckpt, "session_conversation", None)
    if not isinstance(content_state, Mapping) or conversation is None:
        return records

    known_refs = _known_router_content_refs(ckpt)
    requests = plan_deterministic_router_content_lookup_requests(
        ckpt,
        actor_id=actor_id,
        current_input=current_input,
        known_refs=known_refs,
    )
    requests.extend(
        plan_llm_router_content_lookup_requests(
            ckpt,
            actor_id=actor_id,
            current_input=current_input,
            known_refs=known_refs,
            deterministic_requests=requests,
        )
    )
    if not requests:
        return records

    lookup_records = _fetch_lookup_records(ckpt, requests)
    for record in lookup_records:
        conversation.append(ConversationMessage(role="assistant", content=record))
    records.extend(lookup_records)
    return records


def plan_deterministic_router_content_lookup_requests(
    ckpt: Any,
    *,
    actor_id: str,
    current_input: str,
    known_refs: set[tuple[str, str]] | None = None,
) -> list[ContentLookupRequest]:
    session = getattr(ckpt, "session", None)
    content_state = getattr(session, "content_state", None) if session else None
    if not isinstance(content_state, Mapping):
        return []

    search_text = _lookup_search_text(ckpt, actor_id, current_input)
    if not search_text:
        return []

    known = known_refs or set()
    requests: list[ContentLookupRequest] = []
    seen: set[tuple[str, str]] = set()
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        for alias, ref, reason in _iter_alias_catalog_entries(pack_state):
            if not alias or not ref:
                continue
            key = (pack_id, ref)
            if key in seen:
                continue
            if not _contains_alias(search_text, alias):
                continue
            seen.add(key)
            requests.append(
                ContentLookupRequest(
                    pack_id=pack_id,
                    ref=ref,
                    alias=alias,
                    reason=reason or "deterministic_alias_match",
                    already_known=key in known,
                )
            )
    return requests


def plan_llm_router_content_lookup_requests(
    ckpt: Any,
    *,
    actor_id: str,
    current_input: str,
    known_refs: set[tuple[str, str]],
    deterministic_requests: list[ContentLookupRequest],
) -> list[ContentLookupRequest]:
    """Extension point for a future `event_router_content_lookup` prompt.

    The first runtime slice deliberately avoids a second model call. If an LLM
    preflight is added later, keep it bounded to one request/refetch retry and
    return exact ContentLookupRequest objects from here.
    """
    return []


def _fetch_lookup_records(
    ckpt: Any,
    requests: list[ContentLookupRequest],
) -> list[str]:
    session = getattr(ckpt, "session", None)
    content_state = getattr(session, "content_state", {}) if session else {}
    if isinstance(content_state, Mapping):
        pack_states = {
            _pack_id(pack_key, pack_state): pack_state
            for pack_key, pack_state in content_state.items()
        }
    else:
        pack_states = {}
    by_pack: dict[str, list[ContentLookupRequest]] = {}
    for request in requests:
        by_pack.setdefault(request.pack_id, []).append(request)

    resolved: list[tuple[Any, Any, str]] = []
    missing: list[ContentLookupRequest] = []
    for pack_id, pack_requests in by_pack.items():
        pack_state = pack_states.get(pack_id)
        db_path = _pack_db_path(pack_state)
        if db_path is None:
            missing.extend(
                request for request in pack_requests if not request.already_known
            )
            continue
        _assert_pack_runtime_identity(
            db_path,
            pack_id=pack_id,
            pack_state=pack_state,
        )
        refs = [request.ref for request in pack_requests]
        cards = load_content_cards(
            db_path,
            refs=refs,
            pack_id=pack_id,
            runtime_only=True,
        )
        cards_by_ref = {card.ref: card for card in cards}
        for request in pack_requests:
            card = cards_by_ref.get(request.ref)
            if card is None:
                if not request.already_known:
                    missing.append(request)
                continue
            if not content_ref_needs_introduction(
                pack_state,
                card,
                pack_id=pack_id,
                content_hash=card.content_hash,
            ):
                continue
            resolved.append((pack_state, card, pack_id))

    if missing:
        raise MissingContentError(missing)

    records: list[str] = []
    for pack_state, card, pack_id in resolved:
        records.append(format_compact_record(card, pack_id=pack_id))
        mark_ref_introduced(
            pack_state,
            card,
            pack_id=pack_id,
            content_hash=card.content_hash,
            kind=card.kind,
            visibility=card.visibility,
            summary=card.summary,
        )
    return records


def _assert_pack_runtime_identity(
    db_path: Path,
    *,
    pack_id: str,
    pack_state: Any,
) -> None:
    metadata = _metadata(pack_state)
    source_fingerprint = str(metadata.get("source_fingerprint") or "").strip()
    if not source_fingerprint:
        raise CompiledContentPackMismatchError(
            f"Compiled pack source_fingerprint is missing for pack={pack_id or '-'}"
        )
    CompiledContentPackReader(db_path).assert_pack_identity(
        pack_id=pack_id,
        pack_version=str(metadata.get("pack_version") or "").strip(),
        source_fingerprint=source_fingerprint,
        schema_version=str(
            metadata.get("schema_version") or CONTENT_PACK_SCHEMA_VERSION
        ).strip(),
    )


def _lookup_search_text(ckpt: Any, actor_id: str, current_input: str) -> str:
    parts = [current_input or ""]
    actor = next(
        (
            character for character in getattr(ckpt, "characters", [])
            if getattr(character, "character_id", "") == actor_id
        ),
        None,
    )
    if actor is not None:
        parts.append(getattr(actor, "location", "") or "")
    return " ".join(part for part in parts if part).lower()


def _iter_alias_catalog_entries(pack_state: Any) -> Iterable[tuple[str, str, str]]:
    metadata = _metadata(pack_state)
    yield from _iter_metadata_aliases(metadata.get("aliases"), "alias")
    yield from _iter_metadata_aliases(metadata.get("alias_index"), "alias_index")
    yield from _iter_metadata_catalog(metadata.get("catalog"))

    db_path = _pack_db_path(pack_state)
    if db_path is not None:
        yield from _iter_sqlite_aliases(db_path, _pack_id("", pack_state))


def _iter_metadata_aliases(raw: Any, reason: str) -> Iterable[tuple[str, str, str]]:
    if isinstance(raw, Mapping):
        for alias, value in raw.items():
            if isinstance(value, str):
                yield str(alias), value, reason
            elif isinstance(value, Mapping):
                yield str(alias), str(value.get("ref") or value.get("ref_id") or ""), (
                    str(value.get("reason") or reason)
                )
        return
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            ref = str(item.get("ref") or item.get("ref_id") or "")
            for alias in item.get("aliases") or item.get("names") or ():
                yield str(alias), ref, str(item.get("reason") or reason)


def _iter_metadata_catalog(raw: Any) -> Iterable[tuple[str, str, str]]:
    if not isinstance(raw, list):
        return
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        ref = str(item.get("ref") or item.get("ref_id") or "")
        aliases = list(item.get("aliases") or item.get("names") or ())
        for key in ("title", "label", "name", "ref", "ref_id"):
            value = item.get(key)
            if value:
                aliases.append(str(value))
        for alias in dict.fromkeys(aliases):
            yield str(alias), ref, "catalog"


def _iter_sqlite_aliases(
    db_path: Path,
    pack_id: str,
) -> Iterable[tuple[str, str, str]]:
    if not db_path.exists():
        return
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if not _table_exists(conn, "content_aliases"):
            return
        columns = _table_columns(conn, "content_aliases")
        alias_col = (
            "alias" if "alias" in columns else "name" if "name" in columns else ""
        )
        ref_col = (
            "ref" if "ref" in columns else "ref_id" if "ref_id" in columns else ""
        )
        if not alias_col or not ref_col:
            return
        selected = [alias_col, ref_col]
        where_parts: list[str] = []
        params: list[Any] = []
        if pack_id and "pack_id" in columns:
            selected.append("pack_id")
            where_parts.append("pack_id = ?")
            params.append(pack_id)
        if "review_status" in columns:
            where_parts.append(
                "review_status IN ("
                + ",".join("?" for _ in _APPROVED_REVIEW_STATUSES)
                + ")"
            )
            params.extend(sorted(_APPROVED_REVIEW_STATUSES))
        if "confidence" in columns:
            where_parts.append("confidence >= ?")
            params.append(_MIN_ALIAS_CONFIDENCE)
        where = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        sql = f"SELECT {', '.join(selected)} FROM content_aliases{where}"
        rows = conn.execute(sql, params).fetchall()
    for row in rows:
        yield str(row[alias_col] or ""), str(row[ref_col] or ""), "sqlite_alias"


def _known_router_content_refs(ckpt: Any) -> set[tuple[str, str]]:
    known: set[tuple[str, str]] = set()
    session = getattr(ckpt, "session", None)
    content_state = getattr(session, "content_state", {}) if session else {}
    if isinstance(content_state, Mapping):
        for pack_key, pack_state in content_state.items():
            pack_id = _pack_id(pack_key, pack_state)
            introduced = (
                pack_state.get("introduced_refs")
                if isinstance(pack_state, Mapping)
                else getattr(pack_state, "introduced_refs", None)
            )
            if isinstance(introduced, Mapping):
                values = introduced.values()
            else:
                values = introduced or []
            for item in values:
                ref = _record_ref(item)
                if ref:
                    known.add((pack_id, ref))

    for message in getattr(ckpt, "session_conversation", []) or []:
        if getattr(message, "role", "") != "assistant":
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str) or not content.startswith(
            ("content_known ", "location_card ", "front_signal ")
        ):
            continue
        ref_match = _REF_RE.search(content)
        if not ref_match:
            continue
        pack_match = re.search(r"\bpack=([A-Za-z0-9_.:/@+-]+)", content)
        known.add(((pack_match.group(1) if pack_match else ""), ref_match.group(1)))
    return known


def _metadata(pack_state: Any) -> Mapping[str, Any]:
    raw = (
        pack_state.get("metadata")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "metadata", None)
    )
    return raw if isinstance(raw, Mapping) else {}


def _pack_id(pack_key: Any, pack_state: Any) -> str:
    raw = (
        pack_state.get("pack_id")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "pack_id", "")
    )
    return str(raw or pack_key or "").strip()


def _pack_db_path(pack_state: Any) -> Path | None:
    metadata = _metadata(pack_state)
    for key in ("db_path", "pack_path", "sqlite_path", "content_db_path"):
        raw = metadata.get(key)
        if raw:
            return Path(str(raw)).expanduser()
    return None


def _record_ref(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("ref") or item.get("ref_id") or "").strip()
    return str(
        getattr(item, "ref", "")
        or getattr(item, "ref_id", "")
        or ""
    ).strip()


def _contains_alias(text: str, alias: str) -> bool:
    needle = " ".join(str(alias).strip().lower().split())
    if not needle:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text) is not None


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}
