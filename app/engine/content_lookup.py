from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.engine.prompt_manager import PromptManager
from app.engine.router_prompt_projection import router_prompt_history
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
from app.llm.client import LLMClient
from app.schemas.conversation import ConversationMessage
from app.schemas.content_privacy import contains_imported_asset_sentinel


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


class EventRouterContentLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    ref: str = ""
    reason: str = ""
    alias: str = ""
    urgency: Literal["required", "optional"] = "required"
    spoiler_boundary: str = "router_hidden"

    @model_validator(mode="after")
    def _clean(self) -> "EventRouterContentLookupRequest":
        self.pack_id = self.pack_id.strip()
        self.ref = self.ref.strip()
        self.reason = " ".join(self.reason.split())
        self.alias = " ".join(self.alias.split())
        self.spoiler_boundary = self.spoiler_boundary.strip() or "router_hidden"
        return self


class EventRouterContentLookupOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: list[EventRouterContentLookupRequest] = Field(default_factory=list)
    no_lookup_reason: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "EventRouterContentLookupOutput":
        self.no_lookup_reason = " ".join(self.no_lookup_reason.split())
        return self


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
    llm_requests: Iterable[ContentLookupRequest] = (),
) -> list[str]:
    """Append one-shot router content records before the normal router call.

    This is intentionally the smallest deterministic preflight slice: queued
    pending signals are drained first, then pack-local aliases/catalog entries
    are matched against current actor input and location. A future LLM lookup
    prompt can plug into `plan_llm_router_content_lookup_requests` without
    changing the ordinary CanonicalEventRecord schema.
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
    requests.extend(_dedupe_lookup_requests(list(llm_requests), known_refs=known_refs))
    if not requests:
        return records

    lookup_records = _fetch_lookup_records(ckpt, requests)
    for record in lookup_records:
        conversation.append(ConversationMessage(role="assistant", content=record))
    records.extend(lookup_records)
    return records


async def append_router_content_lookup_records_with_llm(
    ckpt: Any,
    *,
    actor_id: str,
    current_input: str,
    client: LLMClient,
    prompt_mgr: PromptManager,
    max_lookup_passes: int = 1,
) -> list[str]:
    """Append deterministic plus bounded model-selected content records."""

    records = append_pending_router_content_records(ckpt)

    session = getattr(ckpt, "session", None)
    content_state = getattr(session, "content_state", None) if session else None
    conversation = getattr(ckpt, "session_conversation", None)
    if not isinstance(content_state, Mapping) or conversation is None:
        return records

    known_refs = _known_router_content_refs(ckpt)
    deterministic_requests = plan_deterministic_router_content_lookup_requests(
        ckpt,
        actor_id=actor_id,
        current_input=current_input,
        known_refs=known_refs,
    )
    deterministic_requests = _dedupe_lookup_requests(
        deterministic_requests,
        known_refs=known_refs,
    )
    if max_lookup_passes < 1:
        lookup_records = _fetch_lookup_records(ckpt, deterministic_requests)
        _append_lookup_records(conversation, lookup_records)
        records.extend(lookup_records)
        return records

    catalog_block = build_router_content_lookup_catalog_block(ckpt)
    llm_requests: list[ContentLookupRequest] = []
    previous_missing: list[ContentLookupRequest] = []
    if catalog_block and not deterministic_requests:
        for pass_index in range(max_lookup_passes):
            llm_requests = await plan_llm_router_content_lookup_requests(
                ckpt,
                actor_id=actor_id,
                current_input=current_input,
                known_refs=known_refs,
                deterministic_requests=deterministic_requests,
                client=client,
                prompt_mgr=prompt_mgr,
                catalog_block=catalog_block,
                previous_missing=previous_missing,
            )
            requests = _dedupe_lookup_requests(
                [*deterministic_requests, *llm_requests],
                known_refs=known_refs,
            )
            try:
                lookup_records = _fetch_lookup_records(ckpt, requests)
            except MissingContentError as exc:
                if pass_index + 1 >= max_lookup_passes or not llm_requests:
                    raise
                previous_missing = exc.requests
                continue
            _append_lookup_records(conversation, lookup_records)
            records.extend(lookup_records)
            return records

    requests = _dedupe_lookup_requests(
        [*deterministic_requests, *llm_requests],
        known_refs=known_refs,
    )
    lookup_records = _fetch_lookup_records(ckpt, requests)
    _append_lookup_records(conversation, lookup_records)
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


async def plan_llm_router_content_lookup_requests(
    ckpt: Any,
    *,
    actor_id: str,
    current_input: str,
    known_refs: set[tuple[str, str]],
    deterministic_requests: list[ContentLookupRequest],
    client: LLMClient,
    prompt_mgr: PromptManager,
    catalog_block: str = "",
    previous_missing: list[ContentLookupRequest] | None = None,
) -> list[ContentLookupRequest]:
    """Ask for exact reviewed refs that should be fetched before routing."""

    catalog_text = catalog_block or build_router_content_lookup_catalog_block(ckpt)
    if not catalog_text:
        return []
    messages = prompt_mgr.render_messages(
        "event_router_content_lookup",
        actor_id=actor_id,
        current_input=current_input,
        router_memory_block=_router_memory_block(ckpt),
        known_refs_block=_known_refs_block(known_refs),
        deterministic_requests_block=_requests_block(deterministic_requests),
        catalog_block=catalog_text,
        previous_missing_block=_requests_block(previous_missing or []),
    )
    response = await client.complete(
        role="event_router",
        messages=messages,
        response_model=EventRouterContentLookupOutput,
        temperature=0.1,
        max_tokens=1200,
        cache=True,
        compact=True,
    )
    parsed = response.parsed
    if not isinstance(parsed, EventRouterContentLookupOutput):
        return []
    requests = [
        ContentLookupRequest(
            pack_id=request.pack_id,
            ref=request.ref,
            reason=request.reason,
            alias=request.alias,
            urgency=request.urgency,
            spoiler_boundary=request.spoiler_boundary,
            already_known=(request.pack_id, request.ref) in known_refs,
        )
        for request in parsed.requests
        if request.pack_id and request.ref
    ]
    return _dedupe_lookup_requests(requests, known_refs=known_refs)


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
                request
                for request in pack_requests
                if not request.already_known and request.urgency != "optional"
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
                if not request.already_known and request.urgency != "optional":
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


def _append_lookup_records(conversation: Any, records: list[str]) -> None:
    for record in records:
        conversation.append(ConversationMessage(role="assistant", content=record))


def _dedupe_lookup_requests(
    requests: Iterable[ContentLookupRequest],
    *,
    known_refs: set[tuple[str, str]],
) -> list[ContentLookupRequest]:
    deduped: list[ContentLookupRequest] = []
    seen: set[tuple[str, str]] = set()
    for request in requests:
        pack_id = str(request.pack_id or "").strip()
        ref = str(request.ref or "").strip()
        if not pack_id or not ref:
            continue
        key = (pack_id, ref)
        if key in seen:
            continue
        seen.add(key)
        if key in known_refs:
            continue
        deduped.append(
            ContentLookupRequest(
                pack_id=pack_id,
                ref=ref,
                reason=" ".join(str(request.reason or "").split()),
                alias=" ".join(str(request.alias or "").split()),
                urgency=(
                    "optional" if request.urgency == "optional" else "required"
                ),
                spoiler_boundary=(
                    str(request.spoiler_boundary or "").strip()
                    or "router_hidden"
                ),
                already_known=False,
            )
        )
    return deduped


def build_router_content_lookup_catalog_block(
    ckpt: Any,
    *,
    max_cards_per_pack: int = 80,
) -> str:
    session = getattr(ckpt, "session", None)
    content_state = getattr(session, "content_state", None) if session else None
    if not isinstance(content_state, Mapping):
        return ""

    lines: list[str] = []
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        db_path = _pack_db_path(pack_state)
        if db_path is None:
            continue
        _assert_pack_runtime_identity(db_path, pack_id=pack_id, pack_state=pack_state)
        projected_lines = _projected_router_lookup_catalog_lines(
            pack_state,
            pack_id=pack_id,
            max_cards=max_cards_per_pack,
        )
        if projected_lines:
            lines.extend(projected_lines)
            continue
        aliases_by_ref = _aliases_by_ref(pack_state)
        cards = load_content_cards(
            db_path,
            pack_id=pack_id,
            limit=max_cards_per_pack,
            runtime_only=True,
        )
        for card in cards:
            aliases = aliases_by_ref.get(card.ref, [])
            parts = [
                f"pack={_safe_token(pack_id)}",
                f"ref={_safe_token(card.ref)}",
            ]
            if card.kind:
                parts.append(f"kind={_safe_token(card.kind)}")
            if card.visibility:
                parts.append(f"visibility={_safe_token(card.visibility)}")
            if aliases:
                parts.append("aliases=" + _quote_value(", ".join(aliases[:8])))
            if card.title:
                parts.append("title=" + _quote_value(card.title))
            if card.summary:
                parts.append("summary=" + _quote_value(card.summary))
            lines.append(" ".join(parts))
    return "\n".join(lines)


def _projected_router_lookup_catalog_lines(
    pack_state: Any,
    *,
    pack_id: str,
    max_cards: int,
) -> list[str]:
    metadata = _metadata(pack_state)
    raw = metadata.get("router_lookup_catalog") or metadata.get("catalog")
    if not isinstance(raw, list):
        return []
    rows: list[str] = []
    for item in raw[: max(0, int(max_cards))]:
        if not isinstance(item, Mapping):
            continue
        ref = _safe_token(item.get("ref") or item.get("ref_id"))
        if not ref:
            continue
        parts = [f"pack={_safe_token(pack_id)}", f"ref={ref}"]
        kind = _safe_token(item.get("kind") or item.get("card_kind"))
        visibility = _safe_token(item.get("visibility"))
        if kind:
            parts.append(f"kind={kind}")
        if visibility:
            parts.append(f"visibility={visibility}")
        aliases = [
            alias
            for alias in (
                _safe_catalog_text(alias)
                for alias in _catalog_alias_values(
                    item.get("aliases") or item.get("names") or []
                )
            )
            if alias
        ]
        if aliases:
            parts.append("aliases=" + _quote_value(", ".join(aliases[:8])))
        title = _safe_catalog_text(item.get("title") or item.get("label") or "")
        summary = _safe_catalog_text(item.get("summary") or "")
        if title:
            parts.append("title=" + _quote_value(title))
        if summary:
            parts.append("summary=" + _quote_value(summary))
        rows.append(" ".join(part for part in parts if part))
    return rows


def _catalog_alias_values(raw: Any) -> list[Any]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return list(raw)
    return []


def _aliases_by_ref(pack_state: Any) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for alias, ref, _reason in _iter_alias_catalog_entries(pack_state):
        safe_alias = _safe_catalog_text(alias)
        safe_ref = _safe_catalog_text(ref)
        if not safe_alias or not safe_ref:
            continue
        values = aliases.setdefault(safe_ref, [])
        if safe_alias not in values:
            values.append(safe_alias)
    return aliases


def _router_memory_block(ckpt: Any, *, limit: int = 18) -> str:
    rows: list[str] = []
    for message in router_prompt_history(ckpt):
        if getattr(message, "role", "") != "assistant":
            continue
        content = _safe_catalog_text(getattr(message, "content", ""))
        if not content:
            continue
        if content.startswith(("prior_event ", "content_known ", "location_card ", "front_signal ")):
            rows.append(content)
    return "\n".join(rows[-limit:]) or "-"


def _known_refs_block(known_refs: set[tuple[str, str]]) -> str:
    rows = [
        f"pack={_safe_token(pack_id)} ref={_safe_token(ref)}"
        for pack_id, ref in sorted(known_refs)
        if _safe_token(pack_id) and _safe_token(ref)
    ]
    return "\n".join(rows) or "-"


def _requests_block(requests: Iterable[ContentLookupRequest]) -> str:
    rows = []
    for request in requests:
        parts = [
            f"pack={_safe_token(request.pack_id)}",
            f"ref={_safe_token(request.ref)}",
        ]
        if request.alias:
            parts.append("alias=" + _quote_value(request.alias))
        if request.reason:
            parts.append("reason=" + _quote_value(request.reason))
        rows.append(" ".join(part for part in parts if part and not part.endswith("=")))
    return "\n".join(rows) or "-"


def _quote_value(value: str) -> str:
    text = _safe_catalog_text(value)
    if not text:
        return '""'
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _safe_token(value: Any) -> str:
    text = _safe_catalog_text(value)
    if not text or not re.match(r"^[A-Za-z0-9_.:/@+-]+$", text):
        return ""
    return text


def _safe_catalog_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text or contains_imported_asset_sentinel(text):
        return ""
    return text


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
