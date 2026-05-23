from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.engine.content_lookup import (
    _assert_pack_runtime_identity,
    _known_router_content_refs,
    _pack_db_path,
    _pack_id,
    build_router_content_lookup_catalog_block,
)
from app.engine.content_resolver import ContentCard, load_content_cards
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.content_manager import (
    ContentManagerContentUpdate,
    ContentManagerOutput,
    ContentManagerTurnHint,
)
from app.schemas.content_privacy import contains_imported_asset_sentinel


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")
_CANDIDATE_KEYS = frozenset((
    "front",
    "faction",
    "location",
    "name",
    "role",
    "status",
    "current_objective",
    "stance",
))


class ContentManagerValidationError(RuntimeError):
    """Raised when proposed router content updates fail validation."""


async def plan_content_manager_updates(
    ckpt: Any,
    *,
    candidate_entities: Mapping[str, Any] | Sequence[str],
    client: LLMClient,
    prompt_mgr: PromptManager,
    max_recent_facts: int = 12,
    max_catalog_cards_per_pack: int = 120,
) -> ContentManagerOutput:
    """Ask for router content deltas and optional turn hints, then validate."""

    catalog_block = build_content_manager_catalog_block(
        ckpt,
        max_cards_per_pack=max_catalog_cards_per_pack,
    )
    candidate_entities_block = build_candidate_turn_entities_block(
        candidate_entities,
    )
    if not catalog_block:
        return ContentManagerOutput(
            content_updates=[],
            turn_hints=[],
            no_update_reason="No reviewed content catalog is available.",
        )

    messages = build_content_manager_messages(
        ckpt,
        candidate_entities=candidate_entities,
        prompt_mgr=prompt_mgr,
        catalog_block=catalog_block,
        candidate_entities_block=candidate_entities_block,
        max_recent_facts=max_recent_facts,
    )
    response = await client.complete(
        role="content_manager",
        messages=messages,
        response_model=ContentManagerOutput,
        temperature=0.1,
        max_tokens=1200,
        cache=True,
        compact=True,
    )
    parsed = response.parsed
    if not isinstance(parsed, ContentManagerOutput):
        raise ContentManagerValidationError(
            "Content manager returned no structured update proposal"
        )
    return validate_content_manager_output(
        ckpt,
        parsed,
        candidate_entities=candidate_entities,
    )


def build_content_manager_messages(
    ckpt: Any,
    *,
    candidate_entities: Mapping[str, Any] | Sequence[str],
    prompt_mgr: PromptManager,
    catalog_block: str | None = None,
    candidate_entities_block: str | None = None,
    max_recent_facts: int = 12,
) -> list[dict[str, str]]:
    return prompt_mgr.render_messages(
        "content_manager",
        recent_fact_limit=str(max_recent_facts),
        recent_facts_block=build_recent_canonical_facts_block(
            ckpt,
            limit=max_recent_facts,
        ),
        known_router_refs_block=build_known_router_refs_block(ckpt),
        candidate_entities_block=(
            candidate_entities_block
            if candidate_entities_block is not None
            else build_candidate_turn_entities_block(candidate_entities)
        ),
        available_catalog_block=(
            catalog_block
            if catalog_block is not None
            else build_content_manager_catalog_block(ckpt)
        ),
    )


def build_content_manager_catalog_block(
    ckpt: Any,
    *,
    max_cards_per_pack: int = 120,
) -> str:
    return build_router_content_lookup_catalog_block(
        ckpt,
        max_cards_per_pack=max_cards_per_pack,
    )


def build_recent_canonical_facts_block(ckpt: Any, *, limit: int = 12) -> str:
    rows: list[tuple[str, str]] = []
    for event in getattr(ckpt, "canonical_events", []) or []:
        canonical = getattr(event, "canonical_event", None)
        if canonical is None and isinstance(event, Mapping):
            canonical = event.get("canonical_event")
        facts = getattr(canonical, "observable_facts", []) if canonical else []
        if isinstance(canonical, Mapping):
            facts = canonical.get("observable_facts") or []
        for fact in facts or []:
            text = _fact_text(fact)
            if not text:
                continue
            audience = _fact_audience(fact)
            rows.append((audience, text))

    recent = rows[-max(0, limit):]
    formatted = [
        f"f{index:02d} audience={audience} text={_quote_value(text)}"
        for index, (audience, text) in enumerate(recent, start=1)
    ]
    return "\n".join(formatted) or "-"


def build_known_router_refs_block(ckpt: Any) -> str:
    rows = [
        f"pack={_safe_token(pack_id)} ref={_safe_token(ref)}"
        for pack_id, ref in sorted(_known_router_content_refs(ckpt))
        if _safe_token(pack_id) and _safe_token(ref)
    ]
    return "\n".join(rows) or "-"


def build_candidate_turn_entities_block(
    candidate_entities: Mapping[str, Any] | Sequence[str],
) -> str:
    rows: list[str] = []
    for raw_entity_id, raw_value in _iter_candidate_entities(candidate_entities):
        entity_id = _safe_token(raw_entity_id)
        if not entity_id:
            continue
        parts = [f"character={entity_id}"]
        if isinstance(raw_value, Mapping):
            for raw_key, raw_item in sorted(
                raw_value.items(),
                key=lambda item: str(item[0]),
            ):
                key = _safe_token(raw_key)
                if key not in _CANDIDATE_KEYS:
                    continue
                value = _format_value(raw_item)
                if value:
                    parts.append(f"{key}={value}")
        rows.append(" ".join(parts))
    return "\n".join(rows) or "-"


def validate_content_manager_output(
    ckpt: Any,
    output: ContentManagerOutput,
    *,
    candidate_entities: Mapping[str, Any] | Sequence[str],
) -> ContentManagerOutput:
    candidate_ids = _candidate_entity_ids(candidate_entities)
    errors: list[str] = []
    cards: dict[tuple[str, str], ContentCard | None] = {}
    validated_updates: list[ContentManagerContentUpdate] = []

    for update in output.content_updates:
        card = _runtime_card_or_none(
            ckpt,
            update.pack_id,
            update.ref,
            cards,
        )
        if card is None:
            errors.append(f"missing content pack={update.pack_id} ref={update.ref}")
            continue
        if update.content_hash and update.content_hash != card.content_hash:
            errors.append(
                "content hash mismatch "
                f"pack={update.pack_id} ref={update.ref} "
                f"expected={card.content_hash} actual={update.content_hash}"
            )
            continue
        validated_updates.append(
            update.model_copy(update={"content_hash": card.content_hash})
        )

    validated_hints: list[ContentManagerTurnHint] = []
    for hint in output.turn_hints:
        if hint.character_id not in candidate_ids:
            errors.append(f"unknown character_id={hint.character_id or '-'}")
            continue
        bad_refs = [
            ref
            for ref in hint.related_content_refs
            if _runtime_card_from_compact_ref(ckpt, ref, cards) is None
        ]
        if bad_refs:
            errors.append(
                f"invalid hint refs character_id={hint.character_id} "
                f"refs={','.join(bad_refs)}"
            )
            continue
        validated_hints.append(hint)

    if errors:
        raise ContentManagerValidationError("; ".join(errors))

    return output.model_copy(update={
        "content_updates": _dedupe_updates(validated_updates),
        "turn_hints": _dedupe_hints(validated_hints),
    })


def format_content_manager_router_records(
    output: ContentManagerOutput,
) -> list[str]:
    records: list[str] = []
    for update in output.content_updates:
        parts = [
            "content_update",
            f"kind={_safe_token(update.update_kind)}",
            f"pack={_safe_token(update.pack_id)}",
            f"ref={_safe_token(update.ref)}",
        ]
        if update.content_hash:
            parts.append(f"hash={_safe_token(update.content_hash)}")
        if update.source_fact_ids:
            parts.append(f"facts={_join_tokens(update.source_fact_ids)}")
        if update.reason:
            parts.append(f"reason={_quote_value(update.reason)}")
        records.append(
            " ".join(part for part in parts if part and not part.endswith("="))
        )

    for hint in output.turn_hints:
        parts = [
            "turn_hint",
            f"character={_safe_token(hint.character_id)}",
            f"priority={_safe_token(hint.priority)}",
        ]
        if hint.related_content_refs:
            parts.append(f"refs={_join_tokens(hint.related_content_refs)}")
        if hint.source_fact_ids:
            parts.append(f"facts={_join_tokens(hint.source_fact_ids)}")
        if hint.reason:
            parts.append(f"reason={_quote_value(hint.reason)}")
        records.append(
            " ".join(part for part in parts if part and not part.endswith("="))
        )

    return records


def _runtime_card_or_none(
    ckpt: Any,
    pack_id: str,
    ref: str,
    cache: dict[tuple[str, str], ContentCard | None],
) -> ContentCard | None:
    key = (pack_id, ref)
    if key not in cache:
        cache[key] = _resolve_runtime_card(
            ckpt,
            pack_id=pack_id,
            ref=ref,
        )
    return cache[key]


def _runtime_card_from_compact_ref(
    ckpt: Any,
    value: str,
    cache: dict[tuple[str, str], ContentCard | None],
) -> ContentCard | None:
    pack_id, ref = _split_compact_ref(value)
    if not pack_id or not ref:
        return None
    return _runtime_card_or_none(ckpt, pack_id, ref, cache)


def _resolve_runtime_card(
    ckpt: Any,
    *,
    pack_id: str,
    ref: str,
) -> ContentCard | None:
    pack_states = _content_pack_states_by_id(ckpt)
    pack_state = pack_states.get(pack_id)
    if pack_state is None:
        return None
    db_path = _pack_db_path(pack_state)
    if db_path is None:
        return None
    _assert_pack_runtime_identity(db_path, pack_id=pack_id, pack_state=pack_state)
    cards = load_content_cards(
        db_path,
        refs=[ref],
        pack_id=pack_id,
        runtime_only=True,
    )
    return cards[0] if cards else None


def _content_pack_states_by_id(ckpt: Any) -> dict[str, Any]:
    session = getattr(ckpt, "session", None)
    content_state = getattr(session, "content_state", None) if session else None
    if not isinstance(content_state, Mapping):
        return {}
    return {
        pack_id: pack_state
        for pack_key, pack_state in content_state.items()
        if (pack_id := _pack_id(pack_key, pack_state))
    }


def _iter_candidate_entities(
    candidate_entities: Mapping[str, Any] | Sequence[str],
) -> Iterable[tuple[Any, Any]]:
    if isinstance(candidate_entities, Mapping):
        yield from sorted(candidate_entities.items(), key=lambda item: str(item[0]))
        return
    for entity_id in candidate_entities:
        yield entity_id, {}


def _candidate_entity_ids(
    candidate_entities: Mapping[str, Any] | Sequence[str],
) -> set[str]:
    return {
        entity_id
        for raw_entity_id, _ in _iter_candidate_entities(candidate_entities)
        if (entity_id := _safe_token(raw_entity_id))
    }


def _dedupe_updates(
    updates: Iterable[ContentManagerContentUpdate],
) -> list[ContentManagerContentUpdate]:
    deduped: dict[tuple[str, str, str], ContentManagerContentUpdate] = {}
    for update in updates:
        deduped.setdefault(update.dedupe_key(), update)
    return list(deduped.values())


def _dedupe_hints(
    hints: Iterable[ContentManagerTurnHint],
) -> list[ContentManagerTurnHint]:
    deduped: dict[str, ContentManagerTurnHint] = {}
    for hint in hints:
        deduped.setdefault(hint.dedupe_key(), hint)
    return list(deduped.values())


def _fact_text(fact: Any) -> str:
    if isinstance(fact, str):
        return _safe_text(fact)
    if isinstance(fact, Mapping):
        return _safe_text(fact.get("text"))
    return _safe_text(getattr(fact, "text", ""))


def _fact_audience(fact: Any) -> str:
    audience = ""
    visible_to: Sequence[Any] = ()
    if isinstance(fact, Mapping):
        audience = _safe_token(fact.get("audience")) or "all_observers"
        visible_to = fact.get("visible_to") or ()
    else:
        audience = _safe_token(getattr(fact, "audience", "")) or "all_observers"
        visible_to = getattr(fact, "visible_to", ()) or ()
    if audience == "only":
        viewers = [
            viewer
            for raw in visible_to
            if (viewer := _safe_token(raw))
        ]
        return "only:" + ",".join(viewers) if viewers else "only"
    return "all"


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [
            formatted
            for item in value
            if (formatted := _format_value(item))
        ]
        if not values:
            return "[]"
        if all(_SAFE_TOKEN_RE.fullmatch(item) for item in values):
            return ",".join(values)
        return json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    text = _safe_text(value)
    if not text:
        return ""
    if _SAFE_TOKEN_RE.fullmatch(text):
        return text
    return json.dumps(text, ensure_ascii=True, separators=(",", ":"))


def _split_compact_ref(value: str) -> tuple[str, str]:
    text = _safe_token(value)
    if ":" not in text:
        return "", ""
    pack_id, ref = text.split(":", 1)
    return _safe_token(pack_id), _safe_token(ref)


def _join_tokens(values: Iterable[str]) -> str:
    return ",".join(token for value in values if (token := _safe_token(value)))


def _safe_token(value: Any) -> str:
    text = _safe_text(value)
    return text if text and _SAFE_TOKEN_RE.fullmatch(text) else ""


def _safe_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text or contains_imported_asset_sentinel(text):
        return ""
    return text


def _quote_value(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return '""'
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
