from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.engine.content_lookup import (
    _assert_pack_runtime_identity,
    _pack_db_path,
    _pack_id,
    build_router_content_lookup_catalog_block,
)
from app.engine.content_resolver import ContentCard, load_content_cards
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.content_manager import (
    ContentManagerEntityUpdate,
    ContentManagerOutput,
)
from app.schemas.content_privacy import contains_imported_asset_sentinel


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")


class ContentManagerValidationError(RuntimeError):
    """Raised when proposed entity knowledge updates fail runtime validation."""


async def plan_content_manager_updates(
    ckpt: Any,
    *,
    entity_knowledge: Mapping[str, Any],
    client: LLMClient,
    prompt_mgr: PromptManager,
    max_recent_facts: int = 12,
    max_catalog_cards_per_pack: int = 120,
) -> ContentManagerOutput:
    """Ask the content manager for entity knowledge updates, then validate refs."""

    catalog_block = build_content_manager_catalog_block(
        ckpt,
        max_cards_per_pack=max_catalog_cards_per_pack,
    )
    entity_knowledge_block = build_entity_knowledge_block(entity_knowledge)
    if not catalog_block:
        return ContentManagerOutput(
            updates=[],
            no_update_reason="No reviewed content catalog is available.",
        )
    if not _entity_ids_from_knowledge(entity_knowledge):
        return ContentManagerOutput(
            updates=[],
            no_update_reason="No entity knowledge map is available.",
        )

    messages = build_content_manager_messages(
        ckpt,
        entity_knowledge=entity_knowledge,
        prompt_mgr=prompt_mgr,
        catalog_block=catalog_block,
        entity_knowledge_block=entity_knowledge_block,
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
        entity_knowledge=entity_knowledge,
    )


def build_content_manager_messages(
    ckpt: Any,
    *,
    entity_knowledge: Mapping[str, Any],
    prompt_mgr: PromptManager,
    catalog_block: str | None = None,
    entity_knowledge_block: str | None = None,
    max_recent_facts: int = 12,
) -> list[dict[str, str]]:
    return prompt_mgr.render_messages(
        "content_manager",
        recent_fact_limit=str(max_recent_facts),
        recent_facts_block=build_recent_canonical_facts_block(
            ckpt,
            limit=max_recent_facts,
        ),
        entity_knowledge_block=(
            entity_knowledge_block
            if entity_knowledge_block is not None
            else build_entity_knowledge_block(entity_knowledge)
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


def build_entity_knowledge_block(entity_knowledge: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for raw_entity_id, raw_value in sorted(
        entity_knowledge.items(),
        key=lambda item: str(item[0]),
    ):
        entity_id = _safe_token(raw_entity_id)
        if not entity_id:
            continue
        value = _format_entity_value(raw_value)
        rows.append(f"entity={entity_id}" + (f" {value}" if value else " knows=[]"))
    return "\n".join(rows) or "-"


def validate_content_manager_output(
    ckpt: Any,
    output: ContentManagerOutput,
    *,
    entity_knowledge: Mapping[str, Any],
) -> ContentManagerOutput:
    entity_ids = _entity_ids_from_knowledge(entity_knowledge)
    errors: list[str] = []
    validated: list[ContentManagerEntityUpdate] = []
    cards: dict[tuple[str, str], ContentCard | None] = {}

    for update in output.updates:
        if update.entity_id not in entity_ids:
            errors.append(f"unknown entity_id={update.entity_id or '-'}")
            continue

        key = (update.pack_id, update.ref)
        if key not in cards:
            cards[key] = _resolve_runtime_card(
                ckpt,
                pack_id=update.pack_id,
                ref=update.ref,
            )
        card = cards[key]
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
        validated.append(
            update.model_copy(update={"content_hash": card.content_hash})
        )

    if errors:
        raise ContentManagerValidationError("; ".join(errors))

    return output.model_copy(update={"updates": _dedupe_updates(validated)})


def format_content_manager_update_records(
    output: ContentManagerOutput,
) -> list[str]:
    records: list[str] = []
    for update in output.updates:
        parts = [
            "content_update",
            f"entity={_safe_token(update.entity_id)}",
            f"state={_safe_token(update.knowledge_state)}",
            f"pack={_safe_token(update.pack_id)}",
            f"ref={_safe_token(update.ref)}",
        ]
        if update.content_hash:
            parts.append(f"hash={_safe_token(update.content_hash)}")
        if update.source_fact_ids:
            facts = ",".join(
                fact_id
                for fact_id in (
                    _safe_token(value) for value in update.source_fact_ids
                )
                if fact_id
            )
            parts.append(
                f"facts={facts}"
            )
        if update.reason:
            parts.append(f"reason={_quote_value(update.reason)}")
        records.append(
            " ".join(part for part in parts if part and not part.endswith("="))
        )
    return records


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


def _entity_ids_from_knowledge(entity_knowledge: Mapping[str, Any]) -> set[str]:
    return {
        entity_id
        for raw_entity_id in entity_knowledge
        if (entity_id := _safe_token(raw_entity_id))
    }


def _dedupe_updates(
    updates: Iterable[ContentManagerEntityUpdate],
) -> list[ContentManagerEntityUpdate]:
    deduped: dict[tuple[str, str, str, str], ContentManagerEntityUpdate] = {}
    for update in updates:
        deduped.setdefault(update.dedupe_key(), update)
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


def _format_entity_value(value: Any) -> str:
    if isinstance(value, Mapping):
        parts = []
        for raw_key, raw_item in sorted(value.items(), key=lambda item: str(item[0])):
            key = _safe_token(raw_key)
            item = _format_entity_value(raw_item)
            if key and item:
                parts.append(f"{key}={item}")
        return " ".join(parts)
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [
            formatted
            for item in value
            if (formatted := _format_entity_value(item))
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
