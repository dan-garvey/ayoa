from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.engine.content_lookup import (
    ContentLookupRequest,
    append_router_content_lookup_records,
    _assert_pack_runtime_identity,
    _known_router_content_refs,
    _pack_db_path,
    _pack_id,
    build_router_content_lookup_catalog_block,
)
from app.engine.content_resolver import ContentCard, load_content_cards
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.conversation import ConversationMessage
from app.schemas.content import ContentKnowledgeEntityState
from app.schemas.content_manager import (
    ContentManagerAgentContextBroadcast,
    ContentManagerKnowledgeUpdate,
    ContentManagerOutput,
    ContentManagerRouterRequiredKnowledge,
    ContentManagerRouterTurnCandidate,
)
from app.schemas.content_privacy import contains_imported_asset_sentinel


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")
CONTENT_MANAGER_MAX_TOKENS = 8000
_DEFAULT_CONTENT_MANAGER_REFRESH_INTERVAL = 3
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
logger = logging.getLogger(__name__)


class ContentManagerValidationError(RuntimeError):
    """Raised when proposed content-manager deltas fail validation."""


def content_manager_enabled(ckpt: Any) -> bool:
    """Return true when a checkpoint carries a content-manager knowledge map."""

    for pack_state in _content_pack_states_by_id(ckpt).values():
        knowledge_map = getattr(pack_state, "knowledge_map", None)
        if isinstance(knowledge_map, Mapping) and knowledge_map:
            return True
    return False


def should_run_content_manager_preflight(ckpt: Any) -> bool:
    """Advance the preflight cycle and decide whether the manager should run."""

    if not _has_recent_canonical_facts(ckpt):
        return False
    session = getattr(ckpt, "session", None)
    if session is None:
        return True
    interval = _content_manager_refresh_interval(ckpt)
    cycle = max(0, int(getattr(session, "content_manager_preflight_cycle", 0)))
    last_run = int(getattr(session, "content_manager_last_run_cycle", -1))
    should_run = last_run < 0 or cycle - last_run >= interval
    session.content_manager_preflight_cycle = cycle + 1
    if should_run:
        session.content_manager_last_run_cycle = cycle
    return should_run


async def plan_content_manager_updates(
    ckpt: Any,
    *,
    client: LLMClient,
    prompt_mgr: PromptManager,
    candidate_entities: Mapping[str, Any] | Sequence[str] | None = None,
    max_recent_facts: int = 12,
    max_catalog_cards_per_pack: int = 120,
) -> ContentManagerOutput:
    """Ask for knowledge-map patches and router deltas, then validate them."""

    catalog_block = build_content_manager_catalog_block(
        ckpt,
        max_cards_per_pack=max_catalog_cards_per_pack,
    )
    resolved_candidates = (
        build_candidate_turn_entities_from_checkpoint(ckpt)
        if candidate_entities is None
        else candidate_entities
    )
    candidate_entities_block = build_candidate_turn_entities_block(
        resolved_candidates,
    )
    if not catalog_block:
        return ContentManagerOutput(
            knowledge_updates=[],
            router_required_knowledge=[],
            router_turn_candidates=[],
            agent_context_broadcasts=[],
            no_update_reason="No reviewed content catalog is available.",
        )

    messages = build_content_manager_messages(
        ckpt,
        candidate_entities=resolved_candidates,
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
        max_tokens=CONTENT_MANAGER_MAX_TOKENS,
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
        candidate_entities=resolved_candidates,
    )


def build_content_manager_messages(
    ckpt: Any,
    *,
    candidate_entities: Mapping[str, Any] | Sequence[str] | None = None,
    prompt_mgr: PromptManager,
    catalog_block: str | None = None,
    candidate_entities_block: str | None = None,
    max_recent_facts: int = 12,
) -> list[dict[str, str]]:
    resolved_candidates = (
        build_candidate_turn_entities_from_checkpoint(ckpt)
        if candidate_entities is None
        else candidate_entities
    )
    return prompt_mgr.render_messages(
        "content_manager",
        recent_fact_limit=str(max_recent_facts),
        recent_facts_block=build_recent_canonical_facts_block(
            ckpt,
            limit=max_recent_facts,
        ),
        engine_knowledge_map_block=build_content_knowledge_map_block(ckpt),
        known_router_refs_block=build_known_router_refs_block(ckpt),
        candidate_entities_block=(
            candidate_entities_block
            if candidate_entities_block is not None
            else build_candidate_turn_entities_block(resolved_candidates)
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


def build_content_knowledge_map_block(ckpt: Any) -> str:
    rows: list[str] = []
    for pack_key, pack_state in _content_pack_states_by_id(ckpt).items():
        pack_id = _safe_token(_pack_id(pack_key, pack_state))
        knowledge_map = getattr(pack_state, "knowledge_map", None)
        if not pack_id or not isinstance(knowledge_map, Mapping):
            continue
        for raw_entity_id, raw_state in sorted(
            knowledge_map.items(),
            key=lambda item: str(item[0]),
        ):
            entity_id = _safe_token(
                getattr(raw_state, "entity_id", raw_entity_id)
            )
            if not entity_id:
                continue
            known = _join_tokens(
                _safe_token(ref)
                for ref in getattr(raw_state, "known_refs", []) or []
            ) or "-"
            suspected = _join_tokens(
                _safe_token(ref)
                for ref in getattr(raw_state, "suspected_refs", []) or []
            ) or "-"
            facts = _join_tokens(
                _safe_token(fact_id)
                for fact_id in getattr(raw_state, "last_source_fact_ids", []) or []
            ) or "-"
            parts = [
                f"pack={pack_id}",
                f"entity={entity_id}",
                f"known={known}",
                f"suspected={suspected}",
                f"facts={facts}",
            ]
            notes = _safe_text(getattr(raw_state, "notes", ""))
            if notes:
                parts.append("notes=" + _quote_value(notes))
            rows.append(" ".join(parts))
    return "\n".join(rows) or "-"


def build_candidate_turn_entities_from_checkpoint(
    ckpt: Any,
) -> dict[str, dict[str, Any]]:
    """Build the compact candidate roster the content manager can hint against."""

    candidates: dict[str, dict[str, Any]] = {}
    for character in getattr(ckpt, "characters", []) or []:
        character_id = _safe_token(getattr(character, "character_id", ""))
        if not character_id:
            continue
        status = getattr(character, "status", "")
        status_value = getattr(status, "value", status)
        if status_value != "active":
            continue
        private_state = getattr(character, "private_state", None)
        public_sheet = getattr(character, "public_sheet", None)
        current_objectives = list(
            getattr(private_state, "current_objectives", []) or []
        )
        data: dict[str, Any] = {
            "name": getattr(character, "name", ""),
            "role": getattr(public_sheet, "role", ""),
            "location": getattr(character, "location", ""),
            "status": status_value,
        }
        if current_objectives:
            data["current_objective"] = current_objectives[:2]
        if bool(getattr(private_state, "intentions_enabled", False)):
            data["stance"] = "intentions_enabled"
        candidates[character_id] = data
    return candidates


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
    validated_knowledge_updates: list[ContentManagerKnowledgeUpdate] = []

    for update in output.knowledge_updates:
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
        validated_knowledge_updates.append(
            update.model_copy(update={"content_hash": card.content_hash})
        )

    validated_router_knowledge: list[ContentManagerRouterRequiredKnowledge] = []
    for item in output.router_required_knowledge:
        card = _runtime_card_or_none(
            ckpt,
            item.pack_id,
            item.ref,
            cards,
        )
        if card is None:
            errors.append(f"missing content pack={item.pack_id} ref={item.ref}")
            continue
        if item.content_hash and item.content_hash != card.content_hash:
            errors.append(
                "content hash mismatch "
                f"pack={item.pack_id} ref={item.ref} "
                f"expected={card.content_hash} actual={item.content_hash}"
            )
            continue
        validated_router_knowledge.append(
            item.model_copy(update={"content_hash": card.content_hash})
        )

    validated_candidates: list[ContentManagerRouterTurnCandidate] = []
    for candidate in output.router_turn_candidates:
        if candidate.character_id not in candidate_ids:
            errors.append(f"unknown character_id={candidate.character_id or '-'}")
            continue
        valid_refs = [
            ref for ref in candidate.related_content_refs
            if _runtime_card_from_compact_ref(ckpt, ref, cards) is not None
        ]
        validated_candidates.append(
            candidate.model_copy(update={"related_content_refs": valid_refs})
        )

    validated_broadcasts: list[ContentManagerAgentContextBroadcast] = []
    for broadcast in output.agent_context_broadcasts:
        if broadcast.character_id not in candidate_ids:
            errors.append(
                f"unknown broadcast character_id={broadcast.character_id or '-'}"
            )
            continue
        card = _runtime_card_or_none(
            ckpt,
            broadcast.pack_id,
            broadcast.ref,
            cards,
        )
        if card is None:
            errors.append(
                f"missing broadcast content pack={broadcast.pack_id} "
                f"ref={broadcast.ref}"
            )
            continue
        if broadcast.content_hash and broadcast.content_hash != card.content_hash:
            errors.append(
                "broadcast content hash mismatch "
                f"pack={broadcast.pack_id} ref={broadcast.ref} "
                f"expected={card.content_hash} actual={broadcast.content_hash}"
            )
            continue
        validated_broadcasts.append(
            broadcast.model_copy(update={"content_hash": card.content_hash})
        )

    if errors:
        raise ContentManagerValidationError("; ".join(errors))

    return output.model_copy(update={
        "knowledge_updates": _dedupe_knowledge_updates(
            validated_knowledge_updates
        ),
        "router_required_knowledge": _dedupe_router_required_knowledge(
            validated_router_knowledge
        ),
        "router_turn_candidates": _dedupe_turn_candidates(validated_candidates),
        "agent_context_broadcasts": _dedupe_agent_context_broadcasts(
            validated_broadcasts
        ),
    })


def format_content_manager_router_records(
    output: ContentManagerOutput,
) -> list[str]:
    records: list[str] = []
    for hint in output.router_turn_candidates:
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


def content_manager_required_lookup_requests(
    output: ContentManagerOutput,
) -> list[ContentLookupRequest]:
    return [
        ContentLookupRequest(
            pack_id=item.pack_id,
            ref=item.ref,
            reason=item.reason,
            urgency="required",
        )
        for item in output.router_required_knowledge
    ]


async def append_content_manager_router_records(
    ckpt: Any,
    *,
    actor_id: str,
    current_input: str,
    client: LLMClient,
    prompt_mgr: PromptManager,
    max_recent_facts: int = 12,
    max_catalog_cards_per_pack: int = 120,
) -> list[str]:
    """Run content-manager preflight and append only router-facing records."""

    if not should_run_content_manager_preflight(ckpt):
        logger.info("Skipping content-manager preflight on throttled cycle")
        return append_router_content_lookup_records(
            ckpt,
            actor_id=actor_id,
            current_input=current_input,
        )

    candidate_entities = build_candidate_turn_entities_from_checkpoint(ckpt)
    output = await plan_content_manager_updates(
        ckpt,
        candidate_entities=candidate_entities,
        client=client,
        prompt_mgr=prompt_mgr,
        max_recent_facts=max_recent_facts,
        max_catalog_cards_per_pack=max_catalog_cards_per_pack,
    )
    apply_content_manager_knowledge_updates(ckpt, output)

    records = append_router_content_lookup_records(
        ckpt,
        actor_id=actor_id,
        current_input=current_input,
        llm_requests=content_manager_required_lookup_requests(output),
    )
    hint_records = format_content_manager_router_records(output)
    conversation = getattr(ckpt, "session_conversation", None)
    if conversation is not None:
        for record in hint_records:
            conversation.append(ConversationMessage(role="assistant", content=record))
    records.extend(hint_records)
    return records


def _content_manager_refresh_interval(ckpt: Any) -> int:
    session = getattr(ckpt, "session", None)
    config = getattr(session, "config", None) if session is not None else None
    settings = getattr(config, "settings", None) if config is not None else None
    raw_value = getattr(
        settings,
        "content_manager_refresh_interval",
        _DEFAULT_CONTENT_MANAGER_REFRESH_INTERVAL,
    )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = _DEFAULT_CONTENT_MANAGER_REFRESH_INTERVAL
    return max(1, value)


def _has_recent_canonical_facts(ckpt: Any) -> bool:
    for event in getattr(ckpt, "canonical_events", []) or []:
        canonical = getattr(event, "canonical_event", None)
        if canonical is None and isinstance(event, Mapping):
            canonical = event.get("canonical_event")
        facts = getattr(canonical, "observable_facts", []) if canonical else []
        if isinstance(canonical, Mapping):
            facts = canonical.get("observable_facts") or []
        if any(_fact_text(fact) for fact in facts or []):
            return True
    return False


def apply_content_manager_knowledge_updates(
    ckpt: Any,
    output: ContentManagerOutput,
) -> None:
    pack_states = _content_pack_states_by_id(ckpt)
    for update in output.knowledge_updates:
        pack_state = pack_states.get(update.pack_id)
        if pack_state is None:
            continue
        knowledge_map = dict(getattr(pack_state, "knowledge_map", {}) or {})
        state = knowledge_map.get(update.entity_id)
        if not isinstance(state, ContentKnowledgeEntityState):
            state = ContentKnowledgeEntityState(entity_id=update.entity_id)
        compact_ref = update.compact_ref()
        known_refs = list(state.known_refs)
        suspected_refs = list(state.suspected_refs)

        if update.operation == "forget":
            known_refs = _remove_ref_base(known_refs, compact_ref)
            suspected_refs = _remove_ref_base(suspected_refs, compact_ref)
        elif update.operation == "mark_suspected":
            if not _contains_ref_base(known_refs, compact_ref):
                suspected_refs = _replace_ref_by_base(suspected_refs, compact_ref)
        else:
            known_refs = _replace_ref_by_base(known_refs, compact_ref)
            suspected_refs = _remove_ref_base(suspected_refs, compact_ref)

        knowledge_map[update.entity_id] = ContentKnowledgeEntityState(
            entity_id=update.entity_id,
            known_refs=known_refs,
            suspected_refs=suspected_refs,
            notes=update.reason or state.notes,
            last_source_fact_ids=update.source_fact_ids or state.last_source_fact_ids,
        )
        pack_state.knowledge_map = knowledge_map


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
    pack_id, ref, content_hash = _split_compact_ref(value)
    if not pack_id or not ref:
        return None
    card = _runtime_card_or_none(ckpt, pack_id, ref, cache)
    if card is None:
        return None
    if content_hash and content_hash != card.content_hash:
        return None
    return card


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


def _dedupe_knowledge_updates(
    updates: Iterable[ContentManagerKnowledgeUpdate],
) -> list[ContentManagerKnowledgeUpdate]:
    deduped: dict[tuple[str, str, str, str], ContentManagerKnowledgeUpdate] = {}
    for update in updates:
        deduped.setdefault(update.dedupe_key(), update)
    return list(deduped.values())


def _dedupe_router_required_knowledge(
    updates: Iterable[ContentManagerRouterRequiredKnowledge],
) -> list[ContentManagerRouterRequiredKnowledge]:
    deduped: dict[tuple[str, str], ContentManagerRouterRequiredKnowledge] = {}
    for update in updates:
        deduped.setdefault(update.dedupe_key(), update)
    return list(deduped.values())


def _dedupe_turn_candidates(
    candidates: Iterable[ContentManagerRouterTurnCandidate],
) -> list[ContentManagerRouterTurnCandidate]:
    deduped: dict[str, ContentManagerRouterTurnCandidate] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.dedupe_key(), candidate)
    return list(deduped.values())


def _dedupe_agent_context_broadcasts(
    broadcasts: Iterable[ContentManagerAgentContextBroadcast],
) -> list[ContentManagerAgentContextBroadcast]:
    deduped: dict[tuple[str, str, str], ContentManagerAgentContextBroadcast] = {}
    for broadcast in broadcasts:
        deduped.setdefault(broadcast.dedupe_key(), broadcast)
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


def _split_compact_ref(value: str) -> tuple[str, str, str]:
    text = _safe_token(value)
    if ":" not in text:
        return "", "", ""
    pack_id, ref_with_hash = text.split(":", 1)
    ref = ref_with_hash
    content_hash = ""
    if "@" in ref_with_hash:
        ref, content_hash = ref_with_hash.rsplit("@", 1)
    return _safe_token(pack_id), _safe_token(ref), _safe_token(content_hash)


def _ref_base(value: str) -> str:
    compact = _safe_token(value)
    return compact.rsplit("@", 1)[0] if "@" in compact else compact


def _contains_ref_base(values: Iterable[str], compact_ref: str) -> bool:
    base = _ref_base(compact_ref)
    return any(_ref_base(value) == base for value in values)


def _remove_ref_base(values: Iterable[str], compact_ref: str) -> list[str]:
    base = _ref_base(compact_ref)
    return [
        value
        for value in values
        if _safe_token(value) and _ref_base(value) != base
    ]


def _replace_ref_by_base(values: Iterable[str], compact_ref: str) -> list[str]:
    cleaned_ref = _safe_token(compact_ref)
    if not cleaned_ref:
        return [
            value
            for value in values
            if _safe_token(value)
        ]
    return [*_remove_ref_base(values, cleaned_ref), cleaned_ref]


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
