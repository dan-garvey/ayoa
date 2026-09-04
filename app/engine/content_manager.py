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
    ContentManagerRouterRequiredKey,
    ContentManagerRouterTurnCandidate,
)
from app.schemas.content_privacy import contains_imported_asset_sentinel


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")
CONTENT_MANAGER_MAX_TOKENS = 16000
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
    """Return true when a checkpoint carries content-manager dispatch state."""

    for pack_state in _content_pack_states_by_id(ckpt).values():
        if _router_knowledge_index_entries(pack_state):
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
    current_input: str = "",
    max_recent_facts: int = 12,
    max_dispatch_keys_per_pack: int = 40,
) -> ContentManagerOutput:
    """Ask for knowledge-map patches and router deltas, then validate them."""

    resolved_candidates = (
        build_candidate_turn_entities_from_checkpoint(ckpt)
        if candidate_entities is None
        else candidate_entities
    )
    dispatch_index_block = build_content_manager_dispatch_index_block(
        ckpt,
        candidate_entities=resolved_candidates,
        current_input=current_input,
        max_keys_per_pack=max_dispatch_keys_per_pack,
    )
    candidate_entities_block = build_candidate_turn_entities_block(
        resolved_candidates,
    )
    if not dispatch_index_block:
        return ContentManagerOutput(
            knowledge_updates=[],
            router_required_keys=[],
            router_turn_candidates=[],
            agent_context_broadcasts=[],
            no_update_reason="No reviewed router knowledge dispatch index is available.",
        )

    messages = build_content_manager_messages(
        ckpt,
        candidate_entities=resolved_candidates,
        prompt_mgr=prompt_mgr,
        dispatch_index_block=dispatch_index_block,
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
    dispatch_index_block: str | None = None,
    candidate_entities_block: str | None = None,
    current_input: str = "",
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
        router_knowledge_state_block=build_router_knowledge_state_block(ckpt),
        candidate_entities_block=(
            candidate_entities_block
            if candidate_entities_block is not None
            else build_candidate_turn_entities_block(resolved_candidates)
        ),
        router_knowledge_dispatch_index_block=(
            dispatch_index_block
            if dispatch_index_block is not None
            else build_content_manager_dispatch_index_block(
                ckpt,
                candidate_entities=resolved_candidates,
                current_input=current_input,
            )
        ),
    )


def build_content_manager_dispatch_index_block(
    ckpt: Any,
    *,
    candidate_entities: Mapping[str, Any] | Sequence[str] | None = None,
    current_input: str = "",
    max_keys_per_pack: int = 40,
) -> str:
    return _build_router_knowledge_dispatch_index_block(
        ckpt,
        candidate_entities=(
            build_candidate_turn_entities_from_checkpoint(ckpt)
            if candidate_entities is None
            else candidate_entities
        ),
        current_input=current_input,
        max_keys_per_pack=max_keys_per_pack,
    )


def build_recent_canonical_facts_block(ckpt: Any, *, limit: int = 12) -> str:
    rows: list[tuple[str, str]] = []
    for event in getattr(ckpt, "canonical_events", []) or []:
        facts = (
            event.get("observable_facts", [])
            if isinstance(event, Mapping)
            else getattr(event, "observable_facts", [])
        )
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


def build_router_knowledge_state_block(ckpt: Any) -> str:
    rows: list[str] = []
    for pack_key, pack_state in _content_pack_states_by_id(ckpt).items():
        pack_id = _safe_token(_pack_id(pack_key, pack_state))
        if not pack_id:
            continue
        active_fronts = _active_front_refs(pack_state)
        if active_fronts:
            rows.append(
                f"pack={pack_id} active_fronts={_join_tokens(active_fronts)}"
            )
    for key, status in sorted(_router_knowledge_key_statuses(ckpt).items()):
        if status["status"] not in {"known", "partial"}:
            continue
        rows.append(
            " ".join(
                part
                for part in (
                    f"key={_safe_token(key)}",
                    f"pack={_safe_token(status['pack_id'])}",
                    f"status={_safe_token(status['status'])}",
                    f"introduced={status['introduced']}/{status['total']}",
                )
                if part and not part.endswith("=")
            )
        )
    return "\n".join(rows) or "-"


def _build_router_knowledge_dispatch_index_block(
    ckpt: Any,
    *,
    candidate_entities: Mapping[str, Any] | Sequence[str],
    current_input: str,
    max_keys_per_pack: int,
) -> str:
    scope = _dispatch_scope(ckpt, candidate_entities, current_input)
    statuses = _router_knowledge_key_statuses(ckpt)
    rows: list[str] = []
    for pack_key, pack_state in _content_pack_states_by_id(ckpt).items():
        pack_id = _safe_token(_pack_id(pack_key, pack_state))
        if not pack_id:
            continue
        db_path = _pack_db_path(pack_state)
        if db_path is not None:
            _assert_pack_runtime_identity(
                db_path,
                pack_id=pack_id,
                pack_state=pack_state,
            )
        entries = _router_knowledge_index_entries(pack_state)
        ranked: list[tuple[int, str, Mapping[str, Any]]] = []
        for entry in entries:
            key = _safe_token(entry.get("key"))
            if not key:
                continue
            if statuses.get(key, {}).get("status") == "known":
                continue
            score = _dispatch_entry_score(entry, scope)
            if score < 50:
                continue
            ranked.append((score, key, entry))
        ranked.sort(key=lambda item: (-item[0], -int(item[2].get("priority") or 0), item[1]))
        for _score, key, entry in ranked[: max(0, int(max_keys_per_pack))]:
            rows.append(_format_dispatch_index_row(pack_id, key, entry, statuses.get(key)))
    return "\n".join(row for row in rows if row)


def _dispatch_scope(
    ckpt: Any,
    candidate_entities: Mapping[str, Any] | Sequence[str],
    current_input: str,
) -> dict[str, Any]:
    candidate_ids = _candidate_entity_ids(candidate_entities)
    active_front_refs: set[str] = set()
    text_parts = [_safe_text(current_input)]
    for _pack_key, pack_state in _content_pack_states_by_id(ckpt).items():
        active_front_refs.update(_active_front_refs(pack_state))
    for raw_entity_id, raw_value in _iter_candidate_entities(candidate_entities):
        if entity_id := _safe_token(raw_entity_id):
            text_parts.append(entity_id)
        if isinstance(raw_value, Mapping):
            for value in raw_value.values():
                text_parts.append(_format_value(value))
    for event in getattr(ckpt, "canonical_events", [])[-12:]:
        facts = (
            event.get("observable_facts", [])
            if isinstance(event, Mapping)
            else getattr(event, "observable_facts", [])
        )
        for fact in facts or []:
            text_parts.append(_fact_text(fact))
    return {
        "candidate_ids": candidate_ids,
        "active_front_refs": active_front_refs,
        "text": " ".join(part for part in text_parts if part).lower(),
    }


def _dispatch_entry_score(entry: Mapping[str, Any], scope: Mapping[str, Any]) -> int:
    priority = int(entry.get("priority") or 0)
    score = 100 if priority >= 100 else 0
    facets = entry.get("scope_facets")
    if not isinstance(facets, Mapping):
        facets = {}
    candidate_ids = set(scope.get("candidate_ids") or ())
    active_fronts = set(scope.get("active_front_refs") or ())
    text = str(scope.get("text") or "")

    if set(_facet_values(facets, "front_refs")).intersection(active_fronts):
        score += 25
    if set(_facet_values(facets, "character_ids")).intersection(candidate_ids):
        score += 80
    for field_name, weight in (
        ("location_refs", 60),
        ("actor_refs", 45),
        ("phase_tags", 35),
        ("region_tags", 35),
    ):
        if any(_scope_token_matches_text(value, text) for value in _facet_values(facets, field_name)):
            score += weight
            break
    if any(_hint_matches_text(hint, text) for hint in _list_values(entry.get("activation_hints"))):
        score += 30
    return score


def _format_dispatch_index_row(
    pack_id: str,
    key: str,
    entry: Mapping[str, Any],
    status: Mapping[str, Any] | None,
) -> str:
    parts = [
        f"key={_safe_token(key)}",
        f"kind={_safe_token(entry.get('kind'))}",
    ]
    if label := _safe_text(entry.get("label")):
        parts.append("label=" + _quote_value(label))
    if summary := _safe_text(entry.get("summary")):
        parts.append("summary=" + _quote_value(summary))
    scope = _format_scope_facets(entry.get("scope_facets"))
    if scope:
        parts.append("scope=" + _quote_value(scope))
    hints = [
        hint for hint in (_safe_text(value) for value in _list_values(entry.get("activation_hints")))
        if hint
    ][:5]
    if hints:
        parts.append("hints=" + _quote_value("; ".join(hints)))
    if priority := int(entry.get("priority") or 0):
        parts.append(f"priority={priority}")
    packet_count = int(entry.get("packet_count") or 0)
    if packet_count:
        parts.append(f"packets={packet_count}")
    kinds = _join_tokens(_list_values(entry.get("packet_kinds")))
    if kinds:
        parts.append(f"packet_kinds={kinds}")
    if status is not None and status.get("status"):
        parts.append(f"status={_safe_token(status.get('status'))}")
    parts.append(f"pack={_safe_token(pack_id)}")
    return " ".join(part for part in parts if part and not part.endswith("="))


def _format_scope_facets(raw: Any) -> str:
    if not isinstance(raw, Mapping):
        return ""
    parts: list[str] = []
    labels = {
        "location_refs": "locations",
        "front_refs": "fronts",
        "actor_refs": "actors",
        "character_ids": "characters",
        "phase_tags": "phases",
        "region_tags": "regions",
    }
    for key, label in labels.items():
        values = _join_tokens(_facet_values(raw, key))
        if values:
            parts.append(f"{label}:{values}")
    return " ".join(parts)


def _facet_values(facets: Mapping[str, Any], key: str) -> list[str]:
    return [
        token for token in (_safe_token(value) for value in _list_values(facets.get(key)))
        if token
    ]


def _list_values(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return list(raw)
    return [raw]


def _scope_token_matches_text(value: str, text: str) -> bool:
    token = _safe_token(value).lower()
    if not token:
        return False
    candidates = {
        token,
        token.rsplit(".", 1)[-1],
        token.rsplit("/", 1)[-1],
        token.replace(".", " "),
        token.replace("_", " "),
    }
    return any(candidate and candidate in text for candidate in candidates)


def _hint_matches_text(hint: str, text: str) -> bool:
    words = [
        word.strip(".,;:!?()[]{}\"'").lower()
        for word in str(hint or "").split()
    ]
    return any(len(word) >= 4 and word in text for word in words)


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
                _prompt_ref_token(ref)
                for ref in getattr(raw_state, "known_refs", []) or []
            ) or "-"
            suspected = _join_tokens(
                _prompt_ref_token(ref)
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
        if _character_is_unresolved_combat_spawn(ckpt, character, character_id):
            continue
        status = getattr(character, "status", "")
        status_value = getattr(status, "value", status)
        if status_value != "active":
            continue
        public_sheet = getattr(character, "public_sheet", None)
        data: dict[str, Any] = {
            "name": getattr(character, "name", ""),
            "role": getattr(public_sheet, "role", ""),
            "location": getattr(character, "location", ""),
            "status": status_value,
        }
        candidates[character_id] = data
    return candidates


def _character_is_unresolved_combat_spawn(
    ckpt: Any,
    character: Any,
    character_id: str,
) -> bool:
    if not _character_has_combat_spawn_marker(character):
        return False
    session = getattr(ckpt, "session", None)
    combat = getattr(session, "active_combat", None)
    if combat is None:
        return False
    for combatant in getattr(combat, "combatants", []) or []:
        if character_id in _combatant_identity_set(combatant):
            return True
    return False


def _character_has_combat_spawn_marker(character: Any) -> bool:
    mechanics = getattr(character, "mechanics", None)
    if not isinstance(mechanics, Mapping):
        return False
    marker = mechanics.get("combat_spawn")
    if isinstance(marker, Mapping) and bool(marker.get("spawned")):
        return True
    return str(mechanics.get("source") or "").strip() == "router_combatant_spawn"


def _combatant_identity_set(combatant: Any) -> set[str]:
    return {
        text
        for text in (
            str(getattr(combatant, "combatant_id", "") or "").strip(),
            str(getattr(combatant, "character_id", "") or "").strip(),
            str(getattr(combatant, "name", "") or "").strip(),
        )
        if text
    }


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

    packet_index = _router_knowledge_packet_index(ckpt)
    validated_required_keys: list[ContentManagerRouterRequiredKey] = []
    for item in output.router_required_keys:
        packet = packet_index.get(item.key)
        if packet is None:
            errors.append(f"unknown router knowledge key={item.key or '-'}")
            continue
        packet_errors = _validate_router_knowledge_packet_refs(
            ckpt,
            item.key,
            packet,
            cards,
        )
        if packet_errors:
            errors.extend(packet_errors)
            continue
        validated_required_keys.append(item)

    validated_candidates: list[ContentManagerRouterTurnCandidate] = []
    for candidate in output.router_turn_candidates:
        if candidate.character_id not in candidate_ids:
            errors.append(f"unknown character_id={candidate.character_id or '-'}")
            continue
        valid_refs = [
            ref for ref in candidate.related_content_refs
            if _runtime_card_from_compact_ref(ckpt, ref, cards) is not None
        ]
        valid_keys = [
            key for key in candidate.related_keys
            if key in packet_index
        ]
        validated_candidates.append(
            candidate.model_copy(update={
                "related_content_refs": valid_refs,
                "related_keys": valid_keys,
            })
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
        "router_required_keys": _dedupe_router_required_keys(
            validated_required_keys
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
            "scope=attention_hint",
            f"character={_safe_token(hint.character_id)}",
            f"priority={_safe_token(hint.priority)}",
        ]
        if hint.related_content_refs:
            parts.append(f"refs={_join_tokens(hint.related_content_refs)}")
        if hint.related_keys:
            parts.append(f"keys={_join_tokens(hint.related_keys)}")
        if hint.source_fact_ids:
            parts.append(f"facts={_join_tokens(hint.source_fact_ids)}")
        if hint.reason:
            parts.append(f"reason={_quote_value(hint.reason)}")
        records.append(
            " ".join(part for part in parts if part and not part.endswith("="))
        )

    return records


def content_manager_required_lookup_requests(
    ckpt: Any,
    output: ContentManagerOutput,
) -> list[ContentLookupRequest]:
    packet_index = _router_knowledge_packet_index(ckpt)
    requests: list[ContentLookupRequest] = []
    for item in output.router_required_keys:
        packet = packet_index.get(item.key)
        if packet is None:
            raise ContentManagerValidationError(
                f"unknown router knowledge key={item.key or '-'}"
            )
        for packet_ref in packet["packet_refs"]:
            pack_id = _safe_token(packet_ref.get("pack_id")) or packet["pack_id"]
            ref = _safe_token(packet_ref.get("ref"))
            if not pack_id or not ref:
                continue
            requests.append(
                ContentLookupRequest(
                    pack_id=pack_id,
                    ref=ref,
                    reason=item.reason or item.key,
                    urgency="required",
                )
            )
    return requests


async def append_content_manager_router_records(
    ckpt: Any,
    *,
    actor_id: str,
    current_input: str,
    client: LLMClient,
    prompt_mgr: PromptManager,
    max_recent_facts: int = 12,
    max_dispatch_keys_per_pack: int = 40,
) -> list[str]:
    """Run content-manager preflight and append only router-facing records."""

    if _checkpoint_has_active_combat(ckpt):
        logger.info("Skipping content-manager preflight during active combat")
        return append_router_content_lookup_records(
            ckpt,
            actor_id=actor_id,
            current_input=current_input,
        )

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
        current_input=current_input,
        client=client,
        prompt_mgr=prompt_mgr,
        max_recent_facts=max_recent_facts,
        max_dispatch_keys_per_pack=max_dispatch_keys_per_pack,
    )
    apply_content_manager_knowledge_updates(ckpt, output)

    records = append_router_content_lookup_records(
        ckpt,
        actor_id=actor_id,
        current_input=current_input,
        llm_requests=content_manager_required_lookup_requests(ckpt, output),
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
        facts = (
            event.get("observable_facts", [])
            if isinstance(event, Mapping)
            else getattr(event, "observable_facts", [])
        )
        if any(_fact_text(fact) for fact in facts or []):
            return True
    return False


def _checkpoint_has_active_combat(ckpt: Any) -> bool:
    session = getattr(ckpt, "session", None)
    return getattr(session, "active_combat", None) is not None


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


def _router_knowledge_index_entries(pack_state: Any) -> list[Mapping[str, Any]]:
    raw = _pack_metadata(pack_state).get("router_knowledge_index")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _router_knowledge_packet_index(ckpt: Any) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    duplicate_keys: set[str] = set()
    for pack_key, pack_state in _content_pack_states_by_id(ckpt).items():
        pack_id = _safe_token(_pack_id(pack_key, pack_state))
        if not pack_id:
            continue
        raw = _pack_metadata(pack_state).get("router_knowledge_packets")
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            key = _safe_token(item.get("key"))
            packet_refs = item.get("packet_refs")
            if not key or not isinstance(packet_refs, list):
                continue
            if key in index:
                duplicate_keys.add(key)
                continue
            refs = [
                ref for ref in packet_refs
                if isinstance(ref, Mapping) and _safe_token(ref.get("ref"))
            ]
            if refs:
                index[key] = {
                    "pack_id": pack_id,
                    "packet_refs": refs,
                }
    if duplicate_keys:
        raise ContentManagerValidationError(
            "duplicate router knowledge key(s): " + ", ".join(sorted(duplicate_keys))
        )
    return index


def _router_knowledge_key_statuses(ckpt: Any) -> dict[str, dict[str, Any]]:
    known_refs = _known_router_content_refs(ckpt)
    statuses: dict[str, dict[str, Any]] = {}
    for key, packet in _router_knowledge_packet_index(ckpt).items():
        refs = [
            (
                _safe_token(ref.get("pack_id")) or packet["pack_id"],
                _safe_token(ref.get("ref")),
            )
            for ref in packet["packet_refs"]
            if _safe_token(ref.get("ref"))
        ]
        total = len(refs)
        introduced = sum(1 for ref in refs if ref in known_refs)
        status = "missing"
        if total and introduced == total:
            status = "known"
        elif introduced:
            status = "partial"
        statuses[key] = {
            "pack_id": packet["pack_id"],
            "status": status,
            "introduced": introduced,
            "total": total,
        }
    return statuses


def _validate_router_knowledge_packet_refs(
    ckpt: Any,
    key: str,
    packet: Mapping[str, Any],
    cache: dict[tuple[str, str], ContentCard | None],
) -> list[str]:
    errors: list[str] = []
    packet_refs = packet.get("packet_refs")
    if not isinstance(packet_refs, list) or not packet_refs:
        return [f"router knowledge key={key} has no packet refs"]
    default_pack_id = _safe_token(packet.get("pack_id"))
    for raw_ref in packet_refs:
        if not isinstance(raw_ref, Mapping):
            errors.append(f"router knowledge key={key} has malformed packet ref")
            continue
        pack_id = _safe_token(raw_ref.get("pack_id")) or default_pack_id
        ref = _safe_token(raw_ref.get("ref"))
        expected_hash = _safe_token(raw_ref.get("content_hash"))
        if not pack_id or not ref:
            errors.append(f"router knowledge key={key} has blank packet ref")
            continue
        card = _runtime_card_or_none(ckpt, pack_id, ref, cache)
        if card is None:
            errors.append(f"missing content key={key} pack={pack_id} ref={ref}")
            continue
        if expected_hash and expected_hash != card.content_hash:
            errors.append(
                "router knowledge hash mismatch "
                f"key={key} pack={pack_id} ref={ref} "
                f"expected={card.content_hash} actual={expected_hash}"
            )
    return errors


def _active_front_refs(pack_state: Any) -> set[str]:
    refs: set[str] = set()
    fronts = getattr(pack_state, "fronts", None)
    if isinstance(fronts, Mapping):
        for raw_front_id, raw_state in fronts.items():
            front_id = _safe_token(getattr(raw_state, "front_id", raw_front_id))
            status = _safe_token(getattr(raw_state, "status", "")) or "active"
            if front_id and status not in {"resolved", "abandoned"}:
                refs.add(front_id)
    metadata_refs = _pack_metadata(pack_state).get("active_front_refs")
    if isinstance(metadata_refs, list):
        refs.update(_safe_token(ref) for ref in metadata_refs if _safe_token(ref))
    return refs


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


def _pack_metadata(pack_state: Any) -> Mapping[str, Any]:
    metadata = (
        pack_state.get("metadata")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "metadata", None)
    )
    return metadata if isinstance(metadata, Mapping) else {}


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


def _dedupe_router_required_keys(
    updates: Iterable[ContentManagerRouterRequiredKey],
) -> list[ContentManagerRouterRequiredKey]:
    deduped: dict[str, ContentManagerRouterRequiredKey] = {}
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


def _prompt_ref_token(value: Any) -> str:
    """Project persisted pack:ref@hash tokens into prompt-facing pack:ref tokens."""

    token = _safe_token(value)
    return _ref_base(token) if token else ""


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
