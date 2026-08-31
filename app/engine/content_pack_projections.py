from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.engine.content_pack_compiler import SCHEMA_VERSION as CONTENT_PACK_SCHEMA_VERSION
from app.schemas.characters import (
    ActorRecord,
    CharacterAgentTier,
    CharacterRecord,
    CharacterStatus,
    PublicSheet,
)
from app.schemas.content import (
    ContentFrontState,
    ContentKnowledgeEntityState,
    ContentPackState,
    PendingContentSignal,
)
from app.schemas.content_pack import (
    ActorDossierRecord,
    ContentPackDomainCatalog,
    ContentPackDomainRecord,
)
from app.schemas.content_privacy import redact_imported_content_metadata_text
from app.schemas.content_projection import (
    CONTENT_PACK_PROJECTION_SCHEMA_VERSION,
    ContentCharacterPatchProjection,
    ContentCharacterProjection,
    ContentCheckpointProjection,
    ContentEngineOverlayProjection,
    ContentFieldStartProjection,
    ContentFrontProjection,
    ContentKnowledgeProjection,
    ContentPackProjectionArtifact,
    ContentProjectionRef,
    ContentRouterKnowledgeKeyProjection,
    ContentRouterProjection,
)


APPROVED_REVIEW_STATUSES = frozenset(("approved", "reviewed"))
RUNTIME_GATE_STATUS = "runtime_ready"


class ContentProjectionBuildError(ValueError):
    """Raised when import-time projection inputs are not runtime-authoritative."""


def build_content_pack_projection_artifact(
    catalog: ContentPackDomainCatalog | Mapping[str, Any],
    *,
    runtime_cards: Sequence[Any],
    initial_router_lookup_refs: Sequence[str],
    field_start_router_lookup_refs: Sequence[str] = (),
    active_front_refs: Sequence[str] = (),
    active_character_ids: Sequence[str] = (),
    character_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    checkpoint: ContentCheckpointProjection | Mapping[str, Any] | None = None,
    field_start: ContentFieldStartProjection | Mapping[str, Any] | None = None,
    router_knowledge_keys: Sequence[
        ContentRouterKnowledgeKeyProjection | Mapping[str, Any]
    ] = (),
    content_pack_schema_version: str = CONTENT_PACK_SCHEMA_VERSION,
    engine_overlay_notes: str = "",
) -> ContentPackProjectionArtifact:
    """Build the durable projection that runtime seeds should consume.

    The reviewed catalog remains the import-time authority. This function
    validates every projected runtime ref against both the typed catalog and the
    compiled runtime-card view, then emits compact packets for router lookup,
    character-agent initialization, and the engine-owned knowledge map.
    """

    domain_catalog = (
        catalog
        if isinstance(catalog, ContentPackDomainCatalog)
        else ContentPackDomainCatalog.model_validate(catalog)
    )
    records_by_ref = _records_by_ref(domain_catalog)
    cards_by_ref = _cards_by_ref(runtime_cards)
    _validate_runtime_cards(cards_by_ref, records_by_ref)

    active_ids = {item for item in (_clean_token(value) for value in active_character_ids) if item}
    overrides = _character_overrides(character_overrides)

    characters: list[ContentCharacterProjection] = []
    knowledge_map: list[ContentKnowledgeProjection] = []
    for actor in domain_catalog.actor_dossiers:
        _assert_runtime_record(actor)
        character_id = actor.character_id_hint or actor.ref.replace(".", "_")
        projection_refs = [
            _projection_ref(cards_by_ref, ref)
            for ref in _refs_for_actor(actor)
        ]
        known_refs = [ref.compact() for ref in projection_refs]
        character_projection = _character_projection(
            actor,
            known_refs=known_refs,
            character_id=character_id,
            active_character_ids=active_ids,
            overrides=overrides.get(character_id, {}),
        )
        characters.append(character_projection)
        knowledge_map.append(
            ContentKnowledgeProjection(
                entity_id=character_id,
                known_refs=known_refs,
                notes="Reviewed module knowledge map seed.",
            )
        )

    active_fronts: list[ContentFrontProjection] = []
    for ref in _unique_refs(active_front_refs):
        front_ref = _projection_ref(cards_by_ref, ref)
        active_fronts.append(
            ContentFrontProjection(
                front_id=ref,
                label=front_ref.title or ref,
                status="active",
                notes=front_ref.summary,
                introduced_ref_keys=[],
            )
        )

    initial_lookup_refs = [
        _projection_ref(cards_by_ref, ref)
        for ref in _unique_refs(initial_router_lookup_refs)
    ]
    field_start_lookup_refs = [
        _projection_ref(cards_by_ref, ref)
        for ref in _unique_refs(field_start_router_lookup_refs)
    ]
    router = ContentRouterProjection(
        initial_lookup_refs=initial_lookup_refs,
        field_start_lookup_refs=field_start_lookup_refs,
        lookup_catalog=[
            _projection_ref(cards_by_ref, ref)
            for ref in sorted(cards_by_ref)
        ],
        knowledge_keys=_coerce_router_knowledge_keys(
            router_knowledge_keys,
            cards_by_ref=cards_by_ref,
            pack_id=domain_catalog.pack_id,
            initial_refs=initial_lookup_refs,
            field_start_refs=field_start_lookup_refs,
            active_front_refs=active_front_refs,
        ),
    )
    field_projection = _coerce_field_start(
        field_start,
        cards_by_ref=cards_by_ref,
        fallback_router_refs=router.field_start_lookup_refs,
    )
    checkpoint_projection = _coerce_checkpoint(checkpoint)

    artifact = ContentPackProjectionArtifact(
        pack_id=domain_catalog.pack_id,
        pack_version=domain_catalog.pack_version,
        source_fingerprint=domain_catalog.source_fingerprint,
        schema_version=CONTENT_PACK_PROJECTION_SCHEMA_VERSION,
        content_pack_schema_version=content_pack_schema_version,
        catalog_schema_version=domain_catalog.schema_version,
        catalog_build_hash=domain_catalog.build_hash,
        router=router,
        characters=characters,
        knowledge_map=knowledge_map,
        fronts=active_fronts,
        checkpoint=checkpoint_projection,
        field_start=field_projection,
        engine_overlay=ContentEngineOverlayProjection(
            domain_catalog=domain_catalog.model_dump(mode="json"),
            domain_groups={
                group: len(getattr(domain_catalog, group))
                for group in _catalog_group_names()
            },
            notes=engine_overlay_notes,
        ),
    )
    return artifact.model_copy(update={"projection_hash": _projection_hash(artifact)})


def content_pack_state_from_projection(
    artifact: ContentPackProjectionArtifact | Mapping[str, Any],
    *,
    db_path: str = "",
    start_mode: str = "startup",
) -> dict[str, ContentPackState]:
    """Create checkpoint content_state from a prepared projection artifact."""

    projection = (
        artifact
        if isinstance(artifact, ContentPackProjectionArtifact)
        else ContentPackProjectionArtifact.model_validate(artifact)
    )
    pending = _pending_signals(
        projection.pack_id,
        projection.router.initial_lookup_refs,
        reason="reviewed pack startup router context",
        signal_prefix="startup",
        priority_start=10,
    )
    if start_mode == "field":
        pending.update(
            _pending_signals(
                projection.pack_id,
                projection.field_start.router_lookup_refs,
                reason="field-start reviewed module context",
                signal_prefix="field",
                priority_start=30,
            )
        )
    knowledge_map = {
        item.entity_id: ContentKnowledgeEntityState(
            entity_id=item.entity_id,
            known_refs=list(item.known_refs),
            suspected_refs=list(item.suspected_refs),
            notes=item.notes,
        )
        for item in projection.knowledge_map
    }
    if start_mode == "field":
        for grant in projection.field_start.knowledge_grants:
            current = knowledge_map.get(grant.entity_id)
            known_refs = list(current.known_refs) if current is not None else []
            for ref in grant.known_refs:
                if ref not in known_refs:
                    known_refs.append(ref)
            knowledge_map[grant.entity_id] = ContentKnowledgeEntityState(
                entity_id=grant.entity_id,
                known_refs=known_refs,
                suspected_refs=list(grant.suspected_refs),
                notes=grant.notes,
            )

    metadata: dict[str, Any] = {
        "pack_version": projection.pack_version,
        "source_fingerprint": projection.source_fingerprint,
        "schema_version": projection.content_pack_schema_version
        or CONTENT_PACK_SCHEMA_VERSION,
        "projection_schema_version": projection.schema_version,
        "projection_hash": projection.projection_hash,
        "catalog_build_hash": projection.catalog_build_hash,
        "active_front_refs": [front.front_id for front in projection.fronts],
        "catalog": projection.router.router_catalog_metadata(),
        "router_lookup_catalog": projection.router.router_catalog_metadata(),
        "router_knowledge_index": projection.router.router_knowledge_index_metadata(),
        "router_knowledge_packets": projection.router.router_knowledge_packet_metadata(),
        "engine_overlay": {
            "domain_groups": dict(projection.engine_overlay.domain_groups),
            "notes": projection.engine_overlay.notes,
        },
    }
    if db_path:
        metadata["db_path"] = db_path
    if projection.engine_overlay.domain_catalog:
        metadata["domain_catalog"] = projection.engine_overlay.domain_catalog

    return {
        projection.pack_id: ContentPackState(
            pack_id=projection.pack_id,
            pending_signals=pending,
            fronts={
                front.front_id: ContentFrontState(
                    front_id=front.front_id,
                    label=front.label,
                    status=front.status,
                    notes=front.notes,
                    introduced_ref_keys=list(front.introduced_ref_keys),
                )
                for front in projection.fronts
            },
            knowledge_map=knowledge_map,
            metadata=metadata,
        )
    }


def character_record_from_projection(
    projection: ContentCharacterProjection | Mapping[str, Any],
    *,
    mechanics: Mapping[str, Any] | None = None,
    agent_tier: CharacterAgentTier = CharacterAgentTier.standard,
) -> CharacterRecord:
    """Create a runtime CharacterRecord from an import-authored projection."""

    character = (
        projection
        if isinstance(projection, ContentCharacterProjection)
        else ContentCharacterProjection.model_validate(projection)
    )
    return CharacterRecord(
        character_id=character.character_id,
        name=character.name,
        status=CharacterStatus(character.status),
        location=character.location,
        is_playable=False,
        agent_tier=agent_tier,
        public_sheet=PublicSheet(
            role=character.public_role,
            appearance=character.appearance,
            faction=character.faction,
            public_context=character.public_context,
        ),
        actor=(
            character.actor.model_copy(deep=True)
            if character.actor is not None
            else None
        ),
        mechanics=dict(mechanics or {}),
    )


def apply_checkpoint_projection(
    ckpt: Any,
    projection: ContentCheckpointProjection | Mapping[str, Any],
) -> None:
    seed = (
        projection
        if isinstance(projection, ContentCheckpointProjection)
        else ContentCheckpointProjection.model_validate(projection)
    )
    if seed.player_primer:
        ckpt.player_primer = seed.player_primer
    if seed.world_facts:
        ckpt.world_state.facts = list(seed.world_facts)
    if seed.narrative_rules:
        ckpt.session.config.narrative_rules = seed.narrative_rules


def apply_field_start_projection(
    ckpt: Any,
    artifact: ContentPackProjectionArtifact | Mapping[str, Any],
) -> None:
    projection = (
        artifact
        if isinstance(artifact, ContentPackProjectionArtifact)
        else ContentPackProjectionArtifact.model_validate(artifact)
    )
    field = projection.field_start
    active_ids = set(field.active_character_ids)
    patches = {patch.character_id: patch for patch in field.character_patches}
    for character in ckpt.characters:
        patch = patches.get(character.character_id)
        if character.character_id in active_ids:
            character.status = CharacterStatus.active
        elif patch is not None and patch.status is not None:
            character.status = CharacterStatus(patch.status)
        else:
            character.status = CharacterStatus.dormant
        if patch is not None:
            _apply_character_patch(character, patch)
        elif field.location_ref and character.character_id in active_ids:
            character.location = field.location_ref
    apply_checkpoint_projection(ckpt, field.checkpoint)


def _pending_signals(
    pack_id: str,
    refs: Sequence[ContentProjectionRef],
    *,
    reason: str,
    signal_prefix: str,
    priority_start: int,
) -> dict[str, PendingContentSignal]:
    pending: dict[str, PendingContentSignal] = {}
    for index, ref in enumerate(refs, start=1):
        signal_id = f"{signal_prefix}_{index:02d}"
        pending[signal_id] = PendingContentSignal(
            signal_id=signal_id,
            pack_id=pack_id,
            ref_id=ref.ref,
            content_hash=ref.content_hash,
            reason=reason,
            priority=max(1, priority_start - index + 1),
            requested_fields=["summary"],
            metadata={
                "kind": ref.kind,
                "visibility": ref.visibility,
                "summary": ref.summary,
            },
        )
    return pending


def _apply_character_patch(
    character: CharacterRecord,
    patch: ContentCharacterPatchProjection,
) -> None:
    if patch.status is not None:
        character.status = CharacterStatus(patch.status)
    if patch.location:
        character.location = patch.location


def _records_by_ref(
    catalog: ContentPackDomainCatalog,
) -> dict[str, ContentPackDomainRecord]:
    return {record.ref: record for record in catalog._domain_records()}


def _cards_by_ref(runtime_cards: Sequence[Any]) -> dict[str, Any]:
    cards: dict[str, Any] = {}
    for card in runtime_cards:
        ref = _value(card, "ref")
        if not ref:
            raise ContentProjectionBuildError("runtime card without ref")
        if ref in cards:
            raise ContentProjectionBuildError(f"duplicate runtime card ref: {ref}")
        cards[ref] = card
    return cards


def _validate_runtime_cards(
    cards_by_ref: Mapping[str, Any],
    records_by_ref: Mapping[str, ContentPackDomainRecord],
) -> None:
    for ref, card in cards_by_ref.items():
        _assert_runtime_card(card)
        record = records_by_ref.get(ref)
        if record is None:
            raise ContentProjectionBuildError(
                f"runtime card is not backed by reviewed catalog ref: {ref}"
            )
        _assert_runtime_record(record)
        record_hash = _value(record, "content_hash")
        card_hash = _value(card, "content_hash")
        if record_hash and card_hash and record_hash != card_hash:
            raise ContentProjectionBuildError(
                f"stale runtime card hash for {ref}: catalog={record_hash} card={card_hash}"
            )


def _assert_runtime_card(card: Any) -> None:
    ref = _value(card, "ref") or "-"
    content_hash = _value(card, "content_hash")
    review_status = _value(card, "review_status")
    gate_status = _value(card, "gate_status")
    if not content_hash:
        raise ContentProjectionBuildError(f"runtime card {ref} is missing content_hash")
    if review_status not in APPROVED_REVIEW_STATUSES:
        raise ContentProjectionBuildError(
            f"runtime card {ref} is not reviewed: {review_status or '-'}"
        )
    if gate_status != RUNTIME_GATE_STATUS:
        raise ContentProjectionBuildError(
            f"runtime card {ref} is not runtime_ready: {gate_status or '-'}"
        )


def _assert_runtime_record(record: ContentPackDomainRecord) -> None:
    if not record.content_hash:
        raise ContentProjectionBuildError(
            f"catalog record {record.ref} is missing content_hash"
        )
    if record.review_status not in APPROVED_REVIEW_STATUSES:
        raise ContentProjectionBuildError(
            f"catalog record {record.ref} is not reviewed: {record.review_status}"
        )
    if record.gate_status != RUNTIME_GATE_STATUS:
        raise ContentProjectionBuildError(
            f"catalog record {record.ref} is not runtime_ready: {record.gate_status}"
        )


def _projection_ref(
    cards_by_ref: Mapping[str, Any],
    ref: str,
) -> ContentProjectionRef:
    cleaned_ref = _clean_token(ref)
    card = cards_by_ref.get(cleaned_ref)
    if card is None:
        raise ContentProjectionBuildError(
            f"projection references missing runtime card: {cleaned_ref or '-'}"
        )
    return ContentProjectionRef(
        pack_id=_value(card, "pack_id"),
        ref=cleaned_ref,
        content_hash=_value(card, "content_hash"),
        kind=_value(card, "kind") or _value(card, "card_kind"),
        visibility=_value(card, "visibility") or "router_hidden",
        title=_safe_character_text(_value(card, "title")),
        summary=_safe_character_text(_value(card, "summary")),
        aliases=_card_aliases(card),
    )


def _refs_for_actor(
    actor: ActorDossierRecord,
) -> list[str]:
    refs = [
        actor.ref,
        *actor.home_location_refs,
        *actor.front_refs,
        *actor.knowledge_channel_refs,
    ]
    return _unique_refs(refs)


def _actor_record_from_dossier(actor: ActorDossierRecord) -> ActorRecord:
    """Compile reviewed import facts once into the runtime actor contract."""

    return ActorRecord(
        may_act_offstage=actor.may_act_offstage,
        facts=[fact.model_copy(deep=True) for fact in actor.facts],
    )


def _character_projection(
    actor: ActorDossierRecord,
    *,
    known_refs: Sequence[str],
    character_id: str,
    active_character_ids: set[str],
    overrides: Mapping[str, Any],
) -> ContentCharacterProjection:
    display_name = _clean_text(overrides.get("name")) or actor.title or character_id
    status = _clean_text(overrides.get("status")) or (
        "active" if character_id in active_character_ids else "dormant"
    )
    location = (
        _clean_text(overrides.get("location"))
        or (actor.home_location_refs[0] if actor.home_location_refs else "")
    )

    return ContentCharacterProjection(
        character_id=character_id,
        name=display_name,
        status=status,
        location=location,
        actor_ref=actor.ref,
        public_role=_clean_text(overrides.get("public_role"))
        or actor.actor_kind.replace("_", " "),
        appearance=_clean_text(overrides.get("appearance"))
        or "A reviewed NPC from the imported module.",
        faction=_clean_text(overrides.get("faction")) or "",
        public_context=_safe_character_text(
            _clean_text(overrides.get("public_context"))
            or actor.public_context
        ),
        actor=_actor_record_from_dossier(actor),
        known_refs=list(known_refs),
    )


def _coerce_checkpoint(
    checkpoint: ContentCheckpointProjection | Mapping[str, Any] | None,
) -> ContentCheckpointProjection:
    if checkpoint is None:
        return ContentCheckpointProjection()
    if isinstance(checkpoint, ContentCheckpointProjection):
        return checkpoint
    return ContentCheckpointProjection.model_validate(checkpoint)


def _coerce_router_knowledge_keys(
    raw_keys: Sequence[ContentRouterKnowledgeKeyProjection | Mapping[str, Any]],
    *,
    cards_by_ref: Mapping[str, Any],
    pack_id: str,
    initial_refs: Sequence[ContentProjectionRef],
    field_start_refs: Sequence[ContentProjectionRef],
    active_front_refs: Sequence[str],
) -> list[ContentRouterKnowledgeKeyProjection]:
    if not raw_keys:
        return _default_router_knowledge_keys(
            pack_id=pack_id,
            initial_refs=initial_refs,
            field_start_refs=field_start_refs,
            active_front_refs=active_front_refs,
        )

    keys: list[ContentRouterKnowledgeKeyProjection] = []
    for raw in raw_keys:
        if isinstance(raw, ContentRouterKnowledgeKeyProjection):
            keys.append(raw)
            continue
        data = dict(raw)
        refs = data.get("packet_refs", [])
        if refs and all(isinstance(ref, str) for ref in refs):
            data["packet_refs"] = [
                _projection_ref(cards_by_ref, ref) for ref in _unique_refs(refs)
            ]
        keys.append(ContentRouterKnowledgeKeyProjection.model_validate(data))
    return keys


def _default_router_knowledge_keys(
    *,
    pack_id: str,
    initial_refs: Sequence[ContentProjectionRef],
    field_start_refs: Sequence[ContentProjectionRef],
    active_front_refs: Sequence[str],
) -> list[ContentRouterKnowledgeKeyProjection]:
    keys: list[ContentRouterKnowledgeKeyProjection] = []
    if initial_refs:
        keys.append(
            ContentRouterKnowledgeKeyProjection(
                key=f"{pack_id}.startup",
                kind="startup",
                label="Startup router context",
                summary="Reviewed content needed for the initial module setup.",
                scope_facets={
                    "front_refs": list(active_front_refs),
                    "phase_tags": ["startup"],
                },
                activation_hints=["startup", "opening", "mission"],
                priority=100,
                packet_refs=list(initial_refs),
            )
        )
    if field_start_refs:
        keys.append(
            ContentRouterKnowledgeKeyProjection(
                key=f"{pack_id}.field_start",
                kind="field_start",
                label="Field-start router context",
                summary="Reviewed content needed when the module starts in the field.",
                scope_facets={
                    "front_refs": list(active_front_refs),
                    "phase_tags": ["field_start", "route", "travel"],
                },
                activation_hints=["field", "route", "travel", "exploration"],
                priority=70,
                packet_refs=list(field_start_refs),
            )
        )
    return keys


def _coerce_field_start(
    field_start: ContentFieldStartProjection | Mapping[str, Any] | None,
    *,
    cards_by_ref: Mapping[str, Any],
    fallback_router_refs: Sequence[ContentProjectionRef],
) -> ContentFieldStartProjection:
    if field_start is None:
        return ContentFieldStartProjection(router_lookup_refs=list(fallback_router_refs))
    if isinstance(field_start, ContentFieldStartProjection):
        return field_start
    data = dict(field_start)
    raw_refs = data.get("router_lookup_refs", [])
    if raw_refs and all(isinstance(ref, str) for ref in raw_refs):
        data["router_lookup_refs"] = [
            _projection_ref(cards_by_ref, ref) for ref in _unique_refs(raw_refs)
        ]
    elif not raw_refs:
        data["router_lookup_refs"] = list(fallback_router_refs)
    raw_grants = data.get("knowledge_grants", [])
    grants: list[ContentKnowledgeProjection] = []
    for raw_grant in raw_grants:
        grant = dict(raw_grant)
        known_refs = grant.get("known_refs", [])
        if known_refs and all(isinstance(ref, str) and "@" not in ref for ref in known_refs):
            grant["known_refs"] = [
                _projection_ref(cards_by_ref, ref).compact()
                for ref in _unique_refs(known_refs)
            ]
        grants.append(ContentKnowledgeProjection.model_validate(grant))
    data["knowledge_grants"] = grants
    return ContentFieldStartProjection.model_validate(data)


def _projection_hash(artifact: ContentPackProjectionArtifact) -> str:
    payload = artifact.model_dump(mode="json")
    payload["projection_hash"] = ""
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _catalog_group_names() -> tuple[str, ...]:
    return (
        "sections",
        "spans",
        "locations",
        "keyed_areas",
        "reveal_edges",
        "handouts",
        "tables",
        "tactical_map_templates",
        "front_dossiers",
        "actor_dossiers",
        "knowledge_graph_edges",
        "statblocks",
        "trap_hazards",
        "treasures",
        "encounter_templates",
        "cross_refs",
    )


_CHARACTER_OVERRIDE_FIELDS = frozenset(
    {
        "name",
        "status",
        "location",
        "public_role",
        "appearance",
        "faction",
        "public_context",
    }
)


def _character_overrides(
    raw_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Validate the small public overlay accepted by content projections."""

    result: dict[str, dict[str, Any]] = {}
    for raw_character_id, raw_values in (raw_overrides or {}).items():
        character_id = _clean_token(raw_character_id)
        if not character_id:
            raise ContentProjectionBuildError("character override needs character id")
        if not isinstance(raw_values, Mapping):
            raise ContentProjectionBuildError(
                f"character override {character_id} must be a mapping"
            )
        values = dict(raw_values)
        unsupported = sorted(set(values) - _CHARACTER_OVERRIDE_FIELDS)
        if unsupported:
            raise ContentProjectionBuildError(
                f"character override {character_id} has unsupported fields: "
                + ", ".join(unsupported)
            )
        result[character_id] = values
    return result


def _unique_refs(refs: Iterable[str]) -> list[str]:
    return [ref for ref in dict.fromkeys(_clean_token(ref) for ref in refs) if ref]


def _card_aliases(card: Any) -> list[str]:
    aliases = [
        *_list_text(_value(card, "aliases")),
        _value(card, "ref").replace(".", " "),
        _value(card, "title"),
    ]
    return [
        alias
        for alias in dict.fromkeys(_safe_character_text(item) for item in aliases)
        if alias
    ]


def _safe_character_text(value: Any) -> str:
    return redact_imported_content_metadata_text(str(value or ""))


def _clean_token(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_clean_text(value)] if _clean_text(value) else []
    if isinstance(value, Iterable):
        return [_clean_text(item) for item in value if _clean_text(item)]
    return [_clean_text(value)] if _clean_text(value) else []


def _value(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        if key == "kind" and "kind" not in record and "card_kind" in record:
            return record.get("card_kind")
        return record.get(key, "")
    if key == "kind" and not hasattr(record, "kind") and hasattr(record, "card_kind"):
        return getattr(record, "card_kind")
    if hasattr(record, key):
        return getattr(record, key)
    if hasattr(record, "model_dump"):
        dumped = record.model_dump()
        if isinstance(dumped, Mapping):
            return dumped.get(key, "")
    return ""
