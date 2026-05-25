from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.content_privacy import contains_imported_asset_sentinel


CONTENT_PACK_PROJECTION_SCHEMA_VERSION = "content-pack-projection-v1"
ProjectionCharacterStatus = Literal["active", "dormant", "culled"]
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clean_unique_strings(values: list[str]) -> list[str]:
    return [item for item in dict.fromkeys(_clean_text(value) for value in values) if item]


def _clean_token(value: Any) -> str:
    text = _clean_text(value)
    return text if text and _SAFE_TOKEN_RE.fullmatch(text) else ""


def _reject_private_asset_text(value: str, *, field_name: str) -> str:
    text = _clean_text(value)
    if contains_imported_asset_sentinel(text):
        raise ValueError(f"{field_name} contains private asset/source text")
    return text


class ContentProjectionRef(BaseModel):
    """Stable import-time projection of one runtime-readable content card."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    ref: str
    content_hash: str
    kind: str = ""
    visibility: str = "router_hidden"
    title: str = ""
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentProjectionRef":
        self.pack_id = _clean_text(self.pack_id)
        self.ref = _clean_text(self.ref)
        self.content_hash = _clean_text(self.content_hash)
        self.kind = _clean_text(self.kind)
        self.visibility = _clean_text(self.visibility) or "router_hidden"
        self.title = _reject_private_asset_text(self.title, field_name="title")
        self.summary = _reject_private_asset_text(self.summary, field_name="summary")
        self.aliases = [
            _reject_private_asset_text(alias, field_name="alias")
            for alias in _clean_unique_strings(self.aliases)
        ]
        if not self.ref:
            raise ValueError("projection refs need ref")
        if not self.content_hash:
            raise ValueError(f"projection ref {self.ref} needs content_hash")
        return self

    def compact(self) -> str:
        return f"{self.pack_id}:{self.ref}@{self.content_hash}"

    def router_catalog_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ref": self.ref,
            "kind": self.kind,
            "visibility": self.visibility,
            "aliases": list(self.aliases),
            "title": self.title,
            "summary": self.summary,
        }
        return {key: value for key, value in entry.items() if value not in ("", [])}


class ContentRouterKnowledgeScopeFacets(BaseModel):
    """Small import-authored selectors for router knowledge dispatch."""

    model_config = ConfigDict(extra="forbid")

    location_refs: list[str] = Field(default_factory=list)
    front_refs: list[str] = Field(default_factory=list)
    actor_refs: list[str] = Field(default_factory=list)
    character_ids: list[str] = Field(default_factory=list)
    phase_tags: list[str] = Field(default_factory=list)
    region_tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentRouterKnowledgeScopeFacets":
        self.location_refs = _clean_unique_strings(self.location_refs)
        self.front_refs = _clean_unique_strings(self.front_refs)
        self.actor_refs = _clean_unique_strings(self.actor_refs)
        self.character_ids = _clean_unique_strings(self.character_ids)
        self.phase_tags = _clean_unique_strings(self.phase_tags)
        self.region_tags = _clean_unique_strings(self.region_tags)
        return self

    def compact_entry(self) -> dict[str, list[str]]:
        return {
            key: value
            for key, value in {
                "location_refs": list(self.location_refs),
                "front_refs": list(self.front_refs),
                "actor_refs": list(self.actor_refs),
                "character_ids": list(self.character_ids),
                "phase_tags": list(self.phase_tags),
                "region_tags": list(self.region_tags),
            }.items()
            if value
        }


class ContentRouterKnowledgeKeyProjection(BaseModel):
    """Tiny selector that expands to reviewed router content packets."""

    model_config = ConfigDict(extra="forbid")

    key: str
    kind: str = "module_scope"
    label: str = ""
    summary: str = ""
    scope_facets: ContentRouterKnowledgeScopeFacets = Field(
        default_factory=ContentRouterKnowledgeScopeFacets
    )
    activation_hints: list[str] = Field(default_factory=list)
    priority: int = 0
    packet_refs: list[ContentProjectionRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentRouterKnowledgeKeyProjection":
        self.key = _clean_token(self.key)
        self.kind = _clean_token(self.kind) or "module_scope"
        self.label = _reject_private_asset_text(self.label or self.key, field_name="label")
        self.summary = _reject_private_asset_text(self.summary, field_name="summary")
        self.activation_hints = [
            _reject_private_asset_text(hint, field_name="activation_hint")
            for hint in _clean_unique_strings(self.activation_hints)
        ]
        self.priority = max(0, int(self.priority or 0))
        self.packet_refs = _dedupe_refs(self.packet_refs)
        if not self.key:
            raise ValueError("router knowledge keys need key")
        if not self.packet_refs:
            raise ValueError(f"router knowledge key {self.key} needs packet_refs")
        return self

    def prompt_index_entry(self) -> dict[str, Any]:
        packet_kinds = [
            kind
            for kind in dict.fromkeys(ref.kind for ref in self.packet_refs)
            if kind
        ]
        entry: dict[str, Any] = {
            "key": self.key,
            "kind": self.kind,
            "label": self.label,
            "summary": self.summary,
            "scope_facets": self.scope_facets.compact_entry(),
            "activation_hints": list(self.activation_hints),
            "priority": self.priority,
            "packet_count": len(self.packet_refs),
            "packet_kinds": packet_kinds,
        }
        return {key: value for key, value in entry.items() if value not in ("", [], {}, 0)}

    def packet_metadata_entry(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "packet_refs": [
                {
                    "pack_id": ref.pack_id,
                    "ref": ref.ref,
                    "content_hash": ref.content_hash,
                    "kind": ref.kind,
                    "visibility": ref.visibility,
                    "title": ref.title,
                    "summary": ref.summary,
                }
                for ref in self.packet_refs
            ],
        }


class ContentCharacterProjection(BaseModel):
    """Reviewed import-time seed for one character agent."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    name: str
    status: ProjectionCharacterStatus = "dormant"
    location: str = ""
    actor_ref: str = ""
    agent_context_ref: str = ""
    public_role: str = ""
    appearance: str = ""
    faction: str = ""
    backstory: str = ""
    personality: str = ""
    known_context: str = ""
    goals: list[str] = Field(default_factory=list)
    current_objectives: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    intentions_enabled: bool = False
    tick_cues: list[str] = Field(default_factory=list)
    known_refs: list[str] = Field(default_factory=list)
    suspected_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentCharacterProjection":
        self.character_id = _clean_text(self.character_id)
        self.name = _reject_private_asset_text(self.name, field_name="name")
        self.location = _clean_text(self.location)
        self.actor_ref = _clean_text(self.actor_ref)
        self.agent_context_ref = _clean_text(self.agent_context_ref)
        for field_name in (
            "public_role",
            "appearance",
            "faction",
            "backstory",
            "personality",
            "known_context",
        ):
            setattr(
                self,
                field_name,
                _reject_private_asset_text(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        self.goals = _clean_unique_strings(self.goals)
        self.current_objectives = _clean_unique_strings(self.current_objectives)
        self.secrets = [
            _reject_private_asset_text(secret, field_name="secret")
            for secret in _clean_unique_strings(self.secrets)
        ]
        self.tick_cues = _clean_unique_strings(self.tick_cues)
        self.known_refs = _clean_unique_strings(self.known_refs)
        self.suspected_refs = [
            ref for ref in _clean_unique_strings(self.suspected_refs) if ref not in self.known_refs
        ]
        if not self.character_id:
            raise ValueError("character projections need character_id")
        if not self.name:
            raise ValueError(f"character projection {self.character_id} needs name")
        return self


class ContentKnowledgeProjection(BaseModel):
    """Initial engine-owned knowledge-map state for one entity."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    known_refs: list[str] = Field(default_factory=list)
    suspected_refs: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "ContentKnowledgeProjection":
        self.entity_id = _clean_text(self.entity_id)
        self.known_refs = _clean_unique_strings(self.known_refs)
        self.suspected_refs = [
            ref for ref in _clean_unique_strings(self.suspected_refs) if ref not in self.known_refs
        ]
        self.notes = _reject_private_asset_text(self.notes, field_name="notes")
        if not self.entity_id:
            raise ValueError("knowledge projections need entity_id")
        return self


class ContentFrontProjection(BaseModel):
    """Initial front/pressure state authored by the import step."""

    model_config = ConfigDict(extra="forbid")

    front_id: str
    label: str = ""
    status: str = "active"
    notes: str = ""
    introduced_ref_keys: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentFrontProjection":
        self.front_id = _clean_text(self.front_id)
        self.label = _reject_private_asset_text(self.label or self.front_id, field_name="label")
        self.status = _clean_text(self.status).lower() or "active"
        self.notes = _reject_private_asset_text(self.notes, field_name="notes")
        self.introduced_ref_keys = _clean_unique_strings(self.introduced_ref_keys)
        if not self.front_id:
            raise ValueError("front projections need front_id")
        return self


class ContentRouterProjection(BaseModel):
    """Router startup and lookup packets prepared by import-time review."""

    model_config = ConfigDict(extra="forbid")

    initial_lookup_refs: list[ContentProjectionRef] = Field(default_factory=list)
    field_start_lookup_refs: list[ContentProjectionRef] = Field(default_factory=list)
    lookup_catalog: list[ContentProjectionRef] = Field(default_factory=list)
    knowledge_keys: list[ContentRouterKnowledgeKeyProjection] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _clean(self) -> "ContentRouterProjection":
        self.initial_lookup_refs = _dedupe_refs(self.initial_lookup_refs)
        self.field_start_lookup_refs = _dedupe_refs(self.field_start_lookup_refs)
        self.lookup_catalog = _dedupe_refs(self.lookup_catalog)
        self.knowledge_keys = _dedupe_router_knowledge_keys(self.knowledge_keys)
        return self

    def router_catalog_metadata(self) -> list[dict[str, Any]]:
        return [ref.router_catalog_entry() for ref in self.lookup_catalog]

    def router_knowledge_index_metadata(self) -> list[dict[str, Any]]:
        return [item.prompt_index_entry() for item in self.knowledge_keys]

    def router_knowledge_packet_metadata(self) -> list[dict[str, Any]]:
        return [item.packet_metadata_entry() for item in self.knowledge_keys]


class ContentCharacterPatchProjection(BaseModel):
    """Checkpoint-start patch for character state when a seed starts in medias res."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    status: ProjectionCharacterStatus | None = None
    location: str = ""
    known_context: str = ""
    current_objectives: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentCharacterPatchProjection":
        self.character_id = _clean_text(self.character_id)
        self.location = _clean_text(self.location)
        self.known_context = _reject_private_asset_text(
            self.known_context,
            field_name="known_context",
        )
        self.current_objectives = _clean_unique_strings(self.current_objectives)
        if not self.character_id:
            raise ValueError("character patches need character_id")
        return self


class ContentCheckpointProjection(BaseModel):
    """Story checkpoint seed text owned by the import artifact."""

    model_config = ConfigDict(extra="forbid")

    player_primer: str = ""
    world_facts: list[str] = Field(default_factory=list)
    narrative_rules: str = ""
    world_lore: str = ""
    player_known_context: str = ""
    player_objectives: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentCheckpointProjection":
        for field_name in (
            "player_primer",
            "narrative_rules",
            "world_lore",
            "player_known_context",
        ):
            setattr(
                self,
                field_name,
                _reject_private_asset_text(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        self.world_facts = [
            _reject_private_asset_text(fact, field_name="world_fact")
            for fact in _clean_unique_strings(self.world_facts)
        ]
        self.player_objectives = _clean_unique_strings(self.player_objectives)
        return self


class ContentFieldStartProjection(BaseModel):
    """Optional alternate seed for beginning after an import-authored handoff."""

    model_config = ConfigDict(extra="forbid")

    router_lookup_refs: list[ContentProjectionRef] = Field(default_factory=list)
    location_ref: str = ""
    active_character_ids: list[str] = Field(default_factory=list)
    knowledge_grants: list[ContentKnowledgeProjection] = Field(default_factory=list)
    character_patches: list[ContentCharacterPatchProjection] = Field(default_factory=list)
    checkpoint: ContentCheckpointProjection = Field(default_factory=ContentCheckpointProjection)

    @model_validator(mode="after")
    def _clean(self) -> "ContentFieldStartProjection":
        self.router_lookup_refs = _dedupe_refs(self.router_lookup_refs)
        self.location_ref = _clean_text(self.location_ref)
        self.active_character_ids = _clean_unique_strings(self.active_character_ids)
        self.knowledge_grants = _dedupe_knowledge(self.knowledge_grants)
        self.character_patches = _dedupe_character_patches(self.character_patches)
        return self


class ContentEngineOverlayProjection(BaseModel):
    """Private engine/rules-adapter overlay prepared at import time."""

    model_config = ConfigDict(extra="forbid")

    domain_catalog: dict[str, Any] = Field(default_factory=dict)
    domain_groups: dict[str, int] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "ContentEngineOverlayProjection":
        self.domain_groups = {
            _clean_text(key): max(0, int(value or 0))
            for key, value in self.domain_groups.items()
            if _clean_text(key)
        }
        self.notes = _reject_private_asset_text(self.notes, field_name="notes")
        return self


class ContentPackProjectionArtifact(BaseModel):
    """Durable import-time projection consumed by runtime seeds."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str
    pack_version: str = ""
    source_fingerprint: str = ""
    schema_version: str = CONTENT_PACK_PROJECTION_SCHEMA_VERSION
    content_pack_schema_version: str = ""
    catalog_schema_version: str = ""
    catalog_build_hash: str = ""
    projection_hash: str = ""
    router: ContentRouterProjection = Field(default_factory=ContentRouterProjection)
    characters: list[ContentCharacterProjection] = Field(default_factory=list)
    knowledge_map: list[ContentKnowledgeProjection] = Field(default_factory=list)
    fronts: list[ContentFrontProjection] = Field(default_factory=list)
    checkpoint: ContentCheckpointProjection = Field(default_factory=ContentCheckpointProjection)
    field_start: ContentFieldStartProjection = Field(default_factory=ContentFieldStartProjection)
    engine_overlay: ContentEngineOverlayProjection = Field(
        default_factory=ContentEngineOverlayProjection
    )

    @model_validator(mode="after")
    def _clean(self) -> "ContentPackProjectionArtifact":
        self.pack_id = _clean_text(self.pack_id)
        self.pack_version = _clean_text(self.pack_version)
        self.source_fingerprint = _clean_text(self.source_fingerprint)
        self.schema_version = (
            _clean_text(self.schema_version)
            or CONTENT_PACK_PROJECTION_SCHEMA_VERSION
        )
        self.content_pack_schema_version = _clean_text(self.content_pack_schema_version)
        self.catalog_schema_version = _clean_text(self.catalog_schema_version)
        self.catalog_build_hash = _clean_text(self.catalog_build_hash)
        self.projection_hash = _clean_text(self.projection_hash)
        if not self.pack_id:
            raise ValueError("projection artifacts need pack_id")
        for ref in [
            *self.router.initial_lookup_refs,
            *self.router.field_start_lookup_refs,
            *self.router.lookup_catalog,
            *[
                packet_ref
                for key in self.router.knowledge_keys
                for packet_ref in key.packet_refs
            ],
            *self.field_start.router_lookup_refs,
        ]:
            if not ref.pack_id:
                ref.pack_id = self.pack_id
            if ref.pack_id != self.pack_id:
                raise ValueError(f"projection ref pack mismatch: {ref.ref}")
        self.characters = _dedupe_characters(self.characters)
        self.knowledge_map = _dedupe_knowledge(self.knowledge_map)
        self.fronts = _dedupe_fronts(self.fronts)
        return self


def _dedupe_refs(refs: list[ContentProjectionRef]) -> list[ContentProjectionRef]:
    deduped: dict[str, ContentProjectionRef] = {}
    for ref in refs:
        deduped[ref.ref] = ref
    return list(deduped.values())


def _dedupe_router_knowledge_keys(
    keys: list[ContentRouterKnowledgeKeyProjection],
) -> list[ContentRouterKnowledgeKeyProjection]:
    deduped: dict[str, ContentRouterKnowledgeKeyProjection] = {}
    for item in keys:
        if item.key in deduped:
            raise ValueError(f"duplicate router knowledge key: {item.key}")
        deduped[item.key] = item
    return list(deduped.values())


def _dedupe_characters(
    characters: list[ContentCharacterProjection],
) -> list[ContentCharacterProjection]:
    deduped: dict[str, ContentCharacterProjection] = {}
    for character in characters:
        deduped[character.character_id] = character
    return list(deduped.values())


def _dedupe_knowledge(
    items: list[ContentKnowledgeProjection],
) -> list[ContentKnowledgeProjection]:
    deduped: dict[str, ContentKnowledgeProjection] = {}
    for item in items:
        deduped[item.entity_id] = item
    return list(deduped.values())


def _dedupe_fronts(fronts: list[ContentFrontProjection]) -> list[ContentFrontProjection]:
    deduped: dict[str, ContentFrontProjection] = {}
    for front in fronts:
        deduped[front.front_id] = front
    return list(deduped.values())


def _dedupe_character_patches(
    patches: list[ContentCharacterPatchProjection],
) -> list[ContentCharacterPatchProjection]:
    deduped: dict[str, ContentCharacterPatchProjection] = {}
    for patch in patches:
        deduped[patch.character_id] = patch
    return list(deduped.values())
