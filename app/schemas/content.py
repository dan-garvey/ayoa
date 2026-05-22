from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)

from app.schemas.content_privacy import (
    FORBIDDEN_MODULE_METADATA_KEYS,
    contains_imported_asset_sentinel,
    sanitize_module_metadata,
    should_include_private_runtime_metadata,
)


ContentSignalStatus = Literal["pending", "resolved", "dismissed"]
ContentSpawnOverlayStatus = Literal["active", "moved", "killed", "removed"]
ContentFrontKnowledgeStatus = Literal["unknown", "suspected", "known", "resolved"]
ContentActivePlanStatus = Literal[
    "planned",
    "active",
    "blocked",
    "resolved",
    "abandoned",
]
ContentModuleOverrideKind = Literal[
    "availability",
    "replacement",
    "state",
    "visibility",
    "other",
]

_CONTENT_OVERLAY_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")


def content_ref_key(pack_id: str, ref_id: str, content_hash: str) -> str:
    """Stable key for deduplicating a concrete content reference."""

    return "::".join(
        (
            (pack_id or "").strip(),
            (ref_id or "").strip(),
            (content_hash or "").strip(),
        )
    )


def content_overlay_key(ref_id: str, content_hash: str = "") -> str:
    """Pack-local key for mutable overlay records."""

    return "::".join(
        part
        for part in (
            _clean_overlay_token(ref_id),
            _clean_overlay_token(content_hash),
        )
        if part
    )


def _clean_overlay_token(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if contains_imported_asset_sentinel(text):
        return ""
    if not _CONTENT_OVERLAY_TOKEN_RE.fullmatch(text):
        return ""
    if text.lower() in FORBIDDEN_MODULE_METADATA_KEYS:
        return ""
    return text


def _clean_overlay_tokens(values: list[str]) -> list[str]:
    return [
        token
        for value in dict.fromkeys(values)
        if (token := _clean_overlay_token(value))
    ]


def _clean_overlay_bool_flags(values: Mapping[str, bool]) -> dict[str, bool]:
    return {
        key: bool(value)
        for raw_key, value in values.items()
        if (key := _clean_overlay_token(raw_key))
    }


def _clean_overlay_ref_map(values: Mapping[str, str]) -> dict[str, str]:
    return {
        key: cleaned_value
        for raw_key, raw_value in values.items()
        if (key := _clean_overlay_token(raw_key))
        if (cleaned_value := _clean_overlay_token(raw_value))
    }


def _rekey_overlay_records(records: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for record in records.values():
        key = record.overlay_key() if hasattr(record, "overlay_key") else ""
        if key:
            cleaned[key] = record
    return cleaned


class IntroducedContentRef(BaseModel):
    """A content reference the router has already brought into play."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    ref_id: str = ""
    content_hash: str = ""
    label: str = ""
    kind: str = ""
    source_event_id: str = ""
    introduced_at_s: int = 0
    notes: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "IntroducedContentRef":
        self.pack_id = self.pack_id.strip()
        self.ref_id = self.ref_id.strip()
        self.content_hash = self.content_hash.strip()
        self.label = self.label.strip()
        self.kind = self.kind.strip().lower()
        self.source_event_id = self.source_event_id.strip()
        self.notes = self.notes.strip()
        if self.introduced_at_s < 0:
            self.introduced_at_s = 0
        return self

    def dedupe_key(self) -> str:
        return content_ref_key(self.pack_id, self.ref_id, self.content_hash)


class PendingContentSignal(BaseModel):
    """A durable signal that more content may need to be looked up."""

    model_config = ConfigDict(extra="forbid")

    signal_id: str = ""
    pack_id: str = ""
    ref_id: str = ""
    content_hash: str = ""
    reason: str = ""
    source_event_id: str = ""
    status: ContentSignalStatus = "pending"
    priority: int = 0
    created_at_s: int = 0
    requested_fields: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "PendingContentSignal":
        self.signal_id = self.signal_id.strip()
        self.pack_id = self.pack_id.strip()
        self.ref_id = self.ref_id.strip()
        self.content_hash = self.content_hash.strip()
        self.reason = self.reason.strip()
        self.source_event_id = self.source_event_id.strip()
        if self.priority < 0:
            self.priority = 0
        if self.created_at_s < 0:
            self.created_at_s = 0
        self.requested_fields = [
            field.strip()
            for field in dict.fromkeys(self.requested_fields)
            if field.strip()
        ]
        return self

    def content_key(self) -> str:
        return content_ref_key(self.pack_id, self.ref_id, self.content_hash)

    @field_serializer("metadata")
    def _serialize_metadata(
        self,
        value: dict[str, Any],
        info: SerializationInfo,
    ) -> dict[str, Any]:
        if should_include_private_runtime_metadata(info.context):
            return value
        return sanitize_module_metadata(value) or {}


class ContentFrontState(BaseModel):
    """Pack-local progress state for a story front or pressure track."""

    model_config = ConfigDict(extra="forbid")

    front_id: str = ""
    label: str = ""
    status: str = ""
    clock: int = 0
    max_clock: int = 0
    villain_ids: list[str] = Field(default_factory=list)
    introduced_ref_keys: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "ContentFrontState":
        self.front_id = self.front_id.strip()
        self.label = self.label.strip()
        self.status = self.status.strip().lower()
        if self.clock < 0:
            self.clock = 0
        if self.max_clock < 0:
            self.max_clock = 0
        if self.max_clock and self.clock > self.max_clock:
            self.clock = self.max_clock
        self.villain_ids = [
            villain_id.strip()
            for villain_id in dict.fromkeys(self.villain_ids)
            if villain_id.strip()
        ]
        self.introduced_ref_keys = [
            ref_key.strip()
            for ref_key in dict.fromkeys(self.introduced_ref_keys)
            if ref_key.strip()
        ]
        self.notes = self.notes.strip()
        return self


class ContentVillainState(BaseModel):
    """Pack-local progress state for an antagonist or other major pressure."""

    model_config = ConfigDict(extra="forbid")

    villain_id: str = ""
    label: str = ""
    status: str = ""
    front_ids: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    introduced_ref_keys: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "ContentVillainState":
        self.villain_id = self.villain_id.strip()
        self.label = self.label.strip()
        self.status = self.status.strip().lower()
        self.front_ids = [
            front_id.strip()
            for front_id in dict.fromkeys(self.front_ids)
            if front_id.strip()
        ]
        self.goals = [
            goal.strip()
            for goal in dict.fromkeys(self.goals)
            if goal.strip()
        ]
        self.introduced_ref_keys = [
            ref_key.strip()
            for ref_key in dict.fromkeys(self.introduced_ref_keys)
            if ref_key.strip()
        ]
        self.notes = self.notes.strip()
        return self


class ContentLocationOverlayState(BaseModel):
    """Mutable pack-local location state, keyed by reviewed module refs."""

    model_config = ConfigDict(extra="forbid")

    location_id: str = ""
    content_hash: str = ""
    revealed: bool = False
    visited: bool = False

    @model_validator(mode="after")
    def _clean(self) -> "ContentLocationOverlayState":
        self.location_id = _clean_overlay_token(self.location_id)
        self.content_hash = _clean_overlay_token(self.content_hash)
        return self

    def overlay_key(self) -> str:
        return content_overlay_key(self.location_id, self.content_hash)


class ContentDoorOverlayState(BaseModel):
    """Mutable pack-local door/passage state."""

    model_config = ConfigDict(extra="forbid")

    door_id: str = ""
    content_hash: str = ""
    opened: bool = False
    locked: bool = False
    sealed: bool = False

    @model_validator(mode="after")
    def _clean(self) -> "ContentDoorOverlayState":
        self.door_id = _clean_overlay_token(self.door_id)
        self.content_hash = _clean_overlay_token(self.content_hash)
        return self

    def overlay_key(self) -> str:
        return content_overlay_key(self.door_id, self.content_hash)


class ContentTrapOverlayState(BaseModel):
    """Mutable pack-local trap state."""

    model_config = ConfigDict(extra="forbid")

    trap_id: str = ""
    content_hash: str = ""
    revealed: bool = False
    disabled: bool = False
    sprung: bool = False

    @model_validator(mode="after")
    def _clean(self) -> "ContentTrapOverlayState":
        self.trap_id = _clean_overlay_token(self.trap_id)
        self.content_hash = _clean_overlay_token(self.content_hash)
        return self

    def overlay_key(self) -> str:
        return content_overlay_key(self.trap_id, self.content_hash)


class ContentTreasureOverlayState(BaseModel):
    """Mutable pack-local treasure or reward state."""

    model_config = ConfigDict(extra="forbid")

    treasure_id: str = ""
    content_hash: str = ""
    revealed: bool = False
    looted: bool = False
    depleted: bool = False
    claimed_ref_ids: list[str] = Field(default_factory=list)
    remaining_ref_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentTreasureOverlayState":
        self.treasure_id = _clean_overlay_token(self.treasure_id)
        self.content_hash = _clean_overlay_token(self.content_hash)
        self.claimed_ref_ids = _clean_overlay_tokens(self.claimed_ref_ids)
        self.remaining_ref_ids = _clean_overlay_tokens(self.remaining_ref_ids)
        return self

    def overlay_key(self) -> str:
        return content_overlay_key(self.treasure_id, self.content_hash)


class ContentSpawnOverlayState(BaseModel):
    """Mutable state for a spawned module reference."""

    model_config = ConfigDict(extra="forbid")

    spawn_ref_id: str = ""
    content_hash: str = ""
    source_ref_id: str = ""
    current_location_ref_id: str = ""
    current_location_hash: str = ""
    status: ContentSpawnOverlayStatus = "active"

    @model_validator(mode="after")
    def _clean(self) -> "ContentSpawnOverlayState":
        self.spawn_ref_id = _clean_overlay_token(self.spawn_ref_id)
        self.content_hash = _clean_overlay_token(self.content_hash)
        self.source_ref_id = _clean_overlay_token(self.source_ref_id)
        self.current_location_ref_id = _clean_overlay_token(
            self.current_location_ref_id
        )
        self.current_location_hash = _clean_overlay_token(self.current_location_hash)
        return self

    def overlay_key(self) -> str:
        return content_overlay_key(self.spawn_ref_id, self.content_hash)


class ContentCombatMapOverlayState(BaseModel):
    """Mutable reveal/fog state for a reviewed combat-map record."""

    model_config = ConfigDict(extra="forbid")

    map_id: str = ""
    content_hash: str = ""
    fog_of_war: bool = True
    fully_revealed: bool = False
    revealed_area_ref_ids: list[str] = Field(default_factory=list)
    fogged_area_ref_ids: list[str] = Field(default_factory=list)
    visible_spawn_ref_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentCombatMapOverlayState":
        self.map_id = _clean_overlay_token(self.map_id)
        self.content_hash = _clean_overlay_token(self.content_hash)
        self.revealed_area_ref_ids = _clean_overlay_tokens(
            self.revealed_area_ref_ids
        )
        self.fogged_area_ref_ids = _clean_overlay_tokens(self.fogged_area_ref_ids)
        self.visible_spawn_ref_ids = _clean_overlay_tokens(
            self.visible_spawn_ref_ids
        )
        return self

    def overlay_key(self) -> str:
        return content_overlay_key(self.map_id, self.content_hash)


class ContentPovRevealState(BaseModel):
    """Pack-local asset/handout/map visibility for one player POV."""

    model_config = ConfigDict(extra="forbid")

    viewer_id: str = ""
    revealed_handout_ref_ids: list[str] = Field(default_factory=list)
    revealed_asset_ids: list[str] = Field(default_factory=list)
    reveal_ref_ids: list[str] = Field(default_factory=list)
    map_overlays: dict[str, ContentCombatMapOverlayState] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _clean(self) -> "ContentPovRevealState":
        self.viewer_id = _clean_overlay_token(self.viewer_id)
        self.revealed_handout_ref_ids = _clean_overlay_tokens(
            self.revealed_handout_ref_ids
        )
        self.revealed_asset_ids = _clean_overlay_tokens(self.revealed_asset_ids)
        self.reveal_ref_ids = _clean_overlay_tokens(self.reveal_ref_ids)
        self.map_overlays = _rekey_overlay_records(self.map_overlays)
        return self

    def overlay_key(self) -> str:
        return _clean_overlay_token(self.viewer_id)


class ContentFrontKnowledgeState(BaseModel):
    """Mutable party-facing knowledge about a module front."""

    model_config = ConfigDict(extra="forbid")

    front_id: str = ""
    content_hash: str = ""
    status: ContentFrontKnowledgeStatus = "unknown"
    known_ref_ids: list[str] = Field(default_factory=list)
    revealed_clue_ref_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentFrontKnowledgeState":
        self.front_id = _clean_overlay_token(self.front_id)
        self.content_hash = _clean_overlay_token(self.content_hash)
        self.known_ref_ids = _clean_overlay_tokens(self.known_ref_ids)
        self.revealed_clue_ref_ids = _clean_overlay_tokens(
            self.revealed_clue_ref_ids
        )
        return self

    def overlay_key(self) -> str:
        return content_overlay_key(self.front_id, self.content_hash)


class ContentActivePlanState(BaseModel):
    """Mutable state for a reviewed module plan or pressure sequence."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = ""
    content_hash: str = ""
    front_id: str = ""
    owner_ref_id: str = ""
    current_step_ref_id: str = ""
    target_ref_ids: list[str] = Field(default_factory=list)
    completed_step_ref_ids: list[str] = Field(default_factory=list)
    status: ContentActivePlanStatus = "planned"

    @model_validator(mode="after")
    def _clean(self) -> "ContentActivePlanState":
        self.plan_id = _clean_overlay_token(self.plan_id)
        self.content_hash = _clean_overlay_token(self.content_hash)
        self.front_id = _clean_overlay_token(self.front_id)
        self.owner_ref_id = _clean_overlay_token(self.owner_ref_id)
        self.current_step_ref_id = _clean_overlay_token(self.current_step_ref_id)
        self.target_ref_ids = _clean_overlay_tokens(self.target_ref_ids)
        self.completed_step_ref_ids = _clean_overlay_tokens(
            self.completed_step_ref_ids
        )
        return self

    def overlay_key(self) -> str:
        return content_overlay_key(self.plan_id, self.content_hash)


class ContentModuleOverrideState(BaseModel):
    """Mutable module override made from refs, hashes, and boolean flags."""

    model_config = ConfigDict(extra="forbid")

    override_id: str = ""
    target_ref_id: str = ""
    content_hash: str = ""
    kind: ContentModuleOverrideKind = "state"
    enabled: bool = True
    replacement_ref_id: str = ""
    replacement_hash: str = ""
    flags: dict[str, bool] = Field(default_factory=dict)
    ref_overrides: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "ContentModuleOverrideState":
        self.override_id = _clean_overlay_token(self.override_id)
        self.target_ref_id = _clean_overlay_token(self.target_ref_id)
        self.content_hash = _clean_overlay_token(self.content_hash)
        self.replacement_ref_id = _clean_overlay_token(self.replacement_ref_id)
        self.replacement_hash = _clean_overlay_token(self.replacement_hash)
        self.flags = _clean_overlay_bool_flags(self.flags)
        self.ref_overrides = _clean_overlay_ref_map(self.ref_overrides)
        return self

    def overlay_key(self) -> str:
        return content_overlay_key(
            self.override_id or self.target_ref_id,
            self.content_hash,
        )


class ContentOverlayState(BaseModel):
    """Checkpoint overlay for mutable state in authored content packs."""

    model_config = ConfigDict(extra="forbid")

    locations: dict[str, ContentLocationOverlayState] = Field(default_factory=dict)
    doors: dict[str, ContentDoorOverlayState] = Field(default_factory=dict)
    traps: dict[str, ContentTrapOverlayState] = Field(default_factory=dict)
    treasures: dict[str, ContentTreasureOverlayState] = Field(default_factory=dict)
    spawn_refs: dict[str, ContentSpawnOverlayState] = Field(default_factory=dict)
    combat_maps: dict[str, ContentCombatMapOverlayState] = Field(default_factory=dict)
    pov_reveals: dict[str, ContentPovRevealState] = Field(default_factory=dict)
    front_knowledge: dict[str, ContentFrontKnowledgeState] = Field(
        default_factory=dict
    )
    active_plans: dict[str, ContentActivePlanState] = Field(default_factory=dict)
    module_overrides: dict[str, ContentModuleOverrideState] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _clean(self) -> "ContentOverlayState":
        self.locations = _rekey_overlay_records(self.locations)
        self.doors = _rekey_overlay_records(self.doors)
        self.traps = _rekey_overlay_records(self.traps)
        self.treasures = _rekey_overlay_records(self.treasures)
        self.spawn_refs = _rekey_overlay_records(self.spawn_refs)
        self.combat_maps = _rekey_overlay_records(self.combat_maps)
        self.pov_reveals = _rekey_overlay_records(self.pov_reveals)
        self.front_knowledge = _rekey_overlay_records(self.front_knowledge)
        self.active_plans = _rekey_overlay_records(self.active_plans)
        self.module_overrides = _rekey_overlay_records(self.module_overrides)
        return self


class ContentPackState(BaseModel):
    """Checkpoint state for one adventure/content pack."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    introduced_refs: dict[str, IntroducedContentRef] = Field(default_factory=dict)
    pending_signals: dict[str, PendingContentSignal] = Field(default_factory=dict)
    fronts: dict[str, ContentFrontState] = Field(default_factory=dict)
    villains: dict[str, ContentVillainState] = Field(default_factory=dict)
    overlay: ContentOverlayState = Field(default_factory=ContentOverlayState)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "ContentPackState":
        self.pack_id = self.pack_id.strip()
        return self

    @field_serializer("metadata")
    def _serialize_metadata(
        self,
        value: dict[str, Any],
        info: SerializationInfo,
    ) -> dict[str, Any]:
        if should_include_private_runtime_metadata(info.context):
            return value
        return sanitize_module_metadata(value) or {}
