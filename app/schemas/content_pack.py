from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ReviewStatus = Literal[
    "unreviewed",
    "needs_review",
    "reviewed",
    "approved",
    "blocked",
    "rejected",
]
SpoilerClass = Literal["none", "low", "moderate", "high"]
CoverageGateStatus = Literal["runtime_ready", "flagged", "blocked"]
AssetRevealAudience = Literal["all_observers", "only"]
AssetPresentation = Literal["inline", "attachment", "reference", "map_overlay"]


class ContentProvenance(BaseModel):
    """Compiled source pointer without raw paths or protected excerpts."""

    model_config = ConfigDict(extra="forbid")

    source_asset_id: str = ""
    page_id: str = ""
    span_id: str = ""
    image_id: str = ""
    bbox: list[float] = Field(default_factory=list)
    section_id: str = ""
    method: str = ""
    confidence: float = 1.0
    importer_version: str = ""
    human_review_status: ReviewStatus = "unreviewed"

    @model_validator(mode="after")
    def _clean(self) -> "ContentProvenance":
        self.source_asset_id = self.source_asset_id.strip()
        self.page_id = self.page_id.strip()
        self.span_id = self.span_id.strip()
        self.image_id = self.image_id.strip()
        self.section_id = self.section_id.strip()
        self.method = self.method.strip()
        self.importer_version = self.importer_version.strip()
        self.confidence = _clamp_confidence(self.confidence)
        self.bbox = [float(value) for value in self.bbox]
        return self


class PageInventoryRecord(BaseModel):
    """One logical source page in the compiled private pack inventory."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    page_id: str
    source_asset_id: str
    pdf_page_index: int = 0
    printed_page_label: str = ""
    source_sha256: str = ""
    section_id: str = ""
    alignment_status: str = ""
    confidence: float = 1.0
    review_status: ReviewStatus = "unreviewed"
    coverage_status: CoverageGateStatus = "runtime_ready"
    notes: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "PageInventoryRecord":
        self.pack_id = self.pack_id.strip()
        self.page_id = self.page_id.strip()
        self.source_asset_id = self.source_asset_id.strip()
        self.printed_page_label = self.printed_page_label.strip()
        self.source_sha256 = self.source_sha256.strip()
        self.section_id = self.section_id.strip()
        self.alignment_status = self.alignment_status.strip()
        self.notes = self.notes.strip()
        if self.pdf_page_index < 0:
            self.pdf_page_index = 0
        self.confidence = _clamp_confidence(self.confidence)
        return self


class CompiledContentCard(BaseModel):
    """Runtime-readable compiled card, already redacted for the pack layer."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    ref: str
    content_hash: str = ""
    card_kind: str = "content"
    visibility: str = "hidden"
    title: str = ""
    summary: str = ""
    body: str = ""
    spoiler_class: SpoilerClass = "none"
    reveal_trigger: str = ""
    confidence: float = 1.0
    review_status: ReviewStatus = "unreviewed"
    gate_status: CoverageGateStatus = "flagged"
    gate_reasons: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    provenance: list[ContentProvenance] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "CompiledContentCard":
        self.pack_id = self.pack_id.strip()
        self.ref = self.ref.strip()
        self.content_hash = self.content_hash.strip()
        self.card_kind = self.card_kind.strip().lower() or "content"
        self.visibility = self.visibility.strip().lower() or "hidden"
        self.title = self.title.strip()
        self.summary = self.summary.strip()
        self.body = self.body.strip()
        self.reveal_trigger = self.reveal_trigger.strip()
        self.confidence = _clamp_confidence(self.confidence)
        self.gate_reasons = [
            reason.strip()
            for reason in dict.fromkeys(self.gate_reasons)
            if reason.strip()
        ]
        self.aliases = [
            alias.strip()
            for alias in dict.fromkeys(self.aliases)
            if alias.strip()
        ]
        return self


class ContentAliasRecord(BaseModel):
    """Search/catalog alias pointing at a compiled card ref."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    alias: str
    ref: str
    kind: str = "alias"
    confidence: float = 1.0
    review_status: ReviewStatus = "unreviewed"

    @model_validator(mode="after")
    def _clean(self) -> "ContentAliasRecord":
        self.pack_id = self.pack_id.strip()
        self.alias = self.alias.strip()
        self.ref = self.ref.strip()
        self.kind = self.kind.strip().lower() or "alias"
        self.confidence = _clamp_confidence(self.confidence)
        return self


class CoverageGateResult(BaseModel):
    """A deterministic preflight result for serving a compiled record."""

    model_config = ConfigDict(extra="forbid")

    status: CoverageGateStatus
    allowed: bool
    reasons: list[str] = Field(default_factory=list)


class CoverageManifest(BaseModel):
    """Pack-level import coverage summary persisted beside compiled records."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str
    pack_version: str = ""
    source_fingerprint: str = ""
    importer_version: str = ""
    schema_version: str = "content-pack-v1"
    source_page_count: int = 0
    compiled_page_count: int = 0
    card_count: int = 0
    alias_count: int = 0
    ready_count: int = 0
    flagged_count: int = 0
    blocked_count: int = 0
    low_confidence_count: int = 0
    high_spoiler_count: int = 0
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "CoverageManifest":
        self.pack_id = self.pack_id.strip()
        self.pack_version = self.pack_version.strip()
        self.source_fingerprint = self.source_fingerprint.strip()
        self.importer_version = self.importer_version.strip()
        self.schema_version = self.schema_version.strip() or "content-pack-v1"
        self.warnings = [
            warning.strip()
            for warning in dict.fromkeys(self.warnings)
            if warning.strip()
        ]
        for field_name in (
            "source_page_count",
            "compiled_page_count",
            "card_count",
            "alias_count",
            "ready_count",
            "flagged_count",
            "blocked_count",
            "low_confidence_count",
            "high_spoiler_count",
        ):
            if getattr(self, field_name) < 0:
                setattr(self, field_name, 0)
        return self


class ContentImageAsset(BaseModel):
    """Private asset catalog row addressed by stable ids, not file paths."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    asset_id: str
    kind: str = "image"
    title: str = ""
    mime_type: str = ""
    width: int = 0
    height: int = 0
    sha256: str = ""
    source_ref: str = ""
    review_status: ReviewStatus = "unreviewed"
    spoiler_class: SpoilerClass = "none"
    player_safe_alt_text: str = ""
    player_safe_caption: str = ""
    delivery_ref: str = ""
    safe_for_players: bool = False
    safe_for_llm: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "ContentImageAsset":
        self.pack_id = self.pack_id.strip()
        self.asset_id = self.asset_id.strip()
        self.kind = self.kind.strip().lower() or "image"
        self.title = self.title.strip()
        self.mime_type = self.mime_type.strip()
        self.sha256 = self.sha256.strip()
        self.source_ref = self.source_ref.strip()
        self.player_safe_alt_text = self.player_safe_alt_text.strip()
        self.player_safe_caption = self.player_safe_caption.strip()
        self.delivery_ref = self.delivery_ref.strip()
        if self.width < 0:
            self.width = 0
        if self.height < 0:
            self.height = 0
        return self


class AssetReveal(BaseModel):
    """Router-owned image/map reveal request using observable-fact visibility."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    asset_id: str
    audience: AssetRevealAudience = "all_observers"
    visible_to_character_ids: list[str] = Field(default_factory=list)
    visible_to_user_ids: list[str] = Field(default_factory=list)
    presentation: AssetPresentation = "reference"
    caption: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "AssetReveal":
        self.pack_id = self.pack_id.strip()
        self.asset_id = self.asset_id.strip()
        self.visible_to_character_ids = [
            value.strip()
            for value in dict.fromkeys(self.visible_to_character_ids)
            if value.strip()
        ]
        self.visible_to_user_ids = [
            value.strip()
            for value in dict.fromkeys(self.visible_to_user_ids)
            if value.strip()
        ]
        self.caption = self.caption.strip()
        return self


class SafeAssetRevealPayload(BaseModel):
    """Player/LLM-safe reveal payload. No source paths, notes, or bytes."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str
    asset_id: str
    kind: str
    title: str = ""
    mime_type: str = ""
    width: int = 0
    height: int = 0
    sha256: str = ""
    delivery_ref: str
    presentation: AssetPresentation = "reference"
    caption: str = ""
    alt_text: str = ""


def _clamp_confidence(value: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence
