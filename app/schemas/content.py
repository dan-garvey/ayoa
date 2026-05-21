from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ContentSignalStatus = Literal["pending", "resolved", "dismissed"]


def content_ref_key(pack_id: str, ref_id: str, content_hash: str) -> str:
    """Stable key for deduplicating a concrete content reference."""

    return "::".join(
        (
            (pack_id or "").strip(),
            (ref_id or "").strip(),
            (content_hash or "").strip(),
        )
    )


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


class ContentPackState(BaseModel):
    """Checkpoint state for one adventure/content pack."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str = ""
    introduced_refs: dict[str, IntroducedContentRef] = Field(default_factory=dict)
    pending_signals: dict[str, PendingContentSignal] = Field(default_factory=dict)
    fronts: dict[str, ContentFrontState] = Field(default_factory=dict)
    villains: dict[str, ContentVillainState] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clean(self) -> "ContentPackState":
        self.pack_id = self.pack_id.strip()
        return self
