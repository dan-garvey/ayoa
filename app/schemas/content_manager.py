from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.content import ContentKnowledgeUpdateOperation
from app.schemas.content_privacy import contains_imported_asset_sentinel


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")


def _clean_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return "" if contains_imported_asset_sentinel(text) else text


def _clean_token(value: Any) -> str:
    text = _clean_text(value)
    return text if text and _SAFE_TOKEN_RE.fullmatch(text) else ""


class _ContentManagerRefMixin(BaseModel):
    pack_id: str
    ref: str
    content_hash: str = ""
    reason: str = ""
    source_fact_ids: list[str] = Field(default_factory=list)

    def _clean_ref_fields(self) -> None:
        self.pack_id = _clean_token(self.pack_id)
        self.ref = _clean_token(self.ref)
        self.content_hash = _clean_token(self.content_hash)
        self.reason = _clean_text(self.reason)
        self.source_fact_ids = [
            fact_id
            for value in dict.fromkeys(self.source_fact_ids)
            if (fact_id := _clean_token(value))
        ]
        if not self.pack_id:
            raise ValueError("Content manager ref requires pack_id")
        if not self.ref:
            raise ValueError("Content manager ref requires ref")

    def compact_ref(self) -> str:
        return (
            f"{self.pack_id}:{self.ref}@{self.content_hash}"
            if self.content_hash
            else f"{self.pack_id}:{self.ref}"
        )


class ContentManagerKnowledgeUpdate(_ContentManagerRefMixin):
    """Patch operation against the engine-owned knowledge map."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    operation: ContentKnowledgeUpdateOperation = "mark_known"

    @model_validator(mode="after")
    def _clean(self) -> "ContentManagerKnowledgeUpdate":
        self.entity_id = _clean_token(self.entity_id)
        self._clean_ref_fields()
        if not self.entity_id:
            raise ValueError("Content manager knowledge update requires entity_id")
        return self

    def dedupe_key(self) -> tuple[str, str, str, str]:
        return (self.entity_id, self.pack_id, self.ref, self.operation)


class ContentManagerRouterRequiredKey(BaseModel):
    """Reviewed router knowledge key required for the next router call."""

    model_config = ConfigDict(extra="forbid")

    key: str
    reason: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "ContentManagerRouterRequiredKey":
        self.key = _clean_token(self.key)
        self.reason = _clean_text(self.reason)
        if not self.key:
            raise ValueError("Content manager router key requires key")
        return self

    def dedupe_key(self) -> str:
        return self.key


class ContentManagerRouterTurnCandidate(BaseModel):
    """Non-binding hint that a candidate character may want router attention."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    priority: Literal["low", "medium", "high"] = "medium"
    reason: str = ""
    source_fact_ids: list[str] = Field(default_factory=list)
    related_content_refs: list[str] = Field(default_factory=list)
    related_keys: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentManagerRouterTurnCandidate":
        self.character_id = _clean_token(self.character_id)
        self.reason = _clean_text(self.reason)
        self.source_fact_ids = [
            fact_id
            for value in dict.fromkeys(self.source_fact_ids)
            if (fact_id := _clean_token(value))
        ]
        self.related_content_refs = [
            ref
            for value in dict.fromkeys(self.related_content_refs)
            if (ref := _clean_token(value))
        ]
        self.related_keys = [
            key
            for value in dict.fromkeys(self.related_keys)
            if (key := _clean_token(value))
        ]
        if not self.character_id:
            raise ValueError("Content manager router candidate requires character_id")
        return self

    def dedupe_key(self) -> str:
        return self.character_id


class ContentManagerAgentContextBroadcast(_ContentManagerRefMixin):
    """Rare validated ref that should be broadcast to a character agent."""

    model_config = ConfigDict(extra="forbid")

    character_id: str

    @model_validator(mode="after")
    def _clean(self) -> "ContentManagerAgentContextBroadcast":
        self.character_id = _clean_token(self.character_id)
        self._clean_ref_fields()
        if not self.character_id:
            raise ValueError("Content manager agent broadcast requires character_id")
        return self

    def dedupe_key(self) -> tuple[str, str, str]:
        return (self.character_id, self.pack_id, self.ref)


class ContentManagerOutput(BaseModel):
    """Structured content-manager response."""

    model_config = ConfigDict(extra="forbid")

    knowledge_updates: list[ContentManagerKnowledgeUpdate] = Field(
        default_factory=list
    )
    router_required_keys: list[ContentManagerRouterRequiredKey] = Field(
        default_factory=list
    )
    router_turn_candidates: list[ContentManagerRouterTurnCandidate] = Field(
        default_factory=list
    )
    agent_context_broadcasts: list[ContentManagerAgentContextBroadcast] = Field(
        default_factory=list
    )
    no_update_reason: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "ContentManagerOutput":
        self.no_update_reason = _clean_text(self.no_update_reason)
        self.knowledge_updates = list({
            update.dedupe_key(): update for update in self.knowledge_updates
        }.values())
        self.router_required_keys = list({
            item.dedupe_key(): item for item in self.router_required_keys
        }.values())
        self.router_turn_candidates = list({
            candidate.dedupe_key(): candidate
            for candidate in self.router_turn_candidates
        }.values())
        self.agent_context_broadcasts = list({
            item.dedupe_key(): item for item in self.agent_context_broadcasts
        }.values())
        return self
