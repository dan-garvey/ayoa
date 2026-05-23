from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.content_privacy import contains_imported_asset_sentinel


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")


def _clean_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return "" if contains_imported_asset_sentinel(text) else text


def _clean_token(value: Any) -> str:
    text = _clean_text(value)
    return text if text and _SAFE_TOKEN_RE.fullmatch(text) else ""


class ContentManagerEntityUpdate(BaseModel):
    """One proposed content-ref knowledge update for one entity."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    pack_id: str
    ref: str
    content_hash: str = ""
    knowledge_state: Literal["known", "suspected"] = "known"
    reason: str = ""
    source_fact_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentManagerEntityUpdate":
        self.entity_id = _clean_token(self.entity_id)
        self.pack_id = _clean_token(self.pack_id)
        self.ref = _clean_token(self.ref)
        self.content_hash = _clean_token(self.content_hash)
        self.reason = _clean_text(self.reason)
        self.source_fact_ids = [
            fact_id
            for value in dict.fromkeys(self.source_fact_ids)
            if (fact_id := _clean_token(value))
        ]
        if not self.entity_id:
            raise ValueError("Content manager update requires entity_id")
        if not self.pack_id:
            raise ValueError("Content manager update requires pack_id")
        if not self.ref:
            raise ValueError("Content manager update requires ref")
        return self

    def dedupe_key(self) -> tuple[str, str, str, str]:
        return (self.entity_id, self.pack_id, self.ref, self.knowledge_state)


class ContentManagerOutput(BaseModel):
    """Structured content-manager response."""

    model_config = ConfigDict(extra="forbid")

    updates: list[ContentManagerEntityUpdate] = Field(default_factory=list)
    no_update_reason: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "ContentManagerOutput":
        self.no_update_reason = _clean_text(self.no_update_reason)
        deduped: dict[tuple[str, str, str, str], ContentManagerEntityUpdate] = {}
        for update in self.updates:
            deduped.setdefault(update.dedupe_key(), update)
        self.updates = list(deduped.values())
        return self
