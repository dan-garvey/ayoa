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


class ContentManagerContentUpdate(BaseModel):
    """One reviewed content ref that should become router-available."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str
    ref: str
    content_hash: str = ""
    update_kind: Literal["introduce", "refresh"] = "introduce"
    reason: str = ""
    source_fact_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentManagerContentUpdate":
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
            raise ValueError("Content manager content update requires pack_id")
        if not self.ref:
            raise ValueError("Content manager content update requires ref")
        return self

    def dedupe_key(self) -> tuple[str, str, str]:
        return (self.pack_id, self.ref, self.update_kind)


class ContentManagerTurnHint(BaseModel):
    """A non-binding hint that a candidate character may want router attention."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    priority: Literal["low", "medium", "high"] = "medium"
    reason: str = ""
    source_fact_ids: list[str] = Field(default_factory=list)
    related_content_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "ContentManagerTurnHint":
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
        if not self.character_id:
            raise ValueError("Content manager turn hint requires character_id")
        return self

    def dedupe_key(self) -> str:
        return self.character_id


class ContentManagerOutput(BaseModel):
    """Structured content-manager response."""

    model_config = ConfigDict(extra="forbid")

    content_updates: list[ContentManagerContentUpdate] = Field(default_factory=list)
    turn_hints: list[ContentManagerTurnHint] = Field(default_factory=list)
    no_update_reason: str = ""

    @model_validator(mode="after")
    def _clean(self) -> "ContentManagerOutput":
        self.no_update_reason = _clean_text(self.no_update_reason)
        content_updates: dict[
            tuple[str, str, str],
            ContentManagerContentUpdate,
        ] = {}
        for update in self.content_updates:
            content_updates.setdefault(update.dedupe_key(), update)
        self.content_updates = list(content_updates.values())

        turn_hints: dict[str, ContentManagerTurnHint] = {}
        for hint in self.turn_hints:
            turn_hints.setdefault(hint.dedupe_key(), hint)
        self.turn_hints = list(turn_hints.values())
        return self
