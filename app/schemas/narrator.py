from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TranscriptEntry(BaseModel):
    user: str
    assistant: str


class NarratorFinalOutput(BaseModel):
    """Produced by Narrator Phase 2. LLM output target."""

    model_config = ConfigDict(extra="forbid")

    final_text: str
    world_updates: dict[str, Any] = Field(default_factory=dict)
    transcript_entry: TranscriptEntry
