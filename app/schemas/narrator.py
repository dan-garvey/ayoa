from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TranscriptEntry(BaseModel):
    user: str
    assistant: str


class NarratorFinalOutput(BaseModel):
    """Produced by Narrator Phase 2. LLM output target."""

    model_config = ConfigDict(extra="forbid")

    final_text: str
    transcript_entry: TranscriptEntry
