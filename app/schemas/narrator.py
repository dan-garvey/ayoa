from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptEntry(BaseModel):
    user: str
    assistant: str


class NarratorFinalOutput(BaseModel):
    """Produced by Narrator Phase 2. LLM output target.

    The model emits a candidate passage plus its delivery judgment. The engine
    constructs transcript entries from the real player input and commits only
    passages whose handoff is accepted.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    handoff: Literal["render", "continue"]
    handoff_reason: str = Field(min_length=1, max_length=500)
    final_text: str
