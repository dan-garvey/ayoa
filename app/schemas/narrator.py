from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="after")
    def _require_accepted_render_text(self) -> "NarratorFinalOutput":
        if self.handoff == "render" and not self.final_text:
            raise ValueError("final_text must be non-empty when handoff='render'")
        return self
