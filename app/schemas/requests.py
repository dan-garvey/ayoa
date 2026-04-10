from __future__ import annotations

from pydantic import BaseModel, Field


class DebugFlags(BaseModel):
    include_discriminator: bool = False
    include_agent_outputs: bool = False
    include_internal_state_deltas: bool = False


class TurnRequest(BaseModel):
    session_id: str
    checkpoint_id: str | None = None
    user_input: str
    stream: bool = False
    debug: bool = False
    debug_flags: DebugFlags = Field(default_factory=DebugFlags)
