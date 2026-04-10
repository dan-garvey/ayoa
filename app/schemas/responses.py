from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DebugPayload(BaseModel):
    canonical_event: dict[str, Any] = Field(default_factory=dict)
    discriminator: dict[str, Any] = Field(default_factory=dict)
    agent_outputs: list[dict[str, Any]] = Field(default_factory=list)
    world_updates: dict[str, Any] = Field(default_factory=dict)


class TurnResponse(BaseModel):
    session_id: str
    checkpoint_id: str = ""
    turn_index: int = 0
    output_text: str
    debug: DebugPayload | None = None
