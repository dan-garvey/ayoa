from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PhaseLatency(BaseModel):
    """Latency info for a single pipeline phase."""
    phase: str
    duration_ms: float
    model: str = ""


class DebugPayload(BaseModel):
    canonical_event: dict[str, Any] = Field(default_factory=dict)
    discriminator: dict[str, Any] = Field(default_factory=dict)
    agent_outputs: list[dict[str, Any]] = Field(default_factory=list)
    world_updates: dict[str, Any] = Field(default_factory=dict)
    latencies: list[PhaseLatency] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    models_used: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    validations: list[dict[str, Any]] = Field(default_factory=list)


class TurnResponse(BaseModel):
    session_id: str
    checkpoint_id: str = ""
    turn_index: int = 0
    output_text: str
    debug: DebugPayload | None = None
