from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WorldAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempted_action: str
    feasible: bool
    resolved_outcome: str


class SceneDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_advanced_seconds: int = 0


class CanonicalEvent(BaseModel):
    """Produced by Narrator Phase 1. LLM output target."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = ""
    user_intent: str
    world_adjudication: WorldAdjudication
    scene_delta: SceneDelta = Field(default_factory=SceneDelta)
    observable_facts: list[str] = Field(default_factory=list)
