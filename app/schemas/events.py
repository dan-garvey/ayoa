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
    new_scene_id: str = ""


class CanonicalEvent(BaseModel):
    """Produced by the event router's adjudication pass. LLM output target.

    `user_intent` was dropped in favor of `world_adjudication.attempted_action`
    — the latter is the normalized form the pipeline actually uses. `event_id`
    was dropped too; the orchestrator tags the visibility log directly from
    turn_index so there's no need for the router to emit one."""

    model_config = ConfigDict(extra="forbid")

    world_adjudication: WorldAdjudication
    scene_delta: SceneDelta = Field(default_factory=SceneDelta)
    observable_facts: list[str] = Field(default_factory=list)
