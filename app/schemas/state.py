from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    narrator: str = "GPT-oss-120B"
    discriminator: str = "GPT-oss-120B"
    agent_default: str = "GPT-oss-120B"


class SessionConfig(BaseModel):
    models: ModelConfig = Field(default_factory=ModelConfig)
    debug: bool = False
    stream_mode: str = "final_only"
    # Long-form narrator style rules: prose discipline, pacing, subtext philosophy
    narrative_rules: str = ""


class SessionState(BaseModel):
    session_id: str
    story_id: str = ""
    turn_index: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    config: SessionConfig = Field(default_factory=SessionConfig)


class TimeState(BaseModel):
    scene_time: datetime = Field(default_factory=datetime.utcnow)
    turn_count: int = 0


class LocationState(BaseModel):
    current_scene_id: str = ""
    scene_graph: dict[str, Any] = Field(default_factory=dict)


class PhysicsRuleset(BaseModel):
    strength_limits: str = "human_baseline"
    magic_enabled: bool = False


class StorySetting(BaseModel):
    """Genre, era, and tone metadata — used to ground character genesis and narrator voice."""
    genre: str = ""
    era: str = ""
    tone: str = ""
    premise: str = ""


class WorldState(BaseModel):
    time: TimeState = Field(default_factory=TimeState)
    locations: LocationState = Field(default_factory=LocationState)
    facts: list[str] = Field(default_factory=list)
    physics_ruleset: PhysicsRuleset = Field(default_factory=PhysicsRuleset)
    global_flags: dict[str, Any] = Field(default_factory=dict)
    setting: StorySetting = Field(default_factory=StorySetting)
    # Long-form world lore: history, factions, laws, magic systems, etc.
    lore: str = ""
