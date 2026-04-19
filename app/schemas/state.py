from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    event_router: str = "claude-sonnet-4-6"
    narrator: str = "claude-sonnet-4-6"
    discriminator: str = "claude-sonnet-4-6"
    agent_default: str = "claude-sonnet-4-6"


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
    player_name: str = ""
    player_character_id: str = ""
    # Freeform description of the player character's physical presence: height,
    # build, hair, clothing, voice quality, notable features. Plumbed into the
    # system prompts of all three engine roles so everyone references the
    # character consistently. Empty until the player runs /describe.
    player_character_description: str = ""
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
    # Hidden lore/facts — available to discriminator and agents for authentic
    # reactions, but NEVER shown to the narrator or the player. These contain
    # spoilers, conspiracy details, and secrets to be discovered through play.
    hidden_lore: str = ""
    hidden_facts: list[str] = Field(default_factory=list)
    # Characters the player has been formally introduced to (by character_id).
    # NP2 uses names only for known characters; others are described by appearance.
    known_characters: list[str] = Field(default_factory=list)
