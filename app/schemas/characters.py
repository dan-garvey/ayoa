from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CharacterStatus(str, Enum):
    active = "active"
    dormant = "dormant"
    culled = "culled"


class PublicSheet(BaseModel):
    role: str = ""
    traits: list[str] = Field(default_factory=list)
    voice: str = ""
    appearance: str = ""
    faction: str = ""


class CharacterMemory(BaseModel):
    episodic: list[str] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)
    observation_queue: list[str] = Field(default_factory=list)


class PrivateState(BaseModel):
    goals: list[str] = Field(default_factory=list)
    # Keys can be "user" or any character_id for inter-character attitudes
    attitudes: dict[str, float] = Field(default_factory=dict)
    secrets: list[str] = Field(default_factory=list)
    intentions_enabled: bool = False


class CharacterRecord(BaseModel):
    character_id: str
    name: str
    status: CharacterStatus = CharacterStatus.active
    location: str = ""
    public_sheet: PublicSheet = Field(default_factory=PublicSheet)
    private_state: PrivateState = Field(default_factory=PrivateState)
    memory: CharacterMemory = Field(default_factory=CharacterMemory)
    # Long-form text fields for rich character content
    backstory: str = ""
    personality: str = ""
    narrative_notes: str = ""
