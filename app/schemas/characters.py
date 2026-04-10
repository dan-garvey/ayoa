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


class CharacterMemory(BaseModel):
    episodic: list[str] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)


class PrivateState(BaseModel):
    goals: list[str] = Field(default_factory=list)
    attitudes: dict[str, float] = Field(default_factory=dict)
    intentions_enabled: bool = False


class CharacterRecord(BaseModel):
    character_id: str
    name: str
    status: CharacterStatus = CharacterStatus.active
    location: str = ""
    public_sheet: PublicSheet = Field(default_factory=PublicSheet)
    private_state: PrivateState = Field(default_factory=PrivateState)
    memory: CharacterMemory = Field(default_factory=CharacterMemory)
