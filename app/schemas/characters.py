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
    # True if this character is a human-player slot. The personalize flow
    # finds the player character by this flag (not by id suffix), and the
    # orchestrator excludes is_player characters from agent fan-out. Multi-
    # player stories can have multiple is_player=True entries.
    is_player: bool = False
    public_sheet: PublicSheet = Field(default_factory=PublicSheet)
    private_state: PrivateState = Field(default_factory=PrivateState)
    # Staging area for observations the character witnessed silently (turns where
    # they didn't respond). Flushed into the next agent user message when the
    # character is asked to respond, then cleared.
    pending_observations: list[str] = Field(default_factory=list)
    # Long-form text fields for rich character content
    backstory: str = ""
    personality: str = ""
    narrative_notes: str = ""
