from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.characters import CharacterRecord
from app.schemas.narrator import TranscriptEntry
from app.schemas.state import SessionConfig, SessionState, WorldState


class CheckpointFile(BaseModel):
    schema_version: str = "1.0"
    session: SessionState
    world_state: WorldState = Field(default_factory=WorldState)
    characters: list[CharacterRecord] = Field(default_factory=list)
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    visibility_log: list[dict[str, Any]] = Field(default_factory=list)
    config: SessionConfig = Field(default_factory=SessionConfig)
    prompt_versions: dict[str, str] = Field(default_factory=lambda: {
        "narrator": "v1",
        "discriminator": "v1",
        "agent": "v1",
    })
