from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.characters import CharacterRecord
from app.schemas.conversation import ConversationMessage
from app.schemas.narrator import TranscriptEntry
from app.schemas.state import SessionConfig, SessionState, WorldState


class CheckpointFile(BaseModel):
    schema_version: str = "2.0"
    session: SessionState
    opening_narrative: str = ""
    world_state: WorldState = Field(default_factory=WorldState)
    characters: list[CharacterRecord] = Field(default_factory=list)
    # Rolling conversation histories: each role sees the full prior exchange
    # on every call, so continuity and caching both work.
    session_conversation: list[ConversationMessage] = Field(default_factory=list)
    narrator_conversation: list[ConversationMessage] = Field(default_factory=list)
    character_conversations: dict[str, list[ConversationMessage]] = Field(default_factory=dict)
    # Display/audit log only — no longer fed into prompts.
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    visibility_log: list[dict[str, Any]] = Field(default_factory=list)
    config: SessionConfig = Field(default_factory=SessionConfig)
    prompt_versions: dict[str, str] = Field(default_factory=lambda: {
        "event_router": "v3",
        "agent": "v5",
        "narrator_phase2": "v4",
    })
