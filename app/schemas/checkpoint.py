from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.characters import CharacterRecord
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import DndEventRouterOutput, EventRouterOutput
from app.schemas.narrator import TranscriptEntry
from app.schemas.state import SessionConfig, SessionState, WorldState


CURRENT_SCHEMA_VERSION = "4.0"


class CheckpointFile(BaseModel):
    # v11 schema is 4.0 (relative-time + private commitments). Older
    # checkpoints are
    # HARD-BREAK on load — the loader in checkpoint_manager raises with a
    # message pointing the user at /story start. No migration shim.
    schema_version: str = CURRENT_SCHEMA_VERSION
    session: SessionState
    # 1-2 paragraph player-facing world primer: the first thing a fresh
    # player sees after /story start. Distinct from the omniscient dossier
    # (which leaked spoilers). Synthetic story checkpoints should author
    # this directly. Empty checkpoints render a fallback stub.
    #
    # Note: there is no longer an authored `opening_narrative` field —
    # the opening beat is composed at runtime by the router (using
    # world_state, character_records, and the `(begin)` OOC directive)
    # and rendered per-POV by the narrator on the first turn. This
    # keeps every turn on a single code path and avoids the POV-binding
    # and race-window problems of an authored opener; see commit log
    # for the rationale.
    player_primer: str = ""
    world_state: WorldState = Field(default_factory=WorldState)
    characters: list[CharacterRecord] = Field(default_factory=list)
    # Rolling conversation histories: each role sees the full prior exchange
    # on every call, so continuity and caching both work.
    session_conversation: list[ConversationMessage] = Field(default_factory=list)
    # v11: per-character narrator rolling history. Each human (by their
    # bound character_id) has their own stream; the narrator_phase2 call
    # reads character_id-keyed history. The old session-wide
    # narrator_conversation is gone in the v11 pipeline.
    narrator_conversations: dict[str, list[ConversationMessage]] = Field(default_factory=dict)
    character_conversations: dict[str, list[ConversationMessage]] = Field(default_factory=dict)
    # v11: the canonical event log. Every closed canonical event appended
    # here. Source of truth for rendering, replay, and debug.
    canonical_events: list[DndEventRouterOutput | EventRouterOutput] = Field(
        default_factory=list
    )
    # Display/audit log only — no longer fed into prompts.
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    visibility_log: list[dict[str, Any]] = Field(default_factory=list)
    config: SessionConfig = Field(default_factory=SessionConfig)
