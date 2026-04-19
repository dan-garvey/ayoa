from __future__ import annotations

from pydantic import BaseModel, Field


class DebugFlags(BaseModel):
    include_discriminator: bool = False
    include_agent_outputs: bool = False
    include_internal_state_deltas: bool = False


class PersonalizeRequest(BaseModel):
    session_id: str
    player_name: str


class TurnRequest(BaseModel):
    session_id: str
    checkpoint_id: str | None = None
    user_input: str
    # Which character is taking this turn. Empty falls back to
    # session.player_character_id so legacy single-player call sites keep
    # working without change. The orchestrator resolves this against the
    # roster each turn to build the {acting_character_name} prompt slot.
    acting_character_id: str = ""
    stream: bool = False
    debug: bool = False
    debug_flags: DebugFlags = Field(default_factory=DebugFlags)
