from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.event_router import SceneCreation


class PublicResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[str] = Field(default_factory=list)
    dialogue: list[str] = Field(default_factory=list)
    expression: str = ""


class DirectiveSend(BaseModel):
    """A message this character is sending to another character.

    Lands on the target's `incoming_directives` queue and is flushed on
    the target's next response or tick. Targets may be NPCs (drives
    off-stage AI behavior) or player-bound characters (rendered as an
    observable fact on the player's next turn — note, whisper, etc.).
    """
    model_config = ConfigDict(extra="forbid")

    to: str  # target character_id
    content: str


class PrivateUpdates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Agent-authoritative: each response emits the FULL current objectives
    # list, replacing any prior list. Completions drop off, revisions land
    # in place, new sub-goals get added. Whole-list replacement keeps the
    # schema simple and avoids diff machinery.
    current_objectives: list[str] = Field(default_factory=list)
    attitude_delta: dict[str, float] = Field(default_factory=dict)
    directives_sent: list[DirectiveSend] = Field(default_factory=list)
    # Self-initiated movement — typically emitted during off-stage ticks
    # ("I go to the garden to wait"). Must be a valid scene_id. Ignored
    # when empty. Applied after the public_response lands.
    moved_to: str = ""
    # Scenes the agent is inventing on a tick — only honored when
    # session setting `agents_can_create_scenes` is True, and only on
    # the tick path (during respond() the router owns world topology).
    # Applied BEFORE moved_to on the same tick, so `moved_to` may
    # reference a scene this list introduces. Same shape and
    # bidirectional-edge handling as the router's scenes_created.
    scenes_created: list[SceneCreation] = Field(default_factory=list)


class CharacterAgentOutput(BaseModel):
    """Produced by a Character Agent. LLM output target."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    public_response: PublicResponse = Field(default_factory=PublicResponse)
    private_updates: PrivateUpdates = Field(default_factory=PrivateUpdates)
