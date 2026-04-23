from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.schemas.event_router import SceneCreation


class PublicResponse(BaseModel):
    """The visible-to-the-scene part of an agent's turn.

    All three fields are REQUIRED on the wire so the LLM commits to its
    public surface every turn. Empty list / empty string is a valid
    commitment ("the character stays still and silent"), but the model
    isn't allowed to omit the fields and let a Pydantic default fill
    them in — defaults teach the LLM that fields are optional and bias
    it toward dropping them, which then makes downstream behavior
    inconsistent. Same rationale as EventRouterOutput's no-defaults
    contract."""

    model_config = ConfigDict(extra="forbid")

    actions: list[str]
    dialogue: list[str]
    expression: str


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
    """Off-screen updates an agent emits alongside its public response.

    All fields REQUIRED on the wire — the LLM emits empty list / empty
    string when it has nothing to add. The on-stage prompt (agent_v9)
    instructs the agent to leave `moved_to` and `scenes_created` empty
    on respond(); they're only populated on the off-stage tick path
    (agent_tick_v2). Defaults removed so the model commits explicitly
    every turn and doesn't drift into omitting fields it should at
    least mark as empty."""

    model_config = ConfigDict(extra="forbid")

    # Agent-authoritative: each response emits the FULL current objectives
    # list, replacing any prior list. Completions drop off, revisions land
    # in place, new sub-goals get added. Whole-list replacement keeps the
    # schema simple and avoids diff machinery.
    current_objectives: list[str]
    directives_sent: list[DirectiveSend]
    # Self-initiated movement — populated during off-stage ticks
    # ("I go to the garden to wait"). Must be a valid scene_id, or empty
    # string when the agent isn't moving. Applied after the public_response
    # lands. On-stage respond() ignores this field (movement is the
    # router's job there); the agent prompt tells the model to leave it
    # empty in that mode.
    moved_to: str
    # Scenes the agent is inventing on a tick — only honored when
    # session setting `agents_can_create_scenes` is True, and only on
    # the tick path. On-stage respond() ignores this field. Applied
    # BEFORE moved_to on the same tick, so `moved_to` may reference a
    # scene this list introduces. Same shape and bidirectional-edge
    # handling as the router's scenes_created.
    scenes_created: list[SceneCreation]


class CharacterAgentOutput(BaseModel):
    """Produced by a Character Agent. LLM output target.

    All fields REQUIRED. Same no-defaults discipline as
    EventRouterOutput — the LLM has to commit to character_id,
    public_response, and private_updates every turn rather than relying
    on Pydantic to backfill anything."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    public_response: PublicResponse
    private_updates: PrivateUpdates
