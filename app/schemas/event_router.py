from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.events import CanonicalEvent


class ObserverEntry(BaseModel):
    """Which characters observed the event and how strongly they should
    respond. Observers at priority 0 should be omitted entirely — they're
    routing noise, not silent responders.

    Observation level is a single-char enum: "d" = direct (in the scene),
    "i" = indirect (adjacent, heard/saw spillover), "f" = inferred
    (aftermath or ambient inference only). Agents see `observable_facts`
    from the canonical event; the level is used downstream to filter that
    set, not to duplicate it per-observer."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    observation_level: str = "d"
    # 1=minimal, 2=low, 3=moderate, 4=high, 5=compelled. Omit 0-priority
    # observers from the output entirely.
    response_priority: int = 1


class SpawnRequest(BaseModel):
    """Router-directed creation of a new character. `seed` is a freeform
    dict (role, reason, location, objectives, ...) consumed by
    character_gen."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    seed: dict[str, Any] = Field(default_factory=dict)


class RosterMove(BaseModel):
    """Router-directed movement of an existing character between scenes.
    Applied by the orchestrator — updates the target character's
    `location` field. Empty list on turns where nobody's moving."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    to_scene: str  # scene_id (must exist in scene_graph)
    reason: str = ""


class SceneCreation(BaseModel):
    """Router-directed creation of a new scene.

    The scene graph is seeded at import time from the master prompt, but
    stories naturally imply spaces the author didn't explicitly enumerate
    — a side room, a vista a balcony overlooks, a shop the player glances
    into. When a player action requires a scene that doesn't exist yet,
    the router creates it here rather than silently dropping the move.

    Applied by the orchestrator BEFORE any movement logic on the same
    turn, so `scene_delta.new_scene_id`, `roster_moves.to_scene`, and
    spawn seed locations may reference a scene this list introduces.

    Reverse edges are added automatically — if this scene connects to
    `hallway`, the orchestrator also adds this scene's id to
    `hallway.connected_to`. The graph stays traversable in both
    directions without the router having to spell that out."""
    model_config = ConfigDict(extra="forbid")

    scene_id: str  # snake_case, unique across the graph
    name: str
    description: str = ""
    # Scene_ids this new scene is directly reachable from. Entries may
    # reference existing scenes OR other scenes in the same
    # scenes_created batch (for newly-paired adjacent spaces).
    connected_to: list[str] = Field(default_factory=list)


class EventRouterOutput(BaseModel):
    """Merged adjudication + perception output — AND in v11, the router's
    beat-pacing decision.

    v11 additions:
      - `requires_responders` + `required_responders`: Cat I (self-closing)
        vs Cat II (contested). Cat II events open and collect responder
        intentions before canonicalization closes.
      - `agent_responder_picks`: NPCs the router wants to cascade into the
        current beat. Capped by settings.max_responders.
      - `ends_beat` + `ends_beat_reason`: the router's DM-pacing signal —
        when true, the beat composes its buffered events into a render
        and the scene's active_act_slot is released. Cat II adjudication
        always ends the beat (implicit, regardless of this field). For
        Cat I events the router decides explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    canonical_event: CanonicalEvent

    # ---- v11: Cat I / Cat II intention classification --------------------
    # When True, this intention is CONTESTED: a canonical event cannot close
    # without the listed required_responders also intending. Examples:
    # violence, consensual physical contact, contested possession, forced
    # movement through a blocker. When False (Cat I), the intention closes
    # immediately — dialogue, passive action, unambiguous movement, OOC
    # directives.
    requires_responders: bool = False
    # Characters whose intentions must be collected before this Cat II
    # event adjudicates. Includes the direct target plus any plausible
    # intercepter / defender / counter-actor. Empty for Cat I.
    required_responders: list[str] = Field(default_factory=list)

    # ---- v11: post-canonicalization agent cascade ------------------------
    # Router-selected NPC agents to dispatch into the current beat as
    # reactive intentions. Cap: clamped to settings.max_responders at the
    # orchestrator layer. Humans are NEVER in this list; humans only
    # enter via /act, gated by active_act_slot. Empty when the router
    # thinks no NPC cascade is warranted.
    agent_responder_picks: list[str] = Field(default_factory=list)

    # ---- v11: DM pacing — end beat now, or let the cascade continue? ----
    # True = render now; false = router will pick next actor and continue.
    # Cat II adjudication implicitly ends the beat regardless of this
    # value. For Cat I events this is the router's judgment call.
    ends_beat: bool = True
    # Short tag for telemetry: "directed_at_player" | "scene_transition" |
    # "state_change" | "cascade_exhausted" | "cat_ii_resolution" |
    # "cat_ii_open" | "max_events_per_beat" | "ambient_pause". Empty
    # string is acceptable — used as a free-form override.
    ends_beat_reason: str = ""

    # ---- Legacy observation / roster plumbing (unchanged) ---------------
    # `observers` is retained for the render-buffer determination: every
    # character in the observer list gets this event into their render
    # buffer if they're human. Agents get it as observation context for
    # future intend() calls. The router_responder_picks above is a
    # DIFFERENT decision — who actually fires next, a subset (or
    # superset) of observers.
    observers: list[ObserverEntry] = Field(default_factory=list)
    spawn: list[SpawnRequest] = Field(default_factory=list)
    dormant: list[str] = Field(default_factory=list)
    cull: list[str] = Field(default_factory=list)
    roster_moves: list[RosterMove] = Field(default_factory=list)
    scenes_created: list[SceneCreation] = Field(default_factory=list)
