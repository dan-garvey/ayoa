from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.events import CanonicalEvent


class ObserverEntry(BaseModel):
    """Which characters observed the event, at what fidelity, and how
    strongly the fiction expects them to respond."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    observation_level: str = "direct"
    facts: list[str] = Field(default_factory=list)
    # 0=silent, 1=minimal, 2=low, 3=moderate, 4=high, 5=compelled
    response_priority: int = 0


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
    """Merged adjudication + perception output. Single source of truth
    for what happened this turn, who noticed, and how the roster shifts.

    (Previously split between a standalone DiscriminatorOutput and this
    merged shape; the discriminator role has been folded into the event
    router and the projection no longer exists.)"""

    model_config = ConfigDict(extra="forbid")

    canonical_event: CanonicalEvent
    observers: list[ObserverEntry] = Field(default_factory=list)
    suggested_response_cap: int = 2
    spawn: list[SpawnRequest] = Field(default_factory=list)
    dormant: list[str] = Field(default_factory=list)
    cull: list[str] = Field(default_factory=list)
    # NPC-to-scene movement, applied after the event itself. Lets the
    # router animate the world — characters arrive, depart, go off-stage.
    # Spawns use `spawn` for new characters; roster_moves is for existing
    # characters changing scenes.
    roster_moves: list[RosterMove] = Field(default_factory=list)
    # New scenes the router introduces on this turn. Applied before any
    # movement logic so scene_delta/roster_moves/spawn seeds can target
    # them. Empty on turns where no new spaces are needed.
    scenes_created: list[SceneCreation] = Field(default_factory=list)
