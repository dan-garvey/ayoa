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
