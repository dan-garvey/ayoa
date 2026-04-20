from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.discriminator import DiscriminatorOutput, ObserverEntry, SpawnRequest
from app.schemas.events import CanonicalEvent


class RosterMove(BaseModel):
    """Router-directed character movement between scenes. Applied by the
    orchestrator — updates the target character's `location` field. Empty
    list on turns where nobody's moving."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    to_scene: str  # scene_id (must exist in scene_graph)
    reason: str = ""


class EventRouterOutput(BaseModel):
    """Merged adjudication + perception output."""

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

    def to_discriminator_output(self) -> DiscriminatorOutput:
        """Project the merged output onto the legacy discriminator schema."""
        return DiscriminatorOutput(
            event_id=self.canonical_event.event_id,
            observers=self.observers,
            suggested_response_cap=self.suggested_response_cap,
            spawn=self.spawn,
            dormant=self.dormant,
            cull=self.cull,
        )
