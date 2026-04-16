from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ObserverEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    observation_level: str = "direct"
    facts: list[str] = Field(default_factory=list)
    response_priority: int = 0  # 0=silent, 1=minimal, 2=low, 3=moderate, 4=high, 5=compelled


class SpawnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    seed: dict[str, Any] = Field(default_factory=dict)


class DiscriminatorOutput(BaseModel):
    """Produced by the Discriminator. LLM output target."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = ""
    observers: list[ObserverEntry] = Field(default_factory=list)
    suggested_response_cap: int = 2
    spawn: list[SpawnRequest] = Field(default_factory=list)
    dormant: list[str] = Field(default_factory=list)
    cull: list[str] = Field(default_factory=list)
