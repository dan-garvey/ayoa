from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PublicResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[str] = Field(default_factory=list)
    dialogue: list[str] = Field(default_factory=list)
    expression: str = ""


class PrivateUpdates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intentions: list[str] = Field(default_factory=list)
    attitude_delta: dict[str, float] = Field(default_factory=dict)


class CharacterAgentOutput(BaseModel):
    """Produced by a Character Agent. LLM output target."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    public_response: PublicResponse = Field(default_factory=PublicResponse)
    private_updates: PrivateUpdates = Field(default_factory=PrivateUpdates)
    memory_writes: list[str] = Field(default_factory=list)
