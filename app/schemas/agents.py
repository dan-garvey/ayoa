"""Engine-internal records for character-agent turns.

Character turns are ordinary observable prose or an explicit ``<silence/>``
beat. Empty and presentation-only turn responses are invalid because they
cannot consume causal input without an observable choice.

The LLM does not target ``CharacterAgentOutput`` directly. The engine parses
the free-form response before constructing this record, and rejects retired
private markers before anything is committed or exposed publicly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CharacterPresentationChoice(BaseModel):
    """One character-authored outward display plus an optional future request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    use: str = Field(default="", max_length=80)
    request: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def _normalize(self) -> "CharacterPresentationChoice":
        self.use = self.use.strip().lower()
        self.request = " ".join(self.request.split()).strip()
        return self


class CharacterPerceptionOutput(BaseModel):
    """Parsed visual-loadout prose and its character-owned display choice."""

    model_config = ConfigDict(extra="forbid")

    public_text: str
    presentation: CharacterPresentationChoice = Field(
        default_factory=CharacterPresentationChoice
    )


class CharacterAgentOutput(BaseModel):
    """Engine-internal record of one parsed character-agent response."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    public_text: str
    is_silence: bool = False
    presentation: CharacterPresentationChoice = Field(
        default_factory=CharacterPresentationChoice
    )
