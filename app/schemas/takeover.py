"""Structured output for custom-character authoring and replacement."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.characters import (
    ActorRecord,
    CharacterRecord,
    CharacterVisuals,
    PublicSheet,
)


class ReplacementCandidate(BaseModel):
    """One existing character a player could replace."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    name: str
    fit_rationale: str


class TakeoverSuggestOutput(BaseModel):
    """The non-mutating candidate response for takeover suggestion."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[ReplacementCandidate] = Field(default_factory=list)
    preamble: str = ""


class AuthoredCharacter(BaseModel):
    """One complete authoring result with public identity and sparse actor facts.

    Every field is required so the structured-output grammar has one stable
    shape. ``actor.facts`` is deliberately allowed to be empty: the author
    should retain only material that the supplied authority supports.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    location: str
    role: str
    appearance: str
    public_context: str
    default_loadout: str
    faction: str
    actor: ActorRecord
    # Frontend-created characters need a compact external update for the next
    # router turn; generated NPC spawns leave this empty.
    router_summary: str

    def to_record(self, character_id: str = "") -> CharacterRecord:
        """Map the authoring result directly onto the current record shape."""

        return CharacterRecord(
            character_id=character_id,
            name=self.name,
            location=self.location,
            public_sheet=PublicSheet(
                role=self.role,
                appearance=self.appearance,
                faction=self.faction,
                public_context=self.public_context,
            ),
            visuals=CharacterVisuals(
                default_loadout=self.default_loadout or self.appearance,
            ),
            actor=self.actor.model_copy(deep=True),
        )


class TakeoverAuthoredOutput(BaseModel):
    """The character result for describe and replace authoring modes."""

    model_config = ConfigDict(extra="forbid")

    character: AuthoredCharacter
    session_note: str = ""
