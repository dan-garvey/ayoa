"""Schemas for the /join_custom flow.

Three modes, one prompt (`takeover_v1`), three output shapes:

- **describe** — user hands the router a character concept. The router
  spins up a full character record (name, role, backstory, objectives,
  placement) that fits the world. No existing NPC is touched.

- **suggest** — user hands the router a character concept plus "I want
  to replace someone." The router surveys the roster and returns 2–4
  candidates with rationale. No mutation.

- **replace** — user picks one candidate from a prior `suggest` call.
  The router returns the final authored character (same shape as
  describe) which the engine will graft onto the chosen NPC's slot —
  name and appearance overwritten, everything else (location,
  objectives, directives) preserved.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.characters import (
    CharacterRecord,
    PrivateState,
    PublicSheet,
)


class ReplacementCandidate(BaseModel):
    """One entry in a suggest-mode response."""
    model_config = ConfigDict(extra="forbid")

    character_id: str
    name: str
    fit_rationale: str = Field(
        description=(
            "Why this NPC would be an interesting or appropriate replacement "
            "for the user's concept, and what narrative consequences the "
            "takeover implies. Player-facing; keep it tight and honest — one "
            "to three sentences."
        ),
    )


class TakeoverSuggestOutput(BaseModel):
    """Router output for mode='suggest'. A short list of candidate NPCs
    the router thinks would work as replacement targets for the user's
    described concept."""
    model_config = ConfigDict(extra="forbid")

    candidates: list[ReplacementCandidate] = Field(default_factory=list)
    # Optional preface the user sees above the picker. Empty string if
    # nothing to add. Kept as a single string so the CLI/Discord layers
    # can render it plainly.
    preamble: str = ""


class AuthoredCharacter(BaseModel):
    """Flat authoring shape for LLM output. Kept intentionally simple —
    the full CharacterRecord has an enum field (status), nested queue
    lists (incoming_directives, pending_observations), and nested
    sub-models (PublicSheet, PrivateState) that pushed Anthropic's
    structured-output grammar compiler past its server-side deadline.
    Flattening into a single-level struct makes the grammar compile
    fast enough, and the engine maps the result back into a
    CharacterRecord before persisting."""
    model_config = ConfigDict(extra="forbid")

    name: str
    location: str = ""
    # public_sheet fields
    role: str = ""
    traits: list[str] = Field(default_factory=list)
    voice: str = ""
    appearance: str = ""
    faction: str = ""
    # long-form interior prose
    backstory: str = ""
    personality: str = ""
    narrative_notes: str = ""
    known_context: str = ""
    # private_state fields
    goals: list[str] = Field(default_factory=list)
    current_objectives: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    intentions_enabled: bool = False

    def to_record(self, character_id: str = "") -> CharacterRecord:
        """Map the authoring shape onto the engine's CharacterRecord.
        Caller sets character_id, is_player, and bound_user_id afterward
        if relevant."""
        return CharacterRecord(
            character_id=character_id,
            name=self.name,
            location=self.location,
            public_sheet=PublicSheet(
                role=self.role,
                traits=list(self.traits),
                voice=self.voice,
                appearance=self.appearance,
                faction=self.faction,
            ),
            private_state=PrivateState(
                goals=list(self.goals),
                current_objectives=list(self.current_objectives),
                secrets=list(self.secrets),
                intentions_enabled=self.intentions_enabled,
            ),
            backstory=self.backstory,
            personality=self.personality,
            narrative_notes=self.narrative_notes,
            known_context=self.known_context,
        )


class TakeoverAuthoredOutput(BaseModel):
    """Router output for mode='describe' or mode='replace'. Returns a
    flat AuthoredCharacter; the engine maps it onto a CharacterRecord
    before persisting."""
    model_config = ConfigDict(extra="forbid")

    character: AuthoredCharacter
    # Short in-fiction note the router wants attached to the session log
    # so the next turn has continuity with the new character's arrival.
    # Empty string if nothing to add.
    session_note: str = ""
