"""Schemas for the /join_custom flow.

Three modes, one prompt (`takeover_v1`), three output shapes:

- **describe** — user hands the router a character concept. The router
  spins up a full character record (name, role, backstory, objectives,
  attitudes, placement) that fits the world. No existing NPC is touched.

- **suggest** — user hands the router a character concept plus "I want
  to replace someone." The router surveys the roster and returns 2–4
  candidates with rationale. No mutation.

- **replace** — user picks one candidate from a prior `suggest` call.
  The router returns the final authored character (same shape as
  describe) which the engine will graft onto the chosen NPC's slot —
  name and appearance overwritten, everything else (location,
  attitudes toward them, objectives, directives) preserved.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.characters import CharacterRecord


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


class TakeoverAuthoredOutput(BaseModel):
    """Router output for mode='describe' or mode='replace'. A complete
    CharacterRecord ready to be spawned (describe) or grafted onto an
    existing slot (replace). The caller is responsible for assigning
    character_id, bound_user_id, and is_player before persisting."""
    model_config = ConfigDict(extra="forbid")

    character: CharacterRecord
    # Short in-fiction note the router wants attached to the session log
    # so the next turn has continuity with the new character's arrival.
    # Empty string if nothing to add.
    session_note: str = ""
