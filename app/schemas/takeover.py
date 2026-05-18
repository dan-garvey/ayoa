"""Schemas for the /join_custom flow.

Three modes, one prompt (`takeover`), three output shapes:

- **describe** — user hands the router a character concept. The router
  spins up a full character record (name, role, backstory, objectives,
  placement) that fits the world. No existing NPC is touched.

- **suggest** — user hands the router a character concept plus "I want
  to replace someone." The router surveys the roster and returns 2–4
  candidates with rationale. No mutation.

- **replace** — user picks one candidate from a prior `suggest` call.
  The router returns the final authored character (same shape as
  describe) which the engine will graft onto the chosen NPC's slot —
  name and appearance overwritten, circumstances (location,
  objectives, pending observations) preserved.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.characters import (
    CharacterRecord,
    CharacterVisuals,
    PrivateState,
    PublicSheet,
)


# One entry in a suggest-mode response.
class ReplacementCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    name: str
    # Prompt instructs: "1-3 sentences on why this is a good replacement
    # AND what narrative consequences follow." Description kept out of
    # the schema to minimize grammar-compile cost.
    fit_rationale: str


# Router output for mode='suggest'. A short list of candidate NPCs the
# router thinks would work as replacement targets for the user's
# described concept.
class TakeoverSuggestOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[ReplacementCandidate] = Field(default_factory=list)
    # Optional preface the user sees above the picker. Empty string if
    # nothing to add. Kept as a single string so the CLI/Discord layers
    # can render it plainly.
    preamble: str = ""


# Flat authoring shape. ALL fields required. Every optional field in a
# flat struct doubles the grammar states the LLM's schema compiler
# must enumerate (2^N combinations); with 14 optionals we blew past
# Anthropic's "Schema is too complex" ceiling. All-required collapses
# the grammar to a single fixed shape.
#
# Fields intentionally merged vs. legacy CharacterRecord:
#   traits + voice + narrative_notes → rolled into `personality`.
#   The LLM writes one prose block covering who the character is, how
#   they talk, and how to play them under pressure. Fewer fields =
#   fewer author-time decisions, less token cost, simpler prompts.
#
# For player-authored characters (takeover describe/replace), the LLM
# emits personality="" — the prose is the player's to develop. Empty
# is a legitimate state; /leave triggers a synthesis pass if needed
# when handing the character back to the agent.
class AuthoredCharacter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    location: str
    role: str
    appearance: str
    default_loadout: str
    faction: str
    backstory: str
    # personality now absorbs traits + voice + portrayal notes: "a
    # disciplined guard captain with dry humor who speaks in clipped
    # formal sentences; his right hand twitches when he's lying; under
    # pressure he closes down rather than lashes out."
    personality: str
    known_context: str
    goals: list[str]
    current_objectives: list[str]
    secrets: list[str]
    intentions_enabled: bool
    # Short identity-and-intent line for player-created characters and
    # replacements. Router-authored NPC spawns live in compact router history;
    # frontend-created characters are external engine state updates, so this
    # line is queued for the next fresh router call. Author it as
    # third-person reportage that reads like one entry in a roster
    # ledger ("Tom — nervous stablehand at the courtyard, stays close
    # to the gate because he expects the captain back at any moment").
    # NOT in-character voice; NOT a goal restatement; NOT a backstory
    # paragraph. For player-created characters this should not be empty; the
    # engine would fall back to a much weaker mechanical line.
    router_summary: str

    def to_record(self, character_id: str = "") -> CharacterRecord:
        """Map onto CharacterRecord. Caller sets character_id, is_playable,
        and binding afterward if relevant.

        `router_summary` deliberately does NOT land on CharacterRecord. It is
        an author-time scratch field consumed by frontend create/replacement
        flows and otherwise discarded.
        """
        return CharacterRecord(
            character_id=character_id,
            name=self.name,
            location=self.location,
            public_sheet=PublicSheet(
                role=self.role,
                appearance=self.appearance,
                faction=self.faction,
            ),
            visuals=CharacterVisuals(
                default_loadout=self.default_loadout or self.appearance,
            ),
            private_state=PrivateState(
                goals=list(self.goals),
                current_objectives=list(self.current_objectives),
                secrets=list(self.secrets),
                intentions_enabled=self.intentions_enabled,
            ),
            backstory=self.backstory,
            personality=self.personality,
            known_context=self.known_context,
        )


# Router output for mode='describe' or mode='replace'. Returns a flat
# AuthoredCharacter; the engine maps it onto a CharacterRecord before
# persisting. Docstring kept off the class so it doesn't get baked into
# the JSON schema as a "description" the grammar compiler has to handle.
class TakeoverAuthoredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character: AuthoredCharacter
    # Short in-fiction note the router wants attached to the session log
    # so the next turn has continuity with the new character's arrival.
    # Empty string if nothing to add.
    session_note: str = ""
