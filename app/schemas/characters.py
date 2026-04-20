from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CharacterStatus(str, Enum):
    active = "active"
    dormant = "dormant"
    culled = "culled"


class PublicSheet(BaseModel):
    role: str = ""
    traits: list[str] = Field(default_factory=list)
    voice: str = ""
    appearance: str = ""
    faction: str = ""


class PrivateState(BaseModel):
    # Existential drives — who this character is at core. Importer seeds
    # from character nature/personality. Rarely changes during play.
    goals: list[str] = Field(default_factory=list)
    # Actionable pursuits — what this character is trying to DO right now.
    # Agent-authoritative: each response emits a full replacement list.
    # Importer seeds 1-3 arc-level objectives per character at import time.
    current_objectives: list[str] = Field(default_factory=list)
    # Keys can be any character_id for inter-character attitudes
    attitudes: dict[str, float] = Field(default_factory=dict)
    secrets: list[str] = Field(default_factory=list)
    # Flag for "this character is significant enough to tick off-screen."
    # Importer sets true for antagonists, rivals, faction leaders — anyone
    # whose goals should keep moving while the player isn't watching.
    intentions_enabled: bool = False


class IncomingDirective(BaseModel):
    """A message or instruction another character has sent to this one.

    Flushed into the recipient's user message on their next response or
    tick, then cleared. `depth` tracks the delegation chain length for
    cycle/over-delegation guarding (warn at >2, cap at 10).
    """
    from_character_id: str
    content: str
    turn: int
    depth: int = 1


class CharacterRecord(BaseModel):
    character_id: str
    name: str
    status: CharacterStatus = CharacterStatus.active
    location: str = ""
    # True if this character is a human-player slot. The personalize flow
    # finds the player character by this flag (not by id suffix), and the
    # orchestrator excludes is_player characters from agent fan-out. Multi-
    # player stories can have multiple is_player=True entries.
    is_player: bool = False
    public_sheet: PublicSheet = Field(default_factory=PublicSheet)
    private_state: PrivateState = Field(default_factory=PrivateState)
    # Staging area for observations the character witnessed silently (turns where
    # they didn't respond). Flushed into the next agent user message when the
    # character is asked to respond, then cleared.
    pending_observations: list[str] = Field(default_factory=list)
    # Queue of directives from other characters awaiting flush on next turn.
    # Same flush-and-clear pattern as pending_observations but for messages
    # addressed to this character by another NPC (or by the world/narrator).
    incoming_directives: list[IncomingDirective] = Field(default_factory=list)
    # Long-form text fields for rich character content
    backstory: str = ""
    personality: str = ""
    narrative_notes: str = ""
    # Per-character world knowledge envelope: the filtered slice of world
    # lore/facts this character plausibly knows, plus their in-world sense
    # of what's going on. Populated at import time (and at spawn time for
    # router-created characters) by an LLM pass that sees the omniscient
    # world plus this character's role/faction/backstory/secrets. Left as
    # a single freeform field on purpose — the LLM picks whatever shape
    # best conveys what THIS character takes for granted.
    known_context: str = ""
