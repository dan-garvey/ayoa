from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CharacterStatus(str, Enum):
    active = "active"
    dormant = "dormant"
    culled = "culled"


class PublicSheet(BaseModel):
    # Trait list and voice absorbed into CharacterRecord.personality as
    # a single prose block — fewer author-time fields, less token cost,
    # smaller structured-output grammar. personality now describes who
    # they are, how they talk, and how to play them.
    role: str = ""
    appearance: str = ""
    faction: str = ""


class PrivateState(BaseModel):
    # Existential drives — who this character is at core. Importer seeds
    # from character nature/personality. Rarely changes during play.
    goals: list[str] = Field(default_factory=list)
    # Actionable pursuits — what this character is trying to DO right now.
    # Importer seeds 1-3 arc-level objectives per character at import
    # time. PRE-Commit-1 the agent's structured output rewrote this list
    # every turn (`private_updates.current_objectives`); Commit 1 dropped
    # structured agent output entirely, so this field is now author-time
    # state only — it seeds the agent's identity prompt and is otherwise
    # immutable during play. Live mutation of an agent's "what I'm
    # trying to do" lives in their rolling conversation history (the
    # trailing parenthetical on every response carries fresh intent).
    current_objectives: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    # Flag for "this character is significant enough to tick off-screen."
    # Importer sets true for antagonists, rivals, faction leaders — anyone
    # whose goals should keep moving while the player isn't watching.
    intentions_enabled: bool = False


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
    # character is asked to respond, then cleared. Commit 2 removed the
    # parallel `incoming_directives` queue: cross-character communication
    # now travels through normal canonical events (a courier walks in and
    # speaks; a note is rendered in `observable_facts` and the recipient
    # is added to that event's `observers`) rather than a structured
    # inter-agent message bus. NOTE: in current v11 production no engine
    # code path appends to this list — it is kept as the surviving
    # narrow channel for "things this character privately observed but
    # nobody else did" and Commit 5's tick wiring will re-establish a
    # population path.
    pending_observations: list[str] = Field(default_factory=list)
    # Long-form text fields for rich character content. `personality`
    # now absorbs what used to live in `narrative_notes` + traits/voice
    # on PublicSheet — one prose block covering who they are, how they
    # speak, and how to play them. Can be empty for freshly-authored
    # player characters (the player fills it through play); the /leave
    # path synthesizes from the character's rolling conversation when
    # handing back to an agent.
    backstory: str = ""
    personality: str = ""
    # Per-character world knowledge envelope: the filtered slice of world
    # lore/facts this character plausibly knows, plus their in-world sense
    # of what's going on. Populated at import time (and at spawn time for
    # router-created characters) by an LLM pass that sees the omniscient
    # world plus this character's role/faction/backstory/secrets. Left as
    # a single freeform field on purpose — the LLM picks whatever shape
    # best conveys what THIS character takes for granted.
    known_context: str = ""
