from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class CharacterStatus(str, Enum):
    active = "active"
    dormant = "dormant"
    culled = "culled"


class CharacterAgentTier(str, Enum):
    # Expensive, high-agency characters whose decisions carry core plot,
    # secrets, or long-running adversarial pressure.
    premium = "premium"
    # Normal story characters with ongoing narrative threads. In the
    # current runtime they use the Haiku-backed standard-agent role.
    standard = "standard"
    # Utility or supporting characters that need continuity but should not
    # consume the expensive plot-agent model. In the current runtime they
    # use the Sonnet-backed convenience role.
    utility = "utility"
    # Legacy tier names kept loadable for older checkpoints.
    plot = "plot"
    convenience = "convenience"


class PublicSheet(BaseModel):
    # Trait list and voice absorbed into CharacterRecord.personality as
    # a single prose block — fewer author-time fields, less token cost,
    # smaller structured-output grammar. personality now describes who
    # they are, how they talk, and how to play them.
    role: str = ""
    appearance: str = ""
    faction: str = ""


class CharacterDescriptions(BaseModel):
    # Player-safe identity context. The narrator may use this to explain
    # why a known name, uniform, rank, or social shorthand matters to the
    # viewpoint character. It must not contain secrets, motives, hidden
    # allegiances, authorial labels, or private body details.
    public: str = ""
    # Omniscient/private character description for future engine use and
    # auditing. This may contain spoiler-bearing identity context, but it is
    # never sent to the narrator.
    private: str = ""


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
    # Author-authored deterministic triggers for off-stage tick selection.
    # These are scheduler metadata, not prompt state: phrases here are
    # matched against recent canonical surface facts and state-change lines
    # to decide who should get a scarce off-screen action slot.
    tick_cues: list[str] = Field(default_factory=list)


class CharacterRecord(BaseModel):
    character_id: str
    name: str
    status: CharacterStatus = CharacterStatus.active
    location: str = ""
    # Fiction-time clock in seconds since session start. Updated from
    # canonical event/fact timing for events this character acts in or observes.
    clock_at_s: int = 0
    # Fiction-time of the last committed NPC-agent turn for this
    # character. Observation-only clock advancement does not touch this;
    # it exists so delayed agent turns can see how much story time has
    # passed since the character last had a chance to act.
    last_agent_turn_at_s: int | None = None
    # True if this character is REASONABLY PLAYABLE BY A HUMAN — i.e.,
    # the author/importer marked them as a slot a player can claim via
    # /join. They run as agent NPCs by default and only stop being
    # agent-controlled when the engine binds a human to them
    # (`session.character_bindings[character_id] = user_id`). Multi-
    # player stories typically have many is_playable=True characters
    # (e.g. every contestant on a dating show); a story can also have
    # zero (then /join surfaces nothing).
    #
    # Backward compat (renamed from `is_player` in playable-2 commit):
    # the field used to be named `is_player` and conflated "this is a
    # player slot" with "this is a human-controlled character." Old
    # checkpoint JSONs stored under that name are still loadable —
    # see the model_validator below that maps it on parse.
    is_playable: bool = False
    agent_tier: CharacterAgentTier = CharacterAgentTier.premium
    public_sheet: PublicSheet = Field(default_factory=PublicSheet)
    descriptions: CharacterDescriptions = Field(default_factory=CharacterDescriptions)
    private_state: PrivateState = Field(default_factory=PrivateState)
    # Staging area for observations the character perceived silently
    # (turns where they didn't respond). Flushed into the next agent
    # user message when the character is asked to respond, then cleared.
    # Commit 2 removed the parallel `incoming_directives` queue:
    # cross-character communication now travels through normal canonical
    # events (a courier walks in and speaks; a note is rendered in
    # `observable_facts` and the recipient is added to that event's
    # `observers`).
    #
    # Population path: `broadcast_event` in `app/engine/turn_loop.py`
    # appends visible `observable_facts` for local NPC observers and for
    # mediated/remote NPC observers explicitly named in fact-level
    # `visible_to`. Broad room facts do not cross a location boundary
    # unless the router scopes them through a concrete perception
    # channel such as live audio, camera feed, magic, radio, or spycraft.
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
    # Optional rules/content adapter state. Narrative-only sessions leave
    # this empty. D&D v1 reads a small conventional subset when present:
    # ruleset_id, ability_scores, proficiency_bonus, skill_proficiencies,
    # saving_throw_proficiencies, armor_class, hit_points, conditions,
    # resources, and raw.
    mechanics: dict[str, Any] = Field(default_factory=dict)
    # Interior continuity (Commit 3 had `last_intent` / `last_intent_turn`
    # mirror fields here; both removed). The agent's freshest interior
    # is the trailing parenthetical at the end of its most recent
    # `respond()` or `tick()` output, which is appended verbatim to
    # `character_conversations[character_id]`. The agent's own future
    # calls see that parenthetical because it's in the history they
    # replay. Cross-actor consumers (router, narrator, other agents)
    # deliberately do NOT see another character's interior — that
    # asymmetry is the entire point of having separate per-actor LLM
    # calls. If you find yourself wanting to surface one character's
    # parenthetical to another LLM, you are about to break that
    # asymmetry; reach for in-fiction signals (a courier, an
    # observable_fact, a witnessed action) instead.

    @model_validator(mode="before")
    @classmethod
    def _migrate_is_player_alias(cls, data: Any) -> Any:
        """Back-compat: old saves and any extant tests serialized the
        playability flag as `is_player`. Map it to `is_playable` on
        parse so legacy checkpoints load without touching them on
        disk. If both keys are present and disagree we trust the new
        name and warn — the explicit `is_playable` wins.
        """
        if not isinstance(data, dict):
            return data
        if "is_player" in data:
            old = data.pop("is_player")
            if "is_playable" in data:
                if data["is_playable"] != old:
                    logger.warning(
                        "CharacterRecord %r got both is_player=%r and "
                        "is_playable=%r; using is_playable.",
                        data.get("character_id", "?"), old, data["is_playable"],
                    )
            else:
                data["is_playable"] = old
        return data
