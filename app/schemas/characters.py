from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)

from app.schemas.content_privacy import should_include_private_runtime_metadata


class CharacterStatus(str, Enum):
    active = "active"
    dormant = "dormant"
    culled = "culled"


class CharacterAgentTier(str, Enum):
    # Expensive, high-agency characters whose decisions carry core plot,
    # secrets, or long-running adversarial pressure.
    premium = "premium"
    # Normal story characters with ongoing narrative threads. In the
    # current runtime they use the Luna-backed standard-agent role.
    standard = "standard"
    # Utility or supporting characters that need continuity but should not
    # consume the expensive plot-agent model. In the current runtime they
    # use the Sonnet-backed convenience role.
    utility = "utility"


class FictionalEntityKind(str, Enum):
    """The kind of fictional agency represented by a roster record.

    ``character`` records own deliberate choices and can therefore produce a
    character-agent or player intention. ``hazard`` records preserve stable
    identity, location, visuals, and state for a patterned non-social entity,
    but have no dialogue, interiority, or character turn. Their established
    behavior is adjudicated as environmental pressure instead.
    """

    character = "character"
    hazard = "hazard"


class PlayerSlotKind(str, Enum):
    """How an authored playable record behaves while no human controls it.

    Ordinary slots remain normal characters and may be agent-driven until a
    player claims them. Player-authored slots are blank casting positions:
    they stay outside the fiction and every LLM surface until a player supplies
    a name and appearance atomically at claim time.
    """

    standard = "standard"
    player_authored = "player_authored"


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
    # why a newly met name, uniform, rank, or social shorthand matters to the
    # viewpoint character when the engine explicitly surfaces first-meeting
    # context. It must not contain secrets, motives, hidden allegiances,
    # authorial labels, or private body details.
    public: str = ""
    # Omniscient/private character description for future engine use and
    # auditing. This may contain spoiler-bearing identity context, but it is
    # never sent to the narrator.
    private: str = ""


class CharacterVisuals(BaseModel):
    # Stable player-safe first-look exterior. This is separate from
    # `public_sheet.appearance` because it is runtime context: the engine may
    # surface it once when a viewpoint first meaningfully sees this character.
    # Keep it free of secrets, motives, concealed traits, and author-only notes.
    default_loadout: str = ""
    # Presentation policy for generated illustrations. `anonymous` permits an
    # unreferenced background figure but never a named visual subject; `omit`
    # keeps the character out of generated art entirely.
    depiction_policy: Literal["normal", "anonymous", "omit"] = "normal"
    # Engine-owned selected identity handle. It may resolve to a generated
    # candidate or a reviewed authored reference. The image and opaque handle
    # are presentation-only and never enter runtime LLM input.
    identity_reference_id: str = ""
    # Engine-owned reviewed visual-novel cutout set. The set and its variant
    # handles are presentation provenance, not character knowledge or player
    # text, and therefore serialize only in private checkpoint contexts.
    sprite_set_id: str = ""

    @field_serializer("identity_reference_id")
    def _serialize_identity_reference_id(
        self,
        value: str,
        info: SerializationInfo,
    ) -> str:
        if should_include_private_runtime_metadata(info.context):
            return value
        return ""

    @field_serializer("sprite_set_id")
    def _serialize_sprite_set_id(
        self,
        value: str,
        info: SerializationInfo,
    ) -> str:
        if should_include_private_runtime_metadata(info.context):
            return value
        return ""


class PrivateState(BaseModel):
    # Existential drives — who this character is at core. Synthetic story
    # authors seed these from character nature/personality. Rarely changes
    # during play.
    goals: list[str] = Field(default_factory=list)
    # Actionable pursuits — what this character is trying to DO right now.
    # Synthetic story authors seed 1-3 arc-level objectives per character.
    # PRE-Commit-1 the agent's structured output rewrote this list
    # every turn (`private_updates.current_objectives`); Commit 1 dropped
    # structured agent output entirely, so this field is now author-time
    # state only — it seeds the agent's identity prompt and is otherwise
    # immutable during play. Live mutation of an agent's "what I'm
    # trying to do" lives in their rolling conversation history (the
    # trailing parenthetical on every response carries fresh intent).
    current_objectives: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    # Flag for "this character is significant enough to act off-screen."
    # Synthetic story authors set true for antagonists, rivals, faction
    # leaders — anyone whose goals should keep moving while the player isn't
    # watching.
    intentions_enabled: bool = False


class CharacterRecord(BaseModel):
    character_id: str
    name: str
    entity_kind: FictionalEntityKind = FictionalEntityKind.character
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
    # the author marked them as a slot a player can claim via
    # /join. They run as agent NPCs by default and only stop being
    # agent-controlled when the engine binds a human to them
    # (`session.character_bindings[character_id] = user_id`). Multi-
    # player stories typically have many is_playable=True characters
    # (e.g. every contestant on a dating show); a story can also have
    # zero (then /join surfaces nothing).
    is_playable: bool = False
    player_slot_kind: PlayerSlotKind = PlayerSlotKind.standard
    # Player-facing control contract for an authored seat. This is deliberately
    # separate from agent personality and private context: it explains what a
    # human controls without teaching an LLM how to play the character.
    player_guidance: str = ""
    agent_tier: CharacterAgentTier = CharacterAgentTier.premium
    # Knowledge tier this character was generated at (0 = untiered). Set at
    # spawn from the story's world_state.knowledge_tiers ladder; a future
    # promotion hook raises it and re-grants known_context. Orthogonal to
    # agent_tier (model cost), though a ladder rung may set both together.
    knowledge_tier: int = 0
    public_sheet: PublicSheet = Field(default_factory=PublicSheet)
    descriptions: CharacterDescriptions = Field(default_factory=CharacterDescriptions)
    visuals: CharacterVisuals = Field(default_factory=CharacterVisuals)
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
    # of what's going on. Seeded in synthetic story checkpoints (and at
    # spawn time for router-created characters) from the omniscient world
    # plus this character's role/faction/backstory/secrets. Left as a
    # single freeform field on purpose so each character gets the shape that
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
    # committed agent turn, which is appended verbatim to
    # `character_conversations[character_id]`. The agent's own future
    # calls see that parenthetical because it's in the history they
    # replay. Cross-actor consumers (router, narrator, other agents)
    # deliberately do NOT see another character's interior — that
    # asymmetry is the entire point of having separate per-actor LLM
    # calls. If you find yourself wanting to surface one character's
    # parenthetical to another LLM, you are about to break that
    # asymmetry; reach for in-fiction signals (a courier, an
    # observable_fact, a witnessed action) instead.

    @model_validator(mode="after")
    def _validate_entity_kind(self) -> "CharacterRecord":
        if self.entity_kind != FictionalEntityKind.hazard:
            return self
        if self.is_playable:
            raise ValueError("non-social hazards cannot be playable seats")
        if self.player_slot_kind != PlayerSlotKind.standard:
            raise ValueError(
                "non-social hazards cannot be player-authored seats"
            )
        if self.player_guidance:
            raise ValueError(
                "non-social hazards cannot carry player guidance"
            )
        if self.personality.strip():
            raise ValueError(
                "non-social hazards cannot carry character portrayal direction"
            )
        if self.backstory.strip() or self.known_context.strip():
            raise ValueError(
                "non-social hazards cannot carry character knowledge fields"
            )
        private_state = self.private_state
        if (
            private_state.goals
            or private_state.current_objectives
            or private_state.secrets
            or private_state.intentions_enabled
        ):
            raise ValueError(
                "non-social hazards cannot carry character interior state"
            )
        return self


def is_player_authored_slot(character: CharacterRecord | None) -> bool:
    return bool(
        character is not None
        and character.player_slot_kind == PlayerSlotKind.player_authored
    )


def is_non_social_hazard(character: CharacterRecord | None) -> bool:
    return bool(
        character is not None
        and character.entity_kind == FictionalEntityKind.hazard
    )
