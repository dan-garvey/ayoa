from __future__ import annotations

from enum import Enum
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
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
    role: str = ""
    appearance: str = ""
    faction: str = ""
    # Player-safe identity context. The narrator may use this to explain
    # why a newly met name, uniform, rank, or social shorthand matters to the
    # viewpoint character when the engine explicitly surfaces first-meeting
    # context. It must not contain secrets, motives, hidden allegiances,
    # authorial labels, or private body details.
    public_context: str = ""


class ActorFactOrigin(str, Enum):
    """How one actor-local fact entered this person's experience."""

    lived = "lived"
    witnessed = "witnessed"
    told = "told"
    inferred = "inferred"


class ActorFact(BaseModel):
    """One concrete piece of actor-owned life or understanding.

    ``origin`` is authoring and audit provenance. The text itself must carry
    any uncertainty, source limitation, promise, debt, habit, or refusal that
    matters to the person; the runtime does not expose the enum as a dialogue
    recipe.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    origin: ActorFactOrigin = ActorFactOrigin.lived
    text: str = Field(min_length=1)


class ActorRecord(BaseModel):
    """Sparse actor-owned material plus an engine scheduling policy.

    Zero facts is valid. Different people should receive different amounts and
    kinds of material; there are deliberately no mandatory voice, secret,
    objective, ritual, or trauma slots.
    """

    model_config = ConfigDict(extra="forbid")

    may_act_offstage: bool = False
    facts: list[ActorFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def _facts_are_unique(self) -> "ActorRecord":
        normalized = [fact.text.casefold() for fact in self.facts]
        if len(normalized) != len(set(normalized)):
            raise ValueError("actor facts must be unique")
        return self


class VisualNovelCustomVariantRequest(BaseModel):
    """Character-authored request awaiting optional background generation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    variant_key: str = Field(min_length=1, max_length=80)
    direction: str = Field(min_length=1, max_length=200)
    sprite_pack_id: str = Field(min_length=1, max_length=200)
    requested_turn_index: int = Field(ge=0)
    generation_round: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def _normalize(self) -> "VisualNovelCustomVariantRequest":
        self.variant_key = self.variant_key.strip().lower()
        self.direction = " ".join(self.direction.split()).strip()
        self.sprite_pack_id = self.sprite_pack_id.strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", self.variant_key):
            raise ValueError("visual-novel custom variant key is invalid")
        if "<" in self.direction or ">" in self.direction:
            raise ValueError("visual-novel custom variant directions must be plain text")
        return self


class CharacterVisualNovelPresentation(BaseModel):
    """Durable self-authored display state; image bytes never live here."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    current_variant_key: str = Field(default="neutral", min_length=1, max_length=80)
    scene_location: str = Field(default="", max_length=500)
    custom_variant_sprite_pack_id: str = Field(default="", max_length=200)
    pending_requests: list[VisualNovelCustomVariantRequest] = Field(
        default_factory=list,
        max_length=2,
    )
    custom_variant_directions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "CharacterVisualNovelPresentation":
        self.current_variant_key = self.current_variant_key.strip().lower()
        self.scene_location = " ".join(self.scene_location.split()).strip()
        self.custom_variant_sprite_pack_id = self.custom_variant_sprite_pack_id.strip()
        cleaned_directions: dict[str, str] = {}
        for raw_key, raw_direction in self.custom_variant_directions.items():
            key = raw_key.strip().lower()
            direction = " ".join(raw_direction.split()).strip()
            if not key or not direction:
                continue
            if key in cleaned_directions:
                raise ValueError(
                    "visual-novel custom variant keys must remain unique"
                )
            if len(direction) > 200 or "<" in direction or ">" in direction:
                raise ValueError(
                    "visual-novel custom variant directions must be bounded plain text"
                )
            cleaned_directions[key] = direction
        self.custom_variant_directions = cleaned_directions
        if len(self.custom_variant_directions) > 20:
            raise ValueError("visual-novel custom variant catalog exceeds 20")
        pending_keys = [request.variant_key for request in self.pending_requests]
        if len(pending_keys) != len(set(pending_keys)):
            raise ValueError("visual-novel pending custom variants must be unique")
        if set(pending_keys).intersection(self.custom_variant_directions):
            raise ValueError(
                "visual-novel pending variants cannot already be in the catalog"
            )
        if len(self.custom_variant_directions) + len(pending_keys) > 20:
            raise ValueError("visual-novel custom variant capacity exceeds 20")
        keys = {
            self.current_variant_key,
            *self.custom_variant_directions,
        }
        if any(
            re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", key) is None
            for key in keys
        ):
            raise ValueError("visual-novel presentation keys must be opaque labels")
        return self


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
    visual_novel_presentation: CharacterVisualNovelPresentation = Field(
        default_factory=CharacterVisualNovelPresentation
    )

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


class CharacterRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    # separate from actor-owned facts: it explains what a human controls
    # without teaching an LLM how to play the character.
    player_guidance: str = ""
    agent_tier: CharacterAgentTier = CharacterAgentTier.premium
    # Knowledge tier this character was generated at (0 = untiered). Set at
    # spawn from the story's world_state.knowledge_tiers ladder; a future
    # promotion hook raises it and grants newly available actor facts. Orthogonal to
    # agent_tier (model cost), though a ladder rung may set both together.
    knowledge_tier: int = 0
    public_sheet: PublicSheet = Field(default_factory=PublicSheet)
    visuals: CharacterVisuals = Field(default_factory=CharacterVisuals)
    # Private, actor-owned life and understanding. ``None`` is valid for a
    # player-authored blank seat or a deliberately exterior-only walk-on.
    actor: ActorRecord | None = None
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
    # Optional rules/content adapter state. Narrative-only sessions leave
    # this empty. D&D v1 reads a small conventional subset when present:
    # ruleset_id, ability_scores, proficiency_bonus, skill_proficiencies,
    # saving_throw_proficiencies, armor_class, hit_points, conditions,
    # resources, and raw.
    mechanics: dict[str, Any] = Field(default_factory=dict)
    # Interior continuity (Commit 3 had `last_intent` / `last_intent_turn`
    # mirror fields here; both removed). The actor's own rolling conversation
    # preserves its observable choices and the causal input that produced
    # them. Cross-actor consumers (router, narrator, other agents) learn only
    # through in-fiction signals (a courier, an observable fact, a witnessed
    # action), never an actor-local hidden summary.
    # Engine-owned revision of the exact owner-only identity packet retained
    # in this character's private conversation. It is checkpoint metadata,
    # not character knowledge or model input.
    agent_identity_seed_sha256: str = Field(
        default="",
        pattern=r"^(?:|[0-9a-f]{64})$",
    )

    @field_serializer("agent_identity_seed_sha256")
    def _serialize_agent_identity_seed_sha256(
        self,
        value: str,
        info: SerializationInfo,
    ) -> str:
        if should_include_private_runtime_metadata(info.context):
            return value
        return ""

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
        if self.actor is not None:
            raise ValueError(
                "non-social hazards cannot carry an actor record"
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
