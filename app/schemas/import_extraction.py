"""Pydantic shapes the story importer asks the model to fill via `output_format`.

These mirror the existing runtime shapes in `app/schemas/` but are kept separate
because:

- The extraction LLM should only populate what's discoverable from the source
  prompt (no session state, no transcripts, no rolling conversations).
- Extraction-time fields like `is_player` don't live on the runtime
  CharacterRecord (yet).
- Pydantic V2 schemas generated here flow directly into the Anthropic
  `output_format` path, which enforces JSON validity server-side.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------- World extraction ----------------

class SettingExtraction(BaseModel):
    genre: str = ""
    era: str = ""
    tone: str = ""
    premise: str = ""


class PhysicsRulesetExtraction(BaseModel):
    strength_limits: str = "human_baseline"
    magic_enabled: bool = False


class SceneExtraction(BaseModel):
    # Scene ID is a field (not a dict key) because the structured-output path
    # doesn't cleanly support dict[str, Model] — the generated schema uses
    # additionalProperties: $ref which the API's validator strips. Converted
    # to a dict at checkpoint-build time.
    scene_id: str
    name: str = ""
    description: str = ""
    connected_to: list[str] = Field(default_factory=list)


class LocationsExtraction(BaseModel):
    current_scene_id: str = ""
    scene_graph: list[SceneExtraction] = Field(default_factory=list)


class WorldExtraction(BaseModel):
    setting: SettingExtraction = Field(default_factory=SettingExtraction)
    lore: str = ""
    facts: list[str] = Field(default_factory=list)
    physics_ruleset: PhysicsRulesetExtraction = Field(default_factory=PhysicsRulesetExtraction)
    locations: LocationsExtraction = Field(default_factory=LocationsExtraction)
    narrative_rules: str = ""
    # Spoiler / conspiracy / hidden-plot content. ADJUDICATION ONLY — never
    # reaches the player-facing narrator output. Extracted from the master
    # prompt when the source contains hidden lore or plot details.
    hidden_lore: str = ""
    hidden_facts: list[str] = Field(default_factory=list)


# ---------------- Character extraction ----------------

class PublicSheetExtraction(BaseModel):
    role: str = ""
    traits: list[str] = Field(default_factory=list)
    voice: str = ""
    appearance: str = ""
    faction: str = ""


class PrivateStateExtraction(BaseModel):
    # Existential drives — who they are at core. Stable across the story.
    goals: list[str] = Field(default_factory=list)
    # Active, actionable pursuits — what they are trying to DO right now,
    # with a target and a time horizon. Seeded by the importer; the
    # character agent revises these at runtime via its output.
    current_objectives: list[str] = Field(default_factory=list)
    # Keys: character_id. Values clamped to [-1, 1] during checkpoint build.
    attitudes: dict[str, float] = Field(default_factory=dict)
    secrets: list[str] = Field(default_factory=list)
    # Marks this character as significant enough to warrant off-stage ticks
    # — they'll pursue their objectives while the player isn't watching.
    # Set true for antagonists, rivals, faction leaders; false for
    # background/incidental characters.
    intentions_enabled: bool = False


class CharacterExtraction(BaseModel):
    character_id: str
    name: str
    status: Literal["active", "dormant"] = "active"
    location: str = ""
    # When True, this character is a human-player slot the personalize flow
    # binds a Discord user to. Master prompts mark the protagonist(s) via this.
    is_player: bool = False
    public_sheet: PublicSheetExtraction = Field(default_factory=PublicSheetExtraction)
    private_state: PrivateStateExtraction = Field(default_factory=PrivateStateExtraction)
    backstory: str = ""
    personality: str = ""
    narrative_notes: str = ""


class CharacterListExtraction(BaseModel):
    """Wrapper so the extraction returns a JSON object (required by output_format)."""
    characters: list[CharacterExtraction] = Field(default_factory=list)


# ---------------- Character knowledge envelope ----------------

class CharacterKnowledgeEnvelope(BaseModel):
    """The filtered slice of world knowledge one character plausibly has.

    Produced by a batch pass that sees the omniscient world (public +
    hidden) and every character's role/backstory/secrets, and decides for
    each character what they would plausibly know. `known_context` is a
    single freeform field on purpose — the LLM picks the shape that best
    conveys what this character takes for granted.
    """
    character_id: str
    known_context: str = ""


class CharacterKnowledgeListExtraction(BaseModel):
    envelopes: list[CharacterKnowledgeEnvelope] = Field(default_factory=list)


# ---------------- Opening extraction ----------------

class OpeningExtraction(BaseModel):
    """The opening scene prose the narrator renders on first turn. Stored as
    authorial guidance; the runtime narrator re-renders it in-scene per the
    rolling-conversation architecture."""
    text: str = ""
