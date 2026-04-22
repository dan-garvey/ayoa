"""Pydantic shapes the story importer asks the model to fill via `output_format`.

These mirror the existing runtime shapes in `app/schemas/` but are kept separate
because:

- The extraction LLM should only populate what's discoverable from the source
  prompt (no session state, no transcripts, no rolling conversations).
- Extraction-time fields like `is_player` don't live on the runtime
  CharacterRecord (yet).
- Pydantic V2 schemas generated here flow directly into the Anthropic
  `output_format` path, which enforces JSON validity server-side.

## Schema-shape policy: no Pydantic defaults

Every field is REQUIRED (no `default=`, no `default_factory=`). Anthropic's
structured-output grammar compiler treats default-bearing fields as
optionals, and the optional permutations explode for deeply-nested schemas
— a 95 KB master prompt failed import with `Grammar compilation timed out.
(400)` after the compiler choked on the WorldExtraction → LocationsExtraction
→ SceneExtraction tree with optionals at every layer. Making everything
required collapses the search space; the LLM emits explicit `""` / `[]` /
`false` for empty content, which downstream `_build_*` helpers handle the
same way they handled the old defaults.

Pre-r7b this file had defaults on most fields and only `CombinedImportExtraction`
(the v3 collapsed pass) was annotated as needing required-only — but the v4
two-call pipeline still hit the timeout because `CoreImportExtraction` wraps
WorldExtraction/CharacterListExtraction/OpeningExtraction whose nested types
still carried defaults. Stripping them here fixes both pipelines.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# ---------------- World extraction ----------------

class SettingExtraction(BaseModel):
    genre: str
    era: str
    tone: str
    premise: str


class PhysicsRulesetExtraction(BaseModel):
    strength_limits: str
    magic_enabled: bool


class SceneExtraction(BaseModel):
    # Scene ID is a field (not a dict key) because the structured-output path
    # doesn't cleanly support dict[str, Model] — the generated schema uses
    # additionalProperties: $ref which the API's validator strips. Converted
    # to a dict at checkpoint-build time.
    scene_id: str
    name: str
    description: str
    connected_to: list[str]


class LocationsExtraction(BaseModel):
    current_scene_id: str
    scene_graph: list[SceneExtraction]


class PublicWorldExtraction(BaseModel):
    """v7 Call-1 schema: PUBLIC world only. Setting, public lore (one
    big string), public facts, physics, narrative rules. No locations,
    no hidden content.

    v6 split locations off and that wasn't enough — for dense
    conspiracy stories the public lore + hidden lore packed into one
    schema STILL truncated mid-string at ~265K JSON chars (≈60K
    output tokens) on a 95KB master prompt. Splitting public from
    hidden gives each call its own 64K budget. The structural
    separation also matches the engine's runtime contract: hidden
    content is adjudication-only (never reaches the player), public
    content is what player-facing renders may draw on."""
    setting: SettingExtraction
    lore: str
    facts: list[str]
    physics_ruleset: PhysicsRulesetExtraction
    narrative_rules: str


class HiddenWorldExtraction(BaseModel):
    """v7 Call-2 schema: HIDDEN world only. Conspiracy / spoiler / plot-
    secret content extracted into adjudication-only fields. Continuation
    that reads the public world (Call 1) as cached history.

    LLM emits "" / [] when the source has no hidden content (most
    stories without conspiracies or secret history)."""
    hidden_lore: str
    hidden_facts: list[str]


class WorldExtraction(BaseModel):
    """Merged world bundle consumed by `build_checkpoint`. From v7
    onwards this is assembled in Python from `PublicWorldExtraction`
    (Call 1) + `HiddenWorldExtraction` (Call 2) + `LocationsExtraction`
    (Call 3); the LLM never emits a `WorldExtraction` directly anymore.
    Retained as a single shape so downstream assembly code stays
    unchanged across importer versions."""
    setting: SettingExtraction
    lore: str
    facts: list[str]
    physics_ruleset: PhysicsRulesetExtraction
    locations: LocationsExtraction
    narrative_rules: str
    hidden_lore: str
    hidden_facts: list[str]


# ---------------- Character extraction ----------------

class PublicSheetExtraction(BaseModel):
    # traits and voice absorbed into CharacterExtraction.personality;
    # importer now emits one prose block covering both (plus portrayal
    # notes) rather than separate fields.
    role: str
    appearance: str
    faction: str


class PrivateStateExtraction(BaseModel):
    # Existential drives — who they are at core. Stable across the story.
    goals: list[str]
    # Active, actionable pursuits — what they are trying to DO right now,
    # with a target and a time horizon. Seeded by the importer; the
    # character agent revises these at runtime via its output.
    current_objectives: list[str]
    secrets: list[str]
    # Marks this character as significant enough to warrant off-stage ticks
    # — they'll pursue their objectives while the player isn't watching.
    # Set true for antagonists, rivals, faction leaders; false for
    # background/incidental characters.
    intentions_enabled: bool


class CharacterExtraction(BaseModel):
    character_id: str
    name: str
    status: Literal["active", "dormant"]
    location: str
    # When True, this character is a human-player slot the personalize flow
    # binds a Discord user to. Master prompts mark the protagonist(s) via this.
    is_player: bool
    public_sheet: PublicSheetExtraction
    private_state: PrivateStateExtraction
    backstory: str
    # personality absorbs narrative_notes (portrayal direction) and the
    # legacy public_sheet traits/voice fields; one prose block per
    # character.
    personality: str


class CharacterListExtraction(BaseModel):
    """Wrapper so the extraction returns a JSON object (required by output_format)."""
    characters: list[CharacterExtraction]


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
    known_context: str


class CharacterKnowledgeListExtraction(BaseModel):
    envelopes: list[CharacterKnowledgeEnvelope]


# ---------------- Opening extraction ----------------

class OpeningExtraction(BaseModel):
    """The opening scene prose the narrator renders on first turn. Stored as
    authorial guidance; the runtime narrator re-renders it in-scene per the
    rolling-conversation architecture."""
    text: str


# ---------------- Five-call pipeline (v7) wrapper schemas ----------------

class CharsAndOpeningExtraction(BaseModel):
    """v7 Call-4 schema (introduced in v5 as Call-2; renumbered each
    time the world side gained another split). Characters + opening
    together as a continuation that reads public + hidden world +
    locations as cached history. Splitting public / hidden / locations
    / chars+opening across separate calls lets each one use the full
    64K Sonnet output budget — v4's single combined call truncated
    mid-JSON on long master prompts, v5's combined-world call did the
    same on dense scene_graphs, and v6's combined public+hidden
    skeleton did the same on dense conspiracy lore."""
    characters: CharacterListExtraction
    opening: OpeningExtraction
