"""Master-prompt → CheckpointFile import pipeline.

Four LLM calls that share a cached source-prompt prefix:
  1. World extraction (setting, lore, scenes, rules, hidden content) —
     serial. Primes the shared cache so subsequent calls read instead of
     re-write.
  2. Character extraction (full roster, is_player flag) — parallel with 3.
  3. Opening extraction (author's guidance prose for the narrator) —
     parallel with 2.
  4. Knowledge envelope extraction — batch call that sees the omniscient
     world + full roster and outputs per-character `known_context`. Runs
     after 2 completes.

Preservation analysis (`run_preservation_analysis`) is a separate fifth
call callers run independently — on the bot path it fires background via
`asyncio.create_task`; on the CLI path it runs inline.

All calls use `response_model=<Pydantic class>` so the API enforces schema
validity server-side.

Used by both `scripts/import_story.py` (CLI) and
`app/bot/engine_bridge.py` (Discord /story import).

## TODO: single-call import experiment

The current 4-stage pipeline was chosen for cache-friendly, focused
prompts — each call narrows the LLM onto one extraction task with the
shared source prefix cached. A tempting optimization: collapse stages
1-4 into one structured-output call that emits `{world, characters,
opening, envelopes}` as a single JSON blob. That would cut per-stage
setup/teardown, halve or quarter API round trips, and let the model
reason about cross-stage consistency (e.g., character envelopes
informed by the world it just extracted) in one pass.

The reason it isn't done: we have no ground-truth measurement of
extraction quality. A drop in lore fidelity, character nuance, or
knowledge-envelope distinctness is easy to produce and hard to detect
from an automated metric — the preservation analysis catches topic-level
drops but not subtle-but-critical compressions. Running A/B would
require:
  1. A fixed set of master prompts with hand-graded extraction quality
     rubrics (faction coverage, character voice preservation, secret
     placement, envelope distinctness).
  2. Running both pipelines on each and scoring against the rubric.
  3. Token and wall-clock deltas per pipeline.

If the comparison shows the single-call pipeline within ~5% of the
4-stage pipeline on quality for meaningful savings in cost or latency,
ship it. Until that measurement exists, don't refactor blindly.

## Versioning

`IMPORTER_VERSION` below stamps every checkpoint this pipeline produces.
The version covers the UNION of the extraction schemas (app/schemas/
import_extraction.py) and the extraction prompts in this file — any
change to either may alter what gets produced from the same source
master prompt.

**Policy: bump sparingly.** A version bump signals to story authors and
operators that the extraction contract changed and their existing stories
may need re-importing to pick up new fields or better output. Because
re-imports cost real time and money (a 95 KB master prompt was ~11
minutes on Sonnet in the last measurement), avoid churn here unless:

  1. A new engine feature requires a field the current version doesn't
     populate, AND the extraction is non-trivial to infer post-hoc.
  2. A guidance change is measurably improving output quality on real
     stories (not just tweaking wording).

Small prompt tweaks that don't change shape or alter downstream
semantics can go in without a version bump — use judgement. When in
doubt, bump."""

from __future__ import annotations

import asyncio
import logging
import re
import time

from pydantic import BaseModel, ConfigDict, Field

from app.llm.client import LLMClient
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile, ImportAnalysis
from app.schemas.import_extraction import (
    CharacterKnowledgeListExtraction,
    CharacterListExtraction,
    CharsAndOpeningExtraction,
    OpeningExtraction,
    WorldExtraction,
)
from app.schemas.state import (
    LocationState,
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)

logger = logging.getLogger(__name__)

# Sonnet 4.6 max output tokens. The extractions don't hit this in practice, but
# setting it at the ceiling means soft-budget truncation never happens — the
# model outputs what the source warrants, no more, no less.
MAX_EXTRACTION_TOKENS = 64_000

# Stamped on every checkpoint produced by this pipeline. Bump only when
# the extraction shape or prompts change in a way that invalidates
# existing imports. See the module docstring for policy.
#
# v1 → v2: added knowledge envelopes (CharacterRecord.known_context),
# removed setting.premise from agent context (agent sees only its own
# envelope), added preservation analysis as a separate pass.
# v2 → v3: combined single-call pipeline (world+characters+opening+knowledge
# in one structured-output call); preservation analysis runs as a
# continuation that reads the first call as cached 1h history.
# v3 → v4: two-call pipeline. Call 1 extracts world+characters+opening
# in a single structured-output pass; Call 2 produces knowledge
# envelopes as a continuation that reads Call 1's conversation as
# cached history. Motivation: v3 over-compressed knowledge envelopes
# (−75% vs v2) and dropped cross-character relational secrets. Giving
# knowledge its own focused pass — while keeping the upstream three
# together for cross-referencing — restores the attention budget
# without re-fragmenting into four separate calls.
# v4 → v5: three-call pipeline. The combined world+characters+opening
# call exceeded Sonnet 4.6's 64K output-token cap on a 95KB master
# prompt (mid-string truncation at ~303K chars of dense JSON ≈ 60K
# tokens), so split it: Call 1 = world only; Call 2 = characters +
# opening as a continuation reading world as cached history; Call 3 =
# knowledge envelopes as a continuation reading the full chain. Each
# call now gets the full 64K output budget. Cache-prefix continuity is
# preserved — every call after Call 1 reads the prior chain warm.
IMPORTER_VERSION = "v5"


class CombinedImportExtraction(BaseModel):
    """(v3) Wraps all four extraction outputs in a single structured-output
    response. Retained so the v3 single-call path stays runnable for
    quality comparisons; the active importer is v4 (two-call).

    Required fields only — the structured-output grammar compiler
    handles deeply-nested schemas more reliably when optionals are
    minimized.
    """
    world: WorldExtraction
    characters: CharacterListExtraction
    opening: OpeningExtraction
    knowledge: CharacterKnowledgeListExtraction


class CoreImportExtraction(BaseModel):
    """(v4) First of two calls. Produces world + characters + opening in
    one structured-output response. Knowledge envelopes are generated
    by a focused Call 2 that reads this conversation as cached history
    — the split gives envelopes their own attention budget and
    preserves cross-character relational detail that the v3 single-
    call pass compressed out.
    """
    world: WorldExtraction
    characters: CharacterListExtraction
    opening: OpeningExtraction


# ---------------- Extraction prompts ----------------

SHARED_SOURCE_SYSTEM = """\
You extract structured data from a master interactive-fiction prompt.

The source prompt below is the single authoritative input. Every extraction
task you receive will ask for a different slice of it. Your outputs must be
faithful to the source: preserve every distinct detail, relationship,
personality beat, faction, secret, and motivation the source describes.
Compression, summarization, or rounding is a quality defect — the engine
consumes your output verbatim, so anything you drop is permanently lost.

## Source Prompt
{source_prompt}"""


WORLD_EXTRACTION_INSTRUCTIONS = """\
Extract the world state from the source prompt into the requested schema.

For each field, be exhaustive. If the source describes five factions, all five
appear; if it names nine locations, all nine appear in the scene graph; if the
narrative-style rules cover ten pages, all ten pages become `narrative_rules`.
There is no length limit — produce whatever volume the source warrants.

## Field guidance

- **setting.premise**: Capture the story's core situation with enough detail
  for someone who has never read the source to understand who is involved,
  what's at stake, and what tension drives the story. Brevity is not a virtue
  here; clarity is.

- **lore**: COMMON-KNOWLEDGE world lore only — history, factions, political
  systems, laws, magic systems, religions, key publicly-known events. If the
  master prompt's lore section is long and detailed, your `lore` field should
  match that length and detail. If something in the source is described as
  mysterious to the in-world public, describe it as a mystery here — do not
  explain the resolution.

- **facts**: Each fact is one concrete, publicly-known current-state
  observation about the world. Include every such fact the source describes.

- **hidden_lore**: Spoiler-tier world information the in-world public does
  NOT know: engineered plagues, conspiracies, secret factions, the real causes
  of mysteries, hidden histories. This is for the engine's adjudication layer
  only and is never shown to the player. If the source describes hidden
  history, conspiracies, or plot-level secrets, capture them here in full.
  If the source has no hidden information, leave this empty.

- **hidden_facts**: Each hidden fact is one concrete, non-public truth about
  the current state of the world. Parallel to `facts` but for the adjudication
  layer only.

- **physics_ruleset.strength_limits**: A short phrase describing the baseline
  physical capability of the story's characters ("human_baseline",
  "augmented_human", "low_magic_enhanced", etc.).

- **physics_ruleset.magic_enabled**: true if the story has any supernatural
  force the characters use or encounter; false otherwise.

- **locations.current_scene_id**: The scene_id (lowercase snake_case) of the
  scene where the story begins. This scene_id MUST match an entry in
  `scene_graph`.

- **locations.scene_graph**: A list with one entry per location the source
  describes — named rooms, wings, districts, outdoor spaces, anywhere the
  player might be or move to. Each entry:
    - `scene_id`: lowercase snake_case unique identifier (e.g. `diplomatic_quarter`)
    - `name`: human-readable label
    - `description`: full sensory description as the source presents it
    - `connected_to`: list of other scene_ids reachable directly from here
  Do not invent connections the source does not describe; do not omit
  connections the source does describe. All scene_ids in `connected_to`
  should match an existing entry's `scene_id`.

  **Sub-scenes are distinct scenes.** Sealed chambers, inner wings,
  restricted sections, hidden rooms, access tunnels that sit inside a
  larger district, archive vaults inside an archive — each gets its
  own entry in `scene_graph`, with its own `scene_id`, and
  `connected_to` the parent scene. Do NOT fold a sealed chamber into
  its parent district's description — sealed/restricted sub-scenes
  are exactly where plot-critical discovery happens, and the engine
  cannot route a player into a scene that does not exist in the
  graph. If the source describes `gatehouse_district` AND a
  `gatehouse_sealed_section` inside it, both must appear as separate
  entries.

- **narrative_rules**: ONLY the story-specific narrative delta. The
  engine's narrator, router, and character-agent prompts already handle
  the generic craft discipline that every literary story shares — prose
  rules against stock phrases and filler, pacing (0-5 short paragraphs
  per response, stop when an NPC finishes speaking), information
  asymmetry enforced structurally by the pipeline, earned-outcomes /
  deferral posture, "never speak for the player," no menu options, no
  unearned sycophancy, no consolation signals on failure, rivals played
  to win, NPCs-have-lives-between-turns. **Do not restate any of that
  here.**

  What DOES belong in `narrative_rules`:
    - The story's signature structural constraint (Article Nineteen / a
      ticking clock / Price of Crossing / an Isolation Mandate — the
      named, world-specific pressure that shapes every scene).
    - Faction-response triggers and escalation logic specific to this
      conspiracy or antagonist structure.
    - Per-story tone or POV deviations from engine defaults (e.g.
      second-person-present, an unusual narrator posture).
    - Moral framing of antagonists that's unique to this world's ethics
      (e.g. "the conspirators' arguments should be genuinely
      compelling, not simple villainy" when the story demands it).

  What does NOT belong (already handled by engine prompts):
    - Forbidden stock phrases lists ("mask slips," "flickers in her
      eyes") — in narrator prompt.
    - Dialogue-and-pacing rules — in narrator prompt.
    - Love-interest / rival structural requirements as generic patterns
      — handled per-character via `narrative_notes` and via agent
      prompt discipline.
    - Information-asymmetry rules — structurally enforced.
    - Success/failure evaluation metrics — these are authoring rubrics,
      not runtime instructions.

  Be aggressive about keeping this field minimal. A 10,000-word
  `narrative_rules` block is a signal the extraction didn't understand
  which parts the engine already handles. Target: the story's
  distinguishing craft guidance only — often 500-2000 characters, not
  10,000+."""


CHARACTER_EXTRACTION_INSTRUCTIONS = """\
Extract every named character from the source prompt into structured records.
Return them in a `characters` array.

Include EVERY named character: protagonists, NPCs, love interests, rivals,
supporting cast, faculty, political figures, servants, background figures
named in passing. If the source introduces a character, they belong in the
output. A roster of 20 characters in the source produces a roster of 20
characters here. Brevity is not a virtue — capture each character's full
depth as the source describes them.

## Field guidance per character

- **character_id**: lowercase snake_case, unique across the roster. Derive
  from the character's name (e.g. "Captain Jorin Vael" → `captain_vael` or
  `jorin_vael`).

- **name**: Full name as the source presents it.

- **status**: "active" for characters currently in play; "dormant" for named
  characters who exist in the world but are not expected to appear in scenes
  (off-planet contacts, distant figures). Default to "active" unless the
  source signals otherwise.

- **location**: starting scene_id. Leave empty if the source does not anchor
  them to a specific starting scene.

- **is_player**: **true** for the protagonist / player character — the one
  the human player controls. Multi-protagonist stories can mark multiple
  characters as `is_player: true` (one per player slot). Every other
  character is `is_player: false`. If no character is clearly the protagonist
  (e.g. a fully-NPC ensemble), leave all false.

- **public_sheet**:
  - `role`: their role, title, or occupation.
  - `appearance`: every physical detail the source gives. Height, build,
    hair, eyes, clothing, bearing, signatures. Full.
  - `faction`: faction/house/group affiliation.

- **private_state**:
  - `goals`: **existential drives only** — who this character is at their
    core, what they fundamentally want from life, what they fear, what
    they are trying to become. These are stable across the story and do
    not change turn-to-turn. Examples: "Matter to someone who chose her
    rather than was assigned her." "Survive long enough to see his
    daughter married." "Prove that his craft deserves the respect of the
    Circle." Do NOT put pursuable tasks here — those go in
    `current_objectives`.
  - `current_objectives`: **actionable pursuits this character is working
    on RIGHT NOW**, with a target and a time horizon. Each objective is
    a thing the character would check off when done. Examples: "Get
    Johnny introduced to the Regent before the end of his first week."
    "Retrieve the sealed letter from Councilor Hanzo's study." "Keep
    Sable away from the Athenaeum tonight." Seed 1–3 per character.
    Leave empty for purely reactive background characters. If the source
    gives the character arc-level drivers (someone wants X, is pursuing
    Y, has been tasked with Z), those belong here.
  - `secrets`: every secret the source says this character keeps. Not just
    the dramatic ones; include minor private truths too if the source names
    them.
  - `intentions_enabled`: **true for narratively significant characters
    who should keep pursuing their objectives while the player isn't
    watching** — antagonists, faction leaders, rivals, romantic targets,
    anyone with a clear agenda. False for servants, background figures,
    walk-on parts, or characters whose only role is to react when the
    player encounters them. A good rule of thumb: if the character's
    `current_objectives` list is empty, `intentions_enabled` should
    almost certainly be false.

- **backstory**: character history from THEIR perspective — only what THEY
  know or believe about themselves. If the source says an NPC is secretly a
  spy, that fact goes in `secrets`, not `backstory`. If the source
  describes a character's past in depth, your backstory field mirrors that
  depth.

- **personality**: ONE prose block that covers three things at once:
    1. Inner world and private self — what they want, fear, believe;
       their contradictions; how they behave under pressure.
    2. Voice — how they speak, verbal tics, register, cadence, lexicon.
    3. Portrayal direction — how to play them in scenes. What they hide,
       what pressures them, how they deflect, what kind of interaction
       changes them, what narrative role they serve. Situational tells
       if the source makes them distinctive.
  Do not split these into separate paragraphs unless the source does.
  Prioritize pressure points, contradictions, loyalties, and
  vulnerabilities over surface quirks. Extract as much as the source
  offers.

## Preserve authored interior passages verbatim — do NOT bullet-compress

If the source prompt contains clearly-authored sections of interior
prose about a character — for example sections explicitly labeled
"Private Sensuality," "Emotional Interior," "Hidden Feelings," "Inner
Life," "Sensual Geography," "What Drives Them," or similar — preserve
that prose **verbatim** into `personality`. Do not flatten it to a
bullet list. Do not rewrite it into your own voice. Do not abstract
the specific sensory, emotional, or psychological detail into a
summary.

Rule of thumb: if the source wrote a paragraph, your field carries a
paragraph. If the source wrote a list of specific erogenous zones or
emotional tells or private habits with rich context, your field
carries that list with its context intact. The engine's character
agent reads `personality` directly and needs the specificity —
"Aldric's two suppressed attractions to Seraphel (curse-driven verse
she won't explain) and Thessaly (psychological strangeness he cannot
categorize), each suppressed for a different reason" is not
interchangeable with "Aldric is attracted to Seraphel and Thessaly."
The importer's preservation-analysis pass explicitly calls out
compression of these passages as a quality defect, so this is not
optional craft advice.

## Information isolation — critical

The source may contain hidden plot details, conspiracies, and secrets.
Separate what each character KNOWS from what the READER knows:

- **backstory, personality, public_sheet**: ONLY what this character knows
  or others can observe. If the protagonist is unaware a plague was
  engineered, their backstory says "the plague", not "the engineered plague".
  Portrayal notes in `personality` may reference hidden dynamics for the
  engine's benefit but should not state secrets as if the character is
  aware of them (unless they are).

- **private_state.secrets**: where hidden knowledge, true allegiances,
  conspiracies, and plot-relevant information belongs. A character who is
  secretly an agent of a conspiracy has that fact here, not in their
  backstory or personality.

## Relational secrets — name the other character explicitly

Many of a story's load-bearing secrets are RELATIONAL: character A
knows something about, hides something from, or holds an unspoken
feeling toward character B. Extract every relational secret the
source describes, and in the secret text, NAME character B
explicitly — not "her superior," not "the opposing Court's leader,"
not "the other resident." Examples of the shape to preserve:

- "The sound she hears is the same sound Nyx experiences in
  nightmares — she does not know this about Nyx, but the
  connection is real."
- "She succeeded in Reaching her son once and decided not to try
  again. She has not told Edric this, because it would devastate
  him — it would mean the thing he is chasing is real and she
  chose not to pursue it."
- "He suspects Captain Vael is feeding information to the Keeper
  conspiracy but has not confronted her."

Relational secrets are the primary source of in-fiction tension.
Unnamed abstractions ("she knows her superior is hiding something")
are not interchangeable with named specifics ("she knows Edric is
hiding something"); the named form is what the engine's agents
need to act on subtext. Do not compress relational secrets into
generic references. If the source names both sides of the
relationship, your secret names both sides of the relationship."""


KNOWLEDGE_EXTRACTION_INSTRUCTIONS = """\
Produce a per-character knowledge envelope for every character in the
roster. The envelope is what THIS character takes for granted about the
world — the filtered slice of lore, facts, rumors, and understanding
their role, education, faction, and personal history would plausibly
expose them to.

## What goes in an envelope

- Public knowledge anyone of this character's education and social class
  would hold.
- Specific knowledge their role grants (a Court Regent reads intelligence
  the kitchen never sees; a physician knows who is ill; a herald knows
  who was introduced when).
- Hidden knowledge ONLY if their position or secrets make their access
  plausible. A Keeper operative knows specific pieces of the conspiracy
  that her handler told her, NOT the full authorial reveal. A scholar
  might suspect something is strange without knowing the truth.
- Rumors, misconceptions, and beliefs that differ from the objective
  truth. Characters can be wrong — great material for later revelations.

## What to exclude

- Spoiler content this character has no plausible access to. Err toward
  exclusion when in doubt; a character can always learn more through
  play, but they can't unlearn what the envelope hands them.
- Other characters' private thoughts or personal secrets.
- Authorial meta-framing. Write from INSIDE the character's perspective,
  not as an omniscient summary of the story.

## Distinctness

Two characters in similar positions still have different envelopes. A
Council member who favors one faction knows the landscape differently
than one who favors another. Two scholars have different specialties.
Two guards overhear different rumors. Do not reuse passages across
envelopes; distinctness is the point.

## Format

`known_context` is a single freeform markdown field. Use whatever shape
best conveys what this character knows — prose, bullets, section
headers, any mixture. Be thorough but do not invent detail the source
does not support. Aim for enough content that the character could
confidently speak to their world without needing to stop and think.

Every character_id from the roster below must appear exactly once in the
`envelopes` array."""


OPENING_EXTRACTION_INSTRUCTIONS = """\
Render the opening scene of the story — the first passage the player reads
when the story begins.

Write prose, not exposition. This is authorial guidance the engine's narrator
reads when rendering the player's first turn in-scene; the text you produce
here shapes tone, atmosphere, and sensory grounding.

## Directives

1. Write in **second person present tense** addressing the player character
   as "you." Other characters in third person as normal.

2. Use `PLAYER_NAME` as a placeholder for the player character's name — the
   personalize step replaces it with the Discord user's chosen name.

3. Begin immediately in-fiction. No preambles, no setup questions to the
   player, no "before we begin."

4. Establish — with whatever depth and length the source's opening warrants:
   - Who PLAYER_NAME is and why they are here
   - Sensory grounding in the starting location (sight, sound, smell, air,
     light, texture as appropriate)
   - The stakes: what's at risk, what they don't yet know, what looms
   - The sense of being an outsider or arrival (when the source supports it)

5. End at a **natural decision point** — the player has arrived, taken in
   the scene, and can now choose what to do. Do not make choices on their
   behalf (don't decide where they sit, what they pick up, who they talk to).

6. Include ONLY common knowledge. No conspiracy details, no hidden lore,
   no secret plots. The player character is a newcomer and the opening
   reflects what they know or can observe.

7. Do not open with the word "you" — lead with something more evocative.

8. Never break the fourth wall — no references to prompts, players,
   character creation, or anything out-of-story.

9. **Match the source's opening in length and texture — do NOT
   compress.** If the master prompt's opening passage runs 900 words
   of arrival, atmosphere, and sensory grounding, your `text` runs
   close to that — same pace, same density of sensory detail, same
   accumulation of in-fiction setup. A terse five-paragraph summary
   of a long authored opening is a quality defect: the runtime
   narrator renders this text on the player's first turn, and
   everything you trim here is trimmed from the player's arrival
   experience permanently. Compressing a rich opening into half its
   length is exactly what this instruction forbids. If the source
   was rich, your output is rich. If the source was terse, your
   output is terse. Mirror, don't summarize.

Produce the prose in the `text` field. No JSON structure beyond that, no
markdown fences, no meta-commentary."""


CORE_EXTRACTION_INSTRUCTIONS = """\
Extract the CORE import bundle for this master prompt in a single
structured response: `world`, `characters`, and `opening`. Knowledge
envelopes are produced in a follow-up call that will read your output
here as cached history — focus THIS call on getting world state,
characters, and the opening scene extracted as completely and
faithfully as the source warrants.

Because you are producing world + characters + opening together, use
the cross-referencing opportunity: the same faction described in
`world.lore` should land on the right
`characters[].public_sheet.faction`; the same secret in
`world.hidden_facts` should be carried on the plausible
`characters[].private_state.secrets`; characters who would reasonably
know secrets about ANOTHER character (relationships, hidden
connections, rivalries, loyalties) should have those relational
secrets recorded on their own `secrets` list, with the other
character named. Do not drop cross-character linkages — they are
the primary source of in-fiction tension and rediscovery.

---

## `world`

{world_instructions}

---

## `characters`

{character_instructions}

---

## `opening`

{opening_instructions}"""


CHARS_AND_OPENING_CONTINUATION_INSTRUCTIONS = """\
You just extracted the `world` section above.

Now produce `characters` and `opening` together in this single
structured response. Use the world state YOU JUST EXTRACTED as
ground truth — do not re-derive it from the master prompt. Keep
the same exhaustiveness standard as Call 1: every character the
source describes, every faction membership, every secret, every
backstory beat.

Cross-referencing opportunity: factions in `world.lore` should
land on the right `characters[].public_sheet.faction`; secrets in
`world.hidden_facts` should be carried on the plausible
`characters[].private_state.secrets`; characters who would
reasonably know secrets about ANOTHER character should record
those relational secrets on their own `secrets` list, with the
other character named.

---

## `characters`

{character_instructions}

---

## `opening`

{opening_instructions}"""


KNOWLEDGE_CONTINUATION_INSTRUCTIONS = """\
You just extracted `world`, `characters`, and `opening` across the
two prior calls.

Now produce the `knowledge` envelopes: one `CharacterKnowledgeEnvelope`
per character in the roster, in the `envelopes` array. Use the world
state and roster YOU JUST EXTRACTED as ground truth — do not
re-derive them from the master prompt.

{knowledge_instructions}

## Cross-character relational knowledge — do NOT drop

Because you have the full roster in the conversation above, each
envelope can and SHOULD reference other characters by name when
appropriate: "She knows Edric is conducting undisclosed experiments
— she has observed the substance donations." "She knows most of the
Deep's secondary access routes — including the sealed tunnel near
the Gatehouse that the Caretaker does not acknowledge." "He
believes Laith's arguments about the Periphery even though Laith
has not made them publicly."

If character A in the roster has a relationship, suspicion,
admiration, or useful knowledge about character B, that belongs in
A's `known_context` — with B named. Do not compress these into
abstract ("she knows about the Return leadership") when the
specific name and specific knowledge were in the source. The
texture of this story lives in who knows what about whom.

Every `character_id` from the roster you extracted MUST appear
exactly once in `envelopes`, and no envelope may reference a
character_id not in the roster."""


COMBINED_EXTRACTION_INSTRUCTIONS = """\
Extract the COMPLETE import bundle for this master prompt in a single
structured response. You will produce four sub-objects in one JSON
output: `world`, `characters`, `opening`, and `knowledge`.

Every sub-object has its own fidelity standard — spelled out below —
and each should be populated as if it were its own dedicated extraction
pass. Because you are producing them together, you have the advantage
of cross-referencing: the same faction described in `world.lore`
should appear on the right `characters[].public_sheet.faction`; the
same secret in `world.hidden_facts` should land on the plausible
`characters[].private_state.secrets`; `knowledge.envelopes` should
reflect the world you just extracted, not hallucinated parallel
content.

---

## `world` — World Extraction

{world_instructions}

---

## `characters` — Character Extraction

{character_instructions}

---

## `opening` — Opening Scene

{opening_instructions}

---

## `knowledge` — Character Knowledge Envelopes

{knowledge_instructions}

Every `character_id` in `characters` MUST have exactly one matching
entry in `knowledge.envelopes`, and no envelope may reference a
character_id not in `characters`. The two lists are siblings, not
independent extractions."""


# ---------------- Per-stage extractions ----------------

def _log_usage(stage: str, response) -> None:
    """Log cache hit/write telemetry so we can see at a glance whether the
    shared-source prefix is actually cache-sharing. Silent on responses
    without usage (tests)."""
    u = getattr(response, "usage", None) or {}
    if not u:
        return
    logger.info(
        "  usage[%s]: in=%d out=%d cache_read=%d cache_write=%d full=%d",
        stage,
        u.get("prompt_tokens", 0),
        u.get("completion_tokens", 0),
        u.get("cache_read_input_tokens", 0),
        u.get("cache_creation_input_tokens", 0),
        u.get("full_input_tokens", 0),
    )


async def extract_world(client: LLMClient, source: str) -> WorldExtraction:
    logger.info("Extracting world state...")
    response = await client.complete(
        role="narrator",
        messages=[
            {"role": "system", "content": SHARED_SOURCE_SYSTEM.format(source_prompt=source)},
            {"role": "user", "content": WORLD_EXTRACTION_INSTRUCTIONS},
        ],
        response_model=WorldExtraction,
        temperature=0.3,
        max_tokens=MAX_EXTRACTION_TOKENS,
    )
    _log_usage("world", response)
    data: WorldExtraction = response.parsed
    logger.info(
        "  Setting: %s / %s", data.setting.genre or "?", data.setting.tone or "?"
    )
    logger.info(
        "  Locations: %d scenes, Facts: %d, Hidden facts: %d",
        len(data.locations.scene_graph), len(data.facts), len(data.hidden_facts),
    )
    logger.info(
        "  Lore: %d chars, Hidden lore: %d chars, Narrative rules: %d chars",
        len(data.lore), len(data.hidden_lore), len(data.narrative_rules),
    )
    return data


async def extract_characters(client: LLMClient, source: str) -> CharacterListExtraction:
    logger.info("Extracting characters...")
    response = await client.complete(
        role="narrator",
        messages=[
            {"role": "system", "content": SHARED_SOURCE_SYSTEM.format(source_prompt=source)},
            {"role": "user", "content": CHARACTER_EXTRACTION_INSTRUCTIONS},
        ],
        response_model=CharacterListExtraction,
        temperature=0.3,
        max_tokens=MAX_EXTRACTION_TOKENS,
    )
    _log_usage("characters", response)
    data: CharacterListExtraction = response.parsed
    logger.info("  Characters: %d extracted", len(data.characters))
    for c in data.characters:
        player_tag = " [PLAYER]" if c.is_player else ""
        logger.info(
            "    - %s (%s) [%s]%s",
            c.name, c.character_id, c.public_sheet.role or "?", player_tag,
        )
    return data


async def extract_character_knowledge(
    client: LLMClient,
    source: str,
    world: WorldExtraction,
    roster: CharacterListExtraction,
) -> CharacterKnowledgeListExtraction:
    """Batch extraction: one call produces knowledge envelopes for every
    character. Sees the omniscient world (public + hidden) plus the full
    roster with backstories and secrets, and filters per-character."""
    logger.info(
        "Extracting knowledge envelopes for %d characters...",
        len(roster.characters),
    )
    user_prompt = (
        f"{KNOWLEDGE_EXTRACTION_INSTRUCTIONS}\n\n"
        f"## World (omniscient)\n{_format_world_for_knowledge(world)}\n\n"
        f"## Roster\n{_format_roster_for_knowledge(roster)}"
    )
    response = await client.complete(
        role="narrator",
        messages=[
            {"role": "system", "content": SHARED_SOURCE_SYSTEM.format(source_prompt=source)},
            {"role": "user", "content": user_prompt},
        ],
        response_model=CharacterKnowledgeListExtraction,
        temperature=0.3,
        max_tokens=MAX_EXTRACTION_TOKENS,
    )
    _log_usage("knowledge", response)
    data: CharacterKnowledgeListExtraction = response.parsed
    logger.info("  Envelopes: %d produced", len(data.envelopes))
    return data


def _format_world_for_knowledge(world: WorldExtraction) -> str:
    parts: list[str] = []
    s = world.setting
    meta_bits = []
    if s.genre:
        meta_bits.append(f"Genre: {s.genre}")
    if s.era:
        meta_bits.append(f"Era: {s.era}")
    if s.tone:
        meta_bits.append(f"Tone: {s.tone}")
    if meta_bits:
        parts.append("\n".join(meta_bits))
    if s.premise:
        parts.append(f"### Authorial premise (omniscient)\n{s.premise}")
    if world.lore:
        parts.append(f"### Public lore\n{world.lore}")
    if world.hidden_lore:
        parts.append(f"### Hidden lore (not public)\n{world.hidden_lore}")
    if world.facts:
        parts.append("### Public facts\n" + "\n".join(f"- {f}" for f in world.facts))
    if world.hidden_facts:
        parts.append(
            "### Hidden facts (not public)\n"
            + "\n".join(f"- {f}" for f in world.hidden_facts)
        )
    return "\n\n".join(parts)


def _format_roster_for_knowledge(roster: CharacterListExtraction) -> str:
    parts: list[str] = []
    for c in roster.characters:
        lines = [f"### {c.name} (character_id: `{c.character_id}`)"]
        if c.public_sheet.role:
            lines.append(f"Role: {c.public_sheet.role}")
        if c.public_sheet.faction:
            lines.append(f"Faction: {c.public_sheet.faction}")
        if c.is_player:
            lines.append("(PLAYER SLOT)")
        if c.backstory:
            lines.append(f"Backstory: {c.backstory}")
        if c.private_state.secrets:
            lines.append(
                "Secrets:\n" + "\n".join(f"- {s}" for s in c.private_state.secrets)
            )
        if c.private_state.goals:
            lines.append(
                "Goals:\n" + "\n".join(f"- {g}" for g in c.private_state.goals)
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


async def extract_opening(client: LLMClient, source: str) -> OpeningExtraction:
    logger.info("Generating opening narrative...")
    response = await client.complete(
        role="narrator",
        messages=[
            {"role": "system", "content": SHARED_SOURCE_SYSTEM.format(source_prompt=source)},
            {"role": "user", "content": OPENING_EXTRACTION_INSTRUCTIONS},
        ],
        response_model=OpeningExtraction,
        temperature=0.5,
        max_tokens=MAX_EXTRACTION_TOKENS,
    )
    _log_usage("opening", response)
    data: OpeningExtraction = response.parsed
    logger.info("  Opening narrative: %d chars", len(data.text))
    return data


# ---------------- Checkpoint assembly ----------------

def build_checkpoint(
    world: WorldExtraction,
    roster: CharacterListExtraction,
    opening: OpeningExtraction,
    knowledge: CharacterKnowledgeListExtraction,
    story_id: str,
) -> CheckpointFile:
    """Assemble the three extractions into a CheckpointFile. Non-fatal warnings
    are emitted for scene-graph integrity issues and duplicate character ids."""
    session = SessionState(
        session_id=story_id,
        story_id=story_id,
        turn_index=0,
    )

    setting = StorySetting(
        genre=world.setting.genre,
        era=world.setting.era,
        tone=world.setting.tone,
        premise=world.setting.premise,
    )

    scene_graph: dict[str, dict] = {}
    for scene in world.locations.scene_graph:
        if not scene.scene_id:
            logger.warning("Scene with empty scene_id dropped: name=%r", scene.name)
            continue
        if scene.scene_id in scene_graph:
            logger.warning("Duplicate scene_id %r — keeping first occurrence", scene.scene_id)
            continue
        scene_graph[scene.scene_id] = {
            "name": scene.name,
            "description": scene.description,
            "connected_to": scene.connected_to,
            "properties": {},
        }
    locations = LocationState(
        current_scene_id=world.locations.current_scene_id,
        scene_graph=scene_graph,
    )

    physics = PhysicsRuleset(
        strength_limits=world.physics_ruleset.strength_limits,
        magic_enabled=world.physics_ruleset.magic_enabled,
    )

    world_state = WorldState(
        locations=locations,
        facts=world.facts,
        physics_ruleset=physics,
        setting=setting,
        lore=world.lore,
        hidden_lore=world.hidden_lore,
        hidden_facts=world.hidden_facts,
    )

    config = SessionConfig(narrative_rules=world.narrative_rules)

    envelopes_by_id = {e.character_id: e for e in knowledge.envelopes}

    characters: list[CharacterRecord] = []
    seen_ids: set[str] = set()
    for cd in roster.characters:
        if cd.character_id in seen_ids:
            logger.warning("Duplicate character_id %r — keeping first occurrence", cd.character_id)
            continue
        seen_ids.add(cd.character_id)

        envelope = envelopes_by_id.get(cd.character_id)
        if envelope is None:
            logger.warning(
                "No knowledge envelope for %s — character will start with empty known_context",
                cd.character_id,
            )

        characters.append(
            CharacterRecord(
                character_id=cd.character_id,
                name=cd.name,
                status=CharacterStatus(cd.status),
                location=cd.location,
                is_player=cd.is_player,
                public_sheet=PublicSheet(
                    role=cd.public_sheet.role,
                    appearance=cd.public_sheet.appearance,
                    faction=cd.public_sheet.faction,
                ),
                private_state=PrivateState(
                    goals=cd.private_state.goals,
                    current_objectives=cd.private_state.current_objectives,
                    secrets=cd.private_state.secrets,
                    intentions_enabled=cd.private_state.intentions_enabled,
                ),
                backstory=cd.backstory,
                personality=cd.personality,
                known_context=envelope.known_context if envelope else "",
            )
        )

    if world.locations.current_scene_id and world.locations.current_scene_id not in scene_graph:
        logger.warning(
            "current_scene_id %r is not in scene_graph — runtime will fall back to empty scene context",
            world.locations.current_scene_id,
        )
    for scene in world.locations.scene_graph:
        for target in scene.connected_to:
            if target not in scene_graph:
                logger.warning(
                    "Scene %r connects to %r but %r is not defined in scene_graph",
                    scene.scene_id, target, target,
                )

    player_chars = [c for c in roster.characters if c.is_player]
    if player_chars:
        logger.info(
            "Player character slot(s): %s",
            ", ".join(f"{c.name} ({c.character_id})" for c in player_chars),
        )
    else:
        logger.info("No character marked is_player=true — roster is fully NPC")

    return CheckpointFile(
        session=session,
        importer_version=IMPORTER_VERSION,
        opening_narrative=opening.text,
        world_state=world_state,
        characters=characters,
        config=config,
    )


# ---------------- Top-level pipeline ----------------

PRESERVATION_ANALYSIS_SYSTEM = """\
You are auditing the fidelity of an interactive-fiction story import.

You will see two things:
1. The original master prompt the author wrote.
2. The structured checkpoint the importer extracted from that prompt.

Your job is to report how much of the source's content, nuance, and
detail survived into the structured form. Be direct and concrete.

## coverage_rating
- "high": essentially everything meaningful survived; at most minor
  reorganizations of phrasing.
- "medium": most substance survived but specific details, characters,
  factions, or nuances were dropped or flattened.
- "low": substantial content is missing or generically summarized.

## dropped_topics
Specific topics, characters, factions, secrets, rules, or details that
appear in the source but are NOT present in the checkpoint. Name each
concretely ("the Regent's memorial refusal", "the Isolation Mandate's
origin story", "Nessa's handler name Calder"). Aim for 0-12 items.

## compressed_topics
Topics that survived but lost meaningful detail (a paragraph became a
sentence; a character's five-point arc became a one-line role). Name
what got compressed. Aim for 0-8 items.

## preservation_notes
One or two short paragraphs of free-form observations — patterns of
loss, quality of character capture, anything the operator should know.
Not a re-listing of the items above.

Respond with ONLY valid JSON matching the schema."""


class _AnalysisLLMOutput(BaseModel):
    """LLM-only fields of ImportAnalysis. Kept internal to this module;
    the deterministic fields (char/word counts, duration) are filled
    programmatically and merged into the final ImportAnalysis."""
    model_config = ConfigDict(extra="forbid")

    coverage_rating: str  # "high" | "medium" | "low" — validated by caller
    dropped_topics: list[str] = []
    compressed_topics: list[str] = []
    preservation_notes: str = ""


def _serialize_checkpoint_for_analysis(ckpt: CheckpointFile) -> str:
    """Render the text content of a checkpoint into a single string so the
    preservation auditor can compare against the source prompt without
    drowning in JSON structural noise."""
    ws = ckpt.world_state
    parts: list[str] = []

    setting_bits = []
    if ws.setting.genre:
        setting_bits.append(f"Genre: {ws.setting.genre}")
    if ws.setting.era:
        setting_bits.append(f"Era: {ws.setting.era}")
    if ws.setting.tone:
        setting_bits.append(f"Tone: {ws.setting.tone}")
    if ws.setting.premise:
        setting_bits.append(f"Premise:\n{ws.setting.premise}")
    if setting_bits:
        parts.append("# Setting\n" + "\n\n".join(setting_bits))

    if ws.lore:
        parts.append(f"# Public Lore\n{ws.lore}")
    if ws.hidden_lore:
        parts.append(f"# Hidden Lore\n{ws.hidden_lore}")
    if ws.facts:
        parts.append("# Public Facts\n" + "\n".join(f"- {f}" for f in ws.facts))
    if ws.hidden_facts:
        parts.append("# Hidden Facts\n" + "\n".join(f"- {f}" for f in ws.hidden_facts))
    if ckpt.config.narrative_rules:
        parts.append(f"# Narrative Rules\n{ckpt.config.narrative_rules}")

    if ws.locations.scene_graph:
        scene_lines = ["# Scene Graph"]
        for sid, scene in ws.locations.scene_graph.items():
            name = scene.get("name", sid) if isinstance(scene, dict) else sid
            desc = scene.get("description", "") if isinstance(scene, dict) else ""
            scene_lines.append(f"## {name} ({sid})\n{desc}")
        parts.append("\n\n".join(scene_lines))

    if ckpt.opening_narrative:
        parts.append(f"# Opening Narrative\n{ckpt.opening_narrative}")

    if ckpt.characters:
        char_lines = ["# Characters"]
        for c in ckpt.characters:
            char_lines.append(f"## {c.name} ({c.character_id})")
            if c.public_sheet.role:
                char_lines.append(f"Role: {c.public_sheet.role}")
            if c.public_sheet.faction:
                char_lines.append(f"Faction: {c.public_sheet.faction}")
            if c.public_sheet.appearance:
                char_lines.append(f"Appearance: {c.public_sheet.appearance}")
            if c.backstory:
                char_lines.append(f"Backstory: {c.backstory}")
            if c.personality:
                char_lines.append(f"Personality: {c.personality}")
            if c.known_context:
                char_lines.append(f"Known Context: {c.known_context}")
            if c.private_state.goals:
                char_lines.append(
                    "Goals:\n" + "\n".join(f"- {g}" for g in c.private_state.goals)
                )
            if c.private_state.current_objectives:
                char_lines.append(
                    "Current Objectives:\n"
                    + "\n".join(f"- {o}" for o in c.private_state.current_objectives)
                )
            if c.private_state.secrets:
                char_lines.append(
                    "Secrets:\n" + "\n".join(f"- {s}" for s in c.private_state.secrets)
                )
        parts.append("\n\n".join(char_lines))

    return "\n\n".join(parts)


async def run_preservation_analysis(
    client: LLMClient,
    source_text: str,
    checkpoint: CheckpointFile,
) -> ImportAnalysis:
    """Audit the fidelity of an import. Runs as a separate LLM call so
    callers can defer or background it. Deterministic metrics (char/word
    counts, duration) are computed locally; the rest comes from the LLM.

    Caller is responsible for persisting the result (patching the
    checkpoint's `import_analysis` field)."""
    t_start = time.monotonic()

    source_chars = len(source_text)
    source_words = len(source_text.split())
    output_text = _serialize_checkpoint_for_analysis(checkpoint)
    output_chars = len(output_text)
    output_words = len(output_text.split())

    logger.info(
        "Preservation analysis: source=%d chars / %d words, "
        "output=%d chars / %d words",
        source_chars, source_words, output_chars, output_words,
    )

    response = await client.complete(
        role="narrator",
        messages=[
            {"role": "system", "content": PRESERVATION_ANALYSIS_SYSTEM},
            {"role": "user", "content": (
                "## Source master prompt\n\n"
                f"{source_text}\n\n"
                "## Extracted checkpoint contents\n\n"
                f"{output_text}"
            )},
        ],
        response_model=_AnalysisLLMOutput,
        temperature=0.2,
        max_tokens=8000,
    )
    llm_out: _AnalysisLLMOutput = response.parsed

    rating = llm_out.coverage_rating.strip().lower()
    if rating not in ("high", "medium", "low"):
        logger.warning("Unexpected coverage_rating %r; coercing to 'unknown'", rating)
        rating = "unknown"

    duration = time.monotonic() - t_start
    model_name = getattr(response.raw_response, "model", "") or ""

    logger.info(
        "Preservation analysis complete: coverage=%s, dropped=%d, compressed=%d (%.1fs)",
        rating, len(llm_out.dropped_topics), len(llm_out.compressed_topics),
        duration,
    )

    return ImportAnalysis(
        source_chars=source_chars,
        source_words=source_words,
        output_chars=output_chars,
        output_words=output_words,
        coverage_rating=rating,
        dropped_topics=llm_out.dropped_topics,
        compressed_topics=llm_out.compressed_topics,
        preservation_notes=llm_out.preservation_notes,
        duration_s=duration,
        model=model_name,
    )


async def run_import(
    client: LLMClient,
    source_text: str,
    story_id: str,
) -> CheckpointFile:
    """Run the four-stage import pipeline and return the assembled checkpoint.

    Pipeline shape:
      Stage 1: world extraction — serial. Primes the shared cache.
      Stages 2 + 3: characters + opening — run in parallel. Both read the
        source prompt from cache (primed by stage 1), so starting them
        concurrently doubles up the compute without re-paying the prefix
        cache write.
      Stage 4: knowledge envelopes — runs after stage 2 completes. Depends
        on the extracted world + roster.

    Preservation analysis is intentionally NOT part of this function.
    Callers kick it off separately (background on the bot path, inline on
    the CLI path) via run_preservation_analysis().

    Does NOT write the checkpoint to disk — callers are responsible for
    persisting via CheckpointManager.save() (or writing the JSON directly).
    """
    t_start = time.monotonic()
    logger.info(
        "Starting import (pipeline %s): source prompt %d chars, ~%d words",
        IMPORTER_VERSION, len(source_text), len(source_text.split()),
    )

    # Stage 1: serial so the shared-source cache prefix writes exactly once
    # before the parallel stages consume it.
    world = await extract_world(client, source_text)

    # Stages 2 + 3: parallel. Both read from the primed cache.
    roster, opening = await asyncio.gather(
        extract_characters(client, source_text),
        extract_opening(client, source_text),
    )

    # Stage 4: depends on world + roster. Sees omniscient world and filters
    # per-character knowledge envelopes.
    knowledge = await extract_character_knowledge(
        client, source_text, world, roster,
    )

    checkpoint = build_checkpoint(world, roster, opening, knowledge, story_id)
    logger.info(
        "Import complete in %.1fs (%d characters, %d scenes)",
        time.monotonic() - t_start,
        len(checkpoint.characters),
        len(checkpoint.world_state.locations.scene_graph),
    )
    return checkpoint


# ---------------- Combined single-call path ----------------


class _CombinedImportResult:
    """Return bundle for `run_import_combined` — ships the assembled
    checkpoint plus the priming messages + raw assistant content so the
    caller can cheaply continue the conversation for the preservation-
    analysis pass.

    We pack these together so the analysis continuation doesn't have to
    re-render the source prompt or re-serialize the combined extraction
    blob — the messages list is the exact conversation to replay."""
    __slots__ = ("checkpoint", "priming_messages", "assistant_text")

    def __init__(
        self,
        checkpoint: CheckpointFile,
        priming_messages: list[dict[str, str]],
        assistant_text: str,
    ):
        self.checkpoint = checkpoint
        self.priming_messages = priming_messages
        self.assistant_text = assistant_text


def _combined_user_prompt() -> str:
    """Assemble the single user message containing all four sets of
    extraction instructions. Reuses the existing per-stage instruction
    blocks so there's one source of truth for extraction guidance."""
    return COMBINED_EXTRACTION_INSTRUCTIONS.format(
        world_instructions=WORLD_EXTRACTION_INSTRUCTIONS,
        character_instructions=CHARACTER_EXTRACTION_INSTRUCTIONS,
        opening_instructions=OPENING_EXTRACTION_INSTRUCTIONS,
        knowledge_instructions=KNOWLEDGE_EXTRACTION_INSTRUCTIONS,
    )


async def run_import_combined(
    client: LLMClient,
    source_text: str,
    story_id: str,
) -> _CombinedImportResult:
    """Single-call importer. Produces world + characters + opening +
    knowledge in one structured-output response and returns the
    checkpoint along with the exact messages used so the preservation
    analysis pass can replay them as cached history.

    Writes a cache breakpoint on the user message (via
    `cache_user_tail=True`) so the follow-up analysis call reads
    [system, user] as cached prefix, paying only for the new assistant
    echo + analysis question.

    Does NOT persist the checkpoint — callers save via CheckpointManager
    or direct JSON write.
    """
    t_start = time.monotonic()
    logger.info(
        "Starting combined import (pipeline %s): source prompt %d chars, ~%d words",
        IMPORTER_VERSION, len(source_text), len(source_text.split()),
    )

    system_text = SHARED_SOURCE_SYSTEM.format(source_prompt=source_text)
    user_text = _combined_user_prompt()
    priming_messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    response = await client.complete(
        role="narrator",
        messages=priming_messages,
        response_model=CombinedImportExtraction,
        temperature=0.3,
        max_tokens=MAX_EXTRACTION_TOKENS,
        cache_user_tail=True,
    )
    _log_usage("combined", response)
    bundle: CombinedImportExtraction = response.parsed

    logger.info(
        "  Setting: %s / %s",
        bundle.world.setting.genre or "?", bundle.world.setting.tone or "?",
    )
    logger.info(
        "  Locations: %d scenes, Facts: %d, Hidden facts: %d",
        len(bundle.world.locations.scene_graph),
        len(bundle.world.facts),
        len(bundle.world.hidden_facts),
    )
    logger.info("  Characters: %d extracted", len(bundle.characters.characters))
    logger.info("  Envelopes: %d produced", len(bundle.knowledge.envelopes))
    logger.info("  Opening narrative: %d chars", len(bundle.opening.text))

    checkpoint = build_checkpoint(
        bundle.world,
        bundle.characters,
        bundle.opening,
        bundle.knowledge,
        story_id,
    )
    logger.info(
        "Combined import complete in %.1fs (%d characters, %d scenes)",
        time.monotonic() - t_start,
        len(checkpoint.characters),
        len(checkpoint.world_state.locations.scene_graph),
    )

    return _CombinedImportResult(
        checkpoint=checkpoint,
        priming_messages=priming_messages,
        assistant_text=response.content or "",
    )


# ---------------- Three-call path (v5) ----------------


def _world_only_user_prompt() -> str:
    """Assemble the Call-1 user message: world only. The per-stage
    WORLD_EXTRACTION_INSTRUCTIONS already cover the entire `world`
    schema in detail; no extra wrapper is needed."""
    return WORLD_EXTRACTION_INSTRUCTIONS


def _chars_and_opening_user_prompt() -> str:
    """Assemble the Call-2 user message: characters + opening as a
    continuation that reads Call 1's world output as cached history."""
    return CHARS_AND_OPENING_CONTINUATION_INSTRUCTIONS.format(
        character_instructions=CHARACTER_EXTRACTION_INSTRUCTIONS,
        opening_instructions=OPENING_EXTRACTION_INSTRUCTIONS,
    )


def _knowledge_user_prompt() -> str:
    """Assemble the Call-3 user message: knowledge envelopes as a
    continuation. Reuses the per-stage knowledge instructions plus the
    cross-character relational emphasis that v3's single-call pass
    tended to drop."""
    return KNOWLEDGE_CONTINUATION_INSTRUCTIONS.format(
        knowledge_instructions=KNOWLEDGE_EXTRACTION_INSTRUCTIONS,
    )


async def run_import_two_call(
    client: LLMClient,
    source_text: str,
    story_id: str,
) -> _CombinedImportResult:
    """Three-call importer (v5; function name retained for caller
    compatibility — bridge + CLI + tests still import this symbol).
    Call 1 extracts `world`; Call 2 extracts `characters` + `opening`
    as a continuation that reads Call 1 as cached history; Call 3
    extracts `knowledge` envelopes as a continuation that reads the
    full two-call chain.

    Why three calls instead of v4's two: the combined
    world+characters+opening pass exceeded Sonnet 4.6's 64K output
    cap on a 95KB master prompt, truncating the response mid-string
    around column 303K (~60K tokens of dense JSON). Splitting world
    off gives each call the full output budget. Cross-referencing
    quality is preserved because every call after Call 1 reads the
    prior chain as cached history.

    Returns a `_CombinedImportResult` shaped the same as v4 so
    downstream callers (EngineBridge, preservation continuation) don't
    care which pipeline produced the checkpoint. `priming_messages`
    contains the FULL three-call conversation up through Call-3's
    user turn; `assistant_text` is the Call-3 assistant reply — the
    preservation analysis tacks itself on as Call 4 with the same
    continuation helper.
    """
    t_start = time.monotonic()
    logger.info(
        "Starting three-call import (pipeline %s): source prompt %d chars, ~%d words",
        IMPORTER_VERSION, len(source_text), len(source_text.split()),
    )

    system_text = SHARED_SOURCE_SYSTEM.format(source_prompt=source_text)
    world_user = _world_only_user_prompt()
    world_messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": world_user},
    ]

    # Call 1 — world only.
    t_world = time.monotonic()
    world_response = await client.complete(
        role="narrator",
        messages=world_messages,
        response_model=WorldExtraction,
        temperature=0.3,
        max_tokens=MAX_EXTRACTION_TOKENS,
        cache_user_tail=True,
    )
    _log_usage("world", world_response)
    world: WorldExtraction = world_response.parsed
    logger.info(
        "  World (%.1fs): setting=%s/%s, scenes=%d, facts=%d, hidden_facts=%d",
        time.monotonic() - t_world,
        world.setting.genre or "?", world.setting.tone or "?",
        len(world.locations.scene_graph),
        len(world.facts),
        len(world.hidden_facts),
    )

    # Call 2 — characters + opening (continuation; reads Call 1 as
    # cached history thanks to cache_user_tail on Call 1's user turn).
    chars_user = _chars_and_opening_user_prompt()
    chars_messages = world_messages + [
        {"role": "assistant", "content": world_response.content or ""},
        {"role": "user", "content": chars_user},
    ]
    t_chars = time.monotonic()
    chars_response = await client.complete(
        role="narrator",
        messages=chars_messages,
        response_model=CharsAndOpeningExtraction,
        temperature=0.3,
        max_tokens=MAX_EXTRACTION_TOKENS,
        cache_user_tail=True,
    )
    _log_usage("chars+opening", chars_response)
    chars_and_opening: CharsAndOpeningExtraction = chars_response.parsed
    logger.info(
        "  Chars+Opening (%.1fs): characters=%d, opening=%d chars",
        time.monotonic() - t_chars,
        len(chars_and_opening.characters.characters),
        len(chars_and_opening.opening.text),
    )

    # Call 3 — knowledge envelopes (continuation; reads the full
    # two-call chain as cached history).
    knowledge_user = _knowledge_user_prompt()
    knowledge_messages = chars_messages + [
        {"role": "assistant", "content": chars_response.content or ""},
        {"role": "user", "content": knowledge_user},
    ]
    t_know = time.monotonic()
    knowledge_response = await client.complete(
        role="narrator",
        messages=knowledge_messages,
        response_model=CharacterKnowledgeListExtraction,
        temperature=0.3,
        max_tokens=MAX_EXTRACTION_TOKENS,
        cache_user_tail=True,
    )
    _log_usage("knowledge", knowledge_response)
    knowledge: CharacterKnowledgeListExtraction = knowledge_response.parsed
    logger.info(
        "  Knowledge (%.1fs): %d envelopes produced",
        time.monotonic() - t_know, len(knowledge.envelopes),
    )

    checkpoint = build_checkpoint(
        world,
        chars_and_opening.characters,
        chars_and_opening.opening,
        knowledge,
        story_id,
    )
    logger.info(
        "Three-call import complete in %.1fs (%d characters, %d scenes)",
        time.monotonic() - t_start,
        len(checkpoint.characters),
        len(checkpoint.world_state.locations.scene_graph),
    )

    # Pack the FULL three-call conversation into priming_messages so
    # the preservation analysis continuation reads the whole chain as
    # its cached prefix.
    return _CombinedImportResult(
        checkpoint=checkpoint,
        priming_messages=knowledge_messages,
        assistant_text=knowledge_response.content or "",
    )


async def run_preservation_analysis_continuation(
    client: LLMClient,
    priming_messages: list[dict[str, str]],
    assistant_text: str,
    source_text: str,
    checkpoint: CheckpointFile,
) -> ImportAnalysis:
    """Preservation analysis that piggybacks on the combined-import call's
    conversation. The second call sends
    [system, user1, assistant1 (combined blob), user2 (analysis question)]
    which reads [system, user1] as cached prefix and pays fresh only for
    assistant_text + the analysis prompt.

    Mirrors `run_preservation_analysis` for fields; differs only in how
    the API call is framed.
    """
    t_start = time.monotonic()

    source_chars = len(source_text)
    source_words = len(source_text.split())
    output_text = _serialize_checkpoint_for_analysis(checkpoint)
    output_chars = len(output_text)
    output_words = len(output_text.split())

    logger.info(
        "Preservation analysis (continuation): source=%d chars / %d words, "
        "output=%d chars / %d words",
        source_chars, source_words, output_chars, output_words,
    )

    analysis_prompt = (
        "Now audit what you just extracted against the source master prompt.\n\n"
        "You are grading your own extraction for fidelity. Rate coverage "
        "honestly — every dropped topic and every compression is a defect. "
        "The checkpoint fields you produced are listed below for reference; "
        "compare them back against the master prompt you were given at the "
        "start of this conversation.\n\n"
        f"{PRESERVATION_ANALYSIS_SYSTEM}\n\n"
        "## Extracted checkpoint contents (for reference)\n\n"
        f"{output_text}"
    )

    followup_messages = priming_messages + [
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": analysis_prompt},
    ]

    response = await client.complete(
        role="narrator",
        messages=followup_messages,
        response_model=_AnalysisLLMOutput,
        temperature=0.2,
        max_tokens=8000,
    )
    _log_usage("preservation_analysis", response)
    llm_out: _AnalysisLLMOutput = response.parsed

    rating = llm_out.coverage_rating.strip().lower()
    if rating not in ("high", "medium", "low"):
        logger.warning(
            "Unexpected coverage_rating %r; coercing to 'unknown'", rating
        )
        rating = "unknown"

    duration = time.monotonic() - t_start
    model_name = getattr(response.raw_response, "model", "") or ""

    logger.info(
        "Preservation analysis continuation complete: coverage=%s, "
        "dropped=%d, compressed=%d (%.1fs)",
        rating, len(llm_out.dropped_topics),
        len(llm_out.compressed_topics), duration,
    )

    return ImportAnalysis(
        source_chars=source_chars,
        source_words=source_words,
        output_chars=output_chars,
        output_words=output_words,
        coverage_rating=rating,
        dropped_topics=llm_out.dropped_topics,
        compressed_topics=llm_out.compressed_topics,
        preservation_notes=llm_out.preservation_notes,
        duration_s=duration,
        model=model_name,
    )
