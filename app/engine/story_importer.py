"""Master-prompt → CheckpointFile import pipeline.

Active path: `run_import_two_call` (name retained for caller
compatibility; v10 now runs six LLM calls that share a cached
source-prompt prefix):
  1. PUBLIC world (setting, public lore, public facts, physics,
     narrative rules) — serial. Primes the shared cache so subsequent
     calls read instead of re-write.
  2. HIDDEN world (hidden_lore, hidden_facts) — continuation that
     reads Call 1 as cached history. Split off from Call 1 because
     dense conspiracy stories pack so much into hidden content that
     a combined public+hidden call still overran the 64K output cap.
  3. Locations — continuation that reads Calls 1-2.
  4. Characters — continuation that reads Calls 1-3. (v9 dropped the
     paired opening-prose extraction; the opening beat is now
     composed at runtime by the router and rendered by the narrator
     on the first `(begin)` turn, so there is no authored opener to
     extract here.)
     v10 adds player-safe/public and private identity descriptions.
  5. Knowledge envelope extraction — continuation that sees the
     full extracted world + roster and outputs per-character
     `known_context`.
  6. Player primer — short player-facing briefing for `/story start`.

Preservation analysis (`run_preservation_analysis_continuation`) is a
separate follow-up call callers run independently — on the bot path it
fires background via `asyncio.create_task`; on the CLI path it runs
inline.

Older single-call (`run_import_combined`) and parallel four-stage
(`run_import`) paths are retained for quality comparisons but are not
the active pipeline.

All calls use `response_model=<Pydantic class>` so the API enforces schema
validity server-side.

Used by both `scripts/import_story.py` (CLI) and
`app/bot/engine_bridge.py` (Discord /story import).

## TODO: single-call import experiment

The current 4-stage pipeline was chosen for cache-friendly, focused
prompts — each call narrows the LLM onto one extraction task with the
shared source prefix cached. A tempting optimization: collapse stages
1-4 into one structured-output call that emits `{world, characters,
envelopes}` as a single JSON blob. That would cut per-stage
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

from app.engine.model_config_sync import sync_checkpoint_runtime_models
from app.llm.client import LLMClient
from app.schemas.characters import (
    CharacterDescriptions,
    CharacterRecord,
    CharacterStatus,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile, ImportAnalysis
from app.schemas.import_extraction import (
    CharacterKnowledgeListExtraction,
    CharacterListExtraction,
    HiddenWorldExtraction,
    PlayerPrimerExtraction,
    PublicWorldExtraction,
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
# v5 → v6: four-call pipeline. v5's Call 1 (`WorldExtraction`) STILL
# truncated mid-string on the same 95KB master prompt. Split world into
# narrower follow-up calls. Cache continuity preserved.
# v6 → v7: five-call pipeline. v6's Call 1 (`WorldSkeletonExtraction`)
# STILL truncated at column 265,192 of JSON on the same 95KB master
# prompt — dense conspiracy stories pack so much into lore + hidden_
# lore + hidden_facts that the skeleton alone overran 64K output
# tokens. Split skeleton into PUBLIC world (Call 1: setting / lore /
# facts / physics / narrative_rules) and HIDDEN world (Call 2:
# hidden_lore / hidden_facts) as two consecutive calls. The structural
# split also matches the engine's adjudication-vs-public contract.
# Pipeline is now: public-world → hidden-world → locations → chars+
# opening → knowledge.
# v7 → v8: six-call pipeline. Adds Call 6 (player_primer) — a short
# (≤200 word, ≤2 paragraph) second-person, truck-kun-framed primer
# the bot displays on /story start before the player picks a
# character. Replaces the omniscient briefing dump that used to
# leak roster names, factions, and lore. Stamped onto
# `CheckpointFile.player_primer` so every session loaded from this
# story shares it. Older checkpoints load with `player_primer=""`
# and `render_briefing` falls back to a setting-only stub.
# v8 → v9: dropped authored opening prose. The `OpeningExtraction`
# schema, `OPENING_EXTRACTION_INSTRUCTIONS`, and the per-stage
# `extract_opening` call are gone; Call 4 now extracts characters
# alone (renamed from "characters + opening"). Rationale: the
# authored opener was a parallel context channel that fought the
# rest of the engine — it was POV-bound to a single character (the
# author writes "you" assuming a specific protagonist), it created
# a special-case render path that bypassed the router→narrator
# pipeline, and it forced race-window mitigations on `/join`. The
# opening beat is now composed at runtime by the omniscient
# router (using world_state, character_records, and the `(begin)`
# OOC directive) and rendered per-POV by the narrator on the first
# turn — every turn now goes through the same code path. Older
# checkpoints with `opening_narrative` populated load cleanly
# (Pydantic v2 default `extra='ignore'` drops the field) and the
# field is no longer consulted by any runtime code.
# v9 → v10: character extraction now emits `descriptions.public` and
# `descriptions.private`. Public descriptions are player-safe identity
# glosses for narrator context; private descriptions retain authorial /
# spoiler-bearing identity context without exposing it to narrator prompts.
IMPORTER_VERSION = "v10"


class CombinedImportExtraction(BaseModel):
    """(v3) Wraps the legacy single-call extraction in one structured-
    output response. Retained so the v3 path stays runnable for
    quality comparisons; the active importer is v10 (six-call).

    Required fields only — the structured-output grammar compiler
    handles deeply-nested schemas more reliably when optionals are
    minimized.

    v9 dropped the `opening` member (authored opening prose is no
    longer extracted; the router composes the opening at runtime).
    """
    world: WorldExtraction
    characters: CharacterListExtraction
    knowledge: CharacterKnowledgeListExtraction


class CoreImportExtraction(BaseModel):
    """(v4) First of two calls in the legacy two-call pipeline.
    Produces world + characters in one structured-output response.
    Knowledge envelopes are generated by a focused Call 2 that reads
    this conversation as cached history — the split gives envelopes
    their own attention budget and preserves cross-character
    relational detail that the v3 single-call pass compressed out.

    v9 dropped the `opening` member here too (see module docstring
    and the IMPORTER_VERSION v8→v9 note).
    """
    world: WorldExtraction
    characters: CharacterListExtraction


# ---------------- Extraction prompts ----------------

SHARED_SOURCE_SYSTEM = """\
<role>
You are a structured extraction specialist for an interactive-fiction engine.
</role>

<instructions>
The source prompt below is the single authoritative input. Every extraction
task you receive will ask for a different slice of it. Your outputs must be
faithful to the source: preserve every distinct detail, relationship,
personality beat, faction, secret, and motivation the source describes.
Compression, summarization, or rounding is a quality defect — the engine
consumes your output verbatim, so anything you drop is permanently lost.
</instructions>

<source_prompt>
## Source Prompt
{source_prompt}
</source_prompt>"""


PUBLIC_WORLD_EXTRACTION_INSTRUCTIONS = """\
Extract the PUBLIC world from the source prompt into the requested
schema. **This call covers ONLY the publicly-known content** — and
only the WORLD layer (setting + lore + facts + physics + narrative
rules). Other layers are produced by focused follow-up calls that
will read your output here as cached history:

- Hidden / spoiler / conspiracy content (hidden_lore, hidden_facts) →
  Call 2. Do not anticipate it here, do not even hint at it.
- **Characters** (rosters, backstories, personalities, secrets,
  goals, factions) → Call 3. Do NOT put character profiles into
  `lore`. Do NOT include character names, character histories, or
  character-specific detail in any field below. The world layer is
  about the WORLD; the character layer is about the people IN it.
- **Opening narrative prose**: There is no authored-opening
  extraction stage in this pipeline. The opening beat is composed
  at runtime by the router using the world + character state you
  extract here. Do NOT include any second-person prose, scene-
  setting passages, or "you walk into…" framing here. The world
  layer is reference material — facts and lore the engine will
  cite; not performance material the player will read.

## Length budget

Each prose field below has a target size. Going substantially over
target signals you've conflated layers — you're putting character or
hidden content into world fields. The 64K output cap
(hard) means "exhaustive on every detail" is incompatible with
"single big string fields"; lean into the per-call decomposition
this pipeline provides.

## Field guidance

- **setting.premise**: 200-1500 chars. The story's core situation
  with enough detail for someone who has never read the source to
  understand who is involved, what's at stake, and what tension
  drives the story. PUBLIC framing — describe the situation as the
  in-world public understands it, not the authorial-omniscient
  version. Do NOT recap entire lore sections here — premise is a
  high-level orientation.

- **lore**: 3,000-12,000 chars target; HARD CEILING 25,000 chars.
  COMMON-KNOWLEDGE world lore only — history, factions, political
  systems, laws, magic systems, religions, key publicly-known
  events. Organize as focused paragraphs (one per major concept).
  Mirror the source's depth on world topics, but do NOT echo the
  source verbatim — extract the substance.

  If the source has multiple distinct world-content sections (e.g.
  "The World" + "Currency" + "The Five Courts" + "The Assembly"),
  cover each in 1-3 focused paragraphs. Don't reproduce 35KB of
  source text into a 35KB lore field; the runtime narrator reads
  this material as reference, not as a script. If something in the
  source is described as mysterious to the in-world public, describe
  it as a mystery here — do not explain the resolution. (The
  resolution belongs in Call 2's `hidden_lore`.)

- **facts**: 5-50 entries. Each fact is one concrete, publicly-known
  current-state observation about the world. Tightly scoped — one
  proposition per entry. Examples: "The borders are sealed by an
  Isolation Mandate." "The Athenaeum has been closed for forty
  years." NOT paragraph-length composites; those go in `lore`.

- **physics_ruleset.strength_limits**: A short phrase describing the
  baseline physical capability of the story's characters
  ("human_baseline", "augmented_human", "low_magic_enhanced", etc.).

- **physics_ruleset.magic_enabled**: true if the story has any
  supernatural force the characters use or encounter; false otherwise.

- **narrative_rules**: ONLY the story-specific narrative delta. The
  engine's narrator, router, and character-agent prompts already
  handle the generic craft discipline that every literary story shares
  — prose rules against stock phrases and filler, pacing (0-5 short
  paragraphs per response, stop when an NPC finishes speaking),
  information asymmetry enforced structurally by the pipeline,
  earned-outcomes / deferral posture, "never speak for the player," no
  menu options, no unearned sycophancy, no consolation signals on
  failure, rivals played to win, NPCs-have-lives-between-turns. **Do
  not restate any of that here.**

  What DOES belong in `narrative_rules`:
    - The story's signature structural constraint (Article Nineteen / a
      ticking clock / Price of Crossing / an Isolation Mandate — the
      named, world-specific pressure that shapes every scene).
    - Faction-response triggers and escalation logic specific to this
      conspiracy or antagonist structure.
    - Per-story tone or POV deviations from engine defaults (e.g.
      second-person-present, an unusual narrator posture).
    - Moral framing of antagonists that's unique to this world's
      ethics (e.g. "the conspirators' arguments should be genuinely
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
  which parts the engine already handles. Target: 500-2000 chars;
  HARD CEILING 5,000 chars."""


HIDDEN_WORLD_EXTRACTION_INSTRUCTIONS = """\
Extract the HIDDEN world content from the source prompt. The PUBLIC
world (setting, lore, facts, physics, narrative rules) is ALREADY in
the conversation above — do not re-extract it.

This call's job is the spoiler / conspiracy / plot-secret content
only. These fields are ADJUDICATION-ONLY — they never reach the
player-facing narrator output. They power the engine's omniscient
adjudication layer (so character agents and the router can act with
full knowledge of the truth even when their characters cannot).

If the source contains no hidden information (a transparent slice-of-
life story, no conspiracy, no engineered events) leave both fields
empty (`""` and `[]`).

## Field guidance

- **hidden_lore**: 0-15,000 chars target; HARD CEILING 25,000 chars.
  Spoiler-tier WORLD information the in-world public does NOT know —
  engineered plagues, conspiracies, secret factions, the real causes
  of mysteries, hidden histories. If the source describes a "Deep
  History" / "What Is Actually Happening" / "Authorial Truth"
  section or similar, that prose belongs here. If the source
  describes the *resolution* of a public mystery (Call 1 recorded
  the public-facing mystery; this field records what's really going
  on), capture the resolution here.

  Like `lore`, mirror the source's depth on hidden world topics but
  do NOT echo verbatim — extract the substance. Do NOT re-summarize
  the public lore here — only the hidden delta. Do NOT include
  character-specific secrets (those go in Call 4 character
  extraction); only WORLD-level hidden content belongs here.

- **hidden_facts**: 0-50 entries. Each hidden fact is one concrete,
  non-public truth about the current state of the world. Parallel to
  `facts` from Call 1 but for the adjudication layer only. Examples:
  "The Caretaker is a deposed Warden hiding her identity." "The
  Aetheri testimony crystals are real and the Regent has read
  them." Each entry is a single concrete proposition the engine can
  act on. Aim for granular facts (one truth per entry), not
  paragraph-length composites — those go in `hidden_lore`."""


# Legacy combined world prompt — used by v3/v4 single-call paths
# (`extract_world`, `run_import_combined`) that still emit a full
# `WorldExtraction` in one shot.
WORLD_EXTRACTION_INSTRUCTIONS = (
    PUBLIC_WORLD_EXTRACTION_INSTRUCTIONS
    + "\n\n## Hidden world (also part of this call)\n\n"
    + HIDDEN_WORLD_EXTRACTION_INSTRUCTIONS
)


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

- **location**: starting location or perceptual context label. Leave empty if
  the source does not anchor them to a specific starting context.

- **is_playable**: **true** for every character a human could reasonably
  STEP INTO — the protagonist, every contestant on a dating show, every
  party member, every named survivor in a shipwreck ensemble. They run
  as agent NPCs by default; binding a Discord user takes them over via
  `/join`. Mark generously: any character whose role makes sense as a
  player slot. Mark **false** only for characters whose function would
  break if a human controlled them: a pure narrator/quest-giver, a
  background watcher, a god, a one-shot courier the plot uses and
  discards. Multi-protagonist stories therefore have many `is_playable:
  true` entries — that is the expected shape, not the exception.

- **public_sheet**:
  - `role`: their role, title, or occupation.
  - `appearance`: every physical detail the source gives. Height, build,
    hair, eyes, clothing, bearing, signatures. Full.
  - `faction`: faction/house/group affiliation.

- **descriptions**:
  - `public`: 1-2 sentences the player-facing narrator may safely use as a
    brief gloss when this character's known name, uniform, rank, faction, or
    social shorthand appears. Write only what a viewpoint character could
    plausibly know or recognize without exposing secrets. Good shape:
    "Sora is the cohort's favored Hero and informal organizer; his blue
    sun-crest tabard marks Crown Hero livery." Bad shape: "Riku is a
    manipulator cultivated by the Cardinal" unless that is already openly
    known to the viewpoint. Do not include private motives, hidden
    allegiances, concealed species markers, private body details, authorial
    labels, or plot spoilers.
  - `private`: 1-3 sentences of omniscient identity context that may include
    secrets, hidden role, true faction, private stakes, and authorial framing.
    This is audit/future-engine context and is not player-facing.

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


CORE_EXTRACTION_INSTRUCTIONS = """\
Extract the CORE import bundle for this master prompt in a single
structured response: `world` and `characters`. Knowledge envelopes
are produced in a follow-up call that will read your output here as
cached history — focus THIS call on getting world state and
characters extracted as completely and faithfully as the source
warrants.

Because you are producing world + characters together, use the
cross-referencing opportunity: the same faction described in
`world.lore` should land on the right
`characters[].public_sheet.faction`; the same secret in
`world.hidden_facts` should be carried on the plausible
`characters[].private_state.secrets`; characters who would reasonably
know secrets about ANOTHER character (relationships, hidden
connections, rivalries, loyalties) should have those relational
secrets recorded on their own `secrets` list, with the other
character named. Do not drop cross-character linkages — they are
the primary source of in-fiction tension and rediscovery.

(There is no authored-opening extraction in this pipeline — the
opening beat is composed at runtime by the router from the world
and character state you extract here.)

---

## `world`

{world_instructions}

---

## `characters`

{character_instructions}"""


HIDDEN_WORLD_CONTINUATION_INSTRUCTIONS = """\
You just extracted the PUBLIC world above (setting, lore, facts,
physics, narrative rules) — what the in-world public knows.

Now produce the HIDDEN world — the spoiler / conspiracy / plot-secret
content that powers the engine's adjudication layer but never reaches
the player. Use the public world YOU JUST EXTRACTED as ground truth
— do NOT re-extract or summarize it here, only the hidden delta.

{hidden_world_instructions}"""


CHARACTERS_CONTINUATION_INSTRUCTIONS = """\
You just extracted the public + hidden world above.

Now produce `characters` in this structured response. Use the
world state YOU JUST EXTRACTED as ground truth — do not re-derive
it from the master prompt. Keep the same exhaustiveness standard
as the prior calls: every character the source describes, every
faction membership, every secret, every backstory beat.

Cross-referencing opportunity: factions in `world.lore` should
land on the right `characters[].public_sheet.faction`; secrets in
`world.hidden_facts` should be carried on the plausible
`characters[].private_state.secrets`; characters who would
reasonably know secrets about ANOTHER character should record
those relational secrets on their own `secrets` list, with the
other character named.

(There is no authored-opening extraction in this pipeline. Do not
include any second-person opening prose, "you walk into the room"
framing, or scene-setting passages; the opening is composed by the
router at runtime when the player issues `(begin)`.)

---

## `characters`

{character_instructions}"""


PLAYER_PRIMER_EXTRACTION_INSTRUCTIONS = """\
Write a short PLAYER PRIMER: 1–2 paragraphs (≤200 words) that orients
a fresh player who is about to /join the story. Strict constraints:

- **Voice**: second person, present tense ("You wake up…", "You're
  on a dating show called…", "You don't remember how you got here").
- **Framing**: truck-kun style. The player has been deposited into
  this world without warning and has no idea what to expect. Their
  experience is curious, off-balance, slightly absurd. They are NOT
  the omniscient narrator and they are NOT a specific character yet.
- **Content scope**: the world's hook (genre + premise as a teaser),
  the broad social situation (e.g. "you're a contestant on a high-
  stakes reality show"), and ONE or TWO concrete tropes/atmosphere
  cues that signal what kind of fiction this is. Surface texture,
  not lore dump.
- **Hard exclusions** — the primer MUST NOT contain:
  - any character name from the roster
  - any faction, location, or item proper noun from `world.lore` or
    `world.facts`
  - any hidden lore, hidden facts, or spoilers (assume the reader
    will discover those by playing)
  - meta instructions ("type /join", "use the menu", etc.) — the
    bot UI handles that
  - the words "dossier", "briefing", "checkpoint", or "session"
- **Tone**: punchy, mood-setting, a little funny if the source is
  funny, a little ominous if it's ominous. Match the source's vibe.
- **Length**: hard cap 200 words. One paragraph is fine; two is the
  ceiling.

If the source is sparse on premise, lean harder on genre+tone and
the truck-kun "where am I?" framing. If the source is rich, pick
the single most evocative slice; do NOT try to summarize everything."""


PLAYER_PRIMER_CONTINUATION_INSTRUCTIONS = """\
You just extracted `world`, `characters`, and the knowledge envelopes
across the prior calls. Now produce ONE more artifact for the player-
facing UI.

{primer_instructions}

Emit exactly one field, `primer`, containing the prose described
above. No headers, no bullet lists, no JSON-style annotations —
just the primer text the player will read."""


KNOWLEDGE_CONTINUATION_INSTRUCTIONS = """\
You just extracted `world` and `characters` across the prior calls.

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
structured response. You will produce three sub-objects in one JSON
output: `world`, `characters`, and `knowledge`.

Every sub-object has its own fidelity standard — spelled out below —
and each should be populated as if it were its own dedicated extraction
pass. Because you are producing them together, you have the advantage
of cross-referencing: the same faction described in `world.lore`
should appear on the right `characters[].public_sheet.faction`; the
same secret in `world.hidden_facts` should land on the plausible
`characters[].private_state.secrets`; `knowledge.envelopes` should
reflect the world you just extracted, not hallucinated parallel
content.

(There is no authored-opening extraction. The opening beat is
composed by the router at runtime; do not include opening prose
anywhere in this output.)

---

## `world` — World Extraction

{world_instructions}

---

## `characters` — Character Extraction

{character_instructions}

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
        "  Facts: %d, Hidden facts: %d",
        len(data.facts), len(data.hidden_facts),
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
        playable_tag = " [PLAYABLE]" if c.is_playable else ""
        logger.info(
            "    - %s (%s) [%s]%s",
            c.name, c.character_id, c.public_sheet.role or "?", playable_tag,
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
        lines = [f"### {c.character_id}"]
        if c.public_sheet.role:
            lines.append(f"Role: {c.public_sheet.role}")
        if c.public_sheet.faction:
            lines.append(f"Faction: {c.public_sheet.faction}")
        if c.is_playable:
            lines.append("(PLAYABLE SLOT)")
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


# ---------------- Checkpoint assembly ----------------

def build_checkpoint(
    world: WorldExtraction,
    roster: CharacterListExtraction,
    knowledge: CharacterKnowledgeListExtraction,
    story_id: str,
) -> CheckpointFile:
    """Assemble the three extractions into a CheckpointFile. Non-fatal warnings
    are emitted for duplicate character ids."""
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

    physics = PhysicsRuleset(
        strength_limits=world.physics_ruleset.strength_limits,
        magic_enabled=world.physics_ruleset.magic_enabled,
    )

    world_state = WorldState(
        locations=LocationState(),
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

        record = CharacterRecord(
            character_id=cd.character_id,
            name=cd.name,
            status=CharacterStatus(cd.status),
            location=cd.location,
            is_playable=cd.is_playable,
            public_sheet=PublicSheet(
                role=cd.public_sheet.role,
                appearance=cd.public_sheet.appearance,
                faction=cd.public_sheet.faction,
            ),
            descriptions=CharacterDescriptions(
                public=cd.descriptions.public,
                private=cd.descriptions.private,
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
        # Seed the location signal: NPCs need to know where they start
        # the story. Player characters never read
        # `pending_observations`, so skip them. Without this push the
        # very first agent dispatch for a seeded NPC arrives with no
        # location context once the agent prompt's redundant `## Scene`
        # block goes away.
        if not cd.is_playable and cd.location:
            record.pending_observations.append(
                f"[your own action] {cd.name} at {cd.location}."
            )
        characters.append(record)

    playable_chars = [c for c in roster.characters if c.is_playable]
    if playable_chars:
        logger.info(
            "Playable character slot(s): %s",
            ", ".join(f"{c.name} ({c.character_id})" for c in playable_chars),
        )
    else:
        logger.info(
            "No character marked is_playable=true — /join will surface "
            "an empty list and the story will run as fully NPC."
        )

    return CheckpointFile(
        session=session,
        importer_version=IMPORTER_VERSION,
        world_state=world_state,
        characters=characters,
        config=config,
    )


# ---------------- Top-level pipeline ----------------

PRESERVATION_ANALYSIS_SYSTEM = """\
<role>
You are auditing the fidelity of an interactive-fiction story import.
</role>

<instructions>
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

Respond with ONLY valid JSON matching the schema.
</instructions>

<output_schema>
{
  "coverage_rating": "high | medium | low",
  "dropped_topics": ["string"],
  "compressed_topics": ["string"],
  "preservation_notes": "string"
}
</output_schema>"""


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

    if ckpt.characters:
        char_lines = ["# Characters"]
        for c in ckpt.characters:
            char_lines.append(f"## {c.character_id}")
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
                "<source_prompt>\n"
                f"{source_text}\n"
                "</source_prompt>\n\n"
                "<extracted_checkpoint>\n"
                f"{output_text}\n"
                "</extracted_checkpoint>\n\n"
                "<audit_request>\n"
                "Compare the extracted checkpoint against the source prompt. "
                "Report coverage, dropped topics, compressed topics, and "
                "preservation notes per the system instructions.\n"
                "</audit_request>"
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
    """Run the legacy three-stage import pipeline and return the
    assembled checkpoint.

    Pipeline shape:
      Stage 1: world extraction — serial. Primes the shared cache.
      Stage 2: characters — runs after stage 1. Reads the source
        prompt from cache.
      Stage 3: knowledge envelopes — runs after stage 2 completes.
        Depends on the extracted world + roster.

    (v9 dropped a parallel `extract_opening` stage that used to run
    alongside stage 2 — the opening beat is now composed at runtime
    by the router instead of being authored at import time.)

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

    # Stage 2: characters. Reads the primed cache.
    roster = await extract_characters(client, source_text)

    # Stage 3: depends on world + roster. Sees omniscient world and filters
    # per-character knowledge envelopes.
    knowledge = await extract_character_knowledge(
        client, source_text, world, roster,
    )

    checkpoint = build_checkpoint(world, roster, knowledge, story_id)
    logger.info(
        "Import complete in %.1fs (%d characters)",
        time.monotonic() - t_start,
        len(checkpoint.characters),
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
    """Assemble the single user message containing all three sets of
    extraction instructions. Reuses the existing per-stage instruction
    blocks so there's one source of truth for extraction guidance."""
    body = COMBINED_EXTRACTION_INSTRUCTIONS.format(
        world_instructions=WORLD_EXTRACTION_INSTRUCTIONS,
        character_instructions=CHARACTER_EXTRACTION_INSTRUCTIONS,
        knowledge_instructions=KNOWLEDGE_EXTRACTION_INSTRUCTIONS,
    )
    return f"<stage_instructions>\n{body}\n</stage_instructions>"


async def run_import_combined(
    client: LLMClient,
    source_text: str,
    story_id: str,
) -> _CombinedImportResult:
    """Single-call importer. Produces world + characters + knowledge
    in one structured-output response and returns the checkpoint
    along with the exact messages used so the preservation analysis
    pass can replay them as cached history.

    (v9 dropped the `opening` member from the combined extraction —
    the opening beat is composed at runtime by the router, not
    authored at import time.)

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
        "  Facts: %d, Hidden facts: %d",
        len(bundle.world.facts),
        len(bundle.world.hidden_facts),
    )
    logger.info("  Characters: %d extracted", len(bundle.characters.characters))
    logger.info("  Envelopes: %d produced", len(bundle.knowledge.envelopes))

    checkpoint = build_checkpoint(
        bundle.world,
        bundle.characters,
        bundle.knowledge,
        story_id,
    )
    logger.info(
        "Combined import complete in %.1fs (%d characters)",
        time.monotonic() - t_start,
        len(checkpoint.characters),
    )

    return _CombinedImportResult(
        checkpoint=checkpoint,
        priming_messages=priming_messages,
        assistant_text=response.content or "",
    )


# ---------------- Five-call path (v7) ----------------


def _public_world_user_prompt() -> str:
    """Assemble the Call-1 user message: PUBLIC world (setting, lore,
    facts, physics, narrative_rules). Hidden world and locations are
    extracted in Calls 2 and 3 to keep each output under Sonnet 4.6's
    64K cap."""
    return (
        "<stage_instructions name=\"public_world\">\n"
        f"{PUBLIC_WORLD_EXTRACTION_INSTRUCTIONS}\n"
        "</stage_instructions>"
    )


def _hidden_world_user_prompt() -> str:
    """Assemble the Call-2 user message: HIDDEN world (hidden_lore,
    hidden_facts) as a continuation that reads the public world from
    Call 1 as cached history."""
    body = HIDDEN_WORLD_CONTINUATION_INSTRUCTIONS.format(
        hidden_world_instructions=HIDDEN_WORLD_EXTRACTION_INSTRUCTIONS,
    )
    return f"<stage_instructions name=\"hidden_world\">\n{body}\n</stage_instructions>"


def _characters_user_prompt() -> str:
    """Assemble the Call-4 user message: characters as a continuation
    that reads the prior chain (public + hidden + locations) as
    cached history.

    (v9 renamed from `_chars_and_opening_user_prompt`; the paired
    opening-prose extraction is gone — see IMPORTER_VERSION v8→v9
    note for rationale.)"""
    body = CHARACTERS_CONTINUATION_INSTRUCTIONS.format(
        character_instructions=CHARACTER_EXTRACTION_INSTRUCTIONS,
    )
    return f"<stage_instructions name=\"characters\">\n{body}\n</stage_instructions>"


def _knowledge_user_prompt() -> str:
    """Assemble the Call-5 user message: knowledge envelopes as a
    continuation. Reuses the per-stage knowledge instructions plus the
    cross-character relational emphasis that v3's single-call pass
    tended to drop."""
    body = KNOWLEDGE_CONTINUATION_INSTRUCTIONS.format(
        knowledge_instructions=KNOWLEDGE_EXTRACTION_INSTRUCTIONS,
    )
    return f"<stage_instructions name=\"knowledge\">\n{body}\n</stage_instructions>"


def _player_primer_user_prompt() -> str:
    """Assemble the Call-6 user message: short player-facing primer as
    a continuation that reads the full prior chain (world + characters +
    knowledge) as cached history. The primer is what a fresh player sees
    on /story start before they pick a character — it replaces the
    omniscient briefing that used to leak roster names, factions, and
    lore the player hadn't earned yet."""
    body = PLAYER_PRIMER_CONTINUATION_INSTRUCTIONS.format(
        primer_instructions=PLAYER_PRIMER_EXTRACTION_INSTRUCTIONS,
    )
    return f"<stage_instructions name=\"player_primer\">\n{body}\n</stage_instructions>"


async def run_import_two_call(
    client: LLMClient,
    source_text: str,
    story_id: str,
) -> _CombinedImportResult:
    """Six-call importer (v10; function name retained for caller
    compatibility — bridge + CLI + tests still import this symbol).
    Call 1 extracts the PUBLIC world; Call 2 extracts the HIDDEN
    world as a continuation that reads Call 1 as cached history;
    Call 3 extracts locations reading the prior chain; Call 4
    extracts `characters` with public/private descriptions
    (v9 dropped the paired `opening` extraction); Call 5 extracts
    `knowledge` envelopes; Call 6
    extracts the player-facing `player_primer` — a 1-2 paragraph
    spoiler-free framing the bot shows on briefing before the
    player ever runs `/join`.

    Why this many calls: v6 split the combined world into skeleton
    (public + hidden) and locations, but a 95KB master prompt with
    deep conspiracy lore (multi-faction, hidden-history section)
    STILL pushed the skeleton call past Sonnet 4.6's 64K output cap
    — truncating mid-string around column 265K of JSON. Splitting
    public from hidden gives each its own budget, and also matches
    the engine's adjudication-vs-public contract structurally
    (hidden content is for the omniscient adjudication layer;
    public content is what player-facing renders may draw on). v8
    added Call 6 (player primer) — it shares the cached prefix from
    earlier calls, so it's effectively paid for once. v9 dropped
    the authored opening from Call 4, leaving the call with
    characters alone (the opening beat is now composed at runtime
    by the router; see IMPORTER_VERSION v8→v9 note for rationale).

    The merged `WorldExtraction` shape is assembled in Python from
    the three world responses (public + hidden + locations), so
    `build_checkpoint` is unchanged.

    Returns a `_CombinedImportResult` shaped the same as prior
    versions so downstream callers (EngineBridge, preservation
    continuation) don't care which pipeline produced the checkpoint.
    `priming_messages` contains the conversation up through Call-5's
    user turn (the primer is intentionally omitted from the priming
    chain so the preservation continuation reads only the structural
    extractions); `assistant_text` is the Call-5 assistant reply —
    the preservation analysis tacks itself on as a downstream call
    with the same continuation helper.
    """
    t_start = time.monotonic()
    logger.info(
        "Starting six-call import (pipeline %s): source prompt %d chars, ~%d words",
        IMPORTER_VERSION, len(source_text), len(source_text.split()),
    )
    # v9+: Call 4 produces characters alone (no paired `opening`).
    # The opening beat is composed at runtime by the router from
    # the world + character state extracted across Calls 1-4.

    system_text = SHARED_SOURCE_SYSTEM.format(source_prompt=source_text)
    public_user = _public_world_user_prompt()
    public_messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": public_user},
    ]

    # Call 1 — PUBLIC world (setting, lore, facts, physics, narrative_rules).
    t_pub = time.monotonic()
    public_response = await client.complete(
        role="narrator",
        messages=public_messages,
        response_model=PublicWorldExtraction,
        temperature=0.3,
        max_tokens=MAX_EXTRACTION_TOKENS,
        cache_user_tail=True,
    )
    _log_usage("public_world", public_response)
    public_world: PublicWorldExtraction = public_response.parsed
    logger.info(
        "  Public world (%.1fs): setting=%s/%s, facts=%d, "
        "lore=%d chars, narrative_rules=%d chars",
        time.monotonic() - t_pub,
        public_world.setting.genre or "?", public_world.setting.tone or "?",
        len(public_world.facts),
        len(public_world.lore), len(public_world.narrative_rules),
    )

    # Call 2 — HIDDEN world (hidden_lore, hidden_facts) as a
    # continuation that reads Call 1 as cached history.
    hidden_user = _hidden_world_user_prompt()
    hidden_messages = public_messages + [
        {"role": "assistant", "content": public_response.content or ""},
        {"role": "user", "content": hidden_user},
    ]
    t_hid = time.monotonic()
    hidden_response = await client.complete(
        role="narrator",
        messages=hidden_messages,
        response_model=HiddenWorldExtraction,
        temperature=0.3,
        max_tokens=MAX_EXTRACTION_TOKENS,
        cache_user_tail=True,
    )
    _log_usage("hidden_world", hidden_response)
    hidden_world: HiddenWorldExtraction = hidden_response.parsed
    logger.info(
        "  Hidden world (%.1fs): hidden_facts=%d, hidden_lore=%d chars",
        time.monotonic() - t_hid,
        len(hidden_world.hidden_facts), len(hidden_world.hidden_lore),
    )

    # Merge public + hidden into the unified WorldExtraction
    # shape downstream `build_checkpoint` already understands. The LLM
    # never emits a WorldExtraction directly under v7; this is the
    # join.
    world = WorldExtraction(
        setting=public_world.setting,
        lore=public_world.lore,
        facts=public_world.facts,
        physics_ruleset=public_world.physics_ruleset,
        narrative_rules=public_world.narrative_rules,
        hidden_lore=hidden_world.hidden_lore,
        hidden_facts=hidden_world.hidden_facts,
    )

    # Call 3 — characters (continuation; reads the public + hidden world
    # as cached history). Pre-v9 this
    # call also extracted the authored opening beat; now characters
    # is the only payload — the runtime router composes the opening
    # from the world + character state.
    chars_user = _characters_user_prompt()
    chars_messages = hidden_messages + [
        {"role": "assistant", "content": hidden_response.content or ""},
        {"role": "user", "content": chars_user},
    ]
    t_chars = time.monotonic()
    chars_response = await client.complete(
        role="narrator",
        messages=chars_messages,
        response_model=CharacterListExtraction,
        temperature=0.3,
        max_tokens=MAX_EXTRACTION_TOKENS,
        cache_user_tail=True,
    )
    _log_usage("characters", chars_response)
    characters: CharacterListExtraction = chars_response.parsed
    logger.info(
        "  Characters (%.1fs): %d extracted",
        time.monotonic() - t_chars,
        len(characters.characters),
    )

    # Call 5 — knowledge envelopes (continuation; reads the full
    # four-call upstream chain as cached history).
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

    # Call 6 — player primer (continuation; reads the full five-call
    # chain as cached history, paying fresh only for the primer
    # instructions and the short response). Stamped on the checkpoint;
    # NOT folded into priming_messages so preservation analysis still
    # branches off the Call-5 prefix and ignores the primer turn.
    primer_user = _player_primer_user_prompt()
    primer_messages = knowledge_messages + [
        {"role": "assistant", "content": knowledge_response.content or ""},
        {"role": "user", "content": primer_user},
    ]
    t_primer = time.monotonic()
    primer_response = await client.complete(
        role="narrator",
        messages=primer_messages,
        response_model=PlayerPrimerExtraction,
        temperature=0.6,
        max_tokens=2_000,
    )
    _log_usage("player_primer", primer_response)
    primer: PlayerPrimerExtraction = primer_response.parsed
    primer_text = (primer.primer or "").strip()
    logger.info(
        "  Player primer (%.1fs): %d chars",
        time.monotonic() - t_primer, len(primer_text),
    )

    checkpoint = build_checkpoint(
        world,
        characters,
        knowledge,
        story_id,
    )
    checkpoint.player_primer = primer_text
    sync_checkpoint_runtime_models(checkpoint, client.config)
    logger.info(
        "Import complete in %.1fs (%d characters, primer=%d chars)",
        time.monotonic() - t_start,
        len(checkpoint.characters),
        len(primer_text),
    )

    # Pack the five-call conversation (NOT the primer turn) into
    # priming_messages so the preservation analysis continuation reads
    # the same cached prefix it used pre-v8 — branching off Call 5
    # rather than chaining through the primer (which is stylistic and
    # would only confuse the audit pass).
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
        "<audit_request>\n"
        "Now audit what you just extracted against the source master prompt. "
        "You are grading your own extraction for fidelity. Rate coverage "
        "honestly — every dropped topic and every compression is a defect. "
        "Compare the checkpoint fields below back against the master prompt "
        "you were given at the start of this conversation.\n"
        "</audit_request>\n\n"
        f"{PRESERVATION_ANALYSIS_SYSTEM}\n\n"
        "<extracted_checkpoint>\n"
        f"{output_text}\n"
        "</extracted_checkpoint>"
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
