# D&D Module Import And Basic D&D Runtime

This document captures the planned path for running D&D-style content in Ayoa.
It separates two related but different projects:

1. basic D&D functionality: dice, arithmetic, mechanical character state, and
   rules-aware Cat II resolution
2. adventure module import: compiling a large owned module into a runtime
   adventure substrate with forward-looking dramatic structure

The first implementation target is basic D&D functionality. Full module import
depends on that runtime layer and should not be built first.

## Core Position

Ayoa should not import a whole D&D module into a prompt. That is too expensive,
too brittle, and too weak dramatically. A published adventure is not just local
room text; it is an arc with future reveals, dependencies, villains, motifs,
and consequences. A flat RAG/snippet layer can answer "what is in this room?"
but cannot reliably answer "what should this moment preserve or foreshadow?"

The eventual module importer should therefore be an adventure compiler, not a
prompt loader. It should transform owned source material into structured local
data and runtime indexes. During play, the router receives compact adventure
guidance for the current turn; character agents receive only their own
knowledge/observations; the narrator continues to render visible facts only.

## Basic D&D First

The immediate goal is not "run Phandelver." The immediate goal is "run Ayoa
with D&D mechanics available when the fiction needs them."

The useful first slice:

* A session can opt into a D&D rules mode.
* Characters can carry optional D&D mechanical state.
* Code owns dice rolling, modifiers, arithmetic, hit point/resource deltas, and
  simple condition bookkeeping.
* The LLM can still reason about D&D rules and fiction, but it does not invent
  roll totals or do arithmetic.
* Cat II final resolution can use a router-owned D&D subflow when D&D mode is
  enabled.
* That subflow returns structured adjudication that compiles back into the
  existing router/narrator pipeline without adding another model role.

This keeps the current Ayoa architecture intact:

* The router still opens Cat II, chooses required responders, determines
  observers, controls beat pacing, and owns final D&D adjudication calls.
* The narrator still sees only canonical observable facts.
* NPC agents still see only their own rolling history, known context, and
  pending observations.

## Basic D&D Scope

Initial D&D support should be deliberately narrow.

In scope:

* ability scores and modifiers
* proficiency bonus
* armor class
* hit points and temporary hit points
* saving throws
* skill checks
* attack rolls
* damage rolls
* simple contested checks
* simple DC checks
* common conditions as tags
* initiative later, but not required for first Cat II integration
* spell/resource slots as generic resource counters, not full spell automation

Out of scope for the first slice:

* full tactical combat engine
* grid movement
* full action economy enforcement
* complete spell text automation
* class feature automation
* complete monster manuals
* published adventure import
* protected D&D content checked into the repo

The first system should be good at adjudicating common contested actions:

* "I shove him away from the door."
* "I try to grapple the goblin."
* "I swing at the bandit."
* "I dive behind the altar before the cultist's spell lands."
* "I try to sneak past the guard."
* "I wrestle the idol out of her hands."

## Mechanical State

Add optional rules-facing state without making every Ayoa story a D&D story.
For imported player sheets, the full D&D character snapshot should follow
`DND_CHARACTER_IMPORT.md` and live under `mechanics.dnd5e_sheet`; the fields
below are the compact compatibility projection used by the current D&D Cat II
path.

Recommended shape on `CharacterRecord`:

```json
{
  "mechanics": {
    "ruleset_id": "dnd5e_basic",
    "stat_block_ref": "",
    "level": 1,
    "proficiency_bonus": 2,
    "ability_scores": {
      "str": 10,
      "dex": 10,
      "con": 10,
      "int": 10,
      "wis": 10,
      "cha": 10
    },
    "saving_throw_proficiencies": [],
    "skill_proficiencies": [],
    "armor_class": 10,
    "hit_points": {
      "current": 10,
      "max": 10,
      "temporary": 0
    },
    "conditions": [],
    "resources": {},
    "raw": {}
  }
}
```

This field should be optional and ignored by non-D&D sessions. The `raw` object
exists as an escape hatch for imported/homebrew data, but core code should use
typed fields when it needs to calculate.

## Dice And Arithmetic

Dice and arithmetic should be code-owned. The LLM may decide that a contested
Athletics check is appropriate; the engine rolls and calculates it.

Core helpers:

* parse/evaluate dice expressions such as `1d20+3`, `2d6+1`, advantage, and
  disadvantage
* compute ability modifiers
* compute skill and saving throw bonuses
* execute roll requests
* return an auditable roll ledger

The roll ledger is persisted in checkpoint roll transactions, not in normal
router/narrator conversation history. Future prompt context receives the
canonical outcome facts, while rewind/debug can inspect the full dice audit.

Example ledger:

```json
{
  "rolls": [
    {
      "roll_id": "roll_01",
      "actor_id": "sildar",
      "kind": "skill_check",
      "ability": "str",
      "skill": "athletics",
      "dice": "1d20",
      "d20": [14],
      "modifier": 4,
      "total": 18,
      "dc": 15
    }
  ]
}
```

## Router-Owned D&D Cat II Flow

Use a two-step adjudication flow for D&D Cat II resolution.

### Step 1: Plan Rolls

The event-router role receives a contested action packet in a dedicated D&D
Cat II prompt:

* initiator id and intention
* responder ids and intentions
* recent canonical facts from the open Cat II event
* relevant locations/positioning as known to the engine
* character mechanical snapshots
* available environmental constraints

It returns a `RollPlan`, not the final outcome:

```json
{
  "needs_rolls": true,
  "roll_requests": [
    {
      "roll_id": "roll_01",
      "actor_id": "gundren",
      "kind": "skill_check",
      "ability": "str",
      "skill": "athletics",
      "dc": null,
      "opposed_by": "roll_02",
      "advantage_state": "normal",
      "reason": "Gundren is trying to shove the guard away from the door."
    },
    {
      "roll_id": "roll_02",
      "actor_id": "guard",
      "kind": "skill_check",
      "ability": "str",
      "skill": "athletics",
      "dc": null,
      "opposed_by": "roll_01",
      "advantage_state": "normal",
      "reason": "The guard braces and contests the shove."
    }
  ],
  "no_roll_reason": ""
}
```

If no roll is needed, the router can say so and explain why. For example,
an impossible action, a freely accepted action, or a purely fictional
non-mechanical resolution can skip dice.

### Step 2: Execute Rolls

The engine resolves every roll request:

* calculates modifiers from character mechanics
* applies proficiency when relevant
* handles advantage/disadvantage
* rolls NPC/agent dice automatically
* rolls player dice automatically by default, or pauses for Discord roll UI
  when `player_roll_mode="interactive"`
* returns totals and roll details

The LLM never invents roll totals.

### Step 3: Finalize Outcome

The event-router role receives the original packet plus the transient roll
ledger and returns
`RulesAdjudication`:

```json
{
  "feasible": true,
  "mechanical_summary": "Gundren wins the opposed Athletics contest.",
  "visible_outcome_facts": [
    "Gundren drives his shoulder into the guard and forces him back from the door.",
    "The guard stumbles two steps and loses the doorway for a breath."
  ],
  "state_deltas": [
    {
      "kind": "condition_add",
      "target_id": "guard",
      "condition": "off_balance",
      "duration": "brief"
    }
  ],
  "roll_ledger_refs": ["roll_01", "roll_02"],
  "rules_notes": [
    "Resolved as an opposed Strength (Athletics) contest."
  ],
  "fallback_reason": ""
}
```

The engine compiles this back into an `EventRouterOutput`-compatible Cat II
resolution:

* `requires_responders=false`
* `required_responders=[]`
* `agent_responder_picks=[]`
* `ends_beat=true`
* `ends_beat_reason="cat_ii_resolution"`
* `canonical_event.observable_facts` from `visible_outcome_facts`
* `observers` inherited from the Cat II context or conservatively recalculated

After finalization, the roll plan and ledger stay in
`session.cat_ii_roll_transactions` for checkpoint rewind and audit. They are not
appended to `session_conversation`; only a compact canonical result note is
queued for the next router call.

## Basic D&D Implementation Plan

1. Add optional mechanics schemas.

   Add a rules-neutral `mechanics` field to `CharacterRecord`, with a concrete
   `dnd5e_basic` shape under it. Keep old checkpoints loadable by making the
   field optional/defaulted.

2. Add dice and D&D arithmetic helpers.

   Implement ability modifiers, proficiency-aware bonuses, simple roll
   execution, advantage/disadvantage, and roll ledgers. Dice parsing and
   evaluation should go through Ayoa's internal wrapper around `d20` so future
   rules code receives stable Ayoa-owned roll request/result objects instead
   of depending directly on the third-party package API.

3. Add session settings.

   Add settings for:

   * `ruleset_id`
   * `player_roll_mode`

   Defaults preserve current behavior:

   * `ruleset_id="narrative"`
   * `player_roll_mode="auto"`

4. Preserve richer Cat II context.

   `OpenCatIIEvent` currently stores intentions and responders. D&D
   arbitration needs the opening event's relevant observable facts,
   participating observers, and a compact mechanics snapshot. Add this without
   changing Cat II open behavior.

5. Add D&D Cat II schemas and prompt.

   Add structured models for `ContestedActionPacket`, `RollPlan`,
   `RollRequest`, `RollLedger`, and `RulesAdjudication`. Add a dedicated prompt
   used with the `event_router` role; it owns only the mechanics-heavy final
   resolution, not fresh Cat II classification.

6. Wire D&D mode into Cat II final resolution.

   In Cat II final resolution only, if `ruleset_id == "dnd5e_basic"`, call the
   roll-planning path, execute rolls in code, call finalization, and compile the
   result back into the existing event shape. Otherwise keep the current router
   path.

   Current implementation status: this branch exists behind
   `ruleset_id == "dnd5e_basic"`. It preserves the Cat II opening context, asks
   the event-router role for a d20 roll plan, executes planned rolls through
   Ayoa's `d20` wrapper, asks for final adjudication, and compiles the outcome
   back into `EventRouterOutput`. The earlier separate
   `cat_ii_resolution_mode` switch has been removed; D&D arbitration is part of
   the selected ruleset.

7. Apply simple state deltas.

   Start with hit point deltas, condition tags, and resource counters. Avoid
   broad inventory or spell automation until the adjudication path is stable.

8. Add focused tests.

   Tests should verify:

   * non-D&D sessions are unchanged
   * D&D mode uses the event-router role for Cat II final resolution only
   * dice totals are code-generated
   * modifiers come from mechanics state
   * roll ledgers are auditable
   * roll planning/ledger details do not append to router message history
   * visible facts compile into the narrator path
   * private/mechanical notes do not leak to narrator prompts

## Module Import Later

Once basic D&D runtime exists, module import can target a real substrate.

The module importer is a compiler, not a runtime RAG layer over OCR. It turns a
private owned source corpus into reviewed compiled records. During play, the
runtime reads only compiled refs, compact text cards, graph data, structured
tables/stat blocks, reveal metadata, and player-safe asset refs.

The import surface has three storage layers:

1. **Raw private corpus**

   The original PDF, rendered page images, extracted image streams, OCR text,
   layout JSON, thumbnails, crop masks, and review snapshots live outside git
   by default. These are protected source derivatives and should be treated as
   private table material.

   Files hold blobs. A raw catalog SQLite database indexes them by hash, source
   asset, page candidate, extraction method, dimensions, bbox, confidence, and
   review status. JSONL may be emitted as review/export interchange, but it is
   not the source of truth because the compiler needs joins across pages,
   spans, images, sections, provenance, and review issues.

2. **Compiled immutable pack**

   The runtime pack is a private SQLite database, with FTS5 and graph indexes,
   plus adjacent media files addressed by content hash. It contains reviewed
   `SceneCard`, `MapAsset`, `Handout`, `Table`, `StatBlock`, `Reveal`,
   `FrontDossier`, alias, provenance, and review records.

   The compiled pack says what the published module starts with. It is not
   mutated during play. A pack has a stable `pack_id`, semantic version,
   source fingerprint, importer version, schema versions, content hash, and ref
   migration table.

3. **Mutable play overlay**

   Checkpoints store `pack_id`, `pack_version`, `pack_content_hash`, introduced
   refs, revealed asset refs, map fog/reveal masks, visited locations,
   discovered clues, consumed table results, spawned module refs, depleted
   encounters, looted treasure, opened doors, sprung traps, killed or moved
   NPCs, front knowledge, clocks, active plans, and local overrides.

   Checkpoints should not store raw images, page scans, full OCR, large
   protected excerpts, or complete source cards by default. They store refs,
   hashes, reveal state, and campaign mutations. The module says what starts
   true; the checkpoint says what is true now.

Runtime context should be composed from:

```text
compiled module pack
+ campaign state overlay
+ current actor/intention
+ local scene position
+ unresolved reveals and safe foreshadowing opportunities
= compact router context packet
```

The router should receive enough forward-looking module structure to preserve
payoffs and foreshadow safely. It should not receive the whole module.

## Adventure Compiler Artifacts

The compiler should produce explicit raw-catalog records and compiled runtime
records. These names are schema concepts, not necessarily Pydantic class names
for the first implementation.

### Raw Catalog Records

`SourceAsset` is any original or derived private blob:

* `asset_id`
* `kind`: `source_pdf`, `rendered_page`, `extracted_image`, `ocr_text`,
  `layout_json`, `thumbnail`, `crop_mask`, `review_export`
* `path`
* `sha256`
* `bytes`
* `mime_type`
* `source_asset_id`
* `extraction_method`
* `importer_version`

`Page` is a reconstructed logical page candidate. Do not assume that image
stream order equals page order; scanned pages, generic text layers, decorative
images, and maps can appear as different asset streams.

* `page_id`
* `source_asset_id`
* `pdf_page_index`
* `printed_page_label`
* `rendered_page_asset_id`
* `image_asset_ids`
* `text_span_ids`
* `section_id`
* `alignment_status`
* `confidence`
* `review_status`

`TextSpan` preserves both raw OCR/generic text and normalized text:

* `span_id`
* `page_id`
* `raw_text`
* `normalized_text`
* `bbox`
* `reading_order`
* `kind`: heading, paragraph, boxed text, sidebar, table cell, stat line,
  caption, map label, footer, page number, unknown
* `confidence`
* `method`

`ImageAsset` describes image streams, rendered pages, crops, masks, and
player-safe derivatives:

* `image_id`
* `asset_id`
* `page_id`
* `stream_index`
* `dimensions`
* `bbox`
* `role`: source page, DM map, player-safe map, handout, portrait, item art,
  scene art, table page, stat block page, decoration, unknown
* `spoiler_level`
* `safe_for_players`
* `safe_for_llm`
* `derived_from_image_id`
* `confidence`
* `review_status`

`Section` reconstructs the source outline:

* `section_id`
* `parent_id`
* `title`
* `path`
* `page_range`
* `span_refs`
* `aliases`
* `confidence`

`Provenance` is a reusable reference from any compiled field back to source:

* `source_asset_id`
* `page_id`
* `span_id`
* `image_id`
* `bbox`
* `section_id`
* `method`
* `confidence`
* `importer_version`
* `human_review_status`

`ReviewIssue` records blocking and non-blocking import problems:

* `issue_id`
* `entity_ref`
* `severity`
* `kind`
* `message`
* `blocking`
* `status`
* `reviewer_notes`

### Campaign Bible

A compact omniscient summary of the module's premise, escalation path, major
villains, factions, mysteries, themes, and tonal motifs. This is stable context
for the router/adventure resolver, not for character agents or the narrator.

### Adventure Graph

Nodes and edges for chapters, regions, locations, rooms, scenes, encounters,
quests, and transitions. This lets the runtime know what is adjacent, what has
been skipped, and what later content a current choice can affect.

Edges must distinguish normal routes, hidden routes, locked routes, one-way
transitions, stairs, vertical connections, portals, blocked paths, and
perception-only relationships. Map-derived edges are not trusted until reviewed.

### Reveal Graph

Structured hidden facts and their reveal triggers:

* what is secret
* who knows it
* where it can be discovered
* what clues point to it
* what later scenes depend on it
* what would spoil it too early

### Foreshadowing Bank

Safe forward-looking signals:

* surface detail
* future payoff
* allowed timing
* allowed observers
* maximum directness
* spoiler boundary

Example:

```json
{
  "surface": "A faint violet shimmer in damaged stone.",
  "payoff_ref": "later_obelisk_revelation",
  "allowed_now": true,
  "max_directness": "ambiguous sensory detail",
  "forbidden": "Do not name the obelisk or explain the psychic cause."
}
```

### Scene Cards

The local precision layer:

* visible description
* hidden features
* exits and adjacent nodes
* occupants
* hazards
* treasure
* clues
* encounter setup
* source refs
* image refs
* review status

Scene cards are the module equivalent of "what is in this room?" They are
necessary but not sufficient without the campaign bible and reveal graph.

### Map Assets

`MapAsset` records are private map images plus reviewed topology:

* source image refs and player-safe derivatives
* keyed labels and linked location refs
* graph nodes and edges
* secret edges and hidden areas
* vertical levels, stairs, portals, blocked routes, and one-way paths
* fog/reveal regions
* scale, orientation, and line-of-sight notes when known
* source refs and review status

The router should consume sanitized topology and reveal-state refs, not raw DM
map images.

For the first module-import slice, map topology is a **manual import task**.
Vision or OCR may identify labels and suggest regions, but a human/import agent
must explicitly author the tactical projection. The required manual output is a
`TacticalMapTemplate` linked back to the `MapAsset`:

* `template_id`
* `derived_from_map_asset_id`
* `target_schema`: usually `DndBattleMapState` for D&D combat maps
* `map_kind`: overland, floorplan, dungeon, tower stack, battle encounter,
  abstract scene map, or reference-only
* `scale_assumption` and `square_size_ft`
* `grid`: width, height, orientation, origin notes, and confidence
* `spawn_anchors`: named entry points, encounter starts, fallback player starts,
  fallback enemy starts, reinforcements, exits, and retreat routes
* `terrain`: walls, doors, windows, cliffs, pits, stairs, balconies, furniture,
  water, difficult ground, cover, blocked movement, and line-of-sight blockers
* `secrets`: secret doors, hidden routes, traps, concealed areas, one-way links,
  vertical drops, teleport/portal links, and reveal conditions
* `area_links`: mapping from keyed labels to scene/location refs
* `review_status`: draft, topology-reviewed, player-safe-reviewed,
  blocked-needs-review
* `review_notes`: uncertainties, assumptions, and source refs

The manual importer should be conservative. If exact walls or secret routes are
uncertain, mark them as `review_status="draft"` and explain the uncertainty
rather than laundering guesses into reviewed topology. Derivative player images
can be produced later; the first goal is a machine-readable tactical projection
that can seed combat state or provide router-safe navigation context.

For D&D combat, the template compiles into `DndBattleMapState`:

```text
TacticalMapTemplate
  -> DndBattleMapState
     present
     map_name
     width / height / square_size_ft
     tokens from bound combatants + spawn anchors
     terrain zones for blockers, cover, hazards, and movement constraints
     areas for active spell/effect templates
```

The live `DndBattleMapState` remains mutable combat state. The imported
template is immutable pack data. Runtime code binds actual combatants to spawn
anchors when combat begins, then spatial deltas move tokens and add/remove
active areas during initiative.

### Handouts

`Handout` records represent letters, cards, inscriptions, notes, journals,
symbols, documents, and other player-facing artifacts:

* title and aliases
* source image refs
* player-safe image/text variants
* language, legibility, damage, lighting, and possession requirements
* partial-reveal rules
* source refs and review status

### Tables

`Table` records preserve structure:

* title and usage context
* columns and rows
* dice ranges, weights, and nested-table refs
* result refs
* malformed flags
* source refs and review status

Malformed tables must not drive runtime randomization.

### Stat Blocks

`StatBlock` records hold typed rules data:

* name and aliases
* ruleset id
* AC, HP, speed, abilities, saves, skills, senses, languages, CR/XP
* traits, actions, reactions, legendary actions, lair actions, spellcasting
* damage resistances, immunities, vulnerabilities, and conditions
* parse warnings
* source refs and review status

Missing combat-critical fields block combat automation.

### Front Dossiers

`FrontDossier` records describe off-stage pressure without making it runtime
truth by themselves:

* goals, constraints, resources, minions, and domains of control
* initial knowledge
* learning rules and plausible information channels
* response palette
* cooldowns and restraint rules
* source refs and review status

The mutable checkpoint records what the front currently knows and what plans are
active.

## Image Use In Play

Adventure images are private table assets, not generic model context. The
default rule is:

* display images to humans only after a router-owned reveal
* send only small, curated, reveal-scoped metadata to LLMs
* store refs and reveal state in checkpoints, never raw image bytes

### Image Categories

`DM/source images` are visible to the host only. Full source pages, DM maps,
table scans, stat block scans, and review screenshots are never player-facing by
default and should not be sent to LLMs.

`Player-safe maps` are reviewed derivatives with spoilers removed or masked.
They can be revealed by region/fog state. Players receive the image; the router
receives sanitized topology; the narrator receives only visible facts or a safe
caption.

`Cropped room maps` are small map regions revealed only to characters who can
perceive the room or possess an in-fiction map. Crops must exclude adjacent
spoilers, secret doors, traps, and labels unless those are already revealed.

`Handouts` are letters, cards, inscriptions, notes, journals, and documents.
They reveal only when found, received, read, shown, remembered, or otherwise
made perceptible in fiction. OCR text is visible only when the character can
read the relevant portion.

`Portraits and item art` reveal on clear sight, identification, wanted poster,
memory, rumor, acquisition, or inspection. Captions and alt text must not leak
proper names, monster identity, allegiance, magical properties, or future role
unless those facts are known.

`Mood plates and decorative art` are optional atmosphere. They require spoiler
review because chapter titles, symbols, depicted NPCs, landmarks, or future
locations can reveal more than intended.

`Table/stat pages` remain host/reference assets. Runtime play uses parsed table
and stat-block records rather than page images.

### Reveal States

Each image asset should carry a reveal state:

* `private_source`: extracted but not prepared for play
* `dm_only`: usable by the human host, never automatic
* `router_reference`: usable through private sanitized metadata
* `prepared_player_safe`: reviewed derivative, not yet revealed
* `revealed_to_character_ids`: visible to exact character ids
* `revealed_to_user_ids`: visible to exact users when user binding matters
* `revealed_to_channel`: visible to the shared table surface
* `expired_or_superseded`: hidden from active play unless history requests it

Metadata is itself content. Asset ids, filenames, captions, alt text, OCR,
embed titles, and local paths must obey the same spoiler boundary as the image.

### Runtime Asset Contract

Content lookup may return text refs and image refs:

```text
ContentLookupResult
  text_refs: ContentRef[]
  image_refs: ContentImageRef[]
```

`ContentImageRef` should be rules-neutral:

```text
pack_id
asset_id
kind
title
mime_type
width
height
sha256
source_ref
review_status
spoiler_class
player_safe_alt_text
```

The router may reveal assets as a sibling payload to canonical facts:

```text
AssetReveal
  asset_ref
  audience: all_observers | only
  visible_to: character ids
  presentation: inline | attachment | reference | map_overlay
  caption
```

`AssetReveal` uses the same visibility semantics as `observable_facts`. The
narrator receives visible facts and safe captions only; it never receives
hidden image refs, raw source paths, DM notes, map labels, or unrevealed
metadata. Character agents receive text observations only. If image content
matters to an NPC, the router should canonicalize what that character perceives
as text.

Discord should attach player-safe revealed images after per-POV filtering. A
private image reveal must not fall back to public posting if a DM/private thread
delivery fails. CLI can initially print safe captions and local asset refs; map
images may also have an ASCII room/exits summary when available.

## Import Strategy

PDF import should be local-only and offline for protected content. It should be
treated as a compilation pipeline with review, not as a perfect one-shot parse.

The compiler pipeline should be staged:

1. **Source inventory**

   Record original source hashes, page count, rendered page images, generic
   text, OCR output, image stream inventory, extraction warnings, and importer
   version. Rendered pages are the visual source of truth; extracted image
   streams are supporting assets, not page identity.

2. **Page alignment**

   Reconstruct logical pages from rendered images, generic text, OCR spans,
   image streams, printed page labels, and section headings. Flag mixed
   image/text pages, decorative-only streams, missing pages, low-confidence OCR,
   crop/rotation problems, and page-label mismatches.

3. **Layout normalization**

   Preserve raw OCR lines and bounding boxes, then produce normalized spans for
   headings, columns, boxed/read-aloud text, sidebars, captions, tables, stat
   blocks, map labels, page numbers, footers, and decorative regions. Every
   normalized span must be auditable back to source.

4. **Section detection**

   Build a section tree from headings, table of contents, appendices, map
   listings, and manual corrections. Stable section ids become the parent
   structure for cards, maps, tables, and review issues.

5. **Domain card extraction**

   Draft cards for locations, keyed rooms, NPCs, monsters, factions, items,
   traps, clues, treasure, encounters, quests, handouts, random tables,
   statblocks, and fronts. Each card records aliases, outbound refs, source
   spans, visibility fields, confidence, and review status.

6. **Image classification and derivative preparation**

   Classify every image, page, crop, mask, and thumbnail. Prepare player-safe
   derivatives only after spoiler review. DM/source images stay host-only.

7. **Map graph building**

   Build reviewed graph nodes and edges for maps: rooms, keyed labels, normal
   exits, secret doors, stairs, vertical levels, portals, blocked paths,
   one-way routes, and fog/reveal regions. LLM/vision may propose labels and
   adjacencies, but human review approves topology.

8. **Table parsing**

   Parse dice ranges, weights, headers, row boundaries, nested-table refs, and
   result refs. Deterministic validation must catch gaps, overlaps, malformed
   ranges, and impossible dice expressions.

9. **Stat block parsing**

   Parse typed mechanics and validate combat-critical fields. Missing AC, HP,
   actions, saves, senses, special traits, reactions, legendary/lair actions,
   or spellcasting details create blocking review issues for combat use.

10. **Spoiler and reveal classification**

    Split visible descriptions from hidden features, traps, treasure, monster
    tactics, secret identities, clue dependencies, future reveals, and front
    knowledge. Every hidden field needs a reveal trigger or runtime condition.

11. **Review workspace generation**

    Generate human review artifacts: rendered page, OCR text, normalized
    layout, extracted cards, image classification, map graph overlay, tables,
    stat blocks, spoiler fields, unresolved refs, and blocking warnings.

12. **Pack compilation**

    Compile only reviewed or explicitly accepted records into the private
    runtime SQLite pack. The runtime resolver should never read raw OCR/page
    files directly.

The importer can use LLMs at import time because the expensive part happens
once. Turn-time should use compact compiled artifacts.

LLMs should propose OCR cleanup, section detection, card drafts, image
classification, spoiler splits, table interpretation, stat-block extraction,
cross-reference candidates, and front summaries. Deterministic code owns
hashing, provenance, confidence aggregation, schema validation, dice-table
checks, graph consistency, duplicate detection, source coverage, and pack
compilation. Manual review is required for page alignment, low-confidence OCR,
map topology, spoiler boundaries, player-safe handouts, randomization tables,
combat-critical stat blocks, and front/villain dossiers.

Runtime use requires:

* 100% page inventory with source hashes and review states
* no unreviewed low-confidence pages in reachable sections
* source refs, confidence, visibility classification, and review state on every
  runtime card
* reviewed map graph nodes/edges for maps used in navigation
* valid dice ranges and row boundaries for runtime tables
* typed mechanics for combat stat blocks, or explicit combat-unavailable flags
* no unresolved critical cross-refs for reachable locations, encounters, NPCs,
  items, clues, maps, appendices, or tables
* no high-spoiler content served without an approved reveal trigger
* zero blocking review issues, unless the pack is explicitly marked private
  playtest-only and blocked areas are disabled

## Module Pack Test Requirements

Before module content is wired into runtime play, tests and harnesses should
prove these contracts:

* Prompt hygiene: hidden/DM-only image metadata, source paths, raw OCR, map
  labels, and protected excerpts never reach narrator prompts, character-agent
  prompts, player responses, or logs by default.
* Checkpoint hygiene: checkpoints store pack refs, hashes, reveal state, and
  mutable adventure overlay only. They do not store raw images, page scans,
  full OCR, full source cards, or protected excerpts unless an explicit
  private-save/export mode is active.
* Visibility filtering: two player-bound characters can receive different
  `AssetReveal` sets from the same canonical event, matching
  `all_observers`/`only` semantics.
* Discord delivery: private or single-POV image reveals never fall back to a
  public channel. If private delivery fails, strip the private asset or fail
  with a non-spoiling player-safe error.
* CLI delivery: CLI can render safe asset captions and local refs without
  exposing hidden source paths, unrevealed filenames, or DM labels.
* Rewind: revealed handouts, map fog, cropped assets, and per-POV reveal state
  restore exactly when a checkpoint is rewound.
* Pack mismatch: missing packs, changed pack hashes, or missing asset refs fail
  loudly instead of prompting the router to improvise.
* Import gates: malformed tables, unreviewed map topology, unresolved critical
  cross-refs, and invalid stat blocks are blocked from runtime use.
* D&D adapter neutrality: battle maps, combat stat blocks, and D&D handouts use
  the same generic content/asset reveal surfaces as non-D&D content.
* Synthetic fixture coverage: committed tests use fake/redacted packs, not
  proprietary excerpts or images. Private live-pack harnesses may run locally
  against owned content but must not write protected artifacts into git,
  Beads, CI logs, or normal test reports.

## Public And Private Content Boundary

Public Ayoa must not ship protected D&D content. The repo can ship schemas,
adapters, and open/public packs only.

Private or owned module content should live outside the git-tracked repo by
default, for example:

```text
~/.ayoa/content/packs/phandelver_obelisk_private/
  pack.json
  compiled/
  indexes/
  raw/
  review/
```

Checkpoints should store content refs, roll ledgers, and campaign mutations.
They should avoid storing large protected excerpts unless the user explicitly
chooses a private-only save/export mode.

## Open Design Questions

These do not block basic D&D functionality:

* Should adventure state live in `world_state.global_flags`, a new typed
  `adventure_state`, or a pack-owned sidecar persisted next to checkpoints?
* Should compiled module packs use SQLite, JSONL, or both?
* How much raw source text may be stored in private checkpoints?
* Should module NPCs be imported as full Ayoa `CharacterRecord`s upfront, or
  spawned lazily from content refs when they enter play?
* Should foreshadowing be injected directly into router context, or should a
  separate adventure director choose safe opportunities first?

None of those need to be solved before the basic D&D mechanics path.
