# D&D Manual Pack Authoring Workflow

Status: initial workflow for manually authored protected module packs.

This document defines the manual authoring path for D&D module content packs
and projection/application profiles. OCR, extraction helpers, local vision
tools, and LLM drafts may assist in private review, but they are never
runtime-authoritative. A record becomes runtime data only after a coding agent
has inspected the private source, rewritten the material into redacted pack
text, attached provenance ids, assigned visibility and spoiler state, and set
the review gate deliberately.

This workflow extends the decisions in:

- `DND_MODULE_PRIVACY_BOUNDARY.md`
- `DND_ASSET_BACKING_AND_DELIVERY.md`
- `DND_ASSET_REVEAL_CONTRACT.md`
- `DND_MODULE_IMPORT.md`

It does not authorize raw OCR RAG, protected excerpts in prompts, checked-in
module text, or runtime fallback to private source pages.

## Authoring Principle

Manual pack authoring is a compile-and-review process, not transcription.

The coding agent may inspect local private source pages and private extraction
artifacts to understand the module, but the committed or runtime-ready output is
a set of redacted records:

- stable ids and hashes
- paraphrased summaries and runtime bodies
- source provenance ids, not source paths or excerpts
- visibility and reveal conditions
- confidence and review state
- gate status and gate reasons
- coverage notes for anything partial, uncertain, blocked, or intentionally
  omitted

The agent must not copy protected prose into compiled card text as a shortcut.
Even private compiled packs are runtime-readable artifacts; they should contain
reviewed derivative play data, not source-page dumps.

## Storage Boundary

Manual authoring uses the same three storage classes as the privacy decision:

| Layer | Purpose | Allowed material |
| --- | --- | --- |
| Local-only raw/review workspace | Inspect source pages, OCR, crops, draft notes, screenshots, and helper output. | Protected source material, raw OCR, source paths, page images, review screenshots, and agent scratch notes under ignored private storage only. |
| Private compiled pack | Runtime-readable SQLite/media records after coding-agent review. | Redacted cards, ids, hashes, provenance ids, review state, safe asset refs, coverage manifest, and reviewed media derivatives. |
| Runtime public surface | Prompts, checkpoints, normal logs, Discord, CLI, git, Beads, PR text, and tests. | Only allowlisted runtime projections, synthetic examples, aggregate counts, non-spoiling errors, and player-safe reveals after filtering. |

The coding agent can use private source paths while working locally, but those
paths must not enter git, Beads, prompt text, normal logs, default checkpoints,
compiled pack metadata, asset delivery refs, or test artifacts. If a helper
prints raw OCR, source paths, protected excerpts, or DM-only labels, treat that
output as private scratch and rewrite from it into redacted records.

## Record And Projection Format

Manual authoring has two outputs:

1. Reviewed content-pack records: the stable module authority.
2. Projection/application profiles: import-authored slices that create a seed
   checkpoint or apply the module to an existing checkpoint.

The compiled-pack vocabulary includes:

- `PageInventoryRecord`
- `ContentProvenance`
- `CompiledContentCard`
- `ContentAliasRecord`
- `ContentImageAsset`
- `CoverageManifest`
- asset catalog rows and reviewed derivative media

Domain-specific records such as encounters, traps, stat blocks, front dossiers,
handouts, tables, tactical map templates, actor dossiers, agent context slices,
and knowledge graph edges should use the typed content-pack schemas when they
exist. Temporary draft formats are authoring aids only; migrate directly rather
than keeping compatibility shims for obsolete record shapes.

Every runtime candidate record must carry these authoring fields:

| Field | Requirement |
| --- | --- |
| `pack_id` | Private stable pack id. Do not mention real private pack ids in git, Beads, PR text, or public docs. Synthetic docs and tests may use ids such as `synthetic-pack`. |
| `ref` or `asset_id` | Stable opaque record id. Prefer semantic prefixes without protected room names, for example `loc.ch02.area_014`, `npc.front_actor_002`, `enc.area_014_entry`, `asset.map_floor_03_player`, or `stat.creature_009`. |
| `card_kind` or `kind` | Runtime domain, such as `location_card`, `encounter`, `stat_block`, `map_asset`, `handout`, `trap`, `loot`, `secret`, `front_dossier`, `villain_dossier`, `alias`, or `tactical_map_template`. |
| `visibility` | The router lookup visibility, usually `hidden`, `router_hidden`, `player_visible_after_reveal`, `host_only`, or another explicitly documented value. Visibility must describe runtime exposure, not source-page layout. |
| `spoiler_class` | One of `none`, `low`, `moderate`, or `high`. Use `high` for future reveals, secret identities, hidden villain plans, unrevealed traps, secret routes, and outcome-changing treasure. |
| `title` | Redacted display/debug title. It must not be a protected source heading unless that heading is already safe to show at the relevant visibility. |
| `summary` | Short paraphrased runtime summary. This is the preferred compact router projection source. |
| `body` | Reviewed derivative play data. Keep it concise and paraphrased; do not use it as a source excerpt container. |
| `reveal_trigger` | Required for high-spoiler records that can ever become runtime-ready. State the in-fiction condition, not implementation mechanics. |
| `confidence` | `0.0` to `1.0`, reflecting source recovery and authoring certainty. Confidence from OCR or LLM drafts is only input evidence; the accepted value belongs to the coding agent. |
| `review_status` | `unreviewed`, `needs_review`, `reviewed`, `approved`, `blocked`, or `rejected`. Runtime-ready records must be `reviewed` or `approved`. |
| `gate_status` | `runtime_ready`, `flagged`, or `blocked`. This is the serving gate, separate from review state. |
| `gate_reasons` | Reviewable reason codes for flagged or blocked records, such as `low_confidence`, `missing_reveal_trigger`, `unresolved_cross_ref`, `map_topology_unreviewed`, `combat_fields_missing`, or `source_boundary_unclear`. |
| `provenance` | Source ids only: `source_asset_id`, `page_id`, `span_id`, `image_id`, `bbox`, `section_id`, `method`, `confidence`, `importer_version`, and `human_review_status`. No local paths or excerpts. |
| `aliases` | Redacted lookup aliases. Do not store DM-only labels, secret room names, protected prose, or future-spoiler names unless the alias is kept private and never projected to prompts/logs. |
| `metadata` | Structured redacted fields for the domain. Recursively exclude source paths, raw OCR, screenshots, protected excerpts, DM-only labels, and unsafe delivery refs. |
| `coverage_notes` | Human-readable private compiled-pack note for partial scope, skipped details, unresolved uncertainty, or handoff context. If it contains protected wording, keep it only in ignored private review storage, not in git or normal logs. |

Synthetic example:

```json
{
  "pack_id": "synthetic-pack",
  "ref": "loc.ch01.area_001",
  "card_kind": "location_card",
  "visibility": "router_hidden",
  "title": "Entry Area",
  "summary": "A cold threshold room with one visible exit and a concealed hazard.",
  "body": "Reviewed redacted notes for routing, navigation, and hazard setup.",
  "spoiler_class": "moderate",
  "reveal_trigger": "A character searches the threshold or crosses the marked floor.",
  "confidence": 0.92,
  "review_status": "approved",
  "gate_status": "runtime_ready",
  "gate_reasons": [],
  "aliases": ["entry", "threshold"],
  "provenance": [
    {
      "source_asset_id": "src.redacted.001",
      "page_id": "page.001",
      "span_id": "span.001.a",
      "image_id": "",
      "bbox": [0.12, 0.18, 0.74, 0.42],
      "section_id": "sec.ch01",
      "method": "manual-agent-review",
      "confidence": 0.92,
      "importer_version": "manual-pack-authoring-v1",
      "human_review_status": "approved"
    }
  ],
  "metadata": {
    "exits": ["loc.ch01.area_002"],
    "hazards": ["trap.ch01.area_001.floor"],
    "coverage_notes": "Synthetic example only; no protected source material."
  }
}
```

Projection/application profiles use `ContentPackProjectionArtifact` and related
schemas. Every accepted module handoff should include at least one explicit
profile or a blocker explaining why no profile is runtime-ready. A profile must
state whether it is for seed creation, existing-checkpoint application, or both,
and must include only reviewed/runtime-ready refs.

Profile outputs may include:

- router startup lookup refs and compact lookup catalog entries
- character-agent initial context projections
- initial engine-owned knowledge map
- active fronts and pressure state
- checkpoint seed text and optional field-start patches
- private D&D adapter overlay data

The engine consumes these profiles. It should not rediscover semantic slice
boundaries from raw domain records on every session.

## Record Lifecycle

Use this lifecycle for every manually authored record:

1. `candidate`

   The agent has identified a possible record from source pages, OCR, helper
   drafts, or cross-reference review. The candidate lives only in private review
   storage. It may contain raw source context because it is not a compiled pack
   record yet.

2. `draft_redacted`

   The agent has rewritten the candidate into redacted pack shape with stable
   ids, paraphrased text, provenance ids, initial visibility, spoiler class,
   confidence, and coverage notes. Set `review_status="unreviewed"` or
   `needs_review`; set `gate_status="flagged"` or `blocked`.

3. `source_reviewed`

   The agent has checked the source page(s), verified the rewrite, and removed
   protected excerpts, raw OCR residue, source paths, and unsafe labels. Set
   `review_status="reviewed"` when the record is accurate but may need a second
   acceptance pass; keep `gate_status="flagged"` if any serving concern remains.

4. `accepted`

   The record is intentionally usable for its declared scope. Set
   `review_status="approved"` or `reviewed`, `gate_status="runtime_ready"`, and
   no blocking `gate_reasons`. High-spoiler records require an approved
   `reveal_trigger` before this state.

5. `flagged`

   The record is useful to retain but not cleanly runtime-ready. Use
   `gate_status="flagged"` with explicit `gate_reasons`. Examples include a
   safe location card whose map topology is not reviewed, a stat block missing
   optional features, or a handout with player-safe text but no reviewed image
   derivative.

6. `blocked`

   The record must not be served. Use `review_status="blocked"` or
   `gate_status="blocked"` when source alignment is uncertain, OCR confidence
   is low, a protected excerpt remains, the spoiler boundary is unclear, a
   combat-critical stat block is incomplete, a map template has unreviewed
   topology, or a cross-reference points to an unauthored required record.

7. `rejected`

   The candidate was wrong, duplicate, unsafe, or outside scope. Keep only a
   sanitized rejection note in compiled-pack coverage if needed. Do not preserve
   protected rejected drafts in git, Beads, or shareable review exports.

Runtime-ready means all of the following are true:

- `review_status` is `reviewed` or `approved`.
- `gate_status` is `runtime_ready`.
- `confidence` meets the pack gate policy.
- `gate_reasons` has no blocking reason.
- high-spoiler content has a reviewed reveal trigger.
- provenance ids are present and source-checkable in the private workspace.
- text and metadata pass the protected-excerpt, source-path, raw-OCR, and
  DM-label hygiene checks.
- referenced records, aliases, assets, maps, tables, and stat blocks either
  exist or are explicitly marked unavailable with a non-spoiling gate reason.

## Domain Authoring Expectations

### Pages And Sections

Author a page inventory before trusting downstream records. Every logical page
needs a stable `page_id`, source hash/fingerprint, printed label if known,
section assignment, confidence, review status, and coverage status. Page
records may note extraction defects, but raw OCR text and source paths stay in
private review storage.

Sections should provide stable parentage for authored cards. A missing or
uncertain section tree is a coverage warning because cross-references and page
range audits depend on it.

### Locations, Encounters, Loot, Traps, And Secrets

Location cards should separate:

- player-perceivable description after arrival or inspection
- router-hidden topology and adjacency
- concealed hazards, locked routes, secret doors, and search triggers
- encounter start conditions and reinforcement hooks
- treasure or loot availability and depletion refs
- clues and future-reveal dependencies

Traps and secrets must be their own refs when they have independent reveal
conditions, mechanics, or campaign state. Do not bury secret state only in a
location prose paragraph. Loot that can be taken, spent, consumed, or moved
needs stable refs so the checkpoint overlay can record depletion.

Encounter records should identify participants by refs, starting conditions,
default tactics in redacted terms, possible noncombat resolutions, and links to
required stat blocks, loot, traps, maps, and front/villain consequences.
Combat-critical missing refs block combat automation for that encounter.

### Maps, Images, And Handouts

Every image-like object starts as a private asset. Do not treat a source image,
rendered page, DM map, crop, or OCR page as player-safe by default.

Author these distinctions explicitly:

- `DM/source image`: private source or host-reference asset only.
- `player-safe derivative`: reviewed image or crop with spoilers removed or
  masked, addressed by safe `asset://<pack_id>/<asset_id>` delivery ref.
- `handout`: player-facing document/artifact with reveal conditions,
  possession/reading constraints, safe caption, safe alt text, and optional
  partial reveals.
- `map overlay`: player-safe map or crop with fog/reveal region ids and
  no hidden labels unless already revealed.
- `reference-only image`: private asset useful for authoring but not deliverable
  in runtime play.

`ContentImageAsset` rows must include `review_status`, `spoiler_class`,
`safe_for_players`, `safe_for_llm`, safe captions/alt text where applicable,
hashes, dimensions, and safe delivery refs. `source_ref` remains private
catalog provenance and must not appear in player payloads, prompts, normal logs,
or default checkpoints.

### Tactical Map Templates

Tactical map templates are manual projections from reviewed maps into runtime
geometry. They are not image OCR output.

Each template should carry:

- `template_id`
- `derived_from_map_asset_id`
- target runtime schema, usually `DndBattleMapState`
- map kind and scale assumption
- grid width, height, origin notes, orientation, square size, and confidence
- spawn anchors for players, enemies, reinforcements, exits, retreats, and
  fallback starts
- terrain features: walls, doors, windows, pits, cliffs, stairs, balconies,
  furniture, water, cover, difficult ground, blocked movement, and line-of-sight
  blockers
- secret features with reveal triggers
- area links from keyed labels to location refs
- provenance ids, review status, confidence, and gate reasons

If exact topology is uncertain, set the template to `flagged` or `blocked`.
Approximate tactical geometry is acceptable only when the record says what was
approximated and the runtime scope can tolerate it.

### Stat Blocks

Stat block records must be typed rules data, not screenshots or copied source
text. Required fields depend on intended use:

- identity refs and redacted aliases
- ruleset id
- AC, HP, speed, ability scores, saves, skills, senses, languages, CR/XP
- traits, actions, reactions, legendary actions, lair actions, and spellcasting
  when present
- damage resistances, immunities, vulnerabilities, condition immunities, and
  special movement
- parse warnings, confidence, provenance, and review status

Missing AC, HP, attack/action data, save/DC data, damage expressions, or
spellcasting details block combat automation. A partial stat block may be
runtime-ready only for noncombat lookup if its gate reasons say combat is
unavailable.

### Front And Villain Dossiers

Front and villain dossiers preserve long-range pressure without exposing future
truth to players or character agents.

Author:

- front/villain refs with redacted labels
- goals, constraints, resources, minions, domains, and current starting state
- what the front initially knows
- how it plausibly learns events
- escalation triggers, clocks, thresholds, cooldowns, and restraint rules
- response palette: spies, threats, relocation, traps, negotiations, attacks,
  rumors, or environmental pressure
- safe foreshadowing opportunities and hard spoiler boundaries
- provenance, confidence, review status, and coverage notes

The dossier is immutable pack data. The checkpoint overlay records what the
front currently knows, active plans, clock progress, introduced refs, and
campaign-specific mutations.

### Aliases And Cross References

Aliases and cross refs make manual packs usable, but they are high leakage risk.

Author aliases only when they help lookup or graph integrity. Classify each as:

- `player_safe`: a name players may know at the current reveal boundary
- `router_hidden`: a hidden lookup aid that must not be logged or projected
- `host_only`: private authoring/review alias, not runtime prompt material

Cross refs should use record ids, not copied headings. Required cross refs for
reachable locations, encounters, maps, handouts, stat blocks, loot, traps,
secrets, fronts, and appendices must resolve or carry a blocking gate reason.
Optional cross refs may be flagged with coverage notes.

## Private Source Inspection Rules

Coding agents may inspect private source pages only inside the local private
workspace. During inspection:

- Keep raw OCR, screenshots, rendered pages, source paths, helper JSON, and LLM
  drafts in ignored private storage.
- Do not paste protected excerpts into git-tracked docs, tests, Beads, commit
  messages, PR descriptions, normal logs, prompt files, or checkpoints.
- Do not use real private source paths in compiled metadata or test fixtures.
- Do not let DM-only labels, hidden room names, secret identities, or unrevealed
  asset ids become normal debug text.
- Rewrite helper output into concise redacted records before compilation.
- Replace source references with stable provenance ids that can be resolved only
  inside the private workspace.
- Run hygiene checks on compiled text and synthetic fixtures before commit.

If a task needs a dump of private extraction evidence, write it only under
ignored private storage and report a sanitized path category, not the raw path,
in public handoff text.

Hosted LLMs must not receive raw protected pages or OCR as import prompts under
this workflow. Hosted LLMs may receive only redacted candidate fields that
already pass the same deny-list checks as compiled pack text.

## Coverage Auditing

Partial manual import must never masquerade as complete. Every pack handoff
needs a `CoverageManifest` plus domain coverage notes.

At minimum, audit:

- source page count versus inventoried page count
- sections found, reviewed, missing, and blocked
- authored locations/keyed areas versus expected locations/keyed areas
- maps, map derivatives, fog regions, and tactical templates
- handouts, inscriptions, letters, symbols, and player-facing documents
- encounters and linked stat blocks
- stat blocks parsed, partial, blocked, or intentionally unavailable
- loot, treasure, traps, secrets, clues, and depletion refs
- tables and randomization records, including malformed or unaudited tables
- front/villain dossiers and escalation hooks
- aliases and unresolved cross refs
- low-confidence records, high-spoiler records, flagged records, and blocked
  records

The manifest should report counts for ready, flagged, blocked, low-confidence,
and high-spoiler records. Domain notes should explain scope, for example:

- `chapter_01_locations_complete`
- `chapter_01_tactical_maps_blocked`
- `appendix_stat_blocks_partial`
- `handouts_player_safe_review_pending`
- `front_dossier_skeleton_only`

Runtime configuration must be able to disable blocked areas or mark the pack as
private playtest-only. A pack is not complete until its manifest says what is
covered, what is intentionally excluded, what is blocked, and what runtime
features are unavailable.

## Handoff Expectations

When handing off manual authoring work, provide only sanitized information:

- changed compiled-pack files or docs by path category, not private source paths
- changed projection/application profile artifacts by path category
- synthetic or redacted record ids
- aggregate coverage counts
- review states and gate reason codes
- blocked domains and next authoring steps
- tests/checks run

Do not include source page screenshots, raw OCR snippets, protected excerpts,
real private module titles, source fingerprints, source paths, DM labels, or
spoiler-bearing aliases in handoff text, Beads, commit messages, or PRs.

If follow-up work is needed, create or update durable task tracking only with
sanitized descriptions and synthetic examples. The task should say which domain
is blocked and why, not quote the protected source.

## Test And Fixture Policy

Tests, fixtures, and docs committed to git must use redacted synthetic material
only. They must not depend on private PDFs, private compiled packs, private
source paths, OCR output, screenshots, or real module excerpts.

Useful tests for this workflow include:

- compiled card hygiene rejects protected-excerpt and absolute-path sentinels
- metadata sanitization drops forbidden nested keys
- coverage gates block low-confidence, unreviewed, blocked, or high-spoiler
  records without reveal triggers
- asset payload filtering returns only player-safe reviewed assets with safe
  delivery refs
- prompt/checkpoint/log hygiene tests use synthetic sentinel strings
- coverage manifests expose partial import rather than reporting success from
  card count alone

Synthetic fixtures should look like authored pack records but must be obviously
fake. Use ids such as `synthetic-pack`, `page.001`, `loc.test.entry`, and
`asset.handout.safe_001`; use prose such as "Reviewed redacted room notes" or
"Synthetic protected-excerpt sentinel" rather than source-derived text.
