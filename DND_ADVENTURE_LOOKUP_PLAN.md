# Delta-Based Adventure Lookup Plan

## Summary

Ayoa should not import a large adventure module into checkpoint lore or repeat a
large router packet every turn. The compiled module pack should behave as a
private content oracle. The router receives only content it has not already been
given, plus short signals that new module knowledge has become relevant.

The runtime needs two related ledgers:

- `router_content_memory`: content refs already introduced into router history.
- `adventure_state`: mutable campaign overlay: visited places, discovered clues,
  front and villain knowledge, active plans, depleted encounters, spawned refs,
  and module overrides.

New content is inserted once as compact assistant-side router history, adjacent
to the existing `prior_event` records. Subsequent turns rely on router history
and canonical events, not repeated cache-write packets.

## Key Changes

- Add generic content-pack state, not D&D-specific state:
  - `session.content_state[pack_id]`
  - introduced router refs with hashes and turn indexes
  - pending content signals waiting to be introduced
  - front and villain knowledge and clocks
  - module refs linked to spawned `CharacterRecord`s
- Add a `ContentResolver` around router calls:
  - deterministic prefetch from current location, actor, aliases, recent events,
    and pending signals
  - optional router lookup preflight when deterministic matching is uncertain
  - no canonical event for lookup-only work
- Add compact router history records such as:
  - `content_known ref=... scope=router visibility=hidden hash=...`
  - `front_signal ref=... actor=... knows=... pressure=...`
  - `location_card ref=... exits=... hazards=... clues=...`
- Do not repeat introduced refs unless the checkpoint says the source changed,
  the prior record was compacted away, or a new delta supersedes it.

## Router Lookup Protocol

Use a bounded two-phase router path for unique player choices:

1. `event_router_content_lookup` sees the normal router history, current
   actor/input, known content refs, and a compact content catalog or alias index.
2. It returns `ready=true` or exact lookup requests: location, entity, rule,
   reveal, front, or source query, plus reason, urgency, and spoiler boundary.
3. The resolver fetches records from the private pack, converts them into compact
   `content_known` records, and the normal event-router call runs with those
   records included.
4. Max one retry by default. If the router still lacks required content, fail
   loudly with an auditable missing-content error instead of improvising from
   nothing.

The final event schema stays clean. The ordinary router still outputs canonical
events, observers, routing roles, spawns, commitments, and location updates.

## Villains And Fronts

Model the main villain as an off-stage front, not a room on the player graph.

The content pack should include `FrontDossier` records:

- goals, constraints, resources, minions, and domains of control
- what the front initially knows
- how it learns public or semi-public events
- trigger thresholds for attention and escalation
- plausible response palette: spies, threats, assassins, bribery, attacks,
  relocation, traps
- cooldown and restraint rules so it does not act every turn

Runtime `FrontState` tracks what the villain currently knows and what plans are
active. When players create public consequences, the resolver queues a one-time
`front_signal`; the router decides whether that information reaches the villain
now, whether to select the villain or a minion as a background `next_output`, or
whether to hold.

If selected, the villain or minion acts through the normal agent/background
route. Their public output is routed back through the router, scoped by
visibility, and any consequences enter canonical history. Hidden plans stay
hidden until surfaced by reports, arrivals, changed locations, attacks, rumors,
missing NPCs, or other observable facts.

## Content Pack Shape

Use a private compiled SQLite pack with FTS5 first. Embeddings can be added
later as an optional recall index only if deterministic refs, aliases, graph
adjacency, and FTS are not enough. The raw extraction corpus is separate:
original PDFs, extracted images, OCR text, layout JSON, thumbnails, crop masks,
and review exports live as private files with a raw catalog SQLite database.
Runtime lookup reads compiled records, not raw OCR/page files.

Pack artifacts:

- raw source catalog refs and compiled page/source chunks with private citations
- section/outline hierarchy
- location and keyed-area cards
- NPC, monster, faction, item, trap, table, stat block, and handout cards
- map assets with reviewed topology, fog/reveal regions, and player-safe
  derivatives
- image assets for maps, handouts, portraits, item art, mood plates, source
  pages, table pages, stat block pages, and decorative art
- reveal/clue graph with spoilage boundaries
- front dossiers and action palettes
- import warnings and human-review flags
- provenance links from every compiled field back to source assets, pages,
  spans, images, bboxes, importer versions, confidence, and review status

The current local `stories/curse_of_strahd.pdf` is a large private source file,
and the local environment does not currently include enough PDF tooling for deep
extraction. The import path should therefore plan for page-addressable import,
outline extraction, maps/images, stat blocks, tables, and appendices as separate
domains.

Images are private table assets. Lookup may return image refs alongside text
refs, but not raw bytes. The router may reveal player-safe assets as sibling
payloads to canonical facts using the same `all_observers` / `only` visibility
semantics. Narrator prompts receive visible facts and safe captions only, never
hidden image refs, source paths, DM notes, map labels, or unrevealed metadata.
Agents receive text observations only.

## Test Plan

- Verify introduced content refs are not re-added to router history on later
  turns.
- Verify newly relevant deltas are introduced exactly once.
- Verify router lookup preflight fetches a requested off-path room, entity, or
  reveal before adjudication.
- Verify hidden content never reaches narrator prompts.
- Verify DM-only or hidden image refs, source paths, map labels, and protected
  excerpts never reach narrator prompts, agent inboxes, player responses, logs,
  or default checkpoints.
- Verify revealed image assets are filtered per POV and can differ between two
  player-bound characters observing the same event.
- Verify Discord private image delivery never falls back to a public channel,
  and CLI output shows only safe captions/refs.
- Verify villain front signals are created from public consequences, can trigger
  a background turn, and do not reveal hidden plans to players.
- Verify rewind restores content memory, front knowledge, active plans, and
  depleted/visited state with the checkpoint.
- Verify rewind also restores handout reveals, map fog, cropped assets, and
  per-POV asset visibility.
- Verify missing or changed pack hashes fail loudly instead of letting the
  router improvise from stale refs.
- Verify non-content stories keep existing prompt shape and runtime behavior.

## Assumptions

- Module content is private/user-owned and stays outside tracked repo files.
- The router is allowed a bounded lookup preflight, but not arbitrary live tool
  access inside a normal event output.
- The first implementation should favor deterministic refs, FTS, and authored
  front dossiers over broad semantic RAG.

## PDF Import Fidelity Deep Dive

The current local `stories/curse_of_strahd.pdf` should be treated as a scanned
or image-dominant source, not a clean text PDF. Local inspection found:

- file size: 226,468,753 bytes
- PDF header: `%PDF-1.6`
- pages visible to the generic reader: 258
- embedded JPEG images: 258
- dominant JPEG dimensions: 257 images at 2550 x 3300
- `/ToUnicode` maps: 0
- `/Font` occurrences in raw bytes: 1
- generic text extraction: mostly empty page markers, with only a small early
  chapter slice visible and visibly damaged by line-break/token artifacts
- local tool availability: no `pdfinfo`, `pdftotext`, OCR, image, or Python PDF
  libraries installed in the current environment

This means the first import problem is not retrieval. It is source recovery.
SQLite FTS, aliases, graph lookup, and runtime LLM preflight cannot compensate
for missing or untrusted extraction. Before runtime lookup exists, the importer
needs a compilation-and-review pipeline that can prove what it recovered and
what it failed to recover.

Minimum fidelity gates before a private pack can be trusted:

- Page inventory: every PDF page has a stable source fingerprint, image
  extraction status, OCR/text extraction status, printed page label if known,
  and review status.
- OCR coverage: every page has measured text coverage and confidence, with
  warnings for blank, low-confidence, rotated, cropped, column-confused, or
  image-only pages.
- Layout recovery: headings, sidebars, boxed/read-aloud text, tables, stat
  blocks, appendices, captions, and map labels are detected as separate layout
  domains rather than flattened into prose.
- Map recovery: full-page maps and keyed images are extracted as image assets
  and reviewed into graph edges; text extraction alone is not accepted as map
  topology.
- Table recovery: every table keeps row boundaries, dice ranges, headers, and
  source spans; malformed tables are blocked from runtime use until reviewed.
- Stat block recovery: stat blocks are parsed into typed mechanics plus source
  refs, with warnings for missing actions, traits, HP/AC, saves, skills, and
  special rules.
- Spoiler classification: visible descriptions, hidden features, traps,
  treasure, monster tactics, secret identities, and future reveals are separated
  into fields with explicit visibility and reveal triggers.
- Cross-reference resolution: cards record outbound refs to appendices, maps,
  encounters, NPCs, items, factions, randomized placements, and later reveals.
- Coverage manifest: the compiler reports expected-vs-found counts for pages,
  sections, keyed areas, maps, tables, handouts, stat blocks, NPCs, monsters,
  items, traps, clues, and unresolved cross-references.
- Human review gates: low-confidence or high-spoiler records cannot be served
  to the runtime until reviewed or explicitly accepted for private playtesting.

Runtime lookup should not be considered until these import artifacts exist.
Without them, the router will be asked to adjudicate from partial OCR, missing
maps, malformed tables, and unverified spoiler boundaries.

## Open Questions From Critical Review

- Can the current `stories/curse_of_strahd.pdf` be extracted with enough
  fidelity to produce complete page chunks, headings, keyed areas, tables,
  appendices, stat blocks, handouts, and maps, or does the import pipeline need
  an explicit OCR/image/table extraction phase before any runtime work?
- How will the importer recover authoritative map topology for map-first
  material such as Castle Ravenloft: room keys, vertical adjacency, stairs,
  secret doors, one-way transitions, and legal exits that may not survive text
  extraction?
- What field-level provenance must every derived card carry: source hash,
  printed page label, PDF page index, section path, extraction method, text span
  or bounding box, confidence, importer version, and human-review status?
- How will the compiler prove import coverage and catch omissions or
  hallucinated records for keyed rooms, tables, stat blocks, traps, clues,
  treasure, and NPC/faction references?
- How will visible read-aloud text, hidden DM-only facts, traps, treasure,
  monster tactics, future reveals, and spoiler boundaries be separated when the
  source PDF interleaves them or loses typography during extraction?
- What is the exact privacy boundary for protected module content once pack
  records are sent to hosted LLM providers, stored in checkpoints, included in
  logs, written to review artifacts, or mentioned in Beads/GitHub issues?
- Should raw PDFs, derived SQLite packs, extracted images, OCR text, handouts,
  and review exports be forced outside the git repo or ignored explicitly before
  any importer is run?
- Can `content_known`, `front_signal`, and `location_card` records safely live
  beside compact `prior_event` router history, or do they need a first-class
  prompt grammar, checkpoint schema, replay path, refresh path, and compaction
  invariant?
- How can `router_content_memory` know that an introduced ref is still present
  and semantically intact in the router's effective context after provider
  compaction, retries, rewind, or checkpoint reload?
- What atomic transaction boundary keeps `content_state`, router history,
  canonical events, narrator render state, and checkpoint saves from diverging
  when lookup, final routing, narration, or save/load fails mid-turn?
- How will immutable source cards be projected through mutable campaign state so
  killed monsters, looted treasure, sprung traps, opened doors, moved NPCs, and
  discovered clues do not revert when a location ref is reused?
- Who owns front and villain knowledge: the router, a durable front ledger, an
  agent's private conversation, or canonical in-fiction observations, and how
  does that avoid hidden side channels into agents, narrators, or players?
- When a public consequence may reach a villain, what observable information
  path makes that fair in fiction before a hidden `front_signal` produces an
  attack, rumor, trap, spy report, or escalation?
- How does lookup interact with D&D Cat II and combat flows that may use
  dedicated ruleset prompts instead of the ordinary fresh router path?
- What concurrency and ordering contract handles simultaneous players in
  different module locations, front clocks firing, background agent turns, and
  pending lookup preflights without blocking the table or corrupting state?
- What player-facing error should replace a generic internal failure when module
  content is missing, extraction is incomplete, or lookup cannot resolve a
  requested ref without improvising?
- What player-facing orientation contract ensures keyed-area narration gives
  enough visible exits, affordances, occupants, and obvious interactables without
  turning into a menu or leaking hidden map data?
- How will multiplayer `/history`, resume, and recap surfaces avoid showing the
  wrong POV, missing private clues, or leaking that another player received
  hidden module information?
- Should runtime lookup use an LLM only as a fallback after import-time
  semantic preparation, deterministic refs, aliases, graph adjacency, and
  validated retrieval fail?
- Which redacted synthetic fixtures and live private-pack harnesses are required
  so tests can validate import/runtime contracts without committing protected
  adventure excerpts?
