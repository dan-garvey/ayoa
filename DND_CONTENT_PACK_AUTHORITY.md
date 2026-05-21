# D&D Content Pack Runtime Authority And Schema Boundaries

Status: accepted decision for `ayoa-3gk`. This document defines runtime
authority and schema boundaries. It does not implement storage, resolver,
schema, or delivery changes.

## Current Grounding

This decision sits downstream of:

- `DND_MODULE_PRIVACY_BOUNDARY.md`, which separates raw/private artifacts,
  compiled private packs, runtime prompt surfaces, checkpoints, logs, CLI, and
  Discord output.
- `DND_ASSET_REVEAL_CONTRACT.md`, which makes asset reveal an event sidecar
  and keeps player-visible assets filtered through safe payloads.
- `DND_ASSET_BACKING_AND_DELIVERY.md`, which makes `asset://<pack>/<asset>` the
  canonical runtime delivery ref and rejects source paths as delivery refs.
- `DND_ADVENTURE_LOOKUP_PLAN.md`, which says runtime lookup reads compiled
  SQLite records, not raw OCR or page files.
- `DND_MODULE_IMPORT.md`, which treats initial module imports as manual
  coding-agent-authored records compiled and reviewed into structured local
  runtime data.

The rules-neutral engine remains router-centered. The router canonicalizes
events, the narrator renders visible facts, character agents receive text
observations, checkpoints persist session state, and D&D mechanics remain an
adapter around that engine.

## Decision

The compiled SQLite content pack is the authoritative runtime representation for
reviewed module records.

JSONL source files, manually authored draft files, PDF/OCR extraction outputs,
review exports, screenshots, private source catalogs, and intermediate import
notes are not runtime fallbacks. They may feed the compiler and review workflow,
but a turn-time resolver must not consult them when a compiled row is missing,
stale, unreviewed, blocked, unsafe, or incomplete. Any future exception must be
accepted by a separate decision that names the exact source, privacy boundary,
failure behavior, and tests.

Runtime uses these authority layers:

1. Compiled SQLite pack: immutable reviewed module data and indexes served to
   runtime by stable `pack_id`, version, schema version, refs, content hashes,
   review gates, and safe asset ids.
2. Checkpoint/session overlay: mutable campaign state, revealed/introduced refs,
   per-pack overlay progress, active D&D adapter state, and player/session
   mutations.
3. Router history: compact assistant-side records that summarize already
   introduced pack facts for router continuity. These are projections, not a
   second content database.
4. Canonical event log: story-time truth authored by the router, including
   visible facts and content-enabled side effects such as asset reveal requests.
5. Private raw/review artifacts: local-only authoring evidence, never queried by
   turn-time runtime.
6. Safe player asset payloads: delivery-ready projections validated from pack
   asset catalog rows and filtered per POV.

## Pack Database Authority

The pack database owns reviewed, immutable module records:

- pack manifest: `pack_id`, pack version, schema version, source fingerprint,
  compiler/importer version, build hash, coverage summary, and dependency
  metadata
- reviewed content cards: keyed areas, locations, NPC dossiers, factions,
  clues, handouts, tables, traps, treasure, items, rules notes, and other
  generic adventure records
- lookup indexes: aliases, graph adjacency, FTS rows, cross references,
  reveal/clue graph links, source-safe labels, and deterministic lookup keys
- provenance and coverage: page ids, span ids, image ids, bboxes, confidence,
  import method, review status, gate status, and coverage warnings
- private asset catalog: source asset refs, reviewed player-safe derivatives,
  safe captions, safe alt text, hashes, dimensions, MIME type, and canonical
  logical delivery refs
- adapter payload rows: D&D statblocks, encounter seeds, tactical map templates,
  loot/treasure definitions, trap mechanics, hazard mechanics, and D&D-specific
  front/action metadata when those records need typed mechanical structure

Only reviewed or approved rows whose gate status allows runtime service may be
served. "Exists in the pack" is not enough. Runtime-ready means the compiler and
review gate have accepted the row for the requested projection.

The pack database does not own mutable session facts. It may define that a room
contains a trap, an NPC has a statblock, an encounter has candidate monsters, or
a front has an escalation palette. It does not own whether this table has
visited the room, sprung the trap, killed the monster, learned the clue, looted
the chest, revealed the map, advanced the clock, or overwritten the module fact.

## Checkpoint And Session Overlay

Checkpoints own mutable play state and portable recovery state:

- installed/active pack identity by `pack_id`, expected pack version, schema
  version, build/content hash, and local locator key
- introduced content refs by `pack_id`, `ref_id`, `content_hash`, label, kind,
  source event id, and introduction time
- pending content signals queued by event consequences, lookup needs, fronts,
  villain pressure, or reveal opportunities
- reveal state, map fog/masks/crops, asset reveal event ids, audience, visible
  character/user ids, captions, and safe derivative ids or hashes
- campaign overlay state such as visited refs, consumed/depleted refs, spawned
  refs, overridden refs, discovered clues, active fronts, villain knowledge,
  clocks, temporary plans, and table-specific notes
- runtime D&D adapter state: active combat, combatants, initiative, HP, death
  saves, active effects, roll transactions, loot offers, XP awards, tactical
  battle map state, and other state created by play
- character inventory/mechanics changes, including claimed loot, spent
  resources, conditions, and sheet/runtime deltas

Checkpoints store refs, hashes, state, and overlays. They do not store the pack
database's full card bodies as a cache, do not store raw source material, and do
not become a compatibility layer for obsolete pack row shapes. Until a future
release-candidate migration decision says otherwise, changed schema fields
should be retired directly rather than kept alive through save shims.

Portable checkpoints must not require absolute local paths. If code needs a
local SQLite file or media cache path, it must resolve a logical pack locator
through local-only configuration. Persisting absolute `db_path`, `pack_path`,
source PDF paths, extraction paths, or raw media paths in shareable checkpoints
is blocked.

## Router History And Canonical Events

Router history owns compact continuity, not content authority.

When a resolver introduces pack content to the router, it appends deterministic
assistant-side compact records such as `content_known`, `location_card`, and
`front_signal`. These records preserve the runtime contract the router needs:
ref, scope, visibility, hash, kind, pack, summary, exits, hazards, clues, actor,
knowledge, and pressure. They are not a fallback copy of the source card and do
not authorize future lookup from stale prose.

If the router later needs a record whose hash differs from the introduced
history, the resolver must fetch the current reviewed pack row and append a new
delta record. If the pack row is missing, unreviewed, blocked, mismatched, or no
longer safe for the requested projection, runtime fails loudly before the router
call rather than asking the router to improvise.

Canonical events remain the story-time authority for what happened and who
perceived it. Content pack refs become player-visible only when the router emits
observable facts or a content side effect such as an asset reveal. A compact
hidden router record is never itself a player-visible fact.

## Generic Content Versus D&D Adapter Mechanics

Generic content-pack data must stay rules-neutral. The core engine may know how
to resolve pack refs, validate hashes, queue generic content signals, append
compact router records, persist content overlay state, and filter safe asset
payloads. It must not bake D&D terms into the baseline router, narrator,
character-agent, checkpoint, or frontend contracts for sessions that are not
using D&D.

Generic pack rows include narrative and structural adventure data:

- locations and keyed areas
- exits, clues, hazards, descriptions, summaries, and reveal triggers
- NPC/faction/front dossiers expressed as story pressure and knowledge
- handout/image asset refs and player-safe captions
- generic item/treasure/trap/hazard records when no D&D mechanics are needed
- graph and lookup indexes

D&D adapter payloads are opt-in mechanical attachments under `dnd5e_basic`:

- monster/NPC statblocks, challenge/XP values, actions, traits, senses,
  languages, proficiencies, saves, and spell/resource data
- encounter seeds, combatant spawn candidates, initiative/combat setup hints,
  and combat-end/XP policy metadata
- tactical maps as D&D battle-map templates: grid dimensions, tokens, terrain,
  area templates, cover/line-of-sight/movement annotations, and safe map asset
  refs
- loot and treasure mechanics: item ids, quantities, currency, value, weight,
  attunement, identification, consumable flags, and eligible characters
- traps, hazards, and environmental mechanics: DCs, saves, attack bonuses,
  damage expressions, conditions, reset/depletion state, and detection/disarm
  affordances
- fronts or villains only where D&D-specific mechanical behavior is needed,
  such as minion statblock refs, encounter palettes, travel/combat triggers, or
  mechanical consequences

D&D adapter schemas may consume pack rows and materialize adapter state in the
checkpoint, but they must use the same generic content-pack authority boundary:
read reviewed compiled rows, store session mutations in overlay state, expose
only compact records or safe payloads to model/frontends, and fail loudly when
required pack data is not serviceable.

## Manual Authoring To Reviewed Pack Rows

Initial module content is manually authored by a coding agent and reviewed into
pack rows. The accepted path is:

1. Inspect private source and private extraction artifacts only in ignored or
   local-only storage.
2. Author draft records with stable refs, redacted/paraphrased runtime text,
   visibility, spoiler classification, aliases, graph links, adapter payloads,
   asset refs, provenance ids, coverage notes, confidence, and review status.
3. Run compiler validation that rejects raw source excerpts, absolute paths,
   unsafe delivery refs, missing required fields, duplicate refs, broken graph
   links, blocked provenance, and unsupported adapter payload shapes.
4. Review records against source evidence. Records remain unavailable to runtime
   while `unreviewed`, `needs_review`, `blocked`, or `rejected`.
5. Compile accepted records into SQLite tables and indexes with deterministic
   row hashes and a pack manifest hash.
6. Runtime opens only the compiled pack. It never reads the draft file to patch
   a missing row and never reads a private review export to fill a gap.

Draft formats are replaceable authoring tools. SQLite rows plus their manifest,
hashes, gates, and reviewed status are the runtime contract.

## Private Raw And Review Artifacts

Private raw/review artifacts may contain original PDFs, OCR, rendered pages,
extracted images, page thumbnails, source paths, DM labels, screenshots, crop
masks, raw notes, and proposed diffs. They are authoring evidence and review
workspace only.

They are forbidden from:

- turn-time lookup
- hosted router, narrator, character-agent, or D&D prompts
- checkpoints
- normal logs and playtest reports
- Discord and CLI output
- `TurnResponse` and frontend view payloads
- git-tracked fixtures and docs, except synthetic examples
- Beads issue text and public review summaries

Runtime failure must not trigger a fallback into private review material. A
missing reviewed pack row is a missing content error, not permission to query a
draft or source export.

## Safe Player Asset Payloads

The pack asset catalog owns private asset metadata and reviewed safe derivative
metadata. The player-visible runtime owns only filtered safe payloads.

An asset may be delivered only after the runtime validates:

- active pack id, version, schema version, and content/build hash
- asset row exists in the compiled pack
- row is reviewed or approved
- row is safe for players for the requested presentation
- `delivery_ref` is a safe logical ref such as `asset://<pack_id>/<asset_id>`
- bytes resolve through approved backing and match the catalog hash when byte
  delivery is attempted
- POV visibility matches the owning canonical event and reveal request

`SafeAssetRevealPayload` may contain player-safe pack id, asset id, kind, title,
MIME type, dimensions, SHA-256, safe delivery ref, presentation, caption, and
alt text. It must not contain source refs, local paths, OCR, provenance, private
metadata, hidden map labels, unsafe filenames, or protected excerpts.

## Loud Failure Rules

The runtime must fail before state commit or before unsafe delivery when a
required content contract is violated.

Hard failures include:

- missing active pack for a required lookup, reveal, combat spawn, statblock,
  encounter, trap, loot table, front, or map
- pack id, version, schema version, manifest hash, content hash, source
  fingerprint, or dependency mismatch
- missing required row, missing required relation, broken graph edge, missing
  adapter payload, missing asset, or missing media bytes
- row review status is `unreviewed`, `needs_review`, `blocked`, or `rejected`
  when runtime service requires `reviewed` or `approved`
- gate status is blocked or not runtime-ready for the requested projection
- unsafe refs, absolute paths, raw source paths, unapproved schemes, unsafe
  delivery refs, or non-hash-verified media backing
- hidden/source-only fields requested for a player, narrator, character-agent,
  normal log, checkpoint, or safe response payload
- compact router history claims a hash that no longer matches the reviewed pack
  row needed for the turn
- D&D adapter asks for a statblock, encounter payload, tactical map, loot/trap
  mechanic, or front mechanical payload that is absent or invalid

The correct failure mode is a non-spoiling runtime error with sanitized ids and
operator-diagnostic logs. The runtime must not silently skip the side effect,
invent replacement content, summarize missing source material, emit a fallback
asset caption as if delivery succeeded, or ask a model to bridge the missing
authority.

## Blockers And Open Technical Choices

These choices are not decided here and should block implementation slices that
depend on them:

- Pack registry and locator: portable checkpoints need logical pack locators
  and local-only resolution. Absolute SQLite/media paths in checkpoints are not
  accepted.
- SQLite schema normalization: exact table layout, FTS tables, graph indexes,
  adapter payload table boundaries, and migration mechanics remain a separate
  storage design.
- Content-pack domain schemas: full field-level schemas for locations,
  encounters, statblocks, traps, loot, tactical maps, fronts, and assets still
  need to be specified before broad implementation.
- Review tooling: the accepted UI/CLI workflow for marking manual records
  reviewed, approved, blocked, or rejected is not defined here.
- Shareable review exports: private review exports may include raw/protected
  material only under ignored storage. A redacted shareable export format needs
  its own schema and deny-list tests.
- Pack update semantics: how a running checkpoint reconciles a pack version or
  content hash change is not decided. Until then, mismatches are hard failures.
- Public/open content packs: this decision targets protected/private module
  packs. Public-pack shortcuts, direct CDN payloads, or relaxed source
  boundaries require a separate mode decision.
- Asset byte resolver integration: the safe payload contract exists, but exact
  resolver configuration, byte limits, cache layout, and Discord/CLI transport
  wiring remain implementation work.
- D&D adapter import completeness gates: the minimum complete set of D&D
  statblock, spell, trap, encounter, loot, and tactical-map fields required for
  runtime-ready rows still needs domain-schema acceptance.
