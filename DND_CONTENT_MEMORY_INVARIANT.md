# D&D Content Memory And Compaction Invariant

Status: accepted decision for `ayoa-6j7`. This document defines the runtime
invariant for introduced content refs across router-history compaction,
checkpoint reload, rewind, changed pack hashes, missing assistant history, and
model cache behavior. It does not implement storage, resolver, schema, prompt,
or test changes.

This decision extends:

- `DND_ADVENTURE_LOOKUP_PLAN.md`
- `DND_CONTENT_PACK_AUTHORITY.md`
- `DND_MODULE_PRIVACY_BOUNDARY.md`
- `DND_ASSET_REVEAL_CONTRACT.md`
- `DND_MANUAL_PACK_AUTHORING.md`

The core constraint remains unchanged: content packs are generic private
runtime data, while D&D mechanics remain adapter-owned. No baseline
rules-neutral story pays prompt, schema, or runtime cost for D&D module lookup.

## Definitions

`introduced_refs` is the checkpoint-durable ledger of content refs that have
already been made available to the event router for this session. It records
refs, hashes, labels, kinds, visibility, source event ids, and introduction
positions. It is not itself model context and is not proof that a restored model
conversation still contains the compact record.

`effective router context` for a router call is the complete message context the
router actually receives for that call:

- stable cached system prompt and story-level rules
- restored compact assistant-side router history, including `prior_event`,
  `content_known`, `location_card`, and `front_signal` records
- current-turn assistant-side content records appended by lookup or
  reintroduction before the router call
- current actor input and ordinary per-turn user-tail state

The effective router context does not include the private pack database,
checkpoint overlay objects, raw source artifacts, local paths, hidden asset
catalog metadata, or model-provider cache internals unless the runtime projects
them into an allowed compact record.

`required content refs` for a router call are the refs the router may rely on
for correct adjudication now. The set is produced by deterministic preflight and
future bounded lookup work from the current actor, location, pending signals,
active front pressure, active/recent reveals, active encounters, unresolved
commitments, and current player input. It is intentionally smaller than "every
ref ever visited."

## Invariant

Before any content-enabled router call, every required content ref must be
covered by exactly one current compact record in the effective router context,
or the runtime must fail loudly before the router call.

A compact record covers a required ref only when all of these match:

- `pack_id`
- `ref_id` or asset/front/location ref
- record kind or projection kind
- router visibility/scope
- content hash for the served pack row
- projection version or compact-record formatter version, once that exists

`introduced_refs` and effective router context are related by this rule:

1. If an `introduced_refs` entry is still present in effective router context
   with matching hashes, the resolver must not repeat it.
2. If an `introduced_refs` entry is required now but its compact record is absent
   after compaction, checkpoint reload, cache loss, or history repair, the
   resolver must reintroduce a compact record from the reviewed pack row before
   the router call.
3. If an `introduced_refs` entry cannot be reintroduced from a reviewed,
   runtime-ready pack row whose hash matches the checkpoint expectation, the
   runtime must fail loudly.
4. If a ref is not required now, absence from effective router context is
   allowed even when the checkpoint overlay still records visited, consumed,
   revealed, or introduced state for that ref.

This means `introduced_refs` is a deduplication and recovery ledger. It prevents
giant content packets every turn, but it also creates a proof obligation: when a
ref matters to the next router call, the compact fact must either still be in
the actual message context or be reintroduced from authoritative pack data.

## Checkpoint Overlay Requirements

The checkpoint overlay must store every content fact needed to recover state
without trusting process memory, provider cache, or old assistant history:

- active pack identity: `pack_id`, pack version, schema version, source
  fingerprint, manifest/build hash, dependency hashes, and logical locator key
- introduced refs: `pack_id`, `ref_id`, content hash, label, kind, visibility,
  compact projection kind, source event id, introduced turn/index, and
  supersession state if a future migration path allows it
- pending content signals: signal id, pack id, target ref, expected content
  hash when known, reason code, source event id, priority, requested fields,
  created turn/index, and status
- campaign overlay: visited refs, discovered clues, consumed/depleted refs,
  spawned refs, overridden refs, table-specific notes, location/front progress,
  and unresolved content-linked commitments
- front overlay: front ids, villain/minion ids, current knowledge, clocks,
  active plans, cooldowns, queued pressure, and refs already introduced for
  that front
- asset reveal overlay: owning event id, pack id, asset id, audience, visible
  character/user ids, presentation, safe caption, safe derivative id/hash,
  fog/mask/crop refs, and reveal time
- adapter state created from content: active combat, combatants, encounter
  refs, statblock refs, loot/trap/hazard refs, map/template refs, HP/resources,
  effects, XP awards, and roll transactions

The overlay stores ids, hashes, compact labels, state, and projections. It must
not store raw source pages, OCR, protected excerpts, full private card bodies,
absolute local paths, raw asset paths, hidden map labels, or delivery refs that
are not approved safe logical refs.

## Facts Allowed Only In Router History

Some facts may live only in compact router history because they are model-facing
continuity, not durable game state:

- the exact compact record string that was appended for a reviewed pack row,
  when the checkpoint stores enough ids and hashes to reconstruct it
- prior-event pacing and responder context that is already represented by
  canonical events plus compact router history
- one-turn lookup rationale, alias-match diagnostics, and why a deterministic
  preflight chose a record
- current-turn candidate catalogs or lookup prompts that were used only to ask
  for missing refs
- model cache read/write metadata and token accounting

These facts cannot be the only source of mutable campaign truth. If losing the
assistant history would make the runtime forget a visited room, depleted loot,
revealed asset, front clock, villain knowledge, active encounter, spawned NPC,
or D&D mechanics state, that fact belongs in the checkpoint overlay.

## Reintroduction After Compaction Or Reload

Router-history compaction may remove raw user messages, full router JSON,
empty fields, rationale boilerplate, and compact records for refs that are no
longer required. It must not leave a required ref in a state where the router is
expected to remember it but no covering compact record exists.

After compaction, checkpoint reload, missing assistant history, prompt-version
repair, or provider cache loss, the resolver must build the required-ref set
before the router call and compare it to the effective router context. It must
append only missing required compact records.

Required reintroduction includes:

- current location and adjacent/topology refs needed for the submitted action
- unresolved pending content signals
- active front or villain signals whose knowledge or pressure may affect this
  beat
- active encounter, trap, loot, statblock, tactical-map, or hazard refs needed
  by a D&D adapter path
- asset reveal continuity that the router must reason about, such as a revealed
  handout, visible map region, or fog/mask state affecting the current action
- any off-path refs returned by the bounded lookup preflight before the final
  normal router call

Reintroduction must not become "send the whole module again." Historical refs
that are only needed for audit, rewind, or distant visited-state bookkeeping
stay in the overlay until a current action, signal, front, or lookup makes them
required again.

## Pack And Content Hash Changes

The active checkpoint names the pack identity and hashes it was using. The v1
runtime rule is conservative:

- If the active pack id, schema version, source fingerprint, manifest/build
  hash, or dependency hash does not match the checkpoint expectation, fail
  loudly before lookup or router calls.
- If a required row is missing, unreviewed, blocked, unsafe for the requested
  projection, or has a content hash that does not match the checkpoint's
  expected hash for an introduced or state-bound ref, fail loudly.
- If a ref has never been introduced and no overlay state depends on it, the
  resolver may introduce the current reviewed row normally under its current
  hash, provided the active pack identity itself matches.
- If a future explicit pack-update workflow proves that a changed row is not
  state-bound and is safe to supersede, that workflow may append a new compact
  record and update `introduced_refs` as a superseding introduction. That is a
  separate implementation decision, not implied by this document.

Reintroduction solves missing effective context. It does not solve pack drift.
A changed row that could reinterpret committed canonical events, revealed
assets, front knowledge, spawned refs, consumed/depleted refs, combat state,
loot, traps, hazards, or D&D mechanics is a migration problem and must fail
until an operator-reviewed migration accepts the new meaning.

## Rewind

Rewind restores the checkpoint target exactly. Content memory is not monotonic
across rewind.

After rewind:

- `introduced_refs` contains only refs introduced at or before the target
  checkpoint.
- pending content signals are exactly the target checkpoint's pending signals;
  signals created by later turns disappear.
- front and villain knowledge, clocks, active plans, cooldowns, and queued
  pressure are restored to the target checkpoint state.
- asset reveal state is restored to the target checkpoint state; reveals after
  the target are not deliverable, while reveals at or before the target remain
  deliverable only if the pack and safe derivative hashes still validate.
- map fog, crop masks, handout reveal state, tactical-map state, consumed loot,
  spawned refs, defeated/depleted refs, and active encounter state are restored
  from the target overlay.
- router history after the target checkpoint is discarded. The next router call
  must rely on the restored compact history plus reintroduced required refs,
  not on later assistant messages, provider cache, or later asset delivery
  state.

If rewind lands before a content ref was introduced, the router must not retain
knowledge of that ref. If rewind lands after a ref was introduced but compaction
or missing assistant history removed the compact record, the resolver may
reintroduce it only when it is required for the next call and the pack hashes
validate.

## Asset Reveal Continuity

Asset reveal continuity follows the event sidecar and overlay decision in
`DND_ASSET_REVEAL_CONTRACT.md`.

The owning canonical event remains the story-time authority for a reveal. The
content overlay records current reveal state for recovery, lookup, rewind, map
fog, crops, and per-POV delivery. Router history may carry a terse compact
asset marker only for continuity; it is not a delivery payload.

On reload or rewind, safe asset payloads must be recomputed or validated from
the active compiled pack and overlay. A missing, unsafe, unreviewed, or
hash-mismatched asset is a loud failure for delivery or for any router call that
requires that reveal as context. The runtime must not replace it with narrator
prose, a source path, a raw image, an unreviewed derivative, or a public-channel
fallback.

## Model Cache Behavior

Provider cache behavior is an optimization only. Cache reads, cache writes, and
missing cached prefixes are not durable content memory.

Stable prompt rules and story-level context may live in cached system prefixes.
Turn-specific content records, current actor state, pending signals, lookup
results, active front pressure, asset reveal deltas, and newly required refs
belong after the template boundary in the per-call message context or compact
assistant history.

The resolver must avoid giant cache-write packets every turn:

- append a compact record only when a new ref is introduced, a required ref must
  be reintroduced because effective history lost it, or a future accepted
  migration supersedes a prior ref
- do not resend every introduced ref on every turn
- do not send full private card bodies, source excerpts, raw aliases, or asset
  metadata to preserve cache locality
- let the router request off-path content through bounded refs/lookup when
  player choices leave the deterministic path

The testable target is delta-based content memory: repeated turns without new
content should not grow a content packet merely because content packs are active.

## Test Proof Without Prompt-Prose Freezes

Implementation should prove this invariant with runtime-contract tests, not
tests that freeze approved prompt prose.

Required test shapes:

- Synthetic pack reload: load a checkpoint with `introduced_refs`, strip or
  truncate assistant history, run the resolver preflight, and assert only the
  required compact records are reintroduced before the router call.
- Compaction coverage: compact router history and assert required refs still
  have matching `content_known`, `location_card`, `front_signal`, or asset
  continuity records, or that the resolver queues reintroduction before the
  next call.
- No giant packet: run multiple turns with no new required content and assert
  the current user packet does not receive a repeated list of all introduced
  refs.
- Hash mismatch rollback: use a synthetic pack with a changed required row and
  assert the runtime fails before the hosted router call and before checkpoint
  mutation.
- Non-required history drop: compact away an old non-required ref and assert it
  is not reintroduced until a current action, pending signal, front, reveal, or
  lookup requires it.
- Rewind restore: create content signals, front changes, introduced refs, and
  asset reveals after a checkpoint; rewind to the prior checkpoint and assert
  those later refs/signals/reveals are absent.
- Rewind continuity: rewind to a checkpoint after an asset reveal and assert
  safe asset payloads are recomputed or validated with the same POV filtering
  and hash checks.
- Prompt hygiene: scan rendered router, narrator, character-agent, and D&D
  adapter prompts for sentinel source paths, raw OCR, protected excerpts, DM
  labels, unsafe delivery refs, private asset metadata, and full card bodies.
- Router lookup preflight: drive an off-path player choice that requires a
  missing ref and assert the lookup fetches that ref, appends a compact record,
  and then calls the normal router once with the ref covered.
- Non-content regression: verify sessions with no active content pack preserve
  the existing prompt shape and runtime behavior.

The assertions should inspect rendered message placement, structured state,
compact-record allowlists, router-call boundaries, rollback behavior, and
checkpoint contents. They should not assert that a particular prose sentence
remains in `app/prompts/*.txt`.

## Blocking Choices

These choices remain open and should block implementation slices that depend on
them:

- Exact schema names for content memory fields, projection hashes,
  supersession markers, and compact-history coverage indexes.
- Pack registry and local locator design for resolving active packs without
  storing absolute paths in portable checkpoints.
- Pack update and migration workflow for accepting changed manifest or content
  hashes after a session has already used earlier rows.
- The exact compaction algorithm for deciding which historical non-required
  refs can be dropped versus retained as compact router continuity.
- Whether compact record formatter versioning should be global, per projection
  kind, or derived from schema version.
- How much asset reveal state can be rebuilt from canonical events alone versus
  stored directly in overlay for map fog, masks, crops, and delivery continuity.
- The router lookup preflight schema for requested refs, urgency, spoiler
  boundary, and max-retry behavior.
