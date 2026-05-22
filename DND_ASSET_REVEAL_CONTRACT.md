# D&D Asset Reveal Authority And Event Contract

Status: accepted decision for `ayoa-d57`. This document defines the contract only;
it does not implement schema or runtime changes.

## Current Grounding

The generic event contract is intentionally small. `CanonicalEvent` contains only
`world_adjudication` and `observable_facts`, and each `ObservableFact` owns text,
timing, and fact-level visibility. `EventRouterOutput` keeps the event id,
timing, event kind, observers, routing roles, and side-effect signals as siblings
around that canonical event. `DndEventRouterOutput` extends the same shape only
for D&D interaction mode, combatants, loot, and battle-map seed data.

Runtime delivery is event-id based. `turn_loop.broadcast_event()` appends the
closed router output to `ckpt.canonical_events`, queues `RenderBufferEntry`
records for human observers, and pushes only visible fact text into NPC
`pending_observations`. Narrator composition resolves buffered event ids against
`ckpt.canonical_events` and formats only POV-visible fact text. Character agents
also receive text observations only.

Content-pack state is already a generic adapter slice on `SessionState`:
`content_state: dict[str, ContentPackState]`. Content lookup is a pre-router
history append path; when no content pack is active, lookup is a no-op and no
current-turn content packet is sent. Existing tests assert that content history
stays assistant-side and out of the current user packet. Content lookup failures
raise before the router call instead of asking the model to improvise missing
module material.

The asset scaffold already exists but is not wired into events or responses.
`ContentImageAsset` represents reviewed private catalog rows, `AssetReveal`
represents a router-owned reveal request with observable-fact-style visibility,
`SafeAssetRevealPayload` is a player/LLM-safe payload shape, and
`safe_asset_reveals_for_viewer()` filters reveal requests against player-safe
catalog rows. `TurnResponse` currently carries per-POV prose, reaction prompts,
loot prompts, dice rolls, XP awards, and pre-turn `TurnResponse` objects; it has
no asset field yet. `frontend_views.py` currently has no asset-specific DTO.

`DND_MODULE_IMPORT.md` is the upstream policy: adventure images are private table
assets, humans see images only after a router-owned reveal, runtime LLMs never
process image bytes or source page scans, and checkpoints store refs and reveal
state but not raw image bytes. The module privacy boundary has three surfaces:
the router may receive reviewed router-only asset candidates needed for
adjudication; narrator and character-agent prompts receive only POV-safe facts,
observations, and safe captions; output/log/export surfaces receive only
post-filter player-safe payloads and sanitized diagnostics. Metadata such as
filenames, ids, captions, alt text, OCR, paths, and map labels obey the strictest
boundary that can contain them.

## Decision

Asset reveal is a content-pack-enabled event sidecar. It is not a generic
`CanonicalEvent` field, not an `ObservableFact` escape hatch, and not narrator
prose with embedded asset refs.

For ordinary rules-neutral sessions with no active reveal-capable content pack,
the router output schema remains the current `EventRouterOutput` or
`DndEventRouterOutput`; no asset field, no asset prompt rules, and no empty
asset arrays are added. When a session has an active content pack that exposes
reviewed router-only reveal-capable asset candidates, the dispatcher may select a
content-enabled router output schema that extends the active router schema with
one required sibling field:

```text
asset_reveals: AssetReveal[]
```

That content-enabled schema must be selected dynamically for the call. It must
not expand the baseline generic schema or cached system prompt for sessions that
cannot reveal pack assets.

## Reveal Authority

The router authors reveal requests. A content lookup or resolver may make
reviewed hidden asset candidates available to the router as router-only
adjudication context, but those candidates are not observable facts and do not
reveal themselves to players. The narrator cannot authorize a reveal. Character
agents cannot authorize a reveal. Frontends cannot infer a reveal from prose.
D&D combat, maps, handouts, portraits, and item art must use this same generic
content reveal sidecar rather than a D&D-specific delivery channel.

Adapter-owned resolvers that produce canonical events, including D&D combat
manager paths, may participate only by returning the same content-enabled event
shape when content-pack reveal support is active. They must not embed hidden
image refs, source refs, filenames, map labels, or pack-private metadata in
observable facts, narrator prompts, agent prompts, or response text.

## Event Contract

`AssetReveal` is an event side effect owned by the same event id as the router
output that carries it:

```text
AssetReveal
  pack_id: string
  asset_id: string
  audience: all_observers | only
  visible_to_character_ids: string[]
  visible_to_user_ids: string[]
  presentation: inline | attachment | reference | map_overlay
  caption: string
```

The owning event id plus the reveal's array index is the stable reveal identity
for the first implementation. A separate `reveal_id` is not required unless a
future delivery system needs idempotent client acknowledgements independent of
the event log.

Visibility mirrors `ObservableFact` with one additional delivery restriction:

- `audience="all_observers"` means every player POV whose character is in the
  owning event's `observers` may receive the safe asset payload.
- `audience="only"` means only the listed character ids, and optionally their
  listed bound user ids, may receive it.
- Character ids listed on a reveal must be valid event observers unless the
  reveal is explicitly a host-only or out-of-band UI surface. A player-facing
  reveal must not create visibility for a character who did not observe the
  event.
- User ids narrow delivery to bound users; they are not an independent fiction
  visibility channel for player POVs.

Every reveal must have a corresponding player-perceivable surface in the same
canonical event. The image is display media, not the only record that something
happened. For example, the facts can say that a folded map is opened or a portrait
is uncovered. The asset sidecar attaches the reviewed image. If an NPC needs to
act on image content later, the router must canonicalize what that NPC perceived
as text in `observable_facts`; agents do not inspect image refs.

Missing, unreviewed, unsafe, or pack-mismatched assets are contract violations.
The event application path should fail loudly before saving the turn rather than
silently dropping a requested reveal or falling back to narrator prose.

## Persistence

The immutable reveal request persists with the canonical event that authored it.
Implementation should add content-enabled event subclasses to the checkpoint
event union rather than widening the baseline `CanonicalEvent`. A future saved
content event should be readable as:

```text
ContentEventRouterOutput(EventRouterOutput)
  asset_reveals: AssetReveal[]

DndContentEventRouterOutput(DndEventRouterOutput)
  asset_reveals: AssetReveal[]
```

The mutable play overlay in `SessionState.content_state[pack_id]` should also
persist accumulated reveal state for lookup, rewind, and map fog:

```text
RevealedAssetState
  pack_id
  asset_id
  source_event_id
  audience
  visible_to_character_ids
  visible_to_user_ids
  presentation
  caption
  revealed_at_s
```

The event sidecar is the authority for what was requested in story time. The
content-state reveal overlay is a projection of current campaign state and must
be rebuildable from the event log plus pack data where practical. Checkpoints
store ids, hashes, visibility state, captions, and fog/mask refs. They do not
store raw images, page scans, source paths, full OCR, DM notes, or protected
source excerpts.

Router history should preserve reveal continuity compactly only when content
packs are active. Router-only history may include hidden reviewed reveal state
needed for future adjudication, but the projection must remain authored text or
structured ids rather than source images, page scans, OCR dumps, raw paths, or
protected excerpts. A prior-event record may include a terse marker such as:

```text
asset reveal pack=<pack_id> asset=<asset_id> audience=only[alice] presentation=attachment caption="A folded note with a wax mark."
```

That history marker is for router continuity only. Narrator and agent prompt
builders must not receive delivery refs, source refs, local paths, hidden map
labels, unredacted filenames, raw OCR, asset catalog metadata, or router-only
asset candidates.

## Response And Frontend Contract

`TurnResponse` should gain player-safe asset payload fields shaped like the
existing per-POV render fields:

```text
asset_reveals: SafeAssetRevealPayload[]
per_player_asset_reveals: dict[character_id, SafeAssetRevealPayload[]]
```

`per_player_asset_reveals` is authoritative. `asset_reveals` mirrors the acting
POV's list for legacy single-POV callers, matching the relationship between
`output_text` and `per_player_renders`.

Each `SafeAssetRevealPayload` is produced only after the engine validates the
router request against the active content pack and then filters by POV. The
payload may include only player-safe fields:

```text
pack_id
asset_id
kind
title
mime_type
width
height
sha256
delivery_ref
presentation
caption
alt_text
```

No `source_ref`, provenance, source asset id, local path, raw OCR, DM notes, map
secret labels, hidden caption, or protected source text is allowed in the
response payload. `frontend_views.py` can either reuse `SafeAssetRevealPayload`
directly or introduce an isomorphic `AssetRevealView`; it must not add private
fields.

Frontends deliver assets from the response payload after per-POV filtering.
Discord must not fall back from failed private delivery to a public channel. CLI
may print the safe caption and safe delivery ref for local inspection, but must
not print private source paths or unrevealed filenames.

## Narrator And Agent Inputs

Narrator inputs may receive only visible facts and LLM-safe caption text for the
current POV. They never receive `asset_id`, `delivery_ref`, `source_ref`, local
paths, source metadata, raw image bytes, page scans, hidden map labels, DM notes,
or protected excerpts. If an asset's curated caption or alt text is not marked
safe for LLM use, the narrator sees only the ordinary `observable_facts`.

Character agents receive text observations only. The existing
`broadcast_event()` behavior should remain the model: push visible
`observable_facts` to NPC inboxes, not asset payloads. If the image content is
fictionally available to an NPC and relevant to later action, it belongs in a
text `ObservableFact` with normal visibility.

The narrator may mention a displayed asset only from the safe caption/text facts,
not by inventing asset provenance or by copying refs into prose. The asset itself
is attached by the frontend from `TurnResponse`, not embedded into the generated
narration.

## Pre-Turn Responses

Pre-turn resolutions keep their own asset reveal payloads. `_with_pre_turn_resolutions()`
should not flatten assets from pre-turn responses into the main response, because
story order and POV filtering matter.

Display order is:

1. render each `pre_turn_resolutions[]` `TurnResponse` in order, including its
   dice rolls, prose, prompts, and asset reveals
2. then render the main `TurnResponse`

If a pre-turn automated combat advance or stale Cat II closure reveals an asset
and invalidates the submitted player action, the asset remains attached to the
pre-turn response. The main "scene changed" rejection response should not
duplicate that asset.

## Canonical Events And Player-Visible Assets

Canonical events remain the textual truth of what happened and who perceived it.
Asset reveals are event-linked presentation side effects. A player-visible asset
must never be the only authoritative expression of a story fact. The same event
must contain safe visible facts that explain why the asset is now perceptible,
while the sidecar says which reviewed asset payload may be delivered to which
POVs.

This preserves the original rules-neutral narrative engine:

- non-content sessions pay no schema or prompt cost
- D&D assets use the same content reveal surface as non-D&D content
- narrator and agents stay text-bound and spoiler-filtered
- frontends receive only post-filter player-safe payloads
- checkpoints persist refs and reveal state, not private source artifacts

## Implementation Gates

Before implementation is accepted, tests should cover:

- no asset schema or asset prompt block when `content_state` has no reveal-capable
  active pack
- content-enabled router output validates reveal recipients against event
  observers
- missing, unreviewed, unsafe, or pack-mismatched requested assets raise before
  save
- narrator prompt contains safe caption/text only, never asset refs or source
  metadata
- agent prompts contain text observations only
- reviewed hidden asset candidates can be projected into router-only
  adjudication context without appearing in narrator prompts, agent prompts,
  player output, ordinary logs, or default checkpoint exports
- two player-bound characters can receive different asset payloads from the same
  event
- pre-turn asset reveals stay on their pre-turn `TurnResponse`
- response payloads never include source refs, local paths, raw OCR, DM notes,
  hidden labels, or protected excerpts

## Blocking Choices To Escalate

These choices should be bubbled up to `ayoa-50c` or `ayoa-57l` before schema
implementation:

- `SafeAssetRevealPayload` currently exposes `asset_id`, `title`, `sha256`, and
  `delivery_ref`. The pack compiler must guarantee these fields are
  player-safe, or the response contract needs an opaque `public_asset_ref`.
- `safe_asset_reveals_for_viewer()` currently lets `visible_to_user_ids` grant
  visibility independent of event observers. Event-bound player reveals should
  tighten this, or reserve user-only delivery for host/out-of-band UI.
- The implementation needs a concrete content-enabled schema selection point for
  both the generic router and D&D combat-manager paths, without widening the
  baseline router schema or prompt for sessions with no reveal-capable pack.
