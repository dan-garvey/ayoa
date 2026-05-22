# D&D Combat Start Source Precedence

Status: accepted decision for `ayoa-don`. This document decides source
precedence for module-backed D&D combat starts. It does not implement resolver,
schema, compiler, prompt, checkpoint, or UI changes.

## Current Grounding

This decision extends:

- `DND_CONTENT_PACK_AUTHORITY.md`
- `DND_TACTICAL_MAP_GEOMETRY.md`
- `DND_MANUAL_PACK_AUTHORING.md`
- `DND_CONTENT_MEMORY_INVARIANT.md`

The existing runtime already has a D&D-only router interaction mode,
`dnd_combat_start`, plus adapter-owned surfaces for `combatant_ids`,
`combatant_spawns`, `battle_map_seed`, active combat state, initiative,
combatants, XP, loot offers, and advisory battle-map state. Those surfaces are
D&D adapter contracts. They must not become generic router, narrator, or
character-agent machinery.

The content-pack decisions add a stricter rule for protected module content:
reviewed compiled pack rows are runtime authority. Raw source, OCR, draft JSON,
private review exports, and router invention are not fallbacks for missing
module encounter data.

## Decision

Compiled encounter templates provide authoritative mechanical facts, not a
deterministic script for the scene. Their job is to supply the router and D&D
combat resolver with reviewed participants, statblocks, map refs, hazards,
spawn anchors, and gates so the LLM can adjudicate from accurate table context.
The adapter should reject missing, unsafe, or unreviewed module authority
instead of asking the model to invent it, but once authority is present the
router/combat resolver remains the fun-first arbiter of how the combat starts
and what the fiction makes possible.

For module-backed combat, compiled encounter templates must resolve before a
router-authored `dnd_combat_start` is executable.

The exact runtime shape can evolve, but the contract is:

1. Deterministic D&D/content preflight identifies any active required encounter
   refs from the current location, visible trigger, pending content signal,
   front pressure, or already introduced encounter state.
2. Required encounter, statblock, map, spawn-anchor, trap, hazard, loot, and
   front refs are validated against the compiled pack and checkpoint overlay
   before the start can commit state.
3. The router may decide that the fiction has crossed into initiative and may
   author the visible triggering event.
4. The adapter materializes combat from the reviewed pack rows plus checkpoint
   overlay. It rejects router-authored replacements for reviewed module facts.

If the resolver cannot prove the required reviewed records before materializing
combat, runtime fails loudly with a sanitized operator error. It must not ask
the router to invent monsters, positions, statblocks, map geometry, loot, traps,
hazards, or front consequences for a reviewed module encounter.

This does not remove improvised combat. When no active pack/template claims
authority over the encounter, the D&D adapter may continue to accept
router-authored combat starts, inline spawned combatants, and advisory map seeds
under a clearly non-pack/improvised mode. Those starts must not masquerade as
reviewed module automation.

## Router Authorship Boundary

For module-backed starts, the router may author:

- the fiction-facing trigger that starts initiative
- `interaction_mode="dnd_combat_start"` when the visible situation calls for
  initiative
- visible observable facts and observers for the start event
- references to already introduced, currently visible participant ids when the
  choice is genuinely in-fiction, such as which present guards join immediately
- non-mechanical narrative framing, morale, threats, warnings, and apparent
  intent
- requests or side effects that use reviewed refs already introduced through the
  content resolver, such as "this encounter starts from `enc.test.entry`" once
  a future schema provides that field

For module-backed starts, reviewed pack records or checkpoint overlay must own:

- encounter identity and template hash
- required and optional participant specs
- monster/NPC statblock refs and mechanical payloads
- spawn anchors, fallback anchors, token sizes, and occupancy constraints
- hidden, delayed, or reinforcement participants
- strict battle-map template refs, geometry, fog, and player-safe asset refs
- linked loot, trap, hazard, lair, and environmental mechanics
- D&D-specific front/villain consequences and encounter-end policies
- review, gate, safety, and projection readiness

The router must not author inline statblocks, protected module substitutions,
secret participant identities, hidden map labels, unreviewed trap mechanics, or
fallback module rewards. If the router emits any such data for a module-backed
encounter, the adapter ignores or rejects it according to the field rules below;
it never treats that data as a pack-authoritative correction.

## Overall Precedence

When a combat start combines multiple sources, apply this precedence:

1. Checkpoint/session overlay for mutable state already created by play:
   defeated, removed, depleted, revealed, hidden, consumed, relocated,
   overridden, occupied, spawned, or front-progress state.
2. Reviewed compiled pack rows for immutable module definitions and typed D&D
   payloads.
3. Deterministic resolver projections from those rows into compact router
   context or adapter state.
4. Router-authored current-event fiction, only for the start trigger,
   visibility, observers, and in-fiction choices that do not replace reviewed
   mechanical data.
5. Improvised non-pack fallback, only when no reviewed encounter/template/ref
   claims authority and the runtime records the start as improvised.

Router history and compact content records are continuity projections, not a
second database. If a compact record is missing, stale, or hash-mismatched, the
resolver reintroduces it from the reviewed pack row or fails loudly.

## Field Precedence

### Encounter Template

A module-backed combat start must name or resolve exactly one active encounter
template before state commit. The template may come from current keyed area,
pending encounter signal, front escalation, explicit reviewed ref, or another
accepted resolver path.

If multiple templates can apply and no reviewed priority rule or router-visible
choice disambiguates them, the start is blocked. The router may choose among
already introduced, fictionally visible alternatives only when the compiled pack
marks that choice as allowed.

### Participants

Participant precedence is:

1. Checkpoint overlay state: already dead, fled, befriended, recruited,
   relocated, hidden, revealed, depleted, or previously spawned actors.
2. Encounter template participant specs and required/optional groups.
3. Current active session characters whose ids satisfy the template role,
   location, visibility, or trigger constraints.
4. Router-authored participant ids for visible, already existing characters
   when the template permits discretionary inclusion.

Required participants missing from the session and not spawnable from reviewed
pack records block the start. Router-authored extra participants not allowed by
the template are rejected for module-backed combat; they may still appear in the
visible fiction as observers or bystanders if the canonical event supports that.

### Statblocks

Statblock precedence is:

1. Character/session mechanics overlay for existing characters, including spent
   resources, HP, effects, inventory, and approved sheet updates.
2. Reviewed compiled statblock refs named by the encounter or participant spec.
3. Reviewed statblock variants or overrides explicitly linked by the pack and
   allowed by current overlay state.
4. Router inline statblocks only for improvised non-pack combat.

Missing AC, HP, initiative-relevant defenses, action data, save/DC data, damage
expressions, spell/resource data required by intended automation, or XP/CR when
the encounter-end policy needs it blocks module combat automation for that
participant. A router-generated statblock must not fill the gap.

### Spawn Anchors

Spawn-anchor precedence is:

1. Checkpoint tactical overlay: current token positions, occupied cells,
   opened/closed connectors, fog/reveal state, removed anchors, and prior
   placement mutations.
2. Reviewed encounter/map spawn anchors, including player starts, enemy starts,
   hidden starts, reinforcement starts, exits, retreats, and fallback starts.
3. Deterministic adapter placement within reviewed fallback rules.
4. Router/advisory placement only for improvised or non-strict combat.

If required anchors are missing, blocked, outside bounds, occupied with no
reviewed fallback, incompatible with token size, or gated for a projection other
than combat start, the start aborts. The adapter does not guess from room prose,
source map labels, image pixels, or router coordinates for module-backed starts.

### Hidden And Revealed Combatants

Hidden participant state belongs to the pack plus overlay, not the generic
router prompt.

The compiled encounter may define hidden, dormant, disguised, delayed,
reinforcement, fleeing, surrendered, or lair-controlled participants. The
checkpoint overlay records which are already revealed, defeated, relocated,
depleted, allied, or triggered.

The router and narrator receive only visible or sanitized compact facts. The
adapter may materialize hidden combatants privately when needed for initiative,
ambush timing, fog, traps, or lair actions, but it must not expose secret ids,
labels, token positions, statblock names, or map refs through player payloads,
normal logs, narrator prompts, or character-agent context before reveal.

If hidden participants are mechanically required but their reveal trigger,
spawn anchor, statblock, or visibility boundary is unreviewed or unsafe, the
module-backed start aborts. It does not downgrade to public monsters with
invented labels.

### Battle-Map Template Refs

Battle-map precedence is:

1. Active combat/checkpoint tactical overlay for mutable map state after combat
   has started.
2. Reviewed strict tactical-map template refs named by the encounter, keyed
   area, or front/hazard hook.
3. Reviewed player-safe asset refs and alignment metadata for display only.
4. Existing advisory `battle_map_seed` only for improvised or explicitly
   non-strict combat.

For module-backed strict maps, geometry is authority; images are display
artifacts. A router-authored `battle_map_seed` may not override reviewed
geometry, spawn anchors, blockers, hazards, fog, keyed-area links, or asset
refs. If strict map automation is requested but required gates are not ready,
that operation blocks. Combat may continue only through an explicit non-strict
mode whose status/logs say strict map automation is unavailable.

### Loot, Traps, Hazards, And Lair Hooks

Loot, trap, hazard, and lair hook precedence is:

1. Checkpoint overlay for depleted loot, sprung traps, disabled hazards,
   ongoing effects, revealed clues, claimed items, and prior saves/damage.
2. Reviewed compiled loot/trap/hazard/lair records linked by the encounter,
   map, keyed area, or front.
3. Deterministic D&D adapter trigger and effect handling for reviewed mechanics.
4. Router-authored narrative description of visible consequences.

The router may describe a visible trap springing or a hazard flaring when the
reviewed hook is active. It may not invent DCs, damage, reset state, treasure,
item mechanics, lair actions, or hidden trigger regions for module content.

If a required hook is missing, unreviewed, blocked, unsafe, hash-mismatched, or
missing mechanical fields needed for the requested automation, the automation
aborts before state commit. Optional hooks may be omitted only when the reviewed
encounter says they are optional and the omission is recorded.

### Front Consequences

Front consequence precedence is:

1. Checkpoint front overlay: current knowledge, clock progress, cooldowns,
   active plans, minion depletion, and prior introduced refs.
2. Reviewed front/villain dossier records and encounter-linked consequence
   policies.
3. Deterministic adapter/content resolver updates to front overlay.
4. Router-authored visible fiction about what the characters observe.

The router may author observable consequences such as an alarm heard, a messenger
fleeing, or a villain's visible response. It may not invent private villain
knowledge, change front clocks, spawn mechanically meaningful reinforcements,
or attach encounter-end consequences unless those outcomes are supported by
reviewed front records or existing overlay state.

## Loud Abort Rules

Module-backed combat start aborts before committing `session.active_combat`,
spawning characters, moving tokens, revealing assets, consuming resources, or
opening initiative when any required ref is not serviceable.

Hard aborts include:

- no active compiled pack for a required module encounter
- pack id, schema version, manifest hash, content hash, source fingerprint, or
  dependency mismatch
- missing encounter template, participant spec, statblock, map template, spawn
  anchor, loot, trap, hazard, lair, front, keyed-area, relation, or asset row
- row review status is `unreviewed`, `needs_review`, `blocked`, or `rejected`
  when runtime service requires `reviewed` or `approved`
- gate status is not `runtime_ready` for the requested projection
- strict geometry, fog, spawn, hazard, or asset-display gates are incomplete
  for the automation being requested
- required hidden participant or secret feature lacks a reviewed reveal boundary
- router output tries to replace reviewed module statblocks, map geometry,
  hidden refs, reward mechanics, trap/hazard mechanics, or front consequences
- safe player display requires an unsafe, unreviewed, missing, unaligned, or
  hash-mismatched asset
- the only available source is raw PDF/OCR, private review notes, draft JSON,
  source map pixels, protected prose, or router invention

The failure surface must be non-spoiling for players and actionable for the
operator. It should name sanitized refs and gate reasons, not protected labels
or source excerpts. It must not silently skip required side effects, create
synthetic substitutes, or save partial combat state as if the reviewed start
succeeded.

## Prompt And Engine Boundary

This decision does not widen generic router or narrator prompt weight.

The generic engine may resolve content refs, validate hashes/gates, persist
overlay state, and append compact allowed records. D&D-specific encounter
templates, statblocks, spawn anchors, strict maps, loot/traps/hazards, and
front mechanics stay in D&D adapter schemas and resolver code. The narrator
renders visible canonical facts and safe asset captions; it does not inspect
private encounter templates or adjudicate D&D mechanics.

Router prompt changes, if any, should be limited to the D&D router extension and
only describe field semantics. Do not explain pack databases, compilers,
orchestrators, dispatchers, private source files, or implementation plumbing to
the model.

## Implementation Blockers

Implementation should remain blocked until these pieces are designed or built:

- field-level compiled encounter template schema with participant specs,
  statblock refs, spawn-anchor refs, optional/reinforcement groups, and
  encounter-end policy
- adapter resolver that preflights module-backed combat starts before
  `session.active_combat` mutation and can roll back failed materialization
- checkpoint overlay fields for encounter refs, template hashes, spawned refs,
  hidden/revealed combatants, consumed/depleted hooks, map template refs, and
  front consequences
- authoritative statblock resolver that refuses router inline statblocks for
  module-backed encounters
- strict map template schema integration with spawn anchors, fog/reveal,
  keyed-area links, player-safe map assets, and advisory-mode downgrade status
- hook schemas for loot, traps, hazards, lair/environment actions, XP/rewards,
  and encounter-linked front consequences
- non-spoiling loud-failure surfaces for CLI, Discord, logs, playtest reports,
  and persisted checkpoint recovery
- tests for module-backed start precedence, missing/unreviewed refs, unsafe
  asset refs, hidden participant filtering, spawn-anchor failure, and router
  inline-data rejection

Until those blockers are cleared, current router-authored starts remain
appropriate for improvised D&D play and synthetic tests, but they are not a
complete module-backed combat-start authority model.
