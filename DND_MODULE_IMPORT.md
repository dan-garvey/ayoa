# D&D Module Content Packs And Checkpoint Application

Status: current module-import contract for `ayoa-3bm`.

This document defines the current shape of D&D module import in Ayoa. A D&D
module is not a story checkpoint and is not a prompt dump. It is modular
importable content that can be applied to a checkpoint, or used to create a new
checkpoint seed.

The rules-neutral engine remains router-centered. D&D mechanics and module
records are adapter/content-pack layers around that engine.

## Core Model

A module import has three durable layers.

### Content Pack Authority

The content pack is the reviewed module authority. It contains typed, reviewed,
runtime-gated records such as:

- sections, locations, keyed areas, exits, clues, reveal edges, and cross refs
- actors, factions, front dossiers, agent context slices, and knowledge graph
  edges
- handouts, tables, tactical map templates, assets, hazards, traps, treasures,
  encounter templates, and statblocks
- provenance ids, review status, gate status, confidence, content hashes, and
  coverage notes

The runtime may read reviewed compiled rows. It must not read raw PDFs, page
images, OCR dumps, source paths, screenshots, private review notes, or protected
source excerpts.

### Projection Profiles

A projection profile is an import-authored application plan over the content
pack. It decides which semantic records matter for a specific use case and
produces runtime-ready packets:

- router startup and lookup catalog packets
- character-agent initial context
- initial engine-owned knowledge map
- active fronts and pressure state
- checkpoint seed text
- optional in-medias-res field-start state
- private engine/rules-adapter overlay inputs

Projection profiles preserve refs and hashes back to reviewed pack records, but
ordinary router lookup catalog prompts should receive compact ref/title/summary
entries, not hashes unless the hash is needed for the runtime contract.

Profiles are authored at import time. The engine should not rediscover semantic
slice boundaries from raw domain records every session.

### Checkpoint Application

Checkpoint application is deterministic code that consumes one projection
profile and either:

- creates a fresh `CheckpointFile` seed, or
- adds the module to an existing checkpoint in a bounded, conflict-aware way.

Application may add `session.content_state`, active or dormant imported
characters, router pending signals, active fronts, knowledge-map entries, D&D
adapter overlay data, and narrow world/character updates. It must not silently
rewrite unrelated campaign state.

## Two Consumers

### Create A Seed From A Module

Seed creation is allowed to be opinionated and complete. It creates
`app/storage/stories/<story_id>/ckpt_0000.json` from a chosen projection
profile.

The generated seed should define:

- `player_primer`
- common `world_state.facts`, `world_state.lore`, setting, tone, and physics
- router-only hidden lore/facts when needed
- playable characters and initial bindings policy
- active and dormant imported NPC roster
- per-character goals, objectives, secrets, known context, and mechanics
- `session.content_state` with pack identity, pending router signals,
  knowledge map, fronts, overlays, and private D&D adapter data
- clean empty histories unless the seed intentionally starts after authored
  canonical events

The opening beat is still produced by normal runtime routing and narration. The
seed should provide the state needed for that first turn, not an authored prose
scene that bypasses the engine.

### Apply A Module To An Existing Checkpoint

Module application is additive by default. It should assume the existing
checkpoint owns:

- player characters and bindings
- current premise, location, and canonical history
- existing world facts/lore and hidden facts
- current combat, inventory, commitments, render buffers, and content packs

An application profile may introduce a hook, location, route, site, front,
handout, NPC, or encounter cluster, but it should do so through explicit
profile fields and deterministic validation.

Valid application examples:

- a patron offers the module hook
- the party finds a route map or handout
- the party reaches a module region
- a site is inserted as a reachable location
- a front becomes active because public facts triggered it
- a later profile activates a villain, encounter set, or treasure/hazard domain

Application should fail loudly when a profile expects missing reviewed records,
stale hashes, blocked records, impossible character ids, incompatible ruleset
state, or conflicting active content that the profile cannot merge safely.

## Checkpoint Expectations

A seed checkpoint is the generic story runtime object. It is not module-specific
storage. Today the minimum hard contract is Pydantic validation plus
`schema_version`, but imported module seeds should satisfy stronger authoring
expectations:

- current `schema_version`
- empty or intentionally authored histories
- no raw source text, source paths, private extraction paths, or protected
  excerpts
- world/player/hidden knowledge separated deliberately
- active characters have enough context to act faithfully
- dormant imported characters are available through content state or roster
  state without being advertised as immediate turn candidates
- router startup signals cover the opening scene
- content manager knowledge map starts from reviewed projection data
- D&D adapter payloads are private runtime metadata and remain optional outside
  D&D rulesets

## Runtime Lookup

Runtime lookup is bounded and validated:

- deterministic pending signals and alias/current-location matches are drained
  first
- optional content-manager or router lookup can request exact refs
- code validates pack id, ref existence, pack identity, review status, gate
  status, content hash, and projection allowance
- compact records are appended to router history before the normal router call
- missing required content raises loudly instead of asking the router to invent
  module facts

The router receives hidden/reviewed module context only as adjudication
authority. Narrator prompts, character-agent prompts, player output, ordinary
logs, Beads, and default checkpoint exports receive only safe projections.

## D&D Adapter Boundary

D&D-specific mechanics remain adapter-owned:

- statblocks and monster spawns
- encounter templates and combat-start resolution
- traps, hazards, treasure, loot offers, XP, and inventory mutations
- tactical map runtime state and player-safe map/status projections
- dice, roll transactions, HP, resources, conditions, and action-source data

The generic engine owns refs, content state, canonical events, routing,
visibility, checkpoint persistence, and prompt boundaries. It must not add D&D
prompt or runtime cost to non-D&D stories.

## Retired Non-Contracts

These are not valid module-import surfaces:

- page-proxy catalogs as the semantic import target
- raw OCR or PDF RAG at runtime
- module-specific bespoke checkpoint assembly scripts as the runtime contract
- `/story import` or an LLM story importer that turns source material directly
  into a checkpoint
- tests that hand-build imported module metadata when they are meant to exercise
  module application
- full module text in `world_state.lore`, character prompts, router prompts,
  checkpoints, logs, Beads, or tracked docs

Manual review and private extraction are allowed only as authoring inputs. The
runtime contract is reviewed pack records plus projection/application profiles.

## Current Implementation Notes

The current reusable projection layer is:

- `app/schemas/content_projection.py`
- `app/engine/content_pack_projections.py`

`ContentPackProjectionArtifact` currently covers seed/startup and field-start
profiles. It can create `ContentPackState`, character records, checkpoint seed
text, active fronts, knowledge map state, and private D&D overlay metadata.

Lost Laboratory is the first local private module using this shape. Its CLI demo
loads a projection artifact instead of reconstructing actor/context/graph slices
from the raw domain catalog.

Still-open design/implementation work belongs under `ayoa-3bm`:

- first-class named application profiles beyond startup/field-start
- an explicit API for applying a module profile to an existing checkpoint
- validation for profile conflicts against existing checkpoint state
- a seed-quality validator for generated `ckpt_0000.json`
- pack locator resolution that avoids shareable checkpoints depending on local
  paths
