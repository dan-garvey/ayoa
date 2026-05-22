# D&D Tactical Map Strictness And Geometry Model

Status: accepted decision for `ayoa-75w`.

This document decides how imported D&D combat maps become runtime geometry. It
does not implement schema, compiler, prompt, or rendering changes.

## Current Grounding

The existing D&D combat map is adapter-owned and intentionally light. Current
`DndBattleMapState` stores a rectangular grid, tokens, rectangular terrain
zones, area templates, and notes. `app/engine/dnd_spatial.py` normalizes
router-seeded maps, computes advisory distance, line of sight, cover, and area
targeting context, applies router-authored spatial deltas, and renders compact
status lines. `DESIGN.md` explicitly describes this v1 surface as advisory:
code persists and summarizes map state while the D&D combat manager decides
legality and outcome.

The content-pack decisions add stricter authority boundaries:

- runtime content comes only from reviewed compiled pack rows
- the router may receive reviewed adapter-private geometry and secret map
  records as router-only adjudication context
- narrator prompts, character-agent prompts, player output, ordinary logs, and
  default checkpoint exports receive only POV-safe projections
- tactical map templates are D&D adapter payloads, not generic engine fields
- images are private assets until safely revealed through the asset sidecar
- checkpoints store refs, hashes, reveal/fog state, and adapter state, not raw
  source pages, images, OCR, or local paths
- missing, unsafe, unreviewed, or blocked pack rows fail loudly before the
  runtime asks a model to improvise the missing authority

## Decision

Imported module combat maps are strict only when their reviewed geometry is
runtime-ready for the requested automation. They are not strict because an image
exists, because a source map was extracted, because a note mentions a room, or
because the router can describe likely positions.

There are two accepted tactical-map modes:

1. Advisory map state: the current router-seeded or manually adjusted
   `DndBattleMapState` surface. It may summarize positions, rough terrain,
   visible area templates, distances, line of sight, and cover to the D&D
   combat manager and status views. It must not be treated as authoritative for
   strict movement, pathing, fog, keyed-area entry, hazards, or player-safe map
   rendering.
2. Strict imported map geometry: a reviewed D&D adapter geometry payload from
   the compiled content pack. It may drive mechanical automation only for the
   layers whose review gates are `runtime_ready`. Geometry-dependent behavior
   outside those ready layers is blocked, not approximated.

The generic narrative engine remains rules-neutral. It may store and resolve
content-pack refs, persist checkpoint overlay state, and deliver player-safe
asset payloads. D&D movement, cover, line of sight, difficult terrain, hazards,
spawn anchors, and tactical fog are D&D adapter/domain-schema concerns.

Engine representations are never images. A map image, crop, or overlay is a
player display artifact addressed through safe asset refs. The authoritative
runtime map is typed geometry plus adapter state. If the geometry and image
asset disagree, strict automation follows reviewed geometry and player display
is blocked until the image alignment is reviewed.

## Geometry Representation

A full imported combat map template is a D&D adapter payload, for example a
future `DndTacticalMapTemplate`, stored in the compiled content pack and
materialized into `DndCombatState` when combat starts. The minimum reviewed
shape is:

- `template_id`, `pack_id`, schema version, content hash, review status, gate
  status, confidence, provenance ids, and gate reasons
- logical map refs: location/keyed-area refs, encounter refs, safe map asset
  refs, and optional player-display derivative refs
- one or more map planes for floors, decks, rooftops, balconies, exterior
  approaches, disconnected submaps, or inset tactical regions
- a coordinate system and scale for every plane
- geometry layers for cells, polygons, blockers, terrain, cover, elevation,
  doors/windows, hazards, spawn anchors, fog/reveal regions, secret features,
  keyed-area links, and asset alignment

### Coordinate System And Scale

Coordinates are adapter tactical coordinates, not pixels.

- Each plane has a stable `plane_id`, `width`, `height`, `square_size_ft`, and
  optional `z_index` or elevation origin.
- Cell coordinates are integer grid cells `(x, y)` with origin at the
  top-left of the authored tactical plane. `x` increases east/right and `y`
  increases south/down for display alignment. A token occupying one square at
  `(3, 4)` occupies the cell from `(3, 4)` to `(4, 5)`.
- Polygon vertices are expressed in grid units on the same plane. Cell centers
  are `(x + 0.5, y + 0.5)`. Polygons may define nonrectangular rooms, partial
  obstructions, reveal masks, hazards, and area boundaries without falling back
  to image pixels.
- `square_size_ft` defaults to 5 only when the reviewed map scale says it is a
  standard 5-foot grid. Nonstandard scales must be explicit. Unknown scale
  blocks strict automation.
- Image asset alignment, if present, is a separate transform from tactical
  coordinates to display pixels. It is for rendering overlays only and is not
  the source of geometry.

Square-grid D&D is the first supported strict mode. Hex grids, gridless maps,
and theater-of-the-mind sketches require their own accepted schema before they
can be strict.

### Floors And Submaps

Multi-level maps are modeled as separate planes connected by reviewed vertical
links. A plane may be a floor, balcony, roof, pit bottom, ledge, bridge, or
separate inset map. Disconnected rooms on one source image may be separate
planes when no continuous path exists between them.

Each vertical or inter-plane link needs:

- source and destination plane ids
- endpoint cells or polygons
- link kind, such as stairs, ladder, shaft, pit, trapdoor, elevator, climb,
  bridge, teleport, or one-way drop
- movement cost or required movement mode
- visibility and line-of-effect implications
- reveal state if the link is secret or initially hidden

The adapter must not infer floor transitions from image layout, labels, or
prose during combat.

### Cells And Polygons

Every strict plane has an explicit walkability model. A cell can be passable,
impassable, void, blocked until opened, occupied by static obstruction, or
special-purpose. Polygons may refine or override cell-level features when a map
has diagonal walls, curved rooms, large furniture, ledges, water, or partial
areas.

The cell layer must be precise enough for token footprints and path validation.
The polygon layer must be precise enough for nonrectangular blockers, area
templates, fog/reveal masks, and keyed-area boundaries. A rectangular bounding
box around a room is not enough for strict automation when walls, doors, cover,
or hazards inside that box affect play.

### Blockers

Blockers are typed geometry, not notes.

At minimum, strict geometry distinguishes:

- movement blockers: walls, locked barriers, furniture too large to cross,
  cliff edges, pits without a traversal link, closed portcullises, and solid
  doors
- line-of-sight blockers: walls, opaque doors, heavy smoke, darkness only when
  the adapter has a lighting/vision decision, curtains, large pillars, and
  terrain that blocks sight but not movement
- line-of-effect blockers: solid barriers that prevent spells, ranged attacks,
  and area effects even when a creature can see through or around them
- token occupancy blockers: features that reduce usable cells for creature
  footprints without necessarily blocking all sight

Blockers may be edge-based, cell-based, or polygon-based. Review must state
which layer owns each blocker so pathing and line tests do not double-count or
miss the same feature.

### Difficult Terrain

Difficult terrain is a geometry layer with movement-cost semantics. It records
affected cells or polygons, movement multiplier or extra cost, applicable
movement modes, duration or depletion state if mutable, and whether it is known
to players.

The D&D adapter may enforce movement cost only when the terrain, token speed,
movement mode, and path are all available in reviewed adapter state. Otherwise
it reports the terrain as context and blocks strict movement-budget automation.

### Cover

Cover is derived from reviewed geometry where possible and authored as an
override where D&D table judgment requires it. The strict representation must
support `none`, `half`, `three_quarters`, and `total`, scoped by source/target
line, obstacle polygon or edge, and elevation when relevant.

The adapter should compute and report cover from strict geometry. It may enforce
target invalidity for total cover when line of effect is blocked. It reports
half and three-quarters cover to the D&D combat manager or dice pipeline; the
existing rules path still owns attack adjudication and AC/bonus application
until a separate damage/attack pipeline decision moves more of that math into
code.

### Elevation And Vertical Links

Elevation is adapter geometry, not prose. A strict map records plane elevation,
cell or polygon elevation deltas, ledges, pits, ceilings, flying/climbing
clearance if known, and vertical links. It also records which vertical changes
block movement, require climbing or falling, provide cover, change range, or
change line of sight.

If elevation affects an action and the elevation layer is missing or flagged,
the adapter may report the ambiguity but must not perform strict range, fall,
cover, or path automation from guesses.

### Doors And Windows

Doors and windows are stateful connectors. They need geometry, open/closed/
locked/barred/broken state, visibility, reveal status if hidden, passability,
line-of-sight behavior, line-of-effect behavior, interaction affordances, and
links to keyed areas when they divide rooms.

The adapter enforces door/window state only from reviewed connector geometry
and checkpoint overlay state. A door drawn on a player image but absent from
geometry is not a runtime door.

### Hazards

Hazards that matter to combat automation need stable hazard refs and geometry:
trigger cells or polygons, save/attack/DC metadata when D&D-specific, damage or
condition payloads, recurrence timing, reset/depletion state, reveal state, and
links to source trap/hazard records.

The adapter may enforce entering, starting turn in, ending turn in, crossing,
forced-movement, and area-overlap triggers only when both the hazard geometry
and mechanical payload are reviewed. If either side is missing, strict hazard
automation is blocked and the hazard remains narrative/manual until authored.

### Spawn Anchors

Spawn anchors are reviewed tactical positions, not suggestions hidden in prose.
They can be cells, polygons, ranked lists, or named groups for players,
monsters, reinforcements, summons, lair effects, retreat routes, fallback
starts, and exits.

Each anchor records allowed token sizes, eligible combatant refs or roles,
visibility, plane id, occupancy constraints, and fallback behavior. If required
anchors are missing, blocked, occupied with no reviewed fallback, or outside the
strict map, the adapter must block automated combat start or the specific spawn
operation.

### Fog And Reveal Regions

Fog and reveal regions are geometry overlays keyed by POV or observer group.
They may reference player-safe map derivatives, but the authoritative reveal
state is region geometry plus checkpoint overlay state.

Each region records:

- region id, plane id, cells or polygons
- initial visibility, reveal trigger, and reveal event id when revealed
- visible asset derivative or crop refs when display changes
- whether region text/caption is safe for narrator or player display
- relation to secret features and keyed areas

Fog hides player display, narrator prompts, character-agent prompts, and
front-end payloads. It does not erase adapter-private geometry needed to
adjudicate hidden doors, traps, blockers, or enemy movement. The router may
receive reviewed hidden geometry and secret-feature records as router-only
adjudication context when needed to DM the map properly. Those records are not
observable facts by themselves; they reach players only after the router emits
canonical visible facts or an asset reveal with normal visibility.

### Secret Areas

Secret doors, rooms, traps, compartments, passages, and alternate exits must be
authored as secret features with reveal triggers and visibility gates. They may
exist in adapter-private geometry before players discover them, but player
status views, map overlays, narrator prompts, and character-agent observations
must not receive their labels, refs, display geometry, or captions before reveal.

Strict automation may use unrevealed secret blockers and hazards internally
only when the action requires the world to be physically consistent. It must not
surface spoiler-bearing reasons such as "blocked by secret door" to players.

### Keyed-Area Links

Keyed areas connect tactical geometry to content-pack location records. Each
keyed-area link records a stable location/keyed-area ref, plane id, cells or
polygons, entrances/exits, associated secrets, encounter refs, and reveal
status. These links are how movement over a tactical map can update or suggest
location context without changing the generic world model directly.

The D&D adapter may report entered/left keyed-area refs as adapter events or
pending content signals. The generic engine must not infer `CharacterRecord`
locations from tactical coordinates.

### Map Asset Refs

Strict geometry may reference display assets, but assets are not geometry.

Map asset refs use the existing asset authority:

- `asset://<pack_id>/<asset_id>` for safe delivery refs
- reviewed player-safe derivatives for player display
- optional host/private refs only inside the compiled pack, private review
  tooling, or router-only adjudication context after sanitization to reviewed
  logical ids; never in narrator prompts, character-agent prompts, player
  payloads, default checkpoint exports, or ordinary logs
- asset alignment transforms reviewed separately from topology

If an image is missing, unsafe, unreviewed, hash-mismatched, or not aligned to
the geometry, player display or overlay delivery is blocked. Strict mechanics
may continue only if the geometry itself is reviewed and the requested action
does not require player-visible map display as part of the UX contract.

## Review Gates

Imported tactical maps are accepted by projection, not all-or-nothing. A map can
be runtime-ready for display, rough advisory context, strict movement, strict
line of sight, strict hazards, or strict fog only when that projection's gate is
ready.

Minimum gates:

- `map_asset_reviewed`: player-safe derivatives, hashes, dimensions, MIME type,
  captions, and delivery refs are reviewed
- `grid_alignment_reviewed`: tactical coordinates align to the map scale and
  display asset transform
- `topology_reviewed`: cells, polygons, keyed-area boundaries, and floor/submap
  planes are complete enough for the declared scope
- `blockers_reviewed`: movement, sight, line-of-effect, and occupancy blockers
  are typed and complete
- `terrain_cover_reviewed`: difficult terrain and cover geometry are accepted
- `elevation_reviewed`: elevation and vertical links are accepted where they
  matter
- `connectors_reviewed`: doors, windows, locks, state, and connector behavior
  are accepted
- `hazards_reviewed`: hazard geometry and D&D mechanics are accepted
- `spawns_reviewed`: required player, enemy, reinforcement, exit, and fallback
  anchors are accepted
- `fog_reveal_reviewed`: fog, reveal regions, secret display rules, and player
  derivatives are accepted
- `secrets_reviewed`: secret features have reveal triggers and non-spoiling
  failure/display behavior
- `cross_refs_reviewed`: keyed areas, encounters, statblocks, hazards, loot,
  doors, and assets resolve

Every gate records status, confidence, reviewer, provenance ids, and gate
reasons. `unreviewed`, `needs_review`, `flagged`, `blocked`, low-confidence,
hash-mismatched, or missing gates are not serviceable for strict automation.

## Adapter Enforcement Versus Reporting

The D&D combat adapter must enforce:

- pack identity, schema version, content hash, review status, and gate status
  before materializing imported geometry
- coordinate bounds, token footprint bounds, valid plane ids, valid refs, and
  valid asset refs
- strict movement only across passable cells, connectors, vertical links, and
  reviewed movement modes
- movement budget only when speed, path, difficult terrain, vertical cost, and
  token state are all known
- total-cover or line-of-effect target invalidity when strict blockers prove the
  action cannot reach the target
- stateful connectors such as opened, closed, locked, barred, broken, or secret
  doors/windows when their connector gate is ready
- hazard triggers and mechanical effects only from reviewed hazard geometry plus
  reviewed D&D mechanical payloads
- spawn placement, token size, occupancy, and reviewed fallback anchors
- fog/reveal filtering for player display payloads and status views
- secret feature visibility boundaries and non-spoiling failure text
- area template placement only within reviewed map bounds and blocker rules
  when strict area automation is requested

The D&D combat adapter may report, without enforcing:

- advisory distances, line of sight, cover, and area targeting from current v1
  maps or from geometry layers whose strict gate is not ready
- half-cover and three-quarters-cover context until the attack math path has a
  separate accepted code-owned cover application contract
- ambiguous climb, jump, squeeze, fly, swim, burrow, teleport, forced-movement,
  mounted, vehicle, or improvised-interaction cases that need table judgment
- narrative affordances such as chandeliers, loose rubble, furniture use, noise,
  smell, morale, tactics, or monster intent unless they have typed mechanics
- approximate positions in theater-of-the-mind or sketch-map combat
- reviewed hidden/private geometry to the router through router-only content
  lookup or compact adapter context. The same geometry must not leak unrevealed
  labels, private asset refs, source metadata, or secret display regions to
  narrator prompts, character-agent prompts, player payloads, ordinary logs, or
  default checkpoint exports.

The narrator receives visible canonical facts and safe display captions. It does
not inspect geometry, recompute D&D mechanics, or reveal hidden map state.
Character agents receive text observations and combat context scoped to what
they can know; they do not receive private asset refs, raw geometry dumps, or
unrevealed secret labels.

## Failure And Blocking Behavior

Strict map automation fails loudly and non-spoilingly when required geometry is
missing, unsafe, unreviewed, blocked, mismatched, or not precise enough for the
requested operation.

Hard blockers include:

- no compiled pack row for the requested tactical map template
- pack id, schema version, manifest hash, or content hash mismatch
- geometry row or required relation is missing, unreviewed, blocked, rejected,
  low-confidence, or gated for a different projection
- image exists but geometry is missing, or geometry exists only as OCR notes,
  source-page pixels, labels, captions, or a bounding box
- scale, origin, plane id, grid alignment, or coordinate transform is unknown
- movement/path blockers, doors/windows, vertical links, difficult terrain,
  cover, hazards, fog, secrets, keyed-area links, or spawn anchors are required
  for the requested automation and their gates are not ready
- player display requires an asset that is missing, unsafe, unreviewed,
  hash-mismatched, oversized, unaligned, or not safe for the recipient
- a requested spawn or movement target is outside bounds, in impassable geometry,
  collides with occupied cells without a reviewed fallback, or crosses a
  blocked connector
- a strict operation would reveal secret geometry or unsafe map labels to a
  player, narrator, character agent, normal log, or frontend payload

The failure result is a sanitized runtime error or blocked-action notice plus
operator-diagnostic logs. It must not silently downgrade to image inspection,
invent a path, create a synthetic room, guess from source map labels, ask the
router to bridge missing topology, or save state as if strict automation
succeeded.

When strict geometry is blocked, play may continue only through an explicit
non-strict path: narrative-first D&D combat, manual operator adjudication, or
the existing advisory map state. That downgrade must be visible in status or
logs and must not claim strict movement, fog, hazard, cover, or keyed-area
automation is active.

## Implementation Blockers

These blockers remain explicit and should not be papered over with shims:

- accepted D&D adapter schema for strict tactical map templates and checkpoint
  overlay state
- compiler/review tooling for map geometry gates and per-projection readiness
- asset alignment model from tactical coordinates to player-display images
- deterministic pathing, line-of-sight, line-of-effect, cover, area-template,
  and occupancy algorithms for square grids
- vertical movement and elevation rules, including flight, climbing, falling,
  pits, balconies, and multi-plane line tests
- secret/fog filtering for status views, Discord map overlays, CLI output,
  narrator prompts, character-agent context, and logs
- door/window/connector mutable state and rewind behavior
- hazard trigger timing and integration with the D&D roll/effect pipeline
- strict-map failure surfaces for CLI and Discord that are non-spoiling but
  actionable for the operator
- migration from current advisory `DndBattleMapState` to a strict imported-map
  state without widening the generic narrative engine or adding D&D prompt
  weight to non-D&D sessions

Until those blockers are solved, imported maps with partial geometry are useful
for authoring, display, and advisory combat context only. They are not a
mechanical authority for full module combat maps.
