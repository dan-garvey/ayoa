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
   * `cat_ii_resolution_mode`
   * `player_roll_mode`

   Defaults preserve current behavior:

   * `ruleset_id="narrative"`
   * `cat_ii_resolution_mode="router"`
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

   In Cat II final resolution only, if `cat_ii_resolution_mode` selects
   `dnd5e_router`, call the roll-planning path, execute rolls in code, call
   finalization, and compile the result back into the existing event shape.
   Otherwise keep the current router path.

   Current implementation status: this branch exists behind
   `cat_ii_resolution_mode="dnd5e_router"`. It preserves the Cat II opening
   context, asks the event-router role for a d20 roll plan, executes planned
   rolls through Ayoa's `d20` wrapper, asks for final adjudication, and compiles
   the outcome back into `EventRouterOutput`. State deltas are returned as notes
   but are not applied yet. The legacy setting value `"rules_arbitrator"` maps
   to `"dnd5e_router"` for old saves.

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

The module importer should compile an owned adventure into:

* raw chunk index
* adventure graph
* location/scene cards
* NPC/faction cards
* encounter cards
* stat block refs
* clue/reveal graph
* foreshadowing bank
* campaign bible
* source/provenance metadata
* import warnings and review notes

The mutable campaign state is separate from the immutable module pack. The
module says what the published adventure starts with; the checkpoint says what
is true now.

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

### Campaign Bible

A compact omniscient summary of the module's premise, escalation path, major
villains, factions, mysteries, themes, and tonal motifs. This is stable context
for the router/adventure resolver, not for character agents or the narrator.

### Adventure Graph

Nodes and edges for chapters, regions, locations, rooms, scenes, encounters,
quests, and transitions. This lets the runtime know what is adjacent, what has
been skipped, and what later content a current choice can affect.

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

Scene cards are the module equivalent of "what is in this room?" They are
necessary but not sufficient without the campaign bible and reveal graph.

## Import Strategy

PDF import should be local-only and offline for protected content. It should be
treated as a compilation pipeline with review, not as a perfect one-shot parse.

Likely passes:

1. segment headings, chapters, keyed areas, appendices, tables, maps, and
   stat blocks
2. extract entities: NPCs, monsters, factions, locations, items, hazards,
   clues, quests
3. build adventure graph
4. build secret/reveal graph
5. build foreshadowing bank
6. produce campaign bible
7. produce warnings for ambiguous or missing structure

The importer can use LLMs at import time because the expensive part happens
once. Turn-time should use compact compiled artifacts.

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
