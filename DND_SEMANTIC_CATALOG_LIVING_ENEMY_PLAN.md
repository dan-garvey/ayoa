# Semantic Catalog And Living Enemy Design Note

Status: draft for review. This is not an accepted implementation plan.

This note captures the current imported-content catalog direction and the
unresolved design tension around making the module's central villain feel alive
from session start. The open question is not only token cost. It is whether the
runtime can give the villain real strategic agency without creating two
independent versions of the same character.

Later design discussion sharpened the conflict:

- Ayoa is built around character agents with enough context and freedom to drive
  the story.
- Published modules are built around a single omniscient table runner who owns
  scenario truth, pacing, and hidden constraints.
- The import layer must preserve agent-driven story while maintaining module
  cohesion when agents go off script.

This points toward a different responsibility split than "DM layer owns truth,
characters get small slices." Major characters need rich role context. The hard
boundary is not context size; it is decision ownership and state authority.

## Why The Page-Card Measurement Was Misleading

The full OCR extraction measured about 235k tokens of raw page text. The later
10k-19k token numbers were not a measured semantic catalog. They were proxy
measurements over one generic card per page:

- a router catalog/index over 258 page proxies
- generic summaries with empty card bodies
- all cards flagged as unreviewed and unavailable to runtime lookup

That proves only that an index is smaller than raw OCR. It does not prove that
the playable module content compresses by 10x, or that one page should become
one card.

The OCR review passes point toward a different shape: hundreds to low-thousands
of semantic records, grouped by adventure sites, keyed areas, NPCs, factions,
fronts, hazards, events, treasures, handouts, tables, maps, and cross refs.
That is the catalog we need to reason about.

## Semantic Catalog Shape

The authored pack should be semantic, not page based. Page and source spans are
provenance. Runtime refs should be gameplay concepts.

Use the existing `ContentPackDomainCatalog` direction as the base:

- regions, sites, levels, routes, and keyed areas
- reveal graph and cross refs
- handouts, tables, and lore records
- tactical map templates, only after manual map/topology review
- front dossiers
- statblocks, traps, hazards, treasures, and encounter templates

Add or formalize two missing domains:

- `actor_dossiers`: noncombat NPCs, faction roles, major monster personas, and
  recurring minions that may need social or strategic agency.
- `agent_context_slices`: durable per-character startup slices for
  `CharacterRecord.known_context`, `private_state`, identity, goals, secrets,
  and misconceptions.

The same semantic catalog should compile into three different runtime products:

- `router_reference_index`: what exists and where to look next.
- `runtime_content_cards`: compact reviewed packets served after validation.
- `checkpoint_seed_slices`: initial checkpoint state, roster, active fronts,
  public lore, locations, and per-character agent context.

## Ideal Knowledge Structure V0

The imported module knowledge model should separate authored content, character
agency, turn orchestration, and engine-owned state.

### Authored Domain Records

These records describe what exists in the module and what can become runtime
authority after review:

- `locations` and `keyed_areas`: where play can happen, including adjacency,
  local clues, hazards, encounters, handouts, maps, and front links.
- `encounter_templates`: reusable conflict or scene seeds, including triggers,
  participants, location refs, possible noncombat resolutions, and reward
  policy.
- `trap_hazards`, `treasures`, `handouts`, `tables`, `statblocks`, and
  `tactical_map_templates`: typed operational records owned by the engine or
  D&D adapter when the fact is not a character decision.
- `front_dossiers`: pressure systems, clocks, resources, escalation palette,
  and constraints.
- `actor_dossiers`: role context for NPCs, villains, faction roles, and major
  monster personas that may drive story.
- `agent_context_slices`: reviewed projections for `CharacterRecord`
  `known_context`, `private_state`, personality, agenda, beliefs,
  uncertainties, and hard boundaries.
- `knowledge_graph_edges`: semantic links for knowledge, reach, ownership,
  containment, triggers, depletion, and influence.

### Responsibility Split

- Character agents own decisions, performance, voice, motive, beliefs, agendas,
  and relationship-driven initiative.
- The router owns turn orchestration: who gets a turn, when they get it, what
  setting/context it interacts with, and whether the attempted intent should be
  denied, delayed, redirected, or adjudicated.
- The engine owns non-decisions: refs, hashes, runtime gates, graph edges,
  topology, inventories, resources, combat state, clocks, roll math, asset
  state, reveal ledgers, and checkpoint mutation.
- A dedicated lookup role may propose missing refs or graph queries when the
  router lacks enough information, but code validates those refs before the
  router sees compact packets.

### Required Projections

Each authored catalog should be able to produce these projections without
duplicating truth:

- Router index: hierarchical heads, hot local refs, active clocks, relevant
  actor/front heads, and enough graph metadata to know when lookup is needed.
- Router packets: compact validated packets for the current turn.
- Agent slices: role-rich context for characters, especially major story
  drivers, without dumping DM-operational records into their prompt.
- Engine overlay: durable mutable state for clocks, depletion, reveals,
  location changes, active plans, and graph-state changes.
- Checkpoint seed: starting public lore, roster, live/dormant actors, selected
  randomized anchors, active front state, and queued start-relevant signals.

## Router Catalog Policy

The router should not see the full authored pack every turn. The default
catalog should be hierarchical:

- domain/site/front/table heads for the full module
- active-start-region child refs
- current location, adjacent location, active front, active encounter, and
  unresolved commitment refs
- aliases sufficient for lookup, not prose sufficient for play

For a large module, this probably means the router gets a small global index
and a hotter local index. Deep refs are fetched by a bounded lookup pass.

Lookup should still be validated by non-LLM code:

- pack id matches
- ref exists
- content hash matches expected pack identity
- record is reviewed and runtime ready
- requested projection is allowed for the router
- requested ref was advertised or reachable through the catalog graph

The router receives compact content packets only after that validation. Missing
or unapproved required content should fail loudly.

## Initial Checkpoint Creation

The initial checkpoint should not preload the whole module and should not seed
rolling conversation history.

It should contain:

- empty `session_conversation` and per-character conversation history
- `session.content_state` with pack identity, private runtime metadata, active
  front/villain state, overlay state, and queued start-relevant signals
- starting public lore and current location state
- player roster and immediately relevant NPCs
- dormant or spawnable refs for later NPCs, encounters, and monsters
- selected hidden setup such as randomized fortunes or start choices, stored as
  state and refs rather than prose dumps
- per-character `known_context` and `private_state` slices for characters who
  may act early

Minor room occupants, combat-only monsters, treasure caches, and late-module
NPCs should stay in typed records until play reaches them.

## The Living Enemy Problem

The original product goal is that the central villain can make decisions from
the start, so the campaign feels like it has a living enemy rather than a series
of scripted encounters.

The semantic-catalog design creates a real concern:

> If the router/front layer advances villain pressure, and a character agent
> later speaks as the villain, do we now have two villain models?

That concern is valid. A bad implementation would create two independent
decision makers:

- a router/front projection that decides strategic pressure from hidden module
  facts
- a character-agent projection that speaks and acts from its own prompt history

If those surfaces can diverge, the villain stops being a coherent character.
The issue is not just cost. It is authorship and continuity.

There is a second problem: giving a bad guy a turn gives them power. If the
central villain can simply choose to appear immediately and crush unprepared
players, the engine has produced agency but destroyed the table experience. If
the router blocks every surprising villain choice, the villain becomes scripted.

So the router cannot merely canonicalize submitted intentions. For module play,
it must also orchestrate who gets turns, when they get them, how those turns
interact with the setting, and whether attempted intents are appropriate for
the current fiction, pacing, knowledge, and balance.

The engine should own anything that is not a decision:

- pack identity, refs, hashes, and runtime gates
- deterministic state mutation and rollback
- D&D mechanics, resources, rolls, combat state, and inventory
- topology, location adjacency, safe asset state, and reveal overlays
- front clocks, cooldowns, introduced-ref ledgers, and required-ref validation
- knowledge graph edges and reachability rules once those exist

Characters should own character decisions. The router should own turn
orchestration and adjudication. The engine should own non-decision state.

## Module-Derived Design Examples

These are sanitized patterns pulled from the private module OCR and range
reviews. They are not source excerpts and should be treated as design evidence,
not reviewed runtime content.

### Overwhelming Villain Agency

The module contains a recurring central antagonist with direct appearances,
remote influence, powerful minions, and the ability to apply pressure far above
what starting characters can handle.

Design implication: a major villain agent can want a proactive turn from the
start, but the router must decide whether granting that turn is appropriate.
Direct lethal intervention against low-level players is usually not just a
character choice; it is a pacing and encounter-balance decision. The router
needs enough context to distinguish "advance pressure through spies, invitations,
threats, tests, or minions" from "end the campaign immediately."

### Knowledge Channels And Reports

The module repeatedly uses indirect knowledge channels: servants, spies,
watchers, rumors, supernatural awareness, public consequences, and minion
reports. These determine what distant antagonists can plausibly know.

Design implication: the villain should not be omniscient just because the
router is. A knowledge graph needs to track who can learn what through which
channel, with latency and reliability. The router can use omniscient module
authority to validate the channel, but the villain agent should receive the
result as belief, suspicion, report, or uncertainty.

### Social Authority Clocks

Several settlements and factions have public authority figures, law/punishment
machinery, festival or morale clocks, hidden factions, and consequences that
can escalate without a dungeon-room trigger.

Design implication: agents can drive these clocks by making choices, but the
router must schedule turns so one faction does not monopolize the story. The
engine should track clock state and public consequences; characters should
decide how they react to those consequences.

### Scripted Trigger Meets Free Agency

The module has many "if/when the characters do X" trigger patterns. Ayoa
characters may instead leave, bargain, send minions, lie, investigate out of
order, destroy a premise, or create a public consequence that the module did
not script directly.

Design implication: triggers should compile into condition/response records,
not hard rails. The router and lookup role need enough catalog knowledge to
find relevant condition records when agents go off script, then adjudicate a
cohesive consequence instead of forcing the scripted branch.

### Protected Secrets And Relics

The module contains hidden objects, thefts, clue chains, sacred/protective
assets, and location-bound secrets that matter to NPC motives and front
pressure.

Design implication: character dossiers need secrets that the character actually
knows or believes. The exact placement, reveal status, mechanics, and depletion
state belong to module authority and engine overlay. When an agent acts on a
secret, code must validate that the secret exists, is reviewed, and is reachable
through that character's knowledge.

### Hard Location Constraints

The module includes locks, traps, secret routes, vertical movement, hazardous
travel constraints, teleport-like links, bridges/gates, and map-only topology.

Design implication: these are not character decisions. They are setting facts
and mechanics. Agents may decide to exploit, avoid, or test them, but engine and
router validation must decide whether the attempt is physically and mechanically
coherent.

### Random Tables And Fortune-Like Setup

The module uses randomized setup, tables, and outcome anchors that affect where
important content lives and what future scenes mean.

Design implication: initial checkpoint creation must resolve and store random
setup as durable state. Character agents may know selected outcomes only when
their role justifies that knowledge. The router and lookup role need the
resolved anchors to retrieve the right records later.

## Knowledge Graph Direction

A knowledge graph is likely necessary for module imports of this size. The
router should not need the entire graph in prompt context, but it needs enough
index and lookup support to ask for more information when a turn requires it.

The graph should represent at least:

- content containment: region, site, level, keyed area, room, encounter
- character knowledge: knows, suspects, believes falsely, can learn through
  channel
- faction/front reach: controls, can influence, can observe, can dispatch
- reveal and clue dependencies
- resource ownership and depletion
- topology and travel constraints
- trigger conditions and response records
- active clocks, cooldowns, and consequences

This probably requires a dedicated LLM role for deciding when the router needs
more information, but that role should not be the authority. It should propose
candidate refs or graph queries. Code validates them, fetches reviewed packets,
and only then does the router adjudicate.

## Candidate Designs For Villain Agency

### Option A: Router-Owned Front, No Strategic Villain Agent

The villain exists as front state plus compact router `front_signal` packets.
The router decides offscreen pressure and only creates an embodied villain agent
when the villain appears directly.

Benefits:

- one main adjudicator for hidden module truth
- cheaper and easier to validate
- less risk of leaking the full catalog into character-agent context

Costs:

- may feel like scripted pressure rather than a thinking enemy
- the villain's strategy is router-authored, not character-authored
- direct speech later may feel disconnected unless the embodied agent receives
  a carefully reconstructed state slice

### Option B: Villain Agent Owns Strategy From Session Start

The villain has a durable `CharacterRecord` from checkpoint creation and can be
called as a background agent on front-clock triggers. The router validates and
canonicalizes the villain's plans and visible consequences.

Benefits:

- closest to the original living-enemy goal
- one character prompt can carry the villain's voice, beliefs, grudges, and
  evolving plans
- offscreen choices can come from the same agent who later appears in scene

Costs:

- the villain agent needs a broad strategic slice from the start
- the slice must be large enough to plan, but not a full catalog dump
- the router still needs hidden content authority, so there are still two LLM
  roles involved unless the router becomes mostly validator
- more chances for hidden knowledge leakage or unsupported plan references

### Option C: One Villain State, Multiple Projections

There is one durable villain state in checkpoint data. The router and character
agent both receive projections of that same state:

- router projection: goals, constraints, knowledge channels, resources, active
  plans, cooldowns, minions, and hidden module refs needed for adjudication
- character-agent projection: what the villain knows, believes, wants, fears,
  misunderstands, remembers, and is currently trying to do

Benefits:

- avoids a full catalog in the character prompt
- keeps strategic state durable and inspectable
- lets the embodied villain speak from the same state the front uses

Costs:

- may still feel like two models if both router and agent can originate plans
- requires clear ownership rules for who proposes strategic action
- requires tests that catch divergence between front state and agent context

This is the current best candidate, but it is not yet proven. The crucial
implementation rule would be: the villain's durable state is the authority, not
either prompt transcript.

## Proposed Ownership Rule

To avoid two independent villains, choose one owner for strategic intent.

Recommended default for review:

- The villain agent proposes strategic intent when a front clock, observation,
  or knowledge-channel trigger says the villain should think or act.
- The router adjudicates whether that intent is feasible, what content refs are
  needed, what becomes observable, and which minions/events/locations change.
- The checkpoint stores the resulting villain-state mutation.
- Future villain-agent calls see the updated state slice.

Under this rule, the router is not inventing the villain's personality or long
term strategy. The router is validating and integrating the villain's intent
against module truth and table state.

This may be more expensive than a pure router-owned front, but it better matches
the living-enemy goal.

## What The Villain Needs In Context

The villain does not need every room, treasure cache, appendix table, or
statblock in prompt context.

The villain probably does need:

- core goals and current priority stack
- constraints, taboos, vulnerabilities, and reasons not to act directly
- domains of control and what resources exist in those domains
- important minions and what each can plausibly do
- knowledge channels: spies, magic, rumors, direct observation, servants, fear
- current beliefs about the party
- current uncertainty and misinformation
- recent mediated observations
- active plans, cooldowns, and pending pressure
- strategic map of major regions/sites, not tactical room detail
- refs the villain can request or activate through minions

The villain should get exact local details only when a plan touches them. For
example, if the villain sends a minion to a specific site, lookup can fetch that
site/minion/hazard packet for adjudication. The villain agent should not carry
that entire site's keyed-area catalog forever unless it becomes part of its
active plan state.

## Interface With The Catalog

The catalog should support villain agency directly:

- `front_dossier` defines broad pressure, resources, constraints, clocks, and
  action palette.
- `actor_dossier` defines the villain as a character/persona, including voice,
  beliefs, relationships, secrets, and play posture.
- `agent_context_slice` defines what the villain's agent prompt receives at
  checkpoint start and after state changes.
- `cross_refs` connect villain resources to minions, regions, items, and
  events.
- `runtime_content_cards` provide precise module authority when a villain plan
  touches a concrete record.

Checkpoint creation should instantiate the villain state early even if the
villain is not physically present. The open question is whether that state is a
full `CharacterRecord` from turn zero or a front/villain state that can be
promoted into a `CharacterRecord` when the first strategic agent call is needed.

## Open Review Questions

1. Is the living enemy allowed to have its own background agent turn from
   session start, or should villain pressure remain router-owned until direct
   appearance?
2. If the villain agent proposes strategy, what prevents the router from also
   inventing strategy in parallel?
3. Should active villain state live primarily in `ContentPackState.fronts` and
   `villains`, in a dormant `CharacterRecord`, or in both with one declared
   source of truth?
4. How large is an acceptable strategic slice for a major villain in tokens?
5. What should happen when the villain wants to act through a module ref that is
   not yet reviewed/runtime-ready?
6. How do we test that later embodied dialogue reflects earlier offscreen
   choices?

## Next Measurement Work

Before accepting an implementation, measure token budgets for actual semantic
drafts, not page proxies:

- hierarchical router index for the full large module
- active-region hot index
- major villain strategic slice
- typical local NPC slice
- typical location packet
- typical front signal
- serve-all compact packets for a representative authored subset

Those measurements should come from authored semantic records, even if only a
representative slice is authored first.
