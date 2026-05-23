# Semantic Catalog And Living Enemy Design Note

Status: draft for review. This is not an accepted implementation plan.

This note captures the current imported-content catalog direction and the
unresolved design tension around making the module's central villain feel alive
from session start. The open question is not only token cost. It is whether the
runtime can give the villain real strategic agency without creating two
independent versions of the same character.

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
