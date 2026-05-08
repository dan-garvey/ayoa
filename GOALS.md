# Ayoa Goals

This document is the product and architecture north star for Ayoa. It is not a
ticket list. `DESIGN.md` describes the current runtime; this file describes the
direction future runtime changes should preserve.

## Vision

Ayoa has two connected goals.

First, Ayoa is a multiplayer, multi-agent interactive fiction engine. Humans
play bound characters. NPCs are represented by agents with private continuity.
The router canonicalizes events, protects perception boundaries, and paces
beats. The narrator renders each human's visible slice of the world.

Second, Ayoa should become a full AI game-master platform. "DM" should not mean
one omniscient chatbot that grants player wishes. It should mean a coordinated
set of roles: character agents with agency, a router/world arbiter, a rules
subflow, an off-stage world scheduler, a narrator, and a content/rules resolver.

## Product Pillars

### 1. Multiplayer Interactive Fiction

The engine must support multiple humans acting through different characters,
including asynchronous play and scenes that may diverge in time, place, and
perception. Player input remains authoritative as an attempted intention, but
not as an automatic outcome.

### 2. NPC Agency

NPCs must act from their own goals, fears, knowledge, loyalties, and constraints.
They should resist, misunderstand, conceal, refuse, bargain, betray, delay,
coordinate, and pursue off-screen objectives when that is what the character or
world demands.

Avoiding AI sycophancy is a core product requirement. The system should not
optimize for making the player's requested outcome true. It should optimize for
making the world respond coherently.

### 3. Information Asymmetry

Information boundaries are structural, not cosmetic. Characters should only
receive facts they could observe, infer, remember, or legitimately know. Private
agent intent remains private to that agent unless surfaced in fiction.

The router is privileged enough to maintain world continuity and hidden facts,
but every cross-role output must still be scoped by visibility. If a future
component needs hidden-state access, its output contract must explicitly state
what may and may not leave that component.

### 4. Living World

The world must move when the players are not looking. Antagonists, factions,
allies, rivals, institutions, hazards, and clocks should progress off stage.
Background ticks are not flavor; they are how the engine prevents the world from
becoming a passive stage around the current player.

The target shape is not "random ambient events." It is goal-driven world motion
that can later collide with player-facing scenes through observable evidence,
changed resources, rumors, arrivals, absences, consequences, and crises.

### 5. Strong Contested-Action Arbitration

Cat II interactions are the main weak point in the current IF loop. The final
outcome of contested action should not be resolved by whichever actor spoke
last, nor by the responder's preferred result. It should be adjudicated from
initiator intent, responder intent, timing, position, capability, surprise,
prior setup, environment, relevant mechanics, and established world rules.

For rules-heavy modes, Cat II resolution should have a mechanics-aware subflow
that returns a structured adjudication for the router/narrator pipeline to
render. The current D&D slice keeps this subflow router-owned rather than
adding another model role.

### 6. Pluggable Rules And Content

Ayoa should not be hard-coded to D&D. The core engine should know about
characters, intentions, observations, events, beats, and consequences. Rulesets
and content should be plugged in.

For personal use, it is acceptable to use the best available digital
representation of owned D&D 5e content. For any general release, the engine
must ship without protected content and support a bring-your-own-content layer.

For the D&D-specific implementation plan, including the split between basic
D&D mechanics and later adventure-module compilation, see
`DND_MODULE_IMPORT.md`.

The content layer should support at least these source classes:

- Open/public packs such as SRD 5.1, SRD 5.2, Open5e, or 5e-bits data.
- User-owned personal imports from local exports, local compendiums, or private
  adapters.
- Hand-authored campaign packs in Ayoa-native JSON.
- Non-D&D rulesets with their own mechanics and content schemas.

## Non-Goals

Ayoa is not trying to be a pure wish-fulfillment engine, a single-player
chatbot, a replacement for human authorship, or a repository of redistributed
commercial RPG content.

The public engine should not require D&D, D&D Beyond, Foundry, Avrae, 5etools,
or any specific third-party service. Those can be adapters, references, or
personal-use integrations, not architectural dependencies.

## Architecture Implications

### AI DM As Role Ensemble

The "DM" should be decomposed into roles with explicit contracts:

- Character agents decide what NPCs intend from inside their own perspective.
- The router canonicalizes intentions into observable facts and visibility.
- The router-owned rules subflow resolves mechanics-heavy contested actions.
- The tick scheduler advances off-stage actors and clocks.
- The content resolver supplies rules, stat blocks, spells, items, lore, and
  source metadata.
- The narrator renders player-facing prose from visible facts only.

This decomposition is the main defense against sycophancy, accidental
omniscience, and incoherent state mutation.

### Rules Subflow Shape

The rules subflow should be a narrow callable, not a full replacement router or
another always-on agent. It should accept a contested-event packet and relevant
rules/content context, then return a structured result:

- legality and feasibility notes
- rolls/checks/saves used, if any
- mechanical outcome
- fictional outcome surface
- consequences and state deltas
- source/rules references where available
- uncertainty or fallback reason when it cannot adjudicate

The router still decides Cat II opens, responders, observers, and beat pacing.
When the session ruleset says mechanics matter, the router-owned subflow owns
the final contested outcome. Roll plans, pending player rolls, and dice ledgers
are checkpoint-persistent audit data; ordinary LLM history receives only the
canonical adjudicated result.

### Content Pack Boundary

Content packs should be data with provenance, not code that freely mutates the
checkpoint. A future pack interface should distinguish:

- `ruleset`: mechanics grammar and adjudication helpers
- `entities`: monsters, NPC templates, spells, items, feats, conditions, etc.
- `source_metadata`: title, version, license or ownership scope, attribution
- `adapter_metadata`: import path, source service, transform version
- `visibility`: whether the pack is redistributable, local-only, or private

Protected or user-owned content should live outside the git-tracked repo by
default and be loaded through explicit local configuration.

## Landscape Reassessment

The open-source AI-D&D landscape should be read through this lens:

- Prompt-only AI DM apps are useful examples of UX, but they do not solve
  Ayoa's core problem. They tend to collapse DM, narrator, rules, and NPC
  agency into one model.
- Mechanically grounded projects are more relevant than full AI-DM apps. The
  useful pattern is "LLM proposes or narrates; typed tools and state decide."
- D&D automation tools are valuable references for data shape, dice, combat,
  initiative, effects, and content import. They should usually remain adapters
  or inspirations, not core dependencies.
- D&D 5e content sources should be layered: SRD/open content for redistributable
  defaults; personal D&D Beyond, Foundry, or other owned-content imports for
  private use; native JSON for homebrew and non-D&D games.

Current high-signal references:

- [`mnehmos.rpg.mcp`](https://github.com/Mnehmos/rpg-mcp): strong reference for
  rules-enforced AI GM architecture.
- [Avrae](https://github.com/avrae/avrae) and its
  [`d20`](https://github.com/avrae/d20) package: strong references for Discord
  D&D automation and dice grammar; `d20` is the most directly reusable Python
  dependency.
- Foundry's [`dnd5e`](https://github.com/foundryvtt/dnd5e) system: strong
  reference for 5e data models and mechanics, but tightly coupled to Foundry's
  document/runtime model.
- [Open5e](https://github.com/open5e/open5e) and
  [5e-bits](https://github.com/5e-bits/5e-database): good redistributable
  SRD/open-data sources.
- [DDB Importer](https://github.com/MrPrimate/ddb-importer) and similar tools:
  practical personal-use paths from owned D&D Beyond content into structured
  local compendiums.
- [FIREBALL](https://github.com/zhudotexe/FIREBALL) and
  [D&D Agents](https://openreview.net/forum?id=3Op7kJOvaD) research: evidence
  that state/tool grounding improves D&D agent behavior and makes failures
  auditable.

## Evaluation Criteria

Future changes should be judged against these questions:

- Did an NPC preserve its own agenda when the player pushed for a convenient
  answer?
- Did hidden information stay hidden unless surfaced through play?
- Did off-stage actors pursue goals without waiting for the player?
- Did Cat II resolution weigh all sides instead of rubber-stamping the latest
  intention?
- Did the narrator render only visible facts?
- Can the same engine run with no protected D&D content checked into the repo?
- Can a personal campaign use richer owned content through a private adapter?
