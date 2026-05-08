# Design Doc: Ayoa Narrative Engine

## 1. Status

This document describes the current v11 architecture. It is no longer a
v1 proposal. The implementation is a Discord-first, checkpointed,
multi-character interactive fiction engine whose core runtime lives in
`app/engine/orchestrator.py`, `app/engine/turn_loop.py`, and
`app/engine/turn_loop_dispatcher.py`.

The old "Narrator Phase 1 + Discriminator + Narrator Phase 2" design has
been collapsed. The current loop is:

1. A human submits an action for a bound character.
2. The event router classifies and canonicalizes the intention.
3. The turn loop broadcasts the canonical event to human render buffers
   and NPC observation inboxes.
4. NPC agents may respond, one routed intention at a time, until the
   router ends the beat or the engine hits the event cap.
5. The narrator renders each observing human's POV from visible
   `observable_facts` only.
6. Spawns, dormancy, culls, off-stage ticks, transcript updates, and
   checkpoint saving are applied around the beat.

The system is intentionally prompt-heavy. Prompt files under
`app/prompts/` are versioned source artifacts. Prompt history is tracked
by git, not by filename suffixes.

The key architectural truth is that Ayoa is currently **router-centered**.
The event router is the world arbiter. Character agents propose
character-local behavior; the narrator renders human POV prose; the
orchestrator and turn loop apply router outputs to checkpoint state. Future
mechanics/content extensions should be understood as narrow subflows that
relieve a specific router responsibility, not as replacements for the current
router-centered runtime.

## 2. Goals

For future product direction, see `GOALS.md`. This design document is the
current-runtime handoff and should be treated as authoritative when it
conflicts with aspirational planning notes.

The engine must:

* support multi-turn interactive fiction in Discord
* support multiple human players bound to different characters
* preserve information boundaries by construction
* keep the player's input and physical agency intact
* let NPC agents carry private continuity without leaking it to other roles
* model shared perception, not just shared physical location
* support contested actions without deciding another actor's response early
* persist every committed state change in portable JSON checkpoints
* provide enough logs and checkpoint artifacts to investigate playtest bugs
* keep prompt context small enough for long-running sessions

## 3. Non-goals

The current engine does not attempt to:

* expose a public HTTP story API
* stream final prose to Discord token-by-token
* expose raw chain-of-thought or thinking blocks
* guarantee deterministic replay across LLM calls
* run external tools from inside story agents
* maintain a separate D&D/rules arbitrator or content resolver
* maintain a fully separate transcript for every human POV in the global
  transcript list

## 4. Design Principles

### 4.1 Facts Are The Contract

The router's `canonical_event.observable_facts` are the authoritative
surface of what happened. The narrator renders those facts. NPCs receive
their visible subset in `pending_observations`. If a detail must be known
by someone, it must exist as an observable fact visible to that character.

### 4.2 Perception Is Contextual

`CharacterRecord.location` is an opaque continuity label, not routing
authority. The router decides who perceives each event from live context:
co-presence, live audio, cameras, radio, telepathy, scrying,
supernatural senses, spies, or any other established channel.

There is no engine-side scene graph and no local/remote observer flag in the
live event model. `broadcast_event()` trusts the router's `observers` list and
then filters by fact-level visibility. In production today, `all_observers`
facts are visible to every listed observer; a remote or mediated character
should only be listed as an observer when the router intends them to receive
the broad facts, or should receive scoped `audience="only"` facts instead.

### 4.3 The Narrator Is A POV Renderer

The narrator is no longer the world adjudicator. It is a per-POV prose
renderer. It receives visible facts, observation levels, relevant
context, the acting player's original input, and the POV's rolling
narrator history. It must not invent actions, dialogue, outcomes,
interiority, or physical business that is not in the visible facts or the
player's input.

### 4.4 The Router Owns Canonicalization

The event router is the adjudication, perception, and beat-pacing
role. It decides:

* Cat I vs Cat II classification
* feasibility
* time advancement
* observable facts and fact-level visibility
* observers and response priorities
* NPC responder picks
* beat end state
* spawns, dormancy, and culls

The router also handles router-shaped special entries: `(begin)`, `(arrive)`,
`(query: ...)`, continuation rescue, Cat II resolution blocks, and off-stage
tick fan-in. These are different user-message framings over the same
`EventRouterOutput` schema, not separate adjudication engines.

### 4.5 Agents Author Intentions, Not State

Character agents produce free-form public prose plus one trailing
parenthetical containing their private intent. The public prose becomes
the next intention that the router canonicalizes. The parenthetical stays
only in that agent's rolling history and is stripped at every cross-role
boundary.

Agents do not write canonical events, mutate roster state, move characters,
or send hidden directives. If an NPC's private plan should affect the world, it
must surface through the agent's public action, an off-stage tick, or a later
observable fact the router canonicalizes.

### 4.6 Checkpoints Are The Source Of Truth

Every committed turn writes a checkpoint. The runtime can be rebuilt from
checkpoint JSON plus the prompt/code version in git. Process memory,
locks, and API caches are not trusted as durable state.

## 5. Runtime Components

### 5.1 Discord Bot And EngineBridge

`app/bot/commands.py` defines the slash-command surface:

* `/session start|resume|list|end`
* `/story list|info|start|characters|import|delete`
* `/join`, `/begin`, `/leave`, `/character`, `/describe`
* `/act`, `/defer`, `/query`, `/status`, `/settings list|set`
* `/rewind`, `/clear`, `/abort_beat`

The bot calls the engine in process through `EngineBridge`; there is no
FastAPI layer in the current runtime. `SessionMap` stores Discord channel
to session mappings, private POV thread mappings, and turn-message refs
used by rewind cleanup.

`EngineBridge` is the frontend boundary for Discord and the CLI REPL. It owns
the live `LLMClient`, `CheckpointManager`, `PromptManager`, and `Orchestrator`;
wraps turns in a per-session frontend lock; runs stale Cat II sweeps before
new `/act` processing; and exposes helper flows for join, begin, arrival,
takeover, leave, settings, query, and rewind.

### 5.2 Orchestrator

`Orchestrator.process_turn()` is the main turn entry point. It:

1. loads the latest checkpoint
2. resolves the acting character
3. acquires a per-session lock
4. checks active act slots
5. calls `run_beat()`
6. applies character lifecycle changes and spawns
7. appends one transcript entry for the beat
8. runs eligible off-stage ticks
9. increments the turn index and saves
10. returns a `TurnResponse`

`Orchestrator.resolve_cat_ii()` is the separate pre-turn closeout path for
stale or newly-ready Cat II events. It re-enters the same router resolution
contract, broadcasts the resolved event, renders any resulting POV output, and
saves before the user's new action proceeds.

### 5.3 Turn Loop

`run_beat()` in `app/engine/turn_loop.py` is the v11 state machine. It
supports fresh human actions, Cat II responder actions, NPC cascades,
observation harvest, partial renders for contested attempts, max-event
backstops, and per-human render fan-out.

Important state:

* `active_act_slots`: per-beat lock state for initiators, Cat II responders,
  and pending D&D player rolls
* `open_cat_ii_events`: contested events awaiting responders
* `cat_ii_roll_transactions`: checkpoint-persistent D&D roll plans, pending
  player rolls, completed roll results, and dice ledgers
* `render_buffers`: per-human queues of canonical event ids waiting for
  narrator render
* `canonical_events`: append-only log of closed canonical events

### 5.4 LLMDispatcher

`LLMDispatcher` binds the abstract turn-loop protocol to the live roles:

* `route_intention()` calls the `event_router` prompt
* `route_continuation()` asks the router to repair an open beat with no
  dispatchable NPC pick
* `route_tick_intentions()` bundles off-stage tick outputs into one
  router call
* `agent_intend()` calls `CharacterAgent.respond()`
* `harvest_perceptions()` calls `CharacterAgent.perceive()` in parallel
* `narrator_compose()` calls `compose_pov_render()`

It also owns the router context-trimming calls that surface initial
rosters, world-fact deltas, and pending state changes.

The dispatcher snapshots and restores context-trim queues around router calls
so a failed router completion does not silently drain `world_facts_delta` or
`pending_router_state_changes`.

### 5.5 Event Router

The event router prompt and `EventRouterOutput` schema replace both the
old narrator adjudication phase and the old discriminator role. The
router emits one structured object per routed intention or tick fan-in.

Current router call shapes:

* fresh human or NPC intention (`## Intention`)
* Cat II final adjudication (`## Cat II Resolution`)
* off-stage tick fan-in (`## Off-Stage Tick`)
* continuation rescue (`## Continuation Required`)
* OOC directives such as `(begin)`, `(arrive)`, `(defer)`, and
  `(query: ...)`

The top-level object carries:

* `event_id`
* `decision_rationale`
* `canonical_event`
* Cat I / Cat II fields
* `agent_responder_picks`
* `ends_beat` and `ends_beat_reason`
* `observers`
* `spawn`, `dormant`, `cull`

The nested `canonical_event` deliberately carries only:

* `world_adjudication.feasible`
* `observable_facts`

Legacy audit fields such as `attempted_action` and `resolved_outcome`
are retired. Old checkpoints that still contain them are loaded by
dropping those fields during validation.

### 5.6 Character Agents

Each character has one rolling conversation in
`checkpoint.character_conversations[character_id]`. The same prompt is
used for:

* on-stage response mode
* off-stage tick mode
* perception/loadout mode

On-stage and tick outputs are free prose followed by a trailing
parenthetical. Those two modes append the raw exchange, including the
parenthetical, to the character's own rolling history. The engine parses the
assistant output into:

```json
{
  "character_id": "rashid",
  "public_text": "Rashid sets his glass down. 'Say that again.'",
  "intent": "force the claim into the open without revealing his own source"
}
```

Only `public_text` leaves the agent. `intent` remains private continuity
inside that character's own history.

Perception/loadout mode is different: it does not require a trailing
parenthetical and does not drain `pending_observations`. It still appends the
perception exchange to the character's rolling history so future agent calls
remember what the character established about their current visual
self-presentation. It is used for observation harvest and private query
enrichment when the engine needs a current visible loadout fragment.

### 5.7 Narrator

`compose_pov_render()` is the only production narrator entry point. It
renders one human POV at a time from render-buffered canonical events.
Events are resolved by id against `checkpoint.canonical_events`, filtered
by fact visibility, and tagged with direct / indirect / inferred
observation level.

The narrator output schema is:

```json
{
  "final_text": "string"
}
```

The engine constructs the transcript entry from the real player input
and the rendered `final_text`.

### 5.8 Character Manager

The character manager applies router-directed roster mutations:

* status changes: active, dormant, culled
* LLM-backed spawns from `SpawnRequest`
* spawn summaries queued into `pending_router_state_changes`

It also purges live bookkeeping when characters are culled or leave: active
act slots, open Cat II responder state, render buffers, and stale queued spawn
state-change lines. Movement is not a roster side effect; arrivals,
departures, and transfers must be written as `observable_facts`.

### 5.9 Story Importer

`story_importer.py` turns a master prompt into a v11 checkpoint. The active
entry point is still named `run_import_two_call` for caller compatibility, but
the current path is a multi-call extraction pipeline:

* public world: setting, public lore/facts, physics, narrative rules
* hidden world: hidden lore/facts
* characters: roster, backstories, personalities, goals, secrets
* knowledge envelopes: per-character `known_context`
* player primer
* optional preservation analysis continuation

Opening prose is not extracted. The first playable scene is composed later by
the router on `(begin)` and rendered through the normal narrator path.

### 5.10 Context Builder And Prompt Manager

`context_builder.py` owns shared context formatting and visibility-aware
helper logic. `PromptManager` loads `app/prompts/{name}.txt`, expands
partials, strips HTML comments, splits system/user sections on
`<<<USER>>>`, and renders rolling conversations.

`turn_loop_contracts.py` owns exact prompt-code markers such as
`## Cat II Resolution`, `## Off-Stage Tick`, `## Continuation Required`,
`## ON-STAGE`, `## TICK`, `## PERCEPTION`, and the partial-render marker.
When code branches on a prompt mode, the marker should live there rather than
as an ad hoc string in the prompt or caller.

Prompt templates are versioned in git. They are not copied into
version-suffixed filenames, and checkpoint JSON does not store prompt
version ids.

### 5.11 LLM Client

`LLMClient` is the provider boundary for live model calls. Callers send
role, messages, sampling settings, and an optional Pydantic response
model; the client selects the configured provider/model for that role
and normalizes the result back into `LLMResponse`.

It supports:

* per-role provider and model selection
* Anthropic Messages and OpenAI Responses adapters
* Pydantic structured output normalized into `response.parsed`
* Anthropic prompt caching and conditional server-side compaction for
  supported models
* per-role Anthropic extended-thinking budgets
* provider-specific retry handling for transient failures

Provider/model selection can be configured with model prefixes such as
`openai:gpt-5.4-mini`, explicit `role_providers`, or environment
overrides like `LLM_PROVIDER_NARRATOR=openai` and
`LLM_MODEL_NARRATOR=gpt-5.4-mini`.

Active live model roles are `event_router`, `narrator`, and `agent`.
`character_gen` remains in configuration and has a prompt file, but the
current spawn path renders `character_gen.txt` and calls the LLM with
`role="agent"`. The old `discriminator` role is vestigial compatibility state
and is not used by live calls.

## 6. Turn Lifecycle

### 6.1 Fresh Action

1. A player submits `/act`.
2. `EngineBridge.run_turn()` sweeps stale Cat II pins before processing
   the new action. Any swept events are resolved first and returned as
   `pre_turn_resolutions`.
3. `Orchestrator.process_turn()` resolves actor, then locks the session.
4. `check_act_slot()` either accepts, rejects, or treats the input as a
   Cat II responder intention.
5. `run_beat()` routes the current intention through the event router.
6. The router emits a Cat I or Cat II event.

### 6.2 Cat I

Cat I is self-closing: dialogue, passive action, unambiguous movement,
ordinary observation, OOC directives, and social attempts where the
speech act itself happens.

The turn loop broadcasts the event, then either:

* ends and renders, if `ends_beat=true`
* performs observation harvest, if `ends_beat_reason=observation_harvest`
* performs private query harvest, if `ends_beat_reason=query_response`
* dispatches the first valid NPC pick and routes that agent's intention
* asks the router for one continuation rescue if the beat remains open but
  no dispatchable NPC pick survives filtering
* forces render if `max_events_per_beat` is reached

### 6.3 Cat II

Cat II is contested: violence, contested possession, forced movement
through opposition, consensual physical contact where reciprocation
matters, and similar actions whose outcome depends on another actor's
response.

Cat II flow:

1. The router emits an attempt-in-progress event.
2. The engine opens an `OpenCatIIEvent`.
3. Human responders are pinned in `active_act_slots`.
4. Agent responders intend immediately.
5. If all responders are present, the router resolves the event inline.
6. If D&D mode is enabled, final resolution may enter a router-owned roll
   planning/finalization subflow. NPC/agent rolls execute automatically; player
   rolls either execute automatically or pause for Discord roll UI depending on
   `player_roll_mode`.
7. If any human responder or player roll is pending, the narrator renders
   partial-mode prose where applicable and the beat pauses.
8. When the human responds or rolls, the router receives the compact resolution
   packet and emits the resolved canonical event.
9. After a Cat II event resolves, an NPC initiator gets the first
   follow-up turn when applicable.

Cat II final resolution is still router-owned. There is no separate rules
arbitrator model role in the live runtime. Roll plans and dice ledgers persist
in checkpoint transactions for rewind/audit but are not appended to
`session_conversation`; future LLM context receives the canonical outcome facts.

### 6.4 Broadcast

`broadcast_event()` appends the router output to `canonical_events` and
fans it out:

* human observers with visible facts get render-buffer entries
* NPC observers with visible facts get their visible facts appended to
  `pending_observations`
* the event actor does not receive their own action as an inbox entry

Important current behavior: `broadcast_event()` does not know whether an
observer is local, remote, mediated, or inferred. It passes
`include_all_observers=True` for every listed observer. Therefore broad
`all_observers` facts reach every observer the router lists. If only one
remote character should receive a mediated fact, that fact must be
`audience="only"` and the router should be careful about whether the remote
character belongs in `observers` at all.

`all_observers` does not mean "all characters in the session." It means all
characters in the event's explicit `observers` list. An event with
`observers=[]` is still appended to `canonical_events`, but no human render
buffer or NPC inbox receives even broad `all_observers` facts.

### 6.5 Render

When the beat ends, `_end_beat()` flushes each human's render buffer,
calls `narrator_compose()`, stores per-POV narrator history, and returns
`per_player_renders`.

The legacy `output_text` mirrors the acting player's render for older
callers.

### 6.6 Roster And World Mutation

After `run_beat()` returns, the orchestrator applies each closed event's:

* `dormant`
* `cull`
* `spawn`

Movement is expressed in `observable_facts`. There is no separate router
movement side-effect.

### 6.7 Off-Stage Ticks

After the on-stage beat, `_run_ticks()` may fire off-stage NPC ticks
based on:

* `ticks_enabled`
* stagnation threshold
* tick concurrency cap
* NPC status, play binding, pin state, prior action this turn, and
  whether the NPC is eligible to act

Successful tick outputs are bundled into one router fan-in call. The
router emits an off-stage canonical event and any implied mutations. No
narrator render happens for tick-only events.

The tick scheduler is intentionally simple today: a stagnation counter plus
eligibility filters. It is not yet a clock/resource/faction scheduler. Tick
fan-out currently runs synchronously on the `/act` critical path.

### 6.8 Begin, Arrive, Defer, And Query

The public commands for non-standard turns are router entries:

* `/begin` runs `(begin)` once after players have joined the lobby.
* Late `/join` runs `(arrive)` after the opening exists.
* `/defer` runs `(defer)` through the same turn path as `/act`.
* `/query` runs `(query: ...)` through the normal router/narrator path.

`/query` is not a read-only side channel. It can append canonical events,
update router/narrator conversations, and save a new checkpoint. The router is
responsible for producing a private visible fact that the narrator renders for
the querying POV.

## 7. Event And Visibility Model

### 7.1 ObservableFact

Each observable fact is an object:

```json
{
  "text": "Rashid says, to the table: 'Say that again.'",
  "audience": "all_observers",
  "visible_to": []
}
```

For scoped facts:

```json
{
  "text": "A producer whispers in Dante's earpiece: 'A late contestant is on site.'",
  "audience": "only",
  "visible_to": ["dante_royale"]
}
```

Facts are split by audience, not by sentence. If the same exact audience
perceives a full exchange, one packet is usually better than many small
packets.

### 7.2 ObserverEntry

Observers carry event-level perception and response pressure:

```json
{
  "character_id": "dante_royale",
  "observation_level": "d",
  "response_priority": 4
}
```

Observation levels:

* `d`: direct
* `i`: indirect
* `f`: inferred

`observation_level` says how the observer encountered the event.
Fact-level `audience` / `visible_to` says which facts they receive.

### 7.3 Responder Picks

`agent_responder_picks` are NPCs the router wants to dispatch next.
Human-controlled characters are stripped by the engine. NPC location and
perception eligibility are router responsibilities, but the schema still
enforces that picks appear in `observers`. For a remote NPC to respond,
the router must include that NPC as an observer and give them a concrete
perceptual path.

Two beat reasons are exceptions: `observation_harvest` and `query_response`.
In those modes, `agent_responder_picks` are perception-harvest targets, not
cascade actors, and the schema does not require them to be event observers.

### 7.4 Visibility Caveat

The schema has a fact-level helper that can exclude broad `all_observers`
facts (`include_all_observers=False`), but production broadcast and narrator
composition currently pass `include_all_observers=True`. The practical rule is:
the router's observer list is the event boundary. Do not list a character as an
observer unless they are meant to receive the event's broad facts; use scoped
facts for partial private channels.

## 8. Character State

### 8.1 CharacterRecord

Important fields:

```json
{
  "character_id": "dante_royale",
  "name": "Dante Royale",
  "status": "active",
  "location": "great_hall",
  "is_playable": true,
  "public_sheet": {
    "role": "host",
    "appearance": "",
    "faction": ""
  },
  "private_state": {
    "goals": [],
    "current_objectives": [],
    "secrets": [],
    "intentions_enabled": true
  },
  "pending_observations": [],
  "backstory": "",
  "personality": "",
  "known_context": ""
}
```

`is_playable` means "claimable by a human", not "currently
human-controlled." Human control is determined by
`session.character_bindings` and the legacy `session.player_character_id`
fallback.

### 8.2 Pending Observations

`pending_observations` is an NPC inbox. It is populated by:

* visible facts from `broadcast_event()`
* private or mediated facts scoped to the character
* arrival/departure/transfer facts when the router writes those as
  `observable_facts`
* a spawn's own location seed when the spawn path needs the new NPC's first
  dispatch to know where they are

The inbox is drained into the agent's next on-stage or tick prompt.

### 8.3 Private Continuity

Mutable interior continuity lives in each agent's rolling conversation,
primarily in the private trailing parenthetical. It is not mirrored onto
the character record as `private_updates`, `last_intent`, or directives.

## 9. Context Management

Context is deliberately one-shot or delta-based where possible.

Router context:

* system prompt contains stable role contract, schema contract, setting,
  world lore/rules, hidden lore, and hidden facts
* initial NPC roster appears only on the first router call
* world facts are surfaced once through a delta tracker
* importer-seeded NPC goals/objectives are surfaced in the initial roster
* router-authored context relies on router history
* external engine changes surface through `pending_router_state_changes`
* actor inbox entries surface through "Arrived For You Since Last Turn"

Agent context:

* stable character identity is in the system prompt
* per-turn user message carries pending observations and mode body
* per-character `known_context` is the normal world-context source for agents
* there is no repeating "Characters Present" block
* local arrivals and exits arrive through the observation inbox

Narrator context:

* each human POV has a separate rolling narrator conversation
* the narrator gets visible facts and observation tags
* agent responses are already folded into those facts

## 10. Dynamic Character And Arrival Flows

### 10.1 Router Spawns

The router may emit `spawn` requests for meaningful recurring
characters. `CharacterManager.spawn_characters()` renders the `character_gen`
prompt to create the record, then queues a compact router summary into
`pending_router_state_changes` so the next router call knows what the
engine added.

This is story-time NPC authoring. The router decides that the world needs a
new actor and provides a `SpawnRequest` with a target `character_id` and seed
facts such as role, location, or objectives. The character manager deduplicates
within the router batch, caps distinct spawns per turn, skips ids that already
exist, renders the prompt with setting/lore/rules/location/existing roster
context, and expects the shared `AuthoredCharacter` flat schema.

The generated `CharacterRecord` is appended to the roster. Its
`router_summary` is not stored on the record; it becomes a single
`pending_router_state_changes` line. If the router or caller supplied a
location, that location overrides the LLM-authored location. Fresh NPCs also
receive a small pending observation of their own location so their first
dispatch has a concrete self-position.

Implementation detail: the spawn path renders `character_gen.txt`, but the
LLM call currently uses `role="agent"` rather than the configured
`character_gen` role. Treat `character_gen` as a prompt/template name, not an
active model role, unless that code path is changed.

### 10.2 LLM-Free Custom Player Characters

`EngineBridge.create_player_character_simple()` creates a player-authored
character directly from name, appearance, and optional backstory. It
leaves location and deeper interior blank, binds the user, and queues a
state-change line telling the router to infer a concrete story role and
immediate on-ramp.

The router must turn that on-ramp into in-fiction observable facts for
the NPCs who would plausibly know. For example, in a production-show
story, a producer/host may receive a quiet roster update that a late
contestant has been added and must be made to work.

### 10.3 Takeover And Replacement

Takeover paths bind a human to an existing or LLM-authored character and
surface the change to the router through the same state-change queue.
When a player leaves, the engine can synthesize enough personality from
their rolling play history for the agent to take the character back over.

The takeover/custom-PC authoring prompt is a frontend authoring flow, not an
on-stage agent. It uses the `takeover` prompt but currently calls the LLM under
the `event_router` role.

There are four related paths:

* plain takeover: bind a user to an existing roster character; identity and
  `is_playable` are not rewritten.
* LLM-backed custom character: mode `describe` in `takeover.txt` authors a full
  `AuthoredCharacter`, creates a new playable record, binds the user, and
  queues the router summary with a `[player-bound]` tag.
* replacement: mode `suggest` returns candidate NPC slots without mutation;
  mode `replace` grafts the authored character onto the picked id, preserving
  circumstances such as location, status, pending observations, and current
  objectives while overwriting identity/interior fields and clearing that id's
  rolling conversation.
* LLM-free custom character: `create_player_character_simple()` builds a sparse
  record directly from user-supplied name/appearance/backstory, binds it, and
  relies on the `(arrive)` router turn to place the character in fiction.

## 11. Discord Session Behavior

Discord is the primary UX.

* Sessions are mapped to channels.
* Human players are mapped to character ids.
* Private POV delivery uses private threads where possible, falling back
  to DMs.
* `TurnResponse.per_player_renders` is delivered to the relevant
  players.
* `/begin` is the canonical opening command; `/join` before begin binds
  players into the lobby, while late joiners get an `(arrive)` turn.
* Each posted turn message is tracked in `SessionMap`.
* `/rewind` deletes later checkpoints and deletes or hides tracked
  Discord messages for the deleted turns.
* `/clear` removes Discord-side session clutter without deleting the save.
* `/abort_beat` is an admin recovery tool for wedged active slots or open
  Cat II events.

## 12. Request And Response Contract

The live engine contract is Pydantic, not HTTP.

Request:

```json
{
  "session_id": "uuid-or-slug",
  "checkpoint_id": null,
  "user_input": "I ask Dante why the new contestant is here.",
  "acting_character_id": "dan",
  "stream": false
}
```

Response:

```json
{
  "session_id": "uuid-or-slug",
  "checkpoint_id": "ckpt_0043",
  "turn_index": 43,
  "output_text": "You ask Dante...",
  "per_player_renders": {
    "dan": "You ask Dante...",
    "dante_royale": "Dan turns toward Dante..."
  },
  "beat_ended_reason": "directed_at_player",
  "pre_turn_resolutions": []
}
```

`debug` and `debug_flags` were removed from `TurnRequest`, and
`TurnResponse.debug` was removed. Per-turn diagnostics live in logs and
checkpoint artifacts. The `stream` field remains on the request for
compatibility but is not a live Discord streaming feature.

## 13. Checkpoint Schema

Current checkpoints use schema version `3.0`.

Top-level shape:

```json
{
  "schema_version": "3.0",
  "importer_version": "string",
  "import_analysis": null,
  "session": {},
  "player_primer": "string",
  "world_state": {},
  "characters": [],
  "session_conversation": [],
  "narrator_conversations": {},
  "character_conversations": {},
  "canonical_events": [],
  "transcript": [],
  "visibility_log": [],
  "config": {}
}
```

Important notes:

* `canonical_events` stores full `EventRouterOutput` objects.
* `transcript` is primarily display/audit state, but takeover/personality
  synthesis flows may read recent transcript entries as authoring context.
* `session_conversation` is the router's rolling history.
* D&D Cat II roll transactions are checkpoint/audit state, not router rolling
  history.
* `narrator_conversations` are per human POV.
* `character_conversations` are per character.
* `world_state.locations` has no runtime topology.
* `CharacterRecord.location` is an opaque continuity label.
* `player_primer` replaces authored opening prose; opening context is
  generated by the router on `(begin)`.
* `CheckpointManager` hard-gates schema version. Checkpoints whose
  `schema_version` does not match the current schema fail to load rather than
  being migrated automatically.

## 14. Prompt Strategy

Prompt files are source. Current prompt files include:

* `event_router.txt`
* `agent.txt`
* `narrator_phase2.txt`
* `character_gen.txt`
* `takeover.txt`

Prompt rules:

* use XML-style tags for major sections
* avoid redundant markdown headers immediately inside tags
* avoid implementation details the LLM does not need
* provide input/output contracts, not pipeline internals
* do not tell a model to remember its own history
* do not add tests that freeze approved prompt prose
* do add tests for runtime contracts and forbidden prompt leakage
* keep exact mode markers in `turn_loop_contracts.py` and test rendered
  helper output rather than hand-copying marker strings through callers

## 15. Error Handling

LLM failures:

* transient API errors are retried with exponential backoff and jitter
* permanent schema/prompt errors raise
* a router or narrator failure aborts the turn before checkpoint commit
* individual off-stage tick failures are logged and swallowed
* a failed tick fan-in router call does not erase the already-rendered
  on-stage turn

Structured output:

* event router and narrator calls use Pydantic response models
* all fields in `EventRouterOutput` are required to keep the structured
  output grammar small
* schema validators assign missing event ids, coerce unknown beat reasons
  where safe, enforce Cat II invariants, and enforce fact visibility
  membership

Discord failures:

* turn-message tracking failures are logged but do not fail the engine
  turn
* rewind cleanup falls back from delete to edit/hide when Discord delete
  fails

## 16. Observability

The system exposes:

* per-role logging
* router decision rationale in logs and checkpoint conversation history
* checkpoint snapshots per turn
* canonical event logs
* per-character rolling conversations
* per-POV narrator conversations
* import preservation analysis
* tracked Discord turn-message refs

The router `decision_rationale` field is temporary diagnostic overhead.
When router behavior is stable enough, remove the schema field, prompt
field, and log plumbing together.

## 17. Current Gaps And Maintenance Notes

Known stale or transitional areas:

* some code comments still reference older architecture names or call counts,
  especially around the early v11 turn-loop skeleton, importer call naming,
  and hidden-context comments
* `SessionConfig.models.discriminator` remains for compatibility even
  though the active LLM config no longer calls a discriminator role
* `character_gen` remains a configured model role, but the live spawn path
  currently calls `role="agent"` while rendering `character_gen.txt`
* `visibility_log` exists but the main v11 flow relies on
  `canonical_events`, render buffers, and NPC inboxes
* global `transcript` stores one selected POV per beat; other POV prose
  lives in `narrator_conversations`
* `/query` is implemented as a mutating router/narrator turn, not a
  read-only information endpoint
* off-stage ticks are synchronous and stagnation-triggered, not yet a
  generalized world-clock or faction-clock system
* debug streaming and public HTTP APIs are not implemented
* prompt version ids are not stored in checkpoints; git history is the
  version source

These are acceptable as long as the design doc names them honestly and
tests cover the live contracts rather than preserving dead architecture.

## 18. Acceptance Criteria

The current engine is healthy when:

1. A bound player can submit `/act` and receive a coherent POV render.
2. Multiple bound players can receive separate POV renders from the same
   beat.
3. Cat I actions close without stealing another actor's response.
4. Cat II actions render the attempt and wait for required responders.
5. NPC observers receive only facts visible to them.
6. Remote or mediated observers are listed only when a concrete perceptual
   channel makes the event available to them; private partial channels use
   scoped `audience="only"` facts.
7. NPC agents do not receive another agent's parenthetical intent.
8. The narrator renders from visible observable facts without adding
   unsupported action or attitude.
9. Router-created spawns, dormancy, and culls persist
   to checkpoint state.
10. `/query` answers through the router/narrator path without leaking
    knowledge outside the querying POV.
11. Off-stage ticks can advance eligible NPCs and land their public results
    as router-canonicalized events.
12. `/rewind` removes later checkpoints and cleans up tracked Discord
    turn messages.
