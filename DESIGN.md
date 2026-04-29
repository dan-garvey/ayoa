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

## 2. Goals

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

### 4.5 Agents Author Intentions, Not State

Character agents produce free-form public prose plus one trailing
parenthetical containing their private intent. The public prose becomes
the next intention that the router canonicalizes. The parenthetical stays
only in that agent's rolling history and is stripped at every cross-role
boundary.

### 4.6 Checkpoints Are The Source Of Truth

Every committed turn writes a checkpoint. The runtime can be rebuilt from
checkpoint JSON plus the prompt/code version in git. Process memory,
locks, and API caches are not trusted as durable state.

## 5. Runtime Components

### 5.1 Discord Bot And EngineBridge

`app/bot/commands.py` defines the slash-command surface:

* `/session start|resume|list|end`
* `/story list|info|start|import|delete`
* `/join`, `/begin`, `/leave`, `/describe`
* `/act`, `/query`, `/status`, `/settings`
* `/rewind`

The bot calls the engine in process through `EngineBridge`; there is no
FastAPI layer in the current runtime. `SessionMap` stores Discord channel
to session mappings, private POV thread mappings, and turn-message refs
used by rewind cleanup.

### 5.2 Orchestrator

`Orchestrator.process_turn()` is the main turn entry point. It:

1. loads the latest checkpoint
2. resolves the acting character
3. acquires a per-session lock
5. checks active act slots
6. calls `run_beat()`
7. applies character lifecycle changes and spawns
8. appends one transcript entry for the beat
9. runs eligible off-stage ticks
10. increments the turn index and saves
11. returns a `TurnResponse`

### 5.3 Turn Loop

`run_beat()` in `app/engine/turn_loop.py` is the v11 state machine. It
supports fresh human actions, Cat II responder actions, NPC cascades,
observation harvest, partial renders for contested attempts, max-event
backstops, and per-human render fan-out.

Important state:

* `active_act_slots`: per-beat lock state for initiators and Cat II
  responders
* `open_cat_ii_events`: contested events awaiting responders
* `render_buffers`: per-human queues of canonical event ids waiting for
  narrator render
* `canonical_events`: append-only log of closed canonical events

### 5.4 LLMDispatcher

`LLMDispatcher` binds the abstract turn-loop protocol to the live roles:

* `route_intention()` calls the `event_router` prompt
* `route_tick_intentions()` bundles off-stage tick outputs into one
  router call
* `agent_intend()` calls `CharacterAgent.respond()`
* `harvest_perceptions()` calls `CharacterAgent.perceive()` in parallel
* `narrator_compose()` calls `compose_pov_render()`

It also owns the router context-trimming calls that surface initial
rosters, world-fact deltas, and pending state changes.

### 5.5 Event Router

The event router prompt and `EventRouterOutput` schema replace both the
old narrator adjudication phase and the old discriminator role. The
router emits one structured object per routed intention or tick fan-in.

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
parenthetical. The engine parses that into:

```json
{
  "character_id": "rashid",
  "public_text": "Rashid sets his glass down. 'Say that again.'",
  "intent": "force the claim into the open without revealing his own source"
}
```

Only `public_text` leaves the agent. `intent` remains private continuity
inside that character's own history.

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

Roster moves are applied by the orchestrator because movement has to
respect act-slot, pinned-character, and player-bound guards.

### 5.9 Context Builder And Prompt Manager

`context_builder.py` owns shared context formatting and visibility-aware
helper logic. `PromptManager` loads `app/prompts/{name}.txt`, expands
partials, strips HTML comments, splits system/user sections on
`<<<USER>>>`, and renders rolling conversations.

Prompt templates are versioned in git. They are not copied into
version-suffixed filenames, and checkpoint JSON does not store prompt
version ids.

### 5.10 LLM Client

`LLMClient` is the provider boundary for live model calls. Callers send
role, messages, sampling settings, and an optional Pydantic response
model; the client selects the configured provider/model for that role
and normalizes the result back into `LLMResponse`.

It supports:

* per-role provider and model selection
* Anthropic Messages and OpenAI Responses adapters
* Pydantic structured output normalized into `response.parsed`
* Anthropic prompt caching and server-side compaction
* per-role Anthropic extended-thinking budgets
* provider-specific retry handling for transient failures

Provider/model selection can be configured with model prefixes such as
`openai:gpt-5.4-mini`, explicit `role_providers`, or environment
overrides like `LLM_PROVIDER_NARRATOR=openai` and
`LLM_MODEL_NARRATOR=gpt-5.4-mini`.

Active model roles include `event_router`, `narrator`, `agent`,
and `character_gen`. The old
`discriminator` role is vestigial and is not used by live calls.

## 6. Turn Lifecycle

### 6.1 Fresh Action

1. A player submits `/act`.
2. `EngineBridge.run_turn()` sweeps stale Cat II pins before processing
   the new action.
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
* dispatches the first valid NPC pick and routes that agent's intention
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
6. If any human is pending, the narrator renders partial-mode prose and
   the beat pauses.
7. When the human responds, the router receives the initiator and
   responder intentions and emits the resolved canonical event.
8. After a Cat II event resolves, an NPC initiator gets the first
   follow-up turn when applicable.

### 6.4 Broadcast

`broadcast_event()` appends the router output to `canonical_events` and
fans it out:

* local human observers get render-buffer entries
* mediated human observers get render-buffer entries only when facts name
  them explicitly
* local NPC observers get their visible facts appended to
  `pending_observations`
* mediated NPC observers get only facts that name them in `visible_to`
* the event actor does not receive their own action as an inbox entry

Broad `all_observers` facts do not cross a physical/channel boundary by
themselves. Remote observers need scoped facts.

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
* movement facts such as "X arrived" or "X left"
* a moved NPC's own action marker

The inbox is drained into the agent's next on-stage or tick prompt.

### 8.3 Private Continuity

Mutable interior continuity lives in each agent's rolling conversation,
primarily in the private trailing parenthetical. It is not mirrored onto
the character record as `private_updates`, `last_intent`, or directives.

## 9. Context Management

Context is deliberately one-shot or delta-based where possible.

Router context:

* system prompt contains stable role contract and schema contract
* initial NPC roster appears only on the first router call
* world facts are surfaced once through a delta tracker
* importer-seeded continuity context is surfaced once
* router-authored context relies on router history
* external engine changes surface through `pending_router_state_changes`
* actor inbox entries surface through "Arrived For You Since Last Turn"

Agent context:

* stable character identity is in the system prompt
* per-turn user message carries pending observations and mode body
* there is no repeating "Characters Present" block
* local arrivals and exits arrive through the observation inbox

Narrator context:

* each human POV has a separate rolling narrator conversation
* the narrator gets visible facts and observation tags
* agent responses are already folded into those facts

## 10. Dynamic Character And Arrival Flows

### 10.1 Router Spawns

The router may emit `spawn` requests for meaningful recurring
characters. `CharacterManager.spawn_characters()` calls `character_gen`
to create the record, then queues a compact router summary into
`pending_router_state_changes` so the next router call knows what the
engine added.

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

## 11. Discord Session Behavior

Discord is the primary UX.

* Sessions are mapped to channels.
* Human players are mapped to character ids.
* Private POV delivery uses private threads where possible, falling back
  to DMs.
* `TurnResponse.per_player_renders` is delivered to the relevant
  players.
* Each posted turn message is tracked in `SessionMap`.
* `/rewind` deletes later checkpoints and deletes or hides tracked
  Discord messages for the deleted turns.

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
* `transcript` is display/audit only and is not fed back into prompts.
* `session_conversation` is the router's rolling history.
* `narrator_conversations` are per human POV.
* `character_conversations` are per character.
* `world_state.locations` has no runtime topology.
* `CharacterRecord.location` is an opaque continuity label.
* `player_primer` replaces authored opening prose; opening context is
  generated by the router on `(begin)`.

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

* some legacy tests and shim classes still reference the old
  `EventRouter` / single narrator flow
* `SessionConfig.models.discriminator` remains for compatibility even
  though the active LLM config no longer calls a discriminator role
* `visibility_log` exists but the main v11 flow relies on
  `canonical_events`, render buffers, and NPC inboxes
* global `transcript` stores one selected POV per beat; other POV prose
  lives in `narrator_conversations`
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
6. Remote observers receive only facts scoped through a concrete live
   channel.
7. NPC agents do not receive another agent's parenthetical intent.
8. The narrator renders from visible observable facts without adding
   unsupported action or attitude.
9. Router-created spawns, dormancy, and culls persist
   to checkpoint state.
10. `/rewind` removes later checkpoints and cleans up tracked Discord
    turn messages.
