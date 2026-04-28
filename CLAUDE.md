# Narrative Engine

## Dev Environment

- Use `.venv/bin/python` and `.venv/bin/pytest` directly instead of sourcing the venv activate script.
- API key is in `.env` as `ANTHROPIC_API_KEY`. Never commit `.env`.
- LLM: Anthropic Messages API via the `anthropic` Python SDK. Player sessions default to `claude-sonnet-4-6`; tests default to `claude-haiku-4-5`.

## Project Structure

- `app/engine/` — turn pipeline: orchestrator, event router, narrator, character agents, context builders
- `app/schemas/` — Pydantic data models (checkpoint, session state, requests, etc.)
- `app/llm/` — Anthropic SDK wrapper (caching, compaction, structured output)
- `app/prompts/` — prompt templates (`event_router.txt`, `narrator_phase2.txt`, `agent.txt`, etc.); versioning lives in git, not in filenames (use `git log <file>` to see history)
- `app/bot/` — Discord frontend: slash commands, EngineBridge, session map, embed rendering
- `app/storage/saves/` — checkpoint save files, one dir per session
- `scripts/play.py` — interactive terminal REPL (alternative frontend, supports multi-character play)
- `scripts/import_story.py` — CLI wrapper for the import pipeline
- `tests/` — pytest tests
- `DESIGN.md` — full design document

## Long-term goals (vision, not tickets)

Direction for the engine over months, not commits. Items here are not
"do next" — they're the questions worth keeping in your head whenever
you make a change that touches the relevant subsystem, so you can
notice when an opportunity to chip at one of them shows up cheaply.

### LT-1: Long-term narrative planning inside the engine

Today the router carries the entire load of "what should happen next."
That's fine for short-horizon adjudication (this beat, the next two
turns) but the router has no notion of a multi-act arc, no concept of
"introduce a new character at hour three of the dating show," no
mechanism to plant Chekhov's gun in turn 5 and fire it in turn 40.
The dating-show story would naturally want to bring in fresh faces
mid-season; the survivor story would want a tribal council cadence;
hollowstone would want the conspiracy to advance off-screen even when
the player is camped in the courtyard.

Possible directions (none chosen, none scoped):

- A separate "showrunner" LLM that runs every N beats and emits
  high-level intents for the router to thread into adjudication
- A persistent "story arc" object on the checkpoint with explicit
  tension/pacing/reveal targets the router consults
- An importer-time arc skeleton (acts, reveals, beats-to-trigger)
  that runtime advances against
- A periodic "casting director" pass that proposes new characters
  to spawn based on roster gaps the LLM identifies

Whatever path we pick, keep the router prompt tax-aware: the router
is already handling adjudication, perception, and roster decisions on
every turn. Adding "long-term planning" to its system prompt without
offloading would push it past its current sweet spot. A separate
actor with a longer cadence is the most likely shape.

### LT-2: Spawn-vs-canonicalize discipline beyond prompt-rules

v11-r7i tightened the router prompt to prefer canonicalize-with-
observer over spawning for one-shot utility characters (couriers,
walk-ons, crowd). The discipline currently lives entirely in the
router's system prompt — there is no engine-side cost or rate-limit
on minting characters beyond `MAX_SPAWNS_PER_TURN=3`. As stories run
longer (50+ beats), even a well-disciplined router can let the
roster bloat to 30+ characters that the next router prompt has to
re-summarize on every call.

Open questions worth keeping in mind:

- Should spawned characters have a "TTL" — auto-dormant after N
  beats with no participation, recoverable on demand?
- Is there a meaningful distinction between "named, plot-relevant
  character" and "ambient world fixture" we should encode in the
  schema, with different cost/visibility profiles?
- Should the router get a "current roster size + recent spawn
  rate" signal so it self-throttles, instead of relying purely on
  prompt language?

Don't solve preemptively — wait for a playtest where the roster
genuinely bloats and shapes the answer.

## Engineering discipline

### Vestigial-field destruction policy

When you add a field to a Pydantic schema, you take on the obligation
to keep it populated and read by ONLY actors authorized to see it.
When a field stops being populated by any code path (or stops being
read by any code path), it becomes a hazard: its value lies on disk
in old saves, it shows up in serialization, it tempts readers into
trusting it, and it accumulates documentation that explains the
1.0 design rather than the live system.

Rule:

1. When you remove the LAST writer of a field, in the SAME commit you
   remove the field from the schema. Don't leave the field behind
   "for back-compat with old saves." Pydantic v2's default
   `extra='ignore'` silently drops the legacy field on load — you do
   not need a deprecation flag for that.
2. When you remove the LAST reader of a field, in the SAME commit you
   remove either the field OR the writers. A write-only field is dead
   freight on every checkpoint serialization.
3. When you change a field's semantics (e.g. "this used to mean X,
   now it means Y"), rename it. Keeping the same name and changing
   the meaning poisons every blame, every reviewer hand-off, every
   future search.
4. When in doubt, list the field under "Vestigial schema fields" in
   `Open architectural concerns` BEFORE the change ships, so the
   next contributor knows not to trust it on read.

The v11 cycle hit two of these the hard way:
the old global location field was set at import and never updated, but
every reader trusted it; `TurnResponse.debug` had a full
schema with no orchestrator writer at all. Both wasted reviewer time
and one of them silently misled a 31-turn playtest summary. Don't do
this again.

## Open architectural concerns (read this before redesigning core systems)

These are real, known sharp edges that have NOT been solved. If you are
touching the tick scheduler, the rolling agent conversations, or
anything that times the world (turn counters, contextual transitions,
parallel play), you must read the relevant entry first or you
will silently make the problem worse.

### Coherent world time across asynchronous play (TODO)

Today the engine carries two distinct notions of "time":

- `session.turn_index` — narrative time. Advances on every closed
  beat (both `process_turn` and `resolve_cat_ii` increment it). This
  is what player-facing transcripts and history are keyed off.
- `session.turns_since_last_tick` — world ticks. Advances ONLY in
  `process_turn`'s tick scheduler block, never in `resolve_cat_ii`.

In single-player single-scene play these stay close enough to be
indistinguishable. As soon as you have multiple players acting in
multiple scenes asynchronously (the design target), they drift:
two players each running their own beats in two scenes both bump
`turn_index`, but the tick clock fires off whichever scheduler ran
last, and "world time" becomes whatever the engine happened to
observe most recently. There is no shared monotonic "world clock"
for off-stage NPCs to reason about.

This matters because:

- Off-stage NPC stagnation triggers (`tick_stagnation_max`) are
  measured in `turns_since_last_tick` — under multi-player load
  that's not a faithful "the player has been camping for N turns"
  signal.
- Cross-scene causality (an antagonist in scene A reacts to a player
  victory elsewhere) needs an ordering primitive richer than a single
  local `last_event_at`.
- "Did this happen before or after that?" gets answered differently
  depending on which participant's perspective you ask from.

We have NOT solved this. If you are reaching for a simple turn
counter to gate behavior, ask whether that gate makes sense across
parallel action. If the answer is "no" or "I don't know," surface it
on a TODO instead of adding a brittle global counter.

### Tick fan-out latency / concurrency (TODO)

`Orchestrator._run_ticks` runs synchronously on the critical path
of `process_turn` — every eligible off-stage NPC's tick is awaited
before the player gets their render back. With a roster of N
intentions-enabled NPCs and concurrency cap C, a tick-fire turn
costs the player roughly `ceil(N/C) * agent_latency` extra wall
time on top of the on-stage beat. With Sonnet/Haiku that's typically
1–4 seconds; for hollowstone-sized rosters it can spike higher.

The right fix is to batch tick fan-out with the next router call
using async synchronization primitives — fire ticks immediately
after the on-stage beat closes, but don't make the player wait for
them; await them inside the next `process_turn`'s router prep so
that the next router call sees their outputs without the current
player's render blocking on them. This needs careful design around
race conditions when two players act in quick succession, session-level
act-slot locking, and what happens when a tick fan-out is
still in flight at checkpoint-save time.

For now: synchronous fan-out is fine, but every change that adds
work inside `_run_ticks` (more LLM calls, deeper context builds,
extra serialization) is paid by the player on tick-fire turns.
Measure before you add.

### Continuity across in-flight code patches (watch item)

`character_conversations[character_id]` is a rolling list of raw
user/assistant message dicts that the agent's next call REPLAYS
verbatim. Old assistant entries from pre-v11 sessions used the
structured-output schema (JSON-shaped agent replies); v11 agents
emit prose + trailing parenthetical and are NOT instructed to
ignore legacy JSON shapes in their own history. If we ever resume
a session whose conversation was written by an older agent prompt,
the new agent will see those legacy entries on replay and may
imitate them — silently regressing format and tone.

This is not a current production problem (we don't yet ship sessions
through prompt-version boundaries), but it becomes one the moment
we do. Two defensive moves are cheap: (a) tag each appended
assistant message with a prompt version, and on resume either
filter or rewrite anything older than the current generation; or
(b) add a one-line "format reminder" to the system prompt so the
LLM ignores legacy shapes regardless. Pick one before the first
patched-mid-session resume goes out.

### Vestigial schema fields (do not trust on read)

- The old global location field is gone from `LocationState` and
  `LocationsExtraction`. `CharacterRecord.location` is an opaque
  continuity label; perception is observer-driven.
- ~~`TurnResponse.debug` (DebugPayload)~~ — **REMOVED in v11-r7j.**
  Field is gone from `TurnResponse`; `TurnRequest.debug` /
  `TurnRequest.debug_flags` are also gone. Per-turn router rationale,
  agent outputs, latency, cache stats, and canonical events live in
  the engine logger (`turn_loop.router[route] …` lines) and the
  per-turn checkpoint files. Old turn requests on the wire that still
  set `debug=true` load cleanly via Pydantic v2's `extra='ignore'`
  (covered by `test_legacy_debug_fields_silently_dropped` and
  `test_legacy_debug_payload_silently_dropped` in `tests/test_schemas.py`).

### Cross-scene communication ✅ WIRED in v11-r7j

`broadcast_event` in `app/engine/turn_loop.py` now appends a
one-line `[off-scene perception] …` entry to every NPC observer
who is NOT in the broadcast scene (and not a human-bound character).
Router rule 13's promise — "add the recipient as an `observer` on
this same event so the recipient's character agent perceives the
message arriving" — now actually lands as engine state: when the
courier-and-note event closes in the courtyard with Marcus listed
as an observer in the citrus_garden, Marcus's
`pending_observations` grows by one line, and his next agent call
flushes that line into his prompt via the existing
`format_pending_observations_block` path.

In-scene NPC observers are intentionally NOT pushed: they are
eligible to be picked into the same beat via
`agent_responder_picks` and read the canonical event live through
their normal context block. Pushing onto their inbox would
double-count the event the next time they fire.

Pre-r7j confirmed via villa playtest turn 13: Jordan asked a
runner to deliver a message to Marcus in citrus_garden; the
message rendered in narration, Marcus's `pending_observations`
stayed empty, and the message functionally never reached him. This
class of bug is closed.

Two remaining knobs the v11-r7j wiring intentionally left alone,
either of which could be tightened later if playtest evidence
appears:

- The summary line is the canonical event's `resolved_outcome`
  (or first `observable_facts` line as fallback). It is verbatim
  router prose, not POV-rewritten for the recipient. For most
  cross-scene messages this is the right shape — the recipient
  perceives "what happened, briefly." For sensitive material
  (whispered threats, secret deliveries) the router can shape the
  outcome line to match what the recipient would actually
  perceive.
- Inbox entries persist across turns until the recipient is asked
  to respond. A cross-scene NPC who never gets called burns inbox
  growth. The /settings registry exposes
  `agent_history_max_turns` for the rolling conversation cap; if
  inbox bloat shows up in long sessions, a parallel cap on
  `pending_observations` length is the obvious knob.

### Spawn discipline (and the canonicalize-vs-mint distinction)

**Engine-side (resolved in v11-r7i):**

- `CharacterManager.spawn_characters` dedups by `character_id`
  within a single batch. Duplicate ids land as a warning + drop, not
  as multiple roster records sharing one id.
- The orchestrator passes the acting actor's location label as
  `acting_actor_location`; spawns whose `seed.location` is empty
  use that label. Router-supplied `seed.location` still wins.
- Covered by `test_spawn_dedups_within_batch`,
  `test_spawn_uses_acting_actor_location_when_seed_omits`, and
  `test_spawn_seed_location_beats_actor_location`.

**Prompt-side (v11-r7i `event_router.txt` rule 10 + rule 13):**

- The router prompt now distinguishes "spawn a real agent" from
  "canonicalize the event with an observer." Spawn is for
  characters who will keep acting; one-shot couriers, walk-ons, and
  plot-utility figures should be written into `resolved_outcome`
  with the recipient/witness added as an `observer`. Rule 10 calls
  out re-using existing in-scene NPCs as the first preference.
- The villa playtest spawned three `production_runner` agents to
  deliver one note. Under the new shape that's a one-line event:
  *"Jordan presses the apology note into a passing runner's hand;
  the runner carries it to Marcus's table"* — Marcus added as
  observer, no spawn needed.

**Resolved (v11-r7j)**: `pending_observations` is now populated by
`broadcast_event` for every off-scene NPC observer on a canonical
event. See the "Cross-scene communication ✅ WIRED in v11-r7j"
entry above for details. The router rule 13 promise now matches
the engine behavior: declaring a recipient as an `observer`
actually delivers a perception signal regardless of co-location.

### Per-character interior asymmetry is load-bearing

The trailing parenthetical at the end of each agent response is
the agent's interior — its plans, its read of the situation, its
honesty with itself. It is appended verbatim to the agent's own
rolling history (so its future calls inherit continuity) and
deliberately NOT mirrored anywhere else. The router decides who
acts on PUBLIC signals (the agent's prose intentions, prior
canonical events, importer-seeded objectives). The narrator sees
externally-observable beat events. Other agents see other agents'
public prose (parens stripped).

If you find yourself adding a `last_intent`-style mirror field on
`CharacterRecord`, or piping an agent's parenthetical into the
router/narrator/another agent's context, stop. That asymmetry is
the entire reason we have separate per-actor LLM calls; collapsing
it back gives you a worse single-LLM-pretending-to-be-many. Surface
the information through in-fiction signals (a courier walks in, a
note is found, a witness sees an action) instead.
