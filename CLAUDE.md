# Narrative Engine

## Dev Environment

- Use `.venv/bin/python` and `.venv/bin/pytest` directly instead of sourcing the venv activate script.
- API key is in `.env` as `ANTHROPIC_API_KEY`. Never commit `.env`.
- LLM: Anthropic Messages API via the `anthropic` Python SDK. Player sessions default to `claude-sonnet-4-6`; tests default to `claude-haiku-4-5`.

## Project Structure

- `app/engine/` — turn pipeline: orchestrator, event router, narrator, character agents, context builders
- `app/schemas/` — Pydantic data models (checkpoint, session state, requests, etc.)
- `app/llm/` — Anthropic SDK wrapper (caching, compaction, structured output)
- `app/prompts/` — versioned prompt templates (event_router_v4, narrator_phase2_v5, agent_v6, etc.)
- `app/bot/` — Discord frontend: slash commands, EngineBridge, session map, embed rendering
- `app/storage/saves/` — checkpoint save files, one dir per session
- `scripts/play.py` — interactive terminal REPL (alternative frontend, supports multi-character play)
- `scripts/import_story.py` — CLI wrapper for the import pipeline
- `tests/` — pytest tests
- `DESIGN.md` — full design document

## Open architectural concerns (read this before redesigning core systems)

These are real, known sharp edges that have NOT been solved. If you are
touching the tick scheduler, the rolling agent conversations, or
anything that times the world (turn counters, scene transitions,
async/multi-scene play), you must read the relevant entry first or you
will silently make the problem worse.

### Coherent world time across asynchronous, multi-scene play (TODO)

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
  victory in scene B) needs an ordering primitive richer than per-
  scene `last_event_at`.
- "Did this happen before or after that?" gets answered differently
  depending on which scene's perspective you ask from.

We have NOT solved this. If you are reaching for a simple turn
counter to gate behavior, ask whether that gate makes sense across
two scenes running concurrently. If the answer is "no" or "I don't
know," surface it on a TODO and prefer a per-scene counter (with an
explicit cross-scene reconciliation step) over a session-global one.

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
race conditions when two players act in quick succession, the
`scene_locks` discipline, and what happens when a tick fan-out is
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
