# Pending: Agent Output Rework + Background Tick Wiring

Working notes from the design conversation. Decisions are committed; review items are deferred until playtest tells us they matter.

---

## Why this exists

The 30-turn playtest found two structural problems:

1. **Background ticks have never fired in v11.** `CharacterAgent.tick()` exists; nothing in `app/` calls it. The "world feels static" symptom traces back to here. If the antagonist can only act when the player is on screen, the player wins by avoiding the antagonist.
2. **The router is blind to character interior.** It picks responders and decides movement using only `name + role + location`. No goals, no objectives, no recent intent. That's why NPCs don't show up when they should and why the world doesn't react to player actions in narratively-aligned ways.

Investigating these surfaced a third problem: most of `CharacterAgentOutput.private_updates` is dead in v11. The schema requires it, the prompt asks for it, nothing consumes it. `current_objectives`, `directives_sent`, and `moved_to` are all dead. Only `public_response.{actions, dialogue, expression}` is read downstream, and even that is just woven back into prose by the narrator.

---

## Decisions (do these)

### 1. Drop structured output from agents

Replace `CharacterAgentOutput` with prose + trailing parenthetical:

```python
class CharacterAgentOutput:
    character_id: str
    public_text: str   # prose; actions in third person, "dialogue" in quotes, nonverbal expression inline
    intent: str        # contents of the trailing parenthetical; engine + router only, never narrator or other agents
```

Drop entirely: `PublicResponse`, `PrivateUpdates`, `DirectiveSend`, `moved_to`, `directives_sent`, `current_objectives` as a writeback target.

Agent emits prose followed by a single trailing `(...)`. Engine's parser extracts only the LAST trailing parenthetical group; warn (don't fail) if absent. Mid-prose parentheticals as stage directions stay in `public_text`.

### 2. Agent prompt discipline

Short responses by default — two to four sentences. Long-form only when the character is genuinely doing long-form (sermon, deposition, formal proposal). Every response ends with one parenthetical.

Tick-prompt urgency rule:

> You are off-stage. The world will move on without you in a moment, so make this beat count. Act, don't deliberate. If you're communicating with someone, write a single exchange — what you said, what they likely said back — not the start of an ongoing dialogue.

### 3. Parenthetical never reaches narrator OR other agents

Strip happens at one chokepoint, parse-time. `public_text` and `intent` are stored separately on the rolling conversation entry too:

- `character_conversations[id]` keeps **both** (full text including parenthetical) so the agent's own future self sees its prior interior. This is how dynamic goal pursuit works without explicit objectives field.
- Other agents' `prior_responses` cascade input — `public_text` only.
- Narrator phase-1 / per-POV cascade input — `public_text` only.
- Router intention serialization — `public_text` only. Interior is delivered separately via the router's character context (see #5), NOT inline in the intention string.

### 4. Kill directives entirely

Drop `incoming_directives`, `directives_sent`, `IncomingDirective`, `DIRECTIVE_DEPTH_*` constants, all the depth machinery. NPC-to-NPC coordination happens implicitly:

- Two off-stage characters with intents toward each other tick in the same fire. The unified router (see #6) reads both intents and emits a single canonical event capturing the exchange.
- Conversations resolve in one router pass per tick. No back-and-forth fan-out within one tick. If they want to keep talking, the next tick (5+ turns later) lets them compose another off-screen scene.
- Length is enforced by the urgency rule above, not by prompt politeness.

### 5. Router gets character interior — initial state in turn-1 user message

Today the router sees only identity (name, id, role, location, status). This is the single biggest router-quality miss in v11.

**REVISED (post-Commit 5 review)**: the original plan added `last_intent` / `last_intent_turn` mirror fields on `CharacterRecord` so the router could see each NPC's freshest parenthetical. That mirror has been **ripped** — see `CLAUDE.md` "Per-character interior asymmetry is load-bearing." The router does NOT get to see any agent's parenthetical. Period.

What the router DOES get on the turn-1 "## Initial Roster" block:
- `goals` (long-term, importer-seed, static)
- `current_objectives` (importer-seed, static — agents do NOT write this)

After turn 1, the router infers interior pressure from public signals only: cascade `intends:` text, prior canonical events, and its own prior character-lifecycle outputs. There is no "freshest interior" channel from agents to the router. If you find yourself wanting one, surface it in fiction (a courier walks in, a witness sees an action) rather than as a private-data leak.

### 6. Background ticks go through the SAME router

One router, one `session_conversation`. Tick fires produce a router call with a different user-message framing:

- **On-stage call**: user message frames "actor X attempts Y" -> router emits canonical event + observers + picks + character lifecycle changes.
- **Tick call**: user message frames "these N off-stage characters ticked, here are their intents" -> router emits off-screen canonical events plus any needed lifecycle changes. No narrator focus.

Both share the same `EventRouterOutput` schema; tick calls produce a narrower subset. The system prompt rule for tick mode is a partial that activates when the user message contains the tick framing.

The router's history is unified: it sees its own on-stage decisions AND its own tick decisions interleaved. State the router authored is in history regardless of which call mode emitted it. No separate `tick_router_conversation`.

### 7. Trim `character_registry` from per-turn user message

`character_registry` (full roster, ~80 tokens × all NPCs, ~1000 tokens/turn for hollowstone) is the bloat. Router has it from history (turn-1 inject + spawn outcomes). Drop entirely from the per-turn user message.

`world_facts` similarly — replace with `world_facts_delta` (only new facts since last call). Empty most turns.

Add to per-turn user message: "## State Changes Since Your Last Call" — surfaces things the router did NOT author:
- Spawn outcomes (full identity + summary for newly-spawned NPCs)
- Player /takeover and /join changes
- Anything else the engine applied that the router didn't decide

Empty in the common case (back-to-back on-stage routing in the same beat). Non-empty when ticks fired (handled implicitly by unified router; this block is for non-router origin only) or when something exotic happened.

### 8. Spawn flow generates a router-consumable summary

Today `char_mgr.spawn_characters` calls a separate LLM (`character_gen_v3`) to produce full backstory/voice/personality/goals. The router never sees the result.

Update the spawn-generation LLM prompt to ALSO produce a short `router_summary: str` field — one or two sentences capturing identity + role + initial intent. The new character's `router_summary` lands on the next router call's "## State Changes Since Your Last Call" block. After that, it's in router history.

### 9. Background tick scheduler

Trigger model:

```python
session.turns_since_last_tick += 1
stagnation_fires = (
    session.turns_since_last_tick >= session.tick_stagnation_max  # default 15
)

if stagnation_fires:
    await _run_ticks(...)
    session.turns_since_last_tick = 0
```

Drop `tick_cadence` / `tick_turn_counter`. Add `turns_since_last_tick` and `tick_stagnation_max=15`.

Eligibility filter (port from pre-v11):
- `private_state.intentions_enabled` is True
- `status == active`
- `character_id NOT in player_ids`
- `character_id NOT in acted_this_turn`
- `character_id NOT in _pinned_character_ids(ckpt)` (v11 addition: pinned NPCs are mid-Cat-II, ticking races resolution)

Concurrency: `asyncio.Semaphore(min(tick_concurrency, TICK_CONCURRENCY_HARD_CAP))` — defaults 4 / 16.

Call site: `Orchestrator.process_turn`, between `_append_transcript_entry(...)` and `ckpt.session.turn_index += 1`. Single save covers both narrator and tick state.

After tick fan-out, bundle N agents' `(name, public_text, intent)` triples into ONE call to the unified router (see #6).

---

## Review items (don't act yet — let playtest tell us)

These are concerns worth tracking but not worth solving until evidence shows they matter.

### R1. State-tracking reliability without `character_registry` re-feed

We don't know empirically how reliably the router tracks state across N turns of its own outputs without the registry rescue. If turn 30's router has lost track of who moved where on turn 14 because attention slipped, the trim was premature.

Mitigations to consider only if observed:
- Periodic anchoring: every K turns, re-inject a "## Current Roster Snapshot" block.
- Disable compaction on the router specifically (compaction summarizes old assistant messages where granular state mutations live).

### R2. Whether to restore a compact roster reminder

Today the router relies on its initial roster and rolling history. If picking quality regresses, consider a compact reminder with goals/objectives, but avoid restoring repeated location or presence dumps.

### R3. Spawning trivial NPCs

The current spawn flow runs a full character-generation LLM call for every spawn — full backstory, personality, voice. For minor NPCs (a guard at a gate, a passing servant, a one-line shopkeeper), this is heavy.

Worth exploring: a "lightweight NPC" path where the router names them as part of the canonical event without generating a full character record. Or a tiered spawn (full vs. silhouette). Or letting the narrator render unnamed bystanders without instantiating them at all.

Don't design until we see actual cases where the heavy spawn pattern is hurting us.

### R4. World-facts delta vs. full

The trim swaps `world_facts` for `world_facts_delta`. If world facts mutate frequently (some genres might do this; others won't), the delta calculation might be wrong-shaped. Defer until playtest shows the rate of change.

### R5. Importer alignment with opening_directive

Mira's turn-2 self-move in the playtest was caused by the importer placing her at `archive_main_hall` while the opening_directive said she should be at `bell_of_arrivals`. Importer should reconcile starting locations with the opening_directive at import time so turn 1 doesn't need to repair it in prose.

---

## Implementation plan

Commit ordering matters because some commits rely on others.

### Commit 1 — Agent prose-output schema swap
- `CharacterAgentOutput` → `{character_id, public_text, intent}`
- Add `_extract_parenthetical(text) -> (public_text, intent)` helper with last-trailing-paren extraction + warning on missing
- `CharacterAgent.respond` and `.tick` emit prose with trailing parenthetical; parser splits
- `agent_v9.txt` and `agent_tick_v2.txt` rewritten short (target ~8 rules instead of current ~22)
- Drop `PublicResponse`, `PrivateUpdates`, and `DirectiveSend` from `app/schemas/agents.py`
- Delete `_serialize_agent_intention` in `turn_loop_dispatcher.py`; dispatcher uses `output.public_text` directly
- Update `narrator.py:88-105` (phase-1 assembly) to consume `public_text` only as a single labeled paragraph per character (not bullet-broken)
- Update `context_builder.py:format_prior_responses` to use `public_text` only
- Update `validators.py:extract_text_from_output` to use `public_text` (validator is dead per CODE_REVIEW.md but keep it consistent)
- Update `turn_loop.py:_is_agent_refusal` to operate on `public_text`

### Commit 2 — Kill directives
- Drop `IncomingDirective`, `incoming_directives` from `CharacterRecord` and `PrivateState`
- Drop `directives_sent` references in `_build_since_last_turn_block`, `event_router.py:37-46`, `context_builder.py:389-406`
- Update `engine_bridge.py:682, 732-733` to drop directive references
- Remove directive blocks from `agent_v9.txt`, `agent_tick_v2.txt`
- Update importer (`story_importer.py`) and takeover (`takeover_v1.txt`) — directives never reach `CharacterRecord`
- Drop `DIRECTIVE_DEPTH_WARN`, `DIRECTIVE_DEPTH_CAP`, `DIRECTIVE_LENGTH_WARN` constants

### Commit 3 — initial roster + state-changes block (no `last_intent`)
**REVISED**: the original Commit 3 added `last_intent` / `last_intent_turn` mirror fields on `CharacterRecord` and surfaced them to the router. That mirror has been ripped (see Decision #5 above and `CLAUDE.md` "Per-character interior asymmetry is load-bearing"). DO NOT re-add it; the parenthetical lives in the agent's own rolling history, not on the character record.

The remaining Commit-3 work is what's actually shipped:
- Add `_build_initial_roster_block(ckpt)` — renders only when `session_conversation` is empty; identity + goals + current_objectives for every non-player character (NO interior parenthetical)
- Add `_build_state_changes_block(ckpt)` — surfaces spawn outcomes, /takeover, /join changes since last router call. Empty in the common case.
- Drop `_build_character_registry` from `_build_router_context`
- Replace `_build_world_facts` (full) with `_build_world_facts_delta` (only new facts)
- Update `event_router_v9.txt` user-template section to match the new variable set
- Router-prompt rule about character context: "Goals are long-term drives; current_objectives are importer-seeded active pursuits. The router does NOT see any agent's parenthetical; read interior pressure from the cascade `intends:` text and prior canonical events."

### Commit 4 — Spawn flow generates router summary
- Update `character_gen_v3.txt` to also produce `router_summary: str` (one or two sentences)
- Update spawn schema in `app/schemas/character_gen.py` (or wherever) to require the new field
- `char_mgr.spawn_characters` writes the summary to a session-scoped queue
- `_build_state_changes_block` reads the queue and surfaces spawn summaries on the next router call
- Queue clears after the router call consumes it

### Commit 5 — Background tick wiring
- Add `turns_since_last_tick: int = 0`, `tick_scene_change_cooldown: int = 5`, `tick_stagnation_max: int = 15` to `SessionState` / `SessionSettings`
- Mark `tick_cadence` / `tick_turn_counter` as deprecated (keep field for migration; new logic ignores)
- Add `Orchestrator._run_ticks(ckpt, acted_this_turn, acting_id) -> list[(char, output)]` — trigger logic + eligibility filter + concurrency-capped fan-out
- Wire call into `process_turn` between `_append_transcript_entry` and turn-index bump
- Surface per-tick `LLMResponse.usage` in the log line

### Commit 6 — Tick router fan-in (unified router, tick mode)
- Add tick-mode framing to the user message: "## Off-Stage Tick — N characters ticked\n{per-character public_text + intent + location}"
- Add `router_tick_mode_block.txt` partial in `app/prompts/_partials/` for the system-prompt-side guidance ("on tick calls, emit only off-screen canonical events plus needed lifecycle changes")
- Add `Orchestrator._route_tick_intentions(ckpt, tick_outputs) -> EventRouterOutput | None`
- Apply path: `apply_roster_updates`, append off-screen `canonical_event_text` to `ckpt.canonical_events`
- Tick-emitted canonical events tagged off-screen so on-stage routing can identify them as recent activity (the actor's `since_last_turn_block` may surface them as observed facts when the player walks into the affected scene)

### Commit 7 — Test hardening
- Test: agent output with parenthetical → narrator's input never contains it
- Test: agent A's prior_responses for agent B contains A's `public_text` only
- Test: `_extract_parenthetical` handles edge cases (no trailing paren, multiple parens mid-prose, nested parens, multiline trailing paren)
- Test: tick fan-in correctly bundles N agents and the router emits expected off-screen `canonical_event_text`
- Test: two off-stage characters with intents toward each other produce a single conversational canonical event in one tick fire
- Test: spawn_characters writes router_summary; next router call's state_changes_block surfaces it; subsequent calls don't
- Un-skip `tests/test_ticks_and_directives.py` scheduler tests; rewrite for the new trigger model

### After-implementation playtest pass
30+ turns. Watch specifically for the R1-R5 review items. Decide which need follow-up commits.

---

## Files most affected

- `app/schemas/agents.py` (rewrite)
- `app/schemas/characters.py` (drop incoming_directives; do NOT add `last_intent` — that mirror was ripped, see CLAUDE.md)
- `app/schemas/state.py` (tick scheduler fields)
- `app/engine/character_agent.py` (parse parenthetical for engine-side logging only; the parenthetical is preserved verbatim in `character_conversations` for the agent's own future replay; no `last_intent` writeback)
- `app/engine/orchestrator.py` (`_run_ticks`, `_route_tick_intentions`)
- `app/engine/turn_loop_dispatcher.py` (drop `_serialize_agent_intention`, drop `_build_character_registry`, add initial-roster + state-changes builders)
- `app/engine/narrator.py` (consume `public_text`, drop bullet structure)
- `app/engine/context_builder.py` (`format_prior_responses`, drop directive plumbing)
- `app/engine/event_router.py` (drop directive references)
- `app/engine/character_manager.py` (spawn flow surfaces router_summary)
- `app/prompts/agent_v9.txt` (rewrite short)
- `app/prompts/agent_tick_v2.txt` (rewrite short with urgency rule)
- `app/prompts/event_router_v9.txt` (new user-template variable set, character interior rule, tick-mode partial reference)
- `app/prompts/_partials/router_tick_mode_block.txt` (new)
- `app/prompts/character_gen_v3.txt` (add `router_summary` to output)
- `tests/test_ticks_and_directives.py` (un-skip, rewrite)
- `tests/test_validators.py` (validator may be deleted)

---

## Background context (don't relitigate)

- Pre-v11 ticks fired at end of `process_turn` after narrator render, before save. `_run_ticks` used a cadence-driven reset model. `_apply_agent_private_updates` handled directive routing + objectives writeback + self-`moved_to`. All deleted in the v11 rewrite, never re-ported. `tests/test_ticks_and_directives.py:14-19` flags this.
- The 30-turn playtest (`playtest_v7i_logs.txt` / `playtest_v7i_console.txt`) had ZERO tick fires across all 30 turns. Proximate cause of "the world feels static."
- The on-stage router today picks responders blindly — registry is identity-only. Root cause of "the antagonist never shows up when he should."
- `private_updates` on `CharacterAgentOutput` is half-real: schema-required, prompt-emitted, never consumed. All four fields dead in v11.
- Router system prompt currently contains: `setting_summary`, `player_characters_block`, `world_lore`, `world_rules`, `hidden_lore`, `hidden_facts`. These are static or session-static, cached as system-prompt prefix. Don't move them.
- Router per-turn user message currently contains `world_facts`, opening/recap/since-last-turn blocks, and the intention block. Avoid restoring repeated roster/location dumps.
- LLM client caches the system prompt prefix and (when `len(messages) > 1`) the last user message. `cache=True` is the v11 default for both router and agent calls.
