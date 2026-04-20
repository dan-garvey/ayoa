# Code Review — Narrative IF Engine

**Date:** 2026-04-19
**Scope:** full codebase — `app/engine`, `app/schemas`, `app/llm`, `app/bot`, `app/prompts`, `scripts/`, `tests/`, `CLAUDE.md`, `DESIGN.md`
**Method:** four parallel deep-dive agents, each owning a slice of the codebase, then synthesized here
**Test suite size at review time:** 193 tests, 2 skipped

This is a working review — findings are grouped thematically and prioritized. Not every finding is bug-tier; many are "this could be cleaner" and can be left alone. The **Prioritized Action Items** section is where to start.

---

## Executive summary

The codebase is in good shape given how fast it's been evolving. Recent phases (multiplayer, NPC autonomy, importer v2) landed cleanly with solid test coverage on the new mechanics. There's no outright broken or abandoned code. What this review found is mostly accumulated-debt patterns: architectural redundancies from successive refactors, error paths that log-and-continue where they should fail loudly, and two or three real time bombs worth fixing before they bite.

**Healthy**
- Turn pipeline is well-encapsulated; rolling-conversation pattern is consistent.
- No circular imports, no dead modules, no zombie scripts.
- New Phase 3 mechanics (directives, ticks, envelopes) have good unit coverage.
- Versioning discipline is in place (IMPORTER_VERSION, prompt versions, schema_version).
- No TODO/FIXME/XXX comments drifting in the code — all known work is in the task list.

**Worth attention**
- **`DiscriminatorOutput` is a live zombie** (see F1.1). It's a legacy schema that EventRouterOutput projects into via `to_discriminator_output()`, passed around to keep narrator + character_manager signatures stable. It works, but it's dragging two modules' imports and one schema file that don't need to exist.
- **Error paths swallow silently** (F3.x). When an agent times out, when the router emits invalid scene ids, when spawn fails, when a validator flags a knowledge leak — the system logs a warning and continues. Individually reasonable; collectively it means the engine rarely tells you something went wrong.
- **Checkpoint save → tick → save-again window is undocumented** (F4.1). If the process dies between the two saves, ticks are lost but the turn persisted. Not fatal, but the semantics aren't written down.
- **`compact-2026-01-12` beta header is hardcoded** (F7.2). This is a real time bomb — Anthropic will rotate this, and when they do, the engine breaks with no graceful fallback.
- **No tests at all** for `app/bot/commands.py`, `app/bot/session_map.py`, `app/bot/embed.py`. These are the user-facing surface area.
- **Documentation is stale** — CLAUDE.md and DESIGN.md both pre-date Phase 3 and the importer v2 work. Anyone reading them to understand the system will be misled on several concepts.

---

## Prioritized action items

### P0 — Real risk, do soon

1. **Audit the Anthropic beta header.** `app/llm/client.py:317` hardcodes `"compact-2026-01-12"`. When Anthropic rotates this (guaranteed, it's a beta), the engine will 400 on every turn. Either make it config-driven with a fallback path, or set a calendar reminder to check it quarterly.

2. **`.bot.log` grows forever.** `app/bot/__main__.py:38` uses `logging.FileHandler(_LOG_FILE, mode="a")` with no rotation. A long-running bot will eventually fill the disk. One-line fix: `RotatingFileHandler(_LOG_FILE, maxBytes=10_000_000, backupCount=5)`.

3. **Checkpoint save failure leaves in-memory state ahead of disk.** `app/engine/orchestrator.py:370`: save() is not wrapped in try/except. If the save fails but `turn_index` was already incremented and agent updates applied in memory, the next `load_latest()` returns the *previous* turn. Player loses a turn silently. Wrap in try/except; on failure, revert the mutations or surface the error.

4. **SessionMap rows persist after `delete_story`.** `app/bot/engine_bridge.py:172-207` explicitly notes this and accepts it, but the result is that a user who played a story that's since been deleted will get "no checkpoint found" on `/story resume` with no path back. Either delete matching SessionMap rows in `delete_story`, or add a `/story clean` command for operators.

### P1 — High value, not urgent

5. **Delete `DiscriminatorOutput` as a live schema.** Merge the two fields EventRouterOutput doesn't already have into EventRouterOutput, delete `to_discriminator_output()`, update the four signature sites in narrator and character_manager. Kills a whole schema file and a projection hop. (F1.1)

6. **Remove `memory_writes` from `CharacterAgentOutput`.** `app/schemas/agents.py:52`. The comment at `app/engine/character_manager.py:54-55` calls it out as "kept for compatibility" but nothing reads it anywhere. Schema bloat with no benefit. Update the agent output format in the prompts too. (F1.2)

7. **Add tests for the bot command layer.** `commands.py`, `session_map.py`, `embed.py` have zero direct test coverage. The embed splitter (`_split_at_paragraph`) and the briefing budget math are both boundary-heavy pure functions — cheap to test, high regression risk when they drift. (F9.1, F9.3)

8. **Consolidate the 4 `acting_character_id` fallback sites.** `orchestrator.py:75`, `event_router.py:73`, `narrator.py:56`, `character_agent.py:70` all do the same `request.acting_character_id or session.player_character_id` fallback. Pull into a helper on `SessionState.resolve_acting_character_id(requested: str) -> str`. (F2.1)

9. **Factor rolling-conversation save-and-append into one helper.** The render → call client → capture user_content → serialize assistant blocks → append both to checkpoint pattern is copy-pasted in `character_agent.py`, `event_router.py`, and `narrator.py`. If the pattern changes once, all three need updating in lockstep. (F2.2)

10. **Validator flags don't do anything.** `app/engine/orchestrator.py:293`: `validate_all_outputs` returns pass/fail + leak flags, the results are written to the visibility log, but the turn proceeds unchanged regardless. Either make it enforcing (retry or drop on leak) or rename it to make clear it's advisory-only logging. (F3.6)

11. **Update CLAUDE.md and DESIGN.md.** Both predate Phase 3 entirely. CLAUDE.md project structure doesn't mention `scripts/play.py`, bot multiplayer, or the tick engine. DESIGN.md §7.3 Character Record example is pre-v2 (no `current_objectives`, `intentions_enabled`, `incoming_directives`, `known_context`). (F10.1, F10.2)

### P2 — Polish

12. **De-duplicate `render_briefing` import** (`commands.py:29` + redundant local import at `commands.py:209`).

13. **Parameterize hardcoded version-string assertions in tests.** `test_checkpoint.py:54` pins `"v4"`; this has already bitten us twice in recent commits. Assert version format (`r"^v\d+$"`), not a specific value.

14. **Collapse duplicate `_run_ticks` skip tests** into a `@pytest.mark.parametrize`. Five near-identical tests.

15. **`_session_locks` dict grows unbounded** (`engine_bridge.py:72`). One entry per session, never cleaned up. Not urgent — a week of uptime with 100 sessions adds 100 entries of ~200 bytes — but worth a TTL eventually.

16. **Add a `--model` override** to both the bot import path and the CLI import script. Haiku handles extraction for ~1/3 the cost of Sonnet; a 95 KB master prompt costs ~$0.50 on Sonnet and ~$0.15 on Haiku. Currently hardcoded to narrator role, which maps to Sonnet. (F7.4)

17. **Emit total-token usage at end of import.** Currently the importer logs wall-clock time but not token totals. If you want to optimize cost, you can't measure it. Roll up `response.usage` across the 4 calls in `run_import`. (F7.5)

---

## Detailed findings

### §1 Dead code and zombie schemas

**F1.1 `DiscriminatorOutput` is a live zombie.**
`app/schemas/discriminator.py` defines `DiscriminatorOutput`. `app/schemas/event_router.py:37` defines `EventRouterOutput.to_discriminator_output()` that returns a strict subset projection. Orchestrator calls it (`orchestrator.py:92`) and hands the projected object to narrator (`narrator.py:44,147`) and character_manager (`character_manager.py:68`) just so their signatures don't have to change. The useful payload from the projection is already on `ObserverEntry` (shared between the two schemas). Collapsing onto EventRouterOutput would delete a schema file and unblock narrator's comment at `narrator.py:21-24` which reads like an apology for legacy compatibility.

**F1.2 `CharacterAgentOutput.memory_writes` is dead.**
`app/schemas/agents.py:52`. Never read by any engine code. Comment at `character_manager.py:54-55` even admits: "kept in the schema for compatibility but no longer written anywhere." The agent prompts (`agent_v7.txt:78`, `agent_tick_v1.txt`) still tell the LLM to emit it. Delete the field, the prompt lines, and the comment.

**F1.3 `LLMConfig` has a `"discriminator"` role mapping that nothing uses.**
`app/llm/config.py:11-17`. Since the discriminator was merged into the event router. Harmless but confusing. Delete the entry.

**F1.4 `LLMConfig.default_temperature` and `default_max_tokens` are never hit.**
`app/llm/config.py:20-21`. Every call site explicitly passes `temperature` and `max_tokens`, so the fallback at `client.py:192-193` never triggers. Remove them or start using them; currently noise.

**F1.5 Duplicate local import of `render_briefing`.**
`app/bot/commands.py:209` re-imports what's already in the module header at line 29.

### §2 Redundancies and duplicated logic

**F2.1 `acting_character_id` fallback is copy-pasted across 4 modules.**
`orchestrator.py:75-77`, `event_router.py:73-76`, `narrator.py:56-62`, `character_agent.py:70-76` all do:
```python
acting_id = request_id or checkpoint.session.player_character_id
acting_char = next((c for c in checkpoint.characters if c.character_id == acting_id), None)
acting_name = acting_char.name if acting_char else (checkpoint.session.player_name or "the protagonist")
```
If the fallback semantics ever change, four sites must be updated in lockstep. Pull into a method on `SessionState` or a helper in `context_builder`.

**F2.2 Rolling-conversation append pattern is triplicated.**
`character_agent.py:107-113`, `event_router.py:95-99`, `narrator.py:117-122` all do: capture user_content before `client.complete`, extract assistant_content via `serialize_assistant_content`, append both to the correct conversation list on the checkpoint. Identical code, three homes. Extract to a helper; it becomes a one-liner at each site and cements the invariant.

**F2.3 `collect_player_ids` scans the full roster on every call, called multiple times per turn.**
Called from `orchestrator.py:118` (roster moves), `orchestrator.py:163` (responder selection), `orchestrator.py:462` (tick eligibility), `event_router.py:259` (character registry). For a 50-character story, that's 200+ character iterations per turn. Compute once at the top of `process_turn` and thread through, or cache on the checkpoint.

**F2.4 Character attitudes can still be keyed by the string `"user"`.**
`app/schemas/import_extraction.py:237` references the legacy key; `app/engine/character_manager.py:62` assumes all keys are real `character_id`s. A legacy-authored checkpoint with `attitudes["user"]` will silently never apply deltas to anyone. Add a migration step in `CheckpointManager.load_file()` that rewrites `"user"` → `session.player_character_id`.

**F2.5 `_build_npc_turn_summary` and `narrator._format_agent_outputs` build similar structures differently.**
`orchestrator.py:629-658` vs `narrator.py:132-184`. Both walk agent outputs and format actions+dialogue; they produce different output formats but the walk is the same. Extract a shared iterator so changes to agent-output shape don't need to be mirrored.

**F2.6 `pending_observations` and `incoming_directives` use near-identical flush-and-clear patterns but diverge in stamping.**
`pending_observations` are `list[str]` with turn embedded in the prose; `incoming_directives` are structured with turn+depth fields. The lifecycle is symmetric (flush on response or tick, clear after) but the stamping isn't. Consider unifying to a single `QueuedMessage` type, or at minimum document the asymmetry.

### §3 Error handling and edge cases

**F3.1 Agent timeouts are logged and skipped; the turn proceeds without the character.**
`orchestrator.py:235-249` and `orchestrator.py:257-279`. If a "compelled" responder (priority 5) fails, the narrator is told "no response" and produces prose that ignores a character the router said *must* respond. Options: retry once, emit a "X remains silent" fallback, or fail the turn loudly. The current silent-continue breaks narrative continuity.

**F3.2 Scene transitions to invalid scene_ids are silently dropped.**
`orchestrator.py:96-105`. Router emits `scene_delta.new_scene_id`; if it's not in the scene_graph, we log and ignore. The resolved outcome prose may reference moving; the player's scene didn't change. Fail the turn or reject the router output.

**F3.3 Roster moves to invalid scenes / missing characters / player-bound characters are logged-then-ignored.**
`orchestrator.py:120-149`. Same pattern. If the router is misbehaving, dozens of roster_moves get dropped and the player sees no world-state changes despite prose that may imply them.

**F3.4 `_spawn_one` failures are caught and silently skipped.**
`character_manager.py:113-121`. If the spawn LLM call fails, the character never appears and no one knows. Consider: if spawn failure rate > 50%, fail the turn instead.

**F3.5 `character_self_introduction` loop assumes `get_character()` non-null.**
`orchestrator.py:316-335`. `get_character()` can return `None`; the loop accesses `char.name` without a null check. Defensive guard or make `get_character()` raise.

**F3.6 `validate_all_outputs` flags are recorded but never acted upon.**
`orchestrator.py:293-313`. A leak flag produces a `visibility_log` entry and nothing else — the leaky agent output is still sent to the narrator and the player sees it. If the validator is advisory, rename to make that explicit; if it's supposed to enforce, implement the enforce path.

**F3.7 Discord interaction timeouts on `/join`.**
`commands.py:521-572` doesn't `defer()` before calling `bind_user` and `build_character_dossier`, both of which do checkpoint I/O. If the checkpoint is large or disk is slow, the 3-second response window is at risk. `/act` and `/describe` both defer; `/join` should too.

**F3.8 `/story import` background analysis has no user-visible completion signal.**
`engine_bridge.py:141-170`. Analysis may take 2-5 more minutes after the import command returns. User has no way to tell when it finishes or whether it failed (exceptions are caught and logged, nothing surfaces). A follow-up DM or a message edit when analysis lands would close the loop.

**F3.9 Admin env var parsing fails closed on one bad entry.**
`commands.py:41-51`. If `DISCORD_ADMIN_USER_IDS="123,456,abc,789"`, the whole list is rejected and all admins lose access. Parse best-effort (skip invalid entries, log them), so a typo doesn't nuke everyone.

**F3.10 `/story resume` on a dead checkpoint tells the user to `/story end` — but the row stays.**
`commands.py:176-186`. Log says "run /story end to clean up," but a user will hit this exact state by following that advice only to hit `/story resume` again later with the same error. Auto-purge the SessionMap row when `load_latest` returns FileNotFoundError.

### §4 State consistency and crash safety

**F4.1 Save → tick → save-again crash window.**
`orchestrator.py:370` saves the player-visible turn. `orchestrator.py:381-391` runs ticks and saves again. If the process dies between the two saves, the turn on disk is correct but tick-updated NPC state is gone. On restart, `_run_ticks` won't re-fire for this turn. Document or fix: either run ticks before the first save (they're off-stage so ordering doesn't break player view), or add a `ticks_pending` flag that's persisted with the first save and consumed on next load.

**F4.2 In-memory mutations happen before the save.**
`orchestrator.py:367-370`: transcript is appended, `turn_index` is incremented, agent outputs applied, *then* save. If save fails, in-memory state is ahead of disk. Save-first, mutate-second — or rollback in-memory state on save failure.

**F4.3 Tick eligibility uses `acted_this_turn` which doesn't distinguish "agent succeeded" from "agent was invoked."**
`orchestrator.py:378`. Set is built from `agent_outputs` + `acting_character_id`. If the agent raised, they're NOT in `agent_outputs` and would be eligible for a tick this turn. If they succeeded but output was invalid and dropped, they may or may not be in the set. Define precisely what "acted" means.

**F4.4 Character roster mutations (spawn, dormancy, cull) are in-memory before the checkpoint save.**
`orchestrator.py:116` appends spawned characters; `character_manager.apply_roster_updates` mutates statuses. Same save-failure window as F4.2.

### §5 Performance

**F5.1 `load_latest()` is called redundantly multiple times per command.**
`commands.py:660`, `commands.py:761`, `commands.py:829`, plus inside `EngineBridge.get_user_binding` (`engine_bridge.py:304`), inside each turn's `process_turn`, inside the embed footer helper. A single `/act` may load the checkpoint 3-4 times. On a 1 MB checkpoint this is real overhead. Pass the loaded checkpoint down through the handler.

**F5.2 Magic-number fan-out caps.**
`orchestrator.py:36` `MAX_RESPONDERS = 3`, `orchestrator.py:491` semaphore-of-4 for ticks. No comments explain why these numbers. Move to `SessionConfig` and document the tradeoff (latency vs concurrency vs cost).

**F5.3 Agent user message embeds turn-indexed strings, weakening cache line-up.**
`pending_observations` are rendered with `[Turn N]` prefixes directly in the prose. The Anthropic cache matches on message content; a per-turn variable embedded in the message weakens hit rates. If the turn index is load-bearing, consider keeping it in a structured separator the cache can still match around; otherwise drop it (the ordering is already preserved).

**F5.4 Prompt render time is not measured.**
`PhaseLatency` captures API call time only. `context_builder` + `render_conversation` can be non-trivial on 50-character stories. Add a `prompt_render_ms` field.

**F5.5 Checkpoint save time is untimed and unlogged.**
`checkpoint_manager.py:27-50`. On a large checkpoint (long conversations), JSON serialize + write can be slow. Log time + file size to spot I/O bottlenecks.

**F5.6 `_format_roster_for_knowledge` repeats work on every import.**
Not a hot path but minor: rebuilds per-character string blocks even for characters the envelope extractor already understands. Fine to leave.

**F5.7 Four-stage extraction could potentially collapse.**
`story_importer.py:910-925`. Stages 1-4 all see the same source; in principle a single structured-output call with a combined schema could produce everything. Would cut API round-trips. Significant refactor; flagged as an opportunity, not a bug.

### §6 Security and input validation

**F6.1 Attachment MIME type not validated.**
`commands.py:384-390`. Only extension is checked. A binary file renamed `.txt` passes the filter; it'll fail on UTF-8 decode but wastes LLM tokens up to that point. Use `attachment.content_type`.

**F6.2 Directive content length is unbounded.**
`app/schemas/agents.py:25` (`DirectiveSend.content`). A runaway agent could emit a 100 KB directive that bloats the recipient's queue and the checkpoint. Cap at 2000 chars.

**F6.3 Observable facts on `CanonicalEvent` aren't checked for hidden-content leakage.**
`orchestrator.py:293` validates agent outputs. The router's observable_facts go straight to observers without the same check. If the router misbehaves and copies hidden_lore content into observable_facts, every observer sees it. Add a validation pass on router output.

**F6.4 `delete_story` does not purge SessionMap rows.**
Already covered under F3.10 / P0 #4.

### §7 LLM client and config

**F7.1 Retry loop doesn't cover Pydantic parsing errors.**
`client.py:333-345`. Retries on connection/timeout/rate-limit, but if the model returns invalid JSON, Pydantic parses at `client.py:231-235` *after* the retry loop. Flaky JSON output → immediate failure instead of retry. Move the parse inside the retry loop.

**F7.2 `compact-2026-01-12` beta header is hardcoded. See P0 #1.**
`client.py:317` and `client.py:309`. No fallback, no config, no version detection.

**F7.3 `raw_response.model` is best-effort read.**
`story_importer.py:854`. If the SDK changes or the attribute is missing, `model` on `ImportAnalysis` silently becomes empty. Pass the model name in from `LLMConfig.model_for_role(role)` so we're not at the SDK's mercy.

**F7.4 Extraction model is not overridable.**
Currently `role="narrator"` → Sonnet for all four extraction stages + the preservation analysis. A 95 KB master prompt is ~$0.50 on Sonnet, ~$0.15 on Haiku. Add `EXTRACTION_MODEL` env var or config knob.

**F7.5 Total-import token usage is not aggregated.**
`run_import` calls `complete()` 4 times + preservation analysis, and each `response.usage` is discarded. No way to report "this import cost X tokens." Log a rollup.

**F7.6 `coverage_rating` isn't a Literal on the LLM output model.**
`story_importer.py:_AnalysisLLMOutput` has `coverage_rating: str`; coercion to the allowed set happens in Python post-parse (line 848-851). If it were `Literal["high","medium","low"]` on the Pydantic model, the Anthropic structured-output path would enforce it server-side.

### §8 Prompts

**F8.1 `known_context` is used but never named to the agent.**
Agents read `{world_context}` in `agent_v7.txt` + `agent_tick_v1.txt`; the runtime substitutes the character's `known_context` envelope when present (v2) or falls back to global lore (v1). The prompt never explains to the agent that this is *their* personalized view. Worth a sentence so the LLM knows it may differ from other NPCs'.

**F8.2 Router "objectives" vs agent "current_objectives" naming is inconsistent.**
`event_router_v4.txt:136` tells the LLM to emit `objectives` on spawn seeds. The runtime `SpawnRequest.seed` is a freeform dict, so whatever key the router emits goes through. `character_gen_v3.txt` and the runtime use `current_objectives`. Audit a real router output on a spawn turn to confirm the key matches; if not, the spawned character starts with empty objectives.

**F8.3 `character_gen` version is not tracked in checkpoint.**
`CheckpointFile.prompt_versions` defaults to `event_router`, `agent`, `narrator_phase2` but not `character_gen`. Each runtime spawn uses whatever's highest on disk; no audit trail.

**F8.4 `_serialize_checkpoint_for_analysis` skips `intentions_enabled` and `attitudes`.**
`story_importer.py:_serialize_checkpoint_for_analysis`. Preservation analysis can't report on tick-eligibility coverage or relationship preservation.

**F8.5 Distinctness instruction in envelope extraction is thin.**
`story_importer.py:KNOWLEDGE_EXTRACTION_INSTRUCTIONS` — "Do not reuse passages across envelopes" is one sentence. LLMs often boilerplate similar characters with minor variations. Concrete worked examples would strengthen it.

### §9 Testing

**F9.1 Zero direct tests for the bot command layer.**
`app/bot/commands.py`, `app/bot/session_map.py`, `app/bot/embed.py`. The embed text-splitter (`_split_at_paragraph`) and briefing budget math are pure functions with boundary cases that are cheap to test and have already regressed once.

**F9.2 Hardcoded version strings in tests.**
`test_checkpoint.py:54` pins `"v4"`. Every prompt bump breaks tests before it hits production. Assert format (`r"^v\d+$"`), not a specific version.

**F9.3 Low-value Pydantic-default tests.**
`test_schemas.py` has several `test_defaults` that assert a freshly-constructed model equals its own defaults. This tests Pydantic, not us. Round-trip and construction tests are the ones that matter.

**F9.4 Duplicate test intent that could be parametrized.**
`TestTickScheduler::test_skips_*` (4 nearly-identical tests, one per skip reason). `TestDirectiveRouting::test_warns_above_depth_warn` and `test_drops_above_depth_cap` (parametrizable over depth). `test_attitude_labels` (7 single-assertion tests).

**F9.5 Envelope application is not tested end-to-end.**
`test_import_envelope_and_analysis.py` tests envelope assignment to characters, but not the follow-through: does `build_world_context(character, checkpoint)` return the envelope when set? Yes — verified in `test_build_world_context_uses_envelope`. But there's no integration test showing that an agent's system prompt *contains* the envelope text (or doesn't contain the global premise).

**F9.6 Preservation analysis output shape has no real test.**
Only `_serialize_checkpoint_for_analysis` is tested. The actual `run_preservation_analysis` flow (deterministic counts, coverage rating coercion, model stamping) has no unit test. A mock of `client.complete` returning a preset `_AnalysisLLMOutput` would cover this.

**F9.7 Shared mock helpers aren't centralized.**
`_make_mock_response`, `_llm_response`, `_install_stream_mock` are each defined in their own test files with identical structure. If the SDK response shape changes, updates needed in 3+ places. Move to `conftest.py`.

**F9.8 No end-to-end test of the full turn pipeline.**
Orchestrator tests use mocked LLM; agent tests use mocked client; but there's no test that runs EventRouter → agents → Narrator together with a coherent scripted LLM. A single integration test would catch contract drift between the roles.

**F9.9 `test_llm_client.py` integration test is not run by default.**
`@pytest.mark.integration` — not run in CI unless explicitly requested. Fine, but no CI smoke means API-level regressions aren't caught until someone runs the bot.

### §10 Documentation

**F10.1 CLAUDE.md is stale.**
- Doesn't mention `scripts/play.py` (the interactive CLI).
- Doesn't mention the Discord multiplayer commands (`/join`, `/leave`, `/story characters`).
- Doesn't mention the tick engine or knowledge envelopes.
- Model-defaults line says "test and playtest scripts default to claude-haiku-4-5" — playtest scripts were deleted.

**F10.2 DESIGN.md §7.3 Character Record example is pre-v2.**
Shows `private_state.goals` without `current_objectives`, `intentions_enabled`, `incoming_directives`. Shows no `known_context`. Shows no `is_player`. A new reader will build wrong mental models.

**F10.3 DESIGN.md §5.7 Prompt Manager doesn't mention caching breakpoint application.**
Critical for performance and omitting it is misleading.

**F10.4 Module docstrings absent on `character_manager.py`, `checkpoint_manager.py`, `validators.py`.**
These three files have no module-level documentation. `validators.py` has complex regex heuristics with no explanation.

**F10.5 Comments reference removed concepts.**
`character_manager.py:54-55` still says `memory_writes` is "kept for compatibility" — but the thing it was compatible with has since been removed. Tidy.

**F10.6 `RosterMove` docstring is thin.**
`app/schemas/event_router.py:9-12`. Doesn't document when the move is observable to the player vs silent, which matters for narrator rendering.

**F10.7 `attitude_delta` semantics are undocumented on the schema.**
`app/schemas/agents.py:36`. Delta vs absolute? What happens if target is unknown? What's the clamp? All in the code, none in the docs.

### §11 Naming and small code-quality items

**F11.1 `extract_entities` and `build_known_entities` in `validators.py` are opaque.**
Short names, no docstrings, regex-driven. Rename or document what they do and what "entity" means in this context.

**F11.2 `_attitude_label` vs rest of the file.**
`context_builder.py:_attitude_label`. Name doesn't hint it converts a float to a label string. `_attitude_to_label` or `_format_attitude` reads better.

**F11.3 Variable naming drift across the pipeline.**
Sometimes `output`, sometimes `agent_output`, sometimes `result`, sometimes `response` (for the same category of thing). Not wrong; mildly tiring.

**F11.4 `observation_level` is a hard-coded string, not an enum.**
`"direct"`, `"indirect"`, `"inferred"` appear as bare strings in `orchestrator.py:611-621`, `narrator.py` agent-output formatting, and the router prompt. An enum would prevent typos and make the valid set explicit.

**F11.5 `SpawnRequest.seed` is `dict[str, Any]`, so SpawnSeed schema is unused.**
Already discussed earlier in Phase 3 planning. Typing the seed would catch router output errors at parse time.

**F11.6 `_session_locks` has no eviction.**
`engine_bridge.py:72`. Low-impact memory leak; not urgent.

**F11.7 Synthetic CLI user_ids don't round-trip cleanly with Discord snowflakes.**
`scripts/play.py:86-100`. If a session was started from Discord (user_id ~10^17) and resumed on CLI, `_next_user_id = max(...) + 1` jumps to ~10^17 range. Minor oddity; doesn't break.

### §12 Missed opportunities (future)

**F12.1 Incremental knowledge envelopes.**
New characters spawned at runtime go through `character_gen` but don't get envelopes. Either call the envelope extractor for each spawn, or extend the spawn prompt to produce `known_context` directly.

**F12.2 Preservation analysis as a gate.**
`run_preservation_analysis` produces a `coverage_rating`. Currently advisory. An optional `reject_threshold` parameter could let operators refuse imports with `low` coverage before they ship.

**F12.3 No `/undo` command.**
Checkpoints are kept turn-by-turn; restoring an earlier one is mechanically trivial. Useful for fat-finger recovery. Low priority.

**F12.4 No `/whoami` in CLI.**
Minor convenience for multi-character sessions where the player forgets which actor they're on.

**F12.5 Directive chains don't carry breadcrumbs.**
`IncomingDirective` stamps depth and immediate sender. The full chain (A → B → C → D) isn't preserved, making deep-chain debugging painful. Adding a `chain: list[str]` field would cost one line per directive.

**F12.6 No resurrection for culled characters.**
`character_manager.apply_roster_updates` handles dormant and culled but never reactivates. If a story ever needs a return-from-the-dead beat, the engine blocks it.

**F12.7 Narrator prompt render time not instrumented.**
If it becomes a bottleneck you won't know.

**F12.8 No token-usage summary at turn-level for operators.**
`PhaseLatency` exists but isn't exposed to any dashboard or command. A `/status --verbose` or `/session cost` would help tune.

---

## Finding counts

| Theme | Findings |
|---|---|
| §1 Dead code | 5 |
| §2 Redundancies | 6 |
| §3 Error handling | 10 |
| §4 State consistency | 4 |
| §5 Performance | 7 |
| §6 Security | 4 |
| §7 LLM client & config | 6 |
| §8 Prompts | 5 |
| §9 Testing | 9 |
| §10 Documentation | 7 |
| §11 Naming / code quality | 7 |
| §12 Future opportunities | 8 |
| **Total** | **78** |

---

## Notes on method

Four agents reviewed slices in parallel and returned ~150 raw findings across the four. This document is the dedup + synthesis. A handful of findings from the raw reports were dropped as either:
- Already on the task list (e.g., the pending task #45: async player off-stage commands).
- Style preferences without clear benefit.
- Contradicted by a check of the actual code.

Verified spot-checks that confirmed agents were grounded, not hallucinating:
- `render_briefing` double-import at `commands.py:29` + `:209` ✓
- `FileHandler` (not rotating) at `__main__.py:38` ✓
- `compact-2026-01-12` at `client.py:317` ✓
- `memory_writes` dead field at `agents.py:52` with apology comment at `character_manager.py:54` ✓
- `DiscriminatorOutput` zombie schema with projection at `event_router.py:37` consumed at `orchestrator.py:92` ✓

Everything else in this report is stated with enough specificity that it can be verified in a minute of grep.
