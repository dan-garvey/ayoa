# Ayoa Agent Instructions

## ALWAYS

Use maximum thinking/reasoning budget for responses. Always consider edge cases and the holistic flow as well as potential downstream effects of your changes.

This repository is a prompt-heavy narrative engine. Prompt files are versioned source artifacts, not hidden implementation details. Edit by tracing behavior through logs and code first, then write tests for runtime contracts and forbidden prompt leakage rather than tests that freeze approved prompt prose.

Every feature change must survive two questions before implementation:

1. Does this preserve the original rules-neutral narrative engine?
2. Does this avoid adding prompt or runtime machinery that is useless in non-D&D narrative contexts?

D&D mechanics must remain modular adapters around the narrative engine, not assumptions baked into generic router, narrator, or character-agent behavior.

Until Ayoa reaches a release-candidate migration posture, prefer retiring changed schema/prompt fields directly over maintaining compatibility with old saves. Avoid adding compatibility shims for obsolete checkpoint shapes unless the user explicitly requests them.

After completing an implementation or bugfix, commit the verified changes and push the branch before handing the work back to the user unless the user explicitly says not to.

Player UX review should assume pseudo-live table norms unless the user says otherwise. Do not treat AFK-player handling as a primary UX requirement; the expected mitigation is social pressure from making the table wait, not engine machinery that keeps play moving around an absent player.

## Design Simplification Discipline

When the user challenges a router, schema, prompt, or model-input design as over-engineered, first re-evaluate whether each extra surface is semantically necessary. Do not preserve headers, wrapper blocks, queues, compatibility paths, or diagnostic summaries just because they already exist.

Prefer the smallest model input that preserves the real runtime contract. If compact canonical history already carries a fact, decision, clock, or state transition, do not re-send a derived copy unless there is a current producer outside that canonical path. If two input paths carry the same semantic content, unify them instead of creating a second prompt mode.

Before claiming latency, correctness, or orchestration tradeoffs, draw or describe the actual loop in terms of who is called, what waits for what, and what context each model receives. Distinguish sequential same-context cascades from genuinely independent parallel work.

Treat router-authored fiction as router-owned truth. Engine code should implement, persist, and validate router decisions, not re-explain those same decisions back to the router as synthetic updates. External engine mutations should surface only when they are not already represented in canonical router history.

When a required router-directed side effect fails, prefer a loud runtime error over a compensating summary or silent skip. Silent drops, fallback summaries, and "failed but continue" paths hide contract violations.

For adapter paths such as D&D, make them obey the same generic contract when their semantic output is the same. Do not create adapter-specific side channels for information that should live in canonical history or shared runtime contracts.

## Prompt Editing

- Treat `app/prompts/*.txt` as carefully reviewed source. Do not add positive tests that assert required prose snippets exist in versioned prompt files just to prevent accidental deletion; human review and git history cover that.
- Do add tests for rendered prompt blocks, helper output, schemas, dispatch behavior, and observable runtime contracts when engine code depends on exact markers or shapes.
- Do add negative prompt hygiene tests for text that should never appear in any prompt: provider/API implementation details, SDK names, API keys, Python class names, test harness references, internal file paths, and engine-only structures the model does not need.
- Keep prompt tests focused on harmful additions, not preserving wording. If a prompt rule matters behaviorally, prefer a unit test that drives the code path or validates the schema/dispatcher contract it relies on.
- Follow XML-style prompt structure without redundant markdown headers immediately inside matching tags. For example, prefer `<instructions> ... </instructions>` over `<instructions>\n## Instructions\n...`.
- Remove navigational filler such as "see the section below/above" or "as described earlier." The model reads the whole prompt at once; spend tokens on the reason a rule exists, not on visual wayfinding.
- When replacing routing cues, explain the underlying contract for each mode: what information is present, what job the model has in that mode, and what it must not do. Do not rely only on "skip to section X" instructions.
- Give each LLM only its input/output contract and the domain decision it must make. Do not explain implementation or pipeline mechanics such as engines, orchestrators, dispatchers, context builders, API clients, or downstream consumers. Translate "the engine will call X" into field semantics such as "put character ids in `agent_responder_picks` when those NPCs should produce follow-up intentions," or remove the sentence.
- Keep turn-specific data out of cached system prefixes. Actor id/name, current player bindings, "acting this turn" markers, pending observations, current intention text, state-change queues, and mode-specific blocks belong after the template's `<<<USER>>>` delimiter. Stable prompt rules and story-level context may live before the delimiter. When moving data for cache efficiency, add a rendered-prompt contract test that checks the system message excludes the volatile values and the user message includes them.
- Do not spend prompt tokens telling a model to read its own conversation, use prior messages, preserve its history, or inherit context across turns. That is inherent to the context window. Keep only concrete knowledge-boundary rules such as "do not reference events this character did not witness."
- Treat comments in `app/prompts/*.txt` as prompt debt. Even if comment blocks are stripped before rendering, maintainer-only notes about template construction or pipeline behavior belong in code, tests, or docs, not inside model prompt source.
- When playtest prose is wrong, determine whether the source is the character agent, router canonicalization, narrator rendering, or stored context before changing prompts.
- Avoid using tests as a substitute for approving major prompt-infra removal. If a change removes or renames prompt infrastructure, surface it explicitly in the final summary.

## Test Strategy

- Prefer tests that fail on meaningful regressions: visibility routing, event broadcast, schema validation, context inclusion/exclusion, turn-loop state transitions, and narrator/agent input construction.
- Do not add tests whose only assertion is that approved wording remains in a prompt file.
- For prompt cache behavior, test rendered message placement rather than prompt wording. Example: assert actor/player turn state is absent from the system message and present in the user tail.
- For prompt hygiene, scan all `app/prompts/*.txt` and fail on banned implementation terms. Keep the banned list concrete and reviewable; add a new banned pattern when a real leak appears.
- Keep tests offline. Do not write tests that require a live Anthropic key unless the user explicitly asks for an integration test.
- Use `.venv/bin/pytest` directly.

## Investigation Workflow

- Start with `rg` over logs/checkpoints/code, then inspect the exact stored router, narrator, and agent messages.
- Separate "who invented this?" from "who rendered this?" The router often canonicalizes facts that the narrator must treat as authoritative.
- For router behavior, inspect `decision_rationale` alongside `canonical_event.observable_facts`, `observers`, `agent_responder_picks`, `ends_beat`, and `ends_beat_reason`. The rationale is the schema's diagnostic field; use it to see whether a bad output came from classification, visibility, responder selection, or beat-pacing logic.
- After major router prompt edits, run a targeted live-router harness before declaring victory. Compare raw outputs against the previous report, not only pass counts: check multi-addressed dialogue, NPC-to-NPC pressure, defer/wait pacing, Cat II open/resolution, mediated perception, custom arrival/onboarding, and off-stage tick fan-in. Also compare LLM usage logs for cache-read vs uncached input when cache layout changed.
- Treat targeted harness checks as aids, not proof. If manual review shows a false pass or false failure, tighten the harness heuristic after the run so future reports catch the same class of regression.
- Before changing schema or prompt shape, check downstream consumers and tests. Remove vestigial schema fields when the last reader or writer goes away.
- When asked for a dump, write raw artifacts to a file under the repo or session storage and report the path.

## Model Input Case Study

Router-history compaction is the reference example for changing model inputs before changing model instructions. The router used to replay both the raw user message and the full JSON output, even though the JSON mostly restated the user input as canonical facts plus empty schema fields. Replacing each stored router exchange with a deterministic assistant-side `prior_event` record kept event id, timing, facts, observers, beat pacing, and non-empty side effects while dropping the user message, rationale, feasibility boilerplate, empty fields, and JSON punctuation.

Result from the relative-time live harness: router checkpoint history fell from 8,345 tokens to 2,792 tokens, final checks improved from 99/101 to 101/101, and the previous advanced-observer backfill failures passed. Use this pattern when model context is bloated or semantically noisy: identify the true runtime contract, store that contract directly, test both token movement and behavioral accuracy, and prefer deterministic input projection over prompt wording changes.

## Existing Context

Read `DESIGN.md` for broader project structure, long-term goals, and architectural concerns (see especially "Repository And Setup", §15 Rules Adapters, §18 Current Gaps And Maintenance Notes, §19 Engineering Discipline, and §20 Future Directions). This file is the concise repo-local editing contract for coding agents.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
