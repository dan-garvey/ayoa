# Ayoa Agent Instructions

## ALWAYS

Use maximum thinking/reasoning budget for responses. Always consider edge cases and the holistic flow as well as potential downstream effects of your changes.

This repository is a prompt-heavy narrative engine. Prompt files are versioned source artifacts, not hidden implementation details. Edit by tracing behavior through logs and code first, then write tests for runtime contracts and forbidden prompt leakage rather than tests that freeze approved prompt prose.

Every feature change must survive two questions before implementation:

1. Does this preserve the original rules-neutral narrative engine?
2. Does this avoid adding prompt or runtime machinery that is useless in non-D&D narrative contexts?

D&D mechanics must remain modular adapters around the narrative engine, not assumptions baked into generic router, narrator, or character-agent behavior.

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

## Existing Context

Read `DESIGN.md` for broader project structure, long-term goals, and architectural concerns (see especially "Repository And Setup", §15 Rules Adapters, §18 Current Gaps And Maintenance Notes, §19 Engineering Discipline, and §20 Future Directions). This file is the concise repo-local editing contract for coding agents.
