# Narrative foundation

Use maximum reasoning effort. Read DESIGN.md and docs/findings.md before changing
the narrative contract. This branch is a standalone, rules-neutral experiment.
Keep work on this branch unless the user explicitly requests integration elsewhere.

## Runtime and prompts

- One author writes world, NPCs and narration in one call. Regeneration is an
  explicit player action, not an automatic editing stage. API and proxy execution
  use the same context builder and persisted turn loop.
- Only active passages and original player inputs enter future history. Preserve
  exact attempts, replaced passages and regeneration instructions as evidence;
  never publish a failed response or replay discarded feedback.
- Treat prompts as reviewed source. Trace a prose problem to its actual response
  or input before editing instructions. Keep reusable writing rules out
  of story canon, and maintain full character histories when moving source text.
- NPC knowledge is bounded by plausible acquisition. Consistent invented history
  is welcome; contradictions and inaccessible knowledge are failures.
- Preserve player ownership and NPC independence. Narrative initiative follows
  player conduct; quiet interaction does not need a compulsory hook or conflict.
- Do not add character-specific sample dialogue, fixed voice recipes, quotas,
  extra model roles or derived state without evidence of a concrete need.
- Runtime model inputs are text only. No images, provider details, credentials,
  SDK names, implementation paths or test machinery belong in narrative prompts.
- Keep session-specific data outside the shared instruction prefix. Avoid duplicate
  representations, obsolete save compatibility and silent recovery fallbacks.

## Validation and evidence

- Use `.venv/bin/pytest` directly; keep automated tests offline. Test actual message
  placement, persistence, replay, failure recovery and forbidden prompt leakage.
  Do not freeze approved prompt prose with required-wording assertions.
- Review complete exchanges for narrative quality. A phrase filter is not a
  literary grader. Save every first result and report mixed outcomes honestly.
- User-authorized narrative playtests use fresh Terra coding-agent proxies at
  maximum reasoning effort unless the user requests another configuration.
  Keep raw evidence on `archive/covenant-prompt-trials`; link immutable commits
  from the active findings document. Never inspect private model reasoning.
- Commit verified changes and push the working branch before handing back work.
  Do not merge main, rewrite history, or delete unrelated worktree changes.

## Task tracking

Use the Beads skill at `.agents/skills/beads/SKILL.md`, run `bd prime`, and use
`bd` for all durable tasks. Preserve the shared Dolt database and worktree
redirects. Historical JSONL exports are archived, not the active issue database.
Create issues for follow-up work, close only completed tasks, run quality gates,
commit and push, and verify branch status before concluding an implementation.
