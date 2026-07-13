# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

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


## Build & Test

Use the checked-in virtualenv directly when it exists, or create one and install
`requirements.txt` before running tests:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/pytest
.venv/bin/python scripts/play.py --session <name>
```

## Architecture Overview

Ayoa is a prompt-heavy, checkpointed narrative engine. The core runtime lives in
`app/engine/orchestrator.py`, `app/engine/turn_loop.py`, and
`app/engine/turn_loop_dispatcher.py`; schemas live under `app/schemas/`; prompt
templates live under `app/prompts/`; Discord integration lives under
`app/bot/`; the terminal REPL is `scripts/play.py`; and tests are in `tests/`.

## Conventions & Patterns

- Treat `app/prompts/*.txt` as reviewed source artifacts.
- Keep D&D behavior in adapter-specific modules, prompts, and schemas rather
  than baking it into the rules-neutral engine.
- Runtime LLM calls receive text and structured data only; images must be
  inspected and converted to authored text/assets during import.
- Prefer `.venv/bin/pytest` for test runs and keep tests offline unless a live
  integration check is explicitly requested.
