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
