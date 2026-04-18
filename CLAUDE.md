# Narrative Engine

## Dev Environment

- Use `.venv/bin/python` and `.venv/bin/pytest` directly instead of sourcing the venv activate script.
- API key is in `.env` as `ANTHROPIC_API_KEY`. Never commit `.env`.
- LLM: Anthropic Messages API via the `anthropic` Python SDK. Player sessions default to `claude-sonnet-4-6`; test and playtest scripts default to `claude-haiku-4-5`.

## Project Structure

- `app/` — main application (FastAPI)
- `app/schemas/` — Pydantic data models
- `app/engine/` — core engine components (narrator, discriminator, orchestrator, etc.)
- `app/llm/` — LLM client wrapper
- `app/prompts/` — versioned prompt templates
- `app/storage/saves/` — checkpoint save files
- `tests/` — pytest tests
- `scripts/` — interactive test/utility scripts
- `DESIGN.md` — full design document
