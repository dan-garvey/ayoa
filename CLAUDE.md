# Narrative Engine

## Dev Environment

- Use `.venv/bin/python` and `.venv/bin/pytest` directly instead of sourcing the venv activate script.
- API key is in `.env` as `LLM_GATEWAY_KEY`. Never commit `.env`.
- LLM gateway: `https://llm-api.amd.com/OnPrem`, OpenAI-compatible, auth via `Ocp-Apim-Subscription-Key` header.

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
