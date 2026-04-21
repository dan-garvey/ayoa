from __future__ import annotations

import os

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    api_key: str = ""

    # The discriminator role was merged into the event router in v2; no
    # caller asks for role="discriminator" anymore, so it's omitted here.
    # Agent defaults to Haiku for playtesting: most of the per-turn spend
    # is in the agent fan-out, and Haiku is cheap enough to iterate on
    # without worrying. Flip back to Sonnet in config for production.
    role_models: dict[str, str] = Field(default_factory=lambda: {
        "event_router": "claude-sonnet-4-6",
        "narrator": "claude-sonnet-4-6",
        "agent": "claude-haiku-4-5",
        "character_gen": "claude-sonnet-4-6",
        # Terse post-turn summarization (delta notes for the router).
        # Cheap, narrow task — Haiku is plenty.
        "summarizer": "claude-haiku-4-5",
    })

    default_model: str = "claude-sonnet-4-6"
    max_retries: int = 2
    retry_base_delay: float = 1.0
    # Anthropic's server-side deadline for a single request is 10 minutes.
    # Client timeouts shorter than that truncate long structured-output
    # grammar-compilation passes and surface as "Request timed out or
    # interrupted." See https://docs.anthropic.com/en/api/errors#long-requests
    # — they explicitly recommend using streaming (we do) plus not undercutting
    # the server's own deadline with a shorter client-side one.
    timeout: float = 600.0
    # Compaction trigger (beta). Sonnet 4.6 has a 1M context window with no
    # long-context pricing tier, so we defer compaction until we're within
    # ~100K tokens of the window limit. Tune down for smaller-window models.
    compact_trigger_tokens: int = 900_000

    # Cache TTL for ephemeral prompt-cache blocks. "1h" is the longest
    # Anthropic supports; "5m" is the cheaper write-cost tier. Sessions
    # with long gaps between turns benefit from "1h"; short playtests
    # are indifferent. Can be overridden per-session via SessionSettings
    # (see app/schemas/state.py::SessionSettings) or globally at boot
    # via the ANTHROPIC_CACHE_TTL env var.
    cache_ttl: str = "1h"

    def model_for_role(self, role: str) -> str:
        return self.role_models.get(role, self.default_model)

    @classmethod
    def from_env(cls) -> LLMConfig:
        ttl = os.environ.get("ANTHROPIC_CACHE_TTL", "1h")
        if ttl not in ("5m", "1h"):
            raise ValueError(
                f"ANTHROPIC_CACHE_TTL must be '5m' or '1h', got {ttl!r}"
            )
        return cls(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            cache_ttl=ttl,
        )
