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
        # Out-of-character /query consultation. Read-only, short
        # answer, single-character POV bound. Latency matters more
        # than depth here — players are staring at Discord waiting
        # for "what do I see?" / "what was her name?" — so Haiku is
        # the default. Flip to Sonnet via env if you want richer
        # in-fiction refusal flavor.
        "query_handler": "claude-haiku-4-5",
    })

    default_model: str = "claude-sonnet-4-6"
    # Retries are for transient API failures: 529 overloaded, 500/503,
    # network blips, streaming disconnects. Anthropic's overload events
    # in particular ride out in 5-30s windows; 4 retries × exp-backoff
    # with jitter clears most of them without dumping the user. The pre-
    # bump default of 2 surfaced "overloaded_error" to the player on the
    # first burst of concurrent /act commands during the playtest.
    max_retries: int = 4
    retry_base_delay: float = 1.0
    # Symmetric jitter band around the exp-backoff delay, expressed as
    # a fraction. 0.3 means each delay is uniformly sampled from
    # [0.7×delay, 1.3×delay]. Prevents thundering-herd lockstep when
    # multiple players /act simultaneously and all hit overloaded at
    # the same wall-clock instant.
    retry_jitter: float = 0.3
    # Hard cap on any single retry sleep. Prevents a 4th retry from
    # waiting 16s+ on a request the user is staring at; we'd rather
    # surface the failure cleanly than hold the slash command for
    # half a minute.
    retry_max_delay: float = 30.0
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
