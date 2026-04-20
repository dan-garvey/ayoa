from __future__ import annotations

import os

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    api_key: str = ""

    # The discriminator role was merged into the event router in v2; no
    # caller asks for role="discriminator" anymore, so it's omitted here.
    role_models: dict[str, str] = Field(default_factory=lambda: {
        "event_router": "claude-sonnet-4-6",
        "narrator": "claude-sonnet-4-6",
        "agent": "claude-sonnet-4-6",
        "character_gen": "claude-sonnet-4-6",
    })

    default_model: str = "claude-sonnet-4-6"
    max_retries: int = 2
    retry_base_delay: float = 1.0
    timeout: float = 60.0
    # Compaction trigger (beta). Sonnet 4.6 has a 1M context window with no
    # long-context pricing tier, so we defer compaction until we're within
    # ~100K tokens of the window limit. Tune down for smaller-window models.
    compact_trigger_tokens: int = 900_000

    def model_for_role(self, role: str) -> str:
        return self.role_models.get(role, self.default_model)

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
