from __future__ import annotations

import os

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    gateway_url: str = "https://llm-api.amd.com/OnPrem"
    api_key: str = "dummy"
    subscription_key: str = ""
    user: str = ""

    role_models: dict[str, str] = Field(default_factory=lambda: {
        "event_router": "GPT-oss-120B",
        "narrator": "GPT-oss-120B",
        "discriminator": "GPT-oss-120B",
        "agent": "GPT-oss-120B",
        "character_gen": "GPT-oss-120B",
    })

    default_model: str = "GPT-oss-120B"
    default_temperature: float = 0.7
    default_max_tokens: int = 1000
    max_retries: int = 2
    retry_base_delay: float = 1.0
    timeout: float = 60.0

    def model_for_role(self, role: str) -> str:
        return self.role_models.get(role, self.default_model)

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            subscription_key=os.environ.get("LLM_GATEWAY_KEY", ""),
            user=os.environ.get("USER", "unknown"),
        )
