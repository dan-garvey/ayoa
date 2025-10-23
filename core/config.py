"""Configuration management for the game engine."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    base_url: str
    api_key: Optional[str]
    model_name: Optional[str]  # Now optional - will be auto-detected from server
    max_context_tokens: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables."""
        return cls(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("MODEL_NAME"),  # Optional - auto-detect if not set
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "8192")),
        )


@dataclass
class RoleParams:
    """Parameters for a specific role's LLM calls."""

    temperature: float
    top_p: float
    max_tokens: int
    json_mode: bool = False


@dataclass
class EngineConfig:
    """Configuration for the game engine."""

    max_active_characters_per_turn: int
    rng_seed: int
    director_params: RoleParams
    storyteller_params: RoleParams
    character_default_params: RoleParams

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Create config from environment variables."""
        return cls(
            max_active_characters_per_turn=int(
                os.getenv("MAX_ACTIVE_CHARACTERS_PER_TURN", "4")
            ),
            rng_seed=int(os.getenv("RNG_SEED", "1337")),
            director_params=RoleParams(
                temperature=float(os.getenv("DIRECTOR_TEMPERATURE", "0.2")),
                top_p=0.9,
                max_tokens=512,
                json_mode=True,
            ),
            storyteller_params=RoleParams(
                temperature=float(os.getenv("STORYTELLER_TEMPERATURE", "0.7")),
                top_p=0.95,
                max_tokens=700,
                json_mode=False,
            ),
            character_default_params=RoleParams(
                temperature=float(os.getenv("CHARACTER_DEFAULT_TEMPERATURE", "0.7")),
                top_p=0.9,
                max_tokens=180,
                json_mode=True,
            ),
        )


@dataclass
class DatabaseConfig:
    """Database configuration."""

    url: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(url=os.getenv("DATABASE_URL", "sqlite:///./core.db"))


# Global config instances
llm_config = LLMConfig.from_env()
engine_config = EngineConfig.from_env()
db_config = DatabaseConfig.from_env()
