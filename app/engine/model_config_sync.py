from __future__ import annotations

from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import ModelConfig


def _runtime_model_label(config: LLMConfig, role: str) -> str:
    provider = config.provider_for_role(role)
    model = config.model_for_role(role)
    return f"{provider}:{model}"


def runtime_model_config(config: LLMConfig) -> ModelConfig:
    return ModelConfig(
        event_router=_runtime_model_label(config, "event_router"),
        narrator=_runtime_model_label(config, "narrator"),
        discriminator=_runtime_model_label(config, "event_router"),
        agent_default=_runtime_model_label(config, "agent"),
        agent_convenience=_runtime_model_label(config, "agent_convenience"),
    )


def sync_checkpoint_runtime_models(
    checkpoint: CheckpointFile,
    config: LLMConfig,
) -> bool:
    models = runtime_model_config(config)
    changed = False

    if checkpoint.config.models != models:
        checkpoint.config.models = models
        changed = True

    if checkpoint.session.config.models != models:
        checkpoint.session.config.models = models
        changed = True

    return changed
