from pathlib import Path
import json

from app.bot.engine_bridge import EngineBridge
from app.engine.model_config_sync import (
    runtime_model_config,
    sync_checkpoint_runtime_models,
)
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import ModelConfig, SessionConfig, SessionState


def _stale_checkpoint(story_id: str) -> CheckpointFile:
    stale = ModelConfig(
        event_router="claude-sonnet-4-6",
        narrator="claude-sonnet-4-6",
        discriminator="claude-sonnet-4-6",
        agent_default="claude-sonnet-4-6",
    )
    return CheckpointFile(
        session=SessionState(
            session_id=story_id,
            story_id=story_id,
            config=SessionConfig(models=stale),
        ),
        config=SessionConfig(models=stale),
    )


def test_sync_checkpoint_runtime_models_uses_actual_llm_config():
    ckpt = _stale_checkpoint("story")
    llm_config = LLMConfig(
        api_key="sk-ant-test",
        openai_api_key="sk-openai-test",
        role_models={
            "event_router": "gpt-5.1",
            "narrator": "openai:gpt-5.1",
            "agent": "claude-sonnet-4-6",
            "agent_convenience": "claude-haiku-4-5",
            "character_gen": "gpt-5.1",
        },
    )

    changed = sync_checkpoint_runtime_models(ckpt, llm_config)

    expected = runtime_model_config(llm_config)
    assert changed is True
    assert ckpt.config.models == expected
    assert ckpt.session.config.models == expected
    assert ckpt.config.models.event_router == "openai:gpt-5.1"
    assert ckpt.config.models.agent_default == "anthropic:claude-sonnet-4-6"
    assert ckpt.config.models.agent_convenience == "anthropic:claude-haiku-4-5"


def test_runtime_model_config_defaults_label_mixed_provider_roles():
    models = runtime_model_config(LLMConfig())

    assert models.event_router == "openai:gpt-5.2"
    assert models.discriminator == "openai:gpt-5.2"
    assert models.narrator == "anthropic:claude-sonnet-4-6"
    assert models.agent_default == "anthropic:claude-opus-4-6"
    assert models.agent_convenience == "anthropic:claude-sonnet-4-6"


def test_load_story_into_session_rewrites_stale_story_models(tmp_path: Path):
    llm_config = LLMConfig(
        api_key="sk-ant-test",
        openai_api_key="sk-openai-test",
    )
    bridge = EngineBridge(
        saves_dir=str(tmp_path),
        prompts_dir="app/prompts",
        llm_config=llm_config,
    )
    story_id = "stale_story"
    story_dir = tmp_path / "stories" / story_id
    story_dir.mkdir(parents=True)
    (story_dir / "ckpt_0000.json").write_text(
        _stale_checkpoint(story_id).model_dump_json(indent=2)
    )

    bridge.create_empty_session("session_1")
    ckpt = bridge.load_story_into_session("session_1", story_id)

    expected = runtime_model_config(llm_config)
    assert ckpt.config.models == expected
    assert ckpt.session.config.models == expected

    ckpt_0000 = CheckpointFile.model_validate_json(
        (tmp_path / "sessions" / "session_1" / "ckpt_0000.json").read_text()
    )
    ckpt_0001 = CheckpointFile.model_validate_json(
        (tmp_path / "sessions" / "session_1" / "ckpt_0001.json").read_text()
    )
    assert ckpt_0000.config.models == expected
    assert ckpt_0000.session.config.models == expected
    assert ckpt_0001.config.models == expected
    assert ckpt_0001.session.config.models == expected


def test_load_story_into_session_rejects_stale_schema(tmp_path: Path):
    bridge = EngineBridge(
        saves_dir=str(tmp_path),
        prompts_dir="app/prompts",
        llm_config=LLMConfig(api_key="sk-ant-test"),
    )
    story_id = "stale_schema_story"
    story_dir = tmp_path / "stories" / story_id
    story_dir.mkdir(parents=True)
    data = json.loads(_stale_checkpoint(story_id).model_dump_json())
    data["schema_version"] = "3.0"
    (story_dir / "ckpt_0000.json").write_text(json.dumps(data))

    bridge.create_empty_session("session_1")

    try:
        bridge.load_story_into_session("session_1", story_id)
    except ValueError as exc:
        assert "schema_version" in str(exc)
        assert "Re-import" in str(exc)
    else:
        raise AssertionError("stale story schema should be rejected")
    assert not (tmp_path / "sessions" / "session_1" / "ckpt_0000.json").exists()
