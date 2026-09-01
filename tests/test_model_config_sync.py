from pathlib import Path
import json

from app.bot.engine_bridge import EngineBridge
from app.engine.model_config_sync import (
    runtime_model_config,
    sync_checkpoint_runtime_models,
)
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CURRENT_SCHEMA_VERSION, CheckpointFile
from app.schemas.state import ModelConfig, SessionConfig, SessionState


def _stale_checkpoint(story_id: str) -> CheckpointFile:
    stale = ModelConfig(
        event_router="claude-sonnet-5",
        narrator="claude-sonnet-5",
        agent_default="claude-sonnet-5",
    )
    return CheckpointFile(
        session=SessionState(
            session_id=story_id,
            story_id=story_id,
            config=SessionConfig(models=stale),
        ),
    )


def test_sync_checkpoint_runtime_models_uses_actual_llm_config():
    ckpt = _stale_checkpoint("story")
    llm_config = LLMConfig(
        api_key="sk-ant-test",
        openai_api_key="sk-openai-test",
        role_models={
            "event_router": "gpt-5.1",
            "narrator": "openai:gpt-5.1",
            "dnd_combat_manager": "gpt-5-mini",
            "agent": "claude-sonnet-5",
            "agent_standard": "gpt-5.6-luna",
            "agent_convenience": "claude-sonnet-5",
            "character_manager": "claude-sonnet-5",
            "image_director": "gpt-5-mini",
        },
    )

    changed = sync_checkpoint_runtime_models(ckpt, llm_config)

    expected = runtime_model_config(llm_config)
    assert changed is True
    assert ckpt.session.config.models == expected
    assert not hasattr(ckpt, "config")
    assert ckpt.session.config.models.event_router == "openai:gpt-5.1"
    assert ckpt.session.config.models.dnd_combat_manager == "openai:gpt-5-mini"
    assert ckpt.session.config.models.agent_default == "anthropic:claude-sonnet-5"
    assert ckpt.session.config.models.agent_standard == "openai:gpt-5.6-luna"
    assert ckpt.session.config.models.agent_convenience == "anthropic:claude-sonnet-5"
    assert ckpt.session.config.models.character_manager == "anthropic:claude-sonnet-5"
    assert ckpt.session.config.models.image_director == "openai:gpt-5-mini"


def test_runtime_model_config_defaults_label_reviewed_openai_roles():
    models = runtime_model_config(LLMConfig())

    assert models.event_router == "openai:gpt-5.6-terra"
    assert models.narrator == "openai:gpt-5.6-terra"
    assert models.dnd_combat_manager == "openai:gpt-5-mini"
    assert models.agent_default == "openai:gpt-5.6-luna"
    assert models.agent_standard == "openai:gpt-5.6-luna"
    assert models.agent_convenience == "openai:gpt-5.6-luna"
    assert models.character_manager == "openai:gpt-5.6-luna"
    assert models.image_director == "openai:gpt-5-mini"


def test_checked_in_story_seeds_use_luna_for_every_character_agent_tier():
    repo_root = Path(__file__).resolve().parent.parent
    seed_paths = [
        repo_root
        / "app/storage/story_templates/synthetic_checkpoint/ckpt_0000.json",
        *sorted((repo_root / "app/storage/stories").glob("*/ckpt_0000.json")),
    ]

    checked_paths = []
    for seed_path in seed_paths:
        raw = json.loads(seed_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != CURRENT_SCHEMA_VERSION:
            continue
        checked_paths.append(seed_path)
        models = CheckpointFile.model_validate(raw).session.config.models
        assert {
            models.agent_default.removeprefix("openai:"),
            models.agent_standard.removeprefix("openai:"),
            models.agent_convenience.removeprefix("openai:"),
        } == {"gpt-5.6-luna"}, seed_path
    assert checked_paths


def test_load_story_into_session_rewrites_stale_story_models(tmp_path: Path):
    llm_config = LLMConfig(
        api_key="sk-ant-test",
        openai_api_key="sk-openai-test",
    )
    bridge = EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
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
    assert ckpt.session.config.models == expected

    ckpt_0000 = CheckpointFile.model_validate_json(
        (tmp_path / "sessions" / "session_1" / "ckpt_0000.json").read_text()
    )
    ckpt_0001 = CheckpointFile.model_validate_json(
        (tmp_path / "sessions" / "session_1" / "ckpt_0001.json").read_text()
    )
    assert ckpt_0000.session.config.models == expected
    assert ckpt_0001.session.config.models == expected
    assert not hasattr(ckpt_0000, "config")
    assert not hasattr(ckpt_0001, "config")


def test_load_story_into_session_rejects_stale_schema(tmp_path: Path):
    bridge = EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
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
        assert "Regenerate" in str(exc)
    else:
        raise AssertionError("stale story schema should be rejected")
    assert not (tmp_path / "sessions" / "session_1" / "ckpt_0000.json").exists()
