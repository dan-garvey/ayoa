"""Tests for the LLM client — unit tests with mocks and integration tests against live API."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.client import (
    LLMClient,
    LLMResponse,
    _openai_strict_json_schema,
    extract_json,
)
from app.llm.config import LLMConfig, live_play_required_roles
from app.schemas.dnd_cat_ii import (
    DndCombatManagerAdjudication,
    DndCombatTurnPlan,
    RollPlan,
    RulesAdjudication,
)
from app.schemas.event_router import DndEventRouterOutput, EventRouterOutput
from app.schemas.events import CanonicalEvent, ObservableFact


def _install_stream_mock(client, *responses):
    """Wire `client._client.messages.stream(**kwargs)` to yield responses in order.

    Each call enters an async context manager whose `get_final_message()` returns
    the next response (or raises it, if it's an Exception).
    """
    iterator = iter(responses)

    def stream_factory(**kwargs):
        nxt = next(iterator)
        ctx = MagicMock()

        async def aenter(_self):
            stream_obj = MagicMock()
            if isinstance(nxt, BaseException):
                stream_obj.get_final_message = AsyncMock(side_effect=nxt)
            else:
                stream_obj.get_final_message = AsyncMock(return_value=nxt)
            return stream_obj

        async def aexit(_self, *args):
            return None

        ctx.__aenter__ = aenter
        ctx.__aexit__ = aexit
        return ctx

    mock = MagicMock(side_effect=stream_factory)
    client._client.messages.stream = mock
    # Structured-output and compaction calls route through the beta
    # client; wire the same mock there so tests hit both paths uniformly.
    client._client.beta.messages.stream = mock
    return mock


# --- extract_json unit tests ---

class TestExtractJson:
    def test_plain_json(self):
        text = '{"key": "value"}'
        assert json.loads(extract_json(text)) == {"key": "value"}

    def test_markdown_fenced(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\n'
        assert json.loads(extract_json(text)) == {"key": "value"}

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        assert json.loads(extract_json(text)) == {"key": "value"}

    def test_surrounded_by_prose(self):
        text = 'Sure! Here is the output:\n{"key": "value"}\nHope that helps!'
        assert json.loads(extract_json(text)) == {"key": "value"}

    def test_nested_braces(self):
        text = '{"outer": {"inner": 1}}'
        result = json.loads(extract_json(text))
        assert result == {"outer": {"inner": 1}}

    def test_no_json(self):
        text = "No JSON here at all"
        assert extract_json(text) == "No JSON here at all"


# --- LLMConfig tests ---

class TestLLMConfig:
    @pytest.mark.parametrize("mode", ["0", "disabled", "shadow"])
    def test_image_director_role_is_not_required_for_disabled_modes(
        self, monkeypatch, mode,
    ):
        monkeypatch.setenv("AYOA_IMAGE_DIRECTOR_ENABLED", mode)

        assert "image_director" not in live_play_required_roles()

    def test_image_director_role_is_required_when_enabled(self, monkeypatch):
        monkeypatch.setenv("AYOA_IMAGE_DIRECTOR_ENABLED", "enabled")

        assert "image_director" in live_play_required_roles()

    def test_defaults_use_gpt_router_narrator_and_anthropic_agents(self):
        config = LLMConfig()
        assert config.default_provider == "openai"
        assert config.default_model == "gpt-5.1"
        assert config.providers_in_use() == {"anthropic", "openai"}
        assert config.role_models["event_router"] == "gpt-5.6-terra"
        assert config.role_models["narrator"] == "gpt-5.6-terra"
        assert config.role_models["dnd_combat_manager"] == "gpt-5-mini"
        assert config.role_models["content_manager"] == "gpt-5-mini"
        assert config.role_models["image_director"] == "gpt-5-mini"
        assert config.role_models["agent"] == "claude-opus-5"
        assert config.role_models["agent_standard"] == "gpt-5.6-luna"
        assert config.role_models["agent_convenience"] == "claude-sonnet-5"
        assert config.role_models["character_manager"] == "claude-sonnet-5"
        assert config.provider_for_role("event_router") == "openai"
        assert config.provider_for_role("narrator") == "openai"
        assert config.provider_for_role("dnd_combat_manager") == "openai"
        assert config.provider_for_role("content_manager") == "openai"
        assert config.provider_for_role("image_director") == "openai"
        assert config.provider_for_role("agent") == "anthropic"
        assert config.provider_for_role("agent_standard") == "openai"
        assert config.provider_for_role("agent_convenience") == "anthropic"
        assert config.provider_for_role("character_manager") == "anthropic"
        assert config.thinking_budget_for_role("agent") == 0
        assert config.thinking_budget_for_role("agent_standard") == 0
        assert config.thinking_budget_for_role("agent_convenience") == 0
        assert config.enable_anthropic_compaction is False
        assert config.openai_reasoning_effort_for_role("content_manager") == "low"
        assert config.openai_reasoning_effort_for_role("image_director") == "low"
        assert config.openai_reasoning_effort_for_role("agent_standard") == "medium"
        assert all(
            effort == "medium"
            for role, effort in config.openai_reasoning_efforts.items()
            if role not in {"content_manager", "image_director"}
        )
        assert config.openai_reasoning_summary_for_role("event_router") == "auto"
        assert config.openai_reasoning_summary_for_role("narrator") == ""

    def test_model_for_role(self):
        config = LLMConfig(role_models={"narrator": "big-model", "agent": "small-model"})
        assert config.model_for_role("narrator") == "big-model"
        assert config.model_for_role("agent") == "small-model"
        assert config.model_for_role("unknown") == config.default_model

    def test_provider_for_role(self):
        config = LLMConfig(
            role_models={
                "narrator": "openai:gpt-5.4-mini",
                "agent": "claude-sonnet-5",
            },
            role_providers={"event_router": "openai"},
        )
        assert config.provider_for_role("narrator") == "openai"
        assert config.model_for_role("narrator") == "gpt-5.4-mini"
        assert config.provider_for_role("agent") == "anthropic"
        assert config.provider_for_role("event_router") == "openai"

    def test_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "test-key",
                "OPENAI_API_KEY": "openai-key",
                "OPEN_AI_NARRATOR": "narrator-openai-key",
                "OPEN_AI_ROUTER": "router-openai-key",
                "OPEN_AI_AGENT": "agent-openai-key",
                "OPEN_AI_COMBAT_MANAGER": "combat-manager-openai-key",
                "OPEN_AI_CONTENT_MANAGER": "content-manager-openai-key",
                "LLM_PROVIDER_ROUTER": "openai",
                "LLM_PROVIDER_NARRATOR": "openai",
                "LLM_MODEL_NARRATOR": "gpt-5.4-mini",
                "LLM_OPENAI_REASONING_ROUTER": "high",
                "LLM_OPENAI_REASONING_SUMMARY_ROUTER": "auto",
                "LLM_REASONING_NARRATOR": "low",
                "ANTHROPIC_COMPACTION_ENABLED": "true",
            },
            clear=True,
        ):
            config = LLMConfig.from_env()
            assert config.api_key == "test-key"
            assert config.openai_api_key == "openai-key"
            assert (
                config.api_key_for_provider("openai", role="agent")
                == "agent-openai-key"
            )
            assert (
                config.api_key_for_provider("openai", role="agent_standard")
                == "agent-openai-key"
            )
            assert (
                config.api_key_for_provider("openai", role="agent_convenience")
                == "agent-openai-key"
            )
            assert (
                config.api_key_for_provider("openai", role="event_router")
                == "router-openai-key"
            )
            assert (
                config.api_key_for_provider("openai", role="narrator")
                == "narrator-openai-key"
            )
            assert (
                config.api_key_for_provider("openai", role="dnd_combat_manager")
                == "combat-manager-openai-key"
            )
            assert (
                config.api_key_for_provider("openai", role="content_manager")
                == "content-manager-openai-key"
            )
            assert config.provider_for_role("event_router") == "openai"
            assert config.provider_for_role("narrator") == "openai"
            assert config.model_for_role("narrator") == "gpt-5.4-mini"
            assert config.openai_reasoning_effort_for_role("event_router") == "high"
            assert config.openai_reasoning_summary_for_role("event_router") == "auto"
            assert config.openai_reasoning_effort_for_role("narrator") == "low"
            assert config.enable_anthropic_compaction is True

    def test_agent_roles_share_one_openai_credential_name(self):
        config = LLMConfig()

        for role in (
            "agent",
            "agent_standard",
            "agent_convenience",
        ):
            assert config.openai_role_api_key_env_names(role) == (
                "OPEN_AI_AGENT",
            )

    def test_dnd_combat_manager_does_not_reuse_router_openai_key(self):
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "test-key",
                "OPEN_AI_ROUTER": "router-openai-key",
            },
            clear=True,
        ):
            config = LLMConfig.from_env()

        assert config.openai_api_key == ""
        assert (
            config.api_key_for_provider("openai", role="dnd_combat_manager")
            == ""
        )

    def test_dnd_combat_manager_openai_key_prefers_explicit_role_key(self):
        with patch.dict(
            "os.environ",
            {
                "OPEN_AI_COMBAT_MANAGER": "combat-manager-key",
            },
            clear=True,
        ):
            config = LLMConfig.from_env()

        assert (
            config.api_key_for_provider("openai", role="dnd_combat_manager")
            == "combat-manager-key"
        )

    def test_missing_credentials_reports_role_env_names(self):
        config = LLMConfig(
            api_key="anthropic-key",
            openai_role_api_keys={"narrator": "narrator-key"},
        )

        missing = config.missing_credentials(
            {"agent", "event_router", "narrator"}
        )

        assert len(missing) == 1
        assert missing[0].role == "event_router"
        assert missing[0].provider == "openai"
        assert "OPEN_AI_ROUTER" in missing[0].env_names
        assert "OPENAI_API_KEY" in missing[0].env_names
        assert "narrator-key" not in missing[0].env_names

    def test_missing_credentials_reports_content_manager_env_name(self):
        config = LLMConfig(api_key="anthropic-key")

        missing = config.missing_credentials({"content_manager"})

        assert len(missing) == 1
        assert missing[0].role == "content_manager"
        assert missing[0].provider == "openai"
        assert "OPEN_AI_CONTENT_MANAGER" in missing[0].env_names
        assert "OPENAI_API_KEY" in missing[0].env_names

    def test_missing_credentials_reports_anthropic_role(self):
        config = LLMConfig(openai_api_key="openai-key")

        missing = config.missing_credentials({"agent", "event_router"})

        assert len(missing) == 1
        assert missing[0].role == "agent"
        assert missing[0].provider == "anthropic"
        assert missing[0].env_names == ("ANTHROPIC_API_KEY",)

    def test_missing_credentials_respects_anthropic_role_model_override(self):
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "anthropic-key",
                "LLM_ROLE_MODELS": ",".join((
                    "event_router=anthropic:claude-sonnet-5",
                    "narrator=anthropic:claude-sonnet-5",
                )),
            },
            clear=True,
        ):
            config = LLMConfig.from_env()

        assert config.missing_credentials({
            "event_router",
            "narrator",
        }) == ()

# --- LLMClient unit tests (mocked API) ---

def _make_mock_response(
    content: str,
    model: str = "claude-haiku-4-5",
    parsed=None,
    *,
    cache_read: int = 0,
    cache_write: int | None = 0,
    cache_write_5m: int = 0,
    cache_write_1h: int = 0,
):
    """Create a mock Anthropic Message response.

    If `parsed` is supplied, the text block carries `parsed_output` — this is
    what the SDK populates when `output_format` is set on the request and the
    API emits a schema-conforming response.
    """
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = content
    text_block.parsed_output = parsed

    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 20
    usage.cache_read_input_tokens = cache_read
    usage.cache_creation_input_tokens = cache_write
    if cache_write_5m or cache_write_1h:
        usage.cache_creation = MagicMock(
            ephemeral_5m_input_tokens=cache_write_5m,
            ephemeral_1h_input_tokens=cache_write_1h,
        )
    else:
        usage.cache_creation = None

    response = MagicMock()
    response.content = [text_block]
    response.model = model
    response.usage = usage
    response.stop_reason = "end_turn"
    return response


def _make_openai_response(
    content: str,
    model: str = "gpt-5.4-mini",
    *,
    output_tokens: int = 7,
    reasoning_tokens: int = 0,
    reasoning_summaries: list[str] | None = None,
):
    usage = MagicMock()
    usage.input_tokens = 15
    usage.output_tokens = output_tokens
    usage.total_tokens = 15 + output_tokens
    usage.input_tokens_details = MagicMock(cached_tokens=5)
    usage.output_tokens_details = MagicMock(reasoning_tokens=reasoning_tokens)

    response = MagicMock()
    response.output_text = content
    response.model = model
    response.usage = usage
    response.status = "completed"
    response.output = []
    if reasoning_summaries:
        item = MagicMock()
        item.type = "reasoning"
        item.summary = []
        for text in reasoning_summaries:
            summary = MagicMock()
            summary.text = text
            item.summary.append(summary)
        response.output.append(item)
    return response


@pytest.fixture
def mock_config():
    return LLMConfig(
        api_key="fake-key",
        role_models={
            "event_router": "claude-sonnet-5",
            "narrator": "claude-haiku-4-5",
            "agent": "claude-opus-5",
            "agent_convenience": "claude-haiku-4-5",
            "character_manager": "claude-sonnet-5",
        },
        max_retries=1,
        retry_base_delay=0.01,
    )


@pytest.fixture
def client(mock_config):
    return LLMClient(config=mock_config)


class TestLLMClientComplete:
    @pytest.mark.asyncio
    async def test_basic_completion(self, client):
        _install_stream_mock(client, _make_mock_response("Hello world"))

        result = await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=100,
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello world"
        assert result.model == "claude-haiku-4-5"
        assert result.usage["total_tokens"] == 30
        assert result.parsed is None

    @pytest.mark.asyncio
    async def test_system_message_peeled_to_top_level(self, client):
        mock = _install_stream_mock(client, _make_mock_response("ok"))

        await client.complete(
            role="narrator",
            messages=[
                {"role": "system", "content": "You are a bard."},
                {"role": "user", "content": "Sing."},
            ],
            cache=False,
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["system"] == "You are a bard."
        assert call_kwargs["messages"] == [{"role": "user", "content": "Sing."}]

    @pytest.mark.asyncio
    async def test_cache_breakpoint_on_system_when_enabled(self, client):
        mock = _install_stream_mock(client, _make_mock_response("ok"))

        await client.complete(
            role="narrator",
            messages=[
                {"role": "system", "content": "You are a bard."},
                {"role": "user", "content": "Sing."},
            ],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["system"] == [
            {
                "type": "text",
                "text": "You are a bard.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]
        assert "cache_control" not in call_kwargs

    @pytest.mark.asyncio
    async def test_single_turn_keeps_user_tail_uncached(self, client):
        mock = _install_stream_mock(client, _make_mock_response("ok"))

        await client.complete(
            role="narrator",
            messages=[
                {"role": "system", "content": "You are a bard."},
                {"role": "user", "content": "Sing."},
            ],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "Sing."}]

    @pytest.mark.asyncio
    async def test_history_caches_last_user_tail(self, client):
        mock = _install_stream_mock(client, _make_mock_response("ok"))

        await client.complete(
            role="narrator",
            messages=[
                {"role": "system", "content": "You are a bard."},
                {"role": "user", "content": "Sing."},
                {"role": "assistant", "content": "A song begins."},
                {"role": "user", "content": "Continue."},
            ],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "Sing."},
            {"role": "assistant", "content": "A song begins."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Continue.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
        ]

    @pytest.mark.asyncio
    async def test_no_cache_when_no_system(self, client):
        """No shared prefix → no cache marker anywhere."""
        mock = _install_stream_mock(client, _make_mock_response("ok"))

        await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock.call_args.kwargs
        assert "system" not in call_kwargs
        assert "cache_control" not in call_kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("role", "expected_model"),
        (
            ("agent", "claude-opus-5"),
            ("character_manager", "claude-sonnet-5"),
        ),
    )
    async def test_claude_5_omits_legacy_thinking_and_sampling_parameters(
        self,
        client,
        role,
        expected_model,
    ):
        mock = _install_stream_mock(client, _make_mock_response("ok"))

        await client.complete(
            role=role,
            messages=[{"role": "user", "content": "Author a character."}],
            temperature=0.6,
            max_tokens=100,
        )

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["model"] == expected_model
        assert "temperature" not in call_kwargs
        assert "thinking" not in call_kwargs

    @pytest.mark.asyncio
    async def test_cache_suppressed_when_disabled(self, client):
        mock = _install_stream_mock(client, _make_mock_response("ok"))

        await client.complete(
            role="narrator",
            messages=[
                {"role": "system", "content": "You are a bard."},
                {"role": "user", "content": "Hi"},
            ],
            cache=False,
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["system"] == "You are a bard."
        assert "cache_control" not in call_kwargs

    @pytest.mark.asyncio
    async def test_structured_output_passes_output_format(self, client):
        """When response_model is set, we hand the Pydantic class to the SDK via output_format."""
        event = CanonicalEvent.model_validate({
            "world_adjudication": {
                "feasible": True,
            },
            "observable_facts": [ObservableFact.all("The door swings open.")],
        })
        mock = _install_stream_mock(client, _make_mock_response("{}", parsed=event))

        result = await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "open door"}],
            response_model=CanonicalEvent,
            temperature=0.5,
            max_tokens=100,
        )

        assert mock.call_args.kwargs["output_format"] is CanonicalEvent
        assert isinstance(result.parsed, CanonicalEvent)
        assert result.parsed.world_adjudication.feasible is True

    @pytest.mark.asyncio
    async def test_structured_output_missing_parsed_raises(self, client):
        """If output_format was set but SDK returned no parsed_output, fail loudly."""
        _install_stream_mock(client, _make_mock_response("not valid", parsed=None))

        with pytest.raises(ValueError, match="no parsed output"):
            await client.complete(
                role="narrator",
                messages=[{"role": "user", "content": "x"}],
                response_model=CanonicalEvent,
                temperature=0.5,
                max_tokens=100,
            )

    @pytest.mark.asyncio
    async def test_openai_provider_uses_responses_api(self):
        config = LLMConfig(
            openai_api_key="fake-openai-key",
            role_models={"narrator": "gpt-5.1"},
            role_providers={"narrator": "openai"},
            max_retries=0,
        )
        client = LLMClient(config=config)
        openai_client = MagicMock()
        openai_client.responses.create = AsyncMock(
            return_value=_make_openai_response("Hello from OpenAI")
        )
        client._openai_clients["narrator"] = openai_client

        result = await client.complete(
            role="narrator",
            messages=[
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Say hi."},
            ],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = openai_client.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5.1"
        assert call_kwargs["instructions"] == "You are concise."
        assert call_kwargs["input"] == [{"role": "user", "content": "Say hi."}]
        assert call_kwargs["reasoning"] == {"effort": "medium"}
        assert "temperature" not in call_kwargs
        assert call_kwargs["max_output_tokens"] == 100
        assert result.content == "Hello from OpenAI"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["cache_read_input_tokens"] == 5
        assert result.usage["visible_completion_tokens"] == 7
        assert result.assistant_content == [
            {"type": "text", "text": "Hello from OpenAI"}
        ]

    @pytest.mark.asyncio
    async def test_anthropic_usage_reads_per_ttl_cache_write_tokens(
        self, client, caplog,
    ):
        _install_stream_mock(
            client,
            _make_mock_response(
                "ok",
                model="claude-opus-5",
                cache_read=512,
                cache_write=None,
                cache_write_1h=4096,
            ),
        )

        with caplog.at_level(logging.INFO, logger="app.llm.client"):
            result = await client.complete(
                role="agent",
                messages=[
                    {"role": "system", "content": "You are a character."},
                    {"role": "user", "content": "Act."},
                ],
                temperature=0.5,
                max_tokens=100,
            )

        assert result.usage["cache_read_input_tokens"] == 512
        assert result.usage["cache_creation_input_tokens"] == 4096
        assert result.usage["cache_creation_1h_input_tokens"] == 4096
        assert result.usage["full_input_tokens"] == 4618
        assert "model=claude-opus-5" in caplog.text
        assert "cache_write=4096" in caplog.text

    @pytest.mark.asyncio
    async def test_openai_usage_logs_reasoning_tokens(self, caplog):
        config = LLMConfig(
            openai_api_key="fake-openai-key",
            role_models={"event_router": "gpt-5.2"},
            role_providers={"event_router": "openai"},
            max_retries=0,
        )
        client = LLMClient(config=config)
        openai_client = MagicMock()
        openai_client.responses.create = AsyncMock(
            return_value=_make_openai_response(
                "ok",
                model="gpt-5.2",
                output_tokens=11,
                reasoning_tokens=4,
                reasoning_summaries=["Checked scene state, then chose a route."],
            )
        )
        client._openai_clients["event_router"] = openai_client

        with caplog.at_level(logging.INFO, logger="app.llm.client"):
            result = await client.complete(
                role="event_router",
                messages=[{"role": "user", "content": "Route this."}],
                temperature=0.5,
                max_tokens=100,
            )

        assert result.usage["completion_tokens"] == 11
        assert result.usage["reasoning_tokens"] == 4
        assert result.usage["visible_completion_tokens"] == 7
        assert result.reasoning_summaries == [
            "Checked scene state, then chose a route."
        ]
        assert "role=event_router" in caplog.text
        assert "model=gpt-5.2" in caplog.text
        assert "reasoning=4" in caplog.text
        assert "visible_out=7" in caplog.text
        assert "Checked scene state" in caplog.text
        call_kwargs = openai_client.responses.create.call_args.kwargs
        assert call_kwargs["reasoning"] == {
            "effort": "medium",
            "summary": "auto",
        }

    @pytest.mark.asyncio
    async def test_openai_reasoning_omitted_for_non_reasoning_models(self):
        config = LLMConfig(
            openai_api_key="fake-openai-key",
            role_models={"narrator": "gpt-4o"},
            role_providers={"narrator": "openai"},
            max_retries=0,
        )
        client = LLMClient(config=config)
        openai_client = MagicMock()
        openai_client.responses.create = AsyncMock(
            return_value=_make_openai_response("Hello from OpenAI", model="gpt-4o")
        )
        client._openai_clients["narrator"] = openai_client

        await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "Say hi."}],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = openai_client.responses.create.call_args.kwargs
        assert "reasoning" not in call_kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_openai_client_uses_role_specific_api_key(self):
        config = LLMConfig(
            openai_api_key="global-openai-key",
            openai_role_api_keys={
                "narrator": "narrator-openai-key",
                "event_router": "router-openai-key",
            },
        )
        client = LLMClient(config=config)

        with patch("app.llm.client.openai.AsyncOpenAI") as openai_ctor:
            client._get_openai_client("narrator")
            client._get_openai_client("event_router")
            client._get_openai_client("character_manager")

        assert openai_ctor.call_args_list[0].kwargs["api_key"] == "narrator-openai-key"
        assert openai_ctor.call_args_list[1].kwargs["api_key"] == "router-openai-key"
        assert openai_ctor.call_args_list[2].kwargs["api_key"] == "global-openai-key"

    def test_openai_client_missing_role_key_error_names_role_envs(self):
        client = LLMClient(config=LLMConfig(openai_api_key=""))

        with pytest.raises(RuntimeError, match="dnd_combat_manager"):
            client._get_openai_client("dnd_combat_manager")

        with pytest.raises(RuntimeError, match="OPEN_AI_COMBAT_MANAGER"):
            client._get_openai_client("dnd_combat_manager")

        with pytest.raises(RuntimeError, match="OPEN_AI_CONTENT_MANAGER"):
            client._get_openai_client("content_manager")

    @pytest.mark.asyncio
    async def test_openai_structured_output_passes_json_schema(self):
        event = CanonicalEvent.model_validate({
            "world_adjudication": {
                "feasible": True,
            },
            "observable_facts": [ObservableFact.all("The door swings open.")],
        })
        config = LLMConfig(
            openai_api_key="fake-openai-key",
            role_models={"event_router": "openai:gpt-5.4"},
            max_retries=0,
        )
        client = LLMClient(config=config)
        openai_client = MagicMock()
        openai_client.responses.create = AsyncMock(
            return_value=_make_openai_response(event.model_dump_json(), model="gpt-5.4")
        )
        client._openai_clients["event_router"] = openai_client

        result = await client.complete(
            role="event_router",
            messages=[{"role": "user", "content": "open door"}],
            response_model=CanonicalEvent,
            temperature=0.5,
            max_tokens=100,
        )

        text_format = openai_client.responses.create.call_args.kwargs["text"]["format"]
        reasoning = openai_client.responses.create.call_args.kwargs["reasoning"]
        assert text_format["type"] == "json_schema"
        assert text_format["name"] == "CanonicalEvent"
        assert text_format["strict"] is True
        assert reasoning == {"effort": "medium", "summary": "auto"}
        assert isinstance(result.parsed, CanonicalEvent)
        assert result.parsed.world_adjudication.feasible is True

    def test_openai_structured_schema_requires_every_object_property(self):
        """OpenAI strict JSON Schema rejects Pydantic default fields unless
        they are still listed as required in the provider-facing schema."""
        for model in (
            EventRouterOutput,
            DndEventRouterOutput,
            RollPlan,
            DndCombatTurnPlan,
            RulesAdjudication,
            DndCombatManagerAdjudication,
        ):
            schema = _openai_strict_json_schema(model)
            failures = []
            defaults = []
            annotations = []

            def walk(node, path=()):
                if isinstance(node, dict):
                    if "default" in node:
                        defaults.append(path)
                    for key in ("description", "title"):
                        if key in node and (not path or path[-1] != "properties"):
                            annotations.append(path + (key,))
                    properties = node.get("properties")
                    if isinstance(properties, dict):
                        required = set(node.get("required") or [])
                        missing = set(properties) - required
                        if missing:
                            failures.append((path, sorted(missing)))
                        if node.get("additionalProperties") is not False:
                            failures.append((path, ["additionalProperties"]))
                    for key, value in node.items():
                        walk(value, path + (key,))
                elif isinstance(node, list):
                    for index, value in enumerate(node):
                        walk(value, path + (str(index),))

            walk(schema)
            assert failures == []
            assert defaults == []
            assert annotations == []

    def test_openai_structured_schema_omits_prompt_irrelevant_internals(self):
        schema_text = json.dumps(
            _openai_strict_json_schema(DndCombatManagerAdjudication)
        ).lower()

        assert "router-authored" not in schema_text
        assert "event-router" not in schema_text
        assert "combat engine" not in schema_text
        assert "structured output schemas" not in schema_text

    @pytest.mark.asyncio
    async def test_compact_false_uses_stable_stream(self, client):
        mock = _install_stream_mock(client, _make_mock_response("ok"))

        await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock.call_args.kwargs
        assert "context_management" not in call_kwargs
        assert "betas" not in call_kwargs

    @pytest.mark.asyncio
    async def test_compact_true_does_not_compact_by_default(self, client):
        mock = _install_stream_mock(client, _make_mock_response("ok"))

        await client.complete(
            role="event_router",
            messages=[{"role": "user", "content": "hi"}],
            compact=True,
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock.call_args.kwargs
        assert "context_management" not in call_kwargs
        assert "betas" not in call_kwargs

    @pytest.mark.asyncio
    async def test_compact_enabled_still_skips_unsupported_haiku(self, client):
        client.config.enable_anthropic_compaction = True
        mock = _install_stream_mock(client, _make_mock_response("ok"))

        await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "hi"}],
            compact=True,
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock.call_args.kwargs
        assert "context_management" not in call_kwargs
        assert "betas" not in call_kwargs

    @pytest.mark.asyncio
    async def test_compact_true_adds_context_management_when_enabled(self, client):
        """compact=True switches to the beta stream with compaction
        config + beta header. Uses `event_router` as the role because
        the client's `compact_supported` gate (see app/llm/client.py)
        only enables compaction on Sonnet/Opus class models — Haiku
        4.5 returns 400 for the `compact_20260112` strategy and is
        silently downgraded. Narrator switched to Haiku in r10
        (Option B's narrowed render contract makes the cheaper model
        sufficient), so this test now uses event_router which remains
        Sonnet."""
        client.config.enable_anthropic_compaction = True
        # Install the mock on beta.messages.stream instead.
        from unittest.mock import AsyncMock, MagicMock

        stream_mock = MagicMock()
        stream_obj = MagicMock()
        stream_obj.get_final_message = AsyncMock(return_value=_make_mock_response("ok"))
        cm = MagicMock()

        async def aenter(_self):
            return stream_obj

        async def aexit(_self, *a):
            return None

        cm.__aenter__ = aenter
        cm.__aexit__ = aexit
        stream_mock.return_value = cm
        client._client.beta.messages.stream = stream_mock

        await client.complete(
            role="event_router",
            messages=[{"role": "user", "content": "hi"}],
            compact=True,
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = stream_mock.call_args.kwargs
        assert call_kwargs["context_management"] == {
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {
                        "type": "input_tokens",
                        "value": client.config.compact_trigger_tokens,
                    },
                }
            ]
        }
        assert call_kwargs["betas"] == ["compact-2026-01-12"]

    @pytest.mark.asyncio
    async def test_max_tokens_truncation_raises_clear_error(self, client):
        """When the API hits max_tokens it returns 200 with a half-written
        body and stop_reason='max_tokens'. The client must raise an
        actionable ValueError BEFORE the structured-output parser tries
        to validate the truncated JSON, otherwise callers see an opaque
        pydantic 'EOF while parsing a string' error that gives no hint
        about the real cause. This was first caught on the old story
        importer truncation regression in v5/v6.

        This test exercises the RAW-content path (non-structured) where
        stop_reason is observable on the final message before any
        parser runs."""
        truncated = _make_mock_response('{"setting":{"genre":"Dark fantasy",')
        truncated.stop_reason = "max_tokens"
        truncated.usage.output_tokens = 64_000
        _install_stream_mock(client, truncated)

        with pytest.raises(ValueError, match="truncated at max_tokens=100"):
            await client.complete(
                role="narrator",
                messages=[{"role": "user", "content": "extract a huge world"}],
                temperature=0.3,
                max_tokens=100,
            )

    @pytest.mark.asyncio
    async def test_structured_output_truncation_wraps_validation_error(self, client):
        """On the STRUCTURED-OUTPUT path, the SDK's stream consumer parses
        text into the response_model inside accumulate_event() — a
        truncated JSON body raises pydantic ValidationError BEFORE
        stream.get_final_message() returns, so we never get to inspect
        stop_reason on the final message. The client must catch the
        pydantic 'Invalid JSON: EOF while parsing a string' shape and
        re-raise with an actionable max_tokens message instead of an
        opaque pydantic error that doesn't identify truncation as the cause.

        Reproduces the old hollowstone v6 importer failure shape (EOF at
        column 265,192 of the structured-output JSON body).
        """
        import pydantic as p

        truncated_text = '{"setting":{"genre":"Dark fantasy"' * 100
        validation_err = p.ValidationError.from_exception_data(
            "WorldSkeletonExtraction",
            [
                {
                    "type": "json_invalid",
                    "loc": (),
                    "input": truncated_text,
                    "ctx": {
                        "error": (
                            "EOF while parsing a string at line 1 column 265192"
                        )
                    },
                }
            ],
        )
        _install_stream_mock(client, validation_err)

        with pytest.raises(ValueError, match="Structured-output JSON truncated"):
            await client.complete(
                role="narrator",
                messages=[{"role": "user", "content": "extract a huge world"}],
                response_model=CanonicalEvent,
                temperature=0.3,
                max_tokens=100,
            )


class TestLLMClientRetry:
    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self, client):
        import anthropic as anth

        mock = _install_stream_mock(
            client,
            anth.APIConnectionError(request=MagicMock()),
            _make_mock_response("ok"),
        )

        result = await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
        )

        assert result.content == "ok"
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, client):
        """Once retries are exhausted the client wraps the underlying
        SDK exception in TransientLLMError so callers (Discord especially)
        can render a clean user-facing message. The original exception is
        preserved as `__cause__` and on `TransientLLMError.last_error`."""
        import anthropic as anth

        from app.llm.client import TransientLLMError

        mock = _install_stream_mock(
            client,
            anth.APIConnectionError(request=MagicMock()),
            anth.APIConnectionError(request=MagicMock()),
        )

        with pytest.raises(TransientLLMError) as excinfo:
            await client.complete(
                role="narrator",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.5,
                max_tokens=100,
            )

        # 1 initial + 1 retry = 2 attempts (max_retries=1)
        assert mock.call_count == 2
        assert excinfo.value.attempts == 2
        assert isinstance(excinfo.value.last_error, anth.APIConnectionError)
        assert isinstance(excinfo.value.__cause__, anth.APIConnectionError)

    @pytest.mark.asyncio
    async def test_retries_on_mid_stream_internal_server_error(self, client):
        """When Anthropic emits an SSE error event mid-stream, the SDK
        constructs the exception via _make_status_error(response=200)
        which falls through the status-code dispatch and surfaces as
        the BASE APIStatusError class — NOT InternalServerError, even
        though the body says 'Internal server error'. The retry path
        must catch this shape; otherwise a single transient kills a
        long-running structured-output batch.

        Reproduces the shape from req_011CaKVv9PMiXwBZ2Eoay6jd which
        hit the old importer Call 1 on a fresh v6 attempt."""
        import anthropic as anth

        sse_body = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Internal server error",
                "details": None,
            },
        }
        sse_error = anth.APIStatusError(
            f"{sse_body}", response=MagicMock(), body=sse_body,
        )
        mock = _install_stream_mock(
            client,
            sse_error,
            _make_mock_response("ok"),
        )

        result = await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
        )

        assert result.content == "ok"
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_non_transient_api_status_error(self, client):
        """An APIStatusError whose body indicates a real client-side
        problem (auth, malformed schema, etc) should NOT be retried —
        the request will keep failing identically and the retry loop
        just delays the inevitable. Only api_error / overloaded_error
        are transient enough to warrant a retry."""
        import anthropic as anth

        sse_body = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Some structural problem with the request",
            },
        }
        sse_error = anth.APIStatusError(
            f"{sse_body}", response=MagicMock(), body=sse_body,
        )
        mock = _install_stream_mock(client, sse_error)

        with pytest.raises(anth.APIStatusError):
            await client.complete(
                role="narrator",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.5,
                max_tokens=100,
            )
        assert mock.call_count == 1


# --- Integration tests (require live API) ---

@pytest.mark.integration
class TestLLMClientIntegration:
    @pytest.fixture
    def live_client(self):
        config = LLMConfig.from_env()
        if not config.api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        return LLMClient(config=config)

    @pytest.mark.asyncio
    async def test_basic_completion(self, live_client):
        result = await live_client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            temperature=0.5,
            max_tokens=50,
        )
        assert len(result.content) > 0
        assert result.usage["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_structured_output(self, live_client):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a world-state engine. Respond ONLY with valid JSON, no markdown.\n"
                    'Schema: {"event_id": string, "user_intent": string, '
                    '"world_adjudication": {"feasible": bool}, '
                    '"observable_facts": [string]}'
                ),
            },
            {"role": "user", "content": "The player tries to fly by flapping their arms."},
        ]

        result = await live_client.complete(
            role="narrator",
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            response_model=CanonicalEvent,
        )

        assert isinstance(result.parsed, CanonicalEvent)
        assert result.parsed.world_adjudication.feasible is False
