"""Tests for the LLM client — unit tests with mocks and integration tests against live API."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.client import LLMClient, LLMResponse, extract_json
from app.llm.config import LLMConfig
from app.schemas.events import CanonicalEvent


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
    def test_model_for_role(self):
        config = LLMConfig(role_models={"narrator": "big-model", "agent": "small-model"})
        assert config.model_for_role("narrator") == "big-model"
        assert config.model_for_role("agent") == "small-model"
        assert config.model_for_role("unknown") == config.default_model

    def test_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            config = LLMConfig.from_env()
            assert config.api_key == "test-key"


# --- LLMClient unit tests (mocked API) ---

def _make_mock_response(content: str, model: str = "claude-haiku-4-5", parsed=None):
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

    response = MagicMock()
    response.content = [text_block]
    response.model = model
    response.usage = usage
    response.stop_reason = "end_turn"
    return response


@pytest.fixture
def mock_config():
    return LLMConfig(
        api_key="fake-key",
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
                "attempted_action": "user opens door",
                "feasible": True,
                "resolved_outcome": "door opens",
            },
            "scene_delta": {"time_advanced_seconds": 2},
            "observable_facts": ["The door swings open."],
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
        assert result.parsed.world_adjudication.attempted_action == "user opens door"

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
    async def test_compact_true_adds_context_management(self, client):
        """compact=True switches to the beta stream with compaction config + beta header."""
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
            role="narrator",
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
        import anthropic as anth

        mock = _install_stream_mock(
            client,
            anth.APIConnectionError(request=MagicMock()),
            anth.APIConnectionError(request=MagicMock()),
        )

        with pytest.raises(anth.APIConnectionError):
            await client.complete(
                role="narrator",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.5,
                max_tokens=100,
            )

        # 1 initial + 1 retry = 2 attempts (max_retries=1)
        assert mock.call_count == 2


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
                    '"world_adjudication": {"attempted_action": string, "feasible": bool, "resolved_outcome": string}, '
                    '"scene_delta": {"time_advanced_seconds": int}, '
                    '"observable_facts": [string]}'
                ),
            },
            {"role": "user", "content": "The player tries to fly by flapping their arms."},
        ]

        result = await live_client.complete(
            role="narrator",
            messages=messages,
            response_model=CanonicalEvent,
        )

        assert isinstance(result.parsed, CanonicalEvent)
        assert result.parsed.world_adjudication.feasible is False
