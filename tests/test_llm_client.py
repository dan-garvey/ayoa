"""Tests for the LLM client — unit tests with mocks and integration tests against live gateway."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.client import LLMClient, LLMResponse, extract_json
from app.llm.config import LLMConfig
from app.schemas.events import CanonicalEvent


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
        with patch.dict("os.environ", {"LLM_GATEWAY_KEY": "test-key", "USER": "testuser"}):
            config = LLMConfig.from_env()
            assert config.subscription_key == "test-key"
            assert config.user == "testuser"


# --- LLMClient unit tests (mocked gateway) ---

def _make_mock_response(content: str, model: str = "test-model"):
    """Create a mock OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = content

    choice = MagicMock()
    choice.message = msg

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    usage.total_tokens = 30

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    return response


@pytest.fixture
def mock_config():
    return LLMConfig(
        gateway_url="http://fake",
        subscription_key="fake-key",
        user="testuser",
        max_retries=1,
        retry_base_delay=0.01,
    )


@pytest.fixture
def client(mock_config):
    return LLMClient(config=mock_config)


class TestLLMClientComplete:
    @pytest.mark.asyncio
    async def test_basic_completion(self, client):
        mock_resp = _make_mock_response("Hello world")
        client._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello world"
        assert result.model == "test-model"
        assert result.usage["total_tokens"] == 30
        assert result.parsed is None

    @pytest.mark.asyncio
    async def test_structured_output_success(self, client):
        valid_json = json.dumps({
            "event_id": "evt_001",
            "user_intent": "open the door",
            "world_adjudication": {
                "attempted_action": "user opens door",
                "feasible": True,
                "resolved_outcome": "door opens",
            },
            "scene_delta": {"time_advanced_seconds": 2},
            "observable_facts": ["The door swings open."],
        })
        mock_resp = _make_mock_response(valid_json)
        client._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "open door"}],
            response_model=CanonicalEvent,
        )

        assert isinstance(result.parsed, CanonicalEvent)
        assert result.parsed.user_intent == "open the door"
        assert result.parsed.world_adjudication.feasible is True

    @pytest.mark.asyncio
    async def test_structured_output_with_markdown_fences(self, client):
        fenced = '```json\n{"event_id":"e1","user_intent":"look","world_adjudication":{"attempted_action":"look around","feasible":true,"resolved_outcome":"sees room"},"scene_delta":{"time_advanced_seconds":1},"observable_facts":["A dim room."]}\n```'
        mock_resp = _make_mock_response(fenced)
        client._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "look"}],
            response_model=CanonicalEvent,
        )

        assert isinstance(result.parsed, CanonicalEvent)
        assert result.parsed.user_intent == "look"

    @pytest.mark.asyncio
    async def test_structured_output_repair_on_bad_json(self, client):
        bad_resp = _make_mock_response("not json at all {broken")
        good_json = json.dumps({
            "event_id": "evt_r",
            "user_intent": "wait",
            "world_adjudication": {
                "attempted_action": "wait",
                "feasible": True,
                "resolved_outcome": "time passes",
            },
            "scene_delta": {"time_advanced_seconds": 5},
            "observable_facts": ["Nothing happens."],
        })
        good_resp = _make_mock_response(good_json)

        client._client.chat.completions.create = AsyncMock(side_effect=[bad_resp, good_resp])

        result = await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "wait"}],
            response_model=CanonicalEvent,
        )

        assert isinstance(result.parsed, CanonicalEvent)
        assert result.parsed.user_intent == "wait"
        # Two calls: original + repair
        assert client._client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_structured_output_fails_after_repair(self, client):
        bad_resp = _make_mock_response("garbage")
        still_bad_resp = _make_mock_response("still garbage")

        client._client.chat.completions.create = AsyncMock(side_effect=[bad_resp, still_bad_resp])

        with pytest.raises(ValueError, match="Failed to parse"):
            await client.complete(
                role="narrator",
                messages=[{"role": "user", "content": "x"}],
                response_model=CanonicalEvent,
            )


class TestLLMClientRetry:
    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self, client):
        import openai as oai

        mock_resp = _make_mock_response("ok")
        client._client.chat.completions.create = AsyncMock(
            side_effect=[oai.APIConnectionError(request=MagicMock()), mock_resp]
        )

        result = await client.complete(
            role="narrator",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result.content == "ok"
        assert client._client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, client):
        import openai as oai

        client._client.chat.completions.create = AsyncMock(
            side_effect=oai.APIConnectionError(request=MagicMock())
        )

        with pytest.raises(oai.APIConnectionError):
            await client.complete(
                role="narrator",
                messages=[{"role": "user", "content": "hi"}],
            )

        # 1 initial + 1 retry = 2 attempts (max_retries=1)
        assert client._client.chat.completions.create.call_count == 2


# --- Integration tests (require live gateway) ---

@pytest.mark.integration
class TestLLMClientIntegration:
    @pytest.fixture
    def live_client(self):
        config = LLMConfig.from_env()
        if not config.subscription_key:
            pytest.skip("LLM_GATEWAY_KEY not set")
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
            temperature=0.3,
        )

        assert isinstance(result.parsed, CanonicalEvent)
        assert result.parsed.world_adjudication.feasible is False
