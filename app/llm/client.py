from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
import openai
from pydantic import BaseModel, ValidationError

from app.llm.config import LLMConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    """Result from an LLM call."""
    content: str = ""
    parsed: Any = None
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None


def extract_json(text: str) -> str:
    """Extract JSON from LLM response text, handling markdown fences and surrounding prose."""
    # Try markdown fenced block first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find a JSON object directly
    # Find the first { and last } to extract the outermost object
    first_brace = text.find("{")
    if first_brace == -1:
        return text.strip()

    depth = 0
    last_brace = -1
    for i in range(first_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                last_brace = i
                break

    if last_brace != -1:
        return text[first_brace : last_brace + 1]

    return text.strip()


class LLMClient:
    """Async LLM client wrapping the OpenAI-compatible gateway."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self._client = openai.AsyncOpenAI(
            base_url=self.config.gateway_url,
            api_key=self.config.api_key,
            http_client=httpx.AsyncClient(verify=False),
            default_headers={
                "Ocp-Apim-Subscription-Key": self.config.subscription_key,
                "user": self.config.user,
            },
        )

    async def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        response_model: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """Call the LLM gateway and optionally parse the response into a Pydantic model.

        Args:
            role: The engine role making this call (narrator, discriminator, agent, etc.).
                  Used to select the model via config.
            messages: Chat messages in OpenAI format.
            response_model: If provided, extract JSON from the response and validate
                            against this Pydantic model. Retries once on parse failure.
            temperature: Sampling temperature. Defaults to config value.
            max_tokens: Max completion tokens. Defaults to config value.
            stream: Whether to stream the response (not yet implemented for structured output).

        Returns:
            LLMResponse with content and optionally parsed model.
        """
        model_name = self.config.model_for_role(role)
        temp = temperature if temperature is not None else self.config.default_temperature
        max_tok = max_tokens if max_tokens is not None else self.config.default_max_tokens

        raw_response = await self._call_with_retry(
            model=model_name,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            stream=stream,
        )

        msg = raw_response.choices[0].message
        content = msg.content or ""
        # Some models put output in reasoning field when content is empty
        if not content and hasattr(msg, "reasoning") and msg.reasoning:
            content = msg.reasoning
        usage = {}
        if raw_response.usage:
            usage = {
                "prompt_tokens": raw_response.usage.prompt_tokens,
                "completion_tokens": raw_response.usage.completion_tokens,
                "total_tokens": raw_response.usage.total_tokens,
            }

        result = LLMResponse(
            content=content,
            model=raw_response.model or model_name,
            usage=usage,
            raw_response=raw_response,
        )

        if response_model is not None:
            result.parsed = await self._parse_structured(
                content=content,
                response_model=response_model,
                messages=messages,
                model=model_name,
                temperature=temp,
                max_tokens=max_tok,
            )

        return result

    async def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> Any:
        """Call the gateway with exponential backoff retry on transient failures."""
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    stream=stream,
                )
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
        raise last_error

    async def _parse_structured(
        self,
        content: str,
        response_model: type[T],
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> T:
        """Extract JSON from content and validate against response_model.

        On parse failure, retries once with a repair prompt that includes
        the raw output and the validation error.
        """
        json_str = extract_json(content)
        parse_error = None

        try:
            data = json.loads(json_str)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            parse_error = e
            logger.warning(f"First parse attempt failed: {parse_error}")

        # Repair attempt: ask the LLM to fix its output
        repair_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    f"Your previous response could not be parsed. Error:\n{parse_error}\n\n"
                    f"Please respond with ONLY valid JSON matching the required schema. "
                    f"No markdown fences, no explanation, just the JSON object."
                ),
            },
        ]

        try:
            repair_response = await self._call_with_retry(
                model=model,
                messages=repair_messages,
                temperature=max(temperature * 0.5, 0.1),
                max_tokens=max_tokens,
                stream=False,
            )
            repair_content = repair_response.choices[0].message.content or ""
            repair_json = extract_json(repair_content)
            data = json.loads(repair_json)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError, Exception) as repair_error:
            raise ValueError(
                f"Failed to parse LLM output into {response_model.__name__} "
                f"after repair attempt. Original error: {parse_error}. "
                f"Repair error: {repair_error}"
            ) from repair_error

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.close()
