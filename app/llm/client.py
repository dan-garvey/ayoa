from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

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
    """Extract a JSON object or array from free-form text.

    Handles markdown fenced code blocks and surrounding prose. Used by callers
    that want raw dict/list output without going through a Pydantic model
    (e.g., the story importer).
    """
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace == -1 and first_bracket == -1:
        return text.strip()

    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        open_ch, close_ch = "[", "]"
        start = first_bracket
    else:
        open_ch, close_ch = "{", "}"
        start = first_brace

    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                end = i
                break

    if end != -1:
        return text[start : end + 1]

    return text.strip()


def _split_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    """Anthropic takes `system` as a top-level param, not a message. Peel it off."""
    system_parts: list[str] = []
    rest: list[dict[str, str]] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            rest.append(m)
    system = "\n\n".join(p for p in system_parts if p) or None
    return system, rest


def _extract_text(response: Any) -> str:
    """Concatenate all text blocks in a response."""
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )


def serialize_assistant_content(raw_content: list) -> list[dict]:
    """Serialize the raw assistant content blocks for persistence in a rolling conversation.

    Strips fields the API emits on output but rejects on input (parsed_output from
    ParsedTextBlock when output_format is used, and citations which are output-only).
    Preserves compaction blocks verbatim — the API requires them back on subsequent
    requests.
    """
    serialized: list[dict] = []
    for block in raw_content:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            # Only `type` and `text` are valid on input TextBlockParam. Read
            # text directly instead of model_dump(), which triggers a pydantic
            # serialization warning because the SDK's ParsedTextBlock declares
            # parsed_output as None-typed but attaches a real Pydantic model
            # when output_format is used.
            serialized.append({"type": "text", "text": getattr(block, "text", "") or ""})
        else:
            # For compaction and other block types, pass through unchanged —
            # but still remove parsed_output/citations defensively if present.
            if hasattr(block, "model_dump"):
                dumped = block.model_dump(exclude={"parsed_output", "citations"})
            else:
                dumped = dict(block)
                dumped.pop("parsed_output", None)
                dumped.pop("citations", None)
            serialized.append(dumped)
    return serialized


def _extract_parsed(response: Any) -> Any:
    """Pull the first `parsed_output` attached to a text block, if present.

    The Anthropic SDK sets `parsed_output` on text blocks when `output_format` was
    supplied on the request (schema-enforced structured output).
    """
    for block in response.content:
        if getattr(block, "type", "") == "text":
            parsed = getattr(block, "parsed_output", None)
            if parsed is not None:
                return parsed
    return None


class LLMClient:
    """Async LLM client wrapping the Anthropic Messages API."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        kwargs: dict[str, Any] = {"timeout": self.config.timeout}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_model: type[T] | None = None,
        cache: bool = True,
        compact: bool = False,
        stream: bool = False,
    ) -> LLMResponse:
        """Call Claude and optionally enforce a Pydantic schema on the response.

        Args:
            role: The engine role making this call (narrator, discriminator, agent, etc.).
                  Used to select the model via config.
            messages: Chat messages in OpenAI-style format. A message with role="system"
                      is peeled off and sent as Anthropic's top-level `system` param.
            response_model: If provided, the API is asked to emit JSON matching this
                            Pydantic model's schema. The SDK parses and validates the
                            response — no client-side repair loop is needed.
            temperature: Sampling temperature. Required — each call site picks
                         a task-appropriate value. Not supported on Opus 4.7
                         — switch the role's model if you need sampling control.
            max_tokens: Max output tokens. Required — pick per task.
            cache: If True and a `system` message is present, place an ephemeral cache
                   breakpoint at the end of the system block so calls that share the
                   same system (but differ in the user tail) hit the same cache entry.
                   No-op when there is no system message (nothing shared to cache).
            compact: If True, enable server-side context compaction (beta). The API
                     automatically summarizes earlier context when input tokens cross
                     `config.compact_trigger_tokens`. Callers that send rolling
                     conversation history are the ones that benefit; single-request
                     calls will never approach the threshold and pay nothing extra.
            stream: Reserved; streaming is used internally to avoid HTTP timeouts.

        Returns:
            LLMResponse with content and optionally parsed model.
        """
        del stream
        model_name = self.config.model_for_role(role)
        temp = temperature
        max_tok = max_tokens
        system, conversation = _split_system(messages)

        raw_response = await self._call_with_retry(
            model=model_name,
            system=system,
            messages=conversation,
            temperature=temp,
            max_tokens=max_tok,
            cache=cache,
            compact=compact,
            response_model=response_model,
        )

        content = _extract_text(raw_response)
        usage: dict[str, int] = {}
        if raw_response.usage:
            # input_tokens is the *uncached* remainder. Cache write/read columns
            # come in separately and are billed at 1.25× / 0.1× respectively.
            prompt_tokens = raw_response.usage.input_tokens
            completion_tokens = raw_response.usage.output_tokens
            cache_read = getattr(
                raw_response.usage, "cache_read_input_tokens", None
            ) or 0
            cache_write = getattr(
                raw_response.usage, "cache_creation_input_tokens", None
            ) or 0
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
                # Full prompt size = uncached + cache read + cache write.
                "full_input_tokens": prompt_tokens + cache_read + cache_write,
            }

        parsed = _extract_parsed(raw_response) if response_model is not None else None
        if response_model is not None and parsed is None:
            raise ValueError(
                f"Model returned no parsed output for {response_model.__name__}. "
                f"Raw text: {content[:500]}"
            )

        return LLMResponse(
            content=content,
            parsed=parsed,
            model=raw_response.model or model_name,
            usage=usage,
            raw_response=raw_response,
        )

    async def _call_with_retry(
        self,
        model: str,
        system: str | None,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        cache: bool,
        compact: bool,
        response_model: type[T] | None,
    ) -> Any:
        """Call the Messages API via streaming to avoid HTTP timeouts on long outputs.

        Retries transient failures with exponential backoff.
        """
        last_error: Exception | None = None
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        cache_control: dict[str, Any] = {"type": "ephemeral"}
        # Anthropic accepts ttl on the cache_control block. "5m" is the
        # default (implicit); "1h" extends to one hour at 2x write cost.
        # We explicitly set it so the TTL is visible in the request and
        # driven by config rather than relying on server-side defaults.
        if self.config.cache_ttl in ("5m", "1h"):
            cache_control = {"type": "ephemeral", "ttl": self.config.cache_ttl}

        if system:
            if cache:
                # Breakpoint at the end of system (the shared prefix). Top-level
                # cache_control would place it at the end of the user message,
                # which differs per call and would never read.
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": cache_control,
                    }
                ]
            else:
                kwargs["system"] = system

        # Rolling-conversation caching: when history exists (messages > 1 pair),
        # also mark the last user message as a cache breakpoint so subsequent
        # turns read the whole prior conversation at cache-hit price.
        if cache and len(messages) > 1:
            last_idx = len(messages) - 1
            last = messages[last_idx]
            if last.get("role") == "user" and isinstance(last.get("content"), str):
                messages = list(messages)
                messages[last_idx] = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": last["content"],
                            "cache_control": cache_control,
                        }
                    ],
                }
                kwargs["messages"] = messages

        if response_model is not None:
            kwargs["output_format"] = response_model

        if compact:
            kwargs["context_management"] = {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {
                            "type": "input_tokens",
                            "value": self.config.compact_trigger_tokens,
                        },
                    }
                ]
            }
            kwargs["betas"] = ["compact-2026-01-12"]
            stream_ctx = self._client.beta.messages.stream
        else:
            stream_ctx = self._client.messages.stream

        for attempt in range(self.config.max_retries + 1):
            try:
                async with stream_ctx(**kwargs) as stream:
                    response = await stream.get_final_message()
                if not response.content:
                    raise anthropic.InternalServerError(
                        message="Claude returned empty content",
                        response=None,
                        body=None,
                    )
                return response
            except (
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
                anthropic.RateLimitError,
            ) as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
            except anthropic.BadRequestError as e:
                # Anthropic's server-side grammar compilation for
                # structured output occasionally times out as a 400 — it
                # is NOT a schema bug, just a transient on their
                # compilation side, and retries succeed. Retry only this
                # specific message; other 400s (schema errors, invalid
                # params) should surface immediately.
                msg = str(e) or ""
                if "Grammar compilation timed out" not in msg:
                    raise
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Grammar compilation timeout (attempt %d), retrying in %.1fs",
                        attempt + 1, delay,
                    )
                    await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.close()
