from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anthropic
import pydantic
from pydantic import BaseModel

from app.llm.config import LLMConfig

logger = logging.getLogger(__name__)


class TransientLLMError(RuntimeError):
    """Surfaced when every retry attempt for a transient API failure has
    been exhausted (overloaded_error, 5xx, network blip, grammar
    compilation timeout, etc.). Carries the underlying exception as
    `__cause__` so callers can inspect details, and a clean
    user-facing message Discord can render directly without leaking
    SDK internals.

    Permanent failures (BadRequestError that isn't grammar-comp,
    AuthenticationError, schema bugs) bypass this wrapper and raise
    their original type — those are programmer errors, not retryable
    user-visible blips.
    """

    def __init__(self, message: str, attempts: int, last_error: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error

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

    Drops `thinking` blocks. Anthropic only requires thinking blocks
    to round-trip when extended thinking is paired with tool use
    (the signature is needed to reconstruct the model's reasoning
    across tool-result turns). Plain agent conversations don't have
    that constraint, and passing thinking blocks back as part of
    the next turn's history would (a) re-bill us for the already-paid
    thinking tokens, (b) bloat the rolling conversation that the
    cache lineage depends on, and (c) potentially confuse a future
    model into treating its old reasoning as fresh.
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
        elif block_type == "thinking":
            continue
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
        cache_user_tail: bool = False,
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
            cache_user_tail: If True, force a cache breakpoint on the last user
                   message even when there's only one user turn (normally breakpoint
                   on user tail is only added when len(messages) > 1). Use when a
                   caller expects a follow-up call to read
                   [system, user1, assistant1] as a cached prefix — e.g. the
                   two-call importer pattern.
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
        thinking_budget = self.config.thinking_budget_for_role(role)
        # Extended thinking has two hard API constraints we backstop
        # here so call sites don't need to know about them:
        #   1. budget_tokens must be < max_tokens. We bump max_tokens
        #      up to budget + 1024 if the caller's max is too tight,
        #      so there's always room for the visible response on
        #      top of the thinking budget.
        #   2. temperature must be 1.0. We override the caller's
        #      temperature when thinking is on.
        # When thinking_budget==0 (no thinking enabled for this role),
        # neither override fires and behavior matches pre-r9b exactly.
        if thinking_budget > 0:
            if temp != 1.0:
                logger.debug(
                    "Extended thinking enabled for role=%s; overriding "
                    "caller temperature %.2f to 1.0 (API requirement)",
                    role, temp,
                )
                temp = 1.0
            min_max = thinking_budget + 1024
            if max_tok < min_max:
                logger.debug(
                    "Extended thinking enabled for role=%s; bumping "
                    "max_tokens %d -> %d to leave room above the "
                    "%d-token thinking budget",
                    role, max_tok, min_max, thinking_budget,
                )
                max_tok = min_max
        system, conversation = _split_system(messages)

        raw_response = await self._call_with_retry(
            model=model_name,
            system=system,
            messages=conversation,
            temperature=temp,
            max_tokens=max_tok,
            cache=cache,
            cache_user_tail=cache_user_tail,
            compact=compact,
            response_model=response_model,
            thinking_budget=thinking_budget,
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
        cache_user_tail: bool,
        compact: bool,
        response_model: type[T] | None,
        thinking_budget: int = 0,
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
        # Extended thinking. The temperature/max_tokens overrides
        # required by the API are applied upstream in `complete()` —
        # see the `thinking_budget > 0` block there. Here we just
        # add the parameter to the request body. Display defaults
        # to "summarized" on Haiku 4.5; we don't pass `display`
        # because we never surface thinking blocks to the user
        # (`_extract_text` filters by `type=="text"`, and
        # `serialize_assistant_content` strips thinking from the
        # rolling history).
        if thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
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
        # turns read the whole prior conversation at cache-hit price. The
        # `cache_user_tail` override forces the same breakpoint on single-turn
        # calls when a follow-up is expected to replay this call as history.
        should_mark_user = cache and (len(messages) > 1 or cache_user_tail)
        if should_mark_user and messages:
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

        # Route through the beta client whenever we're using a beta
        # feature (compaction, structured output). Anthropic's
        # output_format grammar compiler lives on the beta path; the
        # non-beta messages.stream rejects schemas it treats as "too
        # complex" which the beta path compiles without issue. So beta
        # routing is gated on *either* compact OR a response_model.
        # Server-side compaction is currently only implemented on Sonnet-class
        # models. Haiku 4.5 rejects the beta with a 400 ("does not support the
        # 'compact_20260112' context management strategy"). Silently drop the
        # feature for unsupported models — the rolling-conversation trimming
        # we already do keeps inputs far below the 200K/1M ceiling, so the
        # loss is purely an optimization miss.
        compact_supported = "sonnet" in model.lower() or "opus" in model.lower()
        effective_compact = compact and compact_supported
        needs_beta = effective_compact or response_model is not None
        if effective_compact:
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
        if needs_beta:
            stream_ctx = self._client.beta.messages.stream
        else:
            stream_ctx = self._client.messages.stream

        def _backoff_sleep_seconds(attempt: int) -> float:
            """Exponential backoff with symmetric jitter and a hard cap.
            attempt is 0-indexed; the wait BEFORE attempt N+1 is
            base * 2^N, jittered ±retry_jitter, then capped."""
            delay = self.config.retry_base_delay * (2 ** attempt)
            jitter = self.config.retry_jitter
            if jitter > 0:
                delay *= 1.0 + random.uniform(-jitter, jitter)
            return min(delay, self.config.retry_max_delay)

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
                # Detect mid-output truncation on the non-structured path.
                # When the API hits max_tokens it still returns 200 with
                # whatever it had so far. For raw responses we can read
                # stop_reason on the final message and raise a clear error.
                # NOTE: this branch does NOT trigger on structured-output
                # calls — the SDK parses content_block.text into
                # response_model INSIDE accumulate_event() during stream
                # consumption, so a truncated JSON body raises pydantic's
                # ValidationError before stream.get_final_message() ever
                # returns. The structured-output truncation case is
                # handled in the ValidationError except branch below.
                if getattr(response, "stop_reason", None) == "max_tokens":
                    out_tokens = getattr(response.usage, "output_tokens", "?") if response.usage else "?"
                    raise ValueError(
                        f"Output truncated at max_tokens={kwargs['max_tokens']} "
                        f"({out_tokens} tokens emitted, stop_reason='max_tokens'). "
                        f"Increase max_tokens or split the request into smaller calls."
                    )
                return response
            except pydantic.ValidationError as e:
                # The SDK's structured-output streaming parser
                # (anthropic.lib.streaming.accumulate_event) calls
                # `output_format.validate_json(content_block.text)`
                # immediately after the content_block_stop SSE event.
                # When max_tokens truncated mid-string, the resulting
                # error is "Invalid JSON: EOF while parsing a string at
                # line 1 column N" which gives no hint that the cause
                # is output truncation rather than a schema bug.
                # Re-raise with an actionable message in that specific
                # case. Real schema mismatches (model emitted wrong
                # field types, missing required field, etc.) bubble up
                # unchanged.
                msg = str(e)
                if "Invalid JSON" in msg and "EOF" in msg:
                    raise ValueError(
                        f"Structured-output JSON truncated at "
                        f"max_tokens={kwargs['max_tokens']} (parser hit "
                        f"EOF mid-stream). Increase max_tokens or split "
                        f"the request into smaller calls. Underlying "
                        f"pydantic error: {msg.splitlines()[0]}"
                    ) from e
                raise
            except (
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
                anthropic.RateLimitError,
            ) as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = _backoff_sleep_seconds(attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.config.max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)
            except anthropic.APIStatusError as e:
                # Mid-stream server errors arrive here with the BASE
                # APIStatusError class, not InternalServerError, because
                # the HTTP response already returned 200 before the
                # stream started — `_make_status_error(response=200)`
                # falls through to the generic APIStatusError branch
                # (anthropic/_client.py status-code dispatch only fires
                # on the actual HTTP status, not on the SSE error body).
                # The body itself still says
                # `{'type': 'error', 'error': {'type': 'api_error',
                # 'message': 'Internal server error'}}` — definitionally
                # a transient. Retry the same shapes a real 5xx would.
                body = getattr(e, "body", None) or {}
                err = body.get("error", {}) if isinstance(body, dict) else {}
                err_type = err.get("type", "") if isinstance(err, dict) else ""
                msg = (err.get("message", "") if isinstance(err, dict) else "") or str(e)
                transient = err_type in {"api_error", "overloaded_error"} or (
                    "internal server error" in msg.lower()
                    or "overloaded" in msg.lower()
                )
                if not transient:
                    raise
                last_error = e
                if attempt < self.config.max_retries:
                    delay = _backoff_sleep_seconds(attempt)
                    logger.warning(
                        "Mid-stream API error (attempt %d/%d, type=%r), retrying in %.1fs: %s",
                        attempt + 1, self.config.max_retries + 1, err_type,
                        delay, msg,
                    )
                    await asyncio.sleep(delay)
            except anthropic.BadRequestError as e:
                # Anthropic's server-side grammar compilation for
                # structured output occasionally fails as a 400 — it
                # is NOT necessarily a schema bug, sometimes just a
                # transient on their compilation side that retries
                # past. Retry these specific messages only; other 400s
                # (schema errors, invalid params) should surface
                # immediately.
                #
                # Known retryable variants:
                #   - "Grammar compilation timed out" — straightforward
                #     transient.
                #   - "Schema is too complex" — when the schema is
                #     borderline (around the AuthoredCharacter ceiling
                #     or EventRouterOutput's many list fields), the
                #     compiler sometimes fails non-deterministically.
                #     Retry here is mostly free; the structural fix
                #     (strip Pydantic defaults to collapse the grammar)
                #     is the real cure when this is consistent.
                msg = str(e) or ""
                retryable = (
                    "Grammar compilation timed out" in msg
                    or "Schema is too complex" in msg
                )
                if not retryable:
                    raise
                last_error = e
                if attempt < self.config.max_retries:
                    delay = _backoff_sleep_seconds(attempt)
                    logger.warning(
                        "Grammar compilation 400 (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.config.max_retries + 1, delay,
                        msg.splitlines()[0],
                    )
                    await asyncio.sleep(delay)
        assert last_error is not None
        attempts = self.config.max_retries + 1
        # Friendly message: classify the exhausted retry by the last
        # error so callers (Discord especially) can show the user a
        # one-line "the model is busy, try again" rather than the
        # raw SDK exception. The actual exception is preserved as
        # __cause__ for log inspection.
        is_overload = isinstance(last_error, anthropic.APIStatusError) and (
            "overloaded" in str(last_error).lower()
        )
        is_rate = isinstance(last_error, anthropic.RateLimitError)
        if is_overload:
            msg = (
                f"Anthropic is overloaded right now (retried {attempts} times "
                f"with backoff). Please try your action again in a few seconds."
            )
        elif is_rate:
            msg = (
                f"Anthropic rate limit hit (retried {attempts} times). Wait "
                f"a moment before sending another action."
            )
        else:
            msg = (
                f"Anthropic call failed after {attempts} attempts. The "
                f"connection or model is unstable; try again in a moment."
            )
        raise TransientLLMError(msg, attempts, last_error) from last_error

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.close()
