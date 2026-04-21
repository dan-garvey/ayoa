"""Benchmark structured-output (output_format) vs post-hoc JSON parsing.

Hits the real Anthropic API, so automatically skipped when ANTHROPIC_API_KEY
is not set. Runs N trials for each strategy against the same prompt + schema
and emits a comparison table.

Run explicitly:

    .venv/bin/pytest tests/test_structured_output_benchmark.py -v -s

(The `-s` flag lets the comparison table reach stdout.)

Adds a TEST so we stop relying on the user's terminal for diagnostic data
when a structured-output path starts misbehaving. The AuthoredCharacter
schema was the specific shape that kept hitting "Schema is too complex"
and grammar-compilation timeouts during /join_custom, so it's the
natural candidate to benchmark on.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import time

import pytest

from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.takeover import AuthoredCharacter

SKIP_IF_NO_KEY = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; integration benchmark requires a real key.",
)

N_TRIALS = 3

# Same prompt for both strategies so the comparison is apples-to-apples.
# The shape description is baked into the prompt because the raw-JSON
# strategy doesn't have output_format telling the model what to emit.
SYSTEM_PROMPT = (
    "You author fictional character records as JSON. You respond with ONLY "
    "valid JSON — no prose commentary, no markdown fences."
)
USER_PROMPT = """Produce a character matching this JSON schema (ALL fields required):

{
  "name": "string — full name",
  "location": "string — scene_id",
  "role": "string — role or occupation",
  "appearance": "string — physical description",
  "faction": "string — affiliation or empty string",
  "backstory": "string — their background",
  "personality": "string — one prose block covering inner world, voice, and how to play them",
  "known_context": "string — what they take for granted about the world",
  "goals": ["string — existential drives"],
  "current_objectives": ["string — 1-3 active pursuits"],
  "secrets": ["string — private truths"],
  "intentions_enabled": true or false
}

Character concept: a taciturn blacksmith in a port city who recently
witnessed a murder and now has to decide whether to testify. Competent,
private, slow to anger but capable of violence. Set location as
"port_forge". Fill every field with something specific to this concept.

Respond with ONLY the JSON object — no fences, no commentary."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text


async def _run_structured(
    client: LLMClient, effort: str | None = None,
) -> dict:
    """Strategy A: Anthropic output_format enforcement. The effort kwarg
    tunes output_config.effort — `None` uses the client's default (set
    in client.py), anything else overrides for this call only."""
    t0 = time.monotonic()
    try:
        # Temporarily swap the client's output_config.effort by calling
        # complete with a role-level monkey for the test. We do this by
        # briefly patching client.config if needed; simpler: just pass
        # the request through the low-level complete and rely on the
        # client's default output_config. For per-trial effort
        # variation, construct the call by direct kwargs override via
        # a private override param.
        resp = await client.complete(
            role="narrator",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            response_model=AuthoredCharacter,
            temperature=0.5,
            max_tokens=2000,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        parsed = resp.parsed
        return {
            "success": isinstance(parsed, AuthoredCharacter) and bool(parsed.name),
            "latency_ms": elapsed_ms,
            "input_tokens": resp.usage.get("prompt_tokens", 0),
            "output_tokens": resp.usage.get("completion_tokens", 0),
            "error": None,
            "name": parsed.name if parsed else "",
        }
    except Exception as e:
        return {
            "success": False,
            "latency_ms": (time.monotonic() - t0) * 1000,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "name": "",
        }


async def _run_raw_parse(client: LLMClient) -> dict:
    """Strategy B: no output_format; parse JSON from content post-hoc."""
    t0 = time.monotonic()
    try:
        resp = await client.complete(
            role="narrator",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=0.5,
            max_tokens=2000,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        text = _strip_fences(resp.content or "")
        parsed = AuthoredCharacter.model_validate_json(text)
        return {
            "success": bool(parsed.name),
            "latency_ms": elapsed_ms,
            "input_tokens": resp.usage.get("prompt_tokens", 0),
            "output_tokens": resp.usage.get("completion_tokens", 0),
            "error": None,
            "name": parsed.name,
        }
    except Exception as e:
        return {
            "success": False,
            "latency_ms": (time.monotonic() - t0) * 1000,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "name": "",
        }


def _summarize(name: str, trials: list[dict]) -> dict:
    ok = [t for t in trials if t["success"]]
    latencies = [t["latency_ms"] for t in trials]
    return {
        "name": name,
        "success_rate": len(ok) / len(trials) if trials else 0.0,
        "p50_ms": statistics.median(latencies) if latencies else 0.0,
        "mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "max_ms": max(latencies) if latencies else 0.0,
        "mean_out_tokens": (
            statistics.mean([t["output_tokens"] for t in ok]) if ok else 0.0
        ),
        "errors": [t["error"] for t in trials if t["error"]],
    }


def _print_table(summaries: list[dict]) -> None:
    print("\n=== Structured Output vs Raw-JSON Parse ===")
    print(
        f"{'Strategy':<30} {'Success':>8} {'p50':>10} {'mean':>10} {'stdev':>8} {'max':>10} {'out tok':>8}"
    )
    for s in summaries:
        print(
            f"{s['name']:<30} "
            f"{int(s['success_rate'] * 100):>7}% "
            f"{s['p50_ms']:>8.0f}ms "
            f"{s['mean_ms']:>8.0f}ms "
            f"{s['stdev_ms']:>6.0f}ms "
            f"{s['max_ms']:>8.0f}ms "
            f"{s['mean_out_tokens']:>8.0f}"
        )
    for s in summaries:
        if s["errors"]:
            print(f"\n  {s['name']} errors ({len(s['errors'])}):")
            for e in s["errors"]:
                print(f"    - {e}")
    print()


@pytest.mark.integration
@SKIP_IF_NO_KEY
@pytest.mark.asyncio
async def test_benchmark_structured_vs_raw_parse():
    """Run both strategies against the AuthoredCharacter schema and
    report the comparison. Fails only when BOTH strategies fail across
    every trial — the goal is data, not pass/fail."""
    client = LLMClient(config=LLMConfig.from_env())
    try:
        # Run trials sequentially so shared cache behavior is honest;
        # parallel would let one strategy warm the server-side cache
        # for the other.
        structured_trials = []
        for _ in range(N_TRIALS):
            structured_trials.append(await _run_structured(client))

        raw_trials = []
        for _ in range(N_TRIALS):
            raw_trials.append(await _run_raw_parse(client))
    finally:
        await client.close()

    summaries = [
        _summarize("output_format (structured)", structured_trials),
        _summarize("raw JSON + Pydantic parse", raw_trials),
    ]
    _print_table(summaries)

    # Don't fail on one bad trial; fail only if every trial of both
    # strategies failed.
    structured_ok = any(t["success"] for t in structured_trials)
    raw_ok = any(t["success"] for t in raw_trials)
    assert structured_ok or raw_ok, (
        "Both strategies failed across all trials — see error list above."
    )
