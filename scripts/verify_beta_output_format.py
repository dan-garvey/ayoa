"""Verify beta.messages.stream accepts output_format + context_management together.

Makes one small Haiku call to confirm the combination works end-to-end. Run once
before starting the rolling-conversation refactor.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python scripts/verify_beta_output_format.py
"""
from __future__ import annotations

import asyncio
import os
import sys

import anthropic
from pydantic import BaseModel


class Greeting(BaseModel):
    salutation: str
    mood: str


async def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    client = anthropic.AsyncAnthropic()

    print("Calling beta.messages.stream with both output_format and context_management…")
    try:
        async with client.beta.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=200,
            temperature=0.2,
            system="You respond with a cheerful greeting.",
            messages=[
                {"role": "user", "content": "Say hello."},
            ],
            output_format=Greeting,
            context_management={
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": 200_000},
                    }
                ]
            },
            betas=["compact-2026-01-12"],
        ) as stream:
            msg = await stream.get_final_message()
    except anthropic.BadRequestError as e:
        print(f"FAIL: API rejected the combination. {e}")
        return 2
    except Exception as e:
        print(f"FAIL: unexpected error: {type(e).__name__}: {e}")
        return 3

    print(f"  Model:      {msg.model}")
    print(f"  Stop:       {msg.stop_reason}")
    print(f"  Input toks: {msg.usage.input_tokens}")
    print(f"  Output toks:{msg.usage.output_tokens}")

    parsed = None
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            po = getattr(block, "parsed_output", None)
            if po is not None:
                parsed = po
                break

    if parsed is None:
        print("FAIL: no parsed_output on any text block.")
        print(f"  Raw content: {msg.content!r}")
        return 4

    if not isinstance(parsed, Greeting):
        print(f"FAIL: parsed_output is {type(parsed).__name__}, not Greeting.")
        return 5

    print(f"  Parsed:     salutation={parsed.salutation!r}  mood={parsed.mood!r}")
    print()
    print("PASS: beta.messages.stream + output_format + context_management work together.")
    print("The rolling-conversation refactor is unblocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
