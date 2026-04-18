"""Test script for the Anthropic Messages API."""

import argparse
import json
import os
import time

import anthropic


def make_client() -> anthropic.Anthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic()


def _extract_text(response) -> str:
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )


def test_basic_completion(client: anthropic.Anthropic, model: str):
    print(f"=== Basic Completion: {model} ===")
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": "In one sentence, what is interactive fiction?"}],
    )
    elapsed = time.perf_counter() - start

    print(f"Model:   {response.model}")
    print(f"Time:    {elapsed:.2f}s")
    print(f"Reply:   {_extract_text(response)}")
    print(f"Finish:  {response.stop_reason}")
    print(f"Usage:   input={response.usage.input_tokens} output={response.usage.output_tokens}")
    print()


def test_streaming(client: anthropic.Anthropic, model: str):
    print(f"=== Streaming: {model} ===")
    start = time.perf_counter()
    chunks = 0
    print("Reply:   ", end="", flush=True)
    with client.messages.stream(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": "Describe a rainy courtyard in two sentences."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            chunks += 1

    elapsed = time.perf_counter() - start
    print(f"\nChunks:  {chunks}")
    print(f"Time:    {elapsed:.2f}s")
    print()


def test_structured_json(client: anthropic.Anthropic, model: str):
    print(f"=== Structured JSON Output: {model} ===")
    system = (
        "You are a world-state engine. Respond ONLY with valid JSON, no markdown fences.\n"
        'Schema: {"action": string, "feasible": bool, "outcome": string}'
    )
    user = "The player tries to lift a stone castle with bare hands."

    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    elapsed = time.perf_counter() - start

    raw = _extract_text(response)
    print(f"Raw:     {raw}")
    try:
        parsed = json.loads(raw)
        print(f"Parsed:  {json.dumps(parsed, indent=2)}")
        print("Valid JSON: yes")
    except json.JSONDecodeError as e:
        print(f"Valid JSON: NO - {e}")
    print(f"Time:    {elapsed:.2f}s")
    print()


def test_multi_turn(client: anthropic.Anthropic, model: str):
    print(f"=== Multi-Turn Conversation: {model} ===")
    system = "You are a guard captain named Vero. You speak in clipped, formal sentences."
    messages = [{"role": "user", "content": "I approach the gate."}]

    for turn in range(2):
        start = time.perf_counter()
        response = client.messages.create(
            model=model,
            max_tokens=150,
            system=system,
            messages=messages,
        )
        elapsed = time.perf_counter() - start

        reply = _extract_text(response)
        print(f"Turn {turn + 1}: ({elapsed:.2f}s) {reply}")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": "I ask to be let through." if turn == 0 else "I show my papers."})

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Anthropic Messages API")
    parser.add_argument("--model", default="claude-haiku-4-5", help="Model name")
    parser.add_argument("--test", choices=["basic", "stream", "json", "multi", "all"], default="all")
    args = parser.parse_args()

    client = make_client()

    tests = {
        "basic": test_basic_completion,
        "stream": test_streaming,
        "json": test_structured_json,
        "multi": test_multi_turn,
    }

    to_run = tests if args.test == "all" else {args.test: tests[args.test]}
    for name, fn in to_run.items():
        try:
            fn(client, args.model)
        except Exception as e:
            print(f"FAILED [{name}]: {e}\n")
