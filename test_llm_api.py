"""Test script for the LLM gateway chat completion API."""

import openai
import httpx
import os
import getpass
import time
import json
import argparse


def make_client(base_url: str) -> openai.OpenAI:
    headers = {
        "Ocp-Apim-Subscription-Key": os.getenv("LLM_GATEWAY_KEY"),
        "user": os.environ.get("USER", getpass.getuser()),
    }
    return openai.OpenAI(
        base_url=base_url,
        api_key="dummy",
        http_client=httpx.Client(verify=False),
        default_headers=headers,
    )


def test_basic_completion(client: openai.OpenAI, model: str):
    print(f"=== Basic Completion: {model} ===")
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=200,
        temperature=0.7,
        messages=[{"role": "user", "content": "In one sentence, what is interactive fiction?"}],
    )
    elapsed = time.perf_counter() - start

    choice = response.choices[0]
    print(f"Model:   {response.model}")
    print(f"Time:    {elapsed:.2f}s")
    print(f"Reply:   {choice.message.content}")
    print(f"Finish:  {choice.finish_reason}")
    print(f"Usage:   {response.usage.model_dump_json(indent=2)}")
    print()


def test_streaming(client: openai.OpenAI, model: str):
    print(f"=== Streaming: {model} ===")
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        max_completion_tokens=200,
        temperature=0.7,
        messages=[{"role": "user", "content": "Describe a rainy courtyard in two sentences."}],
        stream=True,
    )

    chunks = 0
    print("Reply:   ", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)
            chunks += 1

    elapsed = time.perf_counter() - start
    print(f"\nChunks:  {chunks}")
    print(f"Time:    {elapsed:.2f}s")
    print()


def test_structured_json(client: openai.OpenAI, model: str):
    print(f"=== Structured JSON Output: {model} ===")
    system = (
        "You are a world-state engine. Respond ONLY with valid JSON, no markdown fences.\n"
        'Schema: {"action": string, "feasible": bool, "outcome": string}'
    )
    user = "The player tries to lift a stone castle with bare hands."

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=200,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    elapsed = time.perf_counter() - start

    raw = response.choices[0].message.content
    print(f"Raw:     {raw}")
    try:
        parsed = json.loads(raw)
        print(f"Parsed:  {json.dumps(parsed, indent=2)}")
        print("Valid JSON: yes")
    except json.JSONDecodeError as e:
        print(f"Valid JSON: NO - {e}")
    print(f"Time:    {elapsed:.2f}s")
    print()


def test_multi_turn(client: openai.OpenAI, model: str):
    print(f"=== Multi-Turn Conversation: {model} ===")
    messages = [
        {"role": "system", "content": "You are a guard captain named Vero. You speak in clipped, formal sentences."},
        {"role": "user", "content": "I approach the gate."},
    ]

    for turn in range(2):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=150,
            temperature=0.7,
            messages=messages,
        )
        elapsed = time.perf_counter() - start

        reply = response.choices[0].message.content
        print(f"Turn {turn + 1}: ({elapsed:.2f}s) {reply}")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": "I ask to be let through." if turn == 0 else "I show my papers."})

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM gateway chat completion API")
    parser.add_argument("--url", default="https://llm-api.amd.com/OnPrem", help="Gateway base URL")
    parser.add_argument("--model", default="GPT-oss-120B", help="Model name")
    parser.add_argument("--test", choices=["basic", "stream", "json", "multi", "all"], default="all")
    args = parser.parse_args()

    if not os.getenv("LLM_GATEWAY_KEY"):
        print("ERROR: Set LLM_GATEWAY_KEY environment variable")
        raise SystemExit(1)

    client = make_client(args.url)

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
