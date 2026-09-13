#!/usr/bin/env python3
"""Standalone, resumable one-conversation fiction CLI. No Ayoa imports."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


HERE = Path(__file__).resolve().parent
PROMPTS = ("system.txt", "covenant.txt", "player.json")
INSTRUCTION_ORDER = ("covenant.txt", "system.txt")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def init_run(run: Path, model: str, effort: str, max_tokens: int) -> None:
    run.mkdir(parents=True, exist_ok=False)
    (run / "turns").mkdir()
    (run / "attempts").mkdir()
    for name in PROMPTS:
        shutil.copyfile(HERE / name, run / name)
    write_json(run / "manifest.json", {
        "created_at": utc_now(),
        "model": model,
        "reasoning_effort": effort,
        "max_output_tokens": max_tokens,
        "instruction_order": list(INSTRUCTION_ORDER),
        "prompt_sha256": {name: sha256(run / name) for name in PROMPTS},
        "architecture": "one full-context model; one response per submission",
        "memory": "complete user/assistant prose history; no private state or summaries",
        "transport_retries": 0,
    })


def load_run(run: Path) -> tuple[dict, list[dict]]:
    manifest = read_json(run / "manifest.json")
    for name, expected in manifest["prompt_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"Frozen prompt changed: {name}")
    turns = []
    for index, path in enumerate(sorted((run / "turns").glob("*.json"))):
        turn = read_json(path)
        if path.name != f"{index:02d}.json" or turn["turn"] != index:
            raise ValueError("Turn history has a gap or inconsistent index")
        turns.append(turn)
    return manifest, turns


def make_request(run: Path, manifest: dict, turns: list[dict], text: str) -> dict:
    if not text.strip():
        raise ValueError("A player submission cannot be empty")
    order = manifest.get("instruction_order")
    if not isinstance(order, list) or sorted(order) != sorted(INSTRUCTION_ORDER):
        raise ValueError("Run needs a frozen instruction order; initialize a new run")
    messages = []
    for turn in turns:
        messages.extend([
            {"role": "user", "content": turn["input"]},
            {"role": "assistant", "content": turn["output"]},
        ])
    messages.append({"role": "user", "content": text})
    return {
        "model": manifest["model"],
        "reasoning": {"effort": manifest["reasoning_effort"]},
        "instructions": "\n\n".join(
            (run / name).read_text(encoding="utf-8").strip()
            for name in order
        ),
        "input": messages,
        "max_output_tokens": manifest["max_output_tokens"],
        "store": False,
    }


def accepted_text(response: Any) -> str:
    if response.status != "completed":
        raise RuntimeError(f"Response was {response.status}; raw output preserved")
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "refusal":
                    raise RuntimeError("Model refused; raw output preserved")
    text = response.output_text
    if not text or not text.strip():
        raise RuntimeError("Model returned empty prose; raw output preserved")
    return text


def export_run(run: Path) -> None:
    manifest, turns = load_run(run)
    lines = ["# Covenant: one-model playtest", "", "Exact submitted actions and model prose.", ""]
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    for turn in turns:
        label = "Opening" if turn["turn"] == 0 else f"Turn {turn['turn']}"
        lines.extend([f"## {label}", "", "**Player**", "", turn["input"],
                      "", "**Story**", "", turn["output"], ""])
        usage = turn["usage"]
        totals["input_tokens"] += usage.get("input_tokens", 0)
        totals["output_tokens"] += usage.get("output_tokens", 0)
        totals["cached_tokens"] += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        totals["reasoning_tokens"] += (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    (run / "transcript.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "model": manifest["model"],
        "reasoning_effort": manifest["reasoning_effort"],
        "accepted_calls": len(turns),
        "player_turns": max(0, len(turns) - 1),
        "attempted_calls": len(list((run / "attempts").glob("*"))),
        "total_call_seconds": round(sum(t["duration_s"] for t in turns), 3),
        "visible_words": sum(len(t["output"].split()) for t in turns),
        "visible_output_tokens": totals["output_tokens"] - totals["reasoning_tokens"],
        **totals,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def play_turn(run: Path, client: Any, text: str) -> dict:
    # The lock also prevents two CLI invocations from answering the same history.
    with (run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, turns = load_run(run)
        request = make_request(run, manifest, turns, text)
        attempt = run / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        attempt.mkdir()
        write_json(attempt / "request.json", request)
        started_at = utc_now()
        start = time.monotonic()
        try:
            response = client.responses.create(**request)
            elapsed = time.monotonic() - start
            write_json(attempt / "response.json", response.model_dump(mode="json"))
            output = accepted_text(response)
        except Exception as exc:
            write_json(attempt / "error.json", {
                "type": type(exc).__name__,
                "started_at": started_at,
                "duration_s": round(time.monotonic() - start, 3),
            })
            raise
        turn = {
            "turn": len(turns),
            "input": text,
            "output": output,
            "model": response.model,
            "response_id": response.id,
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_s": round(elapsed, 3),
            "usage": response.usage.model_dump(mode="json") if response.usage else {},
            "attempt": str(attempt.relative_to(run)),
        }
        write_json(run / "turns" / f"{len(turns):02d}.json", turn)
        export_run(run)
        return turn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "turn", "export"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning", default="max")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--text")
    inputs.add_argument("--input-file", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.command == "export":
        export_run(run)
        return
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    # Reuse an existing credential, never the Ayoa runtime or role dispatcher.
    key = os.environ.get("OPEN_AI_ROUTER") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPEN_AI_ROUTER or OPENAI_API_KEY")
    if args.command == "init":
        init_run(run, args.model, args.reasoning, args.max_output_tokens)
        player = read_json(run / "player.json")
        text = f"My character: {player['name']}. {player['description']}\n\n{player['opening']}"
    else:
        text = args.input_file.read_text(encoding="utf-8") if args.input_file else args.text
        if not text:
            parser.error("turn requires --text or --input-file")
    with OpenAI(api_key=key, timeout=300.0, max_retries=0) as client:
        turn = play_turn(run, client, text)
    print(f"Turn {turn['turn']} saved ({turn['duration_s']:.1f}s)", flush=True)
    print(turn["output"], flush=True)


if __name__ == "__main__":
    main()
