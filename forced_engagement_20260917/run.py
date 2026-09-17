"""Replay frozen narrative requests; never modify the source session or retry a call."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = {
    "t16_departure": "c9d71fda988647328be8b37f4949bae0",
    "t20_library": "f356b73111124cd9b7346ef4c0d2a8a6",
    "t21_reading": "b98598905f8f4b3bb851c3b38c607cb6",
}
CONDITIONS = ("api_native", "api_flat", "codex_proxy")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(worktree, session, core):
    if (ROOT / "manifest.json").exists():
        raise SystemExit("Already frozen; use --cell or --pending without --prepare")
    originals = ROOT / "originals"
    originals.mkdir()
    for name in ("manifest.json", "state.json", "transcript.md"):
        shutil.copyfile(session / name, originals / name)
    shutil.copytree(session / "snapshot", originals / "snapshot")
    source_ids = {**CASES, "t16_before_regeneration": "9e4b73c34be843ae9414f7b91db7f753"}
    for label, rid in source_ids.items():
        shutil.copytree(core.attempt_dir(session, rid), originals / label)
    for name in ("core.py", "codex.py", "__main__.py"):
        target = ROOT / "implementation" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(worktree / "narrative" / name, target)
    cells = []
    for repeat in (1, 2):
        for case in CASES:
            source = core.read_json(originals / case / "request.json")
            assert source["model"] == "gpt-5.6-terra"
            assert source["reasoning"] == {"effort": "max", "summary": "auto"}
            assert (originals / case / "request.txt").read_text() == core.render_request(source)
            for condition in CONDITIONS:
                cell_id = f"{case}/{condition}/{repeat}"
                path = ROOT / "cells" / cell_id
                path.mkdir(parents=True)
                if condition == "api_flat":
                    request = {
                        k: v for k, v in source.items() if k not in ("instructions", "input")
                    }
                    request["input"] = [{"role": "user", "content": core.render_request(source)}]
                    core.write_json(path / "request.json", request)
                    (path / "request.txt").write_text(request["input"][0]["content"])
                else:
                    for name in ("request.json", "request.txt"):
                        shutil.copyfile(originals / case / name, path / name)
                cells.append(
                    {
                        "id": cell_id,
                        "case": case,
                        "condition": condition,
                        "repeat": repeat,
                        "request_sha256": sha(path / "request.json"),
                    }
                )
    tracked = [p for p in originals.rglob("*") if p.is_file()]
    core.write_json(
        ROOT / "manifest.json",
        {
            "prepared_at": core.now(),
            "source_session": session.name,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
            ).strip(),
            "codex_version": subprocess.check_output(["codex", "--version"], text=True).strip(),
            "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in sorted(tracked)},
            "cells": cells,
        },
    )
    print(f"Frozen {len(cells)} cells; no calls made", flush=True)


def execute(cell, worktree, env_file, core):
    from dotenv import dotenv_values

    from narrative.codex import CodexProxy

    path = ROOT / "cells" / cell["id"]
    if sha(path / "request.json") != cell["request_sha256"]:
        raise RuntimeError("Prepared request changed")
    if (path / "started.json").exists():
        raise RuntimeError("Refusing to repeat a dispatched call")
    request = core.read_json(path / "request.json")
    transport = "proxy" if cell["condition"] == "codex_proxy" else "api"
    if transport == "api":
        # Explicitly reuse the same direct-OpenAI credential as the archived trials.
        values = dotenv_values(env_file)
        key = (
            os.environ.get("OPENAI_API_KEY")
            or values.get("OPENAI_API_KEY")
            or values.get("OPEN_AI_ROUTER")
        )
        if not key:
            raise RuntimeError("No direct API credential configured")
        from openai import OpenAI

        client = OpenAI(
            api_key=key, base_url="https://api.openai.com/v1/", timeout=300.0, max_retries=0
        )
    else:
        client = CodexProxy()
    print("Starting " + cell["id"], flush=True)
    try:
        core._call_model(path, request, client, transport)
        response = core.read_json(path / "response.json")
        text = core.response_text(response)
        (path / "output.md").write_text(text)
        (path / "summary.txt").write_text(core.response_summary(response))
        raw = response.get("raw")
        core.write_json(
            path / "result.json",
            {
                "status": "completed",
                "word_count": len(text.split()),
                "duration_s": response["duration_s"],
                "usage": raw.get("usage") if isinstance(raw, dict) else None,
                "returned_model": raw.get("model") if isinstance(raw, dict) else None,
            },
        )
        print("Completed " + cell["id"], flush=True)
    except Exception as exc:
        # Save classifications only: provider error strings can contain sensitive data.
        original = exc.__cause__ or exc
        body = getattr(original, "body", None)
        details = body.get("error", body) if isinstance(body, dict) else {}
        core.write_json(
            path / "result.json",
            {
                "status": "failed",
                "type": type(original).__name__,
                "status_code": getattr(original, "status_code", None),
                "provider_type": details.get("type"),
                "provider_code": details.get("code"),
                "provider_param": details.get("param"),
            },
        )
        print("Failed " + cell["id"] + ": " + type(original).__name__, flush=True)
    finally:
        if transport == "api":
            client.close()
    return core.read_json(path / "result.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--prepare", type=Path, metavar="SOURCE_SESSION")
    parser.add_argument("--cell")
    parser.add_argument("--pending", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.worktree))
    from narrative import core

    if args.prepare:
        prepare(args.worktree, args.prepare, core)
        return
    manifest = core.read_json(ROOT / "manifest.json")
    for name, expected in manifest["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Frozen source changed: " + name)
    cells = [
        cell
        for cell in manifest["cells"]
        if cell["id"] == args.cell
        or (args.pending and not (ROOT / "cells" / cell["id"] / "started.json").exists())
    ]
    if not cells:
        raise SystemExit("No selected undispatched calls")
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(execute, cell, args.worktree, args.env_file, core) for cell in cells]
        for task in as_completed(tasks):
            task.result()


if __name__ == "__main__":
    main()
