#!/usr/bin/env python3
"""Live narrator-only probe for lossless semantic compression.

The paired cases differ in one small but meaningful reply gesture. The probe
calls only the narrator role and records the complete constructed messages,
structured output, and lightweight semantic checks for manual review.

Outputs:
  app/storage/playtest_reports/narrator_semantic_compression_<timestamp>/report.json
  app/storage/playtest_reports/narrator_semantic_compression_<timestamp>/report.md
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.narrator import NarratorFinalOutput


REPORT_ROOT = REPO_ROOT / "app/storage/playtest_reports"


@dataclass(frozen=True)
class ProbeCase:
    name: str
    visible_facts: tuple[str, ...]
    required_patterns: tuple[tuple[str, str], ...]


CASES = (
    ProbeCase(
        name="acknowledged_two_finger_signal",
        visible_facts=(
            "Bob raises his index and middle fingers toward Alice.",
            "Alice answers with a slight nod.",
            "Alice shifts her weight onto her back foot in preparation.",
            "Bob lowers his hand, ending the signal.",
            "Bob waits.",
        ),
        required_patterns=(
            ("bob_two_fingers", r"(?:two fingers|index.{0,30}middle)"),
            ("alice_nods", r"\bnod(?:s|ded|ding)?\b"),
            ("alice_back_foot", r"back foot|weight.{0,30}back"),
            ("bob_lowers_hand", r"(?:lower|drop)(?:s|ed|ing)?.{0,30}hand"),
            ("bob_waits", r"\bwait(?:s|ed|ing)?\b"),
        ),
    ),
    ProbeCase(
        name="middle_finger_reversal",
        visible_facts=(
            "Bob raises his index and middle fingers toward Alice.",
            "Alice answers with only her middle finger.",
            "Alice shifts her weight onto her back foot in preparation.",
            "Bob lowers his hand, ending the signal.",
            "Bob waits.",
        ),
        required_patterns=(
            ("bob_two_fingers", r"(?:two fingers|index.{0,30}middle)"),
            ("alice_middle_finger", r"middle finger"),
            ("alice_back_foot", r"back foot|weight.{0,30}back"),
            ("bob_lowers_hand", r"(?:lower|drop)(?:s|ed|ing)?.{0,30}hand"),
            ("bob_waits", r"\bwait(?:s|ed|ing)?\b"),
        ),
    ),
)


def _visible_events(case: ProbeCase) -> str:
    return "Seen directly:\n" + "\n".join(
        f"- {fact}" for fact in case.visible_facts
    )


def _checks(case: ProbeCase, final_text: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, pattern in case.required_patterns:
        passed = re.search(pattern, final_text, flags=re.IGNORECASE) is not None
        checks.append({
            "name": name,
            "passed": passed,
            "pattern": pattern,
        })
    return checks


async def _run_case(
    case: ProbeCase,
    *,
    client: LLMClient,
    prompt_mgr: PromptManager,
) -> dict[str, Any]:
    visible_events = _visible_events(case)
    messages = prompt_mgr.render_conversation(
        "narrator_phase2",
        history=[],
        setting_summary=(
            "A grounded contemporary scene in a quiet warehouse corridor."
        ),
        narrative_rules="Use concise, natural prose.",
        visible_events=visible_events,
        user_input="",
        pov_character_name="Alice",
        player_characters_block="- Alice (you)",
        rendering_note="Write through to the natural handoff point.",
        handoff_policy="forced",
        handoff_context="No unresolved submitted activity.",
    )
    response = await client.complete(
        role="narrator",
        messages=messages,
        response_model=NarratorFinalOutput,
        temperature=0.2,
        max_tokens=1000,
        cache=True,
        compact=True,
    )
    parsed: NarratorFinalOutput | None = response.parsed
    if parsed is None:
        raise RuntimeError("Narrator returned no structured result.")
    checks = _checks(case, parsed.final_text)
    return {
        "name": case.name,
        "visible_facts": list(case.visible_facts),
        "messages": messages,
        "output": parsed.model_dump(),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "error": "",
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Narrator Semantic Compression Probe",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Role/model: `{report['provider']}:{report['model']}`",
        f"All checks passed: `{report['all_passed']}`",
        "",
    ]
    for result in report["results"]:
        lines.extend([f"## {result['name']}", ""])
        if result["error"]:
            lines.extend(["```text", result["error"], "```", ""])
            continue
        lines.extend([
            "Visible facts:",
            "",
            "```text",
            *result["visible_facts"],
            "```",
            "",
            "Narrator output:",
            "",
            "```text",
            result["output"]["final_text"],
            "```",
            "",
            "Checks:",
            "",
        ])
        for check in result["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {mark}: `{check['name']}`")
        lines.append("")
    return "\n".join(lines)


async def main() -> int:
    load_dotenv()
    config = LLMConfig.from_env()
    provider = config.provider_for_role("narrator")
    if not config.api_key_for_provider(provider, role="narrator"):
        raise SystemExit(f"No API key found for narrator provider={provider!r}.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPORT_ROOT / f"narrator_semantic_compression_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    client = LLMClient(config)
    prompt_mgr = PromptManager(str(REPO_ROOT / "app/prompts"))
    results: list[dict[str, Any]] = []
    try:
        for case in CASES:
            print(f"running {case.name}...", flush=True)
            try:
                results.append(await _run_case(
                    case,
                    client=client,
                    prompt_mgr=prompt_mgr,
                ))
            except Exception:
                results.append({
                    "name": case.name,
                    "visible_facts": list(case.visible_facts),
                    "messages": [],
                    "output": {},
                    "checks": [],
                    "passed": False,
                    "error": traceback.format_exc(),
                })
    finally:
        await client.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": config.model_for_role("narrator"),
        "all_passed": all(result["passed"] for result in results),
        "results": results,
    }
    json_path = run_dir / "report.json"
    markdown_path = run_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(_report_markdown(report) + "\n")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
