#!/usr/bin/env python3
"""Live narrator-only probe for per-viewpoint name handling.

The harness loads representative upstream facts from the long D&D CLI
playtest and calls only the narrator role. It does not run the router,
character agents, combat manager, or turn loop.

Outputs:
  app/storage/playtest_reports/narrator_name_probe_<timestamp>/report.json
  app/storage/playtest_reports/narrator_name_probe_<timestamp>/report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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

from app.engine.context_builder import (
    build_narrator_player_characters_block,
    build_narrator_public_character_context_block,
    build_narrator_pov_knowledge_block,
    build_setting_summary,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile
from app.schemas.narrator import NarratorFinalOutput


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "app/storage/sessions/cli_long_dnd_social_20260520/ckpt_0027.json"
)
REPORT_ROOT = REPO_ROOT / "app/storage/playtest_reports"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = REPORT_ROOT / f"narrator_name_probe_{TS}"
JSON_PATH = RUN_DIR / "report.json"
MD_PATH = RUN_DIR / "report.md"
LOG_PATH = RUN_DIR / "run.log"
RAW_CALLS_PATH = RUN_DIR / "raw_calls.jsonl"


@dataclass(frozen=True)
class ProbeCase:
    name: str
    description: str
    pov_character_id: str
    event_ids: tuple[str, ...]
    user_input: str
    forbidden_terms: tuple[str, ...]
    required_terms: tuple[str, ...] = ()


@dataclass
class CaseResult:
    name: str
    description: str
    event_ids: list[str]
    visible_events: str
    final_text: str = ""
    checks: list[dict[str, Any]] | None = None
    error: str = ""


CASES = (
    ProbeCase(
        name="first_reveal_raw_ids_unintroduced",
        description=(
            "The first lookout appears and a second voice answers from brush. "
            "Upstream facts use brigand ids, but no one has spoken Dace or "
            "Thorne yet."
        ),
        pov_character_id="player_protagonist",
        event_ids=("evt_brigand_steps_out_0023",),
        user_input="(defer)",
        forbidden_terms=(
            "Dace",
            "Thorne",
            "Merrow",
            "Vex",
            "brigand_lookout",
            "player_protagonist",
            "korva_sahl",
        ),
    ),
    ProbeCase(
        name="resolver_names_not_introduced",
        description=(
            "The D&D resolver leaked Dace/Thorne labels in public outcome "
            "facts before those names were established in dialogue."
        ),
        pov_character_id="player_protagonist",
        event_ids=("evt_0af9d8b7d2df",),
        user_input=(
            "I keep my voice level and bluff that a Guild patrol is close "
            "enough to hear a whistle."
        ),
        forbidden_terms=(
            "Dace",
            "Thorne",
            "Merrow",
            "Vex",
            "brigand_lookout",
            "player_protagonist",
            "korva_sahl",
        ),
    ),
    ProbeCase(
        name="names_available_after_player_says_them",
        description=(
            "The player has now spoken Dace and Thorne aloud, so those names "
            "are available to this viewpoint."
        ),
        pov_character_id="player_protagonist",
        event_ids=("evt_pp1_middle_path_restrain_open_0029", "evt_039512fdd155"),
        user_input=(
            "I choose the middle path. 'Dace, on your knees, hands behind "
            "your head. Thorne, step out where Lyra can see you. We tie both "
            "of you, no blades unless you force it.'"
        ),
        forbidden_terms=("brigand_lookout", "player_protagonist", "korva_sahl"),
        required_terms=("Dace", "Thorne"),
    ),
)


def _load_checkpoint(path: Path) -> CheckpointFile:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return CheckpointFile.model_validate_json(path.read_text())


def _character_name(ckpt: CheckpointFile, character_id: str) -> str:
    for character in ckpt.characters:
        if character.character_id == character_id:
            return character.name or character_id
    return character_id


def _event_by_id(ckpt: CheckpointFile, event_id: str):
    for event in ckpt.canonical_events:
        if event.event_id == event_id:
            return event
    raise KeyError(f"Event not found in checkpoint: {event_id}")


def _visible_fact_lines(
    ckpt: CheckpointFile,
    *,
    event_ids: tuple[str, ...],
    pov_character_id: str,
) -> list[str]:
    lines: list[str] = []
    for event_id in event_ids:
        event = _event_by_id(ckpt, event_id)
        for fact in event.canonical_event.observable_facts:
            if fact.audience == "all_observers" or fact.is_visible_to(pov_character_id):
                text = (fact.text or "").strip()
                if text:
                    lines.append(text)
    if not lines:
        raise ValueError(
            f"No visible facts found for {pov_character_id} in {event_ids}"
        )
    return lines


def _visible_events_block(
    ckpt: CheckpointFile,
    *,
    case: ProbeCase,
) -> str:
    lines = ["Seen directly:"]
    for fact in _visible_fact_lines(
        ckpt,
        event_ids=case.event_ids,
        pov_character_id=case.pov_character_id,
    ):
        lines.append(f"- {fact}")
    return "\n".join(lines)


def _term_present(text: str, term: str) -> bool:
    return re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", text) is not None


def _checks(case: ProbeCase, final_text: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for term in case.forbidden_terms:
        present = _term_present(final_text, term)
        checks.append({
            "name": f"forbid:{term}",
            "passed": not present,
            "detail": "absent" if not present else "present in narrator output",
        })
    for term in case.required_terms:
        present = _term_present(final_text, term)
        checks.append({
            "name": f"require:{term}",
            "passed": present,
            "detail": "present" if present else "missing from narrator output",
        })
    return checks


async def _run_case(
    *,
    case: ProbeCase,
    ckpt: CheckpointFile,
    client: LLMClient,
    prompt_mgr: PromptManager,
) -> CaseResult:
    visible_events = _visible_events_block(ckpt, case=case)
    result = CaseResult(
        name=case.name,
        description=case.description,
        event_ids=list(case.event_ids),
        visible_events=visible_events,
    )

    pov_name = _character_name(ckpt, case.pov_character_id)
    messages = prompt_mgr.render_conversation(
        "narrator_phase2",
        history=[],
        setting_summary=build_setting_summary(ckpt),
        narrative_rules=ckpt.config.narrative_rules or "No specific narrative rules.",
        visible_events=visible_events,
        user_input=case.user_input,
        pov_character_name=pov_name,
        player_characters_block=build_narrator_player_characters_block(
            ckpt,
            case.pov_character_id,
        ),
        public_character_context_block=build_narrator_public_character_context_block(
            ckpt,
        ),
        pov_knowledge_block=build_narrator_pov_knowledge_block(
            ckpt,
            case.pov_character_id,
            visible_events,
        ),
        rendering_note="Write through to the natural handoff point.",
    )
    with RAW_CALLS_PATH.open("a") as fh:
        fh.write(json.dumps({
            "case": case.name,
            "messages": messages,
        }) + "\n")

    response = await client.complete(
        role="narrator",
        messages=messages,
        response_model=NarratorFinalOutput,
        temperature=0.2,
        max_tokens=1200,
        cache=True,
        compact=True,
    )
    parsed: NarratorFinalOutput = response.parsed
    result.final_text = parsed.final_text
    result.checks = _checks(case, result.final_text)
    return result


def _report_json(
    *,
    checkpoint_path: Path,
    config: LLMConfig,
    results: list[CaseResult],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "role": "narrator",
        "provider": config.provider_for_role("narrator"),
        "model": config.model_for_role("narrator"),
        "all_passed": all(
            not result.error
            and all(check.get("passed") for check in (result.checks or []))
            for result in results
        ),
        "results": [
            {
                "name": result.name,
                "description": result.description,
                "event_ids": result.event_ids,
                "visible_events": result.visible_events,
                "final_text": result.final_text,
                "checks": result.checks or [],
                "error": result.error,
            }
            for result in results
        ],
    }


def _report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Narrator Name Probe",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Checkpoint: `{report['checkpoint_path']}`",
        f"Role/model: `{report['provider']}:{report['model']}`",
        f"All passed: `{report['all_passed']}`",
        "",
    ]
    for result in report["results"]:
        lines.extend([
            f"## {result['name']}",
            "",
            result["description"],
            "",
            f"Events: `{', '.join(result['event_ids'])}`",
            "",
        ])
        if result["error"]:
            lines.extend(["Error:", "", "```text", result["error"], "```", ""])
            continue
        lines.extend(["Narrator output:", "", "```text", result["final_text"], "```", ""])
        lines.append("Checks:")
        for check in result["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {mark}: `{check['name']}` - {check['detail']}")
        lines.append("")
        lines.extend(["Sample facts:", "", "```text", result["visible_events"], "```", ""])
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a narrator-only live name-obfuscation probe.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint to sample from (default: {DEFAULT_CHECKPOINT})",
    )
    args = parser.parse_args()

    load_dotenv()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ckpt = _load_checkpoint(args.checkpoint)
    config = LLMConfig.from_env()
    provider = config.provider_for_role("narrator")
    api_key = config.api_key_for_provider(provider, role="narrator")
    if not api_key:
        names = (
            config.openai_role_api_key_env_names("narrator")
            if provider == "openai"
            else ("ANTHROPIC_API_KEY",)
        )
        raise SystemExit(
            f"No API key found for narrator provider={provider!r}. "
            f"Expected one of: {', '.join(names)}"
        )

    client = LLMClient(config)
    prompt_mgr = PromptManager(str(REPO_ROOT / "app/prompts"))
    results: list[CaseResult] = []
    try:
        for case in CASES:
            print(f"running {case.name}...", flush=True)
            try:
                results.append(await _run_case(
                    case=case,
                    ckpt=ckpt,
                    client=client,
                    prompt_mgr=prompt_mgr,
                ))
            except Exception:
                results.append(CaseResult(
                    name=case.name,
                    description=case.description,
                    event_ids=list(case.event_ids),
                    visible_events="",
                    error=traceback.format_exc(),
                ))
    finally:
        await client.close()

    report = _report_json(
        checkpoint_path=args.checkpoint,
        config=config,
        results=results,
    )
    JSON_PATH.write_text(json.dumps(report, indent=2))
    MD_PATH.write_text(_report_md(report))

    print(f"wrote {JSON_PATH}")
    print(f"wrote {MD_PATH}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
