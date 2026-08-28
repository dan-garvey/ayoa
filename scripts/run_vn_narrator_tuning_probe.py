#!/usr/bin/env python3
"""Replay the 2026-08-26 VN failures through the live model contracts.

The probe first verifies that the router preserves a quoted pronoun plus its
silent identity anchor. It then preserves the source playtest checkpoints,
reconstructs the pre-render per-POV history, records every model input, and
produces neutral-stage cards for accepted narrator renders.
"""

from __future__ import annotations

# ruff: noqa: E402 -- executable scripts add the repository root to sys.path.

import asyncio
import json
import re
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine.narrator import compose_pov_render
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop import _narrator_handoff_context
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.engine.visual_novel_presentation import (
    VisualNovelCardRenderer,
    VisualNovelDeckSection,
)
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.narrator import VisualNovelNarratorOutput, VisualNovelPage
from app.schemas.state import (
    RenderBufferEntry,
    SessionState,
    StorySetting,
    WorldState,
)


SOURCE_SESSION = (
    REPO_ROOT
    / "app/storage/sessions/one-star-vn-luna-playtest-20260826"
)
REPORT_ROOT = REPO_ROOT / "app/storage/playtest_reports"
POV_CHARACTER_ID = "the_master"
_COMPLETE_END_RE = re.compile(r"[.!?…]+(?:[\"”’')\]]+)?$")


@dataclass(frozen=True)
class ProbeCase:
    name: str
    event_id: str
    before_checkpoint: str
    after_checkpoint: str
    handoff_policy: str
    expected_handoff: str
    update_to_pronoun_anchor_contract: bool = False


CASES = (
    ProbeCase(
        name="bracketed_direct_address_forced",
        event_id="event_0004",
        before_checkpoint="ckpt_0004.json",
        after_checkpoint="ckpt_0005.json",
        handoff_policy="forced",
        expected_handoff="render",
        update_to_pronoun_anchor_contract=True,
    ),
    ProbeCase(
        name="long_action_sentence_forced",
        event_id="event_0005",
        before_checkpoint="ckpt_0005.json",
        after_checkpoint="ckpt_0006.json",
        handoff_policy="forced",
        expected_handoff="render",
    ),
    ProbeCase(
        name="spectator_repeated_defer_candidate",
        event_id="event_0005",
        before_checkpoint="ckpt_0005.json",
        after_checkpoint="ckpt_0006.json",
        handoff_policy="candidate",
        expected_handoff="continue",
    ),
)


class _RecordingClient:
    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any):
        self.calls.append({
            "role": kwargs.get("role", ""),
            "messages": deepcopy(kwargs.get("messages", [])),
            "response_model": getattr(
                kwargs.get("response_model"),
                "__name__",
                str(kwargs.get("response_model", "")),
            ),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "cache": kwargs.get("cache"),
            "compact": kwargs.get("compact"),
        })
        return await self.inner.complete(**kwargs)


def _load_checkpoint(filename: str) -> CheckpointFile:
    path = SOURCE_SESSION / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing source checkpoint: {path}")
    return CheckpointFile.model_validate_json(path.read_text(encoding="utf-8"))


def _replay_checkpoint(case: ProbeCase) -> CheckpointFile:
    before = _load_checkpoint(case.before_checkpoint)
    after = _load_checkpoint(case.after_checkpoint)
    after.narrator_conversations[POV_CHARACTER_ID] = deepcopy(
        before.narrator_conversations.get(POV_CHARACTER_ID, [])
    )
    after.session.visual_introductions = deepcopy(
        before.session.visual_introductions
    )
    if case.update_to_pronoun_anchor_contract:
        event = next(
            event
            for event in after.canonical_events
            if event.event_id == case.event_id
        )
        for fact in event.canonical_event.observable_facts:
            fact.text = fact.text.replace(
                "offer [niflheim_first_summon_01] another",
                "offer you [niflheim_first_summon_01] another",
            ).replace(
                "but [niflheim_first_summon_01] may inspect",
                "but you [niflheim_first_summon_01] may inspect",
            )
    return after


def _result_text(result: VisualNovelNarratorOutput) -> str:
    return "\n".join(
        f"{page.speaker}: {page.text}" if page.kind == "dialogue" else page.text
        for beat in result.beats
        for page in beat.pages
    )


def _result_pages(
    result: VisualNovelNarratorOutput,
) -> tuple[VisualNovelPage, ...]:
    return tuple(page for beat in result.beats for page in beat.pages)


async def _run_router_anchor_probe(
    *,
    client: _RecordingClient,
    prompt_mgr: PromptManager,
) -> dict[str, Any]:
    checkpoint = CheckpointFile(
        session=SessionState(
            session_id="vn_quoted_referent_probe",
            player_character_id="alice",
            character_bindings={"alice": "probe-player"},
        ),
        world_state=WorldState(setting=StorySetting(
            genre="grounded fantasy",
            tone="plain and direct",
            premise="Two travelers inspect a modest waystation.",
        )),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                location="waystation",
                is_playable=True,
                public_sheet=PublicSheet(role="traveler"),
            ),
            CharacterRecord(
                character_id="pip",
                name="Pip",
                location="waystation",
                public_sheet=PublicSheet(role="guide"),
            ),
        ],
    )
    dispatcher = LLMDispatcher(client, prompt_mgr)  # type: ignore[arg-type]
    call_start = len(client.calls)
    result = await dispatcher.route_intention(
        ckpt=checkpoint,
        actor_id="pip",
        intention=(
            "Pip holds the side door open for Alice. 'I can't offer you "
            "another route, but you may inspect the hall before deciding.'"
        ),
        cat_ii_event=None,
    )
    facts = "\n".join(
        fact.text for fact in result.canonical_event.observable_facts
    )
    anchor_count = facts.casefold().count("you [alice]")
    checks = [
        {
            "name": "spoken_pronoun_anchor_preserved",
            "passed": anchor_count >= 2,
            "detail": f"you [alice] anchors={anchor_count}",
        },
        {
            "name": "pronoun_not_replaced_by_anchor",
            "passed": "offer [alice]" not in facts.casefold(),
            "detail": "the identity anchor follows rather than replaces you",
        },
        {
            "name": "spoken_meaning_preserved",
            "passed": "route" in facts.casefold()
            and "inspect" in facts.casefold(),
            "detail": "route and inspection remain observable",
        },
    ]
    return {
        "output": result.model_dump(mode="json"),
        "fact_text": facts,
        "model_calls": client.calls[call_start:],
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "error": "",
    }


def _checks(
    case: ProbeCase,
    result: VisualNovelNarratorOutput,
    *,
    card_texts: list[str],
) -> list[dict[str, Any]]:
    output_text = _result_text(result)
    checks: list[tuple[str, bool, str]] = [
        (
            "expected_handoff",
            result.handoff == case.expected_handoff,
            f"expected={case.expected_handoff} actual={result.handoff}",
        ),
        (
            "no_source_identifier",
            "niflheim_" not in output_text.casefold(),
            "no roster source identifier may reach a page",
        ),
    ]
    pages = _result_pages(result)
    if case.name == "bracketed_direct_address_forced":
        dialogue = " ".join(
            page.text
            for page in pages
            if page.kind == "dialogue" and page.speaker.casefold() == "iselle"
        )
        second_person_count = len(
            re.findall(r"\byou(?:r|rs|rself)?\b", dialogue, re.IGNORECASE)
        )
        checks.extend([
            (
                "natural_second_person_address",
                second_person_count >= 2,
                f"second-person references={second_person_count}",
            ),
            (
                "no_repeated_full_name",
                output_text.casefold().count("mara pell") <= 1,
                "Mara Pell may appear in narration once, not twice as address",
            ),
            (
                "route_and_inspection_preserved",
                "route" in dialogue.casefold()
                and "inspect" in dialogue.casefold(),
                "the deployment-route and inspection meanings remain",
            ),
        ])
    elif case.name == "long_action_sentence_forced":
        checks.extend([
            (
                "authored_pages_end_cleanly",
                bool(pages) and all(
                    _COMPLETE_END_RE.search(page.text.rstrip()) is not None
                    for page in pages
                ),
                "each semantic page ends after a complete sentence or utterance",
            ),
            (
                "knife_detail_preserved",
                "knife" in output_text.casefold(),
                "consequential held-knife detail remains",
            ),
            (
                "spoken_warning_preserved",
                "nobody follows me" in output_text.casefold(),
                "Mara's directly witnessed warning remains",
            ),
            (
                "physical_cards_fit",
                bool(card_texts) and all(
                    len(text.splitlines()) <= 4 for text in card_texts
                ),
                "each compositor card uses at most four measured lines",
            ),
        ])
    return [
        {"name": name, "passed": passed, "detail": detail}
        for name, passed, detail in checks
    ]


async def _run_case(
    case: ProbeCase,
    *,
    client: _RecordingClient,
    prompt_mgr: PromptManager,
    run_dir: Path,
) -> dict[str, Any]:
    checkpoint = _replay_checkpoint(case)
    buffer = [RenderBufferEntry(
        event_id=case.event_id,
        observation_level="direct",
    )]
    commitments = [
        commitment.description
        for commitment in checkpoint.session.open_commitments
        if POV_CHARACTER_ID in commitment.actor_ids and commitment.description
    ]
    handoff_context = _narrator_handoff_context(
        checkpoint,
        pov_character_id=POV_CHARACTER_ID,
        buffered_events=buffer,
        commitments=commitments,
        current_user_input="(defer)",
        candidate=case.handoff_policy == "candidate",
    )
    call_start = len(client.calls)
    result, _entry = await compose_pov_render(
        client=client,  # type: ignore[arg-type]
        prompt_mgr=prompt_mgr,
        ckpt=checkpoint,
        pov_character_id=POV_CHARACTER_ID,
        buffered_events=buffer,
        partial_mode=False,
        user_input="(defer)",
        handoff_policy=case.handoff_policy,
        handoff_context=handoff_context,
    )
    if not isinstance(result, VisualNovelNarratorOutput):
        raise TypeError("VN replay returned the prose narrator schema")

    deck_id = ""
    card_paths: list[str] = []
    card_texts: list[str] = []
    if result.handoff == "render":
        renderer = VisualNovelCardRenderer(run_dir / "cards" / case.name)
        deck = renderer.render_deck([
            VisualNovelDeckSection(pages=tuple(beat.pages))
            for beat in result.beats
        ])
        deck_id = deck.deck_id
        card_paths = [str(card.image_path) for card in deck.cards]
        card_texts = [card.text for card in deck.cards]

    checks = _checks(case, result, card_texts=card_texts)
    return {
        "name": case.name,
        "event_id": case.event_id,
        "before_checkpoint": case.before_checkpoint,
        "after_checkpoint": case.after_checkpoint,
        "handoff_policy": case.handoff_policy,
        "source_contract": (
            "archived event updated from replaced-pronoun anchors to the "
            "current spoken-pronoun-plus-identity-anchor contract"
            if case.update_to_pronoun_anchor_contract
            else "archived event unchanged"
        ),
        "handoff_context": handoff_context,
        "output": result.model_dump(mode="json"),
        "deck_id": deck_id,
        "card_paths": card_paths,
        "card_texts": card_texts,
        "model_calls": client.calls[call_start:],
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "error": "",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Visual-novel narrator tuning replay",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Narrator: `{report['provider']}:{report['model']}`",
        f"All checks passed: `{report['all_passed']}`",
        "",
        "## Router quoted-referent producer",
        "",
    ]
    router_probe = report["router_probe"]
    if router_probe["error"]:
        lines.extend(["```text", router_probe["error"], "```", ""])
    else:
        lines.extend(["```text", router_probe["fact_text"], "```", ""])
        for check in router_probe["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {mark}: `{check['name']}` — {check['detail']}")
        lines.append("")
    for result in report["results"]:
        lines.extend([f"## {result['name']}", ""])
        if result["error"]:
            lines.extend(["```text", result["error"], "```", ""])
            continue
        lines.extend([
            f"Handoff: `{result['output']['handoff']}`",
            f"Reason: {result['output']['handoff_reason']}",
            "",
            "```text",
            _result_text(VisualNovelNarratorOutput.model_validate(result["output"])),
            "```",
            "",
            "Checks:",
            "",
        ])
        for check in result["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {mark}: `{check['name']}` — {check['detail']}")
        if result["card_paths"]:
            lines.extend(["", "Cards:", ""])
            lines.extend(f"- `{path}`" for path in result["card_paths"])
        lines.append("")
    return "\n".join(lines)


async def main() -> int:
    load_dotenv()
    config = LLMConfig.from_env()
    provider = config.provider_for_role("narrator")
    if not config.api_key_for_provider(provider, role="narrator"):
        raise SystemExit(f"No API key for narrator provider={provider!r}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPORT_ROOT / f"vn_narrator_tuning_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    inner = LLMClient(config)
    client = _RecordingClient(inner)
    prompt_mgr = PromptManager(str(REPO_ROOT / "app/prompts"))
    results: list[dict[str, Any]] = []
    router_probe: dict[str, Any]
    try:
        print("running router_quoted_referent_producer...", flush=True)
        try:
            router_probe = await _run_router_anchor_probe(
                client=client,
                prompt_mgr=prompt_mgr,
            )
        except Exception:
            router_probe = {
                "error": traceback.format_exc(),
                "passed": False,
            }
        for case in CASES:
            print(f"running {case.name}...", flush=True)
            try:
                results.append(await _run_case(
                    case,
                    client=client,
                    prompt_mgr=prompt_mgr,
                    run_dir=run_dir,
                ))
            except Exception:
                results.append({
                    "name": case.name,
                    "error": traceback.format_exc(),
                    "passed": False,
                })
    finally:
        await inner.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": config.model_for_role("narrator"),
        "router_provider": config.provider_for_role("event_router"),
        "router_model": config.model_for_role("event_router"),
        "source_session": str(SOURCE_SESSION),
        "all_passed": router_probe["passed"] and all(
            result["passed"] for result in results
        ),
        "router_probe": router_probe,
        "results": results,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        _markdown(report) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {run_dir / 'report.json'}")
    print(f"wrote {run_dir / 'report.md'}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
