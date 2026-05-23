#!/usr/bin/env python3
"""Targeted live smoke test for D&D Cat II router-owned arbitration.

This intentionally avoids EngineBridge and persistence. Each case builds a
small in-memory checkpoint, runs `run_beat` through the production
LLMDispatcher, and captures the D&D roll plan, code-generated roll ledger,
final adjudication, compiled router event, and narrator renders.

Outputs:
  app/storage/playtest_reports/dnd_cat_ii_smoke_<timestamp>.json
  app/storage/playtest_reports/dnd_cat_ii_smoke_<timestamp>.md
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import asyncio
import json
import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine.dnd_cat_ii import DndCatIIResolver
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop import run_beat
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.dnd_cat_ii import RollPlan, RulesAdjudication
from app.schemas.state import (
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)


REPORT_DIR = REPO_ROOT / "app/storage/playtest_reports"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
JSON_PATH = REPORT_DIR / f"dnd_cat_ii_smoke_{TS}.json"
MD_PATH = REPORT_DIR / f"dnd_cat_ii_smoke_{TS}.md"
LOG_PATH = REPORT_DIR / f"dnd_cat_ii_smoke_{TS}.log"


@dataclass(frozen=True)
class SmokeCase:
    name: str
    actor_id: str
    action: str
    responder_id: str
    response: str
    expected_responder: str
    expected_roll: bool = True


@dataclass
class Capture:
    packet: str = ""
    roll_plan: dict[str, Any] | None = None
    roll_ledger: list[str] | None = None
    adjudication: dict[str, Any] | None = None


class CapturingDndCatIIResolver(DndCatIIResolver):
    def __init__(self, client: LLMClient, prompt_mgr: PromptManager):
        super().__init__(client, prompt_mgr)
        self.captures: list[Capture] = []
        self._active_capture: Capture | None = None

    async def _plan_rolls(self, packet: str) -> RollPlan:
        plan = await super()._plan_rolls(packet)
        self._active_capture = Capture(
            packet=packet,
            roll_plan=plan.model_dump(mode="json"),
        )
        return plan

    async def _finalize(
        self,
        packet: str,
        ledger_lines: list[str],
    ) -> RulesAdjudication:
        adjudication = await super()._finalize(packet, ledger_lines)
        capture = self._active_capture or Capture(packet=packet)
        capture.roll_ledger = list(ledger_lines)
        capture.adjudication = adjudication.model_dump(mode="json")
        self.captures.append(capture)
        self._active_capture = None
        return adjudication


CASES = [
    SmokeCase(
        name="shove_goblin_from_door",
        actor_id="alice",
        action=(
            "I lower my shoulder and shove Pip hard, trying to drive him "
            "away from the restricted door."
        ),
        responder_id="pip",
        response=(
            "I twist sideways, plant my heel against the doorframe, and try "
            "to slip out of Alice's line of force without giving up the door."
        ),
        expected_responder="pip",
    ),
    SmokeCase(
        name="sword_swing_at_guard",
        actor_id="alice",
        action=(
            "I draw my shortsword and swing at Pip's arm to force him to "
            "drop the key ring."
        ),
        responder_id="pip",
        response=(
            "I jerk the key ring back and duck under the cut, trying to stay "
            "between Alice and the stairs."
        ),
        expected_responder="pip",
    ),
]


def _mechanics(
    *,
    strength: int,
    dexterity: int,
    constitution: int = 10,
    proficiency_bonus: int = 2,
    skills: list[str] | None = None,
    saves: list[str] | None = None,
    armor_class: int = 10,
    hp: int = 10,
) -> dict[str, Any]:
    return {
        "ruleset_id": "dnd5e_basic",
        "level": 1,
        "proficiency_bonus": proficiency_bonus,
        "ability_scores": {
            "str": strength,
            "dex": dexterity,
            "con": constitution,
            "int": 10,
            "wis": 10,
            "cha": 10,
        },
        "skill_proficiencies": skills or [],
        "saving_throw_proficiencies": saves or [],
        "armor_class": armor_class,
        "hit_points": {"current": hp, "max": hp, "temporary": 0},
        "conditions": [],
        "resources": {},
        "raw": {},
    }


def _char(
    *,
    character_id: str,
    name: str,
    role: str,
    location: str,
    mechanics: dict[str, Any],
    playable: bool = False,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        location=location,
        is_playable=playable,
        public_sheet=PublicSheet(
            role=role,
            appearance="adventuring gear, visibly armed",
        ),
        private_state=PrivateState(
            goals=["Survive the fight and keep agency over the situation."],
            current_objectives=["Respond coherently to immediate danger."],
            secrets=[],
            intentions_enabled=False,
        ),
        known_context=(
            "This is a D&D-style scene. Physical contests should respect "
            "positioning, character capabilities, and visible intent."
        ),
        mechanics=mechanics,
    )


def _ckpt(case: SmokeCase) -> CheckpointFile:
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=f"dnd_cat_ii_smoke_{case.name}",
            story_id="dnd_cat_ii_smoke",
            player_character_id=case.actor_id,
            character_bindings={
                "alice": "user-alice",
                "pip": "user-pip",
            },
            config=SessionConfig(),
        ),
        world_state=WorldState(
            facts=[
                "Alice and Pip are within arm's reach at a narrow stone doorway.",
                "Pip is blocking the doorway and trying to keep Alice out.",
                "The scene uses D&D 5e-style physical resolution.",
            ],
            physics_ruleset=PhysicsRuleset(
                strength_limits="dnd5e_basic",
                magic_enabled=False,
            ),
            setting=StorySetting(
                genre="fantasy adventure",
                era="D&D 5e",
                tone="concrete, tactical, fair",
                premise="An adventurer contests a goblin guard at a doorway.",
            ),
            lore=(
                "The corridor is cramped. There is no grid in use, but distance "
                "and leverage matter."
            ),
        ),
        characters=[
            _char(
                character_id="alice",
                name="Alice",
                role="level 1 human fighter",
                location="stone corridor",
                playable=True,
                mechanics=_mechanics(
                    strength=16,
                    dexterity=12,
                    constitution=14,
                    skills=["athletics"],
                    saves=["str", "con"],
                    armor_class=16,
                    hp=12,
                ),
            ),
            _char(
                character_id="pip",
                name="Pip",
                role="goblin guard",
                location="stone corridor",
                playable=True,
                mechanics=_mechanics(
                    strength=8,
                    dexterity=14,
                    constitution=10,
                    skills=["acrobatics", "stealth"],
                    armor_class=15,
                    hp=7,
                ),
            ),
            _char(
                character_id="bob",
                name="Bob",
                role="torch-bearing bystander",
                location="stone corridor",
                playable=False,
                mechanics=_mechanics(
                    strength=10,
                    dexterity=10,
                    armor_class=10,
                    hp=8,
                ),
            ),
        ],
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    return ckpt


def _result_dict(result) -> dict[str, Any]:
    return {
        "ended_reason": result.ended_reason,
        "events_closed": result.events_closed,
        "renders": result.renders,
        "event_actor_ids": result.event_actor_ids,
    }


def _event_dict(event: EventRouterOutput) -> dict[str, Any]:
    dumped = event.model_dump(mode="json")
    dumped["event_kind"] = event.event_kind
    dumped["observer_ids"] = [obs.character_id for obs in event.observers]
    dumped["fact_texts"] = [
        fact.text for fact in event.canonical_event.observable_facts
    ]
    return dumped


def _transaction_dict(ckpt: CheckpointFile) -> list[dict[str, Any]]:
    return [
        transaction.model_dump(mode="json")
        for transaction in ckpt.session.cat_ii_roll_transactions
    ]


def _router_history_text(ckpt: CheckpointFile) -> str:
    parts: list[str] = []
    for message in ckpt.session_conversation:
        content = message.content
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append(json.dumps(content, sort_keys=True))
    return "\n".join(parts)


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _visible_facts_text(event: EventRouterOutput | None) -> str:
    if event is None:
        return ""
    return "\n".join(fact.text for fact in event.canonical_event.observable_facts)


def _case_checks(
    case: SmokeCase,
    ckpt: CheckpointFile,
    opening_result: Any,
    resolution_result: Any,
    capture: Capture | None,
    open_event_context: dict[str, Any],
) -> list[dict[str, Any]]:
    opening_event = ckpt.canonical_events[0] if ckpt.canonical_events else None
    resolution_event = ckpt.canonical_events[-1] if ckpt.canonical_events else None
    plan = capture.roll_plan if capture else {}
    roll_requests = plan.get("roll_requests", []) if isinstance(plan, dict) else []
    ledger = capture.roll_ledger if capture else []
    facts = _visible_facts_text(resolution_event).lower()
    mechanics_leak_terms = ["roll", "dc ", "athletics", "acrobatics", "armor class"]
    transactions = ckpt.session.cat_ii_roll_transactions
    router_history = _router_history_text(ckpt)
    planned_roll_ids = [
        str(request.get("roll_id", ""))
        for request in roll_requests
        if isinstance(request, dict)
    ]
    transaction_persisted = (
        bool(transactions)
        and transactions[-1].status == "finalized"
        and bool(transactions[-1].rolls)
        and bool(transactions[-1].ledger_lines)
        and bool(transactions[-1].final_event_id)
    )
    router_history_clean = (
        not any(
            roll_id and roll_id in router_history
            for roll_id in planned_roll_ids
        )
        and "RollPlan" not in router_history
        and "RulesAdjudication" not in router_history
        and "rolled 1d20" not in router_history
    )
    return [
        _check(
            "opening_pauses_as_cat_ii",
            opening_result.ended_reason == "cat_ii_pending",
            opening_result.ended_reason,
        ),
        _check(
            "opening_requires_expected_responder",
            opening_event is not None
            and case.expected_responder in opening_event.required_responders,
            str(opening_event.required_responders if opening_event else []),
        ),
        _check(
            "opening_context_preserved",
            bool(open_event_context.get("opening_observable_facts")),
            json.dumps(open_event_context, indent=2),
        ),
        _check(
            "dnd_cat_ii_router_captured",
            capture is not None,
        ),
        _check(
            "roll_plan_requested_rolls",
            bool(plan.get("needs_rolls")) == case.expected_roll
            and (not case.expected_roll or bool(roll_requests)),
            json.dumps(plan, indent=2) if plan else "",
        ),
        _check(
            "code_generated_ledger_present",
            bool(ledger) and any("rolled" in line and "=" in line for line in ledger),
            " | ".join(ledger or []),
        ),
        _check(
            "roll_transaction_persisted",
            transaction_persisted,
            (
                f"{len(transactions)} transaction(s), "
                f"status={transactions[-1].status if transactions else 'missing'}"
            )
            if transaction_persisted
            else json.dumps(_transaction_dict(ckpt), indent=2),
        ),
        _check(
            "router_history_omits_roll_details",
            router_history_clean,
            (
                f"{len(ckpt.session_conversation)} router history message(s) checked"
            )
            if router_history_clean
            else router_history,
        ),
        _check(
            "resolution_compiled_to_cat_ii",
            resolution_result.ended_reason == "cat_ii_resolution"
            and resolution_event is not None
            and resolution_event.event_kind == "cat_ii_resolution",
            resolution_result.ended_reason,
        ),
        _check(
            "visible_facts_present",
            bool(resolution_event and resolution_event.canonical_event.observable_facts),
        ),
        _check(
            "visible_facts_do_not_expose_mechanics",
            not any(term in facts for term in mechanics_leak_terms),
            _visible_facts_text(resolution_event),
        ),
        _check(
            "narrator_rendered_actor",
            case.actor_id in resolution_result.renders
            and bool(resolution_result.renders[case.actor_id].strip()),
        ),
    ]


async def _run_case(
    case: SmokeCase,
    dispatcher: LLMDispatcher,
    captures: CapturingDndCatIIResolver,
) -> dict[str, Any]:
    ckpt = _ckpt(case)
    before_capture_count = len(captures.captures)
    opening = await run_beat(
        ckpt=ckpt,
        dispatcher=dispatcher,
        actor_id=case.actor_id,
        intention=case.action,
    )
    open_events = list(ckpt.session.open_cat_ii_events)
    cat_ii_event_id = open_events[0].event_id if open_events else ""
    open_event_context = (
        open_events[0].model_dump(mode="json") if open_events else {}
    )
    resolution = await run_beat(
        ckpt=ckpt,
        dispatcher=dispatcher,
        actor_id=case.responder_id,
        intention=case.response,
        cat_ii_event_id=cat_ii_event_id,
    )
    capture = (
        captures.captures[-1]
        if len(captures.captures) > before_capture_count
        else None
    )
    return {
        "name": case.name,
        "action": case.action,
        "response": case.response,
        "opening": _result_dict(opening),
        "open_event_context": open_event_context,
        "resolution": _result_dict(resolution),
        "canonical_events": [_event_dict(event) for event in ckpt.canonical_events],
        "roll_transactions": _transaction_dict(ckpt),
        "capture": {
            "packet": capture.packet,
            "roll_plan": capture.roll_plan,
            "roll_ledger": capture.roll_ledger,
            "adjudication": capture.adjudication,
        } if capture else {},
        "checks": _case_checks(
            case, ckpt, opening, resolution, capture, open_event_context
        ),
        "error": "",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D&D Cat II Smoke Test",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Router: `{report['roles']['event_router']}`",
        f"D&D Cat II router: `{report['roles']['event_router']}`",
        f"Narrator: `{report['roles']['narrator']}`",
        "",
        "## Summary",
        "",
    ]
    for case in report["cases"]:
        passed = sum(1 for check in case["checks"] if check["passed"])
        total = len(case["checks"])
        status = "ERROR" if case["error"] else f"{passed}/{total}"
        lines.append(f"- `{case['name']}`: {status}")
    lines.append("")

    for case in report["cases"]:
        lines.extend([f"## {case['name']}", ""])
        if case["error"]:
            lines.extend(["```text", case["error"].strip(), "```", ""])
            continue
        lines.extend([
            f"Action: {case['action']}",
            "",
            f"Response: {case['response']}",
            "",
            "Checks:",
        ])
        for check in case["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            detail = f" - {check['detail']}" if check.get("detail") else ""
            lines.append(f"- {mark}: `{check['name']}`{detail}")
        lines.extend(["", "### Roll Plan", "```json"])
        lines.append(json.dumps(case["capture"].get("roll_plan", {}), indent=2))
        lines.extend(["```", "", "### Roll Ledger"])
        for line in case["capture"].get("roll_ledger", []) or []:
            lines.append(f"- {line}")
        lines.extend(["", "### Adjudication", "```json"])
        lines.append(json.dumps(case["capture"].get("adjudication", {}), indent=2))
        lines.extend(["```", "", "### Checkpoint Roll Transactions", "```json"])
        lines.append(json.dumps(case.get("roll_transactions", []), indent=2))
        lines.extend(["```", "", "### Canonical Events"])
        for event in case["canonical_events"]:
            lines.append(
                f"- `{event['event_kind']}` observers={event['observer_ids']}"
            )
            for fact in event["fact_texts"]:
                lines.append(f"  - {fact}")
        lines.extend(["", "### Actor Render", ""])
        lines.append(case["resolution"]["renders"].get("alice", "").strip())
        lines.append("")
    return "\n".join(lines)


def _preflight_api_keys(config: LLMConfig) -> list[str]:
    return [
        f"{item.role} ({item.provider}; set one of {', '.join(item.env_names)})"
        for item in config.missing_credentials(("event_router", "narrator"))
    ]


def _role_label(config: LLMConfig, role: str) -> str:
    return f"{config.provider_for_role(role)}:{config.model_for_role(role)}"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["all", *[case.name for case in CASES]],
        default="all",
    )
    args = parser.parse_args()

    load_dotenv()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = LLMConfig.from_env()
    missing = _preflight_api_keys(config)
    if missing:
        raise SystemExit("Missing API key(s) for: " + ", ".join(missing))

    client = LLMClient(config)
    role_calls: list[dict[str, str]] = []
    real_complete = client.complete

    async def _recording_complete(*call_args, **kwargs):
        role = kwargs.get("role") or (call_args[0] if call_args else "")
        response_model = kwargs.get("response_model")
        role_calls.append({
            "role": str(role),
            "response_model": response_model.__name__ if response_model else "",
        })
        return await real_complete(*call_args, **kwargs)

    client.complete = _recording_complete  # type: ignore[method-assign]

    prompt_mgr = PromptManager(str(REPO_ROOT / "app/prompts"))
    dispatcher = LLMDispatcher(client, prompt_mgr)
    capturing_resolver = CapturingDndCatIIResolver(client, prompt_mgr)
    dispatcher._dnd_cat_ii = capturing_resolver

    selected = CASES if args.case == "all" else [
        case for case in CASES if case.name == args.case
    ]
    case_reports: list[dict[str, Any]] = []
    try:
        for case in selected:
            print(f"running {case.name}...", flush=True)
            try:
                case_reports.append(
                    await _run_case(case, dispatcher, capturing_resolver)
                )
            except Exception:
                case_reports.append({
                    "name": case.name,
                    "action": case.action,
                    "response": case.response,
                    "opening": {},
                    "open_event_context": {},
                    "resolution": {},
                    "canonical_events": [],
                    "capture": {},
                    "checks": [],
                    "error": traceback.format_exc(),
                })
    finally:
        await client.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roles": {
            "event_router": _role_label(config, "event_router"),
            "narrator": _role_label(config, "narrator"),
        },
        "role_calls": role_calls,
        "cases": case_reports,
    }
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(_markdown(report), encoding="utf-8")

    print(JSON_PATH)
    print(MD_PATH)
    for case in case_reports:
        if case["error"]:
            print(f"{case['name']}: ERROR")
            continue
        passed = sum(1 for check in case["checks"] if check["passed"])
        total = len(case["checks"])
        print(
            f"{case['name']}: {passed}/{total} "
            f"opening={case['opening'].get('ended_reason')} "
            f"resolution={case['resolution'].get('ended_reason')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
