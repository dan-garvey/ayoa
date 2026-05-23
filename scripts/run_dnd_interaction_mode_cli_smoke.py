#!/usr/bin/env python3
"""Live CLI smoke for D&D fresh-router interaction modes.

This drives the interactive CLI state object directly, with a temporary story
and session tree under app/storage/playtest_reports. It uses real LLM calls.

Cases:
- non-combat consent/contact contest: should open ordinary Cat II, not combat
- hostile attack declaration against a claimed target: should start D&D
  initiative, not open Cat II
- hostile attack declaration against an unclaimed enemy: should start D&D
  initiative and leave NPC automation in charge of the enemy
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import asyncio
import io
import json
import logging
import sys
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.bot.engine_bridge import EngineBridge
from app.llm.config import LLMConfig
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import (
    DndCombatantState,
    DndCombatState,
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)
from scripts.play import CLIState


REPORT_DIR = REPO_ROOT / "app/storage/playtest_reports"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = REPORT_DIR / f"dnd_interaction_mode_cli_smoke_{TS}"
JSON_PATH = RUN_DIR / "report.json"
MD_PATH = RUN_DIR / "report.md"
LOG_PATH = RUN_DIR / "run.log"


@dataclass(frozen=True)
class SmokeCase:
    name: str
    actor_id: str
    responder_id: str
    user_input: str
    expected: str
    join_responder: bool = True
    starts_in_combat: bool = False
    default: bool = True


CASES = [
    SmokeCase(
        name="kiss_is_cat_ii_not_combat",
        actor_id="seren",
        responder_id="lysara",
        user_input="I try to kiss Lysara.",
        expected="cat_ii",
    ),
    SmokeCase(
        name="attack_starts_combat_not_cat_ii",
        actor_id="seren",
        responder_id="ironjaw",
        user_input=(
            "I don't wait for Ironjaw to swing first; I draw my longsword "
            "and swing at Ironjaw Captain."
        ),
        expected="combat",
    ),
    SmokeCase(
        name="attack_unbound_enemy_starts_combat",
        actor_id="seren",
        responder_id="ironjaw",
        user_input=(
            "I draw my longsword and charge Ironjaw Captain before he can "
            "order the ambush."
        ),
        expected="combat",
        join_responder=False,
    ),
    SmokeCase(
        name="unavailable_spell_in_combat_probe",
        actor_id="seren",
        responder_id="ironjaw",
        user_input="I cast Hold Person on Ironjaw Captain.",
        expected="unsupported_spell_rejected",
        join_responder=True,
        starts_in_combat=True,
        default=False,
    ),
]


def _mechanics(
    *,
    strength: int,
    dexterity: int,
    constitution: int = 10,
    armor_class: int = 10,
    hp: int = 10,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "ruleset_id": "dnd5e_basic",
        "level": 1,
        "proficiency_bonus": 2,
        "ability_scores": {
            "str": strength,
            "dex": dexterity,
            "con": constitution,
            "int": 10,
            "wis": 10,
            "cha": 10,
        },
        "skill_proficiencies": ["athletics"] if strength >= 14 else [],
        "saving_throw_proficiencies": ["str", "con"],
        "armor_class": armor_class,
        "hit_points": {"current": hp, "max": hp, "temporary": 0},
        "conditions": [],
        "resources": {},
        "dnd5e_sheet": {"statblock": {"actions": actions or []}},
        "raw": {},
    }


def _char(
    *,
    character_id: str,
    name: str,
    role: str,
    appearance: str,
    mechanics: dict[str, Any],
    intentions_enabled: bool = False,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        location="broken_bridge",
        is_playable=True,
        public_sheet=PublicSheet(
            role=role,
            appearance=appearance,
        ),
        private_state=PrivateState(
            goals=["Keep agency and respond plausibly to immediate pressure."],
            current_objectives=["Handle the immediate encounter."],
            secrets=[],
            intentions_enabled=intentions_enabled,
        ),
        known_context=(
            "This is a D&D 5e scene on a cracked bridge. Characters are "
            "close enough for speech, touch, and melee attacks."
        ),
        mechanics=mechanics,
    )


def _characters_for(case: SmokeCase) -> list[CharacterRecord]:
    seren = _char(
        character_id="seren",
        name="Seren Pike",
        role="level 3 human fighter",
        appearance="mail, shield, and a longsword at the hip",
        mechanics=_mechanics(
            strength=16,
            dexterity=12,
            constitution=14,
            armor_class=18,
            hp=28,
            actions=[
                {
                    "id": "longsword",
                    "name": "Longsword",
                    "attack": {"bonus": 5, "damage": "1d8+3 slashing"},
                }
            ],
        ),
    )
    if case.responder_id == "lysara":
        return [
            seren,
            _char(
                character_id="lysara",
                name="Lysara Vale",
                role="court envoy",
                appearance="travel cloak, silver clasp, guarded posture",
                mechanics=_mechanics(
                    strength=10,
                    dexterity=14,
                    constitution=12,
                    armor_class=13,
                    hp=20,
                ),
            ),
        ]
    return [
        seren,
        _char(
            character_id="ironjaw",
            name="Ironjaw Captain",
            role="hobgoblin captain",
            appearance="splint armor, hooked blade, hard forward stance",
            intentions_enabled=True,
            mechanics=_mechanics(
                strength=16,
                dexterity=14,
                constitution=14,
                armor_class=17,
                hp=39,
                actions=[
                    {
                        "id": "hooked_blade",
                        "name": "Hooked Blade",
                        "attack": {"bonus": 5, "damage": "1d8+3 slashing"},
                    }
                ],
            ),
        ),
    ]


def _story_checkpoint(story_id: str, case: SmokeCase) -> CheckpointFile:
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=story_id,
            story_id=story_id,
            player_character_id="",
            character_bindings={},
            config=SessionConfig(),
        ),
        world_state=WorldState(
            facts=[
                "Seren Pike and the other character are within arm's reach.",
                "The encounter is tense but not in initiative until combat starts.",
                "D&D 5e rules apply when the scene becomes a fight.",
            ],
            physics_ruleset=PhysicsRuleset(
                strength_limits="dnd5e_basic",
                magic_enabled=True,
            ),
            setting=StorySetting(
                genre="fantasy adventure",
                era="D&D 5e",
                tone="concrete, tactical, fair",
                premise="A focused smoke test for D&D interaction routing.",
            ),
            lore=(
                "The bridge is cracked and exposed. Everyone can see and hear "
                "each other clearly."
            ),
        ),
        characters=_characters_for(case),
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    ckpt.session.config.settings.player_roll_mode = "auto"
    if case.starts_in_combat:
        ckpt.session.active_combat = DndCombatState(
            combat_id=f"combat_{story_id}",
            status="active",
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="seren",
                    character_id="seren",
                    name="Seren Pike",
                    player_controlled=False,
                    armor_class=18,
                    hit_points_current=28,
                    hit_points_max=28,
                    initiative_modifier=1,
                    initiative_roll=15,
                    initiative_total=16,
                    initiative_detail="preset",
                    initiative_order=1,
                ),
                DndCombatantState(
                    combatant_id="ironjaw",
                    character_id="ironjaw",
                    name="Ironjaw Captain",
                    player_controlled=False,
                    armor_class=17,
                    hit_points_current=39,
                    hit_points_max=39,
                    initiative_modifier=2,
                    initiative_roll=10,
                    initiative_total=12,
                    initiative_detail="preset",
                    initiative_order=2,
                ),
            ],
        )
    return ckpt


def _write_story(stories_dir: Path, story_id: str, case: SmokeCase) -> None:
    dst = stories_dir / story_id
    dst.mkdir(parents=True, exist_ok=True)
    ckpt = _story_checkpoint(story_id, case)
    (dst / "ckpt_0000.json").write_text(
        ckpt.model_dump_json(indent=2),
        encoding="utf-8",
    )


async def _run_cli_line(state: CLIState, line: str) -> str:
    output = io.StringIO()
    with redirect_stdout(output):
        await state.handle_line(line)
    return output.getvalue()


def _active_slot_reasons(ckpt: CheckpointFile) -> dict[str, str]:
    return {
        cid: slot.reason
        for cid, slot in (ckpt.session.active_act_slots or {}).items()
    }


def _event_facts(ckpt: CheckpointFile) -> list[str]:
    facts: list[str] = []
    for event in ckpt.canonical_events:
        facts.extend(fact.text for fact in event.canonical_event.observable_facts)
    return facts


def _transactions(ckpt: CheckpointFile) -> list[dict[str, Any]]:
    return [
        txn.model_dump(mode="json")
        for txn in ckpt.session.cat_ii_roll_transactions
    ]


def _combatant_conditions(
    ckpt: CheckpointFile,
    character_id: str,
) -> list[str]:
    combatant = _combatant_dump(ckpt, character_id)
    return list(combatant.get("conditions") or [])


def _roll_kinds(ckpt: CheckpointFile) -> list[str]:
    kinds: list[str] = []
    for txn in ckpt.session.cat_ii_roll_transactions:
        for record in txn.rolls:
            request = record.request or {}
            if isinstance(request, dict):
                kinds.append(str(request.get("kind") or ""))
    return kinds


def _case_checks(
    case: SmokeCase,
    ckpt: CheckpointFile,
    transcript: list[dict[str, str]],
    role_calls: list[dict[str, str]],
    current_actor: str | None,
) -> list[dict[str, Any]]:
    active_combat = ckpt.session.active_combat is not None
    open_cat_ii = list(ckpt.session.open_cat_ii_events or [])
    slot_reasons = _active_slot_reasons(ckpt)
    facts = _event_facts(ckpt)
    latest_cli = transcript[-1]["output"] if transcript else ""
    if "error:" in latest_cli.lower():
        return [_check("turn_completed_without_cli_error", False, latest_cli)]
    dnd_schema_used = any(
        call.get("response_model") == "DndEventRouterOutput"
        for call in role_calls
    )

    if case.expected == "cat_ii":
        checks = [
            _check("dnd_fresh_router_schema_used", dnd_schema_used, role_calls),
            _check("no_active_combat", not active_combat, _combat_dump(ckpt)),
            _check("ordinary_cat_ii_opened", bool(open_cat_ii), _open_events(open_cat_ii)),
            _check(
                "expected_responder_pinned",
                slot_reasons.get(case.responder_id) == "cat_ii_responder",
                slot_reasons,
            ),
            _check("cli_showed_pause", "beat paused" in latest_cli.lower(), latest_cli),
        ]
        if case.join_responder:
            checks.append(_check(
                "cli_switched_to_responder",
                current_actor == case.responder_id,
                {"current_actor": current_actor},
            ))
        return checks

    if case.expected == "unsupported_spell_rejected":
        facts_text = "\n".join(facts).lower()
        transactions = _transactions(ckpt)
        conditions = _combatant_conditions(ckpt, case.responder_id)
        roll_kinds = _roll_kinds(ckpt)
        combat_schema_used = any(
            call.get("response_model") in {"RollPlan", "RulesAdjudication"}
            for call in role_calls
        )
        rejected_text = any(
            phrase in facts_text
            for phrase in (
                "does not know",
                "cannot cast",
                "cannot use",
                "does not cast",
                "no spell",
                "has no spell",
                "not available",
                "unavailable",
                "fails to cast",
                "does not have",
            )
        )
        return [
            _check("combat_resolver_used", combat_schema_used, role_calls),
            _check("active_combat_remains", active_combat, _combat_dump(ckpt)),
            _check(
                "no_paralyzed_condition_applied",
                not any("paraly" in condition.lower() for condition in conditions),
                conditions,
            ),
            _check(
                "no_spell_save_roll_planned",
                "saving_throw" not in roll_kinds,
                transactions,
            ),
            _check(
                "visible_result_rejects_unavailable_spell",
                rejected_text,
                facts,
            ),
        ]

    checks = [
        _check("dnd_fresh_router_schema_used", dnd_schema_used, role_calls),
        _check("active_combat_started", active_combat, _combat_dump(ckpt)),
        _check("no_cat_ii_opened", not open_cat_ii, _open_events(open_cat_ii)),
        _check(
            "no_cat_ii_slots",
            "cat_ii_responder" not in set(slot_reasons.values()),
            slot_reasons,
        ),
        _check(
            "continuity_fact_recorded",
            any("D&D combat begins" in fact for fact in facts),
            facts,
        ),
        _check(
            "cli_showed_combat_start",
            "combat begins" in latest_cli.lower()
            and "initiating action has not resolved" in latest_cli.lower(),
            latest_cli,
        ),
    ]
    responder = _combatant_dump(ckpt, case.responder_id)
    if case.join_responder:
        checks.append(_check(
            "responder_player_controlled",
            responder.get("player_controlled") is True,
            responder,
        ))
    else:
        checks.append(_check(
            "responder_agent_controlled",
            responder.get("player_controlled") is False,
            responder,
        ))
    return checks


def _check(name: str, passed: bool, detail: Any = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _combat_dump(ckpt: CheckpointFile) -> dict[str, Any]:
    combat = ckpt.session.active_combat
    return combat.model_dump(mode="json") if combat is not None else {}


def _combatant_dump(ckpt: CheckpointFile, character_id: str) -> dict[str, Any]:
    combat = ckpt.session.active_combat
    if combat is None:
        return {}
    for combatant in combat.combatants:
        if combatant.character_id == character_id:
            return combatant.model_dump(mode="json")
    return {}


def _open_events(events: list[Any]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json") for event in events]


async def _run_case(
    case: SmokeCase,
    *,
    engine: EngineBridge,
    stories_dir: Path,
    role_calls: list[dict[str, str]],
    role_call_start: int,
) -> dict[str, Any]:
    story_id = f"dnd_interaction_{case.name}"
    session_id = f"{story_id}_{TS.lower()}"
    _write_story(stories_dir, story_id, case)
    engine.create_empty_session(session_id)
    state = CLIState(engine, session_id, "")

    lines = [
        f"/story start {story_id}",
        f"/join {case.actor_id}",
    ]
    if case.join_responder:
        lines.append(f"/join {case.responder_id}")
    lines.extend([
        f"/as {case.actor_id}",
        case.user_input,
    ])

    transcript: list[dict[str, str]] = []
    for line in lines:
        transcript.append({
            "input": line,
            "output": await _run_cli_line(state, line),
        })

    ckpt = engine.load_latest(session_id)
    calls_for_case = list(role_calls[role_call_start:])
    checks = _case_checks(
        case,
        ckpt,
        transcript,
        calls_for_case,
        state.current_actor,
    )
    return {
        "name": case.name,
        "session_id": session_id,
        "story_id": story_id,
        "input": case.user_input,
        "expected": case.expected,
        "join_responder": case.join_responder,
        "current_actor": state.current_actor,
        "turn_index": ckpt.session.turn_index,
        "transcript": transcript,
        "role_calls": calls_for_case,
        "active_combat": _combat_dump(ckpt),
        "open_cat_ii_events": _open_events(list(ckpt.session.open_cat_ii_events)),
        "active_slot_reasons": _active_slot_reasons(ckpt),
        "canonical_facts": _event_facts(ckpt),
        "checks": checks,
        "error": "",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# D&D Interaction Mode CLI Smoke",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Router: `{report['roles']['event_router']}`",
        f"Narrator: `{report['roles']['narrator']}`",
        "",
        "## Summary",
        "",
    ]
    for case in report["cases"]:
        if case.get("error"):
            status = "ERROR"
        else:
            passed = sum(1 for check in case["checks"] if check["passed"])
            status = f"{passed}/{len(case['checks'])}"
        lines.append(f"- `{case['name']}`: {status}")
    lines.append("")

    for case in report["cases"]:
        lines.extend([f"## {case['name']}", ""])
        if case.get("error"):
            lines.extend(["```text", case["error"], "```", ""])
            continue
        lines.append(f"Input: `{case['input']}`")
        lines.append("")
        lines.append("Checks:")
        for check in case["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {mark}: `{check['name']}`")
        lines.extend(["", "### CLI Transcript", "```text"])
        for item in case["transcript"]:
            lines.append(f"> {item['input']}")
            lines.append(item["output"].rstrip())
        lines.extend(["```", "", "### Canonical Facts"])
        for fact in case["canonical_facts"]:
            lines.append(f"- {fact}")
        lines.extend(["", "### Active Combat", "```json"])
        lines.append(json.dumps(case["active_combat"], indent=2))
        lines.extend(["```", "", "### Open Cat II", "```json"])
        lines.append(json.dumps(case["open_cat_ii_events"], indent=2))
        lines.extend(["```", ""])
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
    choices = ["all", "all_with_probes", *[case.name for case in CASES]]
    parser.add_argument(
        "--case",
        choices=choices,
        default="all",
    )
    args = parser.parse_args()

    load_dotenv()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = LLMConfig.from_env()
    missing = _preflight_api_keys(config)
    if missing:
        raise SystemExit("Missing API key(s) for: " + ", ".join(missing))

    stories_dir = RUN_DIR / "stories"
    sessions_dir = RUN_DIR / "sessions"
    engine = EngineBridge(
        stories_dir=str(stories_dir),
        sessions_dir=str(sessions_dir),
        prompts_dir=str(REPO_ROOT / "app/prompts"),
        llm_config=config,
    )
    role_calls: list[dict[str, str]] = []
    real_complete = engine.client.complete

    async def _recording_complete(*call_args, **kwargs):
        role = kwargs.get("role") or (call_args[0] if call_args else "")
        response_model = kwargs.get("response_model")
        role_calls.append({
            "role": str(role),
            "response_model": response_model.__name__ if response_model else "",
        })
        return await real_complete(*call_args, **kwargs)

    engine.client.complete = _recording_complete  # type: ignore[method-assign]

    if args.case == "all":
        selected = [case for case in CASES if case.default]
    elif args.case == "all_with_probes":
        selected = CASES
    else:
        selected = [case for case in CASES if case.name == args.case]
    case_reports: list[dict[str, Any]] = []
    try:
        for case in selected:
            print(f"running {case.name}...", flush=True)
            before = len(role_calls)
            try:
                case_reports.append(await _run_case(
                    case,
                    engine=engine,
                    stories_dir=stories_dir,
                    role_calls=role_calls,
                    role_call_start=before,
                ))
            except Exception:
                case_reports.append({
                    "name": case.name,
                    "input": case.user_input,
                    "expected": case.expected,
                    "checks": [],
                    "error": traceback.format_exc(),
                })
    finally:
        await engine.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roles": {
            "event_router": _role_label(config, "event_router"),
            "narrator": _role_label(config, "narrator"),
        },
        "run_dir": str(RUN_DIR),
        "cases": case_reports,
    }
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(_markdown(report), encoding="utf-8")

    print(JSON_PATH)
    print(MD_PATH)
    failed = False
    for case in case_reports:
        if case.get("error"):
            print(f"{case['name']}: ERROR")
            failed = True
            continue
        passed = sum(1 for check in case["checks"] if check["passed"])
        total = len(case["checks"])
        print(f"{case['name']}: {passed}/{total}")
        if passed != total:
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
