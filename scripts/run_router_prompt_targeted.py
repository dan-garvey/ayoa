#!/usr/bin/env python3
"""Targeted live playtest for the event router prompt.

This is intentionally narrower than the full EngineBridge playtests:
it calls the production router dispatcher directly with small synthetic
checkpoints so regressions in the prompt contract are easier to spot.

The cases focus on behavior we recently compressed or removed detail
around:

- addressed multi-recipient routing
- NPC-to-NPC continuation instead of player-centric defaulting
- defer/wait pacing
- Cat II opening and resolution
- mediated perception without scene topology
- custom player arrival guidance
- frontier-result fan-in

Outputs:
  app/storage/playtest_reports/router_prompt_targeted_<timestamp>.json
  app/storage/playtest_reports/router_prompt_targeted_<timestamp>.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import EventRouterOutput
from app.schemas.router_frontier import RouterFrontierResult
from app.schemas.state import (
    OpenCatIIEvent,
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)


REPORT_DIR = Path("app/storage/playtest_reports")
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
JSON_PATH = REPORT_DIR / f"router_prompt_targeted_{TS}.json"
MD_PATH = REPORT_DIR / f"router_prompt_targeted_{TS}.md"
LOG_PATH = REPORT_DIR / f"router_prompt_targeted_{TS}.log"

TARGETED_FACTS = [
    (
        "The venue is a live immersive show built inside an old estate: "
        "a formal dinner hall, sound-isolated dating pods, a production "
        "control room, and a security annex all matter tonight."
    ),
    (
        "Camera feeds, microphones, earpieces, and intercoms can carry "
        "some events to people who are not physically present. Audio "
        "does not imply sight; video does not imply being seen by the "
        "person on camera."
    ),
    (
        "Current arrangement for these tests: Dan, Ashara, Rashid, and "
        "Thessaly can share the dinner table when a dinner-table action "
        "is described; Dan and Britney can occupy separate pods linked "
        "only by live audio; Maya watches the production feeds from the "
        "control room; Dante can hear production talkback through an "
        "earpiece; Pip works the security annex."
    ),
    (
        "All physical conflict is ordinary human-scale unless a fact "
        "explicitly says otherwise. A responder's desired outcome is not "
        "automatically the resolved outcome."
    ),
]


@dataclass(frozen=True)
class CaseResult:
    name: str
    input_summary: str
    output: dict
    checks: list[dict[str, object]]
    error: str = ""


def _char(
    *,
    character_id: str,
    name: str,
    role: str,
    appearance: str = "",
    personality: str = "",
    goals: list[str] | None = None,
    objectives: list[str] | None = None,
    location: str = "",
    playable: bool = False,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        location=location,
        is_playable=playable,
        public_sheet=PublicSheet(role=role, appearance=appearance),
        personality=personality,
        private_state=PrivateState(
            goals=goals or [],
            current_objectives=objectives or [],
            intentions_enabled=True,
        ),
    )


def _base_characters() -> list[CharacterRecord]:
    return [
        _char(
            character_id="dan",
            name="Dan Gahvey",
            role="human player contestant and heir; currently the acting human",
            appearance="tired, observant, plainly dressed",
            location="dinner hall or pod A depending on the described action",
            playable=True,
        ),
        _char(
            character_id="ashara",
            name="Ashara Vel Kothren",
            role="sharp dinner-table rival seated near Dan",
            personality="controlled, direct, politically dangerous",
            goals=["Protect her leverage without becoming visibly rattled."],
            objectives=["Answer only when the exchange gives her advantage."],
            location="dinner hall",
        ),
        _char(
            character_id="rashid",
            name="Rashid Vel Amara",
            role="dinner-table observer with reasons to test Ashara",
            personality="cool, analytical, willing to speak past the player",
            goals=["Expose false alliances at the table."],
            objectives=["Pressure Ashara when a useful opening appears."],
            location="dinner hall",
        ),
        _char(
            character_id="thessaly",
            name="Thessaly Morrow",
            role="quiet dinner-table specialist",
            personality="careful, slow to speak, precise when addressed",
            goals=["Learn what the room is hiding."],
            objectives=["Watch for magical or social tells."],
            location="dinner hall",
        ),
        _char(
            character_id="dante",
            name="Dante Vale",
            role="show host with an earpiece and responsibility for pacing",
            personality="warm on camera, ruthless about keeping the show alive",
            goals=["Keep contestants oriented and the show moving."],
            objectives=["Give clear in-fiction prompts when the room stalls."],
            location="great hall",
        ),
        _char(
            character_id="britney",
            name="Britney Spears",
            role="late-arriving contestant in pod B with live audio only",
            appearance="glamorous, composed, camera-aware",
            personality="famous, funny, guarded, quick to puncture nonsense",
            goals=["Understand why production put her here."],
            objectives=["Find out whether Dan is a real ally or a setup."],
            location="pod B",
            playable=True,
        ),
        _char(
            character_id="maya",
            name="Maya Cross",
            role="producer in the control room with camera, audio, and talkback",
            personality="decisive, pragmatic, willing to manufacture a cue",
            goals=["Turn production chaos into playable television."],
            objectives=["Feed Dante useful direction without appearing on stage."],
            location="control room",
        ),
        _char(
            character_id="pip",
            name="Pip Arlen",
            role="volatile security worker in the annex",
            personality="jumps to force quickly but is still an ordinary human",
            goals=["Stop contestants from breaching restricted doors."],
            objectives=["Block Dan from getting past the security desk."],
            location="security annex",
        ),
    ]


def _ckpt(
    *,
    session_id: str,
    acting_player: str = "dan",
    bindings: dict[str, str] | None = None,
    pending_changes: list[str] | None = None,
) -> CheckpointFile:
    bindings = bindings if bindings is not None else {acting_player: "user-1"}
    return CheckpointFile(
        session=SessionState(
            session_id=session_id,
            story_id="router_prompt_targeted",
            player_name="Dan",
            player_character_id=acting_player,
            character_bindings=bindings,
            pending_router_state_changes=pending_changes or [],
            config=SessionConfig(),
        ),
        player_primer=(
            "You are inside a live immersive show where some people share "
            "a room and others only share a feed."
        ),
        world_state=WorldState(
            facts=list(TARGETED_FACTS),
            physics_ruleset=PhysicsRuleset(strength_limits="human_baseline"),
            setting=StorySetting(
                genre="social thriller / reality show",
                era="near future",
                tone="tense, playable, character-driven",
                premise=(
                    "Contestants, rivals, production staff, and security "
                    "create overlapping channels of perception."
                ),
            ),
            lore=(
                "The show is filmed live. Production staff can observe feeds "
                "and create prompts, but contestants only perceive what their "
                "room, microphone, earpiece, or direct line actually gives them."
            ),
            hidden_lore=(
                "Production secretly inserted Britney as a ratings emergency. "
                "Dante was not warned early enough to have a polished explanation."
            ),
        ),
        characters=_base_characters(),
    )


def _facts(result: EventRouterOutput) -> list[dict[str, object]]:
    return [
        {
            "text": fact.text,
            "audience": fact.audience,
            "visible_to": fact.visible_to,
        }
        for fact in result.canonical_event.observable_facts
    ]


def _fact_text(result: EventRouterOutput) -> str:
    return "\n".join(fact.text for fact in result.canonical_event.observable_facts)


def _observer_ids(result: EventRouterOutput) -> list[str]:
    return [obs.character_id for obs in result.observers]


def _contains_all(values: list[str], required: list[str]) -> bool:
    return all(item in values for item in required)


def _check(name: str, passed: bool, detail: str = "") -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _result_dict(result: EventRouterOutput) -> dict:
    dumped = result.model_dump(mode="json")
    dumped["fact_texts"] = _facts(result)
    dumped["observer_ids"] = _observer_ids(result)
    return dumped


async def _multi_recipient(dispatcher: LLMDispatcher) -> CaseResult:
    ckpt = _ckpt(session_id="targeted_multi_recipient")
    text = (
        'I look from Ashara to Rashid. "I am Dan. What do you both want '
        'from this room before the night is over?"'
    )
    result = await dispatcher.route_intention(
        ckpt=ckpt,
        actor_id="dan",
        intention=text,
    )
    checks = [
        _check("cat_i_dialogue", not result.requires_responders),
        _check(
            "picks_both_addressed_npcs",
            _contains_all(result.agent_responder_picks, ["ashara", "rashid"]),
            f"picks={result.agent_responder_picks}",
        ),
        _check("keeps_beat_open", result.event_kind == "beat_continues"),
        _check(
            "dialogue_preserved",
            "What do you both want from this room" in _fact_text(result),
        ),
    ]
    return CaseResult(
        name="multi_recipient_address",
        input_summary=text,
        output=_result_dict(result),
        checks=checks,
    )


async def _npc_to_npc(dispatcher: LLMDispatcher) -> CaseResult:
    ckpt = _ckpt(session_id="targeted_npc_to_npc")
    text = (
        'Rashid turns away from Dan and says to Ashara, "You do know the '
        'host is using you as cover, yes?"'
    )
    result = await dispatcher.route_intention(
        ckpt=ckpt,
        actor_id="rashid",
        intention=text,
    )
    checks = [
        _check("cat_i_dialogue", not result.requires_responders),
        _check(
            "routes_to_ashara",
            "ashara" in result.agent_responder_picks,
            f"picks={result.agent_responder_picks}",
        ),
        _check(
            "does_not_player_center_pick",
            "dan" not in result.agent_responder_picks,
            f"picks={result.agent_responder_picks}",
        ),
        _check(
            "keeps_beat_open_for_target",
            result.event_kind == "beat_continues",
            f"kind={result.event_kind}",
        ),
    ]
    return CaseResult(
        name="npc_to_npc_pressure",
        input_summary=text,
        output=_result_dict(result),
        checks=checks,
    )


async def _defer_pacing(dispatcher: LLMDispatcher) -> CaseResult:
    ckpt = _ckpt(session_id="targeted_defer_pacing")
    result = await dispatcher.route_intention(
        ckpt=ckpt,
        actor_id="dan",
        intention="(defer)",
    )
    facts = _fact_text(result).lower()
    invented_player_motion = any(
        phrase in facts
        for phrase in [
            "dan waits",
            "dan stays",
            "dan looks",
            "dan turns",
            "dan hesitates",
            "dan pauses",
            "dan stands",
            "dan sits",
            "dan remains",
        ]
    )
    non_empty_forward_motion = bool(facts.strip()) and any(
        word in facts
        for word in [
            "dante",
            "maya",
            "producer",
            "intercom",
            "door",
            "bell",
            "signal",
            "britney",
            "cue",
        ]
    )
    checks = [
        _check("does_not_invent_player_action", not invented_player_motion),
        _check(
            "no_dead_air_closed_beat",
            not (
                result.event_kind != "beat_continues"
                and result.event_kind == "ambient_pause"
                and not non_empty_forward_motion
            ),
            f"kind={result.event_kind}",
        ),
        _check(
            "open_has_pick_or_closed_has_affordance",
            (
                result.event_kind == "beat_continues"
                and bool(result.agent_responder_picks)
            )
            or (result.event_kind != "beat_continues" and non_empty_forward_motion),
            f"picks={result.agent_responder_picks}",
        ),
    ]
    return CaseResult(
        name="defer_wait_pacing",
        input_summary="(defer)",
        output=_result_dict(result),
        checks=checks,
    )


async def _defer_after_premature_boundary(dispatcher: LLMDispatcher) -> CaseResult:
    ckpt = _ckpt(session_id="targeted_defer_after_premature_boundary")
    ckpt.session_conversation.append(ConversationMessage(
        role="assistant",
        content=(
            "prior_event evt_thin_prompt @120+8 source=dan mode=intention "
            "end=directed_at_player\n"
            "fact all @0+8: Dante says, 'Dan, one name. Who do you trust "
            "here?'\n"
            "obs dan:d5 ashara:d2 rashid:d2 thessaly:d2 dante:d5 maya:d3"
        ),
    ))
    result = await dispatcher.route_intention(
        ckpt=ckpt,
        actor_id="dan",
        intention="(defer)",
    )
    facts = _fact_text(result).lower()
    invented_player_motion = any(
        phrase in facts
        for phrase in [
            "dan waits",
            "dan stays",
            "dan looks",
            "dan turns",
            "dan hesitates",
            "dan pauses",
            "dan stands",
            "dan sits",
            "dan remains",
        ]
    )
    stronger_boundary = any(
        phrase in facts
        for phrase in [
            "door opens",
            "new arrival",
            "bell",
            "producer",
            "signal",
            "message",
            "blocked path",
            "opened route",
            "britney",
        ]
    )
    thin_direct_return = (
        result.event_kind == "directed_at_player"
        and not stronger_boundary
    )
    checks = [
        _check("does_not_invent_player_action", not invented_player_motion),
        _check(
            "does_not_repeat_thin_direct_handoff",
            not thin_direct_return,
            f"kind={result.event_kind} facts={_fact_text(result)}",
        ),
        _check(
            "keeps_open_or_creates_stronger_boundary",
            (
                result.event_kind == "beat_continues"
                and bool(result.agent_responder_picks)
            ) or (
                result.event_kind in {"state_change", "ambient_pause"}
                and stronger_boundary
            ),
            (
                f"kind={result.event_kind} "
                f"picks={result.agent_responder_picks}"
            ),
        ),
    ]
    return CaseResult(
        name="defer_after_premature_boundary",
        input_summary="(defer) after a prior thin directed-at-player handoff",
        output=_result_dict(result),
        checks=checks,
    )


async def _cat_ii_open(dispatcher: LLMDispatcher) -> CaseResult:
    ckpt = _ckpt(session_id="targeted_cat_ii_open")
    text = "I shove Pip aside and try to force my way through the restricted door."
    result = await dispatcher.route_intention(
        ckpt=ckpt,
        actor_id="dan",
        intention=text,
    )
    lowered = _fact_text(result).lower()
    resolving_words = ["knocks pip aside", "pip falls", "dan gets through", "door opens"]
    checks = [
        _check("requires_responder", result.requires_responders),
        _check(
            "requires_pip",
            "pip" in result.required_responders,
            f"required={result.required_responders}",
        ),
        _check(
            "kind_cat_ii_open",
            result.event_kind == "cat_ii_open",
            f"kind={result.event_kind}",
        ),
        _check(
            "does_not_resolve_in_open",
            not any(word in lowered for word in resolving_words),
        ),
    ]
    return CaseResult(
        name="cat_ii_open_physical_contest",
        input_summary=text,
        output=_result_dict(result),
        checks=checks,
    )


async def _cat_ii_resolution(dispatcher: LLMDispatcher) -> CaseResult:
    ckpt = _ckpt(session_id="targeted_cat_ii_resolution")
    evt = OpenCatIIEvent(
        event_id="evt_targeted_cat_ii",
        initiator_id="dan",
        initiator_intention="I swing at Pip's jaw to get him away from the door.",
        required_responders=["pip"],
        collected_intentions={
            "pip": (
                "Pip slips the punch, twists Dan face-down onto the floor, "
                "and knocks him out before he can make another sound."
            ),
        },
        swept_responders=[],
    )
    result = await dispatcher.route_intention(
        ckpt=ckpt,
        actor_id="dan",
        intention="",
        cat_ii_event=evt,
    )
    lowered = _fact_text(result).lower()
    total_incapacitation_terms = [
        "knocks him out",
        "knocked out",
        "unconscious",
        "goes limp",
        "stops responding",
        "motionless",
        "lies limp",
    ]
    checks = [
        _check("resolved_no_more_required", not result.requires_responders),
        _check(
            "kind_cat_ii_resolution",
            result.event_kind == "cat_ii_resolution",
            f"kind={result.event_kind}",
        ),
        _check("terminal_event_kind", result.event_kind != "beat_continues"),
        _check(
            "does_not_blindly_accept_total_knockout",
            not any(term in lowered for term in total_incapacitation_terms),
            _fact_text(result),
        ),
    ]
    return CaseResult(
        name="cat_ii_resolution_responder_overclaim",
        input_summary="Dan swings; Pip overclaims a total knockout.",
        output=_result_dict(result),
        checks=checks,
    )


async def _mediated_pod(dispatcher: LLMDispatcher) -> CaseResult:
    ckpt = _ckpt(session_id="targeted_mediated_pod")
    text = (
        'I stay facing the wall of pod A and say into the microphone, '
        '"Britney, are you hearing this, or am I talking to the room?"'
    )
    result = await dispatcher.route_intention(
        ckpt=ckpt,
        actor_id="dan",
        intention=text,
    )
    all_observer_visual = [
        fact.text
        for fact in result.canonical_event.observable_facts
        if fact.audience == "all_observers"
        and any(term in fact.text.lower() for term in ["facing", "wall", "pod a"])
    ]
    britney_visual_leak = [
        fact.text
        for fact in result.canonical_event.observable_facts
        if fact.audience == "only"
        and "britney" in fact.visible_to
        and any(term in fact.text.lower() for term in ["facing", "wall", "pod a"])
    ]
    checks = [
        _check(
            "audio_recipient_observes",
            "britney" in _observer_ids(result),
            f"observers={_observer_ids(result)}",
        ),
        _check(
            "production_can_observe",
            "maya" in _observer_ids(result) or "dante" in _observer_ids(result),
            f"observers={_observer_ids(result)}",
        ),
        _check(
            "visual_details_not_broadcast_to_all_audio_observers",
            not all_observer_visual,
            " | ".join(all_observer_visual),
        ),
        _check(
            "visual_details_not_visible_to_audio_only_recipient",
            not britney_visual_leak,
            " | ".join(britney_visual_leak),
        ),
        _check(
            "routes_to_britney",
            "britney" in result.agent_responder_picks,
            f"picks={result.agent_responder_picks}",
        ),
    ]
    return CaseResult(
        name="mediated_pod_shared_audio_not_sight",
        input_summary=text,
        output=_result_dict(result),
        checks=checks,
    )


async def _custom_arrival(dispatcher: LLMDispatcher) -> CaseResult:
    ckpt = _ckpt(
        session_id="targeted_custom_arrival",
        acting_player="britney",
        bindings={"britney": "user-2"},
        pending_changes=[
            (
                "Britney has just appeared as a late contestant on the "
                "production roster. No one in the fiction has explained "
                "her role to Dan or Dante yet, and Dante needs an in-world "
                "cue to make the entrance playable."
            ),
        ],
    )
    result = await dispatcher.route_intention(
        ckpt=ckpt,
        actor_id="britney",
        intention="(begin)",
    )
    facts = _fact_text(result).lower()
    forbidden = [
        "state change",
        "player-bound",
        "custom character",
        "checkpoint",
        "router",
        "schema",
    ]
    checks = [
        _check(
            "creates_story_substance",
            any(term in facts for term in ["dante", "maya", "producer", "intercom", "britney"]),
            _fact_text(result),
        ),
        _check(
            "no_procedural_leakage",
            not any(term in facts for term in forbidden),
            _fact_text(result),
        ),
        _check(
            "does_not_dead_end_arrival",
            (
                result.event_kind == "beat_continues"
                and bool(result.agent_responder_picks)
            )
            or result.event_kind in {"state_change", "directed_at_player", "ambient_pause"},
            f"kind={result.event_kind} picks={result.agent_responder_picks}",
        ),
    ]
    return CaseResult(
        name="custom_arrival_story_direction",
        input_summary="Britney enters via (begin) with a pending roster note.",
        output=_result_dict(result),
        checks=checks,
    )


async def _frontier_private_talkback(dispatcher: LLMDispatcher) -> CaseResult:
    ckpt = _ckpt(session_id="targeted_frontier_private_talkback")
    result = await dispatcher.route_frontier_results(
        ckpt=ckpt,
        acting_character_id="dan",
        prior_result=EventRouterOutput.model_validate(
            {
                "event_id": "evt_prior",
                "effective_at_s": 0,
                "duration_s": 0,
                "decision_rationale": "targeted frontier fixture",
                "canonical_event": {
                    "world_adjudication": {"feasible": True},
                    "observable_facts": [],
                },
                "observers": [],
                "requires_responders": False,
                "required_responders": [],
                "agent_responder_picks": ["maya"],
                "event_kind": "beat_continues",
                "spawn": [],
                "dormant": [],
                "cull": [],
            }
        ),
        frontier_results=[
            RouterFrontierResult(
                result_kind="agent_turn",
                character_id="maya",
                frame="background",
                public_text=(
                    "Maya presses the control-room talkback and says into "
                    "Dante's earpiece, \"Bring Britney in now. Tell Dan "
                    "it is the twist.\""
                ),
                source_event_id="evt_prior",
            ),
        ],
    )
    if result is None:
        raise AssertionError(
            "route_frontier_results returned None for a non-empty frontier"
        )
    facts = _fact_text(result).lower()
    checks = [
        _check(
            "kind_private_frontier",
            result.event_kind in {"state_change", "cascade_exhausted"},
            f"kind={result.event_kind}",
        ),
        _check(
            "talkback_target_observes",
            "dante" in _observer_ids(result),
            f"observers={_observer_ids(result)}",
        ),
        _check(
            "player_not_direct_observer_of_private_talkback",
            "dan" not in _observer_ids(result) or "dante's earpiece" not in facts,
            f"observers={_observer_ids(result)} facts={_fact_text(result)}",
        ),
    ]
    return CaseResult(
        name="frontier_private_talkback",
        input_summary="Maya privately cues Dante through frontier fan-in.",
        output=_result_dict(result),
        checks=checks,
    )


CASES: list[tuple[str, Callable[[LLMDispatcher], Awaitable[CaseResult]]]] = [
    ("multi_recipient_address", _multi_recipient),
    ("npc_to_npc_pressure", _npc_to_npc),
    ("defer_wait_pacing", _defer_pacing),
    ("defer_after_premature_boundary", _defer_after_premature_boundary),
    ("cat_ii_open_physical_contest", _cat_ii_open),
    ("cat_ii_resolution_responder_overclaim", _cat_ii_resolution),
    ("mediated_pod_shared_audio_not_sight", _mediated_pod),
    ("custom_arrival_story_direction", _custom_arrival),
    ("frontier_private_talkback", _frontier_private_talkback),
]


async def _run_case(
    name: str,
    fn: Callable[[LLMDispatcher], Awaitable[CaseResult]],
    dispatcher: LLMDispatcher,
) -> CaseResult:
    try:
        return await fn(dispatcher)
    except Exception:
        return CaseResult(
            name=name,
            input_summary="",
            output={},
            checks=[],
            error=traceback.format_exc(),
        )


def _markdown(report: dict) -> str:
    lines = [
        "# Router Prompt Targeted Playtest",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Router provider: `{report['router_provider']}`",
        f"Router model: `{report['router_model']}`",
        f"Router reasoning effort: `{report['router_reasoning_effort']}`",
        "",
        "## Summary",
        "",
    ]
    for item in report["cases"]:
        passed = sum(1 for check in item["checks"] if check["passed"])
        total = len(item["checks"])
        status = "ERROR" if item["error"] else f"{passed}/{total}"
        lines.append(f"- `{item['name']}`: {status}")
    lines.append("")

    for item in report["cases"]:
        lines.extend([
            f"## {item['name']}",
            "",
            f"Input: {item['input_summary'] or '(none)'}",
            "",
        ])
        if item["error"]:
            lines.extend(["```text", item["error"].strip(), "```", ""])
            continue

        out = item["output"]
        lines.extend([
            f"event_kind=`{out.get('event_kind')}`",
            f"requires_responders=`{out.get('requires_responders')}`",
            f"required_responders=`{out.get('required_responders')}`",
            f"agent_responder_picks=`{out.get('agent_responder_picks')}`",
            f"observers=`{out.get('observer_ids')}`",
            "",
            f"Rationale: {out.get('decision_rationale', '').strip()}",
            "",
            "Checks:",
        ])
        for check in item["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            detail = f" - {check['detail']}" if check.get("detail") else ""
            lines.append(f"- {mark}: `{check['name']}`{detail}")
        lines.extend(["", "Observable facts:"])
        for fact in out.get("fact_texts", []):
            lines.append(
                f"- [{fact['audience']} {fact['visible_to']}] {fact['text']}"
            )
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    load_dotenv()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = LLMConfig.from_env()
    provider = config.provider_for_role("event_router")
    model = config.model_for_role("event_router")
    api_key = config.api_key_for_provider(provider, role="event_router")
    if not api_key:
        names = (
            config.openai_role_api_key_env_names("event_router")
            if provider == "openai"
            else ("ANTHROPIC_API_KEY",)
        )
        raise SystemExit(
            f"No API key found for event_router provider={provider!r}. "
            f"Expected one of: {', '.join(names)}"
        )

    client = LLMClient(config)
    dispatcher = LLMDispatcher(client, PromptManager("app/prompts"))
    try:
        results: list[CaseResult] = []
        for name, fn in CASES:
            print(f"running {name}...", flush=True)
            results.append(await _run_case(name, fn, dispatcher))
    finally:
        await client.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "router_provider": provider,
        "router_model": model,
        "router_reasoning_effort": config.openai_reasoning_effort_for_role(
            "event_router"
        ) if provider == "openai" else "",
        "cases": [
            {
                "name": item.name,
                "input_summary": item.input_summary,
                "output": item.output,
                "checks": item.checks,
                "error": item.error,
            }
            for item in results
        ],
    }
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(_markdown(report), encoding="utf-8")

    print(JSON_PATH)
    print(MD_PATH)
    for item in report["cases"]:
        if item["error"]:
            print(f"{item['name']}: ERROR")
        else:
            passed = sum(1 for check in item["checks"] if check["passed"])
            total = len(item["checks"])
            reason = item["output"].get("event_kind")
            picks = item["output"].get("agent_responder_picks")
            print(f"{item['name']}: {passed}/{total} kind={reason} picks={picks}")


if __name__ == "__main__":
    asyncio.run(main())
