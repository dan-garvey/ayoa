#!/usr/bin/env python3
"""Live acceptance harness for the batched event-router contract.

The cases call ``StoryDispatcher.route_batch`` directly. Saved artifacts include
the exact provider messages, raw response, parsed batch, materialized canonical
events, and behavioral checks needed to compare prompt revisions.

Outputs:
  app/storage/playtest_reports/router_prompt_targeted_<timestamp>.json
  app/storage/playtest_reports/router_prompt_targeted_<timestamp>.md
  app/storage/playtest_reports/router_prompt_targeted_<timestamp>.log
"""

from __future__ import annotations

# ruff: noqa: E402 -- executable scripts add the repository root to sys.path.

import argparse
import asyncio
import json
import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine.prompt_manager import PromptManager
from app.engine.story_dispatcher import StoryDispatcher, append_router_history
from app.llm.client import LLMClient, LLMResponse
from app.llm.config import LLMConfig
from app.schemas.characters import (
    ActorFact,
    ActorRecord,
    CharacterRecord,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import (
    CanonicalEventRecord,
    ObserverGroups,
    RouterBatchOutput,
    RouterInputEnvelope,
)
from app.schemas.events import ObservableFact
from app.schemas.state import (
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)


REPORT_DIR = REPO_ROOT / "app/storage/playtest_reports"
BASELINE_PATH = REPORT_DIR / "router_prompt_targeted_20260901T163651Z.json"
TARGETED_FACTS = (
    "The venue is a live immersive show inside an old estate. Its dinner hall, "
    "sound-isolated dating pods, production control room, and security annex "
    "can host simultaneous activity.",
    "Camera feeds, microphones, earpieces, and intercoms can carry some events "
    "to people who are not physically present. Audio does not imply sight; "
    "video does not imply being seen by the person on camera.",
    "Dan, Ashara, and Rashid are at the dinner table. Dan and Britney can speak "
    "between separate pods through live audio. Maya monitors production feeds "
    "from the control room. Dante hears Maya through an earpiece. Pip works in "
    "the security annex.",
    "All physical conflict is ordinary human-scale. A proposed outcome does not "
    "become true merely because its actor wants it.",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    summary: str
    build: Callable[[], tuple[CheckpointFile, list[RouterInputEnvelope]]]
    evaluate: Callable[[RouterBatchOutput, Any], list[dict[str, Any]]]


class RecordingClient:
    """Capture exact router calls without changing provider behavior."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.config = inner.config
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
        response = await self.inner.complete(*args, **kwargs)
        if kwargs.get("role") == "event_router":
            raw_response = response.raw_response
            if hasattr(raw_response, "model_dump"):
                raw_response = raw_response.model_dump(mode="json")
            elif raw_response is not None:
                raw_response = repr(raw_response)
            self.calls.append({
                "messages": kwargs.get("messages", []),
                "response_model": getattr(
                    kwargs.get("response_model"),
                    "__name__",
                    "",
                ),
                "model": response.model,
                "usage": dict(response.usage or {}),
                "raw_content": response.content,
                "parsed": (
                    response.parsed.model_dump(mode="json")
                    if hasattr(response.parsed, "model_dump")
                    else response.parsed
                ),
                "reasoning_summaries": list(response.reasoning_summaries or ()),
                "raw_response": raw_response,
            })
        return response


def _char(
    character_id: str,
    name: str,
    role: str,
    *,
    location: str,
    facts: tuple[str, ...] = (),
    playable: bool = False,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        location=location,
        is_playable=playable,
        public_sheet=PublicSheet(role=role),
        actor=ActorRecord(
            may_act_offstage=True,
            facts=[ActorFact(text=fact) for fact in facts],
        ),
    )


def _characters() -> list[CharacterRecord]:
    return [
        _char(
            "dan",
            "Dan Gahvey",
            "contestant and heir",
            location="dinner hall",
            playable=True,
        ),
        _char(
            "ashara",
            "Ashara Vel Kothren",
            "controlled dinner-table rival",
            location="dinner hall",
            facts=(
                "You are direct and protective of your leverage.",
                "You answer when an exchange gives you an advantage.",
            ),
        ),
        _char(
            "rashid",
            "Rashid Vel Amara",
            "analytical dinner-table rival",
            location="dinner hall",
            facts=(
                "You pressure Ashara when a useful opening appears.",
                "You want to expose false alliances.",
            ),
        ),
        _char(
            "britney",
            "Britney Spears",
            "late-arriving contestant",
            location="pod B",
            facts=(
                "You are funny, guarded, and quick to puncture nonsense.",
                "You want to learn why production brought you here.",
            ),
            playable=True,
        ),
        _char(
            "maya",
            "Maya Cross",
            "producer with camera, audio, and talkback access",
            location="control room",
            facts=(
                "You create practical cues when the production stalls.",
                "You feed Dante useful direction without appearing on stage.",
            ),
        ),
        _char(
            "dante",
            "Dante Vale",
            "show host responsible for pacing",
            location="great hall",
            facts=(
                "You are warm on camera and ruthless about keeping things moving.",
            ),
        ),
        _char(
            "pip",
            "Pip Arlen",
            "volatile security worker",
            location="security annex",
            facts=(
                "You guard the emergency keys and block unauthorized access.",
            ),
        ),
    ]


def _checkpoint(
    session_id: str,
    *,
    player_id: str = "dan",
    bindings: dict[str, str] | None = None,
) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id=session_id,
            story_id="router_prompt_targeted",
            player_name="Player",
            player_character_id=player_id,
            character_bindings=(
                dict(bindings) if bindings is not None else {player_id: "user-1"}
            ),
            config=SessionConfig(),
        ),
        player_primer=(
            "You are inside a live immersive show whose rooms have deliberately "
            "uneven channels of perception."
        ),
        world_state=WorldState(
            facts=list(TARGETED_FACTS),
            physics_ruleset=PhysicsRuleset(strength_limits="human_baseline"),
            setting=StorySetting(
                genre="social thriller / reality show",
                era="near future",
                tone="tense, playable, character-driven",
                premise=(
                    "Contestants, rivals, production staff, and security create "
                    "overlapping channels of pressure and perception."
                ),
            ),
            lore=(
                "The show is filmed live. Production staff can observe feeds and "
                "create cues, but contestants perceive only their room or an "
                "established microphone, earpiece, intercom, or camera feed."
            ),
            hidden_lore=(
                "Production inserted Britney as a ratings emergency. Dante was "
                "not warned early enough to have a polished explanation."
            ),
        ),
        characters=_characters(),
    )


def _input(
    *,
    index: int,
    lane: str,
    kind: str,
    actor_ids: list[str],
    participant_ids: list[str],
    payload: str,
    source_event_ids: list[str] | None = None,
    chosen_at_s: int = 0,
    observed_sequence: int = -1,
) -> RouterInputEnvelope:
    return RouterInputEnvelope(
        submission_id=f"submission_{lane}_{index}",
        input_index=index,
        lane_id=lane,
        kind=kind,
        actor_ids=actor_ids,
        participant_ids=participant_ids,
        source_event_ids=list(source_event_ids or ()),
        chosen_at_s=chosen_at_s,
        observed_through_event_sequence=observed_sequence,
        observed_through_s=chosen_at_s,
        payload=payload,
    )


def _check(name: str, passed: bool, detail: Any = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _fact_text(output: RouterBatchOutput) -> str:
    return "\n".join(
        fact.text for event in output.events for fact in event.observable_facts
    )


def _events_for_input(output: RouterBatchOutput, index: int) -> list[Any]:
    return [
        event
        for event in output.events
        if index in event.feasible_input_indexes
        or index in event.infeasible_input_indexes
    ]


def _next_actors(output: RouterBatchOutput, *, source: int | None = None) -> list[str]:
    return [
        turn.actor_id
        for turn in output.next_turns
        if turn.turn_kind == "character"
        and (source is None or turn.source_event_index == source)
    ]


def _accounting_checks(
    output: RouterBatchOutput,
    count: int,
) -> list[dict[str, Any]]:
    accounted = [
        index
        for event in output.events
        for index in (
            *event.feasible_input_indexes,
            *event.infeasible_input_indexes,
        )
    ]
    return [
        _check(
            "exact_input_accounting",
            sorted(accounted) == list(range(count)),
            accounted,
        ),
        _check(
            "batch_bounds",
            len(output.events) <= 5 and len(output.next_turns) <= 5,
        ),
    ]


def _evaluate_multi(output: RouterBatchOutput, _batch: Any) -> list[dict[str, Any]]:
    event = _events_for_input(output, 0)[0]
    event_index = output.events.index(event)
    source_actors = _next_actors(output, source=event_index)
    addressed = set(source_actors) & {"ashara", "rashid"}
    text = _fact_text(output)
    return [
        *_accounting_checks(output, 1),
        _check("dialogue_preserved", "What do you both want" in text, text),
        _check(
            "one_same_lane_reply_selected",
            len(addressed) == 1,
            sorted(addressed),
        ),
        _check("no_player_centering", "dan" not in source_actors, source_actors),
        _check(
            "dialogue_does_not_request_appearance",
            not event.appearance_target_ids,
            event.appearance_target_ids,
        ),
    ]


def _evaluate_npc_pressure(
    output: RouterBatchOutput,
    _batch: Any,
) -> list[dict[str, Any]]:
    event = _events_for_input(output, 0)[0]
    actors = _next_actors(output, source=output.events.index(event))
    return [
        *_accounting_checks(output, 1),
        _check("routes_to_addressed_npc", actors == ["ashara"], actors),
        _check("does_not_route_to_player", "dan" not in actors, actors),
    ]


def _evaluate_defer(output: RouterBatchOutput, _batch: Any) -> list[dict[str, Any]]:
    player_facts = _fact_text(output).casefold()
    independent = [
        turn.actor_id
        for turn in output.next_turns
        if turn.turn_kind == "character" and turn.source_event_index == -1
    ]
    return [
        *_accounting_checks(output, 1),
        _check(
            "defer_invents_no_player_behavior",
            not any(
                phrase in player_facts
                for phrase in (
                    "dan waits",
                    "dan hesitates",
                    "dan looks",
                    "dan remains",
                )
            ),
            player_facts,
        ),
        _check("off_lane_work_selected", bool(independent), independent),
        _check("player_not_automated", "dan" not in independent, independent),
    ]


def _evaluate_cat_open(output: RouterBatchOutput, _batch: Any) -> list[dict[str, Any]]:
    event = _events_for_input(output, 0)[0]
    event_index = output.events.index(event)
    return [
        *_accounting_checks(output, 1),
        _check(
            "opens_contest_for_pip",
            event.required_responders == ["pip"],
            event.required_responders,
        ),
        _check("opening_is_instantaneous", event.duration_s == 0, event.duration_s),
        _check(
            "responder_has_direct_access",
            "pip" in event.observers.direct,
            event.observers.model_dump(),
        ),
        _check(
            "opening_does_not_also_route",
            not any(
                turn.source_event_index == event_index
                for turn in output.next_turns
            ),
            [turn.model_dump() for turn in output.next_turns],
        ),
    ]


def _evaluate_same_lane_contest(
    output: RouterBatchOutput,
    batch: Any,
) -> list[dict[str, Any]]:
    shared = [
        event
        for event in output.events
        if 0 in event.feasible_input_indexes
        and 1 in event.feasible_input_indexes
    ]
    event = shared[0] if shared else None
    text = _fact_text(output)
    return [
        *_accounting_checks(output, 2),
        _check("same_lane_inputs_share_one_event", len(shared) == 1),
        _check(
            "opening_proposal_is_first_feasible_input",
            event is not None and event.feasible_input_indexes[0] == 0,
            event.feasible_input_indexes if event is not None else [],
        ),
        _check(
            "contest_opens_for_pip",
            event is not None and event.required_responders == ["pip"],
            event.required_responders if event is not None else [],
        ),
        _check(
            "simultaneous_dialogue_is_preserved",
            "Don't touch it" in text,
            text,
        ),
        _check("materializes_one_causal_event", len(batch.events) == 1),
    ]


def _evaluate_cat_resolution(
    output: RouterBatchOutput,
    _batch: Any,
) -> list[dict[str, Any]]:
    event = _events_for_input(output, 0)[0]
    text = "\n".join(fact.text for fact in event.observable_facts)
    return [
        *_accounting_checks(output, 1),
        _check("contest_closes", not event.required_responders, event.required_responders),
        _check("outcome_is_visible", bool(text.strip()), text),
        _check(
            "opening_marker_not_replayed",
            "has not resolved" not in text.casefold(),
            text,
        ),
    ]


def _evaluate_mediated(
    output: RouterBatchOutput,
    _batch: Any,
) -> list[dict[str, Any]]:
    event = _events_for_input(output, 0)[0]
    britney_facts = [
        fact for fact in event.observable_facts if fact.is_visible_to("britney")
    ]
    sight_leaks = [
        fact.model_dump(mode="json")
        for fact in britney_facts
        if fact.visual_subject_ids
    ]
    return [
        *_accounting_checks(output, 1),
        _check(
            "audio_reaches_britney",
            bool(britney_facts),
            [fact.text for fact in britney_facts],
        ),
        _check("audio_does_not_grant_sight", not sight_leaks, sight_leaks),
        _check(
            "britney_can_answer",
            "britney" in _next_actors(output),
            _next_actors(output),
        ),
    ]


def _evaluate_arrival(output: RouterBatchOutput, _batch: Any) -> list[dict[str, Any]]:
    text = _fact_text(output)
    next_actors = _next_actors(output)
    forbidden = ("router", "schema", "dispatcher", "api", "engine")
    return [
        *_accounting_checks(output, 1),
        _check(
            "arrival_materializes",
            any(not event.is_no_event_resolution for event in output.events),
        ),
        _check(
            "no_procedural_leakage",
            not any(term in text.casefold() for term in forbidden),
            text,
        ),
        _check(
            "arrival_does_not_speak_for_player",
            '"' not in text and "“" not in text,
            text,
        ),
        _check(
            "already_active_arrival_is_not_activated",
            not any(event.activate for event in output.events),
            [
                signal.model_dump(mode="json")
                for event in output.events
                for signal in event.activate
            ],
        ),
        _check("story_can_continue", bool(next_actors), next_actors),
    ]


def _evaluate_fan_in(output: RouterBatchOutput, batch: Any) -> list[dict[str, Any]]:
    first = _events_for_input(output, 0)
    second = _events_for_input(output, 1)
    first_observers = set(first[0].observers.all_ids) if first else set()
    second_observers = set(second[0].observers.all_ids) if second else set()
    return [
        *_accounting_checks(output, 2),
        _check(
            "independent_inputs_stay_sibling_events",
            len(first) == len(second) == 1 and first[0] is not second[0],
        ),
        _check(
            "security_does_not_leak_to_dinner",
            "pip" not in first_observers,
            sorted(first_observers),
        ),
        _check(
            "dinner_does_not_leak_to_security",
            not {"dan", "ashara"} & second_observers,
            sorted(second_observers),
        ),
        _check(
            "materializes_two_causal_lanes",
            len({event.record.causal_lane_id for event in batch.events}) == 2,
        ),
    ]


def _simple_case(
    name: str,
    *,
    kind: str,
    actor_ids: list[str],
    participant_ids: list[str],
    payload: str,
) -> tuple[CheckpointFile, list[RouterInputEnvelope]]:
    checkpoint = _checkpoint(name)
    return checkpoint, [
        _input(
            index=0,
            lane=f"lane_{name}",
            kind=kind,
            actor_ids=actor_ids,
            participant_ids=participant_ids,
            payload=payload,
        )
    ]


def _build_resolution() -> tuple[CheckpointFile, list[RouterInputEnvelope]]:
    checkpoint = _checkpoint("cat_ii_resolution")
    opening = CanonicalEventRecord(
        event_id="evt_unresolved_badge_grab",
        causal_lane_id="lane_security_contest",
        effective_at_s=12,
        duration_s=0,
        actor_ids=["dan"],
        source_submission_ids=["submission_opening"],
        feasible_submission_ids=["submission_opening"],
        infeasible_submission_ids=[],
        observable_facts=[
            ObservableFact.all(
                "Dan reaches toward the badge on Pip's belt; the grab has not "
                "resolved.",
                visual_subject_ids=["dan", "pip"],
            )
        ],
        observers=ObserverGroups(
            direct=["dan", "pip"],
            indirect=[],
            inferred=[],
        ),
        spawn=[],
        dormant=[],
        cull=[],
        commitment_opens=[],
        commitment_resolutions=[],
        commitment_interrupts=[],
        location_updates=[],
        activate=[],
    )
    checkpoint.canonical_events.append(opening)
    append_router_history(checkpoint, [opening])
    payload = json.dumps({
        "opening_event_id": opening.event_id,
        "initiator": {
            "character_id": "dan",
            "intention": "I snatch the security badge from Pip's belt.",
        },
        "responses": [
            {
                "character_id": "pip",
                "intention": (
                    "I slap his hand away and step between him and the desk."
                ),
            }
        ],
    })
    return checkpoint, [
        _input(
            index=0,
            lane=opening.causal_lane_id,
            kind="cat_ii_resolution",
            actor_ids=["dan", "pip"],
            participant_ids=["dan", "pip"],
            payload=payload,
            source_event_ids=[opening.event_id],
            chosen_at_s=12,
            observed_sequence=0,
        )
    ]


def _build_cat_open() -> tuple[CheckpointFile, list[RouterInputEnvelope]]:
    checkpoint = _checkpoint("cat_ii_open_physical_contest")
    dan = next(item for item in checkpoint.characters if item.character_id == "dan")
    dan.location = "security annex"
    checkpoint.world_state.facts.append(
        "For this case, Dan and Pip are physically together at the security desk."
    )
    return checkpoint, [
        _input(
            index=0,
            lane="lane_security_contest",
            kind="player",
            actor_ids=["dan"],
            participant_ids=["dan", "pip"],
            payload="I snatch the security badge from Pip's belt.",
        )
    ]


def _build_same_lane_contest() -> tuple[CheckpointFile, list[RouterInputEnvelope]]:
    checkpoint = _checkpoint("same_lane_contest_merge")
    for character in checkpoint.characters:
        if character.character_id in {"dan", "pip", "ashara"}:
            character.location = "security annex"
    checkpoint.world_state.facts.append(
        "Dan, Pip, and Ashara are together beside the security desk."
    )
    return checkpoint, [
        _input(
            index=0,
            lane="lane_security_contest",
            kind="player",
            actor_ids=["dan"],
            participant_ids=["dan", "pip"],
            payload="I snatch the security badge from Pip's belt.",
        ),
        _input(
            index=1,
            lane="lane_security_contest",
            kind="character",
            actor_ids=["ashara"],
            participant_ids=["ashara", "dan", "pip"],
            payload='Ashara tells Dan, "Don\'t touch it."',
        ),
    ]


def _build_arrival() -> tuple[CheckpointFile, list[RouterInputEnvelope]]:
    checkpoint = _checkpoint(
        "custom_arrival",
        player_id="britney",
        bindings={"britney": "user-2"},
    )
    britney = next(
        item for item in checkpoint.characters if item.character_id == "britney"
    )
    britney.location = ""
    britney.public_sheet.appearance = "glamorous, composed, and camera-aware"
    return checkpoint, [
        _input(
            index=0,
            lane="lane_britney_arrival",
            kind="player",
            actor_ids=["britney"],
            participant_ids=["britney"],
            payload="(arrive)",
        )
    ]


def _build_fan_in() -> tuple[CheckpointFile, list[RouterInputEnvelope]]:
    checkpoint = _checkpoint("offstage_fan_in")
    return checkpoint, [
        _input(
            index=0,
            lane="lane_dinner",
            kind="player",
            actor_ids=["dan"],
            participant_ids=["dan", "ashara"],
            payload='I ask Ashara, "Who warned you about the host?"',
            chosen_at_s=20,
        ),
        _input(
            index=1,
            lane="lane_security",
            kind="character",
            actor_ids=["pip"],
            participant_ids=["pip"],
            payload="I inventory the emergency keys and lock the red case.",
            chosen_at_s=20,
        ),
    ]


CASES = (
    CaseSpec(
        "multi_recipient_address",
        "One addressed exchange selects one strongest same-lane respondent.",
        lambda: _simple_case(
            "multi_recipient_address",
            kind="player",
            actor_ids=["dan"],
            participant_ids=["dan", "ashara", "rashid"],
            payload=(
                'I look from Ashara to Rashid. "I am Dan. What do you both want '
                'from this room before the night is over?"'
            ),
        ),
        _evaluate_multi,
    ),
    CaseSpec(
        "npc_to_npc_pressure",
        "NPC dialogue routes to its target rather than centering the player.",
        lambda: _simple_case(
            "npc_to_npc_pressure",
            kind="character",
            actor_ids=["rashid"],
            participant_ids=["rashid", "ashara"],
            payload=(
                'Rashid turns to Ashara. "You know the host is using you as '
                'cover, yes?"'
            ),
        ),
        _evaluate_npc_pressure,
    ),
    CaseSpec(
        "defer_routes_off_lane",
        "A defer invents no action and gives independent story work a turn.",
        lambda: _simple_case(
            "defer_routes_off_lane",
            kind="player",
            actor_ids=["dan"],
            participant_ids=["dan"],
            payload="(defer)",
        ),
        _evaluate_defer,
    ),
    CaseSpec(
        "cat_ii_open_physical_contest",
        "A physical contest opens an obligation without resolving success.",
        _build_cat_open,
        _evaluate_cat_open,
    ),
    CaseSpec(
        "same_lane_contest_merge",
        "A contest and simultaneous same-lane speech remain one event.",
        _build_same_lane_contest,
        _evaluate_same_lane_contest,
    ),
    CaseSpec(
        "cat_ii_resolution",
        "Collected intentions close once without replaying the opening as outcome.",
        _build_resolution,
        _evaluate_cat_resolution,
    ),
    CaseSpec(
        "mediated_audio_not_sight",
        "Established audio supports reply without granting visual access.",
        lambda: _simple_case(
            "mediated_audio_not_sight",
            kind="player",
            actor_ids=["dan"],
            participant_ids=["dan", "britney", "maya"],
            payload=(
                'Through the pod microphone I tell Britney, "Production says '
                'you asked for me. Is that true?"'
            ),
        ),
        _evaluate_mediated,
    ),
    CaseSpec(
        "custom_arrival",
        "A player arrival receives neutral authored placement and story pressure.",
        _build_arrival,
        _evaluate_arrival,
    ),
    CaseSpec(
        "offstage_batch_fan_in",
        "Two disjoint proposals resolve in one router call as sibling lanes.",
        _build_fan_in,
        _evaluate_fan_in,
    ),
)


async def _run_case(
    spec: CaseSpec,
    dispatcher: StoryDispatcher,
    client: RecordingClient,
) -> dict[str, Any]:
    checkpoint, inputs = spec.build()
    call_start = len(client.calls)
    try:
        batch = await dispatcher.route_batch(ckpt=checkpoint, inputs=inputs)
        call = client.calls[-1]
        output = RouterBatchOutput.model_validate(call["parsed"])
        checks = spec.evaluate(output, batch)
        return {
            "name": spec.name,
            "summary": spec.summary,
            "inputs": [item.model_dump(mode="json") for item in inputs],
            "router_output": output.model_dump(mode="json"),
            "materialized": {
                "events": [
                    item.record.model_dump(mode="json") for item in batch.events
                ],
                "next_turns": [
                    item.model_dump(mode="json") for item in batch.next_turns
                ],
                "feasible_submission_ids": list(batch.feasible_submission_ids),
                "infeasible_submission_ids": list(batch.infeasible_submission_ids),
            },
            "provider_calls": client.calls[call_start:],
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
            "error": "",
        }
    except Exception:
        return {
            "name": spec.name,
            "summary": spec.summary,
            "inputs": [item.model_dump(mode="json") for item in inputs],
            "router_output": {},
            "materialized": {},
            "provider_calls": client.calls[call_start:],
            "checks": [],
            "passed": False,
            "error": traceback.format_exc(),
        }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Batched Router Targeted Playtest",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Router: `{report['provider']}:{report['model']}`",
        f"Reasoning effort: `{report['reasoning_effort'] or 'provider default'}`",
        f"Comparison baseline: `{report['comparison_baseline']}`",
        f"All checks passed: `{report['all_passed']}`",
        "",
        "## Summary",
        "",
    ]
    for case in report["cases"]:
        total = len(case["checks"])
        passed = sum(check["passed"] for check in case["checks"])
        status = "ERROR" if case["error"] else f"{passed}/{total}"
        lines.append(f"- `{case['name']}`: {status}")
    for case in report["cases"]:
        lines.extend(["", f"## {case['name']}", "", case["summary"], ""])
        if case["error"]:
            lines.extend(["```text", case["error"].strip(), "```"])
            continue
        for check in case["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            detail = f" - {check['detail']}" if check["detail"] != "" else ""
            lines.append(f"- {mark}: `{check['name']}`{detail}")
        lines.extend([
            "",
            "```json",
            json.dumps(case["router_output"], indent=2, ensure_ascii=False),
            "```",
        ])
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASES],
        help="Run only this named case; repeat to select several.",
    )
    return parser.parse_args()


async def main() -> int:
    args = _parse_args()
    load_dotenv(REPO_ROOT / ".env")
    config = LLMConfig.from_env()
    provider = config.provider_for_role("event_router")
    if not config.api_key_for_provider(provider, role="event_router"):
        raise SystemExit(f"No API key for event_router provider={provider!r}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORT_DIR / f"router_prompt_targeted_{timestamp}.json"
    markdown_path = REPORT_DIR / f"router_prompt_targeted_{timestamp}.md"
    log_path = REPORT_DIR / f"router_prompt_targeted_{timestamp}.log"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    selected = [case for case in CASES if not args.case or case.name in args.case]
    inner = LLMClient(config)
    client = RecordingClient(inner)
    dispatcher = StoryDispatcher(
        client,
        PromptManager(str(REPO_ROOT / "app/prompts")),
    )
    try:
        results = []
        for case in selected:
            print(f"running {case.name}...", flush=True)
            results.append(await _run_case(case, dispatcher, client))
    finally:
        await inner.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": config.model_for_role("event_router"),
        "reasoning_effort": (
            config.openai_reasoning_effort_for_role("event_router")
            if provider == "openai"
            else ""
        ),
        "comparison_baseline": str(BASELINE_PATH),
        "all_passed": all(result["passed"] for result in results),
        "cases": results,
    }
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    for result in results:
        passed = sum(check["passed"] for check in result["checks"])
        status = "ERROR" if result["error"] else f"{passed}/{len(result['checks'])}"
        print(f"{result['name']}: {status}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
