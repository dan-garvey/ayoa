#!/usr/bin/env python3
"""Live smoke harness for relative time and private commitments.

This script uses real LLM calls. It builds a deterministic synthetic story
checkpoint from `scripts/synthetic_relative_time_story_prompt.md`, drives
`EngineBridge.run_turn()` through a set of targeted scenarios, and writes a
JSON + Markdown report under `app/storage/playtest_reports`.

The checkpoint is hand-compiled from the prompt on purpose: this keeps the
run focused on router/narrator/runtime behavior instead of story-import
variance. The prompt is copied into every run directory so the source fiction
is still reviewable alongside the report.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.bot.engine_bridge import EngineBridge
from app.engine.settings import UnknownSettingError
from app.llm.config import LLMConfig
from app.schemas.characters import (
    ActorFact,
    ActorRecord,
    CharacterRecord,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.responses import TurnResponse
from app.schemas.state import (
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)


REPORT_DIR = REPO_ROOT / "app/storage/playtest_reports"
PROMPT_PATH = REPO_ROOT / "scripts/synthetic_relative_time_story_prompt.md"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = REPORT_DIR / f"relative_time_commitment_live_{TS}"
JSON_PATH = RUN_DIR / "report.json"
MD_PATH = RUN_DIR / "report.md"
LOG_PATH = RUN_DIR / "run.log"

PLAYER_IDS = {
    "mira": 1001,
    "theo": 1002,
    "jun": 1003,
    "rhea": 1004,
    "cal": 1005,
}


@dataclass(frozen=True)
class Step:
    actor_id: str
    user_input: str
    label: str = ""


@dataclass(frozen=True)
class LiveCase:
    name: str
    description: str
    players: tuple[str, ...]
    steps: tuple[Step, ...]
    checks: tuple[str, ...]
    default: bool = True


@dataclass(frozen=True)
class ProgressiveStep:
    phase: str
    actor_id: str
    user_input: str
    label: str
    expectations: tuple[str, ...] = ()
    settings: tuple[tuple[str, str], ...] = ()
    slot_rejected_ok: bool = False


CASES: tuple[LiveCase, ...] = (
    LiveCase(
        name="open_commitment_private",
        description=(
            "One player starts a long watch; the commitment should be private."
        ),
        players=("mira",),
        checks=("turns_completed", "open_commitment_for_mira", "no_commitment_leak"),
        steps=(
            Step(
                "mira",
                "I settle at the north gallery window to keep watch for the "
                "red harbor lantern. Do not decide the result yet: I am "
                "committing to watch quietly for up to half an hour unless "
                "something in the gallery changes.",
                "Mira opens a long watch commitment.",
            ),
        ),
    ),
    LiveCase(
        name="interrupted_commitment_revision",
        description=(
            "A second player changes Mira's scene before the watch resolves."
        ),
        players=("mira", "theo"),
        checks=(
            "turns_completed",
            "open_commitment_for_mira_after_first_step",
            "pending_revision_for_mira",
            "no_commitment_leak",
        ),
        steps=(
            Step(
                "mira",
                "I settle at the north gallery window to keep watch for the "
                "red harbor lantern. Do not decide the result yet: I am "
                "committing to watch quietly for up to half an hour unless "
                "something in the gallery changes.",
                "Mira opens a long watch commitment.",
            ),
            Step(
                "theo",
                "Before Mira has had enough time to finish watching, I step "
                "onto the stair landing beside the north gallery, point "
                "through the window toward the harbor, and whisper, 'The "
                "harbor lights just went black.'",
                "Theo visibly changes Mira's scene.",
            ),
        ),
    ),
    LiveCase(
        name="revision_continue_clears_prompt",
        description=(
            "The committed player answers the revision prompt with continue."
        ),
        players=("mira", "theo"),
        checks=(
            "turns_completed",
            "pending_revision_for_mira_after_second_step",
            "revision_prompt_cleared_after_continue",
            "no_commitment_leak",
        ),
        steps=(
            Step(
                "mira",
                "I settle at the north gallery window to keep watch for the "
                "red harbor lantern. Do not decide the result yet: I am "
                "committing to watch quietly for up to half an hour unless "
                "something in the gallery changes.",
                "Mira opens a long watch commitment.",
            ),
            Step(
                "theo",
                "Before Mira has had enough time to finish watching, I step "
                "onto the stair landing beside the north gallery, point "
                "through the window toward the harbor, and whisper, 'The "
                "harbor lights just went black.'",
                "Theo visibly changes Mira's scene.",
            ),
            Step(
                "mira",
                "(continue)",
                "Mira asks to continue the interrupted commitment.",
            ),
        ),
    ),
    LiveCase(
        name="cat_ii_resolution_time",
        description=(
            "A contested key grab should open Cat II and resolve at the "
            "opening event's fiction time."
        ),
        players=("mira", "theo"),
        checks=(
            "turns_completed",
            "cat_ii_opened_first",
            "cat_ii_resolved",
            "cat_ii_resolution_pinned_to_open_time",
            "no_commitment_leak",
        ),
        steps=(
            Step(
                "theo",
                "I lunge for the brass key in Mira's hand and try to take it "
                "before she can close her fist.",
                "Theo opens a contested action.",
            ),
            Step(
                "mira",
                "I clamp my fist shut around the brass key, twist my shoulder "
                "away from Theo, and try to keep the key in my grip.",
                "Mira responds to the Cat II pin.",
            ),
        ),
    ),
    LiveCase(
        name="remote_clock_independence",
        description=(
            "A remote player should act from their local clock after Mira "
            "advances far ahead by watching."
        ),
        players=("mira", "rhea"),
        checks=(
            "turns_completed",
            "mira_clock_advanced_by_wait",
            "rhea_event_stays_before_mira_clock",
            "parallel_clocks_not_flattened",
            "no_commitment_leak",
        ),
        steps=(
            Step(
                "mira",
                "I keep watch from the north gallery for thirty full minutes, "
                "then mark whether the red harbor lantern moved.",
                "Mira advances far ahead.",
            ),
            Step(
                "rhea",
                "At roughly the same early stretch of evening, far away in "
                "the glasshouse, I inspect the cracked sundial and write down "
                "what it shows without waiting on anyone else.",
                "Rhea acts in a remote parallel scene.",
            ),
        ),
    ),
    LiveCase(
        name="five_player_parallel_surface",
        description=(
            "Five players act in separated scenes, then Theo changes Mira's "
            "scene. The clocks should not collapse into one global now."
        ),
        players=("mira", "theo", "jun", "rhea", "cal"),
        checks=(
            "turns_completed",
            "five_player_multiple_clocks_advanced",
            "five_player_clocks_not_all_leading",
            "five_player_mira_revision_or_resolution",
            "no_commitment_leak",
        ),
        steps=(
            Step(
                "mira",
                "I settle at the north gallery window to keep watch for the "
                "red harbor lantern. Do not decide the result yet: I am "
                "committing to watch quietly for up to half an hour unless "
                "something in the gallery changes.",
                "Mira opens a long watch commitment.",
            ),
            Step(
                "jun",
                "In the archive, I spend two minutes indexing the blue tide "
                "ledger, then close it and leave it exactly where I found it.",
                "Jun advances a separate archive clock.",
            ),
            Step(
                "rhea",
                "In the glasshouse, at the same early stretch of evening, I "
                "check the cracked sundial and jot down its reading.",
                "Rhea acts remotely.",
            ),
            Step(
                "cal",
                "At the gatehouse I listen for a courier bell for one minute, "
                "then mark the gate log.",
                "Cal advances a third remote scene.",
            ),
            Step(
                "theo",
                "I pass onto the stair landing beside the north gallery and "
                "ring the small handbell beside Mira's post.",
                "Theo changes Mira's scene.",
            ),
        ),
    ),
)


PROGRESSIVE_STEPS: tuple[ProgressiveStep, ...] = (
    ProgressiveStep(
        phase="1_minimal_setup",
        actor_id="jun",
        label="Advance Jun's archive clock far ahead",
        user_input=(
            "In the archive, I spend twelve full minutes indexing the blue "
            "tide ledger, then close it and leave it exactly where I found it."
        ),
        expectations=(
            "long_wait_duration_encoded",
            "jun_advanced",
            "player_clocks_diverged",
        ),
    ),
    ProgressiveStep(
        phase="1_minimal_setup",
        actor_id="rhea",
        label="Open a private calibration with a secret code",
        user_input=(
            "At the glasshouse, still in my own early slice of evening, I "
            "start a twenty-minute calibration of the cracked sundial. I "
            "write the private code BLUE-FOG on my wrist slate, but I do not "
            "send that code to anyone."
        ),
        expectations=(
            "remote_event_before_leading",
            "open_rhea_commitment",
            "blue_fog_private_to_rhea",
        ),
    ),
    ProgressiveStep(
        phase="1_minimal_setup",
        actor_id="cal",
        label="Open an overlapping gatehouse watch",
        user_input=(
            "At the gatehouse, also in my own early slice of evening, I begin "
            "a ten-minute watch through the peephole for any late visitor. "
            "Do not decide the result yet unless something at the gatehouse "
            "changes."
        ),
        expectations=(
            "remote_event_before_leading",
            "open_cal_commitment",
            "player_clocks_diverged",
        ),
    ),
    ProgressiveStep(
        phase="2_failure_probes",
        actor_id="theo",
        label="Backfill a building-wide bell into advanced archive time",
        user_input=(
            "From the stair landing I ring the clockhouse bell three times. "
            "The bell is loud enough for the archive, glasshouse, and "
            "gatehouse, and I call, 'Hold your posts until Mira confirms the "
            "lantern.'"
        ),
        expectations=(
            "shared_bell_observed_by_rhea_and_cal",
            "rhea_cal_revision_prompts",
            "no_advanced_observer_backfill_without_revision",
        ),
    ),
    ProgressiveStep(
        phase="2_failure_probes",
        actor_id="theo",
        label="Direct a second early call at advanced Jun",
        user_input=(
            "Before waiting for any reply from the archive, I call from the "
            "stair landing toward Jun: 'Archive, answer now if the ledger "
            "says why the lantern went dark.'"
        ),
        expectations=(
            "rhea_cal_revisions_still_pending",
            "no_advanced_observer_backfill_without_revision",
        ),
    ),
    ProgressiveStep(
        phase="2_failure_probes",
        actor_id="jun",
        label="Probe for a remote private code",
        user_input=(
            "I look over the archive notes and ask aloud, 'Has anyone "
            "reported what the glasshouse instruments showed?' I have not "
            "heard from Rhea and I do not know her wrist-slate code."
        ),
        expectations=("no_blue_fog_leak_to_jun", "no_new_global_flattening"),
    ),
    ProgressiveStep(
        phase="2_failure_probes",
        actor_id="cal",
        label="Continue past Cal's watch max duration",
        user_input=(
            "I hear the bell, but keep my eye at the gatehouse peephole for "
            "eleven full minutes anyway. When the watch is over I tell anyone "
            "beside me, 'No late visitor crossed the threshold.'"
        ),
        expectations=(
            "cal_revision_cleared",
            "long_wait_duration_encoded",
            "cal_commitment_resolved_after_wait",
            "player_clocks_diverged",
        ),
    ),
    ProgressiveStep(
        phase="2_failure_probes",
        actor_id="rhea",
        label="Finish Rhea's private calibration after interruption",
        user_input=(
            "I hear the bell but choose to keep calibrating. I work until the "
            "full twenty-minute calibration is complete, then mark the final "
            "glasshouse drift reading under the private code BLUE-FOG."
        ),
        expectations=(
            "rhea_revision_cleared",
            "long_wait_duration_encoded",
            "rhea_commitment_resolved_after_wait",
            "blue_fog_private_to_rhea",
        ),
    ),
    ProgressiveStep(
        phase="2_failure_probes",
        actor_id="rhea",
        label="Attempt a forbidden local backfill after completion",
        user_input=(
            "Actually, before I finished the calibration, I secretly swapped "
            "the sundial label and hid the old label under the cracked lens. "
            "I want that to have already happened earlier."
        ),
        expectations=("actor_event_not_before_actor_clock", "no_actor_clock_regression"),
    ),
    ProgressiveStep(
        phase="2_failure_probes",
        actor_id="rhea",
        label="Reveal the private code only to Cal",
        user_input=(
            "I walk to the gatehouse and quietly tell only Cal, 'The private "
            "glasshouse code is BLUE-FOG. Do not pass that to the archive.'"
        ),
        expectations=(
            "blue_fog_visible_only_to_rhea_cal",
            "no_blue_fog_leak_to_mira_or_jun",
            "player_clocks_diverged",
        ),
    ),
    ProgressiveStep(
        phase="2_failure_probes",
        actor_id="theo",
        label="Late call into advanced archive time",
        user_input=(
            "I stand on the stair landing and call clearly enough for Jun in "
            "the archive, but not the gatehouse: 'If the lantern went dark, "
            "check the ledger and tell me whether to fetch Cal.'"
        ),
        expectations=(
            "no_commitment_leak",
            "player_clocks_diverged",
            "no_advanced_observer_backfill_without_revision",
        ),
    ),
    ProgressiveStep(
        phase="2_failure_probes",
        actor_id="jun",
        label="Ask whether the remote code is known after a scoped reveal",
        user_input=(
            "Still in the archive, I ask plainly, 'Do I know the glasshouse "
            "private code yet, or has no one told me?'"
        ),
        expectations=("no_blue_fog_leak_to_jun", "no_new_global_flattening"),
    ),
)


def _char(
    *,
    character_id: str,
    name: str,
    role: str,
    location: str,
    appearance: str,
    playable: bool = True,
    may_act_offstage: bool = False,
    goals: list[str] | None = None,
    objectives: list[str] | None = None,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        location=location,
        is_playable=playable,
        public_sheet=PublicSheet(
            role=role,
            appearance=appearance,
        ),
        actor=ActorRecord(
            may_act_offstage=may_act_offstage,
            facts=[
                ActorFact(
                    text=(
                        "You are a trusted clockhouse worker present for the "
                        "dusk lantern watch."
                    )
                ),
                ActorFact(
                    text=(
                        "You are concrete, restrained, and attentive to "
                        "visible detail."
                    )
                ),
                *(
                    ActorFact(text=f"You want to {value.rstrip('.').lower()}.")
                    for value in (
                        goals
                        or ["Keep the Lantern Clockhouse stable tonight."]
                    )
                ),
                *(
                    ActorFact(text=f"You intend to {value.rstrip('.').lower()}.")
                    for value in (
                        objectives
                        or ["Respond to visible changes in the clockhouse."]
                    )
                ),
            ],
        ),
    )


def _story_checkpoint(story_id: str) -> CheckpointFile:
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
                "The session begins at relative time 0s, just after dusk.",
                "The north gallery overlooks the red harbor lantern.",
                "The stair landing is adjacent to the north gallery.",
                "Speech, shutters, and handbells near the stair landing can "
                "be perceived from the north gallery when observers are named.",
                "The archive, glasshouse, and gatehouse are separate locations "
                "that can progress in parallel.",
                "A brass key and a sealed tide ledger are visible in the "
                "north gallery.",
                "Open commitments are private routing state and must not be "
                "rendered as facts merely because they exist.",
            ],
            physics_ruleset=PhysicsRuleset(
                strength_limits="ordinary human capability",
                magic_enabled=False,
            ),
            setting=StorySetting(
                genre="coastal mystery",
                era="gaslamp-adjacent but rules-neutral",
                tone="precise, restrained, table-play concrete",
                premise=(
                    "Five playable clockhouse workers coordinate around a "
                    "red harbor lantern and several time-sensitive duties."
                ),
            ),
            lore=(
                "The clockhouse bell marks work intervals. The red harbor "
                "lantern is expected to move after about thirty minutes."
            ),
        ),
        characters=[
            _char(
                character_id="mira",
                name="Mira Vale",
                role="watch officer",
                location="north gallery",
                appearance="dark watch coat, brass timepiece, ink-stained cuffs",
                goals=["Learn whether the harbor lantern moves."],
                objectives=["Keep watch from the north gallery."],
            ),
            _char(
                character_id="theo",
                name="Theo Rusk",
                role="courier",
                location="stair landing",
                appearance="short blue runner's coat, message satchel",
                goals=["Get urgent information to the right person quickly."],
                objectives=["Carry updates between clockhouse rooms."],
            ),
            _char(
                character_id="jun",
                name="Jun Park",
                role="archivist",
                location="archive",
                appearance="rolled sleeves, ledger straps, graphite on fingers",
                goals=["Keep the tide ledger accurate."],
                objectives=["Index the blue tide ledger."],
            ),
            _char(
                character_id="rhea",
                name="Rhea Sol",
                role="glasshouse technician",
                location="glasshouse",
                appearance="green apron, glass dust, cracked lens on a chain",
                goals=["Track instrument drift before the bell interval ends."],
                objectives=["Check the cracked sundial and weather glass."],
            ),
            _char(
                character_id="cal",
                name="Cal Ives",
                role="gatehouse keeper",
                location="gatehouse",
                appearance="weathered cloak, iron key loop, lantern hook",
                goals=["Notice arrivals before they reach the galleries."],
                objectives=["Listen for the courier bell and keep the gate log."],
            ),
            _char(
                character_id="keeper_solan",
                name="Keeper Solan",
                role="off-stage keeper",
                location="map room",
                appearance="grey coat, silver case under one arm",
                playable=False,
                may_act_offstage=True,
                goals=["Move the silver case before the next bell interval."],
                objectives=[
                    "Carry the silver case from the map room to a locked cabinet."
                ],
            ),
            _char(
                character_id="scribe_nell",
                name="Scribe Nell",
                role="off-stage scribe",
                location="lower archive",
                appearance="brown smock, copy slip, capped ink bottle",
                playable=False,
                may_act_offstage=True,
                goals=["Copy one tide-ledger line without being noticed."],
                objectives=[
                    "Copy the marked tide-ledger line and hide the copy."
                ],
            ),
        ],
    )
    settings = ckpt.session.config.settings
    settings.max_events_per_beat = 3
    return ckpt


def _write_story(stories_dir: Path, story_id: str) -> None:
    dst = stories_dir / story_id
    dst.mkdir(parents=True, exist_ok=True)
    ckpt = _story_checkpoint(story_id)
    (dst / "ckpt_0000.json").write_text(
        ckpt.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _checkpoint_snapshot(ckpt: CheckpointFile) -> dict[str, Any]:
    return {
        "turn_index": ckpt.session.turn_index,
        "leading_at_s": ckpt.session.leading_at_s,
        "character_clocks": {
            char.character_id: getattr(char, "clock_at_s", 0)
            for char in ckpt.characters
        },
        "character_locations": {
            char.character_id: char.location for char in ckpt.characters
        },
        "open_commitments": [
            commitment.model_dump(mode="json")
            for commitment in ckpt.session.open_commitments
        ],
        "pending_commitment_revisions": {
            cid: prompt.model_dump(mode="json")
            for cid, prompt in ckpt.session.pending_commitment_revisions.items()
        },
        "active_act_slots": {
            cid: slot.model_dump(mode="json")
            for cid, slot in ckpt.session.active_act_slots.items()
        },
        "open_cat_ii_events": [
            event.model_dump(mode="json")
            for event in ckpt.session.open_cat_ii_events
        ],
        "canonical_events": [
            _event_dump(event) for event in ckpt.canonical_events
        ],
    }


def _event_dump(event: EventRouterOutput) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "effective_at_s": event.effective_at_s,
        "duration_s": event.duration_s,
        "event_kind": event.event_kind,
        "requires_responders": event.requires_responders,
        "required_responders": list(event.required_responders),
        "next_output_character_ids": list(event.next_output_character_ids),
        "perception_enrichment_character_ids": list(
            event.perception_enrichment_character_ids
        ),
        "observers": [
            {
                "character_id": observer.character_id,
                "observation_level": observer.observation_level,
                "routing_role": observer.routing_role,
            }
            for observer in event.observers
        ],
        "facts": [
            {
                "text": fact.text,
                "audience": fact.audience,
                "visible_to": list(fact.visible_to),
                "at_offset_s": fact.at_offset_s,
                "duration_s": fact.duration_s,
            }
            for fact in event.canonical_event.observable_facts
        ],
        "commitment_open": event.commitment_open.model_dump(mode="json"),
        "commitment_resolutions": [
            signal.model_dump(mode="json")
            for signal in event.commitment_resolutions
        ],
        "commitment_interrupts": [
            signal.model_dump(mode="json")
            for signal in event.commitment_interrupts
        ],
        "location_updates": [
            signal.model_dump(mode="json") for signal in event.location_updates
        ],
        "decision_rationale": event.decision_rationale,
    }


def _response_dump(response: TurnResponse) -> dict[str, Any]:
    return {
        "beat_ended_reason": response.beat_ended_reason,
        "turn_index": response.turn_index,
        "output_text": response.output_text,
        "per_player_renders": response.per_player_renders,
        "reaction_prompts": response.reaction_prompts,
        "loot_prompts": response.loot_prompts,
        "commitment_revision_prompts": response.commitment_revision_prompts,
        "pre_turn_resolutions": [
            item.model_dump(mode="json") for item in response.pre_turn_resolutions
        ],
    }


def _render_texts(step_reports: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for step in step_reports:
        response = step.get("response") or {}
        output = response.get("output_text") or ""
        if output:
            texts.append(output)
        for render in (response.get("per_player_renders") or {}).values():
            if render:
                texts.append(render)
    return texts


def _new_events_for_step(
    before: CheckpointFile,
    after: CheckpointFile,
) -> list[EventRouterOutput]:
    return after.canonical_events[len(before.canonical_events):]


async def _run_step(
    *,
    engine: EngineBridge,
    session_id: str,
    step: Step,
) -> dict[str, Any]:
    before = engine.load_latest(session_id)
    response = await engine.run_turn(
        session_id=session_id,
        user_input=step.user_input,
        acting_character_id=step.actor_id,
    )
    after = engine.load_latest(session_id)
    new_events = _new_events_for_step(before, after)
    return {
        "label": step.label,
        "actor_id": step.actor_id,
        "user_input": step.user_input,
        "response": _response_dump(response),
        "new_events": [_event_dump(event) for event in new_events],
        "checkpoint_after": _checkpoint_snapshot(after),
    }


def _check(name: str, passed: bool, detail: Any = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _final_snapshot(case_report: dict[str, Any]) -> dict[str, Any]:
    return case_report["steps"][-1]["checkpoint_after"] if case_report["steps"] else {}


def _step_snapshot(case_report: dict[str, Any], index: int) -> dict[str, Any]:
    steps = case_report.get("steps") or []
    if index >= len(steps):
        return {}
    return steps[index].get("checkpoint_after") or {}


def _all_events(case_report: dict[str, Any]) -> list[dict[str, Any]]:
    return list((_final_snapshot(case_report).get("canonical_events") or []))


def _events_by_kind(case_report: dict[str, Any], event_kind: str) -> list[dict[str, Any]]:
    return [
        event for event in _all_events(case_report)
        if event.get("event_kind") == event_kind
    ]


def _commitments(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return list(snapshot.get("open_commitments") or [])


def _pending_revisions(snapshot: dict[str, Any]) -> dict[str, Any]:
    return dict(snapshot.get("pending_commitment_revisions") or {})


def _render_leak_check(case_report: dict[str, Any]) -> dict[str, Any]:
    joined = "\n".join(_render_texts(case_report.get("steps") or []))
    banned = (
        "## Open Commitments",
        "OpenCommitment",
        "pending_commitment",
        "commit_evt_",
        "commitment_id",
    )
    leaks = [term for term in banned if term in joined]
    return _check("no_commitment_leak", not leaks, {"leaks": leaks})


def _turns_completed(case_report: dict[str, Any]) -> dict[str, Any]:
    errors = [
        step.get("error", "")
        for step in case_report.get("steps", [])
        if step.get("error")
    ]
    slot_rejections = [
        step["response"].get("beat_ended_reason")
        for step in case_report.get("steps", [])
        if (step.get("response") or {}).get("beat_ended_reason") == "slot_rejected"
    ]
    return _check(
        "turns_completed",
        not errors and not slot_rejections,
        {"errors": errors, "slot_rejections": slot_rejections},
    )


def _open_commitment_for(snapshot: dict[str, Any], character_id: str) -> bool:
    return any(
        character_id in commitment.get("actor_ids", [])
        for commitment in _commitments(snapshot)
    )


def _evaluate_case(case: LiveCase, case_report: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    final = _final_snapshot(case_report)
    for name in case.checks:
        if name == "turns_completed":
            checks.append(_turns_completed(case_report))
        elif name == "no_commitment_leak":
            checks.append(_render_leak_check(case_report))
        elif name == "open_commitment_for_mira":
            checks.append(_check(
                name,
                _open_commitment_for(final, "mira"),
                _commitments(final),
            ))
        elif name == "open_commitment_for_mira_after_first_step":
            snap = _step_snapshot(case_report, 0)
            checks.append(_check(
                name,
                _open_commitment_for(snap, "mira"),
                _commitments(snap),
            ))
        elif name == "pending_revision_for_mira":
            revisions = _pending_revisions(final)
            response_prompts = (
                (case_report["steps"][-1].get("response") or {})
                .get("commitment_revision_prompts") or {}
            )
            checks.append(_check(
                name,
                "mira" in revisions or "mira" in response_prompts,
                {
                    "pending": revisions,
                    "response_prompts": response_prompts,
                },
            ))
        elif name == "pending_revision_for_mira_after_second_step":
            snap = _step_snapshot(case_report, 1)
            revisions = _pending_revisions(snap)
            response_prompts = (
                (case_report["steps"][1].get("response") or {})
                .get("commitment_revision_prompts") or {}
            )
            checks.append(_check(
                name,
                "mira" in revisions or "mira" in response_prompts,
                {
                    "pending": revisions,
                    "response_prompts": response_prompts,
                },
            ))
        elif name == "revision_prompt_cleared_after_continue":
            checks.append(_check(
                name,
                "mira" not in _pending_revisions(final),
                _pending_revisions(final),
            ))
        elif name == "cat_ii_opened_first":
            first_step = (case_report.get("steps") or [{}])[0]
            response = first_step.get("response") or {}
            snap = first_step.get("checkpoint_after") or {}
            checks.append(_check(
                name,
                response.get("beat_ended_reason") == "cat_ii_pending"
                and bool(snap.get("open_cat_ii_events")),
                {
                    "response_reason": response.get("beat_ended_reason"),
                    "open_cat_ii_events": snap.get("open_cat_ii_events"),
                    "active_slots": snap.get("active_act_slots"),
                },
            ))
        elif name == "cat_ii_resolved":
            checks.append(_check(
                name,
                bool(_events_by_kind(case_report, "cat_ii_resolution"))
                and not final.get("open_cat_ii_events"),
                {
                    "cat_ii_resolution_events": _events_by_kind(
                        case_report, "cat_ii_resolution"
                    ),
                    "open_cat_ii_events": final.get("open_cat_ii_events"),
                },
            ))
        elif name == "cat_ii_resolution_pinned_to_open_time":
            opens = _events_by_kind(case_report, "cat_ii_open")
            resolutions = _events_by_kind(case_report, "cat_ii_resolution")
            pinned = bool(opens and resolutions) and (
                resolutions[-1].get("effective_at_s") == opens[-1].get("effective_at_s")
            )
            duration_zero = bool(opens) and opens[-1].get("duration_s") == 0
            checks.append(_check(
                name,
                pinned and duration_zero,
                {"opens": opens, "resolutions": resolutions},
            ))
        elif name == "mira_clock_advanced_by_wait":
            snap = _step_snapshot(case_report, 0)
            clocks = snap.get("character_clocks") or {}
            checks.append(_check(
                name,
                int(clocks.get("mira", 0)) >= 600,
                clocks,
            ))
        elif name == "rhea_event_stays_before_mira_clock":
            first = _step_snapshot(case_report, 0)
            mira_clock = int((first.get("character_clocks") or {}).get("mira", 0))
            second_events = (
                case_report["steps"][1].get("new_events")
                if len(case_report.get("steps", [])) > 1 else []
            ) or []
            rhea_start = (
                int(second_events[0].get("effective_at_s", 0))
                if second_events else 0
            )
            checks.append(_check(
                name,
                bool(second_events) and rhea_start < mira_clock,
                {"mira_clock_after_wait": mira_clock, "rhea_events": second_events},
            ))
        elif name == "parallel_clocks_not_flattened":
            clocks = final.get("character_clocks") or {}
            leading = int(final.get("leading_at_s", 0))
            checks.append(_check(
                name,
                int(clocks.get("rhea", 0)) < leading
                and int(clocks.get("mira", 0)) == leading,
                {"leading_at_s": leading, "character_clocks": clocks},
            ))
        elif name == "five_player_multiple_clocks_advanced":
            clocks = final.get("character_clocks") or {}
            advanced = [
                cid for cid in case.players if int(clocks.get(cid, 0)) > 0
            ]
            checks.append(_check(name, len(advanced) >= 4, clocks))
        elif name == "five_player_clocks_not_all_leading":
            clocks = final.get("character_clocks") or {}
            leading = int(final.get("leading_at_s", 0))
            player_clock_values = [int(clocks.get(cid, 0)) for cid in case.players]
            checks.append(_check(
                name,
                bool(player_clock_values)
                and max(player_clock_values) == leading
                and any(value < leading for value in player_clock_values),
                {"leading_at_s": leading, "player_clocks": player_clock_values},
            ))
        elif name == "five_player_mira_revision_or_resolution":
            revisions = _pending_revisions(final)
            mira_commitments = [
                commitment for commitment in _commitments(final)
                if "mira" in commitment.get("actor_ids", [])
            ]
            resolution_events = [
                event for event in _all_events(case_report)
                if any(
                    "mira" in signal.get("actor_ids", [])
                    or "mira" in json.dumps(signal)
                    for signal in event.get("commitment_resolutions", [])
                )
            ]
            checks.append(_check(
                name,
                "mira" in revisions or bool(resolution_events) or not mira_commitments,
                {
                    "pending_revisions": revisions,
                    "mira_commitments": mira_commitments,
                    "resolution_events": resolution_events,
                },
            ))
        else:
            checks.append(_check(name, False, "unknown check"))
    return checks


async def _run_case(
    case: LiveCase,
    *,
    engine: EngineBridge,
    stories_dir: Path,
    role_calls: list[dict[str, Any]],
    role_call_start: int,
) -> dict[str, Any]:
    story_id = f"relative_time_lab_{case.name}"
    session_id = f"{story_id}_{TS.lower()}"
    _write_story(stories_dir, story_id)
    engine.create_empty_session(session_id)
    engine.load_story_into_session(session_id, story_id)
    for char_id in case.players:
        await engine.bind_user(session_id, PLAYER_IDS[char_id], char_id)

    steps: list[dict[str, Any]] = []
    for step in case.steps:
        try:
            steps.append(await _run_step(
                engine=engine,
                session_id=session_id,
                step=step,
            ))
        except Exception:
            steps.append({
                "label": step.label,
                "actor_id": step.actor_id,
                "user_input": step.user_input,
                "error": traceback.format_exc(),
            })
            break

    case_report = {
        "name": case.name,
        "description": case.description,
        "session_id": session_id,
        "story_id": story_id,
        "players": list(case.players),
        "steps": steps,
        "role_calls": list(role_calls[role_call_start:]),
    }
    case_report["checks"] = _evaluate_case(case, case_report)
    return case_report


def _usage_totals(calls: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "prompt_tokens",
        "completion_tokens",
        "visible_completion_tokens",
        "reasoning_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "full_input_tokens",
        "total_tokens",
    )
    totals = {key: 0 for key in keys}
    for call in calls:
        usage = call.get("usage") or {}
        for key in keys:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


def _parse_setting_like_current(current: Any, raw_value: str) -> Any:
    if isinstance(current, bool):
        value = raw_value.strip().lower()
        if value in {"true", "t", "yes", "y", "on", "1", "enable", "enabled"}:
            return True
        if value in {"false", "f", "no", "n", "off", "0", "disable", "disabled"}:
            return False
        raise ValueError(f"Cannot interpret {raw_value!r} as a boolean.")
    if isinstance(current, int):
        return int(raw_value.strip())
    return raw_value


async def _apply_progressive_setting(
    engine: EngineBridge,
    session_id: str,
    key: str,
    raw_value: str,
) -> Any:
    try:
        return await engine.set_setting(session_id, key, raw_value)
    except UnknownSettingError:
        ckpt = engine.load_latest(session_id)
        settings = ckpt.session.config.settings
        if not hasattr(settings, key):
            raise
        parsed = _parse_setting_like_current(getattr(settings, key), raw_value)
        setattr(settings, key, parsed)
        engine.checkpoint_mgr.save(ckpt)
        return parsed


def _role_counts(calls: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for call in calls:
        role = str(call.get("role") or "")
        counts[role] = counts.get(role, 0) + 1
    return counts


def _render_texts_from_response(response: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    output = response.get("output_text") or ""
    if output:
        texts.append(output)
    for render in (response.get("per_player_renders") or {}).values():
        if render:
            texts.append(render)
    return texts


def _progressive_leak_check(step_report: dict[str, Any]) -> dict[str, Any]:
    joined = "\n".join(
        _render_texts_from_response(step_report.get("response") or {})
    )
    banned = (
        "## Open Commitments",
        "OpenCommitment",
        "pending_commitment",
        "commit_evt_",
        "commitment_id",
    )
    leaks = [term for term in banned if term in joined]
    return _check("no_commitment_leak", not leaks, {"leaks": leaks})


def _progressive_clock_regression_check(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    before_clocks = before.get("character_clocks") or {}
    after_clocks = after.get("character_clocks") or {}
    regressions = {
        cid: {
            "before": before_clocks.get(cid),
            "after": after_clocks.get(cid),
        }
        for cid in sorted(set(before_clocks) | set(after_clocks))
        if int(after_clocks.get(cid, 0) or 0) < int(before_clocks.get(cid, 0) or 0)
    }
    leading_ok = int(after.get("leading_at_s", 0) or 0) >= int(
        before.get("leading_at_s", 0) or 0
    )
    return _check(
        "no_clock_regression",
        not regressions and leading_ok,
        {
            "character_regressions": regressions,
            "before_leading": before.get("leading_at_s"),
            "after_leading": after.get("leading_at_s"),
        },
    )


def _progressive_event_timing_check(step_report: dict[str, Any]) -> dict[str, Any]:
    bad: list[dict[str, Any]] = []
    for event in step_report.get("new_events") or []:
        duration = int(event.get("duration_s", 0) or 0)
        if int(event.get("effective_at_s", 0) or 0) < 0 or duration < 0:
            bad.append({"event_id": event.get("event_id"), "event": event})
            continue
        for fact in event.get("facts") or []:
            at = int(fact.get("at_offset_s", 0) or 0)
            fact_duration = int(fact.get("duration_s", 0) or 0)
            if at < 0 or fact_duration < 0 or at + fact_duration > duration:
                bad.append({
                    "event_id": event.get("event_id"),
                    "fact": fact,
                    "event_duration_s": duration,
                })
    return _check("event_timing_windows_valid", not bad, bad)


def _progressive_chronology_check(step_report: dict[str, Any]) -> dict[str, Any]:
    bad: list[dict[str, Any]] = []
    last_start = -1
    for event in step_report.get("new_events") or []:
        start = int(event.get("effective_at_s", 0) or 0)
        if start < last_start:
            bad.append({
                "kind": "event_order",
                "event_id": event.get("event_id"),
                "effective_at_s": start,
                "previous_effective_at_s": last_start,
            })
        last_start = max(last_start, start)

        last_offset = -1
        for fact in event.get("facts") or []:
            offset = int(fact.get("at_offset_s", 0) or 0)
            if offset < last_offset:
                bad.append({
                    "kind": "fact_order",
                    "event_id": event.get("event_id"),
                    "fact": fact,
                    "previous_at_offset_s": last_offset,
                })
            last_offset = max(last_offset, offset)
    return _check("new_events_and_facts_chronological", not bad, bad)


def _player_clock_values(snapshot: dict[str, Any]) -> list[int]:
    clocks = snapshot.get("character_clocks") or {}
    return [int(clocks.get(cid, 0) or 0) for cid in PLAYER_IDS]


def _render_texts_for_characters(
    response: dict[str, Any],
    character_ids: set[str],
) -> list[str]:
    texts: list[str] = []
    renders = response.get("per_player_renders") or {}
    for character_id in character_ids:
        render = renders.get(character_id)
        if render:
            texts.append(render)
    return texts


def _event_observer_ids(event: dict[str, Any]) -> set[str]:
    return {
        str(observer.get("character_id"))
        for observer in event.get("observers", [])
        if observer.get("character_id")
    }


def _event_fact_visible_to(event: dict[str, Any], fact: dict[str, Any]) -> set[str]:
    if fact.get("audience") == "only":
        return {str(cid) for cid in fact.get("visible_to", [])}
    return _event_observer_ids(event)


def _jsonable_for_compare(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _progressive_expectation_check(
    expectation: str,
    *,
    step: ProgressiveStep,
    step_report: dict[str, Any],
) -> dict[str, Any]:
    before = step_report.get("checkpoint_before") or {}
    after = step_report.get("checkpoint_after") or {}
    response = step_report.get("response") or {}
    new_events = step_report.get("new_events") or []
    commitments = _commitments(after)
    revisions = _pending_revisions(after)
    response_prompts = response.get("commitment_revision_prompts") or {}

    if expectation == "no_commitment_leak":
        return _progressive_leak_check(step_report)

    if expectation == "open_mira_commitment":
        return _check(
            expectation,
            _open_commitment_for(after, "mira"),
            commitments,
        )

    if expectation == "open_rhea_commitment":
        return _check(
            expectation,
            _open_commitment_for(after, "rhea"),
            commitments,
        )

    if expectation == "open_cal_commitment":
        return _check(
            expectation,
            _open_commitment_for(after, "cal"),
            commitments,
        )

    if expectation == "mira_and_jun_advanced":
        clocks = after.get("character_clocks") or {}
        return _check(
            expectation,
            int(clocks.get("mira", 0) or 0) >= 600
            and int(clocks.get("jun", 0) or 0) >= 600,
            clocks,
        )

    if expectation == "jun_advanced":
        clocks = after.get("character_clocks") or {}
        return _check(
            expectation,
            int(clocks.get("jun", 0) or 0) >= 600,
            clocks,
        )

    if expectation == "mira_revision_prompt":
        return _check(
            expectation,
            "mira" in revisions or "mira" in response_prompts,
            {"pending": revisions, "response_prompts": response_prompts},
        )

    if expectation == "mira_revision_prompt_cleared":
        return _check(
            expectation,
            "mira" not in revisions and "mira" not in response_prompts,
            {"pending": revisions, "response_prompts": response_prompts},
        )

    if expectation == "cat_ii_open":
        return _check(
            expectation,
            response.get("beat_ended_reason") == "cat_ii_pending"
            and any(event.get("event_kind") == "cat_ii_open" for event in new_events)
            and bool(after.get("open_cat_ii_events")),
            {
                "beat_ended_reason": response.get("beat_ended_reason"),
                "new_events": new_events,
                "open_cat_ii_events": after.get("open_cat_ii_events"),
            },
        )

    if expectation == "slot_rejected_no_state_change":
        same = {
            "character_clocks": (
                before.get("character_clocks") == after.get("character_clocks")
            ),
            "character_locations": (
                before.get("character_locations") == after.get("character_locations")
            ),
            "open_commitments": _jsonable_for_compare(
                before.get("open_commitments") or []
            ) == _jsonable_for_compare(after.get("open_commitments") or []),
            "pending_commitment_revisions": _jsonable_for_compare(
                before.get("pending_commitment_revisions") or {}
            ) == _jsonable_for_compare(
                after.get("pending_commitment_revisions") or {}
            ),
            "open_cat_ii_events": _jsonable_for_compare(
                before.get("open_cat_ii_events") or []
            ) == _jsonable_for_compare(after.get("open_cat_ii_events") or []),
        }
        return _check(
            expectation,
            response.get("beat_ended_reason") == "slot_rejected"
            and not new_events
            and all(same.values()),
            {
                "beat_ended_reason": response.get("beat_ended_reason"),
                "new_events": new_events,
                "same": same,
            },
        )

    if expectation == "cat_ii_resolution_pinned":
        all_events = after.get("canonical_events") or []
        opens = [
            event for event in all_events
            if event.get("event_kind") == "cat_ii_open"
        ]
        resolutions = [
            event for event in new_events
            if event.get("event_kind") == "cat_ii_resolution"
        ]
        passed = bool(opens and resolutions) and (
            int(resolutions[-1].get("effective_at_s", -1))
            == int(opens[-1].get("effective_at_s", -2))
        )
        return _check(
            expectation,
            passed,
            {"last_open": opens[-1:] or [], "new_resolutions": resolutions},
        )

    if expectation == "cat_ii_required_mira_only":
        candidates = [
            event for event in new_events
            if event.get("event_kind") == "cat_ii_open"
        ] or list(after.get("open_cat_ii_events") or [])
        required = [
            sorted(event.get("required_responders") or [])
            for event in candidates
        ]
        return _check(
            expectation,
            bool(required) and all(item == ["mira"] for item in required),
            {"required_responders": required, "candidates": candidates},
        )

    if expectation == "remote_event_before_leading":
        before_leading = int(before.get("leading_at_s", 0) or 0)
        actor_events = [
            event for event in new_events
            if step.actor_id in [
                observer.get("character_id")
                for observer in event.get("observers", [])
            ]
        ]
        passed = bool(actor_events) and (
            before_leading == 0
            or int(actor_events[0].get("effective_at_s", 0) or 0) < before_leading
        )
        return _check(
            expectation,
            passed,
            {"before_leading_at_s": before_leading, "actor_events": actor_events},
        )

    if expectation == "player_clocks_diverged":
        values = _player_clock_values(after)
        nonzero = [value for value in values if value > 0]
        return _check(
            expectation,
            len(set(values)) > 1 and bool(nonzero),
            {
                "leading_at_s": after.get("leading_at_s"),
                "player_clocks": {
                    cid: (after.get("character_clocks") or {}).get(cid)
                    for cid in PLAYER_IDS
                },
            },
        )

    if expectation == "mira_commitment_resolved_or_mira_moved":
        mira_commitments = [
            commitment for commitment in commitments
            if "mira" in commitment.get("actor_ids", [])
        ]
        locations = after.get("character_locations") or {}
        moved = str(locations.get("mira") or "").lower() not in {
            "north gallery",
            "north_gallery",
        }
        return _check(
            expectation,
            not mira_commitments or moved,
            {"mira_commitments": mira_commitments, "mira_location": locations.get("mira")},
        )

    if expectation == "no_new_global_flattening":
        values = _player_clock_values(after)
        leading = int(after.get("leading_at_s", 0) or 0)
        active_values = [value for value in values if value > 0]
        return _check(
            expectation,
            len(set(active_values)) > 1
            and any(value < leading for value in active_values),
            {"leading_at_s": leading, "player_clock_values": values},
        )

    if expectation == "no_actor_clock_regression":
        before_clock = int(
            (before.get("character_clocks") or {}).get(step.actor_id, 0) or 0
        )
        after_clock = int(
            (after.get("character_clocks") or {}).get(step.actor_id, 0) or 0
        )
        return _check(
            expectation,
            after_clock >= before_clock,
            {"before": before_clock, "after": after_clock},
        )

    if expectation == "blue_fog_private_to_rhea":
        blue_facts: list[dict[str, Any]] = []
        bad: list[dict[str, Any]] = []
        for event in new_events:
            for fact in event.get("facts") or []:
                if "BLUE-FOG" not in str(fact.get("text", "")).upper():
                    continue
                visible_to = _event_fact_visible_to(event, fact)
                entry = {
                    "event_id": event.get("event_id"),
                    "fact": fact,
                    "visible_to": sorted(visible_to),
                }
                blue_facts.append(entry)
                if visible_to - {"rhea"}:
                    bad.append(entry)
        return _check(
            expectation,
            bool(blue_facts) and not bad,
            {"blue_facts": blue_facts, "bad": bad},
        )

    if expectation == "shared_bell_observed_by_rhea_and_cal":
        bell_events = [
            event for event in new_events
            if "bell" in json.dumps(event).lower()
        ]
        matching = [
            event for event in bell_events
            if {"rhea", "cal"}.issubset(_event_observer_ids(event))
        ]
        return _check(
            expectation,
            bool(matching),
            {"bell_events": bell_events},
        )

    if expectation == "rhea_cal_revision_prompts":
        available = set(revisions) | set(response_prompts)
        return _check(
            expectation,
            {"rhea", "cal"}.issubset(available),
            {"pending": revisions, "response_prompts": response_prompts},
        )

    if expectation == "rhea_cal_revisions_still_pending":
        available = set(revisions) | set(response_prompts)
        return _check(
            expectation,
            {"rhea", "cal"}.issubset(available),
            {"pending": revisions, "response_prompts": response_prompts},
        )

    if expectation == "rhea_revision_cleared_cal_pending":
        available = set(revisions) | set(response_prompts)
        return _check(
            expectation,
            "rhea" not in available and "cal" in available,
            {"pending": revisions, "response_prompts": response_prompts},
        )

    if expectation == "rhea_revision_cleared":
        available = set(revisions) | set(response_prompts)
        return _check(
            expectation,
            "rhea" not in available,
            {"pending": revisions, "response_prompts": response_prompts},
        )

    if expectation == "cal_revision_cleared":
        available = set(revisions) | set(response_prompts)
        return _check(
            expectation,
            "cal" not in available,
            {"pending": revisions, "response_prompts": response_prompts},
        )

    if expectation == "cal_commitment_resolved_after_wait":
        cal_commitments = [
            commitment for commitment in commitments
            if "cal" in commitment.get("actor_ids", [])
        ]
        available = set(revisions) | set(response_prompts)
        return _check(
            expectation,
            not cal_commitments and "cal" not in available,
            {
                "cal_commitments": cal_commitments,
                "pending": revisions,
                "response_prompts": response_prompts,
            },
        )

    if expectation == "rhea_commitment_resolved_after_wait":
        rhea_commitments = [
            commitment for commitment in commitments
            if "rhea" in commitment.get("actor_ids", [])
        ]
        available = set(revisions) | set(response_prompts)
        return _check(
            expectation,
            not rhea_commitments and "rhea" not in available,
            {
                "rhea_commitments": rhea_commitments,
                "pending": revisions,
                "response_prompts": response_prompts,
            },
        )

    if expectation == "no_blue_fog_leak_to_jun":
        response_texts = _render_texts_for_characters(response, {"jun", "mira"})
        if step.actor_id in {"jun", "mira"} and response.get("output_text"):
            response_texts.append(str(response.get("output_text")))
        leaks = [text for text in response_texts if "BLUE-FOG" in text.upper()]
        return _check(
            expectation,
            not leaks,
            {"leaks": leaks, "checked_texts": response_texts},
        )

    if expectation == "no_blue_fog_leak_to_mira_or_jun":
        response_texts = _render_texts_for_characters(response, {"mira", "jun"})
        leaks = [text for text in response_texts if "BLUE-FOG" in text.upper()]
        return _check(
            expectation,
            not leaks,
            {"leaks": leaks, "checked_texts": response_texts},
        )

    if expectation == "blue_fog_visible_only_to_rhea_cal":
        blue_facts: list[dict[str, Any]] = []
        bad: list[dict[str, Any]] = []
        allowed = {"rhea", "cal"}
        for event in new_events:
            for fact in event.get("facts") or []:
                if "BLUE-FOG" not in str(fact.get("text", "")).upper():
                    continue
                visible_to = _event_fact_visible_to(event, fact)
                entry = {
                    "event_id": event.get("event_id"),
                    "fact": fact,
                    "visible_to": sorted(visible_to),
                }
                blue_facts.append(entry)
                if visible_to - allowed:
                    bad.append(entry)
        return _check(
            expectation,
            bool(blue_facts) and not bad,
            {"blue_facts": blue_facts, "bad": bad},
        )

    if expectation == "actor_event_not_before_actor_clock":
        before_clock = int(
            (before.get("character_clocks") or {}).get(step.actor_id, 0) or 0
        )
        actor_events = [
            event for event in new_events
            if step.actor_id in _event_observer_ids(event)
            or any(
                update.get("character_id") == step.actor_id
                for update in event.get("location_updates") or []
            )
        ]
        bad = [
            event for event in actor_events
            if int(event.get("effective_at_s", 0) or 0) < before_clock
        ]
        return _check(
            expectation,
            bool(actor_events) and not bad,
            {"before_clock": before_clock, "actor_events": actor_events, "bad": bad},
        )

    if expectation == "long_wait_duration_encoded":
        actor_events = [
            event for event in new_events
            if step.actor_id in _event_observer_ids(event)
            or any(
                update.get("character_id") == step.actor_id
                for update in event.get("location_updates") or []
            )
        ]
        long_events = [
            event for event in actor_events
            if int(event.get("duration_s", 0) or 0) >= 300
        ]
        return _check(
            expectation,
            bool(long_events),
            {"actor_events": actor_events},
        )

    if expectation == "no_advanced_observer_backfill_without_revision":
        before_clocks = before.get("character_clocks") or {}
        available_revisions = set(revisions) | set(response_prompts)
        commitment_actors = {
            commitment.get("commitment_id"): set(commitment.get("actor_ids") or [])
            for commitment in before.get("open_commitments") or []
        }
        bad: list[dict[str, Any]] = []
        for event in new_events:
            event_end = int(event.get("effective_at_s", 0) or 0) + int(
                event.get("duration_s", 0) or 0
            )
            interrupted: set[str] = set()
            for signal in event.get("commitment_interrupts") or []:
                interrupted.update(str(cid) for cid in signal.get("actor_ids") or [])
                interrupted.update(
                    commitment_actors.get(signal.get("commitment_id"), set())
                )
            for observer_id in _event_observer_ids(event):
                if observer_id not in PLAYER_IDS or observer_id == step.actor_id:
                    continue
                before_clock = int(before_clocks.get(observer_id, 0) or 0)
                if before_clock <= event_end:
                    continue
                if (
                    observer_id in available_revisions
                    or observer_id in interrupted
                ):
                    continue
                bad.append({
                    "event_id": event.get("event_id"),
                    "observer_id": observer_id,
                    "observer_before_clock": before_clock,
                    "event_start": event.get("effective_at_s"),
                    "event_end": event_end,
                    "interrupted": sorted(interrupted),
                    "available_revisions": sorted(available_revisions),
                })
        return _check(expectation, not bad, {"bad": bad})

    return _check(expectation, False, "unknown progressive expectation")


def _evaluate_progressive_step(
    step: ProgressiveStep,
    step_report: dict[str, Any],
) -> list[dict[str, Any]]:
    before = step_report.get("checkpoint_before") or {}
    after = step_report.get("checkpoint_after") or {}
    response = step_report.get("response") or {}
    if step.slot_rejected_ok:
        turn_check = _check(
            "expected_slot_rejection",
            not step_report.get("error")
            and response.get("beat_ended_reason") == "slot_rejected",
            {
                "error": step_report.get("error", ""),
                "beat_ended_reason": response.get("beat_ended_reason"),
            },
        )
    else:
        turn_check = _check(
            "turn_completed",
            not step_report.get("error")
            and response.get("beat_ended_reason") != "slot_rejected",
            {
                "error": step_report.get("error", ""),
                "beat_ended_reason": response.get("beat_ended_reason"),
            },
        )
    checks = [
        turn_check,
        _progressive_leak_check(step_report),
        _progressive_clock_regression_check(before, after),
        _progressive_event_timing_check(step_report),
        _progressive_chronology_check(step_report),
    ]
    for expectation in step.expectations:
        checks.append(_progressive_expectation_check(
            expectation,
            step=step,
            step_report=step_report,
        ))
    return checks


async def _run_progressive_step(
    *,
    engine: EngineBridge,
    session_id: str,
    step: ProgressiveStep,
    role_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    settings_applied: dict[str, str] = {}
    for key, value in step.settings:
        settings_applied[key] = str(
            await _apply_progressive_setting(engine, session_id, key, value)
        )

    before = engine.load_latest(session_id)
    call_start = len(role_calls)
    started = time.perf_counter()
    try:
        response = await engine.run_turn(
            session_id=session_id,
            user_input=step.user_input,
            acting_character_id=step.actor_id,
        )
        elapsed_s = time.perf_counter() - started
        after = engine.load_latest(session_id)
        new_events = _new_events_for_step(before, after)
        step_report = {
            "phase": step.phase,
            "label": step.label,
            "actor_id": step.actor_id,
            "user_input": step.user_input,
            "settings_applied": settings_applied,
            "elapsed_s": elapsed_s,
            "response": _response_dump(response),
            "new_events": [_event_dump(event) for event in new_events],
            "checkpoint_before": _checkpoint_snapshot(before),
            "checkpoint_after": _checkpoint_snapshot(after),
            "role_calls": list(role_calls[call_start:]),
        }
    except Exception:
        elapsed_s = time.perf_counter() - started
        after = engine.load_latest(session_id)
        step_report = {
            "phase": step.phase,
            "label": step.label,
            "actor_id": step.actor_id,
            "user_input": step.user_input,
            "settings_applied": settings_applied,
            "elapsed_s": elapsed_s,
            "error": traceback.format_exc(),
            "new_events": [],
            "checkpoint_before": _checkpoint_snapshot(before),
            "checkpoint_after": _checkpoint_snapshot(after),
            "role_calls": list(role_calls[call_start:]),
        }
    step_report["checks"] = _evaluate_progressive_step(step, step_report)
    step_report["usage_totals"] = _usage_totals(step_report["role_calls"])
    step_report["role_counts"] = _role_counts(step_report["role_calls"])
    return step_report


def _progressive_analysis(step_reports: list[dict[str, Any]]) -> dict[str, Any]:
    phase_summary: dict[str, dict[str, Any]] = {}
    total_checks = 0
    passed_checks = 0
    total_elapsed_s = 0.0
    all_calls: list[dict[str, Any]] = []
    first_failure: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = []
    clock_spread: list[dict[str, Any]] = []

    for index, step in enumerate(step_reports, start=1):
        phase = step.get("phase") or ""
        phase_data = phase_summary.setdefault(
            phase,
            {"steps": 0, "checks": 0, "passed": 0, "elapsed_s": 0.0},
        )
        checks = step.get("checks") or []
        passed = sum(1 for check in checks if check.get("passed"))
        failed = [check for check in checks if not check.get("passed")]
        total = len(checks)
        phase_data["steps"] += 1
        phase_data["checks"] += total
        phase_data["passed"] += passed
        phase_data["elapsed_s"] += float(step.get("elapsed_s", 0.0) or 0.0)
        total_checks += total
        passed_checks += passed
        total_elapsed_s += float(step.get("elapsed_s", 0.0) or 0.0)
        all_calls.extend(step.get("role_calls") or [])
        if failed and first_failure is None:
            first_failure = {
                "step": index,
                "phase": phase,
                "label": step.get("label"),
                "failed_checks": failed,
            }
        if failed:
            failures.append({
                "step": index,
                "phase": phase,
                "label": step.get("label"),
                "failed_checks": failed,
            })
        after = step.get("checkpoint_after") or {}
        values = _player_clock_values(after)
        if values:
            clock_spread.append({
                "step": index,
                "phase": phase,
                "label": step.get("label"),
                "min": min(values),
                "max": max(values),
                "spread": max(values) - min(values),
                "leading_at_s": after.get("leading_at_s"),
            })

    return {
        "checks": {"passed": passed_checks, "total": total_checks},
        "phase_summary": phase_summary,
        "elapsed_s": total_elapsed_s,
        "role_counts": _role_counts(all_calls),
        "usage_totals": _usage_totals(all_calls),
        "first_failure": first_failure,
        "failures": failures,
        "clock_spread": clock_spread,
    }


async def _run_progressive(
    *,
    engine: EngineBridge,
    stories_dir: Path,
    role_calls: list[dict[str, Any]],
    max_steps: int,
    stop_on_fail: bool,
) -> dict[str, Any]:
    story_id = "relative_time_progressive"
    session_id = f"{story_id}_{TS.lower()}"
    _write_story(stories_dir, story_id)
    engine.create_empty_session(session_id)
    engine.load_story_into_session(session_id, story_id)
    for char_id, user_id in PLAYER_IDS.items():
        await engine.bind_user(session_id, user_id, char_id)

    selected_steps = list(PROGRESSIVE_STEPS)
    if max_steps > 0:
        selected_steps = selected_steps[:max_steps]

    step_reports: list[dict[str, Any]] = []
    for index, step in enumerate(selected_steps, start=1):
        print(
            f"running progressive step {index}/{len(selected_steps)}: "
            f"{step.phase} / {step.label}",
            flush=True,
        )
        report = await _run_progressive_step(
            engine=engine,
            session_id=session_id,
            step=step,
            role_calls=role_calls,
        )
        report["index"] = index
        step_reports.append(report)
        if stop_on_fail and any(not check["passed"] for check in report["checks"]):
            break

    return {
        "mode": "progressive",
        "name": "progressive_single_session",
        "description": (
            "One session that uses a minimal setup to create divergent clocks "
            "and private commitments, then broadens the failure suite across "
            "advanced-observer backfill, stale commitment resolution, private "
            "information leakage, backfill attempts, and "
            "late multi-target play."
        ),
        "session_id": session_id,
        "story_id": story_id,
        "players": list(PLAYER_IDS),
        "steps": step_reports,
        "analysis": _progressive_analysis(step_reports),
    }


def _progressive_markdown(progressive: dict[str, Any]) -> str:
    analysis = progressive["analysis"]
    checks = analysis["checks"]
    lines = [
        "# Progressive Relative-Time Live Harness",
        "",
        progressive["description"],
        "",
        "## Summary",
        "",
        f"- Session: `{progressive['session_id']}`",
        f"- Checks: `{checks['passed']}/{checks['total']}`",
        f"- Elapsed: `{analysis['elapsed_s']:.1f}s`",
        f"- Role calls: `{json.dumps(analysis['role_counts'], sort_keys=True)}`",
        f"- Usage totals: `{json.dumps(analysis['usage_totals'], sort_keys=True)}`",
        "",
        "## Phase Summary",
        "",
    ]
    for phase, data in analysis["phase_summary"].items():
        lines.append(
            f"- `{phase}`: {data['passed']}/{data['checks']} checks, "
            f"{data['steps']} step(s), {data['elapsed_s']:.1f}s"
        )
    lines.append("")

    first_failure = analysis.get("first_failure")
    if first_failure:
        lines.extend(["## First Failure", "```json"])
        lines.append(json.dumps(first_failure, indent=2))
        lines.extend(["```", ""])
    else:
        lines.extend(["## First Failure", "", "None.", ""])

    failures = analysis.get("failures") or []
    if failures:
        lines.extend(["## All Failures", ""])
        for failure in failures:
            names = ", ".join(
                f"`{check.get('name')}`"
                for check in failure.get("failed_checks", [])
            )
            lines.append(
                f"- Step {failure.get('step')} `{failure.get('phase')}` "
                f"{failure.get('label')}: {names}"
            )
        lines.append("")

    lines.extend(["## Complexity Over Time", ""])
    lines.append(
        "| # | Phase | Actor | Beat | Checks | Events | Clocks | Open / Pending | "
        "Calls | Seconds |"
    )
    lines.append(
        "|---|---|---|---|---:|---:|---|---|---:|---:|"
    )
    for step in progressive["steps"]:
        response = step.get("response") or {}
        after = step.get("checkpoint_after") or {}
        checks_passed = sum(1 for check in step["checks"] if check["passed"])
        checks_total = len(step["checks"])
        clocks = after.get("character_clocks") or {}
        clock_summary = ", ".join(
            f"{cid}:{clocks.get(cid, 0)}" for cid in PLAYER_IDS
        )
        open_count = len(after.get("open_commitments") or [])
        pending = ",".join(
            sorted((after.get("pending_commitment_revisions") or {}).keys())
        )
        lines.append(
            f"| {step['index']} | `{step['phase']}` | `{step['actor_id']}` | "
            f"`{response.get('beat_ended_reason', 'error')}` | "
            f"{checks_passed}/{checks_total} | "
            f"{len(step.get('new_events') or [])} | {clock_summary} | "
            f"{open_count} / {pending or '-'} | "
            f"{len(step.get('role_calls') or [])} | "
            f"{float(step.get('elapsed_s', 0.0)):.1f} |"
        )
    lines.append("")

    for step in progressive["steps"]:
        lines.extend([
            f"## Step {step['index']}: {step['label']}",
            "",
            f"Phase: `{step['phase']}`",
            f"Actor: `{step['actor_id']}`",
            f"Elapsed: `{float(step.get('elapsed_s', 0.0)):.1f}s`",
            f"Role calls: `{json.dumps(step.get('role_counts') or {}, sort_keys=True)}`",
            f"Usage: `{json.dumps(step.get('usage_totals') or {}, sort_keys=True)}`",
            "",
            f"> {step['user_input']}",
            "",
        ])
        if step.get("settings_applied"):
            lines.append(f"Settings: `{json.dumps(step['settings_applied'])}`")
            lines.append("")
        if step.get("error"):
            lines.extend(["```text", step["error"], "```", ""])
        response = step.get("response") or {}
        lines.append(f"Beat reason: `{response.get('beat_ended_reason', '')}`")
        prompts = response.get("commitment_revision_prompts") or {}
        if prompts:
            lines.append(f"Commitment prompts: `{json.dumps(prompts)}`")
        lines.append("")
        lines.append("Checks:")
        for check in step["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {mark}: `{check['name']}`")
        lines.append("")
        if step.get("new_events"):
            lines.append("New events:")
            for event in step["new_events"]:
                observers = ",".join(
                    observer["character_id"] for observer in event.get("observers", [])
                )
                lines.append(
                    f"- `{event['event_id']}` kind="
                    f"`{event['event_kind']}` "
                    f"t={event['effective_at_s']}+{event['duration_s']} "
                    f"observers={observers}"
                )
            lines.append("")
        failed = [check for check in step["checks"] if not check["passed"]]
        if failed:
            lines.extend(["Failed check details:", "```json"])
            lines.append(json.dumps(failed, indent=2))
            lines.extend(["```", ""])
    return "\n".join(lines)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Relative Time And Commitment Live Harness",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Run dir: `{report['run_dir']}`",
        f"Router: `{report['roles']['event_router']}`",
        f"Narrator: `{report['roles']['narrator']}`",
        f"Agent: `{report['roles']['agent']}`",
        "",
        "## Summary",
        "",
    ]
    for case in report["cases"]:
        passed = sum(1 for check in case["checks"] if check["passed"])
        total = len(case["checks"])
        lines.append(f"- `{case['name']}`: {passed}/{total}")
    lines.append("")

    for case in report["cases"]:
        lines.extend([f"## {case['name']}", "", case["description"], ""])
        lines.append("Checks:")
        for check in case["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {mark}: `{check['name']}`")
        lines.extend(["", "### Transcript", ""])
        for index, step in enumerate(case.get("steps", []), start=1):
            lines.append(
                f"#### Step {index}: {step.get('actor_id')} - "
                f"{step.get('label', '')}"
            )
            lines.append("")
            lines.append(f"> {step.get('user_input', '')}")
            lines.append("")
            if step.get("error"):
                lines.extend(["```text", step["error"], "```", ""])
                continue
            response = step.get("response") or {}
            lines.append(f"Beat reason: `{response.get('beat_ended_reason', '')}`")
            prompts = response.get("commitment_revision_prompts") or {}
            if prompts:
                lines.append(f"Commitment prompts: `{json.dumps(prompts)}`")
            if response.get("output_text"):
                lines.extend(["", "```text", response["output_text"], "```"])
            if response.get("per_player_renders"):
                lines.extend(["", "Per-player renders:"])
                for cid, render in response["per_player_renders"].items():
                    lines.extend([f"- `{cid}`", "```text", render, "```"])
            if step.get("new_events"):
                lines.extend(["", "New events:"])
                for event in step["new_events"]:
                    lines.append(
                        f"- `{event['event_id']}` kind="
                        f"`{event['event_kind']}` "
                        f"t={event['effective_at_s']}+{event['duration_s']}"
                    )
            lines.append("")

        final = _final_snapshot(case)
        lines.extend(["### Final State", "```json"])
        lines.append(json.dumps({
            "leading_at_s": final.get("leading_at_s"),
            "character_clocks": final.get("character_clocks"),
            "open_commitments": final.get("open_commitments"),
            "pending_commitment_revisions": final.get(
                "pending_commitment_revisions"
            ),
            "open_cat_ii_events": final.get("open_cat_ii_events"),
        }, indent=2))
        lines.extend(["```", ""])

        failed = [check for check in case["checks"] if not check["passed"]]
        if failed:
            lines.extend(["### Failed Check Details", "```json"])
            lines.append(json.dumps(failed, indent=2))
            lines.extend(["```", ""])
    return "\n".join(lines)


def _preflight_api_keys(config: LLMConfig, roles: set[str]) -> list[str]:
    return [
        f"{item.role} ({item.provider}; set one of {', '.join(item.env_names)})"
        for item in config.missing_credentials(roles)
    ]


def _role_label(config: LLMConfig, role: str) -> str:
    return f"{config.provider_for_role(role)}:{config.model_for_role(role)}"


def _case_by_name(name: str) -> LiveCase:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)


async def main() -> None:
    parser = argparse.ArgumentParser()
    choices = ["all", "quick", "progressive", *[case.name for case in CASES]]
    parser.add_argument("--case", choices=choices, default="all")
    parser.add_argument(
        "--progressive",
        action="store_true",
        help=(
            "Run one cumulative single-session scenario instead of the "
            "independent focused cases."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Progressive mode only: stop after this many scripted steps.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Progressive mode only: stop after the first failing step.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List scenario names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("progressive: one cumulative single-session stress run")
        for case in CASES:
            default = " default" if case.default else ""
            print(f"{case.name}{default}: {case.description}")
        return

    load_dotenv()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    progressive_mode = args.progressive or args.case == "progressive"

    if progressive_mode:
        selected: list[LiveCase] = []
    elif args.case == "all":
        selected = [case for case in CASES if case.default]
    elif args.case == "quick":
        selected = [
            _case_by_name("open_commitment_private"),
            _case_by_name("interrupted_commitment_revision"),
            _case_by_name("cat_ii_resolution_time"),
        ]
    else:
        selected = [_case_by_name(args.case)]

    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    (RUN_DIR / "synthetic_story_prompt.md").write_text(
        prompt_text,
        encoding="utf-8",
    )

    config = LLMConfig.from_env()
    roles = {"event_router", "narrator", "agent"}
    missing = _preflight_api_keys(config, roles)
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
    role_calls: list[dict[str, Any]] = []
    real_complete: Callable[..., Any] = engine.client.complete

    async def _recording_complete(*call_args, **kwargs):
        role = kwargs.get("role") or (call_args[0] if call_args else "")
        response_model = kwargs.get("response_model")
        entry: dict[str, Any] = {
            "role": str(role),
            "response_model": response_model.__name__ if response_model else "",
        }
        started = time.perf_counter()
        try:
            response = await real_complete(*call_args, **kwargs)
        except Exception as exc:
            entry["elapsed_s"] = time.perf_counter() - started
            entry["error"] = repr(exc)
            role_calls.append(entry)
            raise
        entry["elapsed_s"] = time.perf_counter() - started
        entry["usage"] = dict(getattr(response, "usage", {}) or {})
        role_calls.append(entry)
        return response

    engine.client.complete = _recording_complete  # type: ignore[method-assign]

    try:
        if progressive_mode:
            progressive_report = await _run_progressive(
                engine=engine,
                stories_dir=stories_dir,
                role_calls=role_calls,
                max_steps=args.max_steps,
                stop_on_fail=args.stop_on_fail,
            )
            report = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(RUN_DIR),
                "synthetic_story_prompt": str(
                    RUN_DIR / "synthetic_story_prompt.md"
                ),
                "roles": {
                    "event_router": _role_label(config, "event_router"),
                    "narrator": _role_label(config, "narrator"),
                    "agent": _role_label(config, "agent"),
                },
                "progressive": progressive_report,
            }
            JSON_PATH.write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )
            MD_PATH.write_text(
                _progressive_markdown(progressive_report),
                encoding="utf-8",
            )
            print(JSON_PATH)
            print(MD_PATH)
            checks = progressive_report["analysis"]["checks"]
            print(f"progressive: {checks['passed']}/{checks['total']}")
            if checks["passed"] != checks["total"]:
                raise SystemExit(1)
            return

        case_reports: list[dict[str, Any]] = []
        for case in selected:
            print(f"running {case.name}...", flush=True)
            before = len(role_calls)
            try:
                case_reports.append(
                    await _run_case(
                        case,
                        engine=engine,
                        stories_dir=stories_dir,
                        role_calls=role_calls,
                        role_call_start=before,
                    )
                )
            except Exception:
                case_reports.append(
                    {
                        "name": case.name,
                        "description": case.description,
                        "checks": [
                            _check(
                                "case_completed",
                                False,
                                traceback.format_exc(),
                            )
                        ],
                        "steps": [],
                        "role_calls": list(role_calls[before:]),
                    }
                )
    finally:
        await engine.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(RUN_DIR),
        "synthetic_story_prompt": str(RUN_DIR / "synthetic_story_prompt.md"),
        "roles": {
            "event_router": _role_label(config, "event_router"),
            "narrator": _role_label(config, "narrator"),
            "agent": _role_label(config, "agent"),
        },
        "cases": case_reports,
    }
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(_markdown(report), encoding="utf-8")

    print(JSON_PATH)
    print(MD_PATH)
    failed = False
    for case in case_reports:
        passed = sum(1 for check in case["checks"] if check["passed"])
        total = len(case["checks"])
        print(f"{case['name']}: {passed}/{total}")
        if passed != total:
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
