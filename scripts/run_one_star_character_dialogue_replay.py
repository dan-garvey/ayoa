#!/usr/bin/env python3
# ruff: noqa: E402 -- executable script adds the repository root to sys.path.
"""Replay sustained One-Star CharacterAgent conversations in isolation.

Each candidate-fed conversation starts from an explicitly supplied tracked
checkpoint and uses the production CharacterAgent prompt builder and commit
path.  Every later prompt therefore sees the candidate-authored public beat
produced immediately before it.  Sustained quiet and conflict transcripts are
reviewed as whole conversations; compact danger cases remain regression checks.
All semantic judgments belong to a human reviewer.

Importing the module and the default command are offline.  ``--live`` is the
only path that calls the configured provider.  Every run saves raw prompts,
responses, timings, and usage.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.engine.character_agent import CharacterAgent
from app.engine.context_builder import (
    build_character_packet,
    build_character_state,
    build_world_context,
)
from app.engine.prompt_manager import PromptManager
from app.schemas.checkpoint import CheckpointFile
from scripts.run_one_star_dialogue_ab import (
    TERRA_MODEL,
    _CapturedClient,
    _write_json,
    _write_jsonl,
)


PRODUCTION_SEED_PATH = (
    REPO_ROOT
    / "app"
    / "storage"
    / "stories"
    / "one_star_ascension_s1"
    / "ckpt_0000.json"
)


@dataclass(frozen=True)
class ConversationScenario:
    """One candidate-fed conversation anchored in tracked checkpoint state."""

    scenario_id: str
    scenario_kind: str
    actor_ids: tuple[str, ...]
    tracked_state_markers: tuple[str, ...]
    setup_observation: str
    review_purpose: str


_RENNA_MIRELLE_EIGHT_TURNS = (
    "renna_holt",
    "mirelle_voss",
    "renna_holt",
    "mirelle_voss",
    "renna_holt",
    "mirelle_voss",
    "renna_holt",
    "mirelle_voss",
)
_MIRELLE_RENNA_EIGHT_TURNS = (
    "mirelle_voss",
    "renna_holt",
    "mirelle_voss",
    "renna_holt",
    "mirelle_voss",
    "renna_holt",
    "mirelle_voss",
    "renna_holt",
)

SUSTAINED_CONVERSATION_SCENARIOS: tuple[ConversationScenario, ...] = (
    ConversationScenario(
        scenario_id="quiet_post_clear_memory",
        scenario_kind="quiet",
        actor_ids=_RENNA_MIRELLE_EIGHT_TURNS,
        tracked_state_markers=(
            "A tangible surge of hard-won strength settles through",
            "Floor Two is unlocked",
        ),
        setup_observation=(
            "With the gate closed and no one in danger, Renna Holt places three "
            "practice arrows close together in the same scarred timber. Mirelle "
            'Voss studies the grouping and asks, "May I count on that on Floor '
            'Two?"'
        ),
        review_purpose=(
            "Review whether a request for future reliance develops the skill's "
            "relief, shame, bodily memory, and social cost without diagnosing "
            "Renna for her."
        ),
    ),
    ConversationScenario(
        scenario_id="quiet_mending_and_work",
        scenario_kind="quiet",
        actor_ids=_RENNA_MIRELLE_EIGHT_TURNS,
        tracked_state_markers=(
            "niflheim_lobby beneath the faceless idol",
            "The Tower gate stands closed and dark again",
        ),
        setup_observation=(
            "No gate is open and no one is bleeding. Mirelle Voss has already "
            "put one conspicuous red stitch through Renna Holt's split bracer "
            "without asking. A bent needle and coarse gray cord lie untouched "
            "beside it."
        ),
        review_purpose=(
            "Review whether Renna contests or accepts a visible favor and whether "
            "ownership, taste, work habits, and status change across the repair."
        ),
    ),
    ConversationScenario(
        scenario_id="conflict_bargain_authority",
        scenario_kind="conflict",
        actor_ids=_MIRELLE_RENNA_EIGHT_TURNS,
        tracked_state_markers=(
            "I gave my word on the bell",
            "If that bell moves, I’ll hear it",
        ),
        setup_observation=(
            "Edren sets the hooked pole at Renna Holt's feet instead of Mirelle "
            'Voss\'s. He says, "At the barricade, Mirelle promised safety and all '
            "three of us paid for it. Next time she makes a promise for the "
            "party, I wait for Renna. If Mirelle wants it sooner, she covers the "
            'price alone." He leaves them with the pole.'
        ),
        review_purpose=(
            "Review whether lost deference and unwanted transferred authority "
            "change across eight turns without instant consensus, a moral "
            "summary, or reset banter."
        ),
    ),
)

PRESSURE_REGRESSION_SCENARIOS: tuple[ConversationScenario, ...] = (
    ConversationScenario(
        scenario_id="goblin_separate_escape",
        scenario_kind="pressure",
        actor_ids=("mirelle_voss", "renna_holt", "mirelle_voss"),
        tracked_state_markers=(
            "barred section of the timber barricade",
            "rubble-strewn side path",
        ),
        setup_observation=(
            "The first goblin points to the rubble-strewn shortcut, then to a "
            "low drainage culvert beneath the barricade. The goblin says, "
            '"Cover us until we reach the culvert. You take the rubble path; '
            "we crawl out the other way. The routes don't meet. Deal?\""
        ),
        review_purpose=(
            "Test direct handling of a cover-for-shortcut bargain while the "
            "party and deserters take explicit nonjoining routes."
        ),
    ),
    ConversationScenario(
        scenario_id="barricade_first_contact",
        scenario_kind="pressure",
        actor_ids=("mirelle_voss", "renna_holt", "mirelle_voss"),
        tracked_state_markers=(
            "waist-high timber barricade",
            "one tugs the bell rope again",
        ),
        setup_observation=(
            "The bell rope jerks above the timber while the second armed figure "
            "stays low; neither goblin has spoken yet."
        ),
        review_purpose=(
            "Test immediate action/referent continuity under threat without a "
            "social or flirtatious setup."
        ),
    ),
)

REPLAY_SCENARIOS = (
    *SUSTAINED_CONVERSATION_SCENARIOS,
    *PRESSURE_REGRESSION_SCENARIOS,
)

CONVERSATION_REVIEW_CONTRACT: dict[str, Any] = {
    "version": "one_star_sustained_conversation_v1",
    "reviewer": "human",
    "model_judge": False,
    "unit_of_review": (
        "Read each sustained transcript from setup through turn eight as one "
        "conversation. Transition rows and compact danger cases are diagnostic; "
        "they cannot establish acceptance from isolated lines."
    ),
    "sustained_scenario_requirements": {
        "quiet_scenarios": "at least two, each at least eight turns",
        "conflict_scenarios": "at least one, at least eight turns",
        "grounding": (
            "explicit tracked checkpoint markers plus the current seed profile"
        ),
    },
    "whole_conversation_dimensions": [
        {
            "id": "public_established_backstory_and_depth",
            "question": (
                "Does public dialogue establish character history, ordinary "
                "experience, values, or vulnerability grounded in the supplied "
                "profile, and does any disclosure affect later turns?"
            ),
            "values": ["pass", "fail", "uncertain"],
        },
        {
            "id": "character_specific_cadence_and_attention",
            "question": (
                "Across all eight turns, does each speaker keep a recognizable "
                "cadence and attend to details this character would notice, "
                "rather than sharing one generic clever voice?"
            ),
            "values": ["pass", "fail", "uncertain"],
        },
        {
            "id": "contradictions_change_later_turns",
            "question": (
                "When a claim, refusal, disclosure, or contradiction appears, "
                "do later turns remember and react to it instead of resetting "
                "the relationship or silently smoothing it away?"
            ),
            "values": ["pass", "fail", "uncertain"],
        },
        {
            "id": "non_neat_conflict",
            "question": (
                "In the conflict scenario, do incompatible stakes remain human "
                "and consequential without instant consensus, a tidy moral, or "
                "a disagreement performed only to be resolved?"
            ),
            "values": ["pass", "fail", "not_applicable", "uncertain"],
        },
        {
            "id": "fact_fidelity",
            "question": (
                "Are public claims consistent with tracked checkpoint events, "
                "the current seed profile, and facts established earlier in the "
                "same conversation, with uncertainty preserved where required?"
            ),
            "values": ["pass", "fail", "uncertain"],
        },
        {
            "id": "voice_swappability",
            "question": (
                "After reading the separate blinded transcript, could the two "
                "speakers' lines be exchanged without damaging cadence, "
                "attention, history, or social behavior?"
            ),
            "values": ["distinct", "swappable", "uncertain"],
        },
        {
            "id": "subtext_authority_support",
            "question": (
                "Can the inferred interpersonal attempt and unavailable direct "
                "sentence be supported by seeded character authority, tracked "
                "state, or an earlier turn rather than post-hoc interpretation?"
            ),
            "values": ["pass", "fail", "uncertain"],
        },
        {
            "id": "conversation_changes_something",
            "question": (
                "By turn eight, has information, status, obligation, ritual, or "
                "relationship changed in a way the next scene must carry, rather "
                "than producing longer therapy dialogue that changes nothing?"
            ),
            "values": ["pass", "fail", "uncertain"],
        },
    ],
    "pressure_regression_review": {
        "purpose": (
            "Keep immediate uptake, direct-question handling, concrete danger "
            "action, and fact fidelity from regressing under time pressure."
        ),
        "acceptance_role": "diagnostic only; never substitutes for sustained review",
    },
    "decision_policy": (
        "No automated semantic score and no model judge. A human records exact "
        "whole-conversation evidence for every dimension; fact-fidelity failures "
        "reject the run, and isolated attractive lines cannot pass it."
    ),
    "supersedes": ("conversation-review-v3 and its 4/3/3-turn acceptance claim"),
}


def validate_scenario_inventory(
    scenarios: Iterable[ConversationScenario],
) -> None:
    """Enforce sustained review depth separately from pressure coverage."""
    materialized = tuple(scenarios)
    scenario_ids = [scenario.scenario_id for scenario in materialized]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Conversation scenario ids must be unique")

    quiet = [scenario for scenario in materialized if scenario.scenario_kind == "quiet"]
    conflicts = [
        scenario for scenario in materialized if scenario.scenario_kind == "conflict"
    ]
    pressure = [
        scenario for scenario in materialized if scenario.scenario_kind == "pressure"
    ]
    unknown_kinds = sorted(
        {
            scenario.scenario_kind
            for scenario in materialized
            if scenario.scenario_kind not in {"quiet", "conflict", "pressure"}
        }
    )
    if unknown_kinds:
        raise ValueError(f"Unknown conversation scenario kinds: {unknown_kinds}")
    if len(quiet) < 2 or any(len(scenario.actor_ids) < 8 for scenario in quiet):
        raise ValueError("Sustained review requires two quiet eight-turn scenarios")
    if not conflicts or any(len(scenario.actor_ids) < 8 for scenario in conflicts):
        raise ValueError("Sustained review requires an eight-turn conflict")
    if not pressure:
        raise ValueError("At least one compact danger regression is required")
    if any(len(scenario.actor_ids) > 3 for scenario in pressure):
        raise ValueError("Danger regressions must remain compact")


validate_scenario_inventory(REPLAY_SCENARIOS)

_TRAILING_INTENT_RE = re.compile(r"\((?P<intent>[^()]*)\)\s*$", re.DOTALL)
_IDENTITY_BLOCK_RE = re.compile(
    r"<character_identity>.*?</character_identity>",
    re.DOTALL,
)
_CURRENT_STATE_BLOCK_RE = re.compile(
    r"<current_state>.*?</current_state>",
    re.DOTALL,
)
_PRESENTATION_FOOTER_RE = re.compile(
    r"<presentation>.*?</presentation>",
    re.DOTALL,
)
_DIALOGUE_RE = re.compile(r'[“"](?P<dialogue>[^”"\n]+)[”"]')
_OTHER_NAME_CUE_RE = re.compile(
    r"\b(?:Edren Marr|Edren)(?P<possessive>['’]s)?\b",
    re.IGNORECASE,
)
_VOICE_WEAPON_ACTION_CUE_RE = re.compile(
    r"\b(?:nocks?(?:\s+an?\s+arrow)?|"
    r"looses?(?:\s+an?\s+arrow)?|"
    r"shoots?(?:\s+an?\s+arrow)?|"
    r"fires?(?:\s+an?\s+arrow)?|"
    r"draws?(?=\s+(?:the\s+)?(?:bow|bowstring|arrow|until)))\b",
    re.IGNORECASE,
)
_VOICE_WEAPON_CUE_RE = re.compile(
    r"\b(?:hooked laundry pole|hooked pole|long spear|spear shaft|"
    r"red shaft|cheap ash|longbow|shortbow|crossbow|bowstring|bow hand|"
    r"arrowshaft|arrowhead|arrows?|spearhead|spearpoint|spears?|polearm|"
    r"bow|shaft|pole)\b",
    re.IGNORECASE,
)


class _ReplayCapturedClient(_CapturedClient):
    """Capture the production call while explicitly disabling compaction."""

    async def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        kwargs["compact"] = False
        return await super().complete(
            role=role,
            messages=messages,
            **kwargs,
        )


def load_checkpoint(path: Path) -> CheckpointFile:
    """Load one checkpoint without mutating the source artifact."""
    return CheckpointFile.model_validate_json(path.read_text(encoding="utf-8"))


def _character(checkpoint: CheckpointFile, actor_id: str) -> Any:
    return next(
        character
        for character in checkpoint.characters
        if character.character_id == actor_id
    )


def _render_profile_blocks(
    profile_checkpoint: CheckpointFile,
    actor_id: str,
    *,
    prompt_manager: PromptManager | None = None,
) -> tuple[str, str]:
    """Render identity/private-state blocks through the production template."""
    manager = prompt_manager or PromptManager()
    actor = _character(profile_checkpoint, actor_id)
    messages = manager.render_messages(
        "agent",
        agent_ruleset_system_addon=manager.render("agent_ruleset_one_star").strip(),
        **build_character_packet(actor, profile_checkpoint),
        **build_character_state(actor, profile_checkpoint),
        world_context=build_world_context(actor, profile_checkpoint),
        pending_observations_block="",
        mode_header="## AGENT-TURN",
        mode_block="## Turn Frame\nforeground",
    )
    user_content = messages[-1]["content"]
    identity_match = _IDENTITY_BLOCK_RE.search(user_content)
    state_match = _CURRENT_STATE_BLOCK_RE.search(user_content)
    if identity_match is None or state_match is None:
        raise ValueError(f"Could not render current profile blocks for {actor_id}")
    return identity_match.group(0), state_match.group(0)


def _replace_profile_blocks(
    content: str,
    *,
    identity_block: str,
    state_block: str,
) -> str:
    """Replace blocks in-place without adding, dropping, or reordering messages."""
    updated, identity_count = _IDENTITY_BLOCK_RE.subn(identity_block, content)
    updated, state_count = _CURRENT_STATE_BLOCK_RE.subn(state_block, updated)
    if identity_count != 1 or state_count != 1:
        raise ValueError(
            "Persisted CharacterAgent user message must contain exactly one "
            "identity block and one current-state block"
        )
    return updated


def overlay_current_seed_profiles(
    checkpoint: CheckpointFile,
    profile_checkpoint: CheckpointFile,
    *,
    actor_ids: tuple[str, ...] = ("renna_holt", "mirelle_voss"),
) -> CheckpointFile:
    """Overlay current seed identity fields without changing message topology.

    This developer replay deliberately includes ``known_context`` because it is
    prompt-facing identity authority and tracked checkpoints retain the seed
    version that existed when they were written.  This is an in-memory test
    overlay, not a mutation of the source checkpoint or runtime state.
    """
    for actor_id in actor_ids:
        target = _character(checkpoint, actor_id)
        source = _character(profile_checkpoint, actor_id)
        target.public_sheet.role = source.public_sheet.role
        target.backstory = source.backstory
        target.personality = source.personality
        target.known_context = source.known_context
        target.private_state.goals = copy.deepcopy(source.private_state.goals)
        target.private_state.current_objectives = copy.deepcopy(
            source.private_state.current_objectives
        )
        target.private_state.secrets = copy.deepcopy(source.private_state.secrets)

        identity_block, state_block = _render_profile_blocks(
            checkpoint,
            actor_id,
        )
        for message in checkpoint.character_conversations.get(actor_id, []):
            if message.role != "user":
                continue
            if not isinstance(message.content, str):
                raise ValueError("CharacterAgent user history must be plain text")
            message.content = _replace_profile_blocks(
                message.content,
                identity_block=identity_block,
                state_block=state_block,
            )
    return checkpoint


def split_public_intent(text: str) -> tuple[str, str]:
    """Split the final private intent for review metrics without engine mutation."""
    match = _TRAILING_INTENT_RE.search(text or "")
    if match is None:
        return (text or "").strip(), ""
    return (text or "")[: match.start()].rstrip(), match.group("intent").strip()


def public_prose(text: str) -> str:
    """Return only observable prose, excluding presentation and private intent."""
    public, _intent = split_public_intent(text)
    return _PRESENTATION_FOOTER_RE.sub("", public).strip()


def extract_direct_questions(text: str) -> list[str]:
    """Extract literal question spans without judging how they were handled."""
    prose = public_prose(text)
    quoted_questions: list[str] = []
    for dialogue_match in _DIALOGUE_RE.finditer(prose):
        dialogue = dialogue_match.group("dialogue")
        for question_match in re.finditer(r"[^.!?\n]{1,400}\?", dialogue):
            question = question_match.group(0).strip()
            if question:
                quoted_questions.append(question)
    if quoted_questions:
        return quoted_questions

    questions: list[str] = []
    for match in re.finditer(r"[^.!?\n]{1,400}\?", prose):
        question = match.group(0).strip().strip('"“”')
        if question:
            questions.append(question)
    return questions


def _system_prompt_sha256(messages: list[dict[str, Any]]) -> str:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("CharacterAgent replay call must begin with a system message")
    content = str(messages[0].get("content") or "")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _validate_scenario_checkpoint(
    checkpoint: CheckpointFile,
    scenario: ConversationScenario,
) -> None:
    """Fail loudly when a scenario path does not contain its tracked state."""
    serialized = checkpoint.model_dump_json()
    missing = [
        marker for marker in scenario.tracked_state_markers if marker not in serialized
    ]
    if missing:
        raise ValueError(
            f"Scenario {scenario.scenario_id!r} checkpoint is missing tracked "
            f"state markers: {missing}"
        )


def build_whole_conversation_review(
    rows: Iterable[Mapping[str, Any]],
    scenarios: Iterable[ConversationScenario],
) -> list[dict[str, Any]]:
    """Build blank human review sheets around complete sustained exchanges."""
    rows_by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_scenario.setdefault(str(row["scenario_id"]), []).append(row)

    reviews: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_rows = sorted(
            rows_by_scenario.get(scenario.scenario_id, []),
            key=lambda row: int(row["scenario_turn"]),
        )
        if len(scenario_rows) != len(scenario.actor_ids):
            raise ValueError(
                f"Scenario {scenario.scenario_id!r} has "
                f"{len(scenario_rows)} rows; expected {len(scenario.actor_ids)}"
            )
        turns = []
        for row in scenario_rows:
            turns.append(
                {
                    "turn": int(row["scenario_turn"]),
                    "actor_id": row["actor_id"],
                    "prior_actor_id": row["prior_actor_id"],
                    "prior_public": row["prior_public"],
                    "public_text": row["parsed"]["public_text"],
                    "manual_turn_review": {
                        "literal_topic": "",
                        "interpersonal_attempt": "",
                        "information_or_epistemology_used": "",
                        "why_optimal_sentence_is_unavailable": "",
                        "status_shift": {
                            "topic_control": "",
                            "answer_debt": "",
                            "interruption_or_repair": "",
                        },
                        "ritual_or_meaningful_deviation": "",
                        "rhythm_change": "",
                        "debt_or_consequence_into_next_turn": "",
                        "authority_support": "",
                        "exact_evidence": "",
                    },
                }
            )
        reviews.append(
            {
                "review_id": f"{scenario.scenario_id}-whole-conversation",
                "scenario_id": scenario.scenario_id,
                "scenario_kind": scenario.scenario_kind,
                "checkpoint_path": scenario_rows[0]["checkpoint_path"],
                "profile_checkpoint_path": scenario_rows[0]["profile_checkpoint_path"],
                "tracked_state_markers": list(scenario.tracked_state_markers),
                "setup_observation": scenario.setup_observation,
                "review_purpose": scenario.review_purpose,
                "turns": turns,
                "manual_whole_conversation_review": {
                    "public_established_backstory_and_depth": "",
                    "character_specific_cadence_and_attention": "",
                    "contradictions_change_later_turns": "",
                    "non_neat_conflict": (
                        "" if scenario.scenario_kind == "conflict" else "not_applicable"
                    ),
                    "fact_fidelity": "",
                    "voice_swappability": "",
                    "subtext_authority_support": "",
                    "conversation_changes_something": "",
                    "speaker_interpersonal_attempts": {
                        actor_id: "" for actor_id in dict.fromkeys(scenario.actor_ids)
                    },
                    "established_conversational_ritual": "",
                    "meaningful_deviations": [],
                    "ending_debt_or_consequence": "",
                    "exact_evidence": [],
                },
            }
        )
    return reviews


def build_pressure_review_rows(
    rows: list[Mapping[str, Any]],
    scenario: ConversationScenario,
) -> list[dict[str, Any]]:
    """Pair compact danger transitions with blank regression-review fields."""
    review_rows: list[dict[str, Any]] = []
    prior_actor_id = "tracked_setup"
    prior_public = scenario.setup_observation
    for row in rows:
        prior_questions = extract_direct_questions(prior_public)
        review_rows.append(
            {
                "review_id": (
                    f"{scenario.scenario_id}-pressure-{int(row['scenario_turn']):02d}"
                ),
                "scenario_id": scenario.scenario_id,
                "scenario_turn": row["scenario_turn"],
                "checkpoint_path": row["checkpoint_path"],
                "profile_checkpoint_path": row["profile_checkpoint_path"],
                "tracked_state_markers": list(scenario.tracked_state_markers),
                "review_purpose": scenario.review_purpose,
                "prior_actor_id": prior_actor_id,
                "prior_public": prior_public,
                "prior_direct_questions": prior_questions,
                "current_actor_id": row["actor_id"],
                "current_public": row["parsed"]["public_text"],
                "manual_review": {
                    "immediate_uptake": "",
                    "referent_thread_continuity": "",
                    "direct_question_handling": (
                        "" if prior_questions else "not_applicable"
                    ),
                    "concrete_danger_action": "",
                    "fact_fidelity": "",
                    "exact_evidence": "",
                },
            }
        )
        prior_actor_id = str(row["actor_id"])
        prior_public = str(row["parsed"]["public_text"])
    return review_rows


def build_blinded_voice_review(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build blinded whole-scenario transcripts and a traceability key."""
    materialized = list(rows)
    actor_ids = {str(row["actor_id"]) for row in materialized}
    if len(actor_ids) != 2:
        raise ValueError("Voice review supports exactly two compared actors")
    ordered_actor_ids = sorted(
        actor_ids,
        key=lambda actor_id: hashlib.sha256(
            f"voice-pseudonym\0{actor_id}".encode("utf-8")
        ).hexdigest(),
    )
    pseudonym_by_actor = {
        actor_id: label
        for actor_id, label in zip(ordered_actor_ids, ("A", "B"), strict=True)
    }

    def redact(text: str) -> str:
        redacted = text
        for actor_id, label in pseudonym_by_actor.items():
            display_name = actor_id.replace("_", " ").title()
            first_name = display_name.split()[0]
            actor_name_re = re.compile(
                rf"\b(?:{re.escape(display_name)}|{re.escape(first_name)})"
                r"(?P<possessive>['’]s)?\b",
                re.IGNORECASE,
            )
            redacted = actor_name_re.sub(
                lambda match: (f"Speaker {label}{match.group('possessive') or ''}"),
                redacted,
            )
        redacted = _OTHER_NAME_CUE_RE.sub(
            lambda match: f"[other]{match.group('possessive') or ''}",
            redacted,
        )
        redacted = _VOICE_WEAPON_ACTION_CUE_RE.sub(
            "[weapon action]",
            redacted,
        )
        return _VOICE_WEAPON_CUE_RE.sub("[weapon]", redacted)

    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in materialized:
        by_scenario.setdefault(str(row["scenario_id"]), []).append(row)

    prepared: list[dict[str, Any]] = []
    for scenario_id, scenario_rows in by_scenario.items():
        scenario_rows.sort(key=lambda row: int(row["scenario_turn"]))
        first = scenario_rows[0]
        setup = str(first.get("setup_observation") or first.get("prior_public") or "")
        turns = []
        present_actor_ids: set[str] = set()
        for row in scenario_rows:
            actor_id = str(row["actor_id"])
            present_actor_ids.add(actor_id)
            parsed = row.get("parsed")
            if isinstance(parsed, Mapping):
                public_text = str(parsed.get("public_text") or "")
            else:
                public_text = public_prose(str(row["response"]))
            turns.append(
                {
                    "turn": int(row["scenario_turn"]),
                    "speaker": pseudonym_by_actor[actor_id],
                    "text": redact(public_text),
                }
            )
        prepared.append(
            {
                "stable_shuffle_key": hashlib.sha256(
                    f"voice-scenario\0{scenario_id}".encode("utf-8")
                ).hexdigest(),
            "scenario_id": scenario_id,
            "shared_context": redact(setup),
            "turns": turns,
            "speaker_map": {
                    pseudonym_by_actor[actor_id]: actor_id
                    for actor_id in sorted(present_actor_ids)
                },
            }
        )

    prepared.sort(key=lambda item: item["stable_shuffle_key"])
    samples: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for index, item in enumerate(prepared, start=1):
        blind_id = f"voice-{index:03d}"
        samples.append(
            {
                "blind_id": blind_id,
                "sample_kind": "blinded_sustained_conversation",
                "shared_context": item["shared_context"],
                "turns": item["turns"],
                "manual_review": {
                    "speaker_a_cadence_attention_notes": "",
                    "speaker_b_cadence_attention_notes": "",
                    "voice_swappability": "",
                    "interpersonal_attempt_and_status_evidence": "",
                    "exact_evidence": "",
                },
            }
        )
        answer_key.append(
            {
                "blind_id": blind_id,
                "scenario_id": item["scenario_id"],
                "speaker_map": item["speaker_map"],
            }
        )
    return samples, answer_key


async def _run_conversation_scenario(
    checkpoint: CheckpointFile,
    client: _ReplayCapturedClient,
    *,
    phase: str,
    checkpoint_path: Path,
    profile_checkpoint_path: Path,
    scenario: ConversationScenario,
) -> list[dict[str, Any]]:
    _validate_scenario_checkpoint(checkpoint, scenario)
    if scenario.setup_observation:
        for actor_id in dict.fromkeys(scenario.actor_ids):
            _character(checkpoint, actor_id).pending_observations.append(
                scenario.setup_observation
            )

    rows: list[dict[str, Any]] = []
    agent = CharacterAgent(client, PromptManager())
    previous_actor_id = ""
    previous_public = ""
    for scenario_turn, actor_id in enumerate(scenario.actor_ids, start=1):
        actor = _character(checkpoint, actor_id)
        incoming_prior_actor_id = previous_actor_id or "tracked_setup"
        incoming_prior_public = previous_public or scenario.setup_observation
        injected_observation = ""
        if previous_public:
            injected_observation = previous_public
            actor.pending_observations.append(injected_observation)
        history_before = len(checkpoint.character_conversations.get(actor_id, []))
        started = time.perf_counter()
        draft = await agent.draft_turn(actor, checkpoint, frame="foreground")
        calls = copy.deepcopy(client.calls)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        raw_response = calls[-1]["response"]
        agent.commit_draft(actor, checkpoint, draft)
        history_after = len(checkpoint.character_conversations.get(actor_id, []))
        if history_after != history_before + 2:
            raise RuntimeError(
                f"Scenario {scenario.scenario_id!r} changed {actor_id!r} "
                "history by something other than one normal user/assistant pair"
            )
        rows.append(
            {
                "phase": phase,
                "scenario_id": scenario.scenario_id,
                "scenario_kind": scenario.scenario_kind,
                "scenario_turn": scenario_turn,
                "actor_id": actor_id,
                "checkpoint_path": str(checkpoint_path),
                "profile_checkpoint_path": str(profile_checkpoint_path),
                "input_contract": "candidate-fed production CharacterAgent scenario",
                "runtime_known_context_source": (
                    "current production seed known_context overlaid in memory onto "
                    "the scenario checkpoint"
                ),
                "tracked_state_markers": list(scenario.tracked_state_markers),
                "setup_observation": scenario.setup_observation,
                "review_purpose": scenario.review_purpose,
                "prior_actor_id": incoming_prior_actor_id,
                "prior_public": incoming_prior_public,
                "prior_direct_questions": extract_direct_questions(
                    incoming_prior_public
                ),
                "injected_observation": injected_observation,
                "prompt": calls[0]["messages"],
                "system_prompt_sha256": _system_prompt_sha256(calls[0]["messages"]),
                "response": raw_response,
                "parsed": draft.output.model_dump(mode="json"),
                "history_message_count_before": history_before,
                "history_message_count_after": history_after,
                "elapsed_ms": elapsed_ms,
                "usage": agent.last_usage,
                "provider_calls": calls,
            }
        )
        previous_actor_id = actor_id
        previous_public = draft.output.public_text
        client.calls.clear()
    return rows


def _artifact_paths(output_dir: Path, phase: str) -> dict[str, Path]:
    phase_dir = output_dir / phase
    return {
        "conversation_calls": (
            phase_dir / "raw" / "candidate_fed_conversation_calls.jsonl"
        ),
        "conversation_contract": phase_dir / "conversation_review_contract.json",
        "whole_conversation_review": (phase_dir / "whole_conversation_review.json"),
        "pressure_review": phase_dir / "pressure_regression_review.json",
        "voice_samples": phase_dir / "voice_blind_samples.json",
        "voice_answer_key": phase_dir / "voice_answer_key.json",
        "report": phase_dir / "report.json",
    }


async def run_replay(
    output_dir: Path,
    *,
    phase: str,
    client: _ReplayCapturedClient,
    mode: str,
    checkpoint_path: Path,
    profile_checkpoint_path: Path = PRODUCTION_SEED_PATH,
) -> dict[str, Any]:
    """Run candidate-fed conversations, then save human-review evidence."""
    validate_scenario_inventory(REPLAY_SCENARIOS)
    profile_checkpoint = load_checkpoint(profile_checkpoint_path)
    scenario_rows: list[dict[str, Any]] = []
    for scenario in REPLAY_SCENARIOS:
        rows = await _run_conversation_scenario(
            overlay_current_seed_profiles(
                load_checkpoint(checkpoint_path),
                profile_checkpoint,
            ),
            client,
            phase=phase,
            checkpoint_path=checkpoint_path,
            profile_checkpoint_path=profile_checkpoint_path,
            scenario=scenario,
        )
        scenario_rows.extend(rows)

    sustained_ids = {
        scenario.scenario_id for scenario in SUSTAINED_CONVERSATION_SCENARIOS
    }
    sustained_rows = [
        row for row in scenario_rows if row["scenario_id"] in sustained_ids
    ]
    pressure_rows = [
        row for row in scenario_rows if row["scenario_id"] not in sustained_ids
    ]
    whole_conversation_reviews = build_whole_conversation_review(
        sustained_rows,
        SUSTAINED_CONVERSATION_SCENARIOS,
    )
    pressure_reviews = [
        review
        for scenario in PRESSURE_REGRESSION_SCENARIOS
        for review in build_pressure_review_rows(
            [
                row
                for row in pressure_rows
                if row["scenario_id"] == scenario.scenario_id
            ],
            scenario,
        )
    ]
    system_prompt_hashes = sorted(
        {str(row["system_prompt_sha256"]) for row in scenario_rows}
    )
    provider_compaction_values = {
        call["request"].get("compact")
        for row in scenario_rows
        for call in row["provider_calls"]
    }
    if provider_compaction_values != {False}:
        raise RuntimeError(
            "Replay evidence must capture compact=false on every provider call"
        )

    paths = _artifact_paths(output_dir, phase)
    _write_jsonl(paths["conversation_calls"], scenario_rows)
    voice_samples, voice_answer_key = build_blinded_voice_review(sustained_rows)
    _write_json(
        paths["conversation_contract"],
        CONVERSATION_REVIEW_CONTRACT,
    )
    _write_json(
        paths["whole_conversation_review"],
        whole_conversation_reviews,
    )
    _write_json(paths["pressure_review"], pressure_reviews)
    _write_json(paths["voice_samples"], voice_samples)
    _write_json(paths["voice_answer_key"], voice_answer_key)
    report = {
        "phase": phase,
        "mode": mode,
        "model": client.model,
        "profile_checkpoint_path": str(profile_checkpoint_path),
        "profile_checkpoint_sha256": hashlib.sha256(
            profile_checkpoint_path.read_bytes()
        ).hexdigest(),
        "tracked_checkpoint_path": str(checkpoint_path),
        "conversation_sample_count": len(scenario_rows),
        "sustained_conversation_sample_count": len(sustained_rows),
        "pressure_regression_sample_count": len(pressure_rows),
        "sustained_conversation_scenario_count": len(SUSTAINED_CONVERSATION_SCENARIOS),
        "pressure_regression_scenario_count": len(PRESSURE_REGRESSION_SCENARIOS),
        "scenario_sample_counts": {
            scenario.scenario_id: sum(
                row["scenario_id"] == scenario.scenario_id for row in scenario_rows
            )
            for scenario in REPLAY_SCENARIOS
        },
        "scenario_kinds": {
            scenario.scenario_id: scenario.scenario_kind
            for scenario in REPLAY_SCENARIOS
        },
        "system_prompt_sha256s": system_prompt_hashes,
        "profile_overlay_fields": [
            "public_sheet.role",
            "backstory",
            "personality",
            "known_context",
            "private_state.goals",
            "private_state.current_objectives",
            "private_state.secrets",
        ],
        "runtime_known_context_contract": (
            "Current production-seed known_context is deliberately overlaid "
            "in memory because it is prompt-facing identity authority. The "
            "tracked source checkpoint file is not mutated, and the overlay "
            "does not add, drop, or reorder stored history messages."
        ),
        "provider_compaction_values": [False],
        "conversation_contract": (
            "Each conversation starts from a fresh in-memory load of the same "
            "explicit tracked checkpoint. Current production-seed profile fields, "
            "including known_context, replace stale prompt-facing profile fields "
            "without adding, dropping, or reordering stored history messages. "
            "Every later prompt includes the previous candidate "
            "public beat and uses the production CharacterAgent draft/commit "
            "path, which appends exactly one user/assistant pair. Sustained quiet "
            "and conflict exchanges receive whole-conversation human review; "
            "compact danger exchanges are diagnostic regressions only. Every "
            "provider request records compact=false. No router, narrator, full "
            "engine, automated semantic score, or model judge is used."
        ),
        "superseded_review_phases": ["conversation-review-v3"],
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["report"], report)
    return report


def _offline_client() -> _ReplayCapturedClient:
    return _ReplayCapturedClient(
        model=TERRA_MODEL,
        fixed_response=(
            "The character makes one observable choice without explaining it. "
            "(I keep the motive private and carry it into the next beat.)"
        ),
    )


def _live_client() -> _ReplayCapturedClient:
    from app.llm.client import LLMClient
    from app.llm.config import LLMConfig

    config = LLMConfig.from_env()
    for role in (
        "agent",
        "agent_standard",
        "agent_convenience",
        "character_manager",
    ):
        config.role_models[role] = TERRA_MODEL
        config.role_providers[role] = "openai"
    return _ReplayCapturedClient(
        model=TERRA_MODEL,
        delegate=LLMClient(config=config),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--offline", action="store_true", help="use deterministic fixtures (default)"
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="call production Terra through the configured client",
    )
    parser.add_argument(
        "--phase",
        required=True,
        help="artifact subdirectory label, such as baseline or final",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="shared artifact root; phase outputs are written beneath it",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=("tracked checkpoint loaded fresh for every candidate-fed conversation"),
    )
    parser.add_argument(
        "--profile-checkpoint",
        type=Path,
        default=PRODUCTION_SEED_PATH,
        help="current production seed whose identity/private-state fields overlay the replay",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    client = _live_client() if args.live else _offline_client()
    report = asyncio.run(
        run_replay(
            args.output_dir,
            phase=args.phase,
            client=client,
            mode="live" if args.live else "offline",
            checkpoint_path=args.checkpoint,
            profile_checkpoint_path=args.profile_checkpoint,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
