#!/usr/bin/env python3
# ruff: noqa: E402 -- executable script adds the repository root to sys.path.
"""Replay One-Star CharacterAgent dialogue without the router or narrator.

The fixed-context lane replays the committed Renna/Mirelle message lineages
from session ``42t24t``.  Each call keeps the persisted role/order/event
history through that turn while overlaying current production-seed profile
blocks.  Candidate-fed scenario lanes start from explicitly supplied tracked
checkpoints and use the production CharacterAgent prompt builder and commit
path.  Each later scenario prompt therefore sees the candidate-authored public
beat produced immediately before it.  Review artifacts preserve prior/current
pairs and leave semantic judgments to a human reviewer.

Importing the module and the default command are offline.  ``--live`` is the
only path that calls the configured provider.  Every run saves raw prompts,
responses, timings, usage, and the historical comparison response.
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
from collections import Counter
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
    BLINDED_RUBRIC,
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
class ReplayTurn:
    """One persisted CharacterAgent turn in global story order."""

    sequence_index: int
    actor_id: str
    actor_turn: int


@dataclass(frozen=True)
class ConversationScenario:
    """One short candidate-fed exchange anchored in tracked checkpoint state."""

    scenario_id: str
    actor_ids: tuple[str, ...]
    tracked_state_markers: tuple[str, ...]
    setup_observation: str
    review_purpose: str


REPLAY_TURNS: tuple[ReplayTurn, ...] = (
    ReplayTurn(1, "renna_holt", 1),
    ReplayTurn(2, "mirelle_voss", 1),
    ReplayTurn(3, "mirelle_voss", 2),
    ReplayTurn(4, "mirelle_voss", 3),
    ReplayTurn(5, "renna_holt", 2),
    ReplayTurn(6, "mirelle_voss", 4),
    ReplayTurn(7, "mirelle_voss", 5),
    ReplayTurn(8, "renna_holt", 3),
    ReplayTurn(9, "mirelle_voss", 6),
    ReplayTurn(10, "mirelle_voss", 7),
    ReplayTurn(11, "renna_holt", 4),
)

CONVERSATION_SCENARIOS: tuple[ConversationScenario, ...] = (
    ConversationScenario(
        scenario_id="post_clear_changed_strength",
        actor_ids=(
            "renna_holt",
            "mirelle_voss",
            "renna_holt",
            "mirelle_voss",
        ),
        tracked_state_markers=(
            "A tangible surge of hard-won strength settles through",
            "Floor Two is unlocked",
        ),
        setup_observation=(
            "Edren flexes his hand after the return surge and asks Renna Holt "
            "and Mirelle Voss, \"Did that new strength feel familiar to either "
            "of you?\""
        ),
        review_purpose=(
            "Test direct-question handling and whether a shared post-clear "
            "thread deepens instead of becoming free-floating banter."
        ),
    ),
    ConversationScenario(
        scenario_id="goblin_separate_escape",
        actor_ids=("mirelle_voss", "renna_holt", "mirelle_voss"),
        tracked_state_markers=(
            "barred section of the timber barricade",
            "rubble-strewn side path",
        ),
        setup_observation=(
            "The first goblin points to the rubble-strewn shortcut, then to a "
            "low drainage culvert beneath the barricade. The goblin says, "
            "\"Cover us until we reach the culvert. You take the rubble path; "
            "we crawl out the other way. The routes don't meet. Deal?\""
        ),
        review_purpose=(
            "Test direct handling of a cover-for-shortcut bargain while the "
            "party and deserters take explicit nonjoining routes."
        ),
    ),
    ConversationScenario(
        scenario_id="barricade_first_contact",
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

CONVERSATION_REVIEW_CONTRACT: dict[str, Any] = {
    "version": "one_star_conversation_v2",
    "reviewer": "human",
    "model_judge": False,
    "unit_of_review": (
        "Read each exact prior/current transition, then the complete scenario. "
        "Do not score isolated lines without their incoming thread."
    ),
    "dimensions": [
        {
            "id": "immediate_uptake",
            "question": (
                "Does the current beat answer, use, resist, or visibly redirect "
                "the immediately prior line rather than merely sharing a mood?"
            ),
            "values": ["pass", "fail", "uncertain"],
        },
        {
            "id": "referent_thread_continuity",
            "question": (
                "Can every important pronoun, object, number, and topic shift be "
                "resolved from tracked state or the prior line, yielding one "
                "reconstructible situation?"
            ),
            "values": ["pass", "fail", "uncertain"],
        },
        {
            "id": "direct_question_handling",
            "question": (
                "When the prior line asks a direct question, does this beat "
                "answer, refuse, or recognizably dodge it? If it dodges, does "
                "the next uptake acknowledge the dodge instead of treating it "
                "as an answer?"
            ),
            "values": [
                "answered",
                "refused",
                "recognized_dodge",
                "unrecognized_dodge",
                "missed",
                "not_applicable",
                "uncertain",
            ],
        },
        {
            "id": "bounded_figurative_anchors",
            "question": (
                "Does the exchange deepen one intelligible figurative anchor, "
                "or stay literal, instead of introducing unrelated images "
                "faster than the listener can resolve them?"
            ),
            "values": ["pass", "fail", "uncertain"],
        },
        {
            "id": "voice_swappability",
            "question": (
                "Across each separate blinded whole-scenario transcript, do "
                "the two speakers sustain distinct cadence, attention, and "
                "social moves without relying on names or weapons?"
            ),
            "values": ["distinct", "swappable", "uncertain"],
        },
    ],
    "decision_policy": (
        "No automated semantic score and no model judge. Record exact evidence "
        "for every fail or uncertain result."
    ),
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
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
_METRIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "is", "it", "its", "me", "my", "not", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "them",
    "they", "this", "to", "was", "we", "were", "will", "with", "you",
}


class _ReplayPromptManager(PromptManager):
    """Return one checkpoint-derived message list to CharacterAgent."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        super().__init__()
        self._messages = copy.deepcopy(messages)

    def render_conversation(
        self,
        template_name: str,
        history: list[Any],
        **variables: Any,
    ) -> list[dict[str, Any]]:
        del history, variables
        if template_name != "agent":
            raise ValueError(f"Replay only supports the agent template, got {template_name!r}")
        return copy.deepcopy(self._messages)


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
        agent_ruleset_system_addon=manager.render(
            "agent_ruleset_one_star"
        ).strip(),
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
    """Transplant current seed profiles while preserving runtime/history shape."""
    for actor_id in actor_ids:
        target = _character(checkpoint, actor_id)
        source = _character(profile_checkpoint, actor_id)
        target.public_sheet.role = source.public_sheet.role
        target.backstory = source.backstory
        target.personality = source.personality
        target.private_state.goals = copy.deepcopy(source.private_state.goals)
        target.private_state.current_objectives = copy.deepcopy(
            source.private_state.current_objectives
        )
        target.private_state.secrets = copy.deepcopy(
            source.private_state.secrets
        )

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


def build_persisted_replay_messages(
    checkpoint: CheckpointFile,
    turn: ReplayTurn,
    *,
    prompt_manager: PromptManager | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Build ``[system, *prior history, current user]`` from ``42t24t``.

    Production intentionally omits volatile One-Star state and presentation
    catalogs when committing the user message to character history.  This
    controlled replay uses that committed lineage exactly; it does not claim
    to reconstruct transient blocks that were never stored.
    """
    manager = prompt_manager or PromptManager()
    addon = manager.render("agent_ruleset_one_star").strip()
    system = manager.render_system_message(
        "agent",
        agent_ruleset_system_addon=addon,
    )
    conversation = checkpoint.character_conversations[turn.actor_id]
    user_index = (turn.actor_turn - 1) * 2
    assistant_index = user_index + 1
    if assistant_index >= len(conversation):
        raise IndexError(
            f"Missing {turn.actor_id} turn {turn.actor_turn} in checkpoint history"
        )
    if conversation[user_index].role != "user":
        raise ValueError(f"Expected user message at history index {user_index}")
    if conversation[assistant_index].role != "assistant":
        raise ValueError(f"Expected assistant message at history index {assistant_index}")
    messages: list[dict[str, Any]] = [system]
    messages.extend(
        {"role": item.role, "content": copy.deepcopy(item.content)}
        for item in conversation[: user_index + 1]
    )
    return messages, copy.deepcopy(conversation[assistant_index].content)


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


def extract_spoken_dialogue(text: str) -> str:
    """Return quoted speech for a name-free voice-swappability sample."""
    return "\n".join(
        match.group("dialogue").strip()
        for match in _DIALOGUE_RE.finditer(public_prose(text))
        if match.group("dialogue").strip()
    )


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
        marker
        for marker in scenario.tracked_state_markers
        if marker not in serialized
    ]
    if missing:
        raise ValueError(
            f"Scenario {scenario.scenario_id!r} checkpoint is missing tracked "
            f"state markers: {missing}"
        )


def build_transition_review_rows(
    rows: list[Mapping[str, Any]],
    scenario: ConversationScenario,
) -> list[dict[str, Any]]:
    """Pair exact incoming/current beats with blank human-review fields."""
    review_rows: list[dict[str, Any]] = []
    prior_actor_id = "tracked_setup"
    prior_public = scenario.setup_observation
    for row in rows:
        prior_questions = extract_direct_questions(prior_public)
        review_rows.append({
            "review_id": (
                f"{scenario.scenario_id}-transition-{int(row['scenario_turn']):02d}"
            ),
            "scenario_id": scenario.scenario_id,
            "scenario_turn": row["scenario_turn"],
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
                "figurative_anchors": [],
                "bounded_figurative_anchors": "",
                "exact_evidence": "",
            },
        })
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
                lambda match: (
                    f"Speaker {label}{match.group('possessive') or ''}"
                ),
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
        setup = str(
            first.get("setup_observation")
            or first.get("prior_public")
            or ""
        )
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
            turns.append({
                "turn": int(row["scenario_turn"]),
                "speaker": pseudonym_by_actor[actor_id],
                "text": redact(public_text),
            })
        prepared.append({
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
        })

    prepared.sort(key=lambda item: item["stable_shuffle_key"])
    samples: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for index, item in enumerate(prepared, start=1):
        blind_id = f"voice-{index:03d}"
        samples.append({
            "blind_id": blind_id,
            "sample_kind": "whole_scenario_transcript",
            "shared_context": item["shared_context"],
            "turns": item["turns"],
            "manual_review": {
                "speaker_a_voice_notes": "",
                "speaker_b_voice_notes": "",
                "voice_swappability": "",
                "cadence_attention_social_move_evidence": "",
            },
        })
        answer_key.append({
            "blind_id": blind_id,
            "scenario_id": item["scenario_id"],
            "speaker_map": item["speaker_map"],
        })
    return samples, answer_key


def public_private_overlap(text: str) -> dict[str, Any]:
    """Report lexical public/private echo as a review aid, not an auto-gate."""
    public, intent = split_public_intent(text)
    public = _PRESENTATION_FOOTER_RE.sub("", public)

    def significant_tokens(value: str) -> list[str]:
        return [
            token.lower()
            for token in _WORD_RE.findall(value)
            if token.lower() not in _METRIC_STOPWORDS
        ]

    public_tokens = significant_tokens(public)
    intent_tokens = significant_tokens(intent)
    public_counts = Counter(public_tokens)
    intent_counts = Counter(intent_tokens)
    shared_count = sum((public_counts & intent_counts).values())
    denominator = min(len(public_tokens), len(intent_tokens))
    return {
        "public_token_count": len(public_tokens),
        "intent_token_count": len(intent_tokens),
        "shared_token_count": shared_count,
        "overlap_of_shorter": round(shared_count / denominator, 3)
        if denominator
        else 0.0,
    }


def repeated_actor_trigrams(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Surface repeated three-word motifs across one actor's generated sequence."""
    by_actor: dict[str, Counter[tuple[str, str, str]]] = {}
    for row in rows:
        actor_id = str(row["actor_id"])
        public, _intent = split_public_intent(str(row["response"]))
        public = _PRESENTATION_FOOTER_RE.sub("", public)
        tokens = [token.lower() for token in _WORD_RE.findall(public)]
        trigrams = set(zip(tokens, tokens[1:], tokens[2:]))
        by_actor.setdefault(actor_id, Counter()).update(trigrams)
    return {
        actor_id: [
            {"phrase": " ".join(trigram), "turn_count": count}
            for trigram, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count > 1
        ]
        for actor_id, counts in by_actor.items()
    }


async def _run_fixed_calls(
    checkpoint: CheckpointFile,
    client: _CapturedClient,
    *,
    phase: str,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in REPLAY_TURNS:
        messages, historical_response = build_persisted_replay_messages(
            checkpoint,
            turn,
        )
        actor = _character(checkpoint, turn.actor_id)
        agent = CharacterAgent(client, _ReplayPromptManager(messages))
        started = time.perf_counter()
        draft = await agent.draft_turn(actor, checkpoint, frame="foreground")
        calls = copy.deepcopy(client.calls)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        raw_response = calls[-1]["response"]
        rows.append({
            "phase": phase,
            "sequence_index": turn.sequence_index,
            "actor_id": turn.actor_id,
            "actor_turn": turn.actor_turn,
            "checkpoint_path": str(checkpoint_path),
            "input_contract": (
                "persisted CharacterAgent role/order/event history with "
                "current production profile blocks"
            ),
            "prompt": messages,
            "historical_response": historical_response,
            "response": raw_response,
            "system_prompt_sha256": _system_prompt_sha256(messages),
            "parsed": draft.output.model_dump(mode="json"),
            "public_private_overlap": public_private_overlap(raw_response),
            "elapsed_ms": elapsed_ms,
            "usage": agent.last_usage,
            "provider_calls": calls,
        })
        client.calls.clear()
    return rows


async def _run_conversation_scenario(
    checkpoint: CheckpointFile,
    client: _CapturedClient,
    *,
    phase: str,
    checkpoint_path: Path,
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
        history_before = len(
            checkpoint.character_conversations.get(actor_id, [])
        )
        started = time.perf_counter()
        draft = await agent.draft_turn(actor, checkpoint, frame="foreground")
        calls = copy.deepcopy(client.calls)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        raw_response = calls[-1]["response"]
        agent.commit_draft(actor, checkpoint, draft)
        history_after = len(
            checkpoint.character_conversations.get(actor_id, [])
        )
        if history_after != history_before + 2:
            raise RuntimeError(
                f"Scenario {scenario.scenario_id!r} changed {actor_id!r} "
                "history by something other than one normal user/assistant pair"
            )
        rows.append({
            "phase": phase,
            "scenario_id": scenario.scenario_id,
            "scenario_turn": scenario_turn,
            "actor_id": actor_id,
            "checkpoint_path": str(checkpoint_path),
            "input_contract": "candidate-fed production CharacterAgent scenario",
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
            "system_prompt_sha256": _system_prompt_sha256(
                calls[0]["messages"]
            ),
            "response": raw_response,
            "parsed": draft.output.model_dump(mode="json"),
            "public_private_overlap": public_private_overlap(raw_response),
            "history_message_count_before": history_before,
            "history_message_count_after": history_after,
            "elapsed_ms": elapsed_ms,
            "usage": agent.last_usage,
            "provider_calls": calls,
        })
        previous_actor_id = actor_id
        previous_public = draft.output.public_text
        client.calls.clear()
    return rows


def _artifact_paths(output_dir: Path, phase: str) -> dict[str, Path]:
    phase_dir = output_dir / phase
    return {
        "fixed_calls": phase_dir / "raw" / "fixed_context_calls.jsonl",
        "scenario_calls": phase_dir / "raw" / "candidate_fed_scenario_calls.jsonl",
        "review_samples": phase_dir / "review_samples.json",
        "rubric": phase_dir / "review_rubric.json",
        "conversation_contract": phase_dir / "conversation_review_contract.json",
        "transition_review": phase_dir / "transition_review_sheet.json",
        "voice_samples": phase_dir / "voice_blind_samples.json",
        "voice_answer_key": phase_dir / "voice_answer_key.json",
        "report": phase_dir / "report.json",
    }


async def run_replay(
    output_dir: Path,
    *,
    phase: str,
    client: _CapturedClient,
    mode: str,
    checkpoint_path: Path,
    scenario_checkpoint_paths: Mapping[str, Path],
    profile_checkpoint_path: Path = PRODUCTION_SEED_PATH,
) -> dict[str, Any]:
    """Run fixed contexts plus candidate-fed scenarios, then save evidence."""
    expected_scenarios = {
        scenario.scenario_id
        for scenario in CONVERSATION_SCENARIOS
    }
    supplied_scenarios = set(scenario_checkpoint_paths)
    if supplied_scenarios != expected_scenarios:
        missing = sorted(expected_scenarios - supplied_scenarios)
        unknown = sorted(supplied_scenarios - expected_scenarios)
        raise ValueError(
            "Scenario checkpoint mapping must match the review scenarios; "
            f"missing={missing}, unknown={unknown}"
        )
    scenarios_with_paths = tuple(
        (
            scenario,
            scenario_checkpoint_paths[scenario.scenario_id],
        )
        for scenario in CONVERSATION_SCENARIOS
    )

    profile_checkpoint = load_checkpoint(profile_checkpoint_path)
    fixed_rows = await _run_fixed_calls(
        overlay_current_seed_profiles(
            load_checkpoint(checkpoint_path),
            profile_checkpoint,
        ),
        client,
        phase=phase,
        checkpoint_path=checkpoint_path,
    )
    scenario_rows: list[dict[str, Any]] = []
    transition_review_rows: list[dict[str, Any]] = []
    for scenario, scenario_path in scenarios_with_paths:
        rows = await _run_conversation_scenario(
            overlay_current_seed_profiles(
                load_checkpoint(scenario_path),
                profile_checkpoint,
            ),
            client,
            phase=phase,
            checkpoint_path=scenario_path,
            scenario=scenario,
        )
        scenario_rows.extend(rows)
        transition_review_rows.extend(
            build_transition_review_rows(rows, scenario)
        )

    paths = _artifact_paths(output_dir, phase)
    _write_jsonl(paths["fixed_calls"], fixed_rows)
    _write_jsonl(paths["scenario_calls"], scenario_rows)
    review_samples = [
        {
            "sample_id": f"fixed-{row['sequence_index']:02d}",
            "lane": "fixed_context",
            "actor_id": row["actor_id"],
            "actor_turn": row["actor_turn"],
            "historical_response": row["historical_response"],
            "response": row["response"],
            "public_private_overlap": row["public_private_overlap"],
        }
        for row in fixed_rows
    ]
    review_samples.extend(
        {
            "sample_id": (
                f"{row['scenario_id']}-{int(row['scenario_turn']):02d}"
            ),
            "lane": "candidate_fed_scenario",
            "scenario_id": row["scenario_id"],
            "scenario_turn": row["scenario_turn"],
            "actor_id": row["actor_id"],
            "prior_public": row["prior_public"],
            "prior_direct_questions": row["prior_direct_questions"],
            "response": row["response"],
            "public_private_overlap": row["public_private_overlap"],
        }
        for row in scenario_rows
    )
    voice_samples, voice_answer_key = build_blinded_voice_review(
        scenario_rows
    )
    _write_json(paths["review_samples"], review_samples)
    _write_json(paths["rubric"], BLINDED_RUBRIC)
    _write_json(
        paths["conversation_contract"],
        CONVERSATION_REVIEW_CONTRACT,
    )
    _write_json(paths["transition_review"], transition_review_rows)
    _write_json(paths["voice_samples"], voice_samples)
    _write_json(paths["voice_answer_key"], voice_answer_key)
    system_prompt_hashes = sorted({
        str(row["system_prompt_sha256"])
        for row in [*fixed_rows, *scenario_rows]
    })
    report = {
        "phase": phase,
        "mode": mode,
        "model": client.model,
        "profile_checkpoint_path": str(profile_checkpoint_path),
        "profile_checkpoint_sha256": hashlib.sha256(
            profile_checkpoint_path.read_bytes()
        ).hexdigest(),
        "fixed_context_checkpoint_path": str(checkpoint_path),
        "scenario_checkpoint_paths": {
            scenario.scenario_id: str(scenario_path)
            for scenario, scenario_path in scenarios_with_paths
        },
        "fixed_context_sample_count": len(fixed_rows),
        "candidate_fed_scenario_sample_count": len(scenario_rows),
        "scenario_sample_counts": {
            scenario.scenario_id: sum(
                row["scenario_id"] == scenario.scenario_id
                for row in scenario_rows
            )
            for scenario, _path in scenarios_with_paths
        },
        "system_prompt_sha256s": system_prompt_hashes,
        "fixed_context_contract": (
            "Each call preserves the persisted 42t24t CharacterAgent role, "
            "order, event input, and assistant history while overlaying the "
            "current production-seed identity/private-state blocks. Candidate "
            "output is not fed into later fixed-context calls."
        ),
        "candidate_fed_scenario_contract": (
            "Each exchange starts from an explicitly supplied checkpoint. "
            "Every later prompt includes the previous candidate "
            "public beat and uses the production CharacterAgent draft/commit "
            "path. Review is manual over exact prior/current pairs; no router, "
            "narrator, full engine, automated semantic score, or model judge "
            "is used."
        ),
        "repeated_public_trigrams": {
            "fixed_context": repeated_actor_trigrams(fixed_rows),
            "candidate_fed_scenarios": repeated_actor_trigrams(
                scenario_rows
            ),
        },
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["report"], report)
    return report


def _offline_client() -> _CapturedClient:
    return _CapturedClient(
        model=TERRA_MODEL,
        fixed_response=(
            "The character makes one observable choice without explaining it. "
            "(I keep the motive private and carry it into the next beat.)"
        ),
    )


def _live_client() -> _CapturedClient:
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
    return _CapturedClient(
        model=TERRA_MODEL,
        delegate=LLMClient(config=config),
    )


def parse_scenario_checkpoint_args(values: Iterable[str]) -> dict[str, Path]:
    """Parse repeatable ``SCENARIO_ID=PATH`` CLI values with no implicit paths."""
    mapping: dict[str, Path] = {}
    for raw in values:
        scenario_id, separator, raw_path = raw.partition("=")
        scenario_id = scenario_id.strip()
        raw_path = raw_path.strip()
        if not separator or not scenario_id or not raw_path:
            raise ValueError(
                "Scenario checkpoints must use SCENARIO_ID=PATH"
            )
        if scenario_id in mapping:
            raise ValueError(
                f"Duplicate scenario checkpoint for {scenario_id!r}"
            )
        mapping[scenario_id] = Path(raw_path)
    return mapping


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="use deterministic fixtures (default)")
    mode.add_argument("--live", action="store_true", help="call production Terra through the configured client")
    parser.add_argument("--phase", required=True, help="artifact subdirectory label, such as baseline or final")
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
        help="checkpoint containing the persisted character histories to replay",
    )
    parser.add_argument(
        "--scenario-checkpoint",
        action="append",
        required=True,
        metavar="SCENARIO_ID=PATH",
        help=(
            "explicit checkpoint for one candidate-fed scenario; repeat once "
            "for every scenario id"
        ),
    )
    parser.add_argument(
        "--profile-checkpoint",
        type=Path,
        default=PRODUCTION_SEED_PATH,
        help="current production seed whose identity/private-state fields overlay the replay",
    )
    args = parser.parse_args(argv)
    try:
        args.scenario_checkpoints = parse_scenario_checkpoint_args(
            args.scenario_checkpoint
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


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
            scenario_checkpoint_paths=args.scenario_checkpoints,
            profile_checkpoint_path=args.profile_checkpoint,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
