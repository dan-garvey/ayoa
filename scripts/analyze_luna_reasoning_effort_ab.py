#!/usr/bin/env python3
"""Mechanically aggregate the blind Luna reasoning-effort A/B review.

This tool intentionally accepts only the answer key, blinded reviewer sheets,
adjudications, and scheduler telemetry.  It does not accept a blind packet,
response ledger, or any other dialogue artifact, and it never emits the
qualitative review fields it validates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any


CELLS = ("M", "X")
EXPECTED_BLIND_ENTRY_COUNT = 8
EXPECTED_CASES_PER_CELL = 4
EXPECTED_DIMENSION_COUNT = 12
EXPECTED_ACCEPTED_ATTEMPTS = 64
EXPECTED_WORKER_COUNT = 8
EXPECTED_ATTEMPTS_PER_CASE = 16
DIMENSION_RESULT_FIELDS = frozenset({"result", "evidence_turns", "notes"})
STATUS_SCORE = {"fail": 0, "mixed": 1, "pass": 2}


class ReasoningEffortAnalysisError(ValueError):
    """The supplied artifacts do not meet the preregistered mechanical gates."""


@dataclass(frozen=True)
class AnswerKeyEntry:
    blind_id: str
    cell: str
    case_id: str


@dataclass(frozen=True)
class ReviewRecord:
    overall_status: str
    dimension_results: dict[str, str]


@dataclass(frozen=True)
class ReviewerSheet:
    reviews: dict[str, ReviewRecord]
    dimension_ids: frozenset[str]


@dataclass(frozen=True)
class SchedulerTelemetry:
    cell: str
    selected_case_ids: tuple[str, ...]
    latencies_ms: tuple[float, ...]
    rate_limit_count: int
    retry_count: int
    retryable_attempt_count: int


@dataclass(frozen=True)
class FinalEntry:
    blind_id: str
    cell: str
    case_id: str
    reviewer_1_status: str
    reviewer_2_status: str
    consensus_status: str


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReasoningEffortAnalysisError(
            f"cannot read {label} at {path}"
        ) from error
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReasoningEffortAnalysisError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ReasoningEffortAnalysisError(f"{label} must be a JSON object")
    return value, digest


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReasoningEffortAnalysisError(f"{label} must be a nonempty string")
    return value


def _status(value: object, *, label: str) -> str:
    if not isinstance(value, str) or value not in STATUS_SCORE:
        allowed = ", ".join(sorted(STATUS_SCORE))
        raise ReasoningEffortAnalysisError(f"{label} must be one of: {allowed}")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReasoningEffortAnalysisError(f"{label} must be a positive integer")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ReasoningEffortAnalysisError(f"{label} must be a boolean")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReasoningEffortAnalysisError(f"{label} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ReasoningEffortAnalysisError(
            f"{label} must be a nonnegative finite number"
        )
    return numeric


def _list_field(payload: Mapping[str, Any], *, key: str, label: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ReasoningEffortAnalysisError(f"{label} needs a top-level {key} list")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReasoningEffortAnalysisError(f"{label} must be an object")
    return value


def _validate_answer_key(payload: Mapping[str, Any]) -> dict[str, AnswerKeyEntry]:
    raw_entries = _list_field(payload, key="conversations", label="answer key")
    if len(raw_entries) != EXPECTED_BLIND_ENTRY_COUNT:
        raise ReasoningEffortAnalysisError(
            "answer key must contain exactly 8 conversations"
        )

    entries: dict[str, AnswerKeyEntry] = {}
    cases_by_cell: dict[str, set[str]] = {cell: set() for cell in CELLS}
    counts_by_cell: Counter[str] = Counter()
    for index, raw_entry in enumerate(raw_entries, start=1):
        entry = _mapping(raw_entry, label=f"answer key conversation {index}")
        blind_id = _nonempty_string(
            entry.get("blind_id"), label=f"answer key conversation {index} blind_id"
        )
        if blind_id in entries:
            raise ReasoningEffortAnalysisError("answer key has duplicate blind_id values")
        cell = entry.get("cell")
        if cell not in CELLS:
            raise ReasoningEffortAnalysisError(
                "answer key conversation cell must be M or X"
            )
        case_id = _nonempty_string(
            entry.get("case_id"), label=f"answer key conversation {index} case_id"
        )
        entries[blind_id] = AnswerKeyEntry(
            blind_id=blind_id,
            cell=cell,
            case_id=case_id,
        )
        counts_by_cell[cell] += 1
        cases_by_cell[cell].add(case_id)

    if counts_by_cell != Counter({"M": 4, "X": 4}):
        raise ReasoningEffortAnalysisError(
            "answer key must contain exactly four M and four X conversations"
        )
    if any(len(cases_by_cell[cell]) != EXPECTED_CASES_PER_CELL for cell in CELLS):
        raise ReasoningEffortAnalysisError(
            "each answer-key cell must contain four unique case ids"
        )
    if cases_by_cell["M"] != cases_by_cell["X"]:
        raise ReasoningEffortAnalysisError(
            "M and X answer-key cells must contain the same four case ids"
        )
    return entries


def _validate_dimension_result(
    raw_result: object,
    *,
    label: str,
) -> str:
    result = _mapping(raw_result, label=label)
    if set(result) != DIMENSION_RESULT_FIELDS:
        raise ReasoningEffortAnalysisError(
            f"{label} must contain exactly result, evidence_turns, and notes"
        )
    status = _status(result.get("result"), label=f"{label} result")
    evidence_turns = result.get("evidence_turns")
    if not isinstance(evidence_turns, list) or any(
        isinstance(turn, bool) or not isinstance(turn, int) for turn in evidence_turns
    ):
        raise ReasoningEffortAnalysisError(f"{label} evidence_turns must be an integer list")
    if not isinstance(result.get("notes"), str):
        raise ReasoningEffortAnalysisError(f"{label} notes must be a string")
    return status


def _validate_reviewer_sheet(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_blind_ids: frozenset[str],
) -> ReviewerSheet:
    raw_conversations = _list_field(payload, key="conversations", label=label)
    reviews: dict[str, ReviewRecord] = {}
    expected_dimension_ids: frozenset[str] | None = None
    for index, raw_conversation in enumerate(raw_conversations, start=1):
        conversation = _mapping(raw_conversation, label=f"{label} conversation {index}")
        if set(conversation) != {"blind_id", "review"}:
            raise ReasoningEffortAnalysisError(
                f"{label} conversation {index} must contain exactly blind_id and review"
            )
        blind_id = _nonempty_string(
            conversation.get("blind_id"), label=f"{label} conversation {index} blind_id"
        )
        if blind_id in reviews:
            raise ReasoningEffortAnalysisError(f"{label} has duplicate blind_id values")
        review = _mapping(
            conversation.get("review"), label=f"{label} conversation {index} review"
        )
        overall_status = _status(
            review.get("overall_status"),
            label=f"{label} conversation {index} overall_status",
        )
        raw_dimensions = review.get("dimensions")
        if not isinstance(raw_dimensions, Mapping) or len(raw_dimensions) != EXPECTED_DIMENSION_COUNT:
            raise ReasoningEffortAnalysisError(
                f"{label} conversation {index} needs exactly 12 dimension results"
            )
        dimension_results: dict[str, str] = {}
        for dimension_id, raw_result in raw_dimensions.items():
            dimension_name = _nonempty_string(
                dimension_id,
                label=f"{label} conversation {index} dimension id",
            )
            dimension_results[dimension_name] = _validate_dimension_result(
                raw_result,
                label=f"{label} conversation {index} dimension {dimension_name}",
            )
        dimension_ids = frozenset(dimension_results)
        if expected_dimension_ids is None:
            expected_dimension_ids = dimension_ids
        elif dimension_ids != expected_dimension_ids:
            raise ReasoningEffortAnalysisError(
                f"{label} must use the same 12 dimension ids for every conversation"
            )
        reviews[blind_id] = ReviewRecord(
            overall_status=overall_status,
            dimension_results=dimension_results,
        )

    found_blind_ids = frozenset(reviews)
    if found_blind_ids != expected_blind_ids:
        raise ReasoningEffortAnalysisError(
            f"{label} must cover exactly the answer-key blind ids"
        )
    if expected_dimension_ids is None:
        raise ReasoningEffortAnalysisError(f"{label} cannot be empty")
    return ReviewerSheet(reviews=reviews, dimension_ids=expected_dimension_ids)


def _validate_adjudications(
    payload: Mapping[str, Any],
    *,
    expected_disagreements: frozenset[str],
) -> dict[str, str]:
    raw_adjudications = _list_field(payload, key="adjudications", label="adjudicator")
    adjudications: dict[str, str] = {}
    required_fields = {
        "blind_id",
        "final_status",
        "controlling_disputed_concern",
        "rationale",
    }
    for index, raw_adjudication in enumerate(raw_adjudications, start=1):
        adjudication = _mapping(raw_adjudication, label=f"adjudication {index}")
        if set(adjudication) != required_fields:
            raise ReasoningEffortAnalysisError(
                "each adjudication must contain exactly blind_id, final_status, "
                "controlling_disputed_concern, and rationale"
            )
        blind_id = _nonempty_string(
            adjudication.get("blind_id"), label=f"adjudication {index} blind_id"
        )
        if blind_id in adjudications:
            raise ReasoningEffortAnalysisError("adjudicator has duplicate blind_id values")
        adjudications[blind_id] = _status(
            adjudication.get("final_status"), label=f"adjudication {index} final_status"
        )
        _nonempty_string(
            adjudication.get("controlling_disputed_concern"),
            label=f"adjudication {index} controlling_disputed_concern",
        )
        _nonempty_string(
            adjudication.get("rationale"), label=f"adjudication {index} rationale"
        )
    if frozenset(adjudications) != expected_disagreements:
        raise ReasoningEffortAnalysisError(
            "adjudication ids must equal the reviewer overall-status disagreements"
        )
    return adjudications


def _validate_scheduler_state(
    payload: Mapping[str, Any],
    *,
    cell: str,
    expected_case_ids: frozenset[str],
) -> SchedulerTelemetry:
    run = _mapping(payload.get("run"), label=f"{cell} scheduler state run")
    if run.get("luna_reasoning_effort") != (
        "medium" if cell == "M" else "max"
    ):
        expected_effort = "medium" if cell == "M" else "max"
        raise ReasoningEffortAnalysisError(
            f"{cell} scheduler state must explicitly record reasoning effort {expected_effort}"
        )
    raw_selected_case_ids = run.get("selected_case_ids")
    if not isinstance(raw_selected_case_ids, list):
        raise ReasoningEffortAnalysisError(
            f"{cell} scheduler state selected_case_ids must be a list"
        )
    selected_case_ids = tuple(
        _nonempty_string(
            case_id,
            label=f"{cell} scheduler state selected_case_ids entry",
        )
        for case_id in raw_selected_case_ids
    )
    if (
        len(selected_case_ids) != EXPECTED_CASES_PER_CELL
        or len(set(selected_case_ids)) != EXPECTED_CASES_PER_CELL
        or frozenset(selected_case_ids) != expected_case_ids
    ):
        raise ReasoningEffortAnalysisError(
            f"{cell} scheduler state selected_case_ids must match the four answer-key cases"
        )

    workers = payload.get("workers")
    if not isinstance(workers, Mapping) or len(workers) != EXPECTED_WORKER_COUNT:
        raise ReasoningEffortAnalysisError(
            f"{cell} scheduler state must contain exactly 8 workers"
        )
    worker_case_counts: Counter[str] = Counter()
    worker_session_ids: set[str] = set()
    for index, raw_worker in enumerate(workers.values(), start=1):
        worker = _mapping(raw_worker, label=f"{cell} scheduler worker {index}")
        case_id = _nonempty_string(
            worker.get("case_id"), label=f"{cell} scheduler worker {index} case_id"
        )
        if case_id not in expected_case_ids:
            raise ReasoningEffortAnalysisError(
                f"{cell} scheduler worker {index} references an unselected case"
            )
        worker_case_counts[case_id] += 1
        session_id = _nonempty_string(
            worker.get("worker_session_id"),
            label=f"{cell} scheduler worker {index} worker_session_id",
        )
        if session_id in worker_session_ids:
            raise ReasoningEffortAnalysisError(
                f"{cell} scheduler state reuses a worker session"
            )
        worker_session_ids.add(session_id)
    if worker_case_counts != Counter(
        {case_id: 2 for case_id in expected_case_ids}
    ):
        raise ReasoningEffortAnalysisError(
            f"{cell} scheduler state must have two workers for each selected case"
        )

    inflight_calls = payload.get("inflight_delta_calls")
    if not isinstance(inflight_calls, Mapping) or inflight_calls:
        raise ReasoningEffortAnalysisError(
            f"{cell} scheduler state must have zero in-flight calls"
        )

    raw_attempts = payload.get("attempts")
    if not isinstance(raw_attempts, list) or len(raw_attempts) != EXPECTED_ACCEPTED_ATTEMPTS:
        raise ReasoningEffortAnalysisError(
            f"{cell} scheduler state must contain exactly 64 attempts"
        )
    latencies_ms: list[float] = []
    attempt_case_counts: Counter[str] = Counter()
    rate_limit_count = 0
    retry_count = 0
    retryable_attempt_count = 0
    for index, raw_attempt in enumerate(raw_attempts, start=1):
        attempt = _mapping(raw_attempt, label=f"{cell} scheduler attempt {index}")
        if attempt.get("status") != "accepted":
            raise ReasoningEffortAnalysisError(
                f"{cell} scheduler state contains a nonaccepted attempt"
            )
        case_id = _nonempty_string(
            attempt.get("case_id"), label=f"{cell} scheduler attempt {index} case_id"
        )
        if case_id not in expected_case_ids:
            raise ReasoningEffortAnalysisError(
                f"{cell} scheduler attempt {index} references an unselected case"
            )
        attempt_case_counts[case_id] += 1
        latencies_ms.append(
            _finite_number(
                attempt.get("latency_ms"), label=f"{cell} scheduler attempt {index} latency_ms"
            )
        )
        attempt_number = _positive_int(
            attempt.get("attempt"), label=f"{cell} scheduler attempt {index} attempt"
        )
        rate_limited = _boolean(
            attempt.get("rate_limited"),
            label=f"{cell} scheduler attempt {index} rate_limited",
        )
        retryable = _boolean(
            attempt.get("retryable"),
            label=f"{cell} scheduler attempt {index} retryable",
        )
        rate_limit_count += int(rate_limited)
        retry_count += int(attempt_number > 1)
        retryable_attempt_count += int(retryable)
    if attempt_case_counts != Counter(
        {case_id: EXPECTED_ATTEMPTS_PER_CASE for case_id in expected_case_ids}
    ):
        raise ReasoningEffortAnalysisError(
            f"{cell} scheduler state must have 16 accepted attempts per selected case"
        )
    return SchedulerTelemetry(
        cell=cell,
        selected_case_ids=selected_case_ids,
        latencies_ms=tuple(latencies_ms),
        rate_limit_count=rate_limit_count,
        retry_count=retry_count,
        retryable_attempt_count=retryable_attempt_count,
    )


def _linear_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_summary(latencies_ms: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(latencies_ms)
    count = len(ordered)
    if not count:
        raise ValueError("latency summary needs at least one value")
    total = math.fsum(ordered)
    midpoint = count // 2
    median = (
        ordered[midpoint]
        if count % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return {
        "n": count,
        "total_ms": total,
        "mean_ms": total / count,
        "median_ms": median,
        "p90_ms": _linear_percentile(ordered, 0.90),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _build_final_entries(
    answer_key: Mapping[str, AnswerKeyEntry],
    reviewer_1: ReviewerSheet,
    reviewer_2: ReviewerSheet,
    adjudications: Mapping[str, str],
) -> list[FinalEntry]:
    entries: list[FinalEntry] = []
    for blind_id, answer in answer_key.items():
        reviewer_1_status = reviewer_1.reviews[blind_id].overall_status
        reviewer_2_status = reviewer_2.reviews[blind_id].overall_status
        consensus_status = (
            reviewer_1_status
            if reviewer_1_status == reviewer_2_status
            else adjudications[blind_id]
        )
        entries.append(
            FinalEntry(
                blind_id=blind_id,
                cell=answer.cell,
                case_id=answer.case_id,
                reviewer_1_status=reviewer_1_status,
                reviewer_2_status=reviewer_2_status,
                consensus_status=consensus_status,
            )
        )
    return sorted(entries, key=lambda entry: (entry.cell, entry.case_id, entry.blind_id))


def _cell_status_summary(entries: Sequence[FinalEntry]) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for cell in CELLS:
        statuses = [entry.consensus_status for entry in entries if entry.cell == cell]
        counts = Counter(statuses)
        summaries[cell] = {
            "pass": counts["pass"],
            "mixed": counts["mixed"],
            "fail": counts["fail"],
            "mean_status_score": math.fsum(STATUS_SCORE[status] for status in statuses)
            / len(statuses),
        }
    return summaries


def _dimension_averages(
    answer_key: Mapping[str, AnswerKeyEntry],
    reviewer_1: ReviewerSheet,
    reviewer_2: ReviewerSheet,
) -> dict[str, dict[str, dict[str, float]]]:
    averages: dict[str, dict[str, dict[str, float]]] = {}
    for cell in CELLS:
        blind_ids = sorted(
            blind_id for blind_id, entry in answer_key.items() if entry.cell == cell
        )
        averages[cell] = {}
        for label, reviewer in (("reviewer_1", reviewer_1), ("reviewer_2", reviewer_2)):
            averages[cell][label] = {
                dimension_id: math.fsum(
                    STATUS_SCORE[reviewer.reviews[blind_id].dimension_results[dimension_id]]
                    for blind_id in blind_ids
                )
                / len(blind_ids)
                for dimension_id in sorted(reviewer.dimension_ids)
            }
    return averages


def _dimension_comparison(
    averages: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    dimension_ids = sorted(averages["M"]["reviewer_1"])
    for dimension_id in dimension_ids:
        medium_reviewer_1 = averages["M"]["reviewer_1"][dimension_id]
        medium_reviewer_2 = averages["M"]["reviewer_2"][dimension_id]
        max_reviewer_1 = averages["X"]["reviewer_1"][dimension_id]
        max_reviewer_2 = averages["X"]["reviewer_2"][dimension_id]
        medium_mean = (medium_reviewer_1 + medium_reviewer_2) / 2
        max_mean = (max_reviewer_1 + max_reviewer_2) / 2
        comparison[dimension_id] = {
            "M_mean_across_original_reviewers": medium_mean,
            "X_mean_across_original_reviewers": max_mean,
            "X_minus_M_mean": max_mean - medium_mean,
            "reviewer_1_X_minus_M": max_reviewer_1 - medium_reviewer_1,
            "reviewer_2_X_minus_M": max_reviewer_2 - medium_reviewer_2,
        }
    return comparison


def _matched_comparison(entries: Sequence[FinalEntry]) -> dict[str, Any]:
    entries_by_cell_and_case = {
        (entry.cell, entry.case_id): entry for entry in entries
    }
    case_ids = sorted(entry.case_id for entry in entries if entry.cell == "M")
    pairs: list[dict[str, Any]] = []
    differences: list[int] = []
    wins = 0
    ties = 0
    losses = 0
    for case_id in case_ids:
        medium_entry = entries_by_cell_and_case[("M", case_id)]
        max_entry = entries_by_cell_and_case[("X", case_id)]
        medium_score = STATUS_SCORE[medium_entry.consensus_status]
        max_score = STATUS_SCORE[max_entry.consensus_status]
        difference = max_score - medium_score
        differences.append(difference)
        if difference > 0:
            wins += 1
        elif difference < 0:
            losses += 1
        else:
            ties += 1
        pairs.append(
            {
                "case_id": case_id,
                "M_status": medium_entry.consensus_status,
                "M_status_score": medium_score,
                "X_status": max_entry.consensus_status,
                "X_status_score": max_score,
                "X_minus_M_status_score": difference,
            }
        )
    return {
        "matched_pairs": pairs,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "net": wins - losses,
        "mean_paired_score_difference": math.fsum(differences) / len(differences),
    }


def _technical_summary(telemetry: SchedulerTelemetry) -> dict[str, Any]:
    return {
        "selected_case_ids": list(telemetry.selected_case_ids),
        "accepted_attempt_count": len(telemetry.latencies_ms),
        "nonaccepted_attempt_count": 0,
        "latency_ms": _latency_summary(telemetry.latencies_ms),
        "rate_limit_count": telemetry.rate_limit_count,
        "retry_count": telemetry.retry_count,
        "retryable_attempt_count": telemetry.retryable_attempt_count,
    }


def _technical_comparison(
    medium: SchedulerTelemetry,
    maximum: SchedulerTelemetry,
) -> dict[str, Any]:
    medium_mean = math.fsum(medium.latencies_ms) / len(medium.latencies_ms)
    max_mean = math.fsum(maximum.latencies_ms) / len(maximum.latencies_ms)
    return {
        "mean_latency_ratio_X_to_M": (
            max_mean / medium_mean if medium_mean else None
        ),
        "mean_latency_difference_ms_X_minus_M": max_mean - medium_mean,
        "reasoning_token_telemetry": {
            "available": False,
            "reason": (
                "The persistent Codex worker scheduler records latency and retry "
                "telemetry but not per-call reasoning-token usage."
            ),
        },
    }


def analyze_reasoning_effort_ab(
    *,
    answer_key_path: Path,
    reviewer_1_path: Path,
    reviewer_2_path: Path,
    adjudicator_path: Path,
    medium_scheduler_state_path: Path,
    max_scheduler_state_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate all artifacts before joining reviewer outcomes to the answer key."""

    answer_key_payload, answer_key_hash = _read_json_object(
        answer_key_path, label="answer key"
    )
    reviewer_1_payload, reviewer_1_hash = _read_json_object(
        reviewer_1_path, label="reviewer 1"
    )
    reviewer_2_payload, reviewer_2_hash = _read_json_object(
        reviewer_2_path, label="reviewer 2"
    )
    adjudicator_payload, adjudicator_hash = _read_json_object(
        adjudicator_path, label="adjudicator"
    )
    medium_state_payload, medium_state_hash = _read_json_object(
        medium_scheduler_state_path, label="medium scheduler state"
    )
    max_state_payload, max_state_hash = _read_json_object(
        max_scheduler_state_path, label="max scheduler state"
    )

    answer_key = _validate_answer_key(answer_key_payload)
    blind_ids = frozenset(answer_key)
    reviewer_1 = _validate_reviewer_sheet(
        reviewer_1_payload, label="reviewer 1", expected_blind_ids=blind_ids
    )
    reviewer_2 = _validate_reviewer_sheet(
        reviewer_2_payload, label="reviewer 2", expected_blind_ids=blind_ids
    )
    if reviewer_1.dimension_ids != reviewer_2.dimension_ids:
        raise ReasoningEffortAnalysisError(
            "reviewer 1 and reviewer 2 must use the same 12 dimension ids"
        )

    disagreements = frozenset(
        blind_id
        for blind_id in blind_ids
        if reviewer_1.reviews[blind_id].overall_status
        != reviewer_2.reviews[blind_id].overall_status
    )
    adjudications = _validate_adjudications(
        adjudicator_payload, expected_disagreements=disagreements
    )

    expected_case_ids = frozenset(entry.case_id for entry in answer_key.values())
    medium_telemetry = _validate_scheduler_state(
        medium_state_payload, cell="M", expected_case_ids=expected_case_ids
    )
    max_telemetry = _validate_scheduler_state(
        max_state_payload, cell="X", expected_case_ids=expected_case_ids
    )
    if medium_telemetry.selected_case_ids != max_telemetry.selected_case_ids:
        raise ReasoningEffortAnalysisError(
            "medium and max scheduler states must use matching selected_case_ids"
        )

    # This is the first point at which blind review outcomes are joined to M/X.
    final_entries = _build_final_entries(
        answer_key, reviewer_1, reviewer_2, adjudications
    )
    dimension_averages = _dimension_averages(
        answer_key, reviewer_1, reviewer_2
    )
    reviewer_agreement_count = sum(
        entry.reviewer_1_status == entry.reviewer_2_status for entry in final_entries
    )
    report: dict[str, Any] = {
        "schema_version": "luna_reasoning_effort_ab_analysis_v1",
        "guardrail": (
            "The preregistered top-four selection makes this a ceiling/robustness "
            "test, not a population estimate."
        ),
        "input_sha256": {
            "answer_key": answer_key_hash,
            "reviewer_1": reviewer_1_hash,
            "reviewer_2": reviewer_2_hash,
            "adjudicator": adjudicator_hash,
            "medium_scheduler_state": medium_state_hash,
            "max_scheduler_state": max_state_hash,
        },
        "entries": [
            {
                "blind_id": entry.blind_id,
                "cell": entry.cell,
                "case_id": entry.case_id,
                "reviewer_1_status": entry.reviewer_1_status,
                "reviewer_2_status": entry.reviewer_2_status,
                "consensus_status": entry.consensus_status,
            }
            for entry in final_entries
        ],
        "cell_status_summary": _cell_status_summary(final_entries),
        "preregistered_matched_X_relative_to_M": _matched_comparison(final_entries),
        "per_cell_original_reviewer_dimension_averages": dimension_averages,
        "secondary_original_reviewer_dimension_X_relative_to_M": (
            _dimension_comparison(dimension_averages)
        ),
        "reviewer_agreement_count": reviewer_agreement_count,
        "technical": {
            "M": _technical_summary(medium_telemetry),
            "X": _technical_summary(max_telemetry),
        },
        "technical_X_relative_to_M": _technical_comparison(
            medium_telemetry, max_telemetry
        ),
    }
    _write_json(output_path, report)
    return report


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write(rendered)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-key", required=True, type=Path)
    parser.add_argument("--reviewer-1", required=True, type=Path)
    parser.add_argument("--reviewer-2", required=True, type=Path)
    parser.add_argument("--adjudicator", required=True, type=Path)
    parser.add_argument("--medium-scheduler-state", required=True, type=Path)
    parser.add_argument("--max-scheduler-state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        analyze_reasoning_effort_ab(
            answer_key_path=args.answer_key,
            reviewer_1_path=args.reviewer_1,
            reviewer_2_path=args.reviewer_2,
            adjudicator_path=args.adjudicator,
            medium_scheduler_state_path=args.medium_scheduler_state,
            max_scheduler_state_path=args.max_scheduler_state,
            output_path=args.output,
        )
    except ReasoningEffortAnalysisError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
