from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.analyze_luna_reasoning_effort_ab import (
    ReasoningEffortAnalysisError,
    analyze_reasoning_effort_ab,
)


CASES = ("case_a", "case_b", "case_c", "case_d")
DIMENSIONS = tuple(f"dimension_{index:02d}" for index in range(1, 13))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _answer_key() -> dict[str, object]:
    conversations = []
    sample_index = 1
    for cell in ("M", "X"):
        for case_id in CASES:
            conversations.append(
                {
                    "blind_id": f"sample-{sample_index:02d}",
                    "cell": cell,
                    "case_id": case_id,
                }
            )
            sample_index += 1
    return {"conversations": conversations}


def _reviewer(
    answer_key: dict[str, object],
    statuses: dict[str, str],
) -> dict[str, object]:
    conversations = []
    for item in answer_key["conversations"]:  # type: ignore[index]
        blind_id = item["blind_id"]  # type: ignore[index]
        status = statuses[blind_id]
        conversations.append(
            {
                "blind_id": blind_id,
                "review": {
                    "overall_status": status,
                    "dimensions": {
                        dimension_id: {
                            "result": status,
                            "evidence_turns": [1, 16],
                            "notes": f"Qualitative note for {blind_id}.",
                        }
                        for dimension_id in DIMENSIONS
                    },
                    "quality_flags": ["reviewer-only qualitative field"],
                    "notes": "Reviewer-only overall note.",
                },
            }
        )
    return {"conversations": conversations}


def _scheduler_state(cell: str) -> dict[str, object]:
    effort = "medium" if cell == "M" else "max"
    workers = {}
    attempts = []
    for case_index, case_id in enumerate(CASES, start=1):
        for actor_index in range(2):
            worker_id = f"{cell}-{case_id}-worker-{actor_index}"
            workers[worker_id] = {
                "case_id": case_id,
                "worker_session_id": worker_id,
            }
        for turn_index in range(16):
            attempts.append(
                {
                    "attempt": 1,
                    "case_id": case_id,
                    "latency_ms": float(case_index * 100 + turn_index),
                    "rate_limited": False,
                    "retryable": False,
                    "status": "accepted",
                }
            )
    return {
        "run": {
            "luna_reasoning_effort": effort,
            "selected_case_ids": list(CASES),
        },
        "workers": workers,
        "attempts": attempts,
        "inflight_delta_calls": {},
    }


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    answer_key = _answer_key()
    reviewer_1_statuses = {
        "sample-01": "fail",
        "sample-02": "mixed",
        "sample-03": "pass",
        "sample-04": "pass",
        "sample-05": "pass",
        "sample-06": "mixed",
        "sample-07": "pass",
        "sample-08": "mixed",
    }
    reviewer_2_statuses = {
        **reviewer_1_statuses,
        "sample-01": "mixed",
    }
    payloads = {
        "answer_key": answer_key,
        "reviewer_1": _reviewer(answer_key, reviewer_1_statuses),
        "reviewer_2": _reviewer(answer_key, reviewer_2_statuses),
        "adjudicator": {
            "adjudications": [
                {
                    "blind_id": "sample-01",
                    "final_status": "mixed",
                    "controlling_disputed_concern": "The arc changes modestly.",
                    "rationale": "The change is enough for mixed, but not pass.",
                }
            ]
        },
        "medium_scheduler_state": _scheduler_state("M"),
        "max_scheduler_state": _scheduler_state("X"),
    }
    paths = {name: tmp_path / f"{name}.json" for name in payloads}
    for name, payload in payloads.items():
        _write_json(paths[name], payload)
    paths["output"] = tmp_path / "analysis.json"
    return paths


def _analyze(paths: dict[str, Path]) -> dict[str, object]:
    return analyze_reasoning_effort_ab(
        answer_key_path=paths["answer_key"],
        reviewer_1_path=paths["reviewer_1"],
        reviewer_2_path=paths["reviewer_2"],
        adjudicator_path=paths["adjudicator"],
        medium_scheduler_state_path=paths["medium_scheduler_state"],
        max_scheduler_state_path=paths["max_scheduler_state"],
        output_path=paths["output"],
    )


def test_analysis_validates_and_joins_only_after_blind_review(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)

    report = _analyze(paths)

    assert json.loads(paths["output"].read_text(encoding="utf-8")) == report
    assert report["cell_status_summary"] == {
        "M": {"pass": 2, "mixed": 2, "fail": 0, "mean_status_score": 1.5},
        "X": {"pass": 2, "mixed": 2, "fail": 0, "mean_status_score": 1.5},
    }
    assert report["preregistered_matched_X_relative_to_M"] == {
        "matched_pairs": [
            {
                "case_id": "case_a",
                "M_status": "mixed",
                "M_status_score": 1,
                "X_status": "pass",
                "X_status_score": 2,
                "X_minus_M_status_score": 1,
            },
            {
                "case_id": "case_b",
                "M_status": "mixed",
                "M_status_score": 1,
                "X_status": "mixed",
                "X_status_score": 1,
                "X_minus_M_status_score": 0,
            },
            {
                "case_id": "case_c",
                "M_status": "pass",
                "M_status_score": 2,
                "X_status": "pass",
                "X_status_score": 2,
                "X_minus_M_status_score": 0,
            },
            {
                "case_id": "case_d",
                "M_status": "pass",
                "M_status_score": 2,
                "X_status": "mixed",
                "X_status_score": 1,
                "X_minus_M_status_score": -1,
            },
        ],
        "wins": 1,
        "ties": 2,
        "losses": 1,
        "net": 0,
        "mean_paired_score_difference": 0.0,
    }
    assert report["reviewer_agreement_count"] == 7
    assert report["technical"]["M"]["accepted_attempt_count"] == 64  # type: ignore[index]
    assert report["technical"]["X"]["accepted_attempt_count"] == 64  # type: ignore[index]
    assert report["technical_X_relative_to_M"] == {
        "mean_latency_ratio_X_to_M": 1.0,
        "mean_latency_difference_ms_X_minus_M": 0.0,
        "reasoning_token_telemetry": {
            "available": False,
            "reason": (
                "The persistent Codex worker scheduler records latency and retry "
                "telemetry but not per-call reasoning-token usage."
            ),
        },
    }
    assert report["secondary_original_reviewer_dimension_X_relative_to_M"][  # type: ignore[index]
        DIMENSIONS[0]
    ] == {
        "M_mean_across_original_reviewers": 1.375,
        "X_mean_across_original_reviewers": 1.5,
        "X_minus_M_mean": 0.125,
        "reviewer_1_X_minus_M": 0.25,
        "reviewer_2_X_minus_M": 0.0,
    }

    encoded = json.dumps(report)
    assert "Qualitative note" not in encoded
    assert "reviewer-only" not in encoded.lower()


def test_analysis_requires_exact_adjudication_set(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    _write_json(paths["adjudicator"], {"adjudications": []})

    with pytest.raises(
        ReasoningEffortAnalysisError,
        match="adjudication ids must equal",
    ):
        _analyze(paths)


def test_analysis_rejects_effort_or_review_shape_drift(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    max_state = json.loads(paths["max_scheduler_state"].read_text(encoding="utf-8"))
    max_state["run"]["luna_reasoning_effort"] = "medium"
    _write_json(paths["max_scheduler_state"], max_state)

    with pytest.raises(ReasoningEffortAnalysisError, match="must explicitly record"):
        _analyze(paths)

    paths = _artifacts(tmp_path / "review-shape")
    reviewer = json.loads(paths["reviewer_1"].read_text(encoding="utf-8"))
    bad_dimension = deepcopy(reviewer["conversations"][0]["review"]["dimensions"][DIMENSIONS[0]])
    bad_dimension["unexpected"] = "drift"
    reviewer["conversations"][0]["review"]["dimensions"][DIMENSIONS[0]] = bad_dimension
    _write_json(paths["reviewer_1"], reviewer)

    with pytest.raises(ReasoningEffortAnalysisError, match="must contain exactly"):
        _analyze(paths)
