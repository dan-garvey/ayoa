from __future__ import annotations

import json
from pathlib import Path

from scripts.run_one_star_dialogue_ab import (
    DEFAULT_CASES,
    DEFAULT_VARIANTS,
    DialogueCase,
    DialogueVariant,
    blind_samples,
    run_offline_ab,
)


def test_blinded_samples_drop_variant_and_model_labels() -> None:
    rows = blind_samples([
        {
            "variant_id": "luna",
            "model": "gpt-5.6-luna",
            "case_id": "opening",
            "actor_id": "edren_marr",
            "prompt": "prompt",
            "contract": "contract",
            "response": "response",
        },
    ])

    assert rows == [{
        "blind_id": "sample-001",
        "case_id": "opening",
        "actor_id": "edren_marr",
        "prompt": "prompt",
        "contract": "contract",
        "response": "response",
    }]
    assert "luna" not in json.dumps(rows)
    assert "gpt-5.6" not in json.dumps(rows)


def test_offline_ab_writes_raw_prompts_and_blinded_review_set(tmp_path: Path) -> None:
    variants = (
        DialogueVariant("a", "gpt-5.6-luna"),
        DialogueVariant("b", "gpt-5.6-terra"),
    )
    cases = (
        DialogueCase(
            "sample",
            "edren_marr",
            "Choose one concrete action.",
            "Do not repeat the objection.",
        ),
    )
    calls: list[tuple[str, str]] = []

    def responder(variant: DialogueVariant, case: DialogueCase) -> str:
        calls.append((variant.variant_id, case.case_id))
        return (
            f"{variant.variant_id}:{case.case_id} "
            "(offline fixture makes one concrete choice.)"
        )

    report = run_offline_ab(
        tmp_path,
        variants=variants,
        cases=cases,
        responder=responder,
    )

    assert calls == [("a", "sample"), ("b", "sample")]
    assert report["mode"] == "offline"
    assert report["sample_count"] == 2

    raw_path = tmp_path / "raw" / "dialogue_calls.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert [row["variant_id"] for row in raw_rows] == ["a", "b"]
    assert [row["response"] for row in raw_rows] == [
        "a:sample (offline fixture makes one concrete choice.)",
        "b:sample (offline fixture makes one concrete choice.)",
    ]
    assert all(row["prompt"][0]["role"] == "system" for row in raw_rows)
    assert all(row["prompt"][-1]["role"] == "user" for row in raw_rows)
    assert all(row["provider_calls"] for row in raw_rows)
    assert all(
        "Edren Marr" in row["prompt"][-1]["content"]
        and "Choose one concrete action." in row["prompt"][-1]["content"]
        for row in raw_rows
    )

    blinded = json.loads((tmp_path / "blinded_samples.json").read_text())
    assert [row["blind_id"] for row in blinded] == ["sample-001", "sample-002"]
    assert all("variant_id" not in row and "model" not in row for row in blinded)
    assert json.loads((tmp_path / "blinded_rubric.json").read_text())["blind_fields"] == [
        "variant_id",
        "model",
    ]


def test_default_harness_covers_approved_dialogue_pressures() -> None:
    case_ids = {case.case_id for case in DEFAULT_CASES}
    assert case_ids == {
        "opening",
        "repeated_pressure",
        "ally_danger",
        "lobby_favoritism",
        "post_mission_loss",
        "iselle_control",
    }
    assert [variant.variant_id for variant in DEFAULT_VARIANTS] == ["luna", "terra"]
