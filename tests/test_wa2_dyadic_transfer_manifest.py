from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from scripts.run_character_dialogue_benchmark import (
    BenchmarkRequest,
    ModelCall,
    load_benchmark_manifest,
    run_conversation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPO_ROOT
    / "app"
    / "storage"
    / "playtest_reports"
    / "character-dialogue-benchmark"
    / "20260830-luna-popular-controls"
    / "white-album-2"
    / "dyadic-transfer-control"
    / "control-manifest.json"
)


def _raw_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _cases():
    return load_benchmark_manifest(MANIFEST_PATH)


def _candidate_responder(request: BenchmarkRequest) -> ModelCall:
    return ModelCall(
        content=f"A visible reply from {request.actor_id} on turn {request.turn_index}.",
        model=request.model,
        provider="synthetic",
    )


def _normalize_replica_ids(raw_case: dict) -> dict:
    normalized = copy.deepcopy(raw_case)
    normalized["case_id"] = "<case-id>"
    normalized["source_metadata"]["conversation_id"] = "<conversation-id>"
    return normalized


@pytest.fixture(scope="module")
def results_by_case():
    return {
        case.case_id: asyncio.run(
            run_conversation(
                case,
                model="synthetic-model",
                responder=_candidate_responder,
                conversation_id=str(case.source_metadata["conversation_id"]),
            )
        )
        for case in _cases()
    }


def test_dyadic_transfer_manifest_has_four_sixteen_turn_cases() -> None:
    cases = _cases()

    assert len(cases) == 4
    assert [case.case_id for case in cases] == [
        "wa2_hs_transfer_a",
        "wa2_hs_transfer_b",
        "wa2_sk_transfer_a",
        "wa2_sk_transfer_b",
    ]
    assert all(case.suite == "ordinary_surface" for case in cases)
    assert all(len(case.actors) == 2 for case in cases)
    assert all(len(case.scenes) == 2 for case in cases)
    assert all([len(scene.turn_order) for scene in case.scenes] == [8, 8] for case in cases)
    assert all(
        sum(len(scene.turn_order) for scene in case.scenes) == 16
        for case in cases
    )

    assert [actor.character_id for actor in cases[0].actors] == [
        "setsuna_ogiso",
        "haruki_kitahara",
    ]
    assert [actor.character_id for actor in cases[2].actors] == [
        "setsuna_ogiso",
        "kazusa_touma",
    ]
    assert cases[0].scenes[0].turn_order == (
        "setsuna_ogiso",
        "haruki_kitahara",
    ) * 4
    assert cases[0].scenes[1].turn_order == (
        "haruki_kitahara",
        "setsuna_ogiso",
    ) * 4
    assert cases[2].scenes[0].turn_order == (
        "kazusa_touma",
        "setsuna_ogiso",
    ) * 4
    assert cases[2].scenes[1].turn_order == (
        "setsuna_ogiso",
        "kazusa_touma",
    ) * 4


def test_replicas_are_semantically_identical_after_only_id_normalization() -> None:
    raw_cases = _raw_manifest()["cases"]

    assert _normalize_replica_ids(raw_cases[0]) == _normalize_replica_ids(raw_cases[1])
    assert _normalize_replica_ids(raw_cases[2]) == _normalize_replica_ids(raw_cases[3])
    assert {
        raw_case["source_metadata"]["conversation_id"] for raw_case in raw_cases
    } == {
        "wa2-hs-transfer-a",
        "wa2-hs-transfer-b",
        "wa2-sk-transfer-a",
        "wa2-sk-transfer-b",
    }


def test_actor_material_reuses_selected_records_and_omits_the_stale_sixth_setsuna_fact() -> None:
    cases = _cases()
    expected_roles = {
        "setsuna_ogiso": "a popular third-year student and the light-music club's singer",
        "haruki_kitahara": "an honors student, former class representative, and the light-music club's second guitarist",
        "kazusa_touma": "an isolated third-year student and gifted pianist",
    }
    expected_counts = {
        "setsuna_ogiso": 5,
        "haruki_kitahara": 5,
        "kazusa_touma": 6,
    }
    stale_setsuna_fact = (
        "This is the first time you have brought school friends into the booth. "
        "You did not tell Haruki or Kazusa that."
    )
    expected_source_urls = [
        "https://aquaplus.jp/wa2/character.html",
        "https://aquaplus.jp/wa2/character02.html",
        "https://aquaplus.jp/wa2/character03.html",
        "https://whitealbum2.jp/character/toumayouko.html",
    ]
    expected_actor_material_source = (
        "app/storage/playtest_reports/character-dialogue-benchmark/"
        "20260830-luna-popular-controls/white-album-2/"
        "spare-umbrella-control/control-manifest.json"
    )

    for case in cases:
        for actor in case.actors:
            assert actor.public_sheet.role == expected_roles[actor.character_id]
            assert actor.actor is not None
            assert len(actor.actor.facts) == expected_counts[actor.character_id]
            assert all(fact.text != stale_setsuna_fact for fact in actor.actor.facts)

        selected = case.source_metadata["selected_actor_facts"]
        assert case.source_metadata["actor_material_source"] == (
            expected_actor_material_source
        )
        assert [entry["url"] for entry in case.source_metadata["source_basis"]] == (
            expected_source_urls
        )
        assert selected["setsuna_ogiso"] == [1, 2, 3, 4, 5]
        for actor in case.actors:
            assert selected[actor.character_id] == list(
                range(1, expected_counts[actor.character_id] + 1)
            )


def test_manifest_has_no_authored_dialogue_pressure_or_stale_benchmark_surfaces() -> None:
    raw = _raw_manifest()
    raw_text = json.dumps(raw, ensure_ascii=False)
    forbidden_markers = (
        "<initial_objectives>",
        "<private_carry>",
        "prior_public_exchange",
        "pressure_pulses",
        "review_targets",
        "reviewer_focus",
        "outcome_policy",
        "evaluator",
        "umbrella_return",
        "cafe_card",
        "rain_begins",
        "loyalty_card_arrives",
        "This is the first time you have brought school friends into the booth.",
    )

    assert all(marker not in raw_text for marker in forbidden_markers)
    for raw_case, case in zip(raw["cases"], _cases(), strict=True):
        assert all("prior_public_exchange" not in scene for scene in raw_case["scenes"])
        assert all("pressure_pulses" not in scene for scene in raw_case["scenes"])
        assert all(scene.prior_public_exchange == () for scene in case.scenes)
        assert all(scene.pressure_pulses == () for scene in case.scenes)
        assert "actor_observations" not in raw_case["scenes"][1]
        assert case.scenes[1].actor_observations == {}


def test_private_scene_observations_stay_owner_bounded_in_current_renderer(
    results_by_case,
) -> None:
    for case in _cases():
        result = results_by_case[case.case_id]
        observations = case.scenes[0].actor_observations

        assert result.conversation_id == case.source_metadata["conversation_id"]
        assert len(result.turns) == 16
        assert set(observations) == {
            actor.character_id for actor in case.actors
        }

        for actor_id, actor_observations in observations.items():
            owner_turn = next(
                turn
                for turn in result.turns
                if turn.scene_index == 0 and turn.actor_id == actor_id
            )
            system_text = str(owner_turn.prompt[0]["content"])
            user_text = str(owner_turn.prompt[-1]["content"])

            assert "<you>" in user_text
            assert "</you>" in user_text
            assert "<now>" in user_text
            assert "</now>" in user_text
            assert f"You are {owner_turn.actor_name}." in user_text
            for observation in actor_observations:
                assert observation not in system_text
                assert observation in user_text.split("<now>", 1)[-1]
                for counterpart_turn in result.turns:
                    if counterpart_turn.actor_id != actor_id:
                        assert observation not in json.dumps(
                            counterpart_turn.prompt,
                            ensure_ascii=False,
                        )

        prompt_text = "\n".join(
            json.dumps(turn.prompt, ensure_ascii=False) for turn in result.turns
        )
        assert "<initial_objectives>" not in prompt_text
        assert "<private_carry>" not in prompt_text
        assert case.source_metadata["conversation_id"] not in prompt_text
        assert case.source_metadata["actor_material_source"] not in prompt_text
