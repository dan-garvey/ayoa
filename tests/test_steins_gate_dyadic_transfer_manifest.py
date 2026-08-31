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
    / "steins-gate"
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


def _tag_content(text: str, tag: str) -> str:
    opening = f"<{tag}>"
    closing = f"</{tag}>"

    assert text.count(opening) == 1
    assert text.count(closing) == 1
    return text.split(opening, 1)[1].split(closing, 1)[0]


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


def test_dyadic_transfer_manifest_has_four_identical_sixteen_turn_cases() -> None:
    cases = _cases()

    assert len(cases) == 4
    assert [case.case_id for case in cases] == [
        "sg_okabe_kurisu_transfer_a",
        "sg_okabe_kurisu_transfer_b",
        "sg_okabe_kurisu_transfer_c",
        "sg_okabe_kurisu_transfer_d",
    ]
    assert all(case.suite == "ordinary_surface" for case in cases)
    assert all(len(case.actors) == 2 for case in cases)
    assert all(len(case.scenes) == 2 for case in cases)
    assert all([len(scene.turn_order) for scene in case.scenes] == [8, 8] for case in cases)
    assert all(
        sum(len(scene.turn_order) for scene in case.scenes) == 16
        for case in cases
    )
    assert all(
        [actor.character_id for actor in case.actors]
        == ["okabe_rintaro", "kurisu_makise"]
        for case in cases
    )
    assert all(
        scene.turn_order == ("okabe_rintaro", "kurisu_makise") * 4
        for case in cases
        for scene in case.scenes
    )


def test_replicas_are_semantically_identical_after_only_id_normalization() -> None:
    raw_cases = _raw_manifest()["cases"]
    normalized = [_normalize_replica_ids(raw_case) for raw_case in raw_cases]

    assert all(raw_case == normalized[0] for raw_case in normalized[1:])
    assert all(
        set(raw_case["source_metadata"]) == {"conversation_id"}
        for raw_case in raw_cases
    )
    assert {
        raw_case["source_metadata"]["conversation_id"] for raw_case in raw_cases
    } == {
        "sg-okabe-kurisu-transfer-a",
        "sg-okabe-kurisu-transfer-b",
        "sg-okabe-kurisu-transfer-c",
        "sg-okabe-kurisu-transfer-d",
    }


def test_manifest_leaves_dialogue_and_outcomes_to_the_agent() -> None:
    raw = _raw_manifest()
    raw_text = json.dumps(raw, ensure_ascii=False).casefold()
    forbidden_markers = (
        "<initial_objectives>",
        "<private_carry>",
        "prior_public_exchange",
        "pressure_pulses",
        "procedure",
        "evidence",
        "ledger",
        "courtroom",
        "evaluator",
        "rubric",
        "canonical",
        "mad scientist",
        "el psy kongroo",
        "prompt_mode",
    )

    assert all(marker not in raw_text for marker in forbidden_markers)
    for raw_case, case in zip(raw["cases"], _cases(), strict=True):
        assert all("prior_public_exchange" not in scene for scene in raw_case["scenes"])
        assert all("pressure_pulses" not in scene for scene in raw_case["scenes"])
        assert all(scene.prior_public_exchange == () for scene in case.scenes)
        assert all(scene.pressure_pulses == () for scene in case.scenes)
        assert set(case.scenes[0].actor_observations) == {
            actor.character_id for actor in case.actors
        }


def test_private_material_stays_owner_bounded_in_current_character_agent_packets(
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
            actor = case.actor(actor_id)
            assert actor.actor is not None
            private_facts = tuple(fact.text for fact in actor.actor.facts)
            assert private_facts
            owner_turn = next(
                turn
                for turn in result.turns
                if turn.scene_index == 0 and turn.actor_id == actor_id
            )
            system_text = str(owner_turn.prompt[0]["content"])
            user_text = str(owner_turn.prompt[-1]["content"])
            owner_prompt_text = json.dumps(owner_turn.prompt, ensure_ascii=False)

            assert owner_turn.prompt[0]["role"] == "system"
            assert owner_turn.prompt[-1]["role"] == "user"
            you_text = _tag_content(user_text, "you")
            now_text = _tag_content(user_text, "now")
            assert f"You are {owner_turn.actor_name}." in you_text
            assert all(fact in you_text for fact in private_facts)
            assert all(fact not in system_text for fact in private_facts)
            for observation in actor_observations:
                assert observation not in system_text
                assert observation in now_text
                for counterpart_turn in result.turns:
                    if counterpart_turn.actor_id != actor_id:
                        counterpart_prompt_text = json.dumps(
                            counterpart_turn.prompt,
                            ensure_ascii=False,
                        )
                        assert observation not in counterpart_prompt_text
                        assert all(
                            fact not in counterpart_prompt_text
                            for fact in private_facts
                        )

            for counterpart_actor_id, counterpart_observations in observations.items():
                if counterpart_actor_id != actor_id:
                    assert all(
                        observation not in owner_prompt_text
                        for observation in counterpart_observations
                    )

        prompt_text = "\n".join(
            json.dumps(turn.prompt, ensure_ascii=False) for turn in result.turns
        )
        assert "<initial_objectives>" not in prompt_text
        assert "<private_carry>" not in prompt_text
        assert case.source_metadata["conversation_id"] not in prompt_text

        public_text = json.dumps(result.public_transcript, ensure_ascii=False)
        assert all(
            observation not in public_text
            for actor_observations in observations.values()
            for observation in actor_observations
        )


def test_replicas_render_the_same_agent_input_without_run_provenance(
    results_by_case,
) -> None:
    cases = _cases()
    reference = results_by_case[cases[0].case_id]
    reference_prompts = [turn.prompt for turn in reference.turns]
    reference_transcript = reference.public_transcript

    for case in cases[1:]:
        result = results_by_case[case.case_id]

        assert [turn.prompt for turn in result.turns] == reference_prompts
        assert result.public_transcript == reference_transcript
