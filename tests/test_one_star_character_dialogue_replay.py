from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.schemas.characters import CharacterStatus
from app.schemas.conversation import ConversationMessage
from scripts.run_one_star_character_dialogue_replay import (
    CONVERSATION_REVIEW_CONTRACT,
    PRESSURE_REGRESSION_SCENARIOS,
    PRODUCTION_SEED_PATH,
    REPLAY_SCENARIOS,
    SUSTAINED_CONVERSATION_SCENARIOS,
    ConversationScenario,
    _offline_client,
    _render_profile_blocks,
    build_blinded_voice_review,
    build_pressure_review_rows,
    build_whole_conversation_review,
    extract_direct_questions,
    load_checkpoint,
    overlay_current_seed_profiles,
    run_replay,
    validate_scenario_inventory,
)


TRACKED_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "storage"
    / "sessions"
    / "42t24t"
    / "ckpt_0007.json"
)


def _write_synthetic_tracked_checkpoint(path: Path) -> Path:
    checkpoint = load_checkpoint(PRODUCTION_SEED_PATH)
    markers = [
        marker
        for scenario in REPLAY_SCENARIOS
        for marker in scenario.tracked_state_markers
    ]
    for actor_id in {actor_id for scenario in REPLAY_SCENARIOS for actor_id in scenario.actor_ids}:
        actor = next(
            character
            for character in checkpoint.characters
            if character.character_id == actor_id
        )
        actor.status = CharacterStatus.active
        actor.location = "niflheim_lobby"
        actor.pending_observations = list(markers)
    path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
    return path


def _synthetic_scenario_rows(
    scenario: ConversationScenario,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prior_actor_id = "tracked_setup"
    prior_public = scenario.setup_observation
    for turn, actor_id in enumerate(scenario.actor_ids, start=1):
        public_text = f'Speaker {turn} answers the present object. "Turn {turn}."'
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_turn": turn,
                "actor_id": actor_id,
                "prior_actor_id": prior_actor_id,
                "prior_public": prior_public,
                "checkpoint_path": str(TRACKED_CHECKPOINT_PATH),
                "profile_checkpoint_path": str(PRODUCTION_SEED_PATH),
                "setup_observation": scenario.setup_observation,
                "parsed": {"public_text": public_text},
                "response": f"{public_text} (Private intent {turn}.)",
            }
        )
        prior_actor_id = actor_id
        prior_public = public_text
    return rows


def test_scenario_inventory_requires_sustained_quiet_conflict_and_pressure() -> None:
    validate_scenario_inventory(REPLAY_SCENARIOS)

    quiet = [
        scenario for scenario in REPLAY_SCENARIOS if scenario.scenario_kind == "quiet"
    ]
    conflict = [
        scenario
        for scenario in REPLAY_SCENARIOS
        if scenario.scenario_kind == "conflict"
    ]
    pressure = [
        scenario
        for scenario in REPLAY_SCENARIOS
        if scenario.scenario_kind == "pressure"
    ]

    assert len(quiet) >= 2
    assert all(len(scenario.actor_ids) >= 8 for scenario in quiet)
    assert conflict
    assert all(len(scenario.actor_ids) >= 8 for scenario in conflict)
    assert pressure
    assert all(len(scenario.actor_ids) <= 3 for scenario in pressure)
    assert tuple(quiet + conflict) == SUSTAINED_CONVERSATION_SCENARIOS
    assert tuple(pressure) == PRESSURE_REGRESSION_SCENARIOS

    too_short = ConversationScenario(
        scenario_id="short_quiet",
        scenario_kind="quiet",
        actor_ids=("renna_holt", "mirelle_voss"),
        tracked_state_markers=("tracked",),
        setup_observation="A present object changes hands.",
        review_purpose="Exercise the inventory contract.",
    )
    with pytest.raises(ValueError, match="two quiet eight-turn"):
        validate_scenario_inventory((too_short, *conflict, *pressure))


def test_current_seed_profile_overlay_includes_known_context_and_preserves_topology(
    tmp_path: Path,
) -> None:
    replay = load_checkpoint(PRODUCTION_SEED_PATH)
    identity, state = _render_profile_blocks(replay, "mirelle_voss")
    replay.character_conversations["mirelle_voss"] = [
        ConversationMessage(role="user", content=f"{identity}\n\n{state}"),
        ConversationMessage(
            role="assistant",
            content="Mirelle makes one prior public choice. (A private motive.)",
        ),
    ]
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(replay.model_dump_json(indent=2), encoding="utf-8")
    replay = load_checkpoint(replay_path)
    profile = load_checkpoint(PRODUCTION_SEED_PATH)
    replay_actor = next(
        character
        for character in replay.characters
        if character.character_id == "mirelle_voss"
    )
    profile_actor = next(
        character
        for character in profile.characters
        if character.character_id == "mirelle_voss"
    )
    old_personality = replay_actor.personality
    old_known_context = replay_actor.known_context
    old_intentions_enabled = replay_actor.private_state.intentions_enabled
    original_roles = [
        message.role for message in replay.character_conversations["mirelle_voss"]
    ]
    original_assistant = replay.character_conversations["mirelle_voss"][1].content
    profile_actor.personality = "CURRENT-SEED-PERSONALITY-SENTINEL"
    profile_actor.known_context = "CURRENT-SEED-KNOWN-CONTEXT-SENTINEL"
    profile_actor.private_state.intentions_enabled = not old_intentions_enabled

    overlaid = overlay_current_seed_profiles(replay, profile)
    overlaid_actor = next(
        character
        for character in overlaid.characters
        if character.character_id == "mirelle_voss"
    )
    history = overlaid.character_conversations["mirelle_voss"]
    user_content = str(history[0].content)

    assert [message.role for message in history] == original_roles
    assert history[1].content == original_assistant
    assert "CURRENT-SEED-PERSONALITY-SENTINEL" in user_content
    assert "CURRENT-SEED-KNOWN-CONTEXT-SENTINEL" in user_content
    assert old_personality not in user_content
    assert old_known_context not in user_content
    assert overlaid_actor.known_context == "CURRENT-SEED-KNOWN-CONTEXT-SENTINEL"
    assert overlaid_actor.private_state.intentions_enabled is old_intentions_enabled
    source_actor_after = next(
        character
        for character in load_checkpoint(replay_path).characters
        if character.character_id == "mirelle_voss"
    )
    assert source_actor_after.personality == old_personality
    assert source_actor_after.known_context == old_known_context


def test_offline_replay_writes_one_candidate_fed_evidence_path(
    tmp_path: Path,
) -> None:
    checkpoint_path = _write_synthetic_tracked_checkpoint(
        tmp_path / "tracked-checkpoint.json"
    )
    report = asyncio.run(
        run_replay(
            tmp_path,
            phase="offline-check",
            client=_offline_client(),
            mode="offline",
            checkpoint_path=checkpoint_path,
        )
    )

    expected_turn_count = sum(len(scenario.actor_ids) for scenario in REPLAY_SCENARIOS)
    expected_sustained_turns = sum(
        len(scenario.actor_ids) for scenario in SUSTAINED_CONVERSATION_SCENARIOS
    )
    expected_pressure_turns = sum(
        len(scenario.actor_ids) for scenario in PRESSURE_REGRESSION_SCENARIOS
    )
    assert report["tracked_checkpoint_path"] == str(checkpoint_path)
    assert report["conversation_sample_count"] == expected_turn_count
    assert report["sustained_conversation_sample_count"] == expected_sustained_turns
    assert report["pressure_regression_sample_count"] == expected_pressure_turns
    assert report["scenario_sample_counts"] == {
        scenario.scenario_id: len(scenario.actor_ids) for scenario in REPLAY_SCENARIOS
    }
    assert report["scenario_kinds"] == {
        scenario.scenario_id: scenario.scenario_kind for scenario in REPLAY_SCENARIOS
    }
    assert report["provider_compaction_values"] == [False]
    assert "known_context" in report["profile_overlay_fields"]
    assert len(report["profile_checkpoint_sha256"]) == 64
    assert len(report["system_prompt_sha256s"]) == 1

    phase_dir = tmp_path / "offline-check"
    raw_paths = list((phase_dir / "raw").glob("*.jsonl"))
    assert raw_paths == [phase_dir / "raw" / "candidate_fed_conversation_calls.jsonl"]
    rows = [
        json.loads(line)
        for line in raw_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == expected_turn_count
    assert all(row["checkpoint_path"] == str(checkpoint_path) for row in rows)
    assert all(
        "current production seed known_context" in row["runtime_known_context_source"]
        for row in rows
    )
    assert all(
        call["request"]["compact"] is False
        for row in rows
        for call in row["provider_calls"]
    )

    for scenario in REPLAY_SCENARIOS:
        scenario_rows = [
            row for row in rows if row["scenario_id"] == scenario.scenario_id
        ]
        assert len(scenario_rows) == len(scenario.actor_ids)
        assert scenario.setup_observation in scenario_rows[0]["prompt"][-1]["content"]
        assert scenario_rows[0]["prior_actor_id"] == "tracked_setup"
        assert scenario_rows[0]["prior_public"] == scenario.setup_observation
        assert scenario_rows[0]["tracked_state_markers"] == list(
            scenario.tracked_state_markers
        )
        for previous, current in zip(scenario_rows, scenario_rows[1:]):
            previous_public = previous["parsed"]["public_text"]
            assert current["injected_observation"] == previous_public
            assert previous_public in current["prompt"][-1]["content"]
        assert all(
            row["history_message_count_after"]
            == row["history_message_count_before"] + 2
            for row in scenario_rows
        )
        rendered_prompts = json.dumps([row["prompt"] for row in scenario_rows])
        assert scenario.scenario_id not in rendered_prompts

    whole_reviews = json.loads(
        (phase_dir / "whole_conversation_review.json").read_text(encoding="utf-8")
    )
    assert len(whole_reviews) == len(SUSTAINED_CONVERSATION_SCENARIOS)
    assert all(len(review["turns"]) >= 8 for review in whole_reviews)
    assert all(
        review["manual_whole_conversation_review"]["non_neat_conflict"] == ""
        for review in whole_reviews
        if review["scenario_kind"] == "conflict"
    )
    assert all(
        review["manual_whole_conversation_review"]["non_neat_conflict"]
        == "not_applicable"
        for review in whole_reviews
        if review["scenario_kind"] == "quiet"
    )

    pressure_reviews = json.loads(
        (phase_dir / "pressure_regression_review.json").read_text(encoding="utf-8")
    )
    assert len(pressure_reviews) == expected_pressure_turns
    assert all(row["manual_review"]["fact_fidelity"] == "" for row in pressure_reviews)
    assert any(row["prior_direct_questions"] for row in pressure_reviews)

    voice_samples = json.loads(
        (phase_dir / "voice_blind_samples.json").read_text(encoding="utf-8")
    )
    voice_key = json.loads(
        (phase_dir / "voice_answer_key.json").read_text(encoding="utf-8")
    )
    assert len(voice_samples) == len(voice_key) == len(SUSTAINED_CONVERSATION_SCENARIOS)
    assert all(
        sample["sample_kind"] == "blinded_sustained_conversation"
        for sample in voice_samples
    )
    assert all(len(sample["turns"]) >= 8 for sample in voice_samples)
    assert all("scenario_id" not in sample for sample in voice_samples)


def test_review_contract_is_human_whole_conversation_analysis() -> None:
    dimensions = {
        dimension["id"]
        for dimension in CONVERSATION_REVIEW_CONTRACT["whole_conversation_dimensions"]
    }

    assert dimensions == {
        "public_established_backstory_and_depth",
        "character_specific_cadence_and_attention",
        "contradictions_change_later_turns",
        "non_neat_conflict",
        "fact_fidelity",
        "voice_swappability",
        "subtext_authority_support",
        "conversation_changes_something",
    }
    assert CONVERSATION_REVIEW_CONTRACT["reviewer"] == "human"
    assert CONVERSATION_REVIEW_CONTRACT["model_judge"] is False
    assert (
        "diagnostic only"
        in CONVERSATION_REVIEW_CONTRACT["pressure_regression_review"]["acceptance_role"]
    )
    assert "conversation-review-v3" in CONVERSATION_REVIEW_CONTRACT["supersedes"]


def test_whole_conversation_sheet_captures_turn_dynamics_and_carrying_debt() -> None:
    scenario = SUSTAINED_CONVERSATION_SCENARIOS[0]

    review = build_whole_conversation_review(
        _synthetic_scenario_rows(scenario),
        (scenario,),
    )[0]

    assert len(review["turns"]) == len(scenario.actor_ids)
    assert set(review["turns"][0]["manual_turn_review"]) == {
        "literal_topic",
        "interpersonal_attempt",
        "information_or_epistemology_used",
        "why_optimal_sentence_is_unavailable",
        "status_shift",
        "ritual_or_meaningful_deviation",
        "rhythm_change",
        "debt_or_consequence_into_next_turn",
        "authority_support",
        "exact_evidence",
    }
    assert set(review["turns"][0]["manual_turn_review"]["status_shift"]) == {
        "topic_control",
        "answer_debt",
        "interruption_or_repair",
    }
    assert set(review["manual_whole_conversation_review"]) == {
        "public_established_backstory_and_depth",
        "character_specific_cadence_and_attention",
        "contradictions_change_later_turns",
        "non_neat_conflict",
        "fact_fidelity",
        "voice_swappability",
        "subtext_authority_support",
        "conversation_changes_something",
        "speaker_interpersonal_attempts",
        "established_conversational_ritual",
        "meaningful_deviations",
        "ending_debt_or_consequence",
        "exact_evidence",
    }
    assert (
        review["manual_whole_conversation_review"]["non_neat_conflict"]
        == "not_applicable"
    )
    assert set(
        review["manual_whole_conversation_review"]["speaker_interpersonal_attempts"]
    ) == {"renna_holt", "mirelle_voss"}


def test_pressure_review_preserves_direct_question_without_scoring_it() -> None:
    scenario = ConversationScenario(
        scenario_id="synthetic_pressure",
        scenario_kind="pressure",
        actor_ids=("renna_holt",),
        tracked_state_markers=("tracked",),
        setup_observation='A witness asks, "Did the latch move?"',
        review_purpose="Exercise the compact regression structure.",
    )
    rows = [
        {
            "scenario_turn": 1,
            "actor_id": "renna_holt",
            "checkpoint_path": str(TRACKED_CHECKPOINT_PATH),
            "profile_checkpoint_path": str(PRODUCTION_SEED_PATH),
            "parsed": {"public_text": 'Renna says, "Maybe."'},
        }
    ]

    review = build_pressure_review_rows(rows, scenario)

    assert extract_direct_questions(review[0]["prior_public"]) == [
        "Did the latch move?",
    ]
    assert review[0]["prior_direct_questions"] == ["Did the latch move?"]
    assert review[0]["manual_review"]["direct_question_handling"] == ""
    assert review[0]["manual_review"]["immediate_uptake"] == ""


def test_blinded_voice_samples_separate_identity_answer_key() -> None:
    scenario = ConversationScenario(
        scenario_id="synthetic",
        scenario_kind="quiet",
        actor_ids=("renna_holt", "mirelle_voss"),
        tracked_state_markers=("tracked",),
        setup_observation=("Edren asks Renna and Mirelle to choose a bow or spear."),
        review_purpose="Exercise identity blinding.",
    )
    rows = _synthetic_scenario_rows(scenario)
    rows[0]["parsed"] = {
        "public_text": (
            'Renna tells Mirelle, "Edren, take the bow; I have the spear."'
        ),
    }
    rows[1]["parsed"] = {
        "public_text": 'Mirelle lowers her spear. "Renna, keep the arrow."',
    }

    samples, answer_key = build_blinded_voice_review(rows)

    assert len(samples) == 1
    assert samples[0]["sample_kind"] == "blinded_sustained_conversation"
    assert [turn["turn"] for turn in samples[0]["turns"]] == [1, 2]
    serialized_sample = json.dumps(samples[0]).lower()
    for identity_cue in (
        "renna",
        "mirelle",
        "edren",
        "bow",
        "spear",
        "arrow",
    ):
        assert identity_cue not in serialized_sample
    assert "scenario_id" not in samples[0]
    assert answer_key == [
        {
            "blind_id": samples[0]["blind_id"],
            "scenario_id": "synthetic",
            "speaker_map": {
                turn["speaker"]: actor_id
                for turn, actor_id in zip(
                    samples[0]["turns"],
                    ("renna_holt", "mirelle_voss"),
                    strict=True,
                )
            },
        }
    ]
