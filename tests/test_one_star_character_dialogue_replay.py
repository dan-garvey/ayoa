from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

from app.schemas.conversation import ConversationMessage
from scripts.run_one_star_character_dialogue_replay import (
    CONVERSATION_REVIEW_CONTRACT,
    CONVERSATION_SCENARIOS,
    PRODUCTION_SEED_PATH,
    REPLAY_TURNS,
    ConversationScenario,
    ReplayTurn,
    _offline_client,
    _render_profile_blocks,
    build_blinded_voice_review,
    build_persisted_replay_messages,
    build_transition_review_rows,
    extract_direct_questions,
    load_checkpoint,
    overlay_current_seed_profiles,
    parse_scenario_checkpoint_args,
    public_private_overlap,
    run_replay,
)


def _write_synthetic_checkpoint(
    path: Path,
    *,
    with_replay_history: bool,
    scenario: ConversationScenario | None = None,
) -> Path:
    checkpoint = load_checkpoint(PRODUCTION_SEED_PATH)
    if with_replay_history:
        turn_counts = {"renna_holt": 4, "mirelle_voss": 7}
        for actor_id, turn_count in turn_counts.items():
            identity, state = _render_profile_blocks(checkpoint, actor_id)
            user = ConversationMessage(
                role="user",
                content=f"{identity}\n\n{state}",
            )
            assistant = ConversationMessage(
                role="assistant",
                content=(
                    f"{actor_id} makes the historical choice for this turn. "
                    "(I carry a distinct private concern forward.)"
                ),
            )
            checkpoint.character_conversations[actor_id] = [
                item
                for _ in range(turn_count)
                for item in (copy.deepcopy(user), copy.deepcopy(assistant))
            ]
    if scenario is not None:
        for actor_id in dict.fromkeys(scenario.actor_ids):
            actor = next(
                character
                for character in checkpoint.characters
                if character.character_id == actor_id
            )
            actor.pending_observations = list(scenario.tracked_state_markers)
    path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
    return path


def _write_scenario_checkpoints(tmp_path: Path) -> dict[str, Path]:
    return {
        scenario.scenario_id: _write_synthetic_checkpoint(
            tmp_path / f"{scenario.scenario_id}.json",
            with_replay_history=False,
            scenario=scenario,
        )
        for scenario in CONVERSATION_SCENARIOS
    }


def test_persisted_replay_keeps_checkpoint_history_order_and_cache_split(
    tmp_path: Path,
) -> None:
    checkpoint = load_checkpoint(
        _write_synthetic_checkpoint(
            tmp_path / "fixed.json",
            with_replay_history=True,
        )
    )
    turn = ReplayTurn(4, "mirelle_voss", 3)

    messages, historical_response = build_persisted_replay_messages(
        checkpoint,
        turn,
    )
    stored = checkpoint.character_conversations[turn.actor_id]
    current_role = next(
        character.public_sheet.role
        for character in checkpoint.characters
        if character.character_id == turn.actor_id
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1:] == [
        {"role": message.role, "content": message.content}
        for message in stored[:5]
    ]
    assert historical_response == stored[5].content
    assert "Mirelle Voss" not in messages[0]["content"]
    assert "Renna Holt" not in messages[0]["content"]
    assert current_role not in messages[0]["content"]
    assert current_role in messages[-1]["content"]


def test_checkpoint_replay_covers_four_renna_and_seven_mirelle_turns() -> None:
    by_actor = {
        actor_id: [turn.actor_turn for turn in REPLAY_TURNS if turn.actor_id == actor_id]
        for actor_id in {turn.actor_id for turn in REPLAY_TURNS}
    }
    assert by_actor == {
        "renna_holt": [1, 2, 3, 4],
        "mirelle_voss": [1, 2, 3, 4, 5, 6, 7],
    }


def test_current_seed_profile_replaces_old_profile_in_every_persisted_user(
    tmp_path: Path,
) -> None:
    replay = load_checkpoint(
        _write_synthetic_checkpoint(
            tmp_path / "replay.json",
            with_replay_history=True,
        )
    )
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
    old_intentions_enabled = replay_actor.private_state.intentions_enabled
    old_known_context = replay_actor.known_context
    sentinel = "CURRENT-SEED-PERSONALITY-SENTINEL"
    profile_actor.personality = sentinel
    profile_actor.private_state.intentions_enabled = not old_intentions_enabled
    profile_actor.known_context = "PROFILE-KNOWLEDGE-SENTINEL"

    overlaid = overlay_current_seed_profiles(replay, profile)
    messages, _historical = build_persisted_replay_messages(
        overlaid,
        ReplayTurn(4, "mirelle_voss", 3),
    )
    user_messages = [
        message["content"]
        for message in messages
        if message["role"] == "user"
    ]

    assert user_messages
    assert all(sentinel in message for message in user_messages)
    assert all(old_personality not in message for message in user_messages)
    assert replay_actor.private_state.intentions_enabled is old_intentions_enabled
    assert replay_actor.known_context == old_known_context
    assert all("PROFILE-KNOWLEDGE-SENTINEL" not in message for message in user_messages)


def test_offline_replay_writes_fixed_and_candidate_fed_evidence(tmp_path: Path) -> None:
    fixed_checkpoint_path = _write_synthetic_checkpoint(
        tmp_path / "fixed.json",
        with_replay_history=True,
    )
    scenario_checkpoint_paths = _write_scenario_checkpoints(tmp_path)
    report = asyncio.run(
        run_replay(
            tmp_path,
            phase="offline-check",
            client=_offline_client(),
            mode="offline",
            checkpoint_path=fixed_checkpoint_path,
            scenario_checkpoint_paths=scenario_checkpoint_paths,
        )
    )

    assert report["fixed_context_sample_count"] == 11
    expected_scenario_count = sum(
        len(scenario.actor_ids)
        for scenario in CONVERSATION_SCENARIOS
    )
    assert report["candidate_fed_scenario_sample_count"] == expected_scenario_count
    assert report["scenario_sample_counts"] == {
        scenario.scenario_id: len(scenario.actor_ids)
        for scenario in CONVERSATION_SCENARIOS
    }
    assert report["fixed_context_checkpoint_path"] == str(
        fixed_checkpoint_path
    )
    assert report["scenario_checkpoint_paths"] == {
        scenario_id: str(path)
        for scenario_id, path in scenario_checkpoint_paths.items()
    }
    assert len(report["profile_checkpoint_sha256"]) == 64
    assert len(report["system_prompt_sha256s"]) == 1
    fixed_rows = [
        json.loads(line)
        for line in (tmp_path / "offline-check/raw/fixed_context_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    scenario_rows = [
        json.loads(line)
        for line in (
            tmp_path / "offline-check/raw/candidate_fed_scenario_calls.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert len(fixed_rows) == 11
    assert len(scenario_rows) == expected_scenario_count
    for scenario in CONVERSATION_SCENARIOS:
        rows = [
            row
            for row in scenario_rows
            if row["scenario_id"] == scenario.scenario_id
        ]
        assert scenario.setup_observation in rows[0]["prompt"][-1]["content"]
        if scenario.scenario_id == "goblin_separate_escape":
            assert rows[0]["prior_actor_id"] == "tracked_setup"
            assert rows[0]["prior_public"] == scenario.setup_observation
            assert rows[0]["prior_direct_questions"]
            assert rows[0]["tracked_state_markers"] == list(
                scenario.tracked_state_markers
            )
        for previous, current in zip(rows, rows[1:]):
            previous_public = previous["parsed"]["public_text"]
            assert current["injected_observation"] == previous_public
            assert previous_public in current["prompt"][-1]["content"]
        assert all(
            row["history_message_count_after"]
            == row["history_message_count_before"] + 2
            for row in rows
        )
        rendered_prompts = json.dumps(
            [row["prompt"] for row in rows],
        )
        assert "review_setup_" not in rendered_prompts
        assert "replay_relay_" not in rendered_prompts
        assert scenario.scenario_id not in rendered_prompts
    assert all(row["provider_calls"] for row in fixed_rows + scenario_rows)

    transition_rows = json.loads(
        (tmp_path / "offline-check/transition_review_sheet.json").read_text()
    )
    assert len(transition_rows) == expected_scenario_count
    assert all("prior_public" in row and "current_public" in row for row in transition_rows)
    assert all(
        row["manual_review"]["immediate_uptake"] == ""
        for row in transition_rows
    )
    assert any(row["prior_direct_questions"] for row in transition_rows)

    voice_samples = json.loads(
        (tmp_path / "offline-check/voice_blind_samples.json").read_text()
    )
    voice_key = json.loads(
        (tmp_path / "offline-check/voice_answer_key.json").read_text()
    )
    assert len(voice_samples) == len(voice_key) == len(CONVERSATION_SCENARIOS)
    assert all(
        sample["sample_kind"] == "whole_scenario_transcript"
        for sample in voice_samples
    )
    assert all("scenario_id" not in sample for sample in voice_samples)
    assert all(
        [turn["turn"] for turn in sample["turns"]]
        == list(range(1, len(sample["turns"]) + 1))
        for sample in voice_samples
    )
    assert all(
        {turn["speaker"] for turn in sample["turns"]} <= {"A", "B"}
        for sample in voice_samples
    )
    assert all(
        set(item) == {
            "blind_id",
            "scenario_id",
            "speaker_map",
        }
        for item in voice_key
    )
    assert all(
        set(item["speaker_map"]) == {"A", "B"}
        for item in voice_key
    )
    source_scenario_order = [
        scenario.scenario_id
        for scenario in CONVERSATION_SCENARIOS
    ]
    blind_scenario_order = [item["scenario_id"] for item in voice_key]
    assert blind_scenario_order != source_scenario_order


def test_conversation_review_contract_requires_human_transition_review() -> None:
    dimensions = {
        dimension["id"]
        for dimension in CONVERSATION_REVIEW_CONTRACT["dimensions"]
    }

    assert dimensions == {
        "immediate_uptake",
        "referent_thread_continuity",
        "direct_question_handling",
        "bounded_figurative_anchors",
        "voice_swappability",
    }
    assert CONVERSATION_REVIEW_CONTRACT["reviewer"] == "human"
    assert CONVERSATION_REVIEW_CONTRACT["model_judge"] is False


def test_transition_review_preserves_question_and_leaves_semantics_unscored() -> None:
    scenario = ConversationScenario(
        scenario_id="synthetic",
        actor_ids=("renna_holt",),
        tracked_state_markers=("tracked",),
        setup_observation='A witness asks, "Did the latch move?"',
        review_purpose="Exercise the review structure.",
    )
    rows = [{
        "scenario_turn": 1,
        "actor_id": "renna_holt",
        "parsed": {"public_text": 'Renna says, "Maybe."'},
    }]

    review = build_transition_review_rows(rows, scenario)

    assert extract_direct_questions(review[0]["prior_public"]) == [
        "Did the latch move?",
    ]
    assert review[0]["prior_direct_questions"] == ["Did the latch move?"]
    assert review[0]["manual_review"]["direct_question_handling"] == ""
    assert review[0]["manual_review"]["immediate_uptake"] == ""


def test_blinded_voice_samples_separate_identity_answer_key() -> None:
    rows = [
        {
            "scenario_id": "synthetic",
            "scenario_turn": 1,
            "actor_id": "renna_holt",
            "setup_observation": (
                "Edren asks Renna and Mirelle to choose a bow or spear."
            ),
            "parsed": {
                "public_text": (
                    'Renna tells Mirelle, "Edren, take the bow; I have the '
                    'spear."'
                ),
            },
            "response": "unused",
        },
        {
            "scenario_id": "synthetic",
            "scenario_turn": 2,
            "actor_id": "mirelle_voss",
            "parsed": {
                "public_text": (
                    'Mirelle lowers her spear. "Renna, keep the arrow."'
                ),
            },
            "response": "unused",
        },
    ]

    samples, answer_key = build_blinded_voice_review(rows)

    assert len(samples) == 1
    assert samples[0]["sample_kind"] == "whole_scenario_transcript"
    assert [turn["turn"] for turn in samples[0]["turns"]] == [1, 2]
    serialized_sample = json.dumps(samples[0]).lower()
    for identity_cue in ("renna", "mirelle", "edren", "bow", "spear", "arrow"):
        assert identity_cue not in serialized_sample
    assert "scenario_id" not in samples[0]
    assert answer_key == [{
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
    }]


def test_scenario_checkpoint_cli_requires_explicit_unique_mapping() -> None:
    parsed = parse_scenario_checkpoint_args([
        "post_clear_changed_strength=/tmp/post.json",
        "goblin_separate_escape=/tmp/goblin.json",
    ])

    assert parsed == {
        "post_clear_changed_strength": Path("/tmp/post.json"),
        "goblin_separate_escape": Path("/tmp/goblin.json"),
    }


def test_public_private_overlap_flags_direct_explanatory_echo() -> None:
    echoed = public_private_overlap(
        'She shuts the red door and says, "The red door stays shut." '
        "(I need the red door to stay shut.)"
    )
    subtextual = public_private_overlap(
        'She palms the latch. "Try me." '
        "(I cannot let him discover that the room is empty.)"
    )

    assert echoed["overlap_of_shorter"] > subtextual["overlap_of_shorter"]
