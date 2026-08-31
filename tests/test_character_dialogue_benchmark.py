from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.engine.character_agent import CharacterAgentOutputError
from app.schemas.characters import CharacterRecord

from scripts.run_character_dialogue_benchmark import (
    DEFAULT_MODEL,
    EXACT_SILENCE,
    HUMAN_REVIEW_FIELDS,
    HUMAN_REVIEW_STATUSES,
    SUITES,
    BenchmarkManifestError,
    BenchmarkRequest,
    ModelCall,
    build_arg_parser,
    build_human_review_packet,
    load_benchmark_manifest,
    new_synthetic_checkpoint,
    run_benchmark,
    run_conversation,
    write_benchmark_artifacts,
)


def _cases():
    return load_benchmark_manifest()


def _run(case, **kwargs):
    return asyncio.run(run_conversation(case, responder=_candidate_responder, **kwargs))


def _candidate_responder(request: BenchmarkRequest) -> ModelCall:
    """A deterministic responder whose public candidates expose scene wiring."""

    if request.scene_turn_index == 1 and request.scene_index == 1:
        content = f"FOLLOWUP-{request.actor_id}-{request.turn_index}"
    else:
        content = f"CANDIDATE-{request.actor_id}-{request.turn_index}"
    return ModelCall(
        content=content,
        model=request.model,
        provider="synthetic",
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    )


def _metadata_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _metadata_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _metadata_strings(item)
    elif isinstance(value, str):
        yield value


def _prompt_text(messages) -> str:
    return json.dumps(messages, ensure_ascii=False)


def _case_with_actor_observations(tmp_path: Path):
    source = json.loads(
        Path("scripts/character_dialogue_benchmark_manifest.json").read_text()
    )
    source["cases"][0]["scenes"][0]["actor_observations"] = {
        "u03": [
            "At the scene start, you notice a blue thread caught under the meter cover."
        ],
        "v61": [
            "At the scene start, you hear the basement fan click twice before it starts."
        ],
    }
    source["cases"][0]["scenes"][1]["actor_observations"] = {
        "u03": [
            "At this return, you notice a fresh pinhole beside the lobby board."
        ],
        "v61": [
            "At this return, you hear water behind the wall before anyone speaks."
        ],
    }
    path = tmp_path / "actor-observations.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return load_benchmark_manifest(path)[0]


def test_manifest_is_serial_and_has_two_scenes_for_every_case() -> None:
    cases = _cases()

    assert len(cases) == 4
    assert set(SUITES) == {"ordinary_surface", "pressure"}
    assert {case.suite for case in cases} == set(SUITES)
    assert {case.suite: sum(item.suite == case.suite for item in cases) for case in cases} == {
        "ordinary_surface": 2,
        "pressure": 2,
    }
    assert all(len(case.scenes) >= 2 for case in cases)
    for case in cases:
        assert len(case.actors) == 2
        assert len({scene.scene_id for scene in case.scenes}) == len(case.scenes)
        assert all(scene.turn_order for scene in case.scenes)
        assert all(scene.frame for scene in case.scenes)
        assert all(
            scene.between_scene_public_history for scene in case.scenes[1:]
        )
        for actor in case.actors:
            assert isinstance(actor, CharacterRecord)
            assert actor.character_id
            assert actor.name
            assert actor.actor is not None


def test_public_scene_material_does_not_repeat_actor_local_material() -> None:
    for case in _cases():
        public_text = "\n".join(
            [scene.frame for scene in case.scenes]
            + [
                entry.text
                for scene in case.scenes
                for entry in scene.prior_public_exchange
            ]
        ).casefold()
        for actor in case.actors:
            for value in (
                fact.text for fact in (actor.actor.facts if actor.actor else ())
            ):
                if value:
                    assert value.casefold() not in public_text


def test_manifest_uses_sparse_uneven_actor_records_without_profile_checklists() -> None:
    source = json.loads(
        Path("scripts/character_dialogue_benchmark_manifest.json").read_text()
    )
    cases = _cases()
    legacy_profile_fields = {
        "assumptions",
        "backstory",
        "concrete_wants",
        "dossier",
        "habits",
        "known_facts",
        "lived_facts",
        "named_actors",
        "objective",
        "objectives",
        "profile",
        "ritual",
        "secret",
        "secrets",
        "speaking_behavior",
        "trauma",
        "voice",
        "withheld_acts",
    }
    all_counts: list[int] = []

    for raw_case, case in zip(source["cases"], cases, strict=True):
        raw_actors = raw_case["actors"]
        assert len(raw_actors) == len(case.actors) == 2
        assert "named_actors" not in raw_case
        assert not any(
            legacy_profile_fields & set(raw_actor) for raw_actor in raw_actors
        )
        counts = [len(actor.actor.facts) for actor in case.actors if actor.actor]
        origin_sets = [
            {fact.origin.value for fact in actor.actor.facts}
            for actor in case.actors
            if actor.actor
        ]
        assert len(counts) == len(origin_sets) == 2
        assert abs(counts[0] - counts[1]) >= 2
        assert origin_sets[0] != origin_sets[1]
        all_counts.extend(counts)

        for raw_actor, actor in zip(raw_actors, case.actors, strict=True):
            assert set(raw_actor["actor"]) == {"may_act_offstage", "facts"}
            assert not (legacy_profile_fields & set(raw_actor["actor"]))
            assert actor.actor is not None
            for raw_fact, fact in zip(
                raw_actor["actor"]["facts"], actor.actor.facts, strict=True
            ):
                assert set(raw_fact) == {"origin", "text"}
                assert fact.text.startswith(("You ", "Your "))

    assert min(all_counts) <= 2
    assert max(all_counts) <= 7
    assert len(set(all_counts)) >= 4


def test_manifest_rejects_a_single_scene(tmp_path: Path) -> None:
    source = json.loads(
        Path("scripts/character_dialogue_benchmark_manifest.json").read_text()
    )
    only_scene = dict(source["cases"][0]["scenes"][0])
    only_scene["scene_id"] = "only_scene"
    source["cases"][0]["scenes"] = [only_scene]
    path = tmp_path / "one-scene.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="at least two separated scenes"):
        load_benchmark_manifest(path)


def test_manifest_supports_a_three_character_conversation(tmp_path: Path) -> None:
    source = json.loads(
        Path("scripts/character_dialogue_benchmark_manifest.json").read_text()
    )
    case = source["cases"][0]
    third_actor = json.loads(json.dumps(case["actors"][0]))
    third_actor["character_id"] = "third_actor"
    third_actor["name"] = "Third Actor"
    third_actor["actor"]["facts"] = [
        {
            "origin": "witnessed",
            "text": "You watched both tenants move the envelope without opening it.",
        }
    ]
    case["actors"].append(third_actor)
    for scene in case["scenes"]:
        scene["turn_order"].append("third_actor")
    case["scenes"][0]["prior_public_exchange"].append(
        {
            "sequence": 5,
            "speaker_slot": 2,
            "text": "Neither of you has opened it.",
        }
    )
    path = tmp_path / "three-actors.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    parsed = load_benchmark_manifest(path)

    assert len(parsed[0].actors) == 3
    assert parsed[0].scenes[0].prior_public_exchange[-1].speaker_slot == 2
    result = _run(parsed[0], turns_per_scene=1)
    assert len(result.checkpoint.characters) == 3
    review = build_human_review_packet(result)
    assert len(review["blinded"]["speaker_sheets"]) == 3
    assert all(
        "across every scene" in sheet["instruction"]
        for sheet in review["blinded"]["speaker_sheets"]
    )


def test_manifest_rejects_prior_exchange_speaker_outside_ensemble(
    tmp_path: Path,
) -> None:
    source = json.loads(
        Path("scripts/character_dialogue_benchmark_manifest.json").read_text()
    )
    source["cases"][0]["scenes"][0]["prior_public_exchange"][0][
        "speaker_slot"
    ] = 2
    path = tmp_path / "bad-speaker-slot.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="reference one of 2 actors"):
        load_benchmark_manifest(path)


def test_manifest_rejects_obsolete_fields_instead_of_loading_legacy_shapes(
    tmp_path: Path,
) -> None:
    source = json.loads(
        Path("scripts/character_dialogue_benchmark_manifest.json").read_text()
    )
    source["cases"][0]["scene_frame"] = "legacy"
    source["cases"][0]["actors"][0]["dossier"] = {"voice": "legacy"}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="unknown fields"):
        load_benchmark_manifest(path)


def test_manifest_parses_per_scene_actor_observations_and_validates_actor_ids(
    tmp_path: Path,
) -> None:
    case = _case_with_actor_observations(tmp_path)

    assert case.scenes[0].actor_observations == {
        "u03": (
            "At the scene start, you notice a blue thread caught under the meter cover.",
        ),
        "v61": (
            "At the scene start, you hear the basement fan click twice before it starts.",
        ),
    }
    assert case.scenes[1].actor_observations["u03"] == (
        "At this return, you notice a fresh pinhole beside the lobby board.",
    )

    source = json.loads(
        Path("scripts/character_dialogue_benchmark_manifest.json").read_text()
    )
    source["cases"][0]["scenes"][0]["actor_observations"] = {
        "not_an_actor": ["A private observation for nobody."]
    }
    path = tmp_path / "unknown-actor-observation.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(BenchmarkManifestError, match="unknown actor"):
        load_benchmark_manifest(path)


def test_actor_startup_observations_are_private_user_tail_and_retained_after_commit(
    tmp_path: Path,
) -> None:
    case = _case_with_actor_observations(tmp_path)
    result = _run(case, turns_per_scene=2)

    scene = case.scenes[0]
    startup_by_actor = {
        actor_id: observations[0]
        for actor_id, observations in scene.actor_observations.items()
    }
    first_turns = {
        turn.actor_id: turn
        for turn in result.turns
        if turn.scene_index == 0
    }
    assert set(first_turns) == set(startup_by_actor)

    for actor_id, observation in startup_by_actor.items():
        prompt = first_turns[actor_id].prompt
        system_text = str(prompt[0]["content"])
        user_text = str(prompt[-1]["content"])
        assert observation not in system_text
        assert observation in user_text
        assert observation in user_text.split("<now>", 1)[-1]
        for other_actor_id, other_observation in startup_by_actor.items():
            if other_actor_id != actor_id:
                assert other_observation not in user_text

        history = result.checkpoint.character_conversations[actor_id]
        history_text = json.dumps(
            [message.model_dump(mode="json") for message in history],
            ensure_ascii=False,
        )
        assert observation in history_text
        assert result.checkpoint.characters[
            next(
                index
                for index, character in enumerate(result.checkpoint.characters)
                if character.character_id == actor_id
            )
        ].pending_observations == []

    public_text = json.dumps(result.public_transcript, ensure_ascii=False)
    assert all(observation not in public_text for observation in startup_by_actor.values())


def test_actor_startup_observations_stay_out_of_public_artifact_surfaces(
    tmp_path: Path,
) -> None:
    case = _case_with_actor_observations(tmp_path)
    result = _run(case, turns_per_scene=1)
    observations = tuple(
        observation
        for scene in case.scenes
        for values in scene.actor_observations.values()
        for observation in values
    )

    artifact = result.artifact()
    assert all(
        observation not in json.dumps(artifact["public_transcript"], ensure_ascii=False)
        for observation in observations
    )
    assert all(
        observation not in json.dumps(artifact["scenes"], ensure_ascii=False)
        for observation in observations
    )
    packet = build_human_review_packet(result)
    assert all(
        observation not in json.dumps(packet["blinded"], ensure_ascii=False)
        for observation in observations
    )

    output = write_benchmark_artifacts([result], tmp_path / "artifacts")
    blinded = (output / "review" / "blinded_transcripts.json").read_text()
    assert all(observation not in blinded for observation in observations)
    raw = (output / "raw" / f"{case.case_id}.json").read_text()
    assert any(observation in raw for observation in observations)


def test_between_scene_public_history_is_rendered_for_the_returning_scene() -> None:
    case = _cases()[0]
    history = case.scenes[1].between_scene_public_history
    result = _run(case, turns_per_scene=1)
    second_scene_turn = next(turn for turn in result.turns if turn.scene_index == 1)
    assert history in _prompt_text(second_scene_turn.prompt)


def test_actor_prompts_are_second_person_owner_bounded_and_metadata_free() -> None:
    for case in _cases():
        result = _run(case, turns_per_scene=2)
        first_turn_by_actor = {}
        for turn in result.turns:
            first_turn_by_actor.setdefault(turn.actor_id, turn)
        assert set(first_turn_by_actor) == {
            actor.character_id for actor in case.actors
        }

        for actor in case.actors:
            turn = first_turn_by_actor[actor.character_id]
            messages = turn.prompt
            system = messages[0]["content"]
            user = messages[-1]["content"]
            prompt = _prompt_text(messages)
            assert f"You are {actor.name}" not in system
            assert f"You are {actor.name}" in user
            assert actor.actor is not None
            assert all(fact.text not in system for fact in actor.actor.facts)
            assert all(fact.text in user for fact in actor.actor.facts)
            for counterpart in case.actors:
                if counterpart.character_id == actor.character_id:
                    continue
                assert counterpart.actor is not None
                assert all(
                    fact.text not in prompt for fact in counterpart.actor.facts
                )
            assert "<initial_objectives>" not in user
            assert "source_metadata" not in prompt
            assert all(
                value not in prompt
                for value in _metadata_strings(case.source_metadata)
                if value.strip()
            )
            assert case.scenes[turn.scene_index].frame in user
            assert turn.compact is False
            assert turn.cache is True


def test_case_character_records_are_deep_copied_into_each_checkpoint() -> None:
    case = _cases()[0]
    checkpoint = new_synthetic_checkpoint(case, model="synthetic-model")
    assert checkpoint.player_primer == ""
    for source, character in zip(case.actors, checkpoint.characters, strict=True):
        assert character is not source
        assert character.model_dump(mode="json") == source.model_dump(mode="json")
        assert character.actor is not source.actor


def test_production_output_parser_accepts_exact_silence_and_rejects_bad_turns() -> None:
    case = _cases()[0]

    def silence(request: BenchmarkRequest) -> ModelCall:
        return ModelCall(content=EXACT_SILENCE, model=request.model)

    silent = asyncio.run(
        run_conversation(case, responder=silence, turns_per_scene=1)
    )
    assert all(turn.public_text == EXACT_SILENCE for turn in silent.turns)
    for invalid in ("", "   ", "<silence/> then speaks", "<private_carry>x</private_carry>"):
        def bad(request: BenchmarkRequest, text=invalid) -> ModelCall:
            return ModelCall(content=text, model=request.model)

        with pytest.raises(CharacterAgentOutputError):
            asyncio.run(
                run_conversation(case, responder=bad, turns_per_scene=1)
            )


def test_arg_parser_has_one_contract_and_explicit_luna_default() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.model == DEFAULT_MODEL
    action_dests = {action.dest for action in parser._actions}
    assert "contract" not in action_dests
    assert "prompt_mode" not in action_dests
    assert "turns" not in action_dests
    explicit = parser.parse_args(["--model", "anthropic:gpt-5.6-luna"])
    assert explicit.model == "anthropic:gpt-5.6-luna"


def test_serial_scene_candidate_is_fed_into_the_next_scene_and_histories_are_full() -> None:
    case = _cases()[0]
    result = _run(case, turns_per_scene=2)
    assert [turn.scene_index for turn in result.turns] == [0, 0, 1, 1]
    scene_one = [turn for turn in result.turns if turn.scene_index == 0]
    scene_two = [turn for turn in result.turns if turn.scene_index == 1]
    scene_two_prompt = "\n".join(_prompt_text(turn.prompt) for turn in scene_two)
    for turn in scene_one:
        assert turn.public_text in scene_two_prompt
    for actor_id, history in result.checkpoint.character_conversations.items():
        assert len(history) == 4
        assert all(message.role in {"user", "assistant"} for message in history)
    assert result.public_transcript[-1]["scene_index"] == 1
    assert any(entry["kind"] == "scene_break" for entry in result.public_transcript)


def test_second_scene_receives_earlier_counterpart_candidate_not_only_own_history() -> None:
    case = _cases()[0]
    result = _run(case, turns_per_scene=2)
    scene_two_first = next(turn for turn in result.turns if turn.scene_index == 1)
    prompt = _prompt_text(scene_two_first.prompt)
    earlier_turns = [turn for turn in result.turns if turn.scene_index == 0]
    assert any(
        turn.actor_id != scene_two_first.actor_id and turn.public_text in prompt
        for turn in earlier_turns
    )


def test_benchmark_runs_one_isolated_result_per_case() -> None:
    case = _cases()[0]
    results = asyncio.run(
        run_benchmark(
            [case],
            model="synthetic-model",
            responder=_candidate_responder,
            turns_per_scene=1,
        )
    )
    assert len(results) == 1
    assert results[0].case is case
    assert results[0].model == "synthetic-model"


def test_pressure_pulses_are_public_scene_inputs() -> None:
    pressure = next(case for case in _cases() if case.suite == "pressure")
    result = _run(pressure, turns_per_scene=3)
    pulse = next(
        pulse
        for scene in pressure.scenes
        for pulse in scene.pressure_pulses
    )
    pulse_turn = next(
        turn for turn in result.turns if pulse.pulse_id in turn.pressure_pulse_ids
    )
    assert pulse.text in pulse_turn.prompt[-1]["content"]
    assert pulse.text in "\n".join(
        entry["text"]
        for entry in result.public_transcript
        if entry["kind"] == "pressure"
    )


def test_review_packet_is_blinded_and_covers_the_whole_serial_sequence(tmp_path: Path) -> None:
    result = _run(_cases()[0], turns_per_scene=2)
    packet = build_human_review_packet(result)
    review = packet["review"]
    assert review["model_judge"] is False
    assert review["auto_semantic_score"] is False
    assert review["unit"] == "whole_serial_conversation"
    assert review["status"] == ""
    assert tuple(review["allowed_statuses"]) == HUMAN_REVIEW_STATUSES
    assert set(review["fields"]) == set(HUMAN_REVIEW_FIELDS)
    assert all(field["value"] == "" for field in review["fields"].values())

    blinded = packet["blinded"]
    assert blinded["unit"] == "whole_serial_conversation"
    assert len(blinded["scenes"]) >= 2
    assert len(blinded["transcript"]) == len(result.public_transcript)
    assert len(blinded["speaker_sheets"]) == 2
    reviewer_text = json.dumps(blinded)
    assert all(set(scene) == {"scene_index", "label"} for scene in blinded["scenes"])
    assert result.case.case_id not in reviewer_text
    assert all(scene.scene_id not in reviewer_text for scene in result.case.scenes)
    assert all(scene.title not in reviewer_text for scene in result.case.scenes)
    assert result.case.actors[0].character_id not in reviewer_text
    assert result.case.actors[0].name not in reviewer_text
    assert "source_metadata" not in reviewer_text
    assert packet["answer_key"]["speaker_mapping"]
    assert packet["answer_key"]["source_metadata"] == result.case.source_metadata

    output = write_benchmark_artifacts([result], tmp_path / "artifacts")
    raw_path = output / "raw" / f"{result.case.case_id}.json"
    assert raw_path.is_file()
    artifact = json.loads(raw_path.read_text())
    assert len(artifact["scenes"]) >= 2
    assert "prompt_mode" not in json.dumps(artifact)
    assert artifact["turns"][0]["request"]["compact"] is False
    assert artifact["turns"][0]["request"]["cache"] is True
    first_prompt = artifact["turns"][0]["prompt"]
    assert [message["role"] for message in first_prompt] == ["system", "user"]
    first_actor = next(
        actor
        for actor in result.case.actors
        if actor.character_id == artifact["turns"][0]["actor_id"]
    )
    assert first_actor.name not in first_prompt[0]["content"]
    assert first_actor.name in first_prompt[-1]["content"]
    assert "structural_inspiration" not in json.dumps(
        artifact["turns"][0]["prompt"]
    )
    assert (output / "review" / "blinded_transcripts.json").is_file()
    assert (output / "review" / "answer_key.json").is_file()
    assert (output / "benchmark_report.json").is_file()


def test_default_runner_keeps_every_scene_and_can_cap_each_scene() -> None:
    cases = _cases()[:2]
    results = asyncio.run(
        run_benchmark(
            cases,
            model="synthetic-model",
            responder=_candidate_responder,
        )
    )
    assert len(results) == len(cases)
    assert all(len(result.case.scenes) >= 2 for result in results)
    assert all(
        len(result.turns)
        == sum(len(scene.turn_order) for scene in result.case.scenes)
        for result in results
    )
    clipped = asyncio.run(
        run_benchmark(
            cases,
            model="synthetic-model",
            responder=_candidate_responder,
            turns_per_scene=1,
        )
    )
    assert all(len(result.turns) == len(result.case.scenes) for result in clipped)


def test_fresh_runs_do_not_share_checkpoint_or_actor_history() -> None:
    case = _cases()[1]
    first = _run(case, model="model-a", turns_per_scene=1)
    second = _run(case, model="model-a", turns_per_scene=1)
    assert first.checkpoint is not second.checkpoint
    assert first.checkpoint_id != second.checkpoint_id
    assert first.initial_checkpoint_sha256 != second.initial_checkpoint_sha256
    assert first.checkpoint.character_conversations is not second.checkpoint.character_conversations
    assert all(
        first.checkpoint.character_conversations[actor.character_id]
        for actor in case.actors
    )
    assert all(
        second.checkpoint.character_conversations[actor.character_id]
        for actor in case.actors
    )
    assert first.checkpoint.session.turn_index == 2
    assert second.checkpoint.session.turn_index == 2
