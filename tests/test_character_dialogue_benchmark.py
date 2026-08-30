from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scripts.run_character_dialogue_benchmark import (
    DEFAULT_IDENTITY_MODE,
    DEFAULT_MODEL,
    EXACT_SILENCE,
    HUMAN_REVIEW_FIELDS,
    IDENTITY_MODES,
    SUITES,
    BenchmarkOutputError,
    BenchmarkRequest,
    ModelCall,
    build_arg_parser,
    build_human_review_packet,
    load_benchmark_manifest,
    parse_observable_response,
    render_actor_messages,
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


def test_manifest_is_serial_and_has_two_scenes_for_every_case() -> None:
    cases = _cases()

    assert len(cases) >= 4
    assert set(SUITES) == {"ordinary_surface", "pressure"}
    assert {case.suite for case in cases} == set(SUITES)
    assert all(len(case.scenes) >= 2 for case in cases)
    for case in cases:
        assert len(case.actors) == 2
        assert len(case.named_actors) == 2
        assert len({scene.scene_id for scene in case.scenes}) == len(case.scenes)
        assert all(scene.turn_order for scene in case.scenes)
        assert all(scene.named_turn_order for scene in case.scenes)
        assert all(scene.frame for scene in case.scenes)
        for actor in (*case.actors, *case.named_actors):
            assert actor.lived_facts
            assert actor.habits
            assert actor.concrete_wants
            assert actor.withheld_acts
            assert not hasattr(actor, "speaking_behavior")


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
        for actor in (*case.actors, *case.named_actors):
            for value in (
                *actor.lived_facts,
                *actor.habits,
                *actor.concrete_wants,
                *actor.withheld_acts,
                *actor.known_facts,
                *actor.assumptions,
            ):
                assert value.casefold() not in public_text


def test_manifest_rejects_a_single_scene(tmp_path: Path) -> None:
    source = json.loads(
        Path("scripts/character_dialogue_benchmark_manifest.json").read_text()
    )
    source["cases"][0]["scenes"] = [
        {
            "scene_id": "only_scene",
            "frame": source["cases"][0]["scene_frame"],
            "prior_public_exchange": source["cases"][0]["prior_public_exchange"],
            "turn_order": source["cases"][0]["turn_order"],
            "named_turn_order": source["cases"][0]["named_turn_order"],
        }
    ]
    path = tmp_path / "one-scene.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(BenchmarkOutputError):
        # Keep the output parser assertion close to the manifest test so this
        # test file never grows a second obsolete contract helper.
        parse_observable_response("<private>hidden</private>")
    with pytest.raises(Exception, match="at least two separated scenes"):
        load_benchmark_manifest(path)


def test_actor_prompt_is_second_person_and_source_bounded() -> None:
    case = _cases()[0]
    actor = case.actors[0]
    counterpart = case.actors[1]
    first_scene = case.scenes[0]
    public_updates = [
        {
            "kind": "scene_start",
            "scene_index": 0,
            "text": first_scene.frame,
        },
        *(
            {
                "kind": "prior_public",
                "scene_index": 0,
                "sequence": entry.sequence,
                "speaker_name": case.actors[entry.speaker_slot].display_name,
                "text": entry.text,
            }
            for entry in first_scene.prior_public_exchange
        ),
    ]
    messages = render_actor_messages(
        case,
        actor.actor_id,
        scene_index=0,
        scene_turn_index=1,
        identity_mode="deidentified",
        public_updates=public_updates,
    )
    system = messages[0]["content"]
    prompt = _prompt_text(messages)
    assert f"You are {actor.display_name}" in system
    assert actor.lived_facts[0] in system
    assert actor.habits[0] in system
    assert actor.concrete_wants[0] in system
    assert actor.withheld_acts[0] in system
    assert counterpart.withheld_acts[0] not in prompt
    assert counterpart.known_facts[0] not in prompt
    assert "source_metadata" not in prompt
    assert "structural_inspiration" not in prompt
    assert "answer_debt" not in prompt
    assert "voice_swappability" not in prompt
    assert "speaking_behavior" not in prompt
    assert all(entry.text in prompt for entry in first_scene.prior_public_exchange)


def test_actor_material_is_not_copied_into_synthetic_checkpoint() -> None:
    result = _run(_cases()[0], identity_mode="deidentified", turns_per_scene=1)
    checkpoint_text = json.dumps(result.checkpoint.model_dump(mode="json"))
    for actor in result.case.actors:
        assert actor.lived_facts[0] not in checkpoint_text
        assert actor.withheld_acts[0] not in checkpoint_text
    assert result.checkpoint.player_primer == ""
    assert all(character.backstory == "" for character in result.checkpoint.characters)
    assert all(character.personality == "" for character in result.checkpoint.characters)


def test_only_observable_prose_or_exact_silence_is_accepted() -> None:
    assert parse_observable_response("  A public action.  ") == "A public action."
    assert parse_observable_response(f"  {EXACT_SILENCE}  ") == EXACT_SILENCE
    for invalid in ("", "   ", "<private>hidden</private>", "a <tag>"):
        with pytest.raises(BenchmarkOutputError):
            parse_observable_response(invalid)


def test_arg_parser_has_one_contract_and_explicit_luna_default() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.model == DEFAULT_MODEL
    assert args.identity_mode == DEFAULT_IDENTITY_MODE
    assert set(IDENTITY_MODES) == {"named", "deidentified", "both"}
    action_dests = {action.dest for action in parser._actions}
    assert "contract" not in action_dests
    assert "prompt_mode" not in action_dests
    assert "turns" not in action_dests
    explicit = parser.parse_args(
        ["--model", "anthropic:gpt-5.6-luna", "--identity-mode", "named"]
    )
    assert explicit.model == "anthropic:gpt-5.6-luna"
    assert explicit.identity_mode == "named"


def test_serial_scene_candidate_is_fed_into_the_next_scene_and_histories_are_full() -> None:
    case = _cases()[0]
    result = _run(case, identity_mode="deidentified", turns_per_scene=2)
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
    result = _run(case, identity_mode="deidentified", turns_per_scene=2)
    scene_two_first = next(turn for turn in result.turns if turn.scene_index == 1)
    prompt = _prompt_text(scene_two_first.prompt)
    earlier_turns = [turn for turn in result.turns if turn.scene_index == 0]
    assert any(
        turn.actor_id != scene_two_first.actor_id and turn.public_text in prompt
        for turn in earlier_turns
    )


def test_named_and_deidentified_runs_keep_identity_and_history_separate() -> None:
    case = _cases()[0]
    results = asyncio.run(
        run_benchmark(
            [case],
            model="synthetic-model",
            identity_mode="both",
            responder=_candidate_responder,
            turns_per_scene=1,
        )
    )
    assert {result.identity_mode for result in results} == {"named", "deidentified"}
    by_mode = {result.identity_mode: result for result in results}
    assert by_mode["named"].checkpoint_id != by_mode["deidentified"].checkpoint_id
    assert {character.name for character in by_mode["named"].checkpoint.characters} == {
        "Mara Venn",
        "Ilan Rowe",
    }
    assert {character.name for character in by_mode["deidentified"].checkpoint.characters} == {
        "A-17",
        "B-04",
    }
    assert "Mara Venn" in _prompt_text(by_mode["named"].turns[0].prompt)
    assert "Mara Venn" not in _prompt_text(by_mode["deidentified"].turns[0].prompt)


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
    assert set(review["fields"]) == set(HUMAN_REVIEW_FIELDS)
    assert all(field["value"] == "" for field in review["fields"].values())

    blinded = packet["blinded"]
    assert blinded["unit"] == "whole_serial_conversation"
    assert len(blinded["scenes"]) >= 2
    assert len(blinded["transcript"]) == len(result.public_transcript)
    assert len(blinded["speaker_sheets"]) == 2
    reviewer_text = json.dumps(blinded)
    assert result.case.actors[0].actor_id not in reviewer_text
    assert result.case.actors[0].display_name not in reviewer_text
    assert "source_metadata" not in reviewer_text
    assert packet["answer_key"]["speaker_mapping"]
    assert packet["answer_key"]["source_metadata"] == result.case.source_metadata

    output = write_benchmark_artifacts([result], tmp_path / "artifacts")
    raw_path = output / "raw" / "deidentified" / f"{result.case.case_id}.json"
    assert raw_path.is_file()
    artifact = json.loads(raw_path.read_text())
    assert len(artifact["scenes"]) >= 2
    assert "private_carry" not in json.dumps(artifact)
    assert "prompt_mode" not in json.dumps(artifact)
    assert "contract" not in json.dumps(artifact)
    assert (output / "review" / "blinded_transcripts.json").is_file()
    assert (output / "review" / "answer_key.json").is_file()
    assert (output / "benchmark_report.json").is_file()


def test_default_runner_keeps_every_scene_and_can_cap_each_scene() -> None:
    cases = _cases()[:2]
    results = asyncio.run(
        run_benchmark(
            cases,
            model="synthetic-model",
            identity_mode="deidentified",
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
            identity_mode="deidentified",
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
    assert first.checkpoint.character_conversations["c12"]
    assert second.checkpoint.character_conversations["c12"]
    assert first.checkpoint.session.turn_index == 2
    assert second.checkpoint.session.turn_index == 2
