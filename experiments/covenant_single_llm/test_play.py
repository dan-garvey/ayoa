import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from . import play
from .play import (
    init_run,
    load_run,
    make_request,
    play_turn,
    read_json,
)


def test_instruction_order_is_frozen_and_player_submission_stays_in_user_tail(tmp_path, monkeypatch):
    run = tmp_path / "run"
    init_run(run, "offline", "max", 12000)
    manifest, turns = load_run(run)
    assert manifest["instruction_order"] == ["covenant.txt", "system.txt"]
    expected = "\n\n".join(
        (run / name).read_text().strip() for name in manifest["instruction_order"]
    )
    monkeypatch.setattr(play, "INSTRUCTION_ORDER", ("system.txt", "covenant.txt"))
    submission = "I privately decide to use the pen name Moss Underhill."
    request = make_request(run, manifest, turns, submission)
    assert request["instructions"] == expected
    assert submission not in request["instructions"]
    assert request["input"] == [{"role": "user", "content": submission}]


@pytest.mark.parametrize("order", [None, ["system.txt", "system.txt"], ["player.json", "system.txt"]])
def test_missing_or_invalid_instruction_order_fails_before_call(tmp_path, order):
    run = tmp_path / "run"
    init_run(run, "offline", "max", 12000)
    manifest, turns = load_run(run)
    manifest["instruction_order"] = order
    with pytest.raises(ValueError, match="frozen instruction order"):
        make_request(run, manifest, turns, "Begin.")


def response(text="The dinner begins.", status="completed"):
    return SimpleNamespace(
        status=status, output=[], output_text=text, model="offline",
        id="response-test", usage=None,
        model_dump=lambda **kwargs: {"status": status, "output_text": text},
    )


def test_single_call_full_shared_context_and_restart(tmp_path):
    run = tmp_path / "run"
    init_run(run, "offline", "max", 12000)
    first = Mock()
    first.responses.create.return_value = response()
    play_turn(run, first, "I enter the dining hall.")
    first.responses.create.assert_called_once()
    request = first.responses.create.call_args.kwargs
    assert "Lord Verantus" in request["instructions"]
    assert "Rashid vel Amara" in request["instructions"]
    assert request["input"] == [{"role": "user", "content": "I enter the dining hall."}]
    second = Mock()
    second.responses.create.return_value = response("Ashara answers.")
    play_turn(run, second, "I ask Ashara a question.")
    second.responses.create.assert_called_once()
    assert second.responses.create.call_args.kwargs["input"] == [
        {"role": "user", "content": "I enter the dining hall."},
        {"role": "assistant", "content": "The dinner begins."},
        {"role": "user", "content": "I ask Ashara a question."},
    ]
    assert len(load_run(run)[1]) == 2


@pytest.mark.parametrize("status,text", [("incomplete", "A partial"), ("completed", "")])
def test_incomplete_or_empty_response_preserved_without_advancing(tmp_path, status, text):
    run = tmp_path / "run"
    init_run(run, "offline", "max", 12000)
    client = Mock()
    client.responses.create.return_value = response(text, status)
    with pytest.raises(RuntimeError):
        play_turn(run, client, "Begin.")
    assert load_run(run)[1] == []
    raw = list((run / "attempts").glob("*/response.json"))
    assert len(raw) == 1
    assert read_json(raw[0])["output_text"] == text


def test_existing_run_and_prompt_mutation_fail_before_model_call(tmp_path):
    run = tmp_path / "run"
    init_run(run, "offline", "max", 12000)
    with pytest.raises(FileExistsError):
        init_run(run, "offline", "max", 12000)
    (run / "system.txt").write_text("Changed instructions.")
    client = Mock()
    with pytest.raises(ValueError, match="Frozen prompt changed"):
        play_turn(run, client, "Begin.")
    client.responses.create.assert_not_called()


def test_rendered_instructions_exclude_implementation_details(tmp_path):
    run = tmp_path / "run"
    init_run(run, "offline", "max", 12000)
    manifest, turns = load_run(run)
    instructions = make_request(run, manifest, turns, "I arrive.")["instructions"]
    forbidden = re.compile(
        r"\b(?:openai|anthropic|claude|sdk|api[_ -]?key|EngineBridge|"
        r"checkpoint|dispatcher|test harness)\b|/home/|/mnt/",
        re.IGNORECASE,
    )
    assert forbidden.search(instructions) is None
