from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .play import (
    init_run,
    load_run,
    play_turn,
    read_json,
)


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
