import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from narrative import core
from narrative.__main__ import main


def raw_response(text="prose", *, status="completed", refusal=False, usage=None):
    block = (
        {"type": "refusal", "refusal": "no"}
        if refusal
        else {
            "type": "output_text",
            "text": text,
            "annotations": [],
        }
    )
    return {
        "id": "response-example",
        "object": "response",
        "created_at": 1,
        "model": "model-example",
        "status": status,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "id": "message-example",
                "status": "completed",
                "content": [block],
            }
        ],
        "usage": usage,
    }


class FakeClient:
    def __init__(self, *responses):
        self.responses = self
        self.results = iter(responses)
        self.requests = []

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        value = next(self.results)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(model_dump=lambda **_: value)


@pytest.fixture
def setup(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "author.txt").write_text("<instructions>Common writing contract.</instructions>\n")
    (prompts / "revision.txt").write_text("Revise this draft.\n")
    story = tmp_path / "story"
    story.mkdir()
    (story / "direction.md").write_text("<direction>Start at dawn.</direction>\n")
    (story / "canon.md").write_text("<world>The bell is silent. Only Ivo knows why.</world>\n")
    (story / "player.json").write_text(
        json.dumps({"name": "PLAYER_BINDING", "description": "PERSONA_DETAIL"})
    )

    def create(name="session", transport="proxy", **kwargs):
        session = tmp_path / name
        core.init_session(session, story, prompts=prompts, transport=transport, **kwargs)
        return session

    return create, prompts, story


def test_api_and_proxy_use_identical_requests_and_only_published_history(setup):
    create, _, _ = setup
    api, proxy = create("api", "api"), create("proxy")
    client = FakeClient(
        *(
            raw_response(text)
            for text in ["discarded draft", "published one", "draft two", "published two"]
        )
    )
    first = core.submit(api, "I knock.", client)
    author = core.submit(proxy, "I knock.")
    assert core.read_json(Path(author["request_json"])) == client.requests[0]
    editor = core.accept(proxy, author["request_id"], "discarded draft")
    assert core.read_json(Path(editor["request_json"])) == client.requests[1]
    final = core.accept(proxy, editor["request_id"], "published one")
    assert first["output"] == final["output"]
    request = client.requests[0]
    assert "PLAYER_BINDING" not in request["instructions"]
    assert "PERSONA_DETAIL" not in request["instructions"]
    assert "I knock." not in request["instructions"]
    assert request["input"][0]["content"] == "My character: PLAYER_BINDING. PERSONA_DETAIL"
    assert client.requests[1]["instructions"] == request["instructions"]
    assert client.requests[1]["input"][:-2] == request["input"]
    assert client.requests[1]["input"][-2:] == [
        {"role": "assistant", "content": "discarded draft"},
        {"role": "user", "content": "Revise this draft."},
    ]
    core.submit(api, "I wait.", client)
    next_request = core.submit(proxy, "I wait.")
    assert core.read_json(Path(next_request["request_json"])) == client.requests[2]
    assert "discarded draft" not in json.dumps(client.requests[2])
    assert "Revise this draft." not in json.dumps(client.requests[2])
    assert client.requests[2]["input"][2] == {"role": "assistant", "content": "published one"}
    assert (
        core.read_json(api / "attempts" / first["author"] / "response.json")["raw"]["output"][0][
            "content"
        ][0]["text"]
        == "discarded draft"
    )
    assert "discarded draft" not in (api / "transcript.md").read_text()


def test_source_updates_only_affect_new_sessions(setup):
    create, prompts, story = setup
    old = create()
    (prompts / "author.txt").write_text("NEW_WRITING_CONTRACT")
    (story / "canon.md").write_text("NEW_WORLD_FACT")
    new = create("new")
    old_request = core.read_json(Path(core.submit(old, "Start")["request_json"]))
    new_request = core.read_json(Path(core.submit(new, "Start")["request_json"]))
    assert "NEW_WORLD_FACT" not in old_request["instructions"]
    assert "NEW_WRITING_CONTRACT" not in old_request["instructions"]
    assert "NEW_WORLD_FACT" in new_request["instructions"]
    assert "NEW_WRITING_CONTRACT" in new_request["instructions"]


@pytest.mark.parametrize(
    "target",
    ["snapshot/canon.md", "snapshot/revision.txt", "snapshot/player.json", "manifest.json"],
)
def test_frozen_source_or_settings_change_blocks_before_dispatch(setup, target):
    create, _, _ = setup
    session = create(transport="api")
    path = session / target
    path.write_text(path.read_text() + "\n")
    client = FakeClient()
    with pytest.raises(core.NarrativeError, match="Frozen"):
        core.submit(session, "Start", client)
    assert client.requests == []
    assert core.read_json(session / "state.json")["pending"] is None


def test_existing_session_and_empty_submission_are_not_overwritten(setup):
    create, _, _ = setup
    session = create()
    before = (session / "state.json").read_bytes()
    with pytest.raises(FileExistsError):
        create()
    with pytest.raises(ValueError):
        core.submit(session, " \n")
    assert (session / "state.json").read_bytes() == before


@pytest.mark.parametrize(
    "bad",
    [
        raw_response("partial", status="incomplete"),
        raw_response("", refusal=True),
        raw_response(" \n"),
    ],
)
def test_editor_rejection_preserves_draft_and_retries_only_editor(setup, bad):
    create, _, _ = setup
    session = create(transport="api")
    client = FakeClient(raw_response("saved draft"), bad, raw_response("revised"))
    with pytest.raises(core.ResponseError):
        core.submit(session, "Start", client)
    state = core.load_session(session)[1]
    assert state["turns"] == []
    assert state["pending"]["author"]
    failed_id = state["pending"]["request_id"]
    assert core.read_json(session / "attempts" / failed_id / "response.json")["raw"] == bad
    with pytest.raises(core.NarrativeError, match="pending"):
        core.submit(session, "Different input", client)
    final = core.resume(session, client)
    assert final["output"] == "revised"
    assert len(client.requests) == 3
    assert client.requests[1] == client.requests[2]
    assert final["editor"] != failed_id


def test_failed_author_does_not_invoke_editor(setup):
    create, _, _ = setup
    session = create(transport="api")
    client = FakeClient(raw_response("", status="incomplete"))
    with pytest.raises(core.ResponseError):
        core.submit(session, "Start", client)
    assert len(client.requests) == 1
    assert core.load_session(session)[1]["pending"]["author"] is None


def test_transport_failure_and_interrupted_dispatch_are_explicit(setup):
    create, _, _ = setup
    session = create(transport="api")
    client = FakeClient(raw_response("draft"), TimeoutError(), raw_response("final"))
    with pytest.raises(core.NarrativeError, match="TimeoutError"):
        core.submit(session, "Start", client)
    state = core.load_session(session)[1]
    failed = session / "attempts" / state["pending"]["request_id"]
    assert (failed / "started.json").is_file()
    assert core.read_json(failed / "error.json")["type"] == "TimeoutError"
    # Losing even the error record after a crash must not lose the durable start marker.
    (failed / "error.json").unlink()
    result = core.resume(session, client)
    assert result["output"] == "final"
    assert client.requests[1] == client.requests[2]
    assert core.export(session)["api_calls_started"] == 3


@pytest.mark.parametrize(
    "stage,after_write", [("author", False), ("editor", False), ("editor", True)]
)
def test_saved_response_recovers_without_regeneration(setup, monkeypatch, stage, after_write):
    create, _, _ = setup
    session = create(transport="api")
    client = FakeClient(raw_response("draft"), raw_response("final"))
    original = core.write_json
    crashed = False

    def crash(path, value):
        nonlocal crashed
        qualifies = path.name == "state.json" and (
            (stage == "author" and value["pending"] and value["pending"]["author"])
            or (stage == "editor" and value["turns"])
        )
        if qualifies and not crashed:
            crashed = True
            if after_write:
                original(path, value)
            raise OSError("simulated crash")
        original(path, value)

    monkeypatch.setattr(core, "write_json", crash)
    with pytest.raises(OSError, match="simulated crash"):
        core.submit(session, "Start", client)
    result = core.resume(session, client)
    assert result["status"] == ("idle" if after_write else "published")
    assert len(client.requests) == 2
    assert len(core.load_session(session)[1]["turns"]) == 1
    assert core.load_session(session)[1]["turns"][0]["output"] == "final"


def test_failed_export_can_be_rebuilt_without_another_call(setup, monkeypatch):
    create, _, _ = setup
    session = create(transport="api")
    client = FakeClient(raw_response("draft"), raw_response("final"))
    original = core._export
    with monkeypatch.context() as patch:
        patch.setattr(
            core, "_export", lambda *a: (_ for _ in ()).throw(OSError("export interrupted"))
        )
        with pytest.raises(OSError):
            core.submit(session, "Start", client)
    assert core.resume(session, client)["status"] == "idle"
    assert len(client.requests) == 2
    assert (session / "transcript.md").is_file()
    assert core._export == original


def test_atomic_replacement_keeps_previous_record_on_write_failure(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    core.write_json(path, {"published": "old"})
    monkeypatch.setattr(core.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError):
        core.write_json(path, {"published": "new"})
    assert core.read_json(path) == {"published": "old"}
    assert list(tmp_path.iterdir()) == [path]


def test_proxy_duplicate_stale_empty_and_resume(setup):
    create, _, _ = setup
    session = create()
    first = core.submit(session, "Start")
    assert core.resume(session) == first
    with pytest.raises(core.ResponseError):
        core.accept(session, first["request_id"], " \n")
    assert (
        core.read_json(session / "attempts" / first["request_id"] / "response.json")["raw"] == " \n"
    )
    second = core.resume(session)
    assert second["request_id"] != first["request_id"]
    with pytest.raises(core.NarrativeError, match="Stale"):
        core.accept(session, first["request_id"], "late draft")
    editor = core.accept(session, second["request_id"], "draft")
    with pytest.raises(core.NarrativeError, match="Stale"):
        core.accept(session, second["request_id"], "duplicate draft")
    final = core.accept(session, editor["request_id"], " final\n")
    assert final["output"] == " final\n"
    with pytest.raises(core.NarrativeError, match="Stale"):
        core.accept(session, editor["request_id"], "duplicate final")
    summary = core.export(session)
    assert summary["requests_prepared"] == 3
    assert summary["responses_received"] == 3
    assert summary["api_calls_started"] == 0
    assert summary["reported_usage"] is None
    assert summary["recorded_api_seconds"] is None


@pytest.mark.parametrize("file", ["request.json", "request.txt"])
def test_changed_pending_request_cannot_be_accepted(setup, file):
    create, _, _ = setup
    session = create()
    result = core.submit(session, "Start")
    path = session / "attempts" / result["request_id"] / file
    if file.endswith("json"):
        request = core.read_json(path)
        request["input"][-1]["content"] = "Different submission"
        core.write_json(path, request)
    else:
        path.write_text("Different context")
    with pytest.raises(core.NarrativeError, match="request"):
        core.accept(session, result["request_id"], "draft")


def test_two_simultaneous_submissions_do_not_overwrite_one_another(setup):
    create, _, _ = setup
    session = create()

    def submit(text):
        try:
            return core.submit(session, text)["status"]
        except core.NarrativeError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, ["first action", "second action"]))
    assert sorted(results) == ["pending", "rejected"]
    assert len(list((session / "attempts").iterdir())) == 1


def test_usage_includes_failed_and_successful_stages(setup):
    create, _, _ = setup
    session = create(transport="api")
    usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 30},
        "output_tokens_details": {"reasoning_tokens": 5},
    }
    client = FakeClient(
        raw_response("draft", usage=usage),
        raw_response("partial", status="incomplete", usage=usage),
        raw_response("final", usage=usage),
    )
    with pytest.raises(core.ResponseError):
        core.submit(session, "Start", client)
    core.resume(session, client)
    summary = core.export(session)
    assert summary["responses_with_usage"] == 3
    assert summary["reported_usage"] == {
        "input_tokens": 300,
        "output_tokens": 60,
        "cached_tokens": 90,
        "reasoning_tokens": 15,
    }
    assert summary["published_turns"] == 1


def test_real_sdk_serialization_with_offline_transport(setup):
    create, _, _ = setup
    session = create(transport="api")
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=raw_response("draft" if len(sent) == 1 else "final"))

    with OpenAI(
        api_key="offline-example",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        assert core.submit(session, "Start", client)["output"] == "final"
    assert len(sent) == 2
    assert sent[0]["model"] == "gpt-5.6-terra"
    assert sent[0]["reasoning"] == {"effort": "max"}
    assert sent[0]["store"] is False
    assert sent[0]["truncation"] == "disabled"
    assert "tools" not in sent[0]


@pytest.mark.parametrize("story", ["covenant", "breakwater"])
def test_bundled_stories_render_in_isolation(tmp_path, story):
    session = tmp_path / story
    bundle = core.ROOT / "stories" / story
    core.init_session(session, bundle)
    request = core.read_json(Path(core.submit(session, "Start here")["request_json"]))
    expected = "\n\n".join(
        path.read_text().strip()
        for path in [core.ROOT / "prompts/author.txt", bundle / "direction.md", bundle / "canon.md"]
    )
    assert request["instructions"] == expected
    assert "Start here" not in request["instructions"]


def test_prompt_hygiene():
    banned = re.compile(
        r"\b(?:OpenAI|Anthropic|Claude|SDK|pytest|EngineBridge|LLMDispatcher|test harness)\b|OPENAI_API_KEY|OPEN_AI_ROUTER|/(?:home|mnt|tmp)/|[A-Z]:\\Users\\",
        re.I,
    )
    paths = list((core.ROOT / "prompts").glob("*.txt")) + list(
        (core.ROOT / "stories").glob("*/*.md")
    )
    assert paths
    for path in paths:
        assert not banned.search(path.read_text()), path


def test_cli_proxy_never_requires_credentials(setup, tmp_path, monkeypatch, capsys):
    _, prompts, story = setup
    session = tmp_path / "cli"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        main(["init", "--session", str(session), "--story", str(story), "--prompts", str(prompts)])
        == 0
    )
    capsys.readouterr()
    assert main(["turn", "--session", str(session), "--text", "Start"]) == 0
    request = json.loads(capsys.readouterr().out)
    output = tmp_path / "output.txt"
    output.write_text("draft")
    assert (
        main(
            [
                "accept",
                "--session",
                str(session),
                "--request-id",
                request["request_id"],
                "--output-file",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["stage"] == "editor"
    assert main(["export", "--session", str(session)]) == 0
