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
    (prompts / "regenerate.txt").write_text("Replace this passage.\n")
    (prompts / "checkup.txt").write_text("CHECKUP_TASK: review adherence and plan possibilities.\n")
    story = tmp_path / "story"
    story.mkdir()
    (story / "direction.md").write_text("<direction>Start at dawn.</direction>\n")
    (story / "canon.md").write_text("<world>The bell is silent. Only Ivo knows why.</world>\n")
    (story / "player.json").write_text(
        json.dumps({"name": "PLAYER_BINDING", "description": "PERSONA_DETAIL"})
    )

    def create(name="session", transport="proxy", checkup_every=0, **kwargs):
        session = tmp_path / name
        core.init_session(
            session,
            story,
            prompts=prompts,
            transport=transport,
            checkup_every=checkup_every,
            **kwargs,
        )
        return session

    return create, prompts, story


def test_api_and_proxy_publish_one_call_and_only_active_history(setup):
    create, _, _ = setup
    api, proxy = create("api", "api"), create("proxy")
    client = FakeClient(raw_response("published one"), raw_response("published two"))
    first = core.submit(api, "I knock.", client)
    prepared = core.submit(proxy, "I knock.")
    assert core.read_json(Path(prepared["request_json"])) == client.requests[0]
    final = core.accept(proxy, prepared["request_id"], "published one")
    assert first["output"] == final["output"] == "published one"
    assert len(client.requests) == 1
    assert len(first["responses"]) == 1
    request = client.requests[0]
    for volatile in ("PLAYER_BINDING", "PERSONA_DETAIL", "I knock."):
        assert volatile not in request["instructions"]
    assert request["input"] == [
        {"role": "user", "content": "My character: PLAYER_BINDING. PERSONA_DETAIL"},
        {"role": "user", "content": "I knock."},
    ]
    core.submit(api, "I wait.", client)
    following = core.submit(proxy, "I wait.")
    assert core.read_json(Path(following["request_json"])) == client.requests[1]
    assert client.requests[1]["input"][2] == {"role": "assistant", "content": "published one"}
    assert "Replace this passage." not in json.dumps(client.requests)
    assert core.export(api)["api_calls_started"] == 2


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


@pytest.mark.parametrize("transport", ["api", "proxy"])
def test_repeated_regeneration_replaces_only_latest_response_and_forgets_feedback(setup, transport):
    create, _, _ = setup
    session = create(transport=transport)
    requests = []

    def run(action, text, output):
        client = FakeClient(raw_response(output))
        result = action(session, text, client if transport == "api" else None)
        if transport == "proxy":
            requests.append(core.read_json(Path(result["request_json"])))
            result = core.accept(session, result["request_id"], output)
        else:
            requests.extend(client.requests)
        return result

    first = run(core.submit, "FIRST_ACTION", "FIRST_SCENE")
    original = run(core.submit, "SECOND_ACTION", "REJECTED_SCENE")
    frozen = {p: p.read_bytes() for p in (session / "attempts").rglob("*") if p.is_file()}
    revised = run(core.regenerate, "FIRST_CORRECTION", "ALSO_REJECTED")
    final = run(core.regenerate, "SECOND_CORRECTION", "CHOSEN_SCENE")
    state = core.load_session(session)[1]
    assert len(state["turns"]) == 2
    assert state["turns"][0] == {k: v for k, v in first.items() if k != "status"}
    assert final["input"] == "SECOND_ACTION"
    assert final["responses"] == original["responses"] + [
        revised["responses"][-1],
        final["responses"][-1],
    ]
    assert requests[2]["input"][:-1] == requests[1]["input"] + [
        {"role": "assistant", "content": "REJECTED_SCENE"}
    ]
    assert requests[2]["input"][-1]["content"].endswith("FIRST_CORRECTION")
    assert requests[3]["input"][-2] == {"role": "assistant", "content": "ALSO_REJECTED"}
    assert "FIRST_CORRECTION" not in json.dumps(requests[3])
    assert '"REJECTED_SCENE"' not in json.dumps(requests[3])
    assert all(p.read_bytes() == data for p, data in frozen.items())
    run(core.submit, "THIRD_ACTION", "NEXT_SCENE")
    history = json.dumps(requests[-1])
    for absent in (
        "REJECTED_SCENE",
        "ALSO_REJECTED",
        "FIRST_CORRECTION",
        "SECOND_CORRECTION",
        "Replace this passage.",
    ):
        assert absent not in history
        assert absent not in (session / "transcript.md").read_text()
    assert requests[-1]["input"][1:] == [
        {"role": "user", "content": "FIRST_ACTION"},
        {"role": "assistant", "content": "FIRST_SCENE"},
        {"role": "user", "content": "SECOND_ACTION"},
        {"role": "assistant", "content": "CHOSEN_SCENE"},
        {"role": "user", "content": "THIRD_ACTION"},
    ]


def test_regeneration_validation_and_concurrency_preserve_existing_story(setup):
    create, _, _ = setup
    session = create()
    before = (session / "state.json").read_bytes()
    with pytest.raises(core.NarrativeError, match="no passage"):
        core.regenerate(session, "Correct it")
    assert (session / "state.json").read_bytes() == before
    prepared = core.submit(session, "Start")
    with pytest.raises(core.NarrativeError, match="pending"):
        core.regenerate(session, "Correct it")
    core.accept(session, prepared["request_id"], "original")
    version = core.state_version(core.load_session(session)[1])
    for text in ("", "  ", None):
        with pytest.raises(ValueError):
            core.regenerate(session, text)
    with ThreadPoolExecutor() as pool:
        futures = [
            pool.submit(core.regenerate, session, text, expected_version=version)
            for text in ("First correction", "Second correction")
        ]
    assert sum(f.exception() is None for f in futures) == 1
    assert sum(isinstance(f.exception(), core.NarrativeError) for f in futures) == 1
    state = core.load_session(session)[1]
    assert state["turns"][0]["output"] == "original"
    pending = state["pending"]["request_id"]
    core.accept(session, pending, "replacement")
    before = (session / "state.json").read_bytes()
    for action in (core.submit, core.regenerate):
        with pytest.raises(core.NarrativeError, match="story changed"):
            action(session, "Stale instruction", expected_version=version)
    assert (session / "state.json").read_bytes() == before


def test_submission_identity_stays_out_of_model_context(setup):
    create, _, _ = setup
    session = create(transport="api")
    client = FakeClient(raw_response("original"), raw_response("replacement"))
    identifiers = ["a" * 32, "b" * 32]
    for action, identifier in zip((core.submit, core.regenerate), identifiers):
        result = action(session, "Instruction", client, submission_id=identifier)
        assert result["submission_id"] == identifier
        assert identifier not in json.dumps(client.requests)


@pytest.mark.parametrize(
    "target",
    ["snapshot/canon.md", "snapshot/regenerate.txt", "snapshot/player.json", "manifest.json"],
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
        raw_response(" "),
    ],
)
def test_rejected_regeneration_keeps_previous_passage_and_retries_same_request(setup, bad):
    create, _, _ = setup
    session = create(transport="api")
    client = FakeClient(raw_response("original"), bad, raw_response("replacement"))
    first = core.submit(session, "Start", client)
    with pytest.raises(core.ResponseError):
        core.regenerate(session, "CORRECTION_ONLY", client)
    state = core.load_session(session)[1]
    assert state["turns"][0]["output"] == first["output"]
    assert state["pending"]["kind"] == "regenerate"
    failed_id = state["pending"]["request_id"]
    assert core.read_json(session / "attempts" / failed_id / "response.json")["raw"] == bad
    with pytest.raises(core.NarrativeError, match="pending"):
        core.submit(session, "Different input", client)
    final = core.resume(session, client)
    assert final["output"] == "replacement" and final["input"] == "Start"
    assert final["responses"] == [first["responses"][0], final["responses"][-1]]
    assert failed_id not in final["responses"]
    assert len(client.requests) == 3 and client.requests[1] == client.requests[2]
    assert len(core.load_session(session)[1]["turns"]) == 1


def test_failed_response_stays_unpublished(setup):
    create, _, _ = setup
    session = create(transport="api")
    client = FakeClient(raw_response("", status="incomplete"))
    with pytest.raises(core.ResponseError):
        core.submit(session, "Start", client)
    assert len(client.requests) == 1
    assert core.load_session(session)[1]["turns"] == []


def test_transport_failure_and_interrupted_dispatch_are_explicit(setup):
    create, _, _ = setup
    session = create(transport="api")
    client = FakeClient(TimeoutError(), raw_response("final"))
    with pytest.raises(core.NarrativeError, match="TimeoutError"):
        core.submit(session, "Start", client)
    state = core.load_session(session)[1]
    failed = session / "attempts" / state["pending"]["request_id"]
    assert (failed / "started.json").is_file()
    assert core.read_json(failed / "error.json")["type"] == "TimeoutError"
    (failed / "error.json").unlink()
    assert core.resume(session, client)["output"] == "final"
    assert client.requests[0] == client.requests[1]
    assert core.export(session)["api_calls_started"] == 2


@pytest.mark.parametrize("regenerating", [False, True])
@pytest.mark.parametrize("after_write", [False, True])
def test_saved_response_recovers_without_another_call(
    setup, monkeypatch, regenerating, after_write
):
    create, _, _ = setup
    session = create(transport="api")
    if regenerating:
        core.submit(session, "Start", FakeClient(raw_response("original")))
    client = FakeClient(raw_response("final"))
    original = core.write_json
    crashed = False

    def crash(path, value):
        nonlocal crashed
        if (
            path.name == "state.json"
            and value["pending"] is None
            and value["turns"]
            and not crashed
        ):
            crashed = True
            if after_write:
                original(path, value)
            raise OSError("simulated crash")
        original(path, value)

    monkeypatch.setattr(core, "write_json", crash)
    action = core.regenerate if regenerating else core.submit
    with pytest.raises(OSError, match="simulated crash"):
        action(session, "Correct it" if regenerating else "Start", client)
    if regenerating and not after_write:
        assert core.load_session(session)[1]["turns"][0]["output"] == "original"
    result = core.resume(session, client)
    assert result["status"] == ("idle" if after_write else "published")
    assert len(client.requests) == 1
    state = core.load_session(session)[1]
    assert len(state["turns"]) == 1 and state["turns"][0]["output"] == "final"
    assert len(state["turns"][0]["responses"]) == (2 if regenerating else 1)


def test_failed_export_can_be_rebuilt_without_another_call(setup, monkeypatch):
    create, _, _ = setup
    session = create(transport="api")
    client = FakeClient(raw_response("final"))
    original = core._export
    with monkeypatch.context() as patch:
        patch.setattr(
            core, "_export", lambda *a: (_ for _ in ()).throw(OSError("export interrupted"))
        )
        with pytest.raises(OSError):
            core.submit(session, "Start", client)
    assert core.resume(session, client)["status"] == "idle"
    assert len(client.requests) == 1
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
        core.accept(session, first["request_id"], " ")
    second = core.resume(session)
    assert second["request_id"] != first["request_id"]
    with pytest.raises(core.NarrativeError, match="Stale"):
        core.accept(session, first["request_id"], "late response")
    final = core.accept(session, second["request_id"], " final\n")
    assert final["output"] == " final\n"
    with pytest.raises(core.NarrativeError, match="Stale"):
        core.accept(session, second["request_id"], "duplicate")
    summary = core.export(session)
    assert summary["requests_prepared"] == summary["responses_received"] == 2
    assert summary["api_calls_started"] == 0 and summary["reported_usage"] is None


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
        raw_response("partial", status="incomplete", usage=usage),
        raw_response("final", usage=usage),
    )
    with pytest.raises(core.ResponseError):
        core.submit(session, "Start", client)
    core.resume(session, client)
    summary = core.export(session)
    assert summary["responses_with_usage"] == 2
    assert summary["reported_usage"] == {
        "input_tokens": 200,
        "output_tokens": 40,
        "cached_tokens": 60,
        "reasoning_tokens": 10,
    }
    assert summary["published_turns"] == 1


def test_real_sdk_serialization_with_offline_transport(setup):
    create, _, _ = setup
    session = create(transport="api")
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        response = raw_response("final")
        response["output"].insert(
            0,
            {
                "id": "rs_example",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "EXPOSED_SUMMARY"}],
            },
        )
        return httpx.Response(200, json=response)

    with OpenAI(
        api_key="offline-example",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        turn = core.submit(session, "Start", client)
        assert turn["output"] == "final"
    assert len(sent) == 1
    assert sent[0]["model"] == "gpt-5.6-terra"
    assert sent[0]["reasoning"] == {"effort": "max", "summary": "detailed"}
    assert sent[0]["store"] is False
    assert sent[0]["truncation"] == "disabled"
    assert "tools" not in sent[0]
    saved = core.read_json(core.attempt_dir(session, turn["responses"][0]) / "response.json")
    assert core.response_summary(saved) == "EXPOSED_SUMMARY"
    assert "EXPOSED_SUMMARY" not in (session / "state.json").read_text()
    assert "EXPOSED_SUMMARY" not in (session / "transcript.md").read_text()


@pytest.mark.parametrize("story", ["covenant", "breakwater"])
def test_bundled_stories_render_in_isolation(tmp_path, story):
    session = tmp_path / story
    bundle = core.ROOT / "stories" / story
    core.init_session(session, bundle, player_name="CHOSEN_PROTAGONIST")
    request = core.read_json(Path(core.submit(session, "Start here")["request_json"]))
    expected = "\n\n".join(
        path.read_text().strip()
        for path in [core.ROOT / "prompts/author.txt", bundle / "direction.md", bundle / "canon.md"]
    )
    assert request["instructions"] == expected
    assert "Start here" not in request["instructions"]
    assert "CHOSEN_PROTAGONIST" not in request["instructions"]
    default = core.read_json(bundle / "player.json")
    assert not re.search(rf"\b{re.escape(default['name'].split()[0])}\b", expected, re.I)
    assert "CHOSEN_PROTAGONIST" in request["input"][0]["content"]
    assert default["description"] in request["input"][0]["content"]


@pytest.mark.parametrize("transport", ["api", "proxy"])
def test_renaming_preserves_evidence_and_reaches_generation_and_regeneration(setup, transport):
    create, _, story = setup
    override = story / "custom.json"
    core.write_json(override, {"name": "DEFAULT_NAME", "description": "FULL_BACKGROUND"})
    session = create(transport=transport, player_file=override, player_name=" Éloi   Vale ")
    client = FakeClient(raw_response("Éloi Vale opens the letter."))
    result = core.submit(session, "I, Éloi, open it.", client if transport == "api" else None)
    if transport == "proxy":
        core.accept(session, result["request_id"], "Éloi Vale opens the letter.")
    _, old = core.load_session(session)
    saved = {
        p: p.read_bytes() for p in session.rglob("*") if p.is_file() and p.name != "state.json"
    }
    core.rename_player(session, "Renée O'Vale")
    core.export(session)
    assert core.load_session(session)[1]["turns"] == old["turns"]
    assert all(p.read_bytes() == data for p, data in saved.items())
    before = (session / "state.json").read_bytes()
    core.rename_player(session, " Renée O'Vale ")
    assert (session / "state.json").read_bytes() == before
    old_request = core.read_json(
        core.attempt_dir(session, old["turns"][0]["responses"][0]) / "request.json"
    )
    for action in (core.regenerate, core.submit):
        client = FakeClient(raw_response("new passage"))
        result = action(session, "New instruction", client if transport == "api" else None)
        request = (
            client.requests[0]
            if transport == "api"
            else core.read_json(Path(result["request_json"]))
        )
        if transport == "proxy":
            core.accept(session, result["request_id"], "new passage")
        assert request["instructions"] == old_request["instructions"]
        identity = request["input"][0]["content"]
        assert (
            "Renée O'Vale" in identity and "Éloi Vale" in identity and "FULL_BACKGROUND" in identity
        )
        assert "DEFAULT_NAME" not in identity
    core.rename_player(session, "Éloi Vale")
    assert core.player_identity(session, core.load_session(session)[1])["previous_names"] == [
        "Renée O'Vale"
    ]


@pytest.mark.parametrize("invalid", ["", " \n\t ", "a" * 81, "bad\x00name", "bad\ud800name", 123])
def test_invalid_name_never_creates_or_changes_a_session(setup, tmp_path, invalid):
    create, _, _ = setup
    with pytest.raises(ValueError):
        create("invalid", player_name=invalid)
    assert not (tmp_path / "invalid").exists()
    session = create()
    before = (session / "state.json").read_bytes()
    with pytest.raises(ValueError):
        core.rename_player(session, invalid)
    assert (session / "state.json").read_bytes() == before


def test_pending_generation_and_regeneration_prevent_rename(setup):
    create, _, _ = setup
    session = create()
    for action in (core.submit, core.regenerate):
        prepared = action(session, "Start")
        before = (session / "state.json").read_bytes()
        with pytest.raises(core.NarrativeError, match="pending response"):
            core.rename_player(session, "Jules")
        assert (session / "state.json").read_bytes() == before
        core.accept(session, prepared["request_id"], "published")
    core.rename_player(session, "Jules")
    request = core.read_json(Path(core.submit(session, "Continue")["request_json"]))
    assert "Jules" in request["input"][0]["content"]


def test_name_changes_and_turns_share_one_stale_state_check(setup):
    create, _, _ = setup
    session = create()
    version = core.state_version(core.load_session(session)[1])
    with ThreadPoolExecutor() as pool:
        futures = [
            pool.submit(core.rename_player, session, name, expected_version=version)
            for name in ("Avery", "Morgan")
        ]
    assert sum(future.exception() is None for future in futures) == 1
    assert sum(isinstance(future.exception(), core.NarrativeError) for future in futures) == 1
    before = (session / "state.json").read_bytes()
    with pytest.raises(core.NarrativeError, match="story changed"):
        core.submit(session, "A stale turn.", expected_version=version)
    assert (session / "state.json").read_bytes() == before
    assert not list((session / "attempts").iterdir())
    current = core.state_version(core.load_session(session)[1])
    assert core.submit(session, "A fresh turn.", expected_version=current)["kind"] == "turn"


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


@pytest.mark.parametrize("transport", ["api", "proxy"])
def test_checkup_cadence_guidance_replacement_and_regeneration(setup, transport):
    create, _, _ = setup
    session = create(transport=transport, checkup_every=2)
    requests = []

    def run(action, text, *replies):
        client = FakeClient(*(raw_response(reply) for reply in replies))
        result = action(session, text, client if transport == "api" else None)
        if transport == "api":
            requests.extend(client.requests)
        else:
            for reply in replies:
                assert result["status"] == "pending"
                requests.append(core.read_json(Path(result["request_json"])))
                result = core.accept(session, result["request_id"], reply)
        assert result["status"] == "published"
        return result

    run(core.submit, "OPENING_INPUT", "FIRST_SCENE")
    run(core.submit, "SECOND_INPUT", "OLD_PRIVATE_PLAN", "REJECTED_SCENE")
    run(core.regenerate, "DISCARDED_FEEDBACK", "REPLACEMENT_SCENE")
    run(core.submit, "THIRD_INPUT", "THIRD_SCENE")
    run(core.submit, "FOURTH_INPUT", "NEW_PRIVATE_PLAN", "FOURTH_SCENE")
    run(core.submit, "FIFTH_INPUT", "FIFTH_SCENE")
    assert len(requests) == 8
    assert "CHECKUP_TASK" in requests[1]["input"][-1]["content"]
    assert requests[1]["input"][-2]["content"] == "SECOND_INPUT"
    assert "CHECKUP_TASK" in requests[5]["input"][-1]["content"]
    assert len({request["instructions"] for request in requests}) == 1
    for index in (2, 3, 4, 5):
        assert "OLD_PRIVATE_PLAN" in json.dumps(requests[index]["input"])
    for index in (6, 7):
        assert "NEW_PRIVATE_PLAN" in json.dumps(requests[index]["input"])
        assert "OLD_PRIVATE_PLAN" not in json.dumps(requests[index])
    for index in (4, 5, 6, 7):
        assert "REJECTED_SCENE" not in json.dumps(requests[index])
        assert "DISCARDED_FEEDBACK" not in json.dumps(requests[index])
        assert "REPLACEMENT_SCENE" in json.dumps(requests[index])
    assert "REJECTED_SCENE" not in json.dumps(requests[1])
    for request in requests:
        assert "PRIVATE_PLAN" not in request["instructions"]
        assert request["reasoning"] == {"effort": "max", "summary": "detailed"}
    _, state = core.load_session(session)
    assert [item["before_turn"] for item in state["checkups"]] == [1, 3]
    published_ids = {item for turn in state["turns"] for item in turn["responses"]}
    assert all(item["request_id"] not in published_ids for item in state["checkups"])
    assert "PRIVATE_PLAN" not in json.dumps(state)
    assert "PRIVATE_PLAN" not in (session / "transcript.md").read_text()
    summary = core.export(session)
    assert summary["published_turns"] == 5
    assert summary["completed_checkups"] == 2
    assert summary["requests_prepared"] == 8


@pytest.mark.parametrize("after_write", [False, True])
def test_completed_checkup_survives_crash_without_another_call(setup, monkeypatch, after_write):
    create, _, _ = setup
    session = create(transport="api", checkup_every=1)
    client = FakeClient(raw_response("SAVED_GUIDANCE"), raw_response("Visible passage."))
    original = core.write_json

    def crash(path, value):
        if (
            path.name == "state.json"
            and value["checkups"]
            and value["pending"]["request_id"] is None
        ):
            if after_write:
                original(path, value)
            raise OSError("simulated checkup crash")
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(core, "write_json", crash)
        with pytest.raises(OSError, match="simulated checkup crash"):
            core.submit(session, "Begin.", client)
    assert not core.load_session(session)[1]["turns"]
    assert core.resume(session, client)["output"] == "Visible passage."
    assert len(client.requests) == 2
    assert "SAVED_GUIDANCE" in json.dumps(client.requests[1])
    assert len(core.load_session(session)[1]["checkups"]) == 1


@pytest.mark.parametrize(
    "bad",
    [
        raw_response(""),
        raw_response("partial", status="incomplete"),
        raw_response(refusal=True),
        OSError("offline"),
    ],
)
def test_checkup_failure_stops_before_author_and_resume_retries_explicitly(setup, bad):
    create, _, _ = setup
    session = create(transport="api", checkup_every=1)
    client = FakeClient(bad, raw_response("PRIVATE_NOTES"), raw_response("Story."))
    with pytest.raises(core.NarrativeError):
        core.submit(session, "Begin.", client)
    state = core.load_session(session)[1]
    assert not state["turns"] and not state["checkups"]
    assert state["pending"]["stage"] == "checkup"
    assert len(client.requests) == 1
    failed_id = state["pending"]["request_id"]
    assert core.resume(session, client)["output"] == "Story."
    assert len(client.requests) == 3
    assert client.requests[0] == client.requests[1]
    assert core.load_session(session)[1]["checkups"][0]["request_id"] != failed_id
    assert len(list((session / "attempts").iterdir())) == 3


def test_author_failure_reuses_checkup_and_failed_regeneration_keeps_it(setup):
    create, _, _ = setup
    session = create(transport="api", checkup_every=1)
    client = FakeClient(
        raw_response("PRIVATE_NOTES"),
        OSError("offline"),
        raw_response("Original."),
        raw_response("rejected", status="incomplete"),
        raw_response("Replacement."),
    )
    with pytest.raises(core.NarrativeError):
        core.submit(session, "Begin.", client)
    state = core.load_session(session)[1]
    assert state["pending"]["stage"] == "author" and len(state["checkups"]) == 1
    assert not state["turns"]
    core.resume(session, client)
    assert client.requests[1] == client.requests[2]
    with pytest.raises(core.ResponseError):
        core.regenerate(session, "Fix it.", client)
    assert core.load_session(session)[1]["turns"][0]["output"] == "Original."
    core.resume(session, client)
    assert client.requests[3] == client.requests[4]
    assert len(core.load_session(session)[1]["checkups"]) == 1
    assert "CHECKUP_TASK" not in json.dumps(client.requests[1:])


def test_default_checkup_interval_and_disabled_mode(setup, tmp_path):
    create, prompts, story = setup
    session = tmp_path / "defaults"
    assert core.init_session(session, story, prompts=prompts)["checkup_every"] == 5
    for number in range(1, 6):
        pending = core.submit(session, f"Player {number}")
        assert pending["stage"] == ("checkup" if number == 5 else "author")
        if number == 5:
            pending = core.accept(session, pending["request_id"], "Private planning.")
            assert pending["stage"] == "author"
        core.accept(session, pending["request_id"], f"Passage {number}")
    disabled = create("disabled", checkup_every=0)
    for _ in range(7):
        pending = core.submit(disabled, "Continue.")
        assert pending["stage"] == "author"
        core.accept(disabled, pending["request_id"], "Story.")
    assert core.load_session(disabled)[1]["checkups"] == []
    assert core.export(disabled)["requests_prepared"] == 7


@pytest.mark.parametrize("interval", [-1, 1.5, True, "5"])
def test_invalid_checkup_interval_does_not_create_session(setup, interval):
    create, prompts, _ = setup
    with pytest.raises(ValueError, match="checkup_every"):
        create(checkup_every=interval)
    assert not (prompts.parent / "session").exists()


def test_automatic_proxy_preserves_line_endings_in_regeneration_and_publication(setup):
    create, _, _ = setup
    session = create()
    outputs = iter(["Original.\r\nAnother line.", "Replacement.\r\nAnother line."])
    requests = []

    def proxy(request):
        requests.append(request)
        return {"raw": next(outputs), "status": "completed"}

    core.submit(session, "My turn.\r\nAnother line.", proxy)
    result = core.regenerate(session, "Correction.\r\nMore instructions.", proxy)
    assert result["output"] == "Replacement.\r\nAnother line."
    assert requests[0]["input"][-1]["content"] == "My turn.\r\nAnother line."
    assert requests[1]["input"][-2]["content"] == "Original.\r\nAnother line."
    assert requests[1]["input"][-1]["content"].endswith("Correction.\r\nMore instructions.")


def test_cli_proxy_never_requires_credentials(setup, tmp_path, monkeypatch, capsys):
    _, prompts, story = setup
    session = tmp_path / "cli"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        main(
            [
                "init",
                "--session",
                str(session),
                "--story",
                str(story),
                "--prompts",
                str(prompts),
                "--player-name",
                "Avery",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert core.player_identity(session, core.load_session(session)[1])["name"] == "Avery"
    assert main(["rename", "--session", str(session), "--player-name", "Morgan"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "Morgan"
    assert main(["turn", "--session", str(session), "--text", "Start"]) == 0
    request = json.loads(capsys.readouterr().out)
    assert "Morgan" in core.read_json(Path(request["request_json"]))["input"][0]["content"]
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
    assert capsys.readouterr().out.strip() == "draft"
    assert main(["export", "--session", str(session)]) == 0
