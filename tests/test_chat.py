import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from narrative import core
from narrative.chat import format_text


def create(service):
    response = service.client.post("/api/sessions", json={"story": "harbor"})
    assert response.status_code == 201
    return response.json()["id"]


def version(service, name):
    return service.client.get("/api/session", params={"id": name}).json()["version"]


def test_proxy_chat_publishes_once_and_only_displays_active_story(chat_service):
    service = chat_service()
    client = service.client
    name = create(service)
    sent = client.post(
        "/api/turn",
        json={
            "id": name,
            "text": "I wait.\n\n*Quietly.*",
            "expected_version": version(service, name),
        },
    )
    assert sent.status_code == 200 and sent.json()["turns"] == []
    assert "<em>Quietly.</em>" in sent.json()["pending"]["input_html"]
    packet = client.get("/api/handoff", params={"id": name}).json()
    assert "SECRET_CANON" in packet["text"] and packet["kind"] == "turn"
    final = "The **bell** rings.\nA second line.\n\n> An old promise.\n\n<script>alert(1)</script>"
    published = client.post(
        "/api/accept", json={"id": name, "request_id": packet["request_id"], "text": final}
    )
    assert published.status_code == 200
    view = published.json()
    assert view["turn_count"] == 1 and view["pending"] is None
    assert view["turns"][0]["output"] == final
    html = view["turns"][0]["output_html"]
    assert "<strong>bell</strong>" in html and "<br" in html and "<blockquote>" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    for route in ("/api/session", "/api/sessions", "/api/bootstrap"):
        response = client.get(route, params={"id": name})
        assert response.status_code == 200 and "SECRET_CANON" not in response.text
    assert final in client.get("/api/transcript", params={"id": name}).text
    duplicate = client.post(
        "/api/accept", json={"id": name, "request_id": packet["request_id"], "text": "replacement"}
    )
    assert duplicate.status_code == 409


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "nested/../../outside"])
def test_browser_routes_cannot_escape_session_directory(chat_service, name):
    service = chat_service()
    response = service.client.get("/api/session", params={"id": name})
    assert response.status_code == 409
    assert service.client.get("/narrative/core.py").status_code == 404
    assert service.client.get("/.env").status_code == 404


def test_session_symlinks_cannot_disclose_outside_files(chat_service, tmp_path):
    service = chat_service()
    name = create(service)
    outside = tmp_path / "outside"
    service.app.session_path(name).rename(outside)
    (service.app.sessions / "escape").symlink_to(outside, target_is_directory=True)
    assert service.client.get("/api/session", params={"id": "escape"}).status_code == 409
    assert service.client.get("/api/sessions").json()["sessions"] == []


def test_http_requires_local_host_origin_and_request_token(chat_service):
    service = chat_service()
    assert (
        service.client.get("/api/bootstrap", headers={"Host": "attacker.example"}).status_code
        == 403
    )
    assert (
        service.client.get(
            "/api/bootstrap", headers={"Origin": "https://attacker.example"}
        ).status_code
        == 403
    )
    assert (
        service.client.post(
            "/api/sessions", json={"story": "harbor"}, headers={"X-Chat-Token": ""}
        ).status_code
        == 403
    )
    assert (
        service.client.post(
            "/api/sessions",
            json={"story": "harbor"},
            headers={"Origin": "https://attacker.example"},
        ).status_code
        == 403
    )
    response = service.client.get("/")
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Cache-Control"] == "no-store"
    assert service.client.post("/api/turn", content="not-json").status_code == 409
    assert service.client.post("/api/sessions", json=[]).status_code == 409


def test_markdown_keeps_formatting_without_executable_html_or_images():
    html = format_text(
        "# A title\n\n*italic* and **bold**\n\n> A letter.\nAnother line.\n\n"
        "- one\n- two\n\n```html\n<script>visible code</script>\n```\n\n"
        '<img src=x onerror="alert(1)">\n\n'
        "[bad](javascript:alert(1)) ![image](https://example.com/tracker.png)"
    )
    assert "<h1>" in html and "<ul>" in html and "<br" in html and "<pre>" in html
    assert "<script" not in html and "<img" not in html and 'href="javascript:' not in html
    assert "&lt;img" in html


def test_progress_reads_stay_responsive_and_stale_tabs_cannot_duplicate_turns(chat_service):
    service = chat_service("Published response.", transport="api", pause_at=1)
    name = create(service)
    payload = {"id": name, "text": "I open the door.", "expected_version": version(service, name)}
    with ThreadPoolExecutor() as pool:
        future = pool.submit(service.client.post, "/api/turn", json=payload)
        try:
            assert service.model.started.wait(5)
            progress = service.client.get("/api/session", params={"id": name}, timeout=1)
            assert progress.status_code == 200 and progress.json()["busy"]
            assert progress.json()["turns"] == [] and "PRIVATE_DRAFT" not in progress.text
            assert service.client.post("/api/turn", json=payload).status_code == 409
            assert (
                service.client.post(
                    "/api/rename", json={**payload, "player_name": "Jules"}
                ).status_code
                == 409
            )
        finally:
            service.model.release.set()
        assert future.result().status_code == 200
    assert service.client.post("/api/turn", json=payload).status_code == 409
    assert len(service.model.requests) == 1
    assert service.model.requests[0]["input"][-1]["content"] == payload["text"]
    assert "expected_version" not in json.dumps(service.model.requests)
    assert payload["expected_version"] not in json.dumps(service.model.requests)
    assert service.app.token not in json.dumps(service.model.requests)


def test_browser_resume_retries_failed_response(chat_service):
    service = chat_service(TimeoutError(), "Published after retry.", transport="api")
    name = create(service)
    result = service.client.post(
        "/api/turn",
        json={"id": name, "text": "I wait.", "expected_version": version(service, name)},
    )
    assert result.status_code == 409
    paused = service.app.view(name)
    assert paused["pending"]["failed"] and not paused["busy"] and paused["turns"] == []
    resumed = service.client.post("/api/resume", json={"id": name})
    assert (
        resumed.status_code == 200
        and resumed.json()["turns"][0]["output"] == "Published after retry."
    )
    assert (
        len(service.model.requests) == 2 and service.model.requests[0] == service.model.requests[1]
    )


def test_proxy_updates_from_cli_are_visible_in_chat(chat_service):
    service = chat_service()
    name = create(service)
    session = service.app.session_path(name)
    author = core.submit(session, "Start")
    assert (
        service.client.get("/api/session", params={"id": name}).json()["pending"]["request_id"]
        == author["request_id"]
    )
    core.accept(session, author["request_id"], "Final from the command line.")
    view = service.client.get("/api/session", params={"id": name}).json()
    assert view["pending"] is None and view["turns"][0]["output"] == "Final from the command line."


def test_chat_name_selection_and_renaming_use_shared_identity_and_guard_stale_tabs(chat_service):
    service = chat_service("published", transport="api")
    client = service.client
    view = client.post(
        "/api/sessions", json={"story": "harbor", "player_name": " Éloi   Vale "}
    ).json()
    assert view["player"] == "Éloi Vale"
    name = view["id"]
    session = service.app.session_path(name)
    payload = {"id": name, "player_name": "Renée", "expected_version": view["version"]}
    assert client.post("/api/rename", json=payload, headers={"X-Chat-Token": ""}).status_code == 403
    renamed = client.post("/api/rename", json=payload)
    assert renamed.status_code == 200 and renamed.json()["player"] == "Renée"
    assert renamed.json()["turns"] == [] and not service.model.requests
    assert client.get("/api/sessions").json()["sessions"][0]["player"] == "Renée"
    assert core.player_identity(session, core.load_session(session)[1])["name"] == "Renée"
    assert client.get("/api/bootstrap").json()["stories"][0]["player"] == "Casey"
    assert client.post("/api/rename", json={**payload, "player_name": "Stale"}).status_code == 409
    assert client.post("/api/turn", json={**payload, "text": "Stale turn."}).status_code == 409
    assert (
        client.post("/api/rename", json={"id": name, "player_name": "No version"}).status_code
        == 409
    )
    assert not service.model.requests
    response = client.post(
        "/api/turn",
        json={"id": name, "text": "Start", "expected_version": renamed.json()["version"]},
    )
    assert response.status_code == 200
    for request in service.model.requests:
        assert "Renée" in request["input"][0]["content"]
        assert "Éloi Vale" in request["input"][0]["content"]
        assert "Renée" not in request["instructions"]
    core.rename_player(session, "Jules")
    assert client.get("/api/session", params={"id": name}).json()["player"] == "Jules"


def test_chat_rejects_invalid_name_before_creation_and_rename_while_pending(chat_service):
    service = chat_service()
    client = service.client
    for value in (" ", "x" * 81, {"name": "Jules"}):
        response = client.post("/api/sessions", json={"story": "harbor", "player_name": value})
        assert response.status_code == 409
    assert not list(service.app.sessions.iterdir())
    name = create(service)
    session = service.app.session_path(name)
    core.submit(session, "Start")
    before = (session / "state.json").read_bytes()
    result = client.post(
        "/api/rename",
        json={"id": name, "player_name": "Jules", "expected_version": version(service, name)},
    )
    assert result.status_code == 409 and "pending response" in result.json()["error"]
    assert (session / "state.json").read_bytes() == before


def test_automatic_proxy_resumes_existing_handoff_and_retries_failed_attempt(chat_service):
    service = chat_service(
        {"status": "failed", "exit_code": 7, "raw": "unfinished response"},
        "Published passage.",
        auto_proxy=True,
    )
    name = create(service)
    session = service.app.session_path(name)
    prepared = core.submit(session, "Begin the story.")
    request = core.read_json(core.attempt_dir(session, prepared["request_id"]) / "request.json")
    assert service.client.post("/api/resume", json={"id": name}).status_code == 409
    assert service.model.requests[0] == request
    view = service.app.view(name)
    assert view["pending"]["failed"] and not view["pending"]["handoff_ready"]
    assert "unfinished response" not in json.dumps(view)
    result = service.client.post("/api/resume", json={"id": name})
    assert result.status_code == 200 and result.json()["turn_count"] == 1
    assert result.json()["turns"][0]["output"] == "Published passage."
    assert (
        len(service.model.requests) == 2 and service.model.requests[0] == service.model.requests[1]
    )
    summary = core.export(session)
    assert summary["api_calls_started"] == 0 and summary["proxy_calls_started"] == 2
    core.resume(session, service.model)
    assert len(service.model.requests) == 2


def test_chat_detects_a_coding_agent_started_from_the_cli_without_blocking_reads(chat_service):
    service = chat_service("final", auto_proxy=True, pause_at=1)
    name = create(service)
    session = service.app.session_path(name)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(core.submit, session, "Start", service.model)
        try:
            assert service.model.started.wait(5)
            response = service.client.get("/api/session", params={"id": name}, timeout=1)
            assert response.status_code == 200
            view = response.json()
            assert view["busy"] and view["pending"]["kind"] == "turn"
            assert not view["pending"]["handoff_ready"]
        finally:
            service.model.release.set()
        assert future.result()["status"] == "published"
    assert not service.app.view(name)["busy"]


@pytest.mark.parametrize("transport", ["api", "proxy"])
def test_comparison_reads_exact_previous_versions_without_changing_history(chat_service, transport):
    original = "OLD_SCENE **verse**\r\nAnother line.\n\n<script>alert(1)</script>"
    final = "A **published** passage."
    service = chat_service(
        original, final, "following", transport=transport, auto_proxy=transport == "proxy"
    )
    name = create(service)
    for action, text in (("turn", "Begin."), ("regenerate", "PRIVATE_FEEDBACK")):
        response = service.client.post(
            "/api/" + action + "?compare=1",
            json={"id": name, "text": text, "expected_version": version(service, name)},
        )
        assert response.status_code == 200
    view = response.json()
    assert view["turn_count"] == 1 and view["turns"][0]["previous"] == original
    session = service.app.session_path(name)
    _, state = core.load_session(session)
    for index, request_id in enumerate(state["turns"][0]["responses"]):
        path = core.attempt_dir(session, request_id) / "response.json"
        raw = core.read_json(path)
        raw["internal_metadata"] = "PRIVATE_METADATA"
        summary = f"EXPOSED_SUMMARY_{index} **formatted**\n<script>alert(2)</script>"
        if transport == "api":
            raw["raw"]["output"].append(
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": summary}],
                    "content": [{"type": "reasoning_text", "text": "PRIVATE_REASONING"}],
                    "encrypted_content": "OPAQUE_CONTENT",
                }
            )
        else:
            raw["reasoning_summaries"] = [summary]
            raw["reasoning"] = "PRIVATE_REASONING"
        core.write_json(path, raw)
    before = {p: p.read_bytes() for p in session.rglob("*") if p.is_file()}
    compared = service.client.get("/api/session", params={"id": name, "compare": "1"})
    html = compared.json()["turns"][0]["previous_html"]
    assert "<strong>verse</strong>" in html and "<br" in html and "<script>" not in html
    turn = compared.json()["turns"][0]
    assert "EXPOSED_SUMMARY_0" in turn["previous_summary_html"]
    assert "EXPOSED_SUMMARY_1" not in turn["previous_summary_html"]
    assert "EXPOSED_SUMMARY_1" in turn["summary_html"]
    assert "EXPOSED_SUMMARY_0" not in turn["summary_html"]
    for key in ("summary_html", "previous_summary_html"):
        assert "<strong>formatted</strong>" in turn[key] and "<script>" not in turn[key]
    for hidden in (
        "SECRET_CANON",
        "PRIVATE_METADATA",
        "PRIVATE_REASONING",
        "PRIVATE_FEEDBACK",
        "OPAQUE_CONTENT",
    ):
        assert hidden not in compared.text
    assert {p: p.read_bytes() for p in session.rglob("*") if p.is_file()} == before
    normal = service.client.get("/api/session", params={"id": name})
    assert "OLD_SCENE" not in normal.text and "previous" not in normal.json()["turns"][0]
    assert "EXPOSED_SUMMARY" not in normal.text and "summary_html" not in normal.text
    exported = service.client.get("/api/transcript", params={"id": name})
    assert (
        final in exported.text
        and "OLD_SCENE" not in exported.text
        and "PRIVATE_FEEDBACK" not in exported.text
        and "EXPOSED_SUMMARY" not in exported.text
    )
    following = service.client.post(
        "/api/turn", json={"id": name, "text": "Continue.", "expected_version": view["version"]}
    )
    assert following.status_code == 200 and len(service.model.requests) == 3
    assert service.model.requests[2]["input"][2] == {"role": "assistant", "content": final}
    for hidden in (
        "PRIVATE_FEEDBACK",
        "OLD_SCENE",
        "compare",
        "EXPOSED_SUMMARY",
        "PRIVATE_REASONING",
    ):
        assert hidden not in json.dumps(service.model.requests[2])


def test_inspection_without_summaries_does_not_invent_them(chat_service):
    service = chat_service("A single passage.", auto_proxy=True)
    name = create(service)
    core.submit(service.app.session_path(name), "Begin.", service.model)
    turn = service.app.view(name, compare=True)["turns"][0]
    assert turn["summary_html"] is None
    assert turn["previous_summary_html"] is None and turn["previous"] is None


def test_regeneration_progress_keeps_existing_passage_and_rejects_concurrent_or_stale_actions(
    chat_service,
):
    service = chat_service("original", "replacement", auto_proxy=True, pause_at=2)
    name = create(service)
    core.submit(service.app.session_path(name), "Begin.", service.model)
    payload = {
        "id": name,
        "text": "Change it.",
        "expected_version": version(service, name),
        "submission_id": "a" * 32,
    }
    assert (
        service.client.post(
            "/api/regenerate", json={"id": name, "text": "Missing version"}
        ).status_code
        == 409
    )
    with ThreadPoolExecutor() as pool:
        future = pool.submit(service.client.post, "/api/regenerate", json=payload)
        try:
            assert service.model.started.wait(5)
            view = service.client.get(
                "/api/session", params={"id": name, "compare": "1"}, timeout=1
            ).json()
            assert (
                view["busy"]
                and view["turn_count"] == 1
                and view["turns"][0]["output"] == "original"
            )
            assert (
                view["pending"]["kind"] == "regenerate"
                and view["pending"]["submission_id"] == "a" * 32
            )
            for action in ("turn", "regenerate"):
                assert service.client.post("/api/" + action, json=payload).status_code == 409
        finally:
            service.model.release.set()
        view = future.result().json()
    assert view["turn_count"] == 1 and view["turns"][0]["input"] == "Begin."
    assert view["turns"][0]["submission_id"] == "a" * 32 and view["pending"] is None
    assert service.client.post("/api/regenerate", json=payload).status_code == 409
    assert len(service.model.requests) == 2


def test_failed_regeneration_keeps_previous_passage_until_successful_resume(chat_service):
    service = chat_service(
        "original", {"status": "failed", "raw": "FAILED_OUTPUT"}, "replacement", auto_proxy=True
    )
    name = create(service)
    session = service.app.session_path(name)
    core.submit(session, "Begin.", service.model)
    result = service.client.post(
        "/api/regenerate?compare=1",
        json={"id": name, "text": "Fix it.", "expected_version": version(service, name)},
    )
    assert result.status_code == 409
    paused = service.client.get("/api/session", params={"id": name, "compare": "1"})
    assert paused.json()["pending"]["failed"]
    assert paused.json()["turns"][0]["output"] == "original"
    assert "FAILED_OUTPUT" not in paused.text
    resumed = service.client.post("/api/resume?compare=1", json={"id": name})
    assert resumed.status_code == 200
    assert resumed.json()["turns"][0]["previous"] == "original"
    assert resumed.json()["turns"][0]["output"] == "replacement"
    assert service.model.requests[1] == service.model.requests[2]


def test_missing_previous_version_is_explicit_without_breaking_current_reading(chat_service):
    service = chat_service("original", "replacement", auto_proxy=True)
    name = create(service)
    session = service.app.session_path(name)
    turn = core.submit(session, "Begin.", service.model)
    core.regenerate(session, "Fix it.", service.model)
    (core.attempt_dir(session, turn["responses"][0]) / "response.json").unlink()
    assert service.client.get("/api/session", params={"id": name}).status_code == 200
    result = service.client.get("/api/session", params={"id": name, "compare": "1"})
    assert result.status_code == 409 and str(session) not in result.text
