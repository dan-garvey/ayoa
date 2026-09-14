import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from narrative import core
from narrative.chat import format_text


def create(service):
    response = service.client.post("/api/sessions", json={"story": "harbor"})
    assert response.status_code == 201
    return response.json()["id"]


def test_proxy_chat_preserves_sources_but_only_displays_published_story(chat_service):
    service = chat_service()
    client = service.client
    name = create(service)
    turn = "I wait.\n\n*Quietly.*"
    sent = client.post("/api/turn", json={"id": name, "text": turn, "expected_turns": 0})
    assert sent.status_code == 200
    assert sent.json()["turns"] == []
    assert sent.json()["pending"]["input"] == turn
    assert "<em>Quietly.</em>" in sent.json()["pending"]["input_html"]
    packet = client.get("/api/handoff", params={"id": name}).json()
    assert "SECRET_CANON" in packet["text"]  # Explicit operator-only disclosure.
    assert packet["stage"] == "draft"
    draft = "PRIVATE_DRAFT with an unfinished idea."
    accepted = client.post(
        "/api/accept", json={"id": name, "request_id": packet["request_id"], "text": draft}
    )
    assert accepted.status_code == 200 and accepted.json()["turns"] == []
    next_packet = client.get("/api/handoff", params={"id": name}).json()
    assert next_packet["stage"] == "revision" and draft in next_packet["text"]
    final = "The **bell** rings.\nA second line.\n\n> An old promise.\n\n<script>alert(1)</script>"
    published = client.post(
        "/api/accept", json={"id": name, "request_id": next_packet["request_id"], "text": final}
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
        assert response.status_code == 200
        assert "PRIVATE_DRAFT" not in response.text and "SECRET_CANON" not in response.text
    saved = core.load_session(service.app.session_path(name))[1]
    attempt = service.app.session_path(name) / "attempts" / saved["turns"][0]["author"]
    assert core.read_json(attempt / "response.json")["raw"] == draft
    transcript = client.get("/api/transcript", params={"id": name})
    assert final in transcript.text and draft not in transcript.text
    assert "attachment" in transcript.headers["Content-Disposition"]
    duplicate = client.post(
        "/api/accept",
        json={"id": name, "request_id": next_packet["request_id"], "text": "replacement"},
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
    service = chat_service("PRIVATE_DRAFT", "Published response.", transport="api", pause_at=2)
    name = create(service)
    payload = {"id": name, "text": "I open the door.", "expected_turns": 0}
    with ThreadPoolExecutor() as pool:
        future = pool.submit(service.client.post, "/api/turn", json=payload)
        try:
            assert service.model.started.wait(5)
            progress = service.client.get("/api/session", params={"id": name}, timeout=1)
            assert progress.status_code == 200 and progress.json()["busy"]
            assert progress.json()["turns"] == [] and "PRIVATE_DRAFT" not in progress.text
            assert service.client.post("/api/turn", json=payload).status_code == 409
        finally:
            service.model.release.set()
        assert future.result().status_code == 200
    assert service.client.post("/api/turn", json=payload).status_code == 409
    assert len(service.model.requests) == 2
    assert service.model.requests[0]["input"][-1]["content"] == payload["text"]
    assert "expected_turns" not in json.dumps(service.model.requests)
    assert service.app.token not in json.dumps(service.model.requests)


def test_browser_resume_reuses_successful_draft_after_editor_failure(chat_service):
    service = chat_service("saved draft", TimeoutError(), "Published after retry.", transport="api")
    name = create(service)
    result = service.client.post(
        "/api/turn", json={"id": name, "text": "I wait.", "expected_turns": 0}
    )
    assert result.status_code == 409
    paused = service.client.get("/api/session", params={"id": name}).json()
    assert paused["pending"]["failed"] and not paused["busy"] and paused["turns"] == []
    resumed = service.client.post("/api/resume", json={"id": name})
    assert (
        resumed.status_code == 200
        and resumed.json()["turns"][0]["output"] == "Published after retry."
    )
    assert len(service.model.requests) == 3
    assert service.model.requests[1] == service.model.requests[2]


def test_proxy_updates_from_cli_are_visible_in_chat(chat_service):
    service = chat_service()
    name = create(service)
    session = service.app.session_path(name)
    author = core.submit(session, "Start")
    assert (
        service.client.get("/api/session", params={"id": name}).json()["pending"]["request_id"]
        == author["request_id"]
    )
    editor = core.accept(session, author["request_id"], "draft")
    core.accept(session, editor["request_id"], "Final from the command line.")
    view = service.client.get("/api/session", params={"id": name}).json()
    assert view["pending"] is None and view["turns"][0]["output"] == "Final from the command line."
