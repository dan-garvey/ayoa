from pathlib import Path

import pytest
from playwright.sync_api import expect

from narrative import core


def begin(page):
    page.get_by_role("button", name="New story", exact=True).click()
    page.get_by_role("dialog").get_by_role("button", name="Begin story", exact=True).click()


def publish_proxy(page, draft, final):
    page.get_by_role("button", name="Response handoff", exact=True).click()
    page.get_by_label("Complete response", exact=True).fill(draft)
    page.get_by_role("button", name="Accept response", exact=True).click()
    expect(page.locator("#handoff-stage")).to_have_text("FINAL RESPONSE HANDOFF")
    assert draft in page.get_by_label("Complete model request").input_value()
    page.get_by_label("Complete response", exact=True).fill(final)
    page.get_by_role("button", name="Accept response", exact=True).click()
    expect(page.locator("#handoff-dialog")).not_to_be_visible()


def test_browser_proxy_turns_markdown_drafts_copy_download_and_reload(browser, chat_service):
    service = chat_service()
    context = browser.new_context(permissions=["clipboard-read", "clipboard-write"])
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        page.goto(service.url)
        begin(page)
        expect(page.get_by_role("button", name="Response handoff", exact=True)).to_be_visible()
        assert "SECRET_CANON" not in page.locator("main").inner_text()
        final = "The **bell** rings.\nOne clear note.\n\n> A folded letter.\n\n*Still sealed.*\n\n<script>window.leaked = true</script>"
        publish_proxy(page, "DISCARDED_DRAFT", final)
        expect(page.locator(".message.story")).to_have_count(1)
        expect(page.locator(".message.story strong")).to_have_text("bell")
        expect(page.locator(".message.story blockquote")).to_have_text("A folded letter.")
        assert page.locator(".message.story br").count() == 1
        assert page.evaluate("window.leaked === undefined")
        assert "DISCARDED_DRAFT" not in page.locator("main").inner_text()
        page.get_by_role("button", name="Copy passage 1").click()
        expect(page.locator("#toast")).to_have_text("Copied to clipboard")
        assert page.evaluate("navigator.clipboard.readText()") == final
        with page.expect_download() as download:
            page.get_by_role("button", name="Download story", exact=True).click()
        exported = Path(download.value.path()).read_text()
        assert final in exported and "DISCARDED_DRAFT" not in exported

        composer = page.get_by_role("textbox", name="Your next turn")
        composer.fill("I wait.")
        composer.press("Enter")
        assert composer.input_value() == "I wait.\n"
        composer.fill("I wait.\n\n*And listen.*")
        page.reload()
        expect(composer).to_have_value("I wait.\n\n*And listen.*")
        expect(composer).to_be_enabled()
        composer.press("Control+Enter")
        expect(composer).to_be_disabled()
        expect(page.get_by_role("button", name="Response handoff", exact=True)).to_be_visible()
        publish_proxy(page, "SECOND_DRAFT", "The tide reaches the last step.")
        expect(page.locator(".message.story")).to_have_count(2)
        expect(composer).to_have_value("")
        page.reload()
        expect(page.locator(".message.story")).to_have_count(2)
        expect(composer).to_be_enabled()
        assert "SECOND_DRAFT" not in page.locator("main").inner_text()
        assert not errors
    finally:
        context.close()


@pytest.mark.parametrize("transport", ["api", "proxy"])
def test_browser_reload_during_generation_preserves_turn_and_hides_draft(
    browser, chat_service, transport
):
    service = chat_service(
        "HIDDEN_DRAFT",
        "The door opens.",
        transport=transport,
        pause_at=2,
        auto_proxy=transport == "proxy",
    )
    page = browser.new_page()
    try:
        page.goto(service.url)
        begin(page)
        assert service.model.started.wait(5)
        expect(page.get_by_role("textbox", name="Your next turn")).to_be_disabled()
        page.reload()
        expect(page.locator("#status-chip")).to_have_text("Writing…")
        expect(page.locator("#response-status-text")).to_have_text("Refining the passage…")
        expect(page.get_by_role("button", name="Response handoff", exact=True)).not_to_be_visible()
        assert "HIDDEN_DRAFT" not in page.locator("main").inner_text()
        assert len(service.model.requests) == 2
        service.model.release.set()
        expect(page.locator(".message.story")).to_have_count(1)
        expect(page.get_by_role("textbox", name="Your next turn")).to_be_enabled()
        expect(page.get_by_role("textbox", name="Your next turn")).to_have_value("")
        assert len(service.model.requests) == 2
    finally:
        service.model.release.set()
        page.close()


@pytest.mark.parametrize("transport", ["api", "proxy"])
def test_browser_failure_can_resume_without_resubmitting_player_turn(
    browser, chat_service, transport
):
    service = chat_service(
        "saved draft",
        TimeoutError(),
        "At last, an answer.",
        transport=transport,
        auto_proxy=transport == "proxy",
    )
    page = browser.new_page()
    try:
        page.goto(service.url)
        begin(page)
        expect(page.get_by_role("button", name="Continue response", exact=True)).to_be_visible()
        page.reload()
        expect(page.get_by_role("button", name="Continue response", exact=True)).to_be_visible()
        page.get_by_role("button", name="Continue response", exact=True).click()
        expect(page.locator(".message.story")).to_have_count(1)
        expect(page.locator(".message.player")).to_have_count(1)
        expect(page.get_by_role("textbox", name="Your next turn")).to_be_enabled()
        assert len(service.model.requests) == 3
        assert service.model.requests[1] == service.model.requests[2]
    finally:
        page.close()


def test_mobile_navigation_reading_preferences_and_unsent_text(browser, chat_service):
    service = chat_service()
    page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    try:
        page.goto(service.url)
        expect(page.get_by_role("button", name="Open story list")).to_be_visible()
        assert page.locator("#sidebar").evaluate("el => el.inert")
        page.get_by_role("button", name="Open story list").click()
        expect(page.locator("#menu")).to_have_attribute("aria-expanded", "true")
        begin(page)
        expect(page.locator("#menu")).to_have_attribute("aria-expanded", "false")
        publish_proxy(page, "draft", "The light moves across the **water**.\n\nA quiet morning.")
        page.get_by_label("Reading preferences").click()
        page.get_by_role("button", name="Larger story text").click()
        page.get_by_role("button", name="Use dark appearance").click()
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        page.get_by_label("Reading preferences").click()
        composer = page.get_by_role("textbox", name="Your next turn")
        composer.fill("A turn I have not sent yet.")
        page.reload()
        expect(composer).to_have_value("A turn I have not sent yet.")
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        assert (
            page.locator("html").evaluate("el => el.style.getPropertyValue('--story-size')")
            == "22px"
        )
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        box = composer.bounding_box()
        assert box and box["y"] + box["height"] <= 844
        page.get_by_role("button", name="Open story list").click()
        page.get_by_role("button", name="Close story list").click()
        expect(composer).to_have_value("A turn I have not sent yet.")
    finally:
        page.close()


def test_long_passages_open_at_start_and_do_not_interrupt_earlier_reading(browser, chat_service):
    long_passage = "The opening of the passage.\n\n" + "\n\n".join(
        f"A long paragraph numbered {number}. " * 12 for number in range(30)
    )
    service = chat_service("draft", long_passage, "next draft", long_passage, transport="api")
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    try:
        page.goto(service.url)
        begin(page)
        expect(page.locator(".message.story")).to_have_count(1)
        top = page.locator("#reader").bounding_box()["y"]
        assert abs(page.locator(".message.story").bounding_box()["y"] - top - 28) < 3
        assert (
            page.locator("#reader").evaluate(
                "el => el.scrollHeight - el.scrollTop - el.clientHeight"
            )
            > 1000
        )
        page.locator("#reader").evaluate("el => { el.scrollTop = 0; }")
        name = service.app.session_list()[0]["id"]
        core.submit(service.app.session_path(name), "A turn from the terminal.", service.model)
        expect(page.locator(".message.story")).to_have_count(2)
        assert page.locator("#reader").evaluate("el => el.scrollTop") == 0
        page.get_by_role("button", name="Latest passage").click()
        assert abs(page.locator(".message.story").last.bounding_box()["y"] - top - 28) < 3
    finally:
        page.close()


def test_browser_choose_and_rename_protagonist_preserves_prose_and_unsent_turn(
    browser, chat_service
):
    service = chat_service()
    page = browser.new_page()
    try:
        page.goto(service.url)
        page.get_by_role("button", name="New story", exact=True).click()
        expect(page.get_by_label("Protagonist name", exact=True).first).to_have_value("Casey")
        page.locator("#new-player").fill("Éloi Vale")
        page.get_by_role("dialog").get_by_role("button", name="Begin story", exact=True).click()
        expect(page.locator("#story-kicker")).to_have_text("PLAYING AS ÉLOI VALE")
        expect(page.get_by_role("button", name="Rename protagonist", exact=True)).to_be_disabled()
        expect(page.get_by_role("button", name="Response handoff", exact=True)).to_be_visible()
        name = service.app.session_list()[0]["id"]
        packet = service.app.handoff(name)
        assert "My character: Éloi Vale." in packet["text"]
        publish_proxy(page, "draft", "Éloi Vale waits by the **water**.")
        composer = page.get_by_role("textbox", name="Your next turn")
        composer.fill("I listen.\nAnd wait.")
        page.get_by_role("button", name="Rename protagonist", exact=True).click()
        expect(page.locator("#rename-name")).to_have_value("Éloi Vale")
        page.locator("#rename-name").fill("Renée O'Vale")
        page.get_by_role("button", name="Save name", exact=True).click()
        expect(page.locator("#rename-dialog")).not_to_be_visible()
        expect(page.locator("#story-kicker")).to_have_text("PLAYING AS RENÉE O'VALE")
        expect(page.locator(".session-meta")).to_contain_text("Renée O'Vale")
        expect(composer).to_have_value("I listen.\nAnd wait.")
        page.reload()
        expect(page.locator("#story-kicker")).to_have_text("PLAYING AS RENÉE O'VALE")
        expect(composer).to_have_value("I listen.\nAnd wait.")
        expect(page.locator(".message.story .prose")).to_have_text("Éloi Vale waits by the water.")
        composer.press("Control+Enter")
        expect(page.get_by_role("button", name="Response handoff", exact=True)).to_be_visible()
        packet = service.app.handoff(name)
        assert "My character: Renée O'Vale." in packet["text"]
        assert '"Éloi Vale"' in packet["text"]
        publish_proxy(page, "next draft", "A reply.")
        expect(page.locator(".message.story")).to_have_count(2)
    finally:
        page.close()


def test_browser_stale_rename_does_not_overwrite_a_new_name(browser, chat_service):
    service = chat_service()
    view = service.app.create("harbor")
    page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True)
    try:
        page.goto(f"{service.url}/#{view['id']}")
        page.get_by_role("button", name="Rename protagonist", exact=True).click()
        page.locator("#rename-name").fill("Avery")
        core.rename_player(service.app.session_path(view["id"]), "Morgan")
        expect(page.locator("#story-kicker")).to_have_text("PLAYING AS MORGAN")
        page.get_by_role("button", name="Save name", exact=True).click()
        expect(page.locator("#rename-error")).to_contain_text("story changed")
        assert service.app.view(view["id"])["player"] == "Morgan"
        expect(page.locator("#rename-name")).to_have_value("Avery")
        page.get_by_role("button", name="Close rename protagonist", exact=True).click()
        page.get_by_role("button", name="Rename protagonist", exact=True).click()
        expect(page.locator("#rename-name")).to_have_value("Morgan")
        long_name = "<b>" + "É" * 73 + "</b>"
        page.locator("#rename-name").fill(long_name)
        page.get_by_role("button", name="Save name", exact=True).click()
        expect(page.locator("#story-kicker")).to_have_text(f"PLAYING AS {long_name.upper()}")
        assert page.locator("#story-kicker b").count() == 0
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.get_by_role("button", name="Open story list").click()
        expect(page.locator(".session-meta")).to_contain_text(long_name)
        assert page.locator(".session-list").evaluate("el => el.scrollWidth <= el.clientWidth")
    finally:
        page.close()
