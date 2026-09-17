from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import pytest
from playwright.sync_api import expect

from narrative import core


def begin(page):
    page.get_by_role("button", name="New story", exact=True).click()
    page.get_by_role("dialog").get_by_role("button", name="Begin story", exact=True).click()


def publish_proxy(page, final):
    page.get_by_role("button", name="Response handoff", exact=True).click()
    page.get_by_label("Complete response", exact=True).fill(final)
    page.get_by_role("button", name="Accept response", exact=True).click()
    expect(page.locator("#handoff-dialog")).not_to_be_visible()


@pytest.mark.parametrize("width", [1280, 390])
def test_browser_checkups_are_inspection_only_and_survive_reload(browser, chat_service, width):
    service = chat_service(
        {
            "raw": "PRIVATE_PLAN **emphasis**\n<script>window.planLeak=true</script>",
            "reasoning_summaries": ["CHECKUP_SUMMARY"],
        },
        "Visible first passage.",
        "PRIVATE_NEXT_PLAN",
        "Visible second passage.",
        checkup_every=1,
        auto_proxy=True,
        pause_at=2,
    )
    page = browser.new_page(viewport={"width": width, "height": 900})
    try:
        page.goto(service.url)
        if width <= 700:
            page.get_by_role("button", name="Open story list", exact=True).click()
        begin(page)
        assert service.model.started.wait(5)
        page.reload()
        expect(page.get_by_role("textbox", name="Your next turn")).to_be_disabled()
        expect(page.locator(".backstage-checkup")).to_have_count(0)
        assert "PRIVATE_PLAN" not in page.locator("main").inner_text()
        toggle = page.get_by_role("button", name="Inspect responses", exact=True)
        toggle.click()
        checkup = page.locator(".backstage-checkup")
        expect(checkup).to_have_count(1)
        checkup.locator(":scope > summary").click()
        expect(checkup.locator(".checkup-prose strong")).to_have_text("emphasis")
        assert page.evaluate("window.planLeak === undefined")
        assert checkup.locator("script").count() == 0
        checkup.get_by_text("Checkup reasoning summary", exact=True).click()
        expect(checkup.locator(".reasoning-prose")).to_have_text("CHECKUP_SUMMARY")
        service.model.release.set()
        expect(page.locator(".message.story > .prose")).to_have_text("Visible first passage.")
        page.reload()
        expect(checkup).to_have_count(1)
        toggle.click()
        expect(checkup).to_have_count(0)
        assert "PRIVATE_PLAN" not in page.locator("main").inner_text()
        composer = page.get_by_role("textbox", name="Your next turn")
        composer.fill("Continue.")
        composer.press("Control+Enter")
        expect(page.locator(".message.story")).to_have_count(2)
        with page.expect_download() as download:
            page.get_by_role("button", name="Download story", exact=True).click()
        text = Path(download.value.path()).read_text()
        assert "PRIVATE" not in text and "SUMMARY" not in text
        toggle.click()
        expect(checkup).to_have_count(2)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert len(service.model.requests) == 4
    finally:
        service.model.release.set()
        page.close()


def test_browser_manual_checkup_then_author_handoffs(browser, chat_service):
    service = chat_service(checkup_every=1)
    page = browser.new_page()
    try:
        page.goto(service.url)
        begin(page)
        page.get_by_role("button", name="Response handoff", exact=True).click()
        expect(page.locator("#handoff-stage")).to_have_text("BACKSTAGE CHECKUP HANDOFF")
        expect(page.locator("#handoff-note")).to_contain_text("stay out of the story")
        page.get_by_label("Complete response", exact=True).fill("PRIVATE_MANUAL_PLAN")
        page.get_by_role("button", name="Accept response", exact=True).click()
        expect(page.locator("#handoff-dialog")).not_to_be_visible()
        expect(page.locator(".message.story")).to_have_count(0)
        publish_proxy(page, "Actual story.")
        expect(page.locator(".message.story > .prose")).to_have_text("Actual story.")
        assert "PRIVATE_MANUAL_PLAN" not in page.locator("main").inner_text()
    finally:
        page.close()


def test_browser_proxy_turns_markdown_copy_download_and_reload(browser, chat_service):
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
        publish_proxy(page, final)
        expect(page.locator(".message.story")).to_have_count(1)
        expect(page.locator(".message.story strong")).to_have_text("bell")
        expect(page.locator(".message.story blockquote")).to_have_text("A folded letter.")
        assert page.locator(".message.story br").count() == 1
        assert page.evaluate("window.leaked === undefined")
        page.get_by_role("button", name="Copy passage 1").click()
        expect(page.locator("#toast")).to_have_text("Copied to clipboard")
        assert page.evaluate("navigator.clipboard.readText()") == final
        with page.expect_download() as download:
            page.get_by_role("button", name="Download story", exact=True).click()
        exported = Path(download.value.path()).read_text()
        assert final in exported

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
        publish_proxy(page, "The tide reaches the last step.")
        expect(page.locator(".message.story")).to_have_count(2)
        expect(composer).to_have_value("")
        page.reload()
        expect(page.locator(".message.story")).to_have_count(2)
        expect(composer).to_be_enabled()
        assert not errors
    finally:
        context.close()


def test_mobile_lan_chat_can_submit_and_reload_without_login_or_https(browser, chat_service):
    service = chat_service(
        "The morning is quiet.", "The door opens.", auto_proxy=True, lan_address="192.168.86.25"
    )
    origin = f"http://192.168.86.25:{service.server.server_port}"
    context = browser.new_context(
        viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True
    )

    def local_network(route):
        headers = {**route.request.all_headers(), "Host": urlsplit(origin).netloc}
        response = route.fetch(
            url=route.request.url.replace(origin, service.url, 1), headers=headers
        )
        route.fulfill(response=response)

    context.route(origin + "/**", local_network)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        assert page.goto(origin).status == 200
        assert page.evaluate("window.isSecureContext") is False
        assert page.evaluate("typeof crypto.randomUUID") == "undefined"
        page.get_by_role("button", name="Open story list").click()
        begin(page)
        expect(page.locator(".message.story")).to_have_count(1)
        composer = page.get_by_role("textbox", name="Your next turn")
        composer.fill("I open the door.")
        page.locator("#send").click()
        expect(page.locator(".message.story")).to_have_count(2)
        page.reload()
        expect(page.locator(".message.story").last).to_contain_text("The door opens.")
        assert len(service.model.requests) == 2
        session = next(service.app.sessions.iterdir())
        turns = core.load_session(session)[1]["turns"]
        assert all(UUID(turn["submission_id"]).version == 4 for turn in turns)
        assert not errors
    finally:
        context.close()


@pytest.mark.parametrize("transport", ["api", "proxy"])
def test_browser_reload_during_generation_preserves_turn(browser, chat_service, transport):
    service = chat_service(
        "The door opens.",
        transport=transport,
        pause_at=1,
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
        expect(page.locator("#response-status-text")).to_have_text("Writing the next passage…")
        expect(page.get_by_role("button", name="Response handoff", exact=True)).not_to_be_visible()
        assert len(service.model.requests) == 1
        service.model.release.set()
        expect(page.locator(".message.story")).to_have_count(1)
        expect(page.get_by_role("textbox", name="Your next turn")).to_be_enabled()
        expect(page.get_by_role("textbox", name="Your next turn")).to_have_value("")
        assert len(service.model.requests) == 1
    finally:
        service.model.release.set()
        page.close()


@pytest.mark.parametrize("transport", ["api", "proxy"])
def test_browser_failure_can_resume_without_resubmitting_player_turn(
    browser, chat_service, transport
):
    service = chat_service(
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
        assert len(service.model.requests) == 2
        assert service.model.requests[0] == service.model.requests[1]
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
        publish_proxy(page, "The light moves across the **water**.\n\nA quiet morning.")
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
    service = chat_service(long_passage, long_passage, transport="api")
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
        publish_proxy(page, "Éloi Vale waits by the **water**.")
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
        publish_proxy(page, "A reply.")
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


@pytest.mark.parametrize("width", [1280, 390])
def test_browser_regeneration_comparison_formats_versions_and_preserves_turns(
    browser, chat_service, width
):
    original = "**Original** verse.\nA second line.\n\n<script>window.draftLeak = true</script>"
    revised = "**Replacement** verse.\nAnother line.\n\n> A quiet reply."
    service = chat_service(original, revised, "Next passage.", auto_proxy=True)
    context = browser.new_context(
        viewport={"width": width, "height": 900}, permissions=["clipboard-read", "clipboard-write"]
    )
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        page.goto(service.url + "/?compare=1")
        if width <= 700:
            page.get_by_role("button", name="Open story list", exact=True).click()
        begin(page)
        expect(page.locator(".message.story")).to_have_count(1)
        expect(page.locator(".comparison.story")).to_have_count(0)
        composer = page.get_by_role("textbox", name="Your next turn")
        composer.fill("REGENERATION_FEEDBACK")
        if width > 700:
            composer.press("Control+Shift+Enter")
        else:
            page.get_by_role("button", name="Regenerate latest response", exact=True).click()
        pair = page.locator(".comparison.story")
        expect(pair.locator(".replacement strong")).to_have_text("Replacement")
        expect(pair.locator(".previous strong")).to_have_text("Original")
        expect(page.locator(".message.player")).to_have_count(1)
        expect(page.locator(".message.regeneration")).to_have_count(0)
        expect(composer).to_have_value("")
        assert (
            pair.locator(".previous br").count() == 1
            and pair.locator(".replacement br").count() == 1
        )
        assert page.evaluate("window.draftLeak === undefined")
        assert "SECRET_CANON" not in page.locator("main").inner_text()
        left = pair.locator(".previous").bounding_box()
        right = pair.locator(".replacement").bounding_box()
        if width > 1000:
            assert left["x"] + left["width"] <= right["x"] and abs(left["y"] - right["y"]) < 1
        else:
            assert left["y"] + left["height"] <= right["y"]
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.get_by_role("button", name="Copy previous version 1", exact=True).click()
        expect(page.locator("#toast")).to_have_text("Copied to clipboard")
        assert page.evaluate("navigator.clipboard.readText()") == original
        page.get_by_role("button", name="Copy current version 1", exact=True).click()
        assert page.evaluate("navigator.clipboard.readText()") == revised
        with page.expect_download() as download:
            page.get_by_role("button", name="Download story", exact=True).click()
        exported = Path(download.value.path()).read_text()
        assert (
            revised in exported
            and original not in exported
            and "REGENERATION_FEEDBACK" not in exported
        )
        composer.fill("I listen.")
        page.reload()
        expect(pair).to_have_count(1)
        expect(composer).to_have_value("I listen.")
        toggle = page.get_by_role("button", name="Inspect responses", exact=True)
        toggle.click()
        expect(pair).to_have_count(0)
        assert "Original" not in page.locator("main").inner_text()
        toggle.click()
        expect(pair.locator(".previous strong")).to_have_text("Original")
        composer.press("Control+Enter")
        expect(page.locator(".message.story")).to_have_count(2)
        expect(page.locator(".message.story").last.locator(".prose")).to_have_text("Next passage.")
        assert len(service.model.requests) == 3
        assert original not in str(service.model.requests[2:])
        assert "REGENERATION_FEEDBACK" not in str(service.model.requests[2:])
        assert not errors
    finally:
        context.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_browser_inspects_each_summary_without_publishing_or_replaying_it(
    browser, chat_service, width
):
    summary = "FIRST_SUMMARY **with emphasis**.\n\n<script>window.summaryLeak = true</script>"
    service = chat_service(
        {"raw": "Original passage.", "reasoning_summaries": [summary]},
        {"raw": "Replacement passage.", "reasoning_summaries": ["SECOND_SUMMARY"]},
        "A passage without a summary.",
        auto_proxy=True,
    )
    page = browser.new_page(viewport={"width": width, "height": 900})
    try:
        page.goto(service.url)
        if width <= 700:
            page.get_by_role("button", name="Open story list", exact=True).click()
        begin(page)
        expect(page.locator(".message.story")).to_have_count(1)
        expect(page.locator(".reasoning-summary")).to_have_count(0)
        toggle = page.get_by_role("button", name="Inspect responses", exact=True)
        toggle.click()
        expect(page.locator(".reasoning-summary")).to_have_count(1)
        page.locator(".reasoning-summary > summary").click()
        expect(page.locator(".reasoning-prose strong")).to_have_text("with emphasis")
        assert page.evaluate("window.summaryLeak === undefined")
        assert page.locator(".reasoning-prose script").count() == 0
        assert len(service.model.requests) == 1
        composer = page.get_by_role("textbox", name="Your next turn")
        composer.fill("Rewrite it.")
        composer.press("Control+Shift+Enter")
        pair = page.locator(".comparison.story")
        expect(pair.locator(".replacement > .prose")).to_have_text("Replacement passage.")
        pair.locator(".previous .reasoning-summary > summary").click()
        pair.locator(".replacement .reasoning-summary > summary").click()
        expect(pair.locator(".previous .reasoning-prose")).to_contain_text("FIRST_SUMMARY")
        expect(pair.locator(".replacement .reasoning-prose")).to_have_text("SECOND_SUMMARY")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        with page.expect_download() as download:
            page.get_by_role("button", name="Download story", exact=True).click()
        exported = Path(download.value.path()).read_text()
        assert "Replacement passage." in exported and "SUMMARY" not in exported
        page.reload()
        expect(toggle).to_have_attribute("aria-pressed", "true")
        expect(pair.locator(".reasoning-summary")).to_have_count(2)
        toggle.click()
        expect(page.locator(".reasoning-summary")).to_have_count(0)
        assert "SUMMARY" not in page.locator("main").inner_text()
        composer.fill("I listen.")
        composer.press("Control+Enter")
        expect(page.locator(".message.story")).to_have_count(2)
        toggle.click()
        expect(page.locator(".message.story").last.locator(".summary-unavailable")).to_be_visible()
        assert len(service.model.requests) == 3
        for request in service.model.requests:
            assert "SUMMARY" not in core.render_request(request)
    finally:
        page.close()


@pytest.mark.parametrize("transport", ["api", "proxy"])
def test_browser_reload_during_regeneration_preserves_old_passage_and_clears_instructions(
    browser, chat_service, transport
):
    service = chat_service(
        "Saved original.\nA line of verse.",
        "Replacement.",
        transport=transport,
        auto_proxy=transport == "proxy",
        pause_at=2,
    )
    page = browser.new_page()
    try:
        page.goto(service.url)
        begin(page)
        expect(page.locator(".message.story")).to_have_count(1)
        composer = page.get_by_role("textbox", name="Your next turn")
        composer.fill("PRIVATE_REWRITE_INSTRUCTIONS")
        composer.press("Meta+Shift+Enter")
        assert service.model.started.wait(5)
        expect(page.locator(".message.story .prose")).to_contain_text("Saved original.")
        page.get_by_role("button", name="Inspect responses", exact=True).click()
        expect(page.locator(".pending-comparison .previous .prose")).to_contain_text(
            "Saved original."
        )
        expect(page.locator(".pending-comparison .replacement .prose")).to_have_text(
            "Regenerating the passage…"
        )
        page.reload()
        expect(page.locator(".pending-comparison .previous .prose")).to_contain_text(
            "Saved original."
        )
        expect(composer).to_be_disabled()
        expect(page.locator(".message.player")).to_have_count(1)
        service.model.release.set()
        expect(page.locator(".comparison .replacement .prose")).to_have_text("Replacement.")
        expect(page.locator(".pending-comparison")).to_have_count(0)
        expect(page.locator(".message.regeneration")).to_have_count(0)
        expect(composer).to_have_value("")
        expect(composer).to_be_enabled()
        assert len(service.model.requests) == 2
    finally:
        service.model.release.set()
        page.close()


def test_rejected_stale_regeneration_keeps_composer_feedback(browser, chat_service):
    service = chat_service("Original.", "Changed in another tab.", auto_proxy=True)
    page = browser.new_page()
    try:
        page.goto(service.url)
        begin(page)
        expect(page.locator(".message.story")).to_have_count(1)
        session = service.app.session_path(service.app.session_list()[0]["id"])

        def change_before_submission(route):
            core.regenerate(session, "Another tab's correction.", service.model)
            route.continue_()

        page.route("**/api/regenerate", change_before_submission, times=1)
        composer = page.get_by_role("textbox", name="Your next turn")
        composer.fill("MY_UNSENT_CORRECTION")
        composer.press("Control+Shift+Enter")
        expect(page.locator("#notice")).to_contain_text("story changed")
        expect(page.locator(".message.story .prose")).to_have_text("Changed in another tab.")
        expect(composer).to_have_value("MY_UNSENT_CORRECTION")
        expect(composer).to_be_enabled()
        page.reload()
        expect(composer).to_have_value("MY_UNSENT_CORRECTION")
        assert len(service.model.requests) == 2
        assert "MY_UNSENT_CORRECTION" not in str(service.model.requests)
    finally:
        page.close()
