import copy
import json
import threading
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace

import httpx
import pytest

from narrative.chat import ChatApp, ChatServer


class StoryClient:
    def __init__(self, replies, pause_at=None):
        self.responses = self
        self.replies = iter(replies)
        self.requests = []
        self.pause_at = pause_at
        self.started = threading.Event()
        self.release = threading.Event()

    def reply(self, request):
        self.requests.append(copy.deepcopy(request))
        if len(self.requests) == self.pause_at:
            self.started.set()
            if not self.release.wait(15):
                raise TimeoutError("The browser test did not release its response")
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def __call__(self, request):
        reply = self.reply(request)
        return reply if isinstance(reply, dict) else {"status": "completed", "raw": reply}

    def create(self, **request):
        reply = self.reply(request)
        value = {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": reply}]}],
        }
        return SimpleNamespace(model_dump=lambda **_: value)


@pytest.fixture
def chat_service(tmp_path):
    stories = tmp_path / "stories"
    story = stories / "harbor"
    story.mkdir(parents=True)
    (story / "canon.md").write_text("SECRET_CANON: Ivo hid the key. Nobody else knows.")
    (story / "direction.md").write_text("Begin by the harbor.")
    (story / "player.json").write_text(json.dumps({"name": "Casey", "description": "A visitor."}))
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "author.txt").write_text("Write the next passage.")
    (prompts / "regenerate.txt").write_text(
        "Rewrite the previous passage using these instructions."
    )
    services = []
    with ExitStack() as stack:

        def start(*replies, transport="proxy", pause_at=None, auto_proxy=False, lan_address=None):
            model = StoryClient(replies, pause_at)
            app = ChatApp(
                tmp_path / f"sessions-{len(services)}",
                stories=stories,
                prompts=prompts,
                transport=transport,
                client_factory=lambda: nullcontext(model),
                proxy_runner=model if auto_proxy else None,
            )
            server = ChatServer(app, 0, lan_address=lan_address)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            client = stack.enter_context(httpx.Client(base_url=url, timeout=20, trust_env=False))
            client.headers["X-Chat-Token"] = app.token
            service = SimpleNamespace(
                app=app, server=server, thread=thread, url=url, client=client, model=model
            )
            services.append(service)
            return service

        yield start
        for service in services:
            service.model.release.set()
            service.server.shutdown()
            service.server.server_close()
            service.thread.join(timeout=5)


def pytest_addoption(parser):
    parser.addoption("--with-browser", action="store_true", help="Run offline Chromium UI tests")


@pytest.fixture(scope="session")
def browser(request):
    if not request.config.getoption("--with-browser"):
        pytest.skip("Use --with-browser after installing Playwright Chromium")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()
