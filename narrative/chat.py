"""Loopback browser delivery over the existing session and publication contract."""

from __future__ import annotations

import json
import secrets
import threading
import uuid
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from markdown_it import MarkdownIt

from . import core

WEB = Path(__file__).with_name("web")


def format_text(text: str) -> str:
    # Disable raw HTML and remote images. Links retain the parser's unsafe-scheme
    # checks. Soft breaks are significant in verse and player-authored passages.
    parser = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table")
    parser.disable("image")
    return parser.render(text)


def title(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def draft_view(session: Path, request_id: str | None) -> dict:
    if request_id is None:
        return {"draft": None, "draft_html": None}
    response = core.read_json(core.attempt_dir(session, request_id) / "response.json")
    text = core.response_text(response)
    return {"draft": text, "draft_html": format_text(text)}


class ChatApp:
    def __init__(
        self,
        sessions: Path,
        *,
        stories: Path = core.ROOT / "stories",
        prompts: Path = core.ROOT / "prompts",
        transport: str = "proxy",
        model: str = "gpt-5.6-terra",
        reasoning: str = "max",
        max_output_tokens: int = 12000,
        client_factory: Callable = nullcontext,
        proxy_runner: Callable | None = None,
    ):
        self.sessions = sessions.resolve()
        self.stories = stories.resolve()
        self.prompts = prompts.resolve()
        self.defaults = dict(
            transport=transport,
            model=model,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
        )
        self.client_factory = client_factory
        self.proxy_runner = proxy_runner
        self.token = secrets.token_urlsafe(32)
        self._active: set[Path] = set()
        self._guard = threading.Lock()
        self.sessions.mkdir(parents=True, exist_ok=True)

    def session_path(self, name: str) -> Path:
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError("Choose an existing session")
        path = (self.sessions / name).resolve()
        if path == self.sessions or not path.is_relative_to(self.sessions):
            raise ValueError("Choose a session inside the session directory")
        if not (path / "state.json").is_file():
            raise ValueError("This session could not be found")
        return path

    def story_list(self) -> list[dict]:
        result = []
        for path in sorted(self.stories.iterdir()):
            if (
                path.is_dir()
                and path.resolve().is_relative_to(self.stories)
                and all(
                    (path / name).is_file() for name in ("canon.md", "direction.md", "player.json")
                )
            ):
                player = core.read_json(path / "player.json")
                result.append(
                    {"id": path.name, "title": title(path.name), "player": player["name"]}
                )
        return result

    def create(
        self, story_id: str, player_name: str | None = None, *, compare: bool = False
    ) -> dict:
        if story_id not in {story["id"] for story in self.story_list()}:
            raise ValueError("Choose one of the available stories")
        name = f"{story_id}-{uuid.uuid4().hex[:12]}"
        core.init_session(
            self.sessions / name,
            self.stories / story_id,
            prompts=self.prompts,
            player_name=player_name,
            **self.defaults,
        )
        return self.view(name, compare=compare)

    @contextmanager
    def operation(self, session: Path):
        # This only describes work executing in this process. All durable turn
        # state and cross-process locking remain owned by core.
        with self._guard:
            if session in self._active:
                raise core.NarrativeError("This story is already responding")
            self._active.add(session)
        try:
            yield
        finally:
            with self._guard:
                self._active.remove(session)

    def busy(self, session: Path) -> bool:
        with self._guard:
            if session in self._active:
                return True
        return core.is_busy(session)

    def session_list(self) -> list[dict]:
        result = []
        for path in self.sessions.rglob("state.json"):
            session = path.parent.resolve()
            if not session.is_relative_to(self.sessions):
                continue
            name = session.relative_to(self.sessions).as_posix()
            try:
                manifest, state = core.load_session(session)
                player = core.player_identity(session, state)
                result.append(
                    {
                        "id": name,
                        "title": title(manifest["story"]),
                        "player": player["name"],
                        "turn_count": len(state["turns"]),
                        "updated_at": state["turns"][-1]["published_at"]
                        if state["turns"]
                        else manifest["created_at"],
                    }
                )
            except (ValueError, OSError, KeyError, core.NarrativeError):
                result.append({"id": name, "title": title(session.name), "unavailable": True})
        return sorted(result, key=lambda item: item.get("updated_at", ""), reverse=True)

    def view(self, name: str, *, compare: bool = False) -> dict:
        session = self.session_path(name)
        # State replacement is atomic. Reading it does not wait on a model call's
        # file lock, so reload and progress remain available during generation.
        manifest, state = core.load_session(session)
        player = core.player_identity(session, state)
        pending = state["pending"]
        waiting = None
        if pending:
            request_id = pending["request_id"]
            ready = False
            failure = False
            if request_id:
                attempt = core.attempt_dir(session, request_id)
                ready = (
                    manifest["transport"] == "proxy"
                    and self.proxy_runner is None
                    and not (attempt / "response.json").exists()
                )
                failure = (attempt / "error.json").exists()
                if (attempt / "response.json").exists():
                    try:
                        core.response_text(core.read_json(attempt / "response.json"))
                    except core.ResponseError:
                        failure = True
            waiting = {
                "input": pending["input"],
                "input_html": format_text(pending["input"]),
                "request_id": request_id,
                "handoff_ready": ready,
                "failed": failure,
                "stage": "revision" if pending["author"] else "draft",
            }
            if compare:
                waiting.update(draft_view(session, pending["author"]))
        return {
            "id": name,
            "title": title(manifest["story"]),
            "player": player["name"],
            "created_at": manifest["created_at"],
            "transport": manifest["transport"],
            "compare": compare,
            "turn_count": len(state["turns"]),
            "version": core.state_version(state),
            "turns": [
                {
                    "number": turn["turn"],
                    "input": turn["input"],
                    "input_html": format_text(turn["input"]),
                    "output": turn["output"],
                    "output_html": format_text(turn["output"]),
                    **(draft_view(session, turn["author"]) if compare else {}),
                }
                for turn in state["turns"]
            ],
            "pending": waiting,
            "busy": self.busy(session),
        }

    def action(self, name: str, command: str, body: dict, *, compare: bool = False) -> dict:
        session = self.session_path(name)
        with self.operation(session):
            manifest, _ = core.load_session(session)
            if command in {"turn", "rename"}:
                if (
                    not isinstance(body.get("expected_version"), str)
                    or not body["expected_version"]
                ):
                    raise ValueError("Refresh the story before trying again")
            if command == "accept":
                text = body.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Paste the complete response before accepting it")
                request_id = body.get("request_id")
                if not isinstance(request_id, str):
                    raise ValueError("Open the current handoff before accepting a response")
                core.accept(session, request_id, text)
            elif command == "rename":
                core.rename_player(
                    session, body.get("player_name"), expected_version=body["expected_version"]
                )
            else:
                if command == "turn":
                    if not isinstance(body.get("text"), str) or not body["text"].strip():
                        raise ValueError("Write a turn before sending it")
                context = (
                    self.client_factory()
                    if manifest["transport"] == "api"
                    else nullcontext(self.proxy_runner)
                )
                with context as client:
                    if command == "turn":
                        core.submit(
                            session, body["text"], client, expected_version=body["expected_version"]
                        )
                    elif command == "resume":
                        core.resume(session, client)
                    else:
                        raise ValueError("Unknown story action")
        return self.view(name, compare=compare)

    def handoff(self, name: str) -> dict:
        session = self.session_path(name)
        manifest, state = core.load_session(session)
        pending = state["pending"]
        if manifest["transport"] != "proxy" or not pending or not pending["request_id"]:
            raise core.NarrativeError("There is no pending handoff")
        path = core.attempt_dir(session, pending["request_id"])
        if (path / "response.json").exists():
            raise core.NarrativeError("A response is already saved; continue the story first")
        # Return the actual prepared packet; do not build a second input surface.
        request = core._check_request(session, manifest, state, path)
        return {
            "request_id": pending["request_id"],
            "stage": "revision" if pending["author"] else "draft",
            "text": core.render_request(request),
        }


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, app: ChatApp, port: int = 8765):
        self.app = app
        super().__init__(("127.0.0.1", port), ChatHandler)


class ChatHandler(BaseHTTPRequestHandler):
    server: ChatServer

    def log_message(self, *_):
        # Turn text, packet contents and credentials do not belong in HTTP logs.
        pass

    def respond(
        self, status: int, data: bytes, content_type="application/json; charset=utf-8", **headers
    ):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'none'; "
            "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # The completed response remains in canonical session storage.

    def json(self, status: int, value: dict):
        self.respond(status, json.dumps(value, ensure_ascii=False).encode())

    def allowed(self) -> bool:
        port = self.server.server_port
        host = self.headers.get("Host", "")
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            self.json(403, {"error": "Open the chat using its localhost address"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            self.json(403, {"error": "This request did not come from the local chat"})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        name = query.get("id", [""])[0]
        app = self.server.app
        try:
            if url.path in {"/", "/app.js", "/style.css"}:
                filename, content_type = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                    "/style.css": ("style.css", "text/css; charset=utf-8"),
                }[url.path]
                self.respond(200, (WEB / filename).read_bytes(), content_type)
            elif url.path == "/api/bootstrap":
                self.json(
                    200,
                    {
                        "token": app.token,
                        "stories": app.story_list(),
                        "transport": app.defaults["transport"],
                        "automatic": app.defaults["transport"] == "api"
                        or app.proxy_runner is not None,
                    },
                )
            elif url.path == "/api/sessions":
                self.json(200, {"sessions": app.session_list()})
            elif url.path == "/api/session":
                self.json(200, app.view(name, compare=query.get("compare") == ["1"]))
            elif url.path == "/api/handoff":
                self.json(200, app.handoff(name))
            elif url.path == "/api/transcript":
                session = app.session_path(name)
                core.export(session)
                self.respond(
                    200,
                    (session / "transcript.md").read_bytes(),
                    "text/markdown; charset=utf-8",
                    Content_Disposition='attachment; filename="story.md"',
                )
            elif url.path == "/favicon.ico":
                self.respond(204, b"")
            else:
                self.json(404, {"error": "Page not found"})
        except (ValueError, core.NarrativeError) as exc:
            self.json(409, {"error": str(exc)})
        except (OSError, KeyError):
            self.json(409, {"error": "This session could not be read. Its files are unchanged."})

    def do_POST(self):
        if not self.allowed():
            return
        app = self.server.app
        if not secrets.compare_digest(self.headers.get("X-Chat-Token", ""), app.token):
            self.json(403, {"error": "Reload the chat before trying again"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("The submitted text is empty or too large")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Send this action from the chat interface")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Invalid story action")
            url = urlsplit(self.path)
            route = url.path
            compare = parse_qs(url.query).get("compare") == ["1"]
            if route == "/api/sessions":
                self.json(
                    201, app.create(body.get("story"), body.get("player_name"), compare=compare)
                )
            elif route in {"/api/turn", "/api/resume", "/api/accept", "/api/rename"}:
                self.json(
                    200,
                    app.action(body.get("id"), route.removeprefix("/api/"), body, compare=compare),
                )
            else:
                self.json(404, {"error": "Unknown story action"})
        except (ValueError, core.NarrativeError) as exc:
            self.json(409, {"error": str(exc)})
        except (OSError, KeyError):
            self.json(
                409, {"error": "Could not finish this action. Reload to check the saved story."}
            )
        except Exception:
            self.json(
                500,
                {
                    "error": "The response stopped unexpectedly. Reload to check the saved turn before retrying."
                },
            )


def serve(sessions: Path, *, port: int = 8765, **settings):
    app = ChatApp(sessions, **settings)
    with ChatServer(app, port) as server:
        print(f"Story chat: http://localhost:{server.server_port}", flush=True)
        print("Press Ctrl+C to stop. Sessions are saved as you play.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
