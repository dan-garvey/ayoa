"""Local entry points; proxy execution never imports an API client."""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from . import core


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Write, regenerate, and resume an interactive story"
    )
    commands = result.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="Open the local browser chat")
    chat.add_argument("--sessions", type=Path, default=core.ROOT / "sessions")
    chat.add_argument("--port", type=int, default=8765)
    chat.add_argument("--lan-address", help="This computer's private IPv4 address for Wi-Fi access")
    chat.add_argument("--transport", choices=("api", "proxy"), default="proxy")
    chat.add_argument("--model", default=core.DEFAULT_MODEL)
    chat.add_argument("--reasoning", default="max")
    chat.add_argument("--max-output-tokens", type=int, default=12000)
    chat.add_argument(
        "--checkup-every", type=int, default=5, help="Player messages per checkup; 0 disables"
    )
    chat.add_argument("--env-file", type=Path)
    chat.add_argument("--manual", action="store_true", help="Use pasted replies for proxy sessions")
    for name in ("init", "turn", "regenerate", "accept", "resume", "export", "rename"):
        command = commands.add_parser(name)
        command.add_argument("--session", type=Path, required=True)
        if name == "init":
            command.add_argument("--story", required=True, help="Bundled story name or directory")
            command.add_argument("--prompts", type=Path, default=core.ROOT / "prompts")
            command.add_argument("--player-file", type=Path)
            command.add_argument("--player-name", help="Choose the protagonist's name")
            command.add_argument("--transport", choices=("api", "proxy"), default="proxy")
            command.add_argument("--model", default=core.DEFAULT_MODEL)
            command.add_argument("--reasoning", default="max")
            command.add_argument("--max-output-tokens", type=int, default=12000)
            command.add_argument(
                "--checkup-every",
                type=int,
                default=5,
                help="Player messages per checkup; 0 disables",
            )
        if name in {"turn", "regenerate"}:
            inputs = command.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--text")
            inputs.add_argument("--input-file", type=Path)
        if name == "accept":
            command.add_argument("--request-id", required=True)
            command.add_argument("--output-file", type=Path, required=True)
        if name == "rename":
            command.add_argument("--player-name", required=True)
        if name in {"turn", "regenerate", "resume"}:
            command.add_argument("--env-file", type=Path)
            command.add_argument("--auto", action="store_true", help="Run proxy replies with Codex")
    return result


def api_client(env_file: Path | None):
    from dotenv import load_dotenv
    from openai import OpenAI

    if env_file is not None:
        load_dotenv(env_file, override=False)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise core.NarrativeError("Set OPENAI_API_KEY or pass --env-file with that variable")
    return OpenAI(api_key=key, timeout=300.0, max_retries=0)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "chat":
            from .chat import serve
            from .codex import CodexProxy

            serve(
                args.sessions,
                port=args.port,
                lan_address=args.lan_address,
                transport=args.transport,
                model=args.model,
                reasoning=args.reasoning,
                max_output_tokens=args.max_output_tokens,
                checkup_every=args.checkup_every,
                client_factory=lambda: api_client(args.env_file),
                proxy_runner=None if args.manual else CodexProxy(),
            )
            return 0
        session = args.session.resolve()
        if args.command == "init":
            story = Path(args.story)
            if not story.is_dir():
                story = core.ROOT / "stories" / args.story
            value = core.init_session(
                session,
                story.resolve(),
                prompts=args.prompts,
                player_file=args.player_file,
                player_name=args.player_name,
                transport=args.transport,
                model=args.model,
                reasoning=args.reasoning,
                max_output_tokens=args.max_output_tokens,
                checkup_every=args.checkup_every,
            )
        elif args.command == "accept":
            value = core.accept(
                session, args.request_id, args.output_file.read_text(encoding="utf-8")
            )
        elif args.command == "export":
            value = core.export(session)
        elif args.command == "rename":
            value = core.rename_player(session, args.player_name)
        else:
            manifest, _ = core.load_session(session)
            if manifest["transport"] == "api":
                context = api_client(args.env_file)
            elif args.auto:
                from .codex import CodexProxy

                context = nullcontext(CodexProxy())
            else:
                context = nullcontext()
            with context as client:
                if args.command in {"turn", "regenerate"}:
                    text = (
                        args.input_file.read_text(encoding="utf-8")
                        if args.input_file
                        else args.text
                    )
                    action = core.regenerate if args.command == "regenerate" else core.submit
                    value = action(session, text, client)
                else:
                    value = core.resume(session, client)
        if value.get("status") == "published":
            print(value["output"])
        else:
            print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except (core.NarrativeError, ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
