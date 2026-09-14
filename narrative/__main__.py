"""Local entry points; proxy execution never imports an API client."""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from . import core


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Write, revise, and resume an interactive story")
    commands = result.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="Open the local browser chat")
    chat.add_argument("--sessions", type=Path, default=core.ROOT / "sessions")
    chat.add_argument("--port", type=int, default=8765)
    chat.add_argument("--transport", choices=("api", "proxy"), default="proxy")
    chat.add_argument("--model", default="gpt-5.6-terra")
    chat.add_argument("--reasoning", default="max")
    chat.add_argument("--max-output-tokens", type=int, default=12000)
    chat.add_argument("--env-file", type=Path)
    for name in ("init", "turn", "accept", "resume", "export"):
        command = commands.add_parser(name)
        command.add_argument("--session", type=Path, required=True)
        if name == "init":
            command.add_argument("--story", required=True, help="Bundled story name or directory")
            command.add_argument("--prompts", type=Path, default=core.ROOT / "prompts")
            command.add_argument("--player-file", type=Path)
            command.add_argument("--transport", choices=("api", "proxy"), default="proxy")
            command.add_argument("--model", default="gpt-5.6-terra")
            command.add_argument("--reasoning", default="max")
            command.add_argument("--max-output-tokens", type=int, default=12000)
        if name == "turn":
            inputs = command.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--text")
            inputs.add_argument("--input-file", type=Path)
        if name == "accept":
            command.add_argument("--request-id", required=True)
            command.add_argument("--output-file", type=Path, required=True)
        if name in {"turn", "resume"}:
            command.add_argument("--env-file", type=Path)
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

            serve(
                args.sessions,
                port=args.port,
                transport=args.transport,
                model=args.model,
                reasoning=args.reasoning,
                max_output_tokens=args.max_output_tokens,
                client_factory=lambda: api_client(args.env_file),
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
                transport=args.transport,
                model=args.model,
                reasoning=args.reasoning,
                max_output_tokens=args.max_output_tokens,
            )
        elif args.command == "accept":
            value = core.accept(
                session, args.request_id, args.output_file.read_text(encoding="utf-8")
            )
        elif args.command == "export":
            value = core.export(session)
        else:
            manifest, _ = core.load_session(session)
            context = api_client(args.env_file) if manifest["transport"] == "api" else nullcontext()
            with context as client:
                if args.command == "turn":
                    text = (
                        args.input_file.read_text(encoding="utf-8")
                        if args.input_file
                        else args.text
                    )
                    value = core.submit(session, text, client)
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
