import json
import sys
from pathlib import Path

import pytest

from narrative import chat, core
from narrative.__main__ import main
from narrative.codex import CodexProxy


def executable(tmp_path, source):
    path = tmp_path / "fake-codex"
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o700)
    return str(path)


def request():
    return {
        "model": "gpt-5.6-terra",
        "reasoning": {"effort": "max", "summary": "auto"},
        "instructions": "STORY_CANON",
        "input": [{"role": "user", "content": "PLAYER_SUBMISSION"}],
    }


def test_codex_keeps_final_text_and_exposed_summaries_from_a_fresh_workspace(tmp_path):
    binary = executable(
        tmp_path,
        """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
result = {'args': args, 'input': sys.stdin.read(), 'cwd': os.getcwd(), 'files': os.listdir('.')}
print(json.dumps({'type': 'item.updated', 'item': {'type': 'reasoning', 'text': 'DISCARDED_PARTIAL'}}))
print(json.dumps({'type': 'item.completed', 'item': {'type': 'reasoning', 'text': 'First summary.\\r\\nSecond line.'}}))
print(json.dumps({'type': 'item.completed', 'item': {'type': 'reasoning', 'text': 'Another summary.'}}))
print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'DISCARDED_PROGRESS'}}))
print(json.dumps({'type': 'raw_reasoning', 'text': 'DISCARDED_RAW_CONTENT'}))
print('DISCARDED_DIAGNOSTIC', file=sys.stderr)
Path(args[args.index('--output-last-message') + 1]).write_bytes(json.dumps(result).encode() + bytes([13, 10]))
""",
    )
    runner = CodexProxy(binary)
    first = runner(request())
    second = runner(request())
    assert first["status"] == "completed" and first["exit_code"] == 0
    assert first["raw"].endswith("\r\n")
    assert first["reasoning_summaries"] == ["First summary.\r\nSecond line.", "Another summary."]
    value = json.loads(first["raw"])
    assert value["input"] == core.render_request(request())
    assert value["files"] == []
    assert not Path(value["cwd"]).exists()
    assert value["cwd"] != json.loads(second["raw"])["cwd"]
    args = value["args"]
    assert args[args.index("--model") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="max"' in args
    assert 'model_reasoning_summary="auto"' in args
    assert "--json" in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    for flag in ("--ephemeral", "--ignore-user-config", "--strict-config", "--skip-git-repo-check"):
        assert flag in args
    assert "project_doc_max_bytes=0" in args and 'web_search="disabled"' in args
    for feature in (
        "view_image",
        "image_generation",
        "shell_tool",
        "plugins",
        "memories",
        "multi_agent",
    ):
        assert args[args.index(feature) - 1] == "--disable"
    assert "resume" not in args and "fork" not in args
    assert "DISCARDED" not in json.dumps(first)


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_codex_failure_preserves_partial_output_but_cannot_publish_it(tmp_path, failure):
    binary = executable(
        tmp_path,
        """
import json, sys, time
from pathlib import Path
args = sys.argv[1:]
sys.stdin.read()
print(json.dumps({'type': 'item.completed', 'item': {'type': 'reasoning', 'text': 'Completed summary before failure.'}}), flush=True)
Path(args[args.index('--output-last-message') + 1]).write_text('Unfinished reply.')
print('{"type":"item.updated","item":', end='', flush=True)
"""
        + ("sys.exit(7)\n" if failure == "exit" else "time.sleep(30)\n"),
    )
    response = CodexProxy(binary, timeout=0.5)(request())
    assert response["status"] == ("failed" if failure == "exit" else "timeout")
    assert response["raw"] == "Unfinished reply."
    assert response["reasoning_summaries"] == ["Completed summary before failure."]
    with pytest.raises(core.ResponseError):
        core.response_text({"transport": "proxy", **response})


def test_missing_codex_is_an_explicit_error(tmp_path):
    with pytest.raises(core.NarrativeError, match="Codex CLI was not found"):
        CodexProxy(str(tmp_path / "missing"))(request())


def test_chat_cli_automates_proxy_by_default_and_manual_mode_is_explicit(monkeypatch):
    calls = []
    monkeypatch.setattr(chat, "serve", lambda *args, **kwargs: calls.append(kwargs))
    assert main(["chat"]) == 0
    assert isinstance(calls[-1]["proxy_runner"], CodexProxy)
    assert main(["chat", "--manual"]) == 0
    assert calls[-1]["proxy_runner"] is None
