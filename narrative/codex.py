"""Fresh, text-only coding-agent execution of the existing proxy request."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from . import core


def reasoning_summaries(events: bytes, *, completed: bool) -> list[str]:
    """Keep only completed, publicly exposed summary items from the CLI stream."""
    summaries = []
    lines = events.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            # A stopped process can leave its final JSON event only partly written.
            if not completed and index == len(lines) - 1:
                break
            raise
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "reasoning":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                summaries.append(text)
    return summaries


class CodexProxy:
    def __init__(self, executable: str = "codex", timeout: float = 900):
        self.executable = executable
        self.timeout = timeout

    def __call__(self, request: dict) -> dict:
        executable = shutil.which(self.executable)
        if executable is None:
            raise core.NarrativeError("Codex CLI was not found; install it and run codex login")
        # An empty workspace and disabled discovery keep repository instructions,
        # memories, plugins and images out of the narrative context.
        with tempfile.TemporaryDirectory(prefix="ayoa-response-") as directory:
            output = Path(directory) / "response.txt"
            command = [
                executable,
                "exec",
                "--ignore-user-config",
                "--strict-config",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--cd",
                directory,
                "--model",
                request["model"],
                "--config",
                "model_reasoning_effort=" + json.dumps(request["reasoning"]["effort"]),
                "--config",
                "model_reasoning_summary=" + json.dumps(request["reasoning"]["summary"]),
                "--config",
                "project_doc_max_bytes=0",
                "--config",
                'web_search="disabled"',
                "--enable",
                "skip_host_skill_discovery",
                "--color",
                "never",
                "--json",
                "--output-last-message",
                str(output),
            ]
            for feature in (
                "shell_tool",
                "view_image",
                "image_generation",
                "browser_use",
                "browser_use_external",
                "in_app_browser",
                "computer_use",
                "apps",
                "plugins",
                "multi_agent",
                "memories",
                "hooks",
                "skill_search",
                "workspace_dependencies",
            ):
                command.extend(["--disable", feature])
            command.append("-")
            # Save exposed summaries separately from the final message. Other
            # events are discarded; each call still starts a fresh process.
            with subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=directory,
            ) as process:
                timed_out = False
                try:
                    events, _ = process.communicate(
                        core.render_request(request).encode(), timeout=self.timeout
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # It exited between the timeout and the signal.
                    events, _ = process.communicate()
                return {
                    "executor": "codex",
                    "exit_code": process.returncode,
                    "status": "timeout"
                    if timed_out
                    else "completed"
                    if process.returncode == 0
                    else "failed",
                    "raw": output.read_bytes().decode("utf-8") if output.exists() else "",
                    "reasoning_summaries": reasoning_summaries(
                        events, completed=process.returncode == 0 and not timed_out
                    ),
                }
