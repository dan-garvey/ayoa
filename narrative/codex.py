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
                'model_reasoning_summary="none"',
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
            # Only the final output is read. Progress/analysis streams are never
            # retained or sent to another model. Each call starts a fresh process.
            with subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=directory,
            ) as process:
                timed_out = False
                try:
                    process.communicate(core.render_request(request).encode(), timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # It exited between the timeout and the signal.
                    process.communicate()
                return {
                    "executor": "codex",
                    "exit_code": process.returncode,
                    "status": "timeout"
                    if timed_out
                    else "completed"
                    if process.returncode == 0
                    else "failed",
                    "raw": output.read_bytes().decode("utf-8") if output.exists() else "",
                }
