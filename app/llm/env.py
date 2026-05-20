from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Iterable


_LLM_ENV_PREFIXES = (
    "ANTHROPIC_",
    "LLM_",
    "OPENAI_",
    "OPEN_AI_",
)


def load_shell_export_env(paths: Iterable[Path] | None = None) -> None:
    """Load exported LLM env vars from shell startup files when absent.

    Local playtest commands are often launched by coding agents whose parent
    process inherited an older environment than the user's interactive shell.
    This reads simple `export NAME=value` lines without executing shell code.
    """

    for path in paths or (Path.home() / ".bashrc",):
        if not path.exists():
            continue
        for name, value in _iter_exported_values(path):
            if not _is_llm_env_name(name) or os.environ.get(name):
                continue
            os.environ[name] = value


def _iter_exported_values(path: Path) -> Iterable[tuple[str, str]]:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        elif "=" not in line:
            continue
        try:
            parts = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        for part in parts:
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            if name:
                yield name, value


def _is_llm_env_name(name: str) -> bool:
    return name.startswith(_LLM_ENV_PREFIXES)
