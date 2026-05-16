from __future__ import annotations

import re


_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_terminal_control(text: str) -> str:
    """Remove terminal control bytes from player-authored or echoed text."""
    cleaned = _ANSI_ESCAPE_RE.sub("", str(text or ""))
    return _CONTROL_CHAR_RE.sub("", cleaned)
