from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.schemas.conversation import ConversationMessage

logger = logging.getLogger(__name__)

# HTML-style comment blocks (see render())
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# `{include "name"}` — loads `app/prompts/_partials/{name}.txt` (recursive)
_INCLUDE_RE = re.compile(r'\{include\s+"([^"]+)"\s*\}')


class PromptManager:
    """Loads, renders, and versions prompt templates."""

    def __init__(self, prompts_dir: str = "app/prompts"):
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

    def _expand_includes(self, text: str, *, depth: int = 0) -> str:
        """Splice in partial templates from `prompts_dir/_partials/`. Recurses for nested includes."""
        if depth > 12:
            raise ValueError("Include nesting exceeded maximum depth (12)")

        def _repl(m: re.Match[str]) -> str:
            name = m.group(1).strip()
            if not name or "/" in name or ".." in name:
                raise ValueError(f"Invalid include name: {name!r}")
            partial_path = self.prompts_dir / "_partials" / f"{name}.txt"
            if not partial_path.is_file():
                raise FileNotFoundError(
                    f"Include not found: {partial_path} (referenced from a template)"
                )
            inner = partial_path.read_text()
            inner = _COMMENT_RE.sub("", inner)
            return self._expand_includes(inner, depth=depth + 1)

        return _INCLUDE_RE.sub(_repl, text)

    def _find_template(self, template_name: str) -> Path:
        """Find a template file by name (without version suffix or extension).

        Searches for files matching {template_name}_v*.txt and returns
        the highest version. Sort key is the integer version, NOT
        lexicographic — so v10 beats v9 once we cross that boundary.
        """
        pattern = f"{template_name}_v*.txt"
        matches = list(self.prompts_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No template found matching '{pattern}' in {self.prompts_dir}"
            )
        version_re = re.compile(r"_v(\d+)$")

        def _version_key(p: Path) -> int:
            m = version_re.search(p.stem)
            return int(m.group(1)) if m else -1

        matches.sort(key=_version_key)
        return matches[-1]  # highest version

    def _find_template_exact(self, template_name: str) -> Path:
        """Find a template by exact filename stem (e.g. 'narrator_phase1_v1')."""
        path = self.prompts_dir / f"{template_name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        return path

    def render(self, template_name: str, **variables) -> str:
        """Load and render a template with the given variables.

        template_name can be either:
        - A base name like 'narrator_phase1' (uses latest version)
        - An exact name like 'narrator_phase1_v1'
        """
        try:
            path = self._find_template_exact(template_name)
        except FileNotFoundError:
            path = self._find_template(template_name)

        raw = path.read_text()

        # Strip HTML-style comment blocks from the required-vars check so
        # templates can carry contract documentation (with `{var_name}`
        # examples) without those example placeholders being treated as
        # required variables. Comments are ALSO stripped from the final
        # rendered output so they don't inflate the prompt.
        #
        # Only whole-line `<!-- ... -->` blocks are recognized; inline HTML
        # comments inside prose would be rare and we'd rather preserve them.
        stripped = _COMMENT_RE.sub("", raw)
        stripped = self._expand_includes(stripped)

        required = set(re.findall(r"\{(\w+)\}", stripped))
        provided = set(variables.keys())
        missing = required - provided
        if missing:
            raise KeyError(
                f"Template '{template_name}' requires variables {missing} "
                f"that were not provided"
            )

        return stripped.format(**variables)

    def render_messages(self, template_name: str, **variables) -> list[dict[str, str]]:
        """Render a template and split it into system + user messages on `<<<USER>>>`.

        The text before the delimiter becomes the system message (frozen, cacheable)
        and the text after becomes the user message (volatile per turn). Templates
        without the delimiter are rejected — every engine template must declare
        which portion is the stable prefix, or caching cannot work.
        """
        rendered = self.render(template_name, **variables)
        marker = "<<<USER>>>"
        if marker not in rendered:
            raise ValueError(
                f"Template '{template_name}' must contain the `{marker}` delimiter "
                f"separating the frozen system prefix from the per-turn user body."
            )
        system, user = rendered.split(marker, 1)
        return [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]

    def render_conversation(
        self,
        template_name: str,
        history: list["ConversationMessage"],
        **variables,
    ) -> list[dict]:
        """Render system + user template, inserting a rolling conversation history.

        Result: `[system, *history, current_user_message]`. Used for rolling-
        conversation call sites (event_router prompt template, CharacterAgent,
        Narrator phase 2).
        """
        messages = self.render_messages(template_name, **variables)
        system_msg, user_msg = messages[0], messages[1]
        result: list[dict] = [system_msg]
        for m in history:
            result.append({"role": m.role, "content": m.content})
        result.append(user_msg)
        return result

    def get_version(self, template_name: str) -> str:
        """Get the version string for a template.

        Extracts from filename: narrator_phase1_v1.txt -> 'v1'
        """
        try:
            path = self._find_template_exact(template_name)
        except FileNotFoundError:
            path = self._find_template(template_name)

        stem = path.stem  # e.g. 'narrator_phase1_v1'
        match = re.search(r"_(v\d+)$", stem)
        if match:
            return match.group(1)
        return "unknown"

    def get_all_versions(self) -> dict[str, str]:
        """Get version strings for all templates in the prompts directory."""
        versions = {}
        for path in sorted(self.prompts_dir.glob("*_v*.txt")):
            stem = path.stem
            match = re.search(r"^(.+?)_(v\d+)$", stem)
            if match:
                base_name = match.group(1)
                version = match.group(2)
                versions[base_name] = version  # latest wins due to sorted
        return versions
