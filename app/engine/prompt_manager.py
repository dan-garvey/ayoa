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
    """Loads and renders prompt templates.

    Versioning policy: prompts are tracked in git, not in their
    filenames. A template named `event_router` lives at
    `app/prompts/event_router.txt`; revisions are normal commits
    (use `git log app/prompts/event_router.txt` to read history,
    `git blame` to see who changed which rule). Pre-cleanup the
    directory accumulated `event_router_v9.txt`,
    `narrator_phase2_v9.txt`, etc.; that scheme was redundant with
    the VCS and made every refactor a multi-file rename.
    """

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
        """Resolve `app/prompts/{template_name}.txt` or raise.

        No versioned-suffix glob: prompts are versioned in git, not
        in their filenames. `template_name` is exactly the file
        stem (no `.txt` suffix, no `_v#` suffix).
        """
        path = self.prompts_dir / f"{template_name}.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Template not found: {path}"
            )
        return path

    def render(self, template_name: str, **variables) -> str:
        """Load and render a template with the given variables.

        `template_name` is the file stem under `app/prompts/`
        (e.g. `'event_router'`, `'narrator_phase2'`) — no version
        suffix. The corresponding `.txt` file is loaded, HTML-style
        comment blocks are stripped, `{include "..."}` directives
        are recursively expanded from `_partials/`, and finally
        the remaining `{var}` placeholders are substituted from
        `variables`.
        """
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
