from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptManager:
    """Loads, renders, and versions prompt templates."""

    def __init__(self, prompts_dir: str = "app/prompts"):
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

    def _find_template(self, template_name: str) -> Path:
        """Find a template file by name (without version suffix or extension).

        Searches for files matching {template_name}_v*.txt and returns
        the highest version.
        """
        pattern = f"{template_name}_v*.txt"
        matches = sorted(self.prompts_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No template found matching '{pattern}' in {self.prompts_dir}"
            )
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

        # Check for missing variables
        required = set(re.findall(r"\{(\w+)\}", raw))
        provided = set(variables.keys())
        missing = required - provided
        if missing:
            raise KeyError(
                f"Template '{template_name}' requires variables {missing} "
                f"that were not provided"
            )

        return raw.format(**variables)

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
