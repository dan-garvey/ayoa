"""Small rules-neutral input contracts shared by story-facing roles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


PARTIAL_MODE_MARKER = (
    "Stop before the attempted action resolves; the player supplies the response."
)


@dataclass(frozen=True)
class AuthoritativeContributionRequest:
    character_id: str
    local_context: str


@dataclass(frozen=True)
class AuthoritativeResultPlan:
    authority_label: str
    result_text: str
    ruleset_actor_id: str
    viewpoint_character_id: str
    submitted_command: str
    contribution_requests: tuple[AuthoritativeContributionRequest, ...] = ()
    location_updates: tuple[tuple[str, str], ...] = ()
    state_updates: tuple[Mapping[str, object], ...] = ()


def format_character_moment(
    *,
    frame: str,
    location: str = "",
    local_context: str = "",
) -> str:
    """Render only the fictional situation a character may act within."""

    frame = (frame or "foreground").strip().lower()
    if frame not in {"foreground", "private", "autonomous"}:
        frame = "foreground"
    context = (local_context or "").strip()
    if frame == "foreground":
        return f"What is immediately true here:\n{context}" if context else ""
    lines: list[str] = []
    place = (location or "").strip()
    if place:
        lines.append(f"You are at {place}.")
    if context:
        lines.extend(["", "What is presently open here:", context])
    return "\n".join(lines)
