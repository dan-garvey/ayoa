"""Shared string constants for prompt-code contracts.

Router mode headers still use structured strings because the router must
choose among several machine-readable turn modes. The narrator partial
instruction is plain language because it is visible to a prose model and
does not need an implementation label.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

PARTIAL_MODE_MARKER = (
    "Stop before the attempted action resolves; the player supplies the response."
)
CAT_II_RESOLUTION_HEADER = "## Cat II Resolution"
UNANSWERED_RESPONDERS_SUBHEADER = "## Responders Without Submitted Intentions"
ROUTER_CONTINUATION_HEADER = "## Continuation Required"
ACTOR_SUBMISSION_HEADER = "## Actor Submission"
AUTHORITATIVE_RESULT_HEADER = "## Authoritative Result"


@dataclass(frozen=True)
class AuthoritativeContributionRequest:
    """One character-local reaction requested before a fixed result closes."""

    character_id: str
    local_context: str


@dataclass(frozen=True)
class AuthoritativeResultPlan:
    """Engine-owned result material for one closed router canonicalization.

    The router sees only fictional authority, collected contributions, and
    the fixed changes it must portray. ``ruleset_actor_id`` and
    ``viewpoint_character_id`` are runtime routing inputs and are never
    serialized as controller metadata.
    """

    authority_label: str
    result_text: str
    ruleset_actor_id: str
    viewpoint_character_id: str
    submitted_command: str
    contribution_requests: tuple[AuthoritativeContributionRequest, ...] = ()
    location_updates: tuple[tuple[str, str], ...] = ()
    state_updates: tuple[Mapping[str, object], ...] = ()

def format_actor_submission(character_id: str, submission_text: str) -> str:
    """One uncommitted fictional-character submission for adjudication.

    The same envelope is used whether the text was supplied directly or
    authored by a character agent.  The router therefore sees only the
    submitting fictional character and proposed motion, never its runtime
    communication source.
    """
    text = (submission_text or "").strip() or "(no public action)"
    return "\n".join([
        ACTOR_SUBMISSION_HEADER,
        "",
        f"submitted_actor_id: {character_id}",
        "submission_text:",
        text,
    ])


def format_authoritative_result_block(
    plan: AuthoritativeResultPlan,
    *,
    character_contributions: Sequence[tuple[str, str]] = (),
) -> str:
    """Serialize one already-established result without an actor submission.

    Empty character contributions have no placeholder section. This keeps a
    later deterministic result from implying that anyone spoke or acted when
    the engine supplied no such contribution.
    """

    lines = [
        AUTHORITATIVE_RESULT_HEADER,
        "",
        f"authority: {plan.authority_label.strip() or 'System'}",
    ]
    contributions = [
        {
            "character_id": character_id.strip(),
            "contribution": text.strip(),
        }
        for character_id, text in character_contributions
        if character_id.strip() and text.strip()
    ]
    if contributions:
        lines.extend([
            "",
            "character_contributions:",
            json.dumps(
                contributions,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ])
    lines.extend([
        "",
        "result:",
        plan.result_text.strip(),
    ])
    if plan.location_updates:
        lines.extend([
            "",
            "fixed_location_updates:",
            json.dumps(
                [
                    {
                        "character_id": character_id,
                        "location_label": location,
                    }
                    for character_id, location in plan.location_updates
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ])
    if plan.state_updates:
        lines.extend([
            "",
            "fixed_state_updates:",
            json.dumps(
                list(plan.state_updates),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ])
    return "\n".join(lines)


def format_cat_ii_resolution_block(
    *,
    initiator_id: str,
    initiator_intention: str,
    responders: list[tuple[str, str]],  # (responder_id, intention_text)
    swept_responders: list[str],
) -> str:
    """Part C Cat II Resolution block. Responder intentions from
    `swept_responders` are listed under a source-neutral unanswered heading,
    not in the responder list, so the prompt receives neither runtime control
    metadata nor the internal sweep sentinel."""
    lines = [CAT_II_RESOLUTION_HEADER, ""]
    lines.append(f"Initiator ({initiator_id}): {initiator_intention}")
    lines.append("")
    live_responders = [
        (rid, itext) for rid, itext in responders
        if rid not in swept_responders
    ]
    if live_responders:
        lines.append("Required responders and their intentions:")
        for rid, itext in live_responders:
            lines.append(f"- {rid}: {itext}")
        lines.append("")
    if swept_responders:
        lines.append(UNANSWERED_RESPONDERS_SUBHEADER)
        for rid in swept_responders:
            lines.append(
                f"- {rid}: no intention received; adjudicate as present "
                "but non-reactive"
            )
        lines.append("")
    lines.append(
        "Compose the resolved canonical event per Part C. Emit "
        "requires_responders=false and event_kind=cat_ii_resolution."
    )
    return "\n".join(lines)


def format_router_continuation_block(
    *,
    prior_rationale: str = "",
    original_action: str = "",
) -> str:
    """Ask the router to author another event for an open visible sequence.

    This is not a new character intention. It is used after a narrator
    continuation handoff says the visible sequence still needs motion. The
    next router output must either create a concrete response boundary or
    supply a real character continuation.
    """
    lines = [
        ROUTER_CONTINUATION_HEADER,
        "",
        (
            "The visible sequence still needs motion, but no character was "
            "selected to act next."
        ),
        "",
        (
            "Author the next canonical event that gives the active moment forward "
            "motion. Use concrete observable facts, a meaningful "
            "environmental or production cue, an existing off-screen "
            "character moving into perception, or a narratively meaningful "
            "spawn. Do not invent an action by the acting character unless "
            "the established situation already makes that action unavoidable."
        ),
        "",
        (
            "Do not open Cat II in this mode. Author a closed cue or select a "
            "character with `routing_role=next_output`. Never emit "
            "`event_kind=beat_continues` without a next-output character."
        ),
    ]
    action = (original_action or "").strip()
    if action:
        lines.extend(["", f"Original submitted action: {action}"])
    cleaned = (prior_rationale or "").strip()
    if cleaned:
        lines.extend(["", f"Prior diagnostic: {cleaned}"])
    return "\n".join(lines)


def format_character_moment(
    *,
    frame: str,
    location: str = "",
    local_context: str = "",
) -> str:
    """Render a character's immediate situation without routing labels.

    The runtime still distinguishes an immediate exchange from an off-stage
    opportunity, but the character receives only the fictional circumstance.
    """

    frame = (frame or "foreground").strip().lower()
    if frame not in {"foreground", "private", "background"}:
        frame = "foreground"
    cleaned_local_context = (local_context or "").strip()
    if frame == "foreground":
        if not cleaned_local_context:
            return ""
        return "\n".join([
            "What is immediately true here:",
            cleaned_local_context,
        ])

    lines = []
    cleaned_location = (location or "").strip()
    if cleaned_location:
        lines.append(f"You are at {cleaned_location}.")
    else:
        lines.append("You are away from the immediate exchange; the place is not established.")
    if cleaned_local_context:
        lines.extend([
            "",
            "What is presently open here:",
            cleaned_local_context,
        ])
    return "\n".join(lines)
