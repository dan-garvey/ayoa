"""Shared string constants for prompt-code contracts.

Router mode headers still use structured strings because the router must
choose among several machine-readable turn modes. The narrator partial
instruction is plain language because it is visible to a prose model and
does not need an implementation label.
"""

PARTIAL_MODE_MARKER = (
    "Stop before the attempted action resolves; the player supplies the response."
)
CAT_II_RESOLUTION_HEADER = "## Cat II Resolution"
UNANSWERED_RESPONDERS_SUBHEADER = "## Responders Without Submitted Intentions"
ROUTER_CONTINUATION_HEADER = "## Continuation Required"
ACTOR_SUBMISSION_HEADER = "## Actor Submission"

# v11 unified-agent turn marker. The agent template (`agent.txt`) is a
# single system prompt for foreground, private/background, and perception
# calls. Character identity/current state and the turn frame live in the
# user tail, so characters on the same model role can share the cached
# system prefix.
AGENT_TURN_HEADER = "## AGENT-TURN"
# Perception mode: the world is asking you to describe how you
# present yourself right now (clothes, grooming, posture, mood-tell).
# Observer-agnostic — the character authors their loadout for the
# moment; whoever's looking sees the same surface. Fired by the
# engine's harvest fork when the router classifies an action as
# pure observation (`event_kind="observation_harvest"`) and
# also reachable later from `/query` for "what does X look like?"
# style questions. NOT part of the on-stage cascade — perception
# calls don't move the beat forward, don't appear in canonical
# events except as enriched observable_facts, and don't pin slots.
AGENT_PERCEPTION_HEADER = "## PERCEPTION"


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


def format_agent_perception_body() -> str:
    """v11: agent perception-mode user-message body.

    The full user message is
    `{AGENT_PERCEPTION_HEADER}\\n\\n{pending_observations_block}\\n\\n{this body}`.
    The mode header flips the agent into Perception Mode; this body
    is fixed prose because perception has no per-turn observation
    surface. The shared agent template puts the character's identity
    and current state (goals, objectives, secrets) in the per-call user
    tail so the cached system prefix stays shared across characters in
    the same model tier.
    No location context: presentation is observer-agnostic and largely
    location-invariant. No "what to advance" prompt: perception is not
    action. The pending_observations slot above this body is sent
    empty for perception calls (set in `character_agent.perceive`),
    so a perception query is never primed by "react to these
    incoming events."
    """
    return (
        "## What The World Sees Of You Right Now\n"
        "Describe your visual loadout for this moment — what someone "
        "in this world would see if they looked your way. Pull from "
        "your character: clothes, grooming, jewelry, marks, posture, "
        "the mood you're carrying in your face and body. Make a "
        "deliberate choice; this is part of how you express yourself, "
        "not a costume sheet. Concrete and specific beats decorative.\n\n"
        "Hard prose constraint: do NOT use reflective-simile or "
        "hypothetical-person frames in this loadout. Banned shapes "
        "include \"with the [quality] of someone/people who...\", "
        "\"the [look/expression/kind/sort] of someone who...\", "
        "\"the way someone...\", and \"like someone who...\". Write the "
        "visible surface directly."
    )


def format_agent_turn_body(
    *,
    frame: str,
    location_context: str = "",
    local_context: str = "",
) -> str:
    """Unified agent-turn body.

    `frame` is a routing label for the character, not hidden engine
    mechanics. Foreground turns react to pending observations. Private and
    background frames advance one objective from the character's current
    location without assuming a player is watching.
    """
    frame = (frame or "foreground").strip().lower()
    if frame not in {"foreground", "private", "background"}:
        frame = "foreground"
    lines = [
        "## Turn Frame",
        frame,
    ]
    if frame == "foreground":
        lines.extend([
            "",
            "React in real time to what reached you. If nothing needs "
            "your action, silence is a valid beat.",
        ])
        cleaned_local_context = (local_context or "").strip()
        if cleaned_local_context:
            lines.extend([
                "",
                "## Local Context",
                cleaned_local_context,
            ])
        return "\n".join(lines)

    location = (
        location_context
        if location_context.strip()
        else "Location: Off-screen / unspecified location."
    )
    lines.extend([
        "",
        "## Where You Are",
        location,
    ])
    cleaned_local_context = (local_context or "").strip()
    if cleaned_local_context:
        lines.extend([
            "",
            "## Local Context",
            cleaned_local_context,
        ])
    lines.extend([
        "",
        "## What You Do",
        "Advance one objective in a single tight beat. If the right move "
        "is to hold, choose silence and record why in the parenthetical.",
    ])
    return "\n".join(lines)
