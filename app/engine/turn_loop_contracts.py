"""v11-r4a: shared string constants for prompt-code contracts.

The router + narrator prompts expect specific structured markers in
their user-message input (e.g. `## Render Mode: PARTIAL`). Those
markers must match EXACTLY. This module is the single source of truth
for the marker strings and the helpers that build them; prompt
templates reference these via {var} interpolation, and the context
builders that populate the prompt's user message use the same helpers
to emit the blocks. Any future drift trips a loud test failure instead
of silently regressing the prompt's behavior.
"""

PARTIAL_MODE_MARKER = "## Render Mode: PARTIAL"
CAT_II_RESOLUTION_HEADER = "## Cat II Resolution"
SWEPT_RESPONDERS_SUBHEADER = "## Swept Responders (AFK)"
INTENTION_BLOCK_HEADER = "## Intention"
TICK_FAN_IN_HEADER = "## Off-Stage Tick"


def format_human_initiator_intention(name: str, user_input: str) -> str:
    """Cat I / Cat II OPEN path: a human player's /act.

    DO NOT use for NPC cascade intentions — the "attempts:" framing
    biases the router toward Cat II classification on dialogue (which
    is Cat I). Use `format_npc_cascade_intention` for cascade steps.
    See the USER-TEMPLATE CONTRACT block in event_router_v9.txt for
    the full shape contract.
    """
    return f"{INTENTION_BLOCK_HEADER}\n{name} attempts: {user_input}"


def format_npc_cascade_intention(name: str, intention: str) -> str:
    """NPC cascade step in a beat (not a fresh player action)."""
    return f"{INTENTION_BLOCK_HEADER}\n{name} intends: {intention}"


def format_ooc_directive(user_input: str) -> str:
    """OOC author-directive like `(begin)` / `(skip to morning)`."""
    return f"{INTENTION_BLOCK_HEADER}\n(OOC) {user_input}"


def format_cat_ii_resolution_block(
    *,
    initiator_id: str,
    initiator_intention: str,
    responders: list[tuple[str, str]],  # (responder_id, intention_text)
    swept_responders: list[str],
) -> str:
    """Part C Cat II Resolution block. Responder intentions from
    swept_responders (AFK-timeout) are LISTED in the swept sub-header
    only — not in the responder list — so the prompt never sees the
    debug sentinel text."""
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
        lines.append(SWEPT_RESPONDERS_SUBHEADER)
        for rid in swept_responders:
            lines.append(f"- {rid} (AFK — render as present-but-non-reactive)")
        lines.append("")
    lines.append(
        "Compose the resolved canonical event per Part C. Emit "
        "requires_responders=false, ends_beat=true, "
        "ends_beat_reason=cat_ii_resolution, empty picks."
    )
    return "\n".join(lines)


def format_partial_render_marker() -> str:
    """Narrator PARTIAL mode marker. Caller prepends this to the user
    input when the human being rendered to is a pinned Cat II
    responder."""
    return PARTIAL_MODE_MARKER


def format_open_attempt_outcome(actor_name: str, intention: str) -> str:
    """v11-r6a: the `resolved_outcome` string for a synthetic 'open Cat
    II attempt' canonical event. The attempt is IN PROGRESS — the
    narrator's PARTIAL mode ends prose on the cliffhanger moment so
    pinned humans see the open attempt before they respond.

    `actor_name` MUST be the display name (e.g. "Pip"), NOT the
    character_id (e.g. "char_0dab1f"). The narrator reads this as
    a prose seed; an id leaks engine structure into the cliffhanger.

    Centralized so tests can pattern-match the shape without hardcoding
    the format."""
    return f"{actor_name} attempts: {intention}"


def format_tick_fan_in_block(
    entries: list[tuple[str, str, str, str]],
) -> str:
    """Commit 6: bundle N off-stage agents' public prose into a single
    user message for the unified router.

    Each entry is `(name, character_id, location, public_text)`. The
    parenthetical (private intent) the agent emitted MUST be stripped
    BEFORE this helper is called — only `public_text` belongs here.
    The information-asymmetry rule (no agent's interior reaches the
    router) is enforced at the call site; this helper just renders
    whatever it's handed.

    Empty list returns "" so the caller can skip the route call when
    no off-stage activity fired.
    """
    if not entries:
        return ""
    lines = [TICK_FAN_IN_HEADER, ""]
    lines.append(
        f"{len(entries)} off-stage NPC(s) acted between the last "
        "player /act and now. Compose ONE canonical event capturing "
        "their actions per the Tick Mode rules in the system prompt. "
        "The player did not see this beat."
    )
    lines.append("")
    for name, char_id, location, public_text in entries:
        loc = location or "(unset)"
        text = (public_text or "").strip() or "(no public action)"
        lines.append(f"- **{name}** (id: `{char_id}`, at `{loc}`): {text}")
    lines.append("")
    return "\n".join(lines)
