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

# v11 unified-agent mode markers. The agent template (`agent_v*.txt`)
# is a SINGLE system prompt for both on-stage and off-stage calls so a
# character keeps one cache lineage across modes — switching between
# respond and tick within the same character does NOT invalidate the
# system-prompt cache. The mode signal is the first line of the user
# message: `## ON-STAGE` or `## TICK`. The agent template's "Mode
# Routing" section reads this header and applies the matching
# mode-specific rules. Constants live here so renames trip the
# prompt-references-constants test instead of silently desynchronizing.
AGENT_ON_STAGE_HEADER = "## ON-STAGE"
AGENT_TICK_HEADER = "## TICK"
# Perception mode: the world is asking you to describe how you
# present yourself right now (clothes, grooming, posture, mood-tell).
# Observer-agnostic — the character authors their loadout for the
# moment; whoever's looking sees the same surface. Fired by the
# engine's harvest fork when the router classifies an action as
# pure observation (`ends_beat_reason="observation_harvest"`) and
# also reachable later from `/query` for "what does X look like?"
# style questions. NOT part of the on-stage cascade — perception
# calls don't move the beat forward, don't appear in canonical
# events except as enriched observable_facts, and don't pin slots.
AGENT_PERCEPTION_HEADER = "## PERCEPTION"


def format_human_initiator_intention(name: str, user_input: str) -> str:
    """Cat I / Cat II OPEN path: a human player's /act.

    DO NOT use for NPC cascade intentions — the "attempts:" framing
    biases the router toward Cat II classification on dialogue (which
    is Cat I). Use `format_npc_cascade_intention` for cascade steps.
    See the USER-TEMPLATE CONTRACT block in event_router.txt for
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


def format_agent_on_stage_body(
    *,
    scene_context: str,
    characters_present: str,
    observed_facts: str,
    prior_character_responses: str,
) -> str:
    """v11 unified-agent on-stage user-message body.

    The full user message is
    `{AGENT_ON_STAGE_HEADER}\\n\\n{agent_user_state_block}\\n\\n{this body}`.
    The mode header is the first-token bitflip that the agent prompt's
    "Mode Routing" section keys off; this helper just shapes the
    on-stage payload (scene + presence + observed facts + prior
    responders) in the order the prompt's On-Stage Mode rules expect.

    All inputs are pre-formatted strings — the helper does not
    interpret or filter them. In particular `prior_character_responses`
    must already have been stripped of other agents' parentheticals
    (see `format_prior_responses` in context_builder); piping raw
    intent here would be a load-bearing information-asymmetry
    violation.
    """
    return (
        f"## Scene\n{scene_context}\n\n"
        f"## Characters Present\n{characters_present}\n\n"
        f"## What You Observe This Turn\n{observed_facts}\n\n"
        f"## Other Characters' Responses This Turn\n"
        f"{prior_character_responses}"
    )


def format_agent_perception_body() -> str:
    """v11: agent perception-mode user-message body.

    The full user message is
    `{AGENT_PERCEPTION_HEADER}\\n\\n{agent_user_state_block}\\n\\n{this body}`.
    The mode header flips the agent into Perception Mode; this body
    is fixed prose because perception has no per-turn observation
    surface — the character's identity (in the cached system prompt)
    + their current state (in the user state block above this body)
    are the only inputs the model needs to author its visual
    loadout. No scene context: presentation is observer-agnostic
    and largely scene-invariant. No "what to advance" prompt:
    perception is not action.
    """
    return (
        "## What The World Sees Of You Right Now\n"
        "Describe your visual loadout for this moment — what someone "
        "in this world would see if they looked your way. Pull from "
        "your character: clothes, grooming, jewelry, marks, posture, "
        "the mood you're carrying in your face and body. Make a "
        "deliberate choice; this is part of how you express yourself, "
        "not a costume sheet. Concrete and specific beats decorative."
    )


def format_agent_tick_body(*, scene_context: str) -> str:
    """v11 unified-agent off-stage tick user-message body.

    The full user message is
    `{AGENT_TICK_HEADER}\\n\\n{agent_user_state_block}\\n\\n{this body}`.
    The mode header flips the agent into Tick Mode. This helper
    renders the location and the standing tick instruction (advance
    one objective in your own location, single tight beat) — kept as
    a fixed string because the off-stage tick has no per-turn
    observation surface to interpolate.
    """
    return (
        f"## Where You Are\n{scene_context}\n\n"
        f"## What You Do This Tick\n"
        "No direct observations — you are off-stage. Advance one "
        "objective in your own location, in a single tight beat."
    )
