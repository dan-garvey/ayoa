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
SWEPT_RESPONDERS_SUBHEADER = "## Swept Responders (AFK)"
INTENTION_BLOCK_HEADER = "## Intention"
TICK_FAN_IN_HEADER = "## Off-Stage Tick"
ROUTER_CONTINUATION_HEADER = "## Continuation Required"

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


def format_router_continuation_block(*, prior_rationale: str = "") -> str:
    """Ask the router to repair an open beat with no continuation path.

    This is not a new character intention. It is a router-only recovery
    mode used after the prior router output kept `ends_beat=false` but
    left no dispatchable NPC pick. The next router output must either
    create a concrete human-facing beat boundary or supply a real NPC
    continuation.
    """
    lines = [
        ROUTER_CONTINUATION_HEADER,
        "",
        (
            "The beat is still open, but no NPC was selected to act next. "
            "Do not hand control back on an unresolved pause."
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
            "If this new event creates a clear human-facing affordance, set "
            "`ends_beat=true` and choose the correct `ends_beat_reason`. "
            "If the beat still needs NPC action, set `ends_beat=false` and "
            "include dispatchable NPC ids in `agent_responder_picks`."
        ),
    ]
    cleaned = (prior_rationale or "").strip()
    if cleaned:
        lines.extend(["", f"Prior diagnostic: {cleaned}"])
    return "\n".join(lines)


def format_agent_on_stage_body() -> str:
    """v11 unified-agent on-stage user-message body — intentionally empty.

    The full user message is
    `{AGENT_ON_STAGE_HEADER}\\n\\n{pending_observations_block}\\n\\n{this body}`.
    The on-stage body USED to carry three blocks (`## Scene`,
    `## What You Observe This Turn`, `## Other Characters' Responses
    This Turn`); all three are gone (v11-r10) because the same
    information already lands on the agent through other channels:

    - **`## Scene`** was a per-turn restatement of where the character
      is standing and a co-located roster. Initial location now lands
      via the importer / spawn helper, which pushes
      `[your own action] <Name> at <Location Label>.` into
      `pending_observations` once at character creation. The agent
      reads this through their inbox (the block above this one), so
      re-emitting the same facts in `## Scene` was duplicate context
      every beat.

    - **`## What You Observe This Turn`** rendered the `observed_facts`
      list every production caller passed as `[]`. Perception lands on
      the cascade NPC's `pending_observations` queue via
      `broadcast_event` (which pushes each observer's visible
      observable_facts onto their inbox when local or explicitly named
      by a mediated fact), not through this body. The block was
      rendering literal "" on every beat.

    - **`## Other Characters' Responses This Turn`** rendered the
      `prior_responses` list every production caller passed as `None`.
      Cascade NPCs see prior cascade responses through the same
      `pending_observations` channel: each cascade event broadcasts
      to its declared observers, so by the time NPC #2 fires they have NPC
      #1's just-broadcast event in their inbox. The block was
      rendering literal "" on every beat.

    The on-stage user message is now just the mode header plus the
    pending-observations block, and that's the entire payload. Kept
    as a function so dispatcher code can stay symmetric with
    `format_agent_tick_body()` / `format_agent_perception_body()` and
    so future on-stage-only context (if any ever surfaces a real
    need) has a documented home.
    """
    return ""


def format_agent_perception_body() -> str:
    """v11: agent perception-mode user-message body.

    The full user message is
    `{AGENT_PERCEPTION_HEADER}\\n\\n{pending_observations_block}\\n\\n{this body}`.
    The mode header flips the agent into Perception Mode; this body
    is fixed prose because perception has no per-turn observation
    surface — the character's identity AND current state (goals,
    objectives, secrets) live in the cached system prompt and are
    the only inputs the model needs to author its visual loadout.
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


def format_agent_tick_body(*, location_context: str) -> str:
    """v11 unified-agent off-stage tick user-message body.

    The full user message is
    `{AGENT_TICK_HEADER}\\n\\n{pending_observations_block}\\n\\n{this body}`.
    The mode header flips the agent into Tick Mode. This helper
    renders the location and the standing tick instruction (advance
    one objective in your own location, single tight beat) — kept as
    a fixed string because the off-stage tick has no per-turn
    observation surface to interpolate. The character's goals,
    objectives, and secrets sit in the cached system prompt, so
    this body just needs to point them at the location and the tick
    instruction.
    """
    return (
        f"## Where You Are\n{location_context}\n\n"
        f"## What You Do This Tick\n"
        "No direct observations — you are off-stage. Advance one "
        "objective in your own location, in a single tight beat."
    )
