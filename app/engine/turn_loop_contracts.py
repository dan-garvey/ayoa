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
ROUTER_CONTINUATION_HEADER = "## Continuation Required"

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


def format_human_initiator_intention(character_id: str, user_input: str) -> str:
    """Cat I / Cat II OPEN path: a human player's /act.

    The router user tail already identifies the acting character by id.
    Keep the input as the player's raw intention so we do not duplicate
    actor labels or add "attempts:"/markdown framing to every call.
    """
    del character_id
    return (user_input or "").strip()


def format_npc_cascade_intention(character_id: str, intention: str) -> str:
    """NPC cascade step in a beat (not a fresh player action)."""
    del character_id
    return (intention or "").strip()


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
        "requires_responders=false and event_kind=cat_ii_resolution."
    )
    return "\n".join(lines)


def format_agent_output_entry(character_id: str, public_text: str) -> str:
    """Router input for one committed agent output.

    Runtime dispatch metadata such as frame and source event id shapes the
    agent call before this point; the router only needs the authoring
    character id and public surface text.
    """
    text = (public_text or "").strip() or "(no public action)"
    return f"{character_id}: {text}"


def format_router_continuation_block(*, prior_rationale: str = "") -> str:
    """Ask the router to repair an open beat with no continuation path.

    This is not a new character intention. It is a router-only recovery
    mode used after the prior router output used `event_kind=beat_continues` but
    left no dispatchable next-output target. The next router output must either
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
            "the correct terminal `event_kind`. "
            "If the beat still needs NPC action, set `event_kind=beat_continues` and "
            "mark dispatchable NPC observers with `routing_role=next_output`."
        ),
    ]
    cleaned = (prior_rationale or "").strip()
    if cleaned:
        lines.extend(["", f"Prior diagnostic: {cleaned}"])
    return "\n".join(lines)


def format_agent_on_stage_body() -> str:
    """v11 unified-agent on-stage user-message body — intentionally empty.

    The full user message is
    `{AGENT_TURN_HEADER}\\n\\n{pending_observations_block}\\n\\n{this body}`.
    The on-stage body USED to carry three blocks (`## Scene`,
    `## What You Observe This Turn`, `## Other Characters' Responses
    This Turn`); all three are gone (v11-r10) because the same
    information already lands on the agent through other channels:

    - **`## Scene`** was a per-turn restatement of where the character
      is standing and a co-located roster. Initial location now lands
      via the story seed / spawn helper, which pushes
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
    `format_agent_turn_body()` / `format_agent_perception_body()` and
    so future foreground-only context (if any ever surfaces a real
    need) has a documented home.
    """
    return ""


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
