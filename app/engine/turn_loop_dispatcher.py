"""v11-A3a: LLMDispatcher — concrete Dispatcher binding turn_loop to the
real router / agent / narrator modules.

`turn_loop.run_beat` talks to the world exclusively through the
`Dispatcher` Protocol. This module provides the production binding:
the three async methods call through to the `event_router` prompt
template, CharacterAgent.respond(), and
narrator.compose_pov_render() (per-POV entry point), constructing
their user-message blocks through the shared helpers in
`turn_loop_contracts` so prompt-code contracts stay in lockstep.

The legacy `EventRouter` engine class that wrapped this same prompt
template was murdered in v11-r7j; this dispatcher is the only
production caller of the `event_router` template now.

Tests should prefer passing fakes into `run_beat` directly; this
class is what the orchestrator constructs at wire-up time.
"""

from __future__ import annotations

import asyncio
import logging

from app.engine import narrator as narrator_module
from app.engine.character_agent import CharacterAgent
from app.engine.context_builder import (
    append_turn_to_conversation,
    build_player_characters_block,
    build_setting_summary,
    clear_character_inbox,
    resolve_acting_character,
    resolve_scene_for_character,
)
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import (
    format_cat_ii_resolution_block,
    format_human_initiator_intention,
    format_npc_cascade_intention,
    format_tick_fan_in_block,
)
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import OpenCatIIEvent, RenderBufferEntry

logger = logging.getLogger(__name__)


# v11-r7j note: a mirror copy of the legacy `EventRouter` engine
# class's private helpers used to live in app/engine/event_router.py
# and these were straight-ported during the v11 transition. The
# legacy class was murdered in v11-r7j; this dispatcher is now the
# only home for these helpers and the duplication concern is gone.


# `build_setting_summary` is now in `app/engine/context_builder.py`
# and imported at the top of this module — pre-v11-r7j three near-
# identical copies of the same helper lived in this module, narrator,
# and engine_bridge.


def _build_world_rules(checkpoint: CheckpointFile) -> str:
    physics = checkpoint.world_state.physics_ruleset
    parts = [f"Strength limits: {physics.strength_limits}"]
    parts.append(f"Magic: {'enabled' if physics.magic_enabled else 'disabled'}")
    return "\n".join(parts)


def _build_scene_context(
    checkpoint: CheckpointFile, character_id: str | None = None,
) -> str:
    """Router-facing scene context block — keyed on `character_id`'s
    actual location read from the roster. Returns an unsited block when
    no character_id is supplied or the character has no location set.
    """
    locations = checkpoint.world_state.locations
    scene_id = resolve_scene_for_character(checkpoint, character_id)
    if not scene_id:
        return "No scene information available."

    scene = locations.scene_graph.get(scene_id, {})
    if not scene:
        return f"Current location: {scene_id}"

    name = scene.get("name", scene_id)
    desc = scene.get("description", "")
    connected = scene.get("connected_to", [])

    parts = [f"Current location: {name} (id: {scene_id})"]
    if desc:
        parts.append(f"Description: {desc}")
    if connected:
        conn_details = []
        for conn_id in connected:
            conn_scene = locations.scene_graph.get(conn_id, {})
            conn_name = conn_scene.get("name", conn_id)
            conn_details.append(f"{conn_name} ({conn_id})")
        parts.append(f"Connected to: {', '.join(conn_details)}")

    return "\n".join(parts)


def _build_scene_graph(checkpoint: CheckpointFile) -> str:
    scene_graph = checkpoint.world_state.locations.scene_graph
    if not scene_graph:
        return "No scene graph available."

    entries = []
    for scene_id, scene in scene_graph.items():
        name = scene.get("name", scene_id)
        desc = scene.get("description", "")
        connected = scene.get("connected_to", [])
        parts = [f"- {name} (id: {scene_id})"]
        if desc:
            parts.append(f"  Description: {desc}")
        if connected:
            parts.append(f"  Connected to: {', '.join(connected)}")
        entries.append("\n".join(parts))

    return "\n".join(entries)


def _build_characters_present(
    checkpoint: CheckpointFile, character_id: str | None = None,
) -> str:
    """Router-facing "characters present" block — keyed on
    `character_id`'s actual location (from the roster). Returns an
    unsited block when no character_id is supplied.
    """
    scene_id = resolve_scene_for_character(checkpoint, character_id)
    if not scene_id:
        return "No other characters are present in this scene."
    present = []
    for char in checkpoint.characters:
        if char.location == scene_id and char.status.value == "active":
            role = char.public_sheet.role or "unknown role"
            present.append(f"- {char.name} ({role})")

    if not present:
        return "No other characters are present in this scene."
    return "\n".join(present)


def _build_world_facts_delta(checkpoint: CheckpointFile) -> str:
    """Render only world facts the router hasn't seen on a prior turn.

    Pre-Commit-3 the per-turn user message carried the full
    `world_state.facts` list every turn (~tens to hundreds of tokens).
    The list is essentially immutable post-import — re-feeding it
    every turn was pure context bloat. Now we surface each fact ONCE,
    on whichever turn it was added (turn 1 for importer-seeded facts,
    later for any rare runtime additions), then track surfacing in
    `session.surfaced_world_facts`.

    Returns "" when nothing is new (the most common case from turn 2
    onward), in which case the template's ## World Facts header
    suppresses entirely. MUTATES `session.surfaced_world_facts` to
    include any newly-surfaced entries — this is intentional and
    atomic with the router LLM call (the caller persists the
    checkpoint after `route_intention` returns).

    Returns the full block including header when non-empty, so the
    template can splice with `{world_facts_delta_block}` and a missing
    delta cleanly disappears with no dangling header.
    """
    facts = checkpoint.world_state.facts or []
    surfaced = set(checkpoint.session.surfaced_world_facts)
    delta = [f for f in facts if f not in surfaced]
    if not delta:
        return ""
    checkpoint.session.surfaced_world_facts = list(surfaced.union(delta))
    body = "\n".join(f"- {fact}" for fact in delta)
    return (
        "## World Facts (Common Knowledge — new entries)\n"
        f"{body}\n\n"
    )


def _build_initial_roster_block(checkpoint: CheckpointFile) -> str:
    """Render the full NPC roster + interior on turn 1 only.

    Pre-Commit-3 the per-turn user message carried `character_registry`
    EVERY turn — name + role + status + location for every non-player
    character (~80 tokens × N NPCs, ~1000 tokens/turn for hollowstone).
    The router has its own session conversation history, so re-feeding
    identity every turn was duplication: turn-1's inject + every
    subsequent `roster_moves` and `spawn` outcome the router itself
    authored is already in history.

    This block lands ONCE — on turn 1 only — and carries richer signal
    than the dropped registry: name, id, role, location, goals,
    current_objectives. The router uses it to seed "who's in this
    world and what are they trying to do" so picking decisions on turn
    1 aren't blind. Returns "" on every turn after the first.

    NOTE: this block does NOT carry per-NPC interior beyond
    importer-seeded goals/objectives. An agent's freshest interior
    (the trailing parenthetical from its last `respond()` / `tick()`)
    lives in that agent's own rolling history and is deliberately
    NOT mirrored to the router — the router decides who acts based on
    public signals (cascade intentions, prior canonical events) +
    seeded objectives, not stolen interior.
    """
    if checkpoint.session_conversation:
        return ""
    if not checkpoint.characters:
        return ""

    from app.engine.context_builder import collect_player_ids
    player_ids = collect_player_ids(checkpoint)
    scene_graph = checkpoint.world_state.locations.scene_graph

    entries: list[str] = []
    for char in checkpoint.characters:
        if char.character_id in player_ids:
            continue
        if char.status.value != "active":
            continue

        location = char.location or "unknown"
        loc_name = scene_graph.get(location, {}).get("name", location)
        role = char.public_sheet.role or "unknown role"

        parts = [
            f"- {char.name} (id: {char.character_id})",
            f"  Role: {role}",
            f"  Location: {loc_name} ({location})",
        ]
        goals = [g for g in (char.private_state.goals or []) if g]
        if goals:
            parts.append(
                "  Goals (long-term): " + "; ".join(goals)
            )
        objs = [o for o in (char.private_state.current_objectives or []) if o]
        if objs:
            parts.append(
                "  Current objectives (active pursuits): " + "; ".join(objs)
            )
        entries.append("\n".join(parts))

    if not entries:
        return ""

    header = (
        "## Initial Roster\n"
        "Every active NPC in this world, with their long-term goals "
        "and active pursuits. See the system prompt's \"Character "
        "interior\" section for how to weight these signals when "
        "picking responders.\n\n"
    )
    return header + "\n\n".join(entries) + "\n"


def _build_state_changes_block(checkpoint: CheckpointFile) -> str:
    """Drain the queue of engine-applied changes the router didn't author.

    Pre-Commit-3 there was no dedicated channel for "the engine did
    something the router needs to know about" — spawn outcomes were
    invisible to the router (it learned about new characters only when
    they showed up in `character_registry` on a later turn), and
    /takeover / /join changes never reached the router at all.

    This block is non-empty whenever `pending_router_state_changes`
    has entries. The queue is populated by:
      - Spawn flow (Commit 4): the LLM-generated `router_summary` for
        each new character lands as one line per spawn.
      - /takeover and /join: a one-line note when a player binds to or
        un-binds from a character.
      - Operator/exotic mutations: anything outside the router's
        own decision loop that flips world state.

    DRAINS the queue as a side effect — atomic with the router LLM
    call (the caller persists the checkpoint after `route_intention`
    returns).
    """
    queued = checkpoint.session.pending_router_state_changes or []
    if not queued:
        return ""
    checkpoint.session.pending_router_state_changes = []
    body = "\n".join(f"- {entry}" for entry in queued)
    return (
        "## State Changes Since Your Last Call\n"
        "These are things the engine applied that you did NOT author "
        "in a prior turn (spawn LLM outcomes, player /takeover or "
        "/join, exotic state mutations). Folded into your model of "
        "the world for this turn's adjudication.\n\n"
        f"{body}\n"
    )


def _build_hidden_facts(checkpoint: CheckpointFile) -> str:
    facts = checkpoint.world_state.hidden_facts
    if not facts:
        return "None."
    return "\n".join(f"- {fact}" for fact in facts)


def _build_since_last_turn_block(acting_char) -> str:
    """Markdown block listing silent observations for the acting
    character. Rendered in the router's user message; the router
    weaves any visible items into observable_facts so the narrator
    can surface them. Returns empty string when the character has
    nothing queued.

    Pre-Commit-2 this also rendered `incoming_directives`, a
    structured inter-agent message bus. Directives are gone;
    cross-character communication now flows through normal scene prose
    (a courier walks in and speaks) plus the v11-r7j off-scene
    perception inbox populated by `broadcast_event`.
    """
    if acting_char is None:
        return ""
    if not acting_char.pending_observations:
        return ""

    lines = [
        "## Arrived For You Since Last Turn",
        "Weave visible items into observable_facts so the narrator can surface them.",
    ]
    for obs in acting_char.pending_observations:
        lines.append(f"- {obs}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_opening_directive(
    checkpoint: CheckpointFile, user_input: str = "",
) -> str:
    """Populate the opening-turn directive block iff this is the first
    turn AND the player's input is the `(begin)` OOC directive AND an
    `opening_narrative` exists on the checkpoint.

    Why all three gates:
    - `session_conversation` empty: never re-inject the opening on
      later turns (covered by the original implementation).
    - `user_input == "(begin)"`: a fresh session can ALSO be entered
      via `(arrive)` (a player joining an empty CLI session, or any
      future flow that opens a session at a non-canonical entry).
      The router prompt's `(arrive)` instructions tell the model to
      ignore the opening narrative and place the character in a
      sensible existing scene; injecting the opening guidance here
      would directly contradict that and re-rail the placement.
    - `opening_narrative` non-empty: nothing to inject if the
      importer didn't author one.

    The empty-string default for `user_input` keeps existing internal
    callers conservative (the tick path explicitly threads an empty
    string; we don't want a forgotten arg path to accidentally fire
    the opening block on an off-stage tick that happened to land on
    turn 1).
    """
    if checkpoint.session_conversation or not checkpoint.opening_narrative:
        return ""
    if user_input.strip() != "(begin)":
        return ""
    return (
        "## Author's Opening Scene Guidance\n"
        "This is the first turn. Read the passage below and apply "
        "rule 14: any character the opening names as present in the "
        "starting scene must be placed via `roster_moves` or `spawn` "
        "and listed in `observers` with priority >= 3 so their agent "
        "produces dialogue. The narrator will NOT transcribe dialogue "
        "from this prose — only you can make characters speak this "
        "turn, by placing them here.\n\n"
        f"{checkpoint.opening_narrative}\n\n"
    )


def _build_recent_turn_recap(checkpoint: CheckpointFile) -> str:
    recap_note = checkpoint.session.pending_recap
    if not recap_note:
        return ""
    return (
        "## Previous Turn — Narrator Delta Note\n"
        "A terse summary of state-level beats the narrator rendered "
        "last turn that aren't already in your prior `canonical_event` "
        "(environmental changes, completed NPC actions, implicit "
        "movement, objects placed). Factor these into this turn's "
        "adjudication.\n\n"
        f"- {recap_note}\n\n"
    )


def _build_router_context(
    ckpt: CheckpointFile,
    acting_character_id: str,
    user_input: str = "",
) -> dict[str, str]:
    """Collect every context variable the event_router template needs
    aside from the two intention-block slots the caller populates
    themselves.

    `user_input` is forwarded to `_build_opening_directive` so the
    opening-narrative author block fires only on a `(begin)` start —
    `(arrive)` on an otherwise-pristine session must NOT inherit the
    opening guidance (the router prompt instructs the model to place
    the character in a sensible existing scene instead). Tick-mode
    callers pass an empty string explicitly.

    Returns a dict ready to splat into `prompt_mgr.render_messages`
    after merging in `{intention_block}` and `{cat_ii_resolution_block}`.
    """
    acting_id, acting_char, acting_name = resolve_acting_character(
        ckpt, acting_character_id,
    )
    since_last_turn_block = _build_since_last_turn_block(acting_char)
    # After the router sees these silent observations, they must not
    # re-deliver on the next turn — drain the inbox here.
    if acting_char is not None:
        clear_character_inbox(acting_char)

    return {
        "setting_summary": build_setting_summary(ckpt),
        "world_lore": ckpt.world_state.lore or "No detailed lore available.",
        "world_rules": _build_world_rules(ckpt),
        "current_scene": _build_scene_context(ckpt, acting_id),
        "scene_graph": _build_scene_graph(ckpt),
        "characters_present": _build_characters_present(ckpt, acting_id),
        "hidden_lore": ckpt.world_state.hidden_lore or "None.",
        "hidden_facts": _build_hidden_facts(ckpt),
        "acting_character_name": acting_name,
        "acting_character_id": acting_id,
        "player_characters_block": build_player_characters_block(
            ckpt, acting_id,
        ),
        "since_last_turn_block": since_last_turn_block,
        "opening_directive": _build_opening_directive(ckpt, user_input),
        "recent_turn_recap": _build_recent_turn_recap(ckpt),
        "world_facts_delta_block": _build_world_facts_delta(ckpt),
        "initial_roster_block": _build_initial_roster_block(ckpt),
        "state_changes_block": _build_state_changes_block(ckpt),
    }


class LLMDispatcher:
    """Production Dispatcher implementation — binds `turn_loop.run_beat`
    to the real router / agent / narrator modules."""

    def __init__(self, client: LLMClient, prompt_mgr: PromptManager):
        self.client = client
        self.prompt_mgr = prompt_mgr
        # Character agent is stateless aside from `last_usage`; reusing one
        # instance avoids per-call allocation.
        self._agent = CharacterAgent(client, prompt_mgr)

    # ------------------------------------------------------------------
    # route_intention
    # ------------------------------------------------------------------

    async def route_intention(
        self,
        *,
        ckpt: CheckpointFile,
        actor_id: str,
        intention: str,
        scene_id: str,
        cat_ii_event: OpenCatIIEvent | None = None,
    ) -> EventRouterOutput:
        """Classify + adjudicate one intention through event_router."""
        del scene_id  # Scene id is picked up from ckpt.world_state.locations.

        # Snapshot the two delta-queue session fields BEFORE
        # `_build_router_context` mutates them, so we can restore on
        # exception and avoid silently losing queued state-change /
        # world-fact entries on a failed router call. Flagged by the
        # Commit-4 reviewers (same pattern the now-deleted legacy
        # EventRouter.run had to handle).
        saved_surfaced_facts = list(ckpt.session.surfaced_world_facts)
        saved_state_changes = list(
            ckpt.session.pending_router_state_changes
        )
        try:
            ctx = _build_router_context(ckpt, actor_id, user_input=intention)

            # Resolve the actor's display name for the intention framing.
            actor_char = next(
                (c for c in ckpt.characters if c.character_id == actor_id),
                None,
            )
            actor_name = actor_char.name if actor_char else actor_id

            if cat_ii_event is None:
                bindings = ckpt.session.character_bindings or {}
                if actor_id in bindings:
                    intention_block = format_human_initiator_intention(
                        actor_name, intention,
                    )
                else:
                    intention_block = format_npc_cascade_intention(
                        actor_name, intention,
                    )
                cat_ii_resolution_block = ""
            else:
                evt = cat_ii_event
                responders: list[tuple[str, str]] = [
                    (rid, evt.collected_intentions[rid])
                    for rid in evt.required_responders
                    if rid in evt.collected_intentions
                ]
                cat_ii_resolution_block = format_cat_ii_resolution_block(
                    initiator_id=evt.initiator_id,
                    initiator_intention=evt.initiator_intention,
                    responders=responders,
                    swept_responders=list(evt.swept_responders),
                )
                intention_block = ""

            template_vars = {
                **ctx,
                "intention_block": intention_block,
                "cat_ii_resolution_block": cat_ii_resolution_block,
                "tick_fan_in_block": "",
            }

            # v11-r6c: event_router explicitly expects the prior router
            # exchanges as conversation history ("The prior messages in this
            # conversation are the full session history"). Use
            # render_conversation so the rolling history rides along, and
            # append this turn's exchange after the call so continuity
            # compounds across turns. The caller (Orchestrator) persists the
            # checkpoint after run_beat returns.
            messages = self.prompt_mgr.render_conversation(
                "event_router",
                history=ckpt.session_conversation,
                **template_vars,
            )

            # Plain-text user content captured before LLMClient wraps with
            # cache_control for this call.
            user_content = messages[-1]["content"]

            logger.info(
                "LLMDispatcher.route_intention: actor=%s cat_ii=%s",
                actor_id, cat_ii_event.event_id if cat_ii_event else None,
            )

            response = await self.client.complete(
                role="event_router",
                messages=messages,
                response_model=EventRouterOutput,
                temperature=0.35,
                max_tokens=5000,
                cache=True,
                compact=True,
            )
        except Exception:
            ckpt.session.surfaced_world_facts = saved_surfaced_facts
            ckpt.session.pending_router_state_changes = saved_state_changes
            raise

        result: EventRouterOutput = response.parsed
        # Persist the user/assistant pair to the rolling session
        # conversation so next turn's router sees it as history.
        append_turn_to_conversation(
            ckpt.session_conversation, user_content, response,
        )
        return result

    # ------------------------------------------------------------------
    # route_tick_intentions  (Commit 6: off-stage tick fan-in)
    # ------------------------------------------------------------------

    async def route_tick_intentions(
        self,
        *,
        ckpt: CheckpointFile,
        tick_outputs: list[tuple[str, str, str, str]],
        acting_character_id: str = "",
    ) -> EventRouterOutput | None:
        """Bundle off-stage agent prose into a single unified-router call.

        Each entry in `tick_outputs` is `(name, character_id, location,
        public_text)` — the parenthetical (private intent) MUST already
        have been stripped upstream by the caller (the orchestrator's
        `_run_ticks` pulls `output.public_text`, never `output.intent`).
        That stripping is the load-bearing information-asymmetry guard:
        the router never sees an agent's interior.

        On empty `tick_outputs`, returns None without any LLM call. On
        any non-empty list, makes ONE router call in tick mode (signaled
        by the `## Off-Stage Tick` user-message header) and returns the
        EventRouterOutput. Caller is responsible for applying
        roster_moves / scenes_created / canonical_event to ckpt; this
        method only handles the LLM call and conversation-history append.

        `acting_character_id` is the player who acted on the on-stage
        beat preceding this tick; used to resolve display name/scene
        framing for the shared router context. The tick itself is not
        attributed to that player — it just has to come from somewhere
        for the existing context-builder helpers (which expect an
        actor frame) to render cleanly.

        Snapshot/restore of the two delta-queue session fields mirrors
        `route_intention` so a failed tick router call doesn't silently
        drain queued state-change / world-fact entries.
        """
        if not tick_outputs:
            return None

        saved_surfaced_facts = list(ckpt.session.surfaced_world_facts)
        saved_state_changes = list(
            ckpt.session.pending_router_state_changes
        )
        try:
            # Tick mode is always a non-(begin) turn — the on-stage
            # beat that triggers it has already happened — so explicitly
            # pass an empty user_input to suppress any first-turn
            # opening directive even on edge cases.
            ctx = _build_router_context(
                ckpt, acting_character_id, user_input="",
            )

            tick_block = format_tick_fan_in_block(tick_outputs)

            template_vars = {
                **ctx,
                # Tick mode owns ONE of the three input-block slots; the
                # other two stay empty. Same exclusivity contract as
                # intention vs cat_ii_resolution on the on-stage path.
                "intention_block": "",
                "cat_ii_resolution_block": "",
                "tick_fan_in_block": tick_block,
            }

            messages = self.prompt_mgr.render_conversation(
                "event_router",
                history=ckpt.session_conversation,
                **template_vars,
            )

            user_content = messages[-1]["content"]

            logger.info(
                "LLMDispatcher.route_tick_intentions: %d off-stage "
                "agents bundled into one router call",
                len(tick_outputs),
            )

            response = await self.client.complete(
                role="event_router",
                messages=messages,
                response_model=EventRouterOutput,
                temperature=0.35,
                max_tokens=5000,
                cache=True,
                compact=True,
            )
        except Exception:
            ckpt.session.surfaced_world_facts = saved_surfaced_facts
            ckpt.session.pending_router_state_changes = saved_state_changes
            raise

        result: EventRouterOutput = response.parsed
        append_turn_to_conversation(
            ckpt.session_conversation, user_content, response,
        )
        return result

    # ------------------------------------------------------------------
    # agent_intend
    # ------------------------------------------------------------------

    async def agent_intend(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        scene_id: str,
    ) -> str:
        """Invoke the character agent and return its prose for the router.

        The agent's prose IS the intention — no serialization layer
        needed. The trailing parenthetical (private intent) is stripped
        at parse time; what we return here is the public surface only,
        which is what the router framing wants ("`{name}` intends: ...").

        Three result shapes the caller (run_beat) must distinguish:
          - **non-empty prose** → real intention, route normally.
          - **`"(remains silent)"`** → the agent had a non-empty intent
            parenthetical but emitted no public prose (the agent
            prompt's "Sparse is valid" shared rule — paren-only
            output is in-character). The cascade MUST treat this as
            a real beat and route it; otherwise the prompt's promise
            that silence is a valid in-character choice gets quietly
            broken by `_is_agent_refusal` collapsing it to
            `cascade_exhausted`.
          - **`""`** → true refusal: no public prose AND no intent (or
            the parser logged a "missing trailing parenthetical"
            warning and we have nothing to route). The cascade ends.
        """
        del scene_id  # Agent pulls scene from the character's own location.

        character = next(
            (c for c in ckpt.characters if c.character_id == character_id),
            None,
        )
        if character is None:
            logger.warning(
                "agent_intend: unknown character_id %s", character_id,
            )
            return ""

        output = await self._agent.respond(
            character=character,
            observed_facts=[],
            checkpoint=ckpt,
            prior_responses=None,
            acting_character_id=character_id,
        )
        public = output.public_text.strip()
        if public:
            return public
        if output.intent.strip():
            # Agent chose deliberate silence (paren-only output). Surface
            # a fixed sentinel so the router can adjudicate a "watches
            # without speaking" beat instead of the cascade dying. The
            # sentinel is intentionally short, parenthesized, and
            # identical every time so the router can recognize it.
            logger.info(
                "Agent %s emitted silent beat (intent=%d chars); "
                "routing via sentinel.",
                character.name, len(output.intent),
            )
            return "(remains silent)"
        return ""

    # ------------------------------------------------------------------
    # harvest_perceptions  (v11-r8a: observation_harvest fork)
    # ------------------------------------------------------------------

    async def harvest_perceptions(
        self,
        *,
        ckpt: CheckpointFile,
        character_ids: list[str],
        acting_character_id: str,
    ) -> list[str]:
        """Fan out CharacterAgent.perceive() across `character_ids` in
        parallel.

        Returns one string per id in input order — empty string for any
        character whose perception call failed (unknown id, LLM error,
        empty output). The harvest fork in `run_beat` filters empties
        and appends the non-empty fragments to the canonical event's
        `observable_facts` block.

        Per-character exceptions are absorbed locally rather than
        bubbled. The harvest is a UX enrichment, not the beat's
        critical path; one failed perception out of three should
        leave the other two on the player's screen instead of taking
        the whole render down. The caller logs dropped fragments at
        WARN so test playthroughs still surface the failure.

        Cache lineage: each character's `perceive` call shares the
        SAME system prompt as that character's `respond` / `tick`
        calls (single unified `agent` template). The system prompt
        cache hits across modes for the same character; only the
        per-call user message changes. Parallel fan-out compounds
        well with this — a 3-character harvest bills three Haiku
        calls in ~1 round-trip wall time, all hitting the cached
        system prefix.
        """
        if not character_ids:
            return []

        by_id = {c.character_id: c for c in ckpt.characters}

        async def _one(cid: str) -> str:
            character = by_id.get(cid)
            if character is None:
                logger.warning(
                    "harvest_perceptions: unknown character_id %s", cid,
                )
                return ""
            try:
                return await self._agent.perceive(
                    character=character,
                    checkpoint=ckpt,
                    acting_character_id=acting_character_id,
                )
            except Exception as exc:  # noqa: BLE001 — see docstring
                logger.warning(
                    "harvest_perceptions: perceive() failed for %s: %s",
                    cid, exc,
                )
                return ""

        logger.info(
            "harvest_perceptions: firing %d parallel perceive calls",
            len(character_ids),
        )
        return list(await asyncio.gather(*(_one(c) for c in character_ids)))

    # ------------------------------------------------------------------
    # narrator_compose
    # ------------------------------------------------------------------

    async def narrator_compose(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        buffered_events: list[RenderBufferEntry],
        partial_mode_override: bool | None = None,
        user_input: str = "",
    ) -> tuple[NarratorFinalOutput, "TranscriptEntry"]:
        """Render per-POV prose via narrator.compose_pov_render.

        Returns `(NarratorFinalOutput, TranscriptEntry)` so run_beat
        can populate ckpt.transcript via the parallel
        BeatResult.transcript_entries map. The transcript entry is
        constructed engine-side from `user_input` (the real player
        utterance for the acting POV; "" for incidental POVs in a
        multi-human beat) and the rendered prose. Pre-r7j the LLM
        owned the transcript entry and emitted `"{name} — "` for the
        user field every time.

        `partial_mode` defaults to True iff this character is currently
        pinned as a Cat II responder in any scene — the narrator renders
        a partial view because the beat still has outstanding resolution
        work. `partial_mode_override`, when not None, wins over the slot
        scan (v11-r6a: Cat II-open render path sets this True for the
        initiator + pinned humans so they see the mid-attempt cliffhanger
        even though the initiator isn't pinned themselves).
        """
        if partial_mode_override is not None:
            partial_mode = partial_mode_override
        else:
            partial_mode = _is_pinned_as_cat_ii_responder(ckpt, character_id)

        envelope, entry = await narrator_module.compose_pov_render(
            client=self.client,
            prompt_mgr=self.prompt_mgr,
            ckpt=ckpt,
            pov_character_id=character_id,
            buffered_events=buffered_events,
            partial_mode=partial_mode,
            user_input=user_input,
        )
        return envelope, entry


def _is_pinned_as_cat_ii_responder(
    ckpt: CheckpointFile, character_id: str,
) -> bool:
    """True iff `character_id` has a slot entry with reason
    `cat_ii_responder` in ANY scene."""
    for slot in ckpt.session.active_act_slots.values():
        entry = slot.get(character_id)
        if entry is not None and entry.reason == "cat_ii_responder":
            return True
    return False
