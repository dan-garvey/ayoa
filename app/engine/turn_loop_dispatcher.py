"""v11-A3a: LLMDispatcher — concrete Dispatcher binding turn_loop to the
real router / agent / narrator modules.

`turn_loop.run_beat` talks to the world exclusively through the
`Dispatcher` Protocol. This module provides the production binding:
the three async methods call through to the EventRouter template,
CharacterAgent.respond(), and narrator.compose_pov_render() (per-POV
entry point), constructing their user-message blocks through the
shared helpers in `turn_loop_contracts` so prompt-code contracts stay
in lockstep.

Tests should prefer passing fakes into `run_beat` directly; this
class is what the orchestrator constructs at wire-up time.
"""

from __future__ import annotations

import logging

from app.engine import narrator as narrator_module
from app.engine.character_agent import CharacterAgent
from app.engine.context_builder import (
    append_turn_to_conversation,
    build_player_characters_block,
    clear_character_inbox,
    resolve_acting_character,
    resolve_scene_for_character,
)
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import (
    format_cat_ii_resolution_block,
    format_human_initiator_intention,
    format_npc_cascade_intention,
)
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import OpenCatIIEvent, RenderBufferEntry

logger = logging.getLogger(__name__)


# TODO: dedupe with event_router's private context helpers after v11
# stabilizes. The router-module methods are instance methods on EventRouter
# and aren't cleanly extractable without more invasive refactoring than is
# safe while sibling agents are rewiring orchestrator/narrator in parallel.
# These are straight-copied from app/engine/event_router.py.


def _build_setting_summary(checkpoint: CheckpointFile) -> str:
    setting = checkpoint.world_state.setting
    parts = []
    if setting.genre:
        parts.append(f"Genre: {setting.genre}")
    if setting.era:
        parts.append(f"Era: {setting.era}")
    if setting.tone:
        parts.append(f"Tone: {setting.tone}")
    if setting.premise:
        parts.append(f"Premise: {setting.premise}")
    return "\n".join(parts) if parts else "No setting information available."


def _build_world_rules(checkpoint: CheckpointFile) -> str:
    physics = checkpoint.world_state.physics_ruleset
    parts = [f"Strength limits: {physics.strength_limits}"]
    parts.append(f"Magic: {'enabled' if physics.magic_enabled else 'disabled'}")
    return "\n".join(parts)


def _build_scene_context(
    checkpoint: CheckpointFile, character_id: str | None = None,
) -> str:
    """Router-facing scene context block — keyed on `character_id`'s
    actual location (with importer current_scene_id fallback). Pre-r7h
    this always returned the importer's pivot scene regardless of the
    actor, so the router believed the actor was at the starting scene
    even after they (logically, narratively) moved."""
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
    `character_id`'s actual location (with importer current_scene_id
    fallback). Pre-r7h this read `current_scene_id` directly, so the
    router saw the starting-scene roster regardless of who the actor
    was or where they had moved."""
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


def _build_world_facts(checkpoint: CheckpointFile) -> str:
    facts = checkpoint.world_state.facts
    if not facts:
        return "No specific world facts recorded."
    return "\n".join(f"- {fact}" for fact in facts)


def _build_hidden_facts(checkpoint: CheckpointFile) -> str:
    facts = checkpoint.world_state.hidden_facts
    if not facts:
        return "None."
    return "\n".join(f"- {fact}" for fact in facts)


def _build_character_registry(checkpoint: CheckpointFile) -> str:
    if not checkpoint.characters:
        return "No characters in the registry."

    from app.engine.context_builder import collect_player_ids

    entries = []
    player_ids = collect_player_ids(checkpoint)
    scene_graph = checkpoint.world_state.locations.scene_graph

    for char in checkpoint.characters:
        if char.character_id in player_ids:
            continue

        location = char.location or "unknown"
        loc_name = scene_graph.get(location, {}).get("name", location)
        role = char.public_sheet.role or "unknown role"
        status = char.status.value

        entry = (
            f"- {char.name} (id: {char.character_id})\n"
            f"  Status: {status}\n"
            f"  Location: {loc_name} ({location})\n"
            f"  Role: {role}"
        )
        entries.append(entry)

    return "\n".join(entries)


def _build_since_last_turn_block(acting_char) -> str:
    """Markdown block listing arrived messages + silent observations for
    the acting character. Rendered in the router's user message; the
    router weaves any visible items into observable_facts so the narrator
    can surface them. Returns empty string when the character has nothing
    queued."""
    if acting_char is None:
        return ""
    if not acting_char.pending_observations and not acting_char.incoming_directives:
        return ""

    lines = [
        "## Arrived For You Since Last Turn",
        "Weave visible items into observable_facts so the narrator can surface them.",
    ]
    for obs in acting_char.pending_observations:
        lines.append(f"- {obs}")
    for d in acting_char.incoming_directives:
        lines.append(
            f"- **Message from {d.from_character_id}** (turn {d.turn}): {d.content}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_opening_directive(checkpoint: CheckpointFile) -> str:
    """Populate the opening-turn directive block iff this is the first
    turn and an opening_narrative exists on the checkpoint."""
    if checkpoint.session_conversation or not checkpoint.opening_narrative:
        return ""
    return (
        "## Author's Opening Scene Guidance\n"
        "This is the first turn. Read the passage below and apply "
        "rule 13: any character the opening names as present in the "
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
    ckpt: CheckpointFile, acting_character_id: str
) -> dict[str, str]:
    """Collect every context variable the event_router template needs
    aside from the two intention-block slots the caller populates
    themselves.

    Returns a dict ready to splat into `prompt_mgr.render_messages`
    after merging in `{intention_block}` and `{cat_ii_resolution_block}`.
    """
    acting_id, acting_char, acting_name = resolve_acting_character(
        ckpt, acting_character_id,
    )
    since_last_turn_block = _build_since_last_turn_block(acting_char)
    # Mirror the flush semantics from EventRouter.run — after the router
    # sees these observations/directives, they must not re-deliver on the
    # next turn.
    if acting_char is not None:
        clear_character_inbox(acting_char)

    return {
        "setting_summary": _build_setting_summary(ckpt),
        "world_lore": ckpt.world_state.lore or "No detailed lore available.",
        "world_rules": _build_world_rules(ckpt),
        "current_scene": _build_scene_context(ckpt, acting_id),
        "scene_graph": _build_scene_graph(ckpt),
        "characters_present": _build_characters_present(ckpt, acting_id),
        "world_facts": _build_world_facts(ckpt),
        "hidden_lore": ckpt.world_state.hidden_lore or "None.",
        "hidden_facts": _build_hidden_facts(ckpt),
        "character_registry": _build_character_registry(ckpt),
        "acting_character_name": acting_name,
        "acting_character_id": acting_id,
        "player_characters_block": build_player_characters_block(
            ckpt, acting_id,
        ),
        "since_last_turn_block": since_last_turn_block,
        "opening_directive": _build_opening_directive(ckpt),
        "recent_turn_recap": _build_recent_turn_recap(ckpt),
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

        ctx = _build_router_context(ckpt, actor_id)

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
        }

        # v11-r6c: event_router_v9 explicitly expects the prior router
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
        result: EventRouterOutput = response.parsed
        # Persist the user/assistant pair to the rolling session
        # conversation so next turn's router sees it as history.
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
            parenthetical but emitted no public prose (the agent_v10
            "silent beat" case, rule 8). The cascade MUST treat this as
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
