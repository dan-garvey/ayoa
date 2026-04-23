"""Merged adjudication + perception routing — single LLM call per turn.

EventRouter maintains a session-wide rolling conversation on the checkpoint.
Each call sees every prior turn's routing decisions as real conversational
history, which lets it remember commitments, who's where, and unfinished
threads across the entire session.
"""

from __future__ import annotations

import logging
import time

from app.engine.prompt_manager import PromptManager
from app.engine.context_builder import (
    append_turn_to_conversation,
    clear_character_inbox,
)
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput

logger = logging.getLogger(__name__)


def _build_since_last_turn_block(acting_char) -> str:
    """Markdown block listing silent observations for the acting
    character. Rendered in the router's user message; the router
    weaves any visible items into observable_facts so the narrator can
    surface them (a note on the desk, a smell drifting in, a distant
    sound). Returns empty string when nothing is queued — the template
    then renders cleanly without a dangling header.

    Pre-Commit-2 this also rendered `incoming_directives`, a structured
    inter-agent message bus. Directives are gone; cross-character
    communication now flows through normal scene prose (a courier
    walks in and speaks).
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


class EventRouter:
    """Single-pass event adjudication + perception routing over a rolling session history."""

    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager
        # Usage from the most recent run() call, for debug introspection.
        self.last_usage: dict[str, int] = {}

    async def run(
        self,
        user_input: str,
        checkpoint: CheckpointFile,
        acting_character_id: str = "",
    ) -> EventRouterOutput:
        """Return both canonical event data and observer routing.

        Appends the current turn's user message and the assistant response (as
        raw content blocks, preserving any compaction blocks) to
        `checkpoint.session_conversation`.
        """
        from app.engine.context_builder import (
            build_player_characters_block,
            resolve_acting_character,
        )

        acting_id, acting_char, acting_name = resolve_acting_character(
            checkpoint, acting_character_id,
        )

        # Flush any silent observations that stacked for this player
        # character between their last action and now (off-stage ticks,
        # scene pushes, environmental beats). The router gets them as
        # a distinct context block so it can fold them into the
        # canonical event (e.g., "Johnny notices the note on his desk
        # as he enters"). Cross-character messages no longer flow
        # through a structured queue — couriers walk into the scene.
        since_last_turn_block = _build_since_last_turn_block(acting_char)
        if acting_char is not None:
            clear_character_inbox(acting_char)

        # Opening-turn hand-off: on the very first turn of the session
        # (session_conversation empty), surface the author's opening prose
        # to the router as population-guidance. It's the router's job to
        # place characters the opening describes — the narrator will be
        # told to use the opening for tone only and to render dialogue
        # exclusively from agent output, so the router has to put the
        # characters in scene or they won't speak at all.
        opening_directive = ""
        if not checkpoint.session_conversation and checkpoint.opening_narrative:
            opening_directive = (
                "## Author's Opening Scene Guidance\n"
                "This is the first turn. Read the passage below and apply "
                "rule 14: any character the opening names as present in the "
                "starting scene must be placed via `roster_moves` or `spawn` "
                "and listed in `observers` with priority ≥ 3 so their agent "
                "produces dialogue. The narrator will NOT transcribe dialogue "
                "from this prose — only you can make characters speak this "
                "turn, by placing them here.\n\n"
                f"{checkpoint.opening_narrative}\n\n"
            )

        # Recap of the previous turn's narrator prose (a terse Haiku-
        # generated delta note). Closes the narrator-context gap: the
        # router normally never sees the narrator's final prose, so any
        # state-level beats the narrator rendered beyond the prior
        # canonical_event are invisible to it. This one-slot buffer
        # carries those beats forward by one turn.
        recent_turn_recap = ""
        recap_note = checkpoint.session.pending_recap
        if recap_note:
            recent_turn_recap = (
                "## Previous Turn — Narrator Delta Note\n"
                "A terse summary of state-level beats the narrator rendered "
                "last turn that aren't already in your prior `canonical_event` "
                "(environmental changes, completed NPC actions, implicit "
                "movement, objects placed). Factor these into this turn's "
                "adjudication.\n\n"
                f"- {recap_note}\n\n"
            )

        # Commit-3: drop `character_registry` and `world_facts` (full)
        # from per-turn context. Replace with `initial_roster_block`
        # (turn-1 only, with goals/objectives/last_intent),
        # `world_facts_delta` (only newly-surfaced facts), and
        # `state_changes_block` (engine-applied changes the router
        # didn't author). The two delta builders MUTATE session state
        # in-place during render: `_build_world_facts_delta` extends
        # `session.surfaced_world_facts`, `_build_state_changes_block`
        # drains `session.pending_router_state_changes`. If the LLM
        # call fails after build but before completion, those queued
        # entries are silently lost — flagged P0 by the Commit-4
        # reviewers. Snapshot the two fields BEFORE render and
        # restore them on exception so a failed router call leaves
        # the next attempt with the same input it had on this one.
        from app.engine.turn_loop_dispatcher import (
            _build_initial_roster_block,
            _build_state_changes_block,
            _build_world_facts_delta,
        )
        saved_surfaced_facts = list(
            checkpoint.session.surfaced_world_facts
        )
        saved_state_changes = list(
            checkpoint.session.pending_router_state_changes
        )
        render_t0 = time.monotonic()
        try:
            messages = self.prompt_manager.render_conversation(
                "event_router",
                history=checkpoint.session_conversation,
                setting_summary=self._build_setting_summary(checkpoint),
                world_lore=checkpoint.world_state.lore or "No detailed lore available.",
                world_rules=self._build_world_rules(checkpoint),
                current_scene=self._build_scene_context(checkpoint),
                scene_graph=self._build_scene_graph(checkpoint),
                characters_present=self._build_characters_present(checkpoint),
                hidden_lore=checkpoint.world_state.hidden_lore or "None.",
                hidden_facts=self._build_hidden_facts(checkpoint),
                user_input=user_input,
                acting_character_name=acting_name,
                acting_character_id=acting_id,
                player_characters_block=build_player_characters_block(
                    checkpoint, acting_id
                ),
                since_last_turn_block=since_last_turn_block,
                opening_directive=opening_directive,
                recent_turn_recap=recent_turn_recap,
                world_facts_delta_block=_build_world_facts_delta(checkpoint),
                initial_roster_block=_build_initial_roster_block(checkpoint),
                state_changes_block=_build_state_changes_block(checkpoint),
            )
            render_ms = (time.monotonic() - render_t0) * 1000

            # Capture the plain-text user content before LLMClient wraps it with
            # cache_control for this call — we persist the plain text.
            user_content = messages[-1]["content"]

            logger.info("EventRouter: adjudicating + routing action: %s", user_input[:80])

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
            # Restore pre-build state so the next attempt re-renders with
            # the same delta queues. Without this, a failed call would
            # leave both queues empty and the retry would silently lose
            # whatever was queued (spawn router_summary lines, world fact
            # entries, etc.).
            checkpoint.session.surfaced_world_facts = saved_surfaced_facts
            checkpoint.session.pending_router_state_changes = (
                saved_state_changes
            )
            raise
        result: EventRouterOutput = response.parsed
        self.last_usage = {**response.usage, "prompt_render_ms": render_ms}

        # Persist the user/assistant pair to the rolling session conversation.
        append_turn_to_conversation(
            checkpoint.session_conversation, user_content, response,
        )
        # Clear the recap buffer now that its content is archived in the
        # just-appended user message. Next turn's router will see it in
        # session_conversation history, and a fresh recap will be
        # generated at end of this turn to populate pending_recap again.
        checkpoint.session.pending_recap = ""

        logger.info(
            "EventRouter result: feasible=%s, facts=%d, observers=%d, spawns=%d",
            result.canonical_event.world_adjudication.feasible,
            len(result.canonical_event.observable_facts),
            len(result.observers),
            len(result.spawn),
        )

        return result

    def _build_setting_summary(self, checkpoint: CheckpointFile) -> str:
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

    def _build_world_rules(self, checkpoint: CheckpointFile) -> str:
        physics = checkpoint.world_state.physics_ruleset
        parts = [f"Strength limits: {physics.strength_limits}"]
        parts.append(f"Magic: {'enabled' if physics.magic_enabled else 'disabled'}")
        return "\n".join(parts)

    def _build_scene_context(self, checkpoint: CheckpointFile) -> str:
        locations = checkpoint.world_state.locations
        scene_id = locations.current_scene_id
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

    def _build_scene_graph(self, checkpoint: CheckpointFile) -> str:
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

    def _build_characters_present(self, checkpoint: CheckpointFile) -> str:
        scene_id = checkpoint.world_state.locations.current_scene_id
        present = []
        for char in checkpoint.characters:
            if char.location == scene_id and char.status.value == "active":
                role = char.public_sheet.role or "unknown role"
                present.append(f"- {char.name} ({role})")

        if not present:
            return "No other characters are present in this scene."
        return "\n".join(present)

    def _build_hidden_facts(self, checkpoint: CheckpointFile) -> str:
        facts = checkpoint.world_state.hidden_facts
        if not facts:
            return "None."
        return "\n".join(f"- {fact}" for fact in facts)
