"""Narrator engine — adjudication (Phase 1) and composition (Phase 2).

Phase 1 takes user input + world state and produces a CanonicalEvent.
Phase 2 (future) takes the CanonicalEvent + agent outputs and produces final prose.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.events import CanonicalEvent
from app.schemas.checkpoint import CheckpointFile

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Narrator:
    """Narrator engine with adjudication (phase 1) and composition (phase 2)."""

    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager

    async def phase_1(
        self,
        user_input: str,
        checkpoint: CheckpointFile,
        max_transcript_entries: int = 10,
    ) -> CanonicalEvent:
        """Adjudicate the player's action against the world state.

        Returns a CanonicalEvent describing what actually happens.
        """
        # Build prompt context from checkpoint
        setting_summary = self._build_setting_summary(checkpoint)
        world_lore = checkpoint.world_state.lore or "No detailed lore available."
        world_rules = self._build_world_rules(checkpoint)
        scene_context = self._build_scene_context(checkpoint)
        characters_present = self._build_characters_present(checkpoint)
        recent_transcript = self._build_recent_transcript(
            checkpoint, max_entries=max_transcript_entries
        )
        world_facts = self._build_world_facts(checkpoint)
        narrative_rules = checkpoint.config.narrative_rules or "No specific narrative rules."

        prompt = self.prompt_manager.render(
            "narrator_phase1",
            setting_summary=setting_summary,
            world_lore=world_lore,
            world_rules=world_rules,
            scene_context=scene_context,
            characters_present=characters_present,
            recent_transcript=recent_transcript,
            world_facts=world_facts,
            narrative_rules=narrative_rules,
            user_input=user_input,
        )

        logger.info("NP1: adjudicating user action: %s", user_input[:80])

        response = await self.client.complete(
            role="narrator",
            messages=[{"role": "user", "content": prompt}],
            response_model=CanonicalEvent,
            temperature=0.4,
            max_tokens=4000,
        )
        event: CanonicalEvent = response.parsed

        # Assign event ID if empty
        if not event.event_id:
            event.event_id = f"evt_{checkpoint.session.turn_index:04d}"

        logger.info(
            "NP1 result: feasible=%s, facts=%d, time=%ds",
            event.world_adjudication.feasible,
            len(event.observable_facts),
            event.scene_delta.time_advanced_seconds,
        )

        return event

    # --- Context builders ---

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

        parts = [f"Location: {name}"]
        if desc:
            parts.append(f"Description: {desc}")
        if connected:
            # Resolve connected location names
            connections = []
            for conn_id in connected:
                conn_scene = locations.scene_graph.get(conn_id, {})
                conn_name = conn_scene.get("name", conn_id)
                connections.append(conn_name)
            parts.append(f"Connected to: {', '.join(connections)}")

        return "\n".join(parts)

    def _build_characters_present(self, checkpoint: CheckpointFile) -> str:
        scene_id = checkpoint.world_state.locations.current_scene_id
        present = []
        for char in checkpoint.characters:
            if char.location == scene_id and char.status.value == "active":
                role = char.public_sheet.role or "unknown role"
                traits = ", ".join(char.public_sheet.traits[:3]) if char.public_sheet.traits else "no notable traits"
                present.append(f"- {char.name} ({role}): {traits}")

        if not present:
            return "No other characters are present in this scene."
        return "\n".join(present)

    def _build_recent_transcript(
        self, checkpoint: CheckpointFile, max_entries: int = 10
    ) -> str:
        if not checkpoint.transcript:
            return "This is the beginning of the story. No prior actions have been taken."

        entries = checkpoint.transcript[-max_entries:]
        lines = []
        for entry in entries:
            lines.append(f"Player: {entry.user}")
            lines.append(f"Narrator: {entry.assistant}")
            lines.append("")
        return "\n".join(lines).strip()

    def _build_world_facts(self, checkpoint: CheckpointFile) -> str:
        facts = checkpoint.world_state.facts
        if not facts:
            return "No specific world facts recorded."
        return "\n".join(f"- {fact}" for fact in facts)
