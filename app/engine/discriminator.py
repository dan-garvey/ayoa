"""Discriminator engine — perception gating and roster management.

Takes a CanonicalEvent + character registry + world state and determines
which characters observe the event, what they perceive, and whether they
should generate a response.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.discriminator import DiscriminatorOutput
from app.schemas.events import CanonicalEvent
from app.schemas.checkpoint import CheckpointFile

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Discriminator:
    """Perception discriminator for observation gating and roster management."""

    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager

    async def run(
        self,
        event: CanonicalEvent,
        checkpoint: CheckpointFile,
    ) -> DiscriminatorOutput:
        """Determine who observes the event and who should respond.

        Returns a DiscriminatorOutput with observer entries, spawn requests,
        dormancy updates, and cull lists.
        """
        setting_summary = self._build_setting_summary(checkpoint)
        scene_context = self._build_scene_context(checkpoint)
        canonical_event = self._format_event(event)
        character_registry = self._build_character_registry(checkpoint)

        prompt = self.prompt_manager.render(
            "discriminator",
            setting_summary=setting_summary,
            scene_context=scene_context,
            canonical_event=canonical_event,
            character_registry=character_registry,
            event_id=event.event_id,
        )

        logger.info(
            "Discriminator: evaluating event %s with %d characters",
            event.event_id,
            len(checkpoint.characters),
        )

        response = await self.client.complete(
            role="discriminator",
            messages=[{"role": "user", "content": prompt}],
            response_model=DiscriminatorOutput,
            temperature=0.3,
            max_tokens=4000,
        )
        result: DiscriminatorOutput = response.parsed

        # Ensure event_id is set
        if not result.event_id:
            result.event_id = event.event_id

        observers_responding = sum(1 for o in result.observers if o.should_respond)
        logger.info(
            "Discriminator result: %d observers, %d responding, %d spawns",
            len(result.observers),
            observers_responding,
            len(result.spawn),
        )

        return result

    # --- Context builders ---

    def _build_setting_summary(self, checkpoint: CheckpointFile) -> str:
        setting = checkpoint.world_state.setting
        parts = []
        if setting.genre:
            parts.append(f"Genre: {setting.genre}")
        if setting.tone:
            parts.append(f"Tone: {setting.tone}")
        if setting.premise:
            parts.append(f"Premise: {setting.premise}")
        return "\n".join(parts) if parts else "No setting information."

    def _build_scene_context(self, checkpoint: CheckpointFile) -> str:
        locations = checkpoint.world_state.locations
        scene_id = locations.current_scene_id
        if not scene_id:
            return "No scene information available."

        scene = locations.scene_graph.get(scene_id, {})
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
                conn_details.append(f"  - {conn_name} (id: {conn_id})")
            parts.append("Connected locations:\n" + "\n".join(conn_details))

        return "\n".join(parts)

    def _format_event(self, event: CanonicalEvent) -> str:
        """Format the canonical event for the prompt."""
        data = event.model_dump()
        return json.dumps(data, indent=2)

    def _build_character_registry(self, checkpoint: CheckpointFile) -> str:
        """Build a summary of all active characters with locations and roles."""
        if not checkpoint.characters:
            return "No characters in the registry."

        entries = []
        scene_id = checkpoint.world_state.locations.current_scene_id

        for char in checkpoint.characters:
            status = char.status.value
            location = char.location or "unknown"

            # Determine spatial relationship to current scene
            scene_graph = checkpoint.world_state.locations.scene_graph
            current_scene = scene_graph.get(scene_id, {})
            connected = current_scene.get("connected_to", [])

            if location == scene_id:
                proximity = "SAME LOCATION"
            elif location in connected:
                proximity = "ADJACENT"
            else:
                proximity = "DISTANT"

            role = char.public_sheet.role or "unknown role"
            loc_name = scene_graph.get(location, {}).get("name", location)

            entry = (
                f"- {char.name} (id: {char.character_id})\n"
                f"  Status: {status} | Location: {loc_name} ({location}) | Proximity: {proximity}\n"
                f"  Role: {role}"
            )
            entries.append(entry)

        return "\n".join(entries)
