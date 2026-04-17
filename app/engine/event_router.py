"""Merged NP1 + discriminator prototype.

Produces the canonical event and observation routing in a single LLM call.
"""

from __future__ import annotations

import logging

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput

logger = logging.getLogger(__name__)


class EventRouter:
    """Single-pass event adjudication + perception routing."""

    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager

    async def run(
        self,
        user_input: str,
        checkpoint: CheckpointFile,
        max_transcript_entries: int = 10,
    ) -> EventRouterOutput:
        """Return both canonical event data and observer routing."""
        prompt = self.prompt_manager.render(
            "event_router",
            setting_summary=self._build_setting_summary(checkpoint),
            world_lore=checkpoint.world_state.lore or "No detailed lore available.",
            world_rules=self._build_world_rules(checkpoint),
            current_scene=self._build_scene_context(checkpoint),
            scene_graph=self._build_scene_graph(checkpoint),
            characters_present=self._build_characters_present(checkpoint),
            recent_transcript=self._build_recent_transcript(
                checkpoint, max_entries=max_transcript_entries
            ),
            world_facts=self._build_world_facts(checkpoint),
            hidden_lore=checkpoint.world_state.hidden_lore or "None.",
            hidden_facts=self._build_hidden_facts(checkpoint),
            character_registry=self._build_character_registry(checkpoint),
            user_input=user_input,
            player_name=checkpoint.session.player_name or "the protagonist",
        )

        logger.info("EventRouter: adjudicating + routing action: %s", user_input[:80])

        response = await self.client.complete(
            role="event_router",
            messages=[{"role": "user", "content": prompt}],
            response_model=EventRouterOutput,
            temperature=0.35,
            max_tokens=5000,
        )
        result: EventRouterOutput = response.parsed

        if not result.canonical_event.event_id:
            result.canonical_event.event_id = f"evt_{checkpoint.session.turn_index:04d}"

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
                traits = (
                    ", ".join(char.public_sheet.traits[:3])
                    if char.public_sheet.traits
                    else "no notable traits"
                )
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

    def _build_hidden_facts(self, checkpoint: CheckpointFile) -> str:
        facts = checkpoint.world_state.hidden_facts
        if not facts:
            return "None."
        return "\n".join(f"- {fact}" for fact in facts)

    def _build_character_registry(self, checkpoint: CheckpointFile) -> str:
        if not checkpoint.characters:
            return "No characters in the registry."

        entries = []
        player_id = checkpoint.session.player_character_id
        scene_graph = checkpoint.world_state.locations.scene_graph

        for char in checkpoint.characters:
            if char.character_id == player_id:
                continue

            location = char.location or "unknown"
            loc_name = scene_graph.get(location, {}).get("name", location)
            role = char.public_sheet.role or "unknown role"
            status = char.status.value

            goals_str = ""
            if char.private_state and char.private_state.goals:
                goals_str = "; ".join(char.private_state.goals[:3])

            attitudes_str = ""
            if char.private_state and char.private_state.attitudes:
                items = list(char.private_state.attitudes.items())[:4]
                attitudes_str = "; ".join(f"{k}={v:+.1f}" for k, v in items)

            secrets_str = ""
            if char.private_state and char.private_state.secrets:
                secrets_str = "; ".join(char.private_state.secrets[:2])

            entry = (
                f"- {char.name} (id: {char.character_id})\n"
                f"  Status: {status}\n"
                f"  Location: {loc_name} ({location})\n"
                f"  Role: {role}"
            )
            if goals_str:
                entry += f"\n  Goals: {goals_str}"
            if attitudes_str:
                entry += f"\n  Attitudes: {attitudes_str}"
            if secrets_str:
                entry += f"\n  Secrets: {secrets_str}"
            entries.append(entry)

        return "\n".join(entries)
