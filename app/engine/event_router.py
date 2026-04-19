"""Merged adjudication + perception routing — single LLM call per turn.

EventRouter maintains a session-wide rolling conversation on the checkpoint.
Each call sees every prior turn's routing decisions as real conversational
history, which lets it remember commitments, who's where, and unfinished
threads across the entire session.
"""

from __future__ import annotations

import logging

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, serialize_assistant_content
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import EventRouterOutput

logger = logging.getLogger(__name__)


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
        from app.engine.context_builder import build_player_characters_block

        acting_id = acting_character_id or checkpoint.session.player_character_id
        acting_char = next(
            (c for c in checkpoint.characters if c.character_id == acting_id), None
        )
        acting_name = (
            acting_char.name if acting_char
            else (checkpoint.session.player_name or "the protagonist")
        )

        messages = self.prompt_manager.render_conversation(
            "event_router",
            history=checkpoint.session_conversation,
            setting_summary=self._build_setting_summary(checkpoint),
            world_lore=checkpoint.world_state.lore or "No detailed lore available.",
            world_rules=self._build_world_rules(checkpoint),
            current_scene=self._build_scene_context(checkpoint),
            scene_graph=self._build_scene_graph(checkpoint),
            characters_present=self._build_characters_present(checkpoint),
            world_facts=self._build_world_facts(checkpoint),
            hidden_lore=checkpoint.world_state.hidden_lore or "None.",
            hidden_facts=self._build_hidden_facts(checkpoint),
            character_registry=self._build_character_registry(checkpoint),
            user_input=user_input,
            acting_character_name=acting_name,
            player_characters_block=build_player_characters_block(
                checkpoint, acting_id
            ),
        )

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
        result: EventRouterOutput = response.parsed
        self.last_usage = response.usage

        if not result.canonical_event.event_id:
            result.canonical_event.event_id = f"evt_{checkpoint.session.turn_index:04d}"

        # Persist the user/assistant pair to the rolling session conversation.
        assistant_content = serialize_assistant_content(response.raw_response.content)
        checkpoint.session_conversation.append(
            ConversationMessage(role="user", content=user_content)
        )
        checkpoint.session_conversation.append(
            ConversationMessage(role="assistant", content=assistant_content)
        )

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
                    ", ".join(char.public_sheet.traits)
                    if char.public_sheet.traits
                    else "no notable traits"
                )
                present.append(f"- {char.name} ({role}): {traits}")

        if not present:
            return "No other characters are present in this scene."
        return "\n".join(present)

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
        # Player-bound characters are rendered in the Player Characters block
        # instead — omit them here so the router doesn't route observations
        # to humans. Falls back to the creator's binding + is_player roster
        # entries when character_bindings is empty (pristine / legacy saves).
        from app.engine.context_builder import collect_player_ids
        player_ids = collect_player_ids(checkpoint)
        scene_graph = checkpoint.world_state.locations.scene_graph

        for char in checkpoint.characters:
            if char.character_id in player_ids:
                continue

            location = char.location or "unknown"
            loc_name = scene_graph.get(location, {}).get("name", location)
            role = char.public_sheet.role or "unknown role"
            status = char.status.value

            goals_str = ""
            if char.private_state and char.private_state.goals:
                goals_str = "; ".join(char.private_state.goals)

            attitudes_str = ""
            if char.private_state and char.private_state.attitudes:
                items = list(char.private_state.attitudes.items())
                attitudes_str = "; ".join(f"{k}={v:+.1f}" for k, v in items)

            secrets_str = ""
            if char.private_state and char.private_state.secrets:
                secrets_str = "; ".join(char.private_state.secrets)

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
