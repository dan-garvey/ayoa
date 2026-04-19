"""Narrator composition (phase 2) — composes polished prose from events + agent outputs.

The narrator carries a session-wide rolling conversation on the checkpoint
(`checkpoint.narrator_conversation`). Every NP2 call sees every prior turn's
prose it has written, so voice and continuity hold across the whole session.
"""

from __future__ import annotations

import json
import logging

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, serialize_assistant_content
from app.schemas.agents import CharacterAgentOutput
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.events import CanonicalEvent
from app.schemas.narrator import NarratorFinalOutput

# DiscriminatorOutput is still used as an input hint (observation levels) — its
# per-observer annotations inform NP2's prose. We don't depend on the Discriminator
# class itself; the EventRouter produces a compatible shape.
from app.schemas.discriminator import DiscriminatorOutput

logger = logging.getLogger(__name__)


class Narrator:
    """Narrator phase 2: composes final prose over a rolling session conversation."""

    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager
        # Usage from the most recent compose() call.
        self.last_usage: dict[str, int] = {}

    async def compose(
        self,
        user_input: str,
        event: CanonicalEvent,
        agent_outputs: list[CharacterAgentOutput],
        checkpoint: CheckpointFile,
        disc_output: DiscriminatorOutput | None = None,
        acting_character_id: str = "",
    ) -> NarratorFinalOutput:
        """Compose final narrative prose and append the turn to narrator_conversation."""
        from app.engine.context_builder import build_player_characters_block

        setting_summary = self._build_setting_summary(checkpoint)
        narrative_rules = checkpoint.config.narrative_rules or "No specific narrative rules."
        scene_context = self._build_scene_context(checkpoint)
        canonical_event = json.dumps(event.model_dump(), indent=2, sort_keys=True)
        formatted_agents = self._format_agent_outputs(agent_outputs, checkpoint, disc_output)

        acting_id = acting_character_id or checkpoint.session.player_character_id
        acting_char = next(
            (c for c in checkpoint.characters if c.character_id == acting_id), None
        )
        acting_name = (
            acting_char.name if acting_char
            else (checkpoint.session.player_name or "the protagonist")
        )
        player_characters_block = build_player_characters_block(checkpoint, acting_id)

        # Opening directive fires only on the very first turn of the story
        # (when the narrator has no prior turns to draw from).
        opening_directive = ""
        if not checkpoint.narrator_conversation and checkpoint.opening_narrative:
            opening_directive = (
                "## Opening Scene Directive\n"
                "This is the OPENING of the story. Render the full opening passage "
                "from the author's guidance below — IN ITS ENTIRETY. If the guidance "
                "describes a journey, approach, border crossing, or any movement INTO "
                "the starting scene, render that sequence from its actual start — "
                "not in media res with the player characters already settled. The "
                "canonical event from the router describes the arrival moment; your "
                "job is to render everything leading up to and including it, "
                "grounded in the author's opening prose.\n"
                "\n"
                "Write this as the opening of a novel — evocative, grounding, giving "
                "the reader a clear sense of who the acting character is, where they "
                "are, how they got here, and what looms. Weave in each listed player "
                "character's physical presence (from the system prompt's Player "
                "Characters block) so observers naturally see them as described.\n"
                "\n"
                "End at a natural first decision point for the acting character.\n"
                "\n"
                "## Author's Opening Guidance\n"
                f"{checkpoint.opening_narrative}\n"
            )

        messages = self.prompt_manager.render_conversation(
            "narrator_phase2",
            history=checkpoint.narrator_conversation,
            setting_summary=setting_summary,
            narrative_rules=narrative_rules,
            canonical_event=canonical_event,
            agent_outputs=formatted_agents,
            scene_context=scene_context,
            user_input=user_input,
            acting_character_name=acting_name,
            player_characters_block=player_characters_block,
            opening_directive=opening_directive,
        )

        user_content = messages[-1]["content"]

        logger.info(
            "Narrator: composing with %d agent outputs (history=%d msgs)",
            len(agent_outputs),
            len(checkpoint.narrator_conversation),
        )

        response = await self.client.complete(
            role="narrator",
            messages=messages,
            response_model=NarratorFinalOutput,
            temperature=0.5,
            max_tokens=4000,
            cache=True,
            compact=True,
        )
        result: NarratorFinalOutput = response.parsed
        self.last_usage = response.usage

        assistant_content = serialize_assistant_content(response.raw_response.content)
        checkpoint.narrator_conversation.append(
            ConversationMessage(role="user", content=user_content)
        )
        checkpoint.narrator_conversation.append(
            ConversationMessage(role="assistant", content=assistant_content)
        )

        logger.info(
            "Narrator result: %d chars, summary: %s",
            len(result.final_text),
            result.turn_summary[:80] if result.turn_summary else "(none)",
        )

        return result

    def _format_agent_outputs(
        self,
        agent_outputs: list[CharacterAgentOutput],
        checkpoint: CheckpointFile,
        disc_output: DiscriminatorOutput | None = None,
    ) -> str:
        """Format agent outputs for the NP2 prompt."""
        if not agent_outputs:
            return "No characters responded to this event."

        obs_levels: dict[str, str] = {}
        if disc_output:
            for obs in disc_output.observers:
                obs_levels[obs.character_id] = obs.observation_level

        known = set(checkpoint.world_state.known_characters)
        sections = []
        for output in agent_outputs:
            char = next(
                (c for c in checkpoint.characters if c.character_id == output.character_id),
                None,
            )

            # Use name only if player has met this character; otherwise describe them
            if output.character_id in known:
                label = char.name if char else output.character_id
            elif char and char.public_sheet.appearance:
                label = char.public_sheet.appearance
            elif char and char.public_sheet.role:
                label = char.public_sheet.role
            else:
                label = output.character_id

            parts = [f"### {label}"]
            obs_level = obs_levels.get(output.character_id, "direct")
            parts.append(f"[Observation: {obs_level}]")

            if output.public_response.actions:
                parts.append("Actions:")
                for action in output.public_response.actions:
                    parts.append(f"  - {action}")

            if output.public_response.dialogue:
                parts.append("Dialogue:")
                for line in output.public_response.dialogue:
                    parts.append(f'  - "{line}"')

            if output.public_response.expression:
                parts.append(f"Expression: {output.public_response.expression}")

            sections.append("\n".join(parts))

        return "\n\n".join(sections)

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
            connections = []
            for conn_id in connected:
                conn_scene = locations.scene_graph.get(conn_id, {})
                conn_name = conn_scene.get("name", conn_id)
                connections.append(conn_name)
            parts.append(f"Connected to: {', '.join(connections)}")

        return "\n".join(parts)
