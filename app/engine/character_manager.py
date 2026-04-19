"""Character manager — registry operations, memory/attitude updates, spawning.

Handles character lookup, state mutations after agent responses,
roster changes from discriminator output, and LLM-powered character genesis.
"""

from __future__ import annotations

import json
import logging

from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    PublicSheet,
    PrivateState,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.discriminator import DiscriminatorOutput, SpawnRequest

logger = logging.getLogger(__name__)

# Max spawns per turn to prevent latency blowups
MAX_SPAWNS_PER_TURN = 3


class CharacterManager:
    """Manages character registry and state updates."""

    def __init__(
        self,
        client: LLMClient | None = None,
        prompt_manager: PromptManager | None = None,
    ):
        self.client = client
        self.prompt_manager = prompt_manager

    def get_character(self, checkpoint: CheckpointFile, character_id: str) -> CharacterRecord | None:
        """Look up a character by ID."""
        for char in checkpoint.characters:
            if char.character_id == character_id:
                return char
        return None

    def apply_agent_output(
        self, checkpoint: CheckpointFile, output: CharacterAgentOutput
    ) -> None:
        """Apply an agent's attitude deltas.

        Memory is handled by the character's rolling conversation; the
        `memory_writes` field on the output is kept in the schema for
        compatibility but no longer written anywhere on the character.
        """
        char = self.get_character(checkpoint, output.character_id)
        if not char:
            logger.warning("Character %s not found for update", output.character_id)
            return

        for target, delta in output.private_updates.attitude_delta.items():
            current = char.private_state.attitudes.get(target, 0.0)
            new_val = max(-1.0, min(1.0, current + delta))
            char.private_state.attitudes[target] = new_val

    def apply_roster_updates(
        self, checkpoint: CheckpointFile, disc_output: DiscriminatorOutput
    ) -> None:
        """Apply discriminator roster changes (dormancy, culling)."""
        for char_id in disc_output.dormant:
            char = self.get_character(checkpoint, char_id)
            if char:
                char.status = CharacterStatus.dormant
                logger.info("Character %s set to dormant", char_id)

        for char_id in disc_output.cull:
            char = self.get_character(checkpoint, char_id)
            if char:
                char.status = CharacterStatus.culled
                logger.info("Character %s culled", char_id)

    async def spawn_characters(
        self,
        checkpoint: CheckpointFile,
        spawn_requests: list[SpawnRequest],
    ) -> list[CharacterRecord]:
        """Generate new characters from discriminator spawn requests via LLM.

        Returns the list of newly created and registered characters.
        """
        if not self.client or not self.prompt_manager:
            logger.warning("CharacterManager has no LLM client; skipping spawns")
            return []

        # Limit spawns per turn
        requests = spawn_requests[:MAX_SPAWNS_PER_TURN]
        if len(spawn_requests) > MAX_SPAWNS_PER_TURN:
            logger.warning(
                "Capping spawns from %d to %d", len(spawn_requests), MAX_SPAWNS_PER_TURN
            )

        # Skip already-existing characters
        requests = [
            r for r in requests
            if self.get_character(checkpoint, r.character_id) is None
        ]

        if not requests:
            return []

        spawned = []
        for req in requests:
            try:
                char = await self._spawn_one(checkpoint, req)
                checkpoint.characters.append(char)
                spawned.append(char)
                logger.info("Spawned character: %s (%s)", char.name, char.character_id)
            except Exception as e:
                logger.warning("Failed to spawn %s: %s", req.character_id, e)

        return spawned

    async def _spawn_one(
        self, checkpoint: CheckpointFile, req: SpawnRequest
    ) -> CharacterRecord:
        """Generate a single character via LLM."""
        setting = checkpoint.world_state.setting
        setting_summary = f"Genre: {setting.genre}\nEra: {setting.era}\nTone: {setting.tone}\nPremise: {setting.premise}"
        world_lore = checkpoint.world_state.lore or "No detailed lore."
        physics = checkpoint.world_state.physics_ruleset
        world_rules = f"Strength limits: {physics.strength_limits}\nMagic: {'enabled' if physics.magic_enabled else 'disabled'}"

        locations = checkpoint.world_state.locations
        scene_id = req.seed.get("location", locations.current_scene_id)
        scene = locations.scene_graph.get(scene_id, {})
        scene_context = f"Location: {scene.get('name', scene_id)}\n{scene.get('description', '')}"

        seed_lines = []
        for k, v in req.seed.items():
            seed_lines.append(f"{k}: {v}")
        spawn_seed = "\n".join(seed_lines) if seed_lines else "No specific seed provided."

        existing = ", ".join(c.name for c in checkpoint.characters)

        messages = self.prompt_manager.render_messages(
            "character_gen",
            setting_summary=setting_summary,
            world_lore=world_lore,
            world_rules=world_rules,
            scene_context=scene_context,
            character_id=req.character_id,
            spawn_seed=spawn_seed,
            existing_characters=existing,
            location=scene_id,
        )

        response = await self.client.complete(
            role="agent",
            messages=messages,
            response_model=CharacterRecord,
            temperature=0.6,
            max_tokens=3000,
        )
        char: CharacterRecord = response.parsed

        # Enforce character_id and location from request
        char.character_id = req.character_id
        char.location = scene_id

        return char
