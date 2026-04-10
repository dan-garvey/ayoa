"""Character manager — registry operations, memory/attitude updates.

Handles character lookup, state mutations after agent responses,
and roster changes from discriminator output.
"""

from __future__ import annotations

import logging

from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord, CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.discriminator import DiscriminatorOutput

logger = logging.getLogger(__name__)


class CharacterManager:
    """Manages character registry and state updates."""

    def get_character(self, checkpoint: CheckpointFile, character_id: str) -> CharacterRecord | None:
        """Look up a character by ID."""
        for char in checkpoint.characters:
            if char.character_id == character_id:
                return char
        return None

    def apply_agent_output(
        self, checkpoint: CheckpointFile, output: CharacterAgentOutput
    ) -> None:
        """Apply an agent's private updates and memory writes to the checkpoint."""
        char = self.get_character(checkpoint, output.character_id)
        if not char:
            logger.warning("Character %s not found for update", output.character_id)
            return

        # Apply attitude deltas
        for target, delta in output.private_updates.attitude_delta.items():
            current = char.private_state.attitudes.get(target, 0.0)
            new_val = max(-1.0, min(1.0, current + delta))
            char.private_state.attitudes[target] = new_val

        # Write memories
        for memory in output.memory_writes:
            char.memory.episodic.append(memory)

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
