"""Character agent implementation with persistent state."""

import uuid
from typing import Optional

from core.models.schemas import (
    CharacterConcept,
    CharacterResponse,
    Dossier,
    InformationPacket,
    StyleCard,
)
from core.roles.character import CharacterRole


class CharacterAgent:
    """A persistent agent representing a character in the story."""

    def __init__(self, agent_id: str, dossier: Dossier):
        """
        Initialize a character agent.

        Args:
            agent_id: Unique identifier for this agent
            dossier: Complete character identity and state
        """
        self.agent_id = agent_id
        self.dossier = dossier
        self.memory_buffer: list[dict] = []  # Full conversation memory
        self.role = CharacterRole()

        # Get temperature from style card or use default
        self.temperature = dossier.style_card.temperature_override or 0.7

    async def perceive_and_respond(
        self, packet: InformationPacket, attention_level: str
    ) -> CharacterResponse:
        """
        Process information and decide whether/how to respond.

        Args:
            packet: What the character perceives
            attention_level: How focused they are (full/partial/peripheral)

        Returns:
            Character's response (may choose not to act)
        """
        response = await self.role.generate_response(
            dossier=self.dossier,
            packet=packet,
            attention_level=attention_level,
            context_memory=self.memory_buffer,
        )

        return response

    def update_memory(self, turn_data: dict):
        """
        Add turn to memory buffer.

        Args:
            turn_data: Data about this turn (action, outcome, etc.)
        """
        self.memory_buffer.append(turn_data)

        # TODO: Future - implement forgetting/summarization
        # Keep last N turns or summarize older memories
        if len(self.memory_buffer) > 20:
            # For now, just truncate to last 20 turns
            self.memory_buffer = self.memory_buffer[-20:]

    def update_beliefs(self, new_info: list[str]):
        """
        Update character's beliefs based on observations.

        Args:
            new_info: New information learned
        """
        for info in new_info:
            if info not in self.dossier.beliefs:
                self.dossier.beliefs.append(info)

    def update_relationships(self, character: str, interaction: str):
        """
        Update relationship stance based on interactions.

        Args:
            character: Name of other character
            interaction: Description of the interaction
        """
        current = self.dossier.relationships.get(character, "neutral")

        # Simple relationship update - in a full implementation,
        # this could use LLM to determine new stance
        self.dossier.relationships[character] = f"{current} (recent: {interaction})"

    def update_emotional_state(self, new_state: str):
        """
        Update the character's emotional state.

        Args:
            new_state: New emotional state
        """
        self.dossier.emotional_state = new_state

    def add_memory(self, memory: str):
        """
        Add a key memory to the character's long-term memory.

        Args:
            memory: Memory to add
        """
        if memory not in self.dossier.memories:
            self.dossier.memories.append(memory)

    @classmethod
    def from_concept(
        cls, concept: CharacterConcept, agent_id: Optional[str] = None
    ) -> "CharacterAgent":
        """
        Create a character agent from a concept.

        Args:
            concept: High-level character concept
            agent_id: Optional specific agent ID (generates one if not provided)

        Returns:
            New character agent
        """
        if agent_id is None:
            agent_id = f"agent_{uuid.uuid4().hex[:8]}"

        # Create style card from concept
        # In a full implementation, might use LLM to generate this
        style_card = StyleCard(
            voice=concept.personality[:2] if len(concept.personality) >= 2 else concept.personality,
            speech_patterns=[f"Tends to be {concept.personality[0]}" if concept.personality else "Neutral"],
            catchphrases=[],
            taboos=[],
            temperature_override=None,
        )

        dossier = Dossier(
            name=concept.name,
            agent_id=agent_id,
            character_concept=concept,
            style_card=style_card,
            beliefs=[],
            current_goals=concept.goals.copy(),
            memories=[],
            relationships={},
            emotional_state="neutral",
        )

        return cls(agent_id=agent_id, dossier=dossier)
