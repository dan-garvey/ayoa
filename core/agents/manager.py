"""Agent manager for spawning and coordinating character agents."""

import asyncio
import uuid
from typing import Optional

from core.agents.character_agent import CharacterAgent
from core.models.schemas import (
    AgentState,
    CharacterConcept,
    CharacterResponse,
    RoutingDecision,
)


class AgentManager:
    """Manages character agent lifecycle and coordination."""

    def __init__(self):
        """Initialize the agent manager."""
        self.agents: dict[str, CharacterAgent] = {}
        self.agent_states: dict[str, AgentState] = {}

    async def spawn_agent(self, concept: CharacterConcept) -> str:
        """
        Create a new character agent from a concept.

        Args:
            concept: High-level character concept

        Returns:
            Unique agent_id for the created agent
        """
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        agent = CharacterAgent.from_concept(concept, agent_id=agent_id)

        self.agents[agent_id] = agent

        # Create agent state for persistence
        state = AgentState(
            agent_id=agent_id,
            dossier=agent.dossier,
            turn_memory=[],
            active=True,
            last_action_turn=0,
        )
        self.agent_states[agent_id] = state

        return agent_id

    async def spawn_agents(self, concepts: list[CharacterConcept]) -> list[str]:
        """
        Spawn multiple agents concurrently.

        Args:
            concepts: List of character concepts

        Returns:
            List of agent IDs
        """
        tasks = [self.spawn_agent(concept) for concept in concepts]
        return await asyncio.gather(*tasks)

    async def get_agent_responses(
        self, routing_decisions: list[RoutingDecision], max_concurrent: int = 4
    ) -> list[CharacterResponse]:
        """
        Fan out to agents in batches to get their responses.

        Args:
            routing_decisions: Routing decisions from Director
            max_concurrent: Maximum agents to query concurrently

        Returns:
            List of character responses
        """
        # Filter for characters that receive information
        active_decisions = [d for d in routing_decisions if d.receives_packet and d.packet]

        if not active_decisions:
            return []

        # Process in batches to respect max_concurrent
        responses = []
        for i in range(0, len(active_decisions), max_concurrent):
            batch = active_decisions[i : i + max_concurrent]

            tasks = []
            for decision in batch:
                agent = self.agents.get(decision.agent_id)
                if agent:
                    task = agent.perceive_and_respond(
                        packet=decision.packet,  # type: ignore
                        attention_level=decision.attention_level,
                    )
                    tasks.append(task)

            batch_responses = await asyncio.gather(*tasks)
            responses.extend(batch_responses)

        return responses

    def get_agent(self, agent_id: str) -> Optional[CharacterAgent]:
        """
        Get an agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Character agent if found, None otherwise
        """
        return self.agents.get(agent_id)

    def get_agent_by_name(self, name: str) -> Optional[CharacterAgent]:
        """
        Get an agent by character name.

        Args:
            name: Character name

        Returns:
            Character agent if found, None otherwise
        """
        for agent in self.agents.values():
            if agent.dossier.name == name:
                return agent
        return None

    def list_agents(self) -> dict[str, AgentState]:
        """
        List all agent states.

        Returns:
            Dictionary of agent_id -> AgentState
        """
        return self.agent_states.copy()

    def update_agent_memory(self, agent_id: str, turn_data: dict):
        """
        Update an agent's memory.

        Args:
            agent_id: Agent identifier
            turn_data: Data about the turn
        """
        agent = self.agents.get(agent_id)
        if agent:
            agent.update_memory(turn_data)
            # Update state
            if agent_id in self.agent_states:
                self.agent_states[agent_id].turn_memory.append(turn_data)

    def persist_agent_state(self, agent_id: str):
        """
        Save agent state (stub for future persistence).

        Args:
            agent_id: Agent to persist
        """
        # TODO: Save to database
        agent = self.agents.get(agent_id)
        if agent and agent_id in self.agent_states:
            state = self.agent_states[agent_id]
            state.dossier = agent.dossier
            state.turn_memory = agent.memory_buffer.copy()

    def restore_agent(self, state: AgentState) -> str:
        """
        Restore an agent from saved state.

        Args:
            state: Saved agent state

        Returns:
            Agent ID
        """
        agent = CharacterAgent(agent_id=state.agent_id, dossier=state.dossier)
        agent.memory_buffer = state.turn_memory.copy()

        self.agents[state.agent_id] = agent
        self.agent_states[state.agent_id] = state

        return state.agent_id
