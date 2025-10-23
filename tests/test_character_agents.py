"""Test character agent system."""

import pytest

from core.agents.character_agent import CharacterAgent
from core.agents.manager import AgentManager
from core.models.schemas import (
    AgentState,
    InformationPacket,
)


def test_character_agent_from_concept(sample_character_concept):
    """Test creating agent from concept."""
    agent = CharacterAgent.from_concept(sample_character_concept)

    assert agent.dossier.name == "Lord Ashford"
    assert agent.dossier.character_concept.role == "antagonist"
    assert len(agent.memory_buffer) == 0


def test_character_agent_memory_update(sample_character_concept):
    """Test updating agent memory."""
    agent = CharacterAgent.from_concept(sample_character_concept)

    turn_data = {"turn": 1, "summary": "Player investigates"}
    agent.update_memory(turn_data)

    assert len(agent.memory_buffer) == 1
    assert agent.memory_buffer[0]["turn"] == 1


def test_character_agent_memory_truncation(sample_character_concept):
    """Test memory truncation after 20 turns."""
    agent = CharacterAgent.from_concept(sample_character_concept)

    # Add 25 memories
    for i in range(25):
        agent.update_memory({"turn": i})

    # Should keep only last 20
    assert len(agent.memory_buffer) == 20
    assert agent.memory_buffer[0]["turn"] == 5  # First 5 removed


def test_character_agent_update_beliefs(sample_character_concept):
    """Test updating beliefs."""
    agent = CharacterAgent.from_concept(sample_character_concept)

    agent.update_beliefs(["The player is trustworthy", "The conspiracy runs deep"])

    assert len(agent.dossier.beliefs) == 2
    assert "The player is trustworthy" in agent.dossier.beliefs


def test_character_agent_update_relationships(sample_character_concept):
    """Test updating relationships."""
    agent = CharacterAgent.from_concept(sample_character_concept)

    agent.update_relationships("Eleanor", "friendly conversation")

    assert "Eleanor" in agent.dossier.relationships


@pytest.mark.asyncio
async def test_agent_manager_spawn(sample_character_concept):
    """Test agent manager spawning."""
    manager = AgentManager()

    agent_id = await manager.spawn_agent(sample_character_concept)

    assert agent_id in manager.agents
    assert agent_id in manager.agent_states
    assert manager.agents[agent_id].dossier.name == "Lord Ashford"


@pytest.mark.asyncio
async def test_agent_manager_spawn_multiple(sample_character_concept):
    """Test spawning multiple agents."""
    manager = AgentManager()

    concepts = [sample_character_concept for _ in range(3)]
    agent_ids = await manager.spawn_agents(concepts)

    assert len(agent_ids) == 3
    assert len(manager.agents) == 3


def test_agent_manager_get_agent(sample_character_concept):
    """Test getting agent by ID."""
    manager = AgentManager()
    agent = CharacterAgent.from_concept(sample_character_concept, agent_id="test_123")
    manager.agents["test_123"] = agent

    retrieved = manager.get_agent("test_123")
    assert retrieved is not None
    assert retrieved.agent_id == "test_123"


def test_agent_manager_get_agent_by_name(sample_character_concept):
    """Test getting agent by character name."""
    manager = AgentManager()
    agent = CharacterAgent.from_concept(sample_character_concept)
    manager.agents[agent.agent_id] = agent

    retrieved = manager.get_agent_by_name("Lord Ashford")
    assert retrieved is not None
    assert retrieved.dossier.name == "Lord Ashford"


def test_agent_manager_update_memory(sample_character_concept):
    """Test updating agent memory through manager."""
    manager = AgentManager()
    agent = CharacterAgent.from_concept(sample_character_concept, agent_id="test_123")
    manager.agents["test_123"] = agent
    manager.agent_states["test_123"] = AgentState(
        agent_id="test_123", dossier=agent.dossier, turn_memory=[]
    )

    turn_data = {"turn": 1, "summary": "Test"}
    manager.update_agent_memory("test_123", turn_data)

    assert len(agent.memory_buffer) == 1
    assert len(manager.agent_states["test_123"].turn_memory) == 1


def test_agent_manager_persist_and_restore(sample_character_concept):
    """Test persisting and restoring agent state."""
    manager = AgentManager()
    agent = CharacterAgent.from_concept(sample_character_concept, agent_id="test_123")
    agent.update_memory({"turn": 1, "summary": "Test"})
    manager.agents["test_123"] = agent
    manager.agent_states["test_123"] = AgentState(
        agent_id="test_123", dossier=agent.dossier, turn_memory=[]
    )

    # Persist
    manager.persist_agent_state("test_123")

    # Get state and restore
    state = manager.agent_states["test_123"]

    # Create new manager and restore
    manager2 = AgentManager()
    restored_id = manager2.restore_agent(state)

    assert restored_id == "test_123"
    assert "test_123" in manager2.agents
