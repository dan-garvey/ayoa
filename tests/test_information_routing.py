"""Test information routing and perception filtering."""

import pytest

from core.engine.information_router import InformationRouter
from core.models.schemas import AgentState, RoutingDecision, Scene
from core.roles.director import Director


@pytest.mark.asyncio
async def test_routing_present_characters(sample_scene, sample_character_concept, monkeypatch):
    """Test routing to characters present in scene."""
    from core.agents.character_agent import CharacterAgent

    # Create mock agent
    agent = CharacterAgent.from_concept(sample_character_concept, agent_id="agent_123")
    agent.dossier.name = "Eleanor Blackwood"  # Present in scene

    agent_state = AgentState(agent_id="agent_123", dossier=agent.dossier, turn_memory=[])

    agents = {"agent_123": agent_state}

    # Mock director response
    mock_decisions = [
        RoutingDecision(
            character="Eleanor Blackwood",
            agent_id="agent_123",
            receives_packet=True,
            packet=None,
            reason="present in scene",
            attention_level="full",
        )
    ]

    async def mock_route_information(scene, user_input, agents, recent_history):
        return mock_decisions

    director = Director()
    monkeypatch.setattr(director, "route_information", mock_route_information)

    router = InformationRouter(director)

    decisions = await router.route_information(
        scene=sample_scene, user_input="I examine the room", agent_registry=agents, recent_events=[]
    )

    assert len(decisions) == 1
    assert decisions[0].character == "Eleanor Blackwood"
    assert decisions[0].receives_packet is True
    assert decisions[0].attention_level == "full"


@pytest.mark.asyncio
async def test_routing_nearby_characters(sample_scene, sample_character_concept, monkeypatch):
    """Test routing to nearby characters with partial attention."""
    from core.agents.character_agent import CharacterAgent

    # Create mock agent for nearby character
    agent = CharacterAgent.from_concept(sample_character_concept)  # Lord Ashford
    agent_state = AgentState(agent_id="agent_123", dossier=agent.dossier, turn_memory=[])

    agents = {"agent_123": agent_state}

    # Mock director response
    mock_decisions = [
        RoutingDecision(
            character="Lord Ashford",
            agent_id="agent_123",
            receives_packet=True,
            packet=None,
            reason="nearby, can overhear",
            attention_level="partial",
        )
    ]

    async def mock_route_information(scene, user_input, agents, recent_history):
        return mock_decisions

    director = Director()
    monkeypatch.setattr(director, "route_information", mock_route_information)

    router = InformationRouter(director)

    decisions = await router.route_information(
        scene=sample_scene, user_input="I whisper to myself", agent_registry=agents, recent_events=[]
    )

    assert len(decisions) == 1
    assert decisions[0].character == "Lord Ashford"
    assert decisions[0].attention_level == "partial"


@pytest.mark.asyncio
async def test_routing_remote_characters_no_info(sample_scene, monkeypatch):
    """Test that remote characters don't receive information by default."""
    from core.agents.character_agent import CharacterAgent
    from core.models.schemas import CharacterConcept

    # Create a character not in scene
    remote_concept = CharacterConcept(
        name="Distant Noble",
        role="rival",
        description="Far away",
        personality=["proud"],
        goals=["gain favor"],
        relationship_to_player="competitive",
    )

    agent = CharacterAgent.from_concept(remote_concept, agent_id="agent_999")
    agent_state = AgentState(agent_id="agent_999", dossier=agent.dossier, turn_memory=[])

    agents = {"agent_999": agent_state}

    # Mock director response - remote character gets no info
    mock_decisions = [
        RoutingDecision(
            character="Distant Noble",
            agent_id="agent_999",
            receives_packet=False,
            packet=None,
            reason="not present or nearby",
            attention_level="full",
        )
    ]

    async def mock_route_information(scene, user_input, agents, recent_history):
        return mock_decisions

    director = Director()
    monkeypatch.setattr(director, "route_information", mock_route_information)

    router = InformationRouter(director)

    decisions = await router.route_information(
        scene=sample_scene, user_input="I search the room", agent_registry=agents, recent_events=[]
    )

    assert len(decisions) == 1
    assert decisions[0].receives_packet is False
