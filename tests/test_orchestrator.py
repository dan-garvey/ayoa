"""Test orchestrator and game loop."""

import pytest

from core.engine.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_create_story(sample_story_config, monkeypatch):
    """Test story creation."""
    orchestrator = Orchestrator()

    # Mock the storyteller to avoid actual LLM calls
    from core.models.schemas import CharacterConcept, StoryOutline

    mock_outline = StoryOutline(
        premise="A test story",
        acts=["Act 1", "Act 2"],
        major_characters=[
            CharacterConcept(
                name="Test Character",
                role="ally",
                description="A test",
                personality=["friendly"],
                goals=["help player"],
                relationship_to_player="trusted friend",
            )
        ],
        key_locations=["Castle"],
        potential_endings=["Happy ending"],
    )

    async def mock_generate_outline(config):
        return mock_outline

    monkeypatch.setattr(orchestrator.storyteller, "generate_outline", mock_generate_outline)

    story_id, outline = await orchestrator.create_story(sample_story_config)

    assert story_id.startswith("story_")
    assert outline.premise == "A test story"
    assert orchestrator.current_story_id == story_id


def test_save_and_load_story_state(sample_story_config, tmp_path, monkeypatch):
    """Test saving and loading story state."""
    import os

    # Use tmp_path for saves
    monkeypatch.chdir(tmp_path)

    orchestrator = Orchestrator()
    orchestrator.current_story_id = "test_123"
    orchestrator.current_config = sample_story_config
    orchestrator.current_scene = None
    orchestrator.turn_history = [{"turn": 1, "summary": "Test"}]

    # Save
    orchestrator._save_story_state("test_123")

    # Check file exists
    assert (tmp_path / "saves" / "test_123.json").exists()

    # Create new orchestrator and load
    orchestrator2 = Orchestrator()
    orchestrator2._load_story_state("test_123")

    assert orchestrator2.current_story_id == "test_123"
    assert len(orchestrator2.turn_history) == 1


def test_meta_command_scene(sample_scene):
    """Test /scene meta command."""
    import asyncio

    orchestrator = Orchestrator()
    orchestrator.current_story_id = "test_123"
    orchestrator.current_scene = sample_scene

    result = asyncio.run(orchestrator._handle_meta_command("/scene", "test_123"))

    assert "Eleanor Blackwood" in result.narrative
    assert "royal physician's quarters" in result.narrative


def test_meta_command_cast():
    """Test /cast meta command."""
    import asyncio

    from core.agents.character_agent import CharacterAgent
    from core.models.schemas import CharacterConcept

    orchestrator = Orchestrator()
    orchestrator.current_story_id = "test_123"

    # Add a test agent
    concept = CharacterConcept(
        name="Test",
        role="ally",
        description="Test",
        personality=["friendly"],
        goals=["help"],
        relationship_to_player="friend",
    )
    agent = CharacterAgent.from_concept(concept, agent_id="test_agent")

    from core.models.schemas import AgentState

    state = AgentState(agent_id="test_agent", dossier=agent.dossier, turn_memory=[])
    orchestrator.agent_manager.agents["test_agent"] = agent
    orchestrator.agent_manager.agent_states["test_agent"] = state

    result = asyncio.run(orchestrator._handle_meta_command("/cast", "test_123"))

    assert "Test" in result.narrative
    assert "ally" in result.narrative
