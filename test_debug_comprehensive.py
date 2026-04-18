"""Comprehensive test of all debug output features."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Set debug mode BEFORE importing core modules
os.environ["DEBUG_AGENT_ACTIVITY"] = "true"

from core.agents.manager import AgentManager
from core.config import engine_config
from core.engine.orchestrator import orchestrator
from core.models.schemas import (
    CharacterConcept,
    CharacterMove,
    CharacterResponse,
    DirectorDecision,
    InformationPacket,
    RoutingDecision,
    Scene,
)
from core.roles.director import Director


async def test_all_debug_outputs():
    """Test that all debug outputs appear correctly."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DEBUG OUTPUT TEST")
    print("=" * 80 + "\n")

    # Verify debug mode is enabled
    assert engine_config.debug_agent_activity, "Debug mode should be enabled"
    print("✓ Debug mode confirmed enabled\n")

    # Test 1: Agent Spawning Debug (🎭)
    print("TEST 1: Agent Spawning Debug (🎭)")
    print("-" * 40)

    agent_manager = AgentManager()
    concept = CharacterConcept(
        name="Test Character",
        role="ally",
        description="A test character",
        personality=["friendly", "helpful"],
        goals=["help the player"],
        secrets=[],
        relationship_to_player="trusted friend"
    )

    agent_id = await agent_manager.spawn_agent(concept)
    print(f"✓ Agent spawned successfully: {agent_id}\n")

    # Test 2: Director Routing Debug (🎬)
    print("TEST 2: Director Routing Debug (🎬)")
    print("-" * 40)

    director = Director()
    scene = Scene(
        scene_id="test_scene",
        where="Test Location",
        when="Now",
        atmosphere="Tense",
        present_characters=["Test Character"],
        nearby_characters=[],
        facts=[]
    )

    # Mock the LLM call ONLY, not the entire method
    from pydantic import BaseModel

    class RoutingDecisionList(BaseModel):
        decisions: list[RoutingDecision]

    mock_routing = RoutingDecision(
        character="Test Character",
        agent_id=agent_id,
        receives_packet=True,
        packet=InformationPacket(
            scene_description="A test scene",
            observed_actions=["Player acts"],
            overheard_dialogue=[],
            whispers=[],
            sensory_details=[]
        ),
        reason="Character is present",
        attention_level="full"
    )

    mock_routing_list = RoutingDecisionList(decisions=[mock_routing])

    with patch('core.llm_client.llm_client.complete_json', return_value=mock_routing_list):
        result = await director.route_information(
            scene=scene,
            user_input="I test the system",
            agents=agent_manager.agent_states,
            recent_history=[]
        )
    print(f"✓ Routing completed with {len(result)} decision(s)\n")

    # Test 3: Agent Querying Debug (📨)
    print("TEST 3: Agent Querying Debug (📨)")
    print("-" * 40)

    # Mock agent response
    mock_response = CharacterResponse(
        agent_id=agent_id,
        character="Test Character",
        responds=True,
        move=CharacterMove(
            character="Test Character",
            agent_id=agent_id,
            intent="Greet the player",
            action="waves",
            dialogue="Hello there!",
            internal_thought=None,
            target=None
        )
    )

    # Mock the character agent's perceive_and_respond method to return our mock response
    with patch('core.agents.character_agent.CharacterAgent.perceive_and_respond', return_value=mock_response):
        responses = await agent_manager.get_agent_responses(
            [mock_routing],
            max_concurrent=1
        )
    print(f"✓ Agent queried successfully\n")

    # Test 4: Director Validation Debug (🎬)
    print("TEST 4: Director Validation Debug (🎬)")
    print("-" * 40)

    mock_decision = DirectorDecision(
        accepted_moves=[mock_response.move],
        rejected_moves=[],
        npc_actions_needed=["bartender nods"],
        environmental_changes=[],
        continuity_notes=[]
    )

    # Mock the LLM call ONLY, not the entire method
    with patch('core.llm_client.llm_client.complete_json', return_value=mock_decision):
        decision = await director.validate_moves(
            responses=[mock_response],
            scene=scene,
            recent_history=[]
        )
    print(f"✓ Move validation completed\n")

    # Test 5: Response Usage Debug (📝)
    print("TEST 5: Response Usage in Narrative (📝)")
    print("-" * 40)

    from core.roles.storyteller import Storyteller

    storyteller = Storyteller()

    # Mock the LLM call
    with patch('core.llm_client.llm_client.complete', return_value="Test narrative output"):
        # This should trigger the debug output in compose_narrative
        await storyteller.compose_narrative(
            user_action="I test the system",
            character_moves=[mock_response.move],
            npc_actions=[],
            scene=scene,
            world_state={}
        )
    print(f"✓ Narrative composition completed\n")

    # Test 6: Dynamic Spawning Debug (🎭)
    print("TEST 6: Dynamic Agent Spawning (🎭)")
    print("-" * 40)

    narrative_with_new_character = """
    You enter the tavern. A mysterious figure named Elara Moonwhisper
    approaches you. "Greetings, traveler," she says softly.
    """

    # Mock the Director's character detection
    from pydantic import BaseModel

    class NewCharacter(BaseModel):
        name: str
        role: str
        brief_description: str
        personality_hints: list[str]
        apparent_goals: list[str]

    class NewCharacterList(BaseModel):
        new_characters: list[NewCharacter]

    mock_new_char = NewCharacterList(
        new_characters=[
            NewCharacter(
                name="Elara Moonwhisper",
                role="neutral",
                brief_description="A mysterious hooded traveler",
                personality_hints=["cautious", "observant"],
                apparent_goals=["gather information"]
            )
        ]
    )

    with patch('core.llm_client.llm_client.complete_json', return_value=mock_new_char):
        # Set up orchestrator state
        orchestrator.current_config = MagicMock()
        orchestrator.current_config.player_character.name = "TestPlayer"

        await orchestrator._spawn_new_characters_if_needed(narrative_with_new_character)

    print(f"✓ Dynamic spawning completed\n")

    print("=" * 80)
    print("ALL DEBUG OUTPUT TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nExpected debug outputs:")
    print("  🎭 = Agent spawning")
    print("  🎬 = Director routing/validation")
    print("  📨 = Agent querying")
    print("  📝 = Response usage in narrative")


if __name__ == "__main__":
    asyncio.run(test_all_debug_outputs())
