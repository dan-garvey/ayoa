"""Test data model contracts and validation."""

import pytest
from pydantic import ValidationError

from core.models.schemas import (
    CharacterConcept,
    CharacterMove,
    CharacterResponse,
    DirectorDecision,
    Dossier,
    InformationPacket,
    PlayerCharacter,
    RoutingDecision,
    Scene,
    StoryConfig,
    StoryOutline,
    StoryPreferences,
    StyleCard,
)


def test_player_character_valid(sample_player_character):
    """Test valid player character creation."""
    assert sample_player_character.name == "Eleanor Blackwood"
    assert len(sample_player_character.traits) == 3
    assert "medicine" in sample_player_character.skills


def test_player_character_required_fields():
    """Test that required fields are enforced."""
    with pytest.raises(ValidationError):
        PlayerCharacter(name="Test")  # Missing required fields


def test_story_preferences_defaults():
    """Test story preferences with defaults."""
    prefs = StoryPreferences(genre="Fantasy", tone="epic")
    assert prefs.length == "short"  # Default
    assert prefs.themes == []
    assert prefs.content_boundaries == []


def test_story_config_valid(sample_story_config):
    """Test valid story config."""
    assert sample_story_config.player_character.name == "Eleanor Blackwood"
    assert sample_story_config.preferences.genre == "Political intrigue"
    assert sample_story_config.seed == 1337  # Default


def test_character_concept_valid(sample_character_concept):
    """Test valid character concept."""
    assert sample_character_concept.name == "Lord Ashford"
    assert sample_character_concept.role == "antagonist"
    assert len(sample_character_concept.personality) == 3


def test_scene_valid(sample_scene):
    """Test valid scene."""
    assert sample_scene.scene_id == "opening"
    assert "Eleanor Blackwood" in sample_scene.present_characters
    assert "Lord Ashford" in sample_scene.nearby_characters


def test_style_card():
    """Test style card creation."""
    style = StyleCard(
        voice=["formal", "sarcastic"],
        speech_patterns=["tends to use metaphors"],
        temperature_override=0.8,
    )
    assert style.temperature_override == 0.8
    assert len(style.voice) == 2


def test_information_packet():
    """Test information packet."""
    packet = InformationPacket(
        scene_description="A tense meeting",
        observed_actions=["Player enters room"],
        overheard_dialogue=["We must act quickly"],
    )
    assert len(packet.observed_actions) == 1
    assert packet.whispers == []  # Default


def test_routing_decision():
    """Test routing decision."""
    decision = RoutingDecision(
        character="Lord Ashford",
        agent_id="agent_123",
        receives_packet=True,
        packet=None,
        reason="present in scene",
        attention_level="full",
    )
    assert decision.receives_packet is True
    assert decision.attention_level == "full"


def test_character_move():
    """Test character move."""
    move = CharacterMove(
        character="Lord Ashford",
        agent_id="agent_123",
        intent="deflect suspicion",
        action="adjusts his cufflinks",
        dialogue="I assure you, I know nothing of this matter",
        target="Eleanor",
    )
    assert move.intent == "deflect suspicion"
    assert move.dialogue is not None


def test_character_response_no_action():
    """Test character response with no action."""
    response = CharacterResponse(
        character="Lord Ashford",
        agent_id="agent_123",
        responds=False,
        observes_only=True,
        observation_notes="Notes the player's nervousness",
    )
    assert response.responds is False
    assert response.move is None


def test_director_decision():
    """Test director decision."""
    decision = DirectorDecision(
        accepted_moves=[],
        rejected_moves=[],
        npc_actions_needed=["guard steps forward", "crowd murmurs"],
        environmental_changes=["candles flicker"],
    )
    assert len(decision.npc_actions_needed) == 2
    assert len(decision.environmental_changes) == 1
