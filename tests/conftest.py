"""Pytest configuration and fixtures."""

import pytest

from core.models.schemas import (
    CharacterConcept,
    PlayerCharacter,
    Scene,
    StoryConfig,
    StoryPreferences,
)


@pytest.fixture
def sample_player_character():
    """Sample player character for testing."""
    return PlayerCharacter(
        name="Eleanor Blackwood",
        background="A former court physician who discovered a conspiracy",
        traits=["analytical", "cautious", "compassionate"],
        motivations=["uncover the truth", "protect the innocent"],
        appearance="Tall with silver-streaked dark hair, sharp green eyes",
        skills=["medicine", "diplomacy", "observation"],
    )


@pytest.fixture
def sample_story_preferences():
    """Sample story preferences for testing."""
    return StoryPreferences(
        genre="Political intrigue",
        tone="witty",
        themes=["betrayal", "redemption"],
        length="short",
    )


@pytest.fixture
def sample_story_config(sample_player_character, sample_story_preferences):
    """Sample story config for testing."""
    return StoryConfig(
        player_character=sample_player_character, preferences=sample_story_preferences
    )


@pytest.fixture
def sample_character_concept():
    """Sample character concept for testing."""
    return CharacterConcept(
        name="Lord Ashford",
        role="antagonist",
        description="Ambitious noble with secrets",
        personality=["calculating", "charismatic", "ruthless"],
        goals=["seize power", "eliminate threats"],
        secrets=["embezzling from the treasury"],
        relationship_to_player="suspicious adversary",
    )


@pytest.fixture
def sample_scene():
    """Sample scene for testing."""
    return Scene(
        scene_id="opening",
        when="Early morning, mist rising",
        where="The royal physician's quarters",
        atmosphere="Tense and uncertain",
        present_characters=["Eleanor Blackwood"],
        nearby_characters=["Lord Ashford"],
        ongoing_events=["Investigation beginning"],
        facts=["A ledger has gone missing", "The king is ill"],
    )
