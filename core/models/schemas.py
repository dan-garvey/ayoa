"""Core data schemas for the multi-agent narrative game."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PlayerCharacter(BaseModel):
    """The player's character definition."""

    name: str
    background: str
    traits: list[str]
    motivations: list[str]
    appearance: str
    skills: list[str] = Field(default_factory=list)
    relationships: dict[str, str] = Field(default_factory=dict)  # name -> relationship


class StoryPreferences(BaseModel):
    """Player's preferences for story generation."""

    genre: str  # "Regency-fantasy intrigue"
    tone: str  # "witty", "dark", "adventurous"
    themes: list[str] = Field(default_factory=list)  # "betrayal", "redemption"
    length: Literal["short", "medium", "long"] = "short"
    content_boundaries: list[str] = Field(default_factory=list)  # MVP: stored but not enforced


class StoryConfig(BaseModel):
    """Complete configuration for a story."""

    player_character: PlayerCharacter
    preferences: StoryPreferences
    temperatures: dict[str, float] = Field(
        default_factory=lambda: {
            "director": 0.2,
            "storyteller": 0.7,
            "character_default": 0.7,
        }
    )
    seed: int = 1337


class CharacterConcept(BaseModel):
    """High-level concept for a character to be spawned as an agent."""

    name: str
    role: str  # "antagonist", "romantic interest", "rival", "ally"
    description: str
    personality: list[str]
    goals: list[str]
    secrets: list[str] = Field(default_factory=list)
    relationship_to_player: str


class StoryOutline(BaseModel):
    """Generated story structure."""

    premise: str
    acts: list[str]  # High-level act descriptions
    major_characters: list[CharacterConcept]
    key_locations: list[str]
    potential_endings: list[str]


class Scene(BaseModel):
    """Current scene state."""

    scene_id: str
    when: str
    where: str
    atmosphere: str
    present_characters: list[str]  # Actually in scene
    nearby_characters: list[str] = Field(default_factory=list)  # Could overhear/observe
    ongoing_events: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)


class StyleCard(BaseModel):
    """Character's speech and personality style."""

    voice: list[str]  # "formal", "sarcastic", "warm"
    speech_patterns: list[str]
    taboos: list[str] = Field(default_factory=list)
    catchphrases: list[str] = Field(default_factory=list)
    temperature_override: Optional[float] = None  # 0.5-0.9


class Dossier(BaseModel):
    """Complete character state and identity."""

    name: str
    agent_id: str  # Unique identifier for agent
    character_concept: CharacterConcept
    style_card: StyleCard
    beliefs: list[str] = Field(default_factory=list)
    current_goals: list[str] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)  # Key events they remember
    relationships: dict[str, str] = Field(default_factory=dict)  # name -> current stance
    emotional_state: str = "neutral"


class InformationPacket(BaseModel):
    """What a character learns this turn."""

    scene_description: str
    observed_actions: list[str]
    overheard_dialogue: list[str]
    whispers: list[str] = Field(default_factory=list)  # Rumors or private info
    sensory_details: list[str] = Field(default_factory=list)  # Smells, sounds, etc


class RoutingDecision(BaseModel):
    """Director's decision on who gets what information."""

    character: str
    agent_id: str
    receives_packet: bool
    packet: Optional[InformationPacket] = None
    reason: str  # "present", "eavesdropping", "spy network", "magical scrying"
    attention_level: Literal["full", "partial", "peripheral"] = "full"


class CharacterMove(BaseModel):
    """A character's proposed action."""

    character: str
    agent_id: str
    intent: str  # "deflect", "charm", "investigate", "escape"
    action: Optional[str] = None  # Physical action
    dialogue: Optional[str] = None  # Spoken words
    internal_thought: Optional[str] = None  # Private thought
    target: Optional[str] = None  # Who/what they're focusing on


class CharacterResponse(BaseModel):
    """Wrapper for character agent responses."""

    character: str
    agent_id: str
    responds: bool  # False if choosing not to act
    move: Optional[CharacterMove] = None
    observes_only: bool = False
    observation_notes: Optional[str] = None


class DirectorValidation(BaseModel):
    """Director's validation of a character move."""

    move: CharacterMove
    valid: bool
    reason: Optional[str] = None  # If rejected
    edit_suggestion: Optional[str] = None  # How to fix it


class DirectorDecision(BaseModel):
    """Director's final decision after character responses."""

    accepted_moves: list[CharacterMove]
    rejected_moves: list[DirectorValidation]
    npc_actions_needed: list[str]  # "guard reacts", "crowd gasps"
    environmental_changes: list[str] = Field(default_factory=list)  # "starts raining", etc
    continuity_notes: list[str] = Field(default_factory=list)


class StoryOutput(BaseModel):
    """Final narrative output for a turn."""

    narrative: str  # The composed prose
    visible_moves: list[CharacterMove]  # What actually happened
    scene_update: Optional[Scene] = None  # If scene changed
    continuity_flags: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """Persistent state for a character agent."""

    agent_id: str
    dossier: Dossier
    turn_memory: list[dict] = Field(default_factory=list)  # Last N turns of context
    active: bool = True
    last_action_turn: int = 0
