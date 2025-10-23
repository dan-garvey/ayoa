"""Storyteller role - composes narrative from character moves and world state."""

import os
from pathlib import Path
from typing import Optional

from core.config import engine_config
from core.llm_client import llm_client
from core.models.schemas import (
    CharacterConcept,
    CharacterMove,
    PlayerCharacter,
    Scene,
    StoryConfig,
    StoryOutline,
    StoryOutput,
)


class Storyteller:
    """The Storyteller composes narrative prose and manages world-building."""

    def __init__(self):
        """Initialize the storyteller."""
        self.params = engine_config.storyteller_params
        prompt_path = Path(__file__).parent.parent / "prompts" / "storyteller.txt"
        with open(prompt_path) as f:
            self.system_prompt = f.read()

    async def generate_outline(self, config: StoryConfig) -> StoryOutline:
        """
        Generate a story outline from player preferences.

        Args:
            config: Story configuration with player character and preferences

        Returns:
            Generated story outline
        """
        player = config.player_character
        prefs = config.preferences

        prompt = f"""Create a story outline for an interactive narrative.

PLAYER CHARACTER:
Name: {player.name}
Background: {player.background}
Traits: {', '.join(player.traits)}
Motivations: {', '.join(player.motivations)}

STORY PREFERENCES:
Genre: {prefs.genre}
Tone: {prefs.tone}
Themes: {', '.join(prefs.themes) if prefs.themes else 'None specified'}
Length: {prefs.length}

Generate a story outline with:
1. A compelling premise that incorporates the player character
2. 3-5 act structure appropriate for the story length
3. 2-4 major characters (allies, rivals, antagonists, etc.) who will interact with the player
4. Key locations where the story unfolds
5. 2-3 potential endings based on different paths

Ensure the major characters have clear goals that will create dramatic tension with the player.

Return JSON matching this structure:
{{
  "premise": "One paragraph premise",
  "acts": ["Act 1 description", "Act 2 description", ...],
  "major_characters": [
    {{
      "name": "Character Name",
      "role": "antagonist/ally/rival/romantic interest",
      "description": "Brief description",
      "personality": ["trait1", "trait2"],
      "goals": ["goal1", "goal2"],
      "secrets": ["secret1"],
      "relationship_to_player": "How they relate to player"
    }}
  ],
  "key_locations": ["Location 1", "Location 2"],
  "potential_endings": ["Ending 1", "Ending 2"]
}}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        outline = await llm_client.complete_json(messages, self.params, StoryOutline)
        return outline

    async def create_opening_scene(
        self, outline: StoryOutline, player: PlayerCharacter
    ) -> Scene:
        """
        Create the opening scene of the story.

        Args:
            outline: Story outline
            player: Player character

        Returns:
            Opening scene description
        """
        prompt = f"""Create the opening scene for this story.

PREMISE: {outline.premise}
FIRST ACT: {outline.acts[0] if outline.acts else 'Beginning of the story'}

PLAYER CHARACTER: {player.name} - {player.background}

Create a scene that:
- Sets up the initial situation
- Introduces the player character in their element
- Does NOT include major characters yet (they'll be introduced later)
- Establishes atmosphere and stakes
- Provides clear hooks for player action

Return JSON:
{{
  "scene_id": "opening",
  "when": "Time of day and context",
  "where": "Location description",
  "atmosphere": "Mood and sensory details",
  "present_characters": ["{player.name}"],
  "nearby_characters": [],
  "ongoing_events": ["Event 1", "Event 2"],
  "facts": ["Important fact about the world/situation"]
}}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        scene = await llm_client.complete_json(messages, self.params, Scene)
        return scene

    async def compose_opening(self, scene: Scene, outline: StoryOutline) -> StoryOutput:
        """
        Compose the opening narrative from the scene.

        Args:
            scene: Opening scene
            outline: Story outline for context

        Returns:
            Story output with opening narrative
        """
        prompt = f"""Compose the opening narrative for this interactive story.

SCENE:
When: {scene.when}
Where: {scene.where}
Atmosphere: {scene.atmosphere}
Ongoing: {', '.join(scene.ongoing_events)}

PREMISE: {outline.premise}

Write 300-500 words of engaging third-person past tense narrative that:
- Establishes the setting vividly
- Introduces the player character in action
- Creates hooks and questions that make the player want to explore
- Ends on a moment where the player can naturally make a choice

Do not include dialogue from major characters (they'll be introduced later).
Focus on atmosphere, sensory details, and the player's situation."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        narrative = await llm_client.complete(messages, self.params)

        return StoryOutput(narrative=narrative, visible_moves=[], scene_update=scene)

    async def compose_narrative(
        self,
        user_action: str,
        character_moves: list[CharacterMove],
        npc_actions: list[str],
        scene: Scene,
        world_state: dict,
    ) -> StoryOutput:
        """
        Compose narrative from player action and character responses.

        Args:
            user_action: What the player tried to do
            character_moves: Accepted character moves
            npc_actions: Required NPC reactions
            scene: Current scene
            world_state: Current world state

        Returns:
            Composed narrative output
        """
        # Build context
        moves_text = "\n".join(
            [
                f"- {move.character}: {move.intent}"
                + (f" (says: '{move.dialogue}')" if move.dialogue else "")
                + (f" (does: {move.action})" if move.action else "")
                for move in character_moves
            ]
        )

        npc_text = "\n".join([f"- {action}" for action in npc_actions]) if npc_actions else "None"

        prompt = f"""Compose the narrative for this turn of the story.

SCENE: {scene.where} - {scene.atmosphere}
PRESENT: {', '.join(scene.present_characters)}

PLAYER ACTION: {user_action}

CHARACTER RESPONSES:
{moves_text if character_moves else "None - characters observe silently"}

NPC REACTIONS NEEDED:
{npc_text}

Write 200-500 words of narrative that:
- Describes the player's action and its immediate effects
- Integrates character responses naturally (preserve exact dialogue!)
- Shows NPC reactions as needed
- Maintains the scene's atmosphere
- Uses third-person past tense
- Ends on a natural pause for the next player input

Preserve character dialogue EXACTLY as provided. Describe actions cinematically."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        narrative = await llm_client.complete(messages, self.params)

        return StoryOutput(
            narrative=narrative,
            visible_moves=character_moves,
            scene_update=None,  # Scene updates handled separately
        )
