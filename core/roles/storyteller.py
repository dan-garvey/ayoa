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

        # Omniscient narrative memory
        self.conversation_history: list[dict] = []  # Full narrative history
        self.world_context: Optional[dict] = None  # Generated world details
        self.max_history_turns: int = engine_config.storyteller_max_history_turns

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

    async def generate_world_context(
        self, outline: StoryOutline, config: StoryConfig
    ) -> dict:
        """
        Generate detailed world-building context before story starts.

        Args:
            outline: Story outline
            config: Story configuration

        Returns:
            Dictionary with world details (culture, history, rules, etc.)
        """
        player = config.player_character
        prefs = config.preferences

        prompt = f"""Generate detailed world-building for this interactive story.

PREMISE: {outline.premise}
GENRE: {prefs.genre}
TONE: {prefs.tone}
KEY LOCATIONS: {', '.join(outline.key_locations)}
MAJOR CHARACTERS: {', '.join([c.name for c in outline.major_characters])}

Create comprehensive world context with:

1. CULTURAL CONTEXT: Social norms, customs, important traditions
2. HISTORICAL BACKGROUND: Recent events that set the stage (last 50-100 years)
3. RULES OF THE WORLD: Magic system, technology level, what's possible/impossible
4. KEY FACTIONS: Political groups, organizations, their goals and conflicts
5. IMPORTANT LOCATIONS: Detailed descriptions of the key locations
6. ESTABLISHED FACTS: 10-15 true facts about this world that must remain consistent
7. TONE GUIDELINES: Specific stylistic notes for maintaining {prefs.tone} tone
8. NPCS AND BACKGROUND CHARACTERS: Types of people who might appear

Return JSON with these sections. Be specific and detailed - this will ensure consistency throughout the story.

Example structure:
{{
  "cultural_context": "Detailed paragraph about society...",
  "historical_background": "Recent history that matters...",
  "world_rules": {{"magic": "...", "technology": "...", "limitations": "..."}},
  "factions": [{{"name": "...", "goals": "...", "conflict": "..."}}],
  "locations": {{"location_name": "detailed description..."}},
  "established_facts": ["fact 1", "fact 2", ...],
  "tone_guidelines": ["guideline 1", "guideline 2", ...],
  "npc_types": ["type 1", "type 2", ...]
}}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Use higher temperature for creativity in world-building
        import json

        response = await llm_client.complete(messages, self.params)

        # Parse the JSON response
        try:
            # Extract JSON from response
            import re

            json_match = re.search(r"(\{.*\})", response, re.DOTALL)
            if json_match:
                world_context = json.loads(json_match.group(1))
            else:
                world_context = json.loads(response)

            self.world_context = world_context
            return world_context
        except json.JSONDecodeError:
            # Fallback to basic structure
            self.world_context = {
                "cultural_context": "Details to be established during play",
                "established_facts": [],
            }
            return self.world_context

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
        import json

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

        # Initialize conversation history with world context
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Add world context if available
        if self.world_context:
            world_context_str = json.dumps(self.world_context, indent=2)
            messages.append(
                {
                    "role": "system",
                    "content": f"WORLD CONTEXT (maintain consistency with these details):\n{world_context_str}",
                }
            )

        messages.append({"role": "user", "content": prompt})

        narrative = await llm_client.complete(messages, self.params)

        # Initialize conversation history
        self.conversation_history = [
            {"role": "user", "content": f"[OPENING SCENE]\n{prompt}"},
            {"role": "assistant", "content": narrative},
        ]

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
            character_moves: Accepted character moves (ephemeral - used only for this turn)
            npc_actions: Required NPC reactions
            scene: Current scene
            world_state: Current world state

        Returns:
            Composed narrative output
        """
        import json

        # Build current turn context (character moves used here, but NOT stored in history)
        moves_text = "\n".join(
            [
                f"- {move.character}: {move.intent}"
                + (f" (says: '{move.dialogue}')" if move.dialogue else "")
                + (f" (does: {move.action})" if move.action else "")
                for move in character_moves
            ]
        )

        npc_text = "\n".join([f"- {action}" for action in npc_actions]) if npc_actions else "None"

        # Current turn prompt (includes character moves as ephemeral input)
        current_turn_prompt = f"""Compose the narrative for this turn.

SCENE: {scene.where} - {scene.atmosphere}
PRESENT: {', '.join(scene.present_characters)}

PLAYER ACTION: {user_action}

CHARACTER RESPONSES (for this turn only):
{moves_text if character_moves else "None - characters observe silently"}

NPC REACTIONS NEEDED:
{npc_text}

Write 200-500 words of narrative that:
- Describes the player's action and its immediate effects
- Integrates character responses naturally (preserve exact dialogue!)
- Shows NPC reactions as needed
- Maintains the scene's atmosphere and continuity with previous narrative
- Uses third-person past tense
- Ends on a natural pause for the next player input

Preserve character dialogue EXACTLY as provided. Describe actions cinematically."""

        # Build messages with full conversation history
        messages = self._build_messages_with_history(current_turn_prompt)

        # Generate narrative
        narrative = await llm_client.complete(messages, self.params)

        # Update history (ONLY user action and narrative - character moves are discarded)
        self._add_to_history(user_action, narrative)

        return StoryOutput(
            narrative=narrative,
            visible_moves=character_moves,
            scene_update=None,  # Scene updates handled separately
        )

    def _build_messages_with_history(self, current_prompt: str) -> list[dict]:
        """
        Build LLM messages including full conversation history.

        Args:
            current_prompt: Current turn prompt

        Returns:
            List of messages for LLM
        """
        import json

        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Add world context once at the beginning (if available)
        if self.world_context:
            world_context_str = json.dumps(self.world_context, indent=2)
            messages.append(
                {
                    "role": "system",
                    "content": f"WORLD CONTEXT (maintain consistency with these details):\n{world_context_str}",
                }
            )

        # Add full conversation history (previous user inputs + narratives)
        messages.extend(self.conversation_history)

        # Add current turn
        messages.append({"role": "user", "content": current_prompt})

        return messages

    def _add_to_history(self, user_action: str, narrative: str):
        """
        Add turn to conversation history.

        Only stores: user action + generated narrative.
        Character moves are NOT stored (used only as ephemeral input).

        Args:
            user_action: What the player did
            narrative: The generated narrative
        """
        self.conversation_history.append(
            {"role": "user", "content": f"PLAYER ACTION: {user_action}"}
        )
        self.conversation_history.append({"role": "assistant", "content": narrative})

        # Truncate if needed
        self._truncate_history_if_needed()

    def _truncate_history_if_needed(self):
        """
        Truncate conversation history to stay within max_history_turns.

        Keeps the most recent turns within the limit.
        """
        # Each turn = 2 messages (user + assistant)
        max_messages = self.max_history_turns * 2

        if len(self.conversation_history) > max_messages:
            # Keep only the most recent turns
            self.conversation_history = self.conversation_history[-max_messages:]

    def set_max_history_turns(self, max_turns: int):
        """
        Set the maximum number of turns to keep in history.

        Args:
            max_turns: Maximum turns to maintain (will be doubled for user+assistant pairs)
        """
        self.max_history_turns = max_turns
