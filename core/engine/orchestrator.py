"""Orchestrator for story initialization and turn processing."""

import json
import uuid
from pathlib import Path
from typing import Optional, Tuple

from core.agents.manager import AgentManager
from core.config import engine_config
from core.engine.information_router import InformationRouter
from core.llm_client import llm_client
from core.models.schemas import (
    CharacterConcept,
    Scene,
    StoryConfig,
    StoryOutline,
    StoryOutput,
)
from core.roles.director import Director
from core.roles.storyteller import Storyteller


class Orchestrator:
    """Orchestrates story creation and turn-by-turn gameplay."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.storyteller = Storyteller()
        self.director = Director()
        self.agent_manager = AgentManager()
        self.router = InformationRouter(self.director)

        # Story state
        self.current_story_id: Optional[str] = None
        self.current_scene: Optional[Scene] = None
        self.current_outline: Optional[StoryOutline] = None
        self.current_config: Optional[StoryConfig] = None
        self.turn_history: list[dict] = []
        self.world_state: dict = {}

    async def create_story(self, config: StoryConfig) -> Tuple[str, StoryOutline]:
        """
        Phase 1: Generate story outline.

        Args:
            config: Story configuration with character and preferences

        Returns:
            Tuple of (story_id, outline)
        """
        # Create story ID using character name + hash
        char_name = config.player_character.name.lower().replace(" ", "_")
        # Remove non-alphanumeric characters except underscore
        char_name = "".join(c for c in char_name if c.isalnum() or c == "_")
        story_id = f"{char_name}_{uuid.uuid4().hex[:8]}"

        # Save configuration IMMEDIATELY (before any LLM calls)
        # This ensures user preferences are persisted even if outline generation fails
        self.current_story_id = story_id
        self.current_outline = None  # Will be filled in after generation
        self.current_config = config
        self.turn_history = []
        self.world_state = {
            "story_id": story_id,
            "player_name": config.player_character.name,
            "genre": config.preferences.genre,
            "tone": config.preferences.tone,
        }

        # Persist preferences to disk BEFORE generating outline
        self._save_story_state(story_id)

        # Now generate outline (this can fail without losing user's preferences)
        outline = await self.storyteller.generate_outline(config)

        # Update with generated outline and save again
        self.current_outline = outline
        self._save_story_state(story_id)

        return story_id, outline

    async def start_story(self, story_id: str) -> StoryOutput:
        """
        Phase 2: Spawn agents and create opening.

        Args:
            story_id: Story to start

        Returns:
            Opening narrative
        """
        # Load story state if not current
        if self.current_story_id != story_id:
            self._load_story_state(story_id)

        if not self.current_outline or not self.current_config:
            raise ValueError(f"Story {story_id} not found or not properly configured")

        # Generate detailed world context (NEW - ensures consistency)
        print("Generating world context...")
        world_context = await self.storyteller.generate_world_context(
            self.current_outline, self.current_config
        )
        self.world_state["world_context"] = world_context

        # Spawn character agents (EXCLUDE player character - they're controlled by the user!)
        npc_characters = [
            char for char in self.current_outline.major_characters
            if char.name.lower() != self.current_config.player_character.name.lower()
        ]
        agent_ids = await self.agent_manager.spawn_agents(npc_characters)

        # Create opening scene
        opening_scene = await self.storyteller.create_opening_scene(
            self.current_outline, self.current_config.player_character
        )

        self.current_scene = opening_scene

        # Compose opening narrative
        opening_output = await self.storyteller.compose_opening(opening_scene, self.current_outline)

        # Update scene based on opening narrative (add any characters mentioned)
        self.current_scene = await self.director.update_scene_from_narrative(
            narrative=opening_output.narrative,
            user_action="Story begins",
            current_scene=self.current_scene,
            available_agents=self.agent_manager.agent_states,
        )

        # Record in history
        self.turn_history.append(
            {
                "turn": 0,
                "type": "opening",
                "narrative": opening_output.narrative,
                "summary": "Story begins",
            }
        )

        # Persist
        self._save_story_state(story_id)

        return opening_output

    def _is_question(self, user_input: str) -> bool:
        """
        Detect if user input is a question rather than an action.

        Args:
            user_input: User's input

        Returns:
            True if this appears to be a question
        """
        input_lower = user_input.lower().strip()

        # Question markers
        if input_lower.endswith("?"):
            return True

        # Common question starters
        question_words = [
            "who", "what", "where", "when", "why", "how",
            "can i", "could i", "should i", "would",
            "is there", "are there", "do i", "does",
            "am i", "will i", "have i"
        ]

        for word in question_words:
            if input_lower.startswith(word + " "):
                return True

        return False

    async def _handle_question(self, question: str, story_id: str) -> StoryOutput:
        """
        Handle a question from the player without progressing the narrative.

        Args:
            question: Player's question
            story_id: Active story

        Returns:
            Answer without advancing the story
        """
        import json

        # Build context about current scene and recent events
        recent_narrative = self.storyteller.conversation_history[-2:] if self.storyteller.conversation_history else []
        recent_text = "\n".join([msg["content"] for msg in recent_narrative])

        prompt = f"""The player has asked a question about the current situation. Answer it directly without progressing the narrative.

CURRENT SCENE:
Where: {self.current_scene.where}
When: {self.current_scene.when}
Atmosphere: {self.current_scene.atmosphere}
Present characters: {', '.join(self.current_scene.present_characters)}
Nearby characters: {', '.join(self.current_scene.nearby_characters) if self.current_scene.nearby_characters else 'None'}

RECENT NARRATIVE:
{recent_text}

PLAYER'S QUESTION: {question}

Provide a direct, informative answer (100-200 words) that:
- Answers the question based on what the player character would know
- Does NOT advance time or the narrative
- Uses second-person present tense ("You see..." "You notice...")
- References details from the current scene and recent events
- If the player wouldn't know something, say so clearly

This is purely informational - no new events occur."""

        messages = [
            {"role": "system", "content": self.storyteller.system_prompt},
        ]

        if self.storyteller.world_context:
            world_context_str = json.dumps(self.storyteller.world_context, indent=2)
            messages.append({
                "role": "system",
                "content": f"WORLD CONTEXT:\n{world_context_str}"
            })

        messages.append({"role": "user", "content": prompt})

        answer = await llm_client.complete(messages, self.storyteller.params)

        return StoryOutput(narrative=answer, visible_moves=[], scene_update=None)

    async def process_turn(self, story_id: str, user_input: str) -> StoryOutput:
        """
        Main game loop for each turn.

        Args:
            story_id: Active story
            user_input: Player's action

        Returns:
            Story output for this turn
        """
        # Load story state if not current
        if self.current_story_id != story_id:
            self._load_story_state(story_id)

        if not self.current_scene:
            raise ValueError(f"Story {story_id} has no active scene")

        # Handle meta commands
        if user_input.startswith("/"):
            return await self._handle_meta_command(user_input, story_id)

        # Handle questions without progressing narrative
        if self._is_question(user_input):
            return await self._handle_question(user_input, story_id)

        turn_number = len(self.turn_history) + 1

        # Get recent history for context
        recent_history = [t.get("summary", "") for t in self.turn_history[-5:]]

        # 1. Director routes information to characters
        routing_decisions = await self.router.route_information(
            scene=self.current_scene,
            user_input=user_input,
            agent_registry=self.agent_manager.agent_states,
            recent_events=recent_history,
        )

        # 2. Fan out to character agents
        character_responses = await self.agent_manager.get_agent_responses(
            routing_decisions, max_concurrent=engine_config.max_active_characters_per_turn
        )

        # 3. Director validates and selects moves
        director_decision = await self.director.validate_moves(
            responses=character_responses, scene=self.current_scene, recent_history=recent_history
        )

        # 4. Storyteller composes final narrative
        story_output = await self.storyteller.compose_narrative(
            user_action=user_input,
            character_moves=director_decision.accepted_moves,
            npc_actions=director_decision.npc_actions_needed,
            scene=self.current_scene,
            world_state=self.world_state,
        )

        # 5. Update scene based on narrative (critical for character tracking!)
        self.current_scene = await self.director.update_scene_from_narrative(
            narrative=story_output.narrative,
            user_action=user_input,
            current_scene=self.current_scene,
            available_agents=self.agent_manager.agent_states,
        )

        # 6. Update agent memories
        turn_data = {
            "turn": turn_number,
            "user_action": user_input,
            "narrative": story_output.narrative,
            "summary": f"Player: {user_input[:50]}...",
        }

        for agent_id in self.agent_manager.agents.keys():
            self.agent_manager.update_agent_memory(agent_id, turn_data)

        # 7. Record history
        self.turn_history.append(turn_data)

        # 8. Check for and spawn new characters dynamically
        await self._spawn_new_characters_if_needed(story_output.narrative)

        # 9. Persist
        self._save_story_state(story_id)

        return story_output

    async def _spawn_new_characters_if_needed(self, narrative: str):
        """
        Detect new significant characters in narrative and spawn agents for them.

        Args:
            narrative: The just-generated narrative text
        """
        # Get list of existing character names
        existing_names = [state.dossier.name for state in self.agent_manager.agent_states.values()]
        if self.current_config:
            existing_names.append(self.current_config.player_character.name)

        # Ask Director to identify new characters
        prompt = f"""Analyze this narrative passage and identify any NEW significant characters that are introduced.

EXISTING CHARACTERS (already tracked): {', '.join(existing_names)}

NARRATIVE:
{narrative}

Identify any NEW named characters who:
- Have a name (not just "a guard" or "the innkeeper")
- Speak dialogue or take significant action
- Seem like they could appear again (not just background mentions)

Do NOT include:
- Characters already in the existing list
- The player character
- Generic unnamed NPCs ("a guard", "the bartender")
- Historical or past-tense references to people not present

Return JSON:
{{
  "new_characters": [
    {{
      "name": "Character Name",
      "role": "ally/antagonist/rival/neutral/other",
      "brief_description": "One sentence about who they are",
      "personality_hints": ["trait1", "trait2"],
      "apparent_goals": ["goal1"]
    }}
  ]
}}

If no new significant characters, return: {{"new_characters": []}}"""

        messages = [
            {"role": "system", "content": self.director.system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            from pydantic import BaseModel

            class NewCharacter(BaseModel):
                name: str
                role: str
                brief_description: str
                personality_hints: list[str]
                apparent_goals: list[str]

            class NewCharacterList(BaseModel):
                new_characters: list[NewCharacter]

            result = await llm_client.complete_json(messages, self.director.params, NewCharacterList)

            # Spawn agents for new characters
            from core.config import engine_config
            for char in result.new_characters:
                # Create CharacterConcept
                concept = CharacterConcept(
                    name=char.name,
                    role=char.role,
                    description=char.brief_description,
                    personality=char.personality_hints,
                    goals=char.apparent_goals,
                    secrets=[],  # Will develop over time
                    relationship_to_player="newly met"
                )

                # Spawn agent
                agent_id = await self.agent_manager.spawn_agent(concept)

                if engine_config.debug_agent_activity:
                    print(f"[DEBUG] 🎭 Dynamically spawned agent: {char.name} ({char.role}) [{agent_id}]")

        except Exception as e:
            # Don't fail the turn if character detection fails
            if engine_config.debug_agent_activity:
                print(f"[DEBUG] ⚠️  Failed to detect new characters: {e}")

    async def _handle_meta_command(self, command: str, story_id: str) -> StoryOutput:
        """
        Handle meta commands like /scene, /save, etc.

        Args:
            command: Meta command
            story_id: Active story

        Returns:
            Story output with command response
        """
        if command == "/scene":
            if self.current_scene:
                narrative = f"""CURRENT SCENE:
Where: {self.current_scene.where}
When: {self.current_scene.when}
Atmosphere: {self.current_scene.atmosphere}
Present: {', '.join(self.current_scene.present_characters)}
Nearby: {', '.join(self.current_scene.nearby_characters) if self.current_scene.nearby_characters else 'None'}"""
            else:
                narrative = "No active scene."

            return StoryOutput(narrative=narrative, visible_moves=[])

        elif command == "/save":
            self._save_story_state(story_id)
            return StoryOutput(narrative="Story saved successfully.", visible_moves=[])

        elif command == "/cast":
            agents = self.agent_manager.list_agents()
            cast_list = "\n".join(
                [f"- {state.dossier.name} ({state.dossier.character_concept.role})" for state in agents.values()]
            )
            narrative = f"ACTIVE CHARACTERS:\n{cast_list}" if cast_list else "No characters spawned yet."
            return StoryOutput(narrative=narrative, visible_moves=[])

        else:
            return StoryOutput(narrative=f"Unknown command: {command}", visible_moves=[])

    def _save_story_state(self, story_id: str):
        """Save story state to disk."""
        saves_dir = Path("./saves")
        saves_dir.mkdir(exist_ok=True)

        state = {
            "story_id": story_id,
            "config": self.current_config.model_dump() if self.current_config else None,
            "outline": self.current_outline.model_dump() if self.current_outline else None,
            "scene": self.current_scene.model_dump() if self.current_scene else None,
            "agents": {
                agent_id: state.model_dump()
                for agent_id, state in self.agent_manager.agent_states.items()
            },
            "turn_history": self.turn_history,
            "world_state": self.world_state,
            # NEW: Storyteller omniscient memory
            "storyteller_history": self.storyteller.conversation_history,
            "storyteller_world_context": self.storyteller.world_context,
        }

        save_path = saves_dir / f"{story_id}.json"
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_story_state(self, story_id: str):
        """Load story state from disk."""
        save_path = Path(f"./saves/{story_id}.json")

        if not save_path.exists():
            raise FileNotFoundError(f"No save file found for story {story_id}")

        with open(save_path) as f:
            state = json.load(f)

        # Restore state
        self.current_story_id = story_id

        if state.get("config"):
            self.current_config = StoryConfig.model_validate(state["config"])

        if state.get("outline"):
            self.current_outline = StoryOutline.model_validate(state["outline"])

        if state.get("scene"):
            self.current_scene = Scene.model_validate(state["scene"])

        # Restore agents
        from core.models.schemas import AgentState

        for agent_id, agent_data in state.get("agents", {}).items():
            agent_state = AgentState.model_validate(agent_data)
            self.agent_manager.restore_agent(agent_state)

        self.turn_history = state.get("turn_history", [])
        self.world_state = state.get("world_state", {})

        # NEW: Restore Storyteller omniscient memory
        self.storyteller.conversation_history = state.get("storyteller_history", [])
        self.storyteller.world_context = state.get("storyteller_world_context", None)


# Global orchestrator instance
orchestrator = Orchestrator()
