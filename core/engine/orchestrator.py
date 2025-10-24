"""Orchestrator for story initialization and turn processing."""

import json
import uuid
from pathlib import Path
from typing import Optional, Tuple

from core.agents.manager import AgentManager
from core.config import engine_config
from core.engine.information_router import InformationRouter
from core.models.schemas import (
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

        # Spawn character agents
        agent_ids = await self.agent_manager.spawn_agents(self.current_outline.major_characters)

        # Create opening scene
        opening_scene = await self.storyteller.create_opening_scene(
            self.current_outline, self.current_config.player_character
        )

        self.current_scene = opening_scene

        # Compose opening narrative
        opening_output = await self.storyteller.compose_opening(opening_scene, self.current_outline)

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

        # 5. Update agent memories
        turn_data = {
            "turn": turn_number,
            "user_action": user_input,
            "narrative": story_output.narrative,
            "summary": f"Player: {user_input[:50]}...",
        }

        for agent_id in self.agent_manager.agents.keys():
            self.agent_manager.update_agent_memory(agent_id, turn_data)

        # 6. Update scene if provided
        if story_output.scene_update:
            self.current_scene = story_output.scene_update

        # 7. Record history
        self.turn_history.append(turn_data)

        # 8. Persist
        self._save_story_state(story_id)

        return story_output

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
