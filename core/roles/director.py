"""Director role - routes information and validates character moves."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from core.config import engine_config
from core.llm_client import llm_client
from core.models.schemas import (
    AgentState,
    CharacterConcept,
    CharacterResponse,
    DirectorDecision,
    DirectorValidation,
    InformationPacket,
    RoutingDecision,
    Scene,
)


class Director:
    """The Director routes information and validates character moves."""

    def __init__(self):
        """Initialize the director."""
        self.params = engine_config.director_params
        prompt_path = Path(__file__).parent.parent / "prompts" / "director.txt"
        with open(prompt_path) as f:
            self.system_prompt = f.read()

    async def route_information(
        self,
        scene: Scene,
        user_input: str,
        agents: dict[str, AgentState],
        recent_history: list[str],
    ) -> list[RoutingDecision]:
        """
        Decide which characters should receive information about this turn.

        Args:
            scene: Current scene
            user_input: Player's action
            agents: Available character agents
            recent_history: Recent story events for context

        Returns:
            List of routing decisions for each character
        """
        # Build character summary
        char_summary = []
        for agent_id, agent in agents.items():
            dossier = agent.dossier
            position = "unknown"
            if dossier.name in scene.present_characters:
                position = "present"
            elif dossier.name in scene.nearby_characters:
                position = "nearby"
            else:
                position = "remote"

            char_summary.append(
                f"- {dossier.name} (agent_id: {agent_id}): {position}, "
                f"current goal: {dossier.current_goals[0] if dossier.current_goals else 'none'}"
            )

        history_text = "\n".join(recent_history[-3:]) if recent_history else "Story just beginning"

        prompt = f"""Decide which characters should be aware of this event and what they perceive.

SCENE: {scene.where}
Present in scene: {', '.join(scene.present_characters)}
Nearby (can potentially observe): {', '.join(scene.nearby_characters) if scene.nearby_characters else 'None'}

PLAYER ACTION: {user_input}

CHARACTERS:
{chr(10).join(char_summary)}

RECENT CONTEXT:
{history_text}

For each character, decide:
1. Should they receive information about this event?
2. If yes, what specifically do they observe?
3. What is their attention level? (full/partial/peripheral)

Consider:
- Characters present in the scene get full information
- Nearby characters might overhear or glimpse things (partial)
- Remote characters normally don't perceive anything unless they have special abilities
- Character goals and abilities might affect what they notice

Create an InformationPacket for each character who receives information.

Return JSON:
{{
  "decisions": [
    {{
      "character": "Character Name",
      "agent_id": "agent_xxx",
      "receives_packet": true/false,
      "packet": {{
        "scene_description": "What they see of the scene",
        "observed_actions": ["What actions they observe"],
        "overheard_dialogue": ["What they hear"],
        "whispers": [],
        "sensory_details": ["Smells, sounds, etc."]
      }} or null,
      "reason": "Why they do/don't receive info",
      "attention_level": "full/partial/peripheral"
    }}
  ]
}}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        from pydantic import BaseModel

        class RoutingDecisionList(BaseModel):
            decisions: list[RoutingDecision]

        result = await llm_client.complete_json(messages, self.params, RoutingDecisionList)

        # Debug output for routing decisions
        if engine_config.debug_agent_activity:
            print(f"[DEBUG] 🎬 Director routing: {len(result.decisions)} character(s) evaluated")
            for decision in result.decisions:
                if decision.receives_packet:
                    print(f"[DEBUG]    ✓ {decision.character} ({decision.attention_level}) - {decision.reason}")
                else:
                    print(f"[DEBUG]    ✗ {decision.character} (no info) - {decision.reason}")

        return result.decisions

    async def validate_moves(
        self,
        responses: list[CharacterResponse],
        scene: Scene,
        recent_history: list[str],
    ) -> DirectorDecision:
        """
        Validate character moves and create final decision.

        Args:
            responses: Character agent responses
            scene: Current scene
            recent_history: Recent events for context

        Returns:
            Director's final decision on accepted/rejected moves
        """
        # Extract moves from responses
        moves = [r.move for r in responses if r.responds and r.move]

        if not moves:
            # No moves to validate
            return DirectorDecision(
                accepted_moves=[],
                rejected_moves=[],
                npc_actions_needed=[],
            )

        moves_text = "\n".join(
            [
                f"- {move.character}: {move.intent}"
                + (f" | Action: {move.action}" if move.action else "")
                + (f" | Says: '{move.dialogue}'" if move.dialogue else "")
                for move in moves
            ]
        )

        history_text = "\n".join(recent_history[-3:]) if recent_history else "Story just beginning"

        prompt = f"""Validate these character moves and decide which to accept.

SCENE: {scene.where} - {scene.atmosphere}
Present: {', '.join(scene.present_characters)}
Facts: {', '.join(scene.facts) if scene.facts else 'None established'}

RECENT CONTEXT:
{history_text}

PROPOSED MOVES:
{moves_text}

For each move, determine:
1. Is it physically possible given the scene?
2. Does it contradict established facts?
3. If multiple moves conflict, which takes priority?
4. What NPC reactions are needed?
5. Are there environmental changes?

Accept moves that are consistent and possible.
Reject moves that break continuity or physics.
Identify needed NPC reactions (guards, crowds, servants, etc.).

Return JSON:
{{
  "accepted_moves": [
    {{
      "character": "Name",
      "agent_id": "agent_xxx",
      "intent": "intent",
      "action": "action or null",
      "dialogue": "dialogue or null",
      "internal_thought": "thought or null",
      "target": "target or null"
    }}
  ],
  "rejected_moves": [
    {{
      "move": {{...}},
      "valid": false,
      "reason": "Why rejected",
      "edit_suggestion": "How to fix"
    }}
  ],
  "npc_actions_needed": ["NPC reaction 1", "NPC reaction 2"],
  "environmental_changes": [],
  "continuity_notes": []
}}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        decision = await llm_client.complete_json(messages, self.params, DirectorDecision)

        # Debug output for move validation
        if engine_config.debug_agent_activity:
            print(f"[DEBUG] 🎬 Director validation: {len(decision.accepted_moves)} accepted, {len(decision.rejected_moves)} rejected")
            for move in decision.accepted_moves:
                action_str = f" | Action: {move.action}" if move.action else ""
                dialogue_str = f" | Says: \"{move.dialogue}\"" if move.dialogue else ""
                print(f"[DEBUG]    ✓ {move.character}: {move.intent}{action_str}{dialogue_str}")
            for validation in decision.rejected_moves:
                print(f"[DEBUG]    ✗ {validation.move.character}: {validation.reason}")
            if decision.npc_actions_needed:
                print(f"[DEBUG]    NPC actions needed: {', '.join(decision.npc_actions_needed)}")

        return decision

    async def update_scene_from_narrative(
        self,
        narrative: str,
        user_action: str,
        current_scene: Scene,
        available_agents: dict[str, "AgentState"],
    ) -> Scene:
        """
        Analyze narrative and update scene with present/nearby characters.

        Args:
            narrative: The composed narrative text
            user_action: Player's action that led to this narrative
            current_scene: Current scene state
            available_agents: All available character agents

        Returns:
            Updated scene with correct character positions
        """
        # Build agent summary
        agent_list = []
        for agent_id, agent_state in available_agents.items():
            agent_list.append({
                "name": agent_state.dossier.name,
                "role": agent_state.dossier.character_concept.role,
                "description": agent_state.dossier.character_concept.description,
            })

        agent_summary = "\n".join([
            f"- {a['name']} ({a['role']}): {a['description']}"
            for a in agent_list
        ])

        prompt = f"""Analyze this narrative and determine which characters are present or nearby.

NARRATIVE:
{narrative}

PLAYER ACTION: {user_action}

CURRENT SCENE: {current_scene.where}

AVAILABLE CHARACTERS:
{agent_summary}

Based on the narrative, determine:
1. Which characters are PRESENT (actively in the scene, interacting, speaking)
2. Which characters are NEARBY (mentioned as nearby, could potentially join, observing from distance)
3. Which characters are mentioned but NOT physically present (memories, references, etc.)

IMPORTANT:
- Match narrative descriptions to character names (e.g., "a group of female students" might be Elara, Seraphine, Elysia)
- If narrative describes people generically but existing characters fit, include them
- If narrative says player approaches a character, that character becomes present
- Always include the player character in present_characters
- Be generous - if unsure whether someone is present, include them

Return JSON:
{{
  "scene_id": "{current_scene.scene_id}",
  "where": "{current_scene.where}",
  "when": "{current_scene.when}",
  "atmosphere": "{current_scene.atmosphere}",
  "present_characters": ["Character 1", "Character 2"],
  "nearby_characters": ["Character 3"],
  "facts": {list(current_scene.facts)}
}}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        from core.models.schemas import Scene
        updated_scene = await llm_client.complete_json(messages, self.params, Scene)

        # Debug output
        if engine_config.debug_agent_activity:
            added_present = set(updated_scene.present_characters) - set(current_scene.present_characters)
            removed_present = set(current_scene.present_characters) - set(updated_scene.present_characters)

            if added_present or removed_present:
                print(f"[DEBUG] 🎬 Scene update:")
                if added_present:
                    print(f"[DEBUG]    + Present: {', '.join(added_present)}")
                if removed_present:
                    print(f"[DEBUG]    - Left: {', '.join(removed_present)}")
                if updated_scene.nearby_characters:
                    print(f"[DEBUG]    ~ Nearby: {', '.join(updated_scene.nearby_characters)}")

        return updated_scene
