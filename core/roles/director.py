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
        return decision
