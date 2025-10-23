"""Character role - provides the base logic for character agent responses."""

from pathlib import Path

from core.config import engine_config
from core.llm_client import llm_client
from core.models.schemas import CharacterResponse, Dossier, InformationPacket


class CharacterRole:
    """Base role for character agents to generate responses."""

    def __init__(self):
        """Initialize the character role."""
        self.params = engine_config.character_default_params
        prompt_path = Path(__file__).parent.parent / "prompts" / "character.txt"
        with open(prompt_path) as f:
            self.system_prompt = f.read()

    async def generate_response(
        self,
        dossier: Dossier,
        packet: InformationPacket,
        attention_level: str,
        context_memory: list[dict],
    ) -> CharacterResponse:
        """
        Generate a character's response to information.

        Args:
            dossier: Character's identity and state
            packet: Information the character perceives
            attention_level: How focused they are (full/partial/peripheral)
            context_memory: Recent turns for context

        Returns:
            Character's response (may choose not to act)
        """
        # Build character context
        char_context = f"""CHARACTER: {dossier.name}
Role: {dossier.character_concept.role}
Personality: {', '.join(dossier.character_concept.personality)}
Current Goals: {', '.join(dossier.current_goals) if dossier.current_goals else 'None specified'}
Emotional State: {dossier.emotional_state}

STYLE:
Voice: {', '.join(dossier.style_card.voice)}
Speech Patterns: {', '.join(dossier.style_card.speech_patterns)}
{'Catchphrases: ' + ', '.join(dossier.style_card.catchphrases) if dossier.style_card.catchphrases else ''}

BELIEFS:
{chr(10).join('- ' + b for b in dossier.beliefs) if dossier.beliefs else 'None established yet'}

SECRETS:
{chr(10).join('- ' + s for s in dossier.character_concept.secrets) if dossier.character_concept.secrets else 'None'}

RELATIONSHIPS:
{chr(10).join(f'- {name}: {stance}' for name, stance in dossier.relationships.items()) if dossier.relationships else 'None established yet'}"""

        # Build recent context
        context_text = ""
        if context_memory:
            recent = context_memory[-3:]  # Last 3 turns
            context_text = "RECENT EVENTS:\n" + "\n".join(
                [f"Turn {i+1}: {turn.get('summary', 'Event occurred')}" for i, turn in enumerate(recent)]
            )

        # Build perception
        perception = f"""WHAT YOU PERCEIVE (attention level: {attention_level}):

Scene: {packet.scene_description}

Observed Actions:
{chr(10).join('- ' + a for a in packet.observed_actions) if packet.observed_actions else '- Nothing notable'}

Overheard Dialogue:
{chr(10).join('- "' + d + '"' for d in packet.overheard_dialogue) if packet.overheard_dialogue else '- No dialogue'}

{('Whispers/Rumors: ' + chr(10).join('- ' + w for w in packet.whispers)) if packet.whispers else ''}

{('Sensory Details: ' + chr(10).join('- ' + s for s in packet.sensory_details)) if packet.sensory_details else ''}"""

        prompt = f"""{char_context}

{context_text}

{perception}

Based on your character's personality, goals, and current state, decide how to respond.

You may:
1. Take physical action (describe it)
2. Speak (keep it brief and in character)
3. Think privately (internal monologue)
4. Simply observe without acting (responds: false)

Consider:
- Does this situation relate to your goals?
- How does your personality guide your response?
- What do your relationships suggest?
- Should you reveal information or keep secrets?
- Sometimes silence is the wisest choice

Return ONLY valid JSON:
{{
  "character": "{dossier.name}",
  "agent_id": "{dossier.agent_id}",
  "responds": true or false,
  "move": {{
    "character": "{dossier.name}",
    "agent_id": "{dossier.agent_id}",
    "intent": "Your intent (e.g., 'deflect', 'charm', 'investigate')",
    "action": "Physical action or null",
    "dialogue": "What you say or null",
    "internal_thought": "Private thought or null",
    "target": "Who/what you're focusing on or null"
  }} or null,
  "observes_only": true/false,
  "observation_notes": "What you notice (if observing only) or null"
}}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Use character-specific temperature if available
        params = self.params
        if dossier.style_card.temperature_override:
            params = RoleParams(
                temperature=dossier.style_card.temperature_override,
                top_p=params.top_p,
                max_tokens=params.max_tokens,
                json_mode=params.json_mode,
            )

        from core.config import RoleParams

        response = await llm_client.complete_json(messages, params, CharacterResponse)
        return response
