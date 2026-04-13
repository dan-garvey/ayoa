"""Knowledge isolation validators.

Heuristic checks that agent outputs don't reference information
the character couldn't plausibly know based on their observation set.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile

logger = logging.getLogger(__name__)


@dataclass
class LeakageFlag:
    """A single leakage detection."""
    character_id: str
    field: str  # e.g. "dialogue", "actions"
    leaked_text: str
    reason: str


@dataclass
class ValidationResult:
    """Result of validating an agent output."""
    character_id: str
    passed: bool
    flags: list[LeakageFlag] = field(default_factory=list)


def extract_entities(text: str) -> set[str]:
    """Extract potential entity references from text.

    Looks for capitalized words/phrases that could be character names,
    location names, or specific proper nouns.
    """
    # Find capitalized words (potential names), excluding sentence starters
    # by looking for mid-sentence capitalized words
    entities = set()

    # Multi-word capitalized phrases (e.g., "Captain Vero", "Article Nineteen")
    for match in re.finditer(r'(?<!\. )(?<!\.\n)(?<!^)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text):
        entity = match.group(1)
        if len(entity) > 2:
            entities.add(entity.lower())

    # Also extract all capitalized words individually
    for match in re.finditer(r'\b([A-Z][a-z]{2,})\b', text):
        entities.add(match.group(1).lower())

    return entities


def extract_text_from_output(output: CharacterAgentOutput) -> str:
    """Combine all public-facing text from an agent output."""
    parts = []
    parts.extend(output.public_response.actions)
    parts.extend(output.public_response.dialogue)
    if output.public_response.expression:
        parts.append(output.public_response.expression)
    return " ".join(parts)


def build_known_entities(
    character: CharacterRecord,
    observed_facts: list[str],
    checkpoint: CheckpointFile,
) -> set[str]:
    """Build the set of entities a character could plausibly know about.

    Includes:
    - Their own name and details
    - Names from observed facts
    - Characters they have attitudes toward
    - Their own backstory/memory entities
    - World facts (common knowledge)
    """
    known = set()

    # Character's own details
    known.add(character.name.lower())
    for part in character.name.lower().split():
        if len(part) > 2:
            known.add(part)

    # Entities from observed facts
    for fact in observed_facts:
        known |= extract_entities(fact)
        # Also add lowercased words from facts
        for word in fact.lower().split():
            if len(word) > 3:
                known.add(word)

    # Characters they know about (from attitudes)
    for target_id in character.private_state.attitudes:
        known.add(target_id.replace("_", " "))
        for part in target_id.split("_"):
            if len(part) > 2:
                known.add(part)
        # Also find the actual character name
        for ch in checkpoint.characters:
            if ch.character_id == target_id:
                known.add(ch.name.lower())
                for part in ch.name.lower().split():
                    if len(part) > 2:
                        known.add(part)

    # Character's own backstory entities
    known |= extract_entities(character.backstory)
    known |= extract_entities(character.personality)

    # Their episodic memories
    for mem in character.memory.episodic:
        known |= extract_entities(mem)

    # World facts are common knowledge
    for fact in checkpoint.world_state.facts:
        known |= extract_entities(fact)

    # Setting info is common knowledge
    known |= extract_entities(checkpoint.world_state.setting.premise)

    return known


def validate_agent_output(
    output: CharacterAgentOutput,
    character: CharacterRecord,
    observed_facts: list[str],
    checkpoint: CheckpointFile,
) -> ValidationResult:
    """Check an agent output for potential knowledge leakage.

    Extracts entities from the agent's public response and checks them
    against the set of entities the character could plausibly know.
    Flags references to unknown entities.
    """
    result = ValidationResult(character_id=output.character_id, passed=True)

    output_text = extract_text_from_output(output)
    output_entities = extract_entities(output_text)
    known_entities = build_known_entities(character, observed_facts, checkpoint)

    # Find entities in output that aren't in the known set
    unknown = output_entities - known_entities

    # Filter out common words that aren't really entity leaks
    common_words = {
        "the", "and", "but", "for", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "are", "has", "his", "how",
        "its", "may", "new", "now", "old", "see", "way", "who", "did",
        "get", "let", "say", "she", "too", "use", "sir", "yes",
    }
    unknown -= common_words

    # Check each unknown entity for character name references
    for entity in unknown:
        # Check if it references unobserved character names
        for ch in checkpoint.characters:
            if ch.character_id == character.character_id:
                continue
            name_parts = {p.lower() for p in ch.name.split() if len(p) > 2}
            if entity in name_parts and entity not in known_entities:
                flag = LeakageFlag(
                    character_id=output.character_id,
                    field="public_response",
                    leaked_text=entity,
                    reason=f"References character '{ch.name}' not in observation set or known contacts",
                )
                result.flags.append(flag)
                result.passed = False

    if result.flags:
        logger.warning(
            "Knowledge leakage detected for %s: %d flags",
            output.character_id,
            len(result.flags),
        )
        for flag in result.flags:
            logger.warning("  %s: %s", flag.leaked_text, flag.reason)

    return result


def validate_all_outputs(
    outputs: list[CharacterAgentOutput],
    checkpoint: CheckpointFile,
    observer_facts: dict[str, list[str]],
) -> list[ValidationResult]:
    """Validate all agent outputs for knowledge leakage.

    Args:
        outputs: Agent outputs to validate.
        checkpoint: Current game state.
        observer_facts: Map of character_id -> observed facts from discriminator.

    Returns list of ValidationResults. Outputs that fail validation are
    logged but not removed (log-and-warn in v1).
    """
    results = []
    for output in outputs:
        character = next(
            (c for c in checkpoint.characters if c.character_id == output.character_id),
            None,
        )
        if not character:
            continue

        facts = observer_facts.get(output.character_id, [])
        result = validate_agent_output(output, character, facts, checkpoint)
        results.append(result)

    return results
