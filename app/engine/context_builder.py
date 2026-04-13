"""Context builder — constructs prompt context for character agents.

Builds character packets, scene context, and recent transcript summaries
that are visibility-aware (only includes information the character can know).
"""

from __future__ import annotations

import logging

from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile

logger = logging.getLogger(__name__)


def build_character_packet(char: CharacterRecord) -> dict[str, str]:
    """Build the context variables for a character agent prompt."""
    traits = ", ".join(char.public_sheet.traits) if char.public_sheet.traits else "none noted"
    goals = "\n".join(f"- {g}" for g in char.private_state.goals) if char.private_state.goals else "None specified."
    secrets = "\n".join(f"- {s}" for s in char.private_state.secrets) if char.private_state.secrets else "None."

    # Format attitudes
    attitudes_lines = []
    for target, value in char.private_state.attitudes.items():
        label = _attitude_label(value)
        attitudes_lines.append(f"- {target}: {value:+.1f} ({label})")
    attitudes = "\n".join(attitudes_lines) if attitudes_lines else "No strong opinions."

    # Format memories
    memories = []
    if char.memory.episodic:
        for m in char.memory.episodic[-10:]:  # Last 10 episodic memories
            memories.append(f"- {m}")
    if char.memory.summaries:
        for s in char.memory.summaries[-3:]:  # Last 3 summary memories
            memories.append(f"- [Summary] {s}")
    memories_str = "\n".join(memories) if memories else "No prior memories of events."

    return {
        "character_id": char.character_id,
        "character_name": char.name,
        "character_role": char.public_sheet.role or "unspecified",
        "character_traits": traits,
        "character_voice": char.public_sheet.voice or "natural speech",
        "character_appearance": char.public_sheet.appearance or "unremarkable",
        "character_faction": char.public_sheet.faction or "unaffiliated",
        "character_backstory": char.backstory or "No detailed backstory available.",
        "character_personality": char.personality or "No detailed personality notes.",
        "character_goals": goals,
        "character_attitudes": attitudes,
        "character_secrets": secrets,
        "character_memories": memories_str,
        "character_narrative_notes": char.narrative_notes or "No special narrative guidance.",
    }


def build_scene_context(checkpoint: CheckpointFile) -> str:
    """Build scene description from the checkpoint's current location."""
    locations = checkpoint.world_state.locations
    scene_id = locations.current_scene_id
    if not scene_id:
        return "No scene information available."

    scene = locations.scene_graph.get(scene_id, {})
    name = scene.get("name", scene_id)
    desc = scene.get("description", "")

    parts = [f"Location: {name}"]
    if desc:
        parts.append(desc)
    return "\n".join(parts)


def build_world_context(checkpoint: CheckpointFile) -> str:
    """Build a condensed world context for character agents."""
    setting = checkpoint.world_state.setting
    parts = []
    if setting.genre:
        parts.append(f"Genre: {setting.genre}")
    if setting.tone:
        parts.append(f"Tone: {setting.tone}")
    if setting.premise:
        parts.append(f"Premise: {setting.premise}")

    # Include key world facts
    if checkpoint.world_state.facts:
        parts.append("\nKey world facts:")
        for fact in checkpoint.world_state.facts:
            parts.append(f"- {fact}")

    # Include world lore for grounding
    if checkpoint.world_state.lore:
        parts.append(f"\nWorld lore:\n{checkpoint.world_state.lore}")

    return "\n".join(parts) if parts else "No world context available."


def build_characters_present(
    character: CharacterRecord, checkpoint: CheckpointFile
) -> str:
    """Build a summary of other characters present in the same scene."""
    scene_id = checkpoint.world_state.locations.current_scene_id
    if not scene_id:
        return "You don't know who else is nearby."

    present = []
    for char in checkpoint.characters:
        if char.character_id == character.character_id:
            continue
        if char.location != scene_id:
            continue
        if char.status.value != "active":
            continue

        role = char.public_sheet.role or "unknown role"
        appearance = char.public_sheet.appearance or "nondescript"
        # Include this character's attitude toward the other character
        attitude = character.private_state.attitudes.get(char.character_id)
        att_note = ""
        if attitude is not None and abs(attitude) >= 0.1:
            if attitude > 0:
                att_note = " (you regard them positively)"
            else:
                att_note = " (you regard them negatively)"

        present.append(f"- {char.name}: {role}, {appearance}{att_note}")

    if not present:
        return "No other characters are present."
    return "\n".join(present)


def build_recent_transcript(
    checkpoint: CheckpointFile, max_entries: int = 5
) -> str:
    """Build recent transcript, limited to prevent context bloat."""
    if not checkpoint.transcript:
        return "No prior conversation."

    entries = checkpoint.transcript[-max_entries:]
    lines = []
    for entry in entries:
        lines.append(f"Player: {entry.user}")
        lines.append(f"Narrator: {entry.assistant}")
        lines.append("")
    return "\n".join(lines).strip()


def format_observed_facts(facts: list[str]) -> str:
    """Format the list of observed facts for the agent prompt."""
    if not facts:
        return "You observe nothing unusual."
    return "\n".join(f"- {fact}" for fact in facts)


def _attitude_label(value: float) -> str:
    """Convert attitude float to a human-readable label."""
    if value >= 0.7:
        return "strong affinity"
    elif value >= 0.3:
        return "positive"
    elif value >= 0.1:
        return "mildly positive"
    elif value > -0.1:
        return "neutral"
    elif value > -0.3:
        return "mildly negative"
    elif value > -0.7:
        return "negative"
    else:
        return "hostile"
