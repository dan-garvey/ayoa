"""Context builder — constructs prompt context for character agents.

Builds character packets and scene context that are visibility-aware (only
includes information the character can know).
"""

from __future__ import annotations

import logging

from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile

logger = logging.getLogger(__name__)


def build_character_packet(char: CharacterRecord) -> dict[str, str]:
    """Build the stable character-identity variables for the agent system prompt.

    Dynamic state (goals/attitudes/secrets) is rendered into the per-turn user
    message separately; this function covers the frozen-identity portion.
    """
    traits = ", ".join(char.public_sheet.traits) if char.public_sheet.traits else "none noted"

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
        "character_narrative_notes": char.narrative_notes or "No special narrative guidance.",
    }


def build_character_state(char: CharacterRecord) -> dict[str, str]:
    """Build the per-turn dynamic state variables for the agent user message."""
    goals = (
        "\n".join(f"- {g}" for g in char.private_state.goals)
        if char.private_state.goals
        else "None specified."
    )
    secrets = (
        "\n".join(f"- {s}" for s in char.private_state.secrets)
        if char.private_state.secrets
        else "None."
    )

    attitudes_lines = []
    for target, value in char.private_state.attitudes.items():
        label = _attitude_label(value)
        attitudes_lines.append(f"- {target}: {value:+.1f} ({label})")
    attitudes = "\n".join(attitudes_lines) if attitudes_lines else "No strong opinions."

    return {
        "character_goals": goals,
        "character_attitudes": attitudes,
        "character_secrets": secrets,
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

    if checkpoint.world_state.facts:
        parts.append("\nKey world facts:")
        for fact in checkpoint.world_state.facts:
            parts.append(f"- {fact}")

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


def format_observed_facts(facts: list[str]) -> str:
    """Format the list of observed facts for the agent prompt."""
    if not facts:
        return "You observe nothing unusual."
    return "\n".join(f"- {fact}" for fact in facts)


def format_prior_responses(
    prior_responses: list[CharacterAgentOutput],
    checkpoint: CheckpointFile,
) -> str:
    """Format other characters' responses that happened earlier this turn."""
    if not prior_responses:
        return "No other characters have responded yet."

    parts = []
    for resp in prior_responses:
        char = next(
            (c for c in checkpoint.characters if c.character_id == resp.character_id),
            None,
        )
        name = char.name if char else resp.character_id
        lines = []
        if resp.public_response.actions:
            lines.extend(resp.public_response.actions)
        if resp.public_response.dialogue:
            for d in resp.public_response.dialogue:
                lines.append(f'"{d}"')
        if resp.public_response.expression:
            lines.append(f"({resp.public_response.expression})")
        parts.append(f"- {name}: {'; '.join(lines)}")

    return "\n".join(parts)


def collect_player_ids(checkpoint: CheckpointFile) -> set[str]:
    """Every character_id that belongs to a human player (bound or flagged).

    Used across the engine to keep player-controlled characters out of NPC
    routing, agent fan-out, and background observation queues. Unions:
    - session.character_bindings keys (the canonical source)
    - session.player_character_id (creator binding, for legacy/pre-bindings
      checkpoints that haven't had their bindings populated)
    - every CharacterRecord.is_player=True in the roster (for pristine
      checkpoints with no session bindings yet)
    """
    ids: set[str] = set(checkpoint.session.character_bindings or {})
    if checkpoint.session.player_character_id:
        ids.add(checkpoint.session.player_character_id)
    for c in checkpoint.characters:
        if c.is_player:
            ids.add(c.character_id)
    return ids


def build_player_characters_block(
    checkpoint: CheckpointFile,
    acting_character_id: str,
) -> str:
    """Render a markdown list of every player character for the prompt header.

    Appears in the frozen (cached) system prompt of the router, narrator, and
    each agent. Marks the turn's acting character so downstream prose knows
    whose action to center. Falls back to the is_player roster entries if
    character_bindings is empty (covers pristine checkpoints and tests).
    """
    bindings = checkpoint.session.character_bindings or {}
    bound_ids = list(bindings.keys()) if bindings else [
        c.character_id for c in checkpoint.characters if c.is_player
    ]
    if not bound_ids:
        pcid = checkpoint.session.player_character_id
        bound_ids = [pcid] if pcid else []

    lines: list[str] = []
    for char_id in bound_ids:
        char = next(
            (c for c in checkpoint.characters if c.character_id == char_id), None
        )
        if char is None:
            continue
        role = char.public_sheet.role or "unspecified role"
        appearance = (char.public_sheet.appearance or "not yet described").strip()
        marker = " (acting this turn)" if char.character_id == acting_character_id else ""
        lines.append(f"- **{char.name}**{marker} — {role}. {appearance}")

    if not lines:
        return "- No player characters bound to this session."
    return "\n".join(lines)


def format_pending_observations_block(character: CharacterRecord) -> str:
    """Render the pending-observations markdown block for the agent user message.

    Returns an empty string when there's nothing pending (so the template
    renders cleanly without a dangling header). The orchestrator flushes and
    clears `character.pending_observations` around this call.
    """
    if not character.pending_observations:
        return ""

    lines = ["## Since your last response"]
    for entry in character.pending_observations:
        lines.append(f"- {entry}")
    lines.append("")  # trailing blank line before next section
    return "\n".join(lines) + "\n"


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
