"""Context builder — constructs prompt context for character agents.

Builds character packets and scene context that are visibility-aware (only
includes information the character can know).
"""

from __future__ import annotations

import logging

from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage

logger = logging.getLogger(__name__)


def iter_agent_beats(
    agent_outputs: list[CharacterAgentOutput],
    checkpoint: CheckpointFile,
):
    """Yield `(output, character | None)` pairs for each agent output.

    Two consumers format agent beats differently — narrator produces
    multi-section markdown with visibility-aware labels (name vs
    appearance vs role, depending on whether the player has met the
    character), while orchestrator's `_build_npc_turn_summary` produces
    a single-line digest for silent observers' pending queues. The
    formatting intentionally diverges because those consumers serve
    different readers, but BOTH need the same roster lookup + the same
    missing-character semantics.

    This helper is the shared kernel: walk outputs, resolve characters
    once via a dict (not N linear scans), hand back a tuple the caller
    shapes as it wishes. If at some point a consumer needs different
    missing-character behavior or a different character-id source, fork
    here — but today they're aligned and a shared iterator keeps them
    aligned.
    """
    chars_by_id = {c.character_id: c for c in checkpoint.characters}
    for output in agent_outputs:
        yield output, chars_by_id.get(output.character_id)


def append_turn_to_conversation(
    conversation: list[ConversationMessage],
    user_content: str,
    response,
) -> None:
    """Append one (user, assistant) exchange to a rolling conversation.

    Every engine role (event_router, narrator, character_agent both in
    respond + tick) persists its turn the same way: capture the user
    message before the LLM call wrapped it with cache_control, then
    re-serialize the raw assistant content blocks from the response and
    append both. Keeping the pattern in one place means any future
    tweak (e.g. truncation, summarization, different cache handling)
    lands once.

    `response` is an LLMResponse; imported lazily to avoid a module-level
    dependency from context_builder on the LLM client.
    """
    from app.llm.client import serialize_assistant_content

    assistant_content = serialize_assistant_content(response.raw_response.content)
    conversation.append(ConversationMessage(role="user", content=user_content))
    conversation.append(ConversationMessage(role="assistant", content=assistant_content))


def build_character_packet(char: CharacterRecord) -> dict[str, str]:
    """Build the stable character-identity variables for the agent system prompt.

    Dynamic state (goals/objectives/secrets) is rendered into the per-turn user
    message separately; this function covers the frozen-identity portion.

    Traits, voice, and narrative_notes are gone as separate fields —
    personality absorbs all of them into one prose block.
    """
    return {
        "character_id": char.character_id,
        "character_name": char.name,
        "character_role": char.public_sheet.role or "unspecified",
        "character_appearance": char.public_sheet.appearance or "unremarkable",
        "character_faction": char.public_sheet.faction or "unaffiliated",
        "character_backstory": char.backstory or "No detailed backstory available.",
        "character_personality": char.personality or "No detailed personality notes.",
    }


def build_character_state(char: CharacterRecord) -> dict[str, str]:
    """Build the per-turn dynamic state variables for the agent user message."""
    goals = (
        "\n".join(f"- {g}" for g in char.private_state.goals)
        if char.private_state.goals
        else "None specified."
    )
    objectives = (
        "\n".join(f"- {o}" for o in char.private_state.current_objectives)
        if char.private_state.current_objectives
        else "None active — let your response emerge from your goals and the moment."
    )
    secrets = (
        "\n".join(f"- {s}" for s in char.private_state.secrets)
        if char.private_state.secrets
        else "None."
    )

    return {
        "character_goals": goals,
        "character_current_objectives": objectives,
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


def build_world_context(
    character: CharacterRecord,
    checkpoint: CheckpointFile,
) -> str:
    """World context as THIS character perceives it.

    With v2 imports, every character carries a `known_context` envelope —
    the filtered slice of world lore/facts/rumors they plausibly know,
    written from their POV by a dedicated extraction pass. That envelope
    is authoritative when present.

    Legacy fallback (pre-v2 checkpoints) concatenates the genre/tone plus
    the global public lore and facts, shared across all characters.
    `setting.premise` is deliberately excluded — it's authorial meta that
    routinely leaks plot-level secrets into every agent's prompt. Router
    and narrator (omniscient roles) still get the premise through their
    own setting summaries.
    """
    if character.known_context:
        return character.known_context

    setting = checkpoint.world_state.setting
    parts: list[str] = []
    if setting.genre:
        parts.append(f"Genre: {setting.genre}")
    if setting.tone:
        parts.append(f"Tone: {setting.tone}")

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
        present.append(f"- {char.name}: {role}, {appearance}")

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


def resolve_acting_character(
    checkpoint: CheckpointFile,
    requested: str,
) -> tuple[str, CharacterRecord | None, str]:
    """Return `(acting_id, acting_char, acting_name)` for a turn.

    Centralizes the fallback chain every role used to re-implement:
      requested id ▶ session.player_character_id ▶ "the protagonist".
    `acting_char` is None when the resolved id has no matching record
    (legacy checkpoints, cull edge cases); name falls back to
    session.player_name in that case, then to a neutral string.
    """
    acting_id = requested or checkpoint.session.player_character_id
    acting_char = next(
        (c for c in checkpoint.characters if c.character_id == acting_id), None
    )
    acting_name = (
        acting_char.name if acting_char
        else (checkpoint.session.player_name or "the protagonist")
    )
    return acting_id, acting_char, acting_name


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


def clear_character_inbox(character: CharacterRecord) -> None:
    """Clear a character's inbox queues after they've been flushed into a
    prompt's user message.

    A character carries two per-turn queues:
      - `pending_observations: list[str]` — things they witnessed silently
        between their own turns. Each entry is rendered prose with a
        `[Turn N]` prefix embedded inline.
      - `incoming_directives: list[IncomingDirective]` — messages sent to
        them by another character, stamped structurally with
        `from_character_id`, `turn`, and `depth` (for delegation-chain
        tracking).

    The two queues diverge intentionally in shape — observations are
    free-text, directives need structured depth-tracking so we can warn
    and cap deep delegation chains. Their LIFECYCLE is symmetric: both
    accumulate silently, both flush on the character's next response or
    tick, both clear at the same moment. This helper guarantees they
    stay in lockstep so nothing can flush one queue while leaving the
    other stale.
    """
    character.pending_observations = []
    character.incoming_directives = []


def format_pending_observations_block(character: CharacterRecord) -> str:
    """Render the "Since your last response" block for the agent user message.

    Combines silent observations (things this character witnessed) with
    incoming directives (messages other characters sent them). Both flush
    on the agent's next response. Returns empty string when nothing is
    pending so the template doesn't render a dangling header.
    """
    if not character.pending_observations and not character.incoming_directives:
        return ""

    lines = ["## Since your last response"]
    for entry in character.pending_observations:
        lines.append(f"- {entry}")
    for d in character.incoming_directives:
        lines.append(
            f"- **Message from {d.from_character_id}** (turn {d.turn}): {d.content}"
        )
    lines.append("")  # trailing blank line before next section
    return "\n".join(lines) + "\n"
