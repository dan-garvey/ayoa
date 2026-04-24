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


def resolve_scene_for_character(
    checkpoint: CheckpointFile, character_id: str | None,
) -> str:
    """The scene a character is currently in. Reads from the roster's
    `character.location` field — the truth-of-record for "where is X."

    Returns "" when the character has no resolvable location:
      - no character_id is given (legacy callers without an actor binding),
      - the character isn't in the roster (pristine tests, mid-spawn races),
      - the character's `location` is unset (importer placed nobody there,
        or a character_gen pass left it blank — also a fixable defect now
        that spawning derives location from the acting actor's scene).

    Callers must handle the empty-string case (most render an empty or
    "no scene information" block). There is no global fallback: the
    pre-v11 `world_state.locations.current_scene_id` was the importer's
    "world pivot" set ONCE and never updated, and reading it gave every
    LLM context-builder a wrong answer the moment any character moved.
    """
    if character_id:
        for c in checkpoint.characters:
            if c.character_id == character_id and c.location:
                return c.location
    return ""


def build_scene_context(
    checkpoint: CheckpointFile, character_id: str | None = None,
) -> str:
    """Build scene description for `character_id`'s current scene.

    `character_id=None` (or an unsited character) yields the empty-scene
    string. New code should always pass the acting/POV character_id so
    the prompt reflects where that character actually is.
    """
    locations = checkpoint.world_state.locations
    scene_id = resolve_scene_for_character(checkpoint, character_id)
    if not scene_id:
        return "No scene information available."

    scene = locations.scene_graph.get(scene_id, {})
    name = scene.get("name", scene_id)
    desc = scene.get("description", "")

    parts = [f"Location: {name}"]
    if desc:
        parts.append(desc)
    return "\n".join(parts)


def build_setting_summary(checkpoint: CheckpointFile) -> str:
    """Render the `## Setting` block shared across the router, narrator,
    takeover, and character_gen prompt templates.

    v11-r7j: collapsed three near-identical helpers (one in
    `turn_loop_dispatcher`, one in `narrator`, plus inline
    constructions in `engine_bridge._build_takeover_context` and
    `character_manager._spawn_one`) into this single source of truth.
    The shape this returns matches the `{setting_summary}` slot in
    `event_router_v9.txt`, `narrator_phase2_v9.txt`, `takeover_v1.txt`,
    and `character_gen_v3.txt`.

    Empty fields are skipped — a story whose importer left, say, `era`
    blank does not get a `Era: ` line. Pre-r7j the takeover and spawn
    paths emitted those empty lines unconditionally and the router /
    narrator paths skipped them; v11-r7j harmonizes on the conditional
    shape (omit-empty is strictly better for the LLM).
    """
    setting = checkpoint.world_state.setting
    parts: list[str] = []
    if setting.genre:
        parts.append(f"Genre: {setting.genre}")
    if setting.era:
        parts.append(f"Era: {setting.era}")
    if setting.tone:
        parts.append(f"Tone: {setting.tone}")
    if setting.premise:
        parts.append(f"Premise: {setting.premise}")
    return "\n".join(parts) if parts else "No setting information available."


def pov_scene_for_user(
    checkpoint: CheckpointFile, user_id: str | None = None,
) -> str:
    """Best-effort "current scene" for player-facing displays (CLI,
    Discord embeds, status lines). Resolves in this order:

    1. The location of the character bound to `user_id` (if `user_id`
       is given and they hold a binding in `session.character_bindings`).
    2. The location of `session.player_character_id` (creator binding /
       single-player default).
    3. The location of the first `is_player=True` character in the
       roster (pristine sessions before any /join).
    4. "" — if nothing resolves, the surface should render
       "(no active scene)" rather than a stale fallback.

    Culled characters are skipped at every step. A culled character
    keeps their last `location` string in the roster (so their backstory
    line still parses), but they are no longer in the scene; rendering
    their dead location as "where the player is now" would mislead
    every downstream UI surface AND the takeover prompt's
    `current_scene` block. Dormant characters are still considered —
    dormancy is "off-screen but alive," which is exactly what a player
    POV usually IS in a single-player checkpoint.

    There is no fallback to a world-level "current scene" anymore —
    that field was murdered in v11. For multi-player sessions the
    "current scene" depends on who's asking; for single-player the
    creator binding is the answer. Either way it's per-character.
    """
    def _alive(c: CharacterRecord) -> bool:
        return c.status != "culled"

    bindings = checkpoint.session.character_bindings or {}
    if user_id and bindings:
        for char_id, bound_uid in bindings.items():
            if bound_uid == user_id:
                for c in checkpoint.characters:
                    if (
                        c.character_id == char_id
                        and c.location
                        and _alive(c)
                    ):
                        return c.location
                break
    pcid = checkpoint.session.player_character_id
    if pcid:
        for c in checkpoint.characters:
            if c.character_id == pcid and c.location and _alive(c):
                return c.location
    for c in checkpoint.characters:
        if c.is_player and c.location and _alive(c):
            return c.location
    return ""


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
    """Build a summary of other characters present in the same scene as
    `character`. Scene is read from `character.location`; an unsited
    character (no location set) gets the empty-scene string.
    """
    scene_id = resolve_scene_for_character(checkpoint, character.character_id)
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
    """Format other characters' responses that happened earlier this turn.

    Renders `public_text` only — `intent` (the trailing parenthetical
    on each agent's output) is private to the emitting agent and the
    engine, and must NEVER reach another agent's prompt. This is one
    of the chokepoints that enforces that contract; the others are
    the router intention block (dispatcher passes `output.public_text`
    directly) and the narrator's canonical-event input (canonical
    events carry resolved-outcome prose authored by the router, not
    raw agent text).
    """
    if not prior_responses:
        return "No other characters have responded yet."

    parts = []
    for resp in prior_responses:
        char = next(
            (c for c in checkpoint.characters if c.character_id == resp.character_id),
            None,
        )
        name = char.name if char else resp.character_id
        body = (resp.public_text or "").strip()
        if not body:
            body = "(silent beat)"
        parts.append(f"- {name}: {body}")

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
        lines.append(
            f"- **{char.name}**{marker} (id: {char.character_id}) — "
            f"{role}. {appearance}"
        )

    if not lines:
        return "- No player characters bound to this session."
    return "\n".join(lines)


def clear_character_inbox(character: CharacterRecord) -> None:
    """Clear a character's `pending_observations` queue after it's been
    flushed into the prompt's user message.

    Pre-Commit-2 there was a parallel `incoming_directives` queue with
    its own structured message envelope (`from_character_id`, `turn`,
    `depth`) and a delegation-chain depth cap. That whole inter-agent
    message bus is gone — cross-character communication now flows
    through normal canonical events: the router authors a courier
    walking in, a note that lands in `observable_facts`, etc.

    Population path (v11-r7j): `broadcast_event` in `turn_loop.py`
    pushes a one-line "[off-scene perception] …" entry onto every NPC
    observer who is NOT in the broadcast scene. This is how the
    perception channel router rule 13 promises actually lands — when
    the router writes a courier delivering a note to Marcus and lists
    Marcus as an observer, Marcus's next agent call sees the inbox
    entry. In-scene NPC observers are NOT pushed (they read the same
    event live via their normal context block when the router picks
    them as a responder; pushing twice would double-count).

    This helper retains the name `clear_character_inbox` so callers
    don't churn, but its job is now scoped to the one remaining queue.
    """
    character.pending_observations = []


def format_pending_observations_block(character: CharacterRecord) -> str:
    """Render the "Since your last response" block for the agent user message.

    Lists silent observations the character witnessed since their last
    response. Returns empty string when nothing is pending so the
    template doesn't render a dangling header.
    """
    if not character.pending_observations:
        return ""

    lines = ["## Since your last response"]
    for entry in character.pending_observations:
        lines.append(f"- {entry}")
    lines.append("")  # trailing blank line before next section
    return "\n".join(lines) + "\n"
