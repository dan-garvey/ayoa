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


def build_setting_summary(checkpoint: CheckpointFile) -> str:
    """Render the `## Setting` block shared across the router, narrator,
    takeover, and character_gen prompt templates.

    v11-r7j: collapsed three near-identical helpers (one in
    `turn_loop_dispatcher`, one in `narrator`, plus inline
    constructions in `engine_bridge._build_takeover_context` and
    `character_manager._spawn_one`) into this single source of truth.
    The shape this returns matches the `{setting_summary}` slot in
    `event_router.txt`, `narrator_phase2.txt`, `takeover.txt`,
    and `character_gen.txt`.

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
    3. The location of the first `is_playable=True` character in the
       roster (pristine sessions before any /join — useful for /status
       previews of an unclaimed story).
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
        if c.is_playable and c.location and _alive(c):
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


# v11-r9b: `build_characters_present` was deleted. It used to render
# "## Characters Present" for the agent's on-stage user message, with
# every co-located character's full role + appearance pasted in EVERY
# beat. Per the r9b context-management pass:
#
#   - The player's appearance is already in the cached
#     `## Player Characters` block of the agent's system prompt.
#     Repeating it in the per-turn user message paid for ~500 tokens
#     of cache-busting duplication and seeded prose echo.
#   - Per-NPC blurbs were the same kind of repeat for the agent's
#     own world_context (which carries each NPC's name + role +
#     appearance once, in the cached system prompt).
#   - Real scene composition CHANGES are now signaled through
#     pending_observations — `_apply_roster_moves` in orchestrator.py
#     pushes "X arrived." / "X left." entries onto every scene-mate's
#     inbox whenever the roster shifts. That is the live channel
#     that actually needed to differ between turns; the static block
#     wasn't carrying any of that information anyway.
#
# Routers still need an in-scene roster (different concern) — that
# helper lives at `turn_loop_dispatcher._build_characters_present`
# and is intentionally separate.


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
    """Every character_id that is currently CONTROLLED BY A HUMAN.

    Used across the engine to keep human-controlled characters out of
    NPC routing, agent fan-out, and background observation queues.
    Unions:
    - `session.character_bindings` keys (the canonical source — the
      Discord user → character map updated by /join).
    - `session.player_character_id` (creator binding, for legacy /
      pre-bindings checkpoints).

    NOTE: `CharacterRecord.is_playable` is INTENTIONALLY excluded from
    this set. Under the playable-2 semantics an `is_playable=True`
    character runs as an agent NPC until a human binds them — they
    DO get cascaded to via the router, DO get ticked off-stage, and
    DO show up in the agent fan-out. Only an explicit binding
    transfers them to "human-controlled" status. The pristine-roster
    fallback the function used to apply (treating every is_player slot
    as already-claimed) caused the playtest's "ticks blocked on every
    contestant before anyone joined" bug.
    """
    ids: set[str] = set(checkpoint.session.character_bindings or {})
    if checkpoint.session.player_character_id:
        ids.add(checkpoint.session.player_character_id)
    return ids


def build_player_characters_block(
    checkpoint: CheckpointFile,
    acting_character_id: str,
) -> str:
    """Render a markdown list of every CURRENTLY-BOUND player character.

    Appears in the frozen (cached) system prompt of the router,
    narrator, and each agent. Marks the turn's acting character so
    downstream prose knows whose action to center.

    Membership: union of `character_bindings` keys and (for legacy
    checkpoints) `session.player_character_id`. We deliberately do NOT
    surface every `is_playable=True` character here — under playable-2
    semantics, an unbound playable character is an agent NPC and
    belongs in the NPC roster, not the protagonists block. The router
    treats a name in this block as "human, never dispatch via picks";
    listing unbound playables here would forbid the cascade from
    reaching the very characters the agents are supposed to drive.

    Returns "- No player characters bound to this session." when empty
    (pristine session before anyone /joins) so prompts still render
    cleanly.
    """
    bindings = checkpoint.session.character_bindings or {}
    bound_ids = list(bindings.keys())
    pcid = checkpoint.session.player_character_id
    if pcid and pcid not in bound_ids:
        bound_ids.append(pcid)

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

    Population paths (v11-r9b):

    - `broadcast_event` in `turn_loop.py` pushes each observer's visible
      `observable_facts` (untagged — the entries are the agent's live
      sensorium for the scene, no routing label needed) onto every
      NPC observer who IS in the broadcast scene (and isn't the
      actor). Pre-r9b this fan-out also covered off-scene observers
      tagged `[off-scene perception]`; that channel was deleted
      because the router's `resolved_outcome` regularly fused private
      and public sub-beats into one omniscient string and any
      off-scene character listed as an observer received the whole
      thing (e.g. Dan's bedroom wardrobe choice landing in Ashara's
      dining-hall queue). Cross-scene awareness now requires a
      separate event whose `scene_id` is where the news lands.

    - `_apply_roster_moves` in `orchestrator.py` pushes
      `[your own action] …` onto the moved NPC's queue (so they
      don't re-narrate their own arrival) and unflagged
      `"X arrived." / "X left."` lines onto every scene-mate's
      queue. The latter is the live channel for "who is in your
      scene right now" — the on-stage user message no longer
      carries a `## Characters Present` block (that block was
      removed in v11-r9b alongside the off-scene perception
      channel).

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
