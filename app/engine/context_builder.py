"""Context builder — constructs prompt context for character agents."""

from __future__ import annotations

import logging
import re

from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage

logger = logging.getLogger(__name__)


def append_turn_to_conversation(
    conversation: list[ConversationMessage],
    user_content: str,
    response,
) -> None:
    """Append one (user, assistant) exchange to a rolling conversation.

    Every engine role (event_router, narrator, character_agent turn/perceive)
    persists its exchange the same way: capture the user
    message before the LLM call wrapped it with cache_control, then
    re-serialize the raw assistant content blocks from the response and
    append both. Keeping the pattern in one place means any future
    tweak (e.g. truncation, summarization, different cache handling)
    lands once.

    `response` is an LLMResponse; imported lazily to avoid a module-level
    dependency from context_builder on the LLM client.
    """
    user_message, assistant_message = conversation_turn_messages(
        user_content, response,
    )
    conversation.append(user_message)
    conversation.append(assistant_message)


def assistant_message_from_response(response) -> ConversationMessage:
    """Build the assistant-side history message for a completed LLM call."""
    from app.llm.client import serialize_assistant_content

    assistant_content = getattr(response, "assistant_content", None)
    if assistant_content is None:
        assistant_content = serialize_assistant_content(response.raw_response.content)
    return ConversationMessage(role="assistant", content=assistant_content)


def conversation_turn_messages(
    user_content: str,
    response,
) -> tuple[ConversationMessage, ConversationMessage]:
    """Build the two rolling-history messages for a completed exchange."""
    return (
        ConversationMessage(role="user", content=user_content),
        assistant_message_from_response(response),
    )


def append_assistant_to_conversation(
    conversation: list[ConversationMessage],
    response,
) -> None:
    """Append only the assistant output for roles with redundant user turns."""
    conversation.append(assistant_message_from_response(response))


def build_character_packet(char: CharacterRecord) -> dict[str, str]:
    """Build the stable character-identity variables for the agent user tail.

    Dynamic state (goals/objectives/secrets) is rendered into the same per-turn
    user message separately; this function covers the stable identity portion.

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


def replace_character_ids_with_names(
    text: str,
    checkpoint: CheckpointFile,
) -> str:
    """Render id-authored facts for prose-facing LLM roles.

    Router and rules prompts use `character_id` as their single character
    handle. Narrator and agent prompts are prose-facing, so they should see
    display names instead of ids. This keeps each model role on one identity
    surface without requiring router prompts to carry name/id pairs.
    """
    if not text:
        return ""
    out = text
    pairs = [
        (char.character_id, char.name)
        for char in checkpoint.characters
        if char.character_id and char.name
    ]
    for char_id, name in sorted(pairs, key=lambda pair: len(pair[0]), reverse=True):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(char_id)}(?![A-Za-z0-9_])"
        )
        out = pattern.sub(name, out)
    return out


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


def resolve_location_for_character(
    checkpoint: CheckpointFile, character_id: str | None,
) -> str:
    """A character's opaque location label from the roster.

    Returns "" when the character has no resolvable location:
      - no character_id is given (legacy callers without an actor binding),
      - the character isn't in the roster (pristine tests, mid-spawn races),
      - the character's `location` is unset (the story seed placed nobody there,
        or a character_gen pass left it blank).

    Callers must handle the empty-string case. There is no global
    location fallback.
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

    Empty fields are skipped — a story whose seed left, say, `era`
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


def pov_location_for_user(
    checkpoint: CheckpointFile, user_id: str | None = None,
) -> str:
    """Best-effort current location label for player-facing displays
    (CLI, Discord embeds, status lines). Resolves in this order:

    1. The location of the character bound to `user_id` (if `user_id`
       is given and they hold a binding in `session.character_bindings`).
    2. The location of `session.player_character_id` (creator binding /
       single-player default).
    3. The location of the first `is_playable=True` character in the
       roster (pristine sessions before any /join — useful for /status
       previews of an unclaimed story).
    4. "" — if nothing resolves, the surface should render
       "(no active location)" rather than a stale fallback.

    Culled characters are skipped at every step. A culled character
    keeps their last `location` string in the roster, but rendering
    their dead location as "where the player is now" would mislead
    every downstream UI surface. Dormant characters are still considered.

    There is no fallback to a world-level current location. For
    multi-player sessions the label depends on who's asking; for
    single-player the creator binding is the answer.
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
    DO get cascaded to via the router, DO get picked for background responses,
    and DO show up in the agent fan-out. Only an explicit binding
    transfers them to "human-controlled" status. The pristine-roster
    fallback the function used to apply (treating every is_player slot
    as already-claimed) caused the playtest's "NPC routing blocked on
    every contestant before anyone joined" bug.
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

    Appears in role prompts where human-bound characters must be
    distinguished from NPCs. For the router this is intentionally
    per-turn user-message context, not frozen system-prefix context,
    because it marks the turn's acting character.

    Membership: union of `character_bindings` keys and (for legacy
    checkpoints) `session.player_character_id`. We deliberately do NOT
    surface every `is_playable=True` character here — under playable-2
    semantics, an unbound playable character is an agent NPC and
    belongs in the NPC roster, not the protagonists block. The router
    treats a name in this block as "human, never dispatch as an agent";
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
        location = char.location or "unknown location"
        marker = " (acting this turn)" if char.character_id == acting_character_id else ""
        lines.append(
            f"- **{char.character_id}**{marker} — {role}. "
            f"Location: {location}. {appearance}"
        )

    if not lines:
        return "- No player characters bound to this session."
    return "\n".join(lines)


def build_narrator_player_characters_block(
    checkpoint: CheckpointFile,
    pov_character_id: str,
) -> str:
    """Render human-bound character names for the narrator.

    The router needs character ids for structural routing. The narrator only
    needs names and which character is "you" to avoid puppeting player-owned
    choices. Player-safe identity/appearance context lives in the stable public
    character context block; raw public_sheet fields are not a narrator gloss
    source.
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
        marker = " (you)" if char.character_id == pov_character_id else ""
        lines.append(f"- {char.name}{marker}")

    if not lines:
        return "- No human-played characters are currently listed."
    return "\n".join(lines)


def _compact_player_context(text: str, *, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    cut = compact.rfind(". ", 0, limit)
    if cut < max(80, limit // 2):
        cut = compact.rfind(" ", 0, limit)
    if cut < 0:
        cut = limit
    return compact[:cut].rstrip(" .") + "."


def build_narrator_public_character_context_block(
    checkpoint: CheckpointFile,
    *,
    max_characters: int = 120,
) -> str:
    """Stable, player-safe character context for narrator glosses.

    This deliberately reads only `descriptions.public`, not
    `public_sheet.role/faction/appearance`. Older imports sometimes put
    authorial or hidden labels in those legacy fields; the narrator's
    public context must be a separate, explicitly player-safe surface.
    """
    lines: list[str] = []
    for char in checkpoint.characters:
        if char.status == "culled":
            continue
        if not char.name.strip():
            continue
        description = _compact_player_context(
            char.descriptions.public, limit=420,
        )
        if description:
            lines.append(f"- {char.name}: {description}")
        if len(lines) >= max_characters:
            break

    if not lines:
        return "- No player-safe public character context is available."
    return "\n".join(lines)


def build_narrator_pov_knowledge_block(
    checkpoint: CheckpointFile,
    pov_character_id: str,
    visible_text: str,
    *,
    max_characters: int = 8,
) -> str:
    """List public-context entries that are relevant to this passage.

    The actual descriptions live in the stable narrator public-character
    context block. This per-turn block only names which public entries
    are eligible for a short local gloss, keeping volatile prompt input
    small and avoiding repetition of public descriptions every turn.
    """
    haystack = visible_text or ""
    if not haystack.strip():
        return "- No additional viewpoint-known glosses for this passage."

    mentioned: list[CharacterRecord] = []
    for char in checkpoint.characters:
        if char.character_id == pov_character_id:
            continue
        probes = [char.character_id, char.name]
        if any(
            probe and re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(probe)}(?![A-Za-z0-9_])",
                haystack,
            )
            for probe in probes
        ):
            mentioned.append(char)
        if len(mentioned) >= max_characters:
            break

    lines: list[str] = []
    for char in mentioned:
        if char.descriptions.public.strip():
            lines.append(f"- {char.name}")

    if not lines:
        return "- No additional viewpoint-known glosses for this passage."
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
      sensorium, no routing label needed) onto every local NPC observer
      who is not the actor, plus mediated/remote NPC observers explicitly
      named by fact-level `visible_to`. Pre-r9b this fan-out also covered
      off-location observers tagged `[off-location perception]`; that
      channel was deleted because the old one-line event summary
      regularly fused private and public sub-beats into one omniscient
      string and any off-location character listed as an observer
      received the whole thing. Cross-location awareness now requires a
      concrete live perceptual path encoded in `observable_facts`, or a
      later local event where the news arrives.

    Movement is no longer a separate router side-effect. Any arrival,
    departure, or transfer other characters perceive must be encoded as
    `observable_facts` on the event.

    This helper retains the name `clear_character_inbox` so callers
    don't churn, but its job is now scoped to the one remaining queue.
    """
    character.pending_observations = []


def _format_elapsed_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        unit = "second" if seconds == 1 else "seconds"
        return f"{seconds} {unit}"

    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        minute_unit = "minute" if minutes == 1 else "minutes"
        if remainder == 0:
            return f"{minutes} {minute_unit}"
        second_unit = "second" if remainder == 1 else "seconds"
        return f"{minutes} {minute_unit} and {remainder} {second_unit}"

    hours, minutes = divmod(minutes, 60)
    hour_unit = "hour" if hours == 1 else "hours"
    if minutes == 0:
        return f"{hours} {hour_unit}"
    minute_unit = "minute" if minutes == 1 else "minutes"
    return f"{hours} {hour_unit} and {minutes} {minute_unit}"


def format_elapsed_agent_turn_block(
    character: CharacterRecord,
    checkpoint: CheckpointFile,
) -> str:
    """Render elapsed story time since this agent last got a turn.

    `clock_at_s` moves when the character observes events, so it cannot
    answer this question by itself. `last_agent_turn_at_s` is updated only
    when a character-agent turn commits.
    """
    last_turn_at_s = character.last_agent_turn_at_s
    if last_turn_at_s is None:
        return ""

    current_at_s = max(
        int(checkpoint.session.leading_at_s),
        int(character.clock_at_s),
    )
    elapsed_s = max(0, current_at_s - int(last_turn_at_s))
    lines = ["## Time Since Your Last Turn"]
    if elapsed_s == 0:
        lines.append(
            "No meaningful story time has passed since you last had a "
            "chance to act."
        )
    else:
        lines.append(
            f"About {_format_elapsed_duration(elapsed_s)} has passed in the "
            "story since you last had a chance to act."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


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
