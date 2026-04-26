"""Out-of-character /query handler — read-only POV consultation.

Players use /query to ask things they can't ask in fiction: what do I
see right now, what was that NPC's name, what day is it, have I met
this person. The handler reads the asking character's envelope (their
known_context, current scene, recent canonical events they observed,
pending observations, and the bound-player roster) and either answers
concisely or refuses in-fiction when the character can't plausibly
know the answer.

Read-only by contract:
- No checkpoint mutation. No append to session_conversation.
- No broadcast. No tick. No turn advancement.
- No `world_state.hidden_lore` / `world_state.hidden_facts` in the
  prompt — the boundary is the same as a character agent's: the
  asking character's envelope only.

The handler runs under role="query_handler" so the LLMConfig can
pick a model independently from /act. Defaults to Haiku — answers
are short, the consultation is bounded, and we want the player to
get a fast reply (the user is staring at Discord waiting for a
response).
"""

from __future__ import annotations

import logging

from app.engine.context_builder import (
    build_player_characters_block,
    build_setting_summary,
    build_world_context,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.events import visible_fact_texts
from app.schemas.query import QueryResponse

logger = logging.getLogger(__name__)


# Last N events the asking character observed are included in the
# recent-events block. Caps the prompt size on long sessions while
# still giving the LLM enough recent context to answer "what just
# happened" / "who walked in" style questions.
_MAX_RECENT_EVENTS = 12


def _format_character_identity(
    checkpoint: CheckpointFile, character_id: str,
) -> str:
    char = next(
        (c for c in checkpoint.characters if c.character_id == character_id),
        None,
    )
    if char is None:
        return (
            f"Character `{character_id}` is not present in the roster. "
            "Treat this as an error state and gate the answer."
        )
    parts = [
        f"You are answering for **{char.name}** (id: `{char.character_id}`)."
    ]
    if char.public_sheet.role:
        parts.append(f"Role: {char.public_sheet.role}")
    if char.public_sheet.faction:
        parts.append(f"Faction: {char.public_sheet.faction}")
    if char.public_sheet.appearance:
        parts.append(f"Appearance: {char.public_sheet.appearance}")
    if char.status.value != "active":
        parts.append(f"Status: {char.status.value}")
    return "\n".join(parts)


def _format_scene(checkpoint: CheckpointFile, character_id: str) -> str:
    """Render the asking character's current scene PLUS who else is
    present there. The handler uses this to answer "what do I see /
    who is here" without having to consult the global roster.
    """
    char = next(
        (c for c in checkpoint.characters if c.character_id == character_id),
        None,
    )
    if char is None or not char.location:
        return "(unknown — the character has no current scene set)"

    scene = checkpoint.world_state.locations.scene_graph.get(char.location, {})
    if isinstance(scene, dict):
        name = scene.get("name", char.location)
        desc = scene.get("description", "")
    else:
        name = char.location
        desc = ""

    others_present: list[str] = []
    for c in checkpoint.characters:
        if c.character_id == character_id:
            continue
        if c.location != char.location or c.status.value != "active":
            continue
        # Surface BOTH name and appearance — the handler decides which
        # to use based on whether the asking character has been
        # "introduced" (its prompt instructs name vs appearance based
        # on context).
        role = c.public_sheet.role or "unknown role"
        appearance = c.public_sheet.appearance or "nondescript"
        others_present.append(
            f"- {c.name} (id: `{c.character_id}`) — {role}; {appearance}"
        )

    parts = [f"Location: {name}"]
    if desc:
        parts.append(desc)
    parts.append("")
    if others_present:
        parts.append("Also present in this scene:")
        parts.extend(others_present)
    else:
        parts.append("No one else is present in this scene.")
    return "\n".join(parts)


def _format_recent_events(
    checkpoint: CheckpointFile,
    character_id: str,
    *,
    max_events: int = _MAX_RECENT_EVENTS,
) -> str:
    """Format the last few canonical events the asking character
    actually observed.

    Walks `canonical_events` newest-first, keeps any event where the
    asking character is in the observer list (along with their
    observation level), then reverses the kept slice so the LLM reads
    them chronologically. Up to `max_events` entries.

    Renders `observable_facts` only — the asking character should
    only get to reflect on what they actually perceived. Same fact
    channel as `broadcast_event`'s in-scene perception path.
    """
    events = checkpoint.canonical_events or []
    if not events:
        return "(no events on record yet — the story has just begun)"

    LEVEL = {
        "d": "directly observed",
        "i": "indirectly perceived",
        "f": "inferred",
    }
    kept: list[tuple[int, str, list[str]]] = []
    for idx in range(len(events) - 1, -1, -1):
        ev = events[idx]
        observer_match = next(
            (o for o in ev.observers if o.character_id == character_id),
            None,
        )
        if observer_match is None:
            continue
        level = LEVEL.get(
            observer_match.observation_level,
            observer_match.observation_level or "observed",
        )
        facts = visible_fact_texts(
            ev.canonical_event.observable_facts, character_id,
        )
        if ev.canonical_event.observable_facts and not facts:
            continue
        kept.append((idx, level, facts))
        if len(kept) >= max_events:
            break

    if not kept:
        return "(this character hasn't observed any events on record yet)"

    kept.reverse()
    lines: list[str] = []
    for idx, level, facts in kept:
        if not facts:
            lines.append(f"- [event #{idx}] ({level}) (no observable surface)")
            continue
        lines.append(f"- [event #{idx}] ({level})")
        for f in facts:
            lines.append(f"    • {f}")
    return "\n".join(lines)


def _format_pending(
    checkpoint: CheckpointFile, character_id: str,
) -> str:
    char = next(
        (c for c in checkpoint.characters if c.character_id == character_id),
        None,
    )
    if char is None or not char.pending_observations:
        return "(none — nothing has happened off-screen since their last beat)"
    return "\n".join(f"- {entry}" for entry in char.pending_observations)


async def answer_query(
    client: LLMClient,
    prompt_manager: PromptManager,
    checkpoint: CheckpointFile,
    character_id: str,
    question: str,
) -> QueryResponse:
    """Run the query handler for one character + one question.

    Pure read: the checkpoint is consulted but never mutated. Caller
    owns delivery (ephemeral Discord followup in the bot path; print
    in the CLI path).
    """
    char = next(
        (c for c in checkpoint.characters if c.character_id == character_id),
        None,
    )
    known_context_block = (
        build_world_context(char, checkpoint) if char is not None
        else "(no character record — answer cannot be POV-grounded)"
    )

    messages = prompt_manager.render_messages(
        "query_handler",
        setting_summary=build_setting_summary(checkpoint),
        character_identity_block=_format_character_identity(
            checkpoint, character_id,
        ),
        known_context_block=known_context_block,
        scene_block=_format_scene(checkpoint, character_id),
        player_characters_block=build_player_characters_block(
            checkpoint, character_id,
        ),
        recent_events_block=_format_recent_events(checkpoint, character_id),
        pending_observations_block=_format_pending(checkpoint, character_id),
        question=question.strip() or "(empty question)",
    )

    response = await client.complete(
        role="query_handler",
        messages=messages,
        response_model=QueryResponse,
        temperature=0.4,
        max_tokens=600,
    )
    parsed: QueryResponse = response.parsed
    parsed.answer = (parsed.answer or "").strip()
    parsed.gate_reason = (parsed.gate_reason or "").strip()
    logger.info(
        "Query for %s: gated=%s reason=%s answer_chars=%d question=%r",
        character_id,
        parsed.knowledge_gated,
        parsed.gate_reason or "-",
        len(parsed.answer),
        question[:120],
    )
    return parsed
