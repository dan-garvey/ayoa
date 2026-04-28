"""Narrator composition — v11 per-POV render path.

`compose_pov_render` is the only production narrator entry point. Each
human POV with a queued perception gets its own render, composed against
a per-character rolling conversation stored on the checkpoint
(`checkpoint.narrator_conversations[pov_character_id]`). Voice and
continuity hold across the session on a per-POV basis.
"""

from __future__ import annotations

import logging
import time

from app.engine.prompt_manager import PromptManager
from app.engine.context_builder import append_turn_to_conversation
from app.engine.turn_loop_contracts import PARTIAL_MODE_MARKER
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.events import visible_fact_texts
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import RenderBufferEntry

logger = logging.getLogger(__name__)


def _resolve_buffered_events(
    ckpt: CheckpointFile,
    buffered_events: list[RenderBufferEntry],
) -> list[tuple[RenderBufferEntry, EventRouterOutput]]:
    """Walk the render buffer and resolve each entry against
    `ckpt.canonical_events`. Missing event_ids are warned and skipped —
    a stale buffer entry must not abort the render.
    """
    by_id: dict[str, EventRouterOutput] = {
        ev.event_id: ev for ev in ckpt.canonical_events
    }
    resolved: list[tuple[RenderBufferEntry, EventRouterOutput]] = []
    for entry in buffered_events:
        ev = by_id.get(entry.event_id)
        if ev is None:
            logger.warning(
                "compose_pov_render: buffered event_id %r not found in "
                "canonical_events; skipping",
                entry.event_id,
            )
            continue
        resolved.append((entry, ev))
    return resolved


_OBS_LEVEL_NAMES = {
    "d": "direct",
    "i": "indirect",
    "f": "inferred",
    "direct": "direct",
    "indirect": "indirect",
    "inferred": "inferred",
}


def _format_canonical_events_block(
    resolved: list[tuple[RenderBufferEntry, EventRouterOutput]],
    pov_character_id: str = "",
) -> str:
    """Serialize the resolved events into a prose block the narrator
    can read. One section per event with its observation level tag.

    The narrator's render input is the visible slice of
    `observable_facts`: the surface-grade list — verbatim dialogue,
    visible gestures, ambient sensory shifts — that drives the prose.
    Audit/framing fields such as `attempted_action` and
    `resolved_outcome` are intentionally absent from this contract; the
    narrator gets only facts visible to this POV.
    """
    if not resolved:
        return "No canonical events to render."
    sections: list[str] = []
    for idx, (entry, ev) in enumerate(resolved, start=1):
        obs = _OBS_LEVEL_NAMES.get(entry.observation_level, entry.observation_level)
        ca = ev.canonical_event
        facts = visible_fact_texts(
            ca.observable_facts,
            pov_character_id,
            include_all_observers=True,
        )
        if pov_character_id and not facts:
            # No fact visible to this POV means the event must not
            # surface in their render at all.
            continue
        lines = [
            f"## Event {idx}: {ev.event_id} [Observation: {obs}]",
        ]
        if facts:
            lines.append("observable_facts:")
            for fact in facts:
                lines.append(f"- {fact}")
        else:
            lines.append("observable_facts: (none)")
        sections.append("\n".join(lines))
    if not sections:
        return "No canonical events visible to this POV."
    return "\n\n".join(sections)


async def compose_pov_render(
    client: LLMClient,
    prompt_mgr: PromptManager,
    ckpt: CheckpointFile,
    pov_character_id: str,
    buffered_events: list[RenderBufferEntry],
    partial_mode: bool,
    user_input: str = "",
) -> tuple[NarratorFinalOutput, "TranscriptEntry"]:
    """v11 per-POV narrator entry point.

    Renders the beat from `pov_character_id`'s point of view in
    second-person present tense, using their per-character rolling
    conversation history (`ckpt.narrator_conversations[pov_character_id]`).

    Events are resolved by `event_id` against `ckpt.canonical_events`;
    observation levels on the buffer entries tag how each event is
    framed (direct / indirect / inferred).

    `user_input` is the actual player utterance for the beat (for the
    acting POV). For non-acting POVs in a multi-human beat it should be
    the empty string — they didn't speak this turn. The string is shown
    to the narrator as player-action context AND is used verbatim as
    `transcript_entry.user` in the returned envelope. Pre-r7j this was
    always "" and the prompt asked the LLM to echo it as the transcript
    user field, which produced `"{name} — "` in /history forever.

    When `partial_mode=True`, the PARTIAL_MODE_MARKER is prepended to
    the user message so the narrator prompt's rule-15 PARTIAL mode
    fires; the rendered passage then ends mid-attempt to prompt the
    pinned responder's /act.

    Returns `(NarratorFinalOutput, TranscriptEntry)`. The schema only
    carries `final_text` now; the engine constructs the transcript
    entry from the real `user_input` + the rendered prose.

    Side-effect unchanged: appends the exchange into
    `ckpt.narrator_conversations[pov_character_id]` in-place — the
    caller is responsible for saving the checkpoint.
    """
    from app.engine.context_builder import build_player_characters_block

    resolved = _resolve_buffered_events(ckpt, buffered_events)

    # POV character identity. Fall back to the raw id if the roster
    # doesn't know them (pristine tests, legacy checkpoints).
    pov_char = next(
        (c for c in ckpt.characters if c.character_id == pov_character_id),
        None,
    )
    acting_name = pov_char.name if pov_char else pov_character_id

    from app.engine.context_builder import build_setting_summary
    setting_summary = build_setting_summary(ckpt)
    narrative_rules = ckpt.config.narrative_rules or "No specific narrative rules."
    player_characters_block = build_player_characters_block(ckpt, pov_character_id)
    canonical_event_block = _format_canonical_events_block(
        resolved, pov_character_id,
    )

    pov_history = ckpt.narrator_conversations.setdefault(pov_character_id, [])

    render_t0 = time.monotonic()
    messages = prompt_mgr.render_conversation(
        "narrator_phase2",
        history=pov_history,
        setting_summary=setting_summary,
        narrative_rules=narrative_rules,
        canonical_event=canonical_event_block,
        user_input=user_input,
        acting_character_name=acting_name,
        player_characters_block=player_characters_block,
    )
    render_ms = (time.monotonic() - render_t0) * 1000

    # Prepend the PARTIAL marker to the per-turn user message body so
    # rule-15 PARTIAL mode fires in the prompt.
    user_content = messages[-1]["content"]
    if partial_mode:
        user_content = f"{PARTIAL_MODE_MARKER}\n\n{user_content}"
        messages[-1] = {"role": "user", "content": user_content}

    logger.info(
        "compose_pov_render: pov=%s events=%d partial=%s history=%d msgs "
        "(prompt_render_ms=%.1f)",
        pov_character_id, len(resolved), partial_mode, len(pov_history),
        render_ms,
    )

    response = await client.complete(
        role="narrator",
        messages=messages,
        response_model=NarratorFinalOutput,
        temperature=0.5,
        max_tokens=4000,
        cache=True,
        compact=True,
    )
    result: NarratorFinalOutput = response.parsed

    # Append user + assistant to the POV's rolling history. We can't
    # reuse `append_turn_to_conversation` here because the user content
    # we want stored is the marker-prepended string (when partial);
    # the helper would pick up `user_content` from the messages list
    # anyway, so just use it directly for clarity.
    append_turn_to_conversation(pov_history, user_content, response)

    final_text = result.final_text if result is not None else ""
    logger.info(
        "compose_pov_render: pov=%s rendered %d chars",
        pov_character_id, len(final_text),
    )
    # Defensive fallback when the SDK gives us no parsed envelope —
    # synthesize an empty one rather than crash run_beat. In practice
    # this never fires; the schema is required and the caller would
    # have raised on the parse error.
    if result is None:
        result = NarratorFinalOutput(final_text="")
    transcript_entry = TranscriptEntry(
        user=user_input, assistant=final_text,
    )
    return result, transcript_entry
