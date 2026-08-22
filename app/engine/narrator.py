"""Narrator composition — v11 per-POV render path.

`compose_pov_render` is the only production narrator entry point. Each
human POV with a queued perception gets its own render, composed against
a per-character rolling conversation stored on the checkpoint
(`checkpoint.narrator_conversations[pov_character_id]`). Voice and
continuity hold across the session on a per-POV basis.
"""

from __future__ import annotations

import json
import logging
import re
import time

from app.engine.prompt_manager import PromptManager
from app.engine.context_builder import (
    build_narrator_player_characters_block,
    replace_character_ids_with_names,
)
from app.schemas.content_privacy import redact_imported_asset_text
from app.engine.turn_loop_contracts import PARTIAL_MODE_MARKER
from app.engine.visual_context import (
    format_visual_introductions,
    mark_visual_introductions,
    plan_render_visual_introductions,
)
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import EventRouterOutput
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import RenderBufferEntry

logger = logging.getLogger(__name__)


def _strip_unmatched_trailing_closers(text: str) -> str:
    """Remove schema/JSON closers the model sometimes leaks into prose.

    Structured narrator output already parsed successfully at this point, so a
    final unmatched `}` or `]` is not needed for JSON validity. Balanced braces
    inside ordinary prose are preserved.
    """
    cleaned = (text or "").rstrip()
    pairs = {"}": "{", "]": "["}
    while cleaned and cleaned[-1] in pairs:
        closer = cleaned[-1]
        opener = pairs[closer]
        if cleaned.count(opener) >= cleaned.count(closer):
            break
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def _resolve_buffered_events(
    ckpt: CheckpointFile,
    buffered_events: list[RenderBufferEntry],
) -> list[tuple[RenderBufferEntry, EventRouterOutput]]:
    """Walk the render buffer and resolve each entry against
    `ckpt.canonical_events`.

    A missing canonical event makes lossless rendering impossible. Fail before
    calling the narrator so the buffer remains available for diagnosis/retry
    instead of silently flushing an incomplete player-visible sequence.
    """
    by_id: dict[str, EventRouterOutput] = {
        ev.event_id: ev for ev in ckpt.canonical_events
    }
    missing_event_ids = [
        entry.event_id for entry in buffered_events if entry.event_id not in by_id
    ]
    if missing_event_ids:
        missing = ", ".join(dict.fromkeys(missing_event_ids))
        raise RuntimeError(
            "Narrator render buffer references missing canonical event(s): "
            f"{missing}"
        )

    resolved: list[tuple[RenderBufferEntry, EventRouterOutput]] = []
    for entry in buffered_events:
        resolved.append((entry, by_id[entry.event_id]))
    return sorted(
        resolved,
        key=lambda pair: (pair[0].visible_at_s, pair[0].event_sequence),
    )


_OBS_LEVEL_HEADERS = {
    "d": "Seen directly:",
    "i": "Partly perceived:",
    "f": "Aftermath only:",
    "direct": "Seen directly:",
    "indirect": "Partly perceived:",
    "inferred": "Aftermath only:",
}

_LOADOUT_TAG_RE = re.compile(r"^\[loadout\s+[—–-]\s*[^\]]+\]\s*")


def _strip_loadout_tag(text: str) -> str:
    """Remove source tags from harvested appearance facts."""
    return _LOADOUT_TAG_RE.sub("", text or "").strip()


def _format_visible_events_block(
    resolved: list[tuple[RenderBufferEntry, EventRouterOutput]],
    pov_character_id: str = "",
    ckpt: CheckpointFile | None = None,
) -> str:
    """Serialize only POV-visible surface facts for prose composition."""
    if not resolved:
        return "Nothing new reaches this viewpoint."
    sections: list[str] = []
    for entry, ev in resolved:
        header = _OBS_LEVEL_HEADERS.get(entry.observation_level, "Perceived:")
        ca = ev.canonical_event
        visible_facts = []
        for index, fact in enumerate(ca.observable_facts):
            if fact.audience == "all_observers" or (
                pov_character_id and fact.is_visible_to(pov_character_id)
            ):
                visible_facts.append((index, fact))
        facts = [
            cleaned
            for _, fact in sorted(
                visible_facts,
                key=lambda item: (
                    item[1].at_offset_s,
                    item[1].duration_s,
                    item[0],
                ),
            )
            if (cleaned := redact_imported_asset_text(_strip_loadout_tag(fact.text)))
        ]
        if ckpt is not None:
            facts = [
                replace_character_ids_with_names(fact, ckpt)
                for fact in facts
            ]
        if pov_character_id and not facts:
            # No fact visible to this POV means the event must not
            # surface in their render at all.
            continue
        lines = [header]
        if facts:
            for fact in facts:
                lines.append(f"- {fact}")
        else:
            lines.append("Nothing concrete is visible.")
        sections.append("\n".join(lines))
    if not sections:
        return "Nothing new is visible to this viewpoint."
    return "\n\n".join(sections)


async def compose_pov_render(
    client: LLMClient,
    prompt_mgr: PromptManager,
    ckpt: CheckpointFile,
    pov_character_id: str,
    buffered_events: list[RenderBufferEntry],
    partial_mode: bool,
    user_input: str = "",
    handoff_policy: str = "forced",
    handoff_context: str = "",
) -> tuple[NarratorFinalOutput, "TranscriptEntry"]:
    """v11 per-POV narrator entry point.

    Renders the beat from `pov_character_id`'s point of view in
    second-person present tense, using their per-character rolling
    assistant-side conversation history
    (`ckpt.narrator_conversations[pov_character_id]`).

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

    When `partial_mode=True`, the user message carries a natural-language
    instruction to stop before the attempted action resolves.

    Returns `(NarratorFinalOutput, TranscriptEntry)`. The schema only
    carries `final_text` now; the engine constructs the transcript
    entry from the real `user_input` + the rendered prose.

    Composition is side-effect free. The caller commits accepted prose with
    `commit_pov_render`; rejected handoff candidates must not affect narrator
    history or visual-introduction state.
    """
    resolved = _resolve_buffered_events(ckpt, buffered_events)

    # POV character identity. Fall back to the raw id if the roster
    # doesn't know them (pristine tests, legacy checkpoints).
    pov_char = next(
        (c for c in ckpt.characters if c.character_id == pov_character_id),
        None,
    )
    pov_name = pov_char.name if pov_char else pov_character_id

    from app.engine.context_builder import build_setting_summary
    setting_summary = build_setting_summary(ckpt)
    narrative_rules = (
        ckpt.session.config.narrative_rules
        or "No specific narrative rules."
    )
    player_characters_block = build_narrator_player_characters_block(
        ckpt, pov_character_id
    )
    visible_events_block = _format_visible_events_block(
        resolved, pov_character_id, ckpt,
    )
    visual_intro_plan = plan_render_visual_introductions(
        ckpt,
        viewer_id=pov_character_id,
        resolved=resolved,
    )
    visual_intro_block = format_visual_introductions(
        visual_intro_plan.loadouts,
    )
    if visual_intro_block:
        visible_events_block = f"{visible_events_block}\n\n{visual_intro_block}"
    rendering_note = (
        PARTIAL_MODE_MARKER
        if partial_mode
        else "Write through to the natural handoff point."
    )

    pov_history = ckpt.narrator_conversations.setdefault(pov_character_id, [])
    assistant_history = [
        message for message in pov_history
        if message.role == "assistant"
    ]

    render_t0 = time.monotonic()
    messages = prompt_mgr.render_conversation(
        "narrator_phase2",
        history=assistant_history,
        setting_summary=setting_summary,
        narrative_rules=narrative_rules,
        visible_events=visible_events_block,
        user_input=user_input,
        pov_character_name=pov_name,
        player_characters_block=player_characters_block,
        rendering_note=rendering_note,
        handoff_policy=handoff_policy,
        handoff_context=(handoff_context or "No unresolved handoff condition."),
    )
    render_ms = (time.monotonic() - render_t0) * 1000

    logger.info(
        "compose_pov_render: pov=%s events=%d partial=%s history=%d msgs "
        "(prompt_render_ms=%.1f)",
        pov_character_id, len(resolved), partial_mode, len(assistant_history),
        render_ms,
    )

    response = await client.complete(
        role="narrator",
        messages=messages,
        response_model=NarratorFinalOutput,
        temperature=0.5,
        max_tokens=8000,
        cache=True,
        compact=True,
    )
    result: NarratorFinalOutput = response.parsed
    if result is not None:
        result.final_text = _strip_unmatched_trailing_closers(result.final_text)
        response.parsed = result

    if result is None:
        raise RuntimeError("Narrator returned no structured result.")
    final_text = result.final_text
    logger.info(
        "compose_pov_render: pov=%s rendered %d chars",
        pov_character_id, len(final_text),
    )
    transcript_entry = TranscriptEntry(
        user=user_input, assistant=final_text,
    )
    return result, transcript_entry


def commit_pov_render(
    ckpt: CheckpointFile,
    *,
    pov_character_id: str,
    buffered_events: list[RenderBufferEntry],
    result: NarratorFinalOutput,
) -> None:
    """Persist one accepted narrator composition and its visual introductions."""
    resolved = _resolve_buffered_events(ckpt, buffered_events)
    visual_intro_plan = plan_render_visual_introductions(
        ckpt,
        viewer_id=pov_character_id,
        resolved=resolved,
    )
    if visual_intro_plan.mark_character_ids:
        mark_visual_introductions(
            ckpt,
            pov_character_id,
            visual_intro_plan.mark_character_ids,
        )
    history = ckpt.narrator_conversations.setdefault(pov_character_id, [])
    history.append(ConversationMessage(
        role="assistant",
        content=[{
            "type": "text",
            "text": json.dumps(
                {"final_text": result.final_text},
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        }],
    ))
