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
    replace_character_ids_for_narrator,
)
from app.engine.text_safety import strip_terminal_control
from app.schemas.content_privacy import redact_imported_content_metadata_text
from app.engine.turn_loop_contracts import PARTIAL_MODE_MARKER
from app.engine.visual_context import (
    format_narrator_visual_introductions,
    mark_visual_introductions,
    plan_render_visual_introductions,
)
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import EventRouterOutput
from app.schemas.narrator import (
    NarratorFinalOutput,
    NarratorOutput,
    TranscriptEntry,
    VisualNovelNarratorOutput,
    narrator_plain_text,
    visual_novel_pages_contain_source_identifiers,
    visual_novel_text_contains_source_identifiers,
)
from app.schemas.state import RenderBufferEntry

logger = logging.getLogger(__name__)


def _safe_narrator_prompt_context(value: object) -> str:
    return redact_imported_content_metadata_text(
        strip_terminal_control(str(value or ""))
    ).strip()


def _checkpoint_roster_source_ids(ckpt: CheckpointFile) -> tuple[str, ...]:
    return tuple(
        character.character_id
        for character in ckpt.characters
        if character.character_id
    )


def _assert_visual_novel_output_is_player_safe(
    ckpt: CheckpointFile,
    result: VisualNovelNarratorOutput,
) -> None:
    if visual_novel_pages_contain_source_identifiers(
        result.pages,
        source_ids=_checkpoint_roster_source_ids(ckpt),
    ):
        raise ValueError(
            "visual-novel narrator output exposed an engine source identifier"
        )


def assert_narrator_handoff_policy(
    result: NarratorOutput,
    *,
    handoff_policy: str,
) -> None:
    """Reject a provider judgment that contradicts a forced boundary."""

    if handoff_policy == "forced" and result.handoff != "render":
        raise ValueError(
            "narrator returned handoff='continue' under forced handoff policy"
        )


def _assert_visual_novel_correction_preserves_contract(
    rejected: VisualNovelNarratorOutput,
    corrected: VisualNovelNarratorOutput,
    *,
    source_ids: tuple[str, ...],
) -> None:
    """Allow one correction to change only fields that exposed source ids."""

    if corrected.handoff != rejected.handoff:
        raise ValueError("visual-novel correction changed the handoff decision")
    if len(corrected.pages) != len(rejected.pages):
        raise ValueError("visual-novel correction changed the page count")
    for index, (before, after) in enumerate(
        zip(rejected.pages, corrected.pages, strict=True)
    ):
        if after.kind != before.kind:
            raise ValueError(
                "visual-novel correction changed page kind/order "
                f"at page {index}"
            )
        for field_name in ("speaker", "text"):
            before_value = getattr(before, field_name)
            if visual_novel_text_contains_source_identifiers(
                before_value,
                source_ids=source_ids,
            ):
                continue
            if getattr(after, field_name) != before_value:
                raise ValueError(
                    "visual-novel correction changed an already-safe "
                    f"{field_name} field at page {index}"
                )


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
            if (
                cleaned := redact_imported_content_metadata_text(
                    strip_terminal_control(_strip_loadout_tag(fact.text))
                )
            )
        ]
        if ckpt is not None:
            facts = [
                replace_character_ids_for_narrator(fact, ckpt)
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
) -> tuple[NarratorOutput, "TranscriptEntry"]:
    """v11 per-POV narrator entry point.

    Renders the beat from `pov_character_id`'s point of view in
    second-person present tense, using their per-character rolling
    player-and-assistant conversation history
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

    Returns the active narrator schema plus a `TranscriptEntry`. Prose mode
    carries `final_text`; visual-novel mode carries ordered semantic pages.
    The engine constructs the shared text projection from the real
    `user_input` and whichever accepted schema was selected.

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
    setting_summary = _safe_narrator_prompt_context(
        build_setting_summary(ckpt)
    )
    narrative_rules = (
        _safe_narrator_prompt_context(ckpt.session.config.narrative_rules)
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
    visual_intro_block = format_narrator_visual_introductions(
        visual_intro_plan.loadouts,
    )
    rendering_note = (
        PARTIAL_MODE_MARKER
        if partial_mode
        else "Write through to the natural handoff point."
    )

    pov_history = ckpt.narrator_conversations.get(pov_character_id, [])

    render_t0 = time.monotonic()
    visual_novel = (
        ckpt.session.config.settings.presentation_mode == "visual_novel"
    )
    messages = prompt_mgr.render_conversation(
        "narrator_visual_novel" if visual_novel else "narrator_phase2",
        history=pov_history,
        setting_summary=setting_summary,
        narrative_rules=narrative_rules,
        visible_events=visible_events_block,
        first_meeting_context=(visual_intro_block or "None."),
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
        pov_character_id, len(resolved), partial_mode, len(pov_history),
        render_ms,
    )

    response_model = (
        VisualNovelNarratorOutput if visual_novel else NarratorFinalOutput
    )
    result: NarratorOutput | None = None
    rejected_result: VisualNovelNarratorOutput | None = None
    roster_source_ids = _checkpoint_roster_source_ids(ckpt)
    for attempt in range(2 if visual_novel else 1):
        response = await client.complete(
            role="narrator",
            messages=messages,
            response_model=response_model,
            temperature=0.5,
            max_tokens=8000,
            cache=True,
            compact=True,
        )
        result = response.parsed
        if result is None:
            raise RuntimeError("Narrator returned no structured result.")
        assert_narrator_handoff_policy(
            result,
            handoff_policy=handoff_policy,
        )
        if isinstance(result, VisualNovelNarratorOutput):
            try:
                _assert_visual_novel_output_is_player_safe(ckpt, result)
                if rejected_result is not None:
                    _assert_visual_novel_correction_preserves_contract(
                        rejected_result,
                        result,
                        source_ids=roster_source_ids,
                    )
            except ValueError:
                if attempt:
                    raise
                rejected_result = result.model_copy(deep=True)
                messages = [
                    *messages,
                    {
                        "role": "assistant",
                        "content": result.model_dump_json(),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Return corrected JSON only. Keep the handoff, page "
                            "count, page kinds, and page order unchanged. Change "
                            "only speaker or text fields that contain a source "
                            "identifier; preserve every other speaker and text "
                            "field exactly. Remove every source identifier. Do "
                            "not transform one into a guessed proper name. Use "
                            "an already established viewpoint-known name when "
                            "the context supplies one; otherwise use a short "
                            "visible description."
                        ),
                    },
                ]
                continue
        break
    assert result is not None
    if isinstance(result, NarratorFinalOutput):
        result.final_text = _strip_unmatched_trailing_closers(result.final_text)
        response.parsed = result

    final_text = narrator_plain_text(result)
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
    result: NarratorOutput,
    user_input: str,
) -> None:
    """Persist one accepted POV conversation turn and visual introductions."""
    if isinstance(result, VisualNovelNarratorOutput):
        _assert_visual_novel_output_is_player_safe(ckpt, result)
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
    if user_input:
        history.append(ConversationMessage(role="user", content=user_input))
    if isinstance(result, VisualNovelNarratorOutput):
        history_payload = {
            "pages": [page.model_dump(mode="json") for page in result.pages]
        }
    else:
        history_payload = {"final_text": result.final_text}
    history.append(ConversationMessage(
        role="assistant",
        content=[{
            "type": "text",
            "text": json.dumps(
                history_payload,
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        }],
    ))
