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
from typing import Literal

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
    visually_staged_character_ids,
)
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import EventRouterOutput
from app.schemas.events import visible_fact_texts
from app.schemas.narrator import (
    NarratorFinalOutput,
    NarratorOutput,
    TranscriptEntry,
    VisualNovelBeatPages,
    VisualNovelNarratorOutput,
    narrator_plain_text,
    visual_novel_pages_contain_source_identifiers,
    visual_novel_text_contains_source_identifiers,
)
from app.schemas.state import RenderBufferEntry

logger = logging.getLogger(__name__)

NarrationMode = Literal["event_aligned", "compressed_sequence"]


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
        (page for beat in result.beats for page in beat.pages),
        source_ids=_checkpoint_roster_source_ids(ckpt),
    ):
        raise ValueError(
            "visual-novel narrator output exposed an engine source identifier"
        )


def _visual_novel_sprite_roster(
    ckpt: CheckpointFile,
    *,
    viewer_id: str,
    resolved: list[tuple[RenderBufferEntry, EventRouterOutput]],
) -> tuple[str, ...]:
    texts: list[str] = []
    present_ids: set[str] = set()
    for entry, event in resolved:
        if entry.observation_level != "direct":
            continue
        for fact in event.canonical_event.observable_facts:
            if fact.is_visible_to(viewer_id):
                present_ids.update(fact.visual_subject_ids)
        texts.extend(
            visible_fact_texts(
                event.canonical_event.observable_facts,
                viewer_id,
                include_all_observers=True,
            )
        )
    present_ids.update(visually_staged_character_ids(ckpt, texts))
    labels: list[str] = []
    for character in ckpt.characters:
        if (
            character.character_id not in present_ids
            or character.character_id == viewer_id
        ):
            continue
        label = " ".join((character.name or "").split()).strip()
        if label:
            labels.append(label)
    unique_labels = tuple(
        label for label in dict.fromkeys(labels) if labels.count(label) == 1
    )
    return unique_labels


def _visual_novel_output_sprite_rosters(
    ckpt: CheckpointFile,
    *,
    viewer_id: str,
    resolved: list[tuple[RenderBufferEntry, EventRouterOutput]],
    narration_mode: NarrationMode,
) -> tuple[tuple[str, ...], ...]:
    """Return the safe foreground roster for each requested output segment."""

    if narration_mode == "compressed_sequence":
        return (
            _visual_novel_sprite_roster(
                ckpt,
                viewer_id=viewer_id,
                resolved=resolved,
            ),
        )
    return tuple(
        _visual_novel_sprite_roster(
            ckpt,
            viewer_id=viewer_id,
            resolved=[pair],
        )
        for pair in resolved
    )


def _assert_visual_novel_sprite_cues(
    result: VisualNovelNarratorOutput,
    *,
    allowed_labels_by_beat: tuple[tuple[str, ...], ...],
) -> None:
    if result.handoff == "continue":
        if result.beats:
            raise ValueError(
                "visual-novel continue decisions cannot contain authored beats"
            )
        return
    if len(result.beats) != len(allowed_labels_by_beat):
        raise ValueError(
            "visual-novel narrator must return one beat for each visible event"
        )
    for beat, allowed_labels in zip(
        result.beats,
        allowed_labels_by_beat,
        strict=True,
    ):
        allowed = set(allowed_labels)
        for page in beat.pages:
            if any(label not in allowed for label in page.sprites):
                raise ValueError(
                    "visual-novel narrator selected an unavailable sprite character"
                )
            if page.kind != "dialogue":
                continue
            if page.speaker in allowed:
                if not page.sprites or page.sprites[0] != page.speaker:
                    raise ValueError(
                        "visual-novel narrator must put the exact available "
                        "dialogue speaker first in the sprite cues"
                    )
            elif page.sprites:
                raise ValueError(
                    "visual-novel narrator used a descriptive dialogue speaker "
                    "with rostered sprite cues"
                )


def _repair_visual_novel_sprite_cues(
    result: VisualNovelNarratorOutput,
    *,
    allowed_labels_by_beat: tuple[tuple[str, ...], ...],
) -> VisualNovelNarratorOutput:
    """Deterministically constrain presentation-only cues after one retry.

    The narrator owns page prose and foreground preference. The engine owns
    which resolved sprites are actually available for each canonical event.
    Filtering an unavailable cue, or placing an available dialogue speaker
    first, cannot change story truth and prevents a presentation mistake from
    aborting an otherwise valid committed turn.
    """

    repaired = result.model_copy(deep=True)
    if repaired.handoff == "continue" or len(repaired.beats) != len(
        allowed_labels_by_beat
    ):
        return repaired
    for beat, allowed_labels in zip(
        repaired.beats,
        allowed_labels_by_beat,
        strict=True,
    ):
        allowed = set(allowed_labels)
        for page in beat.pages:
            available = list(dict.fromkeys(
                label for label in page.sprites if label in allowed
            ))
            if page.kind == "dialogue":
                if page.speaker in allowed:
                    available = [
                        page.speaker,
                        *(label for label in available if label != page.speaker),
                    ]
                else:
                    available = []
            page.sprites = available[:2]
    return repaired


def _visual_novel_page_sprite_cues_are_valid(
    page: object,
    *,
    allowed_sprite_labels: tuple[str, ...],
    source_ids: tuple[str, ...],
) -> bool:
    allowed = set(allowed_sprite_labels)
    sprites = getattr(page, "sprites", ())
    if any(
        label not in allowed
        or visual_novel_text_contains_source_identifiers(
            label,
            source_ids=source_ids,
        )
        for label in sprites
    ):
        return False
    if getattr(page, "kind", "") != "dialogue":
        return True
    speaker = getattr(page, "speaker", "")
    if speaker in allowed:
        return bool(sprites) and sprites[0] == speaker
    return not sprites


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
    allowed_sprite_labels_by_beat: tuple[tuple[str, ...], ...],
) -> None:
    """Allow one correction to change only fields that exposed source ids."""

    if corrected.handoff != rejected.handoff:
        raise ValueError("visual-novel correction changed the handoff decision")
    if len(corrected.beats) != len(rejected.beats):
        raise ValueError("visual-novel correction changed the beat count")
    if not rejected.beats:
        return
    if len(rejected.beats) != len(allowed_sprite_labels_by_beat):
        raise ValueError(
            "visual-novel rejected render lost visible-event alignment"
        )
    before_pages = [page for beat in rejected.beats for page in beat.pages]
    after_pages = [page for beat in corrected.beats for page in beat.pages]
    if [len(beat.pages) for beat in corrected.beats] != [
        len(beat.pages) for beat in rejected.beats
    ]:
        raise ValueError("visual-novel correction changed a beat page count")
    allowed_by_page = [
        allowed
        for allowed, beat in zip(
            allowed_sprite_labels_by_beat,
            rejected.beats,
            strict=True,
        )
        for _page in beat.pages
    ]
    for index, (before, after, allowed_sprite_labels) in enumerate(
        zip(before_pages, after_pages, allowed_by_page, strict=True)
    ):
        if after.kind != before.kind:
            raise ValueError(
                f"visual-novel correction changed page kind/order at page {index}"
            )
        before_sprite_contract_safe = _visual_novel_page_sprite_cues_are_valid(
            before,
            allowed_sprite_labels=allowed_sprite_labels,
            source_ids=source_ids,
        )
        for field_name in ("speaker", "text"):
            before_value = getattr(before, field_name)
            if visual_novel_text_contains_source_identifiers(
                before_value,
                source_ids=source_ids,
            ):
                continue
            if field_name == "speaker" and not before_sprite_contract_safe:
                continue
            if getattr(after, field_name) != before_value:
                raise ValueError(
                    "visual-novel correction changed an already-safe "
                    f"{field_name} field at page {index}"
                )
        if before_sprite_contract_safe and after.sprites != before.sprites:
            raise ValueError(
                "visual-novel correction changed already-safe sprite cues "
                f"at page {index}"
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


def resolve_buffered_events_for_render(
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
            f"Narrator render buffer references missing canonical event(s): {missing}"
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
    sprite_labels_by_beat: tuple[tuple[str, ...], ...] = (),
) -> str:
    """Serialize one explicit, ordered input beat per buffered event."""
    if not resolved:
        return "Nothing new reaches this viewpoint."
    if sprite_labels_by_beat and len(sprite_labels_by_beat) != len(resolved):
        raise ValueError("sprite roster count must match visible event count")
    sections: list[str] = []
    for beat_index, (entry, ev) in enumerate(resolved, start=1):
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
            facts = [replace_character_ids_for_narrator(fact, ckpt) for fact in facts]
        if pov_character_id and not facts:
            raise RuntimeError(
                "Narrator render buffer contains an event with no visible "
                f"facts for {pov_character_id}: {entry.event_id}"
            )
        lines = [f'<visible_beat index="{beat_index}">', header]
        if facts:
            for fact in facts:
                lines.append(f"- {fact}")
        else:
            lines.append("Nothing concrete is visible.")
        if sprite_labels_by_beat:
            labels = sprite_labels_by_beat[beat_index - 1]
            lines.append("Available foreground characters:")
            lines.extend(f"- {label}" for label in labels)
            if not labels:
                lines.append("None.")
        lines.append("</visible_beat>")
        sections.append("\n".join(lines))
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
    narration_mode: NarrationMode = "event_aligned",
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
    if narration_mode not in {"event_aligned", "compressed_sequence"}:
        raise ValueError(f"unknown narrator mode: {narration_mode!r}")
    if narration_mode == "compressed_sequence" and partial_mode:
        raise ValueError("compressed narration cannot render a partial beat")
    resolved = resolve_buffered_events_for_render(ckpt, buffered_events)

    # POV character identity. Fall back to the raw id if the roster
    # doesn't know them (pristine tests, legacy checkpoints).
    pov_char = next(
        (c for c in ckpt.characters if c.character_id == pov_character_id),
        None,
    )
    pov_name = pov_char.name if pov_char else pov_character_id

    from app.engine.context_builder import build_setting_summary

    setting_summary = _safe_narrator_prompt_context(build_setting_summary(ckpt))
    narrative_rules = (
        _safe_narrator_prompt_context(ckpt.session.config.narrative_rules)
        or "No specific narrative rules."
    )
    player_characters_block = build_narrator_player_characters_block(
        ckpt, pov_character_id
    )
    output_sprite_rosters = _visual_novel_output_sprite_rosters(
        ckpt,
        viewer_id=pov_character_id,
        resolved=resolved,
        narration_mode=narration_mode,
    )
    input_sprite_rosters = (
        output_sprite_rosters * len(resolved)
        if narration_mode == "compressed_sequence"
        else output_sprite_rosters
    )
    visible_events_block = _format_visible_events_block(
        resolved,
        pov_character_id,
        ckpt,
        sprite_labels_by_beat=(
            input_sprite_rosters
            if ckpt.session.config.settings.presentation_mode == "visual_novel"
            else ()
        ),
    )
    visual_intro_plan = plan_render_visual_introductions(
        ckpt,
        viewer_id=pov_character_id,
        resolved=resolved,
    )
    visual_intro_block = format_narrator_visual_introductions(
        visual_intro_plan.loadouts,
    )
    if partial_mode:
        rendering_note = PARTIAL_MODE_MARKER
    elif narration_mode == "compressed_sequence":
        rendering_note = (
            "Treat every supplied visible beat as one rapid conflict sequence. "
            "Return one concise passage for the whole sequence, normally one "
            "to three visual-novel pages when that format is active."
        )
    else:
        rendering_note = "Write through to the natural handoff point."

    pov_history = ckpt.narrator_conversations.get(pov_character_id, [])

    render_t0 = time.monotonic()
    visual_novel = ckpt.session.config.settings.presentation_mode == "visual_novel"
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
        "compose_pov_render: pov=%s events=%d partial=%s mode=%s history=%d msgs "
        "(prompt_render_ms=%.1f)",
        pov_character_id,
        len(resolved),
        partial_mode,
        narration_mode,
        len(pov_history),
        render_ms,
    )

    response_model = VisualNovelNarratorOutput if visual_novel else NarratorFinalOutput
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
            if (
                narration_mode == "compressed_sequence"
                and result.handoff == "render"
                and len(result.beats) > 1
            ):
                result.beats = [VisualNovelBeatPages(pages=[
                    page
                    for beat in result.beats
                    for page in beat.pages
                ])]
            if rejected_result is not None:
                _assert_visual_novel_correction_preserves_contract(
                    rejected_result,
                    result,
                    source_ids=roster_source_ids,
                    allowed_sprite_labels_by_beat=output_sprite_rosters,
                )
            try:
                _assert_visual_novel_output_is_player_safe(ckpt, result)
            except ValueError:
                if attempt:
                    raise
            else:
                try:
                    _assert_visual_novel_sprite_cues(
                        result,
                        allowed_labels_by_beat=output_sprite_rosters,
                    )
                except ValueError:
                    if attempt:
                        repaired = _repair_visual_novel_sprite_cues(
                            result,
                            allowed_labels_by_beat=output_sprite_rosters,
                        )
                        _assert_visual_novel_sprite_cues(
                            repaired,
                            allowed_labels_by_beat=output_sprite_rosters,
                        )
                        logger.warning(
                            "visual-novel narrator sprite cues required "
                            "deterministic normalization after correction"
                        )
                        result = repaired
                        break
                else:
                    break
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
                        "Return corrected JSON only. Keep the handoff, beat "
                        "count, page counts, page kinds, and page order "
                        "unchanged. Change only speaker or text fields that "
                        "contain a source identifier, or speaker and sprites "
                        "fields that violate a visible beat's supplied "
                        "Available foreground characters list; preserve every "
                        "other field exactly. Sprite labels never carry from "
                        "one visible beat into another unless the later beat "
                        "lists them again. Remove unavailable sprite labels. "
                        "For rostered dialogue, use the exact supplied label "
                        "as speaker and put it first in sprites. For an "
                        "unrostered speaker, use a short established "
                        "player-safe label and leave sprites empty. Remove "
                        "every source identifier. Do not transform one into a "
                        "guessed proper name."
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
        pov_character_id,
        len(final_text),
    )
    transcript_entry = TranscriptEntry(
        user=user_input,
        assistant=final_text,
    )
    return result, transcript_entry


def commit_pov_render(
    ckpt: CheckpointFile,
    *,
    pov_character_id: str,
    buffered_events: list[RenderBufferEntry],
    result: NarratorOutput,
    user_input: str,
    narration_mode: NarrationMode = "event_aligned",
) -> None:
    """Persist one accepted POV conversation turn and visual introductions."""
    resolved = resolve_buffered_events_for_render(ckpt, buffered_events)
    if isinstance(result, VisualNovelNarratorOutput):
        _assert_visual_novel_output_is_player_safe(ckpt, result)
        _assert_visual_novel_sprite_cues(
            result,
            allowed_labels_by_beat=_visual_novel_output_sprite_rosters(
                ckpt,
                viewer_id=pov_character_id,
                resolved=resolved,
                narration_mode=narration_mode,
            ),
        )
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
            # Sprite cues are transient presentation metadata. Replaying them
            # would spend narrator context on prior layout rather than story
            # continuity, so durable history keeps only the semantic pages.
            "pages": [
                page.model_dump(mode="json", exclude={"sprites"})
                for beat in result.beats
                for page in beat.pages
            ]
        }
    else:
        history_payload = {"final_text": result.final_text}
    history.append(
        ConversationMessage(
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": json.dumps(
                        history_payload,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ),
                }
            ],
        )
    )
