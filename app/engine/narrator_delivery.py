"""Per-lane narrator timing and durable player delivery."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Iterable, Protocol

from app.engine.delivery_outbox import enqueue_delivery
from app.engine.narrator import (
    commit_pov_render,
    resolve_buffered_events_for_render,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.characters import CharacterStatus
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.delivery import (
    DeliveryPayload,
    DeliveryVisualNovelRender,
    DeliveryVisualNovelSegment,
    NarratorEventRef,
    NarratorRenderJob,
)
from app.schemas.narrator import (
    NarratorOutput,
    TranscriptEntry,
    VisualNovelNarratorOutput,
    narrator_plain_text,
)


class NarratorDispatcher(Protocol):
    async def narrator_compose(
        self,
        *,
        ckpt: CheckpointFile,
        character_id: str,
        event_refs: list[NarratorEventRef],
        partial_mode: bool,
        user_input: str,
        handoff_policy: str,
        handoff_context: str,
        narration_mode: str,
    ) -> tuple[NarratorOutput, TranscriptEntry]: ...


@dataclass(slots=True)
class NarratorLaneOutcome:
    lane_id: str
    rendered_pov_ids: list[str] = field(default_factory=list)
    continuing_pov_ids: list[str] = field(default_factory=list)
    failed_pov_ids: list[str] = field(default_factory=list)

    @property
    def lane_is_gated(self) -> bool:
        return bool(self.rendered_pov_ids or self.failed_pov_ids)


def _clone_checkpoint(checkpoint: CheckpointFile) -> CheckpointFile:
    return CheckpointFile.model_validate_json(checkpoint.model_dump_json(
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
    ))


def _replace_checkpoint(target: CheckpointFile, source: CheckpointFile) -> None:
    target.__dict__.clear()
    target.__dict__.update(source.__dict__)
    target.__pydantic_fields_set__ = set(source.__pydantic_fields_set__)
    target.__pydantic_extra__ = source.__pydantic_extra__
    target.__pydantic_private__ = source.__pydantic_private__


def _variant_snapshot_by_label(
    checkpoint: CheckpointFile,
    *,
    refs: list[NarratorEventRef],
    pages: list[object],
) -> dict[str, str]:
    if not refs:
        return {}
    variants = refs[-1].sprite_variant_keys_by_character_id
    labels = {
        str(label).strip()
        for page in pages
        for label in getattr(page, "sprites", ())
        if str(label).strip()
    }
    active = [
        character
        for character in checkpoint.characters
        if character.status != CharacterStatus.culled
    ]
    counts: dict[str, int] = {}
    for character in active:
        label = " ".join(character.name.split())
        if label:
            counts[label] = counts.get(label, 0) + 1
    return {
        label: variants.get(character.character_id, "neutral")
        for character in active
        if (label := " ".join(character.name.split())) in labels
        and counts.get(label) == 1
    }


def _visual_novel_delivery(
    checkpoint: CheckpointFile,
    *,
    job: NarratorRenderJob,
    result: VisualNovelNarratorOutput,
) -> DeliveryVisualNovelRender:
    resolved = resolve_buffered_events_for_render(checkpoint, job.event_refs)
    if job.narration_mode == "compressed_sequence":
        if len(result.beats) != 1:
            raise RuntimeError("compressed narrator output lost sequence alignment")
        pages = [page.model_copy(deep=True) for page in result.beats[0].pages]
        return DeliveryVisualNovelRender(segments=[DeliveryVisualNovelSegment(
            pages=pages,
            rendered_event_ids=[ref.event_id for ref, _event in resolved],
            sprite_variant_keys_by_label=_variant_snapshot_by_label(
                checkpoint,
                refs=[ref for ref, _event in resolved],
                pages=pages,
            ),
        )])
    if len(result.beats) != len(resolved):
        raise RuntimeError("narrator output lost event alignment")
    return DeliveryVisualNovelRender(segments=[
        DeliveryVisualNovelSegment(
            pages=[page.model_copy(deep=True) for page in section.pages],
            rendered_event_ids=[ref.event_id],
            sprite_variant_keys_by_label=_variant_snapshot_by_label(
                checkpoint,
                refs=[ref],
                pages=list(section.pages),
            ),
        )
        for section, (ref, _event) in zip(result.beats, resolved, strict=True)
    ])


def _ungate_pov_lane(
    checkpoint: CheckpointFile,
    *,
    lane_id: str,
    pov_character_id: str,
) -> None:
    for turn in checkpoint.session.router_frontier:
        if turn.lane_id != lane_id:
            continue
        turn.gating_pov_ids = [
            value
            for value in turn.gating_pov_ids
            if value != pov_character_id
        ]


def _reaction_prompt_for_job(
    checkpoint: CheckpointFile,
    job: NarratorRenderJob,
) -> str:
    obligation = checkpoint.session.action_obligations.get(job.pov_character_id)
    if (
        obligation is not None
        and obligation.kind == "combat_reaction"
        and obligation.source_event_id in job.source_event_ids
    ):
        return obligation.source_event_id
    return ""


def _loot_offers_for_job(
    checkpoint: CheckpointFile,
    job: NarratorRenderJob,
) -> list[str]:
    return [
        offer.offer_id
        for offer in checkpoint.session.dnd_inventory_offers
        if offer.status == "open"
        and offer.source_event_id in job.source_event_ids
        and job.pov_character_id in offer.eligible_character_ids
    ]


def _handoff_for_job(
    checkpoint: CheckpointFile,
    job: NarratorRenderJob,
) -> tuple[str, str]:
    obligation = checkpoint.session.action_obligations.get(job.pov_character_id)
    if job.partial_mode or (
        obligation is not None
        and obligation.source_event_id in job.source_event_ids
    ):
        return (
            "forced",
            "The visible event has reached a consequential choice owned by "
            "this viewpoint character.",
        )
    turns = [
        turn
        for turn in checkpoint.session.router_frontier
        if turn.lane_id == job.lane_id
    ]
    if not turns:
        return "forced", "No established motion remains before player control."
    roster = {item.character_id: item.name for item in checkpoint.characters}
    descriptions = [
        (
            f"{roster.get(turn.actor_id, turn.actor_id)} has an immediate "
            "established response"
            if turn.actor_id
            else "established world motion remains unresolved"
        )
        for turn in turns
    ]
    return "candidate", "; ".join(descriptions) + "."


async def process_narrator_lane(
    checkpoint: CheckpointFile,
    dispatcher: NarratorDispatcher,
    *,
    lane_id: str,
) -> NarratorLaneOutcome:
    """Ask every observing POV in one lane and apply render-dominant timing.

    Calls are independent and concurrent. If every successful POV asks to
    continue, their event refs accumulate for the next event in this lane. Any
    render decision makes every successful candidate deliverable. A provider
    failure affects only that POV and keeps this lane gated without undoing
    canonical fiction or another POV's completed delivery.
    """

    return (await process_narrator_lanes(
        checkpoint,
        dispatcher,
        lane_ids=[lane_id],
    ))[0]


def _apply_lane_results(
    checkpoint: CheckpointFile,
    *,
    lane_id: str,
    jobs: list[NarratorRenderJob],
    raw: list[object],
) -> NarratorLaneOutcome:
    outcome = NarratorLaneOutcome(lane_id=lane_id)
    successful: list[tuple[NarratorRenderJob, NarratorOutput, TranscriptEntry]] = []
    for job, value in zip(jobs, raw, strict=True):
        if isinstance(value, BaseException):
            job.status = "failed"
            job.last_error = f"{type(value).__name__}: {value}"[:1000]
            outcome.failed_pov_ids.append(job.pov_character_id)
            continue
        successful.append(value)

    render_dominates = any(
        result.handoff == "render" for _job, result, _entry in successful
    )
    if not render_dominates:
        for job, _result, _entry in successful:
            job.status = "pending"
            job.last_error = ""
            outcome.continuing_pov_ids.append(job.pov_character_id)
            if not outcome.failed_pov_ids:
                _ungate_pov_lane(
                    checkpoint,
                    lane_id=lane_id,
                    pov_character_id=job.pov_character_id,
                )
        return outcome

    for job, result, transcript in successful:
        staged = _clone_checkpoint(checkpoint)
        staged_job = next(
            (
                item
                for item in staged.session.narrator_render_jobs
                if item.job_id == job.job_id
            ),
            None,
        )
        if staged_job is None:
            raise RuntimeError("narrator job disappeared before delivery commit")
        try:
            commit_pov_render(
                staged,
                pov_character_id=staged_job.pov_character_id,
                buffered_events=staged_job.event_refs,
                result=result,
                user_input=transcript.user,
                narration_mode=staged_job.narration_mode,
            )
            visual_novel = (
                _visual_novel_delivery(staged, job=staged_job, result=result)
                if isinstance(result, VisualNovelNarratorOutput)
                else None
            )
            enqueue_delivery(
                staged,
                pov_character_id=staged_job.pov_character_id,
                source_event_ids=list(staged_job.source_event_ids),
                highest_event_sequence=staged_job.highest_event_sequence,
                payload=DeliveryPayload(
                    prose=narrator_plain_text(result),
                    visual_novel=visual_novel,
                    asset_reveals=[],
                    reaction_prompt_event_id=_reaction_prompt_for_job(
                        staged,
                        staged_job,
                    ),
                    loot_offer_ids=_loot_offers_for_job(staged, staged_job),
                    commitment_revision_ids=[
                        prompt.commitment_id
                        for character_id, prompt in (
                            staged.session.pending_commitment_revisions.items()
                        )
                        if character_id == staged_job.pov_character_id
                    ],
                    dice_rolls=list(staged_job.dice_rolls),
                    experience_awards=list(staged_job.experience_awards),
                    owner_error="",
                ),
            )
            staged.session.narrator_render_jobs = [
                item
                for item in staged.session.narrator_render_jobs
                if item.job_id != staged_job.job_id
            ]
        except Exception as exc:  # deterministic delivery failures are retryable
            live_job = next(
                item
                for item in checkpoint.session.narrator_render_jobs
                if item.job_id == job.job_id
            )
            live_job.status = "failed"
            live_job.last_error = f"{type(exc).__name__}: {exc}"[:1000]
            outcome.failed_pov_ids.append(job.pov_character_id)
            continue
        _replace_checkpoint(checkpoint, staged)
        outcome.rendered_pov_ids.append(job.pov_character_id)
    return outcome


async def process_narrator_lanes(
    checkpoint: CheckpointFile,
    dispatcher: NarratorDispatcher,
    *,
    lane_ids: Iterable[str],
) -> list[NarratorLaneOutcome]:
    """Process independent lanes concurrently from one immutable event state."""

    ordered = list(dict.fromkeys(lane_ids))
    if not ordered:
        return []
    jobs_by_lane = {
        lane_id: [
            item
            for item in checkpoint.session.narrator_render_jobs
            if item.lane_id == lane_id and item.status in {"pending", "failed"}
        ]
        for lane_id in ordered
    }

    async def _one(
        job: NarratorRenderJob,
        *,
        force: bool = False,
    ) -> tuple[NarratorRenderJob, NarratorOutput, TranscriptEntry]:
        job.attempts += 1
        handoff_policy, handoff_context = _handoff_for_job(checkpoint, job)
        if force:
            handoff_policy = "forced"
        result, transcript = await dispatcher.narrator_compose(
            ckpt=checkpoint,
            character_id=job.pov_character_id,
            event_refs=list(job.event_refs),
            partial_mode=job.partial_mode,
            user_input=job.user_input,
            handoff_policy=handoff_policy,
            handoff_context=handoff_context,
            narration_mode=job.narration_mode,
        )
        return job, result, transcript

    flattened = [job for lane_id in ordered for job in jobs_by_lane[lane_id]]
    raw = list(await asyncio.gather(
        *(_one(job) for job in flattened),
        return_exceptions=True,
    ))
    lane_cursor = 0
    for lane_id in ordered:
        lane_count = len(jobs_by_lane[lane_id])
        lane_indexes = range(lane_cursor, lane_cursor + lane_count)
        lane_cursor += lane_count
        if not any(
            not isinstance(raw[index], BaseException)
            and raw[index][1].handoff == "render"
            for index in lane_indexes
        ):
            continue
        retry_indexes = [
            index
            for index in lane_indexes
            if not isinstance(raw[index], BaseException)
            and raw[index][1].handoff == "continue"
        ]
        forced = await asyncio.gather(
            *(_one(flattened[index], force=True) for index in retry_indexes),
            return_exceptions=True,
        )
        for index, value in zip(retry_indexes, forced, strict=True):
            raw[index] = value
    cursor = 0
    outcomes: list[NarratorLaneOutcome] = []
    for lane_id in ordered:
        jobs = jobs_by_lane[lane_id]
        lane_raw = raw[cursor:cursor + len(jobs)]
        cursor += len(jobs)
        outcomes.append(_apply_lane_results(
            checkpoint,
            lane_id=lane_id,
            jobs=jobs,
            raw=lane_raw,
        ))
    return outcomes
