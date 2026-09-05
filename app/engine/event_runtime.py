"""Canonical event commit, visibility, obligations, and narrator work.

This module owns durable consequences after a router batch has passed its
complete validation. It deliberately knows nothing about foreground or
background scenes: every event is committed through the same path, every
observer receives the same canonical fact projection, and every bound POV gets
the same retryable narrator-job contract.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Iterable

from app.engine.context_builder import (
    collect_player_ids,
    is_unbound_player_authored_slot,
    replace_character_ids_with_names,
)
from app.engine.visual_context import (
    AGENT_FIRST_MEETING_CAP,
    format_visual_introductions,
    mark_visual_introductions,
    plan_event_visual_introductions,
)
from app.schemas.characters import CharacterStatus, is_non_social_hazard
from app.schemas.checkpoint import CheckpointFile
from app.schemas.delivery import NarratorEventRef, NarratorRenderJob
from app.schemas.event_router import CanonicalEventRecord
from app.schemas.events import ObservableFact
from app.schemas.state import (
    ActionObligation,
    CommitmentRevisionPrompt,
    OpenCatIIEvent,
    OpenCommitment,
)


logger = logging.getLogger(__name__)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def visible_facts_for(
    event: CanonicalEventRecord,
    character_id: str,
) -> list[ObservableFact]:
    return [
        fact
        for fact in event.observable_facts
        if fact.text.strip() and fact.is_visible_to(character_id)
    ]


def visible_at_s(
    event: CanonicalEventRecord,
    facts: Iterable[ObservableFact],
) -> int:
    selected = list(facts)
    if not selected:
        return event.effective_at_s + event.duration_s
    return max(
        event.effective_at_s + fact.at_offset_s + fact.duration_s
        for fact in selected
    )


def _advance_character_clock(
    checkpoint: CheckpointFile,
    character_id: str,
    at_s: int,
) -> None:
    character = next(
        (
            item
            for item in checkpoint.characters
            if item.character_id == character_id
        ),
        None,
    )
    if character is None:
        raise RuntimeError(
            f"canonical event advances unknown character {character_id!r}"
        )
    character.clock_at_s = max(character.clock_at_s, at_s)
    checkpoint.session.leading_at_s = max(
        checkpoint.session.leading_at_s,
        character.clock_at_s,
    )


def _matching_commitments(
    checkpoint: CheckpointFile,
    *,
    actor_ids: list[str],
) -> list[OpenCommitment]:
    actors = set(actor_ids)
    return [
        item
        for item in checkpoint.session.open_commitments
        if actors.intersection(item.actor_ids)
    ]


def _drop_commitments(
    checkpoint: CheckpointFile,
    commitments: Iterable[OpenCommitment],
) -> None:
    selected = list(commitments)
    ids = {item.commitment_id for item in selected}
    if not ids:
        return
    checkpoint.session.open_commitments = [
        item
        for item in checkpoint.session.open_commitments
        if item.commitment_id not in ids
    ]
    for character_id, prompt in list(
        checkpoint.session.pending_commitment_revisions.items()
    ):
        if prompt.commitment_id in ids:
            checkpoint.session.pending_commitment_revisions.pop(character_id)


def _commitment_id(event_id: str, actor_ids: list[str]) -> str:
    suffix = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        "_".join(actor_ids),
    ).strip("_") or "activity"
    return f"commit_{event_id}_{suffix}"[:96]


def _apply_event_state(
    checkpoint: CheckpointFile,
    event: CanonicalEventRecord,
) -> None:
    roster = {item.character_id: item for item in checkpoint.characters}
    end_at_s = event.effective_at_s + event.duration_s
    for actor_id in event.actor_ids:
        _advance_character_clock(checkpoint, actor_id, end_at_s)

    location_ids: set[str] = set()
    for update in event.location_updates:
        if update.character_id in location_ids:
            raise RuntimeError("canonical event repeats a location update")
        location_ids.add(update.character_id)
        character = roster.get(update.character_id)
        if character is None:
            raise RuntimeError(
                "canonical location update targets unknown character "
                f"{update.character_id!r}"
            )
        prior = " ".join(str(character.location or "").split())
        next_location = " ".join(update.location_label.split())
        presentation = character.visuals.visual_novel_presentation
        if (presentation.scene_location or prior) != next_location:
            presentation.current_variant_key = "neutral"
        presentation.scene_location = next_location
        character.location = next_location

    for resolution in event.commitment_resolutions:
        matches = _matching_commitments(
            checkpoint,
            actor_ids=resolution.actor_ids,
        )
        if not matches:
            raise RuntimeError(
                "canonical commitment resolution matched no open activity"
            )
        at_s = event.effective_at_s + resolution.resolved_at_offset_s
        for commitment in matches:
            for actor_id in commitment.actor_ids:
                _advance_character_clock(checkpoint, actor_id, at_s)
        _drop_commitments(checkpoint, matches)

    for directive in event.commitment_opens:
        overlaps = _matching_commitments(
            checkpoint,
            actor_ids=directive.actor_ids,
        )
        _drop_commitments(checkpoint, overlaps)
        checkpoint.session.open_commitments.append(OpenCommitment(
            commitment_id=_commitment_id(event.event_id, directive.actor_ids),
            actor_ids=list(directive.actor_ids),
            description=directive.description,
            trigger_event_id=event.event_id,
            started_at_s=event.effective_at_s,
            expected_end_s=(
                event.effective_at_s + directive.expected_duration_s
            ),
            max_end_s=event.effective_at_s + directive.max_duration_s,
            location_label=directive.location_label,
        ))

    bound_ids = collect_player_ids(checkpoint)
    for interrupt in event.commitment_interrupts:
        matches = _matching_commitments(
            checkpoint,
            actor_ids=interrupt.actor_ids,
        )
        if not matches:
            raise RuntimeError(
                "canonical commitment interrupt matched no open activity"
            )
        for commitment in matches:
            targets = interrupt.actor_ids or commitment.actor_ids
            for character_id in targets:
                if character_id not in bound_ids:
                    continue
                facts = visible_facts_for(event, character_id)
                if not facts:
                    continue
                observed_at = max(
                    event.effective_at_s + interrupt.observed_at_offset_s,
                    visible_at_s(event, facts),
                )
                _advance_character_clock(checkpoint, character_id, observed_at)
                checkpoint.session.pending_commitment_revisions[character_id] = (
                    CommitmentRevisionPrompt(
                        character_id=character_id,
                        commitment_id=commitment.commitment_id,
                        trigger_event_id=event.event_id,
                        observed_at_s=observed_at,
                        reason=interrupt.reason,
                        previous_description=commitment.description,
                    )
                )


def _variant_snapshot(checkpoint: CheckpointFile) -> dict[str, str]:
    return {
        character.character_id: (
            character.visuals.visual_novel_presentation.current_variant_key
            or "neutral"
        )
        for character in checkpoint.characters
        if character.status != CharacterStatus.culled
    }


def _fan_out_event(
    checkpoint: CheckpointFile,
    event: CanonicalEventRecord,
    *,
    event_sequence: int,
) -> dict[str, NarratorEventRef]:
    bound_ids = collect_player_ids(checkpoint)
    roster = {item.character_id: item for item in checkpoint.characters}
    variants = _variant_snapshot(checkpoint)
    narrator_refs: dict[str, NarratorEventRef] = {}

    for character_id in event.observer_ids:
        facts = visible_facts_for(event, character_id)
        if not facts:
            continue
        seen_at = visible_at_s(event, facts)
        _advance_character_clock(checkpoint, character_id, seen_at)
        directness = event.observation_level_for(character_id)
        if not directness:
            raise RuntimeError("canonical observer has no directness group")

        if character_id in bound_ids:
            narrator_refs[character_id] = NarratorEventRef(
                event_id=event.event_id,
                observation_level=directness,
                visible_at_s=seen_at,
                event_sequence=event_sequence,
                sprite_variant_keys_by_character_id=variants,
            )
            continue

        recipient = roster.get(character_id)
        if recipient is None:
            raise RuntimeError(
                f"canonical observer is absent from roster: {character_id!r}"
            )
        if recipient.status == CharacterStatus.culled or is_non_social_hazard(recipient):
            continue
        payload = "\n".join(
            fact.text.strip() if len(facts) == 1 else f"  - {fact.text.strip()}"
            for fact in facts
        )
        recipient.pending_observations.append(
            replace_character_ids_with_names(payload, checkpoint)
        )
        intro = plan_event_visual_introductions(
            checkpoint,
            viewer_id=character_id,
            event=event,
            observation_level=directness,
            priority_target_ids=event.actor_ids,
            max_loadouts=AGENT_FIRST_MEETING_CAP,
        )
        intro_text = format_visual_introductions(intro.loadouts)
        if intro_text:
            recipient.pending_observations.append(intro_text)
        if intro.mark_character_ids:
            mark_visual_introductions(
                checkpoint,
                character_id,
                intro.mark_character_ids,
            )
    return narrator_refs


def _narrator_job_id(
    checkpoint: CheckpointFile,
    pov_character_id: str,
    lane_id: str,
    event_ids: list[str],
) -> str:
    raw = "\x1f".join((
        checkpoint.session.session_id,
        str(checkpoint.session.turn_index + 1),
        pov_character_id,
        lane_id,
        *event_ids,
    ))
    return "narrator_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def commit_event_batch(
    checkpoint: CheckpointFile,
    events: Iterable[CanonicalEventRecord],
    *,
    user_input_by_pov: dict[str, str] | None = None,
    partial_pov_ids: set[str] | None = None,
    dice_rolls_by_event_id: dict[str, list[dict[str, object]]] | None = None,
    experience_awards_by_event_id: dict[str, list[dict[str, object]]] | None = None,
) -> list[NarratorRenderJob]:
    """Commit a validated batch and create one durable job per observing POV.

    The caller must hold the session writer lock and must restore its checkpoint
    snapshot if any step raises. This function never performs provider calls.
    """

    selected = list(events)
    if not selected:
        return []
    existing_ids = {item.event_id for item in checkpoint.canonical_events}
    new_ids = [item.event_id for item in selected]
    if len(new_ids) != len(set(new_ids)) or existing_ids.intersection(new_ids):
        raise RuntimeError("canonical event ids must be unique")

    refs_by_lane_and_pov: dict[tuple[str, str], list[NarratorEventRef]] = {}
    for event in selected:
        _apply_event_state(checkpoint, event)
        sequence = len(checkpoint.canonical_events)
        checkpoint.canonical_events.append(event)
        for pov_id, ref in _fan_out_event(
            checkpoint,
            event,
            event_sequence=sequence,
        ).items():
            refs_by_lane_and_pov.setdefault(
                (event.causal_lane_id, pov_id),
                [],
            ).append(ref)

        from app.engine.content_fronts import queue_front_signals_from_public_event

        queue_front_signals_from_public_event(
            checkpoint,
            event,
            actor_id=event.actor_ids[0] if event.actor_ids else "",
        )

    jobs: list[NarratorRenderJob] = []
    for (lane_id, pov_id), refs in refs_by_lane_and_pov.items():
        refs.sort(key=lambda item: (item.visible_at_s, item.event_sequence))
        dice_rolls = [
            payload
            for ref in refs
            for payload in (dice_rolls_by_event_id or {}).get(ref.event_id, ())
        ]
        experience_awards = [
            payload
            for ref in refs
            for payload in (experience_awards_by_event_id or {}).get(
                ref.event_id,
                (),
            )
        ]
        existing = next(
            (
                item
                for item in checkpoint.session.narrator_render_jobs
                if item.lane_id == lane_id
                and item.pov_character_id == pov_id
                and item.status in {"pending", "failed"}
            ),
            None,
        )
        if existing is not None:
            known = set(existing.source_event_ids)
            additions = [ref for ref in refs if ref.event_id not in known]
            existing.event_refs.extend(additions)
            existing.source_event_ids.extend(ref.event_id for ref in additions)
            existing.highest_event_sequence = max(
                existing.highest_event_sequence,
                *(ref.event_sequence for ref in additions),
            )
            if (user_input_by_pov or {}).get(pov_id):
                existing.user_input = (user_input_by_pov or {})[pov_id]
            existing.partial_mode = (
                existing.partial_mode or pov_id in (partial_pov_ids or set())
            )
            for payload in dice_rolls:
                if payload not in existing.dice_rolls:
                    existing.dice_rolls.append(payload)
            for payload in experience_awards:
                if payload not in existing.experience_awards:
                    existing.experience_awards.append(payload)
            existing.status = "pending"
            existing.last_error = ""
            jobs.append(existing)
            continue
        event_ids = [item.event_id for item in refs]
        job_id = _narrator_job_id(checkpoint, pov_id, lane_id, event_ids)
        job = NarratorRenderJob(
            job_id=job_id,
            lane_id=lane_id,
            pov_character_id=pov_id,
            source_event_ids=event_ids,
            event_refs=refs,
            highest_event_sequence=max(item.event_sequence for item in refs),
            created_revision=checkpoint.session.turn_index + 1,
            user_input=(user_input_by_pov or {}).get(pov_id, ""),
            partial_mode=pov_id in (partial_pov_ids or set()),
            narration_mode="event_aligned",
            dice_rolls=dice_rolls,
            experience_awards=experience_awards,
            status="pending",
            attempts=0,
            last_error="",
        )
        checkpoint.session.narrator_render_jobs.append(job)
        jobs.append(job)
    return jobs


def set_action_obligation(
    checkpoint: CheckpointFile,
    *,
    character_id: str,
    kind: str,
    source_event_id: str,
) -> None:
    if character_id not in collect_player_ids(checkpoint):
        raise RuntimeError("only a bound character may receive an action obligation")
    existing = checkpoint.session.action_obligations.get(character_id)
    if existing is not None:
        raise RuntimeError(
            f"character already has an action obligation: {character_id}"
        )
    checkpoint.session.action_obligations[character_id] = ActionObligation(
        kind=kind,
        source_event_id=source_event_id,
        claimed_at=utcnow_iso(),
    )


def release_action_obligation(
    checkpoint: CheckpointFile,
    character_id: str,
) -> None:
    checkpoint.session.action_obligations.pop(character_id, None)


def open_cat_ii(
    checkpoint: CheckpointFile,
    *,
    initiator_id: str,
    initiator_intention: str,
    required_responder_ids: list[str],
    opening_event: CanonicalEventRecord,
) -> OpenCatIIEvent:
    if not required_responder_ids:
        raise ValueError("Cat II requires at least one responder")
    event_id = "contest_" + hashlib.sha256(
        "\x1f".join((
            checkpoint.session.session_id,
            opening_event.event_id,
            initiator_id,
            *required_responder_ids,
        )).encode("utf-8")
    ).hexdigest()[:12]
    opened = OpenCatIIEvent(
        event_id=event_id,
        initiator_id=initiator_id,
        initiator_intention=initiator_intention,
        required_responders=list(required_responder_ids),
        collected_intentions={},
        opening_event_id=opening_event.event_id,
        opening_observer_ids=list(opening_event.observer_ids),
        opening_observable_facts=[
            fact.text for fact in opening_event.observable_facts
        ],
        opened_at=utcnow_iso(),
    )
    checkpoint.session.open_cat_ii_events.append(opened)
    return opened


def collect_cat_ii_intention(
    checkpoint: CheckpointFile,
    *,
    event_id: str,
    responder_id: str,
    intention: str,
) -> OpenCatIIEvent:
    opened = next(
        (
            item
            for item in checkpoint.session.open_cat_ii_events
            if item.event_id == event_id
        ),
        None,
    )
    if opened is None:
        raise ValueError("Cat II event is no longer open")
    if responder_id not in opened.required_responders:
        raise ValueError("character is not a responder for this Cat II event")
    if responder_id in opened.collected_intentions:
        raise ValueError("Cat II responder already submitted an intention")
    if not intention.strip():
        raise ValueError("Cat II intention cannot be blank")
    opened.collected_intentions[responder_id] = intention
    release_action_obligation(checkpoint, responder_id)
    return opened


def cat_ii_is_ready(opened: OpenCatIIEvent) -> bool:
    return set(opened.required_responders).issubset(opened.collected_intentions)


def close_cat_ii(checkpoint: CheckpointFile, event_id: str) -> None:
    checkpoint.session.open_cat_ii_events = [
        item
        for item in checkpoint.session.open_cat_ii_events
        if item.event_id != event_id
    ]
    for character_id, obligation in list(
        checkpoint.session.action_obligations.items()
    ):
        if obligation.source_event_id == event_id:
            checkpoint.session.action_obligations.pop(character_id)


def purge_character_state(
    checkpoint: CheckpointFile,
    character_id: str,
    *,
    retire_character: bool = True,
) -> None:
    checkpoint.session.action_obligations.pop(character_id, None)
    checkpoint.session.pending_commitment_revisions.pop(character_id, None)
    if retire_character:
        checkpoint.session.router_frontier = [
            item
            for item in checkpoint.session.router_frontier
            if character_id != item.actor_id
            and character_id not in item.participant_ids
        ]
    for item in checkpoint.session.router_frontier:
        item.gating_pov_ids = [
            value for value in item.gating_pov_ids if value != character_id
        ]
    checkpoint.session.narrator_render_jobs = [
        item
        for item in checkpoint.session.narrator_render_jobs
        if item.pov_character_id != character_id
    ]
    checkpoint.session.delivery_outbox = [
        item
        for item in checkpoint.session.delivery_outbox
        if item.pov_character_id != character_id
    ]
    checkpoint.session.last_acknowledged_event_sequence_by_pov.pop(
        character_id,
        None,
    )
    if not retire_character:
        return
    remaining: list[OpenCatIIEvent] = []
    for opened in checkpoint.session.open_cat_ii_events:
        if opened.initiator_id == character_id:
            continue
        opened.required_responders = [
            item for item in opened.required_responders if item != character_id
        ]
        opened.collected_intentions.pop(character_id, None)
        remaining.append(opened)
    checkpoint.session.open_cat_ii_events = remaining


def autonomous_character_is_eligible(
    checkpoint: CheckpointFile,
    character_id: str,
) -> bool:
    character = next(
        (
            item
            for item in checkpoint.characters
            if item.character_id == character_id
        ),
        None,
    )
    return bool(
        character is not None
        and character.status == CharacterStatus.active
        and character_id not in checkpoint.session.character_bindings
        and not is_unbound_player_authored_slot(checkpoint, character)
        and not is_non_social_hazard(character)
    )


def character_has_pending_choice(
    checkpoint: CheckpointFile,
    character_id: str,
) -> bool:
    """Whether a character is already committed to an unresolved choice."""

    if character_id in checkpoint.session.action_obligations:
        return True
    return any(
        character_id == opened.initiator_id
        or character_id in opened.required_responders
        for opened in checkpoint.session.open_cat_ii_events
    )


def autonomous_character_is_ready(
    checkpoint: CheckpointFile,
    character_id: str,
) -> bool:
    """Whether ordinary frontier work may schedule this character now."""

    return (
        autonomous_character_is_eligible(checkpoint, character_id)
        and not character_has_pending_choice(checkpoint, character_id)
    )
