"""Unified causal-frontier coordination for interactive story advancement.

Character, player, ruleset, and sourced world proposals all enter the same
batched router contract.  Provider work may be prepared concurrently from an
immutable checkpoint snapshot; validated fiction is committed by one writer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Iterable

from app.engine.character_agent import CharacterAgentTurnDraft
from app.engine.event_runtime import (
    autonomous_character_is_eligible,
    autonomous_character_is_ready,
    cat_ii_is_ready,
    character_has_pending_choice,
    close_cat_ii,
    collect_cat_ii_intention,
    commit_event_batch,
    open_cat_ii,
    set_action_obligation,
    visible_facts_for,
)
from app.engine.narrator_delivery import (
    NarratorLaneOutcome,
    process_narrator_lanes,
)
from app.engine.router_batch import (
    MaterializedEvent,
    MaterializedRouterBatch,
    log_rejected_router_batch,
)
from app.engine.story_dispatcher import StoryDispatcher, append_router_history
from app.schemas.checkpoint import CheckpointFile
from app.schemas.characters import CharacterStatus, is_player_authored_slot
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.event_router import (
    MAX_ROUTER_BATCH_INPUTS,
    CanonicalEventRecord,
    FrontierTurn,
    RouterInputEnvelope,
    RouterInputKind,
)
from app.schemas.state import OpenCatIIEvent
from app.engine.dnd_cat_ii import DndResolvedCanonicalEvent


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedRouterInput:
    envelope: RouterInputEnvelope
    frontier_turn_id: str = ""
    character_draft: CharacterAgentTurnDraft | None = None
    contest_event_id: str = ""
    contest_responder_drafts: tuple[
        tuple[str, CharacterAgentTurnDraft, str], ...
    ] = ()
    attached_character_drafts: tuple[
        tuple[str, CharacterAgentTurnDraft], ...
    ] = ()
    contest_player_response: tuple[str, str] | None = None


@dataclass(slots=True)
class AdvanceResult:
    events_committed: int = 0
    event_ids: list[str] = field(default_factory=list)
    feasible_submission_ids: list[str] = field(default_factory=list)
    infeasible_submission_ids: list[str] = field(default_factory=list)
    lane_outcomes: list[NarratorLaneOutcome] = field(default_factory=list)
    prepared_followups: list[PreparedRouterInput] = field(default_factory=list)
    pause_reason: str = ""


def immutable_checkpoint(checkpoint: CheckpointFile) -> CheckpointFile:
    return CheckpointFile.model_validate_json(checkpoint.model_dump_json(
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
    ))


def replace_checkpoint_state(target: CheckpointFile, source: CheckpointFile) -> None:
    """Replace one in-memory Pydantic aggregate after an atomic staged commit."""

    target.__dict__.clear()
    target.__dict__.update(source.__dict__)
    target.__pydantic_fields_set__ = set(source.__pydantic_fields_set__)
    target.__pydantic_extra__ = source.__pydantic_extra__
    target.__pydantic_private__ = source.__pydantic_private__


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _event_sequence(checkpoint: CheckpointFile, event_id: str) -> int:
    for index, event in enumerate(checkpoint.canonical_events):
        if event.event_id == event_id:
            return index
    raise RuntimeError(f"causal source event is absent: {event_id!r}")


def _event_by_id(
    checkpoint: CheckpointFile,
    event_id: str,
) -> CanonicalEventRecord:
    for event in checkpoint.canonical_events:
        if event.event_id == event_id:
            return event
    raise RuntimeError(f"canonical event is absent: {event_id!r}")


def _character_clock(checkpoint: CheckpointFile, character_id: str) -> int:
    character = next(
        (item for item in checkpoint.characters if item.character_id == character_id),
        None,
    )
    if character is None:
        raise ValueError(f"unknown character {character_id!r}")
    return max(0, int(character.clock_at_s))


def _latest_acknowledged_source(
    checkpoint: CheckpointFile,
    character_id: str,
) -> tuple[list[str], int, int]:
    sequence = checkpoint.session.last_acknowledged_event_sequence_by_pov.get(
        character_id,
        -1,
    )
    if sequence < 0:
        return [], -1, 0
    if sequence >= len(checkpoint.canonical_events):
        raise RuntimeError("POV knowledge cutoff exceeds canonical history")
    event = checkpoint.canonical_events[sequence]
    facts = visible_facts_for(event, character_id)
    if not facts:
        raise RuntimeError("acknowledged event carries no visible POV facts")
    seen_at = max(
        event.effective_at_s + fact.at_offset_s + fact.duration_s
        for fact in facts
    )
    return [event.event_id], sequence, seen_at


def _player_lane(
    checkpoint: CheckpointFile,
    character_id: str,
    source_event_ids: list[str],
) -> str:
    obligation = checkpoint.session.action_obligations.get(character_id)
    if obligation is not None:
        contest = next(
            (
                item
                for item in checkpoint.session.open_cat_ii_events
                if item.event_id == obligation.source_event_id
            ),
            None,
        )
        if contest is None:
            raise RuntimeError("action obligation references no open contest")
        return _event_by_id(checkpoint, contest.opening_event_id).causal_lane_id
    if source_event_ids:
        return _event_by_id(checkpoint, source_event_ids[-1]).causal_lane_id
    return _stable_id("lane_", checkpoint.session.session_id, character_id)


def player_input(
    checkpoint: CheckpointFile,
    *,
    character_id: str,
    payload: str,
    kind: RouterInputKind = "player",
) -> PreparedRouterInput:
    source_ids, observed_sequence, observed_at_s = _latest_acknowledged_source(
        checkpoint,
        character_id,
    )
    revision = checkpoint.session.turn_index + 1
    envelope = RouterInputEnvelope(
        submission_id=_stable_id(
            "submission_",
            checkpoint.session.session_id,
            revision,
            character_id,
            payload,
        ),
        input_index=0,
        lane_id=_player_lane(checkpoint, character_id, source_ids),
        kind=kind,
        actor_ids=[character_id],
        participant_ids=[character_id],
        source_event_ids=source_ids,
        chosen_at_s=max(_character_clock(checkpoint, character_id), observed_at_s),
        observed_through_event_sequence=observed_sequence,
        observed_through_s=observed_at_s,
        payload=payload.strip(),
    )
    return PreparedRouterInput(envelope=envelope)


def release_frontier_gates_for_pov_action(
    checkpoint: CheckpointFile,
    character_id: str,
) -> int:
    """Release every causal lane waiting for this POV to resume play.

    A narrator render names all bound viewpoints that can resume a lane. The
    first one of them to act or explicitly defer releases the lane as a whole;
    delivery acknowledgement alone never advances story causality.
    """

    released = 0
    for turn in checkpoint.session.router_frontier:
        if character_id not in turn.gating_pov_ids:
            continue
        turn.gating_pov_ids = []
        released += 1
    return released


def _frontier_sort_key(turn: FrontierTurn) -> tuple[int, str]:
    return turn.created_event_sequence, turn.turn_id


def ready_frontier_turns(
    checkpoint: CheckpointFile,
    *,
    preferred_lane_ids: set[str] | None = None,
    preferred_participant_ids: set[str] | None = None,
    excluded_turn_ids: set[str] | None = None,
    limit: int = MAX_ROUTER_BATCH_INPUTS,
) -> list[FrontierTurn]:
    """Select disjoint runnable work, preferring interaction with new input."""

    preferred_lanes = set(preferred_lane_ids or ())
    preferred_participants = set(preferred_participant_ids or ())
    excluded = set(excluded_turn_ids or ())
    lanes: set[str] = set()
    participants: set[str] = set()
    selected: list[FrontierTurn] = []
    candidates = sorted(
        checkpoint.session.router_frontier,
        key=lambda turn: (
            0
            if turn.lane_id in preferred_lanes
            or bool(preferred_participants.intersection(turn.participant_ids))
            else 1,
            *_frontier_sort_key(turn),
        ),
    )
    for turn in candidates:
        if len(selected) >= max(0, limit):
            break
        if turn.turn_id in excluded or turn.gating_pov_ids or turn.lane_id in lanes:
            continue
        if participants.intersection(turn.participant_ids):
            continue
        if turn.turn_kind == "character":
            if not autonomous_character_is_ready(checkpoint, turn.actor_id):
                continue
        if any(
            character_has_pending_choice(checkpoint, character_id)
            for character_id in turn.participant_ids
        ):
            continue
        selected.append(turn)
        lanes.add(turn.lane_id)
        participants.update(turn.participant_ids)
    return selected


async def _narrate_and_prepare_followups(
    checkpoint: CheckpointFile,
    dispatcher: StoryDispatcher,
    *,
    lane_ids: Iterable[str],
) -> tuple[list[NarratorLaneOutcome], list[PreparedRouterInput]]:
    """Overlap visible rendering with independent newly-ready agent work."""

    snapshot = immutable_checkpoint(checkpoint)
    turns = ready_frontier_turns(snapshot)

    async def _prepare() -> list[PreparedRouterInput]:
        if not turns:
            return []
        try:
            return await prepare_frontier_inputs(snapshot, dispatcher, turns)
        except Exception:
            logger.exception("speculative autonomous preparation failed")
            return []

    lane_outcomes, prepared = await asyncio.gather(
        process_narrator_lanes(
            checkpoint,
            dispatcher,
            lane_ids=lane_ids,
        ),
        _prepare(),
    )
    return lane_outcomes, prepared


def _source_context(
    checkpoint: CheckpointFile,
    turn: FrontierTurn,
) -> str:
    lines: list[str] = []
    for event_id in turn.source_event_ids:
        event = _event_by_id(checkpoint, event_id)
        facts = visible_facts_for(event, turn.actor_id) if turn.actor_id else []
        if turn.actor_id and not facts:
            raise RuntimeError("frontier actor did not observe its causal source")
        lines.extend(fact.text for fact in facts)
    return "\n".join(line.strip() for line in lines if line.strip())


async def prepare_frontier_inputs(
    checkpoint: CheckpointFile,
    dispatcher: StoryDispatcher,
    turns: Iterable[FrontierTurn],
    *,
    starting_index: int = 0,
) -> list[PreparedRouterInput]:
    selected = list(turns)
    frozen = checkpoint.model_dump_json(
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
    )

    async def _prepare(turn: FrontierTurn) -> PreparedRouterInput:
        observed_sequence = max(
            (_event_sequence(checkpoint, item) for item in turn.source_event_ids),
            default=-1,
        )
        observed_at_s = max(
            (
                _event_by_id(checkpoint, item).effective_at_s
                + _event_by_id(checkpoint, item).duration_s
                for item in turn.source_event_ids
            ),
            default=0,
        )
        draft: CharacterAgentTurnDraft | None = None
        if turn.turn_kind == "character":
            snapshot = CheckpointFile.model_validate_json(frozen)
            draft = await dispatcher.draft_character_turn(
                ckpt=snapshot,
                character_id=turn.actor_id,
                local_context=_source_context(snapshot, turn),
            )
            payload = draft.output.public_text if not draft.output.is_silence else "(defer)"
            kind = "character"
            actors = [turn.actor_id]
        else:
            payload = "Continue only the motion in the listed source events."
            kind = "world"
            actors = []
        return PreparedRouterInput(
            envelope=RouterInputEnvelope(
                submission_id=turn.turn_id,
                input_index=0,
                lane_id=turn.lane_id,
                kind=kind,
                actor_ids=actors,
                participant_ids=list(turn.participant_ids),
                source_event_ids=list(turn.source_event_ids),
                chosen_at_s=max(
                    observed_at_s,
                    _character_clock(checkpoint, turn.actor_id)
                    if turn.actor_id
                    else observed_at_s,
                ),
                observed_through_event_sequence=observed_sequence,
                observed_through_s=observed_at_s,
                payload=payload,
            ),
            frontier_turn_id=turn.turn_id,
            character_draft=draft,
        )

    prepared = list(await asyncio.gather(*(_prepare(turn) for turn in selected)))
    return [
        PreparedRouterInput(
            envelope=item.envelope.model_copy(update={
                "input_index": starting_index + index,
            }),
            frontier_turn_id=item.frontier_turn_id,
            character_draft=item.character_draft,
            contest_event_id=item.contest_event_id,
            contest_responder_drafts=item.contest_responder_drafts,
            attached_character_drafts=item.attached_character_drafts,
            contest_player_response=item.contest_player_response,
        )
        for index, item in enumerate(prepared)
    ]


def _contest_for_materialized_event(
    prepared: list[PreparedRouterInput],
    materialized: MaterializedEvent,
) -> tuple[str, str]:
    source_id = materialized.record.feasible_submission_ids[0]
    candidate = next(
        (
            item
            for item in prepared
            if item.envelope.submission_id == source_id
        ),
        None,
    )
    if candidate is None or len(candidate.envelope.actor_ids) != 1:
        raise RuntimeError(
            "contested event must identify one feasible initiating actor proposal"
        )
    return candidate.envelope.actor_ids[0], candidate.envelope.payload


def _install_contests(
    checkpoint: CheckpointFile,
    prepared: list[PreparedRouterInput],
    batch: MaterializedRouterBatch,
) -> None:
    bound = set(checkpoint.session.character_bindings)
    for event in batch.events:
        responders = list(event.required_responder_ids)
        if not responders:
            continue
        initiator_id, intention = _contest_for_materialized_event(prepared, event)
        opened = open_cat_ii(
            checkpoint,
            initiator_id=initiator_id,
            initiator_intention=intention,
            required_responder_ids=responders,
            opening_event=event.record,
        )
        for responder_id in responders:
            if responder_id in bound:
                set_action_obligation(
                    checkpoint,
                    character_id=responder_id,
                    kind="cat_ii_response",
                    source_event_id=opened.event_id,
                )


def _commit_prepared_character_drafts(
    checkpoint: CheckpointFile,
    dispatcher: StoryDispatcher,
    prepared: list[PreparedRouterInput],
    batch: MaterializedRouterBatch,
) -> None:
    event_end_by_submission = {
        submission_id: event.record.effective_at_s + event.record.duration_s
        for event in batch.events
        for submission_id in event.record.source_submission_ids
    }
    for item in prepared:
        if item.character_draft is not None:
            actor_id = item.envelope.actor_ids[0]
            dispatcher.commit_character_turn(
                ckpt=checkpoint,
                character_id=actor_id,
                draft=item.character_draft,
                committed_at_s=event_end_by_submission.get(
                    item.envelope.submission_id,
                    item.envelope.chosen_at_s,
                ),
            )
        for responder_id, draft, intention in item.contest_responder_drafts:
            collect_cat_ii_intention(
                checkpoint,
                event_id=item.contest_event_id,
                responder_id=responder_id,
                intention=intention,
            )
            opening = next(
                opened
                for opened in checkpoint.session.open_cat_ii_events
                if opened.event_id == item.contest_event_id
            )
            source = _event_by_id(checkpoint, opening.opening_event_id)
            dispatcher.commit_character_turn(
                ckpt=checkpoint,
                character_id=responder_id,
                draft=draft,
                committed_at_s=source.effective_at_s + source.duration_s,
            )
        if item.contest_player_response is not None:
            responder_id, intention = item.contest_player_response
            collect_cat_ii_intention(
                checkpoint,
                event_id=item.contest_event_id,
                responder_id=responder_id,
                intention=intention,
            )
        for character_id, draft in item.attached_character_drafts:
            dispatcher.commit_character_turn(
                ckpt=checkpoint,
                character_id=character_id,
                draft=draft,
                committed_at_s=event_end_by_submission.get(
                    item.envelope.submission_id,
                    item.envelope.chosen_at_s,
                ),
            )


def _stage_contest_responses(
    checkpoint: CheckpointFile,
    prepared: Iterable[PreparedRouterInput],
) -> None:
    for item in prepared:
        if not item.contest_event_id:
            continue
        for responder_id, _draft, intention in item.contest_responder_drafts:
            collect_cat_ii_intention(
                checkpoint,
                event_id=item.contest_event_id,
                responder_id=responder_id,
                intention=intention,
            )
        if item.contest_player_response is not None:
            responder_id, intention = item.contest_player_response
            collect_cat_ii_intention(
                checkpoint,
                event_id=item.contest_event_id,
                responder_id=responder_id,
                intention=intention,
            )


def _commit_staged_contest_drafts(
    checkpoint: CheckpointFile,
    dispatcher: StoryDispatcher,
    prepared: Iterable[PreparedRouterInput],
) -> None:
    for item in prepared:
        if not item.contest_event_id:
            continue
        opened = next(
            event
            for event in checkpoint.session.open_cat_ii_events
            if event.event_id == item.contest_event_id
        )
        source = _event_by_id(checkpoint, opened.opening_event_id)
        committed_at_s = source.effective_at_s + source.duration_s
        for responder_id, draft, _intention in item.contest_responder_drafts:
            dispatcher.commit_character_turn(
                ckpt=checkpoint,
                character_id=responder_id,
                draft=draft,
                committed_at_s=committed_at_s,
            )


def _consume_and_extend_frontier(
    checkpoint: CheckpointFile,
    prepared: list[PreparedRouterInput],
    batch: MaterializedRouterBatch,
) -> None:
    consumed = {
        item.frontier_turn_id for item in prepared if item.frontier_turn_id
    }
    checkpoint.session.router_frontier = [
        item
        for item in checkpoint.session.router_frontier
        if item.turn_id not in consumed
    ]
    newly_occupied_lanes = {item.lane_id for item in batch.next_turns}
    newly_occupied_participants = {
        character_id
        for item in batch.next_turns
        for character_id in item.participant_ids
    }
    superseded = [
        item
        for item in checkpoint.session.router_frontier
        if item.lane_id in newly_occupied_lanes
        or bool(newly_occupied_participants.intersection(item.participant_ids))
    ]
    if superseded:
        logger.info(
            "new canonical frontier superseded older pending turns: %s",
            ", ".join(item.turn_id for item in superseded),
        )
        superseded_ids = {item.turn_id for item in superseded}
        checkpoint.session.router_frontier = [
            item
            for item in checkpoint.session.router_frontier
            if item.turn_id not in superseded_ids
        ]
    existing = {item.turn_id for item in checkpoint.session.router_frontier}
    occupied_lanes = {
        item.lane_id for item in checkpoint.session.router_frontier
    }
    occupied_participants = {
        character_id
        for item in checkpoint.session.router_frontier
        for character_id in item.participant_ids
    }
    for turn in batch.next_turns:
        if turn.turn_id in existing:
            raise RuntimeError("router frontier contains a duplicate turn id")
        if turn.lane_id in occupied_lanes:
            raise RuntimeError("router produced concurrent work in one causal lane")
        overlap = occupied_participants.intersection(turn.participant_ids)
        if overlap:
            raise RuntimeError(
                "router frontier contains concurrent shared participants: "
                + ", ".join(sorted(overlap))
            )
        checkpoint.session.router_frontier.append(turn)
        existing.add(turn.turn_id)
        occupied_lanes.add(turn.lane_id)
        occupied_participants.update(turn.participant_ids)


def _append_adapter_frontier(
    checkpoint: CheckpointFile,
    event: CanonicalEventRecord,
    actor_ids: Iterable[str],
) -> None:
    """Project adapter follow-ups onto the same durable causal frontier."""

    roster = {item.character_id: item for item in checkpoint.characters}
    bound = set(checkpoint.session.character_bindings)
    candidates = list(dict.fromkeys(actor_ids))
    autonomous = [character_id for character_id in candidates if character_id not in bound]
    if len(autonomous) > 1:
        raise RuntimeError("adapter returned ambiguous simultaneous follow-up actors")
    for character_id in candidates:
        character = roster.get(character_id)
        if character is None or character.status != CharacterStatus.active:
            raise RuntimeError("adapter follow-up actor is not active")
        if character_id not in event.observer_ids or not visible_facts_for(
            event,
            character_id,
        ):
            raise RuntimeError("adapter follow-up actor did not observe its source")
        if character_id in bound:
            continue
        if is_player_authored_slot(character):
            raise RuntimeError("adapter cannot automate a player-authored character")
        participants = list(dict.fromkeys([character_id, *event.actor_ids]))
        turn = FrontierTurn(
            turn_id=_stable_id(
                "turn_",
                checkpoint.session.session_id,
                event.event_id,
                character_id,
            ),
            lane_id=event.causal_lane_id,
            turn_kind="character",
            actor_id=character_id,
            participant_ids=participants,
            source_event_ids=[event.event_id],
            created_event_sequence=len(checkpoint.canonical_events) - 1,
            gating_pov_ids=[
                observer_id
                for observer_id in event.observer_ids
                if observer_id in bound
            ],
        )
        if any(
            existing.lane_id == turn.lane_id
            or set(existing.participant_ids).intersection(turn.participant_ids)
            for existing in checkpoint.session.router_frontier
        ):
            raise RuntimeError("adapter follow-up conflicts with the live frontier")
        checkpoint.session.router_frontier.append(turn)


async def commit_adapter_resolution(
    checkpoint: CheckpointFile,
    dispatcher: StoryDispatcher,
    *,
    working: CheckpointFile,
    resolution: DndResolvedCanonicalEvent,
    user_input_by_pov: dict[str, str] | None = None,
    partial_pov_ids: set[str] | None = None,
    character_drafts: Iterable[
        tuple[str, CharacterAgentTurnDraft, int]
    ] = (),
    close_contest_event_id: str = "",
    player_input_included: bool = False,
    dice_rolls: list[dict[str, object]] | None = None,
    pause_reason: str = "",
) -> AdvanceResult:
    """Commit one adapter-resolved event through canonical delivery machinery."""

    event = resolution.event
    type(event).model_validate(event.model_dump())
    commit_event_batch(
        working,
        [event],
        user_input_by_pov=user_input_by_pov,
        partial_pov_ids=set(partial_pov_ids or ()),
        dice_rolls_by_event_id={event.event_id: list(dice_rolls or ())},
        experience_awards_by_event_id={
            event.event_id: [
                award.model_dump() for award in resolution.experience_awards
            ],
        },
    )
    _append_adapter_frontier(
        working,
        event,
        resolution.next_turn_actor_ids,
    )
    for character_id, draft, committed_at_s in character_drafts:
        dispatcher.commit_character_turn(
            ckpt=working,
            character_id=character_id,
            draft=draft,
            committed_at_s=committed_at_s,
        )
    if close_contest_event_id:
        close_cat_ii(working, close_contest_event_id)
    append_router_history(working, [event])
    if player_input_included:
        working.session.autonomous_router_batches_since_player = 0
    else:
        working.session.autonomous_router_batches_since_player += 1

    replace_checkpoint_state(checkpoint, working)
    lane_outcomes, prepared_followups = await _narrate_and_prepare_followups(
        checkpoint,
        dispatcher,
        lane_ids=[event.causal_lane_id],
    )
    checkpoint.session.turn_index += 1
    return AdvanceResult(
        events_committed=1,
        event_ids=[event.event_id],
        feasible_submission_ids=list(event.feasible_submission_ids),
        infeasible_submission_ids=list(event.infeasible_submission_ids),
        lane_outcomes=lane_outcomes,
        prepared_followups=prepared_followups,
        pause_reason=(
            "narrator_delivery_failed"
            if any(item.failed_pov_ids for item in lane_outcomes)
            else pause_reason
        ),
    )


async def _advance_dnd_contest(
    checkpoint: CheckpointFile,
    dispatcher: StoryDispatcher,
    prepared: list[PreparedRouterInput],
    *,
    user_input_by_pov: dict[str, str] | None,
) -> AdvanceResult:
    if len(prepared) != 1 or not prepared[0].contest_event_id:
        raise RuntimeError("D&D contested resolution must run as one adapter input")
    from app.engine.dnd_cat_ii import DndCatIIRollsPending
    from app.engine.dnd_roll_display import (
        completed_automatic_roll_keys,
        dice_roll_displays_since,
    )

    item = prepared[0]
    working = immutable_checkpoint(checkpoint)
    _stage_contest_responses(working, prepared)
    opened = next(
        event
        for event in working.session.open_cat_ii_events
        if event.event_id == item.contest_event_id
    )
    roll_keys_before = completed_automatic_roll_keys(working)
    try:
        resolution = await dispatcher.resolve_dnd_cat_ii(
            ckpt=working,
            opened=opened,
        )
    except DndCatIIRollsPending:
        _commit_staged_contest_drafts(working, dispatcher, prepared)
        if item.contest_player_response is not None:
            working.session.autonomous_router_batches_since_player = 0
        else:
            working.session.autonomous_router_batches_since_player += 1
        replace_checkpoint_state(checkpoint, working)
        checkpoint.session.turn_index += 1
        return AdvanceResult(pause_reason="cat_ii_pending_rolls")

    submission_id = item.envelope.submission_id
    event = resolution.event
    feasible = bool(event.feasible_submission_ids)
    event.source_submission_ids = [submission_id]
    event.feasible_submission_ids = [submission_id] if feasible else []
    event.infeasible_submission_ids = [] if feasible else [submission_id]
    _commit_staged_contest_drafts(working, dispatcher, prepared)
    return await commit_adapter_resolution(
        checkpoint,
        dispatcher,
        working=working,
        resolution=resolution,
        user_input_by_pov=user_input_by_pov,
        close_contest_event_id=item.contest_event_id,
        player_input_included=item.contest_player_response is not None,
        dice_rolls=[
            display.model_dump()
            for display in dice_roll_displays_since(working, roll_keys_before)
        ],
    )


def _close_resolved_contests(
    checkpoint: CheckpointFile,
    prepared: list[PreparedRouterInput],
    batch: MaterializedRouterBatch,
) -> None:
    accounted = {
        *batch.feasible_submission_ids,
        *batch.infeasible_submission_ids,
    }
    for item in prepared:
        if item.contest_event_id and item.envelope.submission_id in accounted:
            close_cat_ii(checkpoint, item.contest_event_id)


async def advance_story(
    checkpoint: CheckpointFile,
    dispatcher: StoryDispatcher,
    prepared: list[PreparedRouterInput],
    *,
    user_input_by_pov: dict[str, str] | None = None,
) -> AdvanceResult:
    """Route and atomically commit one logical batch, then render its POVs."""

    if not prepared or len(prepared) > MAX_ROUTER_BATCH_INPUTS:
        raise ValueError("story advancement requires one through five inputs")
    normalized = [
        PreparedRouterInput(
            envelope=item.envelope.model_copy(update={"input_index": index}),
            frontier_turn_id=item.frontier_turn_id,
            character_draft=item.character_draft,
            contest_event_id=item.contest_event_id,
            contest_responder_drafts=item.contest_responder_drafts,
            attached_character_drafts=item.attached_character_drafts,
            contest_player_response=item.contest_player_response,
        )
        for index, item in enumerate(prepared)
    ]
    if (
        checkpoint.session.config.settings.ruleset_id == "dnd5e_basic"
        and any(item.envelope.kind == "cat_ii_resolution" for item in normalized)
    ):
        return await _advance_dnd_contest(
            checkpoint,
            dispatcher,
            normalized,
            user_input_by_pov=user_input_by_pov,
        )
    working = immutable_checkpoint(checkpoint)
    batch = await dispatcher.route_batch(
        ckpt=working,
        inputs=[item.envelope for item in normalized],
    )
    player_actor_ids = {
        actor_id
        for item in normalized
        if item.envelope.kind == "player"
        for actor_id in item.envelope.actor_ids
    }
    try:
        await dispatcher.prepare_batch(
            ckpt=working,
            batch=batch,
            inputs=[item.envelope for item in normalized],
            player_actor_ids=player_actor_ids,
        )
        bound = set(working.session.character_bindings)
        partial_pov_ids = {
            character_id
            for event in batch.events
            for character_id in (
                *event.required_responder_ids,
                *event.dnd_reaction_ids,
            )
            if character_id in bound
        }
        commit_event_batch(
            working,
            [item.record for item in batch.events],
            user_input_by_pov=user_input_by_pov,
            partial_pov_ids=partial_pov_ids,
        )
        _install_contests(working, normalized, batch)
        if working.session.config.settings.ruleset_id == "dnd5e_basic":
            from app.engine.dnd_story_adapter import install_dnd_reactions

            install_dnd_reactions(working, batch)
        _commit_prepared_character_drafts(working, dispatcher, normalized, batch)
        _consume_and_extend_frontier(working, normalized, batch)
        _close_resolved_contests(working, normalized, batch)
        append_router_history(working, [item.record for item in batch.events])
        if any(
            item.envelope.kind in {"player", "authoritative_result"}
            for item in normalized
        ):
            working.session.autonomous_router_batches_since_player = 0
        else:
            working.session.autonomous_router_batches_since_player += 1
    except Exception as exc:
        log_rejected_router_batch(
            session_id=working.session.session_id,
            correlation_id=batch.correlation_id,
            stage="atomic_apply",
            error=exc,
            raw_output=batch.raw_output,
            materialized=batch,
        )
        raise

    replace_checkpoint_state(checkpoint, working)
    lane_ids = [item.record.causal_lane_id for item in batch.events]
    lane_outcomes, prepared_followups = await _narrate_and_prepare_followups(
        checkpoint,
        dispatcher,
        lane_ids=lane_ids,
    )
    checkpoint.session.turn_index += 1
    return AdvanceResult(
        events_committed=len(batch.events),
        event_ids=[item.record.event_id for item in batch.events],
        feasible_submission_ids=list(batch.feasible_submission_ids),
        infeasible_submission_ids=list(batch.infeasible_submission_ids),
        lane_outcomes=lane_outcomes,
        prepared_followups=prepared_followups,
        pause_reason=(
            "narrator_delivery_failed"
            if any(item.failed_pov_ids for item in lane_outcomes)
            else "cat_ii_pending"
            if any(item.required_responder_ids for item in batch.events)
            else ""
        ),
    )


def _contest_payload(opened: OpenCatIIEvent) -> str:
    return json.dumps(
        {
            "initiator": {
                "character_id": opened.initiator_id,
                "intention": opened.initiator_intention,
            },
            "responses": [
                {
                    "character_id": responder_id,
                    "intention": opened.collected_intentions[responder_id],
                }
                for responder_id in opened.required_responders
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _resolution_input(
    checkpoint: CheckpointFile,
    opened: OpenCatIIEvent,
    *,
    responder_drafts: tuple[
        tuple[str, CharacterAgentTurnDraft, str], ...
    ] = (),
    player_response: tuple[str, str] | None = None,
) -> list[PreparedRouterInput]:
    if not cat_ii_is_ready(opened):
        return []
    source = _event_by_id(checkpoint, opened.opening_event_id)
    participant_ids = list(dict.fromkeys([
        opened.initiator_id,
        *opened.required_responders,
    ]))
    chosen_at_s = max(
        source.effective_at_s + source.duration_s,
        *(_character_clock(checkpoint, item) for item in participant_ids),
    )
    return [PreparedRouterInput(
        envelope=RouterInputEnvelope(
            submission_id=_stable_id(
                "resolution_",
                checkpoint.session.session_id,
                opened.event_id,
            ),
            input_index=0,
            lane_id=source.causal_lane_id,
            kind="cat_ii_resolution",
            actor_ids=participant_ids,
            participant_ids=participant_ids,
            source_event_ids=[source.event_id],
            chosen_at_s=chosen_at_s,
            observed_through_event_sequence=_event_sequence(
                checkpoint,
                source.event_id,
            ),
            observed_through_s=source.effective_at_s + source.duration_s,
            payload=_contest_payload(opened),
        ),
        character_draft=None,
        contest_event_id=opened.event_id,
        contest_responder_drafts=responder_drafts,
        contest_player_response=player_response,
    )]


async def prepare_autonomous_contest_resolutions(
    checkpoint: CheckpointFile,
    dispatcher: StoryDispatcher,
    *,
    limit: int = MAX_ROUTER_BATCH_INPUTS,
) -> list[PreparedRouterInput]:
    """Collect agent-owned Cat II responses concurrently, bundle per contest."""

    selected: list[OpenCatIIEvent] = []
    used_participants: set[str] = set()
    bound = set(checkpoint.session.character_bindings)
    for opened in checkpoint.session.open_cat_ii_events:
        missing = [
            item
            for item in opened.required_responders
            if item not in opened.collected_intentions
        ]
        if any(item in bound for item in missing):
            continue
        participants = {opened.initiator_id, *opened.required_responders}
        if participants.intersection(used_participants):
            continue
        if any(
            not autonomous_character_is_eligible(checkpoint, item)
            for item in missing
        ):
            raise RuntimeError("open contest has an unavailable autonomous responder")
        selected.append(opened)
        used_participants.update(participants)
        if len(selected) >= limit:
            break

    frozen = checkpoint.model_dump_json(
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
    )

    async def _draft(opened: OpenCatIIEvent, responder_id: str):
        snapshot = CheckpointFile.model_validate_json(frozen)
        context = "\n".join(opened.opening_observable_facts)
        return responder_id, await dispatcher.draft_character_turn(
            ckpt=snapshot,
            character_id=responder_id,
            local_context=context,
        )

    requests = [
        (opened, responder_id)
        for opened in selected
        for responder_id in opened.required_responders
        if responder_id not in opened.collected_intentions
    ]
    responses = list(await asyncio.gather(*(
        _draft(opened, responder_id) for opened, responder_id in requests
    )))
    response_map = {responder_id: draft for responder_id, draft in responses}
    staged_drafts_by_contest: dict[
        str,
        list[tuple[str, CharacterAgentTurnDraft, str]],
    ] = {}
    for opened, responder_id in requests:
        draft = response_map[responder_id]
        intention = (
            draft.output.public_text if not draft.output.is_silence else "(defer)"
        )
        staged_drafts_by_contest.setdefault(opened.event_id, []).append(
            (responder_id, draft, intention)
        )
    prepared: list[PreparedRouterInput] = []
    for opened in selected:
        staged = opened.model_copy(deep=True)
        staged.collected_intentions.update({
            responder_id: intention
            for responder_id, _draft_value, intention in (
                staged_drafts_by_contest.get(opened.event_id, [])
            )
        })
        prepared.extend(_resolution_input(
            checkpoint,
            staged,
            responder_drafts=tuple(
                staged_drafts_by_contest.get(opened.event_id, [])
            ),
        ))
    return [
        PreparedRouterInput(
            envelope=item.envelope.model_copy(update={"input_index": index}),
            frontier_turn_id=item.frontier_turn_id,
            character_draft=item.character_draft,
            contest_event_id=item.contest_event_id,
            contest_responder_drafts=item.contest_responder_drafts,
            attached_character_drafts=item.attached_character_drafts,
            contest_player_response=item.contest_player_response,
        )
        for index, item in enumerate(prepared)
    ]


def collect_player_contest_response(
    checkpoint: CheckpointFile,
    *,
    character_id: str,
    intention: str,
) -> list[PreparedRouterInput]:
    obligation = checkpoint.session.action_obligations.get(character_id)
    if obligation is None or obligation.kind != "cat_ii_response":
        raise ValueError("character has no pending contested-action response")
    opened = next(
        (
            item
            for item in checkpoint.session.open_cat_ii_events
            if item.event_id == obligation.source_event_id
        ),
        None,
    )
    if opened is None:
        raise RuntimeError("action obligation references no open contest")
    if character_id not in opened.required_responders:
        raise RuntimeError("action obligation names a non-responder")
    if character_id in opened.collected_intentions:
        raise ValueError("Cat II responder already submitted an intention")
    if not intention.strip():
        raise ValueError("Cat II intention cannot be blank")
    staged = opened.model_copy(deep=True)
    staged.collected_intentions[character_id] = intention
    if not cat_ii_is_ready(staged):
        collect_cat_ii_intention(
            checkpoint,
            event_id=opened.event_id,
            responder_id=character_id,
            intention=intention,
        )
        return []
    return _resolution_input(
        checkpoint,
        staged,
        player_response=(character_id, intention),
    )


async def prepare_ready_frontier_batch(
    checkpoint: CheckpointFile,
    dispatcher: StoryDispatcher,
    *,
    initial: list[PreparedRouterInput] | None = None,
    preprepared: list[PreparedRouterInput] | None = None,
) -> list[PreparedRouterInput]:
    prepared = list(initial or ())
    if (
        checkpoint.session.config.settings.ruleset_id == "dnd5e_basic"
        and any(item.envelope.kind == "cat_ii_resolution" for item in prepared)
    ):
        return prepared
    preferred_lanes = {item.envelope.lane_id for item in prepared}
    preferred_participants = {
        character_id
        for item in prepared
        for character_id in item.envelope.participant_ids
    }
    turns = ready_frontier_turns(
        checkpoint,
        preferred_lane_ids=preferred_lanes,
        preferred_participant_ids=preferred_participants,
        limit=MAX_ROUTER_BATCH_INPUTS - len(prepared),
    )
    cached_by_turn_id = {
        item.frontier_turn_id: item
        for item in preprepared or ()
        if item.frontier_turn_id
    }

    def _matches(turn: FrontierTurn, item: PreparedRouterInput) -> bool:
        envelope = item.envelope
        return (
            envelope.submission_id == turn.turn_id
            and envelope.lane_id == turn.lane_id
            and envelope.actor_ids == ([turn.actor_id] if turn.actor_id else [])
            and envelope.participant_ids == turn.participant_ids
            and envelope.source_event_ids == turn.source_event_ids
            and envelope.kind
            == ("character" if turn.turn_kind == "character" else "world")
        )

    missing = [
        turn
        for turn in turns
        if not (
            (cached := cached_by_turn_id.get(turn.turn_id)) is not None
            and _matches(turn, cached)
        )
    ]
    newly_prepared = await prepare_frontier_inputs(
        checkpoint,
        dispatcher,
        missing,
    )
    available = {
        item.frontier_turn_id: item
        for item in [*(preprepared or ()), *newly_prepared]
        if item.frontier_turn_id
    }
    prepared.extend(available[turn.turn_id] for turn in turns)
    return [
        PreparedRouterInput(
            envelope=item.envelope.model_copy(update={"input_index": index}),
            frontier_turn_id=item.frontier_turn_id,
            character_draft=item.character_draft,
            contest_event_id=item.contest_event_id,
            contest_responder_drafts=item.contest_responder_drafts,
            attached_character_drafts=item.attached_character_drafts,
            contest_player_response=item.contest_player_response,
        )
        for index, item in enumerate(prepared)
    ]
