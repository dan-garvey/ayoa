"""Validation and materialization for batched canonical routing.

The router proposes fiction and causal follow-ups. This module is the only
place that assigns durable identity and absolute story time, so concurrent
agent preparation cannot race checkpoint order or flatten independent actor
clocks into one global timeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.schemas.characters import (
    CharacterStatus,
    is_non_social_hazard,
    is_player_authored_slot,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import (
    CanonicalEventRecord,
    DndCanonicalEventRecord,
    DndRouterEventDraft,
    FrontierTurn,
    RouterBatchOutput,
    RouterEventDraft,
    RouterInputEnvelope,
    materialized_event_id,
)
from app.schemas.one_star import (
    OneStarCanonicalEventRecord,
    OneStarRouterEventDraft,
)


class RouterBatchContractError(ValueError):
    """A complete provider response violates the batch routing contract."""


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MaterializedEvent:
    draft_index: int
    record: CanonicalEventRecord
    required_responder_ids: tuple[str, ...]
    appearance_target_ids: tuple[str, ...]
    dnd_reaction_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MaterializedRouterBatch:
    events: tuple[MaterializedEvent, ...]
    next_turns: tuple[FrontierTurn, ...]
    feasible_submission_ids: tuple[str, ...]
    infeasible_submission_ids: tuple[str, ...]
    correlation_id: str
    raw_output: str


def router_batch_correlation(inputs: list[RouterInputEnvelope]) -> str:
    """Return the stable identifier used to connect one router call to its logs."""

    return hashlib.sha256(
        "\x1f".join(item.submission_id for item in inputs).encode("utf-8")
    ).hexdigest()[:12]


def log_rejected_router_batch(
    *,
    session_id: str,
    correlation_id: str,
    stage: str,
    error: Exception,
    raw_output: str,
    materialized: MaterializedRouterBatch | None = None,
) -> None:
    """Log a rejected provider response without exposing it to story participants."""

    payload: dict[str, Any] = {
        "session_id": session_id,
        "correlation_id": correlation_id,
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "raw_output": raw_output,
    }
    if materialized is not None:
        payload["materialized_events"] = [
            {
                "draft_index": item.draft_index,
                "event_id": item.record.event_id,
                "causal_lane_id": item.record.causal_lane_id,
            }
            for item in materialized.events
        ]
        payload["materialized_next_turns"] = [
            item.model_dump(mode="json") for item in materialized.next_turns
        ]
    logger.error(
        "rejected router batch %s",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def _event_end_by_id(checkpoint: CheckpointFile) -> dict[str, int]:
    return {
        event.event_id: event.effective_at_s + event.duration_s
        for event in checkpoint.canonical_events
    }


def _record_class(draft: RouterEventDraft) -> type[CanonicalEventRecord]:
    if isinstance(draft, OneStarRouterEventDraft):
        return OneStarCanonicalEventRecord
    if isinstance(draft, DndRouterEventDraft):
        return DndCanonicalEventRecord
    return CanonicalEventRecord


def _adapter_record_fields(draft: RouterEventDraft) -> dict[str, Any]:
    if isinstance(draft, OneStarRouterEventDraft):
        return {"state_updates": list(draft.state_updates)}
    if isinstance(draft, DndRouterEventDraft):
        return {
            "interaction_mode": draft.interaction_mode,
            "combatant_ids": list(draft.combatant_ids),
            "combatant_spawns": list(draft.combatant_spawns),
            "loot_offer": draft.loot_offer,
            "battle_map_seed": draft.battle_map_seed,
        }
    return {}


def _draft_structured_character_ids(draft: RouterEventDraft) -> set[str]:
    ids = {
        *draft.observers.all_ids,
        *draft.required_responders,
        *draft.appearance_target_ids,
        *draft.dormant,
        *draft.cull,
        *(request.character_id for request in draft.spawn),
        *(signal.character_id for signal in draft.activate),
        *(signal.character_id for signal in draft.location_updates),
        *(
            character_id
            for fact in draft.observable_facts
            for character_id in (*fact.visible_to, *fact.visual_subject_ids)
        ),
        *(
            character_id
            for signal in draft.commitment_opens
            for character_id in signal.actor_ids
        ),
        *(
            character_id
            for signal in draft.commitment_resolutions
            for character_id in signal.actor_ids
        ),
        *(
            character_id
            for signal in draft.commitment_interrupts
            for character_id in signal.actor_ids
        ),
    }
    if isinstance(draft, DndRouterEventDraft):
        ids.update(draft.combatant_ids)
        ids.update(spawn.character_id for spawn in draft.combatant_spawns)
        ids.update(draft.dnd_reaction_ids)
    return {value for value in ids if value}


def _mutation_targets(draft: RouterEventDraft) -> set[str]:
    targets = {
        *draft.dormant,
        *draft.cull,
        *(request.character_id for request in draft.spawn),
        *(signal.character_id for signal in draft.activate),
        *(signal.character_id for signal in draft.location_updates),
        *(
            character_id
            for signal in draft.commitment_opens
            for character_id in signal.actor_ids
        ),
        *(
            character_id
            for signal in draft.commitment_resolutions
            for character_id in signal.actor_ids
        ),
        *(
            character_id
            for signal in draft.commitment_interrupts
            for character_id in signal.actor_ids
        ),
    }
    if isinstance(draft, OneStarRouterEventDraft) and draft.state_updates:
        targets.update(
            f"one_star:{update.kind}:{update.target_id}"
            for update in draft.state_updates
        )
    if isinstance(draft, DndRouterEventDraft) and draft.interaction_mode != "narrative":
        targets.add("dnd:active_combat")
    return {value for value in targets if value}


def _validate_known_ids(
    checkpoint: CheckpointFile,
    drafts: list[RouterEventDraft],
) -> None:
    known = {character.character_id for character in checkpoint.characters}
    def _spawn_ids(draft: RouterEventDraft) -> list[str]:
        return [
            *(request.character_id for request in draft.spawn),
            *(
                [spawn.character_id for spawn in draft.combatant_spawns]
                if isinstance(draft, DndRouterEventDraft)
                else []
            ),
        ]

    sibling_spawn_values = [
        character_id
        for draft in drafts
        for character_id in _spawn_ids(draft)
    ]
    all_sibling_spawns = set(sibling_spawn_values)
    if len(all_sibling_spawns) != len(sibling_spawn_values):
        raise RouterBatchContractError("sibling events request a duplicate spawn id")
    if known & all_sibling_spawns:
        raise RouterBatchContractError("router spawn id already exists in the roster")

    for draft_index, draft in enumerate(drafts):
        local_spawns = set(_spawn_ids(draft))
        sibling_only = all_sibling_spawns - local_spawns
        references = _draft_structured_character_ids(draft)
        dependency = references & sibling_only
        if dependency:
            raise RouterBatchContractError(
                "one sibling event depends on a character spawned by another: "
                + ", ".join(sorted(dependency))
            )
        unknown = references - known - local_spawns
        if unknown:
            raise RouterBatchContractError(
                f"event draft {draft_index} references unknown character ids: "
                + ", ".join(sorted(unknown))
            )


def _validate_sibling_mutations(drafts: list[RouterEventDraft]) -> None:
    owners: dict[str, int] = {}
    for index, draft in enumerate(drafts):
        for target in _mutation_targets(draft):
            prior = owners.get(target)
            if prior is not None:
                raise RouterBatchContractError(
                    "sibling event drafts carry conflicting shared mutations for "
                    f"{target!r}; merge drafts {prior} and {index}"
                )
            owners[target] = index


def _validate_contests(
    checkpoint: CheckpointFile,
    inputs: list[RouterInputEnvelope],
    drafts: list[RouterEventDraft],
) -> None:
    """Reject obligations that cannot become a resolvable actor choice."""

    roster = {
        character.character_id: character for character in checkpoint.characters
    }
    bound = set(checkpoint.session.character_bindings)
    for draft_index, draft in enumerate(drafts):
        if not draft.required_responders:
            continue
        if not draft.feasible_input_indexes:
            raise RouterBatchContractError(
                "a contested opening must identify a feasible initiating proposal"
            )
        source = inputs[draft.feasible_input_indexes[0]]
        if len(source.actor_ids) != 1:
            raise RouterBatchContractError(
                "a contested opening must have exactly one initiating actor"
            )
        initiator_id = source.actor_ids[0]
        if initiator_id in draft.required_responders:
            raise RouterBatchContractError(
                "a contested initiator cannot also be its own responder"
            )
        retired = set(draft.dormant) | set(draft.cull)
        if retired.intersection(draft.required_responders):
            raise RouterBatchContractError(
                "a contested responder cannot become dormant or culled in its opening"
            )
        local_spawns = {request.character_id for request in draft.spawn}
        unavailable: list[str] = []
        for responder_id in draft.required_responders:
            if responder_id in local_spawns:
                continue
            responder = roster.get(responder_id)
            if (
                responder is None
                or responder.status != CharacterStatus.active
                or is_non_social_hazard(responder)
                or (
                    is_player_authored_slot(responder)
                    and responder_id not in bound
                )
            ):
                unavailable.append(responder_id)
        if unavailable:
            raise RouterBatchContractError(
                f"event draft {draft_index} has unavailable contested responders: "
                + ", ".join(sorted(unavailable))
            )


def _validate_conflicting_inputs_are_merged(
    inputs: list[RouterInputEnvelope],
    drafts: list[RouterEventDraft],
) -> None:
    """Overlapping proposals are legal inputs but cannot become siblings."""

    owner_by_index = {
        input_index: draft_index
        for draft_index, draft in enumerate(drafts)
        for input_index in (
            *draft.feasible_input_indexes,
            *draft.infeasible_input_indexes,
        )
    }
    for left_index, left in enumerate(inputs):
        for right_index in range(left_index + 1, len(inputs)):
            right = inputs[right_index]
            conflict = (
                left.lane_id == right.lane_id
                or bool(set(left.participant_ids).intersection(right.participant_ids))
            )
            if conflict and owner_by_index[left_index] != owner_by_index[right_index]:
                raise RouterBatchContractError(
                    "overlapping router inputs must resolve in one canonical event"
                )


def _causal_lane_id(
    checkpoint: CheckpointFile,
    selected: list[RouterInputEnvelope],
) -> str:
    lanes = list(dict.fromkeys(item.lane_id for item in selected))
    if len(lanes) == 1:
        return lanes[0]
    basis = "\x1f".join((checkpoint.session.session_id, *sorted(lanes)))
    return "lane_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


def _event_lower_bound(
    inputs: list[RouterInputEnvelope],
    indexes: list[int],
    event_ends: dict[str, int],
    actor_clocks: dict[str, int],
) -> int:
    selected = [inputs[index] for index in indexes]
    source_ends: list[int] = []
    for envelope in selected:
        for source_id in envelope.source_event_ids:
            if source_id not in event_ends:
                raise RouterBatchContractError(
                    f"router input references missing source event {source_id!r}"
                )
            source_ends.append(event_ends[source_id])
    return max(
        0,
        *(envelope.chosen_at_s for envelope in selected),
        *(envelope.observed_through_s for envelope in selected),
        *(actor_clocks.get(actor_id, 0) for envelope in selected for actor_id in envelope.actor_ids),
        *source_ends,
    )


def materialize_router_batch(
    *,
    checkpoint: CheckpointFile,
    inputs: list[RouterInputEnvelope],
    output: RouterBatchOutput,
    correlation_id: str = "",
    raw_output: str | None = None,
) -> MaterializedRouterBatch:
    """Validate a whole response, then assign canonical ids, time, and frontier.

    Nothing mutates ``checkpoint``. Callers may stage adapter validation for all
    returned records and commit them atomically at the sole checkpoint writer.
    """

    try:
        output.validate_for_inputs(inputs)
        _validate_conflicting_inputs_are_merged(inputs, list(output.events))
        _validate_known_ids(checkpoint, list(output.events))
        _validate_sibling_mutations(list(output.events))
        _validate_contests(checkpoint, inputs, list(output.events))
    except ValueError as exc:
        if isinstance(exc, RouterBatchContractError):
            raise
        raise RouterBatchContractError(str(exc)) from exc

    roster = {character.character_id: character for character in checkpoint.characters}
    event_ends = _event_end_by_id(checkpoint)
    actor_clocks = {
        character_id: max(0, int(character.clock_at_s))
        for character_id, character in roster.items()
    }
    revision = checkpoint.session.turn_index + 1
    records: list[MaterializedEvent] = []
    records_by_draft: dict[int, CanonicalEventRecord] = {}
    sequence_by_draft: dict[int, int] = {}
    feasible_ids: list[str] = []
    infeasible_ids: list[str] = []

    for draft_index, draft in enumerate(output.events):
        indexes = [
            *draft.feasible_input_indexes,
            *draft.infeasible_input_indexes,
        ]
        selected = [inputs[index] for index in indexes]
        draft_feasible = [
            inputs[index].submission_id for index in draft.feasible_input_indexes
        ]
        draft_infeasible = [
            inputs[index].submission_id for index in draft.infeasible_input_indexes
        ]
        feasible_ids.extend(draft_feasible)
        infeasible_ids.extend(draft_infeasible)
        if draft.is_no_event_resolution:
            continue

        submission_ids = [envelope.submission_id for envelope in selected]
        actor_ids = list(dict.fromkeys(
            actor_id for envelope in selected for actor_id in envelope.actor_ids
        ))
        effective_at_s = _event_lower_bound(
            inputs,
            indexes,
            event_ends,
            actor_clocks,
        )
        record_type = _record_class(draft)
        record = record_type.model_validate({
            "event_id": materialized_event_id(
                session_id=checkpoint.session.session_id,
                revision=revision,
                event_index=draft_index,
                submission_ids=submission_ids,
            ),
            "causal_lane_id": _causal_lane_id(checkpoint, selected),
            "effective_at_s": effective_at_s,
            "duration_s": draft.duration_s,
            "actor_ids": actor_ids,
            "source_submission_ids": submission_ids,
            "feasible_submission_ids": draft_feasible,
            "infeasible_submission_ids": draft_infeasible,
            "observable_facts": list(draft.observable_facts),
            "observers": draft.observers,
            "spawn": list(draft.spawn),
            "dormant": list(draft.dormant),
            "cull": list(draft.cull),
            "commitment_opens": list(draft.commitment_opens),
            "commitment_resolutions": list(draft.commitment_resolutions),
            "commitment_interrupts": list(draft.commitment_interrupts),
            "location_updates": list(draft.location_updates),
            "activate": list(draft.activate),
            **_adapter_record_fields(draft),
        })
        records_by_draft[draft_index] = record
        sequence_by_draft[draft_index] = len(checkpoint.canonical_events) + len(records)
        records.append(MaterializedEvent(
            draft_index=draft_index,
            record=record,
            required_responder_ids=tuple(draft.required_responders),
            appearance_target_ids=tuple(draft.appearance_target_ids),
            dnd_reaction_ids=tuple(
                draft.dnd_reaction_ids
                if isinstance(draft, DndRouterEventDraft)
                else ()
            ),
        ))

    next_turns: list[FrontierTurn] = []
    known_active = {
        character.character_id
        for character in checkpoint.characters
        if character.status == CharacterStatus.active
    }
    for turn_index, turn in enumerate(output.next_turns):
        source = records_by_draft.get(turn.source_event_index)
        sourced_spawns = (
            {request.character_id for request in source.spawn}
            if source is not None
            else set()
        )
        sourced_activations = (
            {signal.character_id for signal in source.activate}
            if source is not None
            else set()
        )
        unknown_participants = (
            set(turn.participant_ids) - set(roster) - sourced_spawns
        )
        if unknown_participants:
            raise RouterBatchContractError(
                "next turn references unknown participants: "
                + ", ".join(sorted(unknown_participants))
            )
        if turn.source_event_index >= 0 and source is None:
            raise RouterBatchContractError(
                "next turn source did not materialize a canonical event"
            )
        active_for_turn = known_active | sourced_spawns | sourced_activations
        inactive_participants = set(turn.participant_ids) - active_for_turn
        if inactive_participants:
            raise RouterBatchContractError(
                "next turn references inactive participants: "
                + ", ".join(sorted(inactive_participants))
            )
        if (
            turn.turn_kind == "character"
            and turn.actor_id in checkpoint.session.character_bindings
        ):
            raise RouterBatchContractError(
                "router cannot choose a player-owned character's next action"
            )
        if (
            turn.turn_kind == "character"
            and turn.actor_id in roster
            and is_player_authored_slot(roster[turn.actor_id])
        ):
            raise RouterBatchContractError(
                "router cannot choose a player-authored character's next action"
            )
        if source is not None and set(turn.participant_ids).intersection({
            *source.dormant,
            *source.cull,
        }):
            raise RouterBatchContractError(
                "a sourced next turn cannot include newly dormant or culled characters"
            )
        source_ids = [source.event_id] if source is not None else []
        basis = "\x1f".join((
            checkpoint.session.session_id,
            str(revision),
            "frontier",
            str(turn_index),
            *source_ids,
            turn.actor_id,
        ))
        turn_id = "turn_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]
        gating = (
            [
                character_id
                for character_id in source.observer_ids
                if character_id in checkpoint.session.character_bindings
            ]
            if source is not None
            else []
        )
        next_turns.append(FrontierTurn(
            turn_id=turn_id,
            lane_id=(
                source.causal_lane_id
                if source is not None
                else "lane_" + hashlib.sha256(
                    "\x1f".join((
                        checkpoint.session.session_id,
                        "autonomous",
                        turn.actor_id,
                    )).encode("utf-8")
                ).hexdigest()[:12]
            ),
            turn_kind=turn.turn_kind,
            actor_id=turn.actor_id,
            participant_ids=list(turn.participant_ids),
            source_event_ids=source_ids,
            created_event_sequence=(
                sequence_by_draft[turn.source_event_index]
                if source is not None
                else len(checkpoint.canonical_events) + len(records)
            ),
            gating_pov_ids=gating,
        ))

    return MaterializedRouterBatch(
        events=tuple(records),
        next_turns=tuple(next_turns),
        feasible_submission_ids=tuple(feasible_ids),
        infeasible_submission_ids=tuple(infeasible_ids),
        correlation_id=correlation_id or router_batch_correlation(inputs),
        raw_output=output.model_dump_json() if raw_output is None else raw_output,
    )
