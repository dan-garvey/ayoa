"""Semantic projections for durable router history.

Canonical records keep engine-owned identities for persistence and joins.  Model
context uses only stable fictional facts plus prompt-local sequence/group
coordinates; durable ids never need to cross that boundary.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import CanonicalEventRecord, RouterInputEnvelope


def event_sequences(checkpoint: CheckpointFile) -> dict[str, int]:
    return {
        event.event_id: sequence
        for sequence, event in enumerate(checkpoint.canonical_events)
    }


def causal_groups(
    checkpoint: CheckpointFile,
    inputs: Sequence[RouterInputEnvelope] = (),
) -> dict[str, int]:
    lanes = list(dict.fromkeys([
        *(event.causal_lane_id for event in checkpoint.canonical_events),
        *(envelope.lane_id for envelope in inputs),
    ]))
    return {lane_id: group for group, lane_id in enumerate(lanes)}


def _compact(value: str, limit: int = 900) -> str:
    return " ".join(value.split())[:limit]


def router_history_record(
    checkpoint: CheckpointFile,
    event: CanonicalEventRecord,
) -> str:
    sequences = event_sequences(checkpoint)
    groups_by_lane = causal_groups(checkpoint)
    return _router_history_record(
        checkpoint,
        event,
        sequences=sequences,
        groups_by_lane=groups_by_lane,
    )


def _router_history_record(
    checkpoint: CheckpointFile,
    event: CanonicalEventRecord,
    *,
    sequences: dict[str, int],
    groups_by_lane: dict[str, int],
) -> str:
    if event.event_id not in sequences:
        raise RuntimeError("router history event is absent from canonical history")
    groups = event.observers
    lines = [
        f"prior_event sequence={sequences[event.event_id]} "
        f"causal_group={groups_by_lane[event.causal_lane_id]} "
        f"@{event.effective_at_s}+{event.duration_s} "
        f"actors={','.join(event.actor_ids) or '-'}",
        "outcomes "
        f"feasible_inputs={len(event.feasible_submission_ids)} "
        f"infeasible_inputs={len(event.infeasible_submission_ids)}",
        "observers "
        f"direct={','.join(groups.direct) or '-'} "
        f"indirect={','.join(groups.indirect) or '-'} "
        f"inferred={','.join(groups.inferred) or '-'}",
    ]
    for fact in event.observable_facts:
        audience = (
            "all" if fact.audience == "all_observers" else ",".join(fact.visible_to)
        )
        lines.append(
            f"fact +{fact.at_offset_s}/{fact.duration_s} to={audience}: "
            + _compact(fact.text)
        )
    roster = {item.character_id: item for item in checkpoint.characters}
    for request in event.spawn:
        character = roster.get(request.character_id)
        lines.append(
            f"spawn {request.character_id}"
            + (f" name={character.name}" if character is not None else "")
            + f" role={_compact(request.seed.role, 160)}"
        )
    for signal in event.activate:
        lines.append(f"activate {signal.character_id} at={signal.location_label}")
    if event.dormant:
        lines.append("dormant=" + ",".join(event.dormant))
    if event.cull:
        lines.append("cull=" + ",".join(event.cull))
    for update in event.location_updates:
        lines.append(f"location {update.character_id}={update.location_label}")
    for directive in event.commitment_opens:
        lines.append(
            "commitment_open actors="
            + ",".join(directive.actor_ids)
            + " description="
            + _compact(directive.description)
        )
    for signal in event.commitment_resolutions:
        lines.append(
            f"commitment_{signal.reason} actors={','.join(signal.actor_ids)}"
        )
    for signal in event.commitment_interrupts:
        lines.append(
            "commitment_interrupted actors="
            + ",".join(signal.actor_ids)
            + f" reason={_compact(signal.reason)}"
        )
    for update in getattr(event, "state_updates", ()):
        details = ",".join(update.details)
        lines.append(
            f"one_star_update {update.kind} {update.target_id}={update.value}"
            + (f" [{details}]" if details else "")
        )
    mode = getattr(event, "interaction_mode", "narrative")
    if mode != "narrative":
        lines.append(f"dnd_interaction={mode}")
    return "\n".join(lines)


def _strip_router_hash_metadata(content: str) -> str:
    if not content.startswith(("content_known ", "location_card ", "front_signal ")):
        return content
    return re.sub(r'\s+hash=(?:"[^"]*"|\S+)', "", content)


def router_prompt_history(
    checkpoint: CheckpointFile,
) -> list[ConversationMessage]:
    """Rebuild stored router memory without durable engine identity.

    Older checkpoints can contain event/lane ids and content hashes in compact
    assistant records.  Canonical history is authoritative, so rebuild each
    event row from its canonical sequence and scrub legacy content metadata.
    """

    sequences = event_sequences(checkpoint)
    groups_by_lane = causal_groups(checkpoint)
    represented_sequences: set[int] = set()
    projected: list[ConversationMessage] = []
    for message in checkpoint.session_conversation:
        content = message.content
        if (
            message.role == "assistant"
            and isinstance(content, str)
            and content.startswith("prior_event ")
        ):
            sequence_match = re.match(r"prior_event sequence=(\d+)\b", content)
            if sequence_match is not None:
                sequence = int(sequence_match.group(1))
            else:
                event_id = content.split(maxsplit=2)[1]
                if event_id not in sequences:
                    raise RuntimeError(
                        "stored router history references a missing canonical event"
                    )
                sequence = sequences[event_id]
            if sequence < 0 or sequence >= len(checkpoint.canonical_events):
                raise RuntimeError("stored router history has an invalid event sequence")
            if sequence in represented_sequences:
                raise RuntimeError("stored router history repeats a canonical event")
            represented_sequences.add(sequence)
            projected.append(message.model_copy(update={
                "content": _router_history_record(
                    checkpoint,
                    checkpoint.canonical_events[sequence],
                    sequences=sequences,
                    groups_by_lane=groups_by_lane,
                ),
            }))
            continue
        if isinstance(content, str):
            content = _strip_router_hash_metadata(content)
        projected.append(message.model_copy(update={"content": content}))

    for sequence, event in enumerate(checkpoint.canonical_events):
        if sequence in represented_sequences:
            continue
        projected.append(ConversationMessage(
            role="assistant",
            content=_router_history_record(
                checkpoint,
                event,
                sequences=sequences,
                groups_by_lane=groups_by_lane,
            ),
        ))
    return projected
