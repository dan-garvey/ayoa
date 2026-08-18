from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from app.engine.image_director import source_event_fingerprint
from app.engine.spawn_authoring import (
    SpawnAuthoringCoordinator,
    SpawnAuthoringKey,
)
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput


class ClosedEventImageSink(Protocol):
    def on_closed_event(
        self,
        *,
        checkpoint: CheckpointFile,
        event: EventRouterOutput,
        event_sequence: int,
        transaction_id: str,
        source_turn_index: int,
        spawn_key: SpawnAuthoringKey | None,
        actor_id: str,
    ) -> None:
        ...


SpawnRecordApplier = Callable[
    [
        "ClosedEventRuntime",
        CheckpointFile,
        tuple[CharacterRecord, ...],
    ],
    list[str],
]


@dataclass
class ClosedEventRuntime:
    transaction_id: str
    source_turn_index: int
    spawn_authoring: SpawnAuthoringCoordinator
    image_sink: ClosedEventImageSink | None = None
    record_applier: SpawnRecordApplier | None = None
    spawn_keys_by_event_id: dict[str, SpawnAuthoringKey] = field(
        default_factory=dict
    )
    applied_character_ids: set[str] = field(default_factory=set)

    def close_event(
        self,
        *,
        checkpoint: CheckpointFile,
        event: EventRouterOutput,
        event_sequence: int,
        actor_id: str,
    ) -> None:
        key = self.start_spawn_authoring(
            checkpoint=checkpoint,
            event=event,
            actor_id=actor_id,
        )
        if self.image_sink is not None:
            self.image_sink.on_closed_event(
                checkpoint=checkpoint,
                event=event,
                event_sequence=event_sequence,
                transaction_id=self.transaction_id,
                source_turn_index=self.source_turn_index,
                spawn_key=key,
                actor_id=actor_id,
            )

    def start_spawn_authoring(
        self,
        *,
        checkpoint: CheckpointFile,
        event: EventRouterOutput,
        actor_id: str,
    ) -> SpawnAuthoringKey | None:
        fingerprint = source_event_fingerprint(event)
        existing_ids = {
            character.character_id for character in checkpoint.characters
        }
        missing_requests = [
            request
            for request in event.spawn
            if request.character_id not in existing_ids
        ]
        if not missing_requests:
            return None
        authoring_event = event.model_copy(
            deep=True,
            update={"spawn": missing_requests},
        )
        location = ""
        actor = next(
            (
                character
                for character in checkpoint.characters
                if character.character_id == actor_id
            ),
            None,
        )
        if actor is not None:
            location = actor.location
        key = self.spawn_authoring.start(
            checkpoint=checkpoint,
            event=authoring_event,
            transaction_id=self.transaction_id,
            event_fingerprint=fingerprint,
            acting_actor_location=location,
        )
        if key is not None:
            self.spawn_keys_by_event_id[event.event_id] = key
        return key

    async def authored_records(
        self,
        *,
        checkpoint: CheckpointFile,
        event: EventRouterOutput,
        actor_id: str,
    ) -> tuple[CharacterRecord, ...]:
        key = self.spawn_keys_by_event_id.get(event.event_id)
        if key is None:
            key = self.start_spawn_authoring(
                checkpoint=checkpoint,
                event=event,
                actor_id=actor_id,
            )
        return await self.spawn_authoring.result(key)

    def apply_records(
        self,
        checkpoint: CheckpointFile,
        records: tuple[CharacterRecord, ...],
    ) -> list[str]:
        if self.record_applier is None:
            raise RuntimeError(
                "closed-event runtime has no orchestrator spawn applier"
            )
        return self.record_applier(
            self,
            checkpoint,
            records,
        )


def install_closed_event_runtime(
    checkpoint: CheckpointFile,
    runtime: ClosedEventRuntime,
) -> None:
    checkpoint.__dict__["_closed_event_runtime"] = runtime


def closed_event_runtime_for(
    checkpoint: CheckpointFile,
) -> ClosedEventRuntime | None:
    value = checkpoint.__dict__.get("_closed_event_runtime")
    return value if isinstance(value, ClosedEventRuntime) else None
