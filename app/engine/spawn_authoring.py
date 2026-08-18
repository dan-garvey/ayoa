from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

from app.engine.character_manager import CharacterManager
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, SpawnRequest


@dataclass(frozen=True)
class SpawnAuthoringKey:
    session_id: str
    transaction_id: str
    event_id: str
    event_fingerprint: str


class SpawnAuthoringCoordinator:
    """Shares one immutable-snapshot character-authoring task per event."""

    def __init__(self, character_manager: CharacterManager) -> None:
        self.character_manager = character_manager
        self._tasks: dict[
            SpawnAuthoringKey,
            asyncio.Task[tuple[CharacterRecord, ...]],
        ] = {}

    def start(
        self,
        *,
        checkpoint: CheckpointFile,
        event: EventRouterOutput,
        transaction_id: str,
        event_fingerprint: str,
        acting_actor_location: str = "",
    ) -> SpawnAuthoringKey | None:
        if not event.spawn:
            return None
        key = SpawnAuthoringKey(
            session_id=checkpoint.session.session_id,
            transaction_id=transaction_id,
            event_id=event.event_id,
            event_fingerprint=event_fingerprint,
        )
        if key in self._tasks:
            return key
        matching = next(
            (
                existing
                for existing in self._tasks
                if existing.session_id == key.session_id
                and existing.event_id == key.event_id
                and existing.event_fingerprint == key.event_fingerprint
            ),
            None,
        )
        if matching is not None:
            return matching

        # `ClosedEventRuntime` is attached to the live model's `__dict__` as
        # non-schema process state and owns asyncio tasks/locks. A Pydantic
        # deep copy follows that transient object graph and fails on its
        # non-pickleable locks. JSON round-tripping copies only durable
        # checkpoint fields and gives the authoring task a genuinely immutable
        # snapshot.
        immutable_checkpoint = CheckpointFile.model_validate_json(
            checkpoint.model_dump_json()
        )
        immutable_requests = [
            request.model_copy(deep=True) for request in event.spawn
        ]
        task = asyncio.create_task(
            self._author(
                immutable_checkpoint,
                immutable_requests,
                acting_actor_location=acting_actor_location,
            ),
            name=f"spawn-authoring:{event.event_id}",
        )
        task.add_done_callback(_consume_task_exception)
        self._tasks[key] = task
        return key

    async def result(
        self,
        key: SpawnAuthoringKey | None,
    ) -> tuple[CharacterRecord, ...]:
        if key is None:
            return ()
        task = self._tasks.get(key)
        if task is None:
            raise RuntimeError(
                f"spawn-authoring task is unavailable for {key.event_id}"
            )
        return await asyncio.shield(task)

    def task(
        self,
        key: SpawnAuthoringKey,
    ) -> asyncio.Task[tuple[CharacterRecord, ...]] | None:
        return self._tasks.get(key)

    def discard_transaction(
        self,
        transaction_id: str,
        *,
        cancel_running: bool,
    ) -> None:
        for key, task in list(self._tasks.items()):
            if key.transaction_id != transaction_id:
                continue
            if cancel_running and not task.done():
                task.cancel()
            if task.done() or cancel_running:
                del self._tasks[key]

    async def _author(
        self,
        checkpoint: CheckpointFile,
        requests: Sequence[SpawnRequest],
        *,
        acting_actor_location: str,
    ) -> tuple[CharacterRecord, ...]:
        records = await self.character_manager.spawn_characters(
            checkpoint,
            list(requests),
            acting_actor_location=acting_actor_location,
        )
        return tuple(
            character.model_copy(deep=True) for character in records
        )


def _consume_task_exception(
    task: asyncio.Task[tuple[CharacterRecord, ...]],
) -> None:
    if not task.cancelled():
        task.exception()
