from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
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


@dataclass
class _SpeculativeRoster:
    """Process-local roster overlay owned by one closed-event transaction.

    The generated ``CharacterRecord`` objects are shared by every same-beat
    consumer.  They temporarily sit in a new shallow ``characters`` list so
    agent and narrator context can resolve public identity without making the
    records durable before narration is accepted.
    """

    session_id: str
    records_by_id: dict[str, CharacterRecord] = field(default_factory=dict)
    introduction_buckets: dict[str, list[str]] = field(default_factory=dict)
    active: bool = False
    introductions_active: bool = False
    accepted: bool = False


class SpawnAuthoringCoordinator:
    """Shares one immutable-snapshot character-authoring task per event."""

    def __init__(self, character_manager: CharacterManager) -> None:
        self.character_manager = character_manager
        self._tasks: dict[
            SpawnAuthoringKey,
            asyncio.Task[tuple[CharacterRecord, ...]],
        ] = {}
        self._rosters: dict[str, _SpeculativeRoster] = {}

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
        one_star_hero_ids: set[str] | None = None
        if (
            checkpoint.session.config.settings.ruleset_id
            == "one_star_ascension"
        ):
            spawned_ids = {
                request.character_id for request in immutable_requests
            }
            from app.engine.one_star_adapter import (
                one_star_state_updates_to_transaction,
            )

            transaction = one_star_state_updates_to_transaction(
                immutable_checkpoint,
                getattr(event, "state_updates", ()),
                canonical_at_s=event.effective_at_s + event.duration_s,
            )
            one_star_hero_ids = {
                hero_id
                for operation in getattr(transaction, "operations", ())
                if getattr(operation, "operation", "") == "summon"
                for hero_id in getattr(operation, "hero_ids", ())
                if hero_id in spawned_ids
            }
        task = asyncio.create_task(
            self._author(
                immutable_checkpoint,
                immutable_requests,
                acting_actor_location=acting_actor_location,
                one_star_hero_ids=one_star_hero_ids,
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
        self._rosters.pop(transaction_id, None)

    def stage_roster(
        self,
        *,
        checkpoint: CheckpointFile,
        transaction_id: str,
        records: Sequence[CharacterRecord],
    ) -> list[str]:
        """Expose authored records through one reversible shallow roster.

        Records are not copied here.  The exact objects returned by the shared
        authoring task are reused for immediate agent dispatch, narrator and
        visual planning, and the eventual accepted checkpoint.
        """

        roster = self._rosters.get(transaction_id)
        if roster is None:
            roster = _SpeculativeRoster(
                session_id=checkpoint.session.session_id,
            )
            self._rosters[transaction_id] = roster
        elif roster.session_id != checkpoint.session.session_id:
            raise RuntimeError(
                "spawn roster transaction crossed session boundaries"
            )

        if not roster.active and roster.records_by_id:
            self._restore_roster_records(checkpoint, roster)

        existing_by_id = {
            character.character_id: character
            for character in checkpoint.characters
        }
        additions: list[CharacterRecord] = []
        for record in records:
            character_id = record.character_id
            prior = roster.records_by_id.get(character_id)
            if prior is not None:
                if prior is not record and prior != record:
                    raise RuntimeError(
                        "spawn authoring returned different records for "
                        f"{character_id} in one transaction"
                    )
                continue
            if character_id in existing_by_id:
                raise ValueError(
                    "router-authored spawn targets existing character id: "
                    f"{character_id}"
                )
            roster.records_by_id[character_id] = record
            existing_by_id[character_id] = record
            additions.append(record)

        if additions:
            checkpoint.characters = [*checkpoint.characters, *additions]
        roster.active = bool(roster.records_by_id)
        return [record.character_id for record in additions]

    def rollback_roster(
        self,
        *,
        checkpoint: CheckpointFile,
        transaction_id: str,
    ) -> list[str]:
        """Remove an unaccepted overlay and its introduction-ledger edges."""

        roster = self._rosters.get(transaction_id)
        if roster is None or not roster.active or roster.accepted:
            return []
        character_ids = set(roster.records_by_id)

        affected_buckets = self._speculative_introduction_subset(
            checkpoint,
            character_ids,
        )
        self._merge_introduction_buckets(
            roster.introduction_buckets,
            affected_buckets,
        )
        for viewer_id, introduced_ids in list(
            checkpoint.session.visual_introductions.items()
        ):
            if viewer_id in character_ids:
                checkpoint.session.visual_introductions.pop(viewer_id, None)
                continue
            retained = [
                cid for cid in introduced_ids if cid not in character_ids
            ]
            if retained:
                checkpoint.session.visual_introductions[viewer_id] = retained
            else:
                checkpoint.session.visual_introductions.pop(viewer_id, None)

        checkpoint.characters = [
            character
            for character in checkpoint.characters
            if character.character_id not in character_ids
        ]
        roster.active = False
        roster.introductions_active = False
        return list(roster.records_by_id)

    def restore_roster(
        self,
        *,
        checkpoint: CheckpointFile,
        transaction_id: str,
    ) -> list[str]:
        """Restore a rejected candidate's overlay for continued same-beat work."""

        roster = self._rosters.get(transaction_id)
        if roster is None or not roster.records_by_id:
            return []
        restored = (
            self._restore_roster_records(checkpoint, roster)
            if not roster.active
            else []
        )
        if not roster.introductions_active:
            self._merge_introduction_buckets(
                checkpoint.session.visual_introductions,
                roster.introduction_buckets,
            )
            roster.introductions_active = True
        return restored

    def load_pending_introductions(
        self,
        *,
        checkpoint: CheckpointFile,
        transaction_id: str,
        introductions: dict[str, list[str]],
    ) -> None:
        """Load rejected-render edges without exposing them to the retry."""

        if not introductions:
            return
        roster = self._rosters.get(transaction_id)
        if roster is None or not roster.records_by_id:
            raise RuntimeError(
                "pending spawn introductions require staged spawn records"
            )
        if roster.introductions_active:
            raise RuntimeError(
                "pending spawn introductions are already active"
            )

        speculative_ids = set(roster.records_by_id)
        known_ids = {
            character.character_id for character in checkpoint.characters
        }
        for viewer_id, introduced_ids in introductions.items():
            if viewer_id not in known_ids:
                raise RuntimeError(
                    "pending spawn introductions contain an unknown viewer: "
                    f"{viewer_id}"
                )
            unknown_ids = set(introduced_ids) - known_ids
            if unknown_ids:
                raise RuntimeError(
                    "pending spawn introductions contain unknown characters: "
                    + ", ".join(sorted(unknown_ids))
                )
            if (
                viewer_id not in speculative_ids
                and any(cid not in speculative_ids for cid in introduced_ids)
            ):
                raise RuntimeError(
                    "pending spawn introduction edge does not involve a "
                    "speculative character"
                )

        active_subset = self._speculative_introduction_subset(
            checkpoint,
            speculative_ids,
        )
        if active_subset:
            raise RuntimeError(
                "pending spawn introductions must remain outside the active "
                "ledger"
            )
        self._merge_introduction_buckets(
            roster.introduction_buckets,
            introductions,
        )

    def pending_introductions(
        self,
        transaction_id: str,
    ) -> dict[str, list[str]]:
        roster = self._rosters.get(transaction_id)
        if roster is None:
            return {}
        return {
            viewer_id: list(introduced_ids)
            for viewer_id, introduced_ids in roster.introduction_buckets.items()
        }

    def accept_roster(
        self,
        *,
        checkpoint: CheckpointFile,
        transaction_id: str,
    ) -> list[str]:
        """Make the current overlay the accepted roster exactly once."""

        roster = self._rosters.get(transaction_id)
        if roster is None:
            return []
        if not roster.active:
            self._restore_roster_records(checkpoint, roster)
        if not roster.introductions_active:
            self._merge_introduction_buckets(
                checkpoint.session.visual_introductions,
                roster.introduction_buckets,
            )
            roster.introductions_active = True
        if roster.accepted:
            return []
        roster.accepted = True
        return list(roster.records_by_id)

    @staticmethod
    def _restore_roster_records(
        checkpoint: CheckpointFile,
        roster: _SpeculativeRoster,
    ) -> list[str]:
        existing_ids = {
            character.character_id for character in checkpoint.characters
        }
        conflicts = set(roster.records_by_id) & existing_ids
        if conflicts:
            raise ValueError(
                "cannot restore speculative spawn roster over existing ids: "
                + ", ".join(sorted(conflicts))
            )
        records = list(roster.records_by_id.values())
        checkpoint.characters = [*checkpoint.characters, *records]
        roster.active = True
        return [record.character_id for record in records]

    @staticmethod
    def _speculative_introduction_subset(
        checkpoint: CheckpointFile,
        speculative_ids: set[str],
    ) -> dict[str, list[str]]:
        subset: dict[str, list[str]] = {}
        for viewer_id, introduced_ids in (
            checkpoint.session.visual_introductions.items()
        ):
            if viewer_id in speculative_ids:
                retained = list(introduced_ids)
            else:
                retained = [
                    cid for cid in introduced_ids if cid in speculative_ids
                ]
            if retained:
                subset[viewer_id] = retained
        return subset

    @staticmethod
    def _merge_introduction_buckets(
        target: dict[str, list[str]],
        additions: dict[str, list[str]],
    ) -> None:
        for viewer_id, introduced_ids in additions.items():
            bucket = target.setdefault(viewer_id, [])
            seen = set(bucket)
            for character_id in introduced_ids:
                if character_id and character_id not in seen:
                    bucket.append(character_id)
                    seen.add(character_id)

    async def _author(
        self,
        checkpoint: CheckpointFile,
        requests: Sequence[SpawnRequest],
        *,
        acting_actor_location: str,
        one_star_hero_ids: set[str] | None,
    ) -> tuple[CharacterRecord, ...]:
        if not one_star_hero_ids:
            records = await self.character_manager.spawn_characters(
                checkpoint,
                list(requests),
                acting_actor_location=acting_actor_location,
            )
        else:
            records = await self.character_manager.spawn_characters(
                checkpoint,
                list(requests),
                acting_actor_location=acting_actor_location,
                one_star_hero_ids=one_star_hero_ids,
            )
        return tuple(
            character.model_copy(deep=True) for character in records
        )


def _consume_task_exception(
    task: asyncio.Task[tuple[CharacterRecord, ...]],
) -> None:
    if not task.cancelled():
        task.exception()
