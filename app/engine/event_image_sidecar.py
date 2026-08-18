from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

from app.engine.image_director import (
    ImageDirector,
    VisibleEventProjection,
    build_projection_groups,
    projection_checkpoint_snapshot,
)
from app.engine.image_generation import (
    ImageDeliveryTarget,
    ImageGenerationCoordinator,
)
from app.engine.spawn_authoring import (
    SpawnAuthoringCoordinator,
    SpawnAuthoringKey,
)
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.image_director import ImageDirection
from app.schemas.image_generation import ImageDeliveryKind


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventImageSidecarConfig:
    mode: str = "disabled"
    recent_illustration_limit: int = 8

    @property
    def director_enabled(self) -> bool:
        return self.mode in {"shadow", "enabled"}

    @property
    def diffusion_enabled(self) -> bool:
        return self.mode == "enabled"

    @classmethod
    def from_environment(cls) -> "EventImageSidecarConfig":
        raw = os.getenv("AYOA_IMAGE_DIRECTOR_ENABLED", "0").strip().lower()
        aliases = {
            "0": "disabled",
            "false": "disabled",
            "off": "disabled",
            "disabled": "disabled",
            "shadow": "shadow",
            "1": "enabled",
            "true": "enabled",
            "on": "enabled",
            "enabled": "enabled",
        }
        try:
            mode = aliases[raw]
        except KeyError as exc:
            raise ValueError(
                "AYOA_IMAGE_DIRECTOR_ENABLED must be 0, shadow, or 1"
            ) from exc
        return cls(mode=mode)


class EventImageSidecar:
    """Runs event projections independently from narration and turn latency."""

    def __init__(
        self,
        *,
        director: ImageDirector,
        generation: ImageGenerationCoordinator,
        spawn_authoring: SpawnAuthoringCoordinator,
        delivery_kind: ImageDeliveryKind,
        config: EventImageSidecarConfig | None = None,
    ) -> None:
        self.director = director
        self.generation = generation
        self.spawn_authoring = spawn_authoring
        self.delivery_kind = delivery_kind
        self.config = config or EventImageSidecarConfig.from_environment()
        self._wake = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._preparation_tasks: dict[
            str,
            set[asyncio.Task[None]],
        ] = {}
        self._closing = False

    async def start(self) -> None:
        if not self.config.director_enabled:
            return
        if self._runner is None or self._runner.done():
            self._closing = False
            self.generation.store.recover_expired_director_runs()
            self._runner = asyncio.create_task(
                self._run_director_queue(),
                name="ayoa-image-director",
            )
            self._wake.set()

    async def close(self) -> None:
        self._closing = True
        tasks = [
            task
            for transaction_tasks in self._preparation_tasks.values()
            for task in transaction_tasks
        ]
        if self._runner is not None:
            self._runner.cancel()
            tasks.append(self._runner)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._runner = None
        self._preparation_tasks.clear()

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
        if not self.config.director_enabled:
            return
        snapshot = projection_checkpoint_snapshot(checkpoint)
        event_snapshot = event.model_copy(deep=True)
        spawn_task = (
            self.spawn_authoring.task(spawn_key)
            if spawn_key is not None
            else None
        )
        task = asyncio.create_task(
            self._prepare_projections(
                checkpoint=snapshot,
                event=event_snapshot,
                event_sequence=event_sequence,
                transaction_id=transaction_id,
                source_turn_index=source_turn_index,
                spawn_task=spawn_task,
                actor_id=actor_id,
            ),
            name=f"image-project:{event.event_id}",
        )
        self._track_transaction_task(transaction_id, task)

    async def cancel_transaction(self, transaction_id: str) -> None:
        tasks = self._preparation_tasks.pop(transaction_id, set())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _prepare_projections(
        self,
        *,
        checkpoint: CheckpointFile,
        event: EventRouterOutput,
        event_sequence: int,
        transaction_id: str,
        source_turn_index: int,
        spawn_task: asyncio.Task[tuple[CharacterRecord, ...]] | None,
        actor_id: str,
    ) -> None:
        spawned = (
            await asyncio.shield(spawn_task)
            if spawn_task is not None
            else ()
        )
        projections = build_projection_groups(
            checkpoint=checkpoint,
            event=event,
            event_sequence=event_sequence,
            transaction_id=transaction_id,
            source_turn_index=source_turn_index,
            spawned_records=spawned,
            actor_id=actor_id,
            active_identity_character_ids=(
                self.generation.active_identity_character_ids(
                    checkpoint.session.session_id
                )
            ),
            active_location_labels=(
                self.generation.active_reviewed_location_labels(
                    checkpoint.session.session_id
                )
            ),
            delivery_kind=self.delivery_kind.value,
        )
        for projection in projections:
            try:
                self.generation.store.enqueue_director_run(projection)
            except RuntimeError as exc:
                if "active image transaction" in str(exc):
                    # A rewind or failed turn won the race while this immutable
                    # projection was waiting for spawn metadata.
                    return
                raise
        if projections:
            self._wake.set()

    async def _run_director_queue(self) -> None:
        while not self._closing:
            try:
                run = self.generation.store.claim_next_director_run()
            except Exception:
                logger.exception("image director queue claim failed")
                await asyncio.sleep(1)
                continue
            if run is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=5)
                except TimeoutError:
                    self.generation.store.recover_expired_director_runs()
                continue
            heartbeat = asyncio.create_task(
                self._heartbeat_director_run(run.run_id),
                name=f"image-director-heartbeat:{run.run_id}",
            )
            try:
                output = await self.director.decide(
                    run.projection,
                    recent_illustrations=(
                        self.generation.store.recent_illustrations(
                            run.projection.session_id,
                            viewer_character_ids=(
                                run.projection.viewer_character_ids
                            ),
                            limit=self.config.recent_illustration_limit,
                        )
                    ),
                )
                completed = self.generation.store.complete_director_run(
                    run.run_id,
                    output,
                )
                if (
                    self.config.diffusion_enabled
                    and completed is not None
                    and completed.status == "succeeded"
                ):
                    materialization = asyncio.create_task(
                        self._materialize_requests(
                            run.projection,
                            output.requests,
                        ),
                        name=(
                            "image-materialize:"
                            f"{run.projection.event_id}"
                        ),
                    )
                    self._track_transaction_task(
                        run.projection.transaction_id,
                        materialization,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.generation.store.fail_director_run(
                    run.run_id,
                    type(exc).__name__,
                )
                logger.exception(
                    "image director run failed event=%s",
                    run.projection.event_id,
                )
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    async def _heartbeat_director_run(self, run_id: str) -> None:
        while True:
            await asyncio.sleep(30)
            if not self.generation.store.heartbeat_director_run(run_id):
                return

    async def _materialize_requests(
        self,
        projection: VisibleEventProjection,
        requests: Sequence[ImageDirection],
    ) -> None:
        if not self.generation.available:
            logger.warning(
                "image director produced requests but diffusion is unavailable"
            )
            return
        targets = _delivery_targets(projection)
        for ordinal, direction in enumerate(requests):
            establishes_unknown_identity = bool(
                direction.kind == "portrait"
                and len(direction.subject_character_ids) == 1
                and direction.subject_character_ids[0]
                not in self.generation.active_identity_character_ids(
                    projection.session_id
                )
            )
            job = await self.generation.enqueue_direction(
                projection=projection,
                direction=direction,
                request_ordinal=ordinal,
                visual_style=projection.engine_visual_style,
                delivery_targets=targets,
            )
            # Keep later durable scene requests behind a first portrait so
            # their frozen reference set can include the successful identity.
            # This wait occurs only inside the presentation sidecar.
            if establishes_unknown_identity and job is not None:
                await self.generation.wait_for_terminal(job.job_id)

    def _preparation_done(
        self,
        transaction_id: str,
        task: asyncio.Task[None],
    ) -> None:
        tasks = self._preparation_tasks.get(transaction_id)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                self._preparation_tasks.pop(transaction_id, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception(
                "event image projection failed transaction=%s",
                transaction_id,
            )

    def _track_transaction_task(
        self,
        transaction_id: str,
        task: asyncio.Task[None],
    ) -> None:
        self._preparation_tasks.setdefault(transaction_id, set()).add(task)
        task.add_done_callback(
            lambda completed, transaction_id=transaction_id: (
                self._preparation_done(transaction_id, completed)
            )
        )


def _delivery_targets(
    projection: VisibleEventProjection,
) -> list[ImageDeliveryTarget]:
    try:
        kind = ImageDeliveryKind(projection.delivery_kind)
    except ValueError:
        raise ValueError("unsupported event image delivery kind") from None
    user_by_pov = dict(projection.viewer_delivery_bindings)
    return [
        ImageDeliveryTarget(
            pov_character_id=pov_character_id,
            delivery_kind=kind,
            delivery={
                "character_id": pov_character_id,
                "user_id": user_by_pov.get(pov_character_id, ""),
            },
        )
        for pov_character_id in projection.viewer_character_ids
    ]
