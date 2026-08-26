from __future__ import annotations

import asyncio
import logging
import uuid

from app.engine.image_director import (
    ImageDirector,
    VisibleEventProjection,
    build_render_batch_projection_groups,
    projection_checkpoint_snapshot,
)
from app.engine.image_generation import ImageGenerationCoordinator
from app.engine.spawn_authoring import (
    SpawnAuthoringCoordinator,
    SpawnAuthoringKey,
)
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.image_director import ImageDirectorOutput
from app.schemas.state import RenderBufferEntry


logger = logging.getLogger(__name__)


class EventImageSidecar:
    """Runs speculative render-batch direction alongside narration."""

    def __init__(
        self,
        *,
        director: ImageDirector,
        generation: ImageGenerationCoordinator,
        spawn_authoring: SpawnAuthoringCoordinator,
    ) -> None:
        self.director = director
        self.generation = generation
        self.spawn_authoring = spawn_authoring
        self._wake = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._preparation_tasks: dict[
            str,
            set[asyncio.Task[None]],
        ] = {}
        self._projection_tasks: dict[str, set[asyncio.Task[None]]] = {}
        self._closing = False

    async def start(self) -> None:
        if self._runner is None or self._runner.done():
            self._closing = False
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
        self._projection_tasks.clear()

    async def wait_for_stage_discovery(self, session_id: str) -> None:
        """Wait until this session's candidate projections are durable."""

        while True:
            tasks = tuple(
                task
                for task in self._projection_tasks.get(session_id, set())
                if not task.done()
            )
            if not tasks:
                return
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=True,
            )

    async def start_render_candidate(
        self,
        *,
        checkpoint: CheckpointFile,
        buffered_events_by_pov: dict[str, list[RenderBufferEntry]],
        source_turn_index: int,
        source_checkpoint_sha256: str,
        spawn_keys_by_event_id: dict[str, SpawnAuthoringKey],
        actor_ids_by_event_id: dict[str, str],
    ) -> str | None:
        """Start one speculative direction/diffusion transaction per render."""
        if (
            checkpoint.session.config.settings.presentation_mode
            != "visual_novel"
            or self.generation.config.max_requests <= 0
        ):
            return None
        session_id = checkpoint.session.session_id
        eligible_viewer_ids = {
            viewer_id
            for viewer_id, entries in buffered_events_by_pov.items()
            if entries and self.generation.can_generate_render()
        }
        if not eligible_viewer_ids:
            return None
        transaction_id = f"imgtx_{uuid.uuid4().hex}"
        try:
            self.generation.begin_transaction(
                transaction_id=transaction_id,
                session_id=session_id,
                source_turn_index=source_turn_index,
                source_checkpoint_sha256=source_checkpoint_sha256,
            )
        except Exception:
            logger.exception("render image transaction setup failed")
            return None
        buffers = {
            viewer_id: [entry.model_copy(deep=True) for entry in entries]
            for viewer_id, entries in buffered_events_by_pov.items()
            if viewer_id in eligible_viewer_ids
        }
        buffered_event_ids = {
            entry.event_id for entries in buffers.values() for entry in entries
        }
        snapshot = projection_checkpoint_snapshot(
            checkpoint,
            event_ids=buffered_event_ids,
        )
        spawn_tasks = [
            task
            for event_id in buffered_event_ids
            if (key := spawn_keys_by_event_id.get(event_id)) is not None
            and (task := self.spawn_authoring.task(key)) is not None
        ]
        task = asyncio.create_task(
            self._prepare_projections(
                checkpoint=snapshot,
                buffered_events_by_pov=buffers,
                eligible_viewer_ids=eligible_viewer_ids,
                transaction_id=transaction_id,
                source_turn_index=source_turn_index,
                spawn_tasks=spawn_tasks,
                actor_ids_by_event_id=dict(actor_ids_by_event_id),
            ),
            name=f"image-render-project:{transaction_id}",
        )
        self._track_transaction_task(transaction_id, task)
        self._projection_tasks.setdefault(session_id, set()).add(task)
        task.add_done_callback(
            lambda completed, session_id=session_id: (
                self._projection_done(session_id, completed)
            )
        )
        return transaction_id

    async def cancel_transaction(
        self,
        transaction_id: str,
        *,
        reason: str = "render_candidate_rejected",
    ) -> None:
        tasks = self._preparation_tasks.pop(transaction_id, set())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.generation.cancel_transaction(
            transaction_id,
            reason=reason,
        )

    async def commit_transaction(
        self,
        transaction_id: str,
        *,
        target_checkpoint_sha256: str,
    ) -> None:
        await self.generation.commit_transaction(
            transaction_id,
            target_checkpoint_sha256=target_checkpoint_sha256,
        )

    async def _prepare_projections(
        self,
        *,
        checkpoint: CheckpointFile,
        buffered_events_by_pov: dict[str, list[RenderBufferEntry]],
        eligible_viewer_ids: set[str],
        transaction_id: str,
        source_turn_index: int,
        spawn_tasks: list[asyncio.Task[tuple[CharacterRecord, ...]]],
        actor_ids_by_event_id: dict[str, str],
    ) -> None:
        spawn_groups = await asyncio.gather(
            *(asyncio.shield(task) for task in spawn_tasks),
        ) if spawn_tasks else []
        spawned = tuple(
            record for records in spawn_groups for record in records
        )
        projections = build_render_batch_projection_groups(
            checkpoint=checkpoint,
            buffered_events_by_pov=buffered_events_by_pov,
            eligible_viewer_ids=eligible_viewer_ids,
            transaction_id=transaction_id,
            source_turn_index=source_turn_index,
            spawned_records=spawned,
            actor_ids_by_event_id=actor_ids_by_event_id,
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
                # Recovery is a queue cadence, not an idle-only maintenance
                # task. A stranded run in one busy session must not wait for
                # unrelated director work to drain before its lease is noticed.
                _, cancelled_attempts = (
                    self.generation.store
                    .recover_expired_director_runs_with_cleanup()
                )
                await self.generation.abort_cancelled_attempts(
                    cancelled_attempts
                )
                run = self.generation.store.claim_next_director_run()
            except Exception:
                logger.exception("image director queue recovery or claim failed")
                await asyncio.sleep(1)
                continue
            if run is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=5)
                except TimeoutError:
                    pass
                continue
            heartbeat = asyncio.create_task(
                self._heartbeat_director_run(run.run_id, run.attempts),
                name=f"image-director-heartbeat:{run.run_id}",
            )
            try:
                store = self.generation.store
                stage_context = store.visual_novel_stage_context_before_run(
                    run.run_id
                )
                output = await self.director.decide(
                    run.projection,
                    stage_context=stage_context,
                )
                completed = self.generation.store.complete_director_run(
                    run.run_id,
                    output,
                    attempt=run.attempts,
                )
                if (
                    completed is not None
                    and completed.status == "materializing"
                ):
                    # Admission is cheap and defines the durable stage that the
                    # next director run must observe. Diffusion still runs in
                    # the generation worker after enqueue; only the intentional
                    # first-portrait identity dependency waits for its result.
                    await self._materialize_requests(
                        run.run_id,
                        run.attempts,
                        run.projection,
                        output,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.generation.store.fail_director_run(
                    run.run_id,
                    type(exc).__name__,
                    attempt=run.attempts,
                )
                logger.exception(
                    "image director run failed event=%s",
                    run.projection.event_id,
                )
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    async def _heartbeat_director_run(
        self,
        run_id: str,
        attempt: int,
    ) -> None:
        while True:
            await asyncio.sleep(30)
            if not self.generation.store.heartbeat_director_run(
                run_id,
                attempt=attempt,
            ):
                return

    async def _materialize_requests(
        self,
        run_id: str,
        attempt: int,
        projection: VisibleEventProjection,
        output: ImageDirectorOutput,
    ) -> None:
        admitted_job_ids: list[str] = []
        try:
            if not self.generation.can_generate_render():
                logger.warning(
                    "image director produced requests but diffusion is unavailable"
                )
            else:
                # Story-stage images are consumed by the shared card composer.
                # Raw delivery remains reserved for explicit identity-review jobs.
                for ordinal, direction in enumerate(output.requests):
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
                        delivery_targets=[],
                        director_run_id=run_id,
                        director_attempt=attempt,
                    )
                    if job is None:
                        continue
                    admitted_job_ids.append(job.job_id)
                    # Keep later durable scene requests behind a first portrait
                    # so their frozen reference set can include the successful
                    # identity. This wait occurs only inside the presentation
                    # sidecar.
                    if establishes_unknown_identity:
                        await self.generation.wait_for_terminal(job.job_id)
            self.generation.store.finalize_director_materialization(
                run_id,
                attempt=attempt,
                projection=projection,
                admitted_job_ids=admitted_job_ids,
            )
        except asyncio.CancelledError:
            await self._fail_materialization(
                run_id,
                attempt,
                error_code="materialization_cancelled",
            )
            raise
        except Exception as exc:
            await self._fail_materialization(
                run_id,
                attempt,
                error_code=f"materialization_{type(exc).__name__}",
            )
            raise

    async def _fail_materialization(
        self,
        run_id: str,
        attempt: int,
        *,
        error_code: str,
    ) -> None:
        failed, cancelled_attempts = (
            self.generation.store.fail_director_run_with_cleanup(
                run_id,
                error_code,
                attempt=attempt,
            )
        )
        if failed is None:
            return
        await self.generation.abort_cancelled_attempts(cancelled_attempts)

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

    def _projection_done(
        self,
        session_id: str,
        task: asyncio.Task[None],
    ) -> None:
        tasks = self._projection_tasks.get(session_id)
        if tasks is None:
            return
        tasks.discard(task)
        if not tasks:
            self._projection_tasks.pop(session_id, None)

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
