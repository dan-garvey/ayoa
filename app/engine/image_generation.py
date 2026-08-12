from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.engine.image_job_store import ImageJobStore
from app.engine.image_worker_client import (
    ImageWorkerClient,
    ImageWorkerConfig,
    ImageWorkerError,
)
from app.engine.player_media import (
    PlayerMediaError,
    ResolvedPlayerMedia,
    finalize_generated_webp,
    resolve_generated_media,
)
from app.engine.text_safety import strip_terminal_control
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import redact_imported_asset_text
from app.schemas.image_generation import (
    ImageDeliveryKind,
    ImageGenerationJob,
    ImageGenerationRequest,
    ImageGenerationStatus,
    ImageTriggerKind,
)
from app.schemas.responses import TurnResponse


logger = logging.getLogger(__name__)

_ELIGIBLE_TRIGGERS = {
    ImageTriggerKind.act,
    ImageTriggerKind.begin,
    ImageTriggerKind.arrival,
    ImageTriggerKind.roll_resolution,
    ImageTriggerKind.render_retry,
}
_INELIGIBLE_REASON_PARTS = {
    "blocked",
    "deferred",
    "no_pending_render",
    "pending",
    "pre_turn_resolution",
    "rejected",
    "stale",
}
_SAFE_CHECKPOINT_RE = re.compile(r"^ckpt_[0-9]{4,}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ImageWorker(Protocol):
    @property
    def available(self) -> bool:
        ...

    async def generate(
        self,
        request: ImageGenerationRequest,
        *,
        output_path: str | Path,
    ) -> Any:
        ...

    async def abort_current(self) -> None:
        ...

    async def close(self) -> None:
        ...


ImageDeliveryHandler = Callable[
    [ImageGenerationJob, ResolvedPlayerMedia],
    Awaitable[bool],
]


@dataclass(frozen=True)
class ImageGenerationConfig:
    runtime_root: Path
    queue_limit: int = 16
    cli_wait_timeout_seconds: float = 120.0
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance: float = 1.0
    max_prose_chars: int = 3_000
    max_style_chars: int = 600
    max_character_context_chars: int = 1_500
    max_character_context_count: int = 4

    @classmethod
    def from_environment(
        cls,
        *,
        runtime_root: str | Path,
    ) -> "ImageGenerationConfig":
        return cls(
            runtime_root=Path(runtime_root),
            queue_limit=max(1, int(os.getenv("AYOA_IMAGE_QUEUE_LIMIT", "16"))),
            cli_wait_timeout_seconds=max(
                1.0,
                float(os.getenv("AYOA_IMAGE_CLI_WAIT_SECONDS", "120")),
            ),
            width=int(os.getenv("AYOA_IMAGE_WIDTH", "1024")),
            height=int(os.getenv("AYOA_IMAGE_HEIGHT", "1024")),
        )


class ImageGenerationCoordinator:
    """Durable, optional output-only local illustration coordinator."""

    def __init__(
        self,
        *,
        sessions_dir: str | Path,
        config: ImageGenerationConfig,
        worker: ImageWorker | None = None,
        store: ImageJobStore | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.config = config
        self.config.runtime_root.mkdir(parents=True, exist_ok=True)
        self.store = store or ImageJobStore(
            self.config.runtime_root / "jobs.sqlite"
        )
        self.worker = worker or ImageWorkerClient(
            ImageWorkerConfig.from_environment(
                runtime_root=self.config.runtime_root,
                repo_root=repo_root,
            )
        )
        self._delivery_handlers: dict[ImageDeliveryKind, ImageDeliveryHandler] = {}
        self._runner: asyncio.Task[None] | None = None
        self._delivery_runner: asyncio.Task[None] | None = None
        self._ownership_runner: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._changed = asyncio.Condition()
        self._started = False
        self._closing = False
        self._current_job: ImageGenerationJob | None = None
        self._unavailable_logged = False
        self._queue_lock_handle: Any = None
        self._queue_owner = False

    @property
    def available(self) -> bool:
        return bool(self.worker.available)

    def register_delivery_handler(
        self,
        delivery_kind: ImageDeliveryKind,
        handler: ImageDeliveryHandler,
    ) -> None:
        self._delivery_handlers[delivery_kind] = handler
        if (
            self._started
            and self.available
            and self._delivery_runner is None
        ):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return
            self._delivery_runner = asyncio.create_task(
                self._run_delivery_poll(),
                name="ayoa-image-delivery",
            )

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        preflight = getattr(self.worker, "preflight", None)
        if self.available and callable(preflight):
            if not await preflight():
                if not self._unavailable_logged:
                    logger.warning(
                        "local image generation preflight failed; "
                        "text play remains enabled"
                    )
                    self._unavailable_logged = True
                return
        if not self.available:
            if not self._unavailable_logged:
                logger.info(
                    "local image generation unavailable; text play remains enabled"
                )
                self._unavailable_logged = True
            return
        self._closing = False
        self._queue_owner = self._acquire_queue_owner()
        if self._queue_owner:
            self._activate_queue_owner()
        else:
            logger.info(
                "another Ayoa process owns the local image queue; "
                "this process will enqueue and observe durable jobs"
            )
        if self._delivery_handlers:
            self._delivery_runner = asyncio.create_task(
                self._run_delivery_poll(),
                name="ayoa-image-delivery",
            )
        self._ownership_runner = asyncio.create_task(
            self._run_owner_election(),
            name="ayoa-image-owner-election",
        )
        self._wake.set()

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        if self._ownership_runner is not None:
            self._ownership_runner.cancel()
            await asyncio.gather(self._ownership_runner, return_exceptions=True)
            self._ownership_runner = None
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
            self._runner = None
        if self._delivery_runner is not None:
            self._delivery_runner.cancel()
            await asyncio.gather(self._delivery_runner, return_exceptions=True)
            self._delivery_runner = None
        await self.worker.close()
        self._release_queue_owner()
        self._started = False

    async def enqueue_turn(
        self,
        *,
        ckpt: CheckpointFile,
        response: TurnResponse,
        actor_character_id: str,
        trigger_kind: ImageTriggerKind | str,
        delivery_kind: ImageDeliveryKind | str,
        delivery: dict[str, Any] | None = None,
    ) -> ImageGenerationJob | None:
        await self.start()
        if not self.available:
            return None
        try:
            trigger = ImageTriggerKind(trigger_kind)
            destination = ImageDeliveryKind(delivery_kind)
        except ValueError:
            return None
        if not image_turn_is_eligible(
            ckpt=ckpt,
            response=response,
            actor_character_id=actor_character_id,
            trigger_kind=trigger,
        ):
            return None

        prose = response.per_player_renders.get(actor_character_id, "")
        prompt = build_image_prompt(
            ckpt,
            actor_character_id=actor_character_id,
            prose=prose,
            config=self.config,
        )
        if not prompt:
            return None
        checkpoint_path = self._checkpoint_path(
            response.session_id,
            response.checkpoint_id,
        )
        try:
            checkpoint_sha256 = _sha256_file(checkpoint_path)
        except OSError:
            logger.warning(
                "image generation skipped: source checkpoint is unavailable"
            )
            return None
        every = max(
            1,
            int(
                getattr(
                    ckpt.session.config.settings,
                    "image_generation_every_n_beats",
                    1,
                )
            ),
        )
        eligible_ordinal = self.store.register_eligible_beat(
            session_id=response.session_id,
            actor_character_id=actor_character_id,
            checkpoint_sha256=checkpoint_sha256,
            turn_index=response.turn_index,
        )
        if eligible_ordinal % every:
            return None

        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        generation_payload = {
            "actor": actor_character_id,
            "checkpoint_sha256": checkpoint_sha256,
            "guidance": self.config.guidance,
            "height": self.config.height,
            "model_id": self._worker_model_id(),
            "model_revision": self._worker_model_revision(),
            "prompt_sha256": prompt_sha256,
            "steps": self.config.steps,
            "trigger": trigger.value,
            "width": self.config.width,
        }
        delivery_value = dict(delivery or {})
        delivery_identity = {
            key: delivery_value.get(key)
            for key in (
                "session_channel_id",
                "user_id",
                "character_id",
            )
            if delivery_value.get(key) is not None
        }
        dedupe_payload = {
            "generation": generation_payload,
            "delivery_kind": destination.value,
            "delivery_identity": delivery_identity,
        }
        dedupe_key = hashlib.sha256(
            json.dumps(
                dedupe_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        existing = self.store.get(f"img_{dedupe_key[:32]}")
        if existing is not None:
            if existing.status in {
                ImageGenerationStatus.failed,
                ImageGenerationStatus.cancelled,
            } and (existing.artifact is not None or existing.attempts < 2):
                retried = self.store.requeue_retryable(existing.job_id)
                if retried is not None:
                    existing = retried
                    if existing.status == ImageGenerationStatus.queued:
                        self._wake.set()
            if existing.status == ImageGenerationStatus.succeeded:
                asyncio.create_task(self._dispatch_completion(existing))
            return existing
        if self.store.active_count() >= self.config.queue_limit:
            logger.warning("image generation queue is full; illustration skipped")
            return None

        seed_payload = {
            **generation_payload,
            "session_id": response.session_id,
            "turn_index": response.turn_index,
        }
        seed = int.from_bytes(
            hashlib.sha256(
                json.dumps(
                    seed_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        request = ImageGenerationRequest(
            session_id=response.session_id,
            checkpoint_id=response.checkpoint_id,
            checkpoint_sha256=checkpoint_sha256,
            turn_index=response.turn_index,
            actor_character_id=actor_character_id,
            trigger_kind=trigger,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            model_id=self._worker_model_id(),
            model_revision=self._worker_model_revision(),
            width=self.config.width,
            height=self.config.height,
            steps=self.config.steps,
            guidance=self.config.guidance,
            seed=seed,
            dedupe_key=dedupe_key,
            delivery_kind=destination,
            delivery=delivery_value,
        )
        job = self.store.enqueue(request)
        logger.info(
            "queued local illustration job=%s turn=%d delivery=%s depth=%d",
            job.job_id,
            request.turn_index,
            destination.value,
            self.store.active_count(),
        )
        self._wake.set()
        await self._notify_changed()
        return job

    async def wait_for_terminal(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
    ) -> ImageGenerationJob | None:
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            async with self._changed:
                job = self.store.get(job_id)
                if job is None or job.status in {
                    ImageGenerationStatus.succeeded,
                    ImageGenerationStatus.delivered,
                    ImageGenerationStatus.failed,
                    ImageGenerationStatus.cancelled,
                }:
                    return job
                remaining = None if deadline is None else deadline - loop.time()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError
                poll_timeout = 0.5 if remaining is None else min(0.5, remaining)
                try:
                    await asyncio.wait_for(
                        self._changed.wait(),
                        timeout=poll_timeout,
                    )
                except TimeoutError:
                    continue

    def resolve_job_media(self, job: ImageGenerationJob) -> ResolvedPlayerMedia:
        if job.artifact is None:
            raise PlayerMediaError("job_has_no_artifact")
        return resolve_generated_media(
            job.artifact,
            runtime_root=self.config.runtime_root,
        )

    def pending_for_delivery(
        self,
        delivery_kind: ImageDeliveryKind,
        *,
        session_id: str,
        actor_character_ids: set[str] | None = None,
    ) -> list[ImageGenerationJob]:
        jobs = self.store.pending_delivery(
            delivery_kind,
            session_id=session_id,
        )
        if actor_character_ids is None:
            return jobs
        return [
            job
            for job in jobs
            if job.request.actor_character_id in actor_character_ids
        ]

    async def mark_delivered(self, job_id: str) -> ImageGenerationJob | None:
        current = self.store.get(job_id)
        if current is not None and current.status == ImageGenerationStatus.succeeded:
            self.store.claim_delivery(job_id)
        job = self.store.mark_delivered(job_id)
        await self._notify_changed()
        return job

    def delivery_is_current(self, job_id: str) -> bool:
        job = self.store.get(job_id)
        current = bool(
            job is not None
            and job.status == ImageGenerationStatus.delivering
            and self._source_checkpoint_matches(job.request)
        )
        return current and self.store.heartbeat_delivery(job_id)

    async def cancel_after(self, session_id: str, turn_index: int) -> int:
        cancelled = self.store.cancel_after(session_id, turn_index)
        current = self._current_job
        if (
            current is not None
            and current.request.session_id == session_id
            and current.request.turn_index > turn_index
        ):
            await self.worker.abort_current()
        if cancelled:
            logger.info(
                "cancelled %d image job(s) after rewind turn %d",
                cancelled,
                turn_index,
            )
            await self._notify_changed()
        return cancelled

    async def cancel_job(
        self,
        job_id: str,
        *,
        error_code: str = "cancelled",
    ) -> ImageGenerationJob | None:
        current = self._current_job
        job = self.store.mark_cancelled(job_id, error_code)
        if current is not None and current.job_id == job_id:
            await self.worker.abort_current()
        await self._notify_changed()
        return job

    async def cancel_session(self, session_id: str) -> int:
        cancelled = self.store.cancel_session(session_id)
        current = self._current_job
        if current is not None and current.request.session_id == session_id:
            await self.worker.abort_current()
        if cancelled:
            await self._notify_changed()
        return cancelled

    async def cancel_discord_destination(
        self,
        *,
        session_id: str,
        session_channel_id: int,
    ) -> int:
        cancelled = self.store.cancel_discord_destination(
            session_id=session_id,
            session_channel_id=session_channel_id,
        )
        current = self._current_job
        if current is not None and current.request.session_id == session_id:
            try:
                current_channel = int(
                    current.request.delivery.get("session_channel_id")
                )
            except (TypeError, ValueError):
                current_channel = -1
            if current_channel == int(session_channel_id):
                await self.worker.abort_current()
        if cancelled:
            await self._notify_changed()
        return cancelled

    async def _run(self) -> None:
        while not self._closing:
            self.store.recover_expired_deliveries()
            job = self.store.claim_next()
            if job is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=5)
                except TimeoutError:
                    pass
                continue
            self._current_job = job
            try:
                await self._process(job)
            finally:
                self._current_job = None

    async def _run_delivery_poll(self) -> None:
        while not self._closing:
            recovered = self.store.recover_expired_deliveries()
            if recovered:
                logger.warning(
                    "recovered %d expired image delivery lease(s)",
                    recovered,
                )
            await self._redeliver_succeeded()
            await asyncio.sleep(1)

    async def _run_owner_election(self) -> None:
        while not self._closing:
            if not self._queue_owner and self._acquire_queue_owner():
                self._queue_owner = True
                logger.info("this process acquired the local image queue lease")
                self._activate_queue_owner()
                self._wake.set()
            elif (
                self._queue_owner
                and self._runner is not None
                and self._runner.done()
            ):
                logger.error("local image queue runner stopped; restarting")
                self._runner = asyncio.create_task(
                    self._run(),
                    name="ayoa-image-generation",
                )
            await asyncio.sleep(1)

    async def _process(self, job: ImageGenerationJob) -> None:
        temp_dir = self.config.runtime_root / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{job.job_id}.webp"
        temp_path.unlink(missing_ok=True)
        if not self._source_checkpoint_matches(job.request):
            self.store.mark_cancelled(job.job_id, "stale_checkpoint")
            await self._notify_changed()
            return
        try:
            result = await self._generate_with_cancellation(job, temp_path)
            current = self.store.get(job.job_id)
            if (
                current is None
                or current.status != ImageGenerationStatus.running
                or not self._source_checkpoint_matches(job.request)
            ):
                temp_path.unlink(missing_ok=True)
                if current is not None and current.status == ImageGenerationStatus.running:
                    self.store.mark_cancelled(job.job_id, "stale_checkpoint")
                await self._notify_changed()
                return
            artifact = finalize_generated_webp(
                temp_path,
                runtime_root=self.config.runtime_root,
                worker_result=result,
                expected_width=job.request.width,
                expected_height=job.request.height,
            )
            completed = self.store.mark_succeeded(job.job_id, artifact)
            await self._notify_changed()
            if completed is not None:
                logger.info(
                    "local illustration succeeded job=%s bytes=%d seconds=%.2f",
                    completed.job_id,
                    artifact.byte_count,
                    float(result.generation_seconds),
                )
                if completed.request.delivery_kind in self._delivery_handlers:
                    await self._dispatch_completion(completed)
        except (ImageWorkerError, PlayerMediaError) as exc:
            temp_path.unlink(missing_ok=True)
            current = self.store.get(job.job_id)
            if current is not None and current.status == ImageGenerationStatus.running:
                self.store.mark_failed(job.job_id, exc.code)
                logger.warning(
                    "local illustration failed job=%s code=%s",
                    job.job_id,
                    exc.code,
                )
            await self._notify_changed()
        except Exception:
            temp_path.unlink(missing_ok=True)
            current = self.store.get(job.job_id)
            if current is not None and current.status == ImageGenerationStatus.running:
                self.store.mark_failed(job.job_id, "generation_failed")
            logger.exception("unexpected local illustration failure job=%s", job.job_id)
            await self._notify_changed()

    async def _generate_with_cancellation(
        self,
        job: ImageGenerationJob,
        temp_path: Path,
    ) -> Any:
        generation = asyncio.create_task(
            self.worker.generate(job.request, output_path=temp_path)
        )
        cancellation = asyncio.create_task(
            self._wait_for_external_cancellation(job.job_id)
        )
        try:
            done, _pending = await asyncio.wait(
                {generation, cancellation},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation in done:
                await self.worker.abort_current()
                await asyncio.gather(generation, return_exceptions=True)
                raise ImageWorkerError("worker_cancelled")
            cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)
            return await generation
        finally:
            if not generation.done():
                generation.cancel()
            if not cancellation.done():
                cancellation.cancel()

    async def _wait_for_external_cancellation(self, job_id: str) -> None:
        while True:
            current = self.store.get(job_id)
            if current is None or current.status != ImageGenerationStatus.running:
                return
            await asyncio.sleep(0.2)

    async def _dispatch_completion(self, job: ImageGenerationJob) -> None:
        current = self.store.claim_delivery(job.job_id)
        if current is None:
            return
        if not self._source_checkpoint_matches(current.request):
            self.store.mark_cancelled(current.job_id, "stale_checkpoint")
            await self._notify_changed()
            return
        handler = self._delivery_handlers.get(current.request.delivery_kind)
        if handler is None:
            self.store.release_delivery(current.job_id)
            return
        delivered = False
        try:
            media = self.resolve_job_media(current)
            delivered = await handler(current, media)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "generated illustration delivery failed job=%s",
                current.job_id,
            )
        finally:
            latest = self.store.get(current.job_id)
            if (
                latest is not None
                and latest.status == ImageGenerationStatus.delivering
            ):
                if delivered:
                    self.store.mark_delivered(current.job_id)
                else:
                    self.store.release_delivery(current.job_id)
            await self._notify_changed()

    async def _redeliver_succeeded(self) -> None:
        for kind in tuple(self._delivery_handlers):
            for job in self.store.succeeded_undelivered(kind):
                await self._dispatch_completion(job)

    def _source_checkpoint_matches(self, request: ImageGenerationRequest) -> bool:
        try:
            return (
                _sha256_file(
                    self._checkpoint_path(
                        request.session_id,
                        request.checkpoint_id,
                    )
                )
                == request.checkpoint_sha256
            )
        except OSError:
            return False

    def _checkpoint_path(self, session_id: str, checkpoint_id: str) -> Path:
        if not _SAFE_ID_RE.fullmatch(session_id) or not _SAFE_CHECKPOINT_RE.fullmatch(
            checkpoint_id
        ):
            raise OSError("unsafe checkpoint identity")
        return self.sessions_dir / session_id / f"{checkpoint_id}.json"

    def _worker_model_id(self) -> str:
        config = getattr(self.worker, "config", None)
        return str(getattr(config, "model_id", "") or "test-local-image-model")

    def _worker_model_revision(self) -> str:
        config = getattr(self.worker, "config", None)
        return str(getattr(config, "model_revision", "") or "test-revision")

    async def _notify_changed(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    def _acquire_queue_owner(self) -> bool:
        if self._queue_lock_handle is not None:
            return True
        lock_path = self.config.runtime_root / "coordinator.lock"
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        self._queue_lock_handle = handle
        return True

    def _release_queue_owner(self) -> None:
        handle = self._queue_lock_handle
        self._queue_lock_handle = None
        self._queue_owner = False
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _activate_queue_owner(self) -> None:
        recovered = self.store.recover_interrupted()
        if recovered:
            logger.info(
                "requeued %d interrupted image generation job(s)",
                recovered,
            )
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(
                self._run(),
                name="ayoa-image-generation",
            )


def image_turn_is_eligible(
    *,
    ckpt: CheckpointFile,
    response: TurnResponse,
    actor_character_id: str,
    trigger_kind: ImageTriggerKind,
) -> bool:
    settings = ckpt.session.config.settings
    if trigger_kind not in _ELIGIBLE_TRIGGERS:
        return False
    if str(getattr(settings, "image_generation_mode", "actor")) != "actor":
        return False
    if not actor_character_id or response.session_id != ckpt.session.session_id:
        return False
    if response.turn_index != ckpt.session.turn_index:
        return False
    if response.checkpoint_id != f"ckpt_{response.turn_index:04d}":
        return False
    prose = (response.per_player_renders or {}).get(actor_character_id, "")
    if not prose or not prose.strip():
        return False
    reason = str(response.beat_ended_reason or "").strip().lower()
    return not any(part in reason for part in _INELIGIBLE_REASON_PARTS)


def build_image_prompt(
    ckpt: CheckpointFile,
    *,
    actor_character_id: str,
    prose: str,
    config: ImageGenerationConfig,
) -> str:
    scene = _safe_prompt_text(prose, config.max_prose_chars)
    if not scene:
        return ""

    setting = ckpt.world_state.setting
    authored_style = _safe_prompt_text(
        getattr(setting, "visual_style", ""),
        config.max_style_chars,
    )
    if authored_style:
        style = authored_style
    else:
        style_parts = [
            _safe_prompt_text(getattr(setting, field, ""), 200)
            for field in ("genre", "era", "tone")
        ]
        style = "; ".join(part for part in style_parts if part)
    if not style:
        style = "cinematic narrative illustration, coherent natural composition"

    character_lines: list[str] = []
    introduced = set(
        (ckpt.session.visual_introductions or {}).get(actor_character_id, [])
    )
    for character in ckpt.characters:
        if len(character_lines) >= config.max_character_context_count:
            break
        character_id = str(character.character_id or "").strip()
        if character_id != actor_character_id and character_id not in introduced:
            continue
        name = str(character.name or "").strip()
        mentioned_by_name = bool(name and _mentions(scene, name))
        mentioned_by_id = bool(
            "_" in character_id and _mentions(scene, character_id)
        )
        if not (mentioned_by_name or mentioned_by_id):
            continue
        appearance = _safe_prompt_text(
            getattr(character.public_sheet, "appearance", ""),
            400,
        )
        loadout = _safe_prompt_text(
            getattr(character.visuals, "default_loadout", ""),
            500,
        )
        visual = "; ".join(part for part in (appearance, loadout) if part)
        if not visual:
            continue
        name = _safe_prompt_text(character.name or character.character_id, 100)
        character_lines.append(f"- {name}: {visual}")

    character_block = "\n".join(character_lines)
    if len(character_block) > config.max_character_context_chars:
        character_block = character_block[: config.max_character_context_chars].rstrip()
    parts = [
        "Create one story illustration of the visible moment below.",
        f"Visual direction: {style}",
        f"Scene:\n{scene}",
    ]
    if character_block:
        parts.append(f"Visible character continuity:\n{character_block}")
    parts.append(
        "Show only what this scene makes visible. Do not add a caption, "
        "interface, watermark, or explanatory text."
    )
    return "\n\n".join(parts)


def _safe_prompt_text(value: str, max_chars: int) -> str:
    text = strip_terminal_control(redact_imported_asset_text(value))
    text = " ".join(text.split())
    return text[:max_chars].rstrip() if max_chars > 0 else ""


def _mentions(text: str, probe: str) -> bool:
    if not probe:
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(probe)}(?![A-Za-z0-9_])",
            text,
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
