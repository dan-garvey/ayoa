from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.engine.image_director import (
    PublicCharacterVisual,
    VisibleEventProjection,
    source_event_fingerprint,
    text_names_public_character,
)
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
from app.schemas.image_director import ImageDirection, ImageGenerationMode
from app.schemas.checkpoint import CheckpointFile
from app.schemas.image_generation import (
    FrozenReferenceInput,
    IdentityReferenceCandidate,
    IdentityReferenceStatus,
    ImageDelivery,
    ImageDeliveryKind,
    ImageGenerationJob,
    ImageGenerationRequest,
    ImageGenerationStatus,
)


logger = logging.getLogger(__name__)


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
    [
        ImageGenerationJob,
        ImageDelivery,
        ResolvedPlayerMedia,
        str,
    ],
    Awaitable[bool],
]


@dataclass(frozen=True)
class ImageDeliveryTarget:
    pov_character_id: str
    delivery_kind: ImageDeliveryKind
    delivery: dict[str, object]


@dataclass(frozen=True)
class ImageGenerationConfig:
    runtime_root: Path
    queue_limit: int = 48
    per_session_queue_limit: int = 16
    max_requests: int = 6
    max_subjects: int = 4
    max_references: int = 4
    max_reference_bytes: int = 20_000_000
    max_scene_prompt_chars: int = 2_000
    max_style_chars: int = 800
    tandem_delivery_wait_seconds: float = 20.0
    transaction_recovery_lease_seconds: float = 3_600
    steps: int = 50
    guidance: float = 4.0

    @classmethod
    def from_environment(
        cls,
        *,
        runtime_root: str | Path,
    ) -> "ImageGenerationConfig":
        image_backend = os.getenv(
            "AYOA_IMAGE_WORKER_BACKEND",
            "local",
        ).strip().lower()
        default_steps = "20" if image_backend == "remote" else "50"
        return cls(
            runtime_root=Path(runtime_root),
            queue_limit=max(
                1,
                int(os.getenv("AYOA_IMAGE_QUEUE_LIMIT", "48")),
            ),
            per_session_queue_limit=max(
                1,
                int(os.getenv("AYOA_IMAGE_SESSION_QUEUE_LIMIT", "16")),
            ),
            max_requests=max(
                0,
                int(os.getenv("AYOA_IMAGE_MAX_REQUESTS", "6")),
            ),
            tandem_delivery_wait_seconds=max(
                0.0,
                float(os.getenv("AYOA_IMAGE_TANDEM_WAIT_SECONDS", "20")),
            ),
            max_subjects=max(
                1,
                int(os.getenv("AYOA_IMAGE_MAX_SUBJECTS", "4")),
            ),
            max_references=max(
                0,
                int(os.getenv("AYOA_IMAGE_MAX_REFERENCES", "4")),
            ),
            max_reference_bytes=max(
                1,
                int(os.getenv("AYOA_IMAGE_MAX_REFERENCE_BYTES", "20000000")),
            ),
            steps=max(
                1,
                int(os.getenv("AYOA_IMAGE_STEPS", default_steps)),
            ),
            guidance=float(os.getenv("AYOA_IMAGE_GUIDANCE", "4.0")),
        )


class ImageGenerationCoordinator:
    """Durable event-owned diffusion and independent delivery workers."""

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
        self._delivery_handlers: dict[
            ImageDeliveryKind,
            ImageDeliveryHandler,
        ] = {}
        self._runner: asyncio.Task[None] | None = None
        self._delivery_runner: asyncio.Task[None] | None = None
        self._ownership_runner: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._changed = asyncio.Condition()
        self._started = False
        self._closing = False
        self._current_job: ImageGenerationJob | None = None
        self._queue_lock_handle: Any = None
        self._queue_owner = False
        self._unavailable_logged = False

    @property
    def available(self) -> bool:
        return bool(self.worker.available)

    @property
    def supported_generation_modes(self) -> tuple[ImageGenerationMode, ...]:
        configured = getattr(
            self.worker,
            "supported_generation_modes",
            ("compose",),
        )
        return tuple(configured)

    def register_delivery_handler(
        self,
        delivery_kind: ImageDeliveryKind,
        handler: ImageDeliveryHandler,
    ) -> None:
        self._delivery_handlers[delivery_kind] = handler
        if self._started and self._delivery_runner is None:
            self._delivery_runner = asyncio.create_task(
                self._run_delivery_poll(),
                name="ayoa-image-delivery",
            )

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._closing = False
        self._reconcile_speculative_transactions()
        preflight = getattr(self.worker, "preflight", None)
        if self.available and callable(preflight) and not await preflight():
            self._log_unavailable("image generation preflight failed")
        if self.available:
            self._queue_owner = self._acquire_queue_owner()
            if self._queue_owner:
                self._activate_queue_owner()
            self._ownership_runner = asyncio.create_task(
                self._run_owner_election(),
                name="ayoa-image-owner-election",
            )
        else:
            self._log_unavailable("image generation unavailable")
        # Delivery is a separate worker and may drain artifacts created by
        # another process even when this process has no diffusion runtime.
        if self._delivery_handlers:
            self._delivery_runner = asyncio.create_task(
                self._run_delivery_poll(),
                name="ayoa-image-delivery",
            )
        self._wake.set()

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        tasks = (
            self._ownership_runner,
            self._runner,
            self._delivery_runner,
        )
        for task in tasks:
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in tasks if task is not None),
            return_exceptions=True,
        )
        self._ownership_runner = None
        self._runner = None
        self._delivery_runner = None
        await self.worker.close()
        self._release_queue_owner()
        self._started = False

    def begin_transaction(
        self,
        *,
        transaction_id: str,
        session_id: str,
        source_turn_index: int,
        source_checkpoint_sha256: str,
        lineage_bound: bool = True,
    ) -> None:
        self.store.begin_transaction(
            transaction_id=transaction_id,
            session_id=session_id,
            source_turn_index=source_turn_index,
            source_checkpoint_sha256=source_checkpoint_sha256,
            lineage_bound=lineage_bound,
        )

    async def commit_transaction(
        self,
        transaction_id: str,
        *,
        target_checkpoint_sha256: str,
    ) -> bool:
        committed = self.store.commit_transaction(
            transaction_id,
            target_checkpoint_sha256=target_checkpoint_sha256,
        )
        self._wake.set()
        await self._notify_changed()
        return committed

    async def cancel_transaction(
        self,
        transaction_id: str,
        *,
        reason: str = "transaction_aborted",
    ) -> int:
        cancelled = self.store.cancel_transaction(
            transaction_id,
            reason=reason,
        )
        current = self._current_job
        if (
            current is not None
            and current.request.transaction_id == transaction_id
        ):
            await self.worker.abort_current()
        await self._notify_changed()
        return cancelled

    async def enqueue_direction(
        self,
        *,
        projection: VisibleEventProjection,
        direction: ImageDirection,
        request_ordinal: int,
        visual_style: str,
        delivery_targets: Sequence[ImageDeliveryTarget],
        reroll_of_reference_id: str = "",
        diffusion_prompt_override: str = "",
    ) -> ImageGenerationJob | None:
        if not delivery_targets:
            return None
        _validate_direction_for_generation(
            projection=projection,
            direction=direction,
            max_subjects=self.config.max_subjects,
            max_scene_prompt_chars=self.config.max_scene_prompt_chars,
        )
        if direction.generation_mode not in self.supported_generation_modes:
            raise ValueError("requested image generation mode is unavailable")
        if self.store.active_count() >= self.config.queue_limit:
            logger.warning(
                "image generation queue is full; rejecting event request"
            )
            return None
        session_active = sum(
            job.request.session_id == projection.session_id
            and job.status
            in {
                ImageGenerationStatus.queued,
                ImageGenerationStatus.running,
            }
            for job in self.store.all_jobs()
        )
        if session_active >= self.config.per_session_queue_limit:
            logger.warning(
                "session image backlog is full; rejecting event request"
            )
            return None

        width, height = _dimensions_for_kind(direction.kind)
        references = self._resolve_references(
            projection=projection,
            direction=direction,
            exclude_generated_reference_id=reroll_of_reference_id,
        )
        prompt = (
            diffusion_prompt_override
            if diffusion_prompt_override
            else build_diffusion_prompt(
                projection=projection,
                direction=direction,
                visual_style=visual_style,
                max_scene_prompt_chars=self.config.max_scene_prompt_chars,
                max_style_chars=self.config.max_style_chars,
                style_trigger=self._worker_style_trigger(),
                reference_inputs=references,
            )
        )
        if not prompt.strip() or len(prompt) > 8_000:
            raise ValueError("diffusion prompt is empty or exceeds worker limit")
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        generation_identity = {
            "transaction_id": projection.transaction_id,
            "event_fingerprint": projection.event_fingerprint,
            "request_ordinal": request_ordinal,
            "kind": direction.kind,
            "title": direction.title,
            "subjects": direction.subject_character_ids,
            "prompt_sha256": prompt_sha256,
            "generation_mode": direction.generation_mode,
            "model_id": self._worker_model_id(direction.generation_mode),
            "model_revision": self._worker_model_revision(
                direction.generation_mode
            ),
            "width": width,
            "height": height,
            "steps": self.config.steps,
            "guidance": self.config.guidance,
            "references": [
                (reference.reference_id, reference.sha256)
                for reference in references
            ],
            "reroll_of_reference_id": reroll_of_reference_id,
        }
        dedupe_key = _stable_hash(generation_identity)
        seed = int.from_bytes(
            hashlib.sha256(
                json.dumps(
                    {
                        "event": projection.event_fingerprint,
                        "ordinal": request_ordinal,
                        "kind": direction.kind,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        ) & ((1 << 63) - 1)
        request = ImageGenerationRequest(
            session_id=projection.session_id,
            transaction_id=projection.transaction_id,
            source_event_id=projection.event_id,
            source_event_fingerprint=projection.event_fingerprint,
            source_event_sequence=projection.event_sequence,
            source_turn_index=projection.source_turn_index,
            request_ordinal=request_ordinal,
            kind=direction.kind,
            generation_mode=direction.generation_mode,
            title=direction.title,
            subject_character_ids=direction.subject_character_ids,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            model_id=self._worker_model_id(direction.generation_mode),
            model_revision=self._worker_model_revision(
                direction.generation_mode
            ),
            width=width,
            height=height,
            steps=self.config.steps,
            guidance=self.config.guidance,
            seed=seed,
            dedupe_key=dedupe_key,
            reference_inputs=references,
            reroll_of_reference_id=reroll_of_reference_id,
        )
        existing = self.store.get(f"img_{dedupe_key[:32]}")
        if existing is None:
            job = self.store.enqueue(request)
        else:
            job = existing
            if (
                job.status == ImageGenerationStatus.failed
                and job.attempts < 2
            ):
                job = self.store.requeue_retryable(job.job_id) or job
        for target in delivery_targets:
            self.store.add_delivery(
                job_id=job.job_id,
                session_id=projection.session_id,
                source_turn_index=projection.source_turn_index,
                pov_character_id=target.pov_character_id,
                delivery_kind=target.delivery_kind,
                delivery=target.delivery,
            )
        self._wake.set()
        await self._notify_changed()
        return job

    def open_prose_gates(
        self,
        *,
        transaction_id: str,
        rendered_event_ids_by_pov: dict[str, Sequence[str]],
    ) -> int:
        opened = 0
        by_event: dict[str, list[str]] = {}
        for pov_character_id, event_ids in rendered_event_ids_by_pov.items():
            for event_id in event_ids:
                by_event.setdefault(event_id, []).append(pov_character_id)
        for event_id, pov_ids in by_event.items():
            opened += self.store.open_prose_gate(
                transaction_id=transaction_id,
                source_event_id=event_id,
                pov_character_ids=pov_ids,
            )
        if opened:
            self._wake.set()
        return opened

    def open_prose_gates_for_session(
        self,
        *,
        session_id: str,
        rendered_event_ids_by_pov: dict[str, Sequence[str]],
    ) -> int:
        by_event: dict[str, list[str]] = {}
        for pov_character_id, event_ids in rendered_event_ids_by_pov.items():
            for event_id in event_ids:
                by_event.setdefault(event_id, []).append(pov_character_id)
        opened = sum(
            self.store.open_prose_gate_for_session_event(
                session_id=session_id,
                source_event_id=event_id,
                pov_character_ids=pov_ids,
            )
            for event_id, pov_ids in by_event.items()
        )
        if opened:
            self._wake.set()
        return opened

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
                    ImageGenerationStatus.failed,
                    ImageGenerationStatus.cancelled,
                }:
                    return job
                remaining = None if deadline is None else deadline - loop.time()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError
                try:
                    await asyncio.wait_for(
                        self._changed.wait(),
                        timeout=(
                            0.5
                            if remaining is None
                            else min(0.5, remaining)
                        ),
                    )
                except TimeoutError:
                    continue

    async def wait_for_rendered_event_images(
        self,
        *,
        session_id: str,
        rendered_event_ids_by_pov: dict[str, Sequence[str]],
        timeout: float | None = None,
        discovery_grace_seconds: float = 0.5,
    ) -> bool:
        """Wait briefly for first-pass event illustrations before prose posts."""

        if timeout is None:
            timeout = self.config.tandem_delivery_wait_seconds
        timeout = max(0.0, float(timeout))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        discovery_deadline = loop.time() + max(0.0, discovery_grace_seconds)
        saw_work = False

        while True:
            has_work, ready = self.store.rendered_event_image_status(
                session_id=session_id,
                rendered_event_ids_by_pov=rendered_event_ids_by_pov,
            )
            saw_work = saw_work or has_work
            if ready and (saw_work or loop.time() >= discovery_deadline):
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            async with self._changed:
                try:
                    await asyncio.wait_for(
                        self._changed.wait(),
                        timeout=min(0.2, remaining),
                    )
                except TimeoutError:
                    continue

    def resolve_job_media(
        self,
        job: ImageGenerationJob,
    ) -> ResolvedPlayerMedia:
        if job.artifact is None:
            raise PlayerMediaError("job_has_no_artifact")
        return resolve_generated_media(
            job.artifact,
            runtime_root=self.config.runtime_root,
        )

    def delivery_is_current(self, delivery_id: str) -> bool:
        return (
            self.store.delivery_is_current(delivery_id)
            and self.store.heartbeat_delivery(delivery_id)
        )

    async def cancel_after(self, session_id: str, turn_index: int) -> int:
        cancelled = self.store.cancel_after(session_id, turn_index)
        current = self._current_job
        if (
            current is not None
            and current.request.session_id == session_id
            and current.request.source_turn_index > turn_index
        ):
            await self.worker.abort_current()
        if cancelled:
            await self._notify_changed()
        return cancelled

    async def cancel_job(
        self,
        job_id: str,
        *,
        error_code: str = "cancelled",
    ) -> ImageGenerationJob | None:
        job = self.store.mark_cancelled(job_id, error_code)
        if (
            self._current_job is not None
            and self._current_job.job_id == job_id
        ):
            await self.worker.abort_current()
        await self._notify_changed()
        return job

    async def cancel_delivery(
        self,
        delivery_id: str,
    ) -> ImageDelivery | None:
        delivery = self.store.cancel_delivery(delivery_id)
        await self._notify_changed()
        return delivery

    async def cancel_session(self, session_id: str) -> int:
        cancelled = self.store.cancel_session(session_id)
        if (
            self._current_job is not None
            and self._current_job.request.session_id == session_id
        ):
            await self.worker.abort_current()
        if cancelled:
            await self._notify_changed()
        return cancelled

    def reconcile_lineage(
        self,
        *,
        session_id: str,
        canonical_event_fingerprints: dict[str, str],
    ) -> int:
        return self.store.reconcile_lineage(
            session_id=session_id,
            event_fingerprints=canonical_event_fingerprints.values(),
            event_ids=canonical_event_fingerprints.keys(),
        )

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
        if cancelled:
            await self._notify_changed()
        return cancelled

    def active_identity_character_ids(self, session_id: str) -> set[str]:
        return self.store.active_identity_character_ids(session_id)

    def active_reviewed_location_labels(self, session_id: str) -> set[str]:
        return self.store.active_reviewed_location_labels(session_id)

    def register_reviewed_visual_references(
        self,
        *,
        checkpoint: CheckpointFile,
        frozen_references: dict[str, FrozenReferenceInput],
    ) -> None:
        metadata = {
            item.reference_id: item
            for item in checkpoint.reviewed_visual_references
        }
        unknown_frozen = set(frozen_references) - set(metadata)
        if unknown_frozen:
            raise ValueError(
                "frozen reviewed references are absent from checkpoint registry"
            )
        selected_reviewed = {
            reference.reference_id
            for reference in checkpoint.reviewed_visual_references
            if reference.diffusion_authorized
            and reference.scope == "character"
            and reference.purpose == "identity"
        }
        selected_reviewed.update(
            reference_id
            for reference_ids in (
                checkpoint.location_visual_reference_ids.values()
            )
            for reference_id in reference_ids
        )
        missing_frozen = selected_reviewed - set(frozen_references)
        if missing_frozen:
            raise RuntimeError(
                "selected reviewed references have no frozen runtime input"
            )
        references = {
            reference_id: (
                frozen,
                metadata[reference_id].purpose,
                metadata[reference_id].scope,
            )
            for reference_id, frozen in frozen_references.items()
        }
        identity_bindings: dict[str, list[str]] = {}
        active_authored_characters = {
            character.character_id
            for character in checkpoint.characters
            if character.visuals.identity_reference_id
        }
        for reference in checkpoint.reviewed_visual_references:
            if (
                reference.reference_id in references
                and reference.scope == "character"
                and reference.purpose == "identity"
                and reference.scope_id in active_authored_characters
            ):
                identity_bindings.setdefault(reference.scope_id, []).append(
                    reference.reference_id
                )
        for character in checkpoint.characters:
            primary = character.visuals.identity_reference_id
            bindings = identity_bindings.get(character.character_id, [])
            if primary in bindings:
                bindings.insert(0, bindings.pop(bindings.index(primary)))
        location_bindings = {
            label: list(reference_ids)
            for label, reference_ids in (
                checkpoint.location_visual_reference_ids.items()
            )
        }
        self.store.replace_reviewed_references(
            session_id=checkpoint.session.session_id,
            references=references,
            identity_bindings=identity_bindings,
            location_bindings=location_bindings,
        )

    def active_identity_candidate(
        self,
        *,
        session_id: str,
        character_id: str,
    ) -> IdentityReferenceCandidate | None:
        return self.store.active_identity_candidate(
            session_id=session_id,
            character_id=character_id,
        )

    def lock_identity_candidate(
        self,
        *,
        session_id: str,
        candidate_id: str,
    ) -> IdentityReferenceCandidate:
        return self.store.lock_identity_candidate(
            session_id=session_id,
            candidate_id=candidate_id,
        )

    def retire_character_identity(
        self,
        *,
        session_id: str,
        character_id: str,
        source_turn_index: int,
    ) -> int:
        return self.store.retire_character_identity(
            session_id=session_id,
            character_id=character_id,
            source_turn_index=source_turn_index,
        )

    def suppress_reviewed_identity_binding(
        self,
        *,
        session_id: str,
        character_id: str,
    ) -> int:
        return self.store.suppress_reviewed_identity_binding(
            session_id=session_id,
            character_id=character_id,
        )

    def allow_character_identity_after(
        self,
        *,
        session_id: str,
        character_id: str,
        minimum_source_turn: int,
    ) -> None:
        self.store.allow_character_identity_after(
            session_id=session_id,
            character_id=character_id,
            minimum_source_turn=minimum_source_turn,
        )

    async def reroll_identity_reference(
        self,
        *,
        session_id: str,
        reference_id: str,
        delivery_targets: Sequence[ImageDeliveryTarget],
        checkpoint: CheckpointFile | None = None,
        source_checkpoint_sha256: str = "",
    ) -> ImageGenerationJob:
        candidate = self.store.get_identity_candidate(reference_id)
        manual_transaction_id = ""
        if candidate is not None:
            if candidate.session_id != session_id:
                raise KeyError(
                    f"unknown identity reference: {reference_id}"
                )
            if candidate.status == IdentityReferenceStatus.retired:
                raise ValueError(
                    "retired identity references cannot be rerolled"
                )
            source_job = self.store.get(candidate.job_id)
            if (
                source_job is None
                or source_job.status != ImageGenerationStatus.succeeded
            ):
                raise RuntimeError(
                    "identity reference source job is unavailable"
                )
            projection = _projection_from_request(source_job.request)
            direction = ImageDirection(
                kind="portrait",
                title=source_job.request.title,
                subject_character_ids=[candidate.character_id],
                scene_prompt=_director_scene_from_prompt(
                    source_job.request.prompt
                ),
            )
            visual_style = source_job.request.prompt.split("\n\n", 1)[0]
            diffusion_prompt_override = source_job.request.prompt
            reroll_ordinal = (
                source_job.request.request_ordinal
                + 1_000_000
                + int(time_monotonic_token())
            )
        else:
            reviewed = self.store.reviewed_identity_binding(
                session_id=session_id,
                reference_id=reference_id,
            )
            if reviewed is None:
                raise KeyError(
                    f"unknown identity reference: {reference_id}"
                )
            if checkpoint is None:
                raise ValueError(
                    "authored identity rerolls require current checkpoint"
                )
            if checkpoint.session.session_id != session_id:
                raise ValueError(
                    "authored identity reroll checkpoint/session mismatch"
                )
            if (
                len(source_checkpoint_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in source_checkpoint_sha256.lower()
                )
            ):
                raise ValueError(
                    "authored identity rerolls require checkpoint SHA-256"
                )
            character_id, _frozen = reviewed
            (
                projection,
                direction,
                visual_style,
            ) = _authored_identity_reroll_input(
                checkpoint=checkpoint,
                character_id=character_id,
                transaction_id=f"imgtx_{uuid.uuid4().hex}",
                max_scene_prompt_chars=self.config.max_scene_prompt_chars,
            )
            manual_transaction_id = projection.transaction_id
            self.begin_transaction(
                transaction_id=manual_transaction_id,
                session_id=session_id,
                source_turn_index=projection.source_turn_index,
                source_checkpoint_sha256=source_checkpoint_sha256,
                lineage_bound=False,
            )
            await self.commit_transaction(
                manual_transaction_id,
                target_checkpoint_sha256=source_checkpoint_sha256,
            )
            diffusion_prompt_override = ""
            reroll_ordinal = int(time_monotonic_token())

        try:
            job = await self.enqueue_direction(
                projection=projection,
                direction=direction,
                request_ordinal=reroll_ordinal,
                visual_style=visual_style,
                delivery_targets=delivery_targets,
                reroll_of_reference_id=reference_id,
                diffusion_prompt_override=diffusion_prompt_override,
            )
            if job is None:
                raise RuntimeError(
                    "identity reroll rejected by image capacity"
                )
        except Exception:
            if manual_transaction_id:
                await self.cancel_transaction(
                    manual_transaction_id,
                    reason="identity_reroll_enqueue_failed",
                )
            raise
        self.store.open_prose_gate(
            transaction_id=job.request.transaction_id,
            source_event_id=job.request.source_event_id,
            pov_character_ids=[
                target.pov_character_id for target in delivery_targets
            ],
        )
        return job

    async def _run(self) -> None:
        while not self._closing:
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

    async def _process(self, job: ImageGenerationJob) -> None:
        temp_dir = self.config.runtime_root / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{job.job_id}.webp"
        temp_path.unlink(missing_ok=True)
        try:
            self._revalidate_references(job.request.reference_inputs)
            result = await self._generate_with_cancellation(job, temp_path)
            current = self.store.get(job.job_id)
            if (
                current is None
                or current.status != ImageGenerationStatus.running
            ):
                temp_path.unlink(missing_ok=True)
                return
            artifact = finalize_generated_webp(
                temp_path,
                runtime_root=self.config.runtime_root,
                worker_result=result,
                expected_width=job.request.width,
                expected_height=job.request.height,
            )
            completed = self.store.mark_succeeded(job.job_id, artifact)
            if (
                completed is not None
                and completed.status == ImageGenerationStatus.succeeded
            ):
                self._establish_identity_if_applicable(completed)
                logger.info(
                    "illustration succeeded job=%s bytes=%d",
                    completed.job_id,
                    artifact.byte_count,
                )
            await self._notify_changed()
        except (ImageWorkerError, PlayerMediaError, ValueError) as exc:
            temp_path.unlink(missing_ok=True)
            code = getattr(exc, "code", "") or "invalid_reference_input"
            current = self.store.get(job.job_id)
            if (
                current is not None
                and current.status == ImageGenerationStatus.running
            ):
                self.store.mark_failed(job.job_id, str(code))
            logger.warning(
                "illustration failed job=%s code=%s",
                job.job_id,
                code,
            )
            await self._notify_changed()
        except Exception:
            temp_path.unlink(missing_ok=True)
            current = self.store.get(job.job_id)
            if (
                current is not None
                and current.status == ImageGenerationStatus.running
            ):
                self.store.mark_failed(job.job_id, "generation_failed")
            logger.exception(
                "unexpected illustration failure job=%s",
                job.job_id,
            )
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
            done, _ = await asyncio.wait(
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
            for task in (generation, cancellation):
                if not task.done():
                    task.cancel()

    async def _wait_for_external_cancellation(self, job_id: str) -> None:
        while True:
            current = self.store.get(job_id)
            if (
                current is None
                or current.status != ImageGenerationStatus.running
            ):
                return
            await asyncio.sleep(0.2)

    async def _run_delivery_poll(self) -> None:
        while not self._closing:
            try:
                self.store.recover_expired_deliveries()
                for kind in tuple(self._delivery_handlers):
                    while claim := self.store.claim_next_delivery(kind):
                        await self._dispatch_delivery(*claim)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("image delivery queue poll failed")
            await asyncio.sleep(0.5)

    async def _dispatch_delivery(
        self,
        delivery: ImageDelivery,
        job: ImageGenerationJob,
    ) -> None:
        handler = self._delivery_handlers.get(delivery.delivery_kind)
        if handler is None:
            self.store.release_delivery(delivery.delivery_id)
            return
        delivered = False
        candidate = self._candidate_for_job(job)
        instructions = ""
        if candidate is not None and candidate.reminder_required:
            instructions = (
                "Identity reference is provisional. Use "
                f"`/image lock id:{candidate.candidate_id}` or "
                f"`/image reroll id:{candidate.candidate_id}`."
            )
        try:
            delivered = await handler(
                job,
                delivery,
                self.resolve_job_media(job),
                instructions,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "generated illustration delivery failed delivery=%s",
                delivery.delivery_id,
            )
        finally:
            if self.store.delivery_is_current(delivery.delivery_id):
                if delivered:
                    self.store.mark_delivered(delivery.delivery_id)
                else:
                    self.store.release_delivery(delivery.delivery_id)
            await self._notify_changed()

    async def _run_owner_election(self) -> None:
        while not self._closing:
            self._reconcile_speculative_transactions()
            if not self._queue_owner and self._acquire_queue_owner():
                self._queue_owner = True
                self._activate_queue_owner()
                self._wake.set()
            elif (
                self._queue_owner
                and self._runner is not None
                and self._runner.done()
            ):
                logger.error("image queue runner stopped; restarting")
                self._runner = asyncio.create_task(
                    self._run(),
                    name="ayoa-image-generation",
                )
            await asyncio.sleep(1)

    def _reconcile_speculative_transactions(self) -> None:
        now = time.time()
        for transaction in self.store.speculative_transactions():
            transaction_id = str(transaction["transaction_id"])
            events = list(transaction["events"])
            committed = False
            if events:
                turns = {int(event[2]) for event in events}
                if len(turns) == 1:
                    target_turn = next(iter(turns))
                    target_path = (
                        self.sessions_dir
                        / str(transaction["session_id"])
                        / f"ckpt_{target_turn:04d}.json"
                    )
                    if target_path.is_file():
                        try:
                            checkpoint = CheckpointFile.model_validate_json(
                                target_path.read_text()
                            )
                            fingerprints = {
                                event.event_id: source_event_fingerprint(event)
                                for event in checkpoint.canonical_events
                            }
                            if all(
                                fingerprints.get(str(event_id))
                                == str(event_fingerprint)
                                for event_id, event_fingerprint, _ in events
                            ):
                                committed = self.store.commit_transaction(
                                    transaction_id,
                                    target_checkpoint_sha256=(
                                        _sha256_path(target_path)
                                    ),
                                )
                        except Exception:
                            logger.exception(
                                "failed to inspect speculative image lineage"
                            )
            if committed:
                self._wake.set()
                continue
            if now - float(transaction["updated_at"]) >= max(
                1.0,
                self.config.transaction_recovery_lease_seconds,
            ):
                self.store.cancel_transaction(
                    transaction_id,
                    reason="orphaned_speculative_transaction",
                )

    def _activate_queue_owner(self) -> None:
        recovered = self.store.recover_interrupted()
        if recovered:
            logger.info(
                "recovered %d interrupted image queue row(s)",
                recovered,
            )
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(
                self._run(),
                name="ayoa-image-generation",
            )

    def _resolve_references(
        self,
        *,
        projection: VisibleEventProjection,
        direction: ImageDirection,
        exclude_generated_reference_id: str = "",
    ) -> list[FrozenReferenceInput]:
        references: list[FrozenReferenceInput] = []
        if direction.reference_ids:
            allowed_ids = {
                reference.reference_id
                for reference in projection.reference_options
            }
            unavailable = set(direction.reference_ids) - allowed_ids
            if unavailable:
                raise ValueError("selected visual reference is unavailable")
            for reference_id in direction.reference_ids:
                reference = self.store.reviewed_reference(
                    session_id=projection.session_id,
                    reference_id=reference_id,
                )
                if reference is None:
                    raise RuntimeError(
                        "selected authored visual reference is unavailable"
                    )
                references.append(reference)
            self._validate_reference_limits(
                references,
                generation_mode=direction.generation_mode,
            )
            self._revalidate_references(references)
            return references
        missing_required_identities: list[str] = []
        public_by_id = {
            character.character_id: character
            for character in projection.characters
        }
        for character_id in direction.subject_character_ids:
            candidate = self.store.active_identity_candidate(
                session_id=projection.session_id,
                character_id=character_id,
            )
            excluded_current_generated = bool(
                candidate is not None
                and candidate.candidate_id
                == exclude_generated_reference_id
            )
            if (
                candidate is not None
                and candidate.candidate_id
                != exclude_generated_reference_id
            ):
                artifact = candidate.artifact
                references.append(
                    FrozenReferenceInput(
                        reference_id=candidate.candidate_id,
                        sha256=artifact.sha256,
                        mime_type=artifact.mime_type,
                        width=artifact.width,
                        height=artifact.height,
                        byte_count=artifact.byte_count,
                        relative_path=artifact.relative_path,
                        allowed_root="artifacts",
                    )
                )
                continue
            reviewed = self.store.reviewed_identity_reference(
                session_id=projection.session_id,
                character_id=character_id,
            )
            if reviewed is not None:
                references.append(reviewed)
                continue
            if excluded_current_generated:
                continue
            if public_by_id.get(character_id) is not None and (
                public_by_id[character_id].has_identity_reference
            ):
                missing_required_identities.append(character_id)

        if missing_required_identities:
            raise RuntimeError(
                "required identity references are unavailable for: "
                + ", ".join(missing_required_identities)
            )

        if (
            projection.has_location_reference
            and direction.kind in {"establishing", "action", "detail"}
        ):
            location_references = self.store.reviewed_location_references(
                session_id=projection.session_id,
                location_label=projection.engine_location_label,
            )
            if not location_references:
                raise RuntimeError(
                    "required authored location references are unavailable"
                )
            references.extend(location_references)

        references = list(
            {
                reference.reference_id: reference
                for reference in references
            }.values()
        )
        self._validate_reference_limits(
            references,
            generation_mode=direction.generation_mode,
        )
        self._revalidate_references(references)
        return references

    def _validate_reference_limits(
        self,
        references: Sequence[FrozenReferenceInput],
        *,
        generation_mode: ImageGenerationMode,
    ) -> None:
        if len(references) > self.config.max_references:
            raise ValueError(
                "required authored reference set exceeds configured limit"
            )
        if generation_mode == "edit" and not references:
            raise ValueError("edit generation requires a reference")
        if generation_mode == "edit" and len(references) > 3:
            raise ValueError("edit generation accepts at most 3 references")
        if sum(item.byte_count for item in references) > (
            self.config.max_reference_bytes
        ):
            raise ValueError("image references exceed configured byte limit")

    def _revalidate_references(
        self,
        references: Sequence[FrozenReferenceInput],
    ) -> None:
        if references and len(references) > self.config.max_references:
            raise ValueError("image reference count exceeds configured limit")
        total = 0
        runtime_root = self.config.runtime_root.resolve()
        for reference in references:
            if reference.allowed_root != "artifacts":
                raise ValueError("reference root is not authorized")
            path = (runtime_root / reference.relative_path).resolve()
            allowed = (runtime_root / "artifacts").resolve()
            if path != allowed and allowed not in path.parents:
                raise ValueError("reference path escapes authorized root")
            try:
                with path.open("rb") as handle:
                    data = handle.read(reference.byte_count + 1)
            except OSError as exc:
                raise ValueError("reference file is unavailable") from exc
            total += len(data)
            if len(data) != reference.byte_count:
                raise ValueError("reference byte count changed")
            if hashlib.sha256(data).hexdigest() != reference.sha256:
                raise ValueError("reference hash changed")
        if total > self.config.max_reference_bytes:
            raise ValueError("image references exceed configured byte limit")

    def _establish_identity_if_applicable(
        self,
        job: ImageGenerationJob,
    ) -> None:
        if (
            job.request.kind != "portrait"
            or len(job.request.subject_character_ids) != 1
        ):
            return
        character_id = job.request.subject_character_ids[0]
        if (
            character_id
            in self.store.active_identity_character_ids(
                job.request.session_id
            )
            and not job.request.reroll_of_reference_id
        ):
            return
        self.store.create_identity_candidate(
            job=job,
            character_id=character_id,
        )

    def _candidate_for_job(
        self,
        job: ImageGenerationJob,
    ) -> IdentityReferenceCandidate | None:
        if (
            job.request.kind != "portrait"
            or len(job.request.subject_character_ids) != 1
        ):
            return None
        candidate = self.store.active_identity_candidate(
            session_id=job.request.session_id,
            character_id=job.request.subject_character_ids[0],
        )
        return (
            candidate
            if candidate is not None and candidate.job_id == job.job_id
            else None
        )

    def _worker_model_id(self, generation_mode: ImageGenerationMode) -> str:
        config = getattr(self.worker, "config", None)
        resolver = getattr(config, "model_id_for", None)
        if callable(resolver):
            return str(resolver(generation_mode))
        return str(
            getattr(config, "model_id", "")
            or "test-local-image-model"
        )

    def _worker_model_revision(
        self,
        generation_mode: ImageGenerationMode,
    ) -> str:
        config = getattr(self.worker, "config", None)
        resolver = getattr(config, "model_revision_for", None)
        if callable(resolver):
            return str(resolver(generation_mode))
        return str(
            getattr(config, "runtime_revision", "")
            or getattr(config, "model_revision", "")
            or "test-revision"
        )

    def _worker_style_trigger(self) -> str:
        config = getattr(self.worker, "config", None)
        return _bounded_text(
            getattr(config, "style_trigger", ""),
            100,
        )

    def _log_unavailable(self, message: str) -> None:
        if not self._unavailable_logged:
            logger.info("%s; text play remains enabled", message)
            self._unavailable_logged = True

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
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    async def _notify_changed(self) -> None:
        async with self._changed:
            self._changed.notify_all()


def build_diffusion_prompt(
    *,
    projection: VisibleEventProjection,
    direction: ImageDirection,
    visual_style: str,
    max_scene_prompt_chars: int,
    max_style_chars: int,
    style_trigger: str = "",
    reference_inputs: Sequence[FrozenReferenceInput] = (),
) -> str:
    style = _bounded_text(visual_style, max_style_chars)
    trigger = _bounded_text(style_trigger, 100)
    scene = _bounded_text(
        direction.scene_prompt,
        max_scene_prompt_chars,
    )
    if not scene:
        raise ValueError("director scene prompt is empty")
    by_id = {
        character.character_id: character
        for character in projection.characters
    }
    continuity = [
        _character_continuity_line(by_id[character_id])
        for character_id in direction.subject_character_ids
        if character_id in by_id
    ]
    composition = {
        "portrait": (
            "single subject, three-quarter full-body portrait, head and boots visible"
        ),
        "group_portrait": (
            "group composition, exact subject count, each person separated and readable"
        ),
        "action": (
            "dynamic action, coherent physical staging, clear foreground depth"
        ),
        "establishing": (
            "wide establishing scene, distinctive architecture and material scale"
        ),
        "detail": (
            "focused in-world detail, restrained surrounding context"
        ),
    }[direction.kind]
    style_direction = " ".join(
        part
        for part in (
            trigger,
            style or "Story-consistent narrative illustration.",
        )
        if part
    )
    parts = [scene]
    reference_lines = _reference_binding_lines(
        direction=direction,
        references=reference_inputs,
        by_id=by_id,
        reference_options={
            option.reference_id: option
            for option in projection.reference_options
        },
    )
    if reference_lines:
        parts.append(
            "Reference binding: "
            + " ".join(reference_lines)
            + " Preserve each referenced identity exactly; reference identity, "
            "proportions, face, hair, clothing, prop, and silhouette override "
            "general style text. Do not transfer visual traits, anatomy, "
            "costume pieces, props, scale, wings, hats, uniforms, weapons, "
            "or proportions from one referenced subject to another."
        )
    if continuity:
        parts.append("Identity: " + " ".join(continuity))
    parts.append(f"Style and composition: {style_direction}; {composition}.")
    parts.append(
        "Purely pictorial in-world image, no readable text, letters, labels, "
        "UI, HUD, buttons, minimap, dialogue box, card frame, or watermark; "
        "light effects are abstract nonlinguistic geometry."
    )
    return "\n\n".join(parts)


def _reference_binding_lines(
    *,
    direction: ImageDirection,
    references: Sequence[FrozenReferenceInput],
    by_id: dict[str, PublicCharacterVisual],
    reference_options: dict[str, Any],
) -> list[str]:
    if not references:
        return []
    described: list[str] = []
    for index, reference in enumerate(references, start=1):
        option = reference_options.get(reference.reference_id)
        if option is None:
            break
        if option.scope == "character":
            character = by_id.get(option.scope_id)
            name = character.name if character is not None else option.scope_id
            described.append(
                f"Reference image {index} is {name} ({option.scope_id})."
            )
        else:
            described.append(
                f"Reference image {index} is the visible location guide."
            )
    if len(described) == len(references):
        return described

    subject_ids = list(direction.subject_character_ids)
    if len(references) == len(subject_ids):
        return [
            (
                f"Reference image {index} is "
                f"{by_id.get(character_id).name if by_id.get(character_id) else character_id} "
                f"({character_id})."
            )
            for index, character_id in enumerate(subject_ids, start=1)
        ]
    return [
        f"Reference image {index} is {reference.reference_id}."
        for index, reference in enumerate(references, start=1)
    ]


def _validate_direction_for_generation(
    *,
    projection: VisibleEventProjection,
    direction: ImageDirection,
    max_subjects: int,
    max_scene_prompt_chars: int,
) -> None:
    if len(direction.scene_prompt) > max_scene_prompt_chars:
        raise ValueError("director scene prompt exceeds configured limit")
    if len(direction.subject_character_ids) > max_subjects:
        raise ValueError("director subject count exceeds configured limit")
    allowed = {
        character.character_id
        for character in projection.characters
        if character.depiction_policy == "normal"
    }
    if any(
        character_id not in allowed
        for character_id in direction.subject_character_ids
    ):
        raise ValueError(
            "image direction contains unavailable or non-depictable subjects"
        )
    if any(
        character.depiction_policy != "normal"
        and text_names_public_character(
            direction.scene_prompt,
            character,
        )
        for character in projection.characters
    ):
        raise ValueError(
            "image direction names an anonymous or omitted character"
        )
    references = {
        reference.reference_id: reference
        for reference in projection.reference_options
    }
    unknown_references = set(direction.reference_ids) - set(references)
    if unknown_references:
        raise ValueError("image direction contains unavailable references")
    if direction.generation_mode == "edit" and not direction.reference_ids:
        raise ValueError("edit generation requires a selected reference")
    if direction.generation_mode == "edit" and len(direction.reference_ids) > 3:
        raise ValueError("edit generation accepts at most 3 references")
    if any(
        references[reference_id].scope == "character"
        and references[reference_id].scope_id
        not in direction.subject_character_ids
        for reference_id in direction.reference_ids
    ):
        raise ValueError("selected identity reference belongs to another subject")
    if direction.kind == "portrait" and len(
        direction.subject_character_ids
    ) != 1:
        raise ValueError("portrait requests require exactly one subject")
    if direction.kind == "group_portrait" and len(
        direction.subject_character_ids
    ) < 2:
        raise ValueError(
            "group portrait requests require at least two subjects"
        )


def _character_continuity_line(
    character: PublicCharacterVisual,
) -> str:
    parts = [character.name or character.character_id]
    if character.appearance:
        parts.append(character.appearance)
    if character.default_loadout:
        parts.append(character.default_loadout)
    return "- " + "; ".join(parts)


def _dimensions_for_kind(kind: str) -> tuple[int, int]:
    if kind == "portrait":
        return 768, 1024
    if kind in {"group_portrait", "action", "establishing"}:
        return 1024, 768
    return 768, 768


def _bounded_text(value: object, max_chars: int) -> str:
    return " ".join(str(value or "").split())[:max_chars].rstrip()


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authored_identity_reroll_input(
    *,
    checkpoint: CheckpointFile,
    character_id: str,
    transaction_id: str,
    max_scene_prompt_chars: int,
) -> tuple[VisibleEventProjection, ImageDirection, str]:
    character = next(
        (
            item
            for item in checkpoint.characters
            if item.character_id == character_id
        ),
        None,
    )
    if character is None:
        raise KeyError(f"unknown identity character: {character_id}")
    name = _bounded_text(character.name or character_id, 200)
    appearance = _bounded_text(
        character.public_sheet.appearance,
        600,
    )
    loadout = _bounded_text(character.visuals.default_loadout, 700)
    scene = _bounded_text(
        "; ".join(
            part
            for part in (
                f"Individual portrait of {name}",
                appearance,
                loadout,
            )
            if part
        ),
        max_scene_prompt_chars,
    )
    setting = checkpoint.world_state.setting
    event_id = f"manual_identity_reroll_{transaction_id.removeprefix('imgtx_')}"
    projection = VisibleEventProjection(
        session_id=checkpoint.session.session_id,
        transaction_id=transaction_id,
        source_turn_index=checkpoint.session.turn_index,
        event_id=event_id,
        event_sequence=len(checkpoint.canonical_events),
        event_fingerprint=_stable_hash(
            {
                "event_id": event_id,
                "turn_index": checkpoint.session.turn_index,
            }
        ),
        viewer_character_ids=tuple(
            item
            for item in dict.fromkeys(
                (
                    *checkpoint.session.character_bindings.keys(),
                    checkpoint.session.player_character_id,
                )
            )
            if item
        ),
        perception_level="direct",
        effective_at_s=max(0, checkpoint.session.leading_at_s),
        duration_s=0,
        visible_facts=((scene, 0, 0),),
        characters=(
            PublicCharacterVisual(
                character_id=character.character_id,
                name=name,
                appearance=appearance,
                default_loadout=loadout,
                depiction_policy=character.visuals.depiction_policy,
                is_new_character=False,
                has_identity_reference=True,
                public_role=_bounded_text(
                    character.public_sheet.role,
                    300,
                ),
                is_playable=character.is_playable,
                recurring_actor=(
                    character.private_state.intentions_enabled
                    and not character.is_playable
                ),
            ),
        ),
        story_genre=_bounded_text(setting.genre, 300),
        story_era=_bounded_text(setting.era, 300),
        story_tone=_bounded_text(setting.tone, 300),
        story_premise=_bounded_text(setting.premise, 1_000),
        canonical_event_count=len(checkpoint.canonical_events),
        active_roster_count=sum(
            item.status.value == "active"
            for item in checkpoint.characters
        ),
        total_roster_count=sum(
            item.status.value != "culled"
            for item in checkpoint.characters
        ),
        engine_visual_style=_bounded_text(setting.visual_style, 800),
    )
    return (
        projection,
        ImageDirection(
            kind="portrait",
            title=_bounded_text(f"{name} Portrait", 80),
            subject_character_ids=[character_id],
            scene_prompt=scene,
        ),
        setting.visual_style,
    )


def _projection_from_request(
    request: ImageGenerationRequest,
) -> VisibleEventProjection:
    # Rerolls intentionally preserve the already-authored diffusion prompt;
    # this minimal projection carries only immutable provenance.
    return VisibleEventProjection(
        session_id=request.session_id,
        transaction_id=request.transaction_id,
        source_turn_index=request.source_turn_index,
        event_id=request.source_event_id,
        event_sequence=request.source_event_sequence,
        event_fingerprint=request.source_event_fingerprint,
        viewer_character_ids=(),
        perception_level="direct",
        effective_at_s=0,
        duration_s=0,
        visible_facts=(),
        characters=tuple(
            PublicCharacterVisual(
                character_id=character_id,
                name=character_id,
                appearance="",
                default_loadout="",
                depiction_policy="normal",
                is_new_character=False,
                has_identity_reference=True,
            )
            for character_id in request.subject_character_ids
        ),
        story_genre="",
        story_era="",
        story_tone="",
        story_premise="",
        canonical_event_count=request.source_event_sequence + 1,
        active_roster_count=0,
        total_roster_count=0,
    )


def _director_scene_from_prompt(prompt: str) -> str:
    marker = "Visible scene:"
    if marker not in prompt:
        return _bounded_text(prompt, 2_000)
    value = prompt.split(marker, 1)[1].split("\n\n", 1)[0]
    return _bounded_text(value, 2_000)


def time_monotonic_token() -> int:
    return int(asyncio.get_running_loop().time() * 1_000_000)
