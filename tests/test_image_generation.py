from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.engine.image_generation import (
    ImageGenerationConfig,
    ImageGenerationCoordinator,
    build_image_prompt,
    image_turn_is_eligible,
)
from app.engine.image_worker_client import ImageWorkerError
from app.schemas.characters import (
    CharacterRecord,
    CharacterVisuals,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.image_generation import (
    ImageDeliveryKind,
    ImageGenerationStatus,
    ImageTriggerKind,
    ImageWorkerResult,
)
from app.schemas.responses import TurnResponse
from app.schemas.state import SessionState, StorySetting, WorldState


class FakeImageWorker:
    def __init__(self, *, wait: bool = False, error_code: str = "") -> None:
        self.available = True
        self.config = SimpleNamespace(
            model_id="fake/flux-klein",
            model_revision="test-revision",
        )
        self.wait = wait
        self.error_code = error_code
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.aborted = False
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def generate(self, request, *, output_path):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.wait:
                await self.release.wait()
            if self.aborted:
                raise ImageWorkerError("worker_cancelled")
            if self.error_code:
                raise ImageWorkerError(self.error_code)
            data = _fake_webp(request.width, request.height)
            Path(output_path).write_bytes(data)
            return ImageWorkerResult(
                ok=True,
                sha256=hashlib.sha256(data).hexdigest(),
                mime_type="image/webp",
                width=request.width,
                height=request.height,
                byte_count=len(data),
                generation_seconds=0.01,
            )
        finally:
            self.active -= 1

    async def abort_current(self):
        self.aborted = True
        self.release.set()

    async def close(self):
        self.release.set()


def _checkpoint(session_id: str = "image_test", *, turn_index: int = 1):
    ckpt = CheckpointFile(
        session=SessionState(session_id=session_id, turn_index=turn_index),
        world_state=WorldState(
            setting=StorySetting(
                genre="rain-washed campus romance",
                era="contemporary",
                tone="earnest",
                visual_style="soft anime-inspired cinematic illustration",
            ),
            hidden_lore="PRIVATE WORLD SECRET",
        ),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                public_sheet=PublicSheet(
                    appearance="short dark hair and a yellow raincoat",
                ),
                visuals=CharacterVisuals(
                    default_loadout="canvas satchel and rain-spotted shoes",
                ),
                private_state=PrivateState(secrets=["PRIVATE ALICE SECRET"]),
            ),
            CharacterRecord(
                character_id="bob",
                name="Bob",
                public_sheet=PublicSheet(appearance="silver hair"),
                visuals=CharacterVisuals(default_loadout="black formal coat"),
                private_state=PrivateState(secrets=["PRIVATE BOB SECRET"]),
            ),
        ],
    )
    ckpt.session.config.narrative_rules = "PRIVATE NARRATIVE RULE"
    return ckpt


def _response(
    session_id: str = "image_test",
    *,
    turn_index: int = 1,
    prose: str = "Alice pauses beneath the station awning as rain brightens the street.",
    reason: str = "cascade_exhausted",
):
    return TurnResponse(
        session_id=session_id,
        checkpoint_id=f"ckpt_{turn_index:04d}",
        turn_index=turn_index,
        output_text=prose,
        per_player_renders={
            "alice": prose,
            "bob": "Bob sees a private and completely different scene.",
        },
        beat_ended_reason=reason,
    )


def _write_checkpoint(root: Path, ckpt: CheckpointFile) -> None:
    session_dir = root / ckpt.session.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"ckpt_{ckpt.session.turn_index:04d}.json"
    path.write_text(ckpt.model_dump_json(indent=2))


def _config(tmp_path: Path) -> ImageGenerationConfig:
    return ImageGenerationConfig(
        runtime_root=tmp_path / "runtime",
        width=256,
        height=256,
        queue_limit=4,
        cli_wait_timeout_seconds=2,
    )


def _fake_webp(width: int, height: int) -> bytes:
    encoded_width = width - 1
    encoded_height = height - 1
    payload = b"\x2f" + bytes(
        (
            encoded_width & 0xFF,
            ((encoded_width >> 8) & 0x3F) | ((encoded_height & 0x03) << 6),
            (encoded_height >> 2) & 0xFF,
            (encoded_height >> 10) & 0x0F,
        )
    )
    chunk = b"VP8L" + len(payload).to_bytes(4, "little") + payload + b"\x00"
    body = b"WEBP" + chunk
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def test_prompt_uses_only_actor_prose_and_public_visual_context(tmp_path):
    ckpt = _checkpoint()
    config = _config(tmp_path)

    prompt = build_image_prompt(
        ckpt,
        actor_character_id="alice",
        prose=(
            "Alice waits under /private/module/source-map.png while rain falls."
        ),
        config=config,
    )

    assert "Alice waits" in prompt
    assert "yellow raincoat" in prompt
    assert "soft anime-inspired" in prompt
    assert "silver hair" not in prompt
    assert "PRIVATE" not in prompt
    assert "/private/module" not in prompt
    assert "Bob sees" not in prompt


@pytest.mark.parametrize(
    ("trigger", "reason", "expected"),
    [
        (ImageTriggerKind.act, "cascade_exhausted", True),
        (ImageTriggerKind.query, "query_response", False),
        (ImageTriggerKind.act, "cat_ii_pending", False),
        (ImageTriggerKind.roll_resolution, "cat_ii_resolution", True),
        (ImageTriggerKind.render_retry, "cascade_exhausted", True),
    ],
)
def test_image_turn_eligibility_is_positive_and_terminal(trigger, reason, expected):
    assert image_turn_is_eligible(
        ckpt=_checkpoint(),
        response=_response(reason=reason),
        actor_character_id="alice",
        trigger_kind=trigger,
    ) is expected


def test_session_mode_disables_generation_before_cadence():
    ckpt = _checkpoint(turn_index=2)
    response = _response(turn_index=2)
    ckpt.session.config.settings.image_generation_mode = "off"
    assert not image_turn_is_eligible(
        ckpt=ckpt,
        response=response,
        actor_character_id="alice",
        trigger_kind=ImageTriggerKind.act,
    )
    ckpt.session.config.settings.image_generation_mode = "actor"
    ckpt.session.config.settings.image_generation_every_n_beats = 3
    assert image_turn_is_eligible(
        ckpt=ckpt,
        response=response,
        actor_character_id="alice",
        trigger_kind=ImageTriggerKind.act,
    )


@pytest.mark.asyncio
async def test_cadence_counts_only_eligible_completed_beats(tmp_path):
    sessions = tmp_path / "sessions"
    worker = FakeImageWorker()
    coordinator = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=_config(tmp_path),
        worker=worker,
    )
    try:
        queued = []
        for turn in (1, 2, 3):
            ckpt = _checkpoint(turn_index=turn)
            ckpt.session.config.settings.image_generation_every_n_beats = 3
            _write_checkpoint(sessions, ckpt)
            queued.append(
                await coordinator.enqueue_turn(
                    ckpt=ckpt,
                    response=_response(
                        turn_index=turn,
                        prose=f"Alice crosses courtyard number {turn}.",
                    ),
                    actor_character_id="alice",
                    trigger_kind=ImageTriggerKind.act,
                    delivery_kind=ImageDeliveryKind.cli,
                )
            )
        assert queued[0] is None
        assert queued[1] is None
        assert queued[2] is not None
        completed = await coordinator.wait_for_terminal(
            queued[2].job_id,
            timeout=2,
        )
        assert completed is not None
        assert completed.status == ImageGenerationStatus.succeeded
        assert worker.calls == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_coordinator_generates_exactly_once_and_reuses_artifact(tmp_path):
    sessions = tmp_path / "sessions"
    ckpt = _checkpoint()
    _write_checkpoint(sessions, ckpt)
    worker = FakeImageWorker()
    coordinator = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=_config(tmp_path),
        worker=worker,
    )
    try:
        first = await coordinator.enqueue_turn(
            ckpt=ckpt,
            response=_response(),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert first is not None
        completed = await coordinator.wait_for_terminal(first.job_id, timeout=2)
        assert completed is not None
        assert completed.status == ImageGenerationStatus.succeeded
        media = coordinator.resolve_job_media(completed)
        assert media.width == 256
        assert media.height == 256

        repeated = await coordinator.enqueue_turn(
            ckpt=ckpt,
            response=_response(),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert repeated is not None
        assert repeated.job_id == first.job_id
        assert worker.calls == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_bounded_queue_skips_new_auto_job_when_full(tmp_path):
    sessions = tmp_path / "sessions"
    worker = FakeImageWorker(wait=True)
    config = _config(tmp_path)
    config = ImageGenerationConfig(
        **{**config.__dict__, "queue_limit": 1},
    )
    coordinator = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=config,
        worker=worker,
    )
    ckpt1 = _checkpoint(turn_index=1)
    ckpt2 = _checkpoint(turn_index=2)
    _write_checkpoint(sessions, ckpt1)
    _write_checkpoint(sessions, ckpt2)
    try:
        first = await coordinator.enqueue_turn(
            ckpt=ckpt1,
            response=_response(turn_index=1),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert first is not None
        await asyncio.wait_for(worker.started.wait(), timeout=1)
        second = await coordinator.enqueue_turn(
            ckpt=ckpt2,
            response=_response(
                turn_index=2,
                prose="Alice enters the empty rehearsal room.",
            ),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert second is None
    finally:
        worker.release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_worker_is_globally_serial_and_queue_is_fifo(tmp_path):
    sessions = tmp_path / "sessions"
    worker = FakeImageWorker(wait=True)
    coordinator = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=_config(tmp_path),
        worker=worker,
    )
    ckpt1 = _checkpoint(turn_index=1)
    ckpt2 = _checkpoint(turn_index=2)
    _write_checkpoint(sessions, ckpt1)
    _write_checkpoint(sessions, ckpt2)
    try:
        first = await coordinator.enqueue_turn(
            ckpt=ckpt1,
            response=_response(turn_index=1),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        second = await coordinator.enqueue_turn(
            ckpt=ckpt2,
            response=_response(
                turn_index=2,
                prose="Alice crosses the shining courtyard under the rain.",
            ),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert first is not None and second is not None
        await asyncio.wait_for(worker.started.wait(), timeout=1)
        worker.release.set()
        await coordinator.wait_for_terminal(first.job_id, timeout=2)
        await coordinator.wait_for_terminal(second.job_id, timeout=2)
        assert worker.calls == 2
        assert worker.max_active == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_second_process_enqueues_without_recovering_or_claiming_owner_job(
    tmp_path,
):
    sessions = tmp_path / "sessions"
    ckpt = _checkpoint()
    _write_checkpoint(sessions, ckpt)
    config = _config(tmp_path)
    owner_worker = FakeImageWorker()
    observer_worker = FakeImageWorker()
    owner = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=config,
        worker=owner_worker,
    )
    observer = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=config,
        worker=observer_worker,
    )
    try:
        await owner.start()
        await observer.start()
        job = await observer.enqueue_turn(
            ckpt=ckpt,
            response=_response(),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert job is not None
        completed = await observer.wait_for_terminal(job.job_id, timeout=7)
        assert completed is not None
        assert completed.status == ImageGenerationStatus.succeeded
        assert owner_worker.calls == 1
        assert observer_worker.calls == 0
    finally:
        await observer.close()
        await owner.close()


@pytest.mark.asyncio
async def test_observer_takes_queue_ownership_after_owner_exits(tmp_path):
    sessions = tmp_path / "sessions"
    ckpt = _checkpoint()
    _write_checkpoint(sessions, ckpt)
    config = _config(tmp_path)
    owner = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=config,
        worker=FakeImageWorker(),
    )
    successor_worker = FakeImageWorker()
    successor = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=config,
        worker=successor_worker,
    )
    await owner.start()
    await successor.start()
    await owner.close()
    try:
        await asyncio.sleep(1.2)
        job = await successor.enqueue_turn(
            ckpt=ckpt,
            response=_response(),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert job is not None
        completed = await successor.wait_for_terminal(job.job_id, timeout=3)
        assert completed is not None
        assert completed.status == ImageGenerationStatus.succeeded
        assert successor_worker.calls == 1
    finally:
        await successor.close()


@pytest.mark.asyncio
async def test_cross_process_rewind_aborts_owner_inference(tmp_path):
    sessions = tmp_path / "sessions"
    ckpt = _checkpoint(turn_index=2)
    _write_checkpoint(sessions, ckpt)
    config = _config(tmp_path)
    owner_worker = FakeImageWorker(wait=True)
    owner = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=config,
        worker=owner_worker,
    )
    observer = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=config,
        worker=FakeImageWorker(),
    )
    try:
        await owner.start()
        await observer.start()
        job = await observer.enqueue_turn(
            ckpt=ckpt,
            response=_response(turn_index=2),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert job is not None
        await asyncio.wait_for(owner_worker.started.wait(), timeout=6)
        assert await observer.cancel_after("image_test", 1) == 1
        cancelled = await observer.wait_for_terminal(job.job_id, timeout=3)
        assert cancelled is not None
        assert cancelled.status == ImageGenerationStatus.cancelled
        await asyncio.sleep(0.4)
        assert owner_worker.aborted is True
    finally:
        await observer.close()
        await owner.close()


@pytest.mark.asyncio
async def test_checkpoint_hash_change_suppresses_stale_completion(tmp_path):
    sessions = tmp_path / "sessions"
    ckpt = _checkpoint(turn_index=2)
    _write_checkpoint(sessions, ckpt)
    worker = FakeImageWorker(wait=True)
    coordinator = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=_config(tmp_path),
        worker=worker,
    )
    try:
        job = await coordinator.enqueue_turn(
            ckpt=ckpt,
            response=_response(turn_index=2),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert job is not None
        await asyncio.wait_for(worker.started.wait(), timeout=1)
        path = sessions / "image_test" / "ckpt_0002.json"
        path.write_text(path.read_text() + "\n")
        worker.release.set()
        completed = await coordinator.wait_for_terminal(job.job_id, timeout=2)
        assert completed is not None
        assert completed.status == ImageGenerationStatus.cancelled
        assert completed.error_code == "stale_checkpoint"
        assert completed.artifact is None
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_restart_requeues_interrupted_running_job(tmp_path):
    sessions = tmp_path / "sessions"
    ckpt = _checkpoint()
    _write_checkpoint(sessions, ckpt)
    config = _config(tmp_path)
    first_worker = FakeImageWorker(wait=True)
    first_coordinator = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=config,
        worker=first_worker,
    )
    job = await first_coordinator.enqueue_turn(
        ckpt=ckpt,
        response=_response(),
        actor_character_id="alice",
        trigger_kind=ImageTriggerKind.act,
        delivery_kind=ImageDeliveryKind.cli,
    )
    assert job is not None
    await asyncio.wait_for(first_worker.started.wait(), timeout=1)
    await first_coordinator.close()
    interrupted = first_coordinator.store.get(job.job_id)
    assert interrupted is not None
    assert interrupted.status == ImageGenerationStatus.running

    second_worker = FakeImageWorker()
    second_coordinator = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=config,
        worker=second_worker,
    )
    try:
        await second_coordinator.start()
        completed = await second_coordinator.wait_for_terminal(job.job_id, timeout=2)
        assert completed is not None
        assert completed.status == ImageGenerationStatus.succeeded
        assert second_worker.calls == 1
    finally:
        await second_coordinator.close()


@pytest.mark.asyncio
async def test_rewind_cancels_running_job_and_prevents_artifact(tmp_path):
    sessions = tmp_path / "sessions"
    ckpt = _checkpoint(turn_index=2)
    _write_checkpoint(sessions, ckpt)
    worker = FakeImageWorker(wait=True)
    coordinator = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=_config(tmp_path),
        worker=worker,
    )
    try:
        job = await coordinator.enqueue_turn(
            ckpt=ckpt,
            response=_response(turn_index=2),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert job is not None
        await asyncio.wait_for(worker.started.wait(), timeout=1)
        assert await coordinator.cancel_after("image_test", 1) == 1
        cancelled = await coordinator.wait_for_terminal(job.job_id, timeout=2)
        assert cancelled is not None
        assert cancelled.status == ImageGenerationStatus.cancelled
        assert cancelled.artifact is None
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_worker_error_is_typed_and_does_not_raise_to_caller(tmp_path):
    sessions = tmp_path / "sessions"
    ckpt = _checkpoint()
    _write_checkpoint(sessions, ckpt)
    worker = FakeImageWorker(error_code="cuda_oom")
    coordinator = ImageGenerationCoordinator(
        sessions_dir=sessions,
        config=_config(tmp_path),
        worker=worker,
    )
    try:
        job = await coordinator.enqueue_turn(
            ckpt=ckpt,
            response=_response(),
            actor_character_id="alice",
            trigger_kind=ImageTriggerKind.act,
            delivery_kind=ImageDeliveryKind.cli,
        )
        assert job is not None
        failed = await coordinator.wait_for_terminal(job.job_id, timeout=2)
        assert failed is not None
        assert failed.status == ImageGenerationStatus.failed
        assert failed.error_code == "cuda_oom"
    finally:
        await coordinator.close()
