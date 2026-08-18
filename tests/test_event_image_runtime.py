from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.engine.closed_event_runtime import (
    ClosedEventRuntime,
    install_closed_event_runtime,
)
from app.engine.event_image_sidecar import (
    EventImageSidecar,
    EventImageSidecarConfig,
)
from app.engine.image_generation import (
    ImageGenerationConfig,
    ImageGenerationCoordinator,
)
from app.engine.orchestrator import Orchestrator
from app.engine.spawn_authoring import SpawnAuthoringCoordinator
from app.engine.turn_loop import broadcast_event, run_beat
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.image_director import ImageDirectorOutput
from app.schemas.image_generation import ImageDeliveryKind
from app.schemas.event_router import EventRouterOutput
from tests.support.factories import (
    InstanceFakeDispatcher,
    checkpoint,
    router_output,
)


class BlockingCharacterManager:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.seen_premise = ""
        self.seen_roles: list[str] = []

    async def spawn_characters(
        self,
        checkpoint,
        requests,
        *,
        acting_actor_location,
    ):
        self.calls += 1
        self.seen_premise = checkpoint.world_state.setting.premise
        self.seen_roles = [request.seed.role for request in requests]
        self.started.set()
        await self.release.wait()
        return [
            CharacterRecord(
                character_id=request.character_id,
                name=request.character_id.title(),
                public_sheet=PublicSheet(
                    role=request.seed.role,
                    appearance=f"{request.seed.role} in practical travel clothes",
                ),
            )
            for request in requests
        ]


class RecordingSink:
    def __init__(self) -> None:
        self.calls = []

    def on_closed_event(self, **kwargs) -> None:
        self.calls.append(kwargs)


class BlockingDirector:
    def __init__(self) -> None:
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def decide(self, projection, *, recent_illustrations=()):
        self.calls.append(projection)
        self.started.set()
        await self.release.wait()
        return ImageDirectorOutput(requests=[])


class BlockingNarratorDispatcher(InstanceFakeDispatcher):
    def __init__(self) -> None:
        super().__init__()
        self.narrator_started = asyncio.Event()
        self.narrator_release = asyncio.Event()

    async def narrator_compose(self, **kwargs):
        self.narrator_started.set()
        await self.narrator_release.wait()
        return await super().narrator_compose(**kwargs)


class UnavailableWorker:
    available = False
    config = SimpleNamespace(model_id="none", model_revision="none")

    async def close(self):
        return None

    async def abort_current(self):
        return None


def _spawn_event():
    return router_output(
        event_id="evt_spawn",
        observer_ids=["alice", "bob"],
        spawn=[
            {
                "character_id": "guide",
                "seed": {
                    "role": "guide",
                    "reason": "The travelers need local help.",
                    "location": "station",
                    "objectives": ["lead them through the storm"],
                },
            },
            {
                "character_id": "porter",
                "seed": {
                    "role": "porter",
                    "reason": "The station is busy.",
                    "location": "station",
                    "objectives": ["move the luggage"],
                },
            },
        ],
    )


def test_event_router_contract_has_no_image_responsibility():
    assert not {
        field_name
        for field_name in EventRouterOutput.model_fields
        if "image" in field_name or "illustration" in field_name
    }


@pytest.mark.asyncio
async def test_spawn_authoring_uses_one_immutable_task_across_consumers():
    ckpt = checkpoint()
    ckpt.session.character_bindings = {"alice": "11", "bob": "22"}
    ckpt.world_state.setting.premise = "Original immutable premise."
    ckpt.__dict__["_closed_event_runtime"] = SimpleNamespace(
        lock=threading.RLock()
    )
    event = _spawn_event()
    manager = BlockingCharacterManager()
    coordinator = SpawnAuthoringCoordinator(manager)

    first = coordinator.start(
        checkpoint=ckpt,
        event=event,
        transaction_id="tx_first",
        event_fingerprint="a" * 64,
    )
    second = coordinator.start(
        checkpoint=ckpt,
        event=event,
        transaction_id="tx_second",
        event_fingerprint="a" * 64,
    )
    assert first is not None
    assert second == first

    ckpt.world_state.setting.premise = "Mutated after event closure."
    event.spawn[0].seed.role = "mutated"
    await asyncio.wait_for(manager.started.wait(), timeout=1)
    manager.release.set()
    result_a, result_b = await asyncio.gather(
        coordinator.result(first),
        coordinator.result(second),
    )

    assert manager.calls == 1
    assert manager.seen_premise == "Original immutable premise."
    assert manager.seen_roles == ["guide", "porter"]
    assert result_a is result_b
    assert [record.character_id for record in result_a] == ["guide", "porter"]


@pytest.mark.asyncio
async def test_closed_event_starts_spawn_and_notifies_image_sink_without_awaiting():
    ckpt = checkpoint()
    event = _spawn_event()
    manager = BlockingCharacterManager()
    coordinator = SpawnAuthoringCoordinator(manager)
    sink = RecordingSink()
    runtime = ClosedEventRuntime(
        transaction_id="tx_event",
        source_turn_index=1,
        spawn_authoring=coordinator,
        image_sink=sink,
        record_applier=Orchestrator._apply_authored_spawn_records,
    )

    runtime.close_event(
        checkpoint=ckpt,
        event=event,
        event_sequence=4,
        actor_id="alice",
    )

    assert len(sink.calls) == 1
    assert sink.calls[0]["event"] is event
    assert sink.calls[0]["event_sequence"] == 4
    assert sink.calls[0]["spawn_key"] is not None
    await asyncio.wait_for(manager.started.wait(), timeout=1)
    assert coordinator.task(sink.calls[0]["spawn_key"]) is not None
    manager.release.set()
    records = await runtime.authored_records(
        checkpoint=ckpt,
        event=event,
        actor_id="alice",
    )
    assert runtime.apply_records(ckpt, records) == ["guide", "porter"]
    assert runtime.apply_records(ckpt, records) == []


def test_broadcast_notifies_sidecar_after_event_is_finalized_in_canonical_log():
    ckpt = checkpoint()
    coordinator = SpawnAuthoringCoordinator(BlockingCharacterManager())

    class FinalizationSink(RecordingSink):
        def on_closed_event(self, **kwargs) -> None:
            assert kwargs["checkpoint"].canonical_events[-1] is kwargs["event"]
            super().on_closed_event(**kwargs)

    sink = FinalizationSink()
    runtime = ClosedEventRuntime(
        transaction_id="tx_broadcast",
        source_turn_index=1,
        spawn_authoring=coordinator,
        image_sink=sink,
    )
    install_closed_event_runtime(ckpt, runtime)
    event = router_output(event_id="evt_final", observer_ids=["alice"])

    broadcast_event(ckpt, event, actor_id="alice")

    assert len(sink.calls) == 1
    assert sink.calls[0]["event_sequence"] == 0


@pytest.mark.asyncio
async def test_sidecar_groups_equivalent_viewers_and_persists_empty_decision(
    tmp_path: Path,
):
    ckpt = checkpoint()
    ckpt.session.session_id = "sidecar"
    ckpt.session.character_bindings = {"alice": "11", "bob": "22"}
    event = router_output(
        event_id="evt_shared",
        observer_ids=["alice", "bob"],
    )
    spawn = SpawnAuthoringCoordinator(BlockingCharacterManager())
    generation = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=ImageGenerationConfig(runtime_root=tmp_path / "runtime"),
        worker=UnavailableWorker(),
    )
    director = BlockingDirector()
    sidecar = EventImageSidecar(
        director=director,
        generation=generation,
        spawn_authoring=spawn,
        delivery_kind=ImageDeliveryKind.cli,
        config=EventImageSidecarConfig(mode="shadow"),
    )
    generation.begin_transaction(
        transaction_id="tx_sidecar",
        session_id="sidecar",
        source_turn_index=1,
        source_checkpoint_sha256="a" * 64,
    )
    await sidecar.start()
    try:
        sidecar.on_closed_event(
            checkpoint=ckpt,
            event=event,
            event_sequence=0,
            transaction_id="tx_sidecar",
            source_turn_index=1,
            spawn_key=None,
            actor_id="alice",
        )
        await asyncio.wait_for(director.started.wait(), timeout=1)
        assert len(director.calls) == 1
        assert director.calls[0].viewer_character_ids == ("alice", "bob")
        director.release.set()

        async def decision_persisted() -> bool:
            with generation.store._connect() as db:
                row = db.execute(
                    """
                    SELECT status, output_json FROM image_director_runs
                    WHERE source_event_id = 'evt_shared'
                    """
                ).fetchone()
            return bool(
                row is not None
                and row["status"] == "succeeded"
                and row["output_json"] == '{"requests":[]}'
            )

        deadline = asyncio.get_running_loop().time() + 2
        while not await decision_persisted():
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError
            await asyncio.sleep(0.02)
    finally:
        director.release.set()
        await sidecar.close()
        await generation.close()


@pytest.mark.asyncio
async def test_director_starts_while_narrator_is_still_blocked(tmp_path: Path):
    ckpt = checkpoint()
    ckpt.session.session_id = "concurrent_sidecar"
    ckpt.session.character_bindings = {"alice": "11"}
    spawn = SpawnAuthoringCoordinator(BlockingCharacterManager())
    generation = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=ImageGenerationConfig(runtime_root=tmp_path / "runtime"),
        worker=UnavailableWorker(),
    )
    director = BlockingDirector()
    sidecar = EventImageSidecar(
        director=director,
        generation=generation,
        spawn_authoring=spawn,
        delivery_kind=ImageDeliveryKind.cli,
        config=EventImageSidecarConfig(mode="shadow"),
    )
    generation.begin_transaction(
        transaction_id="tx_concurrent",
        session_id="concurrent_sidecar",
        source_turn_index=1,
        source_checkpoint_sha256="a" * 64,
    )
    install_closed_event_runtime(
        ckpt,
        ClosedEventRuntime(
            transaction_id="tx_concurrent",
            source_turn_index=1,
            spawn_authoring=spawn,
            image_sink=sidecar,
            record_applier=Orchestrator._apply_authored_spawn_records,
        ),
    )
    dispatcher = BlockingNarratorDispatcher()
    dispatcher.queue_route(
        router_output(
            event_id="evt_concurrent",
            observer_ids=["alice"],
        )
    )

    await sidecar.start()
    beat = asyncio.create_task(
        run_beat(
            ckpt=ckpt,
            dispatcher=dispatcher,
            actor_id="alice",
            intention="I look out into the rain.",
        )
    )
    try:
        await asyncio.wait_for(dispatcher.narrator_started.wait(), timeout=1)
        await asyncio.wait_for(director.started.wait(), timeout=1)
        assert not beat.done()
    finally:
        director.release.set()
        dispatcher.narrator_release.set()
        await asyncio.gather(beat, return_exceptions=True)
        await sidecar.close()
        await generation.close()
