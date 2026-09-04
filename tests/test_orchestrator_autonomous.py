from __future__ import annotations

import asyncio

import pytest

from app.engine.orchestrator import Orchestrator
from app.engine.session_writer import SessionWriterLocks
from app.engine.story_coordinator import AdvanceResult, player_input
from app.schemas.event_router import FrontierTurn
from tests.support.factories import character_record, checkpoint


class MemoryCheckpointManager:
    def __init__(self, value):
        self.value = value
        self.saved: list[object] = []

    def load_latest(self, _session_id: str):
        return self.value

    def save(self, value):
        self.value = value
        self.saved.append(value)
        return f"ckpt_{value.session.turn_index:04d}"


def _orchestrator(value) -> Orchestrator:
    runtime = object.__new__(Orchestrator)
    runtime.checkpoint_mgr = MemoryCheckpointManager(value)
    runtime.session_locks = SessionWriterLocks()
    runtime.dispatcher = object()
    runtime._autonomous_tasks = {}
    runtime._autonomous_phase = {}
    runtime._prepared_autonomous = {}
    return runtime


def _checkpoint_with_ready_work():
    value = checkpoint(
        session_id="session",
        characters=[character_record("bob")],
    )
    value.session.router_frontier.append(FrontierTurn(
        turn_id="turn_bob",
        lane_id="lane_bob",
        turn_kind="character",
        actor_id="bob",
        participant_ids=["bob"],
        source_event_ids=[],
        created_event_sequence=0,
        gating_pov_ids=[],
    ))
    return value


def test_prepared_cache_ignores_delivery_revision_but_not_story_mutation() -> None:
    value = checkpoint(
        session_id="session",
        characters=[character_record("alice")],
    )
    runtime = _orchestrator(value)
    prepared = player_input(value, character_id="alice", payload="Wait.")
    runtime._remember_prepared(value, [prepared])

    value.session.turn_index += 1
    value.narrator_conversations["alice"] = []
    assert runtime._cached_prepared(value) == [prepared]

    value.world_state.facts.append("The actual story changed.")
    assert runtime._cached_prepared(value) == []


@pytest.mark.asyncio
async def test_autonomous_provider_work_runs_outside_session_writer_lock(
    monkeypatch,
) -> None:
    runtime = _orchestrator(_checkpoint_with_ready_work())
    prepared = object()

    async def fake_prepare(*_args, **_kwargs):
        return [prepared]

    async def fake_advance(snapshot, _dispatcher, values):
        assert values == [prepared]
        lock = await runtime.session_locks.for_session("session")
        await asyncio.wait_for(lock.acquire(), timeout=0.2)
        lock.release()
        snapshot.session.router_frontier = []
        snapshot.session.autonomous_router_batches_since_player += 1
        return AdvanceResult()

    monkeypatch.setattr(
        "app.engine.orchestrator.prepare_ready_frontier_batch",
        fake_prepare,
    )
    monkeypatch.setattr("app.engine.orchestrator.advance_story", fake_advance)

    await asyncio.wait_for(runtime._autonomous_worker("session"), timeout=1)

    assert runtime.checkpoint_mgr.value.session.router_frontier == []
    assert len(runtime.checkpoint_mgr.saved) == 1


@pytest.mark.asyncio
async def test_autonomous_result_is_discarded_after_concurrent_player_commit(
    monkeypatch,
) -> None:
    runtime = _orchestrator(_checkpoint_with_ready_work())
    routing_started = asyncio.Event()
    finish_routing = asyncio.Event()

    async def fake_prepare(*_args, **_kwargs):
        return [object()]

    async def fake_advance(snapshot, _dispatcher, _values):
        routing_started.set()
        await finish_routing.wait()
        snapshot.world_state.facts.append("stale autonomous result")
        snapshot.session.router_frontier = []
        return AdvanceResult()

    monkeypatch.setattr(
        "app.engine.orchestrator.prepare_ready_frontier_batch",
        fake_prepare,
    )
    monkeypatch.setattr("app.engine.orchestrator.advance_story", fake_advance)

    task = asyncio.create_task(runtime._autonomous_worker("session"))
    await asyncio.wait_for(routing_started.wait(), timeout=1)
    lock = await runtime.session_locks.for_session("session")
    async with lock:
        runtime.checkpoint_mgr.value.world_state.facts.append("player commit")
        runtime.checkpoint_mgr.value.session.router_frontier = []
    finish_routing.set()
    await asyncio.wait_for(task, timeout=1)

    assert runtime.checkpoint_mgr.saved == []
    assert runtime.checkpoint_mgr.value.world_state.facts == ["player commit"]
