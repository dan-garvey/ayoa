from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts.run_character_dialogue_benchmark import load_benchmark_manifest
from scripts.run_persistent_luna_dialogue_silos import (
    LUNA_MODEL,
    LunaWorkerError,
    LunaWorkerResponse,
    SchedulerConfig,
    SchedulerError,
    SiloIsolationError,
    WorkerBinding,
    _session_id_from_jsonl,
    build_conversation_plans,
    run_persistent_luna_silos,
)


def _case(case_id: str, left_id: str, right_id: str):
    """Make a compact two-scene, four-turn dyad with actor-private sentinels."""

    source = load_benchmark_manifest()[0]
    actors = []
    for source_actor, actor_id in (
        (source.actors[0], left_id),
        (source.actors[1], right_id),
    ):
        assert source_actor.actor is not None
        facts = list(source_actor.actor.facts)
        facts[0] = facts[0].model_copy(
            update={"text": f"You alone know PRIVATE-{case_id}-{actor_id}."}
        )
        profile = source_actor.actor.model_copy(update={"facts": facts})
        actors.append(
            source_actor.model_copy(
                update={
                    "character_id": actor_id,
                    "name": actor_id.upper(),
                    "actor": profile,
                }
            )
        )
    first = replace(
        source.scenes[0],
        scene_id=f"{case_id}-first",
        turn_order=(left_id, right_id),
    )
    second = replace(
        source.scenes[1],
        scene_id=f"{case_id}-second",
        turn_order=(left_id, right_id),
    )
    return replace(
        source,
        case_id=case_id,
        title=case_id,
        actors=tuple(actors),
        scenes=(first, second),
    )


class FakeLunaExecutor:
    """Offline fake that records every exact request without inspecting prose."""

    def __init__(
        self,
        *,
        outcomes: list[str] | None = None,
        delay_seconds: float = 0.0,
        shared_session_id: str | None = None,
    ) -> None:
        self.outcomes = list(outcomes or [])
        self.delay_seconds = delay_seconds
        self.shared_session_id = shared_session_id
        self.sessions: dict[tuple[str, str], str] = {}
        self.calls: list[dict[str, Any]] = []
        self.start_calls: list[dict[str, Any]] = []
        self.resume_calls: list[dict[str, Any]] = []
        self.active_calls = 0
        self.max_active_calls = 0

    async def start(
        self,
        *,
        binding: WorkerBinding,
        request: Mapping[str, Any],
        attempt: int,
    ) -> LunaWorkerResponse:
        return await self._answer(
            mode="start",
            binding=binding,
            request=request,
            attempt=attempt,
            worker_session_id=None,
        )

    async def resume(
        self,
        *,
        binding: WorkerBinding,
        worker_session_id: str,
        request: Mapping[str, Any],
        attempt: int,
    ) -> LunaWorkerResponse:
        return await self._answer(
            mode="resume",
            binding=binding,
            request=request,
            attempt=attempt,
            worker_session_id=worker_session_id,
        )

    async def _answer(
        self,
        *,
        mode: str,
        binding: WorkerBinding,
        request: Mapping[str, Any],
        attempt: int,
        worker_session_id: str | None,
    ) -> LunaWorkerResponse:
        request_copy = json.loads(json.dumps(request, ensure_ascii=False))
        record = {
            "mode": mode,
            "binding": binding,
            "request": request_copy,
            "attempt": attempt,
            "worker_session_id": worker_session_id,
        }
        self.calls.append(record)
        if mode == "start":
            self.start_calls.append(record)
        else:
            self.resume_calls.append(record)
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            outcome = self.outcomes.pop(0) if self.outcomes else "accepted"
            if outcome == "rate_limited":
                raise LunaWorkerError(
                    "429 rate limited",
                    retryable=True,
                    rate_limited=True,
                    elapsed_ms=1.0,
                )
            key = (binding.conversation_id, binding.actor_id)
            if mode == "start":
                assert key not in self.sessions
                session_id = self.shared_session_id or f"session-{len(self.sessions) + 1}"
                self.sessions[key] = session_id
            else:
                assert self.sessions[key] == worker_session_id
                session_id = worker_session_id
            if outcome == "rate_limited_after_session":
                raise LunaWorkerError(
                    "429 rate limited after thread creation",
                    retryable=True,
                    rate_limited=True,
                    elapsed_ms=1.0,
                    session_id=session_id,
                )
            raw_response = "" if outcome == "invalid_shape" else (
                f'{binding.actor_id} says "turn {request["turn_index"]}".'
            )
            return LunaWorkerResponse(
                worker_session_id=session_id,
                raw_response=raw_response,
                elapsed_ms=1.0,
            )
        finally:
            self.active_calls -= 1


def _run(cases, tmp_path: Path, *, config: SchedulerConfig, executor: FakeLunaExecutor):
    return asyncio.run(
        run_persistent_luna_silos(
            cases,
            output_dir=tmp_path,
            run_id="frozen-run",
            manifest_sha256="a" * 64,
            config=config,
            executor=executor,
        )
    )


def test_scheduler_keeps_workers_persistent_and_private_per_actor_and_silo(
    tmp_path: Path,
) -> None:
    cases = (
        _case("case_a", "a_left", "a_right"),
        _case("case_b", "b_left", "b_right"),
    )
    executor = FakeLunaExecutor(delay_seconds=0.002)
    config = SchedulerConfig(
        conversation_count=2,
        turn_count=4,
        initial_concurrency=2,
        max_concurrency=2,
        auto_fanout=False,
        max_technical_retries=0,
        retry_backoff_seconds=0,
    )

    result = _run(cases, tmp_path, config=config, executor=executor)

    assert LUNA_MODEL == "gpt-5.6-luna"
    assert [len(conversation.turns) for conversation in result.results] == [4, 4]
    assert len(executor.sessions) == 4
    assert len(executor.start_calls) == 4
    assert len(executor.resume_calls) == 4
    assert executor.max_active_calls == 2

    private_sentinels = {
        "a_left": "PRIVATE-case_a-a_left",
        "a_right": "PRIVATE-case_a-a_right",
        "b_left": "PRIVATE-case_b-b_left",
        "b_right": "PRIVATE-case_b-b_right",
    }
    for call in executor.calls:
        binding = call["binding"]
        request = call["request"]
        rendered = json.dumps(request, ensure_ascii=False)
        assert request["conversation_id"] == binding.conversation_id
        assert request["actor_id"] == binding.actor_id
        assert private_sentinels[binding.actor_id] in rendered
        for actor_id, sentinel in private_sentinels.items():
            if actor_id != binding.actor_id:
                assert sentinel not in rendered

    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    workers = state["workers"]
    assert len(workers) == 4
    assert len({entry["worker_session_id"] for entry in workers.values()}) == 4
    accepted = [event for event in state["attempts"] if event["status"] == "accepted"]
    assert len(accepted) == 8
    assert all(len(event["request_sha256"]) == 64 for event in accepted)
    assert all(event["worker_session_id"] for event in accepted)
    for index, case in enumerate(cases, start=1):
        ledger = json.loads(
            (
                tmp_path
                / "conversations"
                / f"{index:02d}-{case.case_id}"
                / "response_ledger.json"
            ).read_text(encoding="utf-8")
        )
        for entry in ledger["responses"]:
            event = next(
                item
                for item in accepted
                if item["conversation_id"] == ledger["conversation_id"]
                and item["ledger_sequence"] == entry["sequence"]
            )
            assert event["request_sha256"] == entry["request_fingerprint"]
            assert event["raw_response"] == entry["response"]["content"]


def test_scheduler_rejects_a_codex_session_reused_by_two_silos(tmp_path: Path) -> None:
    cases = (
        _case("case_one", "one_left", "one_right"),
        _case("case_two", "two_left", "two_right"),
    )
    executor = FakeLunaExecutor(shared_session_id="not-a-silo")
    config = SchedulerConfig(
        conversation_count=2,
        turn_count=4,
        initial_concurrency=1,
        max_concurrency=1,
        auto_fanout=False,
        max_technical_retries=0,
        retry_backoff_seconds=0,
    )

    with pytest.raises(SiloIsolationError, match="cannot be reused"):
        _run(cases, tmp_path, config=config, executor=executor)


def test_scheduler_retries_only_technical_failures_and_freezes_fanout_after_throttle(
    tmp_path: Path,
) -> None:
    case = _case("case_retry", "retry_left", "retry_right")
    executor = FakeLunaExecutor(
        outcomes=["rate_limited_after_session", "invalid_shape"]
    )
    config = SchedulerConfig(
        conversation_count=1,
        turn_count=4,
        initial_concurrency=1,
        max_concurrency=2,
        auto_fanout=True,
        fanout_after_successes=1,
        max_technical_retries=2,
        retry_backoff_seconds=0,
    )

    result = _run((case,), tmp_path, config=config, executor=executor)

    assert len(result.results[0].turns) == 4
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    statuses = [event["status"] for event in state["attempts"]]
    assert "rate_limited" in statuses
    assert "invalid_output_shape" in statuses
    assert statuses.count("accepted") == 4
    assert state["adaptive_concurrency"] == {
        "current_limit": 1,
        "successful_since_fanout": 0,
        "observed_throttling": True,
    }
    ledger = json.loads(
        (tmp_path / "conversations" / "01-case_retry" / "response_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert [entry["sequence"] for entry in ledger["responses"]] == [0, 1, 2, 3]
    # The retry resumes the thread that was emitted before rate limiting; it
    # never starts a replacement session for the same actor/conversation.
    assert len(executor.start_calls) == 2
    assert len(executor.resume_calls) == 4
    rate_limit = next(event for event in state["attempts"] if event["status"] == "rate_limited")
    assert rate_limit["worker_session_id"] == "session-1"


def test_scheduler_requires_the_fixed_authored_turn_count(tmp_path: Path) -> None:
    case = _case("too_short", "left", "right")
    config = SchedulerConfig(
        conversation_count=1,
        turn_count=5,
        initial_concurrency=1,
        max_concurrency=1,
        retry_backoff_seconds=0,
    )

    with pytest.raises(SchedulerError, match="has 4 authored turns"):
        build_conversation_plans(
            (case,),
            output_dir=tmp_path,
            run_id="fixed-run",
            config=config,
        )


def test_session_id_parser_accepts_codex_jsonl_event_envelopes() -> None:
    assert _session_id_from_jsonl(
        '{"type":"thread.started","thread_id":"top-level-thread"}\n'
    ) == "top-level-thread"
    assert _session_id_from_jsonl(
        '{"type":"event_msg","payload":{"thread_id":"payload-thread"}}\n'
    ) == "payload-thread"
