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
    ReflectionOutputError,
    SchedulerConfig,
    SchedulerError,
    SiloIsolationError,
    WorkerBinding,
    _parse_private_reflection,
    _proxy_prompt,
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
        self.next_session_number = 1
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
        reflection_nonce: str | None = None,
        persistent_delta_proxy: bool = False,
    ) -> LunaWorkerResponse:
        return await self._answer(
            mode="start",
            binding=binding,
            request=request,
            attempt=attempt,
            worker_session_id=None,
            reflection_nonce=reflection_nonce,
            persistent_delta_proxy=persistent_delta_proxy,
        )

    async def resume(
        self,
        *,
        binding: WorkerBinding,
        worker_session_id: str,
        request: Mapping[str, Any],
        attempt: int,
        reflection_nonce: str | None = None,
        persistent_delta_proxy: bool = False,
    ) -> LunaWorkerResponse:
        return await self._answer(
            mode="resume",
            binding=binding,
            request=request,
            attempt=attempt,
            worker_session_id=worker_session_id,
            reflection_nonce=reflection_nonce,
            persistent_delta_proxy=persistent_delta_proxy,
        )

    async def _answer(
        self,
        *,
        mode: str,
        binding: WorkerBinding,
        request: Mapping[str, Any],
        attempt: int,
        worker_session_id: str | None,
        reflection_nonce: str | None,
        persistent_delta_proxy: bool,
    ) -> LunaWorkerResponse:
        request_copy = json.loads(json.dumps(request, ensure_ascii=False))
        record = {
            "mode": mode,
            "binding": binding,
            "request": request_copy,
            "attempt": attempt,
            "worker_session_id": worker_session_id,
            "reflection_nonce": reflection_nonce,
            "persistent_delta_proxy": persistent_delta_proxy,
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
                session_id = (
                    self.shared_session_id or f"session-{self.next_session_number}"
                )
                self.next_session_number += 1
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
            raw_response = (
                ""
                if outcome == "invalid_shape"
                else (f'{binding.actor_id} says "turn {request["turn_index"]}".')
            )
            if outcome == "malformed_reflection":
                raw_response += '\n\n(<actor_private_reflection id="R-00000000000000000000000000000000">{})'
            elif reflection_nonce is not None:
                raw_response += (
                    '\n\n(<actor_private_reflection id="'
                    f'{reflection_nonce}">'
                    '{"present_true_position":"I am guarding the point.",'
                    '"public_attempt":"I answer without yielding ground.",'
                    '"deliberately_unsaid_truth":"NONE",'
                    '"unavailable_because":"NONE",'
                    '"relationship_status_cost":"NONE",'
                    '"continuity_pressure":"Keep the unanswered question present."}'
                    "</actor_private_reflection>)"
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
        (
            tmp_path / "conversations" / "01-case_retry" / "response_ledger.json"
        ).read_text(encoding="utf-8")
    )
    assert [entry["sequence"] for entry in ledger["responses"]] == [0, 1, 2, 3]
    # The retry resumes the thread that was emitted before rate limiting; it
    # never starts a replacement session for the same actor/conversation.
    assert len(executor.start_calls) == 2
    assert len(executor.resume_calls) == 4
    rate_limit = next(
        event for event in state["attempts"] if event["status"] == "rate_limited"
    )
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
    assert (
        _session_id_from_jsonl(
            '{"type":"thread.started","thread_id":"top-level-thread"}\n'
        )
        == "top-level-thread"
    )
    assert (
        _session_id_from_jsonl(
            '{"type":"event_msg","payload":{"thread_id":"payload-thread"}}\n'
        )
        == "payload-thread"
    )


def test_private_reflection_parser_requires_the_exact_anchored_shape() -> None:
    nonce = "R-0123456789abcdef0123456789abcdef"
    raw = (
        "She keeps the cup between them.\n\n"
        f'(<actor_private_reflection id="{nonce}">'
        '{"present_true_position":"I want an answer before I leave.",'
        '"public_attempt":"I make the request sound casual.",'
        '"deliberately_unsaid_truth":"NONE",'
        '"unavailable_because":"NONE",'
        '"relationship_status_cost":"I risk sounding colder than I feel.",'
        '"continuity_pressure":"The unpaid favor still needs an answer."}'
        "</actor_private_reflection>)\n"
    )

    parsed = _parse_private_reflection(raw, expected_nonce=nonce)

    assert parsed.public_body == "She keeps the cup between them."
    assert parsed.reflection.nonce == nonce
    assert (
        parsed.reflection.fields["public_attempt"] == "I make the request sound casual."
    )
    with pytest.raises(ReflectionOutputError, match="exact, ordered, and unique"):
        _parse_private_reflection(
            raw.replace('"public_attempt":"I make the request sound casual.",', ""),
            expected_nonce=nonce,
        )
    with pytest.raises(ReflectionOutputError, match="nonce does not match"):
        _parse_private_reflection(
            raw, expected_nonce="R-ffffffffffffffffffffffffffffffff"
        )
    with pytest.raises(ReflectionOutputError, match="missing or malformed"):
        _parse_private_reflection(raw + "extra", expected_nonce=nonce)
    with pytest.raises(ReflectionOutputError, match="appears in public prose"):
        _parse_private_reflection(
            raw.replace("She keeps", "<actor_private_reflection> She keeps"),
            expected_nonce=nonce,
        )


def test_private_reflection_mode_keeps_raw_suffixes_out_of_public_relay(
    tmp_path: Path,
) -> None:
    case = _case("private", "left", "right")
    executor = FakeLunaExecutor()
    config = SchedulerConfig(
        conversation_count=1,
        turn_count=4,
        initial_concurrency=1,
        max_concurrency=1,
        auto_fanout=False,
        max_technical_retries=0,
        retry_backoff_seconds=0,
        private_reflections=True,
        persistent_delta_proxy=True,
    )

    _run((case,), tmp_path, config=config, executor=executor)

    state = json.loads((tmp_path / "scheduler_state.json").read_text(encoding="utf-8"))
    turns = state["private_reflection_turns"]
    assert len(turns) == 4
    assert all(item["status"] == "accepted" for item in turns)
    assert all(item["nonce"].startswith("R-") for item in turns)
    state_text = json.dumps(state, ensure_ascii=False)
    assert "present_true_position" not in state_text
    assert (
        "raw_response" in state_text
    )  # Existing non-reflection telemetry shape remains stable.
    accepted = [item for item in state["attempts"] if item["status"] == "accepted"]
    assert all(item["raw_response"] is None for item in accepted)
    assert all(len(item["request_sha256"]) == 64 for item in accepted)
    assert all(len(item["projected_request_sha256"]) == 64 for item in accepted)
    assert all(call["persistent_delta_proxy"] for call in executor.calls)

    ledger_text = (
        tmp_path / "conversations" / "01-private" / "response_ledger.json"
    ).read_text(encoding="utf-8")
    assert "actor_private_reflection" not in ledger_text
    assert "present_true_position" not in ledger_text
    assert all(item["nonce"] not in ledger_text for item in turns)
    for call in executor.calls:
        request_text = json.dumps(call["request"], ensure_ascii=False)
        assert "actor_private_reflection" not in request_text
        assert "present_true_position" not in request_text
        assert all(item["nonce"] not in request_text for item in turns)

    qa_files = sorted((tmp_path / "private_reflection_qa").rglob("*.json"))
    assert len(qa_files) == 4
    qa_text = qa_files[0].read_text(encoding="utf-8")
    assert "raw_response" in qa_text
    assert "present_true_position" in qa_text


def test_malformed_private_reflection_discards_the_conversation_and_restarts(
    tmp_path: Path,
) -> None:
    case = _case("restart", "left", "right")
    executor = FakeLunaExecutor(outcomes=["malformed_reflection"])
    config = SchedulerConfig(
        conversation_count=1,
        turn_count=4,
        initial_concurrency=1,
        max_concurrency=1,
        auto_fanout=False,
        max_technical_retries=0,
        retry_backoff_seconds=0,
        private_reflections=True,
    )

    result = _run((case,), tmp_path, config=config, executor=executor)

    assert len(result.results[0].turns) == 4
    assert len(executor.start_calls) == 3  # Rejected left worker is never resumed.
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    statuses = [item["status"] for item in state["attempts"]]
    assert "rejected_malformed_suffix" in statuses
    assert all(item["raw_response"] is None for item in state["attempts"])
    rejected = list((tmp_path / "private_reflection_qa").rglob("*.json"))
    assert any(
        "rejected_malformed_suffix" in path.read_text(encoding="utf-8")
        for path in rejected
    )


def test_persistent_delta_resume_excludes_history_and_carries_current_packet_and_nonce() -> (
    None
):
    request = {
        "conversation_id": "run:case",
        "case_id": "case",
        "actor_id": "actor",
        "scene_id": "scene",
        "scene_index": 1,
        "scene_turn_index": 2,
        "turn_index": 3,
        "messages": [
            {"role": "system", "content": "SYSTEM-PROMPT"},
            {"role": "user", "content": "<you>OLD-YOU</you><now>OLD-NOW</now>"},
            {"role": "assistant", "content": "EARLIER-ASSISTANT"},
            {"role": "user", "content": "<you>CURRENT-YOU</you><now>CURRENT-NOW</now>"},
        ],
    }
    nonce = "R-0123456789abcdef0123456789abcdef"

    prompt = _proxy_prompt(
        request,
        reflection_nonce=nonce,
        persistent_delta_proxy=True,
        starts_session=False,
    )

    assert "CURRENT-NOW" in prompt
    assert "CURRENT-YOU" in prompt
    assert nonce in prompt
    assert "OLD-YOU" not in prompt
    assert "OLD-NOW" not in prompt
    assert "SYSTEM-PROMPT" not in prompt
    assert "EARLIER-ASSISTANT" not in prompt
    assert "Follow the embedded observable-body contract" not in prompt
    assert "present_true_position" not in prompt


def test_default_proxy_prompt_remains_the_original_full_request_wrapper() -> None:
    request = {
        "actor_id": "actor",
        "messages": [{"role": "user", "content": "now"}],
    }
    payload = json.dumps(
        request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )

    assert _proxy_prompt(request) == (
        "Act only as an isolated CharacterAgent completion proxy. Return exactly the "
        "raw assistant content for this request: no analysis, labels, markdown, "
        "tools, file inspection, or commentary. The JSON is the full current "
        "continuation for one actor; use nothing outside it.\n"
        f"<exact_character_agent_request>\n{payload}\n</exact_character_agent_request>\n"
    )
