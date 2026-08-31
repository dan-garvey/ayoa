#!/usr/bin/env python3
# ruff: noqa: E402 -- executable script adds the repository root to sys.path.
"""Run isolated persistent Codex Luna proxies through the exact benchmark relay.

The benchmark relay still owns every CharacterAgent prompt, parser, ledger,
and conversation history. This script only schedules one raw response at a
time from a long-lived Codex thread owned by one actor in one conversation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Protocol, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.engine.character_agent import (
    CharacterAgentOutputError,
    sanitize_character_public_text,
)
from scripts.run_character_dialogue_benchmark import (
    BenchmarkCase,
    ConversationResult,
    RelayPendingRequest,
    _atomic_write_json,
    _benchmark_request_fingerprint,
    _benchmark_request_payload,
    _sha256_json,
    append_relay_response,
    load_benchmark_manifest,
    run_relay_conversation,
)


SCHEDULER_SCHEMA_VERSION = "persistent_luna_dialogue_silos_v1"
LUNA_MODEL = "gpt-5.6-luna"
DEFAULT_CONVERSATION_COUNT = 8
FIXED_TURN_COUNT = 16
_RATE_LIMIT_RE = re.compile(
    r"(?:\b429\b|rate[ -]?limit|throttl|too many requests|capacity)", re.I
)


class SchedulerError(RuntimeError):
    """The scheduler cannot continue without violating its technical contract."""


class SiloIsolationError(SchedulerError):
    """A persistent worker session crossed actor or conversation ownership."""


class LunaWorkerError(SchedulerError):
    """A proxy call failed before it produced a valid raw response."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        rate_limited: bool = False,
        elapsed_ms: float = 0.0,
        session_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.rate_limited = rate_limited
        self.elapsed_ms = elapsed_ms
        self.session_id = session_id


@dataclass(frozen=True)
class WorkerBinding:
    conversation_id: str
    case_id: str
    actor_id: str


@dataclass(frozen=True)
class LunaWorkerResponse:
    worker_session_id: str
    raw_response: str
    elapsed_ms: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class LunaWorkerExecutor(Protocol):
    async def start(
        self, *, binding: WorkerBinding, request: Mapping[str, Any], attempt: int
    ) -> LunaWorkerResponse: ...

    async def resume(
        self,
        *,
        binding: WorkerBinding,
        worker_session_id: str,
        request: Mapping[str, Any],
        attempt: int,
    ) -> LunaWorkerResponse: ...


@dataclass(frozen=True)
class SchedulerConfig:
    conversation_count: int = DEFAULT_CONVERSATION_COUNT
    turn_count: int = FIXED_TURN_COUNT
    initial_concurrency: int = 1
    max_concurrency: int = 8
    auto_fanout: bool = True
    fanout_after_successes: int = 16
    max_technical_retries: int = 2
    retry_backoff_seconds: float = 1.0
    worker_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.conversation_count < 1 or self.turn_count < 1:
            raise ValueError("conversation_count and turn_count must be positive")
        if not 1 <= self.initial_concurrency <= self.max_concurrency:
            raise ValueError("concurrency bounds are invalid")
        if self.fanout_after_successes < 1 or self.max_technical_retries < 0:
            raise ValueError("fanout and retry limits are invalid")
        if not 0 <= self.retry_backoff_seconds <= 60:
            raise ValueError("retry_backoff_seconds must be between 0 and 60")
        if not 1 <= self.worker_timeout_seconds <= 3600:
            raise ValueError("worker_timeout_seconds must be between 1 and 3600")


@dataclass(frozen=True)
class ConversationPlan:
    case: BenchmarkCase
    conversation_id: str
    ledger_path: Path
    pending_path: Path
    expected_turn_count: int

    @property
    def actor_ids(self) -> tuple[str, str]:
        actor_ids = tuple(actor.character_id for actor in self.case.actors)
        if len(actor_ids) != 2:
            raise SchedulerError("conversation plan is not dyadic")
        return actor_ids[0], actor_ids[1]


@dataclass(frozen=True)
class SchedulerRunResult:
    results: tuple[ConversationResult, ...]
    state_path: Path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SchedulerError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SchedulerError(f"{label} {path} must be a JSON object")
    return value


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_sha256(value: str) -> str:
    value = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SchedulerError("frozen manifest SHA-256 must be 64 hexadecimal characters")
    return value


def _safe_path_piece(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-") or "conversation"


def _binding_key(binding: WorkerBinding) -> str:
    return json.dumps([binding.conversation_id, binding.actor_id], separators=(",", ":"))


def _validate_case(case: BenchmarkCase, turn_count: int) -> None:
    actor_ids = tuple(actor.character_id for actor in case.actors)
    if len(actor_ids) != 2 or len(set(actor_ids)) != 2:
        raise SchedulerError(f"case {case.case_id!r} must have exactly two actors")
    order = tuple(actor_id for scene in case.scenes for actor_id in scene.turn_order)
    if len(order) != turn_count:
        raise SchedulerError(
            f"case {case.case_id!r} has {len(order)} authored turns; "
            f"the scheduler requires exactly {turn_count}"
        )
    if set(actor_ids) - set(order):
        raise SchedulerError(f"case {case.case_id!r} leaves an actor without a turn")


def build_conversation_plans(
    cases: Sequence[BenchmarkCase],
    *,
    output_dir: Path,
    run_id: str,
    config: SchedulerConfig,
) -> tuple[ConversationPlan, ...]:
    if len(cases) != config.conversation_count:
        raise SchedulerError(
            f"expected exactly {config.conversation_count} conversations, got {len(cases)}"
        )
    if not run_id.strip() or len({case.case_id for case in cases}) != len(cases):
        raise SchedulerError("run_id and every scheduled case_id must be unique and nonblank")
    plans = []
    for index, case in enumerate(cases, start=1):
        _validate_case(case, config.turn_count)
        root = output_dir / "conversations" / f"{index:02d}-{_safe_path_piece(case.case_id)}"
        ledger_path = root / "response_ledger.json"
        plans.append(
            ConversationPlan(
                case=case,
                conversation_id=f"{run_id}:{case.case_id}",
                ledger_path=ledger_path,
                pending_path=ledger_path.with_name(ledger_path.name + ".pending.json"),
                expected_turn_count=config.turn_count,
            )
        )
    return tuple(plans)


class SchedulerAudit:
    """Small durable ledger for worker ownership and technical telemetry."""

    def __init__(
        self,
        path: Path,
        *,
        manifest_sha256: str,
        run_id: str,
        config: SchedulerConfig,
    ) -> None:
        self.path = path
        expected = {
            "manifest_sha256": _frozen_sha256(manifest_sha256),
            "run_id": run_id,
            "model": LUNA_MODEL,
            "conversation_count": config.conversation_count,
            "turn_count": config.turn_count,
        }
        if path.exists():
            self.data = _read_json(path, label="scheduler state")
            if self.data.get("schema_version") != SCHEDULER_SCHEMA_VERSION:
                raise SchedulerError("scheduler state has an unsupported schema version")
            if self.data.get("run") != expected:
                raise SchedulerError("scheduler state does not match this frozen run")
            if not all(
                isinstance(self.data.get(key), kind)
                for key, kind in (
                    ("workers", dict),
                    ("attempts", list),
                    ("adaptive_concurrency", dict),
                )
            ):
                raise SchedulerError("scheduler state has an invalid shape")
        else:
            self.data = {
                "schema_version": SCHEDULER_SCHEMA_VERSION,
                "run": expected,
                "workers": {},
                "attempts": [],
                "adaptive_concurrency": {
                    "current_limit": config.initial_concurrency,
                    "successful_since_fanout": 0,
                    "observed_throttling": False,
                },
            }
            self.flush()
        self._validate_workers()

    def flush(self) -> None:
        _atomic_write_json(self.path, self.data)

    def _validate_workers(self) -> None:
        sessions: set[str] = set()
        for key, entry in self.data["workers"].items():
            if not isinstance(key, str) or not isinstance(entry, Mapping):
                raise SchedulerError("scheduler worker table is malformed")
            values = (
                entry.get("conversation_id"),
                entry.get("case_id"),
                entry.get("actor_id"),
                entry.get("worker_session_id"),
            )
            if not all(isinstance(value, str) and value for value in values):
                raise SchedulerError("scheduler worker entry lacks ownership data")
            binding = WorkerBinding(values[0], values[1], values[2])
            if key != _binding_key(binding):
                raise SiloIsolationError("scheduler worker entry has stale ownership")
            if values[3] in sessions:
                raise SiloIsolationError("one Codex session is assigned to multiple silos")
            sessions.add(values[3])

    def session(self, binding: WorkerBinding) -> str | None:
        entry = self.data["workers"].get(_binding_key(binding))
        if entry is None:
            return None
        if not isinstance(entry, Mapping) or any(
            entry.get(field) != getattr(binding, field)
            for field in ("conversation_id", "case_id", "actor_id")
        ):
            raise SiloIsolationError("worker ownership does not match its requested silo")
        session_id = entry.get("worker_session_id")
        if not isinstance(session_id, str) or not session_id:
            raise SchedulerError("worker entry lacks a session id")
        return session_id

    def bind(self, binding: WorkerBinding, session_id: str | None) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise SiloIsolationError("Codex did not return a persistent session id")
        session_id = session_id.strip()
        known = self.session(binding)
        if known is not None:
            if known != session_id:
                raise SiloIsolationError("a worker attempted to replace its persistent session")
            return
        if any(
            isinstance(entry, Mapping) and entry.get("worker_session_id") == session_id
            for entry in self.data["workers"].values()
        ):
            raise SiloIsolationError("a Codex session cannot be reused across silos")
        self.data["workers"][_binding_key(binding)] = {
            "conversation_id": binding.conversation_id,
            "case_id": binding.case_id,
            "actor_id": binding.actor_id,
            "worker_session_id": session_id,
        }
        self.flush()

    def record(self, event: Mapping[str, Any]) -> None:
        self.data["attempts"].append(dict(event))
        self.flush()

    def save_gate(self, gate: Mapping[str, Any]) -> None:
        self.data["adaptive_concurrency"] = dict(gate)
        self.flush()


class AdaptiveGate:
    """Permit fan-out only after accepted calls without throttling."""

    def __init__(self, config: SchedulerConfig, audit: SchedulerAudit) -> None:
        self.config = config
        self.audit = audit
        saved = audit.data["adaptive_concurrency"]
        self.limit = saved.get("current_limit")
        self.successes = saved.get("successful_since_fanout")
        self.throttled = saved.get("observed_throttling")
        if not isinstance(self.limit, int) or not 1 <= self.limit <= config.max_concurrency:
            raise SchedulerError("persisted concurrency limit is invalid")
        if not isinstance(self.successes, int) or self.successes < 0 or not isinstance(self.throttled, bool):
            raise SchedulerError("persisted concurrency telemetry is invalid")
        self.active = 0
        self.condition = asyncio.Condition()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self.condition:
            while self.active >= self.limit:
                await self.condition.wait()
            self.active += 1
        try:
            yield
        finally:
            async with self.condition:
                self.active -= 1
                self.condition.notify_all()

    async def observe(self, *, accepted: bool, rate_limited: bool) -> None:
        async with self.condition:
            if rate_limited:
                self.throttled = True
                self.successes = 0
            elif accepted and not self.throttled:
                self.successes += 1
                if (
                    self.config.auto_fanout
                    and self.successes >= self.config.fanout_after_successes
                    and self.limit < self.config.max_concurrency
                ):
                    self.limit += 1
                    self.successes = 0
                    self.condition.notify_all()
        self.audit.save_gate(
            {
                "current_limit": self.limit,
                "successful_since_fanout": self.successes,
                "observed_throttling": self.throttled,
            }
        )


def _proxy_prompt(request: Mapping[str, Any]) -> str:
    payload = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        "Act only as an isolated CharacterAgent completion proxy. Return exactly the "
        "raw assistant content for this request: no analysis, labels, markdown, "
        "tools, file inspection, or commentary. The JSON is the full current "
        "continuation for one actor; use nothing outside it.\n"
        f"<exact_character_agent_request>\n{payload}\n</exact_character_agent_request>\n"
    )


def _session_id_from_jsonl(stdout: str) -> str:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        containers = [event]
        if isinstance(event.get("payload"), Mapping):
            containers.append(event["payload"])
        for container in containers:
            for key in ("thread_id", "session_id"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for key in ("thread", "session"):
                nested = container.get(key)
                if isinstance(nested, Mapping) and isinstance(nested.get("id"), str):
                    return nested["id"].strip()
    raise LunaWorkerError("Codex did not emit a persistent thread id", retryable=False)


def _maybe_session_id(stdout: str) -> str | None:
    try:
        return _session_id_from_jsonl(stdout)
    except LunaWorkerError:
        return None


class CodexLunaCliExecutor:
    """Starts a Codex Luna thread once and uses ``exec resume`` thereafter."""

    def __init__(
        self,
        *,
        workers_root: Path,
        timeout_seconds: float = 300.0,
        command: Sequence[str] = ("codex",),
    ) -> None:
        if not command:
            raise ValueError("Codex command cannot be empty")
        self.workers_root = workers_root
        self.timeout_seconds = timeout_seconds
        self.command = tuple(command)

    def _response_path(
        self, binding: WorkerBinding, request: Mapping[str, Any], attempt: int
    ) -> Path:
        worker_hash = hashlib.sha256(_binding_key(binding).encode()).hexdigest()[:12]
        root = self.workers_root / f"{_safe_path_piece(binding.actor_id)}-{worker_hash}"
        root.mkdir(parents=True, exist_ok=True)
        request_hash = _sha256_json(request)
        return root / f"{request_hash}-attempt-{attempt}.txt"

    async def _invoke(
        self,
        *,
        command: Sequence[str],
        binding: WorkerBinding,
        request: Mapping[str, Any],
        attempt: int,
    ) -> tuple[str, str, float]:
        response_path = self._response_path(binding, request, attempt)
        started_at = time.perf_counter()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(response_path.parent),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(_proxy_prompt(request).encode()),
                timeout=self.timeout_seconds,
            )
        except OSError as error:
            raise LunaWorkerError(
                "could not start Codex worker",
                retryable=True,
                rate_limited=bool(_RATE_LIMIT_RE.search(str(error))),
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
            ) from error
        except TimeoutError as error:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise LunaWorkerError(
                "Codex worker timed out",
                retryable=True,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
            ) from error
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        if process.returncode != 0:
            detail = f"{stdout}\n{stderr}"
            raise LunaWorkerError(
                "Codex rate limited" if _RATE_LIMIT_RE.search(detail) else "Codex call failed",
                retryable=True,
                rate_limited=bool(_RATE_LIMIT_RE.search(detail)),
                elapsed_ms=elapsed_ms,
                session_id=_maybe_session_id(stdout),
            )
        try:
            return response_path.read_text(encoding="utf-8"), stdout, elapsed_ms
        except OSError as error:
            raise LunaWorkerError(
                "Codex completed without a readable final response",
                retryable=True,
                elapsed_ms=elapsed_ms,
            ) from error

    async def start(
        self, *, binding: WorkerBinding, request: Mapping[str, Any], attempt: int
    ) -> LunaWorkerResponse:
        response_path = self._response_path(binding, request, attempt)
        command = (
            *self.command, "exec", "--model", LUNA_MODEL, "--sandbox", "read-only",
            "--skip-git-repo-check", "--ignore-rules", "--json", "--output-last-message",
            str(response_path), "-",
        )
        raw, stdout, elapsed_ms = await self._invoke(
            command=command, binding=binding, request=request, attempt=attempt
        )
        return LunaWorkerResponse(_session_id_from_jsonl(stdout), raw, elapsed_ms)

    async def resume(
        self,
        *,
        binding: WorkerBinding,
        worker_session_id: str,
        request: Mapping[str, Any],
        attempt: int,
    ) -> LunaWorkerResponse:
        response_path = self._response_path(binding, request, attempt)
        command = (
            *self.command, "exec", "resume", worker_session_id, "--model", LUNA_MODEL,
            "--skip-git-repo-check", "--ignore-rules", "--json", "--output-last-message",
            str(response_path), "-",
        )
        raw, stdout, elapsed_ms = await self._invoke(
            command=command, binding=binding, request=request, attempt=attempt
        )
        emitted = _maybe_session_id(stdout)
        if emitted is not None and emitted != worker_session_id:
            raise SiloIsolationError("Codex resume emitted a different persistent thread id")
        return LunaWorkerResponse(worker_session_id, raw, elapsed_ms)


def _pending_request(plan: ConversationPlan, pending: RelayPendingRequest) -> tuple[dict[str, Any], str]:
    request = _benchmark_request_payload(pending.request)
    request_hash = _benchmark_request_fingerprint(pending.request)
    if (
        pending.request.conversation_id != plan.conversation_id
        or pending.request.case_id != plan.case.case_id
        or pending.request.actor_id not in plan.actor_ids
        or pending.request.model != LUNA_MODEL
    ):
        raise SiloIsolationError("relay pending request belongs to another silo")
    document = _read_json(plan.pending_path, label="pending request")
    if (
        document.get("sequence") != pending.sequence
        or document.get("request") != request
        or document.get("request_fingerprint") != request_hash
    ):
        raise SchedulerError("pending request does not preserve exact relay identity")
    ledger = _read_json(plan.ledger_path, label="response ledger")
    responses = ledger.get("responses")
    if (
        ledger.get("conversation_id") != plan.conversation_id
        or ledger.get("case_id") != plan.case.case_id
        or ledger.get("model") != LUNA_MODEL
        or not isinstance(responses, list)
        or len(responses) != pending.sequence
        or any(
            not isinstance(item, Mapping) or item.get("sequence") != index
            for index, item in enumerate(responses)
        )
    ):
        raise SchedulerError("response ledger continuity is invalid")
    return request, request_hash


def _event(
    binding: WorkerBinding,
    *,
    session_id: str | None,
    request_hash: str,
    sequence: int,
    attempt: int,
    status: str,
    raw_response: str | None,
    latency_ms: float,
    retryable: bool,
    rate_limited: bool,
) -> dict[str, Any]:
    return {
        "conversation_id": binding.conversation_id,
        "case_id": binding.case_id,
        "actor_id": binding.actor_id,
        "worker_session_id": session_id,
        "request_sha256": request_hash,
        "ledger_sequence": sequence,
        "attempt": attempt,
        "status": status,
        "raw_response": raw_response,
        "latency_ms": latency_ms,
        "retryable": retryable,
        "rate_limited": rate_limited,
    }


async def _worker_call(
    executor: LunaWorkerExecutor,
    audit: SchedulerAudit,
    binding: WorkerBinding,
    request: Mapping[str, Any],
    attempt: int,
) -> LunaWorkerResponse:
    session_id = audit.session(binding)
    if session_id is None:
        response = await executor.start(binding=binding, request=request, attempt=attempt)
        audit.bind(binding, response.worker_session_id)
    else:
        response = await executor.resume(
            binding=binding,
            worker_session_id=session_id,
            request=request,
            attempt=attempt,
        )
    if response.worker_session_id != audit.session(binding):
        raise SiloIsolationError("worker response does not match the owning session")
    return response


async def _append_pending_response(
    *,
    plan: ConversationPlan,
    pending: RelayPendingRequest,
    request: Mapping[str, Any],
    request_hash: str,
    executor: LunaWorkerExecutor,
    audit: SchedulerAudit,
    gate: AdaptiveGate,
    config: SchedulerConfig,
) -> None:
    binding = WorkerBinding(plan.conversation_id, plan.case.case_id, pending.request.actor_id)
    for attempt in range(1, config.max_technical_retries + 2):
        try:
            async with gate.slot():
                response = await _worker_call(executor, audit, binding, request, attempt)
        except LunaWorkerError as error:
            if error.session_id:
                audit.bind(binding, error.session_id)
            audit.record(
                _event(
                    binding,
                    session_id=audit.session(binding),
                    request_hash=request_hash,
                    sequence=pending.sequence,
                    attempt=attempt,
                    status="rate_limited" if error.rate_limited else "technical_error",
                    raw_response=None,
                    latency_ms=error.elapsed_ms,
                    retryable=error.retryable,
                    rate_limited=error.rate_limited,
                )
            )
            await gate.observe(accepted=False, rate_limited=error.rate_limited)
            if not error.retryable or attempt > config.max_technical_retries:
                raise SchedulerError("bounded technical retry exhausted") from error
            if config.retry_backoff_seconds:
                await asyncio.sleep(config.retry_backoff_seconds * attempt)
            continue
        try:
            if not isinstance(response.raw_response, str):
                raise CharacterAgentOutputError("worker response must be raw text")
            sanitize_character_public_text(response.raw_response)
        except CharacterAgentOutputError as error:
            audit.record(
                _event(
                    binding,
                    session_id=response.worker_session_id,
                    request_hash=request_hash,
                    sequence=pending.sequence,
                    attempt=attempt,
                    status="invalid_output_shape",
                    raw_response=(
                        response.raw_response
                        if isinstance(response.raw_response, str)
                        else None
                    ),
                    latency_ms=response.elapsed_ms,
                    retryable=True,
                    rate_limited=False,
                )
            )
            await gate.observe(accepted=False, rate_limited=False)
            if attempt > config.max_technical_retries:
                raise SchedulerError("bounded output-shape retry exhausted") from error
            if config.retry_backoff_seconds:
                await asyncio.sleep(config.retry_backoff_seconds * attempt)
            continue
        append_relay_response(plan.ledger_path, response.raw_response, pending_path=plan.pending_path)
        audit.record(
            _event(
                binding,
                session_id=response.worker_session_id,
                request_hash=request_hash,
                sequence=pending.sequence,
                attempt=attempt,
                status="accepted",
                raw_response=response.raw_response,
                latency_ms=response.elapsed_ms,
                retryable=False,
                rate_limited=False,
            )
        )
        await gate.observe(accepted=True, rate_limited=False)
        return
    raise AssertionError("bounded retry loop should return or raise")


async def _run_one(
    plan: ConversationPlan,
    *,
    executor: LunaWorkerExecutor,
    audit: SchedulerAudit,
    gate: AdaptiveGate,
    config: SchedulerConfig,
    manifest_sha256: str,
) -> ConversationResult:
    while True:
        try:
            result = await run_relay_conversation(
                plan.case,
                ledger_path=plan.ledger_path,
                pending_path=plan.pending_path,
                model=LUNA_MODEL,
                conversation_id=plan.conversation_id,
                manifest_fingerprint=manifest_sha256,
            )
        except RelayPendingRequest as pending:
            request, request_hash = _pending_request(plan, pending)
            await _append_pending_response(
                plan=plan,
                pending=pending,
                request=request,
                request_hash=request_hash,
                executor=executor,
                audit=audit,
                gate=gate,
                config=config,
            )
            await asyncio.sleep(0)
            continue
        if len(result.turns) != plan.expected_turn_count:
            raise SchedulerError("conversation stopped before its fixed turn count")
        return result


async def run_persistent_luna_silos(
    cases: Sequence[BenchmarkCase],
    *,
    output_dir: Path,
    run_id: str,
    manifest_sha256: str,
    config: SchedulerConfig = SchedulerConfig(),
    executor: LunaWorkerExecutor,
) -> SchedulerRunResult:
    manifest_sha256 = _frozen_sha256(manifest_sha256)
    plans = build_conversation_plans(cases, output_dir=output_dir, run_id=run_id, config=config)
    audit = SchedulerAudit(
        output_dir / "scheduler_state.json",
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        config=config,
    )
    gate = AdaptiveGate(config, audit)
    results = await asyncio.gather(
        *(
            _run_one(
                plan,
                executor=executor,
                audit=audit,
                gate=gate,
                config=config,
                manifest_sha256=manifest_sha256,
            )
            for plan in plans
        )
    )
    return SchedulerRunResult(tuple(results), audit.path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conversations", type=int, default=DEFAULT_CONVERSATION_COUNT)
    parser.add_argument("--initial-concurrency", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--fanout-after-successes", type=int, default=16)
    parser.add_argument("--no-auto-fanout", action="store_true")
    parser.add_argument("--max-technical-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--execute", action="store_true")
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    frozen_sha = _frozen_sha256(args.frozen_manifest_sha256)
    if _manifest_sha256(args.manifest) != frozen_sha:
        raise SystemExit("frozen manifest hash mismatch; refusing to start Luna sessions")
    config = SchedulerConfig(
        conversation_count=args.conversations,
        initial_concurrency=args.initial_concurrency,
        max_concurrency=args.max_concurrency,
        auto_fanout=not args.no_auto_fanout,
        fanout_after_successes=args.fanout_after_successes,
        max_technical_retries=args.max_technical_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    cases = load_benchmark_manifest(args.manifest)
    plans = build_conversation_plans(cases, output_dir=args.output, run_id=args.run_id, config=config)
    if not args.execute:
        print(json.dumps({"status": "validated_not_executed", "model": LUNA_MODEL, "conversation_count": len(plans), "worker_count": len(plans) * 2, "turn_count_per_conversation": FIXED_TURN_COUNT}))
        return 0
    result = await run_persistent_luna_silos(
        cases,
        output_dir=args.output,
        run_id=args.run_id,
        manifest_sha256=frozen_sha,
        config=config,
        executor=CodexLunaCliExecutor(
            workers_root=args.output / "worker_sessions",
            timeout_seconds=config.worker_timeout_seconds,
        ),
    )
    print(json.dumps({"status": "complete", "state": str(result.state_path), "conversation_count": len(result.results), "model": LUNA_MODEL}))
    return 0


def main() -> int:
    return asyncio.run(_run_cli(build_arg_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
