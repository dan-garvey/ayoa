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
import secrets
import shutil
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


SCHEDULER_SCHEMA_VERSION = "persistent_luna_dialogue_silos_v2"
LUNA_MODEL = "gpt-5.6-luna"
LUNA_REASONING_EFFORT_CHOICES = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
DEFAULT_CONVERSATION_COUNT = 8
FIXED_TURN_COUNT = 16
REFLECTION_FIELD_NAMES = (
    "present_true_position",
    "public_attempt",
    "deliberately_unsaid_truth",
    "unavailable_because",
    "relationship_status_cost",
    "continuity_pressure",
)
REFLECTION_MAX_SUFFIX_BYTES = 1200
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")
_REFLECTION_NONCE_RE = re.compile(r"R-[0-9a-f]{32}")
_REFLECTION_SUFFIX_RE = re.compile(
    r"\A(?P<body>.*?)\n+"
    r"(?P<suffix>\(<actor_private_reflection id=\"(?P<nonce>R-[0-9a-f]{32})\">"
    r"(?P<payload>[^\n]*)</actor_private_reflection>\))\n?\Z",
    re.DOTALL,
)
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


class ReflectionOutputError(SchedulerError):
    """An experiment-only private reflection violated its sealed contract."""


def _normalize_luna_reasoning_effort(value: str | None) -> str | None:
    """Validate a Codex effort without conflating it with an omitted override."""

    if value is None:
        return None
    if value not in LUNA_REASONING_EFFORT_CHOICES:
        supported = ", ".join(LUNA_REASONING_EFFORT_CHOICES)
        raise ValueError(f"luna_reasoning_effort must be one of: {supported}")
    return value


@dataclass(frozen=True)
class PrivateReflection:
    nonce: str
    fields: Mapping[str, str]
    suffix: str


@dataclass(frozen=True)
class ReflectionParseResult:
    public_body: str
    reflection: PrivateReflection


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
        self,
        *,
        binding: WorkerBinding,
        request: Mapping[str, Any],
        attempt: int,
        reflection_nonce: str | None = None,
        persistent_delta_proxy: bool = False,
    ) -> LunaWorkerResponse: ...

    async def resume(
        self,
        *,
        binding: WorkerBinding,
        worker_session_id: str,
        request: Mapping[str, Any],
        attempt: int,
        reflection_nonce: str | None = None,
        persistent_delta_proxy: bool = False,
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
    private_reflections: bool = False
    max_reflection_conversation_restarts: int = 1
    persistent_delta_proxy: bool = False
    system_prompt_override: str | None = None
    luna_reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "luna_reasoning_effort",
            _normalize_luna_reasoning_effort(self.luna_reasoning_effort),
        )
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
        if self.max_reflection_conversation_restarts < 0:
            raise ValueError("max_reflection_conversation_restarts cannot be negative")
        if self.system_prompt_override is not None:
            if not self.system_prompt_override.strip():
                raise ValueError("system_prompt_override cannot be blank")
            if (
                "<<<USER>>>" in self.system_prompt_override
                or "{request_packet}" in self.system_prompt_override
            ):
                raise ValueError(
                    "system_prompt_override must contain only system content"
                )


@dataclass(frozen=True)
class ConversationPlan:
    case: BenchmarkCase
    conversation_id: str
    ledger_path: Path
    pending_path: Path
    private_reflections_root: Path
    expected_turn_count: int

    @property
    def actor_ids(self) -> tuple[str, str]:
        actor_ids = tuple(actor.character_id for actor in self.case.actors)
        if len(actor_ids) != 2:
            raise SchedulerError("conversation plan is not dyadic")
        return actor_ids[0], actor_ids[1]

    def private_reflection_path(
        self, binding: WorkerBinding, *, sequence: int, attempt: int
    ) -> Path:
        if (
            binding.conversation_id != self.conversation_id
            or binding.case_id != self.case.case_id
        ):
            raise SiloIsolationError(
                "private reflection owner does not match conversation"
            )
        owner_hash = hashlib.sha256(_binding_key(binding).encode()).hexdigest()[:16]
        return (
            self.private_reflections_root
            / owner_hash
            / f"turn-{sequence:02d}-attempt-{attempt}.json"
        )


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
        raise SchedulerError(
            "frozen manifest SHA-256 must be 64 hexadecimal characters"
        )
    return value


def _safe_path_piece(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-") or "conversation"


def _binding_key(binding: WorkerBinding) -> str:
    return json.dumps(
        [binding.conversation_id, binding.actor_id], separators=(",", ":")
    )


def _reflection_artifact_payload(
    *,
    binding: WorkerBinding,
    sequence: int,
    attempt: int,
    status: str,
    raw_response: str,
    parsed: ReflectionParseResult | None,
    error: str | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "conversation_id": binding.conversation_id,
        "case_id": binding.case_id,
        "actor_id": binding.actor_id,
        "ledger_sequence": sequence,
        "attempt": attempt,
        "status": status,
        "raw_response": raw_response,
    }
    if parsed is not None:
        artifact["reflection"] = {
            "nonce": parsed.reflection.nonce,
            "suffix": parsed.reflection.suffix,
            "fields": dict(parsed.reflection.fields),
        }
    if error is not None:
        artifact["error"] = error
    return artifact


def _write_private_reflection_artifact(
    plan: ConversationPlan,
    *,
    binding: WorkerBinding,
    sequence: int,
    attempt: int,
    status: str,
    raw_response: str,
    parsed: ReflectionParseResult | None,
    error: str | None = None,
) -> Path:
    path = plan.private_reflection_path(binding, sequence=sequence, attempt=attempt)
    raw_hash = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()[:16]
    path = path.with_name(f"{path.stem}-{raw_hash}{path.suffix}")
    _atomic_write_json(
        path,
        _reflection_artifact_payload(
            binding=binding,
            sequence=sequence,
            attempt=attempt,
            status=status,
            raw_response=raw_response,
            parsed=parsed,
            error=error,
        ),
    )
    return path


def _parse_private_reflection(
    raw_response: str, *, expected_nonce: str
) -> ReflectionParseResult:
    """Split the sealed experiment suffix before public output parsing.

    This deliberately accepts no approximation: repairing an output would keep
    private material in a persistent actor session whose continuation is no
    longer trustworthy for the experiment.
    """

    if not _REFLECTION_NONCE_RE.fullmatch(expected_nonce):
        raise ReflectionOutputError("reflection nonce has an invalid shape")
    if not raw_response:
        raise ReflectionOutputError("reflection response is blank")
    match = _REFLECTION_SUFFIX_RE.fullmatch(raw_response)
    if match is None:
        raise ReflectionOutputError(
            "missing or malformed anchored private reflection suffix"
        )
    if (
        raw_response.count("<actor_private_reflection") != 1
        or raw_response.count("</actor_private_reflection>") != 1
    ):
        raise ReflectionOutputError(
            "private reflection marker is duplicated or appears in public prose"
        )
    public_body = match.group("body")
    if "actor_private_reflection" in public_body:
        raise ReflectionOutputError("private reflection marker appears in public prose")
    suffix = match.group("suffix")
    if len(suffix.encode("utf-8")) > REFLECTION_MAX_SUFFIX_BYTES:
        raise ReflectionOutputError(
            "private reflection suffix exceeds 1200 UTF-8 bytes"
        )
    nonce = match.group("nonce")
    if nonce != expected_nonce:
        raise ReflectionOutputError("private reflection nonce does not match this turn")
    payload = match.group("payload")
    try:
        decoded = json.loads(payload, object_pairs_hook=lambda pairs: pairs)
    except json.JSONDecodeError as error:
        raise ReflectionOutputError(
            "private reflection payload is not valid JSON"
        ) from error
    if not isinstance(decoded, list):
        raise ReflectionOutputError("private reflection payload must be a JSON object")
    keys = tuple(key for key, _ in decoded)
    if keys != REFLECTION_FIELD_NAMES:
        raise ReflectionOutputError(
            "private reflection keys must be exact, ordered, and unique"
        )
    fields: dict[str, str] = {}
    for key, value in decoded:
        if not isinstance(value, str):
            raise ReflectionOutputError(f"private reflection {key} must be a string")
        if (
            not 1 <= len(value) <= 180
            or value != value.strip()
            or any(character in _LINE_BREAKS for character in value)
            or any(ord(character) < 32 for character in value)
        ):
            raise ReflectionOutputError(
                f"private reflection {key} must be a trimmed one-line 1-180 character clause"
            )
        fields[key] = value
    if fields["present_true_position"] == "NONE" or fields["public_attempt"] == "NONE":
        raise ReflectionOutputError(
            "present_true_position and public_attempt cannot be NONE"
        )
    return ReflectionParseResult(
        public_body=public_body,
        reflection=PrivateReflection(nonce=nonce, fields=fields, suffix=suffix),
    )


def _assert_reflection_free(value: str, *, nonces: Sequence[str], label: str) -> None:
    forbidden = ("actor_private_reflection", *REFLECTION_FIELD_NAMES, *nonces)
    leaked = [item for item in forbidden if item and item in value]
    if leaked:
        raise SiloIsolationError(f"private reflection leaked into {label}: {leaked[0]}")


def _assert_public_artifacts_reflection_free(
    plan: ConversationPlan, audit: SchedulerAudit
) -> None:
    """Fail before a public artifact can silently preserve a sealed suffix."""

    records = audit.data.get("private_reflection_turns")
    if not isinstance(records, list):
        raise SchedulerError("private reflection telemetry is not enabled")
    nonces = tuple(
        record["nonce"]
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("nonce"), str)
    )
    output_root = plan.private_reflections_root.parent
    excluded = {plan.private_reflections_root.resolve(), audit.path.resolve()}
    if not output_root.exists():
        return
    for path in output_root.rglob("*"):
        if not path.is_file() or any(
            parent in excluded for parent in (path.resolve(), *path.resolve().parents)
        ):
            continue
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        _assert_reflection_free(contents, nonces=nonces, label=str(path))


def _discard_reflection_conversation(
    plan: ConversationPlan, audit: SchedulerAudit
) -> None:
    """Remove public relay state so both actors restart in fresh sealed silos."""

    conversation_root = plan.ledger_path.parent.resolve()
    expected_parent = (plan.private_reflections_root.parent / "conversations").resolve()
    if conversation_root.parent != expected_parent:
        raise SiloIsolationError("refusing to discard a conversation outside this run")
    if conversation_root.exists():
        shutil.rmtree(conversation_root)
    audit.discard_conversation_workers(plan)


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
        raise SchedulerError(
            "run_id and every scheduled case_id must be unique and nonblank"
        )
    plans = []
    for index, case in enumerate(cases, start=1):
        _validate_case(case, config.turn_count)
        root = (
            output_dir
            / "conversations"
            / f"{index:02d}-{_safe_path_piece(case.case_id)}"
        )
        ledger_path = root / "response_ledger.json"
        plans.append(
            ConversationPlan(
                case=case,
                conversation_id=f"{run_id}:{case.case_id}",
                ledger_path=ledger_path,
                pending_path=ledger_path.with_name(ledger_path.name + ".pending.json"),
                private_reflections_root=output_dir / "private_reflection_qa",
                expected_turn_count=config.turn_count,
            )
        )
    return tuple(plans)


def select_benchmark_cases(
    cases: Sequence[BenchmarkCase], case_ids: Sequence[str] | None = None
) -> tuple[BenchmarkCase, ...]:
    """Return an explicitly requested, ordered subset of one frozen manifest."""

    available: dict[str, BenchmarkCase] = {}
    for case in cases:
        if not case.case_id.strip() or case.case_id in available:
            raise SchedulerError("benchmark manifest case ids must be unique and nonblank")
        available[case.case_id] = case
    requested = tuple(case_ids or ())
    if not requested:
        return tuple(cases)
    if any(not case_id.strip() for case_id in requested):
        raise SchedulerError("--case-id cannot be blank")
    if len(set(requested)) != len(requested):
        raise SchedulerError("--case-id values must be unique")
    missing = [case_id for case_id in requested if case_id not in available]
    if missing:
        raise SchedulerError(f"--case-id is absent from the manifest: {missing[0]}")
    return tuple(available[case_id] for case_id in requested)


class SchedulerAudit:
    """Small durable ledger for worker ownership and technical telemetry."""

    def __init__(
        self,
        path: Path,
        *,
        manifest_sha256: str,
        run_id: str,
        config: SchedulerConfig,
        selected_case_ids: Sequence[str],
    ) -> None:
        if (
            len(selected_case_ids) != config.conversation_count
            or not all(case_id.strip() for case_id in selected_case_ids)
            or len(set(selected_case_ids)) != len(selected_case_ids)
        ):
            raise SchedulerError("scheduled case ids do not match the frozen run")
        self.path = path
        expected = {
            "manifest_sha256": _frozen_sha256(manifest_sha256),
            "run_id": run_id,
            "model": LUNA_MODEL,
            "conversation_count": config.conversation_count,
            "turn_count": config.turn_count,
            "selected_case_ids": list(selected_case_ids),
        }
        if config.private_reflections:
            expected["private_reflections"] = True
            expected["max_reflection_conversation_restarts"] = (
                config.max_reflection_conversation_restarts
            )
        if config.persistent_delta_proxy:
            expected["persistent_delta_proxy"] = True
        if config.system_prompt_override is not None:
            expected["system_prompt_override_sha256"] = hashlib.sha256(
                config.system_prompt_override.encode("utf-8")
            ).hexdigest()
        if config.luna_reasoning_effort is not None:
            expected["luna_reasoning_effort"] = config.luna_reasoning_effort
        if path.exists():
            self.data = _read_json(path, label="scheduler state")
            if self.data.get("schema_version") != SCHEDULER_SCHEMA_VERSION:
                raise SchedulerError(
                    "scheduler state has an unsupported schema version"
                )
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
            if config.private_reflections and not isinstance(
                self.data.get("private_reflection_turns"), list
            ):
                raise SchedulerError(
                    "scheduler state lacks private reflection telemetry"
                )
            if config.private_reflections and not isinstance(
                self.data.get("reflection_restarts"), dict
            ):
                raise SchedulerError("scheduler state lacks reflection restart counts")
            if config.private_reflections and any(
                not isinstance(count, int)
                or count < 0
                or count > config.max_reflection_conversation_restarts
                for count in self.data["reflection_restarts"].values()
            ):
                raise SchedulerError(
                    "private reflection restart budget is already exhausted"
                )
            if config.persistent_delta_proxy and not isinstance(
                self.data.get("inflight_delta_calls"), dict
            ):
                raise SchedulerError("scheduler state lacks delta in-flight markers")
            if config.persistent_delta_proxy and self.data["inflight_delta_calls"]:
                raise SchedulerError(
                    "an interrupted delta call has uncertain model state; "
                    "restart this cell in a fresh output directory"
                )
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
            if config.private_reflections:
                self.data["private_reflection_turns"] = []
                self.data["reflection_restarts"] = {}
            if config.persistent_delta_proxy:
                self.data["inflight_delta_calls"] = {}
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
                raise SiloIsolationError(
                    "one Codex session is assigned to multiple silos"
                )
            sessions.add(values[3])

    def session(self, binding: WorkerBinding) -> str | None:
        entry = self.data["workers"].get(_binding_key(binding))
        if entry is None:
            return None
        if not isinstance(entry, Mapping) or any(
            entry.get(field) != getattr(binding, field)
            for field in ("conversation_id", "case_id", "actor_id")
        ):
            raise SiloIsolationError(
                "worker ownership does not match its requested silo"
            )
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
                raise SiloIsolationError(
                    "a worker attempted to replace its persistent session"
                )
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

    def reserve_reflection_nonce(
        self,
        binding: WorkerBinding,
        *,
        sequence: int,
    ) -> str:
        records = self.data.get("private_reflection_turns")
        if not isinstance(records, list):
            raise SchedulerError("private reflection telemetry is not enabled")
        for record in reversed(records):
            if not isinstance(record, Mapping):
                raise SchedulerError("private reflection telemetry is malformed")
            if (
                record.get("conversation_id") == binding.conversation_id
                and record.get("actor_id") == binding.actor_id
                and record.get("ledger_sequence") == sequence
                and record.get("status") == "reserved"
            ):
                nonce = record.get("nonce")
                if isinstance(nonce, str) and _REFLECTION_NONCE_RE.fullmatch(nonce):
                    return nonce
                raise SchedulerError("reserved private reflection nonce is malformed")
        nonce = f"R-{secrets.token_hex(16)}"
        if any(
            isinstance(record, Mapping) and record.get("nonce") == nonce
            for record in records
        ):
            raise SchedulerError("private reflection nonce collision")
        records.append(
            {
                "conversation_id": binding.conversation_id,
                "case_id": binding.case_id,
                "actor_id": binding.actor_id,
                "ledger_sequence": sequence,
                "nonce": nonce,
                "status": "reserved",
            }
        )
        self.flush()
        return nonce

    def record_reflection_result(
        self,
        binding: WorkerBinding,
        *,
        sequence: int,
        nonce: str,
        status: str,
        public_response_sha256: str | None = None,
        suffix_sha256: str | None = None,
    ) -> None:
        records = self.data.get("private_reflection_turns")
        if not isinstance(records, list):
            raise SchedulerError("private reflection telemetry is not enabled")
        matches = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("conversation_id") == binding.conversation_id
            and record.get("actor_id") == binding.actor_id
            and record.get("ledger_sequence") == sequence
            and record.get("nonce") == nonce
        ]
        if len(matches) != 1:
            raise SchedulerError("private reflection result lacks its reserved nonce")
        record = matches[0]
        if record.get("status") != "reserved":
            raise SchedulerError("private reflection nonce was already finalized")
        record["status"] = status
        if public_response_sha256 is not None:
            record["public_response_sha256"] = public_response_sha256
        if suffix_sha256 is not None:
            record["suffix_sha256"] = suffix_sha256
        self.flush()

    def discard_conversation_workers(self, plan: ConversationPlan) -> None:
        stale = [
            key
            for key, entry in self.data["workers"].items()
            if isinstance(entry, Mapping)
            and entry.get("conversation_id") == plan.conversation_id
        ]
        for key in stale:
            del self.data["workers"][key]
        self.flush()

    def record_reflection_restart(self, plan: ConversationPlan) -> int:
        restarts = self.data.get("reflection_restarts")
        if not isinstance(restarts, dict):
            raise SchedulerError("private reflection restart telemetry is not enabled")
        current = restarts.get(plan.conversation_id, 0)
        if not isinstance(current, int) or current < 0:
            raise SchedulerError("private reflection restart count is malformed")
        current += 1
        restarts[plan.conversation_id] = current
        self.flush()
        return current

    def begin_delta_call(
        self,
        binding: WorkerBinding,
        *,
        sequence: int,
        attempt: int,
        request_sha256: str,
    ) -> None:
        calls = self.data.get("inflight_delta_calls")
        if not isinstance(calls, dict):
            raise SchedulerError("persistent delta telemetry is not enabled")
        key = _binding_key(binding)
        if key in calls:
            raise SchedulerError("a delta worker already has an in-flight call")
        calls[key] = {
            "conversation_id": binding.conversation_id,
            "case_id": binding.case_id,
            "actor_id": binding.actor_id,
            "ledger_sequence": sequence,
            "attempt": attempt,
            "request_sha256": request_sha256,
        }
        self.flush()

    def finish_delta_call(self, binding: WorkerBinding) -> None:
        calls = self.data.get("inflight_delta_calls")
        if not isinstance(calls, dict):
            raise SchedulerError("persistent delta telemetry is not enabled")
        if calls.pop(_binding_key(binding), None) is None:
            raise SchedulerError("delta worker has no matching in-flight call")
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
        if (
            not isinstance(self.limit, int)
            or not 1 <= self.limit <= config.max_concurrency
        ):
            raise SchedulerError("persisted concurrency limit is invalid")
        if (
            not isinstance(self.successes, int)
            or self.successes < 0
            or not isinstance(self.throttled, bool)
        ):
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


def _persistent_delta_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Project one full relay request to the next user packet for a live silo."""

    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SchedulerError("persistent delta proxy requires a nonempty message list")
    last_message = messages[-1]
    if not isinstance(last_message, Mapping) or last_message.get("role") != "user":
        raise SchedulerError(
            "persistent delta proxy requires the request to end in one user message"
        )
    required = ("conversation_id", "case_id", "actor_id", "turn_index")
    if any(not isinstance(request.get(key), (str, int)) for key in required):
        raise SchedulerError("persistent delta proxy request lacks binding metadata")
    return {
        "binding": {
            "conversation_id": request["conversation_id"],
            "case_id": request["case_id"],
            "actor_id": request["actor_id"],
        },
        "turn": {
            "scene_id": request.get("scene_id"),
            "scene_index": request.get("scene_index"),
            "scene_turn_index": request.get("scene_turn_index"),
            "turn_index": request["turn_index"],
        },
        "user_message": dict(last_message),
    }


def _replace_system_prompt(
    request: Mapping[str, Any], override: str | None
) -> Mapping[str, Any]:
    """Project an experiment-only system prompt without mutating relay truth."""

    if override is None:
        return request
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SchedulerError("system prompt override requires a message list")
    if not isinstance(messages[0], Mapping) or messages[0].get("role") != "system":
        raise SchedulerError("system prompt override requires a leading system message")
    projected = dict(request)
    projected_messages = [dict(message) for message in messages]
    projected_messages[0]["content"] = override
    projected["messages"] = projected_messages
    return projected


def _projected_proxy_request(
    request: Mapping[str, Any],
    *,
    persistent_delta_proxy: bool,
    starts_session: bool,
) -> Mapping[str, Any]:
    if not persistent_delta_proxy:
        return request
    delta = _persistent_delta_request(request)
    return request if starts_session else delta


def _proxy_prompt(
    request: Mapping[str, Any],
    *,
    reflection_nonce: str | None = None,
    persistent_delta_proxy: bool = False,
    starts_session: bool = True,
) -> str:
    projected = _projected_proxy_request(
        request,
        persistent_delta_proxy=persistent_delta_proxy,
        starts_session=starts_session,
    )
    payload = json.dumps(
        projected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if persistent_delta_proxy and not starts_session:
        wrapper: dict[str, Any] = {"request": projected}
        if reflection_nonce is not None:
            wrapper["expected_reflection_nonce"] = reflection_nonce
        return (
            "<persistent_character_agent_turn_delta>\n"
            f"{json.dumps(wrapper, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
            "</persistent_character_agent_turn_delta>\n"
        )
    if reflection_nonce is not None:
        if not _REFLECTION_NONCE_RE.fullmatch(reflection_nonce):
            raise SchedulerError("reflection proxy was given an invalid nonce")
        return (
            "Act only as an isolated CharacterAgent completion proxy. The embedded JSON is\n"
            "the full current continuation for one actor; use nothing outside it except your\n"
            "own earlier actor-private reflections in this same persistent session.\n\n"
            "Follow the embedded observable-body contract. After the complete observable\n"
            "body, append exactly one actor-private suffix using the supplied id and exact\n"
            "shape. No present witness can see this suffix. Never quote it, explain it, or\n"
            "let its field names enter the scene prose.\n\n"
            "The final line must keep these outer parentheses, tag, JSON keys, and key\n"
            "order exactly; replace only the six quoted value descriptions:\n"
            f'(<actor_private_reflection id="{reflection_nonce}">'
            '{"present_true_position":"one concrete clause",'
            '"public_attempt":"one concrete clause",'
            '"deliberately_unsaid_truth":"one concrete clause or NONE",'
            '"unavailable_because":"one concrete clause or NONE",'
            '"relationship_status_cost":"one concrete clause or NONE",'
            '"continuity_pressure":"one concrete clause or NONE"}'
            "</actor_private_reflection>)\n"
            "Do not rename, reorder, omit, or add keys, and keep that suffix on one\n"
            "physical line.\n\n"
            "Use one short concrete clause per value, grounded only in established actor\n"
            "knowledge and the public exchange. Use NONE rather than inventing a withheld\n"
            "truth, reason, cost, or continuity pressure. Distinguish the actor's actual\n"
            "present position from the public face or tactic just used; do not retrofit a\n"
            "virtuous or coherent motive merely to justify the line. unavailable_because\n"
            "means why an established truth cannot be risked now, not why it is unknown.\n"
            "relationship_status_cost names a cost incurred or risked by this move or\n"
            "withholding. continuity_pressure names one unresolved pressure to remember,\n"
            "not a script, theme, rubric, or prediction of another person's behavior.\n\n"
            "Return no analysis, markdown, labels, or text after the suffix.\n"
            + (
                "Future messages in this persistent session contain only the newest "
                "user packet and its binding; apply this same contract without repeating it.\n"
                if persistent_delta_proxy
                else ""
            )
            + f"<actor_private_reflection_id>{reflection_nonce}</actor_private_reflection_id>\n"
            f"<exact_character_agent_request>\n{payload}\n</exact_character_agent_request>\n"
        )
    return (
        "Act only as an isolated CharacterAgent completion proxy. Return exactly the "
        "raw assistant content for this request: no analysis, labels, markdown, "
        "tools, file inspection, or commentary. The JSON is the full current "
        "continuation for one actor; use nothing outside it.\n"
        + (
            "Future messages in this persistent session contain only the newest user "
            "packet and its binding; apply this same contract without repeating it.\n"
            if persistent_delta_proxy
            else ""
        )
        + f"<exact_character_agent_request>\n{payload}\n</exact_character_agent_request>\n"
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
        reasoning_effort: str | None = None,
    ) -> None:
        if not command:
            raise ValueError("Codex command cannot be empty")
        self.workers_root = workers_root
        self.timeout_seconds = timeout_seconds
        self.command = tuple(command)
        self.reasoning_effort = _normalize_luna_reasoning_effort(reasoning_effort)

    def _reasoning_effort_args(self) -> tuple[str, ...]:
        if self.reasoning_effort is None:
            return ()
        return ("-c", f'model_reasoning_effort="{self.reasoning_effort}"')

    def _start_command(self, response_path: Path) -> tuple[str, ...]:
        return (
            *self.command,
            "exec",
            *self._reasoning_effort_args(),
            "--model",
            LUNA_MODEL,
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ignore-rules",
            "--json",
            "--output-last-message",
            str(response_path),
            "-",
        )

    def _resume_command(
        self, worker_session_id: str, response_path: Path
    ) -> tuple[str, ...]:
        return (
            *self.command,
            "exec",
            "resume",
            *self._reasoning_effort_args(),
            worker_session_id,
            "--model",
            LUNA_MODEL,
            "--skip-git-repo-check",
            "--ignore-rules",
            "--json",
            "--output-last-message",
            str(response_path),
            "-",
        )

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
        reflection_nonce: str | None,
        persistent_delta_proxy: bool,
        starts_session: bool,
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
                process.communicate(
                    _proxy_prompt(request, reflection_nonce=reflection_nonce).encode()
                    if not persistent_delta_proxy
                    else _proxy_prompt(
                        request,
                        reflection_nonce=reflection_nonce,
                        persistent_delta_proxy=True,
                        starts_session=starts_session,
                    ).encode()
                ),
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
                "Codex rate limited"
                if _RATE_LIMIT_RE.search(detail)
                else "Codex call failed",
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
        self,
        *,
        binding: WorkerBinding,
        request: Mapping[str, Any],
        attempt: int,
        reflection_nonce: str | None = None,
        persistent_delta_proxy: bool = False,
    ) -> LunaWorkerResponse:
        response_path = self._response_path(binding, request, attempt)
        command = self._start_command(response_path)
        raw, stdout, elapsed_ms = await self._invoke(
            command=command,
            binding=binding,
            request=request,
            attempt=attempt,
            reflection_nonce=reflection_nonce,
            persistent_delta_proxy=persistent_delta_proxy,
            starts_session=True,
        )
        return LunaWorkerResponse(_session_id_from_jsonl(stdout), raw, elapsed_ms)

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
        response_path = self._response_path(binding, request, attempt)
        command = self._resume_command(worker_session_id, response_path)
        raw, stdout, elapsed_ms = await self._invoke(
            command=command,
            binding=binding,
            request=request,
            attempt=attempt,
            reflection_nonce=reflection_nonce,
            persistent_delta_proxy=persistent_delta_proxy,
            starts_session=False,
        )
        emitted = _maybe_session_id(stdout)
        if emitted is not None and emitted != worker_session_id:
            raise SiloIsolationError(
                "Codex resume emitted a different persistent thread id"
            )
        return LunaWorkerResponse(worker_session_id, raw, elapsed_ms)


def _pending_request(
    plan: ConversationPlan, pending: RelayPendingRequest
) -> tuple[dict[str, Any], str]:
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
    reflection_nonce: str | None = None,
    public_response_sha256: str | None = None,
    suffix_sha256: str | None = None,
    projected_request_sha256: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
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
    if reflection_nonce is not None:
        event["reflection_nonce"] = reflection_nonce
    if public_response_sha256 is not None:
        event["public_response_sha256"] = public_response_sha256
    if suffix_sha256 is not None:
        event["suffix_sha256"] = suffix_sha256
    if projected_request_sha256 is not None:
        event["projected_request_sha256"] = projected_request_sha256
    return event


async def _worker_call(
    executor: LunaWorkerExecutor,
    audit: SchedulerAudit,
    binding: WorkerBinding,
    request: Mapping[str, Any],
    attempt: int,
    reflection_nonce: str | None = None,
    persistent_delta_proxy: bool = False,
) -> LunaWorkerResponse:
    session_id = audit.session(binding)
    if session_id is None:
        response = await executor.start(
            binding=binding,
            request=request,
            attempt=attempt,
            reflection_nonce=reflection_nonce,
            persistent_delta_proxy=persistent_delta_proxy,
        )
        audit.bind(binding, response.worker_session_id)
    else:
        response = await executor.resume(
            binding=binding,
            worker_session_id=session_id,
            request=request,
            attempt=attempt,
            reflection_nonce=reflection_nonce,
            persistent_delta_proxy=persistent_delta_proxy,
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
    binding = WorkerBinding(
        plan.conversation_id, plan.case.case_id, pending.request.actor_id
    )
    reflection_nonce = (
        audit.reserve_reflection_nonce(binding, sequence=pending.sequence)
        if config.private_reflections
        else None
    )
    proxy_request = _replace_system_prompt(request, config.system_prompt_override)
    if config.private_reflections:
        _assert_reflection_free(
            json.dumps(request, ensure_ascii=False, sort_keys=True),
            nonces=tuple(
                record["nonce"]
                for record in audit.data["private_reflection_turns"]
                if isinstance(record, Mapping) and isinstance(record.get("nonce"), str)
            ),
            label="canonical inner request",
        )
    for attempt in range(1, config.max_technical_retries + 2):
        starts_session = audit.session(binding) is None
        projected_request_sha256 = (
            _sha256_json(
                _projected_proxy_request(
                    proxy_request,
                    persistent_delta_proxy=True,
                    starts_session=starts_session,
                )
            )
            if config.persistent_delta_proxy
            else None
        )
        delta_call_started = False
        try:
            async with gate.slot():
                if config.persistent_delta_proxy:
                    audit.begin_delta_call(
                        binding,
                        sequence=pending.sequence,
                        attempt=attempt,
                        request_sha256=request_hash,
                    )
                    delta_call_started = True
                response = await _worker_call(
                    executor,
                    audit,
                    binding,
                    proxy_request,
                    attempt,
                    reflection_nonce=reflection_nonce,
                    persistent_delta_proxy=config.persistent_delta_proxy,
                )
        except LunaWorkerError as error:
            if error.session_id:
                audit.bind(binding, error.session_id)
            retryable = error.retryable and (
                not config.persistent_delta_proxy
                or (starts_session and audit.session(binding) is None)
            )
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
                    retryable=retryable,
                    rate_limited=error.rate_limited,
                    projected_request_sha256=projected_request_sha256,
                )
            )
            if delta_call_started:
                audit.finish_delta_call(binding)
            await gate.observe(accepted=False, rate_limited=error.rate_limited)
            if not retryable or attempt > config.max_technical_retries:
                raise SchedulerError("bounded technical retry exhausted") from error
            if config.retry_backoff_seconds:
                await asyncio.sleep(config.retry_backoff_seconds * attempt)
            continue
        try:
            parsed_reflection: ReflectionParseResult | None = None
            if not isinstance(response.raw_response, str):
                raise CharacterAgentOutputError("worker response must be raw text")
            parsed_reflection = (
                _parse_private_reflection(
                    response.raw_response,
                    expected_nonce=reflection_nonce,
                )
                if reflection_nonce is not None
                else None
            )
            public_response = (
                parsed_reflection.public_body
                if parsed_reflection is not None
                else response.raw_response
            )
            sanitize_character_public_text(public_response)
        except ReflectionOutputError as error:
            if reflection_nonce is None or not isinstance(response.raw_response, str):
                raise AssertionError(
                    "reflection parser ran without a raw response and nonce"
                )
            _write_private_reflection_artifact(
                plan,
                binding=binding,
                sequence=pending.sequence,
                attempt=attempt,
                status="rejected_malformed_suffix",
                raw_response=response.raw_response,
                parsed=None,
                error=str(error),
            )
            audit.record_reflection_result(
                binding,
                sequence=pending.sequence,
                nonce=reflection_nonce,
                status="rejected_malformed_suffix",
            )
            audit.record(
                _event(
                    binding,
                    session_id=response.worker_session_id,
                    request_hash=request_hash,
                    sequence=pending.sequence,
                    attempt=attempt,
                    status="rejected_malformed_suffix",
                    raw_response=None,
                    latency_ms=response.elapsed_ms,
                    retryable=False,
                    rate_limited=False,
                    reflection_nonce=reflection_nonce,
                    projected_request_sha256=projected_request_sha256,
                )
            )
            if delta_call_started:
                audit.finish_delta_call(binding)
            await gate.observe(accepted=False, rate_limited=False)
            raise
        except CharacterAgentOutputError as error:
            if reflection_nonce is not None:
                if isinstance(response.raw_response, str):
                    _write_private_reflection_artifact(
                        plan,
                        binding=binding,
                        sequence=pending.sequence,
                        attempt=attempt,
                        status="rejected_public_output_shape",
                        raw_response=response.raw_response,
                        parsed=parsed_reflection,
                        error=str(error),
                    )
                audit.record_reflection_result(
                    binding,
                    sequence=pending.sequence,
                    nonce=reflection_nonce,
                    status="rejected_public_output_shape",
                    suffix_sha256=(
                        hashlib.sha256(
                            parsed_reflection.reflection.suffix.encode()
                        ).hexdigest()
                        if parsed_reflection is not None
                        else None
                    ),
                )
            retryable = not config.persistent_delta_proxy
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
                        and reflection_nonce is None
                        else None
                    ),
                    latency_ms=response.elapsed_ms,
                    retryable=retryable,
                    rate_limited=False,
                    reflection_nonce=reflection_nonce,
                    suffix_sha256=(
                        hashlib.sha256(
                            parsed_reflection.reflection.suffix.encode()
                        ).hexdigest()
                        if parsed_reflection is not None
                        else None
                    ),
                    projected_request_sha256=projected_request_sha256,
                )
            )
            if delta_call_started:
                audit.finish_delta_call(binding)
            await gate.observe(accepted=False, rate_limited=False)
            if reflection_nonce is not None:
                raise ReflectionOutputError(
                    "reflection-bearing response had invalid public output"
                ) from error
            if config.persistent_delta_proxy:
                raise SchedulerError(
                    "invalid output contaminated a persistent delta session"
                ) from error
            if attempt > config.max_technical_retries:
                raise SchedulerError("bounded output-shape retry exhausted") from error
            if config.retry_backoff_seconds:
                await asyncio.sleep(config.retry_backoff_seconds * attempt)
            continue
        append_relay_response(
            plan.ledger_path, public_response, pending_path=plan.pending_path
        )
        public_response_sha256 = hashlib.sha256(public_response.encode()).hexdigest()
        suffix_sha256 = (
            hashlib.sha256(parsed_reflection.reflection.suffix.encode()).hexdigest()
            if parsed_reflection is not None
            else None
        )
        if parsed_reflection is not None:
            _write_private_reflection_artifact(
                plan,
                binding=binding,
                sequence=pending.sequence,
                attempt=attempt,
                status="accepted",
                raw_response=response.raw_response,
                parsed=parsed_reflection,
            )
            audit.record_reflection_result(
                binding,
                sequence=pending.sequence,
                nonce=parsed_reflection.reflection.nonce,
                status="accepted",
                public_response_sha256=public_response_sha256,
                suffix_sha256=suffix_sha256,
            )
        audit.record(
            _event(
                binding,
                session_id=response.worker_session_id,
                request_hash=request_hash,
                sequence=pending.sequence,
                attempt=attempt,
                status="accepted",
                raw_response=response.raw_response
                if reflection_nonce is None
                else None,
                latency_ms=response.elapsed_ms,
                retryable=False,
                rate_limited=False,
                reflection_nonce=reflection_nonce,
                public_response_sha256=(
                    public_response_sha256 if reflection_nonce is not None else None
                ),
                suffix_sha256=suffix_sha256,
                projected_request_sha256=projected_request_sha256,
            )
        )
        if reflection_nonce is not None:
            _assert_public_artifacts_reflection_free(plan, audit)
        if delta_call_started:
            audit.finish_delta_call(binding)
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
            try:
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
            except ReflectionOutputError as error:
                _discard_reflection_conversation(plan, audit)
                reflection_restarts = audit.record_reflection_restart(plan)
                if reflection_restarts > config.max_reflection_conversation_restarts:
                    raise SchedulerError(
                        "repeated private reflection shape failure invalidated the conversation"
                    ) from error
                continue
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
    plans = build_conversation_plans(
        cases, output_dir=output_dir, run_id=run_id, config=config
    )
    audit = SchedulerAudit(
        output_dir / "scheduler_state.json",
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        config=config,
        selected_case_ids=tuple(plan.case.case_id for plan in plans),
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
    parser.add_argument(
        "--case-id",
        action="append",
        help="repeatable ordered case id selector from the frozen manifest",
    )
    parser.add_argument(
        "--conversations",
        type=int,
        help="scheduled conversation count; defaults to the selected case count",
    )
    parser.add_argument("--initial-concurrency", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--fanout-after-successes", type=int, default=16)
    parser.add_argument("--no-auto-fanout", action="store_true")
    parser.add_argument("--max-technical-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--private-reflections",
        action="store_true",
        help="enable the experiment-only sealed actor-private reflection suffix",
    )
    parser.add_argument(
        "--persistent-delta-proxy",
        "--persistent-delta",
        dest="persistent_delta_proxy",
        action="store_true",
        help="send only the newest user packet when resuming a persistent worker",
    )
    parser.add_argument(
        "--system-prompt-override",
        type=Path,
        help="experiment-only system-message body used instead of the rendered prompt",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=LUNA_REASONING_EFFORT_CHOICES,
        default=None,
        help="explicit Codex Luna reasoning setting; omit to preserve the provider default",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    frozen_sha = _frozen_sha256(args.frozen_manifest_sha256)
    if _manifest_sha256(args.manifest) != frozen_sha:
        raise SystemExit(
            "frozen manifest hash mismatch; refusing to start Luna sessions"
        )
    try:
        cases = select_benchmark_cases(
            load_benchmark_manifest(args.manifest), args.case_id
        )
    except SchedulerError as error:
        raise SystemExit(str(error)) from error
    conversation_count = (
        args.conversations if args.conversations is not None else len(cases)
    )
    system_prompt_override = None
    if args.system_prompt_override is not None:
        try:
            system_prompt_override = args.system_prompt_override.read_text(
                encoding="utf-8"
            )
        except OSError as error:
            raise SystemExit(f"cannot read system prompt override: {error}") from error
    config = SchedulerConfig(
        conversation_count=conversation_count,
        initial_concurrency=args.initial_concurrency,
        max_concurrency=args.max_concurrency,
        auto_fanout=not args.no_auto_fanout,
        fanout_after_successes=args.fanout_after_successes,
        max_technical_retries=args.max_technical_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        worker_timeout_seconds=args.worker_timeout_seconds,
        private_reflections=args.private_reflections,
        persistent_delta_proxy=args.persistent_delta_proxy,
        system_prompt_override=system_prompt_override,
        luna_reasoning_effort=args.reasoning_effort,
    )
    plans = build_conversation_plans(
        cases, output_dir=args.output, run_id=args.run_id, config=config
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "validated_not_executed",
                    "model": LUNA_MODEL,
                    "conversation_count": len(plans),
                    "worker_count": len(plans) * 2,
                    "turn_count_per_conversation": FIXED_TURN_COUNT,
                    "selected_case_ids": [plan.case.case_id for plan in plans],
                    "luna_reasoning_effort": config.luna_reasoning_effort,
                    "system_prompt_override_sha256": (
                        hashlib.sha256(
                            system_prompt_override.encode("utf-8")
                        ).hexdigest()
                        if system_prompt_override is not None
                        else None
                    ),
                }
            )
        )
        return 0
    result = await run_persistent_luna_silos(
        cases,
        output_dir=args.output,
        run_id=args.run_id,
        manifest_sha256=frozen_sha,
        config=config,
        executor=CodexLunaCliExecutor(
            workers_root=(
                args.output / "private_reflection_qa" / "worker_sessions"
                if config.private_reflections
                else args.output / "worker_sessions"
            ),
            timeout_seconds=config.worker_timeout_seconds,
            reasoning_effort=config.luna_reasoning_effort,
        ),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "state": str(result.state_path),
                "conversation_count": len(result.results),
                "model": LUNA_MODEL,
                "luna_reasoning_effort": config.luna_reasoning_effort,
            }
        )
    )
    return 0


def main() -> int:
    return asyncio.run(_run_cli(build_arg_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
