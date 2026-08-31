#!/usr/bin/env python3
# ruff: noqa: E402 -- executable script adds the repository root to sys.path.
"""Run an offline or explicit-model serial character dialogue benchmark.

This is a small laboratory for the actor boundary, not a second game engine.
Each actor receives its own lived material and its own complete rolling
conversation.  The ensemble then meets in at least two separated scenes.  The
public candidate prose from the first scene is part of the second scene's
input, so reviewers can see whether a harmless routine or line acquires a
different meaning later.

The only output contract is observable prose or the exact token ``<silence/>``.
There is no hidden carry channel, evaluator-facing prompt mode, or alternate
format flag.  The default is deterministic and offline.  ``--live`` is the
only path that constructs an LLM client, and ``--model`` is always passed to
the actor role explicitly (Luna is the inexpensive default).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import inspect
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from pydantic import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.engine.character_agent import CharacterAgent
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.llm.config import LLMConfig
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import (
    ModelConfig,
    SessionConfig,
    SessionSettings,
    SessionState,
    WorldState,
)


DEFAULT_MODEL = "gpt-5.6-luna"
SUITES = ("ordinary_surface", "pressure")
DEFAULT_MANIFEST_PATH = REPO_ROOT / "scripts" / "character_dialogue_benchmark_manifest.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "app" / "storage" / "playtest_reports" / "character-dialogue-benchmark"
)
EXACT_SILENCE = "<silence/>"

# The relay is deliberately a small, provider-free adapter around the same
# CharacterAgent call used by the normal benchmark.  Keep its file format
# versioned: a response ledger is an experiment artifact, not an implicit
# compatibility format for arbitrary old checkpoints.
RELAY_LEDGER_SCHEMA_VERSION = "character_dialogue_benchmark_response_ledger_v1"
RELAY_PENDING_REQUEST_SCHEMA_VERSION = (
    "character_dialogue_benchmark_pending_request_v1"
)
RELAY_PENDING_EXIT_CODE = 75

# These are intentionally reviewer-only.  They never occur in actor system or
# user messages.  A reviewer gets the complete serial transcript and fills in
# the fields by hand rather than having a second model grade a line in
# isolation.
HUMAN_REVIEW_FIELDS = (
    "setup_and_model_authorship",
    "physical_continuity",
    "attempts",
    "literal_and_interpersonal_subject",
    "knowledge_sources_and_unknowns",
    "unavailable_line",
    "status_and_topic_control",
    "answer_debt",
    "misreading_and_repair_cost",
    "ritual_deviation",
    "rhythm",
    "biography_consequence",
    "conversation_change",
    "articulation_ceiling",
    "voice_swappability",
)

HUMAN_REVIEW_QUESTIONS: Mapping[str, str] = {
    "setup_and_model_authorship": "Which facts, outcomes, symbols, or repairs were supplied by the setup, and which changes were actually earned by the speakers?",
    "physical_continuity": "Do people and persistent objects have valid locations, holders, conditions, and observable transitions throughout the sequence?",
    "attempts": "What is each speaker trying to make the other person do, admit, or avoid at each point?",
    "literal_and_interpersonal_subject": "What is the ordinary subject on the surface, and what are the speakers doing to one another through it?",
    "knowledge_sources_and_unknowns": "For each consequential fact, what does each speaker know, suspect, misunderstand, or not know, and how could they know it?",
    "unavailable_line": "What plain sentence could each speaker say but cannot presently afford to say?",
    "status_and_topic_control": "Who chooses the topic, who must answer, who may refuse, and where does that control change?",
    "answer_debt": "Which question, offer, correction, or bid goes unanswered, and what later line inherits that debt?",
    "misreading_and_repair_cost": "Which reasonable misreading or overreach survives beyond one response, and what does any repair cost or fail to restore?",
    "ritual_deviation": "What repeated routine establishes their relationship, and what does a deviation from it cost?",
    "rhythm": "How do length, interruption, silence, repetition, and pressure or release shape the whole sequence?",
    "biography_consequence": "Which lived detail changes a present choice, and which details are only decorative explanation?",
    "conversation_change": "What permission, obligation, belief, plan, or relationship is materially different when the sequence ends?",
    "articulation_ceiling": "Do the speakers differ in precision, wit, emotional insight, and willingness to explain, or do both sound equally model-like?",
    "voice_swappability": "Without names, what attention, vocabulary, rhythm, or social behavior identifies each speaker? Could either deliver the other’s lines unchanged?",
}

HUMAN_REVIEW_STATUSES = (
    "invalid_setup_or_relay",
    "invalid_physical_state",
    "invalid_prompt_contract",
    "human_quality_pass",
    "human_quality_fail",
    "prompt_architecture_candidate",
    "model_strength_candidate",
    "unresolved_variance",
)


class BenchmarkManifestError(ValueError):
    """The benchmark manifest is malformed or missing required data."""


@dataclass(frozen=True)
class PriorPublicExchange:
    """A concrete, witnessed exchange available to both actors."""

    sequence: int
    speaker_slot: int
    text: str


@dataclass(frozen=True)
class PressurePulse:
    pulse_id: str
    after_turn: int
    text: str


@dataclass(frozen=True)
class SceneSpec:
    """One meeting in a serial benchmark case."""

    scene_id: str
    title: str
    frame: str
    prior_public_exchange: tuple[PriorPublicExchange, ...]
    turn_order: tuple[str, ...]
    pressure_pulses: tuple[PressurePulse, ...] = ()
    between_scene_public_history: str = ""
    # Actor-local setup material is queued through the same inbox as runtime
    # observations.  It is deliberately absent from the public transcript
    # and from the reviewer-facing scene artifact; the exact user-tail prompt
    # and committed actor history remain in the raw run artifact.
    actor_observations: Mapping[str, tuple[str, ...]] = dataclass_field(
        default_factory=dict
    )


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    title: str
    suite: str
    scenes: tuple[SceneSpec, ...]
    actors: tuple[CharacterRecord, ...]
    source_metadata: Mapping[str, Any]

    def actor(self, actor_id: str) -> CharacterRecord:
        for actor in self.actors:
            if actor.character_id == actor_id:
                return actor
        raise BenchmarkManifestError(
            f"case {self.case_id!r} has no actor {actor_id!r}"
        )


@dataclass(frozen=True)
class BenchmarkRequest:
    """Exact production CharacterAgent request plus benchmark provenance."""

    conversation_id: str
    case_id: str
    scene_id: str
    scene_index: int
    scene_turn_index: int
    turn_index: int
    actor_id: str
    actor_name: str
    model: str
    role: str
    messages: tuple[Mapping[str, Any], ...]
    temperature: float
    max_tokens: int
    cache: bool
    compact: bool


@dataclass(frozen=True)
class ModelCall:
    """Provider-independent response envelope used by the harness."""

    content: str
    model: str = ""
    provider: str = "offline"
    usage: Mapping[str, Any] | None = None
    raw_response: Any = None
    assistant_content: Any = None


@dataclass(frozen=True)
class TurnResult:
    scene_index: int
    scene_id: str
    scene_turn_index: int
    turn_index: int
    actor_id: str
    actor_name: str
    public_text: str
    raw_response: str
    prompt: tuple[Mapping[str, Any], ...]
    provider_response: Mapping[str, Any]
    response_model: str
    provider: str
    usage: Mapping[str, Any]
    elapsed_ms: float
    prompt_sha256: str
    response_sha256: str
    pressure_pulse_ids: tuple[str, ...]
    role: str
    cache: bool
    compact: bool


@dataclass
class ConversationResult:
    conversation_id: str
    case: BenchmarkCase
    model: str
    checkpoint: CheckpointFile
    initial_checkpoint_sha256: str
    turns: list[TurnResult]
    public_transcript: list[dict[str, Any]]

    @property
    def checkpoint_id(self) -> str:
        return self.checkpoint.session.session_id

    def artifact(self) -> dict[str, Any]:
        """Return raw calls plus the full public serial transcript."""

        return {
            "schema_version": "character_dialogue_benchmark_artifact_v3",
            "conversation_id": self.conversation_id,
            "case_id": self.case.case_id,
            "title": self.case.title,
            "model": self.model,
            "source_metadata": _json_safe(self.case.source_metadata),
            "suite": self.case.suite,
            "scenes": [_scene_artifact(scene) for scene in self.case.scenes],
            "checkpoint": {
                "session_id": self.checkpoint_id,
                "ruleset_id": self.checkpoint.session.config.settings.ruleset_id,
                "initial_sha256": self.initial_checkpoint_sha256,
                "final": self.checkpoint.model_dump(mode="json"),
            },
            "public_transcript": self.public_transcript,
            "turns": [
                {
                    "scene_index": turn.scene_index,
                    "scene_id": turn.scene_id,
                    "scene_turn_index": turn.scene_turn_index,
                    "turn_index": turn.turn_index,
                    "actor_id": turn.actor_id,
                    "actor_name": turn.actor_name,
                    "public_text": turn.public_text,
                    "raw_response": turn.raw_response,
                    "prompt": [dict(message) for message in turn.prompt],
                    "provider_response": _json_safe(turn.provider_response),
                    "response_model": turn.response_model,
                    "provider": turn.provider,
                    "usage": _json_safe(turn.usage),
                    "elapsed_ms": turn.elapsed_ms,
                    "prompt_sha256": turn.prompt_sha256,
                    "response_sha256": turn.response_sha256,
                    "request": {
                        "model": self.model,
                        "role": turn.role,
                        "scene_index": turn.scene_index,
                        "scene_id": turn.scene_id,
                        "scene_turn_index": turn.scene_turn_index,
                        "turn_index": turn.turn_index,
                        "temperature": 0.6,
                        "max_tokens": 2000,
                        "cache": turn.cache,
                        "compact": turn.compact,
                        "messages": [dict(message) for message in turn.prompt],
                    },
                    "pressure_pulse_ids": list(turn.pressure_pulse_ids),
                }
                for turn in self.turns
            ],
            "review": build_human_review_packet(self)["review"],
            "review_policy": {
                "reviewer": "human",
                "model_judge": False,
                "auto_semantic_score": False,
                "unit": "whole_serial_conversation",
            },
        }


def _scene_artifact(scene: SceneSpec) -> dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "title": scene.title,
        "frame": scene.frame,
        "between_scene_public_history": scene.between_scene_public_history,
        "prior_public_exchange": [
            {
                "sequence": entry.sequence,
                "speaker_slot": entry.speaker_slot,
                "text": entry.text,
            }
            for entry in scene.prior_public_exchange
        ],
        "turn_order": list(scene.turn_order),
        "pressure_pulses": [
            {
                "pulse_id": pulse.pulse_id,
                "after_turn": pulse.after_turn,
                "text": pulse.text,
            }
            for pulse in scene.pressure_pulses
        ],
    }


def _json_safe(value: Any) -> Any:
    """Make provider metadata safe to write without leaking SDK objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump(mode="json"))
        except TypeError:
            return _json_safe(model_dump())
    return repr(value)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(encoded)


class RelayLedgerError(ValueError):
    """A response ledger cannot be safely replayed against this benchmark."""


class RelayPendingRequest(RuntimeError):
    """The production CharacterAgent reached a call with no ledger response."""

    def __init__(self, request: BenchmarkRequest, sequence: int) -> None:
        self.request = request
        self.sequence = sequence
        super().__init__(
            "response ledger is missing model response "
            f"at sequence {sequence} for actor {request.actor_id!r}"
        )


def _benchmark_request_payload(request: BenchmarkRequest) -> dict[str, Any]:
    """Return the JSON form of the exact request sent to CharacterAgent's client."""

    # ``messages`` is intentionally copied at this boundary.  Prompt builders
    # may retain mutable content blocks, while a pending request must remain a
    # byte-stable record of what the production client saw.
    return {
        "conversation_id": request.conversation_id,
        "case_id": request.case_id,
        "scene_id": request.scene_id,
        "scene_index": request.scene_index,
        "scene_turn_index": request.scene_turn_index,
        "turn_index": request.turn_index,
        "actor_id": request.actor_id,
        "actor_name": request.actor_name,
        "model": request.model,
        "role": request.role,
        "messages": copy.deepcopy(_json_safe(request.messages)),
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "cache": request.cache,
        "compact": request.compact,
    }


def _benchmark_request_fingerprint(request: BenchmarkRequest) -> str:
    return _sha256_json(_benchmark_request_payload(request))


def _case_fingerprint(case: BenchmarkCase) -> str:
    """Hash every manifest value that can affect a production actor prompt."""

    return _sha256_json(
        {
            "case_id": case.case_id,
            "title": case.title,
            "suite": case.suite,
            "source_metadata": case.source_metadata,
            "actors": [
                actor.model_dump(mode="json") for actor in case.actors
            ],
            "scenes": [
                {
                    **_scene_artifact(scene),
                    # Startup observations are intentionally omitted from the
                    # public/raw scene artifact, but they still belong in the
                    # relay identity because they change the actor prompt.
                    "actor_observations": {
                        actor_id: list(observations)
                        for actor_id, observations in scene.actor_observations.items()
                    },
                }
                for scene in case.scenes
            ],
        }
    )


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RelayLedgerError(f"cannot hash manifest {path}: {error}") from error


def _atomic_write_json(path: str | Path, value: Any) -> None:
    """Write one JSON artifact via same-directory fsync + replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RelayLedgerError(f"cannot read {label} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RelayLedgerError(f"{label} {path} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RelayLedgerError(f"{label} {path} must contain a JSON object")
    return value


def _default_relay_pending_path(ledger_path: Path) -> Path:
    return ledger_path.with_name(ledger_path.name + ".pending.json")


def _new_relay_ledger(
    case: BenchmarkCase,
    *,
    model: str,
    conversation_id: str,
    turns_per_scene: int | None,
    manifest_fingerprint: str | None,
) -> dict[str, Any]:
    case_hash = _case_fingerprint(case)
    manifest_hash = _clean_string(
        manifest_fingerprint or case_hash,
        "manifest_fingerprint",
    )
    return {
        "schema_version": RELAY_LEDGER_SCHEMA_VERSION,
        "manifest_sha256": manifest_hash,
        "case_fingerprint": case_hash,
        "case_id": case.case_id,
        "conversation_id": conversation_id,
        "model": model,
        "turns_per_scene": turns_per_scene,
        "responses": [],
    }


def _validate_relay_ledger_shape(data: Mapping[str, Any], path: Path) -> None:
    if data.get("schema_version") != RELAY_LEDGER_SCHEMA_VERSION:
        raise RelayLedgerError(
            f"response ledger {path} has unsupported schema_version "
            f"{data.get('schema_version')!r}; expected {RELAY_LEDGER_SCHEMA_VERSION!r}"
        )
    required = {
        "manifest_sha256",
        "case_fingerprint",
        "case_id",
        "conversation_id",
        "model",
        "turns_per_scene",
        "responses",
    }
    missing = sorted(field for field in required if field not in data)
    if missing:
        raise RelayLedgerError(
            f"response ledger {path} is missing required fields {missing}"
        )
    if not isinstance(data.get("responses"), list):
        raise RelayLedgerError(f"response ledger {path}.responses must be a list")
    for sequence, raw_entry in enumerate(data["responses"]):
        if not isinstance(raw_entry, Mapping):
            raise RelayLedgerError(
                f"response ledger {path} response {sequence} must be an object"
            )
        if raw_entry.get("sequence") != sequence:
            raise RelayLedgerError(
                f"response ledger {path} is out of order at response {sequence}: "
                f"sequence={raw_entry.get('sequence')!r}"
            )
        request_payload = raw_entry.get("request")
        if not isinstance(request_payload, Mapping):
            raise RelayLedgerError(
                f"response ledger {path} response {sequence} lacks request provenance"
            )
        fingerprint = raw_entry.get("request_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            fingerprint = raw_entry.get("request_sha256")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise RelayLedgerError(
                f"response ledger {path} response {sequence} lacks request fingerprint"
            )
        expected_fingerprint = _sha256_json(request_payload)
        if fingerprint != expected_fingerprint:
            raise RelayLedgerError(
                f"response ledger {path} response {sequence} has a stale request "
                "fingerprint"
            )
        if "request_fingerprint" in raw_entry and raw_entry["request_fingerprint"] != fingerprint:
            raise RelayLedgerError(
                f"response ledger {path} response {sequence} has conflicting request fingerprints"
            )
        if "request_sha256" in raw_entry and raw_entry["request_sha256"] != fingerprint:
            raise RelayLedgerError(
                f"response ledger {path} response {sequence} has conflicting request hashes"
            )
        if "response" not in raw_entry:
            raise RelayLedgerError(
                f"response ledger {path} response {sequence} lacks raw response data"
            )


def _load_relay_ledger(
    path: str | Path,
    case: BenchmarkCase,
    *,
    model: str,
    conversation_id: str,
    turns_per_scene: int | None,
    manifest_fingerprint: str | None,
) -> tuple[Path, dict[str, Any]]:
    ledger_path = Path(path)
    selected_model = _clean_string(model, "model")
    selected_conversation = _clean_string(conversation_id, "conversation_id")
    expected = _new_relay_ledger(
        case,
        model=selected_model,
        conversation_id=selected_conversation,
        turns_per_scene=turns_per_scene,
        manifest_fingerprint=manifest_fingerprint,
    )
    if not ledger_path.exists():
        _atomic_write_json(ledger_path, expected)
        return ledger_path, expected
    data = _read_json_object(ledger_path, label="response ledger")
    _validate_relay_ledger_shape(data, ledger_path)
    for field in (
        "manifest_sha256",
        "case_fingerprint",
        "case_id",
        "conversation_id",
        "model",
        "turns_per_scene",
    ):
        if data.get(field) != expected[field]:
            raise RelayLedgerError(
                f"response ledger {ledger_path} {field} mismatch: "
                f"stored={data.get(field)!r}, current={expected[field]!r}"
            )
    return ledger_path, data


def _pending_document(
    request: BenchmarkRequest,
    *,
    sequence: int,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    request_payload = _benchmark_request_payload(request)
    fingerprint = _sha256_json(request_payload)
    return {
        "schema_version": RELAY_PENDING_REQUEST_SCHEMA_VERSION,
        "ledger_schema_version": RELAY_LEDGER_SCHEMA_VERSION,
        "sequence": sequence,
        "request_fingerprint": fingerprint,
        "request_sha256": fingerprint,
        "request": request_payload,
        "ledger": {
            field: ledger[field]
            for field in (
                "manifest_sha256",
                "case_fingerprint",
                "case_id",
                "conversation_id",
                "model",
                "turns_per_scene",
            )
        },
        "response_entry_template": {
            "sequence": sequence,
            "request_fingerprint": fingerprint,
            "request_sha256": fingerprint,
            "request": request_payload,
            "response": {"content": ""},
        },
    }


def _validate_pending_document(
    data: Mapping[str, Any],
    path: Path,
    *,
    ledger: Mapping[str, Any],
) -> None:
    if data.get("schema_version") != RELAY_PENDING_REQUEST_SCHEMA_VERSION:
        raise RelayLedgerError(
            f"pending request {path} has unsupported schema_version "
            f"{data.get('schema_version')!r}"
        )
    if data.get("ledger_schema_version") != RELAY_LEDGER_SCHEMA_VERSION:
        raise RelayLedgerError(
            f"pending request {path} targets an incompatible ledger schema"
        )
    pending_ledger = data.get("ledger")
    if not isinstance(pending_ledger, Mapping):
        raise RelayLedgerError(f"pending request {path} lacks ledger provenance")
    for field in (
        "manifest_sha256",
        "case_fingerprint",
        "case_id",
        "conversation_id",
        "model",
        "turns_per_scene",
    ):
        if pending_ledger.get(field) != ledger.get(field):
            raise RelayLedgerError(
                f"pending request {path} {field} does not match response ledger"
            )
    request_payload = data.get("request")
    if not isinstance(request_payload, Mapping):
        raise RelayLedgerError(f"pending request {path} lacks exact request data")
    fingerprint = data.get("request_fingerprint")
    if fingerprint != _sha256_json(request_payload):
        raise RelayLedgerError(
            f"pending request {path} has a stale request fingerprint"
        )
    if data.get("request_sha256") != fingerprint:
        raise RelayLedgerError(
            f"pending request {path} has conflicting request hashes"
        )


def _write_relay_pending_request(
    path: str | Path,
    request: BenchmarkRequest,
    *,
    sequence: int,
    ledger: Mapping[str, Any],
) -> Path:
    pending_path = Path(path)
    document = _pending_document(request, sequence=sequence, ledger=ledger)
    if pending_path.exists():
        existing = _read_json_object(pending_path, label="pending request")
        _validate_pending_document(existing, pending_path, ledger=ledger)
        existing_sequence = existing.get("sequence")
        if not isinstance(existing_sequence, int):
            raise RelayLedgerError(
                f"pending request {pending_path} has a non-integer sequence"
            )
        if existing_sequence < 0:
            raise RelayLedgerError(
                f"pending request {pending_path} has a negative sequence"
            )
        if existing_sequence > sequence:
            raise RelayLedgerError(
                f"pending request {pending_path} is ahead of the current missing "
                f"request ({existing_sequence} > {sequence})"
            )
        if existing_sequence < sequence:
            recorded_responses = ledger.get("responses")
            if (
                not isinstance(recorded_responses, list)
                or existing_sequence >= len(recorded_responses)
            ):
                raise RelayLedgerError(
                    f"pending request {pending_path} points past the recorded "
                    "response ledger"
                )
            prior_entry = recorded_responses[existing_sequence]
            if (
                not isinstance(prior_entry, Mapping)
                or prior_entry.get("request") != existing.get("request")
                or (
                    prior_entry.get("request_fingerprint")
                    or prior_entry.get("request_sha256")
                )
                != existing.get("request_fingerprint")
            ):
                raise RelayLedgerError(
                    f"pending request {pending_path} does not match its already "
                    f"recorded response at sequence {existing_sequence}"
                )
        if existing_sequence == sequence:
            if (
                existing.get("request_fingerprint")
                != document["request_fingerprint"]
                or existing.get("request") != document["request"]
            ):
                raise RelayLedgerError(
                    f"pending request {pending_path} does not match the current "
                    "production prompt"
                )
            return pending_path
        # A prior pending request has since been appended manually.  The next
        # missing request gets a fresh atomic pending document, but only after
        # the old request was validated against the ledger above.
    _atomic_write_json(pending_path, document)
    return pending_path


def _relay_response_payload(value: Any) -> dict[str, Any]:
    """Normalize a raw response for the ledger without dropping provider data."""

    if isinstance(value, ModelCall):
        return _raw_response_payload(value)
    if isinstance(value, LLMResponse):
        call = ModelCall(
            content=value.content,
            model=value.model,
            provider="configured",
            usage=value.usage,
            raw_response=value.raw_response,
            assistant_content=value.assistant_content,
        )
        return _raw_response_payload(call)
    if isinstance(value, str):
        return {"content": value}
    if isinstance(value, Mapping):
        # Accept the ordinary artifact envelope when a coding agent copies a
        # response object from a calls JSONL file.  The ledger still stores the
        # response object itself, not a prompt or a derived public_text.
        if "content" not in value and "response" in value:
            nested = value["response"]
            if isinstance(nested, str):
                return {"content": nested}
            if isinstance(nested, Mapping):
                value = nested
        if "content" not in value:
            raise RelayLedgerError(
                "raw relay response must contain a string content field"
            )
        if not isinstance(value["content"], str):
            raise RelayLedgerError(
                "raw relay response content must be a string"
            )
        return copy.deepcopy(_json_safe(dict(value)))
    raise RelayLedgerError(
        f"raw relay response must be text or a JSON object, got {type(value).__name__}"
    )


def _model_call_from_relay_payload(
    payload: Any,
    request: BenchmarkRequest,
) -> ModelCall:
    if isinstance(payload, str):
        return ModelCall(content=payload, model=request.model, provider="relay")
    if not isinstance(payload, Mapping):
        raise RelayLedgerError("stored relay response is not a JSON object")
    content = payload.get("content")
    if not isinstance(content, str):
        raise RelayLedgerError(
            "stored relay response must contain string content for CharacterAgent"
        )
    usage = payload.get("usage", {})
    if usage is None:
        usage = {}
    if not isinstance(usage, Mapping):
        raise RelayLedgerError("stored relay response usage must be an object")
    return ModelCall(
        content=content,
        model=str(payload.get("model") or request.model),
        provider=str(payload.get("provider") or "relay"),
        usage=dict(usage),
        raw_response=copy.deepcopy(payload.get("raw_response")),
        assistant_content=copy.deepcopy(payload.get("assistant_content")),
    )


class RelayResponder:
    """Replay ordered raw responses while preserving the production call seam."""

    def __init__(self, ledger_path: Path, ledger: Mapping[str, Any]) -> None:
        self.ledger_path = ledger_path
        self.ledger = ledger
        self.next_sequence = 0

    async def __call__(self, request: BenchmarkRequest) -> ModelCall:
        sequence = self.next_sequence
        responses = self.ledger.get("responses")
        if not isinstance(responses, list):
            raise RelayLedgerError(
                f"response ledger {self.ledger_path}.responses must be a list"
            )
        if sequence >= len(responses):
            raise RelayPendingRequest(request, sequence)
        entry = responses[sequence]
        if not isinstance(entry, Mapping):
            raise RelayLedgerError(
                f"response ledger {self.ledger_path} response {sequence} is not an object"
            )
        current_request = _benchmark_request_payload(request)
        current_fingerprint = _sha256_json(current_request)
        stored_request = entry.get("request")
        stored_fingerprint = entry.get("request_fingerprint") or entry.get(
            "request_sha256"
        )
        if isinstance(stored_request, Mapping) and (
            stored_request.get("actor_id") != request.actor_id
        ):
            raise RelayLedgerError(
                f"response ledger {self.ledger_path} actor out of order at "
                f"sequence {sequence}: stored={stored_request.get('actor_id')!r}, "
                f"current={request.actor_id!r}"
            )
        if stored_fingerprint != current_fingerprint:
            raise RelayLedgerError(
                f"response ledger {self.ledger_path} request fingerprint mismatch "
                f"at sequence {sequence}; prompt or request order changed"
            )
        if stored_request != current_request:
            raise RelayLedgerError(
                f"response ledger {self.ledger_path} request changed at sequence "
                f"{sequence}; exact production prompt no longer matches"
            )
        if "response" not in entry:
            raise RelayLedgerError(
                f"response ledger {self.ledger_path} response {sequence} lacks raw data"
            )
        call = _model_call_from_relay_payload(entry["response"], request)
        self.next_sequence += 1
        return call

    def assert_exhausted(self) -> None:
        responses = self.ledger.get("responses")
        if not isinstance(responses, list):
            raise RelayLedgerError(
                f"response ledger {self.ledger_path}.responses must be a list"
            )
        if self.next_sequence != len(responses):
            raise RelayLedgerError(
                f"response ledger {self.ledger_path} has extra responses: "
                f"consumed {self.next_sequence}, stored {len(responses)}"
            )


def append_relay_response(
    ledger_path: str | Path,
    raw_response: Any,
    *,
    pending_path: str | Path | None = None,
) -> Path:
    """Append one response using the exact pending request provenance."""

    resolved_ledger = Path(ledger_path)
    ledger = _read_json_object(resolved_ledger, label="response ledger")
    _validate_relay_ledger_shape(ledger, resolved_ledger)
    resolved_pending = Path(pending_path) if pending_path else _default_relay_pending_path(
        resolved_ledger
    )
    if not resolved_pending.exists():
        raise RelayLedgerError(
            f"cannot append response: pending request {resolved_pending} does not exist"
        )
    pending = _read_json_object(resolved_pending, label="pending request")
    _validate_pending_document(pending, resolved_pending, ledger=ledger)
    expected_sequence = len(ledger["responses"])
    if pending.get("sequence") != expected_sequence:
        raise RelayLedgerError(
            f"pending request sequence {pending.get('sequence')!r} does not identify "
            f"the next ledger response {expected_sequence}"
        )
    request_payload = pending["request"]
    fingerprint = pending["request_fingerprint"]
    entry = {
        "sequence": expected_sequence,
        "request_fingerprint": fingerprint,
        "request_sha256": fingerprint,
        "request": copy.deepcopy(request_payload),
        "response": _relay_response_payload(raw_response),
    }
    updated = copy.deepcopy(ledger)
    updated["responses"].append(entry)
    _atomic_write_json(resolved_ledger, updated)
    return resolved_ledger


def _clean_string(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise BenchmarkManifestError(f"{label} cannot be blank")
    if any(ord(char) < 32 and char not in "\n\t" for char in result):
        raise BenchmarkManifestError(f"{label} contains a control character")
    return result


def _parse_character(raw: Any, *, case_id: str, index: int) -> CharacterRecord:
    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError(f"{case_id}.actors[{index}] must be an object")
    allowed_fields = {
        "character_id",
        "name",
        "location",
        "agent_tier",
        "public_sheet",
        "actor",
    }
    unknown_fields = sorted(set(raw) - allowed_fields)
    if unknown_fields:
        raise BenchmarkManifestError(
            f"{case_id}.actors[{index}] has unknown fields {unknown_fields}"
        )
    try:
        character = CharacterRecord.model_validate(dict(raw))
    except ValidationError as error:
        raise BenchmarkManifestError(
            f"{case_id}.actors[{index}] is not a CharacterRecord: {error}"
        ) from error
    if not character.character_id.strip() or not character.name.strip():
        raise BenchmarkManifestError(
            f"{case_id}.actors[{index}] requires character_id and name"
        )
    if character.pending_observations:
        raise BenchmarkManifestError(
            f"{case_id}.actors[{index}] cannot start with pending observations"
        )
    return character


def _parse_prior_public_exchange(
    raw: Any,
    *,
    case_id: str,
    actor_count: int,
    required: bool = True,
) -> tuple[PriorPublicExchange, ...]:
    if raw is None and not required:
        return ()
    if not isinstance(raw, list):
        raise BenchmarkManifestError(
            f"{case_id}.prior_public_exchange must be a list"
        )
    if required and len(raw) < 2:
        raise BenchmarkManifestError(
            f"{case_id}.prior_public_exchange must contain at least two entries"
        )
    entries: list[PriorPublicExchange] = []
    seen_sequences: set[int] = set()
    seen_slots: set[int] = set()
    for index, entry_raw in enumerate(raw):
        if not isinstance(entry_raw, Mapping):
            raise BenchmarkManifestError(
                f"{case_id}.prior_public_exchange[{index}] must be an object"
            )
        try:
            sequence = int(entry_raw.get("sequence"))
            speaker_slot = int(entry_raw.get("speaker_slot"))
        except (TypeError, ValueError) as error:
            raise BenchmarkManifestError(
                f"{case_id}.prior_public_exchange[{index}] sequence and speaker_slot must be integers"
            ) from error
        if sequence < 1 or sequence in seen_sequences:
            raise BenchmarkManifestError(
                f"{case_id}.prior_public_exchange sequence values must be unique and positive"
            )
        if speaker_slot < 0 or speaker_slot >= actor_count:
            raise BenchmarkManifestError(
                f"{case_id}.prior_public_exchange[{index}].speaker_slot must "
                f"reference one of {actor_count} actors"
            )
        text = _clean_string(
            entry_raw.get("text"),
            f"{case_id}.prior_public_exchange[{index}].text",
        )
        if "<" in text or ">" in text:
            raise BenchmarkManifestError(
                f"{case_id}.prior_public_exchange[{index}].text cannot contain prompt markup"
            )
        seen_sequences.add(sequence)
        seen_slots.add(speaker_slot)
        entries.append(PriorPublicExchange(sequence, speaker_slot, text))
    if entries and sorted(seen_sequences) != list(range(1, len(entries) + 1)):
        raise BenchmarkManifestError(
            f"{case_id}.prior_public_exchange sequence values must be contiguous"
        )
    if required and len(seen_slots) < 2:
        raise BenchmarkManifestError(
            f"{case_id}.prior_public_exchange must include at least two speakers"
        )
    return tuple(sorted(entries, key=lambda entry: entry.sequence))


def _parse_pressure_pulses(
    raw: Any,
    *,
    case_id: str,
    turn_count: int,
) -> tuple[PressurePulse, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise BenchmarkManifestError(f"{case_id}.pressure_pulses must be a list")
    pulses: list[PressurePulse] = []
    seen_ids: set[str] = set()
    for index, pulse_raw in enumerate(raw):
        if not isinstance(pulse_raw, Mapping):
            raise BenchmarkManifestError(
                f"{case_id}.pressure_pulses[{index}] must be an object"
            )
        pulse_id = _clean_string(
            pulse_raw.get("pulse_id"), f"{case_id}.pressure_pulses[{index}].pulse_id"
        )
        if pulse_id in seen_ids:
            raise BenchmarkManifestError(f"{case_id} repeats pressure pulse {pulse_id!r}")
        try:
            after_turn = int(pulse_raw.get("after_turn"))
        except (TypeError, ValueError) as error:
            raise BenchmarkManifestError(
                f"{case_id}/{pulse_id}.after_turn must be an integer"
            ) from error
        if after_turn < 0 or after_turn >= turn_count:
            raise BenchmarkManifestError(
                f"{case_id}/{pulse_id}.after_turn must point between turns"
            )
        seen_ids.add(pulse_id)
        pulses.append(
            PressurePulse(
                pulse_id=pulse_id,
                after_turn=after_turn,
                text=_clean_string(pulse_raw.get("text"), f"{case_id}/{pulse_id}.text"),
            )
        )
    return tuple(pulses)


def _parse_order(
    raw: Any,
    *,
    case_id: str,
    actor_ids: set[str],
    field_name: str = "turn_order",
) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise BenchmarkManifestError(f"{case_id}.{field_name} must be non-empty")
    order = tuple(_clean_string(item, f"{case_id}.{field_name}") for item in raw)
    unknown = set(order) - actor_ids
    if unknown:
        raise BenchmarkManifestError(
            f"{case_id}.{field_name} references unknown actors {sorted(unknown)}"
        )
    return order


def _parse_actor_observations(
    raw: Any,
    *,
    case_id: str,
    actor_ids: set[str],
) -> Mapping[str, tuple[str, ...]]:
    """Parse actor-local observations authored for one scene start.

    These observations are setup input for the owning CharacterAgent.  They
    use the same concrete-string shape as the runtime inbox and deliberately
    do not become a shared scene event or transcript entry.
    """

    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError(
            f"{case_id}.actor_observations must be an object mapping actor ids to lists"
        )
    parsed: dict[str, tuple[str, ...]] = {}
    for raw_actor_id, raw_observations in raw.items():
        if not isinstance(raw_actor_id, str) or not raw_actor_id.strip():
            raise BenchmarkManifestError(
                f"{case_id}.actor_observations keys must be non-empty actor ids"
            )
        actor_id = raw_actor_id.strip()
        if actor_id not in actor_ids:
            raise BenchmarkManifestError(
                f"{case_id}.actor_observations references unknown actor {actor_id!r}"
            )
        if actor_id in parsed:
            raise BenchmarkManifestError(
                f"{case_id}.actor_observations repeats actor {actor_id!r}"
            )
        if not isinstance(raw_observations, list):
            raise BenchmarkManifestError(
                f"{case_id}.actor_observations[{actor_id!r}] must be a list"
            )
        observations: list[str] = []
        seen_observation_texts: set[str] = set()
        for observation_index, observation_raw in enumerate(raw_observations):
            observation = _clean_string(
                observation_raw,
                f"{case_id}.actor_observations[{actor_id!r}][{observation_index}]",
            )
            if "<" in observation or ">" in observation:
                raise BenchmarkManifestError(
                    f"{case_id}.actor_observations[{actor_id!r}] text cannot contain prompt markup"
                )
            normalized_observation = observation.casefold()
            if normalized_observation in seen_observation_texts:
                raise BenchmarkManifestError(
                    f"{case_id}.actor_observations[{actor_id!r}] contains duplicate text"
                )
            seen_observation_texts.add(normalized_observation)
            observations.append(observation)
        parsed[actor_id] = tuple(observations)
    return parsed


def _parse_scene(
    raw: Any,
    *,
    case_id: str,
    scene_index: int,
    actor_ids: set[str],
    actor_count: int,
) -> SceneSpec:
    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError(f"{case_id}.scenes[{scene_index}] must be an object")
    prefix = f"{case_id}.scenes[{scene_index}]"
    allowed_fields = {
        "scene_id",
        "title",
        "frame",
        "between_scene_public_history",
        "actor_observations",
        "prior_public_exchange",
        "turn_order",
        "pressure_pulses",
    }
    unknown_fields = sorted(set(raw) - allowed_fields)
    if unknown_fields:
        raise BenchmarkManifestError(f"{prefix} has unknown fields {unknown_fields}")
    scene_id = _clean_string(raw.get("scene_id"), f"{prefix}.scene_id")
    title = _clean_string(raw.get("title", scene_id), f"{prefix}.title")
    frame = _clean_string(raw.get("frame"), f"{prefix}.frame")
    if "<" in frame or ">" in frame:
        raise BenchmarkManifestError(f"{prefix}.frame cannot contain prompt markup")
    raw_prior = raw.get("prior_public_exchange")
    prior = _parse_prior_public_exchange(
        raw_prior,
        case_id=prefix,
        actor_count=actor_count,
        required=raw_prior is not None,
    )
    order = _parse_order(raw.get("turn_order"), case_id=prefix, actor_ids=actor_ids)
    pulses = _parse_pressure_pulses(
        raw.get("pressure_pulses", []), case_id=prefix, turn_count=len(order)
    )
    history = raw.get("between_scene_public_history", "")
    if not isinstance(history, str):
        raise BenchmarkManifestError(
            f"{prefix}.between_scene_public_history must be a string"
        )
    actor_observations = _parse_actor_observations(
        raw.get("actor_observations", {}),
        case_id=prefix,
        actor_ids=actor_ids,
    )
    return SceneSpec(
        scene_id=scene_id,
        title=title,
        frame=frame,
        prior_public_exchange=prior,
        turn_order=order,
        pressure_pulses=pulses,
        between_scene_public_history=history.strip(),
        actor_observations=actor_observations,
    )


def _parse_actor_list(
    raw: Any,
    *,
    case_id: str,
    field_name: str,
) -> tuple[CharacterRecord, ...]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise BenchmarkManifestError(
            f"{case_id} must contain at least two {field_name} actors"
        )
    actors: list[CharacterRecord] = []
    actor_ids: set[str] = set()
    for index, actor_raw in enumerate(raw):
        character = _parse_character(actor_raw, case_id=case_id, index=index)
        actor_id = character.character_id
        if actor_id in actor_ids:
            raise BenchmarkManifestError(
                f"{case_id} repeats {field_name} actor {actor_id!r}"
            )
        actor_ids.add(actor_id)
        actors.append(character)
    return tuple(actors)


def _parse_case(raw: Any, index: int) -> BenchmarkCase:
    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError(f"case {index} must be an object")
    case_id = _clean_string(raw.get("case_id"), f"cases[{index}].case_id")
    allowed_fields = {"case_id", "title", "suite", "source_metadata", "actors", "scenes"}
    unknown_fields = sorted(set(raw) - allowed_fields)
    if unknown_fields:
        raise BenchmarkManifestError(
            f"{case_id} has unknown fields {unknown_fields}; scene data belongs in scenes"
        )
    suite = _clean_string(raw.get("suite"), f"{case_id}.suite").lower()
    if suite not in SUITES:
        raise BenchmarkManifestError(
            f"{case_id}.suite must be one of {list(SUITES)}, got {suite!r}"
        )
    actors = _parse_actor_list(raw.get("actors"), case_id=case_id, field_name="actors")
    actor_ids = {actor.character_id for actor in actors}
    scenes_raw = raw.get("scenes")
    if not isinstance(scenes_raw, list) or len(scenes_raw) < 2:
        raise BenchmarkManifestError(
            f"{case_id}.scenes must contain at least two separated scenes"
        )
    scenes: list[SceneSpec] = []
    for scene_index, scene_raw in enumerate(scenes_raw):
        scene = _parse_scene(
            scene_raw,
            case_id=case_id,
            scene_index=scene_index,
            actor_ids=actor_ids,
            actor_count=len(actors),
        )
        if scene_index > 0 and not scene.between_scene_public_history:
            raise BenchmarkManifestError(
                f"{case_id}.scenes[{scene_index}].between_scene_public_history "
                "must describe the public consequence between scenes"
            )
        scenes.append(scene)
    if len({scene.scene_id for scene in scenes}) != len(scenes):
        raise BenchmarkManifestError(f"{case_id}.scenes must have unique scene_id values")
    source_metadata = raw.get("source_metadata", {})
    if not isinstance(source_metadata, Mapping):
        raise BenchmarkManifestError(f"{case_id}.source_metadata must be an object")

    public_text = "\n".join(
        [scene.frame for scene in scenes]
        + [
            entry.text
            for scene in scenes
            for entry in scene.prior_public_exchange
        ]
        + [scene.between_scene_public_history for scene in scenes]
    ).casefold()
    for actor in actors:
        actor_local_values = (
            tuple(fact.text for fact in actor.actor.facts)
            if actor.actor is not None
            else ()
        )
        for private_value in actor_local_values:
            if private_value and private_value.casefold() in public_text:
                raise BenchmarkManifestError(
                    f"{case_id} public text repeats actor-local material for {actor.character_id}"
                )
    for scene in scenes:
        for actor_id, observations in scene.actor_observations.items():
            for observation in observations:
                if observation.casefold() in public_text:
                    raise BenchmarkManifestError(
                        f"{case_id} public text repeats actor observation for {actor_id}"
                    )
    if suite == "ordinary_surface" and any(scene.pressure_pulses for scene in scenes):
        raise BenchmarkManifestError(
            f"{case_id} ordinary_surface cases cannot contain pressure pulses"
        )
    if suite == "pressure" and not any(scene.pressure_pulses for scene in scenes):
        raise BenchmarkManifestError(
            f"{case_id} pressure cases must contain at least one pressure pulse"
        )
    return BenchmarkCase(
        case_id=case_id,
        title=_clean_string(raw.get("title"), f"{case_id}.title"),
        suite=suite,
        scenes=tuple(scenes),
        actors=actors,
        source_metadata=dict(source_metadata),
    )


def load_benchmark_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> tuple[BenchmarkCase, ...]:
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkManifestError(
            f"cannot read manifest {manifest_path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise BenchmarkManifestError(
            f"manifest {manifest_path} is not valid JSON: {error}"
        ) from error
    if not isinstance(raw, Mapping) or not isinstance(raw.get("cases"), list):
        raise BenchmarkManifestError("manifest must contain a cases list")
    if raw.get("schema_version") != "character_dialogue_benchmark_manifest_v7":
        raise BenchmarkManifestError(
            "manifest schema_version must be character_dialogue_benchmark_manifest_v7"
        )
    unknown_fields = sorted(set(raw) - {"schema_version", "description", "cases"})
    if unknown_fields:
        raise BenchmarkManifestError(f"manifest has unknown fields {unknown_fields}")
    cases = tuple(_parse_case(case, index) for index, case in enumerate(raw["cases"]))
    if not cases:
        raise BenchmarkManifestError("manifest must contain at least one case")
    return cases


def _model_config(model: str) -> ModelConfig:
    return ModelConfig(
        event_router="",
        narrator="",
        image_director="",
        dnd_combat_manager="",
        agent_default=model,
        agent_standard=model,
        agent_convenience=model,
        character_manager="",
    )


def new_synthetic_checkpoint(
    case: BenchmarkCase,
    *,
    model: str,
    conversation_id: str | None = None,
) -> CheckpointFile:
    """Create the same actor records the production CharacterAgent consumes."""

    selected_model = _clean_string(model, "model")
    run_id = conversation_id or f"{case.case_id}-{uuid.uuid4().hex}"
    characters = [actor.model_copy(deep=True) for actor in case.actors]
    return CheckpointFile(
        session=SessionState(
            session_id=f"character-dialogue-benchmark:{run_id}",
            story_id=f"character-dialogue-benchmark:{case.case_id}",
            config=SessionConfig(
                models=_model_config(selected_model),
                settings=SessionSettings(ruleset_id=""),
            ),
        ),
        player_primer="",
        world_state=WorldState(),
        characters=characters,
        character_conversations={
            actor.character_id: [] for actor in case.actors
        },
    )


def _render_public_updates(public_updates: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for entry in public_updates:
        kind = str(entry.get("kind", ""))
        text = str(entry.get("text", "")).strip()
        if kind == "turn":
            speaker = str(entry.get("speaker_name", "Someone") or "Someone")
            if text == EXACT_SILENCE:
                lines.append(f"{speaker} remains silent.")
            else:
                lines.append(
                    f"Scene {entry.get('scene_index')}, turn {entry.get('turn_index')}, "
                    f"{speaker}: {text}"
                )
        elif kind == "prior_public":
            lines.append(
                f"Earlier in this scene, {entry.get('speaker_name')}: {text}"
            )
        elif kind == "pressure":
            lines.append(f"Something interrupts the moment: {text}")
        elif kind == "scene_start":
            lines.append(f"The scene: {text}")
        elif kind == "scene_break":
            lines.append(f"Time passes: {text}")
        elif kind == "between_scene_history":
            lines.append(f"Since the last meeting: {text}")
        else:
            lines.append(text)
    return "\n".join(lines) or "Nothing new has reached you."


def _offline_responder(request: BenchmarkRequest) -> ModelCall:
    """Deterministic plumbing response for tests and no-cost benchmark runs."""

    if request.scene_index == 0:
        content = (
            f"{request.actor_name} sets the shared object where they can both see it. "
            '"I can answer for this much."'
        )
    else:
        content = (
            f"{request.actor_name} returns to the choice they left behind. "
            '"It did not stay as small as we made it."'
        )
    return ModelCall(
        content=content,
        model=request.model,
        provider="offline",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


def _provider_response_to_model_call(response: Any, request: BenchmarkRequest) -> ModelCall:
    if isinstance(response, ModelCall):
        return response
    if isinstance(response, str):
        return ModelCall(content=response, model=request.model, provider="test")
    if isinstance(response, LLMResponse):
        return ModelCall(
            content=response.content,
            model=response.model or request.model,
            provider="configured",
            usage=dict(response.usage or {}),
            raw_response=response.raw_response,
            assistant_content=response.assistant_content,
        )
    content = str(getattr(response, "content", "") or "")
    return ModelCall(
        content=content,
        model=str(getattr(response, "model", "") or request.model),
        provider=str(getattr(response, "provider", "test") or "test"),
        usage=dict(getattr(response, "usage", {}) or {}),
        raw_response=getattr(response, "raw_response", None),
        assistant_content=getattr(response, "assistant_content", None),
    )


Responder = Callable[
    [BenchmarkRequest],
    ModelCall | str | LLMResponse | Awaitable[ModelCall | str | LLMResponse],
]


async def _call_responder(responder: Responder, request: BenchmarkRequest) -> ModelCall:
    value = responder(request)
    if inspect.isawaitable(value):
        value = await value
    return _provider_response_to_model_call(value, request)


@dataclass(frozen=True)
class _BenchmarkCallContext:
    conversation_id: str
    case_id: str
    scene_id: str
    scene_index: int
    scene_turn_index: int
    turn_index: int
    actor_id: str
    actor_name: str


class _RecordingCharacterClient:
    """Record exact CharacterAgent calls while delegating only model I/O.

    The benchmark does not render, parse, or commit actor turns.  Those jobs
    belong to the production ``CharacterAgent``.  This adapter merely attaches
    scene provenance to the exact request that CharacterAgent sends and then
    either invokes a deterministic responder or a real ``LLMClient``.
    """

    def __init__(
        self,
        *,
        model: str,
        responder: Responder | None = None,
        live_client: LLMClient | None = None,
    ) -> None:
        if (responder is None) == (live_client is None):
            raise ValueError("provide exactly one benchmark responder or live client")
        self.model = model
        self.responder = responder
        self.live_client = live_client
        self._context: _BenchmarkCallContext | None = None
        self.last_request: BenchmarkRequest | None = None
        self.last_call: ModelCall | None = None

    def begin(self, context: _BenchmarkCallContext) -> None:
        self._context = context
        self.last_request = None
        self.last_call = None

    async def complete(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        cache: bool,
        compact: bool,
        **kwargs: Any,
    ) -> LLMResponse:
        del kwargs
        context = self._context
        if context is None:
            raise RuntimeError("benchmark CharacterAgent call has no turn context")
        request = BenchmarkRequest(
            conversation_id=context.conversation_id,
            case_id=context.case_id,
            scene_id=context.scene_id,
            scene_index=context.scene_index,
            scene_turn_index=context.scene_turn_index,
            turn_index=context.turn_index,
            actor_id=context.actor_id,
            actor_name=context.actor_name,
            model=self.model,
            role=role,
            messages=tuple(copy.deepcopy(message) for message in messages),
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            compact=compact,
        )
        if self.live_client is not None:
            response = await self.live_client.complete(
                role=role,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                cache=cache,
                compact=compact,
            )
            call = _provider_response_to_model_call(response, request)
        else:
            assert self.responder is not None
            call = await _call_responder(self.responder, request)
            response = LLMResponse(
                content=call.content,
                model=call.model or self.model,
                usage=dict(call.usage or {}),
                raw_response=call.raw_response,
                assistant_content=(
                    call.assistant_content
                    if call.assistant_content is not None
                    else [{"type": "text", "text": call.content}]
                ),
            )
        self.last_request = request
        self.last_call = call
        return response


def _raw_response_payload(call: ModelCall) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "content": call.content,
        "model": call.model,
        "provider": call.provider,
        "usage": _json_safe(call.usage or {}),
    }
    if call.assistant_content is not None:
        payload["assistant_content"] = _json_safe(call.assistant_content)
    if call.raw_response is not None:
        payload["raw_response"] = _json_safe(call.raw_response)
    return payload


def _pulses_after(scene: SceneSpec, previous_turn_count: int) -> tuple[PressurePulse, ...]:
    return tuple(
        pulse
        for pulse in scene.pressure_pulses
        if pulse.after_turn == previous_turn_count
    )


def _append_scene_public_start(
    public_transcript: list[dict[str, Any]],
    *,
    scene: SceneSpec,
    scene_index: int,
    actors: Sequence[CharacterRecord],
) -> None:
    if scene_index > 0:
        public_transcript.append(
            {
                "kind": "scene_break",
                "scene_index": scene_index,
                "text": "The previous meeting ended; time has passed before this return.",
            }
        )
        if scene.between_scene_public_history:
            public_transcript.append(
                {
                    "kind": "between_scene_history",
                    "scene_index": scene_index,
                    "text": scene.between_scene_public_history,
                }
            )
    public_transcript.append(
        {
            "kind": "scene_start",
            "scene_index": scene_index,
            "scene_id": scene.scene_id,
            "title": scene.title,
            "text": scene.frame,
        }
    )
    for entry in scene.prior_public_exchange:
        public_transcript.append(
            {
                "kind": "prior_public",
                "scene_index": scene_index,
                "sequence": entry.sequence,
                "actor_id": actors[entry.speaker_slot].character_id,
                "speaker_name": actors[entry.speaker_slot].name,
                "text": entry.text,
            }
        )


def _enqueue_scene_actor_observations(
    checkpoint: CheckpointFile,
    scene: SceneSpec,
) -> None:
    """Queue each scene-start observation only for its owning actor."""

    characters = {
        character.character_id: character for character in checkpoint.characters
    }
    for actor_id, observations in scene.actor_observations.items():
        character = characters.get(actor_id)
        if character is None:
            raise BenchmarkManifestError(
                f"scene {scene.scene_id!r} has no checkpoint actor {actor_id!r}"
            )
        character.pending_observations.extend(observations)


async def run_conversation(
    case: BenchmarkCase,
    *,
    model: str = DEFAULT_MODEL,
    responder: Responder | None = None,
    live_client: LLMClient | None = None,
    turns_per_scene: int | None = None,
    conversation_id: str | None = None,
) -> ConversationResult:
    """Run every authored scene through the production CharacterAgent."""

    selected_model = _clean_string(model, "model")
    if turns_per_scene is not None and turns_per_scene < 1:
        raise ValueError("turns_per_scene must be at least 1")
    run_id = conversation_id or f"{case.case_id}-{uuid.uuid4().hex}"
    checkpoint = new_synthetic_checkpoint(
        case,
        model=selected_model,
        conversation_id=run_id,
    )
    initial_checkpoint_sha256 = _sha256_json(checkpoint.model_dump(mode="json"))
    if responder is not None and live_client is not None:
        raise ValueError("responder and live_client are mutually exclusive")
    recording_client = _RecordingCharacterClient(
        model=selected_model,
        responder=responder or (_offline_responder if live_client is None else None),
        live_client=live_client,
    )
    character_agent = CharacterAgent(
        recording_client,  # type: ignore[arg-type]
        PromptManager(str(REPO_ROOT / "app" / "prompts")),
    )
    public_transcript: list[dict[str, Any]] = []
    seen_public_counts = {actor.character_id: 0 for actor in case.actors}
    turn_results: list[TurnResult] = []
    global_turn_index = 0

    for scene_index, scene in enumerate(case.scenes):
        _enqueue_scene_actor_observations(checkpoint, scene)
        _append_scene_public_start(
            public_transcript,
            scene=scene,
            scene_index=scene_index,
            actors=case.actors,
        )
        order = scene.turn_order
        scene_turn_count = len(order)
        if turns_per_scene is not None:
            scene_turn_count = min(turns_per_scene, scene_turn_count)
        for scene_turn_index, actor_id in enumerate(order[:scene_turn_count], start=1):
            global_turn_index += 1
            pulses = _pulses_after(scene, scene_turn_index - 1)
            for pulse in pulses:
                public_transcript.append(
                    {
                        "kind": "pressure",
                        "scene_index": scene_index,
                        "pulse_id": pulse.pulse_id,
                        "after_turn": pulse.after_turn,
                        "text": pulse.text,
                    }
                )
            actor = case.actor(actor_id)
            updates = public_transcript[seen_public_counts[actor_id] :]
            character = next(
                character
                for character in checkpoint.characters
                if character.character_id == actor_id
            )
            witnessed = _render_public_updates(updates)
            if witnessed:
                character.pending_observations.append(witnessed)
            recording_client.begin(_BenchmarkCallContext(
                conversation_id=run_id,
                case_id=case.case_id,
                scene_id=scene.scene_id,
                scene_index=scene_index,
                scene_turn_index=scene_turn_index,
                turn_index=global_turn_index,
                actor_id=actor_id,
                actor_name=actor.name,
            ))
            started_at = time.perf_counter()
            output = await character_agent.turn(
                character,
                checkpoint,
                frame="foreground",
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            request = recording_client.last_request
            call = recording_client.last_call
            if request is None or call is None:
                raise RuntimeError("CharacterAgent completed without a recorded request")
            messages = [dict(message) for message in request.messages]
            raw_text = str(call.content or "").strip()
            public_text = (
                EXACT_SILENCE
                if output.is_silence
                else output.public_text
            )
            prompt_hash = _sha256_json(messages)
            response_hash = _sha256_text(raw_text)
            provider_payload = _raw_response_payload(call)
            public_transcript.append(
                {
                    "kind": "turn",
                    "scene_index": scene_index,
                    "scene_id": scene.scene_id,
                    "scene_turn_index": scene_turn_index,
                    "turn_index": global_turn_index,
                    "actor_id": actor_id,
                    "speaker_name": actor.name,
                    "text": public_text,
                }
            )
            checkpoint.session.turn_index = global_turn_index
            checkpoint.session.leading_at_s = global_turn_index
            seen_public_counts[actor_id] = len(public_transcript)
            turn_results.append(
                TurnResult(
                    scene_index=scene_index,
                    scene_id=scene.scene_id,
                    scene_turn_index=scene_turn_index,
                    turn_index=global_turn_index,
                    actor_id=actor_id,
                    actor_name=actor.name,
                    public_text=public_text,
                    raw_response=raw_text,
                    prompt=tuple(dict(message) for message in messages),
                    provider_response=provider_payload,
                    response_model=call.model or selected_model,
                    provider=call.provider,
                    usage=dict(call.usage or {}),
                    elapsed_ms=elapsed_ms,
                    prompt_sha256=prompt_hash,
                    response_sha256=response_hash,
                    pressure_pulse_ids=tuple(pulse.pulse_id for pulse in pulses),
                    role=request.role,
                    cache=request.cache,
                    compact=request.compact,
                )
            )

    return ConversationResult(
        conversation_id=run_id,
        case=case,
        model=selected_model,
        checkpoint=checkpoint,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        turns=turn_results,
        public_transcript=public_transcript,
    )


async def run_relay_conversation(
    case: BenchmarkCase,
    *,
    ledger_path: str | Path,
    model: str = DEFAULT_MODEL,
    conversation_id: str,
    pending_path: str | Path | None = None,
    turns_per_scene: int | None = None,
    manifest_fingerprint: str | None = None,
) -> ConversationResult:
    """Replay a case through production CharacterAgent using a response ledger.

    The checkpoint and every actor history are rebuilt from the beginning on
    each invocation.  Only the model responses are externalized, so prompt
    construction, parser behavior, public-history fan-in, and commits remain
    the ordinary benchmark path.  A missing response writes an exact pending
    request and raises ``RelayPendingRequest``; callers must append that raw
    response before trying again.
    """

    selected_model = _clean_string(model, "model")
    selected_conversation = _clean_string(conversation_id, "conversation_id")
    if turns_per_scene is not None and turns_per_scene < 1:
        raise ValueError("turns_per_scene must be at least 1")
    resolved_ledger, ledger = _load_relay_ledger(
        ledger_path,
        case,
        model=selected_model,
        conversation_id=selected_conversation,
        turns_per_scene=turns_per_scene,
        manifest_fingerprint=manifest_fingerprint,
    )
    resolved_pending = (
        Path(pending_path)
        if pending_path is not None
        else _default_relay_pending_path(resolved_ledger)
    )
    responder = RelayResponder(resolved_ledger, ledger)
    try:
        result = await run_conversation(
            case,
            model=selected_model,
            responder=responder,
            turns_per_scene=turns_per_scene,
            conversation_id=selected_conversation,
        )
    except RelayPendingRequest as pending:
        _write_relay_pending_request(
            resolved_pending,
            pending.request,
            sequence=pending.sequence,
            ledger=ledger,
        )
        raise
    responder.assert_exhausted()
    return result


async def run_benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    model: str = DEFAULT_MODEL,
    responder: Responder | None = None,
    live_client: LLMClient | None = None,
    turns_per_scene: int | None = None,
) -> list[ConversationResult]:
    """Run cases independently; no checkpoint or actor history is shared."""

    results: list[ConversationResult] = []
    for case in cases:
        results.append(
            await run_conversation(
                case,
                model=model,
                responder=responder,
                live_client=live_client,
                turns_per_scene=turns_per_scene,
            )
        )
    return results


def _blind_transcript(result: ConversationResult) -> tuple[dict[str, Any], dict[str, str]]:
    actor_labels: dict[str, str] = {}
    for entry in result.public_transcript:
        actor_id = str(entry.get("actor_id", ""))
        if actor_id and actor_id not in actor_labels:
            actor_labels[actor_id] = chr(ord("A") + len(actor_labels))

    actor_names = {
        actor_id: result.case.actor(actor_id).name for actor_id in actor_labels
    }

    def blind_text(value: Any) -> str:
        text = str(value)
        # A model may mention an actor by name or id in its public prose.  A
        # blinded review packet must not let that bypass the speaker labels.
        for actor_id, label in sorted(
            actor_labels.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = text.replace(actor_id, label)
            text = text.replace(actor_names[actor_id], label)
        return text

    transcript: list[dict[str, Any]] = []
    for entry in result.public_transcript:
        kind = str(entry.get("kind", ""))
        if kind == "turn":
            transcript.append(
                {
                    "kind": "turn",
                    "scene_index": entry["scene_index"],
                    "scene_turn_index": entry["scene_turn_index"],
                    "turn_index": entry["turn_index"],
                    "speaker": actor_labels[entry["actor_id"]],
                    "text": blind_text(entry["text"]),
                }
            )
        elif kind == "prior_public":
            transcript.append(
                {
                    "kind": "prior_exchange",
                    "scene_index": entry["scene_index"],
                    "sequence": entry["sequence"],
                    "speaker": actor_labels[entry["actor_id"]],
                    "text": blind_text(entry["text"]),
                }
            )
        elif kind == "pressure":
            transcript.append(
                {
                    "kind": "public_pressure",
                    "scene_index": entry["scene_index"],
                    "after_turn": entry["after_turn"],
                    "text": blind_text(entry["text"]),
                }
            )
        else:
            transcript.append(
                {
                    "kind": kind,
                    "scene_index": entry.get("scene_index"),
                    "text": blind_text(entry["text"]),
                }
            )

    whole_transcript = {
        "conversation_label": f"conversation-{_sha256_text(result.conversation_id)[:10]}",
        "unit": "whole_serial_conversation",
        "scenes": [
            {"scene_index": index, "label": f"Scene {index + 1}"}
            for index, _scene in enumerate(result.case.scenes)
        ],
        "transcript": transcript,
        "speaker_sheets": [
            {
                "speaker": label,
                "whole_transcript": transcript,
                "instruction": "Review the complete sequence across every scene; do not score isolated lines.",
            }
            for label in actor_labels.values()
        ],
    }
    return whole_transcript, actor_labels


def build_human_review_packet(result: ConversationResult) -> dict[str, Any]:
    """Build a blank reviewer form around a blinded whole serial transcript."""

    blinded, actor_labels = _blind_transcript(result)
    review = {
        "schema_version": "character_dialogue_human_review_v3",
        "reviewer": "human",
        "model_judge": False,
        "auto_semantic_score": False,
        "unit": "whole_serial_conversation",
        "status": "",
        "allowed_statuses": list(HUMAN_REVIEW_STATUSES),
        "fields": {
            field: {
                "question": HUMAN_REVIEW_QUESTIONS[field],
                "value": "",
                "notes": "",
            }
            for field in HUMAN_REVIEW_FIELDS
        },
    }
    return {
        "blinded": blinded,
        "review": review,
        "answer_key": {
            "schema_version": "character_dialogue_answer_key_v3",
            "conversation_id": result.conversation_id,
            "case_id": result.case.case_id,
            "suite": result.case.suite,
            "title": result.case.title,
            "scenes": [_scene_artifact(scene) for scene in result.case.scenes],
            "model": result.model,
            "checkpoint_id": result.checkpoint_id,
            "speaker_mapping": {
                label: actor_id for actor_id, label in actor_labels.items()
            },
            "actor_names": {
                label: result.case.actor(actor_id).name
                for actor_id, label in actor_labels.items()
            },
            "source_metadata": _json_safe(result.case.source_metadata),
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_benchmark_artifacts(
    results: Sequence[ConversationResult], output_dir: str | Path
) -> Path:
    """Write raw calls and blinded reviewer packets without collapsing scenes."""

    root = Path(output_dir)
    raw_dir = root / "raw"
    review_dir = root / "review"
    blinded_dir = review_dir / "blinded"
    raw_dir.mkdir(parents=True, exist_ok=True)
    blinded_dir.mkdir(parents=True, exist_ok=True)
    blinded_packets: list[dict[str, Any]] = []
    answer_keys: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for result in results:
        artifact = result.artifact()
        review_packet = build_human_review_packet(result)
        _write_json(raw_dir / f"{result.case.case_id}.json", artifact)
        calls_path = raw_dir / f"{result.case.case_id}.jsonl"
        call_lines = []
        for turn in result.turns:
            call_lines.append(
                json.dumps(
                    {
                        "conversation_id": result.conversation_id,
                        "case_id": result.case.case_id,
                        "scene_index": turn.scene_index,
                        "scene_id": turn.scene_id,
                        "scene_turn_index": turn.scene_turn_index,
                        "turn_index": turn.turn_index,
                        "request": {
                            "model": result.model,
                            "role": turn.role,
                            "scene_index": turn.scene_index,
                            "scene_id": turn.scene_id,
                            "scene_turn_index": turn.scene_turn_index,
                            "turn_index": turn.turn_index,
                            "temperature": 0.6,
                            "max_tokens": 2000,
                            "cache": turn.cache,
                            "compact": turn.compact,
                            "messages": turn.prompt,
                        },
                        "response": _json_safe(turn.provider_response),
                        "public_text": turn.public_text,
                        "prompt_sha256": turn.prompt_sha256,
                        "response_sha256": turn.response_sha256,
                        "elapsed_ms": turn.elapsed_ms,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        calls_path.parent.mkdir(parents=True, exist_ok=True)
        calls_path.write_text("\n".join(call_lines) + "\n", encoding="utf-8")
        blinded_packets.append(review_packet["blinded"])
        answer_keys.append(review_packet["answer_key"])
        _write_json(
            blinded_dir / f"{review_packet['blinded']['conversation_label']}.json",
            {"blinded": review_packet["blinded"], "review": review_packet["review"]},
        )
        summaries.append(
            {
                "case_id": result.case.case_id,
                "suite": result.case.suite,
                "conversation_id": result.conversation_id,
                "checkpoint_id": result.checkpoint_id,
                "scene_count": len(result.case.scenes),
                "turn_count": len(result.turns),
                "model": result.model,
            }
        )

    _write_json(
        review_dir / "blinded_transcripts.json",
        {
            "schema_version": "character_dialogue_blinded_review_v3",
            "model_judge": False,
            "auto_semantic_score": False,
            "unit": "whole_serial_conversation",
            "conversations": blinded_packets,
        },
    )
    _write_json(
        review_dir / "answer_key.json",
        {
            "schema_version": "character_dialogue_answer_key_collection_v3",
            "conversations": answer_keys,
        },
    )
    _write_json(
        root / "benchmark_report.json",
        {
            "schema_version": "character_dialogue_benchmark_report_v3",
            "model_judge": False,
            "auto_semantic_score": False,
            "unit": "whole_serial_conversation",
            "cases": summaries,
            "review_fields": list(HUMAN_REVIEW_FIELDS),
        },
    )
    return root


def _live_client(model: str) -> LLMClient:
    """Build a production client whose every actor tier uses one model."""

    configured_model = _clean_string(model, "model")
    provider: str | None = None
    if ":" in configured_model:
        provider, configured_model = configured_model.split(":", 1)
        provider = provider.strip().lower()
        if provider not in {"openai", "anthropic"}:
            raise ValueError("model provider prefix must be openai or anthropic")
        configured_model = _clean_string(configured_model, "model")
    config = LLMConfig.from_env()
    actor_roles = ("agent", "agent_standard", "agent_convenience")
    role_models = dict(config.role_models)
    for role in actor_roles:
        role_models[role] = configured_model
    role_providers = dict(config.role_providers)
    if provider:
        for role in actor_roles:
            role_providers[role] = provider
    return LLMClient(
        config.model_copy(
            update={"role_models": role_models, "role_providers": role_providers}
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--case", dest="case_ids", action="append", default=[])
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default=None,
        help="Run only the ordinary-surface or pressure cases.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Explicit actor model (default: gpt-5.6-luna).",
    )
    parser.add_argument(
        "--turns-per-scene",
        type=int,
        default=None,
        help="Optionally cap each authored scene while retaining every scene.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--relay",
        action="store_true",
        help=(
            "Replay raw responses through production CharacterAgent, writing "
            "one pending request when the ledger runs out."
        ),
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Stable conversation id required by --relay.",
    )
    parser.add_argument(
        "--relay-ledger",
        "--ledger",
        dest="relay_ledger",
        type=Path,
        default=None,
        help="Versioned JSON response ledger used by --relay.",
    )
    parser.add_argument(
        "--pending-request",
        dest="pending_request",
        type=Path,
        default=None,
        help="Exact missing production request written by --relay.",
    )
    parser.add_argument(
        "--append-response",
        type=Path,
        default=None,
        help=(
            "Append one JSON or plain-text raw response from this file before "
            "replaying --relay."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the explicitly selected actor model. The default is offline.",
    )
    return parser


def _read_relay_response_file(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RelayLedgerError(f"cannot read relay response {path}: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Plain text is a useful lowest-friction handoff for a coding agent;
        # JSON provider envelopes remain supported when a raw response was
        # saved from an API log.
        return text


async def _run_cli(args: argparse.Namespace) -> int:
    cases = load_benchmark_manifest(args.manifest)
    if args.case_ids:
        selected_ids = set(args.case_ids)
        cases = tuple(case for case in cases if case.case_id in selected_ids)
        missing = selected_ids - {case.case_id for case in cases}
        if missing:
            raise SystemExit(f"Unknown benchmark case(s): {', '.join(sorted(missing))}")
        if not cases:
            raise SystemExit("No benchmark cases were selected")
    if args.suite:
        cases = tuple(case for case in cases if case.suite == args.suite)
        if not cases:
            raise SystemExit(f"No benchmark cases belong to suite {args.suite!r}")
    if args.turns_per_scene is not None and args.turns_per_scene < 1:
        raise SystemExit("--turns-per-scene must be at least 1")
    if args.relay:
        if args.live:
            raise SystemExit("--relay is offline and cannot be combined with --live")
        if len(cases) != 1:
            raise SystemExit(
                "--relay requires exactly one case; pass one --case and omit "
                "a multi-case suite"
            )
        if not args.conversation_id:
            raise SystemExit("--relay requires --conversation-id")
        case = cases[0]
        ledger_path = args.relay_ledger or (args.output / "response_ledger.json")
        pending_path = args.pending_request or _default_relay_pending_path(
            ledger_path
        )
        manifest_fingerprint = _file_sha256(args.manifest)
        if args.append_response is not None:
            # Validate the fixed case/model/conversation before mutating the
            # ledger with a manually supplied response.
            _load_relay_ledger(
                ledger_path,
                case,
                model=args.model,
                conversation_id=args.conversation_id,
                turns_per_scene=args.turns_per_scene,
                manifest_fingerprint=manifest_fingerprint,
            )
            append_relay_response(
                ledger_path,
                _read_relay_response_file(args.append_response),
                pending_path=pending_path,
            )
        try:
            result = await run_relay_conversation(
                case,
                ledger_path=ledger_path,
                pending_path=pending_path,
                model=args.model,
                conversation_id=args.conversation_id,
                turns_per_scene=args.turns_per_scene,
                manifest_fingerprint=manifest_fingerprint,
            )
        except RelayPendingRequest as pending:
            print(
                json.dumps(
                    {
                        "status": "pending",
                        "exit_code": RELAY_PENDING_EXIT_CODE,
                        "ledger": str(ledger_path),
                        "pending_request": str(pending_path),
                        "sequence": pending.sequence,
                        "actor_id": pending.request.actor_id,
                        "turn_index": pending.request.turn_index,
                    },
                    ensure_ascii=False,
                )
            )
            return RELAY_PENDING_EXIT_CODE
        output_dir = write_benchmark_artifacts([result], args.output)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(output_dir),
                    "ledger": str(ledger_path),
                    "model": args.model,
                    "live": False,
                    "cases": [case.case_id],
                    "scene_count": len(case.scenes),
                    "turn_count": len(result.turns),
                },
                ensure_ascii=False,
            )
        )
        return 0
    live_client = _live_client(args.model) if args.live else None
    results = await run_benchmark(
        cases,
        model=args.model,
        responder=None if args.live else _offline_responder,
        live_client=live_client,
        turns_per_scene=args.turns_per_scene,
    )
    output_dir = write_benchmark_artifacts(results, args.output)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "model": args.model,
                "live": bool(args.live),
                "cases": [case.case_id for case in cases],
                "scene_count": min(len(case.scenes) for case in cases),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
