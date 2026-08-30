#!/usr/bin/env python3
# ruff: noqa: E402 -- executable script adds the repository root to sys.path.
"""Run an offline or explicit-model serial character dialogue benchmark.

This is a small laboratory for the actor boundary, not a second game engine.
Each actor receives its own lived material and its own complete rolling
conversation.  The pair then meets in at least two separated scenes.  The
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
import hashlib
import inspect
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.llm.client import LLMClient, LLMResponse
from app.llm.config import LLMConfig
from app.schemas.characters import CharacterRecord, PrivateState, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.state import (
    ModelConfig,
    SessionConfig,
    SessionSettings,
    SessionState,
    WorldState,
)


DEFAULT_MODEL = "gpt-5.6-luna"
IDENTITY_MODES = ("named", "deidentified", "both")
RUN_IDENTITY_MODES = ("named", "deidentified")
SUITES = ("ordinary_surface", "pressure")
DEFAULT_IDENTITY_MODE = "both"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "scripts" / "character_dialogue_benchmark_manifest.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "app" / "storage" / "playtest_reports" / "character-dialogue-benchmark"
)
EXACT_SILENCE = "<silence/>"

# These are intentionally reviewer-only.  They never occur in actor system or
# user messages.  A reviewer gets the complete serial transcript and fills in
# the fields by hand rather than having a second model grade a line in
# isolation.
HUMAN_REVIEW_FIELDS = (
    "attempts",
    "literal_and_interpersonal_subject",
    "knowledge_sources_and_unknowns",
    "unavailable_line",
    "status_and_topic_control",
    "answer_debt",
    "ritual_deviation",
    "rhythm",
    "biography_consequence",
    "articulation_ceiling",
    "voice_swappability",
)

HUMAN_REVIEW_QUESTIONS: Mapping[str, str] = {
    "attempts": "What is each speaker trying to make the other person do, admit, or avoid at each point?",
    "literal_and_interpersonal_subject": "What is the ordinary subject on the surface, and what are the speakers doing to one another through it?",
    "knowledge_sources_and_unknowns": "For each consequential fact, what does each speaker know, suspect, misunderstand, or not know, and how could they know it?",
    "unavailable_line": "What plain sentence could each speaker say but cannot presently afford to say?",
    "status_and_topic_control": "Who chooses the topic, who must answer, who may refuse, and where does that control change?",
    "answer_debt": "Which question, offer, correction, or bid goes unanswered, and what later line inherits that debt?",
    "ritual_deviation": "What repeated routine establishes their relationship, and what does a deviation from it cost?",
    "rhythm": "How do length, interruption, silence, repetition, and pressure or release shape the whole sequence?",
    "biography_consequence": "Which lived detail changes a present choice, and which details are only decorative explanation?",
    "articulation_ceiling": "Do the speakers differ in precision, wit, emotional insight, and willingness to explain, or do both sound equally model-like?",
    "voice_swappability": "Without names, what attention, vocabulary, rhythm, or social behavior identifies each speaker? Could either deliver the other’s lines unchanged?",
}


class BenchmarkOutputError(ValueError):
    """The actor response is not observable prose or exact silence."""


class BenchmarkManifestError(ValueError):
    """The benchmark manifest is malformed or missing required data."""


@dataclass(frozen=True)
class ActorDossier:
    """Lived material available to one actor and no other actor."""

    actor_id: str
    display_name: str
    lived_facts: tuple[str, ...]
    habits: tuple[str, ...]
    concrete_wants: tuple[str, ...]
    withheld_acts: tuple[str, ...]
    known_facts: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()


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
    named_turn_order: tuple[str, ...] = ()

    def order_for(self, identity_mode: str) -> tuple[str, ...]:
        if identity_mode == "named":
            if not self.named_turn_order:
                raise BenchmarkManifestError(
                    f"scene {self.scene_id!r} has no named turn order"
                )
            return self.named_turn_order
        return self.turn_order


@dataclass(frozen=True)
class IdentityVariant:
    identity_mode: str
    actors: tuple[ActorDossier, ...]
    scenes: tuple[SceneSpec, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    title: str
    suite: str
    scenes: tuple[SceneSpec, ...]
    actors: tuple[ActorDossier, ...]
    source_metadata: Mapping[str, Any]
    named_actors: tuple[ActorDossier, ...] = ()

    def variant(self, identity_mode: str) -> IdentityVariant:
        selected = validate_identity_mode(identity_mode, allow_both=False)
        if selected == "deidentified":
            actors = self.actors
        else:
            if not self.named_actors:
                raise BenchmarkManifestError(
                    f"case {self.case_id!r} has no named semantic twin"
                )
            actors = self.named_actors
        return IdentityVariant(selected, actors, self.scenes)

    def actor(self, actor_id: str, identity_mode: str = "deidentified") -> ActorDossier:
        for actor in self.variant(identity_mode).actors:
            if actor.actor_id == actor_id:
                return actor
        raise BenchmarkManifestError(
            f"case {self.case_id!r} has no actor {actor_id!r}"
        )


@dataclass(frozen=True)
class BenchmarkRequest:
    """Complete input for one isolated actor call."""

    conversation_id: str
    case_id: str
    scene_id: str
    scene_index: int
    scene_turn_index: int
    turn_index: int
    actor_id: str
    actor_name: str
    model: str
    identity_mode: str
    messages: tuple[Mapping[str, Any], ...]
    temperature: float = 0.7
    max_tokens: int = 900


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


@dataclass
class ConversationResult:
    conversation_id: str
    case: BenchmarkCase
    model: str
    identity_mode: str
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
            "identity_mode": self.identity_mode,
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
                        "identity_mode": self.identity_mode,
                        "scene_index": turn.scene_index,
                        "scene_id": turn.scene_id,
                        "scene_turn_index": turn.scene_turn_index,
                        "turn_index": turn.turn_index,
                        "temperature": 0.7,
                        "max_tokens": 900,
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


def _clean_string(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise BenchmarkManifestError(f"{label} cannot be blank")
    if any(ord(char) < 32 and char not in "\n\t" for char in result):
        raise BenchmarkManifestError(f"{label} contains a control character")
    return result


def _clean_text_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise BenchmarkManifestError(f"{label} must be a non-empty list")
    return tuple(
        _clean_string(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


def _clean_optional_text_list(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise BenchmarkManifestError(f"{label} must be a list when supplied")
    return tuple(
        _clean_string(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


def _parse_dossier(raw: Any, *, case_id: str, actor_id: str) -> ActorDossier:
    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError(f"dossier for {case_id}/{actor_id} must be an object")
    return ActorDossier(
        actor_id=actor_id,
        display_name=_clean_string(
            raw.get("display_name"), f"{case_id}/{actor_id}.display_name"
        ),
        lived_facts=_clean_text_list(
            raw.get("lived_facts"), f"{case_id}/{actor_id}.lived_facts"
        ),
        habits=_clean_text_list(raw.get("habits"), f"{case_id}/{actor_id}.habits"),
        concrete_wants=_clean_text_list(
            raw.get("concrete_wants"), f"{case_id}/{actor_id}.concrete_wants"
        ),
        withheld_acts=_clean_text_list(
            raw.get("withheld_acts"), f"{case_id}/{actor_id}.withheld_acts"
        ),
        known_facts=_clean_optional_text_list(
            raw.get("known_facts"), f"{case_id}/{actor_id}.known_facts"
        ),
        assumptions=_clean_optional_text_list(
            raw.get("what_you_take_for_granted"),
            f"{case_id}/{actor_id}.what_you_take_for_granted",
        ),
    )


def _parse_prior_public_exchange(
    raw: Any,
    *,
    case_id: str,
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
        if speaker_slot not in (0, 1):
            raise BenchmarkManifestError(
                f"{case_id}.prior_public_exchange[{index}].speaker_slot must be 0 or 1"
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
    if required and seen_slots != {0, 1}:
        raise BenchmarkManifestError(
            f"{case_id}.prior_public_exchange must include both speaker slots"
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


def _parse_scene(
    raw: Any,
    *,
    case_id: str,
    scene_index: int,
    actor_ids: set[str],
    named_actor_ids: set[str],
) -> SceneSpec:
    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError(f"{case_id}.scenes[{scene_index}] must be an object")
    prefix = f"{case_id}.scenes[{scene_index}]"
    scene_id = _clean_string(raw.get("scene_id"), f"{prefix}.scene_id")
    title = _clean_string(raw.get("title", scene_id), f"{prefix}.title")
    frame_value = raw.get("frame", raw.get("scene_frame"))
    frame = _clean_string(frame_value, f"{prefix}.frame")
    if "<" in frame or ">" in frame:
        raise BenchmarkManifestError(f"{prefix}.frame cannot contain prompt markup")
    raw_prior = raw.get("prior_public_exchange")
    prior = _parse_prior_public_exchange(
        raw_prior,
        case_id=prefix,
        required=raw_prior is not None,
    )
    order = _parse_order(raw.get("turn_order"), case_id=prefix, actor_ids=actor_ids)
    named_order_raw = raw.get("named_turn_order")
    if named_order_raw is None:
        # Named and de-identified manifests normally have the same positional
        # order.  The explicit named order can still be supplied per scene.
        named_order = ()
    else:
        named_order = _parse_order(
            named_order_raw,
            case_id=prefix,
            actor_ids=named_actor_ids,
            field_name="named_turn_order",
        )
    if named_order and len(named_order) != len(order):
        raise BenchmarkManifestError(
            f"{prefix} named and de-identified twins must have matching turn counts"
        )
    pulses = _parse_pressure_pulses(
        raw.get("pressure_pulses", []), case_id=prefix, turn_count=len(order)
    )
    return SceneSpec(
        scene_id=scene_id,
        title=title,
        frame=frame,
        prior_public_exchange=prior,
        turn_order=order,
        pressure_pulses=pulses,
        named_turn_order=named_order,
    )


def _parse_actor_list(
    raw: Any,
    *,
    case_id: str,
    field_name: str,
) -> tuple[ActorDossier, ...]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise BenchmarkManifestError(
            f"{case_id} must contain exactly two {field_name} actors"
        )
    actors: list[ActorDossier] = []
    actor_ids: set[str] = set()
    for index, actor_raw in enumerate(raw):
        if not isinstance(actor_raw, Mapping):
            raise BenchmarkManifestError(
                f"{case_id}.{field_name}[{index}] must be an object"
            )
        actor_id = _clean_string(
            actor_raw.get("actor_id"), f"{case_id}.{field_name}[{index}].actor_id"
        )
        if actor_id in actor_ids:
            raise BenchmarkManifestError(
                f"{case_id} repeats {field_name} actor {actor_id!r}"
            )
        actor_ids.add(actor_id)
        dossier_raw = dict(actor_raw.get("dossier") or {})
        dossier_raw["display_name"] = actor_raw.get("display_name")
        actors.append(_parse_dossier(dossier_raw, case_id=case_id, actor_id=actor_id))
    return tuple(actors)


def _fallback_serial_scenes(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Turn the original one-meeting shape into an explicit two-scene case.

    New manifests should author ``scenes`` directly.  This narrow import path
    permits old evidence to be replayed while making the runtime and review
    unit unambiguously serial.  Scene two can become specific only through the
    public candidate transcript produced by scene one.
    """

    return [
        {
            "scene_id": "scene_1",
            "title": "First meeting",
            "frame": raw.get("scene_frame"),
            "prior_public_exchange": raw.get("prior_public_exchange"),
            "turn_order": raw.get("turn_order"),
            "named_turn_order": raw.get("named_turn_order"),
            "pressure_pulses": raw.get("pressure_pulses", []),
        },
        {
            "scene_id": "scene_2",
            "title": "The return",
            "frame": (
                "Later, after the first meeting has ended, the same two people meet "
                "again. The practical arrangement they left between them now needs "
                "another decision."
            ),
            "turn_order": raw.get("turn_order"),
            "named_turn_order": raw.get("named_turn_order"),
            "pressure_pulses": [],
        },
    ]


def _parse_case(raw: Any, index: int) -> BenchmarkCase:
    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError(f"case {index} must be an object")
    case_id = _clean_string(raw.get("case_id"), f"cases[{index}].case_id")
    suite = _clean_string(raw.get("suite"), f"{case_id}.suite").lower()
    if suite not in SUITES:
        raise BenchmarkManifestError(
            f"{case_id}.suite must be one of {list(SUITES)}, got {suite!r}"
        )
    actors = _parse_actor_list(raw.get("actors"), case_id=case_id, field_name="actors")
    named_actors = _parse_actor_list(
        raw.get("named_actors"), case_id=case_id, field_name="named"
    )
    actor_ids = {actor.actor_id for actor in actors}
    named_actor_ids = {actor.actor_id for actor in named_actors}
    scenes_raw = raw.get("scenes")
    if scenes_raw is None:
        scenes_raw = _fallback_serial_scenes(raw)
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
            named_actor_ids=named_actor_ids,
        )
        # The legacy root shape carries a named order on every scene.  New
        # scene objects may omit it only when they use the same actor ids.
        if not scene.named_turn_order:
            named_order = tuple(
                named_actors[0 if actor_id == actors[0].actor_id else 1].actor_id
                for actor_id in scene.turn_order
            )
            scene = SceneSpec(
                scene_id=scene.scene_id,
                title=scene.title,
                frame=scene.frame,
                prior_public_exchange=scene.prior_public_exchange,
                turn_order=scene.turn_order,
                pressure_pulses=scene.pressure_pulses,
                named_turn_order=named_order,
            )
        scenes.append(scene)
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
    ).casefold()
    for actor in (*actors, *named_actors):
        actor_local_values = (
            *actor.lived_facts,
            *actor.habits,
            *actor.concrete_wants,
            *actor.withheld_acts,
            *actor.known_facts,
            *actor.assumptions,
        )
        for private_value in actor_local_values:
            if private_value.casefold() in public_text:
                raise BenchmarkManifestError(
                    f"{case_id} public text repeats actor-local material for {actor.actor_id}"
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
        named_actors=named_actors,
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
    identity_mode: str = "deidentified",
    conversation_id: str | None = None,
) -> CheckpointFile:
    """Create an audit envelope without duplicating actor material in state."""

    selected_identity_mode = validate_identity_mode(identity_mode, allow_both=False)
    selected_model = _clean_string(model, "model")
    run_id = conversation_id or f"{case.case_id}-{uuid.uuid4().hex}"
    variant = case.variant(selected_identity_mode)
    characters = [
        CharacterRecord(
            character_id=actor.actor_id,
            name=actor.display_name,
            location="benchmark_room",
            public_sheet=PublicSheet(role="participant"),
            backstory="",
            personality="",
            private_state=PrivateState(),
        )
        for actor in variant.actors
    ]
    return CheckpointFile(
        session=SessionState(
            session_id=f"character-dialogue-benchmark:{run_id}",
            story_id=f"character-dialogue-benchmark:{case.case_id}:{selected_identity_mode}",
            config=SessionConfig(
                models=_model_config(selected_model),
                settings=SessionSettings(ruleset_id=""),
            ),
        ),
        player_primer="",
        world_state=WorldState(),
        characters=characters,
        character_conversations={actor.actor_id: [] for actor in variant.actors},
    )


def validate_identity_mode(identity_mode: str, *, allow_both: bool = True) -> str:
    selected = str(identity_mode or "").strip().lower()
    choices = IDENTITY_MODES if allow_both else RUN_IDENTITY_MODES
    if selected not in choices:
        raise BenchmarkManifestError(
            f"identity_mode must be one of {list(choices)}, got {identity_mode!r}"
        )
    return selected


def _render_dossier(actor: ActorDossier) -> str:
    def render_list(label: str, values: Sequence[str]) -> str:
        return "\n".join(
            [f"<{label}>", *(f"- {value}" for value in values), f"</{label}>"]
        )

    sections = [
        "<your_life>",
        f"<your_name>{actor.display_name}</your_name>",
        render_list("things_you_have_lived", actor.lived_facts),
        render_list("things_you_do", actor.habits),
        render_list("what_you_want_now", actor.concrete_wants),
        render_list("things_you_have_not_done", actor.withheld_acts),
    ]
    if actor.known_facts:
        sections.append(render_list("things_you_have_witnessed", actor.known_facts))
    if actor.assumptions:
        sections.append(render_list("things_you_assume", actor.assumptions))
    sections.append("</your_life>")
    return "\n".join(sections)


def _render_actor_system(actor: ActorDossier) -> str:
    """Render a compact actor-local prompt with no reviewer vocabulary."""

    return "\n\n".join(
        (
            f"You are {actor.display_name}. This is your life, not an exercise "
            "about a character. Let the moment reach you through what you have "
            "lived, noticed, wanted, and chosen not to do.",
            _render_dossier(actor),
            "<instructions>\n"
            "Use only what you have witnessed and the material in your life. "
            "Another person's undisclosed life is unavailable to you; do not "
            "invent it or expose it. Choose only your own words, actions, and "
            "silences. Make one public move in response to the latest moment, "
            "then stop and leave room for the other person. An intentional "
            f"public pause is written exactly as {EXACT_SILENCE}. Do not explain "
            "your motives, the writing, or the exchange, and do not write the "
            "other person's reply.\n"
            "</instructions>",
        )
    )


def _render_public_updates(public_updates: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for entry in public_updates:
        kind = str(entry.get("kind", ""))
        text = str(entry.get("text", "")).strip()
        if kind == "turn":
            lines.append(
                f"Scene {entry.get('scene_index')}, turn {entry.get('turn_index')}, "
                f"{entry.get('speaker_name')}: {text}"
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
        else:
            lines.append(text)
    return "\n".join(lines) or "Nothing new has reached you."


def render_actor_messages(
    case: BenchmarkCase,
    actor_id: str,
    *,
    scene_index: int = 0,
    scene_turn_index: int = 1,
    turn_index: int | None = None,
    identity_mode: str = "deidentified",
    history: Sequence[Mapping[str, Any] | ConversationMessage] = (),
    public_updates: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Build one actor request from its full, untrimmed rolling history."""

    variant = case.variant(identity_mode)
    if scene_index < 0 or scene_index >= len(variant.scenes):
        raise BenchmarkManifestError(
            f"scene_index must be between 0 and {len(variant.scenes) - 1}"
        )
    actor = case.actor(actor_id, identity_mode=identity_mode)
    actual_turn_index = turn_index if turn_index is not None else scene_turn_index
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _render_actor_system(actor)}
    ]
    for message in history:
        if isinstance(message, ConversationMessage):
            messages.append({"role": message.role, "content": message.content})
        else:
            messages.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": message.get("content", ""),
                }
            )
    messages.append(
        {
            "role": "user",
            "content": "\n\n".join(
                (
                    f"<scene_index>{scene_index}</scene_index>",
                    f"<scene_turn>{scene_turn_index}</scene_turn>",
                    f"<turn>{actual_turn_index}</turn>",
                    "<public_moment>",
                    _render_public_updates(public_updates),
                    "</public_moment>",
                )
            ),
        }
    )
    return messages


_MARKUP_RE = re.compile(r"<[^>]*>")


def parse_observable_response(response: str) -> str:
    """Accept public prose or exact silence, and reject hidden channels."""

    text = str(response or "").strip()
    if not text:
        raise BenchmarkOutputError("response cannot be blank")
    if text == EXACT_SILENCE:
        return text
    if _MARKUP_RE.search(text):
        raise BenchmarkOutputError(
            "response must be observable prose or the exact <silence/> token"
        )
    return text


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
    variant: IdentityVariant,
) -> None:
    if scene_index > 0:
        public_transcript.append(
            {
                "kind": "scene_break",
                "scene_index": scene_index,
                "text": "The previous meeting ended; time has passed before this return.",
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
                "actor_id": variant.actors[entry.speaker_slot].actor_id,
                "speaker_name": variant.actors[entry.speaker_slot].display_name,
                "text": entry.text,
            }
        )


async def run_conversation(
    case: BenchmarkCase,
    *,
    model: str = DEFAULT_MODEL,
    identity_mode: str = "deidentified",
    responder: Responder | None = None,
    turns_per_scene: int | None = None,
    conversation_id: str | None = None,
) -> ConversationResult:
    """Run every authored scene with independent actors and full histories."""

    selected_model = _clean_string(model, "model")
    selected_identity_mode = validate_identity_mode(identity_mode, allow_both=False)
    variant = case.variant(selected_identity_mode)
    if turns_per_scene is not None and turns_per_scene < 1:
        raise ValueError("turns_per_scene must be at least 1")
    run_id = conversation_id or (
        f"{case.case_id}-{selected_identity_mode}-{uuid.uuid4().hex}"
    )
    checkpoint = new_synthetic_checkpoint(
        case,
        model=selected_model,
        identity_mode=selected_identity_mode,
        conversation_id=run_id,
    )
    initial_checkpoint_sha256 = _sha256_json(checkpoint.model_dump(mode="json"))
    responder_fn = responder or _offline_responder
    public_transcript: list[dict[str, Any]] = []
    seen_public_counts = {actor.actor_id: 0 for actor in variant.actors}
    turn_results: list[TurnResult] = []
    global_turn_index = 0

    for scene_index, scene in enumerate(variant.scenes):
        _append_scene_public_start(
            public_transcript,
            scene=scene,
            scene_index=scene_index,
            variant=variant,
        )
        order = scene.order_for(selected_identity_mode)
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
            actor = case.actor(actor_id, identity_mode=selected_identity_mode)
            updates = public_transcript[seen_public_counts[actor_id] :]
            messages = render_actor_messages(
                case,
                actor_id,
                scene_index=scene_index,
                scene_turn_index=scene_turn_index,
                turn_index=global_turn_index,
                identity_mode=selected_identity_mode,
                history=checkpoint.character_conversations[actor_id],
                public_updates=updates,
            )
            request = BenchmarkRequest(
                conversation_id=run_id,
                case_id=case.case_id,
                scene_id=scene.scene_id,
                scene_index=scene_index,
                scene_turn_index=scene_turn_index,
                turn_index=global_turn_index,
                actor_id=actor_id,
                actor_name=actor.display_name,
                model=selected_model,
                identity_mode=selected_identity_mode,
                messages=tuple(messages),
            )
            started_at = time.perf_counter()
            call = await _call_responder(responder_fn, request)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            raw_text = str(call.content or "").strip()
            public_text = parse_observable_response(raw_text)
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
                    "speaker_name": actor.display_name,
                    "text": public_text,
                }
            )
            checkpoint.character_conversations[actor_id].extend(
                (
                    ConversationMessage(role="user", content=messages[-1]["content"]),
                    ConversationMessage(role="assistant", content=public_text),
                )
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
                    actor_name=actor.display_name,
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
                )
            )

    return ConversationResult(
        conversation_id=run_id,
        case=case,
        model=selected_model,
        identity_mode=selected_identity_mode,
        checkpoint=checkpoint,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        turns=turn_results,
        public_transcript=public_transcript,
    )


async def run_benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    model: str = DEFAULT_MODEL,
    identity_mode: str = DEFAULT_IDENTITY_MODE,
    responder: Responder | None = None,
    turns_per_scene: int | None = None,
) -> list[ConversationResult]:
    """Run cases independently; no checkpoint or actor history is shared."""

    selected_identity_mode = validate_identity_mode(identity_mode)
    identity_modes = (
        RUN_IDENTITY_MODES
        if selected_identity_mode == "both"
        else (selected_identity_mode,)
    )
    results: list[ConversationResult] = []
    for identity_variant in identity_modes:
        for case in cases:
            results.append(
                await run_conversation(
                    case,
                    model=model,
                    identity_mode=identity_variant,
                    responder=responder,
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
        actor_id: result.case.actor(
            actor_id, identity_mode=result.identity_mode
        ).display_name
        for actor_id in actor_labels
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
            {"scene_index": index, "scene_id": scene.scene_id, "title": scene.title}
            for index, scene in enumerate(result.case.scenes)
        ],
        "transcript": transcript,
        "speaker_sheets": [
            {
                "speaker": label,
                "whole_transcript": transcript,
                "instruction": "Review the complete sequence, including both scenes; do not score isolated lines.",
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
            "identity_mode": result.identity_mode,
            "suite": result.case.suite,
            "title": result.case.title,
            "scenes": [_scene_artifact(scene) for scene in result.case.scenes],
            "model": result.model,
            "checkpoint_id": result.checkpoint_id,
            "speaker_mapping": {
                label: actor_id for actor_id, label in actor_labels.items()
            },
            "actor_names": {
                label: result.case.actor(
                    actor_id, identity_mode=result.identity_mode
                ).display_name
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
        mode_raw_dir = raw_dir / result.identity_mode
        _write_json(mode_raw_dir / f"{result.case.case_id}.json", artifact)
        calls_path = mode_raw_dir / f"{result.case.case_id}.jsonl"
        call_lines = []
        for turn in result.turns:
            call_lines.append(
                json.dumps(
                    {
                        "conversation_id": result.conversation_id,
                        "case_id": result.case.case_id,
                        "identity_mode": result.identity_mode,
                        "scene_index": turn.scene_index,
                        "scene_id": turn.scene_id,
                        "scene_turn_index": turn.scene_turn_index,
                        "turn_index": turn.turn_index,
                        "request": {
                            "model": result.model,
                            "identity_mode": result.identity_mode,
                            "scene_index": turn.scene_index,
                            "scene_id": turn.scene_id,
                            "scene_turn_index": turn.scene_turn_index,
                            "turn_index": turn.turn_index,
                            "temperature": 0.7,
                            "max_tokens": 900,
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
                "identity_mode": result.identity_mode,
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


def _live_responder(model: str) -> Responder:
    """Build an actor client with the requested model, never role defaults."""

    configured_model = _clean_string(model, "model")
    provider: str | None = None
    if ":" in configured_model:
        provider, configured_model = configured_model.split(":", 1)
        provider = provider.strip().lower()
        if provider not in {"openai", "anthropic"}:
            raise ValueError("model provider prefix must be openai or anthropic")
        configured_model = _clean_string(configured_model, "model")
    config = LLMConfig.from_env()
    role_models = dict(config.role_models)
    role_models["agent"] = configured_model
    role_providers = dict(config.role_providers)
    if provider:
        role_providers["agent"] = provider
    client = LLMClient(
        config.model_copy(
            update={"role_models": role_models, "role_providers": role_providers}
        )
    )

    async def call(request: BenchmarkRequest) -> ModelCall:
        response = await client.complete(
            role="agent",
            messages=[dict(message) for message in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            cache=False,
            compact=False,
        )
        return ModelCall(
            content=response.content,
            model=response.model or request.model,
            provider=provider or "configured",
            usage=response.usage,
            raw_response=response.raw_response,
            assistant_content=response.assistant_content,
        )

    return call


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
        "--identity-mode",
        choices=IDENTITY_MODES,
        default=DEFAULT_IDENTITY_MODE,
        help="Run named, de-identified, or both semantic-twin identities.",
    )
    parser.add_argument(
        "--turns-per-scene",
        type=int,
        default=None,
        help="Optionally cap each authored scene while retaining every scene.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the explicitly selected actor model. The default is offline.",
    )
    return parser


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
    responder = _live_responder(args.model) if args.live else _offline_responder
    results = await run_benchmark(
        cases,
        model=args.model,
        identity_mode=args.identity_mode,
        responder=responder,
        turns_per_scene=args.turns_per_scene,
    )
    output_dir = write_benchmark_artifacts(results, args.output)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "model": args.model,
                "identity_mode": args.identity_mode,
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
