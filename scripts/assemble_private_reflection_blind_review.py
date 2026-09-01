#!/usr/bin/env python3
"""Assemble a blinded, offline QA packet for sealed private reflections.

The packet is intentionally a projection, rather than a copy, of the run
artifacts.  It contains public dialogue, the authorized control context, and
the parsed six-field reflection for each accepted turn.  Raw model output and
run telemetry remain in the sealed source directories.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL_MANIFEST = (
    REPO_ROOT
    / "app/storage/playtest_reports/character-dialogue-benchmark/"
    "20260831-popular-dialogue-transfer/fixed-prompt-factorial/control-manifest.json"
)
EXPECTED_CONVERSATIONS_PER_CELL = 8
EXPECTED_TURN_COUNT = 16
REFLECTION_FIELDS = (
    "present_true_position",
    "public_attempt",
    "deliberately_unsaid_truth",
    "unavailable_because",
    "relationship_status_cost",
    "continuity_pressure",
)
_REFLECTION_TAG_RE = re.compile(r"actor_private_reflection", re.I)
_REFLECTION_NONCE_RE = re.compile(r"R-[0-9a-f]{32}", re.I)


class PrivateReflectionAssemblyError(ValueError):
    """The supplied artifacts cannot produce a safe blind-review packet."""


@dataclass(frozen=True)
class ControlActor:
    actor_id: str
    name: str
    public_sheet: dict[str, str]
    facts: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ControlCase:
    case_id: str
    actors: tuple[ControlActor, ControlActor]
    turn_actor_ids: tuple[str, ...]
    scenes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LedgerConversation:
    cell: str
    case_id: str
    conversation_id: str
    public_turns: tuple[str, ...]
    actor_ids: tuple[str, ...]


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrivateReflectionAssemblyError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PrivateReflectionAssemblyError(f"{label} must be a JSON object")
    return value


def _string(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise PrivateReflectionAssemblyError(f"{label} must be a nonempty string")
    return value


def _string_map(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise PrivateReflectionAssemblyError(f"{label} must be an object")
    result = {str(key): item for key, item in value.items()}
    if any(not isinstance(item, str) for item in result.values()):
        raise PrivateReflectionAssemblyError(f"{label} values must be strings")
    return result


def _load_control_actor(value: Any, *, case_id: str) -> ControlActor:
    if not isinstance(value, Mapping):
        raise PrivateReflectionAssemblyError(f"control case {case_id} actor is invalid")
    actor_id = _string(value.get("character_id"), label="control actor id")
    name = _string(value.get("name"), label="control actor name")
    public_sheet = _string_map(value.get("public_sheet"), label="control public_sheet")
    raw_actor = value.get("actor")
    if not isinstance(raw_actor, Mapping) or not isinstance(raw_actor.get("facts"), list):
        raise PrivateReflectionAssemblyError(f"control actor {actor_id} has no facts")
    facts: list[dict[str, str]] = []
    for raw_fact in raw_actor["facts"]:
        if not isinstance(raw_fact, Mapping):
            raise PrivateReflectionAssemblyError(f"control actor {actor_id} fact is invalid")
        facts.append(
            {
                "origin": _string(raw_fact.get("origin"), label="control fact origin"),
                "text": _string(raw_fact.get("text"), label="control fact text"),
            }
        )
    return ControlActor(actor_id, name, public_sheet, tuple(facts))


def _project_scene(scene: Mapping[str, Any], *, actor_names: Mapping[str, str]) -> dict[str, Any]:
    result = {
        "title": _string(scene.get("title"), label="control scene title"),
        "setup": _string(scene.get("frame"), label="control scene frame"),
    }
    between = scene.get("between_scene_public_history")
    if between is not None:
        result["between_scene_public_history"] = _string(
            between, label="control between-scene history"
        )
    raw_observations = scene.get("actor_observations")
    if raw_observations is not None:
        if not isinstance(raw_observations, Mapping):
            raise PrivateReflectionAssemblyError("control scene observations are invalid")
        observations: dict[str, list[str]] = {}
        for actor_id, raw_values in raw_observations.items():
            if actor_id not in actor_names or not isinstance(raw_values, list):
                raise PrivateReflectionAssemblyError("control scene observations are invalid")
            observations[actor_names[actor_id]] = [
                _string(item, label="control actor observation") for item in raw_values
            ]
        result["actor_observations"] = observations
    return result


def _load_controls(path: Path) -> dict[str, ControlCase]:
    document = _read_object(path, label="control manifest")
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != EXPECTED_CONVERSATIONS_PER_CELL:
        raise PrivateReflectionAssemblyError(
            f"control manifest must define exactly {EXPECTED_CONVERSATIONS_PER_CELL} cases"
        )
    controls: dict[str, ControlCase] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise PrivateReflectionAssemblyError("control case is invalid")
        case_id = _string(raw_case.get("case_id"), label="control case_id")
        raw_actors = raw_case.get("actors")
        raw_scenes = raw_case.get("scenes")
        if not isinstance(raw_actors, list) or len(raw_actors) != 2:
            raise PrivateReflectionAssemblyError(f"control case {case_id} must be dyadic")
        if not isinstance(raw_scenes, list) or not raw_scenes:
            raise PrivateReflectionAssemblyError(f"control case {case_id} has no scenes")
        actors = tuple(_load_control_actor(item, case_id=case_id) for item in raw_actors)
        if len({actor.actor_id for actor in actors}) != 2:
            raise PrivateReflectionAssemblyError(f"control case {case_id} has duplicate actors")
        actor_names = {actor.actor_id: actor.name for actor in actors}
        turn_actor_ids: list[str] = []
        scenes: list[dict[str, Any]] = []
        for raw_scene in raw_scenes:
            if not isinstance(raw_scene, Mapping):
                raise PrivateReflectionAssemblyError(f"control case {case_id} scene is invalid")
            raw_order = raw_scene.get("turn_order")
            if not isinstance(raw_order, list):
                raise PrivateReflectionAssemblyError(f"control case {case_id} scene has no turn order")
            for actor_id in raw_order:
                if actor_id not in actor_names:
                    raise PrivateReflectionAssemblyError(
                        f"control case {case_id} has an unknown turn actor"
                    )
                turn_actor_ids.append(actor_id)
            scenes.append(_project_scene(raw_scene, actor_names=actor_names))
        if len(turn_actor_ids) != EXPECTED_TURN_COUNT or set(turn_actor_ids) != set(actor_names):
            raise PrivateReflectionAssemblyError(
                f"control case {case_id} must define {EXPECTED_TURN_COUNT} dyadic turns"
            )
        if case_id in controls:
            raise PrivateReflectionAssemblyError(f"duplicate control case {case_id}")
        controls[case_id] = ControlCase(
            case_id=case_id,
            actors=(actors[0], actors[1]),
            turn_actor_ids=tuple(turn_actor_ids),
            scenes=tuple(scenes),
        )
    return controls


def _public_response(entry: Mapping[str, Any], *, label: str) -> str:
    payload = entry.get("response")
    if isinstance(payload, str):
        return _string(payload, label=label, allow_empty=True)
    if isinstance(payload, Mapping):
        return _string(payload.get("content"), label=label, allow_empty=True)
    raise PrivateReflectionAssemblyError(f"{label} must contain public response text")


def _assert_no_sealed_material(value: str, *, label: str) -> None:
    if _REFLECTION_TAG_RE.search(value) or _REFLECTION_NONCE_RE.search(value):
        raise PrivateReflectionAssemblyError(f"{label} contains sealed reflection material")


def _load_ledger(path: Path, *, cell: str, control: ControlCase) -> LedgerConversation:
    document = _read_object(path, label="response ledger")
    if document.get("case_id") != control.case_id:
        raise PrivateReflectionAssemblyError("response ledger case does not match control")
    conversation_id = _string(document.get("conversation_id"), label="ledger conversation_id")
    responses = document.get("responses")
    if not isinstance(responses, list) or len(responses) != EXPECTED_TURN_COUNT:
        raise PrivateReflectionAssemblyError(
            f"response ledger must have exactly {EXPECTED_TURN_COUNT} public turns"
        )
    public_turns: list[str] = []
    actor_ids: list[str] = []
    for sequence, entry in enumerate(responses):
        if not isinstance(entry, Mapping) or entry.get("sequence") != sequence:
            raise PrivateReflectionAssemblyError("response ledger has an invalid response sequence")
        request = entry.get("request")
        if not isinstance(request, Mapping):
            raise PrivateReflectionAssemblyError("response ledger response lacks a request")
        actor_id = _string(request.get("actor_id"), label="ledger actor_id")
        if actor_id != control.turn_actor_ids[sequence]:
            raise PrivateReflectionAssemblyError("response ledger actor order differs from control")
        text = _public_response(entry, label="ledger public response")
        _assert_no_sealed_material(text, label="ledger public response")
        public_turns.append(text)
        actor_ids.append(actor_id)
    return LedgerConversation(cell, control.case_id, conversation_id, tuple(public_turns), tuple(actor_ids))


def _load_cell_ledgers(cell: str, root: Path, controls: Mapping[str, ControlCase]) -> list[LedgerConversation]:
    ledgers = sorted((root / "conversations").glob("*/response_ledger.json"))
    if len(ledgers) != EXPECTED_CONVERSATIONS_PER_CELL:
        raise PrivateReflectionAssemblyError(
            f"cell {cell} must contain exactly {EXPECTED_CONVERSATIONS_PER_CELL} response ledgers"
        )
    result: list[LedgerConversation] = []
    for path in ledgers:
        document = _read_object(path, label="response ledger")
        case_id = _string(document.get("case_id"), label="ledger case_id")
        control = controls.get(case_id)
        if control is None:
            raise PrivateReflectionAssemblyError("response ledger case is not in the control manifest")
        result.append(_load_ledger(path, cell=cell, control=control))
    if {ledger.case_id for ledger in result} != set(controls):
        raise PrivateReflectionAssemblyError("cell ledgers must match control cases exactly once")
    if len({ledger.conversation_id for ledger in result}) != len(result):
        raise PrivateReflectionAssemblyError("cell has duplicate conversation ids")
    return result


def _reflection_fields(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(REFLECTION_FIELDS):
        raise PrivateReflectionAssemblyError("accepted reflection must have exactly six parsed fields")
    fields = {field: value[field] for field in REFLECTION_FIELDS}
    if any(not isinstance(item, str) for item in fields.values()):
        raise PrivateReflectionAssemblyError("accepted reflection fields must be strings")
    return fields


def _load_reflections(root: Path, ledger: LedgerConversation) -> dict[int, dict[str, str]]:
    artifacts = sorted((root / "private_reflection_qa").glob("**/turn-*-attempt-*.json"))
    matched: dict[int, dict[str, str]] = {}
    for path in artifacts:
        artifact = _read_object(path, label="private reflection artifact")
        if artifact.get("status") != "accepted":
            continue
        if artifact.get("conversation_id") != ledger.conversation_id:
            continue
        if artifact.get("case_id") != ledger.case_id:
            raise PrivateReflectionAssemblyError("accepted reflection case does not match its ledger")
        sequence = artifact.get("ledger_sequence")
        if not isinstance(sequence, int) or not 0 <= sequence < EXPECTED_TURN_COUNT:
            raise PrivateReflectionAssemblyError("accepted reflection has an invalid turn sequence")
        if artifact.get("actor_id") != ledger.actor_ids[sequence]:
            raise PrivateReflectionAssemblyError("accepted reflection actor does not match its ledger turn")
        reflection = artifact.get("reflection")
        if not isinstance(reflection, Mapping):
            raise PrivateReflectionAssemblyError("accepted artifact has no parsed reflection")
        fields = _reflection_fields(reflection.get("fields"))
        if sequence in matched:
            raise PrivateReflectionAssemblyError("duplicate accepted reflection for one ledger turn")
        matched[sequence] = fields
    if set(matched) != set(range(EXPECTED_TURN_COUNT)):
        raise PrivateReflectionAssemblyError(
            f"conversation must have exactly {EXPECTED_TURN_COUNT} accepted reflections"
        )
    return matched


def _project_conversation(
    *,
    blind_id: str,
    control: ControlCase,
    ledger: LedgerConversation,
    reflections: Mapping[int, Mapping[str, str]],
) -> dict[str, Any]:
    names = {actor.actor_id: actor.name for actor in control.actors}
    result = {
        "blind_id": blind_id,
        "characters": [
            {
                "name": actor.name,
                "public_sheet": actor.public_sheet,
                "authorized_facts": list(actor.facts),
            }
            for actor in control.actors
        ],
        "scenes": list(control.scenes),
        "turns": [
            {
                "turn_index": sequence + 1,
                "speaker": names[actor_id],
                "public_response": ledger.public_turns[sequence],
                "private_reflection": dict(reflections[sequence]),
            }
            for sequence, actor_id in enumerate(ledger.actor_ids)
        ],
    }
    _assert_packet_safe(result)
    return result


def _assert_packet_safe(value: Any) -> None:
    forbidden_keys = {
        "cell",
        "cell_id",
        "case_id",
        "conversation_id",
        "source_path",
        "source_protocol",
        "nonce",
        "suffix",
        "raw_response",
        "worker_session_id",
        "session_id",
        "prompt_architecture",
        "sha256",
        "hash",
    }

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            if forbidden_keys & set(item):
                raise PrivateReflectionAssemblyError("blind packet contains source metadata")
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            _assert_no_sealed_material(item, label="blind packet")

    visit(value)


def assemble_private_reflection_blind_review(
    *,
    cell_directories: Mapping[str, Path],
    shuffle_seed: str,
    control_manifest_path: Path = DEFAULT_CONTROL_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not shuffle_seed:
        raise PrivateReflectionAssemblyError("shuffle seed must be nonempty")
    if not cell_directories or any(not label for label in cell_directories):
        raise PrivateReflectionAssemblyError("at least one cell directory is required")
    controls = _load_controls(control_manifest_path)
    candidates = [
        ledger
        for cell, root in sorted(cell_directories.items())
        for ledger in _load_cell_ledgers(cell, root, controls)
    ]
    random.Random(shuffle_seed).shuffle(candidates)
    conversations: list[dict[str, Any]] = []
    answer_key: list[dict[str, str]] = []
    for index, ledger in enumerate(candidates, start=1):
        blind_id = f"reflection-{index:02d}"
        reflections = _load_reflections(cell_directories[ledger.cell], ledger)
        conversations.append(
            _project_conversation(
                blind_id=blind_id,
                control=controls[ledger.case_id],
                ledger=ledger,
                reflections=reflections,
            )
        )
        answer_key.append(
            {
                "blind_id": blind_id,
                "cell": ledger.cell,
                "case": ledger.case_id,
                "conversation": ledger.conversation_id,
            }
        )
    return (
        {
            "schema_version": "private_reflection_blind_review_v1",
            "source_hidden": True,
            "conversations": conversations,
        },
        {
            "schema_version": "private_reflection_blind_review_key_v1",
            "conversations": answer_key,
        },
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def parse_cell_directory(value: str) -> tuple[str, Path]:
    cell, separator, raw_path = value.partition("=")
    if not separator or not cell or not raw_path:
        raise argparse.ArgumentTypeError("cell directory must use LABEL=DIR")
    return cell, Path(raw_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", action="append", type=parse_cell_directory, required=True)
    parser.add_argument("--control-manifest", type=Path, default=DEFAULT_CONTROL_MANIFEST)
    parser.add_argument("--shuffle-seed", required=True)
    parser.add_argument("--packet-output", type=Path, required=True)
    parser.add_argument("--answer-key-output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cell_directories = dict(args.cell_dir)
    if len(cell_directories) != len(args.cell_dir):
        raise SystemExit("each --cell-dir label must be unique")
    try:
        packet, answer_key = assemble_private_reflection_blind_review(
            cell_directories=cell_directories,
            shuffle_seed=args.shuffle_seed,
            control_manifest_path=args.control_manifest,
        )
        _atomic_write_json(args.packet_output, packet)
        _atomic_write_json(args.answer_key_output, answer_key)
    except PrivateReflectionAssemblyError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
