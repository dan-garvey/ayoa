#!/usr/bin/env python3
"""Assemble blinded whole-conversation packets from fixed Luna silo ledgers.

This tool deliberately reads only public ``response_ledger.json`` files.  It
does not inspect scheduler telemetry or ``private_reflection_qa`` artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL_MANIFEST = (
    REPO_ROOT
    / "app/storage/playtest_reports/character-dialogue-benchmark/"
    "20260831-popular-dialogue-transfer/fixed-prompt-factorial/control-manifest.json"
)
FIXED_TURN_COUNT = 16
SOURCE_PROTOCOL = "persistent_luna_dialogue_silos_v2"
REFLECTION_FIELD_NAMES = (
    "present_true_position",
    "public_attempt",
    "deliberately_unsaid_truth",
    "unavailable_because",
    "relationship_status_cost",
    "continuity_pressure",
)
_REFLECTION_TAG_RE = re.compile(r"actor_private_reflection", re.I)
_REFLECTION_NONCE_RE = re.compile(r"R-[0-9a-f]{32}", re.I)


class BlindReviewAssemblyError(ValueError):
    """The supplied artifacts cannot produce a safe, comparable review packet."""


@dataclass(frozen=True)
class CaseControl:
    case_id: str
    actor_ids: tuple[str, str]
    actor_names: tuple[str, str]
    expected_turn_actor_ids: tuple[str, ...]


@dataclass(frozen=True)
class LedgerConversation:
    cell: str
    case_id: str
    conversation_id: str
    public_texts: tuple[str, ...]
    actor_ids: tuple[str, ...]


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlindReviewAssemblyError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise BlindReviewAssemblyError(f"{label} {path} must be a JSON object")
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BlindReviewAssemblyError(f"{label} must be a nonempty string")
    return value


def _load_controls(path: Path) -> dict[str, CaseControl]:
    document = _read_object(path, label="control manifest")
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise BlindReviewAssemblyError("control manifest must contain cases")
    controls: dict[str, CaseControl] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise BlindReviewAssemblyError("control manifest case must be an object")
        case_id = _string(raw_case.get("case_id"), label="control case_id")
        raw_actors = raw_case.get("actors")
        if not isinstance(raw_actors, list) or len(raw_actors) != 2:
            raise BlindReviewAssemblyError(f"control case {case_id} must be dyadic")
        actor_ids: list[str] = []
        actor_names: list[str] = []
        for raw_actor in raw_actors:
            if not isinstance(raw_actor, Mapping):
                raise BlindReviewAssemblyError(f"control case {case_id} actor is invalid")
            actor_ids.append(_string(raw_actor.get("character_id"), label="actor id"))
            actor_names.append(_string(raw_actor.get("name"), label="actor name"))
        raw_scenes = raw_case.get("scenes")
        if not isinstance(raw_scenes, list):
            raise BlindReviewAssemblyError(f"control case {case_id} has no scenes")
        turn_actor_ids = tuple(
            actor_id
            for raw_scene in raw_scenes
            if isinstance(raw_scene, Mapping)
            for actor_id in raw_scene.get("turn_order", [])
        )
        if (
            len(turn_actor_ids) != FIXED_TURN_COUNT
            or set(turn_actor_ids) != set(actor_ids)
            or any(not isinstance(actor_id, str) for actor_id in turn_actor_ids)
        ):
            raise BlindReviewAssemblyError(
                f"control case {case_id} must define {FIXED_TURN_COUNT} dyadic turns"
            )
        if case_id in controls:
            raise BlindReviewAssemblyError(f"duplicate control case {case_id}")
        controls[case_id] = CaseControl(
            case_id=case_id,
            actor_ids=(actor_ids[0], actor_ids[1]),
            actor_names=(actor_names[0], actor_names[1]),
            expected_turn_actor_ids=turn_actor_ids,
        )
    return controls


def _load_template_index(
    packet_path: Path, key_path: Path, controls: Mapping[str, CaseControl]
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    packet = _read_object(packet_path, label="blind template")
    answer_key = _read_object(key_path, label="template answer key")
    raw_packets = packet.get("conversations")
    raw_keys = answer_key.get("conversations")
    if not isinstance(raw_packets, list) or not isinstance(raw_keys, list):
        raise BlindReviewAssemblyError("template packet and key need conversations")
    key_by_blind_id: dict[str, dict[str, Any]] = {}
    for raw_key in raw_keys:
        if not isinstance(raw_key, dict):
            raise BlindReviewAssemblyError("template answer-key conversation is invalid")
        blind_id = _string(raw_key.get("blind_id"), label="template key blind_id")
        if blind_id in key_by_blind_id:
            raise BlindReviewAssemblyError(f"duplicate template key {blind_id}")
        key_by_blind_id[blind_id] = raw_key
    templates: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for raw_packet in raw_packets:
        if not isinstance(raw_packet, dict):
            raise BlindReviewAssemblyError("template conversation is invalid")
        blind_id = _string(raw_packet.get("blind_id"), label="template blind_id")
        raw_key = key_by_blind_id.get(blind_id)
        if raw_key is None:
            raise BlindReviewAssemblyError(f"template packet {blind_id} lacks a key")
        case_id = _string(raw_key.get("case_id"), label="template case_id")
        if case_id not in controls or case_id in templates:
            raise BlindReviewAssemblyError(f"template case {case_id} is invalid or duplicate")
        _validate_template(raw_packet, controls[case_id])
        templates[case_id] = (raw_packet, raw_key)
    if set(templates) != set(controls):
        raise BlindReviewAssemblyError("template cases do not exactly match control cases")
    return templates


def _validate_template(template: Mapping[str, Any], control: CaseControl) -> None:
    transcript = template.get("transcript")
    if not isinstance(transcript, list):
        raise BlindReviewAssemblyError(f"template {control.case_id} lacks transcript")
    turns = [entry for entry in transcript if isinstance(entry, Mapping) and entry.get("kind") == "turn"]
    if len(turns) != FIXED_TURN_COUNT:
        raise BlindReviewAssemblyError(
            f"template {control.case_id} must contain {FIXED_TURN_COUNT} turn slots"
        )
    expected_labels: dict[str, str] = {}
    for turn_index, (turn, actor_id) in enumerate(
        zip(turns, control.expected_turn_actor_ids, strict=True), start=1
    ):
        if turn.get("turn_index") != turn_index:
            raise BlindReviewAssemblyError(f"template {control.case_id} has invalid turn index")
        label = _string(turn.get("speaker"), label="template speaker")
        existing = expected_labels.get(actor_id)
        if existing is None:
            expected_labels[actor_id] = label
        elif existing != label:
            raise BlindReviewAssemblyError(
                f"template {control.case_id} changes a speaker label mid-conversation"
            )
    if set(expected_labels) != set(control.actor_ids) or len(set(expected_labels.values())) != 2:
        raise BlindReviewAssemblyError(f"template {control.case_id} speaker labels mismatch")


def _find_case_ledger(cell_root: Path, case_id: str) -> Path:
    conversations = cell_root / "conversations"
    if not conversations.is_dir():
        raise BlindReviewAssemblyError(f"cell output {cell_root} lacks conversations")
    candidates = sorted(conversations.glob(f"*-{case_id}/response_ledger.json"))
    if len(candidates) != 1:
        raise BlindReviewAssemblyError(
            f"cell output {cell_root} needs exactly one ledger for {case_id}; found {len(candidates)}"
        )
    return candidates[0]


def _public_response(entry: Mapping[str, Any], *, label: str) -> str:
    payload = entry.get("response")
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        return _string(payload.get("content"), label=f"{label} response content")
    raise BlindReviewAssemblyError(f"{label} response must be public text or content")


def _load_ledger(cell: str, path: Path, control: CaseControl) -> LedgerConversation:
    document = _read_object(path, label="response ledger")
    case_id = _string(document.get("case_id"), label="ledger case_id")
    if case_id != control.case_id:
        raise BlindReviewAssemblyError(f"ledger {path} case_id does not match its directory")
    conversation_id = _string(document.get("conversation_id"), label="ledger conversation_id")
    responses = document.get("responses")
    if not isinstance(responses, list) or len(responses) != FIXED_TURN_COUNT:
        raise BlindReviewAssemblyError(f"ledger {path} must have {FIXED_TURN_COUNT} responses")
    public_texts: list[str] = []
    actor_ids: list[str] = []
    for sequence, entry in enumerate(responses):
        if not isinstance(entry, Mapping) or entry.get("sequence") != sequence:
            raise BlindReviewAssemblyError(f"ledger {path} has invalid response sequence {sequence}")
        request = entry.get("request")
        if not isinstance(request, Mapping):
            raise BlindReviewAssemblyError(f"ledger {path} response {sequence} lacks request")
        actor_id = _string(request.get("actor_id"), label="ledger actor_id")
        if actor_id != control.expected_turn_actor_ids[sequence]:
            raise BlindReviewAssemblyError(f"ledger {path} actor order differs at turn {sequence + 1}")
        text = _public_response(entry, label=f"ledger {path} response {sequence}")
        _assert_safe_public_text(text, label=f"ledger {path} response {sequence}")
        public_texts.append(text)
        actor_ids.append(actor_id)
    return LedgerConversation(cell, case_id, conversation_id, tuple(public_texts), tuple(actor_ids))


def _assert_safe_public_text(text: str, *, label: str) -> None:
    if _REFLECTION_TAG_RE.search(text) or _REFLECTION_NONCE_RE.search(text):
        raise BlindReviewAssemblyError(f"{label} contains private reflection material")
    if any(field in text for field in REFLECTION_FIELD_NAMES):
        raise BlindReviewAssemblyError(f"{label} contains private reflection field names")


def _identity_replacements(
    control: CaseControl, template_key: Mapping[str, Any], labels: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    replacements: dict[str, str] = {
        control.actor_ids[0]: labels[control.actor_ids[0]],
        control.actor_ids[1]: labels[control.actor_ids[1]],
        control.actor_names[0]: labels[control.actor_ids[0]],
        control.actor_names[1]: labels[control.actor_ids[1]],
    }
    source_speakers = template_key.get("speaker_mapping")
    if isinstance(source_speakers, Mapping):
        for label, name in source_speakers.items():
            if isinstance(label, str) and isinstance(name, str) and label in labels.values():
                replacements[name] = label
    absent_person = template_key.get("absent_person")
    if isinstance(absent_person, str) and absent_person:
        replacements[absent_person] = "the absent person"
    return tuple(sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True))


def _anonymize(text: str, replacements: Sequence[tuple[str, str]]) -> str:
    for original, replacement in replacements:
        text = re.sub(rf"(?<![\w]){re.escape(original)}(?![\w])", replacement, text)
    return text


def _template_labels(template: Mapping[str, Any], control: CaseControl) -> dict[str, str]:
    turns = [entry for entry in template["transcript"] if entry.get("kind") == "turn"]
    return {
        actor_id: _string(turn.get("speaker"), label="template speaker")
        for turn, actor_id in zip(turns, control.expected_turn_actor_ids, strict=True)
    }


def _build_conversation(
    ledger: LedgerConversation,
    control: CaseControl,
    template: Mapping[str, Any],
    template_key: Mapping[str, Any],
    blind_id: str,
) -> dict[str, Any]:
    result = copy.deepcopy(template)
    result["blind_id"] = blind_id
    labels = _template_labels(result, control)
    replacements = (
        (ledger.conversation_id, "this conversation"),
        (ledger.case_id, "this conversation"),
        (ledger.cell, "this conversation"),
        *_identity_replacements(control, template_key, labels),
    )
    response_iter = iter(ledger.public_texts)
    for entry in result["transcript"]:
        if entry.get("kind") == "turn":
            entry["text"] = _anonymize(next(response_iter), replacements)
        elif isinstance(entry.get("text"), str):
            entry["text"] = _anonymize(entry["text"], replacements)
    _assert_blind_packet_safe(result)
    return result


def _assert_blind_packet_safe(packet: Mapping[str, Any]) -> None:
    forbidden_keys = {"cell", "case_id", "conversation_id", "source_protocol", "actor_names"}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            keys = set(value)
            if keys & forbidden_keys:
                raise BlindReviewAssemblyError("blind packet contains unblinded metadata")
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            _assert_safe_public_text(value, label="blind packet")

    visit(packet)


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
        raise argparse.ArgumentTypeError("cell directory must use CELL=DIR")
    return cell, Path(raw_path)


def assemble_blind_review(
    *,
    cell_directories: Mapping[str, Path],
    template_path: Path,
    template_key_path: Path,
    shuffle_seed: str,
    control_manifest_path: Path = DEFAULT_CONTROL_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not shuffle_seed:
        raise BlindReviewAssemblyError("shuffle seed must be nonempty")
    if not cell_directories or any(not cell for cell in cell_directories):
        raise BlindReviewAssemblyError("at least one named cell output is required")
    controls = _load_controls(control_manifest_path)
    templates = _load_template_index(template_path, template_key_path, controls)
    candidates = [
        _load_ledger(cell, _find_case_ledger(root, case_id), control)
        for cell, root in sorted(cell_directories.items())
        for case_id, control in sorted(controls.items())
    ]
    if len({(item.cell, item.case_id) for item in candidates}) != len(candidates):
        raise BlindReviewAssemblyError("duplicate or missing cell/case conversations")
    random.Random(shuffle_seed).shuffle(candidates)
    conversations: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for index, ledger in enumerate(candidates, start=1):
        blind_id = f"sample-{index:02d}"
        control = controls[ledger.case_id]
        template, template_key = templates[ledger.case_id]
        conversations.append(
            _build_conversation(ledger, control, template, template_key, blind_id)
        )
        answer_key.append(
            {
                "blind_id": blind_id,
                "cell": ledger.cell,
                "case_id": ledger.case_id,
                "conversation_id": ledger.conversation_id,
                "source_protocol": SOURCE_PROTOCOL,
            }
        )
    return (
        {
            "schema_version": "persistent_luna_whole_conversation_blind_v1",
            "model_judge": False,
            "source_hidden": True,
            "review_contract": copy.deepcopy(_read_object(template_path, label="blind template")["review_contract"]),
            "conversations": conversations,
        },
        {
            "schema_version": "persistent_luna_whole_conversation_key_v1",
            "shuffle_seed_sha256": hashlib.sha256(shuffle_seed.encode()).hexdigest(),
            "conversations": answer_key,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", action="append", type=parse_cell_directory, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-key", type=Path, required=True)
    parser.add_argument("--shuffle-seed", required=True)
    parser.add_argument("--packet-output", type=Path, required=True)
    parser.add_argument("--answer-key-output", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, default=DEFAULT_CONTROL_MANIFEST)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cell_directories = dict(args.cell_dir)
    if len(cell_directories) != len(args.cell_dir):
        raise SystemExit("each --cell-dir cell name must be unique")
    try:
        packet, answer_key = assemble_blind_review(
            cell_directories=cell_directories,
            template_path=args.template,
            template_key_path=args.template_key,
            shuffle_seed=args.shuffle_seed,
            control_manifest_path=args.control_manifest,
        )
        _atomic_write_json(args.packet_output, packet)
        _atomic_write_json(args.answer_key_output, answer_key)
    except BlindReviewAssemblyError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
