from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.assemble_private_reflection_blind_review import (
    PrivateReflectionAssemblyError,
    REFLECTION_FIELDS,
    assemble_private_reflection_blind_review,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _controls() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for case_index in range(8):
        case_id = f"case_{case_index}"
        actor_ids = (f"left_{case_index}", f"right_{case_index}")
        cases.append(
            {
                "case_id": case_id,
                "actors": [
                    {
                        "character_id": actor_id,
                        "name": f"Character {actor_index}-{case_index}",
                        "public_sheet": {"role": f"role {actor_index}"},
                        "actor": {
                            "facts": [
                                {"origin": "lived", "text": f"fact {actor_index}"}
                            ]
                        },
                    }
                    for actor_index, actor_id in enumerate(actor_ids, start=1)
                ],
                "scenes": [
                    {
                        "title": "first scene",
                        "frame": "authorized setup one",
                        "actor_observations": {
                            actor_ids[0]: ["left observation"],
                            actor_ids[1]: ["right observation"],
                        },
                        "turn_order": [actor_ids[turn % 2] for turn in range(8)],
                    },
                    {
                        "title": "second scene",
                        "frame": "authorized setup two",
                        "between_scene_public_history": "authorized history",
                        "turn_order": [actor_ids[turn % 2] for turn in range(8)],
                    },
                ],
            }
        )
    return {"cases": cases}


def _write_cell(root: Path, label: str, controls: dict[str, object]) -> Path:
    cell = root / label
    for case in controls["cases"]:  # type: ignore[index]
        case_id = case["case_id"]  # type: ignore[index]
        actor_ids = [actor["character_id"] for actor in case["actors"]]  # type: ignore[index]
        conversation = f"conversation-{label}-{case_id}"
        responses = []
        for sequence in range(16):
            actor_id = actor_ids[sequence % 2]
            responses.append(
                {
                    "sequence": sequence,
                    "request": {"actor_id": actor_id},
                    "response": {"content": f"public turn {sequence}"},
                }
            )
            _write_json(
                cell
                / "private_reflection_qa"
                / f"actor-{actor_id}"
                / f"turn-{sequence:02d}-attempt-1.json",
                {
                    "conversation_id": conversation,
                    "case_id": case_id,
                    "actor_id": actor_id,
                    "ledger_sequence": sequence,
                    "attempt": 1,
                    "status": "accepted",
                    "raw_response": "sealed source only",
                    "reflection": {
                        "nonce": "R-0123456789abcdef0123456789abcdef",
                        "suffix": "sealed source only",
                        "fields": {
                            field: f"reflection {sequence} {field}"
                            for field in REFLECTION_FIELDS
                        },
                    },
                },
            )
        _write_json(
            cell / "conversations" / f"01-{case_id}" / "response_ledger.json",
            {"case_id": case_id, "conversation_id": conversation, "responses": responses},
        )
    return cell


def _prepare(tmp_path: Path) -> tuple[Path, dict[str, object], Path, Path]:
    controls = _controls()
    control_path = tmp_path / "control.json"
    _write_json(control_path, controls)
    return (
        control_path,
        controls,
        _write_cell(tmp_path, "CELL_CURRENT", controls),
        _write_cell(tmp_path, "CELL_MINIMAL", controls),
    )


def test_assembler_projects_only_reviewable_reflection_fields_and_hides_sources(
    tmp_path: Path,
) -> None:
    control_path, _, current, minimal = _prepare(tmp_path)

    packet, answer_key = assemble_private_reflection_blind_review(
        cell_directories={"CELL_CURRENT": current, "CELL_MINIMAL": minimal},
        shuffle_seed="stable-seed",
        control_manifest_path=control_path,
    )
    repeat_packet, repeat_key = assemble_private_reflection_blind_review(
        cell_directories={"CELL_CURRENT": current, "CELL_MINIMAL": minimal},
        shuffle_seed="stable-seed",
        control_manifest_path=control_path,
    )

    assert packet == repeat_packet and answer_key == repeat_key
    assert len(packet["conversations"]) == 16
    assert len(answer_key["conversations"]) == 16
    assert {entry["blind_id"] for entry in answer_key["conversations"]} == {
        f"reflection-{index:02d}" for index in range(1, 17)
    }
    assert set(answer_key["conversations"][0]) == {"blind_id", "cell", "case", "conversation"}
    encoded_packet = json.dumps(packet)
    for forbidden in (
        "CELL_CURRENT",
        "CELL_MINIMAL",
        "case_0",
        "conversation-",
        "actor_private_reflection",
        "R-0123456789abcdef0123456789abcdef",
        "raw_response",
        "nonce",
        "suffix",
    ):
        assert forbidden not in encoded_packet
    for conversation in packet["conversations"]:
        assert len(conversation["characters"]) == 2
        assert len(conversation["scenes"]) == 2
        assert "actor_observations" in conversation["scenes"][0]
        assert "actor_observations" not in conversation["scenes"][1]
        assert len(conversation["turns"]) == 16
        for turn_index, turn in enumerate(conversation["turns"], start=1):
            assert turn["turn_index"] == turn_index
            assert turn["speaker"].startswith("Character ")
            assert set(turn["private_reflection"]) == set(REFLECTION_FIELDS)


def test_assembler_rejects_duplicate_accepted_reflection(tmp_path: Path) -> None:
    control_path, _, current, _ = _prepare(tmp_path)
    source = next((current / "private_reflection_qa").glob("**/turn-00-attempt-1.json"))
    duplicate = source.with_name("turn-00-attempt-2.json")
    duplicate.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(PrivateReflectionAssemblyError, match="duplicate accepted reflection"):
        assemble_private_reflection_blind_review(
            cell_directories={"CELL_CURRENT": current},
            shuffle_seed="seed",
            control_manifest_path=control_path,
        )


def test_assembler_rejects_sealed_public_text_and_missing_reflection(tmp_path: Path) -> None:
    control_path, _, current, _ = _prepare(tmp_path)
    ledger_path = current / "conversations" / "01-case_0" / "response_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["responses"][0]["response"]["content"] = "<actor_private_reflection>"
    _write_json(ledger_path, ledger)

    with pytest.raises(PrivateReflectionAssemblyError, match="sealed reflection"):
        assemble_private_reflection_blind_review(
            cell_directories={"CELL_CURRENT": current},
            shuffle_seed="seed",
            control_manifest_path=control_path,
        )

    ledger["responses"][0]["response"]["content"] = "public text"
    _write_json(ledger_path, ledger)
    next((current / "private_reflection_qa").glob("**/turn-15-attempt-1.json")).unlink()
    with pytest.raises(PrivateReflectionAssemblyError, match="exactly 16 accepted reflections"):
        assemble_private_reflection_blind_review(
            cell_directories={"CELL_CURRENT": current},
            shuffle_seed="seed",
            control_manifest_path=control_path,
        )
