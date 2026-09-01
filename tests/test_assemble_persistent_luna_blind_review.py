from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.assemble_persistent_luna_blind_review import (
    BlindReviewAssemblyError,
    assemble_blind_review,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _control_manifest() -> dict[str, object]:
    cases = []
    for case_id, names in (
        ("case_a", ("Ada North", "Bea South")),
        ("case_b", ("Cal East", "Dee West")),
    ):
        actor_ids = tuple(name.lower().replace(" ", "_") for name in names)
        order = [actor_ids[index % 2] for index in range(16)]
        cases.append(
            {
                "case_id": case_id,
                "actors": [
                    {"character_id": actor_id, "name": name}
                    for actor_id, name in zip(actor_ids, names, strict=True)
                ],
                "scenes": [
                    {"turn_order": order[:8]},
                    {"turn_order": order[8:]},
                ],
            }
        )
    return {"cases": cases}


def _template(case_ids: list[str]) -> tuple[dict[str, object], dict[str, object]]:
    conversations = []
    key_conversations = []
    for sample, case_id in enumerate(case_ids, start=1):
        transcript: list[dict[str, object]] = [
            {"kind": "scene_start", "scene_index": 0, "label": "Scene 1", "text": "Ada North waits."}
        ]
        for turn_index in range(1, 17):
            transcript.append(
                {
                    "kind": "turn",
                    "scene_index": 0 if turn_index <= 8 else 1,
                    "scene_turn_index": (turn_index - 1) % 8 + 1,
                    "turn_index": turn_index,
                    "speaker": "Speaker A" if turn_index % 2 else "Speaker B",
                    "text": "TEMPLATE TURN",
                }
            )
        conversations.append(
            {
                "blind_id": f"template-{sample}",
                "transcript": transcript,
                "review": {"dimensions": {"whole": "keep rubric"}, "notes": "", "overall_status": ""},
            }
        )
        key_conversations.append(
            {
                "blind_id": f"template-{sample}",
                "case_id": case_id,
                "speaker_mapping": {"Speaker A": "Ada North", "Speaker B": "Bea South"},
                "absent_person": "Mara Elsewhere",
            }
        )
    return (
        {
            "review_contract": {"unit": "whole conversation", "dimensions": {"whole": "keep rubric"}},
            "conversations": conversations,
        },
        {"conversations": key_conversations},
    )


def _write_cell(root: Path, cell: str, controls: dict[str, object]) -> Path:
    cell_root = root / cell
    for case in controls["cases"]:  # type: ignore[index]
        case_id = case["case_id"]  # type: ignore[index]
        actors = [actor["character_id"] for actor in case["actors"]]  # type: ignore[index]
        responses = []
        for sequence in range(16):
            actor_id = actors[sequence % 2]
            responses.append(
                {
                    "sequence": sequence,
                    "request": {"actor_id": actor_id},
                    "response": {"content": f"{cell}-{case_id}-{sequence} Ada North mentions Mara Elsewhere."},
                }
            )
        _write_json(
            cell_root / "conversations" / f"01-{case_id}" / "response_ledger.json",
            {
                "case_id": case_id,
                "conversation_id": f"{cell}:{case_id}",
                "responses": responses,
            },
        )
    return cell_root


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    controls = _control_manifest()
    control_path = tmp_path / "control.json"
    template_path = tmp_path / "template.json"
    key_path = tmp_path / "key.json"
    template, key = _template(["case_a", "case_b"])
    _write_json(control_path, controls)
    _write_json(template_path, template)
    _write_json(key_path, key)
    return control_path, template_path, key_path, controls


def test_assemble_blind_review_replaces_only_turn_text_and_keeps_key_private(
    tmp_path: Path,
) -> None:
    control_path, template_path, key_path, controls = _artifacts(tmp_path)
    packet, answer_key = assemble_blind_review(
        cell_directories={
            "CELL_MN": _write_cell(tmp_path, "CELL_MN", controls),
            "CELL_PR": _write_cell(tmp_path, "CELL_PR", controls),
        },
        template_path=template_path,
        template_key_path=key_path,
        shuffle_seed="opaque-test-seed",
        control_manifest_path=control_path,
    )

    assert len(packet["conversations"]) == 4
    assert packet["review_contract"] == {"unit": "whole conversation", "dimensions": {"whole": "keep rubric"}}
    assert {item["blind_id"] for item in answer_key["conversations"]} == {
        "sample-01", "sample-02", "sample-03", "sample-04"
    }
    assert {(item["cell"], item["case_id"]) for item in answer_key["conversations"]} == {
        ("CELL_MN", "case_a"),
        ("CELL_MN", "case_b"),
        ("CELL_PR", "case_a"),
        ("CELL_PR", "case_b"),
    }
    assert all(item["source_protocol"] == "persistent_luna_dialogue_silos_v2" for item in answer_key["conversations"])

    encoded = json.dumps(packet)
    assert "CELL_MN" not in encoded and "CELL_PR" not in encoded
    assert "case_a" not in encoded and "case_b" not in encoded
    assert "Ada North" not in encoded and "Mara Elsewhere" not in encoded
    for conversation in packet["conversations"]:
        turns = [item for item in conversation["transcript"] if item["kind"] == "turn"]
        assert len(turns) == 16
        assert {item["speaker"] for item in turns} == {"Speaker A", "Speaker B"}
        assert all("TEMPLATE TURN" not in item["text"] for item in turns)


def test_assemble_blind_review_rejects_private_reflection_leak(tmp_path: Path) -> None:
    control_path, template_path, key_path, controls = _artifacts(tmp_path)
    cell = _write_cell(tmp_path, "MN", controls)
    ledger_path = cell / "conversations" / "01-case_a" / "response_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["responses"][0]["response"]["content"] = "<actor_private_reflection id=\"R-0123456789abcdef0123456789abcdef\">"
    _write_json(ledger_path, ledger)

    with pytest.raises(BlindReviewAssemblyError, match="private reflection"):
        assemble_blind_review(
            cell_directories={"MN": cell},
            template_path=template_path,
            template_key_path=key_path,
            shuffle_seed="seed",
            control_manifest_path=control_path,
        )


def test_assemble_blind_review_requires_one_complete_ledger_per_case(tmp_path: Path) -> None:
    control_path, template_path, key_path, controls = _artifacts(tmp_path)
    cell = _write_cell(tmp_path, "MN", controls)
    missing = cell / "conversations" / "01-case_b" / "response_ledger.json"
    missing.unlink()

    with pytest.raises(BlindReviewAssemblyError, match="exactly one ledger"):
        assemble_blind_review(
            cell_directories={"MN": cell},
            template_path=template_path,
            template_key_path=key_path,
            shuffle_seed="seed",
            control_manifest_path=control_path,
        )
