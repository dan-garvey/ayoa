from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from scripts.run_character_dialogue_benchmark import (
    RELAY_LEDGER_SCHEMA_VERSION,
    RELAY_PENDING_EXIT_CODE,
    RelayLedgerError,
    RelayPendingRequest,
    RelayResponder,
    _benchmark_request_payload,
    _benchmark_request_fingerprint,
    _sha256_json,
    append_relay_response,
    load_benchmark_manifest,
    run_relay_conversation,
    write_benchmark_artifacts,
)


def _case():
    return load_benchmark_manifest()[0]


def _run(case, **kwargs):
    return asyncio.run(run_relay_conversation(case, **kwargs))


def _paths(tmp_path: Path):
    ledger = tmp_path / "responses.json"
    pending = tmp_path / "pending.json"
    return ledger, pending


def _relay_kwargs(ledger: Path, pending: Path):
    return {
        "ledger_path": ledger,
        "pending_path": pending,
        "model": "synthetic-luna",
        "conversation_id": "fixed-conversation",
        "turns_per_scene": 1,
        "manifest_fingerprint": "manifest-for-test",
    }


def test_relay_writes_exact_production_request_before_any_response(tmp_path: Path):
    case = _case()
    ledger, pending = _paths(tmp_path)

    with pytest.raises(RelayPendingRequest) as caught:
        _run(case, **_relay_kwargs(ledger, pending))

    pending_document = json.loads(pending.read_text(encoding="utf-8"))
    request = caught.value.request
    assert caught.value.sequence == 0
    assert pending_document["sequence"] == 0
    assert pending_document["request"] == _benchmark_request_payload(request)
    assert pending_document["request_fingerprint"] == _benchmark_request_fingerprint(
        request
    )
    assert pending_document["request"]["compact"] is False
    assert pending_document["request"]["cache"] is True
    assert pending_document["request"]["actor_id"] == case.scenes[0].turn_order[0]
    assert json.loads(ledger.read_text(encoding="utf-8"))["responses"] == []
    assert (
        json.loads(ledger.read_text(encoding="utf-8"))["schema_version"]
        == RELAY_LEDGER_SCHEMA_VERSION
    )
    assert RELAY_PENDING_EXIT_CODE != 0


def test_appending_response_replays_history_and_advances_to_next_actor(
    tmp_path: Path,
):
    case = _case()
    ledger, pending = _paths(tmp_path)
    kwargs = _relay_kwargs(ledger, pending)

    with pytest.raises(RelayPendingRequest):
        _run(case, **kwargs)
    append_relay_response(ledger, "first actor response", pending_path=pending)

    with pytest.raises(RelayPendingRequest) as caught:
        _run(case, **kwargs)
    next_request = caught.value.request
    assert caught.value.sequence == 1
    assert next_request.actor_id == case.scenes[1].turn_order[0]
    rendered = json.dumps(next_request.messages, ensure_ascii=False)
    assert "first actor response" in rendered
    assert "first actor response" not in json.dumps(
        next_request.messages[0], ensure_ascii=False
    )
    next_pending = json.loads(pending.read_text(encoding="utf-8"))
    assert next_pending["sequence"] == 1
    assert next_pending["request"] == _benchmark_request_payload(next_request)

    append_relay_response(ledger, {"content": "second actor response"}, pending_path=pending)
    result = _run(case, **kwargs)
    assert len(result.turns) == 2
    assert [turn.actor_id for turn in result.turns] == [
        case.scenes[0].turn_order[0],
        case.scenes[1].turn_order[0],
    ]
    assert result.turns[0].public_text == "first actor response"
    assert result.turns[1].public_text == "second actor response"
    own_history = result.checkpoint.character_conversations[
        case.scenes[0].turn_order[0]
    ]
    assert any(
        message.role == "assistant"
        and isinstance(message.content, list)
        and any(
            block.get("type") == "text"
            and block.get("text") == "first actor response"
            for block in message.content
            if isinstance(block, dict)
        )
        for message in own_history
    )

    output = write_benchmark_artifacts([result], tmp_path / "artifacts")
    artifact = json.loads(
        (output / "raw" / f"{case.case_id}.json").read_text(encoding="utf-8")
    )
    assert artifact["schema_version"] == "character_dialogue_benchmark_artifact_v3"
    assert [turn["public_text"] for turn in artifact["turns"]] == [
        "first actor response",
        "second actor response",
    ]


def test_relay_rejects_stale_request_fingerprint_loudly(tmp_path: Path):
    case = _case()
    ledger, pending = _paths(tmp_path)
    kwargs = _relay_kwargs(ledger, pending)
    with pytest.raises(RelayPendingRequest):
        _run(case, **kwargs)
    append_relay_response(ledger, "response", pending_path=pending)

    document = json.loads(ledger.read_text(encoding="utf-8"))
    document["responses"][0]["request"]["messages"][-1]["content"] += " changed"
    ledger.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RelayLedgerError, match="stale request fingerprint"):
        _run(case, **kwargs)


def test_relay_can_represent_an_additional_same_turn_call(tmp_path: Path):
    """A parser/format repair call is another ordered request, never a bypass."""

    case = _case()
    ledger_path, _pending = _paths(tmp_path)
    request = None

    # Capture a real production request without a provider by using the normal
    # relay's first pending boundary, then create a second request for the
    # same actor/turn with a changed user tail, as a repair would do.
    with pytest.raises(RelayPendingRequest) as caught:
        _run(
            case,
            ledger_path=ledger_path,
            model="synthetic-luna",
            conversation_id="repair-conversation",
            turns_per_scene=1,
            manifest_fingerprint="repair-manifest",
        )
    request = caught.value.request
    second_payload = _benchmark_request_payload(request)
    second_payload["messages"] = copy.deepcopy(second_payload["messages"])
    second_payload["messages"][-1]["content"] += "\nPlease retry the format."

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    first_payload = _benchmark_request_payload(request)
    ledger["responses"] = [
        {
            "sequence": 0,
            "request_fingerprint": _sha256_json(first_payload),
            "request_sha256": _sha256_json(first_payload),
            "request": first_payload,
            "response": {"content": "not yet"},
        },
        {
            "sequence": 1,
            "request_fingerprint": _sha256_json(second_payload),
            "request_sha256": _sha256_json(second_payload),
            "request": second_payload,
            "response": {"content": "repaired"},
        },
    ]
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    responder = RelayResponder(ledger_path, ledger)
    first = asyncio.run(responder(request))
    assert first.content == "not yet"
    # BenchmarkRequest remains immutable; construct a second request through
    # its dataclass constructor so the changed prompt is a real request.
    from dataclasses import replace

    repaired_request = replace(
        request,
        messages=tuple(second_payload["messages"]),
    )
    repaired = asyncio.run(responder(repaired_request))
    assert repaired.content == "repaired"
    assert responder.next_sequence == 2
    responder.assert_exhausted()
