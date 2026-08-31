"""Offline contract tests for the Mirelle depth-by-instruction runner."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.run_mirelle_dialogue_factorial import (
    FactorialManifestError,
    FactorialTechnicalInvalidity,
    ModelCall,
    POST_UNBLIND_AUDIT_FIELDS,
    PendingRequest,
    ProxyResponder,
    _assert_prompt_contract,
    import_response,
    load_factorial_manifest,
    run_conversation,
    run_factorial,
    write_artifacts,
)


MANIFEST = Path("scripts/mirelle_dialogue_factorial_manifest.json")


def _run(cell, scenario, *, content_factory=None, conversation_id="test-run"):
    calls = []

    def responder(request):
        calls.append(request)
        content = (
            content_factory(request)
            if content_factory is not None
            else f"action-{request.turn_index}"
        )
        return ModelCall(
            content=content,
            model=request.model,
            provider="coding_agent_proxy",
        )

    result = asyncio.run(
        run_conversation(
            cell,
            scenario,
            responder=responder,
            conversation_id=conversation_id,
            conversation_token=f"opaque-{conversation_id}",
        )
    )
    return result, calls


def test_manifest_has_the_frozen_factorial_and_phase_topology():
    manifest = load_factorial_manifest(MANIFEST)

    assert {(cell.depth, cell.instruction) for cell in manifest.cells} == {
        ("sparse", "lean"),
        ("rich", "lean"),
        ("sparse", "normalized_legacy"),
        ("rich", "normalized_legacy"),
    }
    assert [scenario.scenario_id for scenario in manifest.scenarios] == [
        "seventh_stone_open_gap",
        "reliance_without_crisis",
        "lost_authority",
    ]
    assert [scenario.scene_2_elapsed_s for scenario in manifest.scenarios] == [
        86400,
        86400,
        7200,
    ]
    assert [scenario.depth_relevance for scenario in manifest.scenarios] == [
        "low",
        "medium",
        "high",
    ]
    for scenario in manifest.scenarios:
        assert [len(scene.turn_order) for scene in scenario.scenes] == [6, 6]
        assert all(
            left != right
            for left, right in zip(scenario.turn_order, scenario.turn_order[1:])
        )
        assert scenario.scenes[0].frame
        assert scenario.scenes[1].between_scene_public_history

    assert len(manifest.cell("sparse", "lean").actor("mirelle_voss").actor.facts) == 4
    assert len(manifest.cell("rich", "lean").actor("mirelle_voss").actor.facts) == 28


def test_run_uses_checkpoint_roster_and_exact_packet_without_repeating_scene_setup():
    manifest = load_factorial_manifest(MANIFEST)
    cell = manifest.cell("rich", "lean")
    scenario = manifest.scenarios[0]
    result, calls = _run(cell, scenario, conversation_id="packet-run")

    assert result.status == "valid"
    assert len(calls) == len(result.turns) == 12
    checkpoint_mirelle = next(
        actor for actor in result.checkpoint.characters
        if actor.character_id == "mirelle_voss"
    )
    assert len(checkpoint_mirelle.actor.facts) == 28
    assert result.checkpoint.session.leading_at_s >= scenario.scene_2_elapsed_s + 12

    setup = scenario.scenes[0].frame
    bridge = scenario.scenes[1].between_scene_public_history
    for request in calls:
        system = str(request.messages[0]["content"])
        user = str(request.messages[-1]["content"])
        actor = next(
            candidate for candidate in result.checkpoint.characters
            if candidate.character_id == request.actor_id
        )
        assert request.messages[-1]["role"] == "user"
        assert request.compact is False
        assert request.cache is True
        assert "<you>" in user and "</you>" in user
        assert "<now>" in user and "</now>" in user
        assert f"You are {actor.name}." in user
        assert actor.name not in system
        assert actor.character_id not in system
        for fact in actor.actor.facts:
            assert fact.text in user
            assert fact.text not in system
        assert setup not in system
        assert bridge not in system

    # Setup and bridge arrive through the public observation path.  They are
    # present in each actor's first packet for that scene, then only the newly
    # accepted public beat is added to later current-user tails.
    assert setup in calls[0].messages[-1]["content"]
    assert setup in calls[1].messages[-1]["content"]
    assert setup not in calls[2].messages[-1]["content"]
    assert bridge in calls[6].messages[-1]["content"]
    assert bridge in calls[7].messages[-1]["content"]
    assert bridge not in calls[8].messages[-1]["content"]

    first_actor = manifest.cell("rich", "lean").actor(calls[0].actor_id)
    second_turn_user = str(calls[1].messages[-1]["content"])
    assert f"{first_actor.name}: action-1" in second_turn_user
    assert f"{first_actor.name} says:" not in second_turn_user
    assert "has passed since you last had a chance" in calls[6].messages[-1]["content"]
    assert sum(
        update["kind"] == "scene_break" for update in result.public_transcript
    ) == 1

    # Every accepted turn contributes exactly one user/assistant pair, and the
    # request history grows only for that actor.
    for turn in result.turns:
        assert turn["history"]["message_count_after"] == (
            turn["history"]["message_count_before"] + 2
        )
    assert all(
        len(messages) == 12
        for messages in result.checkpoint.character_conversations.values()
        if messages
    )


def test_profile_and_instruction_variants_are_isolated():
    manifest = load_factorial_manifest(MANIFEST)
    first_requests = {}
    for cell in manifest.cells:
        result, calls = _run(
            cell,
            manifest.scenarios[1],
            conversation_id=f"variant-{cell.cell_id}",
        )
        assert result.status == "valid"
        first_requests[cell.cell_id] = next(
            request for request in calls if request.actor_id == "mirelle_voss"
        )

    assert first_requests["SL"].messages[0] == first_requests["RL"].messages[0]
    assert first_requests["SD"].messages[0] == first_requests["RD"].messages[0]
    assert first_requests["SL"].messages[0] != first_requests["SD"].messages[0]
    assert first_requests["SL"].messages[-1] != first_requests["RL"].messages[-1]

    sparse_fact = manifest.cell("sparse", "lean").actor("mirelle_voss").actor.facts[3].text
    rich_fact = manifest.cell("rich", "lean").actor("mirelle_voss").actor.facts[4].text
    assert sparse_fact in first_requests["SL"].messages[-1]["content"]
    assert rich_fact not in first_requests["SL"].messages[-1]["content"]
    assert rich_fact in first_requests["RL"].messages[-1]["content"]


def test_phase_jobs_are_parallelizable_and_disjoint():
    manifest = load_factorial_manifest(MANIFEST)

    def response(request):
        return ModelCall(
            content=f"action-{request.turn_index}",
            model=request.model,
            provider="coding_agent_proxy",
        )

    async def run():
        pilot, exploratory = await asyncio.gather(
            run_factorial(
                manifest,
                phase="pilot",
                replicates=1,
                parallelism=4,
                responder=response,
            ),
            run_factorial(
                manifest,
                phase="all",
                replicates=1,
                parallelism=4,
                responder=response,
            ),
        )
        return pilot, exploratory

    pilot, exploratory = asyncio.run(run())
    assert len(pilot) == 8
    assert len(exploratory) == 12
    assert all(len(item.turns) == 12 for item in pilot + exploratory)
    assert {item.conversation_token for item in pilot}.isdisjoint(
        item.conversation_token for item in exploratory
    )
    assert {item.conversation_id for item in pilot}.isdisjoint(
        item.conversation_id for item in exploratory
    )
    with pytest.raises(FactorialManifestError):
        asyncio.run(
            run_factorial(
                manifest,
                phase="confirmatory",
                replicates=1,
                responder=response,
            )
        )


def test_pending_request_is_opaque_exact_and_import_resumes(tmp_path):
    manifest = load_factorial_manifest(MANIFEST)
    cell = manifest.cell("sparse", "lean")
    scenario = manifest.scenarios[0]
    ledger = tmp_path / "opaque-run.json"
    pending = tmp_path / "opaque-run.pending.json"
    token = "opaque-fixed-token"

    with pytest.raises(PendingRequest) as pending_error:
        asyncio.run(
            run_conversation(
                cell,
                scenario,
                ledger_responder=ProxyResponder(ledger, pending, token),
                conversation_id="conversation-fixed",
                conversation_token=token,
            )
        )
    assert pending_error.value.sequence == 0
    document = json.loads(pending.read_text(encoding="utf-8"))
    assert set(document) == {
        "schema_version",
        "conversation",
        "proxy_session_id",
        "sequence",
        "request",
        "request_sha256",
    }
    wrapper = {key: value for key, value in document.items() if key != "request"}
    serialized = json.dumps(wrapper, ensure_ascii=False).casefold()
    for forbidden in (
        "mirelle",
        "factorial",
        "experiment",
        "sparse",
        "rich",
        "lean",
        "legacy",
        "pilot",
        "confirmatory",
        "cell_id",
        "scenario_id",
    ):
        assert forbidden not in serialized
    assert document["request"]["messages"][-1]["role"] == "user"
    assert document["request"]["compact"] is False

    import_response(
        ledger,
        pending,
        {"content": "action-1", "proxy_agent_id": "luna-session-a"},
    )
    with pytest.raises(PendingRequest) as next_pending:
        asyncio.run(
            run_conversation(
                cell,
                scenario,
                ledger_responder=ProxyResponder(ledger, pending, token),
                conversation_id="conversation-fixed",
                conversation_token=token,
            )
    )
    assert next_pending.value.sequence == 1
    ledger_document = json.loads(ledger.read_text(encoding="utf-8"))
    assert ledger_document["proxy_agent_id"] == "luna-session-a"
    with pytest.raises(FactorialTechnicalInvalidity):
        import_response(
            ledger,
            next_pending.value.path,
            {"content": "action-2", "proxy_agent_id": "luna-session-b"},
        )

    stale = copy.deepcopy(document)
    stale["request"]["compact"] = True
    pending.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(FactorialTechnicalInvalidity):
        import_response(
            ledger,
            pending,
            {"content": "action-2", "proxy_agent_id": "luna-session-a"},
        )


def test_artifacts_blind_the_whole_transcript_and_keep_hashes(tmp_path):
    manifest = load_factorial_manifest(MANIFEST)
    result, _calls = _run(
        manifest.cell("rich", "normalized_legacy"),
        manifest.scenarios[2],
        conversation_id="review-run",
    )
    root = write_artifacts(manifest, [result], tmp_path)

    raw = json.loads((root / "raw" / "review-run.json").read_text())
    assert raw["execution"]["mode"] == "coding_agent_proxy"
    assert raw["execution"]["model"] == "gpt-5.6-luna"
    assert raw["execution"]["provider_live_calls"] is False
    assert raw["execution"]["proxy_agent_id"]
    assert len(raw["turns"]) == 12
    for turn in raw["turns"]:
        assert turn["request"]["request_sha256"]
        assert turn["response"]["response_sha256"]
        assert turn["history"]["before_sha256"]
        assert turn["history"]["after_sha256"]
    assert raw["final_history_sha256"]

    review = json.loads(
        (root / "review" / "whole_conversation_review.json").read_text()
    )
    sheet = review["sheets"][0]
    blind_text = json.dumps(sheet, ensure_ascii=False)
    assert sheet["unit"] == "whole_conversation"
    assert set(sheet["scores"]) >= {"B", "Q"}
    assert all(
        set(dimension) >= {"question", "score", "evidence"}
        for score in (sheet["scores"]["B"], sheet["scores"]["Q"])
        for dimension in score["dimensions"].values()
    )
    assert "Mirelle Voss" not in blind_text
    assert "Renna Holt" not in blind_text
    assert "Mirelle" not in blind_text
    assert "Renna" not in blind_text
    assert "Edren" not in blind_text
    assert "mirelle_dialogue" not in blind_text
    assert "mirelle_voss" not in blind_text
    assert "renna_holt" not in blind_text

    answer_keys = json.loads(
        (root / "review" / "answer_key.json").read_text()
    )["answer_key"]
    answer_key = next(item for item in answer_keys if item["cell_id"] == "RD")
    assert answer_key["blind_id"] == sheet["blind_id"]
    assert answer_key["cell_id"] == "RD"
    assert answer_key["scenario_id"] == "lost_authority"
    assert set(answer_key["post_unblind_audit"]) == set(POST_UNBLIND_AUDIT_FIELDS)


def test_model_failure_and_technical_invalidity_are_separate():
    manifest = load_factorial_manifest(MANIFEST)
    result, calls = _run(
        manifest.cell("sparse", "lean"),
        manifest.scenarios[0],
        content_factory=lambda _request: "",
        conversation_id="malformed-model",
    )
    assert result.status == "model_failure"
    assert result.model_failure
    assert not result.technical_invalidity

    valid, _ = _run(
        manifest.cell("sparse", "lean"),
        manifest.scenarios[0],
        conversation_id="contract-request",
    )
    request = calls[0]
    actor = next(
        item for item in valid.checkpoint.characters
        if item.character_id == request.actor_id
    )
    with pytest.raises(FactorialTechnicalInvalidity):
        _assert_prompt_contract(
            replace(request, compact=True),
            actor,
            cell=manifest.cell("sparse", "lean"),
            scenario=manifest.scenarios[0],
        )
