from __future__ import annotations

import json

from app.engine.content_fronts import (
    FRONT_RUNTIME_METADATA_KEY,
    queue_front_signal_from_consequence,
)
from app.engine.content_resolver import append_pending_router_content_records
from app.engine.orchestrator import (
    _automated_turn_snapshot,
    _rollback_automated_turn_snapshot,
)
from app.schemas.checkpoint import CheckpointFile
from tests.support.factories import checkpoint


def test_public_consequence_updates_front_state_and_queues_router_signal():
    ckpt = checkpoint()
    hidden_plan = "HIDDEN: abduct the burgomaster before dawn"

    update = queue_front_signal_from_consequence(
        ckpt.session.content_state,
        pack_id="curse",
        front_id="strahd",
        villain_id="strahd",
        actor_id="strahd",
        source_event_id="evt_public_1",
        known=["Ireena was named in the tavern", "the party angered the guard"],
        pressure="send spies to the tavern",
        summary="Public tavern trouble may reach Strahd.",
        consequence_visibility="semi-public",
        now_s=40,
        cooldown_until_s=100,
        restraint="wait for a deniable opening",
        active_plan="watch the tavern road",
        hidden_plan=hidden_plan,
        priority=3,
    )

    assert update.queued
    pack = ckpt.session.content_state["curse"]
    assert pack.fronts["strahd"].villain_ids == ["strahd"]
    assert pack.villains["strahd"].front_ids == ["strahd"]

    front_runtime = pack.metadata[FRONT_RUNTIME_METADATA_KEY]["fronts"]["strahd"]
    assert front_runtime["known_facts"] == [
        "Ireena was named in the tavern",
        "the party angered the guard",
    ]
    assert front_runtime["source_event_ids"] == ["evt_public_1"]
    assert front_runtime["cooldown_until_s"] == 100
    assert front_runtime["restraint"] == {
        "reason": "wait for a deniable opening",
        "source_event_id": "evt_public_1",
    }
    assert front_runtime["active_plans"] == [
        {
            "actor": "strahd",
            "plan": "watch the tavern road",
            "source_event_id": "evt_public_1",
            "status": "active",
        }
    ]

    durable_dump = json.dumps(pack.model_dump(mode="json"), sort_keys=True)
    assert hidden_plan not in durable_dump

    records = append_pending_router_content_records(ckpt)

    assert len(records) == 1
    record = records[0]
    assert record.startswith("front_signal ref=front/strahd actor=strahd")
    assert (
        'knows=["Ireena was named in the tavern","the party angered the guard"]'
        in record
    )
    assert 'pressure="send spies to the tavern"' in record
    assert 'summary="Public tavern trouble may reach Strahd."' in record
    assert "visibility=hidden" in record
    assert hidden_plan not in record
    assert "watch the tavern road" not in record
    assert "evt_public_1" not in record
    assert ckpt.session.content_state["curse"].pending_signals == {}
    assert ckpt.session_conversation[-1].role == "assistant"
    assert ckpt.session_conversation[-1].content == record


def test_front_cooldown_suppresses_extra_signal_but_records_known_event():
    ckpt = checkpoint()
    first = queue_front_signal_from_consequence(
        ckpt.session.content_state,
        pack_id="curse",
        front_id="wolves",
        villain_id="dire_wolf",
        source_event_id="evt_alarm_1",
        known="the west gate alarm rang",
        pressure="send wolves to test the gate",
        summary="The alarm is public enough to reach the wolf front.",
        now_s=10,
        cooldown_until_s=60,
    )
    assert first.queued
    assert append_pending_router_content_records(ckpt)

    second = queue_front_signal_from_consequence(
        ckpt.session.content_state,
        pack_id="curse",
        front_id="wolves",
        villain_id="dire_wolf",
        source_event_id="evt_alarm_2",
        known="smoke rose over the same gate",
        pressure="send a second probe immediately",
        summary="A second public cue lands during cooldown.",
        now_s=20,
    )

    assert not second.queued
    assert second.suppressed_reason == "cooldown"
    assert append_pending_router_content_records(ckpt) == []

    front_runtime = (
        ckpt.session.content_state["curse"]
        .metadata[FRONT_RUNTIME_METADATA_KEY]["fronts"]["wolves"]
    )
    assert front_runtime["known_facts"] == [
        "the west gate alarm rang",
        "smoke rose over the same gate",
    ]
    assert front_runtime["source_event_ids"] == ["evt_alarm_1", "evt_alarm_2"]
    assert front_runtime["suppressed_source_event_ids"] == ["evt_alarm_2"]


def test_front_runtime_state_round_trips_and_rolls_back_with_content_state():
    ckpt = checkpoint()
    update = queue_front_signal_from_consequence(
        ckpt.session.content_state,
        pack_id="curse",
        front_id="abbot",
        villain_id="abbot",
        source_event_id="evt_chapel_1",
        known="the chapel bell rang three times",
        pressure="send a servant to listen at the chapel",
        summary="The abbot front can react to the public bell.",
        now_s=5,
        active_plan="dispatch a servant observer",
    )
    assert update.queued

    rebuilt = CheckpointFile.model_validate(ckpt.model_dump(mode="json"))
    rebuilt_runtime = (
        rebuilt.session.content_state["curse"]
        .metadata[FRONT_RUNTIME_METADATA_KEY]["fronts"]["abbot"]
    )
    assert rebuilt_runtime["known_facts"] == ["the chapel bell rang three times"]
    assert rebuilt_runtime["active_plans"][0]["plan"] == (
        "dispatch a servant observer"
    )

    snapshot = _automated_turn_snapshot(ckpt)
    runtime = (
        ckpt.session.content_state["curse"]
        .metadata[FRONT_RUNTIME_METADATA_KEY]["fronts"]["abbot"]
    )
    runtime["known_facts"].append("corrupted future fact")
    runtime["active_plans"][0]["plan"] = "corrupted plan"
    ckpt.session.content_state["curse"].pending_signals.clear()

    _rollback_automated_turn_snapshot(ckpt, snapshot)

    restored_runtime = (
        ckpt.session.content_state["curse"]
        .metadata[FRONT_RUNTIME_METADATA_KEY]["fronts"]["abbot"]
    )
    assert restored_runtime["known_facts"] == ["the chapel bell rang three times"]
    assert restored_runtime["active_plans"][0]["plan"] == (
        "dispatch a servant observer"
    )
    assert sorted(ckpt.session.content_state["curse"].pending_signals) == [
        update.queued_signal.signal_id
    ]
