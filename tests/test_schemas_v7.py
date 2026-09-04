"""Current schema-boundary contracts for checkpoint format 7."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.checkpoint import CheckpointFile, CURRENT_SCHEMA_VERSION
from app.schemas.delivery import NarratorEventRef, NarratorRenderJob
from app.schemas.event_router import (
    ObserverGroups,
    RouterBatchOutput,
    RouterEventDraft,
    RouterInputEnvelope,
)
from app.schemas.events import ObservableFact
from app.schemas.state import SessionSettings
from tests.support.factories import canonical_event, checkpoint


def test_checkpoint_v7_round_trip_preserves_unified_runtime_state() -> None:
    original = checkpoint(session_id="schema-v7")
    original.canonical_events = [canonical_event(observer_ids=[])]
    original.session.narrator_render_jobs = [NarratorRenderJob(
        job_id="job_1",
        lane_id="lane_test",
        pov_character_id="alice",
        source_event_ids=["evt_test"],
        event_refs=[NarratorEventRef(
            event_id="evt_test",
            observation_level="direct",
            visible_at_s=0,
            event_sequence=0,
            sprite_variant_keys_by_character_id={},
        )],
        highest_event_sequence=0,
        created_revision=1,
        user_input="",
        partial_mode=False,
        narration_mode="event_aligned",
        status="pending",
        attempts=0,
        last_error="",
    )]

    rebuilt = CheckpointFile.model_validate_json(original.model_dump_json())

    assert rebuilt.schema_version == CURRENT_SCHEMA_VERSION == "7.0"
    assert rebuilt.canonical_events[0].causal_lane_id == "lane_test"
    assert rebuilt.session.narrator_render_jobs[0].job_id == "job_1"


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (SessionSettings, {"retired_beat_cap": 4}),
        (ObserverGroups, {"direct": [], "indirect": [], "inferred": [], "x": 1}),
    ],
)
def test_runtime_schemas_reject_retired_or_unknown_fields(model, payload) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate(payload)


def test_router_batch_schema_has_only_events_and_next_turns() -> None:
    assert set(RouterBatchOutput.model_fields) == {"events", "next_turns"}
    assert "background_threads" not in RouterBatchOutput.model_json_schema()[
        "properties"
    ]


def test_router_input_is_one_compact_lane_envelope() -> None:
    item = RouterInputEnvelope(
        submission_id="sub_1",
        input_index=0,
        lane_id="lane_1",
        kind="character",
        actor_ids=["alice"],
        participant_ids=["alice"],
        source_event_ids=[],
        chosen_at_s=3,
        observed_through_event_sequence=-1,
        observed_through_s=0,
        payload="Alice keeps watch.",
    )
    assert item.lane_id == "lane_1"
    assert "background" not in item.model_dump_json()


def test_observer_groups_require_each_observer_to_receive_a_fact() -> None:
    with pytest.raises(ValidationError, match="every observer"):
        RouterEventDraft(
            feasible_input_indexes=[0],
            infeasible_input_indexes=[],
            duration_s=0,
            observable_facts=[ObservableFact.only("Alice sees it.", ["alice"])],
            observers=ObserverGroups(
                direct=["alice", "bob"],
                indirect=[],
                inferred=[],
            ),
            required_responders=[],
            appearance_target_ids=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_opens=[],
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
            activate=[],
        )
