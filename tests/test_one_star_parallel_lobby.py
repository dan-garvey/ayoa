"""Coordination contracts for Master turns beside human-led missions."""

from __future__ import annotations

import asyncio

import pytest

from app.engine.one_star_adapter import (
    OneStarTransactionError,
    one_star_active_mission_has_bound_party_member,
    one_star_lobby_liveness_request_after_result,
    one_star_master_has_human_led_mission,
    one_star_master_may_act_while_mission_responder_pinned,
    one_star_should_autonomous_mission_batch_after_result,
    validate_one_star_autonomous_mission_batch_result,
    validate_one_star_human_led_mission_result,
    validate_one_star_lobby_liveness_activity,
    validate_one_star_lobby_liveness_cue,
)
from app.engine.turn_loop import run_beat
from app.schemas.characters import PlayerSlotKind
from app.schemas.events import ObservableFact
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    OneStarEventRouterOutput,
)
from app.schemas.state import OpenCatIIEvent, OpenCommitment, SlotEntry
from tests.support.factories import (
    InstanceFakeDispatcher,
    character_record,
    checkpoint,
    router_output,
)
from tests.test_one_star_atomicity import _checkpoint, _hero, _mission


def _active_checkpoint(*, bind_party: bool, include_guide: bool = False):
    party = _hero(location="tower_floor_1")
    reserve = _hero(location="lobby")
    reserve.character_id = "reserve"
    reserve.name = "Reserve"
    checkpoint = _checkpoint(
        heroes=[party, reserve],
        active_mission=_mission(),
    )
    if bind_party:
        checkpoint.session.character_bindings["hero"] = "player-hero"
    if include_guide:
        checkpoint.characters.append(character_record("iselle", location="lobby"))
        checkpoint.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["state"][
            "guide_character_ids"
        ] = ["iselle"]
    return checkpoint


def _one_star_result(
    *,
    state_updates: list[dict[str, object]] | None = None,
    observer_ids: list[str] | None = None,
    agent_ids: list[str] | None = None,
    location_updates: list[dict[str, str]] | None = None,
    event_kind: str = "state_change",
    duration_s: int = 0,
    facts: list[ObservableFact] | None = None,
) -> OneStarEventRouterOutput:
    payload = router_output(
        event_id="evt_parallel",
        event_kind=event_kind,
        observer_ids=observer_ids or ["account_owner"],
        agent_ids=agent_ids,
        duration_s=duration_s,
        facts=facts,
        location_updates=location_updates,
    ).model_dump(mode="json")
    payload["state_updates"] = state_updates or []
    return OneStarEventRouterOutput.model_validate(payload)


def _pin_live_party_responder(checkpoint, *, reason: str) -> None:
    checkpoint.session.open_cat_ii_events = [OpenCatIIEvent(
        event_id="cat_two",
        initiator_id="enemy",
        initiator_intention="I strike at Hero.",
        required_responders=["hero"],
    )]
    checkpoint.session.active_act_slots["hero"] = SlotEntry(
        reason=reason,
        cat_ii_event_id="cat_two",
    )


def test_human_led_guard_is_active_without_a_slot_conflict() -> None:
    checkpoint = _active_checkpoint(bind_party=True)

    assert one_star_master_has_human_led_mission(
        checkpoint,
        actor_id="account_owner",
    ) is True
    assert one_star_master_may_act_while_mission_responder_pinned(
        checkpoint,
        actor_id="account_owner",
    ) is False


def test_pinned_admission_requires_only_live_human_cat_ii_responders() -> None:
    checkpoint = _active_checkpoint(bind_party=True)
    _pin_live_party_responder(checkpoint, reason="cat_ii_responder")

    assert one_star_active_mission_has_bound_party_member(checkpoint) is True
    assert one_star_master_may_act_while_mission_responder_pinned(
        checkpoint,
        actor_id="account_owner",
    ) is True


@pytest.mark.parametrize(
    "reason",
    ["initiator", "cat_ii_roll", "combat_reaction"],
)
def test_pinned_admission_rejects_non_responder_conflicts(reason: str) -> None:
    checkpoint = _active_checkpoint(bind_party=True)
    _pin_live_party_responder(checkpoint, reason=reason)

    assert one_star_master_may_act_while_mission_responder_pinned(
        checkpoint,
        actor_id="account_owner",
    ) is False


def test_human_led_validator_accepts_disjoint_account_control() -> None:
    checkpoint = _active_checkpoint(bind_party=True)
    result = _one_star_result(state_updates=[{
        "kind": "catalogue_apply",
        "target_id": "synthesis_chamber",
        "value": "1",
        "details": [],
    }])

    validate_one_star_human_led_mission_result(
        checkpoint,
        actor_id="account_owner",
        result=result,
    )


@pytest.mark.parametrize(
    "result",
    [
        _one_star_result(observer_ids=["account_owner", "hero"]),
        _one_star_result(location_updates=[{
            "character_id": "reserve",
            "location_label": "tower_floor_1",
        }]),
    ],
)
def test_human_led_validator_rejects_mission_leakage(result) -> None:
    checkpoint = _active_checkpoint(bind_party=True)

    with pytest.raises(OneStarTransactionError, match="human-led mission"):
        validate_one_star_human_led_mission_result(
            checkpoint,
            actor_id="account_owner",
            result=result,
        )


def test_human_led_validator_rejects_floor_progress_without_a_pin() -> None:
    checkpoint = _active_checkpoint(bind_party=True)
    result = _one_star_result(state_updates=[{
        "kind": "mission_update",
        "target_id": "mission_1",
        "value": "",
        "details": ["counter=clear:1/1", "report_kind=progress"],
    }])

    with pytest.raises(OneStarTransactionError, match="mission_update"):
        validate_one_star_human_led_mission_result(
            checkpoint,
            actor_id="account_owner",
            result=result,
        )
    with pytest.raises(OneStarTransactionError, match="autonomous lobby followers"):
        validate_one_star_human_led_mission_result(
            checkpoint,
            actor_id="reserve",
            result=result,
        )


def test_human_led_validator_allows_guide_induction_in_the_lobby() -> None:
    checkpoint = _active_checkpoint(bind_party=True, include_guide=True)
    summon = _one_star_result(
        state_updates=[{
            "kind": "summon",
            "target_id": "basic",
            "value": "1",
            "details": [],
        }],
        observer_ids=["account_owner", "iselle"],
        agent_ids=["iselle"],
    )
    tutorial = _one_star_result(
        state_updates=[{
            "kind": "tutorial_delivery",
            "target_id": "tower_gate",
            "value": "",
            "details": ["recipient=reserve"],
        }],
        observer_ids=["iselle", "reserve"],
        facts=[ObservableFact.only(
            "Iselle teaches Reserve how the tower gate works.",
            ["reserve"],
        )],
    )

    validate_one_star_human_led_mission_result(
        checkpoint,
        actor_id="account_owner",
        result=summon,
    )
    validate_one_star_human_led_mission_result(
        checkpoint,
        actor_id="iselle",
        result=tutorial,
    )

    tutorial.state_updates[0].details = ["recipient=hero"]
    with pytest.raises(OneStarTransactionError, match="tutorial recipients"):
        validate_one_star_human_led_mission_result(
            checkpoint,
            actor_id="iselle",
            result=tutorial,
        )


def test_human_led_validator_allows_only_a_pure_master_watch_query() -> None:
    checkpoint = _active_checkpoint(bind_party=True)
    query = _one_star_result(
        event_kind="query_response",
        observer_ids=["account_owner"],
        facts=[ObservableFact.only(
            "The Master sees Hero holding position on the first floor.",
            ["account_owner"],
            visual_subject_ids=["hero"],
        )],
    )

    validate_one_star_human_led_mission_result(
        checkpoint,
        actor_id="account_owner",
        result=query,
    )

    query.duration_s = 1
    with pytest.raises(OneStarTransactionError, match="watch queries"):
        validate_one_star_human_led_mission_result(
            checkpoint,
            actor_id="account_owner",
            result=query,
        )

    query.duration_s = 0
    query.observers[0].routing_role = "perception_enrichment"
    with pytest.raises(OneStarTransactionError, match="watch queries"):
        validate_one_star_human_led_mission_result(
            checkpoint,
            actor_id="account_owner",
            result=query,
        )

    extra_observer = _one_star_result(
        event_kind="query_response",
        observer_ids=["account_owner", "hero"],
        facts=[ObservableFact.only(
            "The Master sees Hero holding position on the first floor.",
            ["account_owner"],
            visual_subject_ids=["hero"],
        )],
    )
    with pytest.raises(OneStarTransactionError, match="watch queries"):
        validate_one_star_human_led_mission_result(
            checkpoint,
            actor_id="account_owner",
            result=extra_observer,
        )

    broadcast_fact = _one_star_result(
        event_kind="query_response",
        observer_ids=["account_owner"],
        facts=[ObservableFact.all(
            "The Master sees Hero holding position on the first floor.",
            visual_subject_ids=["hero"],
        )],
    )
    with pytest.raises(OneStarTransactionError, match="watch queries"):
        validate_one_star_human_led_mission_result(
            checkpoint,
            actor_id="account_owner",
            result=broadcast_fact,
        )


def test_autonomous_mission_batch_requires_owner_and_npc_only_party() -> None:
    checkpoint = _active_checkpoint(bind_party=False)
    mutation = _one_star_result(state_updates=[{
        "kind": "catalogue_apply",
        "target_id": "synthesis_chamber",
        "value": "1",
        "details": [],
    }])
    query = _one_star_result()

    assert one_star_should_autonomous_mission_batch_after_result(
        checkpoint,
        actor_id="account_owner",
        result=mutation,
    ) is True
    assert one_star_should_autonomous_mission_batch_after_result(
        checkpoint,
        actor_id="account_owner",
        result=query,
        user_input="Watch the deployed party on the first floor.",
    ) is True
    assert one_star_should_autonomous_mission_batch_after_result(
        checkpoint,
        actor_id="account_owner",
        result=query,
        user_input="How much does the catalogue item cost?",
    ) is False
    guide_handoff = _one_star_result(
        state_updates=[{
            "kind": "summon",
            "target_id": "basic",
            "value": "1",
            "details": [],
        }],
        observer_ids=["account_owner", "iselle"],
        agent_ids=["iselle"],
    )
    assert one_star_should_autonomous_mission_batch_after_result(
        checkpoint,
        actor_id="account_owner",
        result=guide_handoff,
    ) is False
    guide_handoff.clear_routing_roles()
    assert one_star_should_autonomous_mission_batch_after_result(
        checkpoint,
        actor_id="account_owner",
        result=guide_handoff,
    ) is True

    bound_checkpoint = _active_checkpoint(bind_party=True)
    assert one_star_should_autonomous_mission_batch_after_result(
        bound_checkpoint,
        actor_id="account_owner",
        result=mutation,
    ) is False


def test_lobby_liveness_request_uses_idle_nonparty_autonomous_heroes() -> None:
    checkpoint = _active_checkpoint(bind_party=False)
    result = _one_star_result(observer_ids=["account_owner"])

    request = one_star_lobby_liveness_request_after_result(
        checkpoint,
        actor_id="account_owner",
        result=result,
        user_input="(defer)",
    )

    assert request is not None
    assert request.mission_id == "mission_1"
    assert request.hero_ids == ("reserve",)
    assert "hero" not in request.hero_ids
    assert request.lobby_location == "lobby"

    checkpoint.session.character_bindings["reserve"] = "reserve-player"
    assert (
        one_star_lobby_liveness_request_after_result(
            checkpoint,
            actor_id="account_owner",
            result=result,
            user_input="(defer)",
        )
        is None
    )


@pytest.mark.parametrize("unavailable_kind", ["pinned", "committed", "authored"])
def test_lobby_liveness_request_excludes_unavailable_lobby_heroes(
    unavailable_kind: str,
) -> None:
    checkpoint = _active_checkpoint(bind_party=False)
    reserve = next(
        character
        for character in checkpoint.characters
        if character.character_id == "reserve"
    )
    if unavailable_kind == "pinned":
        checkpoint.session.active_act_slots["reserve"] = SlotEntry(
            reason="cat_ii_responder",
            cat_ii_event_id="cat_two",
        )
    elif unavailable_kind == "committed":
        checkpoint.session.open_commitments = [
            OpenCommitment(
                commitment_id="reserve_work",
                actor_ids=["reserve"],
                description="Reserve is already repairing equipment.",
                location_label="lobby",
            )
        ]
    else:
        reserve.player_slot_kind = PlayerSlotKind.player_authored

    assert (
        one_star_lobby_liveness_request_after_result(
            checkpoint,
            actor_id="account_owner",
            result=_one_star_result(observer_ids=["account_owner"]),
            user_input="(defer)",
        )
        is None
    )


def test_lobby_liveness_request_respects_existing_frontiers_and_cadence() -> None:
    checkpoint = _active_checkpoint(bind_party=False)
    quiet = _one_star_result(observer_ids=["account_owner"])

    assert (
        one_star_lobby_liveness_request_after_result(
            checkpoint,
            actor_id="account_owner",
            result=quiet,
            user_input="How much gold do I have?",
        )
        is None
    )
    assert (
        one_star_lobby_liveness_request_after_result(
            checkpoint,
            actor_id="reserve",
            result=quiet,
            user_input="(defer)",
        )
        is None
    )

    existing_frontier = _one_star_result(
        observer_ids=["account_owner", "reserve"],
        agent_ids=["reserve"],
    )
    assert (
        one_star_lobby_liveness_request_after_result(
            checkpoint,
            actor_id="account_owner",
            result=existing_frontier,
            user_input="(defer)",
        )
        is None
    )

    mutation = _one_star_result(
        state_updates=[
            {
                "kind": "catalogue_apply",
                "target_id": "synthesis_chamber",
                "value": "1",
                "details": [],
            }
        ]
    )
    assert (
        one_star_lobby_liveness_request_after_result(
            checkpoint,
            actor_id="account_owner",
            result=mutation,
        )
        is not None
    )


def test_lobby_liveness_cue_preserves_hero_choice_and_floor_separation() -> None:
    checkpoint = _active_checkpoint(bind_party=False)
    request = one_star_lobby_liveness_request_after_result(
        checkpoint,
        actor_id="account_owner",
        result=_one_star_result(observer_ids=["account_owner"]),
        user_input="(defer)",
    )
    assert request is not None
    cue = _one_star_result(
        observer_ids=["reserve"],
        agent_ids=["reserve"],
        event_kind="public_fact",
        facts=[ObservableFact.all("The training room is quiet and available.")],
    )

    validate_one_star_lobby_liveness_cue(
        checkpoint,
        request=request,
        result=cue,
    )

    cue.canonical_event.observable_facts[0].visual_subject_ids = ["reserve"]
    with pytest.raises(ValueError, match="pre-author"):
        validate_one_star_lobby_liveness_cue(
            checkpoint,
            request=request,
            result=cue,
        )

    leaked = _one_star_result(
        observer_ids=["account_owner", "reserve"],
        agent_ids=["reserve"],
        event_kind="public_fact",
        facts=[ObservableFact.all("The training room is quiet and available.")],
    )
    with pytest.raises(ValueError, match="private"):
        validate_one_star_lobby_liveness_cue(
            checkpoint,
            request=request,
            result=leaked,
        )


def test_lobby_liveness_activity_can_open_work_but_not_route_a_chat_chain() -> None:
    checkpoint = _active_checkpoint(bind_party=False)
    request = one_star_lobby_liveness_request_after_result(
        checkpoint,
        actor_id="account_owner",
        result=_one_star_result(observer_ids=["account_owner"]),
        user_input="(defer)",
    )
    assert request is not None
    activity = _one_star_result(
        observer_ids=["reserve"],
        facts=[
            ObservableFact.all(
                "Reserve begins a measured practice routine.",
                visual_subject_ids=["reserve"],
            )
        ],
    )
    activity.commitment_open.present = True
    activity.commitment_open.actor_ids = ["reserve"]
    activity.commitment_open.description = "practice in the lobby"
    activity.commitment_open.expected_duration_s = 1800
    activity.commitment_open.max_duration_s = 3600
    activity.commitment_open.location_label = "lobby"

    validate_one_star_lobby_liveness_activity(
        checkpoint,
        request=request,
        actor_id="reserve",
        result=activity,
    )

    activity.duration_s = 1
    with pytest.raises(ValueError, match="canonical instant"):
        validate_one_star_lobby_liveness_activity(
            checkpoint,
            request=request,
            actor_id="reserve",
            result=activity,
        )
    activity.duration_s = 0

    routed = _one_star_result(
        observer_ids=["reserve"],
        agent_ids=["reserve"],
        event_kind="beat_continues",
        facts=[ObservableFact.all("Reserve pauses over the practice blade.")],
    )
    with pytest.raises(ValueError, match="close without another routed output"):
        validate_one_star_lobby_liveness_activity(
            checkpoint,
            request=request,
            actor_id="reserve",
            result=routed,
        )

    floor_leak = _one_star_result(
        observer_ids=["reserve", "hero"],
        facts=[ObservableFact.all("The deployed Hero appears in the room.")],
    )
    with pytest.raises(ValueError, match="deployed party"):
        validate_one_star_lobby_liveness_activity(
            checkpoint,
            request=request,
            actor_id="reserve",
            result=floor_leak,
        )


def test_master_defer_runs_one_private_lobby_activity_before_floor_batch() -> None:
    checkpoint = _active_checkpoint(bind_party=False)
    checkpoint.session.character_bindings["account_owner"] = "master-player"
    checkpoint.session.config.settings.max_agent_cascades_per_beat = 1
    checkpoint.session.config.settings.max_events_per_beat = 3
    dispatcher = InstanceFakeDispatcher()

    initial = _one_star_result(
        observer_ids=["account_owner"],
        facts=[
            ObservableFact.only(
                "The watched floor remains active.",
                ["account_owner"],
            )
        ],
    )
    initial.event_id = "evt_master_defer"
    cue = _one_star_result(
        observer_ids=["reserve"],
        agent_ids=["reserve"],
        event_kind="public_fact",
        facts=[ObservableFact.all("The practice room is open and quiet.")],
    )
    cue.event_id = "evt_lobby_cue"
    activity = _one_star_result(
        observer_ids=["reserve"],
        facts=[
            ObservableFact.all(
                "Reserve begins a deliberate practice routine.",
                visual_subject_ids=["reserve"],
            )
        ],
    )
    activity.event_id = "evt_lobby_activity"
    floor_frontier = _one_star_result(
        observer_ids=["account_owner", "hero"],
        agent_ids=["hero"],
        event_kind="beat_continues",
        facts=[
            ObservableFact.only(
                "Hero sees the goblins regroup on the floor.",
                ["account_owner", "hero"],
                visual_subject_ids=["hero"],
            )
        ],
    )
    floor_frontier.event_id = "evt_floor_frontier"
    floor_action = _one_star_result(
        observer_ids=["account_owner", "hero"],
        agent_ids=["hero"],
        event_kind="beat_continues",
        facts=[
            ObservableFact.only(
                "Hero drives into the regrouping goblins.",
                ["account_owner", "hero"],
                visual_subject_ids=["hero"],
            )
        ],
    )
    floor_action.event_id = "evt_floor_action"

    for response in (initial, cue, activity, floor_frontier, floor_action):
        dispatcher.queue_route(response)
    dispatcher.queue_agent("Reserve begins a deliberate practice routine.")
    dispatcher.queue_agent("Hero drives into the regrouping goblins.")

    result = asyncio.run(
        run_beat(
            ckpt=checkpoint,
            dispatcher=dispatcher,
            actor_id="account_owner",
            intention="(defer)",
        )
    )

    assert result.ended_reason == "max_events_cap"
    assert [call["character_id"] for call in dispatcher.agent_calls] == [
        "reserve",
        "hero",
    ]
    assert dispatcher.agent_calls[0]["frame"] == "background"
    assert (
        "A stretch of mission-time is yours"
        in (dispatcher.agent_calls[0]["local_context"])
    )
    assert dispatcher.continuation_calls[0]["one_star_lobby_liveness"]
    assert dispatcher.continuation_calls[1]["actor_id"] == ""
    assert "one_star_lobby_liveness" not in dispatcher.continuation_calls[1]
    assert [event.event_id for event in checkpoint.canonical_events] == [
        "evt_master_defer",
        "evt_lobby_cue",
        "evt_lobby_activity",
        "evt_floor_frontier",
        "evt_floor_action",
    ]
    assert dispatcher.narrator_calls[0]["narration_mode"] == ("compressed_sequence")
    rendered_ids = {
        event.event_id for event in dispatcher.narrator_calls[0]["buffered_events"]
    }
    assert "evt_lobby_cue" not in rendered_ids
    assert "evt_lobby_activity" not in rendered_ids


def test_lobby_activity_leaves_a_human_led_floor_untouched() -> None:
    checkpoint = _active_checkpoint(bind_party=True)
    checkpoint.session.character_bindings["account_owner"] = "master-player"
    dispatcher = InstanceFakeDispatcher()

    initial = _one_star_result(
        observer_ids=["account_owner"],
        facts=[
            ObservableFact.only(
                "The Master leaves the deployed party to its own choices.",
                ["account_owner"],
            )
        ],
    )
    initial.event_id = "evt_human_floor_defer"
    cue = _one_star_result(
        observer_ids=["reserve"],
        agent_ids=["reserve"],
        event_kind="public_fact",
        facts=[ObservableFact.all("An unused workbench stands ready.")],
    )
    cue.event_id = "evt_human_floor_lobby_cue"
    activity = _one_star_result(
        observer_ids=["reserve"],
        facts=[
            ObservableFact.all(
                "Reserve begins restoring a battered shield.",
                visual_subject_ids=["reserve"],
            )
        ],
    )
    activity.event_id = "evt_human_floor_lobby_activity"
    activity.commitment_open.present = True
    activity.commitment_open.actor_ids = ["reserve"]
    activity.commitment_open.description = "restore the battered shield"
    activity.commitment_open.expected_duration_s = 1800
    activity.commitment_open.max_duration_s = 3600
    activity.commitment_open.location_label = "lobby"

    for response in (initial, cue, activity):
        dispatcher.queue_route(response)
    dispatcher.queue_agent("Reserve begins restoring a battered shield.")

    result = asyncio.run(
        run_beat(
            ckpt=checkpoint,
            dispatcher=dispatcher,
            actor_id="account_owner",
            intention="(defer)",
            one_star_human_led_mission_guard=True,
        )
    )

    assert result.ended_reason == "state_change"
    assert [call["character_id"] for call in dispatcher.agent_calls] == ["reserve"]
    assert [event.event_id for event in checkpoint.canonical_events] == [
        "evt_human_floor_defer",
        "evt_human_floor_lobby_cue",
        "evt_human_floor_lobby_activity",
    ]
    assert checkpoint.session.open_commitments[0].actor_ids == ["reserve"]
    party = next(
        character
        for character in checkpoint.characters
        if character.character_id == "hero"
    )
    assert party.location == "tower_floor_1"
    assert party.clock_at_s == 0
    assert party.pending_observations == []


def test_autonomous_mission_validator_allows_floor_fiction_and_passive_owner() -> None:
    checkpoint = _active_checkpoint(bind_party=False)
    result = _one_star_result(
        state_updates=[{
            "kind": "mission_update",
            "target_id": "mission_1",
            "value": "",
            "details": ["counter.clear=0/1"],
        }],
        observer_ids=["account_owner", "hero"],
        agent_ids=["hero"],
        facts=[ObservableFact.all("Hero advances through the watched floor.")],
    )

    validate_one_star_autonomous_mission_batch_result(
        checkpoint,
        actor_id="",
        result=result,
    )
    validate_one_star_autonomous_mission_batch_result(
        checkpoint,
        actor_id="hero",
        result=result,
    )


def test_autonomous_mission_validator_allows_scoped_terminal_system_recipient() -> None:
    checkpoint = _active_checkpoint(bind_party=False)
    newcomer = _hero(location="lobby")
    newcomer.character_id = "newcomer"
    newcomer.name = "Newcomer"
    newcomer.mechanics[ONE_STAR_HERO_KEY]["innate_system_sight"] = True
    checkpoint.characters.append(newcomer)
    result = _one_star_result(
        state_updates=[{
            "kind": "mission_end",
            "target_id": "mission_1",
            "value": "completed",
            "details": [
                "return_destination=lobby",
                "mvp_character_id=hero",
                "mvp_evidence_event_id=evt_parallel",
            ],
        }],
        observer_ids=["account_owner", "hero", "newcomer"],
        facts=[
            ObservableFact.only(
                "Hero reaches the floor objective.",
                ["account_owner", "hero"],
                visual_subject_ids=["hero"],
            ),
            ObservableFact.only(
                "The System posts the terminal floor report.",
                ["account_owner", "newcomer"],
            ),
        ],
    )

    validate_one_star_autonomous_mission_batch_result(
        checkpoint,
        actor_id="",
        result=result,
    )

    result.canonical_event.observable_facts[0].audience = "all_observers"
    result.canonical_event.observable_facts[0].visible_to = []
    with pytest.raises(OneStarTransactionError, match="broad floor facts"):
        validate_one_star_autonomous_mission_batch_result(
            checkpoint,
            actor_id="",
            result=result,
        )


@pytest.mark.parametrize(
    ("actor_id", "result", "error"),
    [
        (
            "account_owner",
            _one_star_result(observer_ids=["account_owner", "hero"]),
            "event actor",
        ),
        (
            "",
            _one_star_result(
                observer_ids=["account_owner", "hero"],
                agent_ids=["account_owner"],
            ),
            "observe_only",
        ),
        (
            "",
            _one_star_result(observer_ids=["account_owner", "iselle"]),
            "non-floor observers",
        ),
        (
            "",
            _one_star_result(state_updates=[{
                "kind": "catalogue_apply",
                "target_id": "synthesis_chamber",
                "value": "1",
                "details": [],
            }]),
            "catalogue_apply",
        ),
        (
            "",
            _one_star_result(location_updates=[{
                "character_id": "hero",
                "location_label": "lobby",
            }]),
            "lifecycle",
        ),
    ],
)
def test_autonomous_mission_validator_rejects_cross_front_authority(
    actor_id: str,
    result: OneStarEventRouterOutput,
    error: str,
) -> None:
    checkpoint = _active_checkpoint(bind_party=False, include_guide=True)

    with pytest.raises(OneStarTransactionError, match=error):
        validate_one_star_autonomous_mission_batch_result(
            checkpoint,
            actor_id=actor_id,
            result=result,
        )

def test_human_led_helpers_leave_rules_neutral_sessions_unchanged() -> None:
    generic = checkpoint()
    result = router_output(event_id="evt_generic")

    assert one_star_active_mission_has_bound_party_member(generic) is False
    assert one_star_master_has_human_led_mission(
        generic,
        actor_id="alice",
    ) is False
    assert one_star_master_may_act_while_mission_responder_pinned(
        generic,
        actor_id="alice",
    ) is False
    validate_one_star_human_led_mission_result(
        generic,
        actor_id="alice",
        result=result,
    )
    validate_one_star_autonomous_mission_batch_result(
        generic,
        actor_id="alice",
        result=result,
    )
    assert one_star_should_autonomous_mission_batch_after_result(
        generic,
        actor_id="alice",
        result=result,
        user_input="watch the mission",
    ) is False
    assert (
        one_star_lobby_liveness_request_after_result(
            generic,
            actor_id="alice",
            result=result,
            user_input="(defer)",
        )
        is None
    )
