"""Cross-path contracts for semantic selections, player handoff, and privacy."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.bot.engine_bridge import EngineBridge
from app.engine.event_runtime import (
    close_cat_ii,
    commit_event_batch,
    frontier_head_turns,
    open_cat_ii,
    visible_facts_for,
)
from app.engine.dnd_cat_ii import DndResolvedCanonicalEvent
from app.engine.narrator_delivery import _handoff_for_job, merge_narrator_lanes
from app.engine.prompt_manager import PromptManager
from app.engine.router_batch import RouterBatchContractError, materialize_router_batch
from app.engine.story_coordinator import (
    _append_adapter_frontier,
    advance_story,
    commit_adapter_resolution,
    immutable_checkpoint,
    player_input,
    prepare_frontier_inputs,
    ready_frontier_turns,
)
from app.engine.story_dispatcher import StoryDispatcher
from app.llm.client import LLMClient
from app.schemas.characters import CharacterStatus, PlayerSlotKind
from app.schemas.event_router import (
    DndRouterBatchOutput,
    FrontierTurn,
    RouterBatchOutput,
    RouterNextTurn,
    WakeSignal,
)
from app.schemas.events import ObservableFact
from tests.support.factories import (
    canonical_event,
    character_record,
    checkpoint,
    router_event_draft,
)
from tests.test_orchestrator_autonomous import _orchestrator
from tests.test_story_coordinator import FakeDispatcher


def _state():
    return checkpoint(
        session_id="handoff-contract",
        bindings={"alice": "1"},
        player_character_id="alice",
        characters=[
            character_record(name, is_playable=True)
            for name in ("alice", "bob", "cara", "dora")
        ],
    )


def _turn(event, actor="alice", *, turn_id="selected", participants=None):
    return FrontierTurn(
        turn_id=turn_id,
        lane_id=event.causal_lane_id,
        turn_kind="character",
        actor_id=actor,
        participant_ids=participants or [actor],
        source_event_ids=[event.event_id],
        created_event_sequence=0,
        gating_pov_ids=[],
    )


def _pick(actor="alice", group=0):
    return RouterNextTurn(
        turn_kind="character",
        actor_id=actor,
        participant_ids=[actor],
        causal_group=group,
    )


def _no_event():
    return router_event_draft(
        feasible_input_indexes=[],
        infeasible_input_indexes=[0],
        observer_ids=[],
        facts=[],
    )


class CapturingDispatcher(FakeDispatcher):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.policies = []
        self.contexts = []

    async def narrator_compose(self, **kwargs):
        self.policies.append(kwargs["handoff_policy"])
        return await super().narrator_compose(**kwargs)

    async def draft_character_turn(self, **kwargs):
        self.contexts.append(kwargs["local_context"])
        return await super().draft_character_turn(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("authored", [False, True])
async def test_human_selection_forces_handoff_without_agent_draft_or_obligation(
    authored,
):
    ckpt = _state()
    if authored:
        ckpt.characters[0].player_slot_kind = PlayerSlotKind.player_authored
    dispatcher = CapturingDispatcher(
        [
            RouterBatchOutput(
                events=[router_event_draft(observer_ids=["alice", "bob"])],
                next_turns=[_pick()],
            )
        ]
    )
    await advance_story(
        ckpt,
        dispatcher,
        [player_input(ckpt, character_id="bob", payload="What do you think?")],
    )
    assert dispatcher.policies == ["forced"]
    assert dispatcher.draft_started == set()
    assert ckpt.session.action_obligations == {}
    assert ckpt.session.router_frontier[0].actor_id == "alice"
    assert len(ckpt.session.delivery_outbox) == 1
    assert ready_frontier_turns(ckpt) == []

    live_dispatcher = StoryDispatcher(
        MagicMock(spec=LLMClient), PromptManager("app/prompts")
    )
    live_dispatcher.character_agent.draft_turn = AsyncMock()
    with pytest.raises(RuntimeError, match="player-owned"):
        await live_dispatcher.draft_character_turn(
            ckpt=ckpt, character_id="alice", local_context=""
        )
    live_dispatcher.character_agent.draft_turn.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("defer", [False, True])
async def test_player_reply_consumes_matching_selection_without_acknowledging_private_anchor(
    defer,
):
    ckpt = _state()
    public = canonical_event(
        event_id="invite", observer_ids=["alice", "bob"], effective_at_s=10
    )
    private = canonical_event(
        event_id="secret", observer_ids=["bob"], effective_at_s=100
    )
    commit_event_batch(ckpt, [public, private])
    ckpt.session.narrator_render_jobs = []
    ckpt.session.last_acknowledged_event_sequence_by_pov["alice"] = 0
    ckpt.session.router_frontier = [_turn(private)]
    submission = player_input(
        ckpt, character_id="alice", payload="(defer)" if defer else "I answer."
    )
    assert submission.frontier_turn_id == "selected"
    assert submission.envelope.source_event_ids == ["secret"]
    assert submission.envelope.observed_through_event_sequence == 0
    assert submission.envelope.observed_through_s == 10
    assert submission.envelope.chosen_at_s == 100
    dispatcher = FakeDispatcher(
        [
            RouterBatchOutput(
                events=[_no_event() if defer else router_event_draft()],
                next_turns=[],
            )
        ]
    )
    await advance_story(ckpt, dispatcher, [submission])
    assert ckpt.session.router_frontier == []
    assert ckpt.session.last_acknowledged_event_sequence_by_pov["alice"] == 0


@pytest.mark.asyncio
async def test_claim_preserves_selection_but_stops_autonomous_dispatch():
    ckpt = _state()
    event = canonical_event(observer_ids=["alice", "bob"])
    jobs = commit_event_batch(ckpt, [event])
    ckpt.session.router_frontier = [_turn(event, "bob")]
    assert [turn.actor_id for turn in ready_frontier_turns(ckpt)] == ["bob"]
    bridge = EngineBridge.__new__(EngineBridge)
    bridge._lock_for = AsyncMock(return_value=asyncio.Lock())
    bridge.orchestrator = SimpleNamespace(cancel_autonomous=MagicMock())
    bridge.checkpoint_mgr = SimpleNamespace(
        load_latest=lambda _sid: ckpt, save=MagicMock()
    )
    await bridge.claim_player_character(ckpt.session.session_id, "bob", 2)
    assert ckpt.session.router_frontier[0].actor_id == "bob"
    assert ready_frontier_turns(immutable_checkpoint(ckpt)) == []
    assert _handoff_for_job(ckpt, jobs[0])[0] == "forced"
    assert ckpt.session.action_obligations == {}


def test_adapter_keeps_humans_and_serializes_multiple_followups():
    ckpt = _state()
    event = canonical_event(observer_ids=["bob"])
    commit_event_batch(ckpt, [event])
    _append_adapter_frontier(ckpt, event, ["alice", "bob", "cara"])
    assert [turn.actor_id for turn in ckpt.session.router_frontier] == [
        "alice",
        "bob",
        "cara",
    ]
    assert ready_frontier_turns(ckpt) == []
    assert ckpt.session.action_obligations == {}


@pytest.mark.asyncio
async def test_unseen_scheduling_source_orders_time_without_granting_knowledge():
    ckpt = _state()
    invite = canonical_event(
        event_id="invite", observer_ids=["alice", "bob"], effective_at_s=10
    )
    private = canonical_event(
        event_id="private",
        observer_ids=["alice"],
        effective_at_s=100,
        facts=[ObservableFact.all("A private preparation Bob cannot see.")],
    )
    commit_event_batch(ckpt, [invite, private])
    assert visible_facts_for(private, "bob") == []
    dispatcher = CapturingDispatcher([])
    prepared = await prepare_frontier_inputs(ckpt, dispatcher, [_turn(private, "bob")])
    assert dispatcher.contexts == [""]
    assert prepared[0].envelope.chosen_at_s == 100
    assert prepared[0].envelope.observed_through_event_sequence == 0
    assert prepared[0].envelope.observed_through_s == 10
    assert ckpt.characters[1].clock_at_s == 10


@pytest.mark.asyncio
async def test_independent_turn_uses_actual_fact_cutoff_not_whole_event_duration():
    ckpt = _state()
    event = canonical_event(
        observer_ids=["alice", "bob"],
        duration_s=80,
        facts=[
            ObservableFact.only(
                "Bob hears the bell.", ["bob"], at_offset_s=2, duration_s=3
            ),
            ObservableFact.only(
                "Alice waits alone.", ["alice"], at_offset_s=70, duration_s=10
            ),
        ],
    )
    commit_event_batch(ckpt, [event])
    selected = _turn(event, "bob").model_copy(update={"source_event_ids": []})
    prepared = await prepare_frontier_inputs(ckpt, CapturingDispatcher([]), [selected])
    assert prepared[0].envelope.observed_through_event_sequence == 0
    assert prepared[0].envelope.observed_through_s == 5


def test_input_group_aliases_resolve_to_merged_event_in_router_order():
    ckpt = _state()
    first = canonical_event(event_id="old_a", lane_id="lane_a", observer_ids=["alice"])
    second = canonical_event(event_id="old_b", lane_id="lane_b", observer_ids=["bob"])
    commit_event_batch(ckpt, [first, second])
    inputs = [
        player_input(ckpt, character_id=actor, payload="We meet.").envelope.model_copy(
            update={
                "input_index": index,
                "lane_id": event.causal_lane_id,
            }
        )
        for index, (actor, event) in enumerate((("alice", first), ("bob", second)))
    ]
    batch = materialize_router_batch(
        checkpoint=ckpt,
        inputs=inputs,
        output=RouterBatchOutput(
            events=[
                router_event_draft(
                    feasible_input_indexes=[0, 1], observer_ids=["alice", "bob"]
                )
            ],
            next_turns=[_pick("bob", 1), _pick("alice", 0)],
        ),
    )
    assert [turn.actor_id for turn in batch.next_turns] == ["bob", "alice"]
    assert all(
        turn.source_event_ids == [batch.events[0].record.event_id]
        for turn in batch.next_turns
    )
    assert len({turn.lane_id for turn in batch.next_turns}) == 1


def test_historical_group_survives_no_event_without_replaying_lifecycle():
    ckpt = _state()
    event = canonical_event(
        event_id="earlier",
        activate=[WakeSignal(character_id="bob", location_label="hall")],
    )
    latest = canonical_event(event_id="latest", effective_at_s=10)
    ckpt.canonical_events = [event, latest]
    inputs = [player_input(ckpt, character_id="alice", payload="(defer)").envelope]
    output = RouterBatchOutput(events=[_no_event()], next_turns=[_pick("bob")])
    batch = materialize_router_batch(checkpoint=ckpt, inputs=inputs, output=output)
    assert batch.events == ()
    assert batch.next_turns[0].source_event_ids == ["latest"]
    assert batch.next_turns[0].gating_pov_ids == []
    ckpt.characters[1].status = CharacterStatus.dormant
    # Even selecting the historical activation itself cannot reactivate Bob.
    ckpt.canonical_events = [event]
    with pytest.raises(RouterBatchContractError, match="inactive participants"):
        materialize_router_batch(checkpoint=ckpt, inputs=inputs, output=output)


@pytest.mark.parametrize(
    "group,error", [(99, "unknown causal group"), (0, "no established event")]
)
def test_missing_group_authority_remains_a_hard_error(group, error):
    ckpt = _state()
    with pytest.raises(RouterBatchContractError, match=error):
        materialize_router_batch(
            checkpoint=ckpt,
            inputs=[
                player_input(ckpt, character_id="alice", payload="(defer)").envelope
            ],
            output=RouterBatchOutput(
                events=[_no_event()], next_turns=[_pick("bob", group)]
            ),
        )


def test_world_requires_group_and_obsolete_source_field_is_not_accepted():
    with pytest.raises(ValidationError, match="causal group"):
        RouterNextTurn(
            turn_kind="world", actor_id="", participant_ids=[], causal_group=None
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        RouterNextTurn.model_validate({**_pick().model_dump(), "source_event_index": 0})


def test_blocked_human_reserves_transitive_conflicts_even_under_preference():
    ckpt = _state()
    event = canonical_event()
    human = _turn(event, "alice", participants=["alice", "bob"])
    second = _turn(event, "bob", turn_id="second", participants=["bob", "cara"])
    second.lane_id = "other_lane"
    third = _turn(event, "cara", turn_id="third")
    third.lane_id = "third_lane"
    independent = _turn(event, "dora", turn_id="independent")
    independent.lane_id = "independent_lane"
    ckpt.session.router_frontier = [human, second, third, independent]
    assert [
        turn.actor_id for turn in frontier_head_turns(ckpt, lane_id="third_lane")
    ] == ["alice"]
    assert [
        turn.actor_id
        for turn in ready_frontier_turns(ckpt, preferred_lane_ids={"third_lane"})
    ] == ["dora"]


@pytest.mark.asyncio
async def test_serial_follower_rebases_to_predecessor_result_and_hands_off():
    ckpt = _state()
    event = canonical_event(observer_ids=["alice", "bob"])
    commit_event_batch(ckpt, [event])
    ckpt.session.router_frontier = [_turn(event, "bob", turn_id="first"), _turn(event)]
    dispatcher = CapturingDispatcher(
        [RouterBatchOutput(events=[router_event_draft()], next_turns=[])]
    )
    prepared = await prepare_frontier_inputs(
        ckpt, dispatcher, ready_frontier_turns(ckpt)
    )
    await advance_story(ckpt, dispatcher, prepared)
    saved = immutable_checkpoint(ckpt)
    assert saved.session.router_frontier[0].source_event_ids == [
        saved.canonical_events[-1].event_id
    ]
    assert dispatcher.policies == ["forced"]
    assert saved.session.narrator_render_jobs == []
    assert ready_frontier_turns(saved) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("new_human_selection", [False, True])
async def test_no_event_consumption_or_human_selection_flushes_pending_narration(
    new_human_selection,
):
    ckpt = _state()
    event = canonical_event(observer_ids=["alice", "bob"])
    commit_event_batch(ckpt, [event])
    ckpt.session.router_frontier = [_turn(event, "bob")]
    dispatcher = CapturingDispatcher(
        [
            RouterBatchOutput(
                events=[_no_event()],
                next_turns=[_pick()] if new_human_selection else [],
            )
        ]
    )
    prepared = await prepare_frontier_inputs(
        ckpt, dispatcher, ready_frontier_turns(ckpt)
    )
    await advance_story(ckpt, dispatcher, prepared)
    assert len(ckpt.canonical_events) == 1
    assert dispatcher.policies == ["forced"]
    assert len(ckpt.session.delivery_outbox) == 1
    assert ckpt.session.narrator_render_jobs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining_npc", [False, True])
async def test_restart_flushes_pending_narration_without_routing_even_at_batch_limit(
    remaining_npc,
):
    ckpt = _state()
    event = canonical_event()
    commit_event_batch(ckpt, [event])
    if remaining_npc:
        ckpt.session.router_frontier = [_turn(event, "bob")]
    ckpt.session.autonomous_router_batches_since_player = 1000
    runtime = _orchestrator(immutable_checkpoint(ckpt))
    runtime.dispatcher = CapturingDispatcher([])
    await asyncio.wait_for(
        runtime._autonomous_worker(ckpt.session.session_id), timeout=1
    )
    saved = runtime.checkpoint_mgr.value
    assert saved.session.narrator_render_jobs == []
    assert len(saved.session.delivery_outbox) == 1
    assert len(saved.canonical_events) == 1
    assert runtime.dispatcher.route_inputs == []


def test_narrator_merge_preserves_each_povs_private_refs_and_retry_state():
    ckpt = _state()
    ckpt.session.character_bindings["bob"] = "2"
    a = canonical_event(event_id="a", lane_id="a_lane", observer_ids=["alice"])
    b = canonical_event(event_id="b", lane_id="b_lane", observer_ids=["alice", "bob"])
    commit_event_batch(ckpt, [a, b])
    ckpt.session.narrator_render_jobs[0].status = "failed"
    ckpt.session.narrator_render_jobs[0].last_error = "retry me"
    merge_narrator_lanes(ckpt, {"a_lane": "merged", "b_lane": "merged"})
    saved = immutable_checkpoint(ckpt)
    jobs = {job.pov_character_id: job for job in saved.session.narrator_render_jobs}
    assert jobs["alice"].source_event_ids == ["a", "b"]
    assert jobs["alice"].status == "failed"
    assert jobs["bob"].source_event_ids == ["b"]
    assert all(job.lane_id == "merged" for job in jobs.values())


def test_observer_only_without_a_fact_remains_rejected():
    with pytest.raises(ValidationError, match="every observer"):
        router_event_draft(
            observer_ids=["alice", "bob", "cara"],
            facts=[ObservableFact.only("Alice quietly invites Bob.", ["bob"])],
        )


def test_unclaimed_authored_selection_is_still_forbidden():
    ckpt = _state()
    ckpt.characters[1].player_slot_kind = PlayerSlotKind.player_authored
    with pytest.raises(RouterBatchContractError, match="unclaimed player-authored"):
        materialize_router_batch(
            checkpoint=ckpt,
            inputs=[player_input(ckpt, character_id="alice", payload="Ask.").envelope],
            output=RouterBatchOutput(
                events=[router_event_draft()], next_turns=[_pick("bob")]
            ),
        )


@pytest.mark.parametrize("historical", [False, True])
def test_current_and_historical_contests_cannot_source_followups(historical):
    ckpt = _state()
    event = canonical_event(observer_ids=["alice", "bob"])
    if historical:
        commit_event_batch(ckpt, [event])
        open_cat_ii(
            ckpt,
            initiator_id="alice",
            initiator_intention="Take the key.",
            required_responder_ids=["bob"],
            opening_event=event,
        )
        draft = _no_event()
    else:
        draft = router_event_draft(
            observer_ids=["alice", "bob"], required_responders=["bob"]
        )
    with pytest.raises(RouterBatchContractError, match="unresolved contest"):
        materialize_router_batch(
            checkpoint=ckpt,
            inputs=[
                player_input(
                    ckpt, character_id="alice", payload="Take the key."
                ).envelope
            ],
            output=RouterBatchOutput(events=[draft], next_turns=[_pick("cara")]),
        )


def test_preexisting_serial_follower_waits_for_contest_resolution():
    ckpt = _state()
    event = canonical_event(observer_ids=["alice", "bob"])
    commit_event_batch(ckpt, [event])
    opened = open_cat_ii(
        ckpt,
        initiator_id="alice",
        initiator_intention="Take the key.",
        required_responder_ids=["bob"],
        opening_event=event,
    )
    ckpt.session.router_frontier = [_turn(event, "cara")]
    assert ready_frontier_turns(ckpt) == []
    close_cat_ii(ckpt, opened.event_id)
    assert [turn.actor_id for turn in ready_frontier_turns(ckpt)] == ["cara"]


@pytest.mark.asyncio
async def test_adapter_resolution_consumes_human_selection_and_rebases_serial_follower():
    ckpt = _state()
    event = canonical_event(observer_ids=["alice", "bob"])
    commit_event_batch(ckpt, [event])
    ckpt.session.router_frontier = [_turn(event), _turn(event, "bob", turn_id="second")]
    resolved = canonical_event(
        event_id="resolved", lane_id="adapter_lane", actor_ids=["alice"]
    )
    dispatcher = CapturingDispatcher([])
    await commit_adapter_resolution(
        ckpt,
        dispatcher,
        working=immutable_checkpoint(ckpt),
        resolution=DndResolvedCanonicalEvent(event=resolved),
    )
    assert [turn.actor_id for turn in ckpt.session.router_frontier] == ["bob"]
    assert ckpt.session.router_frontier[0].source_event_ids == ["resolved"]
    assert ckpt.session.delivery_outbox[0].source_event_ids == [
        event.event_id,
        "resolved",
    ]


@pytest.mark.asyncio
async def test_off_lane_event_does_not_strand_exhausted_foreground_buffer():
    ckpt = _state()
    event = canonical_event(observer_ids=["alice", "bob"])
    commit_event_batch(ckpt, [event])
    foreground = _turn(event, "bob", turn_id="foreground")
    offstage = _turn(event, "cara", turn_id="offstage").model_copy(
        update={
            "lane_id": "offstage_lane",
            "source_event_ids": [],
        }
    )
    ckpt.session.router_frontier = [foreground, offstage]
    dispatcher = CapturingDispatcher(
        [
            RouterBatchOutput(
                events=[
                    _no_event(),
                    router_event_draft(
                        feasible_input_indexes=[1], observer_ids=["cara"]
                    ),
                ],
                next_turns=[],
            )
        ]
    )
    prepared = await prepare_frontier_inputs(
        ckpt, dispatcher, ready_frontier_turns(ckpt)
    )
    await advance_story(ckpt, dispatcher, prepared)
    assert len(ckpt.canonical_events) == 2
    assert ckpt.session.router_frontier == []
    assert ckpt.session.narrator_render_jobs == []
    assert ckpt.session.delivery_outbox[0].source_event_ids == [event.event_id]


@pytest.mark.asyncio
async def test_failed_restart_render_is_saved_once_and_does_not_spin():
    ckpt = _state()
    commit_event_batch(ckpt, [canonical_event()])
    runtime = _orchestrator(immutable_checkpoint(ckpt))
    runtime.dispatcher = CapturingDispatcher([])
    runtime.dispatcher.failed_narrator_ids.add("alice")
    await asyncio.wait_for(
        runtime._autonomous_worker(ckpt.session.session_id), timeout=1
    )
    assert (
        runtime.checkpoint_mgr.value.session.narrator_render_jobs[0].status == "failed"
    )
    await asyncio.wait_for(
        runtime._autonomous_worker(ckpt.session.session_id), timeout=1
    )
    assert runtime.dispatcher.policies == ["forced"]


@pytest.mark.asyncio
async def test_covenant_rejected_player_garvey_batch_now_delivers_human_handoff():
    # Exact first rejected playtest output, with only its retired source-index
    # coordinate adapted to the new group contract. No fictional facts edited.
    output = DndRouterBatchOutput.model_validate_json(
        (
            Path(__file__).parent / "fixtures" / "covenant_player_handoff.json"
        ).read_text()
    )
    ckpt = checkpoint(
        session_id="covenant-rejected-01",
        bindings={"player_garvey": "1"},
        player_character_id="player_garvey",
        characters=[
            character_record(actor) for actor in output.events[0].observers.all_ids
        ],
    )
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    dispatcher = CapturingDispatcher([output])
    await advance_story(
        ckpt,
        dispatcher,
        [
            player_input(
                ckpt,
                character_id="ashara_vel_kothren",
                payload="Rowan Garvey. You have my attention.",
            )
        ],
    )
    assert dispatcher.policies == ["forced"]
    assert dispatcher.draft_started == set()
    assert ckpt.session.router_frontier[0].actor_id == "player_garvey"
    assert ckpt.session.action_obligations == {}
    assert ckpt.session.delivery_outbox[0].pov_character_id == "player_garvey"
