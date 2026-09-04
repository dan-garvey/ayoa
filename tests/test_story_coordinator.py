from __future__ import annotations

import asyncio

import pytest

from app.engine.character_agent import CharacterAgentTurnDraft
from app.engine.router_batch import materialize_router_batch
from app.engine.story_coordinator import (
    advance_story,
    player_input,
    prepare_autonomous_contest_resolutions,
    prepare_ready_frontier_batch,
    ready_frontier_turns,
    release_frontier_gates_for_pov_action,
)
from app.engine.story_dispatcher import (
    _router_input_block,
    eligible_autonomous_character_ids,
)
from app.schemas.agents import CharacterAgentOutput
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import (
    FrontierTurn,
    ObserverGroups,
    RouterBatchOutput,
    RouterEventDraft,
    RouterNextTurn,
)
from app.schemas.events import ObservableFact
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from tests.support.factories import character_record, checkpoint


def _event(
    index: int,
    *,
    observers: list[str],
    fact: str,
    responders: list[str] | None = None,
) -> RouterEventDraft:
    return RouterEventDraft(
        feasible_input_indexes=[index],
        infeasible_input_indexes=[],
        duration_s=0,
        observable_facts=[ObservableFact.all(fact)],
        observers=ObserverGroups(direct=observers, indirect=[], inferred=[]),
        required_responders=responders or [],
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


def _draft(character_id: str, text: str) -> CharacterAgentTurnDraft:
    return CharacterAgentTurnDraft(
        output=CharacterAgentOutput(character_id=character_id, public_text=text),
        user_message=ConversationMessage(role="user", content="moment"),
        assistant_message=ConversationMessage(role="assistant", content=text),
    )


class FakeDispatcher:
    def __init__(
        self,
        outputs: list[RouterBatchOutput],
        *,
        parallel_draft_count: int = 1,
    ):
        self.outputs = list(outputs)
        self.route_inputs: list[list[object]] = []
        self.prepared_batches = 0
        self.committed_drafts: list[tuple[str, int]] = []
        self.draft_started: set[str] = set()
        self.parallel_draft_count = parallel_draft_count
        self.all_drafts_started = asyncio.Event()
        self.failed_narrator_ids: set[str] = set()

    async def route_batch(self, *, ckpt, inputs):
        self.route_inputs.append(list(inputs))
        return materialize_router_batch(
            checkpoint=ckpt,
            inputs=inputs,
            output=self.outputs.pop(0),
        )

    async def prepare_batch(self, *, ckpt, batch, inputs, player_actor_ids):
        del ckpt, batch, inputs, player_actor_ids
        self.prepared_batches += 1

    async def draft_character_turn(self, *, ckpt, character_id, local_context):
        self.draft_started.add(character_id)
        if len(self.draft_started) >= self.parallel_draft_count:
            self.all_drafts_started.set()
        await asyncio.wait_for(self.all_drafts_started.wait(), timeout=1)
        return _draft(character_id, f"{character_id} acts")

    def commit_character_turn(
        self,
        *,
        ckpt,
        character_id,
        draft,
        committed_at_s,
    ):
        self.committed_drafts.append((character_id, committed_at_s))
        ckpt.character_conversations.setdefault(character_id, []).extend([
            draft.user_message,
            draft.assistant_message,
        ])

    async def narrator_compose(
        self,
        *,
        ckpt,
        character_id,
        event_refs,
        partial_mode,
        user_input,
        handoff_policy,
        handoff_context,
        narration_mode,
    ):
        del ckpt, event_refs, partial_mode, handoff_policy, handoff_context, narration_mode
        if character_id in self.failed_narrator_ids:
            raise RuntimeError(f"narrator failed for {character_id}")
        text = f"render for {character_id}"
        return (
            NarratorFinalOutput(
                handoff="render",
                handoff_reason="visible motion is ready",
                final_text=text,
            ),
            TranscriptEntry(user=user_input, assistant=text),
        )


@pytest.mark.asyncio
async def test_player_and_independent_frontier_route_in_one_batch() -> None:
    ckpt = checkpoint(
        bindings={"alice": "1"},
        player_character_id="alice",
        characters=[character_record("alice"), character_record("bob")],
    )
    ckpt.session.router_frontier.append(FrontierTurn(
        turn_id="turn_bob",
        lane_id="lane_bob",
        turn_kind="character",
        actor_id="bob",
        participant_ids=["bob"],
        source_event_ids=[],
        created_event_sequence=0,
        gating_pov_ids=[],
    ))
    dispatcher = FakeDispatcher([RouterBatchOutput(
        events=[
            _event(0, observers=["alice"], fact="Alice moves."),
            _event(1, observers=["bob"], fact="Bob trains."),
        ],
        next_turns=[],
    )])
    initial = [player_input(ckpt, character_id="alice", payload="I move.")]
    prepared = await prepare_ready_frontier_batch(
        ckpt,
        dispatcher,
        initial=initial,
    )
    result = await advance_story(
        ckpt,
        dispatcher,
        prepared,
        user_input_by_pov={"alice": "I move."},
    )

    assert len(dispatcher.route_inputs[0]) == 2
    assert {item.lane_id for item in dispatcher.route_inputs[0]} == {
        initial[0].envelope.lane_id,
        "lane_bob",
    }
    assert result.events_committed == 2
    assert [item.actor_ids for item in ckpt.canonical_events] == [
        ["alice"],
        ["bob"],
    ]
    assert ckpt.session.router_frontier == []
    assert dispatcher.committed_drafts == [("bob", 0)]


@pytest.mark.asyncio
async def test_merged_contest_uses_first_feasible_proposal_as_initiator() -> None:
    ckpt = checkpoint(
        bindings={"alice": "1"},
        player_character_id="alice",
        characters=[
            character_record("alice"),
            character_record("bob"),
            character_record("cara"),
        ],
    )
    initial = [player_input(ckpt, character_id="alice", payload="I take the key.")]
    ckpt.session.router_frontier.append(FrontierTurn(
        turn_id="turn_cara",
        lane_id=initial[0].envelope.lane_id,
        turn_kind="character",
        actor_id="cara",
        participant_ids=["cara"],
        source_event_ids=[],
        created_event_sequence=0,
        gating_pov_ids=[],
    ))
    merged = _event(
        0,
        observers=["alice", "bob", "cara"],
        fact="Alice reaches for the key while Cara speaks.",
        responders=["bob"],
    ).model_copy(update={"feasible_input_indexes": [0, 1]})
    dispatcher = FakeDispatcher([RouterBatchOutput(events=[merged], next_turns=[])])

    prepared = await prepare_ready_frontier_batch(
        ckpt,
        dispatcher,
        initial=initial,
    )
    await advance_story(
        ckpt,
        dispatcher,
        prepared,
        user_input_by_pov={"alice": "I take the key."},
    )

    assert len(prepared) == 2
    assert ckpt.session.open_cat_ii_events[0].initiator_id == "alice"
    assert ckpt.session.open_cat_ii_events[0].initiator_intention == "I take the key."


@pytest.mark.asyncio
async def test_successful_render_gates_frontier_until_a_pov_acts() -> None:
    ckpt = checkpoint(
        bindings={"alice": "1"},
        player_character_id="alice",
        characters=[character_record("alice"), character_record("bob")],
    )
    dispatcher = FakeDispatcher([RouterBatchOutput(
        events=[_event(0, observers=["alice", "bob"], fact="A door opens.")],
        next_turns=[RouterNextTurn(
            turn_kind="character",
            actor_id="bob",
            participant_ids=["bob"],
            source_event_index=0,
        )],
    )])
    result = await advance_story(
        ckpt,
        dispatcher,
        [player_input(ckpt, character_id="alice", payload="Open it.")],
    )

    assert result.pause_reason == ""
    assert ckpt.session.router_frontier[0].gating_pov_ids == ["alice"]
    assert len(ckpt.session.delivery_outbox) == 1
    assert ckpt.session.delivery_outbox[0].payload.prose == "render for alice"
    assert ckpt.session.turn_index == 1


@pytest.mark.asyncio
async def test_newer_frontier_supersedes_an_overlapping_gated_turn() -> None:
    ckpt = checkpoint(
        bindings={"alice": "1"},
        characters=[
            character_record("alice"),
            character_record("bob"),
            character_record("cara"),
        ],
    )
    ckpt.session.router_frontier.append(FrontierTurn(
        turn_id="turn_old_cara",
        lane_id="lane_old",
        turn_kind="character",
        actor_id="cara",
        participant_ids=["cara"],
        source_event_ids=[],
        created_event_sequence=0,
        gating_pov_ids=["alice"],
    ))
    dispatcher = FakeDispatcher([RouterBatchOutput(
        events=[_event(0, observers=["alice", "bob"], fact="Pressure changes.")],
        next_turns=[RouterNextTurn(
            turn_kind="character",
            actor_id="bob",
            participant_ids=["bob", "cara"],
            source_event_index=0,
        )],
    )])

    await advance_story(
        ckpt,
        dispatcher,
        [player_input(ckpt, character_id="alice", payload="Act.")],
    )

    assert len(ckpt.session.router_frontier) == 1
    assert ckpt.session.router_frontier[0].actor_id == "bob"
    assert ckpt.session.router_frontier[0].participant_ids == ["bob", "cara"]


@pytest.mark.asyncio
async def test_narrator_failure_gates_only_its_causal_lane() -> None:
    ckpt = checkpoint(
        bindings={"alice": "1", "bea": "2"},
        characters=[
            character_record("alice"),
            character_record("bea"),
            character_record("bob"),
            character_record("cara"),
        ],
    )
    dispatcher = FakeDispatcher([RouterBatchOutput(
        events=[
            _event(0, observers=["alice", "bob"], fact="Alice sees Bob."),
            _event(1, observers=["bea", "cara"], fact="Bea sees Cara."),
        ],
        next_turns=[
            RouterNextTurn(
                turn_kind="character",
                actor_id="bob",
                participant_ids=["bob"],
                source_event_index=0,
            ),
            RouterNextTurn(
                turn_kind="character",
                actor_id="cara",
                participant_ids=["cara"],
                source_event_index=1,
            ),
        ],
    )])
    dispatcher.failed_narrator_ids.add("bea")
    first = player_input(ckpt, character_id="alice", payload="First.")
    second = player_input(ckpt, character_id="bea", payload="Second.")
    second = type(second)(envelope=second.envelope.model_copy(update={
        "input_index": 1,
        "lane_id": "lane_bea",
    }))
    result = await advance_story(ckpt, dispatcher, [first, second])

    assert result.pause_reason == "narrator_delivery_failed"
    gates = {item.actor_id: item.gating_pov_ids for item in ckpt.session.router_frontier}
    assert gates == {"bob": ["alice"], "cara": ["bea"]}
    assert [item.pov_character_id for item in ckpt.session.delivery_outbox] == [
        "alice"
    ]


def test_first_named_pov_action_releases_the_whole_lane_gate() -> None:
    ckpt = checkpoint(
        bindings={"alice": "1", "bea": "2"},
        characters=[
            character_record("alice"),
            character_record("bea"),
            character_record("bob"),
        ],
    )
    ckpt.session.router_frontier.append(FrontierTurn(
        turn_id="turn_bob",
        lane_id="lane_shared",
        turn_kind="character",
        actor_id="bob",
        participant_ids=["bob"],
        source_event_ids=[],
        created_event_sequence=0,
        gating_pov_ids=["alice", "bea"],
    ))

    assert release_frontier_gates_for_pov_action(ckpt, "alice") == 1
    assert ckpt.session.router_frontier[0].gating_pov_ids == []


@pytest.mark.asyncio
async def test_player_input_joins_overlapping_frontier_for_router_arbitration() -> None:
    ckpt = checkpoint(
        bindings={"alice": "1"},
        player_character_id="alice",
        characters=[character_record("alice"), character_record("bob")],
    )
    initial = player_input(ckpt, character_id="alice", payload="Stop Bob.")
    ckpt.session.router_frontier.append(FrontierTurn(
        turn_id="turn_bob",
        lane_id=initial.envelope.lane_id,
        turn_kind="character",
        actor_id="bob",
        participant_ids=["alice", "bob"],
        source_event_ids=[],
        created_event_sequence=0,
        gating_pov_ids=[],
    ))
    merged = _event(0, observers=["alice", "bob"], fact="They collide.")
    merged.feasible_input_indexes = [0, 1]
    dispatcher = FakeDispatcher([
        RouterBatchOutput(events=[merged], next_turns=[]),
    ])

    prepared = await prepare_ready_frontier_batch(
        ckpt,
        dispatcher,
        initial=[initial],
    )
    result = await advance_story(ckpt, dispatcher, prepared)

    assert len(dispatcher.route_inputs[0]) == 2
    assert result.events_committed == 1
    assert ckpt.canonical_events[0].actor_ids == ["alice", "bob"]
    assert ckpt.session.router_frontier == []


@pytest.mark.asyncio
async def test_followup_drafting_overlaps_narrator_rendering() -> None:
    class ConcurrentDispatcher(FakeDispatcher):
        def __init__(self, outputs):
            super().__init__(outputs)
            self.narrator_started = asyncio.Event()
            self.followup_started = asyncio.Event()

        async def draft_character_turn(self, *, ckpt, character_id, local_context):
            del ckpt, local_context
            await asyncio.wait_for(self.narrator_started.wait(), timeout=1)
            self.followup_started.set()
            return _draft(character_id, f"{character_id} acts")

        async def narrator_compose(self, **kwargs):
            self.narrator_started.set()
            await asyncio.wait_for(self.followup_started.wait(), timeout=1)
            return await super().narrator_compose(**kwargs)

    ckpt = checkpoint(
        bindings={"alice": "1"},
        player_character_id="alice",
        characters=[character_record("alice"), character_record("bob")],
    )
    dispatcher = ConcurrentDispatcher([RouterBatchOutput(
        events=[_event(0, observers=["alice"], fact="Alice waits.")],
        next_turns=[RouterNextTurn(
            turn_kind="character",
            actor_id="bob",
            participant_ids=["bob"],
            source_event_index=-1,
        )],
    )])

    result = await advance_story(
        ckpt,
        dispatcher,
        [player_input(ckpt, character_id="alice", payload="Wait.")],
    )

    assert dispatcher.narrator_started.is_set()
    assert dispatcher.followup_started.is_set()
    assert [item.frontier_turn_id for item in result.prepared_followups] == [
        ckpt.session.router_frontier[0].turn_id
    ]


@pytest.mark.asyncio
async def test_cat_ii_agent_responses_are_parallel_and_resolve_as_one_input() -> None:
    ckpt = checkpoint(
        bindings={"alice": "1"},
        player_character_id="alice",
        characters=[
            character_record("alice"),
            character_record("bob"),
            character_record("cara"),
        ],
    )
    dispatcher = FakeDispatcher([
        RouterBatchOutput(
            events=[_event(
                0,
                observers=["alice", "bob", "cara"],
                fact="Alice reaches for the disputed object.",
                responders=["bob", "cara"],
            )],
            next_turns=[],
        ),
        RouterBatchOutput(
            events=[_event(
                0,
                observers=["alice", "bob", "cara"],
                fact="Cara gets there first.",
            )],
            next_turns=[],
        ),
    ], parallel_draft_count=2)
    await advance_story(
        ckpt,
        dispatcher,
        [player_input(ckpt, character_id="alice", payload="I grab it.")],
    )
    assert eligible_autonomous_character_ids(ckpt) == []
    ckpt.session.router_frontier.append(FrontierTurn(
        turn_id="turn_bob_while_contested",
        lane_id="lane_elsewhere",
        turn_kind="character",
        actor_id="bob",
        participant_ids=["bob"],
        source_event_ids=[],
        created_event_sequence=1,
        gating_pov_ids=[],
    ))
    assert ready_frontier_turns(ckpt) == []
    ckpt.session.router_frontier.clear()
    prepared = await prepare_autonomous_contest_resolutions(ckpt, dispatcher)

    assert dispatcher.draft_started == {"bob", "cara"}
    assert len(prepared) == 1
    assert prepared[0].envelope.kind == "cat_ii_resolution"
    resolution_block = _router_input_block(
        ckpt,
        [prepared[0].envelope],
    )
    assert "<autonomous_character_ids>\nbob,cara\n" in resolution_block
    assert '"character_id":"bob"' in prepared[0].envelope.payload
    assert '"character_id":"cara"' in prepared[0].envelope.payload
    await advance_story(ckpt, dispatcher, prepared)
    assert ckpt.session.open_cat_ii_events == []
    assert len(dispatcher.route_inputs[1]) == 1


@pytest.mark.asyncio
async def test_adapter_failure_rolls_back_every_staged_batch_change() -> None:
    class FailingDispatcher(FakeDispatcher):
        async def prepare_batch(self, *, ckpt, batch, inputs, player_actor_ids):
            del batch, inputs, player_actor_ids
            ckpt.world_state.facts.append("should roll back")
            raise RuntimeError("adapter rejected batch")

    ckpt = checkpoint(
        bindings={"alice": "1"},
        characters=[character_record("alice")],
    )
    dispatcher = FailingDispatcher([RouterBatchOutput(
        events=[_event(0, observers=["alice"], fact="Never committed.")],
        next_turns=[],
    )])
    before = ckpt.model_dump()
    with pytest.raises(RuntimeError, match="adapter rejected"):
        await advance_story(
            ckpt,
            dispatcher,
            [player_input(ckpt, character_id="alice", payload="Try.")],
        )
    assert ckpt.model_dump() == before
