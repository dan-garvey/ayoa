from __future__ import annotations

import pytest

from app.engine.router_batch import (
    RouterBatchContractError,
    materialize_router_batch,
)
from app.engine.story_dispatcher import (
    StoryDispatcher,
    _materialize_adapter_lifecycle,
    _router_input_block,
)
from app.schemas.event_router import (
    LocationUpdateSignal,
    ObserverGroups,
    RouterBatchOutput,
    RouterEventDraft,
    RouterInputEnvelope,
    RouterNextTurn,
    SpawnRequest,
    SpawnSeed,
    WakeSignal,
)
from app.schemas.events import ObservableFact
from app.schemas.one_star import (
    OneStarRouterBatchOutput,
    OneStarRouterEventDraft,
    OneStarStateUpdate,
)
from tests.support.factories import character_record, checkpoint
from tests.support.one_star_factories import one_star_checkpoint, one_star_hero


def _input(
    index: int,
    actor_id: str,
    *,
    chosen_at_s: int = 0,
    source_event_ids: list[str] | None = None,
) -> RouterInputEnvelope:
    return RouterInputEnvelope(
        submission_id=f"sub_{index}",
        input_index=index,
        lane_id=f"lane_{actor_id}",
        kind="player" if index == 0 else "character",
        actor_ids=[actor_id],
        participant_ids=[actor_id],
        source_event_ids=source_event_ids or [],
        chosen_at_s=chosen_at_s,
        observed_through_event_sequence=-1,
        observed_through_s=chosen_at_s,
        payload=f"proposal {index}",
    )


def _draft(
    *,
    feasible: list[int],
    infeasible: list[int] | None = None,
    observers: list[str],
    fact: str = "Something changes.",
) -> RouterEventDraft:
    return RouterEventDraft(
        feasible_input_indexes=feasible,
        infeasible_input_indexes=infeasible or [],
        duration_s=1,
        observable_facts=[ObservableFact.all(fact, duration_s=1)],
        observers=ObserverGroups(
            direct=observers,
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


def test_mixed_feasibility_is_preserved_per_submission() -> None:
    ckpt = checkpoint(
        session_id="batch",
        turn_index=7,
        characters=[character_record("alice"), character_record("bob")],
    )
    inputs = [_input(0, "alice", chosen_at_s=4), _input(1, "bob", chosen_at_s=9)]
    output = RouterBatchOutput(
        events=[
            _draft(
                feasible=[0],
                infeasible=[1],
                observers=["alice", "bob"],
            )
        ],
        next_turns=[],
    )

    result = materialize_router_batch(
        checkpoint=ckpt,
        inputs=inputs,
        output=output,
    )

    assert result.feasible_submission_ids == ("sub_0",)
    assert result.infeasible_submission_ids == ("sub_1",)
    event = result.events[0].record
    assert event.source_submission_ids == ["sub_0", "sub_1"]
    assert event.effective_at_s == 9
    assert event.event_id.startswith("evt_")


def test_router_input_exposes_one_agent_ownership_list() -> None:
    ckpt = checkpoint(
        bindings={"alice": "1"},
        characters=[character_record("alice"), character_record("bob")],
    )

    rendered = _router_input_block(ckpt, [_input(0, "alice")])

    assert "<autonomous_character_ids>\nbob\n</autonomous_character_ids>" in rendered
    assert "eligible_autonomous_character_ids" not in rendered
    assert "agent_owned_appearance_target_ids" not in rendered


def test_overlapping_inputs_are_legal_when_merged_for_arbitration() -> None:
    ckpt = checkpoint(
        characters=[character_record("alice"), character_record("bob")]
    )
    left = _input(0, "alice")
    right = _input(1, "bob").model_copy(update={
        "lane_id": left.lane_id,
        "participant_ids": ["alice", "bob"],
    })
    result = materialize_router_batch(
        checkpoint=ckpt,
        inputs=[left, right],
        output=RouterBatchOutput(
            events=[_draft(feasible=[0, 1], observers=["alice", "bob"])],
            next_turns=[],
        ),
    )
    assert result.events[0].record.actor_ids == ["alice", "bob"]


def test_overlapping_inputs_cannot_become_sibling_events() -> None:
    ckpt = checkpoint(
        characters=[character_record("alice"), character_record("bob")]
    )
    left = _input(0, "alice")
    right = _input(1, "bob").model_copy(update={"lane_id": left.lane_id})
    with pytest.raises(RouterBatchContractError, match="must resolve in one"):
        materialize_router_batch(
            checkpoint=ckpt,
            inputs=[left, right],
            output=RouterBatchOutput(
                events=[
                    _draft(feasible=[0], observers=["alice"]),
                    _draft(feasible=[1], observers=["bob"]),
                ],
                next_turns=[],
            ),
        )


def test_exclusively_infeasible_no_effect_input_creates_no_event() -> None:
    ckpt = checkpoint(characters=[character_record("alice")])
    draft = _draft(feasible=[], infeasible=[0], observers=[])
    draft.observable_facts = []
    # Assignment does not rerun Pydantic validation; rebuild the intended strict
    # no-event shape at the schema boundary.
    draft = RouterEventDraft.model_validate(draft.model_dump())
    result = materialize_router_batch(
        checkpoint=ckpt,
        inputs=[_input(0, "alice")],
        output=RouterBatchOutput(events=[draft], next_turns=[]),
    )
    assert result.events == ()
    assert result.infeasible_submission_ids == ("sub_0",)


def test_batch_rejects_missing_or_duplicate_input_accounting() -> None:
    inputs = [_input(0, "alice"), _input(1, "bob")]
    output = RouterBatchOutput(
        events=[_draft(feasible=[0], observers=["alice"])],
        next_turns=[],
    )
    with pytest.raises(ValueError, match="exactly once"):
        output.validate_for_inputs(inputs)


def test_next_turn_directness_derives_from_source_observers() -> None:
    ckpt = checkpoint(
        bindings={"alice": "1"},
        characters=[character_record("alice"), character_record("bob")],
    )
    output = RouterBatchOutput(
        events=[_draft(feasible=[0], observers=["alice", "bob"])],
        next_turns=[
            RouterNextTurn(
                turn_kind="character",
                actor_id="bob",
                participant_ids=["bob"],
                source_event_index=0,
            )
        ],
    )
    result = materialize_router_batch(
        checkpoint=ckpt,
        inputs=[_input(0, "alice")],
        output=output,
    )
    assert result.events[0].record.observation_level_for("bob") == "direct"
    assert result.next_turns[0].source_event_ids == [
        result.events[0].record.event_id
    ]
    assert result.next_turns[0].gating_pov_ids == ["alice"]


def test_sibling_events_cannot_share_mutation_targets() -> None:
    ckpt = checkpoint(
        characters=[character_record("alice"), character_record("bob")]
    )
    first = _draft(feasible=[0], observers=["alice"])
    second = _draft(feasible=[1], observers=["bob"])
    first.location_updates = [
        LocationUpdateSignal(character_id="alice", location_label="hall")
    ]
    second.location_updates = [
        LocationUpdateSignal(character_id="alice", location_label="yard")
    ]
    with pytest.raises(RouterBatchContractError, match="conflicting shared"):
        materialize_router_batch(
            checkpoint=ckpt,
            inputs=[_input(0, "alice"), _input(1, "bob")],
            output=RouterBatchOutput(events=[first, second], next_turns=[]),
        )


def test_parallel_next_turns_require_disjoint_participants() -> None:
    output = RouterBatchOutput(
        events=[_draft(feasible=[0], observers=["alice", "bob"])],
        next_turns=[
            RouterNextTurn(
                turn_kind="character",
                actor_id="alice",
                participant_ids=["alice", "bob"],
                source_event_index=0,
            ),
            RouterNextTurn(
                turn_kind="character",
                actor_id="bob",
                participant_ids=["bob"],
                source_event_index=0,
            ),
        ],
    )
    with pytest.raises(ValueError, match="share participants"):
        output.validate_for_inputs([_input(0, "alice")])


def _contested_draft(
    responder_ids: list[str],
    *,
    feasible: list[int] | None = None,
) -> RouterEventDraft:
    return RouterEventDraft.model_validate({
        **_draft(
            feasible=feasible or [0],
            observers=list(dict.fromkeys(["alice", *responder_ids])),
        ).model_dump(),
        "duration_s": 0,
        "observable_facts": [ObservableFact.all("Alice reaches for the key.")],
        "required_responders": responder_ids,
    })


def test_contested_opening_can_merge_after_its_feasible_actor_proposal() -> None:
    ckpt = checkpoint(
        characters=[
            character_record("alice"),
            character_record("bob"),
            character_record("cara"),
        ]
    )
    result = materialize_router_batch(
        checkpoint=ckpt,
        inputs=[_input(0, "alice"), _input(1, "cara")],
        output=RouterBatchOutput(
            events=[_contested_draft(["bob"], feasible=[0, 1])],
            next_turns=[],
        ),
    )

    assert result.events[0].record.feasible_submission_ids == ["sub_0", "sub_1"]


def test_contested_opening_rejects_initiator_as_responder() -> None:
    ckpt = checkpoint(characters=[character_record("alice")])
    with pytest.raises(RouterBatchContractError, match="own responder"):
        materialize_router_batch(
            checkpoint=ckpt,
            inputs=[_input(0, "alice")],
            output=RouterBatchOutput(
                events=[_contested_draft(["alice"])],
                next_turns=[],
            ),
        )


def test_contested_opening_rejects_unavailable_responder() -> None:
    from app.schemas.characters import CharacterStatus

    ckpt = checkpoint(
        characters=[
            character_record("alice"),
            character_record("bob", status=CharacterStatus.dormant),
        ]
    )
    with pytest.raises(RouterBatchContractError, match="unavailable"):
        materialize_router_batch(
            checkpoint=ckpt,
            inputs=[_input(0, "alice")],
            output=RouterBatchOutput(
                events=[_contested_draft(["bob"])],
                next_turns=[],
            ),
        )


def test_next_turn_rejects_inactive_participant() -> None:
    from app.schemas.characters import CharacterStatus

    ckpt = checkpoint(
        characters=[
            character_record("alice"),
            character_record("bob", status=CharacterStatus.dormant),
        ]
    )
    with pytest.raises(RouterBatchContractError, match="inactive participants"):
        materialize_router_batch(
            checkpoint=ckpt,
            inputs=[_input(0, "alice")],
            output=RouterBatchOutput(
                events=[_draft(feasible=[0], observers=["alice"])],
                next_turns=[RouterNextTurn(
                    turn_kind="character",
                    actor_id="alice",
                    participant_ids=["alice", "bob"],
                    source_event_index=0,
                )],
            ),
        )


@pytest.mark.parametrize("lifecycle", ["spawn", "activate"])
def test_sourced_newly_active_character_can_take_next_turn(lifecycle: str) -> None:
    from app.schemas.characters import CharacterStatus

    bob = character_record(
        "bob",
        status=(
            CharacterStatus.dormant
            if lifecycle == "activate"
            else CharacterStatus.active
        ),
    )
    ckpt = checkpoint(
        characters=[character_record("alice"), bob]
        if lifecycle == "activate"
        else [character_record("alice")]
    )
    draft = _draft(feasible=[0], observers=["alice", "bob"])
    if lifecycle == "spawn":
        draft.spawn = [SpawnRequest(
            character_id="bob",
            seed=SpawnSeed(
                role="new arrival",
                reason="Alice calls for Bob.",
                location="gatehouse",
                objectives=[],
                knowledge_tier=0,
            ),
        )]
    else:
        draft.activate = [WakeSignal(
            character_id="bob",
            location_label="gatehouse",
        )]
    output = RouterBatchOutput(
        events=[RouterEventDraft.model_validate(draft.model_dump())],
        next_turns=[RouterNextTurn(
            turn_kind="character",
            actor_id="bob",
            participant_ids=["bob"],
            source_event_index=0,
        )],
    )

    result = materialize_router_batch(
        checkpoint=ckpt,
        inputs=[_input(0, "alice")],
        output=output,
    )

    assert result.next_turns[0].actor_id == "bob"


def test_adapter_owned_activation_precedes_common_next_turn_validation() -> None:
    from app.schemas.characters import CharacterStatus

    reserve = one_star_hero(
        status=CharacterStatus.dormant,
        owner="",
    ).model_copy(update={"character_id": "reserve", "name": "Reserve"})
    ckpt = one_star_checkpoint(heroes=[reserve])
    generic = _draft(
        feasible=[0],
        observers=["account_owner", "reserve"],
    )
    output = OneStarRouterBatchOutput(
        events=[OneStarRouterEventDraft.model_validate({
            **generic.model_dump(),
            "state_updates": [OneStarStateUpdate(
                kind="summon",
                target_id="basic",
                value="1",
                details=[],
            )],
        })],
        next_turns=[RouterNextTurn(
            turn_kind="character",
            actor_id="reserve",
            participant_ids=["reserve"],
            source_event_index=0,
        )],
    )

    _materialize_adapter_lifecycle(ckpt, output)
    result = materialize_router_batch(
        checkpoint=ckpt,
        inputs=[_input(0, "account_owner")],
        output=output,
    )

    assert [item.character_id for item in result.events[0].record.activate] == [
        "reserve"
    ]
    assert result.next_turns[0].actor_id == "reserve"


@pytest.mark.asyncio
async def test_one_star_preparation_applies_projected_activation_to_roster() -> None:
    from app.engine.one_star_adapter import load_one_star_hero
    from app.schemas.characters import CharacterStatus

    reserve = one_star_hero(
        status=CharacterStatus.dormant,
        owner="",
    ).model_copy(update={"character_id": "reserve", "name": "Reserve"})
    ckpt = one_star_checkpoint(heroes=[reserve])
    generic = _draft(
        feasible=[0],
        observers=["account_owner", "reserve"],
    )
    output = OneStarRouterBatchOutput(
        events=[OneStarRouterEventDraft.model_validate({
            **generic.model_dump(),
            "state_updates": [OneStarStateUpdate(
                kind="summon",
                target_id="basic",
                value="1",
                details=[],
            )],
        })],
        next_turns=[],
    )
    inputs = [_input(0, "account_owner")]
    _materialize_adapter_lifecycle(ckpt, output)
    batch = materialize_router_batch(
        checkpoint=ckpt,
        inputs=inputs,
        output=output,
    )
    dispatcher = StoryDispatcher(None, None)  # type: ignore[arg-type]

    await dispatcher.prepare_batch(
        ckpt=ckpt,
        batch=batch,
        inputs=inputs,
        player_actor_ids={"account_owner"},
    )

    activated = next(
        character for character in ckpt.characters
        if character.character_id == "reserve"
    )
    assert activated.status == CharacterStatus.active
    assert activated.location == "lobby"
    assert load_one_star_hero(activated).acquisition_event_id == (
        batch.events[0].record.event_id
    )
