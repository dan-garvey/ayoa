from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.one_star_adapter import (
    load_one_star_account,
    load_one_star_hero,
    one_star_event_fingerprint,
)
from app.engine.prompt_manager import PromptManager
from app.engine.router_batch import (
    RouterBatchContractError,
    materialize_router_batch,
    router_batch_correlation,
)
from app.engine.spawn_authoring import SpawnAuthoringCoordinator
from app.engine.story_dispatcher import (
    StoryDispatcher,
    _materialize_adapter_lifecycle,
    _router_input_block,
    _router_prompt_history,
    eligible_autonomous_character_ids,
)
from app.llm.client import (
    LLMClient,
    LLMResponse,
    StructuredOutputValidationError,
)
from app.schemas.event_router import (
    CommitmentInterruptSignal,
    CommitmentResolutionSignal,
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
from app.schemas.conversation import ConversationMessage
from app.schemas.events import ObservableFact
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    OneStarRouterBatchOutput,
    OneStarRouterEventDraft,
    OneStarStateUpdate,
)
from app.schemas.one_star_character_gen import AuthoredOneStarCharacter
from tests.support.factories import canonical_event, character_record, checkpoint
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


@pytest.mark.asyncio
async def test_structured_validation_rejection_logs_complete_raw_router_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ckpt = checkpoint(characters=[character_record("alice")])
    inputs = [_input(0, "alice")]
    raw_output = (
        '{"events":[],"padding":"'
        + ("x" * 600)
        + '","malformed_tail":"still retained"}'
    )
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(side_effect=StructuredOutputValidationError(
        "RouterBatchOutput",
        raw_output,
    ))
    dispatcher = StoryDispatcher(client, PromptManager("app/prompts"))

    with (
        caplog.at_level(logging.ERROR, logger="app.engine.router_batch"),
        pytest.raises(StructuredOutputValidationError),
    ):
        await dispatcher.route_batch(ckpt=ckpt, inputs=inputs)

    rejection = next(
        record.message.removeprefix("rejected router batch ")
        for record in caplog.records
        if record.message.startswith("rejected router batch ")
    )
    payload = json.loads(rejection)
    assert payload["session_id"] == ckpt.session.session_id
    assert payload["correlation_id"] == router_batch_correlation(inputs)
    assert payload["stage"] == "structured_validation"
    assert payload["raw_output"] == raw_output


@pytest.mark.asyncio
async def test_materialization_rejection_logs_exact_raw_router_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ckpt = checkpoint(characters=[character_record("alice")])
    inputs = [_input(0, "alice")]
    output = RouterBatchOutput(
        events=[_draft(feasible=[0], observers=["alice"])],
        next_turns=[RouterNextTurn(
            turn_kind="character",
            actor_id="unknown_actor",
            participant_ids=["unknown_actor"],
            source_event_index=-1,
        )],
    )
    raw_output = output.model_dump_json(indent=2)
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=LLMResponse(
        parsed=output,
        content=raw_output,
        model="offline-fixture",
    ))
    dispatcher = StoryDispatcher(client, PromptManager("app/prompts"))

    with (
        caplog.at_level(logging.ERROR, logger="app.engine.router_batch"),
        pytest.raises(
            RouterBatchContractError,
            match="unknown participants: unknown_actor",
        ),
    ):
        await dispatcher.route_batch(ckpt=ckpt, inputs=inputs)

    rejection = next(
        record.message.removeprefix("rejected router batch ")
        for record in caplog.records
        if record.message.startswith("rejected router batch ")
    )
    payload = json.loads(rejection)
    assert payload == {
        "session_id": ckpt.session.session_id,
        "correlation_id": router_batch_correlation(inputs),
        "stage": "materialization",
        "error_type": "RouterBatchContractError",
        "error": "next turn references unknown participants: unknown_actor",
        "raw_output": raw_output,
    }


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


def test_router_projection_replaces_durable_hashes_with_local_coordinates() -> None:
    event_id = "evt_0123456789ab"
    lane_id = "lane_abcdef012345"
    event = canonical_event(
        event_id=event_id,
        lane_id=lane_id,
        actor_ids=["alice"],
        observer_ids=["alice", "bob"],
    )
    event.source_submission_ids = ["submission_0123456789abcdef"]
    event.feasible_submission_ids = ["submission_0123456789abcdef"]
    ckpt = checkpoint(
        characters=[character_record("alice"), character_record("bob")],
        canonical_events=[event],
    )
    ckpt.session_conversation = [ConversationMessage(
        role="assistant",
        content=(
            f"prior_event {event_id} lane={lane_id} @0+0 actors=alice\n"
            "outcomes feasible=submission_0123456789abcdef infeasible=-"
        ),
    )]
    sourced = _input(0, "alice", source_event_ids=[event_id]).model_copy(
        update={
            "submission_id": "turn_0123456789ab",
            "lane_id": lane_id,
        },
    )
    independent = _input(1, "bob").model_copy(update={
        "submission_id": "turn_fedcba987654",
        "lane_id": "lane_1234567890ab",
    })

    history = _router_prompt_history(ckpt)
    rendered = "\n".join(
        [str(message.content) for message in history]
        + [_router_input_block(ckpt, [sourced, independent])]
    )

    for durable_id in (
        event_id,
        lane_id,
        "submission_0123456789abcdef",
        "turn_0123456789ab",
        "turn_fedcba987654",
        "lane_1234567890ab",
    ):
        assert durable_id not in rendered
    assert "prior_event sequence=0 causal_group=0" in rendered
    assert '"input_index":0,"causal_group":0' in rendered
    assert '"source_event_sequences":[0]' in rendered
    assert '"input_index":1,"causal_group":1' in rendered
    assert '"source_event_sequences":[]' in rendered


def test_router_commitment_signals_are_actor_addressed_only() -> None:
    resolution = CommitmentResolutionSignal(
        actor_ids=["liora_fen"],
        reason="resolved",
        resolved_at_offset_s=2,
    )
    interrupt = CommitmentInterruptSignal(
        actor_ids=["liora_fen"],
        observed_at_offset_s=1,
        reason="The footing gives way.",
    )

    assert set(resolution.model_json_schema()["properties"]) == {
        "actor_ids",
        "reason",
        "resolved_at_offset_s",
    }
    assert set(interrupt.model_json_schema()["properties"]) == {
        "actor_ids",
        "observed_at_offset_s",
        "reason",
    }
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CommitmentResolutionSignal.model_validate({
            **resolution.model_dump(),
            "commitment_id": "evt_dc4b0a3b83fa",
        })


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


@pytest.mark.asyncio
async def test_one_star_preparation_keeps_atomic_opening_history_activations() -> None:
    from app.schemas.characters import CharacterStatus

    renna = one_star_hero(
        status=CharacterStatus.dormant,
        owner="",
    ).model_copy(update={"character_id": "renna", "name": "Renna"})
    edren = one_star_hero(
        status=CharacterStatus.dormant,
        owner="",
    ).model_copy(update={"character_id": "edren", "name": "Edren"})
    ckpt = one_star_checkpoint(heroes=[renna, edren])
    owner = ckpt.characters[0]
    owner.mechanics[ONE_STAR_ACCOUNT_KEY]["config"]["summon_pools"]["opening"] = {
        "usage": "opening_roster",
        "slots": [
            {"kind": "fixed", "character_id": "renna"},
            {"kind": "fixed", "character_id": "edren"},
        ],
    }
    generic = _draft(
        feasible=[0],
        observers=["account_owner", "renna", "edren"],
    )
    output = OneStarRouterBatchOutput(
        events=[OneStarRouterEventDraft.model_validate({
            **generic.model_dump(),
            "state_updates": [
                OneStarStateUpdate(
                    kind="summon",
                    target_id="opening",
                    value="2",
                    details=[],
                ),
                OneStarStateUpdate(
                    kind="mission_start",
                    target_id="mission_1",
                    value="1",
                    details=[
                        "party=renna",
                        "party=edren",
                        "destination=tower_floor_1",
                        "completion=the floor is cleared",
                        "failure=the party is broken",
                        "counter.clear=0/1",
                    ],
                ),
                OneStarStateUpdate(
                    kind="hero_delta",
                    target_id="edren",
                    value="",
                    details=[
                        "hp_current=0",
                        "terminal_action=death",
                        "death_cause=A goblin pinned him against the wall.",
                    ],
                ),
                OneStarStateUpdate(
                    kind="mission_update",
                    target_id="mission_1",
                    value="",
                    details=["counter.clear=1/1"],
                ),
                OneStarStateUpdate(
                    kind="mission_end",
                    target_id="mission_1",
                    value="completed",
                    details=["return_destination=lobby"],
                ),
            ],
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

    event = batch.events[0].record
    characters = {item.character_id: item for item in ckpt.characters}
    assert [signal.character_id for signal in event.activate] == [
        "renna",
        "edren",
    ]
    assert characters["renna"].status == CharacterStatus.active
    assert characters["renna"].location == "lobby"
    assert characters["edren"].status == CharacterStatus.culled
    assert characters["edren"].location == "tower_floor_1"
    assert load_one_star_account(ckpt)[1].state.active_mission is None


@pytest.mark.asyncio
async def test_fresh_one_star_preparation_commits_authored_name_as_hero_id() -> None:
    ckpt = one_star_checkpoint()
    owner = ckpt.characters[0]
    pool = owner.mechanics[ONE_STAR_ACCOUNT_KEY]["config"]["summon_pools"][
        "basic"
    ]
    pool["eligible_existing_ids"] = []
    pool["fresh_generation_allowed"] = True
    generic = _draft(
        feasible=[0],
        observers=["account_owner"],
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
    transient_id = output.events[0].spawn[0].character_id
    assert transient_id == "lobby_a_basic_0001"
    batch = materialize_router_batch(
        checkpoint=ckpt,
        inputs=inputs,
        output=output,
    )
    authored = AuthoredOneStarCharacter.model_validate({
        "name": "Mara Venn",
        "location": "lobby",
        "role": "newly summoned Hero",
        "appearance": "A weathered baker holding a walking staff.",
        "public_context": "",
        "default_loadout": "A worn walking staff.",
        "faction": "",
        "actor": {"may_act_offstage": False, "facts": []},
        "router_summary": "",
        "one_star_hero": {
            "strong_stat_id": "power",
            "weak_stat_id": "spirit",
            "equipment": [],
            "skills": [],
            "conditions": [],
            "persistent_injuries": [],
            "innate_system_sight": False,
            "hidden_capabilities": [],
        },
    })
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=LLMResponse(
        parsed=authored,
        content=authored.model_dump_json(),
        model="offline-fixture",
    ))
    dispatcher = StoryDispatcher(client, PromptManager("app/prompts"))

    await dispatcher.prepare_batch(
        ckpt=ckpt,
        batch=batch,
        inputs=inputs,
        player_actor_ids={"account_owner"},
    )

    event = batch.events[0].record
    assert [request.character_id for request in event.spawn] == ["mara_venn"]
    assert transient_id not in {
        character.character_id for character in ckpt.characters
    }
    hero_record = next(
        character
        for character in ckpt.characters
        if character.character_id == "mara_venn"
    )
    hero = load_one_star_hero(hero_record)
    assert hero is not None
    assert hero.owner_lobby_id == "lobby_a"
    assert hero.acquisition_event_id == event.event_id
    account = load_one_star_account(ckpt)[1]
    assert account.state.summon_draw_counters == {
        "basic": 1,
    }
    assert account.state.applied_event_fingerprints[event.event_id] == (
        one_star_event_fingerprint(event.model_dump(mode="json"))
    )
    eligible_ids = eligible_autonomous_character_ids(ckpt)
    assert "mara_venn" in eligible_ids
    assert transient_id not in eligible_ids


@pytest.mark.asyncio
async def test_one_star_reauthoring_preserves_canonical_name_derived_id() -> None:
    ckpt = one_star_checkpoint()
    pool = ckpt.characters[0].mechanics[ONE_STAR_ACCOUNT_KEY]["config"][
        "summon_pools"
    ]["basic"]
    pool["eligible_existing_ids"] = []
    pool["fresh_generation_allowed"] = True
    generic = _draft(feasible=[0], observers=["account_owner"])
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
    _materialize_adapter_lifecycle(ckpt, output)
    batch = materialize_router_batch(
        checkpoint=ckpt,
        inputs=[_input(0, "account_owner")],
        output=output,
    )
    event = batch.events[0].record
    event.spawn = [
        event.spawn[0].model_copy(update={"character_id": "mara_venn"})
    ]
    generated = character_record("mara_venn").model_copy(
        update={"name": "Mara Venn"}
    )
    character_manager = MagicMock()
    character_manager.spawn_characters = AsyncMock(return_value=[generated])
    coordinator = SpawnAuthoringCoordinator(character_manager)

    key = coordinator.start(
        checkpoint=ckpt,
        event=event,
        transaction_id="tx_replay",
        event_fingerprint="fingerprint",
    )
    records = await coordinator.result(key)

    assert [record.character_id for record in records] == ["mara_venn"]
    call = character_manager.spawn_characters.await_args
    assert [request.character_id for request in call.args[1]] == ["mara_venn"]
    assert call.kwargs["one_star_hero_ids"] == {"mara_venn"}
    assert "name_derived_character_ids" not in call.kwargs
