"""Rules-neutral selection and merge contracts for semantic background work."""

from __future__ import annotations

import asyncio

import pytest

from app.engine.background_threads import (
    BackgroundThreadContractError,
    BackgroundThreadRequest,
    anchor_background_thread_result,
    background_thread_candidate_ids,
    background_thread_requests,
    validate_background_thread_result,
    validate_background_thread_selection,
)
from app.engine.turn_loop import (
    _PreparedBackgroundThread,
    _assert_background_thread_participants_are_unchanged,
    run_beat,
)
from app.schemas.characters import ActorRecord
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import (
    BackgroundThreadPick,
    EventRouterOutput,
    LocationUpdateSignal,
)
from app.schemas.events import ObservableFact
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import DndCombatState, DndCombatantState, SlotEntry
from tests.support.factories import character_record, checkpoint, router_output


def _offstage_actor(character_id: str, *, location: str = ""):
    return character_record(
        character_id,
        location=location,
        actor=ActorRecord(may_act_offstage=True),
    )


def _background_result(
    actor_id: str,
    *,
    event_id: str | None = None,
) -> EventRouterOutput:
    return router_output(
        event_id=event_id or f"evt_background_{actor_id}",
        event_kind="state_change",
        observer_ids=[actor_id],
        facts=[
            ObservableFact.all(
                f"{actor_id} makes one concrete move.",
                visual_subject_ids=[actor_id],
            )
        ],
    )


def test_candidate_filter_is_safety_only_and_does_not_group_by_location() -> None:
    player = character_record("player", location="stage", is_playable=True)
    cook = _offstage_actor("cook", location="same_label")
    smith = _offstage_actor("smith", location="same_label")
    pinned = _offstage_actor("pinned", location="elsewhere")
    combatant = _offstage_actor("combatant", location="elsewhere")
    no_offstage = character_record(
        "no_offstage",
        location="elsewhere",
        actor=ActorRecord(may_act_offstage=False),
    )
    ckpt = checkpoint(
        bindings={"player": "human"},
        player_character_id="player",
        characters=[player, cook, smith, pinned, combatant, no_offstage],
    )
    ckpt.session.active_act_slots["pinned"] = SlotEntry(reason="actor_turn")
    ckpt.session.active_combat = DndCombatState(
        combatants=[
            DndCombatantState(
                combatant_id="combatant",
                character_id="combatant",
            )
        ]
    )

    assert background_thread_candidate_ids(ckpt) == ["cook", "smith"]


def test_router_must_select_an_offscreen_candidate_without_location_inference() -> None:
    ckpt = checkpoint(
        bindings={"player": "human"},
        player_character_id="player",
        characters=[
            character_record("player", location="stage", is_playable=True),
            _offstage_actor("renna", location="stale_floor_one"),
            _offstage_actor("mirelle", location="lobby"),
        ],
    )
    focal = router_output(
        event_id="evt_party_two_enters",
        observer_ids=["player"],
        facts=[ObservableFact.only("The new party enters.", ["player"])],
    )

    with pytest.raises(BackgroundThreadContractError, match="at least one"):
        validate_background_thread_selection(
            ckpt,
            result=focal,
            actor_id="player",
            candidate_ids=["renna", "mirelle"],
            require_when_offscreen=True,
        )

    focal.background_threads = [
        BackgroundThreadPick(
            actor_id="renna",
            participant_ids=["renna", "mirelle"],
        )
    ]
    requests = background_thread_requests(
        ckpt,
        result=focal,
        actor_id="player",
    )

    assert requests == [
        BackgroundThreadRequest(
            actor_id="renna",
            participant_ids=("renna", "mirelle"),
            canonical_at_s=0,
        )
    ]


def test_parallel_thread_selections_cannot_share_participants() -> None:
    ckpt = checkpoint(
        bindings={"player": "human"},
        player_character_id="player",
        characters=[
            character_record("player", is_playable=True),
            _offstage_actor("renna"),
            _offstage_actor("mirelle"),
        ],
    )
    focal = router_output(
        observer_ids=["player"],
        background_threads=[
            {"actor_id": "renna", "participant_ids": ["renna", "mirelle"]},
            {"actor_id": "mirelle", "participant_ids": ["mirelle"]},
        ],
    )

    with pytest.raises(BackgroundThreadContractError, match="share participant"):
        background_thread_requests(ckpt, result=focal, actor_id="player")


def test_background_result_must_close_inside_semantic_participants() -> None:
    ckpt = checkpoint(
        bindings={"player": "human"},
        characters=[
            character_record("player", location="stage", is_playable=True),
            _offstage_actor("cook", location="incorrect_old_label"),
            _offstage_actor("helper", location="different_label"),
        ],
    )
    request = BackgroundThreadRequest(
        actor_id="cook",
        participant_ids=("cook", "helper"),
        canonical_at_s=0,
    )
    result = _background_result("cook")
    anchor_background_thread_result(result, request=request)

    validate_background_thread_result(ckpt, request=request, result=result)

    result.location_updates = [
        LocationUpdateSignal(character_id="cook", location_label="stage")
    ]
    with pytest.raises(ValueError, match="location"):
        validate_background_thread_result(ckpt, request=request, result=result)

    result.location_updates = []
    result.observers[0].character_id = "player"
    with pytest.raises(ValueError, match="actor must observe"):
        validate_background_thread_result(ckpt, request=request, result=result)


class _ConcurrentBackgroundDispatcher:
    def __init__(
        self,
        *,
        invalid_actor_id: str = "",
        first_route_actor_id: str = "smith",
    ) -> None:
        self.invalid_actor_id = invalid_actor_id
        self.first_route_actor_id = first_route_actor_id
        self.narrator_started = asyncio.Event()
        self.all_agents_started = asyncio.Event()
        self.release_agents = asyncio.Event()
        self.first_actor_routed = asyncio.Event()
        self.active_agents: set[str] = set()
        self.agent_calls: list[dict] = []
        self.route_completion_order: list[str] = []
        self.narrator_event_ids: list[str] = []

    async def route_intention(self, **kwargs) -> EventRouterOutput:
        actor_id = kwargs["actor_id"]
        background_thread = kwargs.get("background_thread")
        if background_thread is None:
            return router_output(
                event_id="evt_foreground",
                event_kind="state_change",
                observer_ids=["player"],
                background_threads=[
                    {"actor_id": "cook", "participant_ids": ["cook"]},
                    {"actor_id": "smith", "participant_ids": ["smith"]},
                ],
                facts=[
                    ObservableFact.only(
                        "The player closes the ledger.",
                        ["player"],
                        visual_subject_ids=["player"],
                    )
                ],
            )

        if actor_id == self.first_route_actor_id:
            self.first_actor_routed.set()
        else:
            await self.first_actor_routed.wait()
        self.route_completion_order.append(actor_id)
        kwargs["ckpt"].session_conversation.append(
            ConversationMessage(
                role="assistant",
                content=f"background result for {actor_id}",
            )
        )
        result = _background_result(actor_id)
        if actor_id == self.invalid_actor_id:
            result.observers[0].character_id = "player"
        return result

    async def route_continuation(self, **_kwargs) -> EventRouterOutput:
        raise AssertionError("foreground continuation was not expected")

    async def agent_intend(self, **kwargs) -> str:
        actor_id = kwargs["character_id"]
        self.agent_calls.append(kwargs)
        self.active_agents.add(actor_id)
        await self.narrator_started.wait()
        if self.active_agents == {"cook", "smith"}:
            self.all_agents_started.set()
        await self.release_agents.wait()
        ckpt = kwargs["ckpt"]
        ckpt.character_conversations.setdefault(actor_id, []).extend([
            ConversationMessage(role="user", content="semantic opportunity"),
            ConversationMessage(role="assistant", content=f"{actor_id} acts"),
        ])
        return f"{actor_id} makes one concrete move."

    async def narrator_compose(self, **kwargs):
        self.narrator_event_ids = [
            entry.event_id for entry in kwargs["buffered_events"]
        ]
        self.narrator_started.set()
        await asyncio.wait_for(self.all_agents_started.wait(), timeout=1)
        self.release_agents.set()
        return (
            NarratorFinalOutput(
                handoff="render",
                handoff_reason="The visible action is complete.",
                final_text="The player closes the ledger.",
            ),
            TranscriptEntry(
                user=kwargs.get("user_input", ""),
                assistant="The player closes the ledger.",
            ),
        )


def _concurrent_checkpoint():
    return checkpoint(
        bindings={"player": "human"},
        player_character_id="player",
        characters=[
            character_record("player", location="stage", is_playable=True),
            _offstage_actor("cook", location="same_stale_label"),
            _offstage_actor("smith", location="same_stale_label"),
        ],
    )


def test_independent_threads_overlap_narration_and_merge_deterministically() -> None:
    async def _run():
        dispatcher = _ConcurrentBackgroundDispatcher()
        ckpt = _concurrent_checkpoint()
        result = await run_beat(
            ckpt=ckpt,
            dispatcher=dispatcher,
            actor_id="player",
            intention="I close the ledger.",
        )
        return ckpt, dispatcher, result

    ckpt, dispatcher, result = asyncio.run(_run())

    assert {call["character_id"] for call in dispatcher.agent_calls} == {
        "cook",
        "smith",
    }
    assert all(call["frame"] == "background" for call in dispatcher.agent_calls)
    assert all(
        "same_stale_label" not in call["local_context"]
        for call in dispatcher.agent_calls
    )
    assert dispatcher.route_completion_order == ["smith", "cook"]
    assert dispatcher.narrator_event_ids == ["evt_foreground"]
    assert [event.event_id for event in ckpt.canonical_events] == [
        "evt_foreground",
        "evt_background_cook",
        "evt_background_smith",
    ]
    assert result.events_closed == 3
    assert result.event_actor_ids == ["player", "cook", "smith"]
    assert ckpt.session.render_buffers["player"] == []
    assert len(ckpt.character_conversations["cook"]) == 2
    assert len(ckpt.character_conversations["smith"]) == 2


def test_background_failure_is_loud_and_merges_none_of_the_thread_set() -> None:
    async def _run():
        dispatcher = _ConcurrentBackgroundDispatcher(invalid_actor_id="smith")
        ckpt = _concurrent_checkpoint()
        with pytest.raises(ValueError, match="actor must observe"):
            await run_beat(
                ckpt=ckpt,
                dispatcher=dispatcher,
                actor_id="player",
                intention="I close the ledger.",
            )
        return ckpt

    ckpt = asyncio.run(_run())

    assert [event.event_id for event in ckpt.canonical_events] == [
        "evt_foreground"
    ]
    assert "cook" not in ckpt.character_conversations
    assert "smith" not in ckpt.character_conversations
    assert all(
        "background result" not in message.content
        for message in ckpt.session_conversation
    )


def test_merge_rejects_foreground_change_to_non_actor_participant() -> None:
    source = checkpoint(
        bindings={"player": "human"},
        player_character_id="player",
        characters=[
            character_record("player", is_playable=True),
            _offstage_actor("cook"),
            _offstage_actor("helper"),
        ],
    )
    request = BackgroundThreadRequest(
        actor_id="cook",
        participant_ids=("cook", "helper"),
        canonical_at_s=0,
    )
    prepared = _PreparedBackgroundThread(
        request=request,
        source_checkpoint=source.model_copy(deep=True),
        checkpoint=source.model_copy(deep=True),
        result=_background_result("cook"),
        router_history_suffix=[],
    )
    live = source.model_copy(deep=True)
    helper = next(
        character
        for character in live.characters
        if character.character_id == "helper"
    )
    helper.pending_observations.append(
        "A foreground event reaches the helper before merge."
    )

    with pytest.raises(RuntimeError, match="participant 'helper'"):
        _assert_background_thread_participants_are_unchanged(live, prepared)


class _PostDepartureBackgroundDispatcher:
    def __init__(self) -> None:
        self.route_calls: list[dict] = []
        self.renna_observations: list[str] = []
        self.renna_call: dict | None = None
        self.narrator_calls = 0

    async def route_intention(self, **kwargs) -> EventRouterOutput:
        actor_id = kwargs["actor_id"]
        background_thread = kwargs.get("background_thread")
        self.route_calls.append(kwargs)
        if background_thread is not None:
            assert actor_id in {"halcyon", "renna"}
            return _background_result(actor_id)
        if actor_id == "player":
            return router_output(
                event_id="evt_departure",
                event_kind="beat_continues",
                agent_ids=["expedition"],
                observer_ids=["player", "expedition", "renna", "mirelle"],
                facts=[
                    ObservableFact.all(
                        "The other party enters, leaving Renna and Mirelle behind.",
                        visual_subject_ids=["expedition", "renna", "mirelle"],
                    )
                ],
                background_threads=[{
                    "actor_id": "halcyon",
                    "participant_ids": ["halcyon"],
                }],
            )
        assert actor_id == "expedition"
        return router_output(
            event_id="evt_beyond_departure",
            event_kind="state_change",
            observer_ids=["player", "expedition"],
            background_threads=[{
                "actor_id": "renna",
                "participant_ids": ["renna", "mirelle"],
            }],
            facts=[
                ObservableFact.all(
                    "The expedition advances beyond the threshold.",
                    visual_subject_ids=["expedition"],
                )
            ],
        )

    async def route_continuation(self, **_kwargs) -> EventRouterOutput:
        raise AssertionError("continuation was not expected")

    async def agent_intend(self, **kwargs) -> str:
        actor_id = kwargs["character_id"]
        if actor_id == "expedition":
            return "I advance beyond the threshold."
        if actor_id == "halcyon":
            return "I close one distant bargain."
        assert actor_id == "renna"
        self.renna_call = kwargs
        actor = next(
            character
            for character in kwargs["ckpt"].characters
            if character.character_id == "renna"
        )
        self.renna_observations = list(actor.pending_observations)
        return "I ask Mirelle what she makes of what we survived."

    async def narrator_compose(self, **kwargs):
        self.narrator_calls += 1
        if self.narrator_calls == 1:
            return (
                NarratorFinalOutput(
                    handoff="continue",
                    handoff_reason="The selected expedition still has to move.",
                    final_text="Discarded candidate.",
                ),
                TranscriptEntry(
                    user=kwargs.get("user_input", ""),
                    assistant="Discarded candidate.",
                ),
            )
        return (
            NarratorFinalOutput(
                handoff="render",
                handoff_reason="The foreground move is complete.",
                final_text="The expedition advances.",
            ),
            TranscriptEntry(
                user=kwargs.get("user_input", ""),
                assistant="The expedition advances.",
            ),
        )


def test_selection_remains_enabled_until_post_departure_thread_can_act() -> None:
    async def _run():
        dispatcher = _PostDepartureBackgroundDispatcher()
        ckpt = checkpoint(
            bindings={"player": "human"},
            player_character_id="player",
            characters=[
                character_record("player", is_playable=True),
                _offstage_actor("expedition"),
                _offstage_actor("renna", location="stale_floor_one"),
                _offstage_actor("mirelle", location="lobby"),
                _offstage_actor("halcyon", location="distant_march"),
            ],
        )
        result = await run_beat(
            ckpt=ckpt,
            dispatcher=dispatcher,
            actor_id="player",
            intention="Send the other party through.",
        )
        return ckpt, dispatcher, result

    ckpt, dispatcher, result = asyncio.run(_run())

    foreground_calls = {
        call["actor_id"]: call
        for call in dispatcher.route_calls
        if call.get("background_thread") is None
    }
    assert foreground_calls["player"]["background_thread_selection"] is True
    assert foreground_calls["player"]["background_thread_excluded_ids"] == ()
    assert foreground_calls["player"]["background_thread_max_threads"] == 4
    assert foreground_calls["expedition"]["background_thread_selection"] is True
    assert foreground_calls["expedition"][
        "background_thread_excluded_ids"
    ] == ("halcyon",)
    assert foreground_calls["expedition"]["background_thread_max_threads"] == 3
    assert all(
        not call.get("background_thread_selection", False)
        for call in dispatcher.route_calls
        if call.get("background_thread") is not None
    )
    assert dispatcher.renna_observations == [
        "The other party enters, leaving Renna and Mirelle behind."
    ]
    assert dispatcher.renna_call is not None
    assert dispatcher.renna_call["frame"] == "background"
    assert dispatcher.renna_call["include_location"] is False
    assert "Mirelle" in dispatcher.renna_call["local_context"]
    assert "stale_floor_one" not in dispatcher.renna_call["local_context"]
    assert [event.event_id for event in ckpt.canonical_events] == [
        "evt_departure",
        "evt_background_halcyon",
        "evt_beyond_departure",
        "evt_background_renna",
    ]
    assert result.event_actor_ids == [
        "player",
        "halcyon",
        "expedition",
        "renna",
    ]


def test_checkpoint_is_identical_across_thread_completion_orders() -> None:
    async def _run(first_actor_id: str, source):
        dispatcher = _ConcurrentBackgroundDispatcher(
            first_route_actor_id=first_actor_id,
        )
        ckpt = source.model_copy(deep=True)
        result = await run_beat(
            ckpt=ckpt,
            dispatcher=dispatcher,
            actor_id="player",
            intention="I close the ledger.",
        )
        return ckpt, dispatcher, result

    async def _compare():
        source = _concurrent_checkpoint()
        first = await _run("cook", source)
        second = await _run("smith", source)
        return first, second

    (first_ckpt, first_dispatcher, first_result), (
        second_ckpt,
        second_dispatcher,
        second_result,
    ) = asyncio.run(_compare())

    assert first_dispatcher.route_completion_order == ["cook", "smith"]
    assert second_dispatcher.route_completion_order == ["smith", "cook"]
    assert first_result.event_actor_ids == second_result.event_actor_ids
    assert first_ckpt.model_dump_json() == second_ckpt.model_dump_json()
