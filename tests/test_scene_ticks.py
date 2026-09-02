"""Rules-neutral scheduling and merge contracts for unnarrated scenes."""

from __future__ import annotations

import asyncio

import pytest

from app.engine.scene_ticks import (
    anchor_scene_tick_result,
    discover_scene_tick_requests,
    validate_scene_tick_result,
)
from app.engine.turn_loop import run_beat
from app.schemas.characters import ActorRecord
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import EventRouterOutput, LocationUpdateSignal
from app.schemas.events import ObservableFact
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import DndCombatState, OpenCommitment
from tests.support.factories import character_record, checkpoint, router_output


def _offstage_actor(character_id: str, *, location: str):
    return character_record(
        character_id,
        location=location,
        actor=ActorRecord(may_act_offstage=True),
    )


def _scene_result(
    actor_id: str,
    *,
    event_id: str | None = None,
) -> EventRouterOutput:
    return router_output(
        event_id=event_id or f"evt_tick_{actor_id}",
        event_kind="state_change",
        observer_ids=[actor_id],
        facts=[
            ObservableFact.all(
                f"{actor_id} begins a local task.",
                visual_subject_ids=[actor_id],
            )
        ],
    )


def test_discovery_selects_one_fair_actor_per_unnarrated_scene() -> None:
    player = character_record("player", location="stage", is_playable=True)
    cook = _offstage_actor("cook", location="kitchen")
    helper = _offstage_actor("helper", location="kitchen")
    smith = _offstage_actor("smith", location="forge")
    onstage = _offstage_actor("onstage", location="stage")
    unplaced = _offstage_actor("unplaced", location="")
    ckpt = checkpoint(
        bindings={"player": "human"},
        player_character_id="player",
        characters=[player, cook, helper, smith, onstage, unplaced],
    )
    ckpt.character_conversations["cook"] = [
        ConversationMessage(role="user", content="older work"),
        ConversationMessage(role="assistant", content="done"),
    ]

    requests = discover_scene_tick_requests(
        ckpt,
        blocked_character_ids=set(),
        max_scenes=4,
    )

    assert [(request.location, request.actor_id) for request in requests] == [
        ("kitchen", "helper"),
        ("forge", "smith"),
    ]
    assert requests[0].participant_ids == ("cook", "helper")
    assert all(request.location != "stage" for request in requests)
    capped = discover_scene_tick_requests(
        ckpt,
        blocked_character_ids=set(),
        max_scenes=1,
    )
    assert [(request.location, request.actor_id) for request in capped] == [
        ("kitchen", "helper")
    ]


def test_discovery_excludes_committed_and_rules_active_work() -> None:
    player = character_record("player", location="stage", is_playable=True)
    cook = _offstage_actor("cook", location="kitchen")
    ckpt = checkpoint(
        bindings={"player": "human"},
        characters=[player, cook],
    )
    ckpt.session.open_commitments = [
        OpenCommitment(
            commitment_id="meal",
            actor_ids=["cook"],
            description="prepare the meal",
            location_label="kitchen",
        )
    ]

    assert discover_scene_tick_requests(
        ckpt,
        max_scenes=4,
    ) == []

    ckpt.session.open_commitments = []
    ckpt.session.active_combat = DndCombatState()
    assert discover_scene_tick_requests(
        ckpt,
        max_scenes=4,
    ) == []


def test_scene_result_must_close_inside_its_scene() -> None:
    player = character_record("player", location="stage", is_playable=True)
    cook = _offstage_actor("cook", location="kitchen")
    helper = _offstage_actor("helper", location="kitchen")
    ckpt = checkpoint(
        bindings={"player": "human"},
        characters=[player, cook, helper],
    )
    request = discover_scene_tick_requests(
        ckpt,
        max_scenes=1,
    )[0]
    result = _scene_result("cook")
    anchor_scene_tick_result(result, request=request)

    validate_scene_tick_result(ckpt, request=request, result=result)

    result.location_updates = [
        LocationUpdateSignal(character_id="cook", location_label="stage")
    ]
    with pytest.raises(ValueError, match="shared lifecycle or rules state"):
        validate_scene_tick_result(ckpt, request=request, result=result)

    result.location_updates = []
    result.observers[0].character_id = "player"
    with pytest.raises(ValueError, match="actor must observe"):
        validate_scene_tick_result(ckpt, request=request, result=result)


class _ConcurrentSceneDispatcher:
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
        self.agent_calls: list[str] = []
        self.route_completion_order: list[str] = []
        self.narrator_event_ids: list[str] = []

    async def route_intention(self, **kwargs) -> EventRouterOutput:
        actor_id = kwargs["actor_id"]
        scene_tick = kwargs.get("scene_tick")
        if scene_tick is None:
            return router_output(
                event_id="evt_foreground",
                event_kind="state_change",
                observer_ids=["player"],
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
                content=f"scene result for {actor_id}",
            )
        )
        result = _scene_result(actor_id)
        if actor_id == self.invalid_actor_id:
            result.observers[0].character_id = "player"
        return result

    async def route_continuation(self, **_kwargs) -> EventRouterOutput:
        raise AssertionError("foreground continuation was not expected")

    async def agent_intend(self, **kwargs) -> str:
        actor_id = kwargs["character_id"]
        self.agent_calls.append(actor_id)
        self.active_agents.add(actor_id)
        await self.narrator_started.wait()
        if self.active_agents == {"cook", "smith"}:
            self.all_agents_started.set()
        await self.release_agents.wait()
        ckpt = kwargs["ckpt"]
        ckpt.character_conversations.setdefault(actor_id, []).extend([
            ConversationMessage(role="user", content="local opportunity"),
            ConversationMessage(role="assistant", content=f"{actor_id} acts"),
        ])
        return f"{actor_id} begins a local task."

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
            _offstage_actor("cook", location="kitchen"),
            _offstage_actor("smith", location="forge"),
        ],
    )


def test_independent_scenes_overlap_narration_and_merge_deterministically() -> None:
    async def _run():
        dispatcher = _ConcurrentSceneDispatcher()
        ckpt = _concurrent_checkpoint()
        result = await run_beat(
            ckpt=ckpt,
            dispatcher=dispatcher,
            actor_id="player",
            intention="I close the ledger.",
        )
        return ckpt, dispatcher, result

    ckpt, dispatcher, result = asyncio.run(_run())

    assert set(dispatcher.agent_calls) == {"cook", "smith"}
    assert dispatcher.route_completion_order == ["smith", "cook"]
    assert dispatcher.narrator_event_ids == ["evt_foreground"]
    assert [event.event_id for event in ckpt.canonical_events] == [
        "evt_foreground",
        "evt_tick_cook",
        "evt_tick_smith",
    ]
    assert result.events_closed == 3
    assert result.event_actor_ids == ["player", "cook", "smith"]
    assert ckpt.session.render_buffers["player"] == []
    assert len(ckpt.character_conversations["cook"]) == 2
    assert len(ckpt.character_conversations["smith"]) == 2


def test_scene_tick_failure_is_loud_and_merges_none_of_the_tick_set() -> None:
    async def _run():
        dispatcher = _ConcurrentSceneDispatcher(invalid_actor_id="smith")
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
        "scene result" not in message.content
        for message in ckpt.session_conversation
    )


def test_scene_tick_checkpoint_is_identical_across_completion_orders() -> None:
    async def _run(first_actor_id: str, source):
        dispatcher = _ConcurrentSceneDispatcher(
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
