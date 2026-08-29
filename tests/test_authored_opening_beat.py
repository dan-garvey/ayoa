from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.engine.orchestrator as orchestrator_module
from app.bot.engine_bridge import EngineBridge
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.one_star_adapter import one_star_opening_roster_preview
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.schemas.characters import CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.events import ObservableFact, visible_fact_texts
from app.schemas.one_star import OneStarEventRouterOutput
from app.schemas.requests import TurnRequest
from app.schemas.state import (
    AuthoredOpeningCharacterBeat,
    AuthoredOpeningDialogueSegment,
    OpeningPolicy,
    WorldState,
)
from tests.support.factories import (
    ClassFakeDispatcher,
    character_record,
    checkpoint,
    llm_response,
    router_output,
)


HERO_BRIEFING = "You arrived together. The danger beyond the gate is real."
PLAYER_BRIEFING = "Patron, arrange this group and send them through the gate."
PRIVATE_INTENT = (
    "I have delivered the rules. The Patron has one concrete action, no "
    "question queue is open, and obstruction will meet an immediate "
    "physical consequence."
)


class OneStarOpeningRouterDispatcher(LLMDispatcher, ClassFakeDispatcher):
    """Use the production router path with deterministic narrator/agent fakes."""

    def __init__(self, client, prompt_mgr):
        LLMDispatcher.__init__(self, client, prompt_mgr)

    async def agent_intend(self, **kwargs):
        return await ClassFakeDispatcher.agent_intend(self, **kwargs)

    async def narrator_compose(self, **kwargs):
        return await ClassFakeDispatcher.narrator_compose(self, **kwargs)


@pytest.fixture(autouse=True)
def _reset_fake_dispatcher() -> None:
    ClassFakeDispatcher.reset()
    OneStarOpeningRouterDispatcher.reset()
    yield
    ClassFakeDispatcher.reset()
    OneStarOpeningRouterDispatcher.reset()


def _opening_checkpoint(
    *,
    presentation_mode: str = "prose",
    speaker_presentation: str = "voice_only",
    required_participant_ids: list[str] | None = None,
):
    source = checkpoint(
        session_id="authored-opening",
        bindings={"patron": "user-1"},
        characters=[
            character_record(
                "patron",
                name="Patron",
                is_playable=True,
                location="hall",
            ),
            character_record("guide", name="Guide", location="hall"),
            character_record(
                "arrival_a",
                name="Ari",
                status=CharacterStatus.dormant,
                location="not_yet_fictional",
            ),
            character_record(
                "arrival_b",
                name="Bram",
                status=CharacterStatus.dormant,
                location="not_yet_fictional",
            ),
        ],
        world_state=WorldState(
            opening=OpeningPolicy(
                context="Introduce the two selected arrivals in the hall.",
                authored_character_beat=AuthoredOpeningCharacterBeat(
                    speaker_character_id="guide",
                    required_participant_ids=(
                        ["arrival_a", "arrival_b"]
                        if required_participant_ids is None
                        else required_participant_ids
                    ),
                    introduced_character_count=2,
                    segments=[
                        AuthoredOpeningDialogueSegment(
                            audiences=["introduced_characters"],
                            speaker_presentation=speaker_presentation,
                            text=HERO_BRIEFING,
                        ),
                        AuthoredOpeningDialogueSegment(
                            audiences=[
                                "opening_players",
                                "introduced_characters",
                            ],
                            speaker_presentation=speaker_presentation,
                            text=PLAYER_BRIEFING,
                        ),
                    ],
                    private_intent=PRIVATE_INTENT,
                ),
            )
        ),
    )
    source.session.config.settings.presentation_mode = presentation_mode
    return source


def _opening_router_output():
    return router_output(
        event_id="evt_arrival",
        event_kind="state_change",
        facts=[ObservableFact.all("Two figures take shape in the hall.")],
        observer_ids=["patron", "guide", "arrival_a", "arrival_b"],
        activate=[
            {"character_id": "arrival_a", "location_label": "hall"},
            {"character_id": "arrival_b", "location_label": "hall"},
        ],
    )


def _one_star_story_opening_source(*, session_id: str) -> CheckpointFile:
    checkpoint_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "storage"
        / "stories"
        / "one_star_ascension_s1"
        / "ckpt_0000.json"
    )
    source = CheckpointFile.model_validate_json(checkpoint_path.read_text())
    source.session.session_id = session_id
    source.session.character_bindings = {"the_master": "user-1"}
    source.session.player_character_id = ""
    return source


def _one_star_live_shaped_opening_output(
    source: CheckpointFile,
    *,
    event_id: str,
    guide_delivery: bool,
    guide_handoff: bool,
) -> OneStarEventRouterOutput:
    introduced_ids = [
        draw.existing_character_id
        for draw in one_star_opening_roster_preview(
            source,
            "master_opening_roster",
        )
    ]
    facts = [
        ObservableFact.all(
            "Cold light separates into three embodied arrivals."
        )
    ]
    if guide_delivery:
        facts.append(ObservableFact.only(
            "The starter summon transaction completes.",
            ["the_master", "iselle_the_guide"],
        ))
    base_event = router_output(
        event_id=event_id,
        event_kind=("response_requested" if guide_handoff else "state_change"),
        facts=facts,
        observer_ids=[
            "the_master",
            "iselle_the_guide",
            *introduced_ids,
        ],
        agent_ids=["iselle_the_guide"] if guide_handoff else [],
    )
    return OneStarEventRouterOutput(
        **base_event.model_dump(mode="json"),
        state_updates=[{
            "kind": "summon",
            "target_id": "master_opening_roster",
            "value": "3",
            "details": [],
        }],
    )


def _orchestrator(
    monkeypatch,
    tmp_path,
    source,
    *,
    image_generation=None,
    dispatcher_cls=ClassFakeDispatcher,
    prompt_mgr=None,
    client=None,
):
    monkeypatch.setattr(
        "app.engine.orchestrator.LLMDispatcher",
        dispatcher_cls,
    )
    manager = CheckpointManager(str(tmp_path))
    manager.save(source)
    client = client or MagicMock()
    client.config = MagicMock()
    return Orchestrator(
        client,
        manager,
        prompt_mgr or MagicMock(),
        image_generation=image_generation,
    ), manager


@pytest.mark.asyncio
async def test_prose_authored_opening_has_text_without_visual_novel_render(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(_opening_router_output())

    response = await orchestrator.process_turn(TurnRequest(
        session_id=source.session.session_id,
        user_input="(begin)",
        acting_character_id="patron",
    ))

    assert response.output_text == f"POV_RENDER Guide: {PLAYER_BRIEFING}"
    assert HERO_BRIEFING not in response.per_player_renders["patron"]
    assert response.per_player_visual_novel_renders == {}
    assert len(ClassFakeDispatcher.narrator_calls) == 1
    assert [
        item.event_id
        for item in ClassFakeDispatcher.narrator_calls[0]["buffered_events"]
    ] == ["evt_arrival"]
    assert ClassFakeDispatcher.narrator_calls[0]["handoff_policy"] == "forced"

    saved = manager.load_latest(source.session.session_id)
    assert len(saved.canonical_events) == 2
    authored_event = saved.canonical_events[-1]
    assert authored_event.event_kind == "public_fact"
    assert all(
        fact.visual_subject_ids == []
        for fact in authored_event.canonical_event.observable_facts
    )
    assert visible_fact_texts(
        authored_event.canonical_event.observable_facts,
        "patron",
    ) == [f"Guide: {PLAYER_BRIEFING}"]
    assert visible_fact_texts(
        authored_event.canonical_event.observable_facts,
        "arrival_a",
    ) == [
        f"Guide: {HERO_BRIEFING}",
        f"Guide: {PLAYER_BRIEFING}",
    ]
    assert saved.session.render_buffers.get("patron", []) == []
    assert all(
        next(
            character
            for character in saved.characters
            if character.character_id == character_id
        ).status == CharacterStatus.active
        for character_id in ("arrival_a", "arrival_b")
    )

    history = saved.character_conversations["guide"]
    assert [message.role for message in history] == ["user", "assistant"]
    assert HERO_BRIEFING in history[-1].content
    assert PLAYER_BRIEFING in history[-1].content
    assert history[-1].content.endswith(f"({PRIVATE_INTENT})")
    authored_history = [
        message.content
        for message in saved.session_conversation
        if message.role == "assistant"
        and isinstance(message.content, str)
        and authored_event.event_id in message.content
    ]
    assert len(authored_history) == 1
    history_payload = json.loads(
        saved.narrator_conversations["patron"][-1].content[0]["text"]
    )
    assert history_payload == {
        "final_text": f"Guide: {PLAYER_BRIEFING}"
    }
    bridge = EngineBridge.__new__(EngineBridge)
    bridge.checkpoint_mgr = manager
    turn_history = bridge.turn_history(
        source.session.session_id,
        "patron",
    )
    assert turn_history[-1].turn_index == 1
    assert turn_history[-1].entry.user == ""
    assert turn_history[-1].entry.assistant == f"Guide: {PLAYER_BRIEFING}"


@pytest.mark.asyncio
async def test_voice_only_authored_opening_builds_exact_vn_accessibility_pages(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint(presentation_mode="visual_novel")
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(_opening_router_output())

    response = await orchestrator.process_turn(TurnRequest(
        session_id=source.session.session_id,
        user_input="(begin)",
        acting_character_id="patron",
    ))

    assert response.per_player_renders["patron"].endswith(
        f"Guide: {PLAYER_BRIEFING}"
    )
    assert HERO_BRIEFING not in response.per_player_renders["patron"]
    authored_segment = response.per_player_visual_novel_renders[
        "patron"
    ].segments[-1]
    assert authored_segment.rendered_event_id == "evt_arrival"
    assert [page.model_dump() for page in authored_segment.pages] == [{
        "kind": "dialogue",
        "speaker": "Guide",
        "text": PLAYER_BRIEFING,
        "sprites": [],
    }]
    assert authored_segment.sprite_variant_keys_by_label == {}

    saved = manager.load_latest(source.session.session_id)
    authored_event = saved.canonical_events[-1]
    assert all(
        fact.visual_subject_ids == []
        for fact in authored_event.canonical_event.observable_facts
    )
    history_payload = json.loads(
        saved.narrator_conversations["patron"][-1].content[0]["text"]
    )
    assert history_payload == {
        "pages": [{
            "kind": "dialogue",
            "speaker": "Guide",
            "text": PLAYER_BRIEFING,
        }]
    }


@pytest.mark.asyncio
async def test_visible_authored_opening_adds_only_explicit_speaker_sprite(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint(
        presentation_mode="visual_novel",
        speaker_presentation="visible",
    )
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(_opening_router_output())

    response = await orchestrator.process_turn(TurnRequest(
        session_id=source.session.session_id,
        user_input="(begin)",
        acting_character_id="patron",
    ))

    authored_segment = response.per_player_visual_novel_renders[
        "patron"
    ].segments[-1]
    assert [page.sprites for page in authored_segment.pages] == [["Guide"]]
    assert authored_segment.sprite_variant_keys_by_label == {
        "Guide": "neutral"
    }
    authored_event = manager.load_latest(
        source.session.session_id
    ).canonical_events[-1]
    assert [
        fact.visual_subject_ids
        for fact in authored_event.canonical_event.observable_facts
    ] == [["guide"], ["guide"]]


@pytest.mark.asyncio
async def test_authored_opening_rejects_a_human_controlled_speaker_atomically(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    source.session.character_bindings = {
        "patron": "user-1",
        "guide": "user-2",
    }
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(_opening_router_output())

    with pytest.raises(ValueError, match="player-controlled character"):
        await orchestrator.process_turn(TurnRequest(
            session_id=source.session.session_id,
            user_input="(begin)",
            acting_character_id="patron",
        ))

    saved = manager.load_latest(source.session.session_id)
    assert saved.session.turn_index == 0
    assert saved.canonical_events == []
    assert saved.character_conversations.get("guide", []) == []
    assert {
        character.character_id: character.status
        for character in saved.characters
        if character.character_id.startswith("arrival_")
    } == {
        "arrival_a": CharacterStatus.dormant,
        "arrival_b": CharacterStatus.dormant,
    }
    assert orchestrator.spawn_authoring._rosters == {}
    assert orchestrator.spawn_authoring._tasks == {}


@pytest.mark.asyncio
async def test_authored_opening_player_observers_are_deterministically_sorted(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    source.characters.extend([
        character_record("zulu", name="Zulu", location="hall"),
        character_record("alpha", name="Alpha", location="hall"),
    ])
    source.session.character_bindings = {
        "zulu": "user-3",
        "patron": "user-1",
        "alpha": "user-2",
    }
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(router_output(
        event_id="evt_arrival",
        event_kind="state_change",
        facts=[ObservableFact.all("Two figures take shape in the hall.")],
        observer_ids=[
            "zulu",
            "patron",
            "alpha",
            "guide",
            "arrival_a",
            "arrival_b",
        ],
        activate=[
            {"character_id": "arrival_a", "location_label": "hall"},
            {"character_id": "arrival_b", "location_label": "hall"},
        ],
    ))

    await orchestrator.process_turn(TurnRequest(
        session_id=source.session.session_id,
        user_input="(begin)",
        acting_character_id="patron",
    ))

    authored_event = manager.load_latest(
        source.session.session_id
    ).canonical_events[-1]
    assert [
        observer.character_id for observer in authored_event.observers[-3:]
    ] == ["alpha", "patron", "zulu"]


@pytest.mark.asyncio
async def test_authored_opening_beat_is_added_once_after_render_retry(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(_opening_router_output())
    ClassFakeDispatcher.queue_narrator_error(RuntimeError("narrator unavailable"))

    with pytest.raises(RuntimeError, match="narrator unavailable"):
        await orchestrator.process_turn(TurnRequest(
            session_id=source.session.session_id,
            user_input="(begin)",
            acting_character_id="patron",
        ))

    pending = manager.load_latest(source.session.session_id)
    assert pending.session.pending_narrator_render is not None
    assert [event.event_id for event in pending.canonical_events] == [
        "evt_arrival"
    ]
    assert pending.character_conversations.get("guide", []) == []

    response = await orchestrator.retry_pending_narrator_render(
        source.session.session_id
    )

    assert response.output_text.count(PLAYER_BRIEFING) == 1
    saved = manager.load_latest(source.session.session_id)
    authored_events = [
        event
        for event in saved.canonical_events
        if event.event_id.startswith("evt_authored_opening_")
    ]
    assert len(authored_events) == 1
    assert len(saved.character_conversations["guide"]) == 2
    assert len(ClassFakeDispatcher.route_calls) == 1
    assert len(ClassFakeDispatcher.narrator_calls) == 2
    assert saved.session.pending_narrator_render is None


@pytest.mark.asyncio
async def test_matching_authored_opening_rejects_improvised_next_output(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(router_output(
        event_id="evt_bad_handoff",
        event_kind="response_requested",
        facts=[ObservableFact.all("Two figures take shape in the hall.")],
        observer_ids=["patron", "guide", "arrival_a", "arrival_b"],
        agent_ids=["guide"],
        activate=[
            {"character_id": "arrival_a", "location_label": "hall"},
            {"character_id": "arrival_b", "location_label": "hall"},
        ],
    ))

    with pytest.raises(
        ValueError,
        match="authored opening arrival cannot request responders or next_output",
    ):
        await orchestrator.process_turn(TurnRequest(
            session_id=source.session.session_id,
            user_input="(begin)",
            acting_character_id="patron",
        ))

    assert ClassFakeDispatcher.agent_calls == []
    saved = manager.load_latest(source.session.session_id)
    assert saved.canonical_events == []
    assert saved.character_conversations.get("guide", []) == []
    assert orchestrator.spawn_authoring._rosters == {}
    assert orchestrator.spawn_authoring._tasks == {}


@pytest.mark.asyncio
async def test_one_star_adapter_lifecycle_is_guarded_before_opening_handoff(
    monkeypatch,
    tmp_path,
) -> None:
    class OneStarOpeningFakeDispatcher(
        ClassFakeDispatcher,
        LLMDispatcher,
    ):
        def __init__(self, client, prompt_mgr):
            LLMDispatcher.__init__(self, client, prompt_mgr)

    OneStarOpeningFakeDispatcher.reset()
    checkpoint_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "storage"
        / "stories"
        / "one_star_ascension_s1"
        / "ckpt_0000.json"
    )
    source = CheckpointFile.model_validate_json(checkpoint_path.read_text())
    source.session.session_id = "authored-one-star-opening-guard"
    source.session.character_bindings = {"the_master": "user-1"}
    source.session.player_character_id = ""
    draws = one_star_opening_roster_preview(
        source,
        "master_opening_roster",
    )
    introduced_ids = [draw.existing_character_id for draw in draws]
    base_event = router_output(
        event_id="evt_master_opening_bad_handoff",
        event_kind="response_requested",
        facts=[
            ObservableFact.all(
                "Cold light separates into three embodied arrivals."
            ),
            ObservableFact.only(
                "The starter summon transaction completes.",
                ["the_master", "iselle_the_guide"],
            ),
        ],
        observer_ids=[
            "the_master",
            "iselle_the_guide",
            *introduced_ids,
        ],
        agent_ids=["iselle_the_guide"],
    )
    routed = OneStarEventRouterOutput(
        **base_event.model_dump(mode="json"),
        state_updates=[{
            "kind": "summon",
            "target_id": "master_opening_roster",
            "value": "3",
            "details": [],
        }],
    )
    assert routed.activate == []

    orchestrator, manager = _orchestrator(
        monkeypatch,
        tmp_path,
        source,
        dispatcher_cls=OneStarOpeningFakeDispatcher,
    )
    OneStarOpeningFakeDispatcher.queue_route(routed)

    with pytest.raises(
        ValueError,
        match="authored opening arrival cannot request responders or next_output",
    ):
        await orchestrator.process_turn(TurnRequest(
            session_id=source.session.session_id,
            user_input="(begin)",
            acting_character_id="the_master",
        ))

    assert {
        signal.character_id for signal in routed.activate
    } == set(introduced_ids)
    assert OneStarOpeningFakeDispatcher.agent_calls == []
    saved = manager.load_latest(source.session.session_id)
    assert saved.canonical_events == []
    assert all(
        character.status == CharacterStatus.dormant
        for character in saved.characters
        if character.character_id in introduced_ids
    )
    assert orchestrator.spawn_authoring._rosters == {}
    assert orchestrator.spawn_authoring._tasks == {}


@pytest.mark.asyncio
async def test_live_shaped_one_star_opening_gets_one_full_envelope_correction(
    monkeypatch,
    tmp_path,
) -> None:
    source = _one_star_story_opening_source(
        session_id="authored-one-star-opening-correction",
    )
    introduced_ids = [
        draw.existing_character_id
        for draw in one_star_opening_roster_preview(
            source,
            "master_opening_roster",
        )
    ]
    invalid = _one_star_live_shaped_opening_output(
        source,
        event_id="evt_invalid_master_opening",
        guide_delivery=False,
        guide_handoff=True,
    )
    corrected = _one_star_live_shaped_opening_output(
        source,
        event_id="evt_corrected_master_opening",
        guide_delivery=True,
        guide_handoff=False,
    )
    assert invalid.activate == corrected.activate == []

    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        llm_response(invalid),
        llm_response(corrected),
    ])
    orchestrator, manager = _orchestrator(
        monkeypatch,
        tmp_path,
        source,
        dispatcher_cls=OneStarOpeningRouterDispatcher,
        prompt_mgr=PromptManager("app/prompts"),
        client=client,
    )

    response = await orchestrator.process_turn(TurnRequest(
        session_id=source.session.session_id,
        user_input="(begin)",
        acting_character_id="the_master",
    ))

    assert client.complete.await_count == 2
    assert [
        call.kwargs["response_model"]
        for call in client.complete.await_args_list
    ] == [OneStarEventRouterOutput, OneStarEventRouterOutput]
    correction = client.complete.await_args_list[1].kwargs["messages"][-1][
        "content"
    ]
    assert "configured guide" in correction
    assert "authored opening arrival cannot request responders" in correction
    authored = source.world_state.opening.authored_character_beat
    assert authored is not None
    player_segment = next(
        segment.text
        for segment in authored.segments
        if "opening_players" in segment.audiences
    )
    assert player_segment in response.output_text
    assert OneStarOpeningRouterDispatcher.agent_calls == []

    saved = manager.load_latest(source.session.session_id)
    assert len(saved.canonical_events) == 2
    assert saved.canonical_events[0].event_id == "evt_corrected_master_opening"
    assert saved.canonical_events[1].event_id.startswith(
        "evt_authored_opening_"
    )
    assert all(
        next(
            character
            for character in saved.characters
            if character.character_id == character_id
        ).status == CharacterStatus.active
        for character_id in introduced_ids
    )
    assert len(saved.character_conversations["iselle_the_guide"]) == 2


@pytest.mark.asyncio
async def test_invalid_corrected_one_star_opening_rolls_back_without_briefing(
    monkeypatch,
    tmp_path,
) -> None:
    source = _one_star_story_opening_source(
        session_id="authored-one-star-opening-invalid-correction",
    )
    introduced_ids = [
        draw.existing_character_id
        for draw in one_star_opening_roster_preview(
            source,
            "master_opening_roster",
        )
    ]
    invalid = _one_star_live_shaped_opening_output(
        source,
        event_id="evt_invalid_master_opening",
        guide_delivery=False,
        guide_handoff=True,
    )
    still_invalid = _one_star_live_shaped_opening_output(
        source,
        event_id="evt_still_invalid_master_opening",
        guide_delivery=True,
        guide_handoff=True,
    )
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        llm_response(invalid),
        llm_response(still_invalid),
    ])
    orchestrator, manager = _orchestrator(
        monkeypatch,
        tmp_path,
        source,
        dispatcher_cls=OneStarOpeningRouterDispatcher,
        prompt_mgr=PromptManager("app/prompts"),
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="remained invalid after one correction",
    ):
        await orchestrator.process_turn(TurnRequest(
            session_id=source.session.session_id,
            user_input="(begin)",
            acting_character_id="the_master",
        ))

    assert client.complete.await_count == 2
    assert all(
        call.kwargs["response_model"] is OneStarEventRouterOutput
        for call in client.complete.await_args_list
    )
    assert OneStarOpeningRouterDispatcher.agent_calls == []
    saved = manager.load_latest(source.session.session_id)
    assert saved.session.turn_index == 0
    assert saved.canonical_events == []
    assert saved.session_conversation == []
    assert saved.character_conversations.get("iselle_the_guide", []) == []
    assert all(
        character.status == CharacterStatus.dormant
        for character in saved.characters
        if character.character_id in introduced_ids
    )
    assert orchestrator.spawn_authoring._rosters == {}
    assert orchestrator.spawn_authoring._tasks == {}


@pytest.mark.asyncio
async def test_authored_opening_beat_skips_an_unrelated_opening_branch(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    source.characters.append(character_record(
        "newcomer",
        name="Newcomer",
        status=CharacterStatus.dormant,
        location="not_yet_fictional",
        is_playable=True,
    ))
    source.session.character_bindings = {"newcomer": "user-2"}
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(router_output(
        event_id="evt_newcomer_arrival",
        event_kind="response_requested",
        facts=[ObservableFact.all("One newcomer arrives alone.")],
        observer_ids=["newcomer", "guide"],
        agent_ids=["guide"],
        activate=[
            {"character_id": "newcomer", "location_label": "hall"},
        ],
    ))
    ClassFakeDispatcher.queue_agent("I give the newcomer a brisk orientation.")
    ClassFakeDispatcher.queue_route(router_output(
        event_id="evt_newcomer_orientation",
        event_kind="public_fact",
        facts=[ObservableFact.all("Guide gives the newcomer a short briefing.")],
        observer_ids=["newcomer", "guide"],
    ))

    response = await orchestrator.process_turn(TurnRequest(
        session_id=source.session.session_id,
        user_input="(begin)",
        acting_character_id="newcomer",
    ))

    assert response.output_text == "POV_RENDER"
    assert response.per_player_visual_novel_renders == {}
    saved = manager.load_latest(source.session.session_id)
    assert [event.event_id for event in saved.canonical_events] == [
        "evt_newcomer_arrival",
        "evt_newcomer_orientation",
    ]
    assert saved.character_conversations.get("guide", []) == []
    assert len(ClassFakeDispatcher.agent_calls) == 1
    assert len(ClassFakeDispatcher.route_calls) == 2


@pytest.mark.asyncio
async def test_partial_authored_opening_branch_fails_without_committing(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(router_output(
        event_id="evt_partial_arrival",
        event_kind="state_change",
        facts=[ObservableFact.all("Only one expected arrival takes shape.")],
        observer_ids=["patron", "guide", "arrival_a"],
        activate=[
            {"character_id": "arrival_a", "location_label": "hall"},
        ],
    ))

    with pytest.raises(
        ValueError,
        match="missing required participants: arrival_b",
    ):
        await orchestrator.process_turn(TurnRequest(
            session_id=source.session.session_id,
            user_input="(begin)",
            acting_character_id="patron",
        ))

    saved = manager.load_latest(source.session.session_id)
    assert saved.canonical_events == []
    assert saved.character_conversations.get("guide", []) == []


@pytest.mark.asyncio
async def test_empty_required_ids_remain_idempotent_on_repeated_begin(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint(required_participant_ids=[])
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(_opening_router_output())

    await orchestrator.process_turn(TurnRequest(
        session_id=source.session.session_id,
        user_input="(begin)",
        acting_character_id="patron",
    ))

    ClassFakeDispatcher.queue_route(router_output(
        event_id="evt_repeated_begin",
        event_kind="public_fact",
        facts=[ObservableFact.all("The already-open hall remains unchanged.")],
        observer_ids=["patron", "guide"],
    ))
    await orchestrator.process_turn(TurnRequest(
        session_id=source.session.session_id,
        user_input="(begin)",
        acting_character_id="patron",
    ))

    saved = manager.load_latest(source.session.session_id)
    assert len([
        event
        for event in saved.canonical_events
        if event.event_id.startswith("evt_authored_opening_")
    ]) == 1
    assert len(saved.character_conversations["guide"]) == 2
    assert orchestrator.spawn_authoring._rosters == {}
    assert orchestrator.spawn_authoring._tasks == {}


@pytest.mark.asyncio
async def test_authored_commit_failure_rolls_back_normal_begin_and_runtime(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(_opening_router_output())
    original_commit = (
        orchestrator_module._commit_authored_opening_character_beat
    )

    def _fail_after_authored_mutation(*args, **kwargs):
        result = original_commit(*args, **kwargs)
        assert result is not None
        raise RuntimeError("authored commit failed after mutation")

    monkeypatch.setattr(
        orchestrator_module,
        "_commit_authored_opening_character_beat",
        _fail_after_authored_mutation,
    )

    with pytest.raises(
        RuntimeError,
        match="authored commit failed after mutation",
    ):
        await orchestrator.process_turn(TurnRequest(
            session_id=source.session.session_id,
            user_input="(begin)",
            acting_character_id="patron",
        ))

    saved = manager.load_latest(source.session.session_id)
    assert saved.session.turn_index == 0
    assert saved.session.pending_narrator_render is None
    assert saved.canonical_events == []
    assert saved.session_conversation == []
    assert saved.character_conversations.get("guide", []) == []
    assert all(
        character.status == CharacterStatus.dormant
        for character in saved.characters
        if character.character_id.startswith("arrival_")
    )
    assert orchestrator.spawn_authoring._rosters == {}
    assert orchestrator.spawn_authoring._tasks == {}


@pytest.mark.asyncio
async def test_authored_commit_failure_restores_pending_retry_and_can_retry_once(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    orchestrator, manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(_opening_router_output())
    ClassFakeDispatcher.queue_narrator_error(RuntimeError("narrator unavailable"))

    with pytest.raises(RuntimeError, match="narrator unavailable"):
        await orchestrator.process_turn(TurnRequest(
            session_id=source.session.session_id,
            user_input="(begin)",
            acting_character_id="patron",
        ))

    original_commit = (
        orchestrator_module._commit_authored_opening_character_beat
    )
    with monkeypatch.context() as failure_patch:
        def _fail_after_authored_mutation(*args, **kwargs):
            result = original_commit(*args, **kwargs)
            assert result is not None
            raise RuntimeError("retry authored commit failed")

        failure_patch.setattr(
            orchestrator_module,
            "_commit_authored_opening_character_beat",
            _fail_after_authored_mutation,
        )
        with pytest.raises(RuntimeError, match="retry authored commit failed"):
            await orchestrator.retry_pending_narrator_render(
                source.session.session_id
            )

    pending = manager.load_latest(source.session.session_id)
    assert pending.session.pending_narrator_render is not None
    assert [event.event_id for event in pending.canonical_events] == [
        "evt_arrival"
    ]
    assert pending.character_conversations.get("guide", []) == []
    assert all(
        character.status == CharacterStatus.dormant
        for character in pending.characters
        if character.character_id.startswith("arrival_")
    )
    assert orchestrator.spawn_authoring._rosters == {}
    assert orchestrator.spawn_authoring._tasks == {}

    response = await orchestrator.retry_pending_narrator_render(
        source.session.session_id
    )
    assert response.output_text.count(PLAYER_BRIEFING) == 1
    saved = manager.load_latest(source.session.session_id)
    assert saved.session.pending_narrator_render is None
    assert len([
        event
        for event in saved.canonical_events
        if event.event_id.startswith("evt_authored_opening_")
    ]) == 1
    assert len(saved.character_conversations["guide"]) == 2
    assert len(ClassFakeDispatcher.route_calls) == 1
    assert len(ClassFakeDispatcher.narrator_calls) == 3


@pytest.mark.asyncio
async def test_terminal_identity_retirement_waits_for_authored_validation(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    guide = next(
        character
        for character in source.characters
        if character.character_id == "guide"
    )
    assert guide.visuals.identity_reference_id == ""
    image_generation = MagicMock()
    image_generation.sync_visual_novel_character_presentations = AsyncMock(
        return_value=False
    )
    orchestrator, manager = _orchestrator(
        monkeypatch,
        tmp_path,
        source,
        image_generation=image_generation,
    )
    ClassFakeDispatcher.queue_route(router_output(
        event_id="evt_invalid_arrival",
        event_kind="state_change",
        facts=[ObservableFact.all("The guide vanishes as two figures arrive.")],
        observer_ids=["patron", "guide", "arrival_a", "arrival_b"],
        activate=[
            {"character_id": "arrival_a", "location_label": "hall"},
            {"character_id": "arrival_b", "location_label": "hall"},
        ],
        cull=["guide"],
    ))

    with pytest.raises(ValueError, match="speaker must be an active character"):
        await orchestrator.process_turn(TurnRequest(
            session_id=source.session.session_id,
            user_input="(begin)",
            acting_character_id="patron",
        ))

    image_generation.retire_character_identity.assert_not_called()
    saved_guide = next(
        character
        for character in manager.load_latest(source.session.session_id).characters
        if character.character_id == "guide"
    )
    assert saved_guide.status == CharacterStatus.active
    assert saved_guide.visuals.identity_reference_id == ""
    assert orchestrator.spawn_authoring._rosters == {}
    assert orchestrator.spawn_authoring._tasks == {}


@pytest.mark.asyncio
async def test_authored_validation_precedes_single_roster_acceptance(
    monkeypatch,
    tmp_path,
) -> None:
    source = _opening_checkpoint()
    orchestrator, _manager = _orchestrator(monkeypatch, tmp_path, source)
    ClassFakeDispatcher.queue_route(_opening_router_output())
    order: list[str] = []
    original_commit = (
        orchestrator_module._commit_authored_opening_character_beat
    )
    original_accept = orchestrator.spawn_authoring.accept_roster

    def _record_commit(*args, **kwargs):
        order.append("authored")
        return original_commit(*args, **kwargs)

    def _record_accept(*args, **kwargs):
        order.append("accept")
        return original_accept(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator_module,
        "_commit_authored_opening_character_beat",
        _record_commit,
    )
    monkeypatch.setattr(
        orchestrator.spawn_authoring,
        "accept_roster",
        _record_accept,
    )

    await orchestrator.process_turn(TurnRequest(
        session_id=source.session.session_id,
        user_input="(begin)",
        acting_character_id="patron",
    ))

    assert order == ["authored", "accept"]
