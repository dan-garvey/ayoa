from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.checkpoint_manager import CheckpointManager
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.engine.spawn_authoring import SpawnAuthoringCoordinator
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.llm.client import LLMClient
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, SpawnRequest, SpawnSeed
from app.schemas.events import ObservableFact
from app.schemas.requests import TurnRequest
from tests.support.factories import (
    ClassFakeDispatcher,
    checkpoint,
    character_record,
    llm_response,
    router_output,
)


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "storage"
    / "stories"
    / "one_star_ascension_s1"
    / "ckpt_0000.json"
)


def _load_one_star() -> CheckpointFile:
    return CheckpointFile.model_validate(
        json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    )


def _spawn_request(character_id: str, role: str) -> SpawnRequest:
    return SpawnRequest(
        character_id=character_id,
        seed=SpawnSeed(
            role=role,
            reason="Niflheim's first-summon wave.",
            location="niflheim_lobby",
            objectives=[f"Find a useful place as {role}."],
            knowledge_tier=1,
        ),
    )


def _opening_output(
    *,
    observer_ids: list[str],
    agent_ids: list[str] | None = None,
    spawn: list[SpawnRequest] | None = None,
    activate: list[dict[str, str]] | None = None,
    location_updates: list[dict[str, str]] | None = None,
    fact_text: str = "The first summon-light fills the lobby.",
) -> EventRouterOutput:
    return router_output(
        event_kind="state_change",
        facts=[ObservableFact.all(fact_text)],
        agent_ids=agent_ids,
        observer_ids=observer_ids,
        spawn=spawn,
        activate=activate,
        location_updates=location_updates,
    )


def _last_user_content(messages: list[dict]) -> str:
    user_messages = [
        message for message in messages if message.get("role") == "user"
    ]
    assert user_messages
    content = user_messages[-1]["content"]
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return str(content)


async def _route_opening(
    ckpt: CheckpointFile,
    *,
    actor_id: str,
    output: EventRouterOutput,
) -> tuple[EventRouterOutput, str]:
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(
        return_value=llm_response(output, content=output.model_dump_json())
    )
    result = await LLMDispatcher(
        client,
        PromptManager("app/prompts"),
    ).route_intention(
        ckpt=ckpt,
        actor_id=actor_id,
        intention="(begin)",
    )
    messages = client.complete.await_args.kwargs["messages"]
    return result, _last_user_content(messages)


@pytest.fixture(autouse=True)
def _reset_fake_dispatcher():
    ClassFakeDispatcher.reset()
    yield
    ClassFakeDispatcher.reset()


def test_claimed_newcomer_receives_existing_character_opening_contract() -> None:
    ckpt = _load_one_star()
    newcomer = next(
        character for character in ckpt.characters
        if character.character_id == "one_star_newcomer"
    )
    newcomer.name = "Mara Vale"
    newcomer.public_sheet.appearance = "scarlet coat and iron-gray braid"
    ckpt.session.character_bindings = {"one_star_newcomer": "user-1"}
    output = _opening_output(
        observer_ids=["one_star_newcomer"],
        fact_text=(
            "The Newcomer takes shape in the first summon-light at Niflheim."
        ),
        activate=[
            {
                "character_id": "one_star_newcomer",
                "location_label": "niflheim_lobby",
            }
        ],
    )

    result, user_content = asyncio.run(
        _route_opening(
            ckpt,
            actor_id="one_star_newcomer",
            output=output,
        )
    )

    turn_context = user_content.split("<turn_context>", 1)[1].split(
        "</turn_context>", 1
    )[0]
    opening_input = user_content.split("<input>", 1)[1].split("</input>", 1)[0]
    assert "## Acting Character\none_star_newcomer" in turn_context
    assert "the_master" not in turn_context
    assert "## Authored Opening Participants" in opening_input
    assert "- one_star_newcomer" in opening_input
    assert "Name: Mara Vale" in opening_input
    assert "Appearance: scarlet coat and iron-gray braid" in opening_input
    assert "emit no spawn requests" in opening_input
    for forbidden in (
        "human-bound",
        "human player",
        "player-controlled",
        "player-owned",
        "ai control",
        "character binding",
    ):
        assert forbidden not in opening_input.lower()
    assert result.spawn == []
    assert [
        update.character_id for update in result.activate
    ] == ["one_star_newcomer"]
    assert result.location_updates == []


def test_master_only_opening_accepts_varied_generated_wave_contract() -> None:
    ckpt = _load_one_star()
    ckpt.session.character_bindings = {"the_master": "user-1"}
    spawn = [
        _spawn_request("niflheim_first_summon_01", "timid field medic"),
        _spawn_request("niflheim_first_summon_02", "blunt quarry worker"),
        _spawn_request("niflheim_first_summon_03", "watchful trail scout"),
    ]

    result, user_content = asyncio.run(
        _route_opening(
            ckpt,
            actor_id="the_master",
            output=_opening_output(
                observer_ids=["the_master"],
                spawn=spawn,
            ),
        )
    )

    turn_context = user_content.split("<turn_context>", 1)[1].split(
        "</turn_context>", 1
    )[0]
    opening_input = user_content.split("<input>", 1)[1].split("</input>", 1)[0]
    assert "## Acting Character\nthe_master" in turn_context
    assert "one_star_newcomer" not in turn_context
    assert "## Authored Opening Participants" in opening_input
    assert "- the_master" in opening_input
    assert "- one_star_newcomer" not in opening_input
    assert "New-character spawn requests: allowed only" in opening_input
    assert "niflheim_first_summon_03" in opening_input
    assert [request.character_id for request in result.spawn] == [
        "niflheim_first_summon_01",
        "niflheim_first_summon_02",
        "niflheim_first_summon_03",
    ]
    assert len({request.seed.role for request in result.spawn}) == 3
    assert all(request.seed.location == "niflheim_lobby" for request in result.spawn)
    assert all(request.seed.objectives for request in result.spawn)
    assert all(request.seed.knowledge_tier == 1 for request in result.spawn)


def test_multiple_selected_opening_participants_are_semantic_only() -> None:
    ckpt = _load_one_star()
    claimed_ids = [
        "one_star_newcomer",
        "the_master",
        "halcyon_of_the_gilded_march",
    ]
    ckpt.session.character_bindings = {
        character_id: f"user-{index}"
        for index, character_id in enumerate(claimed_ids, start=1)
    }

    result, user_content = asyncio.run(
        _route_opening(
            ckpt,
            actor_id="one_star_newcomer",
            output=_opening_output(
                observer_ids=claimed_ids,
                activate=[
                    {
                        "character_id": "one_star_newcomer",
                        "location_label": "niflheim_lobby",
                    }
                ],
            ),
        )
    )

    turn_context = user_content.split("<turn_context>", 1)[1].split(
        "</turn_context>", 1
    )[0]
    opening_input = user_content.split("<input>", 1)[1].split("</input>", 1)[0]
    participant_block = opening_input.split(
        "## Authored Opening Participants",
        1,
    )[1].split("## Authored Opening Context", 1)[0]
    for character_id in claimed_ids:
        assert f"- {character_id}\n" in participant_block
    assert "## Acting Character\none_star_newcomer" in turn_context
    for forbidden in (
        "human-bound",
        "human player",
        "player-controlled",
        "player-owned",
        "ai control",
        "character binding",
    ):
        assert forbidden not in participant_block.lower()
    assert {observer.character_id for observer in result.observers} == set(
        claimed_ids
    )
    assert result.spawn == []


def test_story_without_opening_spawn_authority_rejects_router_spawn() -> None:
    ckpt = checkpoint(
        bindings={"alice": "user-1"},
        characters=[
            character_record(
                "alice",
                role="player",
                is_playable=True,
            )
        ],
    )
    output = _opening_output(
        observer_ids=["alice"],
        spawn=[_spawn_request("unexpected_arrival", "traveler")],
    )
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(
        return_value=llm_response(output, content=output.model_dump_json())
    )

    with pytest.raises(ValueError, match="not authorized opening spawns"):
        asyncio.run(
            LLMDispatcher(
                client,
                PromptManager("app/prompts"),
            ).route_intention(
                ckpt=ckpt,
                actor_id="alice",
                intention="(begin)",
            )
        )

    user_content = _last_user_content(
        client.complete.await_args.kwargs["messages"]
    )
    assert "New-character spawn requests: forbidden" in user_content
    assert ckpt.session_conversation == []


def test_opening_spawn_cannot_duplicate_claimed_existing_character() -> None:
    ckpt = _load_one_star()
    ckpt.session.character_bindings = {"one_star_newcomer": "user-1"}
    duplicate = _spawn_request("one_star_newcomer", "replacement protagonist")

    with pytest.raises(ValueError, match="genuinely new character ids"):
        asyncio.run(
            _route_opening(
                ckpt,
                actor_id="one_star_newcomer",
                output=_opening_output(
                    observer_ids=["one_star_newcomer"],
                    spawn=[duplicate],
                ),
            )
        )


class RecordingCharacterManager:
    def __init__(self) -> None:
        self.calls = 0
        self.request_batches: list[list[str]] = []

    async def spawn_characters(
        self,
        checkpoint,
        requests,
        *,
        acting_actor_location,
    ):
        self.calls += 1
        self.request_batches.append(
            [request.character_id for request in requests]
        )
        return [
            CharacterRecord(
                character_id=request.character_id,
                name=request.seed.role.title(),
                location=request.seed.location,
                public_sheet=PublicSheet(role=request.seed.role),
            )
            for request in requests
        ]


def _orchestrator_with_fake_dispatcher(
    monkeypatch,
    ckpt: CheckpointFile,
    spawn_manager: RecordingCharacterManager,
) -> tuple[Orchestrator, MagicMock]:
    monkeypatch.setattr(
        "app.engine.orchestrator.LLMDispatcher",
        ClassFakeDispatcher,
    )
    client = MagicMock()
    client.config = MagicMock()
    checkpoint_manager = MagicMock()
    checkpoint_manager.load_latest.return_value = ckpt
    orchestrator = Orchestrator(
        client,
        checkpoint_manager,
        MagicMock(),
        spawn_authoring=SpawnAuthoringCoordinator(spawn_manager),
    )
    return orchestrator, checkpoint_manager


@pytest.mark.asyncio
async def test_claimed_newcomer_is_activated_without_materialization(
    monkeypatch,
) -> None:
    ckpt = _load_one_star()
    newcomer = next(
        character
        for character in ckpt.characters
        if character.character_id == "one_star_newcomer"
    )
    newcomer.name = "Mara Vale"
    newcomer.public_sheet.appearance = "scarlet coat and iron-gray braid"
    ckpt.session.character_bindings = {"one_star_newcomer": "user-1"}
    assert newcomer.status.value == "dormant"
    assert newcomer.location == "not_yet_fictional"
    spawn_manager = RecordingCharacterManager()
    orchestrator, _checkpoint_manager = _orchestrator_with_fake_dispatcher(
        monkeypatch,
        ckpt,
        spawn_manager,
    )
    ClassFakeDispatcher.queue_route(
        _opening_output(
            observer_ids=["one_star_newcomer"],
            fact_text=(
                "The Newcomer takes shape in the first summon-light at "
                "Niflheim."
            ),
            activate=[
                {
                    "character_id": "one_star_newcomer",
                    "location_label": "niflheim_lobby",
                }
            ],
        )
    )

    await orchestrator.process_turn(
        TurnRequest(
            session_id=ckpt.session.session_id,
            user_input="(begin)",
            acting_character_id="one_star_newcomer",
        )
    )

    assert ckpt.canonical_events[-1].spawn == []
    assert "Newcomer takes shape" in (
        ckpt.canonical_events[-1].canonical_event.observable_facts[0].text
    )
    assert newcomer.status.value == "active"
    assert newcomer.location == "niflheim_lobby"
    assert spawn_manager.calls == 0
    assert sum(
        character.character_id == "one_star_newcomer"
        for character in ckpt.characters
    ) == 1
    assert ClassFakeDispatcher.agent_calls == []


@pytest.mark.asyncio
async def test_claimed_newcomer_mid_session_arrival_activates_existing_record(
    monkeypatch,
) -> None:
    ckpt = _load_one_star()
    newcomer = next(
        character
        for character in ckpt.characters
        if character.character_id == "one_star_newcomer"
    )
    newcomer.name = "Mara Vale"
    newcomer.public_sheet.appearance = "scarlet coat and iron-gray braid"
    ckpt.session.character_bindings = {"one_star_newcomer": "user-1"}
    ckpt.session.turn_index = 4
    spawn_manager = RecordingCharacterManager()
    orchestrator, _checkpoint_manager = _orchestrator_with_fake_dispatcher(
        monkeypatch,
        ckpt,
        spawn_manager,
    )
    ClassFakeDispatcher.queue_route(
        _opening_output(
            observer_ids=["one_star_newcomer"],
            fact_text=(
                "Mara Vale takes shape in the summon-light at Niflheim."
            ),
            activate=[
                {
                    "character_id": "one_star_newcomer",
                    "location_label": "niflheim_lobby",
                }
            ],
        )
    )

    await orchestrator.process_turn(
        TurnRequest(
            session_id=ckpt.session.session_id,
            user_input="(arrive)",
            acting_character_id="one_star_newcomer",
        )
    )

    assert ckpt.canonical_events[-1].spawn == []
    assert newcomer.name == "Mara Vale"
    assert newcomer.public_sheet.appearance == (
        "scarlet coat and iron-gray braid"
    )
    assert newcomer.status.value == "active"
    assert newcomer.location == "niflheim_lobby"
    assert spawn_manager.calls == 0
    assert sum(
        character.character_id == "one_star_newcomer"
        for character in ckpt.characters
    ) == 1
    assert len(ClassFakeDispatcher.route_calls) == 1
    assert ClassFakeDispatcher.agent_calls == []


@pytest.mark.asyncio
async def test_failed_opening_render_retry_materializes_wave_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ckpt = _load_one_star()
    ckpt.session.character_bindings = {"the_master": "user-1"}
    spawn = [
        _spawn_request("niflheim_first_summon_01", "timid field medic"),
        _spawn_request("niflheim_first_summon_02", "blunt quarry worker"),
        _spawn_request("niflheim_first_summon_03", "watchful trail scout"),
    ]
    spawn_manager = RecordingCharacterManager()
    monkeypatch.setattr(
        "app.engine.orchestrator.LLMDispatcher",
        ClassFakeDispatcher,
    )
    checkpoint_manager = CheckpointManager(str(tmp_path))
    checkpoint_manager.save(ckpt)
    client = MagicMock()
    client.config = MagicMock()
    orchestrator = Orchestrator(
        client,
        checkpoint_manager,
        MagicMock(),
        spawn_authoring=SpawnAuthoringCoordinator(spawn_manager),
    )
    ClassFakeDispatcher.queue_route(
        _opening_output(
            observer_ids=["the_master"],
            spawn=spawn,
        )
    )
    ClassFakeDispatcher.queue_narrator_error(RuntimeError("narrator offline"))

    with pytest.raises(RuntimeError, match="narrator offline"):
        await orchestrator.process_turn(
            TurnRequest(
                session_id=ckpt.session.session_id,
                user_input="(begin)",
                acting_character_id="the_master",
            )
        )

    expected_ids = [request.character_id for request in spawn]
    assert spawn_manager.calls == 1
    assert spawn_manager.request_batches == [expected_ids]
    pending_checkpoint = checkpoint_manager.load_latest(
        ckpt.session.session_id
    )
    pending_render = pending_checkpoint.session.pending_narrator_render
    assert pending_render is not None
    # A failed narrator candidate must not make the wave part of the active
    # roster.  The exact authored records live only in the durable retry
    # payload until prose is accepted.
    assert all(
        sum(
            character.character_id == character_id
            for character in pending_checkpoint.characters
        )
        == 0
        for character_id in expected_ids
    )
    assert [
        character.character_id
        for character in pending_render.pending_spawn_records
    ] == expected_ids

    response = await orchestrator.retry_pending_narrator_render(
        ckpt.session.session_id
    )

    assert response.output_text == "POV_RENDER"
    assert spawn_manager.calls == 1
    completed_checkpoint = checkpoint_manager.load_latest(
        ckpt.session.session_id
    )
    assert completed_checkpoint.session.pending_narrator_render is None
    assert all(
        sum(
            character.character_id == character_id
            for character in completed_checkpoint.characters
        )
        == 1
        for character_id in expected_ids
    )
    assert len(ClassFakeDispatcher.route_calls) == 1


@pytest.mark.asyncio
async def test_master_opening_reaches_selected_guide_before_first_render(
    monkeypatch,
) -> None:
    ckpt = _load_one_star()
    ckpt.session.character_bindings = {"the_master": "user-1"}
    spawn = [
        _spawn_request("niflheim_first_summon_01", "timid field medic"),
        _spawn_request("niflheim_first_summon_02", "blunt quarry worker"),
        _spawn_request("niflheim_first_summon_03", "watchful trail scout"),
    ]
    spawned_ids = [request.character_id for request in spawn]
    spawn_manager = RecordingCharacterManager()
    orchestrator, _checkpoint_manager = _orchestrator_with_fake_dispatcher(
        monkeypatch,
        ckpt,
        spawn_manager,
    )
    ClassFakeDispatcher.queue_route(
        _opening_output(
            observer_ids=["the_master", "iselle_the_guide", *spawned_ids],
            agent_ids=["iselle_the_guide"],
            spawn=spawn,
        )
    )
    ClassFakeDispatcher.queue_agent(
        "I brief the new Heroes, then recommend that the Master form their "
        "party and deploy it to Floor 1."
    )
    ClassFakeDispatcher.queue_route(router_output(
        event_id="evt_iselle_briefing",
        event_kind="response_requested",
        facts=[ObservableFact.all(
            "Iselle briefs the new Heroes, then recommends that the Master "
            "form their party and deploy it to Floor 1."
        )],
        agent_ids=["the_master"],
        observer_ids=["the_master", "iselle_the_guide", *spawned_ids],
    ))

    response = await orchestrator.process_turn(
        TurnRequest(
            session_id=ckpt.session.session_id,
            user_input="(begin)",
            acting_character_id="the_master",
        )
    )

    assert response.output_text == "POV_RENDER"
    assert len(ckpt.canonical_events) == 2
    assert ckpt.canonical_events[-1].event_id == "evt_iselle_briefing"
    assert len(ClassFakeDispatcher.route_calls) == 2
    assert len(ClassFakeDispatcher.agent_calls) == 1
    assert len(ClassFakeDispatcher.narrator_calls) == 1
    assert len(
        ClassFakeDispatcher.narrator_calls[0]["buffered_events"]
    ) == 2
    guide_checkpoint = ClassFakeDispatcher.agent_calls[0]["ckpt"]
    assert all(
        any(
            character.character_id == character_id
            for character in guide_checkpoint.characters
        )
        for character_id in spawned_ids
    )
    assert spawn_manager.calls == 1
    assert all(
        sum(
            character.character_id == character_id
            for character in ckpt.characters
        )
        == 1
        for character_id in spawned_ids
    )
