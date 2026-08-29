"""Behavioral contracts for the authored opening-roster guide boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine import turn_loop_dispatcher as dispatcher_module
from app.engine.one_star_adapter import (
    load_one_star_account,
    load_one_star_hero,
    one_star_opening_roster_preview,
)
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.llm.client import LLMClient
from app.schemas.characters import CharacterStatus
from app.schemas.checkpoint import CheckpointFile
from app.schemas.events import ObservableFact
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    OneStarEventRouterOutput,
    OneStarPendingOperation,
)
from app.schemas.state import OpenCatIIEvent
from tests.support.factories import checkpoint, llm_response, router_output


STORY_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "storage"
    / "stories"
    / "one_star_ascension_s1"
    / "ckpt_0000.json"
)


def _floor_one_scenario():
    checkpoint = CheckpointFile.model_validate(
        json.loads(STORY_PATH.read_text(encoding="utf-8"))
    )
    _owner, account = load_one_star_account(checkpoint)
    return account.config.floor_scenarios[1]


def _dispatcher(*responses: object) -> tuple[LLMDispatcher, MagicMock]:
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(
        side_effect=[
            llm_response(response, content="{}") for response in responses
        ]
    )
    return LLMDispatcher(client, PromptManager("app/prompts")), client


def _one_star_output(base: object, *, state_updates: list[dict]) -> OneStarEventRouterOutput:
    data = base.model_dump(mode="json")
    data["state_updates"] = state_updates
    return OneStarEventRouterOutput.model_validate(data)


def _store_hero(character: object, hero: object) -> None:
    character.mechanics = dict(character.mechanics)
    character.mechanics[ONE_STAR_HERO_KEY] = hero.model_dump(mode="json")


def _store_account(owner: object, account: object) -> None:
    owner.mechanics = dict(owner.mechanics)
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")


def _opening_roster_checkpoint(
    *,
    with_pending: bool = True,
) -> tuple[CheckpointFile, str, list[str], str, OpenCatIIEvent | None]:
    ckpt = CheckpointFile.model_validate(
        json.loads(STORY_PATH.read_text(encoding="utf-8"))
    )
    owner, account = load_one_star_account(ckpt)
    scenario = account.config.floor_scenarios[1]
    pool_id = next(
        pool_id
        for pool_id, pool in account.config.summon_pools.items()
        if getattr(pool, "initial_deployment_requires_guide_handoff", False)
    )
    roster_ids = [
        draw.existing_character_id
        for draw in one_star_opening_roster_preview(ckpt, pool_id)
    ]
    assert all(roster_ids)
    roster_ids = [character_id for character_id in roster_ids if character_id]
    guide_id = account.state.guide_character_ids[0]
    opening_event_id = "event_opening_roster"
    opening = _one_star_output(
        router_output(
            event_id=opening_event_id,
            observer_ids=[owner.character_id, guide_id, *roster_ids],
            activate=[
                {
                    "character_id": character_id,
                    "location_label": account.config.lobby_location_label,
                }
                for character_id in roster_ids
            ],
            facts=[],
        ),
        state_updates=[
            {
                "kind": "summon",
                "target_id": pool_id,
                "value": str(len(roster_ids)),
                "details": [],
            }
        ],
    )
    ckpt.canonical_events.append(opening)

    characters = {
        character.character_id: character for character in ckpt.characters
    }
    for character_id in roster_ids:
        character = characters[character_id]
        hero = load_one_star_hero(character)
        assert hero is not None
        character.status = CharacterStatus.active
        character.location = account.config.lobby_location_label
        hero.owner_lobby_id = account.config.lobby_id
        hero.acquisition_event_id = opening_event_id
        _store_hero(character, hero)

    if not with_pending:
        return ckpt, owner.character_id, roster_ids, guide_id, None

    pending_event_id = "event_initial_deployment_open"
    pending_operation_id = "initial_floor_1_deployment"
    pending_open = _pending_open_output(
        owner_id=owner.character_id,
        roster_ids=roster_ids,
        guide_id=guide_id,
        operation_id=pending_operation_id,
    )
    pending_open.event_id = pending_event_id
    ckpt.canonical_events.append(pending_open)
    account.state.pending_operation = OneStarPendingOperation(
        operation_id=pending_operation_id,
        kind="deployment",
        participant_ids=roster_ids,
        target_id="",
        destination=scenario.destination,
        opened_at_s=4,
        synthesis_preview=None,
    )
    _store_account(owner, account)
    cat_ii_event = OpenCatIIEvent(
        event_id="cat_initial_deployment",
        initiator_id=owner.character_id,
        initiator_intention="Deploy the opening roster to Floor 1.",
        required_responders=roster_ids,
        collected_intentions={
            character_id: f"{character_id} responds independently."
            for character_id in roster_ids
        },
        opening_event_id=pending_event_id,
        opening_observer_ids=[owner.character_id, guide_id, *roster_ids],
    )
    return ckpt, owner.character_id, roster_ids, guide_id, cat_ii_event


def _pending_open_output(
    *,
    owner_id: str,
    roster_ids: list[str],
    guide_id: str,
    operation_id: str = "initial_floor_1_deployment",
    responder_ids: list[str] | None = None,
) -> OneStarEventRouterOutput:
    responders = roster_ids if responder_ids is None else responder_ids
    return _one_star_output(
        router_output(
            event_id="pending_open_candidate",
            requires_responders=True,
            required_responders=responders,
            observer_ids=[owner_id, guide_id, *roster_ids],
            facts=[
                ObservableFact.all("The Master selects the opening party."),
                ObservableFact.only(
                    "The System gives the guide the deployment selection.",
                    [guide_id],
                ),
            ],
        ),
        state_updates=[
            {
                "kind": "pending_open",
                "target_id": operation_id,
                "value": "deployment",
                "details": [
                    *(f"participant={character_id}" for character_id in roster_ids),
                    f"destination={_floor_one_scenario().destination}",
                ],
            }
        ],
    )


def _guide_bridge_output(
    *,
    owner_id: str,
    roster_ids: list[str],
    guide_id: str,
    event_id: str = "guide_bridge",
) -> OneStarEventRouterOutput:
    return _one_star_output(
        router_output(
            event_id=event_id,
            event_kind="response_requested",
            observer_ids=[owner_id, *roster_ids, guide_id],
            agent_ids=[guide_id],
            facts=[ObservableFact.all("The collected Hero responses stand.")],
        ),
        state_updates=[],
    )


def _deployment_resolution_output(
    *,
    owner_id: str,
    roster_ids: list[str],
    guide_id: str,
    event_id: str = "direct_deployment_resolution",
) -> OneStarEventRouterOutput:
    operation_id = "initial_floor_1_deployment"
    scenario = _floor_one_scenario()
    formation_by_character_id = {
        "mirelle_voss": "front",
        "one_star_newcomer": (
            "middle-left" if len(roster_ids) == 4 else "front"
        ),
        "edren_marr": (
            "middle-right" if len(roster_ids) == 4 else "middle"
        ),
        "renna_holt": "rear",
    }
    return _one_star_output(
        router_output(
            event_id=event_id,
            event_kind="cat_ii_resolution",
            observer_ids=[owner_id, guide_id, *roster_ids],
            location_updates=[
                {
                    "character_id": character_id,
                    "location_label": _floor_one_scenario().destination,
                }
                for character_id in roster_ids
            ],
            effective_at_s=4,
            duration_s=6,
        ),
        state_updates=[
            {
                "kind": "pending_resolve",
                "target_id": operation_id,
                "value": "",
                "details": [],
            },
            {
                "kind": "mission_start",
                "target_id": scenario.mission_id,
                "value": "1",
                "details": [
                    f"pending_operation_id={operation_id}",
                    *(f"party={character_id}" for character_id in roster_ids),
                    *(
                        f"formation.{character_id}="
                        f"{formation_by_character_id[character_id]}"
                        for character_id in roster_ids
                    ),
                    f"destination={scenario.destination}",
                    f"completion={scenario.completion_declaration}",
                    f"failure={scenario.failure_declaration}",
                    *(
                        f"counter.{counter.counter_id}="
                        f"{counter.current}/{counter.target}"
                        for counter in scenario.counters
                    ),
                ],
            },
        ],
    )


def _disable_content_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dispatcher_module,
        "_append_router_content_lookup_records",
        AsyncMock(return_value=[]),
    )


def test_initial_deployment_responder_order_is_corrected_before_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_content_preflight(monkeypatch)
    ckpt, owner_id, roster_ids, guide_id, _cat_ii = (
        _opening_roster_checkpoint(with_pending=False)
    )
    invalid = _pending_open_output(
        owner_id=owner_id,
        roster_ids=roster_ids,
        guide_id=guide_id,
        responder_ids=list(reversed(roster_ids)),
    )
    corrected = _pending_open_output(
        owner_id=owner_id,
        roster_ids=roster_ids,
        guide_id=guide_id,
    )
    invalid.event_id = "invalid_responder_order"
    corrected.event_id = "corrected_responder_order"
    dispatcher, client = _dispatcher(invalid, corrected)

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=ckpt,
            actor_id=owner_id,
            intention="Deploy the opening roster.",
        )
    )

    assert result is corrected
    assert client.complete.await_count == 2
    correction = client.complete.await_args_list[1].kwargs["messages"][-1][
        "content"
    ]
    assert "opening-roster slot order" in correction
    assert f"expected={roster_ids!r}" in correction
    assert f"received={list(reversed(roster_ids))!r}" in correction
    history = "\n".join(str(message.content) for message in ckpt.session_conversation)
    assert corrected.event_id in history
    assert invalid.event_id not in history


def test_direct_initial_crossing_is_corrected_to_side_effect_free_guide_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_content_preflight(monkeypatch)
    ckpt, owner_id, roster_ids, guide_id, cat_ii_event = (
        _opening_roster_checkpoint()
    )
    assert cat_ii_event is not None
    invalid = _deployment_resolution_output(
        owner_id=owner_id,
        roster_ids=roster_ids,
        guide_id=guide_id,
    )
    corrected = _guide_bridge_output(
        owner_id=owner_id,
        roster_ids=roster_ids,
        guide_id=guide_id,
    )
    dispatcher, client = _dispatcher(invalid, corrected)

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=ckpt,
            actor_id=owner_id,
            intention=cat_ii_event.initiator_intention,
            cat_ii_event=cat_ii_event,
        )
    )

    assert result is corrected
    assert result.state_updates == []
    assert result.location_updates == []
    assert result.next_output_character_ids == [guide_id]
    assert client.complete.await_count == 2
    correction = client.complete.await_args_list[1].kwargs["messages"][-1][
        "content"
    ]
    assert "side-effect-free post-collection guide handoff" in correction
    history = "\n".join(str(message.content) for message in ckpt.session_conversation)
    assert corrected.event_id in history
    assert invalid.event_id not in history
    _owner, account = load_one_star_account(ckpt)
    assert account.state.pending_operation is not None
    assert all(
        next(
            character
            for character in ckpt.characters
            if character.character_id == character_id
        ).location
        == account.config.lobby_location_label
        for character_id in roster_ids
    )


def test_invalid_guide_bridge_correction_restores_router_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_content_preflight(monkeypatch)
    ckpt, owner_id, roster_ids, guide_id, cat_ii_event = (
        _opening_roster_checkpoint()
    )
    assert cat_ii_event is not None
    first = _deployment_resolution_output(
        owner_id=owner_id,
        roster_ids=roster_ids,
        guide_id=guide_id,
        event_id="invalid_direct_resolution_1",
    )
    second = _deployment_resolution_output(
        owner_id=owner_id,
        roster_ids=roster_ids,
        guide_id=guide_id,
        event_id="invalid_direct_resolution_2",
    )
    dispatcher, client = _dispatcher(first, second)
    before = ckpt.model_dump(mode="json")

    with pytest.raises(ValueError, match="remained invalid after one correction"):
        asyncio.run(
            dispatcher.route_intention(
                ckpt=ckpt,
                actor_id=owner_id,
                intention=cat_ii_event.initiator_intention,
                cat_ii_event=cat_ii_event,
            )
        )

    assert client.complete.await_count == 2
    assert ckpt.model_dump(mode="json") == before


def test_later_guide_turn_may_resolve_and_start_the_mission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_content_preflight(monkeypatch)
    ckpt, owner_id, roster_ids, guide_id, _cat_ii_event = (
        _opening_roster_checkpoint()
    )
    resolution = _deployment_resolution_output(
        owner_id=owner_id,
        roster_ids=roster_ids,
        guide_id=guide_id,
        event_id="guide_owned_resolution",
    )
    dispatcher, client = _dispatcher(resolution)

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=ckpt,
            actor_id=guide_id,
            intention="I force the holdout through and close the gate.",
        )
    )
    asyncio.run(
        dispatcher.prepare_ruleset_event(
            ckpt=ckpt,
            result=result,
            actor_id=guide_id,
        )
    )

    assert client.complete.await_count == 1
    _owner, account = load_one_star_account(ckpt)
    assert account.state.pending_operation is None
    assert account.state.active_mission is not None
    assert account.state.active_mission.party_ids == roster_ids
    assert all(
        next(
            character
            for character in ckpt.characters
            if character.character_id == character_id
        ).location
        == _floor_one_scenario().destination
        for character_id in roster_ids
    )


def test_newcomer_pool_does_not_trigger_opening_roster_guide_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_content_preflight(monkeypatch)
    ckpt, owner_id, _roster_ids, guide_id, _cat_ii = (
        _opening_roster_checkpoint(with_pending=False)
    )
    newcomer_id = "one_star_newcomer"
    opening = ckpt.canonical_events[0]
    opening.state_updates[0].target_id = "newcomer_opening"
    opening.state_updates[0].value = "1"
    opening.activate = [
        type(opening.activate[0])(
            character_id=newcomer_id,
            location_label="niflheim_lobby",
        )
    ]
    for character in ckpt.characters:
        hero = load_one_star_hero(character)
        if hero is None:
            continue
        hero.owner_lobby_id = ""
        hero.acquisition_event_id = ""
        if character.character_id == newcomer_id:
            character.status = CharacterStatus.active
            character.location = "niflheim_lobby"
            hero.owner_lobby_id = "niflheim"
            hero.acquisition_event_id = opening.event_id
        _store_hero(character, hero)
    _owner, account = load_one_star_account(ckpt)
    account.state.pending_operation = OneStarPendingOperation(
        operation_id="newcomer_deployment",
        kind="deployment",
        participant_ids=[newcomer_id],
        target_id="",
        destination=_floor_one_scenario().destination,
        opened_at_s=4,
        synthesis_preview=None,
    )
    _store_account(_owner, account)
    pending_open = _pending_open_output(
        owner_id=owner_id,
        roster_ids=[newcomer_id],
        guide_id=guide_id,
        operation_id="newcomer_deployment",
    )
    pending_open.event_id = "newcomer_pending_open"
    ckpt.canonical_events.append(pending_open)
    cat_ii_event = OpenCatIIEvent(
        event_id="newcomer_cat_ii",
        initiator_id=owner_id,
        initiator_intention="Deploy the newcomer.",
        required_responders=[newcomer_id],
        collected_intentions={newcomer_id: "I enter."},
        opening_event_id=pending_open.event_id,
    )
    direct = _deployment_resolution_output(
        owner_id=owner_id,
        roster_ids=[newcomer_id],
        guide_id=guide_id,
        event_id="newcomer_direct_resolution",
    )
    direct.state_updates[0].target_id = "newcomer_deployment"
    direct.state_updates[1].details[0] = (
        "pending_operation_id=newcomer_deployment"
    )
    direct.state_updates[1].details = [
        detail
        for detail in direct.state_updates[1].details
        if not detail.startswith("formation.")
    ]
    direct.state_updates[1].details.append(f"formation.{newcomer_id}=front")
    direct.canonical_event.observable_facts.append(
        ObservableFact.only(
            "The System gives the guide the completed deployment state.",
            [guide_id],
        )
    )
    dispatcher, client = _dispatcher(direct)

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=ckpt,
            actor_id=owner_id,
            intention=cat_ii_event.initiator_intention,
            cat_ii_event=cat_ii_event,
        )
    )

    assert result is direct
    assert client.complete.await_count == 1


def test_generic_cat_ii_resolution_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_content_preflight(monkeypatch)
    ckpt = checkpoint()
    cat_ii_event = OpenCatIIEvent(
        event_id="generic_cat_ii",
        initiator_id="alice",
        initiator_intention="I take the letter.",
        required_responders=["pip"],
        collected_intentions={"pip": "I keep hold of it."},
    )
    resolved = router_output(
        event_id="generic_resolution",
        event_kind="cat_ii_resolution",
    )
    dispatcher, client = _dispatcher(resolved)

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention=cat_ii_event.initiator_intention,
            cat_ii_event=cat_ii_event,
        )
    )

    assert result is resolved
    assert client.complete.await_count == 1
