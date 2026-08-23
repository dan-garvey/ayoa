"""Regression coverage for failed One-Star spawn preparation.

This stays fully offline: the only mocked model response is the bounded
transaction-only repair, which deliberately repeats an invalid empty
transaction.  The test exercises the shared closed-event roster overlay so a
failed prepare cannot leave generated identities behind.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.action_rejection import PlayerActionRejected
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.closed_event_runtime import (
    ClosedEventRuntime,
    install_closed_event_runtime,
)
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.engine.spawn_authoring import SpawnAuthoringCoordinator
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.engine.one_star_adapter import OneStarTransactionError
from app.llm.client import LLMClient
from app.schemas.characters import CharacterRecord
from app.schemas.event_router import SpawnRequest, SpawnSeed
from app.schemas.one_star import (
    OneStarEventRouterOutput,
    OneStarStateUpdateList,
)
from tests.support.factories import character_record, llm_response, router_output
from tests.test_one_star_adapter import _checkpoint


class _SpawnAuthor:
    """Author two deterministic records without touching a provider."""

    async def spawn_characters(
        self,
        checkpoint,
        requests,
        *,
        acting_actor_location,
        one_star_hero_ids,
    ) -> list[CharacterRecord]:
        records = [
            character_record(
                request.character_id,
                name=f"Authored {request.character_id}",
                location=request.seed.location or acting_actor_location,
            )
            for request in requests
        ]
        for record in records:
            if record.character_id not in one_star_hero_ids:
                continue
            record.mechanics["one_star_hero"] = {
                "birth_stars": 1,
                "current_stars": 1,
                "level": 1,
                "experience_points": 0,
                "hp_current": 7,
                "hp_max": 7,
                "stats": {"power": 6, "agility": 5, "resilience": 4},
                "terminal_event_id": "",
                "progression_seed": f"{record.character_id}_progression_seed",
                "strong_stat_id": "power",
                "weak_stat_id": "resilience",
                "potential_grade": 1,
            }
        return records


def _failed_spawn_event() -> OneStarEventRouterOutput:
    spawns = [
        SpawnRequest(
            character_id=character_id,
            seed=SpawnSeed(
                role="fresh one-star",
                reason="the summon light opens",
                location="lobby",
                objectives=["survive the first moment"],
                knowledge_tier=1,
            ),
        )
        for character_id in ("fresh_alpha", "fresh_beta")
    ]
    data = router_output(
        event_id="evt_failed_spawn_prepare",
        observer_ids=["account_owner", "fresh_alpha", "fresh_beta"],
        spawn=spawns,
    ).model_dump()
    # The standard draw adds its adapter-authored Hero lifecycle before an
    # invalid catalogue command fails. An empty repair cannot detach that
    # staged Hero from its summon, exercising complete roster rollback.
    data["state_updates"] = [
        {
            "kind": "summon",
            "target_id": "basic",
            "value": "2",
            "details": [],
        },
        {
            "kind": "catalogue_apply",
            "target_id": "not_a_real_catalogue_entry",
            "value": "1",
            "details": [],
        },
    ]
    return OneStarEventRouterOutput.model_validate(data)


@pytest.mark.asyncio
async def test_failed_one_star_prepare_rolls_back_fresh_spawn_overlay_after_bad_repair(
    tmp_path: Path,
):
    checkpoint = _checkpoint()
    checkpoint.session.session_id = "one_star_spawn_rollback"
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    manager.save(checkpoint)
    persisted_before = manager._checkpoint_path(
        checkpoint.session.session_id,
        checkpoint.session.turn_index,
    ).read_bytes()
    resources_before = (
        checkpoint.characters[0].mechanics["one_star_account"]["state"]
        ["resources"]
        .copy()
    )
    roster_before = {
        character.character_id for character in checkpoint.characters
    }

    coordinator = SpawnAuthoringCoordinator(_SpawnAuthor())
    runtime = ClosedEventRuntime(
        transaction_id="tx_failed_one_star_prepare",
        source_turn_index=checkpoint.session.turn_index + 1,
        spawn_authoring=coordinator,
        record_applier=Orchestrator._apply_authored_spawn_records,
    )
    install_closed_event_runtime(checkpoint, runtime)

    invalid_repair = OneStarStateUpdateList(state_updates=[])
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(
        return_value=llm_response(invalid_repair),
    )
    dispatcher = LLMDispatcher(client, PromptManager("app/prompts"))
    orchestrator = Orchestrator(
        client,
        manager,
        PromptManager("app/prompts"),
        spawn_authoring=coordinator,
    )
    event = _failed_spawn_event()

    with pytest.raises(
        OneStarTransactionError,
        match="repair cannot change a summon",
    ):
        await dispatcher.prepare_ruleset_event(
            ckpt=checkpoint,
            result=event,
            actor_id="account_owner",
        )

    assert client.complete.await_count == 1
    assert {
        character.character_id for character in checkpoint.characters
    } == roster_before
    assert not runtime.applied_character_ids
    assert not coordinator.pending_introductions(runtime.transaction_id)
    assert (
        checkpoint.characters[0].mechanics["one_star_account"]["state"]
        ["resources"]
        == resources_before
    )
    assert checkpoint.canonical_events == []
    assert manager._checkpoint_path(
        checkpoint.session.session_id,
        checkpoint.session.turn_index,
    ).read_bytes() == persisted_before

    await orchestrator._cancel_closed_event_runtime(
        checkpoint,
        reason="turn_failed_before_commit",
    )
    assert not coordinator.pending_introductions(runtime.transaction_id)


@pytest.mark.asyncio
async def test_unaffordable_summon_rejects_before_spawn_authoring_or_repair():
    checkpoint = _checkpoint()
    checkpoint.characters[0].mechanics["one_star_account"]["state"][
        "resources"
    ]["gold"] = 2
    before = checkpoint.model_dump_json()
    data = router_output(
        event_id="evt_unaffordable_summon",
        observer_ids=["account_owner"],
        facts=[],
    ).model_dump(mode="json")
    data["state_updates"] = [{
        "kind": "summon",
        "target_id": "basic",
        "value": "2",
        "details": [],
    }]
    event = OneStarEventRouterOutput.model_validate(data)
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock()
    dispatcher = LLMDispatcher(client, PromptManager("app/prompts"))
    dispatcher.materialize_spawns = AsyncMock()

    with pytest.raises(
        PlayerActionRejected,
        match=r"2 pulls cost 4 Gold, but only 2 Gold are available",
    ):
        await dispatcher.prepare_ruleset_event(
            ckpt=checkpoint,
            result=event,
            actor_id="account_owner",
        )

    dispatcher.materialize_spawns.assert_not_awaited()
    client.complete.assert_not_awaited()
    assert event.spawn == []
    assert event.activate == []
    assert checkpoint.model_dump_json() == before


@pytest.mark.asyncio
async def test_lifecycle_failure_restores_prepared_one_star_ledger(monkeypatch):
    checkpoint = _checkpoint()
    before = checkpoint.model_dump_json()
    data = router_output(
        event_id="evt_lifecycle_failure",
        observer_ids=["account_owner"],
        facts=[],
    ).model_dump(mode="json")
    data["state_updates"] = [{
        "kind": "catalogue_apply",
        "target_id": "synthesis_chamber_1",
        "value": "1",
        "details": [],
    }]
    event = OneStarEventRouterOutput.model_validate(data)
    dispatcher = LLMDispatcher(MagicMock(spec=LLMClient), PromptManager("app/prompts"))

    def fail_lifecycle(*_args, **_kwargs):
        raise RuntimeError("injected lifecycle failure")

    monkeypatch.setattr(
        "app.engine.turn_loop_dispatcher.CharacterManager.apply_roster_updates",
        fail_lifecycle,
    )
    with pytest.raises(RuntimeError, match="injected lifecycle failure"):
        await dispatcher.prepare_ruleset_event(
            ckpt=checkpoint,
            result=event,
            actor_id="account_owner",
        )

    assert checkpoint.model_dump_json() == before


@pytest.mark.asyncio
async def test_dispatcher_rejects_rebroadcast_of_committed_one_star_event():
    checkpoint = _checkpoint()
    data = router_output(
        event_id="evt_duplicate",
        observer_ids=["account_owner"],
        facts=[],
    ).model_dump(mode="json")
    data["state_updates"] = []
    event = OneStarEventRouterOutput.model_validate(data)
    dispatcher = LLMDispatcher(MagicMock(spec=LLMClient), PromptManager("app/prompts"))

    await dispatcher.prepare_ruleset_event(
        ckpt=checkpoint,
        result=event,
        actor_id="account_owner",
    )
    with pytest.raises(OneStarTransactionError, match="cannot be broadcast"):
        await dispatcher.prepare_ruleset_event(
            ckpt=checkpoint,
            result=event,
            actor_id="account_owner",
        )
