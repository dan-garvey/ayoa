"""Offline integration contracts for fixed One-Star System results."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.checkpoint_manager import CheckpointManager
from app.engine.one_star_adapter import (
    load_one_star_account,
    load_one_star_hero,
)
from app.engine.one_star_projection import one_star_synthesis_authoritative_plan
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.llm.config import LLMConfig
from app.schemas.characters import (
    ActorRecord,
    CharacterRecord,
    CharacterStatus,
    PublicSheet,
)
from app.schemas.event_router import ClosedEventRouterOutput
from app.schemas.events import ObservableFact
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    OneStarAccountEnvelope,
)
from tests.support.factories import (
    llm_response,
    narrator_llm_response,
    router_output,
    text_llm_response,
)
from tests.test_one_star_projection import _checkpoint, _hero_state


def _command_checkpoint():
    checkpoint, owner, target, guide = _checkpoint()
    checkpoint.session.character_bindings = {owner.character_id: "user-1"}
    account = OneStarAccountEnvelope.model_validate(
        owner.mechanics[ONE_STAR_ACCOUNT_KEY]
    )
    account.state.pending_operation = None
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    donor_state = _hero_state()
    donor_state.equipment = []
    donor_state.skills = []
    donor = CharacterRecord(
        character_id="donor",
        name="Edric",
        location="lobby",
        public_sheet=PublicSheet(role="guard"),
        actor=ActorRecord(),
        mechanics={ONE_STAR_HERO_KEY: donor_state.model_dump(mode="json")},
    )
    checkpoint.characters.append(donor)
    return checkpoint, owner, target, guide, donor


def _closed_router_result(*, event_id: str = "system_synthesis"):
    return ClosedEventRouterOutput.model_validate(
        router_output(
            event_id=event_id,
            event_kind="state_change",
            observer_ids=["owner", "hero", "guide", "donor"],
            location_updates=[
                {
                    "character_id": character_id,
                    "location_label": "synthesis_room",
                }
                for character_id in ("donor", "hero", "guide")
            ],
            facts=[
                ObservableFact.all(
                    "donor says, 'I will not disappear.' The transparent barrier "
                    "holds donor in the synthesis chamber."
                ),
                ObservableFact.all(
                    "The System irreversibly synthesizes donor into hero while "
                    "owner watches through the live chamber camera."
                ),
            ],
            duration_s=2,
        ).model_dump(mode="python")
    )


def _orchestrator(tmp_path, checkpoint, responses):
    manager = CheckpointManager(str(tmp_path / "sessions"))
    manager.save(checkpoint)
    client = MagicMock()
    client.config = LLMConfig()
    client.complete = AsyncMock(side_effect=responses)
    orchestrator = Orchestrator(
        client,
        manager,
        PromptManager("app/prompts"),
    )
    return orchestrator, manager, client


@pytest.mark.asyncio
async def test_first_synthesis_collects_last_words_then_commits_fixed_result(
    tmp_path,
) -> None:
    checkpoint, owner, target, guide, donor = _command_checkpoint()
    routed = _closed_router_result()
    orchestrator, manager, client = _orchestrator(
        tmp_path,
        checkpoint,
        [
            text_llm_response("I will not disappear. (Edric is terrified.)"),
            llm_response(routed),
            narrator_llm_response("Edric's protest ends in synthesis-light."),
        ],
    )

    response = await orchestrator.process_authoritative_result(
        session_id=checkpoint.session.session_id,
        viewpoint_character_id=owner.character_id,
        plan_builder=lambda current: one_star_synthesis_authoritative_plan(
            current,
            owner.character_id,
            target_ref=target.character_id,
            source_refs=(donor.character_id,),
        ),
    )

    saved = manager.load_latest(checkpoint.session.session_id)
    saved_by_id = {character.character_id: character for character in saved.characters}
    account = load_one_star_account(saved)[1]
    event = saved.canonical_events[-1]
    assert response.output_text == "Edric's protest ends in synthesis-light."
    assert event.event_kind == "ruleset_resolution"
    assert event.next_output_character_ids == []
    assert account.state.pending_operation is None
    assert account.state.synthesis_resolution_count == 1
    assert saved_by_id[donor.character_id].status is CharacterStatus.culled
    assert saved_by_id[target.character_id].status is CharacterStatus.active
    assert (
        load_one_star_hero(saved_by_id[donor.character_id]).terminal_event_id
        == event.event_id
    )
    assert "one_star_update" in saved.session_conversation[-1].content
    assistant_content = saved.character_conversations[donor.character_id][-1].content
    assert assistant_content == [
        {
            "type": "text",
            "text": "I will not disappear. (Edric is terrified.)",
        }
    ]
    agent_messages = client.complete.await_args_list[0].kwargs["messages"]
    agent_user = next(
        message["content"]
        for message in reversed(agent_messages)
        if message["role"] == "user"
    )
    assert "transparent synthesis barrier" in agent_user
    assert "synthesis" in agent_user.casefold()
    router_messages = client.complete.await_args_list[1].kwargs["messages"]
    router_user = next(
        message["content"]
        for message in reversed(router_messages)
        if message["role"] == "user"
    )
    assert "character_contributions:" in router_user
    assert "I will not disappear." in router_user
    assert "human" not in router_user.casefold()
    assert "synthesis_resolution_count" not in router_user


@pytest.mark.asyncio
async def test_failed_authoritative_routing_rolls_back_drafts_and_ledger(
    tmp_path,
) -> None:
    checkpoint, owner, target, _guide, donor = _command_checkpoint()
    orchestrator, manager, _client = _orchestrator(
        tmp_path,
        checkpoint,
        [
            text_llm_response("No. (Edric braces.)"),
            RuntimeError("router unavailable"),
        ],
    )

    with pytest.raises(RuntimeError, match="router unavailable"):
        await orchestrator.process_authoritative_result(
            session_id=checkpoint.session.session_id,
            viewpoint_character_id=owner.character_id,
            plan_builder=lambda current: one_star_synthesis_authoritative_plan(
                current,
                owner.character_id,
                target_ref=target.character_id,
                source_refs=(donor.character_id,),
            ),
        )

    saved = manager.load_latest(checkpoint.session.session_id)
    saved_donor = next(
        character
        for character in saved.characters
        if character.character_id == donor.character_id
    )
    assert saved_donor.status is CharacterStatus.active
    assert saved.character_conversations.get(donor.character_id, []) == []
    assert load_one_star_account(saved)[1].state.synthesis_resolution_count == 0
    assert saved.canonical_events == []
