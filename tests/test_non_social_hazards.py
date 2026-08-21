from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.engine.character_agent import CharacterAgent
from app.engine.character_manager import CharacterManager
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop import (
    _filter_routed_agents_for_dispatch,
    broadcast_event,
    run_beat,
)
from app.schemas.characters import (
    CharacterRecord,
    CharacterStatus,
    FictionalEntityKind,
    PrivateState,
    is_non_social_hazard,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.events import ObservableFact
from app.schemas.state import SessionState
from tests.support.factories import (
    InstanceFakeDispatcher,
    gatehouse_checkpoint,
    router_output,
)


def _hazard(
    *,
    status: CharacterStatus = CharacterStatus.active,
) -> CharacterRecord:
    return CharacterRecord(
        character_id="clockwork_gate",
        name="the Clockwork Gate",
        entity_kind=FictionalEntityKind.hazard,
        status=status,
        location="vault_threshold",
    )


def test_character_records_default_to_deliberate_character_agency() -> None:
    character = CharacterRecord(character_id="pip", name="Pip")

    assert character.entity_kind == FictionalEntityKind.character
    assert is_non_social_hazard(character) is False
    assert is_non_social_hazard(_hazard()) is True


def test_non_social_hazard_cannot_be_a_playable_seat() -> None:
    with pytest.raises(
        ValidationError,
        match="non-social hazards cannot be playable seats",
    ):
        CharacterRecord(
            character_id="clockwork_gate",
            name="the Clockwork Gate",
            entity_kind=FictionalEntityKind.hazard,
            is_playable=True,
        )


def test_non_social_hazard_cannot_carry_character_interior_state() -> None:
    with pytest.raises(
        ValidationError,
        match="non-social hazards cannot carry character interior state",
    ):
        CharacterRecord(
            character_id="clockwork_gate",
            name="the Clockwork Gate",
            entity_kind=FictionalEntityKind.hazard,
            private_state=PrivateState(
                current_objectives=["outwit the intruders"],
            ),
        )


def test_non_social_hazard_cannot_carry_character_portrayal_direction() -> None:
    with pytest.raises(
        ValidationError,
        match="cannot carry character portrayal direction",
    ):
        CharacterRecord(
            character_id="clockwork_gate",
            name="the Clockwork Gate",
            entity_kind=FictionalEntityKind.hazard,
            personality="Play it as a taciturn guard.",
        )


@pytest.mark.parametrize("field", ["backstory", "known_context"])
def test_non_social_hazard_cannot_carry_character_knowledge_fields(
    field: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match="cannot carry character knowledge fields",
    ):
        CharacterRecord(
            character_id="clockwork_gate",
            name="the Clockwork Gate",
            entity_kind=FictionalEntityKind.hazard,
            **{field: "The gate remembers its maker's command."},
        )


def test_non_social_hazard_never_reaches_character_agent() -> None:
    client = MagicMock()
    client.complete = AsyncMock()
    agent = CharacterAgent(client, PromptManager("app/prompts"))
    checkpoint = CheckpointFile(
        session=SessionState(session_id="hazard-agent-contract"),
        characters=[_hazard()],
    )

    with pytest.raises(
        ValueError,
        match="no character-agent intention turn",
    ):
        asyncio.run(agent.turn(_hazard(), checkpoint))
    with pytest.raises(
        ValueError,
        match="no character-agent perception turn",
    ):
        asyncio.run(agent.perceive(_hazard(), checkpoint))

    client.complete.assert_not_awaited()


def test_routed_agent_filter_drops_hazard_but_keeps_social_observer() -> None:
    checkpoint = gatehouse_checkpoint(bindings={"alice": "u1"})
    checkpoint.characters.append(_hazard())
    event = router_output(
        event_kind="beat_continues",
        agent_ids=["pip"],
        observer_ids=["alice", "pip", "clockwork_gate"],
    )

    assert _filter_routed_agents_for_dispatch(
        checkpoint,
        ["clockwork_gate", "pip"],
        event=event,
    ) == ["pip"]


def test_hazard_observe_only_preserves_canonical_state_without_agent_inbox() -> None:
    checkpoint = gatehouse_checkpoint(bindings={"alice": "u1"})
    hazard = _hazard()
    checkpoint.characters.append(hazard)
    event = router_output(
        event_id="evt_gate_trigger",
        event_kind="state_change",
        observer_ids=["alice", "clockwork_gate"],
        facts=[ObservableFact.all(
            "The clockwork_gate registers the pressure plate and its blade "
            "track begins to turn."
        )],
    )

    visible_humans = broadcast_event(
        checkpoint,
        event,
        actor_id="alice",
    )

    assert visible_humans == ["alice"]
    assert checkpoint.canonical_events == [event]
    assert hazard.pending_observations == []
    assert hazard.last_agent_turn_at_s is None


def test_hazard_next_output_fails_before_event_persistence() -> None:
    checkpoint = gatehouse_checkpoint(bindings={"alice": "u1"})
    checkpoint.characters.append(_hazard())
    event = router_output(
        event_kind="beat_continues",
        agent_ids=["clockwork_gate"],
        observer_ids=["alice", "clockwork_gate"],
    )

    with pytest.raises(
        RuntimeError,
        match="character-owned work to a non-social hazard",
    ):
        broadcast_event(checkpoint, event, actor_id="alice")

    assert checkpoint.canonical_events == []
    assert checkpoint.session.render_buffers == {}


def test_hazard_required_responder_fails_before_cat_ii_opens() -> None:
    checkpoint = gatehouse_checkpoint(bindings={"alice": "u1"})
    checkpoint.characters.append(_hazard())
    dispatcher = InstanceFakeDispatcher()
    dispatcher.queue_route(router_output(
        event_kind="cat_ii_open",
        requires_responders=True,
        required_responders=["clockwork_gate"],
        observer_ids=["alice", "clockwork_gate"],
    ))

    with pytest.raises(
        RuntimeError,
        match="required_responder=clockwork_gate",
    ):
        asyncio.run(run_beat(
            ckpt=checkpoint,
            dispatcher=dispatcher,
            actor_id="alice",
            intention="I jam my arm into the moving gate.",
        ))

    assert checkpoint.canonical_events == []
    assert checkpoint.session.open_cat_ii_events == []
    assert dispatcher.agent_calls == []
    assert dispatcher.materialize_calls == []


def test_hazard_activation_pattern_and_defeat_keep_one_canonical_identity() -> None:
    checkpoint = gatehouse_checkpoint(bindings={"alice": "u1"})
    hazard = _hazard(status=CharacterStatus.dormant)
    checkpoint.characters.append(hazard)
    manager = CharacterManager()

    activation = router_output(
        event_id="evt_gate_wakes",
        event_kind="state_change",
        observer_ids=["alice", "clockwork_gate"],
        facts=[ObservableFact.all(
            "The clockwork_gate wakes with three measured iron clicks."
        )],
        activate=[{
            "character_id": "clockwork_gate",
            "location_label": "vault_threshold",
        }],
    )
    broadcast_event(checkpoint, activation, actor_id="alice")
    manager.apply_roster_updates(checkpoint, activation)
    assert hazard.status == CharacterStatus.active

    pattern = router_output(
        event_id="evt_gate_sweeps",
        event_kind="state_change",
        observer_ids=["alice", "clockwork_gate"],
        facts=[ObservableFact.all(
            "The clockwork_gate completes its established low-high-low sweep."
        )],
    )
    broadcast_event(checkpoint, pattern)

    defeated = router_output(
        event_id="evt_gate_breaks",
        event_kind="state_change",
        observer_ids=["alice", "clockwork_gate"],
        facts=[ObservableFact.all(
            "The clockwork_gate's axle shears and the mechanism goes still."
        )],
        cull=["clockwork_gate"],
    )
    broadcast_event(checkpoint, defeated, actor_id="alice")
    manager.apply_roster_updates(checkpoint, defeated)

    assert hazard.status == CharacterStatus.culled
    assert [event.event_id for event in checkpoint.canonical_events] == [
        "evt_gate_wakes",
        "evt_gate_sweeps",
        "evt_gate_breaks",
    ]
    assert hazard.pending_observations == []


def test_dnd_character_data_does_not_imply_non_social_hazard_kind() -> None:
    creature = CharacterRecord(
        character_id="animated_armor",
        name="Animated Armor",
        mechanics={"ruleset_id": "dnd5e", "armor_class": 18},
    )

    assert creature.entity_kind == FictionalEntityKind.character
    assert is_non_social_hazard(creature) is False


def test_one_star_warden_keeps_stable_public_and_visual_identity() -> None:
    checkpoint = CheckpointFile.model_validate_json(
        Path(
            "app/storage/stories/one_star_ascension_s1/ckpt_0000.json"
        ).read_text(encoding="utf-8")
    )
    warden = next(
        character
        for character in checkpoint.characters
        if character.character_id == "warden_of_the_eighth"
    )

    assert is_non_social_hazard(warden)
    assert warden.name
    assert warden.public_sheet.appearance
    assert warden.visuals.default_loadout
    assert warden.visuals.identity_reference_id
