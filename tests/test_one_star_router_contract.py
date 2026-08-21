"""Focused One-Star router schema and prompt-cache contracts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.engine.prompt_manager import PromptManager
from app.engine.one_star_adapter import OneStarTransactionError
from app.engine.turn_loop_dispatcher import (
    LLMDispatcher,
    _build_router_context,
    _router_history_record,
    _router_ruleset_template_vars,
    _validate_one_star_cat_ii_transaction,
    _validate_one_star_guide_routing,
    _validate_one_star_pending_response_routing,
    _validate_one_star_tutorial_routing,
)
from app.llm.client import LLMClient, _openai_strict_json_schema
from app.schemas.one_star import (
    ClosedOneStarEventRouterOutput,
    OneStarEventRouterOutput,
    ONE_STAR_RULESET_ID,
    OneStarTransaction,
)
from app.schemas.state import OpenCatIIEvent
from app.engine.turn_loop_contracts import format_actor_submission
from app.schemas.event_router import ObserverEntry
from app.schemas.events import ObservableFact
from tests.support.factories import (
    character_record,
    checkpoint,
    llm_response,
    router_output,
)


def _one_star_output(*, rationale: str = "router test") -> OneStarEventRouterOutput:
    data = router_output(facts=[], observer_ids=[]).model_dump()
    data["decision_rationale"] = rationale
    data["one_star_transaction"] = {"present": False, "operations": []}
    return OneStarEventRouterOutput.model_validate(data)


def _closed_one_star_output() -> ClosedOneStarEventRouterOutput:
    return ClosedOneStarEventRouterOutput.model_validate(
        _one_star_output().model_dump()
    )


def _pending_selection_output(
    *,
    event_id: str,
    requires_responders: bool,
    target_id: str = "",
) -> OneStarEventRouterOutput:
    data = router_output(
        event_id=event_id,
        requires_responders=requires_responders,
        required_responders=["pip"] if requires_responders else [],
        observer_ids=["alice", "pip"],
    ).model_dump(mode="json")
    data["one_star_transaction"] = {
        "present": True,
        "operations": [{
            "operation": "pending_open",
            "pending": {
                "operation_id": "deployment_1",
                "kind": "deployment",
                "participant_ids": ["pip"],
                "target_id": target_id,
                "destination": "tower_floor_1",
                "opened_at_s": 0,
            },
        }],
    }
    return OneStarEventRouterOutput.model_validate(data)


@pytest.mark.parametrize(
    "response_model",
    [OneStarEventRouterOutput, ClosedOneStarEventRouterOutput],
)
def test_one_star_provider_schema_contains_only_closed_objects(response_model):
    schema = _openai_strict_json_schema(response_model)
    invalid_objects: list[tuple[str, ...]] = []
    required_mismatches: list[tuple[str, ...]] = []
    unsupported_composition: list[tuple[str, ...]] = []

    def walk(node, path=()):
        if isinstance(node, dict):
            for keyword in ("oneOf", "allOf", "not", "discriminator"):
                if keyword in node:
                    unsupported_composition.append(path + (keyword,))
            if node.get("type") == "object":
                if node.get("additionalProperties") is not False:
                    invalid_objects.append(path)
                properties = node.get("properties")
                if isinstance(properties, dict) and node.get("required") != list(properties):
                    required_mismatches.append(path)
            for key, value in node.items():
                walk(value, path + (str(key),))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path + (str(index),))

    walk(schema)

    assert invalid_objects == []
    assert required_mismatches == []
    assert unsupported_composition == []


def _one_star_checkpoint():
    ckpt = checkpoint(characters=[
        character_record("alice", is_playable=True),
        character_record("pip"),
    ])
    ckpt.session.config.settings.ruleset_id = ONE_STAR_RULESET_ID
    return ckpt


def _dispatcher(*responses):
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(side_effect=[
        llm_response(response, content="{}") for response in responses
    ])
    return LLMDispatcher(client, PromptManager("app/prompts")), client


def _stub_one_star_router_context(monkeypatch):
    from app.engine import one_star_router_context

    monkeypatch.setattr(
        one_star_router_context,
        "render_one_star_router_static_config",
        lambda *_args, **_kwargs: "<one_star_rules_config>\nmax_batch=5\n</one_star_rules_config>",
    )
    monkeypatch.setattr(
        one_star_router_context,
        "render_one_star_router_ledger",
        lambda *_args, **_kwargs: "<one_star_current_ledger>\nGold: 34\n</one_star_current_ledger>",
    )


def _stub_local_one_star_account(monkeypatch, ckpt):
    from app.engine import one_star_adapter

    account = SimpleNamespace(
        config=SimpleNamespace(
            lobby_id="local",
            lobby_location_label="lobby",
            operation_requirements={},
        ),
        state=SimpleNamespace(guide_character_ids=[]),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_account",
        lambda _ckpt: (ckpt.characters[0], account),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_hero",
        lambda _character: SimpleNamespace(owner_lobby_id="local"),
    )


def test_one_star_transaction_is_required_and_has_an_explicit_empty_shape():
    data = _one_star_output().model_dump()
    assert data["one_star_transaction"] == {"present": False, "operations": []}

    data.pop("one_star_transaction")
    with pytest.raises(ValidationError, match="one_star_transaction"):
        OneStarEventRouterOutput.model_validate(data)


def test_one_star_router_schema_is_used_for_fresh_and_cat_ii_routes(monkeypatch):
    _stub_one_star_router_context(monkeypatch)
    fresh = _one_star_output()
    dispatcher, client = _dispatcher(fresh)

    result = asyncio.run(dispatcher.route_intention(
        ckpt=_one_star_checkpoint(),
        actor_id="alice",
        intention="I ask Pip a question.",
    ))

    assert result is fresh
    assert client.complete.await_args.kwargs["response_model"] is OneStarEventRouterOutput

    cat_ii = OpenCatIIEvent(
        event_id="evt_open",
        initiator_id="alice",
        initiator_intention="I reach for Pip's letter.",
        required_responders=["pip"],
        collected_intentions={"pip": "I pull it away."},
    )
    closed = _one_star_output()
    dispatcher, client = _dispatcher(closed)
    result = asyncio.run(dispatcher.route_intention(
        ckpt=_one_star_checkpoint(),
        actor_id="pip",
        intention="I pull it away.",
        cat_ii_event=cat_ii,
    ))

    assert result is closed
    assert client.complete.await_args.kwargs["response_model"] is OneStarEventRouterOutput


def test_invalid_embodied_selection_is_retried_before_router_history(
    monkeypatch,
):
    _stub_one_star_router_context(monkeypatch)
    ckpt = _one_star_checkpoint()
    _stub_local_one_star_account(monkeypatch, ckpt)
    invalid = _pending_selection_output(
        event_id="invalid_selection",
        requires_responders=False,
    )
    corrected = _pending_selection_output(
        event_id="corrected_selection",
        requires_responders=True,
    )
    dispatcher, client = _dispatcher(invalid, corrected)

    result = asyncio.run(dispatcher.route_intention(
        ckpt=ckpt,
        actor_id="alice",
        intention="Send Pip through the Tower gate.",
    ))

    assert result is corrected
    assert client.complete.await_count == 2
    correction_messages = client.complete.await_args_list[1].kwargs["messages"]
    assert correction_messages[-2] == {
        "role": "assistant",
        "content": invalid.model_dump_json(),
    }
    assert "must open Cat II" in correction_messages[-1]["content"]
    assert "read-only System or status inspection" in (
        correction_messages[-1]["content"]
    )
    assert "Send Pip through the Tower gate." in correction_messages[-3]["content"]
    stored_history = "\n".join(
        str(message.content) for message in ckpt.session_conversation
    )
    assert "corrected_selection" in stored_history
    assert "invalid_selection" not in stored_history


def test_invalid_deployment_target_is_corrected_before_cat_ii_history(
    monkeypatch,
):
    _stub_one_star_router_context(monkeypatch)
    ckpt = _one_star_checkpoint()
    _stub_local_one_star_account(monkeypatch, ckpt)
    invalid = _pending_selection_output(
        event_id="invalid_deployment_target",
        requires_responders=True,
        target_id="alice",
    )
    corrected = _pending_selection_output(
        event_id="corrected_deployment_target",
        requires_responders=True,
    )
    dispatcher, client = _dispatcher(invalid, corrected)

    result = asyncio.run(dispatcher.route_intention(
        ckpt=ckpt,
        actor_id="alice",
        intention="Deploy Pip to Floor 1.",
    ))

    assert result is corrected
    assert client.complete.await_count == 2
    correction = client.complete.await_args_list[1].kwargs["messages"][-1]
    assert "deployment has no separate target Hero" in correction["content"]
    stored_history = "\n".join(
        str(message.content) for message in ckpt.session_conversation
    )
    assert "corrected_deployment_target" in stored_history
    assert "invalid_deployment_target" not in stored_history


def test_repeated_invalid_embodied_selection_restores_router_snapshot(
    monkeypatch,
):
    _stub_one_star_router_context(monkeypatch)
    ckpt = _one_star_checkpoint()
    _stub_local_one_star_account(monkeypatch, ckpt)
    first = _pending_selection_output(
        event_id="invalid_selection_1",
        requires_responders=False,
    )
    second = _pending_selection_output(
        event_id="invalid_selection_2",
        requires_responders=False,
    )
    dispatcher, client = _dispatcher(first, second)
    before = ckpt.model_dump(mode="json")

    with pytest.raises(ValueError, match="remained invalid after one correction"):
        asyncio.run(dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="Send Pip through the Tower gate.",
        ))

    assert client.complete.await_count == 2
    assert ckpt.model_dump(mode="json") == before


def test_one_star_continuation_uses_the_closed_schema(monkeypatch):
    _stub_one_star_router_context(monkeypatch)
    prior = _one_star_output()
    continuation = _closed_one_star_output()
    dispatcher, client = _dispatcher(continuation)

    result = asyncio.run(dispatcher.route_continuation(
        ckpt=_one_star_checkpoint(),
        actor_id="alice",
        prior_result=prior,
    ))

    assert result is continuation
    assert (
        client.complete.await_args.kwargs["response_model"]
        is ClosedOneStarEventRouterOutput
    )


def test_one_star_repair_accepts_only_the_transaction_shape(monkeypatch):
    _stub_one_star_router_context(monkeypatch)
    repaired_transaction = OneStarTransaction(present=False, operations=[])
    dispatcher, client = _dispatcher(repaired_transaction)

    repaired = asyncio.run(dispatcher._repair_one_star_transaction(
        ckpt=_one_star_checkpoint(),
        result=_one_star_output(),
        actor_id="alice",
        validation_error="insufficient Gold",
    ))

    assert repaired is repaired_transaction
    assert client.complete.await_args.kwargs["response_model"] is OneStarTransaction
    repair_packet = client.complete.await_args.kwargs["messages"][-1]["content"]
    assert "one_star_transaction_repair" in repair_packet
    assert "Candidate transaction" in repair_packet
    assert 'deployment uses participant_ids' in repair_packet
    assert 'target_id=""' in repair_packet
    assert "change the offending field" in repair_packet
    assert "canonical_event" not in repair_packet


def test_one_star_spawn_activation_overlap_fails_before_materialization():
    data = _one_star_output().model_dump(mode="json")
    data["spawn"] = [{
        "character_id": "new_hero",
        "seed": {
            "role": "baker",
            "reason": "cold light deposits a stranger",
            "location": "niflheim_lobby",
            "objectives": ["find an exit"],
            "knowledge_tier": 1,
        },
    }]
    data["activate"] = [{
        "character_id": "new_hero",
        "location_label": "niflheim_lobby",
    }]
    result = OneStarEventRouterOutput.model_validate(data)
    dispatcher, _client = _dispatcher()
    dispatcher.materialize_spawns = AsyncMock()

    with pytest.raises(OneStarTransactionError, match="both spawn and activate"):
        asyncio.run(dispatcher.prepare_ruleset_event(
            ckpt=_one_star_checkpoint(),
            result=result,
            actor_id="alice",
        ))

    dispatcher.materialize_spawns.assert_not_awaited()


def test_default_and_dnd_ruleset_addons_stay_isolated(monkeypatch):
    prompt_mgr = PromptManager("app/prompts")
    _stub_one_star_router_context(monkeypatch)

    default = _router_ruleset_template_vars(
        prompt_mgr,
        ruleset_id="narrative",
        dnd_fresh=False,
    )
    dnd = _router_ruleset_template_vars(
        prompt_mgr,
        ruleset_id="dnd5e_basic",
        dnd_fresh=True,
    )
    one_star = _router_ruleset_template_vars(
        prompt_mgr,
        ruleset_id=ONE_STAR_RULESET_ID,
        dnd_fresh=False,
        ckpt=_one_star_checkpoint(),
    )

    assert "Category II" in default["router_ruleset_addon"]
    assert "dnd_combat_start" in dnd["router_ruleset_addon"]
    assert "one_star_transaction" not in default["router_ruleset_addon"]
    assert "one_star_transaction" in one_star["router_ruleset_addon"]


def test_one_star_state_stays_in_the_volatile_router_message_tail():
    prompt_mgr = PromptManager("app/prompts")
    messages = prompt_mgr.render_messages(
        "event_router",
        setting_summary="setting",
        world_lore="lore",
        world_rules="rules",
        hidden_lore="none",
        hidden_facts="none",
        acting_character_id="alice",
        initial_roster_block="",
        engine_state_updates_block="",
        router_ruleset_addon=prompt_mgr.render(
            "event_router_ruleset_one_star",
            one_star_static_config="<one_star_rules_config>\nmax_batch=5\n</one_star_rules_config>",
        ),
        one_star_state_section=(
            "<one_star_state>\nGold: 34\nPending operation: none\n"
            "</one_star_state>"
        ),
        router_input_block="submitted action",
    )

    system, user = messages
    assert "one_star_transaction" in system["content"]
    assert "max_batch=5" in system["content"]
    assert "Gold: 34" not in system["content"]
    assert "max_batch=5" not in user["content"]
    assert "Gold: 34" in user["content"]
    assert "submitted action" in user["content"]


def test_one_star_router_context_requests_the_full_ledger_tail(monkeypatch):
    from app.engine import one_star_router_context

    monkeypatch.setattr(
        one_star_router_context,
        "render_one_star_router_ledger",
        lambda *_args, **_kwargs: "<one_star_current_ledger>\nGold: 34\n</one_star_current_ledger>",
    )
    context = _build_router_context(
        _one_star_checkpoint(),
        "alice",
        include_engine_state_updates=False,
    )

    assert "Gold: 34" in context["one_star_state_section"]


def test_account_owner_lobby_mutation_requires_scoped_guide_delivery(monkeypatch):
    from app.engine import one_star_adapter

    ckpt = _one_star_checkpoint()
    ckpt.characters.append(character_record("iselle", location="niflheim_lobby"))
    result = _one_star_output()
    result.one_star_transaction = OneStarTransaction.model_construct(
        present=True,
        operations=[SimpleNamespace(operation="summon")],
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_account",
        lambda _: (
            ckpt.characters[0],
            SimpleNamespace(
                config=SimpleNamespace(
                    lobby_id="niflheim",
                    lobby_location_label="niflheim_lobby",
                    operation_requirements={},
                ),
                state=SimpleNamespace(guide_character_ids=["iselle"]),
            ),
        ),
    )

    with pytest.raises(ValueError, match="mediated observer"):
        _validate_one_star_guide_routing(ckpt, actor_id="alice", result=result)

    result.observers.append(ObserverEntry(
        character_id="iselle",
        observation_level="d",
        routing_role="observe_only",
    ))
    result.canonical_event.observable_facts.append(ObservableFact.only(
        "A System receipt records the lobby action.",
        ["iselle"],
    ))

    _validate_one_star_guide_routing(ckpt, actor_id="alice", result=result)


@pytest.mark.parametrize("lifecycle_kind", ["dormant", "activate"])
def test_account_owner_local_hero_lifecycle_reaches_guide(
    monkeypatch,
    lifecycle_kind,
):
    from app.engine import one_star_adapter

    ckpt = _one_star_checkpoint()
    pip = next(character for character in ckpt.characters if character.character_id == "pip")
    pip.location = "niflheim_lobby"
    ckpt.characters.append(character_record("iselle", location="niflheim_lobby"))
    base = router_output(
        observer_ids=["alice"],
        dormant=["pip"] if lifecycle_kind == "dormant" else [],
        activate=(
            [{"character_id": "pip", "location_label": "niflheim_lobby"}]
            if lifecycle_kind == "activate"
            else []
        ),
        facts=[],
    ).model_dump(mode="json")
    base["one_star_transaction"] = {"present": False, "operations": []}
    result = OneStarEventRouterOutput.model_validate(base)
    account = SimpleNamespace(
        config=SimpleNamespace(
            lobby_id="niflheim",
            lobby_location_label="niflheim_lobby",
            operation_requirements={},
        ),
        state=SimpleNamespace(guide_character_ids=["iselle"]),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_account",
        lambda _: (ckpt.characters[0], account),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_hero",
        lambda character: (
            SimpleNamespace(owner_lobby_id="niflheim")
            if character is not None and character.character_id == "pip"
            else None
        ),
    )

    with pytest.raises(ValueError, match="mediated observer"):
        _validate_one_star_guide_routing(
            ckpt,
            actor_id="alice",
            result=result,
        )

    result.observers.append(ObserverEntry(
        character_id="iselle",
        observation_level="d",
        routing_role="observe_only",
    ))
    result.canonical_event.observable_facts.append(ObservableFact.only(
        "The System reports the roster change.",
        ["iselle"],
    ))
    _validate_one_star_guide_routing(
        ckpt,
        actor_id="alice",
        result=result,
    )


def test_cat_ii_open_rejects_one_star_mechanical_side_effects():
    data = router_output(
        requires_responders=True,
        required_responders=["pip"],
        observer_ids=["alice", "pip"],
    ).model_dump(mode="json")
    pending_open = {
        "operation": "pending_open",
        "pending": {
            "operation_id": "deployment_1",
            "kind": "deployment",
            "participant_ids": ["pip"],
            "target_id": "",
            "destination": "tower_floor_1",
            "opened_at_s": 0,
        },
    }
    terminal_delta = {
        "operation": "hero_delta",
        "hero_id": "pip",
        "hp_current": 0,
        "hp_max": None,
        "level": None,
        "experience_delta": 0,
        "stats_delta": [],
        "equipment_add": [],
        "equipment_remove_ids": [],
        "skills_add": [],
        "skills_remove_ids": [],
        "equipment_durability": [],
        "skill_rank_updates": [],
        "conditions": None,
        "persistent_injuries": None,
        "terminal_action": "death",
        "death_cause": "forced before response",
    }
    data["one_star_transaction"] = {
        "present": True,
        "operations": [pending_open, terminal_delta],
    }
    result = OneStarEventRouterOutput.model_validate(data)

    with pytest.raises(ValueError, match="only one pending_open"):
        _validate_one_star_cat_ii_transaction(result)

    result.one_star_transaction = OneStarTransaction.model_validate({
        "present": True,
        "operations": [pending_open],
    })
    _validate_one_star_cat_ii_transaction(result)


def test_embodied_selection_requires_cat_ii_from_every_affected_hero(
    monkeypatch,
):
    from app.engine import one_star_adapter

    ckpt = _one_star_checkpoint()
    ckpt.characters.append(character_record("bob"))
    data = router_output(
        requires_responders=True,
        required_responders=["pip"],
        agent_ids=["bob"],
        observer_ids=["alice", "pip", "bob"],
    ).model_dump(mode="json")
    data["one_star_transaction"] = {
        "present": True,
        "operations": [{
            "operation": "pending_open",
            "pending": {
                "operation_id": "synthesis_1",
                "kind": "synthesis",
                "participant_ids": ["pip", "bob"],
                "target_id": "alice",
                "destination": "synthesis_room",
                "opened_at_s": 0,
            },
        }],
    }
    result = OneStarEventRouterOutput.model_validate(data)
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_account",
        lambda _: (
            ckpt.characters[0],
            SimpleNamespace(config=SimpleNamespace(lobby_id="local")),
        ),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_hero",
        lambda character: SimpleNamespace(owner_lobby_id="local"),
    )

    with pytest.raises(ValueError, match="bob"):
        _validate_one_star_pending_response_routing(
            ckpt,
            actor_id="alice",
            result=result,
        )

    result.required_responders.append("bob")
    _validate_one_star_pending_response_routing(
        ckpt,
        actor_id="alice",
        result=result,
    )

    result.requires_responders = False
    result.required_responders = []
    with pytest.raises(ValueError, match="must open Cat II"):
        _validate_one_star_pending_response_routing(
            ckpt,
            actor_id="alice",
            result=result,
        )


def test_tutorial_delivery_requires_direct_visible_observation():
    data = router_output(
        observer_ids=["pip"],
        facts=[ObservableFact.only("Iselle explains the gate.", ["pip"])],
    ).model_dump(mode="json")
    data["one_star_transaction"] = {
        "present": True,
        "operations": [{
            "operation": "tutorial_delivery",
            "tutorial_key": "tower_gate",
            "delivered_to_ids": ["pip"],
        }],
    }
    result = OneStarEventRouterOutput.model_validate(data)
    _validate_one_star_tutorial_routing(result)

    result.observers[0].observation_level = "i"
    with pytest.raises(ValueError, match="direct observer"):
        _validate_one_star_tutorial_routing(result)

    result.observers[0].observation_level = "d"
    result.canonical_event.observable_facts = [
        ObservableFact.only("Alice alone hears something else.", ["alice"])
    ]
    with pytest.raises(ValueError, match="visible"):
        _validate_one_star_tutorial_routing(result)


def test_one_star_router_projections_split_static_rules_from_live_ledger(
    monkeypatch,
):
    from app.engine import one_star_router_context

    def cost(*, gold=0, gems=0, building_resources=0, materials=None):
        return SimpleNamespace(
            gold=gold,
            gems=gems,
            building_resources=building_resources,
            materials=materials or {},
        )

    config = SimpleNamespace(
        lobby_id="niflheim",
        lobby_location_label="Niflheim Lobby",
        starting_lobby_floor=1,
        starting_capacity=20,
        starting_resources=cost(gold=40, gems=5, building_resources=3),
        max_summon_batch=5,
        summon_pools={
            "premium": SimpleNamespace(
                cost=cost(gems=5),
                minimum_birth_stars=2,
                maximum_birth_stars=5,
                star_weights={2: 7500, 3: 2300, 4: 175, 5: 25},
                eligible_existing_ids=["veil"],
                fresh_generation_allowed=True,
                usage="standard",
            ),
        },
        catalogue={
            "synthesis_chamber_i": SimpleNamespace(
                kind="facility_build",
                cost=cost(gold=8, building_resources=2),
                inventory_item_id="",
                facility_id="synthesis_chamber",
                target_level=1,
                required_cleared_floor=0,
                required_lobby_floor=1,
                resulting_lobby_floor=0,
                resulting_capacity=0,
                research_key="",
                research_level=0,
            ),
        },
        star_level_caps={1: 10, 2: 20},
        hero_constraints=SimpleNamespace(
            minimum_hp_max=1,
            maximum_hp_max=100,
            maximum_xp=1000,
            maximum_stat_value=100,
            maximum_equipment_entries=10,
            maximum_skill_entries=10,
        ),
        deployment_stamina_cost=1,
        maximum_stamina=5,
        stamina_recovery_seconds=1800,
        floor_rewards={1: cost(gold=4, building_resources=1)},
        repeat_gold_numerator=1,
        repeat_gold_denominator=4,
        repeat_gold_minimum=1,
        promotion_cost=cost(gold=16, building_resources=3),
        operation_requirements={
            "deployment": SimpleNamespace(
                facility_id="tower_gate",
                required_location="tower_gate",
            ),
            "synthesis": SimpleNamespace(
                facility_id="synthesis_chamber",
                required_location="synthesis_chamber",
            ),
            "promotion": SimpleNamespace(
                facility_id="promotion_chamber",
                required_location="promotion_chamber",
            ),
        },
        lobby_return_healing=True,
        hero_system_visibility_research_key="hero_reaction_research",
    )
    mission = SimpleNamespace(
        mission_id="tower_1",
        floor=1,
        destination="tower_floor_1",
        started_at_s=20,
        deadline_at_s=320,
        completion_declaration="reach the exit",
        failure_declaration="all party members fall",
        party_ids=["pip"],
        formation_labels=[SimpleNamespace(character_id="pip", label="front")],
        counters=[SimpleNamespace(counter_id="exit", current=0, target=1)],
    )
    state = SimpleNamespace(
        resources=cost(gold=34, gems=5, building_resources=3, materials={"stone": 1}),
        inventory={"rope": 1},
        facilities={"training_camp": 1},
        research_levels={"hero_reaction_research": 1},
        lobby_floor=1,
        capacity=20,
        highest_unlocked_floor=2,
        highest_cleared_floor=1,
        stamina_current=4,
        stamina_recovery_anchor_s=20,
        active_master_feed_id="pip",
        guide_character_ids=["iselle"],
        system_observer_ids=["iselle"],
        tutorial_deliveries={"summoning": ["pip"]},
        active_mission=mission,
        pending_operation=SimpleNamespace(
            operation_id="op_1",
            kind="deployment",
            participant_ids=["pip"],
            target_id="",
            destination="tower_gate",
            opened_at_s=30,
        ),
    )
    hero = SimpleNamespace(
        owner_lobby_id="niflheim",
        birth_stars=1,
        current_stars=1,
        level=1,
        experience_points=2,
        hp_current=4,
        hp_max=6,
        innate_system_sight=False,
        stats={"strength": 2},
        equipment=[],
        skills=[],
        conditions=["bruised"],
        persistent_injuries=[],
        terminal_cause="",
        hidden_capabilities={"potential": "locked"},
        private_potential="unseen",
    )
    ckpt = _one_star_checkpoint()
    ckpt.session.leading_at_s = 60
    monkeypatch.setattr(one_star_router_context, "is_one_star_checkpoint", lambda _: True)
    monkeypatch.setattr(
        one_star_router_context,
        "load_one_star_account",
        lambda _: (ckpt.characters[0], SimpleNamespace(config=config, state=state)),
    )
    monkeypatch.setattr(
        one_star_router_context,
        "load_one_star_hero",
        lambda character: hero if character.character_id == "pip" else None,
    )
    monkeypatch.setattr(
        one_star_router_context,
        "one_star_summon_draw_preview",
        lambda _checkpoint, _pool_id: (
            SimpleNamespace(slot=1, birth_stars=2, existing_character_id="veil"),
            SimpleNamespace(slot=2, birth_stars=3, existing_character_id=""),
        ),
    )

    static = one_star_router_context.render_one_star_router_static_config(ckpt)
    dynamic = one_star_router_context.render_one_star_router_ledger(ckpt)

    assert "synthesis_chamber_i" in static
    assert "birth_stars=2-5" in static
    assert "birth_star_rates[2=75%,3=23%,4=1.75%,5=0.25%]" in static
    assert "repeat_clear_gold" in static
    assert "hero_constraints" in static
    assert "gold=34" not in static
    assert "gold=34" in dynamic
    assert "research_levels" in dynamic
    assert "mission_counters" in dynamic
    assert "pending_operation" in dynamic
    assert "hidden_capabilities" in dynamic
    assert "eligible_unowned_reserves" in dynamic
    assert "source=reserve:veil" in dynamic
    assert "source=fresh" in dynamic
    assert "summon_draw_counters" not in dynamic
    assert "one-star-gacha" not in dynamic
    mission_header = dynamic.split("mission_completion=", 1)[0]
    assert "status=" not in mission_header


def test_local_hero_cull_is_rejected_from_generic_lifecycle(monkeypatch):
    from app.engine import one_star_adapter

    ckpt = _one_star_checkpoint()
    account = SimpleNamespace(
        config=SimpleNamespace(
            lobby_id="niflheim",
            lobby_location_label="niflheim_lobby",
        ),
            state=SimpleNamespace(
                applied_event_fingerprints={},
                active_mission=None,
                pending_operation=None,
            ),
    )
    monkeypatch.setattr(one_star_adapter, "is_one_star_checkpoint", lambda _: True)
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_account",
        lambda _: (ckpt.characters[0], account),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_hero",
        lambda character: (
            SimpleNamespace(owner_lobby_id="niflheim")
            if character.character_id == "pip" else None
        ),
    )

    with pytest.raises(
        one_star_adapter.OneStarTransactionError,
        match="culls belong only in one_star_transaction",
    ):
        one_star_adapter.prepare_one_star_transaction(
            ckpt,
            event_id="evt_local_hero_cull",
            transaction=OneStarTransaction(present=False, operations=[]),
            generic_culled_character_ids=["pip"],
        )


def test_one_star_history_omits_legacy_mission_status_only_for_one_star():
    rationale = (
        "why\nmission_status: id=floor_one; state=active; "
        "completion=clear; progress=0; failure=dead; timing=untimed"
    )

    one_star_record = _router_history_record(
        acting_character_id="alice",
        result=_one_star_output(rationale=rationale),
    )
    closed_one_star_record = _router_history_record(
        acting_character_id="alice",
        result=ClosedOneStarEventRouterOutput.model_validate({
            **_closed_one_star_output().model_dump(),
            "decision_rationale": rationale,
        }),
    )
    generic_record = _router_history_record(
        acting_character_id="alice",
        result=router_output(facts=[], observer_ids=[]).model_copy(
            update={"decision_rationale": rationale},
        ),
    )

    assert "mission_status" not in one_star_record
    assert "mission_status" not in closed_one_star_record
    assert "mission_status id=floor_one" in generic_record


def test_human_and_agent_submissions_share_the_same_router_envelope():
    human_submission = format_actor_submission("alice", "I open the gate.")
    agent_submission = format_actor_submission("alice", "I open the gate.")

    assert human_submission == agent_submission
    assert "player" not in human_submission
    assert "agent" not in human_submission


def test_one_star_prompt_addon_does_not_leak_runtime_implementation_terms():
    addon = (PromptManager("app/prompts").prompts_dir / "event_router_ruleset_one_star.txt").read_text()

    for forbidden in ("engine", "pipeline", "dispatcher", "SDK", "API", "Python"):
        assert forbidden not in addon
