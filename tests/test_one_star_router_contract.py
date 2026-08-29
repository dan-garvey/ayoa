"""Focused One-Star router schema and prompt-cache contracts."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.engine.prompt_manager import PromptManager
from app.engine.one_star_adapter import OneStarTransactionError
from app.engine.turn_loop_dispatcher import (
    LLMDispatcher,
    _build_router_context,
    refresh_router_history_record,
    _router_history_record,
    _router_ruleset_template_vars,
    _one_star_transaction_for_result,
    _include_one_star_synthesis_guide_responders,
    _validate_one_star_cat_ii_transaction,
    _validate_one_star_guide_routing,
    _validate_one_star_pending_response_routing,
    _validate_one_star_standard_summon_guide_handoff,
    _validate_one_star_standard_summon_induction,
    _validate_one_star_tutorial_routing,
)
from app.llm.client import LLMClient, _openai_strict_json_schema
from app.schemas.one_star import (
    ClosedOneStarEventRouterOutput,
    OneStarEventRouterOutput,
    ONE_STAR_RULESET_ID,
    OneStarCost,
    OneStarOpeningRosterBoundPlayerActorSlot,
    OneStarOpeningRosterFixedSlot,
    OneStarOpeningRosterRandomExistingGradeSlot,
    OneStarOpeningRosterSummonPool,
    OneStarStandardSummonPool,
    OneStarStateUpdate,
    OneStarStateUpdateList,
    OneStarTransaction,
)
from app.schemas.conversation import ConversationMessage
from app.schemas.state import OpenCatIIEvent
from app.engine.turn_loop_contracts import format_actor_submission
from app.schemas.event_router import (
    EventRouterOutput,
    LocationUpdateSignal,
    ObserverEntry,
)
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
    data["state_updates"] = []
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
    details = ["participant=pip", "destination=tower_floor_1"]
    if target_id:
        details.append(f"target_id={target_id}")
    data["state_updates"] = [
        {
            "kind": "pending_open",
            "target_id": "deployment_1",
            "value": "deployment",
            "details": details,
        }
    ]
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
                if isinstance(properties, dict) and node.get("required") != list(
                    properties
                ):
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


def test_one_star_provider_schema_adds_only_one_compact_update_definition():
    generic = _openai_strict_json_schema(EventRouterOutput)
    one_star = _openai_strict_json_schema(OneStarEventRouterOutput)
    generic_defs = set(generic.get("$defs", {}))
    one_star_defs = set(one_star.get("$defs", {}))
    rendered = json.dumps(one_star, separators=(",", ":"))

    assert one_star_defs - generic_defs == {"OneStarStateUpdate"}
    assert len(rendered) < 6_000
    assert len(rendered) - len(json.dumps(generic, separators=(",", ":"))) < 1_000
    for adapter_private_type in (
        "OneStarTransaction",
        "OneStarHeroDeltaOperation",
        "OneStarMissionState",
        "OneStarEquipmentEntry",
        "OneStarSkillEntry",
        "OneStarPendingOperation",
        "OneStarCombatantState",
    ):
        assert adapter_private_type not in rendered


def test_one_star_provider_schema_requires_structured_visual_subjects():
    schema = _openai_strict_json_schema(OneStarEventRouterOutput)
    observable_fact = schema["$defs"]["ObservableFact"]

    assert observable_fact["properties"]["visual_subject_ids"] == {
        "items": {"type": "string"},
        "type": "array",
    }
    assert "visual_subject_ids" in observable_fact["required"]


def _one_star_checkpoint():
    ckpt = checkpoint(
        characters=[
            character_record("alice", is_playable=True),
            character_record("pip"),
        ]
    )
    ckpt.session.config.settings.ruleset_id = ONE_STAR_RULESET_ID
    return ckpt


def _dispatcher(*responses):
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(
        side_effect=[llm_response(response, content="{}") for response in responses]
    )
    return LLMDispatcher(client, PromptManager("app/prompts")), client


def _stub_one_star_router_context(monkeypatch):
    from app.engine import one_star_router_context

    monkeypatch.setattr(
        one_star_router_context,
        "render_one_star_router_static_config",
        lambda *_args,
        **_kwargs: "<one_star_rules_config>\nmax_batch=5\n</one_star_rules_config>",
    )
    monkeypatch.setattr(
        one_star_router_context,
        "render_one_star_repair_evidence",
        lambda *_args, **_kwargs: (
            "<one_star_conflict_evidence>\n"
            "current_resources: gold=34\n"
            "</one_star_conflict_evidence>"
        ),
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


def test_one_star_state_updates_are_required_and_have_an_explicit_empty_shape():
    data = _one_star_output().model_dump()
    assert data["state_updates"] == []

    data.pop("state_updates")
    with pytest.raises(ValidationError, match="state_updates"):
        OneStarEventRouterOutput.model_validate(data)


def test_compact_update_translation_rejects_unknown_or_duplicate_details():
    from app.engine.one_star_adapter import one_star_state_updates_to_transaction

    ckpt = _one_star_checkpoint()
    update = OneStarStateUpdate(
        kind="hero_delta",
        target_id="pip",
        value="",
        details=["hp_current=4"],
    )
    transaction = one_star_state_updates_to_transaction(
        ckpt,
        [update],
        canonical_at_s=12,
    )
    operation = transaction.operations[0]

    assert operation.hero_id == "pip"
    assert operation.hp_current == 4

    for details, error in (
        (["hp_currnt=4"], "unsupported details"),
        (["hp_current=4", "hp_current=3"], "exactly once"),
        (["experience_delta=3"], "unsupported details"),
        (["stat.grit=2"], "unsupported details"),
    ):
        with pytest.raises(OneStarTransactionError, match=error):
            one_star_state_updates_to_transaction(
                ckpt,
                [update.model_copy(update={"details": details})],
                canonical_at_s=12,
            )

    with pytest.raises(OneStarTransactionError, match="does not use value"):
        one_star_state_updates_to_transaction(
            ckpt,
            [update.model_copy(update={"value": "ignored"})],
            canonical_at_s=12,
        )


def test_compact_mission_start_derives_canonical_timestamps_and_counters():
    from app.engine.one_star_adapter import one_star_state_updates_to_transaction

    update = OneStarStateUpdate(
        kind="mission_start",
        target_id="floor_1_attempt",
        value="1",
        details=[
            "pending_operation_id=deployment_1",
            "party=pip",
            "formation.pip=front",
            "destination=tower_floor_1",
            "completion=defeat four goblins",
            "failure=no party member remains able to fight",
            "duration_s=300",
            "counter.goblins=0/4",
        ],
    )
    transaction = one_star_state_updates_to_transaction(
        _one_star_checkpoint(),
        [update],
        canonical_at_s=12,
    )
    mission = transaction.operations[0].mission

    assert mission.started_at_s == 12
    assert mission.deadline_at_s == 312
    assert [counter.model_dump() for counter in mission.counters] == [
        {"counter_id": "goblins", "current": 0, "target": 4}
    ]

    for invalid_details, error_type, message in (
        (
            [detail for detail in update.details if not detail.startswith("counter.")],
            ValidationError,
            "mission counter ids must be non-empty and unique",
        ),
        (
            [*update.details[:-1], "counter.=0/4"],
            OneStarTransactionError,
            "empty detail id",
        ),
        (
            [*update.details, "counter.goblins=1/4"],
            OneStarTransactionError,
            "must appear exactly once",
        ),
        (
            [*update.details[:-1], "counter.goblins=0"],
            OneStarTransactionError,
            "must use current/target",
        ),
    ):
        with pytest.raises(error_type, match=message):
            one_star_state_updates_to_transaction(
                _one_star_checkpoint(),
                [update.model_copy(update={"details": invalid_details})],
                canonical_at_s=12,
            )


def test_compact_pending_resolve_uses_only_target_id():
    from app.engine.one_star_adapter import one_star_state_updates_to_transaction

    update = OneStarStateUpdate(
        kind="pending_resolve",
        target_id="deployment_1",
        value="",
        details=[],
    )
    transaction = one_star_state_updates_to_transaction(
        _one_star_checkpoint(),
        [update],
        canonical_at_s=12,
    )

    assert transaction.operations[0].operation_id == "deployment_1"

    for changed_field, changed_value, message in (
        ("value", "resolved", "does not use value"),
        ("details", ["participant=pip"], "unsupported details"),
    ):
        with pytest.raises(OneStarTransactionError, match=message):
            one_star_state_updates_to_transaction(
                _one_star_checkpoint(),
                [update.model_copy(update={changed_field: changed_value})],
                canonical_at_s=12,
            )


def test_one_star_router_schema_is_used_for_fresh_and_cat_ii_routes(monkeypatch):
    _stub_one_star_router_context(monkeypatch)
    fresh = _one_star_output()
    dispatcher, client = _dispatcher(fresh)

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=_one_star_checkpoint(),
            actor_id="alice",
            intention="I ask Pip a question.",
        )
    )

    assert result is fresh
    assert (
        client.complete.await_args.kwargs["response_model"] is OneStarEventRouterOutput
    )

    cat_ii = OpenCatIIEvent(
        event_id="evt_open",
        initiator_id="alice",
        initiator_intention="I reach for Pip's letter.",
        required_responders=["pip"],
        collected_intentions={"pip": "I pull it away."},
    )
    closed = _one_star_output()
    dispatcher, client = _dispatcher(closed)
    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=_one_star_checkpoint(),
            actor_id="pip",
            intention="I pull it away.",
            cat_ii_event=cat_ii,
        )
    )

    assert result is closed
    assert (
        client.complete.await_args.kwargs["response_model"] is OneStarEventRouterOutput
    )


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

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="Send Pip through the Tower gate.",
        )
    )

    assert result is corrected
    assert client.complete.await_count == 2
    correction_messages = client.complete.await_args_list[1].kwargs["messages"]
    assert correction_messages[-2] == {
        "role": "assistant",
        "content": invalid.model_dump_json(),
    }
    assert "must open Cat II" in correction_messages[-1]["content"]
    assert (
        "read-only System or status inspection" in (correction_messages[-1]["content"])
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
    repaired_updates = OneStarStateUpdateList(
        state_updates=corrected.state_updates,
    )
    dispatcher, client = _dispatcher(invalid, repaired_updates)

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="Deploy Pip to Floor 1.",
        )
    )

    assert result is invalid
    assert result.state_updates == corrected.state_updates
    assert client.complete.await_count == 2
    correction = client.complete.await_args_list[1].kwargs["messages"][-1]
    assert "deployment has no separate target Hero" in correction["content"]
    assert (
        client.complete.await_args_list[1].kwargs["response_model"]
        is OneStarStateUpdateList
    )
    stored_history = "\n".join(
        str(message.content) for message in ckpt.session_conversation
    )
    assert "invalid_deployment_target" in stored_history
    assert "target_id=alice" not in stored_history


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
        asyncio.run(
            dispatcher.route_intention(
                ckpt=ckpt,
                actor_id="alice",
                intention="Send Pip through the Tower gate.",
            )
        )

    assert client.complete.await_count == 2
    assert ckpt.model_dump(mode="json") == before


def test_one_star_continuation_uses_the_closed_schema(monkeypatch):
    _stub_one_star_router_context(monkeypatch)
    prior = _one_star_output()
    continuation = _closed_one_star_output()
    dispatcher, client = _dispatcher(continuation)

    result = asyncio.run(
        dispatcher.route_continuation(
            ckpt=_one_star_checkpoint(),
            actor_id="alice",
            prior_result=prior,
        )
    )

    assert result is continuation
    assert (
        client.complete.await_args.kwargs["response_model"]
        is ClosedOneStarEventRouterOutput
    )


def _lobby_return_continuation(*, guide_delivery: bool):
    result = _closed_one_star_output()
    result.event_id = (
        "corrected_lobby_return" if guide_delivery else "invalid_lobby_return"
    )
    result.location_updates = [
        LocationUpdateSignal(character_id="pip", location_label="lobby")
    ]
    result.observers = [
        ObserverEntry(
            character_id="alice",
            observation_level="d",
            routing_role="observe_only",
        )
    ]
    if guide_delivery:
        result.observers.append(
            ObserverEntry(
                character_id="iselle",
                observation_level="d",
                routing_role="observe_only",
            )
        )
        result.canonical_event.observable_facts.append(
            ObservableFact.only(
                "The System reports Pip's return to the lobby.",
                ["iselle"],
            )
        )
    return result


def _stub_one_star_guide_account(monkeypatch, ckpt) -> None:
    from app.engine import one_star_adapter

    account = SimpleNamespace(
        config=SimpleNamespace(
            lobby_id="local",
            lobby_location_label="lobby",
            operation_requirements={},
        ),
        state=SimpleNamespace(
            guide_character_ids=["iselle"],
            pending_operation=None,
        ),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_account",
        lambda _ckpt: (ckpt.characters[0], account),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_hero",
        lambda character: (
            SimpleNamespace(owner_lobby_id="local")
            if character is not None and character.character_id == "pip"
            else None
        ),
    )


def test_one_star_continuation_retries_missing_guide_delivery_before_commit(
    monkeypatch,
):
    _stub_one_star_router_context(monkeypatch)
    ckpt = _one_star_checkpoint()
    next(character for character in ckpt.characters if character.character_id == "pip").location = "synthesis"
    ckpt.characters.append(character_record("iselle", location="lobby"))
    _stub_one_star_guide_account(monkeypatch, ckpt)
    invalid = _lobby_return_continuation(guide_delivery=False)
    corrected = _lobby_return_continuation(guide_delivery=True)
    dispatcher, client = _dispatcher(invalid, corrected)

    result = asyncio.run(
        dispatcher.route_continuation(
            ckpt=ckpt,
            actor_id="alice",
            prior_result=_one_star_output(),
            original_action="I wait for Pip to return.",
        )
    )

    assert result is corrected
    assert client.complete.await_count == 2
    correction_call = client.complete.await_args_list[1]
    assert (
        correction_call.kwargs["response_model"]
        is ClosedOneStarEventRouterOutput
    )
    assert "configured guide" in correction_call.kwargs["messages"][-1]["content"]
    history = "\n".join(message.content for message in ckpt.session_conversation)
    assert "corrected_lobby_return" in history
    assert "invalid_lobby_return" not in history


def test_repeated_invalid_one_star_continuation_restores_router_snapshot(
    monkeypatch,
):
    _stub_one_star_router_context(monkeypatch)
    ckpt = _one_star_checkpoint()
    next(character for character in ckpt.characters if character.character_id == "pip").location = "synthesis"
    ckpt.characters.append(character_record("iselle", location="lobby"))
    _stub_one_star_guide_account(monkeypatch, ckpt)
    dispatcher, client = _dispatcher(
        _lobby_return_continuation(guide_delivery=False),
        _lobby_return_continuation(guide_delivery=False),
    )
    before = ckpt.model_dump(mode="json")

    with pytest.raises(
        ValueError,
        match="continuation output remained invalid after one correction",
    ):
        asyncio.run(
            dispatcher.route_continuation(
                ckpt=ckpt,
                actor_id="alice",
                prior_result=_one_star_output(),
                original_action="I wait for Pip to return.",
            )
        )

    assert client.complete.await_count == 2
    assert all(
        call.kwargs["response_model"] is ClosedOneStarEventRouterOutput
        for call in client.complete.await_args_list
    )
    assert ckpt.model_dump(mode="json") == before


def test_corrected_one_star_continuation_cannot_open_cat_ii(monkeypatch):
    _stub_one_star_router_context(monkeypatch)
    ckpt = _one_star_checkpoint()
    next(
        character
        for character in ckpt.characters
        if character.character_id == "pip"
    ).location = "synthesis"
    ckpt.characters.append(character_record("iselle", location="lobby"))
    _stub_one_star_guide_account(monkeypatch, ckpt)
    invalid = _lobby_return_continuation(guide_delivery=False)
    corrected_data = _lobby_return_continuation(
        guide_delivery=True,
    ).model_dump(mode="json")
    corrected_data.update({
        "event_kind": "cat_ii_open",
        "requires_responders": True,
        "required_responders": ["pip"],
    })
    corrected_cat_ii = OneStarEventRouterOutput.model_validate(corrected_data)
    dispatcher, client = _dispatcher(invalid, corrected_cat_ii)
    before = ckpt.model_dump(mode="json")

    with pytest.raises(
        ValueError,
        match="continuation output remained invalid after one correction",
    ):
        asyncio.run(
            dispatcher.route_continuation(
                ckpt=ckpt,
                actor_id="alice",
                prior_result=_one_star_output(),
                original_action="I wait for Pip to return.",
            )
        )

    assert all(
        call.kwargs["response_model"] is ClosedOneStarEventRouterOutput
        for call in client.complete.await_args_list
    )
    assert ckpt.model_dump(mode="json") == before


def test_one_star_repair_accepts_only_the_state_update_shape(monkeypatch):
    _stub_one_star_router_context(monkeypatch)
    repaired_updates = OneStarStateUpdateList(state_updates=[])
    dispatcher, client = _dispatcher(repaired_updates)

    repaired = asyncio.run(
        dispatcher._repair_one_star_transaction(
            ckpt=_one_star_checkpoint(),
            result=_one_star_output(),
            actor_id="alice",
            validation_error="insufficient Gold",
        )
    )

    assert repaired is repaired_updates
    assert client.complete.await_args.kwargs["response_model"] is OneStarStateUpdateList
    repair_packet = client.complete.await_args.kwargs["messages"][-1]["content"]
    assert "one_star_state_update_repair" in repair_packet
    assert "Candidate state updates" in repair_packet
    assert "deployment repeats participant details" in repair_packet
    assert "omits target_id" in repair_packet
    assert "change the offending field" in repair_packet
    scalar_contract = next(
        line
        for line in repair_packet.splitlines()
        if "pending_resolve" in line and 'value=""' in line
    )
    normalized_scalar_contract = scalar_contract.lower()
    for required_shape in (
        "pending_resolve",
        'value=""',
        "target_id",
        "details=[]",
        "mission_update",
        "complete declared counter set",
        "including unchanged counters",
        "mission_start",
        "counter.<nonempty_id>=<current>/<target>",
    ):
        assert required_shape in normalized_scalar_contract
    assert "<one_star_conflict_evidence>" in repair_packet
    assert "current_resources: gold=34" in repair_packet
    assert "one_star_current_ledger" not in repair_packet
    assert "canonical_event" not in repair_packet
    system_prompt = client.complete.await_args.kwargs["messages"][0]["content"]
    assert "counter.<nonempty_id>=<current>/<target>" in system_prompt
    assert 'pending_resolve' in system_prompt
    assert 'value=""' in system_prompt
    assert 'details=[]' in system_prompt


def test_invalid_compact_value_bounds_enter_the_one_star_repair_contract():
    result = _one_star_output()
    result.state_updates = [
        OneStarStateUpdate(
            kind="hero_delta",
            target_id="pip",
            value="",
            details=["hp_current=-3"],
        )
    ]

    with pytest.raises(
        OneStarTransactionError,
        match="typed value bounds",
    ):
        _one_star_transaction_for_result(_one_star_checkpoint(), result)


def test_invalid_hp_uses_narrow_conflict_repair_without_rewriting_fiction(
    monkeypatch,
):
    _stub_one_star_router_context(monkeypatch)
    from app.engine import one_star_router_context

    monkeypatch.setattr(
        one_star_router_context,
        "render_one_star_repair_evidence",
        lambda *_args, **_kwargs: (
            "<one_star_conflict_evidence>\n"
            "hero pip: hp=2/2\n"
            "</one_star_conflict_evidence>"
        ),
    )
    invalid = _one_star_output()
    invalid.state_updates = [
        OneStarStateUpdate(
            kind="hero_delta",
            target_id="pip",
            value="",
            details=["hp_current=-3"],
        )
    ]
    repaired = OneStarStateUpdateList(state_updates=[])
    dispatcher, client = _dispatcher(invalid, repaired)

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=_one_star_checkpoint(),
            actor_id="alice",
            intention="Pip is hurt.",
        )
    )

    assert result is invalid
    assert result.state_updates == []
    assert client.complete.await_count == 2
    assert (
        client.complete.await_args_list[1].kwargs["response_model"]
        is OneStarStateUpdateList
    )
    repair_packet = client.complete.await_args_list[1].kwargs["messages"][-1]["content"]
    assert "hp_current=-3" in repair_packet
    assert "hero pip: hp=2/2" in repair_packet
    assert "current_resources" not in repair_packet


def test_one_star_spawn_activation_overlap_fails_before_materialization():
    data = _one_star_output().model_dump(mode="json")
    data["spawn"] = [
        {
            "character_id": "new_hero",
            "seed": {
                "role": "baker",
                "reason": "cold light deposits a stranger",
                "location": "niflheim_lobby",
                "objectives": ["find an exit"],
                "knowledge_tier": 1,
            },
        }
    ]
    data["activate"] = [
        {
            "character_id": "new_hero",
            "location_label": "niflheim_lobby",
        }
    ]
    result = OneStarEventRouterOutput.model_validate(data)
    dispatcher, _client = _dispatcher()
    dispatcher.materialize_spawns = AsyncMock()

    with pytest.raises(OneStarTransactionError, match="both spawn and activate"):
        asyncio.run(
            dispatcher.prepare_ruleset_event(
                ckpt=_one_star_checkpoint(),
                result=result,
                actor_id="alice",
            )
        )

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
    assert "state_updates" not in default["router_ruleset_addon"]
    assert "state_updates" in one_star["router_ruleset_addon"]


def test_one_star_normal_router_request_has_no_live_ledger_tail():
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
        router_input_block="submitted action",
    )

    system, user = messages
    assert "state_updates" in system["content"]
    assert "max_batch=5" in system["content"]
    assert "Gold: 34" not in system["content"]
    assert "active_master_feed_id" not in system["content"]
    assert "max_batch=5" not in user["content"]
    assert "Gold: 34" not in user["content"]
    assert "active_master_feed_id" not in user["content"]
    assert "one_star_current_ledger" not in user["content"]
    assert "submitted action" in user["content"]


def test_one_star_router_context_has_no_live_ledger_surface():
    context = _build_router_context(
        _one_star_checkpoint(),
        "alice",
        include_engine_state_updates=False,
    )

    assert "one_star_state_section" not in context


def test_account_owner_lobby_mutation_requires_scoped_guide_delivery(monkeypatch):
    from app.engine import one_star_adapter

    ckpt = _one_star_checkpoint()
    ckpt.characters.append(character_record("iselle", location="niflheim_lobby"))
    result = _one_star_output()
    result.state_updates = [
        OneStarStateUpdate(
            kind="inventory_delta",
            target_id="receipt",
            value="1",
            details=[],
        )
    ]
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

    result.observers.append(
        ObserverEntry(
            character_id="iselle",
            observation_level="d",
            routing_role="observe_only",
        )
    )
    result.canonical_event.observable_facts.append(
        ObservableFact.only(
            "A System receipt records the lobby action.",
            ["iselle"],
        )
    )

    _validate_one_star_guide_routing(ckpt, actor_id="alice", result=result)


@pytest.mark.parametrize("lifecycle_kind", ["dormant", "activate"])
def test_account_owner_local_hero_lifecycle_reaches_guide(
    monkeypatch,
    lifecycle_kind,
):
    from app.engine import one_star_adapter

    ckpt = _one_star_checkpoint()
    pip = next(
        character for character in ckpt.characters if character.character_id == "pip"
    )
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
    base["state_updates"] = []
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

    result.observers.append(
        ObserverEntry(
            character_id="iselle",
            observation_level="d",
            routing_role="observe_only",
        )
    )
    result.canonical_event.observable_facts.append(
        ObservableFact.only(
            "The System reports the roster change.",
            ["iselle"],
        )
    )
    _validate_one_star_guide_routing(
        ckpt,
        actor_id="alice",
        result=result,
    )


def test_cat_ii_open_rejects_one_star_mechanical_side_effects():
    ckpt = _one_star_checkpoint()
    data = router_output(
        requires_responders=True,
        required_responders=["pip"],
        observer_ids=["alice", "pip"],
    ).model_dump(mode="json")
    pending_open = {
        "kind": "pending_open",
        "target_id": "deployment_1",
        "value": "deployment",
        "details": ["participant=pip", "destination=tower_floor_1"],
    }
    terminal_delta = {
        "kind": "hero_delta",
        "target_id": "pip",
        "value": "",
        "details": [
            "hp_current=0",
            "terminal_action=death",
            "death_cause=forced before response",
        ],
    }
    data["state_updates"] = [pending_open, terminal_delta]
    result = OneStarEventRouterOutput.model_validate(data)

    with pytest.raises(ValueError, match="only one pending_open"):
        _validate_one_star_cat_ii_transaction(ckpt, result)

    result.state_updates = [OneStarStateUpdate.model_validate(pending_open)]
    _validate_one_star_cat_ii_transaction(ckpt, result)


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
    data["state_updates"] = [
        {
            "kind": "pending_open",
            "target_id": "synthesis_1",
            "value": "synthesis",
            "details": [
                "participant=pip",
                "participant=bob",
                "target_id=alice",
                "destination=synthesis_room",
            ],
        }
    ]
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


def test_synthesis_selection_collects_configured_guide_intention(monkeypatch):
    from app.engine import one_star_adapter

    ckpt = _one_star_checkpoint()
    ckpt.characters.append(character_record("guide"))
    data = router_output(
        requires_responders=True,
        required_responders=["pip"],
        observer_ids=["alice", "pip", "guide"],
    ).model_dump(mode="json")
    data["state_updates"] = [
        {
            "kind": "pending_open",
            "target_id": "synthesis_1",
            "value": "synthesis",
            "details": [
                "participant=pip",
                "target_id=alice",
                "destination=synthesis_room",
            ],
        }
    ]
    result = OneStarEventRouterOutput.model_validate(data)
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_account",
        lambda _: (
            ckpt.characters[0],
            SimpleNamespace(
                state=SimpleNamespace(guide_character_ids=["guide"]),
            ),
        ),
    )

    _include_one_star_synthesis_guide_responders(
        ckpt,
        actor_id="alice",
        result=result,
    )
    _include_one_star_synthesis_guide_responders(
        ckpt,
        actor_id="alice",
        result=result,
    )

    assert result.required_responders == ["pip", "guide"]


def test_tutorial_delivery_requires_direct_visible_observation():
    ckpt = _one_star_checkpoint()
    data = router_output(
        observer_ids=["pip"],
        facts=[ObservableFact.only("Iselle explains the gate.", ["pip"])],
    ).model_dump(mode="json")
    data["state_updates"] = [
        {
            "kind": "tutorial_delivery",
            "target_id": "tower_gate",
            "value": "",
            "details": ["recipient=pip"],
        }
    ]
    result = OneStarEventRouterOutput.model_validate(data)
    _validate_one_star_tutorial_routing(ckpt, result)

    result.observers[0].observation_level = "i"
    with pytest.raises(ValueError, match="direct observer"):
        _validate_one_star_tutorial_routing(ckpt, result)

    result.observers[0].observation_level = "d"
    result.canonical_event.observable_facts = [
        ObservableFact.only("Alice alone hears something else.", ["alice"])
    ]
    with pytest.raises(ValueError, match="visible"):
        _validate_one_star_tutorial_routing(ckpt, result)


def test_standard_summon_requires_sole_direct_guide_handoff(monkeypatch):
    from app.engine import one_star_adapter

    ckpt = _one_star_checkpoint()
    ckpt.characters.append(character_record("iselle", location="niflheim_lobby"))
    account = SimpleNamespace(
        config=SimpleNamespace(
            summon_pools={
                "basic": SimpleNamespace(usage="standard"),
                "opening": SimpleNamespace(usage="opening_roster"),
            },
        ),
        state=SimpleNamespace(guide_character_ids=["iselle"]),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_account",
        lambda _: (ckpt.characters[0], account),
    )
    data = router_output(
        observer_ids=["alice", "iselle"],
        facts=[ObservableFact.only("A new Hero wakes.", ["alice", "iselle"])],
    ).model_dump(mode="json")
    data["state_updates"] = [
        {
            "kind": "summon",
            "target_id": "basic",
            "value": "1",
            "details": [],
        }
    ]
    result = OneStarEventRouterOutput.model_validate(data)

    with pytest.raises(ValueError, match="immediate guide induction handoff"):
        _validate_one_star_standard_summon_guide_handoff(
            ckpt,
            actor_id="alice",
            result=result,
        )

    guide_observer = next(
        observer
        for observer in result.observers
        if observer.character_id == "iselle"
    )
    guide_observer.routing_role = "next_output"
    _validate_one_star_standard_summon_guide_handoff(
        ckpt,
        actor_id="alice",
        result=result,
    )

    guide_observer.routing_role = "observe_only"
    result.state_updates[0].target_id = "opening"
    _validate_one_star_standard_summon_guide_handoff(
        ckpt,
        actor_id="alice",
        result=result,
    )


def test_standard_summon_handoff_gets_one_full_routing_correction(monkeypatch):
    from app.engine import one_star_adapter, turn_loop_dispatcher

    _stub_one_star_router_context(monkeypatch)
    ckpt = _one_star_checkpoint()
    ckpt.characters.append(character_record("iselle", location="niflheim_lobby"))
    account = SimpleNamespace(
        config=SimpleNamespace(
            summon_pools={"basic": SimpleNamespace(usage="standard")},
        ),
        state=SimpleNamespace(guide_character_ids=["iselle"]),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_account",
        lambda _: (ckpt.characters[0], account),
    )
    monkeypatch.setattr(
        turn_loop_dispatcher,
        "_one_star_transaction_for_result",
        lambda _checkpoint, _result: OneStarTransaction(
            present=False,
            operations=[],
        ),
    )

    def summon_result(*, guide_role: str) -> OneStarEventRouterOutput:
        data = router_output(
            observer_ids=["alice", "iselle"],
            facts=[
                ObservableFact.only(
                    "A new Hero wakes in the summoning circle.",
                    ["alice", "iselle"],
                )
            ],
        ).model_dump(mode="json")
        data["state_updates"] = [
            {
                "kind": "summon",
                "target_id": "basic",
                "value": "1",
                "details": [],
            }
        ]
        parsed = OneStarEventRouterOutput.model_validate(data)
        next(
            observer
            for observer in parsed.observers
            if observer.character_id == "iselle"
        ).routing_role = guide_role
        return parsed

    first = summon_result(guide_role="observe_only")
    corrected = summon_result(guide_role="next_output")
    dispatcher, client = _dispatcher(first, corrected)

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="Use the basic summon circle once.",
        )
    )

    assert result is corrected
    assert client.complete.await_count == 2
    correction = client.complete.await_args_list[1].kwargs["messages"][-1]["content"]
    assert "immediate guide induction handoff" in correction
    assert "sole immediate next_output" in correction
    assert all(first.event_id not in message.content for message in ckpt.session_conversation)


def _standard_summon_induction_case(monkeypatch, *, pool_usage="standard"):
    from app.engine import one_star_adapter

    ckpt = _one_star_checkpoint()
    ckpt.characters.extend((
        character_record("iselle", location="niflheim_lobby"),
        character_record("fresh_hero", location="niflheim_lobby"),
        character_record("reserve_hero", location="niflheim_lobby"),
    ))
    account = SimpleNamespace(
        config=SimpleNamespace(
            summon_pools={"basic": SimpleNamespace(usage=pool_usage)},
        ),
        state=SimpleNamespace(
            guide_character_ids=["iselle"],
            applied_event_fingerprints={},
        ),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "load_one_star_account",
        lambda _: (ckpt.characters[0], account),
    )
    data = router_output(
        event_id="evt_standard_summon",
        observer_ids=["alice", "iselle"],
        agent_ids=["iselle"],
        facts=[ObservableFact.only(
            "Two summoned Heroes wake for Iselle to receive.",
            ["iselle"],
        )],
        spawn=[{
            "character_id": "fresh_hero",
            "seed": {
                "role": "newly summoned Hero",
                "reason": "cold light resolves into a stranger",
                "location": "niflheim_lobby",
                "objectives": ["survive"],
                "knowledge_tier": 1,
            },
        }],
        activate=[{
            "character_id": "reserve_hero",
            "location_label": "niflheim_lobby",
        }],
    ).model_dump(mode="json")
    data["state_updates"] = [{
        "kind": "summon",
        "target_id": "basic",
        "value": "2",
        "details": [],
    }]
    ckpt.canonical_events.append(
        OneStarEventRouterOutput.model_validate(data)
    )
    return ckpt


def _standard_summon_induction_result(
    *,
    event_id: str,
    recipients: list[str] | None,
    self_cascade: bool = False,
) -> OneStarEventRouterOutput:
    observer_ids = ["iselle", "fresh_hero", "reserve_hero", "pip"]
    visible_to = recipients or ["iselle"]
    data = router_output(
        event_id=event_id,
        observer_ids=observer_ids,
        agent_ids=["iselle"] if self_cascade else [],
        facts=[ObservableFact.only(
            "Iselle delivers a compact Niflheim survival induction.",
            visible_to,
        )],
    ).model_dump(mode="json")
    data["state_updates"] = (
        []
        if recipients is None
        else [{
            "kind": "tutorial_delivery",
            "target_id": "niflheim_survival_induction",
            "value": "",
            "details": [
                f"recipient={character_id}"
                for character_id in recipients
            ],
        }]
    )
    return OneStarEventRouterOutput.model_validate(data)


@pytest.mark.parametrize(
    ("defect", "expected_error"),
    (
        ("omission", "exactly one tutorial_delivery"),
        ("wrong_recipient", "recipients must exactly equal"),
        ("self_cascade", "cannot select another next_output"),
    ),
)
def test_standard_summon_induction_gets_one_full_routing_correction(
    monkeypatch,
    defect: str,
    expected_error: str,
):
    _stub_one_star_router_context(monkeypatch)
    ckpt = _standard_summon_induction_case(monkeypatch)
    first = _standard_summon_induction_result(
        event_id=f"evt_bad_induction_{defect}",
        recipients=(
            None
            if defect == "omission"
            else ["pip"]
            if defect == "wrong_recipient"
            else ["fresh_hero", "reserve_hero"]
        ),
        self_cascade=defect == "self_cascade",
    )
    corrected = _standard_summon_induction_result(
        event_id=f"evt_corrected_induction_{defect}",
        recipients=["fresh_hero", "reserve_hero"],
    )
    dispatcher, client = _dispatcher(first, corrected)

    result = asyncio.run(
        dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="iselle",
            intention="I teach both arrivals what keeps them alive here.",
        )
    )

    assert result is corrected
    assert client.complete.await_count == 2
    correction = client.complete.await_args_list[1].kwargs["messages"][-1]["content"]
    assert "standard summon guide induction is incomplete" in correction
    assert expected_error in correction
    stored_history = "\n".join(
        str(message.content) for message in ckpt.session_conversation
    )
    assert first.event_id not in stored_history
    assert corrected.event_id in stored_history


def test_standard_summon_induction_accepts_exact_arrival_set(monkeypatch):
    ckpt = _standard_summon_induction_case(monkeypatch)
    result = _standard_summon_induction_result(
        event_id="evt_complete_induction",
        recipients=["reserve_hero", "fresh_hero"],
    )

    _validate_one_star_standard_summon_induction(
        ckpt,
        actor_id="iselle",
        result=result,
    )


@pytest.mark.parametrize(
    ("defect", "expected_error"),
    (
        ("cat_ii", "must remain Cat I"),
        ("enrichment", "cannot request perception enrichment"),
        ("extra_update", "exactly one state update"),
    ),
)
def test_standard_summon_induction_rejects_extra_routing(
    monkeypatch,
    defect: str,
    expected_error: str,
):
    ckpt = _standard_summon_induction_case(monkeypatch)
    result = _standard_summon_induction_result(
        event_id=f"evt_induction_{defect}",
        recipients=["fresh_hero", "reserve_hero"],
    )
    if defect == "cat_ii":
        result.requires_responders = True
        result.required_responders = ["fresh_hero"]
    elif defect == "enrichment":
        next(
            observer
            for observer in result.observers
            if observer.character_id == "reserve_hero"
        ).routing_role = "perception_enrichment"
    else:
        result.state_updates.append(OneStarStateUpdate(
            kind="hero_delta",
            target_id="fresh_hero",
            value="",
            details=["hp_current=1"],
        ))

    with pytest.raises(ValueError, match=expected_error):
        _validate_one_star_standard_summon_induction(
            ckpt,
            actor_id="iselle",
            result=result,
        )


def test_standard_summon_induction_is_rechecked_before_prepare(monkeypatch):
    ckpt = _standard_summon_induction_case(monkeypatch)
    omitted = _standard_summon_induction_result(
        event_id="evt_omitted_induction_before_prepare",
        recipients=None,
    )
    dispatcher, client = _dispatcher()

    with pytest.raises(ValueError, match="exactly one tutorial_delivery"):
        asyncio.run(dispatcher.prepare_ruleset_event(
            ckpt=ckpt,
            actor_id="iselle",
            result=omitted,
        ))

    assert client.complete.await_count == 0


def test_standard_summon_induction_does_not_capture_other_routes(monkeypatch):
    opening_ckpt = _standard_summon_induction_case(
        monkeypatch,
        pool_usage="opening_roster",
    )
    omitted = _standard_summon_induction_result(
        event_id="evt_unrelated_route",
        recipients=None,
    )

    _validate_one_star_standard_summon_induction(
        opening_ckpt,
        actor_id="iselle",
        result=omitted,
    )
    non_guide_ckpt = _standard_summon_induction_case(monkeypatch)
    _validate_one_star_standard_summon_induction(
        non_guide_ckpt,
        actor_id="pip",
        result=omitted,
    )
    generic = checkpoint(characters=[character_record("iselle")])
    generic.canonical_events.append(router_output(agent_ids=["iselle"]))
    _validate_one_star_standard_summon_induction(
        generic,
        actor_id="iselle",
        result=router_output(),
    )


def test_one_star_router_projections_split_static_rules_from_narrow_repair_evidence(
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
            "premium": OneStarStandardSummonPool(
                usage="standard",
                cost=OneStarCost(
                    gold=0,
                    gems=5,
                    building_resources=0,
                    materials={},
                ),
                minimum_birth_stars=2,
                maximum_birth_stars=5,
                star_weights={2: 7500, 3: 2300, 4: 175, 5: 25},
                eligible_existing_ids=["private_reserve_candidate"],
                fresh_generation_allowed=True,
            ),
            "newcomer_opening": OneStarOpeningRosterSummonPool(
                usage="opening_roster",
                slots=[OneStarOpeningRosterBoundPlayerActorSlot(
                    kind="bound_player_actor",
                    character_id="one_star_newcomer",
                )],
            ),
            "starter_roster": OneStarOpeningRosterSummonPool(
                usage="opening_roster",
                slots=[
                    OneStarOpeningRosterFixedSlot(
                        kind="fixed",
                        character_id="renna_holt",
                    ),
                    OneStarOpeningRosterRandomExistingGradeSlot(
                        kind="random_existing_grade",
                        birth_stars=3,
                    ),
                    OneStarOpeningRosterFixedSlot(
                        kind="fixed",
                        character_id="edren_marr",
                    ),
                ],
                initial_deployment_requires_guide_handoff=True,
            ),
            "unflagged_roster": OneStarOpeningRosterSummonPool(
                usage="opening_roster",
                slots=[
                    OneStarOpeningRosterRandomExistingGradeSlot(
                        kind="random_existing_grade",
                        birth_stars=2,
                    ),
                ],
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
        progression=SimpleNamespace(
            grade_multiplier_milli=1250,
            variance_basis_points=500,
            xp_threshold_factor=50,
            cap_bank_extra_levels=1,
        ),
        deployment_stamina_cost=1,
        maximum_stamina=5,
        stamina_recovery_seconds=1800,
        floor_rewards={1: cost(gold=4, building_resources=1)},
        floor_scenarios={1: SimpleNamespace(
            mission_id="tower_1",
            destination="tower_floor_1",
            premise="Reach the first-floor exit.",
            completion_declaration="reach the exit",
            failure_declaration="all party members fall",
            counters=[SimpleNamespace(
                counter_id="exit",
                current=0,
                target=1,
            )],
            pressure_beats=["The tower presses the party forward."],
        )},
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
        gem_purchase=SimpleNamespace(
            funds_label="$",
            starting_funds=200,
            periodic_income=100,
            income_interval_seconds=604_800,
            funds_cost=100,
            gems_granted=20,
        ),
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
        discretionary_funds=200,
        funds_accrual_anchor_s=0,
        guide_character_ids=["iselle"],
        system_observer_ids=["iselle"],
        tutorial_deliveries={"summoning": ["pip"]},
        active_mission=mission,
        stored_equipment=[
            SimpleNamespace(
                item_id="stored_hidden_ledger_item",
                name="Stored Ledger Item",
                slot="hand",
                quantity=1,
                durability_current=1,
                durability_max=1,
            ),
        ],
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
        terminal_event_id="",
        hidden_capabilities={"potential": "locked"},
        progression_seed="private-stream",
        strong_stat_id="strength",
        weak_stat_id="agility",
        potential_grade=1,
    )
    ckpt = _one_star_checkpoint()
    ckpt.session.leading_at_s = 60
    monkeypatch.setattr(
        one_star_router_context, "is_one_star_checkpoint", lambda _: True
    )
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
    static = one_star_router_context.render_one_star_router_static_config(ckpt)
    newcomer_opening_repair_evidence = (
        one_star_router_context.render_one_star_repair_evidence(
            ckpt,
            state_updates=[
                OneStarStateUpdate(
                    kind="summon",
                    target_id="newcomer_opening",
                    value="1",
                    details=[],
                )
            ],
        )
    )
    opening_repair_evidence = one_star_router_context.render_one_star_repair_evidence(
        ckpt,
        state_updates=[
            OneStarStateUpdate(
                kind="summon",
                target_id="starter_roster",
                value="3",
                details=[],
            )
        ],
    )
    unflagged_opening_repair_evidence = (
        one_star_router_context.render_one_star_repair_evidence(
            ckpt,
            state_updates=[
                OneStarStateUpdate(
                    kind="summon",
                    target_id="unflagged_roster",
                    value="1",
                    details=[],
                )
            ],
        )
    )
    hp_evidence = one_star_router_context.render_one_star_repair_evidence(
        ckpt,
        state_updates=[
            OneStarStateUpdate(
                kind="hero_delta",
                target_id="pip",
                value="",
                details=["hp_current=-3"],
            )
        ],
    )
    purchase_evidence = one_star_router_context.render_one_star_repair_evidence(
        ckpt,
        state_updates=[
            OneStarStateUpdate(
                kind="catalogue_apply",
                target_id="synthesis_chamber_i",
                value="1",
                details=[],
            )
        ],
    )
    gem_purchase_evidence = one_star_router_context.render_one_star_repair_evidence(
        ckpt,
        state_updates=[
            OneStarStateUpdate(
                kind="gem_purchase",
                target_id="gems",
                value="60",
                details=[],
            )
        ],
    )
    equipment_evidence = one_star_router_context.render_one_star_repair_evidence(
        ckpt,
        state_updates=[
            OneStarStateUpdate(
                kind="equipment_move",
                target_id="stored_hidden_ledger_item",
                value="pip",
                details=[],
            )
        ],
    )
    state.pending_operation = SimpleNamespace(
        operation_id="op_synthesis",
        kind="synthesis",
        participant_ids=["donor"],
        target_id="pip",
        destination="synthesis_chamber",
        opened_at_s=30,
        synthesis_preview=SimpleNamespace(
            offered_xp=100,
            applied_xp=95,
            wasted_xp=5,
            returned_equipment=[
                SimpleNamespace(item_id="donor_blade"),
            ],
            skill_transfer_chance_basis_points=500,
            input_state_fingerprint="private-stale-state-hash",
        ),
    )
    pending_evidence = one_star_router_context.render_one_star_repair_evidence(
        ckpt,
        state_updates=[
            OneStarStateUpdate(
                kind="pending_resolve",
                target_id="op_synthesis",
                value="",
                details=[],
            )
        ],
    )

    assert "synthesis_chamber_i" in static
    assert "stars=2-5" in static
    assert "rates[2=75%,3=23%,4=1.75%,5=0.25%]" in static
    assert (
        "newcomer_opening: usage=opening_roster; "
        "count=1; slots[1=bound_player_actor]"
    ) in static
    assert (
        "starter_roster: usage=opening_roster; "
        "count=3; slots[1=fixed,2=random_existing_grade:3,3=fixed]"
    ) in static
    starter_pool_line = next(
        line for line in static.splitlines() if line.startswith("- starter_roster:")
    )
    unflagged_pool_line = next(
        line for line in static.splitlines() if line.startswith("- unflagged_roster:")
    )
    assert "initial_deployment_requires_guide_handoff=true" in starter_pool_line
    assert "initial_deployment_requires_guide_handoff" not in unflagged_pool_line
    for private_character_id in (
        "one_star_newcomer",
        "renna_holt",
        "edren_marr",
        "private_reserve_candidate",
    ):
        assert private_character_id not in static
    assert "eligible_existing_ids" not in static
    assert (
        "summon_pool newcomer_opening: usage=opening_roster; "
        "cost_per_pull=free; required_count=1; first_event_only=true"
    ) in newcomer_opening_repair_evidence
    assert (
        "summon_pool starter_roster: usage=opening_roster; "
        "cost_per_pull=free; required_count=3; first_event_only=true"
    ) in opening_repair_evidence
    assert (
        "initial_deployment_requires_guide_handoff=true"
        in opening_repair_evidence
    )
    assert (
        "initial_deployment_requires_guide_handoff"
        not in unflagged_opening_repair_evidence
    )
    for repair_evidence in (
        newcomer_opening_repair_evidence,
        opening_repair_evidence,
        unflagged_opening_repair_evidence,
    ):
        for private_character_id in (
            "one_star_newcomer",
            "renna_holt",
            "edren_marr",
            "private_reserve_candidate",
        ):
            assert private_character_id not in repair_evidence
    assert "repeat_clear_gold" in static
    assert "starting_funds=$200" in static
    assert "periodic_income=$100/604800s" in static
    assert "pack=20gems/$100" in static
    assert "current_funds" not in static
    assert "hero_bounds" not in static
    assert "grade_multiplier_milli" not in static
    assert "variance_basis_points" not in static
    assert "xp_threshold_factor" not in static
    assert "stored_hidden_ledger_item" not in static
    assert "lobby=niflheim" not in static
    assert "Niflheim Lobby" not in static
    assert "gold=34" not in static
    assert "hero pip:" in hp_evidence
    assert "hp=4/6" in hp_evidence
    assert "gold=34" not in hp_evidence
    assert "active_mission" not in hp_evidence
    assert "pending_operation" not in hp_evidence
    assert "hidden_capabilities" not in hp_evidence
    assert "progression_seed" not in hp_evidence
    assert "potential_grade" not in hp_evidence
    assert "level=" not in hp_evidence
    assert "xp=" not in hp_evidence
    assert "Stored Ledger Item" not in hp_evidence
    assert "current_resources: gold=34" in purchase_evidence
    assert "catalogue synthesis_chamber_i" in purchase_evidence
    assert "gem_purchase: current_funds=$200" in gem_purchase_evidence
    assert "pack=20gems/$100" in gem_purchase_evidence
    assert "hero pip:" not in purchase_evidence
    assert "active_mission" not in purchase_evidence
    assert (
        "equipment stored_hidden_ledger_item: holder=account; "
        "name=Stored Ledger Item; slot=hand; quantity=1; durability=1/1"
    ) in equipment_evidence
    assert "stored_equipment" not in equipment_evidence
    assert (
        "synthesis_preview: offered_xp=100; applied_xp=95; wasted_xp=5; "
        "returned_equipment=donor_blade; skill_transfer_chance=5%"
    ) in pending_evidence
    assert "hero pip progression: level=1/10; xp=2/100" in pending_evidence
    assert "private-stale-state-hash" not in pending_evidence
    assert "progression_seed" not in pending_evidence
    assert "potential_grade" not in pending_evidence


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
            stored_equipment=[],
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
            SimpleNamespace(owner_lobby_id="niflheim", equipment=[])
            if character.character_id == "pip"
            else None
        ),
    )
    monkeypatch.setattr(
        one_star_adapter,
        "_validate_all_hero_progression_states",
        lambda *_args: None,
    )

    with pytest.raises(
        one_star_adapter.OneStarTransactionError,
        match="culls belong only in One-Star state updates",
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
        result=ClosedOneStarEventRouterOutput.model_validate(
            {
                **_closed_one_star_output().model_dump(),
                "decision_rationale": rationale,
            }
        ),
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


def test_one_star_history_preserves_router_updates_and_one_time_authority_state():
    result = _one_star_output()
    result.state_updates = [
        OneStarStateUpdate(
            kind="hero_delta",
            target_id="pip",
            value="",
            details=["hp_current=2", "condition=bleeding"],
        )
    ]
    hero = character_record("new_hero", name="Edric")
    hero.mechanics = {
        "one_star_hero": {
            "birth_stars": 1,
            "current_stars": 1,
            "level": 1,
            "experience_points": 0,
            "hp_current": 7,
            "hp_max": 7,
            "stats": {"strength": 3},
            "equipment": [
                {
                    "item_id": "bent_knife",
                    "name": "Bent knife",
                    "slot": "hand",
                    "quantity": 1,
                    "durability_current": 2,
                    "durability_max": 3,
                    "tags": ["blade"],
                    "visible": True,
                }
            ],
            "skills": [
                {
                    "skill_id": "knead_dough",
                    "name": "Knead Dough",
                    "rank": 1,
                    "capability": "works dough by hand",
                    "tags": ["craft"],
                    "visible": True,
                }
            ],
            "owner_lobby_id": "niflheim",
            "acquisition_event_id": result.event_id,
            "hidden_capabilities": {"potential": "unknown"},
            "terminal_event_id": "",
            "progression_seed": "private-stream",
            "strong_stat_id": "strength",
            "weak_stat_id": "agility",
            "potential_grade": 1,
        },
    }

    conversation = [
        ConversationMessage(
            role="assistant",
            content=_router_history_record(
                acting_character_id="the_master",
                result=result,
            ),
        )
    ]
    assert refresh_router_history_record(
        conversation,
        result=result,
        one_star_engine_characters=[hero],
        one_star_engine_updates=["stamina_recovered current=5 recovery_anchor_s=1800"],
        force=True,
    )

    record = conversation[0].content
    assert 'one_star_update {"kind":"hero_delta","target_id":"pip"' in record
    assert '"hp_current=2"' in record
    assert "one_star_authority_hero" in record
    assert '"character_id":"new_hero"' in record
    assert '"strength":3' in record
    assert '"skill_id":"knead_dough"' in record
    assert '"item_id":"bent_knife"' in record
    assert "one_star_authority_update stamina_recovered current=5" in record
    assert "owner_lobby_id" not in record
    assert "acquisition_event_id" not in record
    assert "hidden_capabilities" not in record
    assert "progression_seed" not in record
    assert "potential_grade" not in record
    assert record.count("one_star_authority_hero") == 1

    # Later fact/observer refreshes must not erase one-time authority records.
    assert refresh_router_history_record(
        conversation,
        result=result,
        force=True,
    )
    refreshed = conversation[0].content
    assert refreshed.count("one_star_authority_hero") == 1
    assert refreshed.count("one_star_authority_update") == 1


def test_human_and_agent_submissions_share_the_same_router_envelope():
    human_submission = format_actor_submission("alice", "I open the gate.")
    agent_submission = format_actor_submission("alice", "I open the gate.")

    assert human_submission == agent_submission
    assert "player" not in human_submission
    assert "agent" not in human_submission


def test_one_star_prompt_addon_does_not_leak_runtime_implementation_terms():
    addon = (
        PromptManager("app/prompts").prompts_dir / "event_router_ruleset_one_star.txt"
    ).read_text()

    for forbidden in ("engine", "pipeline", "dispatcher", "SDK", "API", "Python"):
        assert forbidden not in addon
