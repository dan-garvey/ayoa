from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.engine.checkpoint_manager import CheckpointManager
from app.engine.closed_event_runtime import (
    ClosedEventRuntime,
    closed_event_runtime_for,
    install_closed_event_runtime,
)
from app.engine.narrator import compose_pov_render
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.engine.spawn_authoring import SpawnAuthoringCoordinator
from app.engine.turn_loop import _end_beat, run_beat
from app.engine.turn_loop_dispatcher import (
    _router_history_record,
    refresh_router_history_record,
)
from app.schemas.characters import (
    CharacterDescriptions,
    CharacterRecord,
    CharacterVisuals,
    PublicSheet,
)
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import EventRouterOutput, SpawnRequest
from app.schemas.events import ObservableFact
from tests.support.factories import (
    gatehouse_checkpoint,
    narrator_llm_response,
    router_output,
)


_AUTHORED = {
    "summon_sword": (
        "Davan Corse",
        "cautious one-star swordsman",
        "A broad rusted sword hangs from a patched leather belt.",
    ),
    "summon_staff": (
        "Petra Vale",
        "earnest one-star staff-wielder",
        "A plain ash staff rests in both callused hands.",
    ),
    "summon_knife": (
        "Tam Rill",
        "skittish one-star knife-fighter",
        "A short utility knife sits in a frayed cloth sheath.",
    ),
}


class RecordingCharacterManager:
    def __init__(self, *, fail_if_called: bool = False) -> None:
        self.fail_if_called = fail_if_called
        self.calls = 0

    async def spawn_characters(
        self,
        checkpoint,
        requests,
        *,
        acting_actor_location,
    ):
        if self.fail_if_called:
            raise AssertionError("durable pending spawns were re-authored")
        self.calls += 1
        records = []
        for request in requests:
            name, role, loadout = _AUTHORED[request.character_id]
            records.append(CharacterRecord(
                character_id=request.character_id,
                name=name,
                location=request.seed.location or acting_actor_location,
                public_sheet=PublicSheet(
                    role=role,
                    appearance=f"Visible appearance for {name}.",
                ),
                descriptions=CharacterDescriptions(
                    public=f"Player-safe public identity for {name}.",
                ),
                visuals=CharacterVisuals(default_loadout=loadout),
            ))
        return records


class QueueNarratorClient:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        handoff, reason, text = outcome
        return narrator_llm_response(
            text,
            handoff=handoff,
            handoff_reason=reason,
        )


class TransactionDispatcher:
    def __init__(
        self,
        *,
        routes=(),
        narrator_outcomes=(("render", "ready", "RENDERED"),),
        agent_outputs=(),
        mutate_intro_on_continue: bool = False,
    ) -> None:
        self.routes = list(routes)
        self.agent_outputs = list(agent_outputs)
        self.narrator_client = QueueNarratorClient(narrator_outcomes)
        self.prompt_manager = PromptManager("app/prompts")
        self.narrator_rosters: list[dict[str, CharacterRecord]] = []
        self.narrator_introductions: list[dict[str, list[str]]] = []
        self.agent_records: list[CharacterRecord] = []
        self.continuation_rosters: list[dict[str, CharacterRecord]] = []
        self.continuation_introductions: list[dict[str, list[str]]] = []
        self.mutate_intro_on_continue = mutate_intro_on_continue

    @staticmethod
    def _remember_router_result(
        ckpt,
        result: EventRouterOutput,
        *,
        actor_id: str,
        mode: str,
    ) -> None:
        ckpt.session_conversation.append(ConversationMessage(
            role="assistant",
            content=_router_history_record(
                acting_character_id=actor_id,
                result=result,
                mode=mode,
            ),
        ))

    async def route_intention(self, **kwargs) -> EventRouterOutput:
        result = self.routes.pop(0)
        self._remember_router_result(
            kwargs["ckpt"],
            result,
            actor_id=kwargs["actor_id"],
            mode="intention",
        )
        return result

    async def route_continuation(self, **kwargs) -> EventRouterOutput:
        ckpt = kwargs["ckpt"]
        self.continuation_rosters.append({
            character.character_id: character
            for character in ckpt.characters
        })
        self.continuation_introductions.append({
            viewer_id: list(character_ids)
            for viewer_id, character_ids in (
                ckpt.session.visual_introductions.items()
            )
        })
        result = self.routes.pop(0)
        self._remember_router_result(
            ckpt,
            result,
            actor_id=kwargs["actor_id"],
            mode="continuation",
        )
        return result

    async def materialize_spawns(self, **kwargs) -> list[str]:
        ckpt = kwargs["ckpt"]
        result = kwargs["result"]
        runtime = closed_event_runtime_for(ckpt)
        assert runtime is not None
        records = await runtime.authored_records(
            checkpoint=ckpt,
            event=result,
            actor_id=kwargs["actor_id"],
        )
        applied = runtime.apply_records(ckpt, records)
        refresh_router_history_record(
            ckpt.session_conversation,
            result=result,
            spawned_characters=records,
        )
        return applied

    async def agent_intend(self, **kwargs) -> str:
        character = next(
            character
            for character in kwargs["ckpt"].characters
            if character.character_id == kwargs["character_id"]
        )
        self.agent_records.append(character)
        return self.agent_outputs.pop(0)

    async def narrator_compose(self, **kwargs):
        ckpt = kwargs["ckpt"]
        self.narrator_rosters.append({
            character.character_id: character
            for character in ckpt.characters
        })
        self.narrator_introductions.append({
            viewer_id: list(character_ids)
            for viewer_id, character_ids in (
                ckpt.session.visual_introductions.items()
            )
        })
        result = await compose_pov_render(
            self.narrator_client,
            self.prompt_manager,
            ckpt,
            kwargs["character_id"],
            kwargs["buffered_events"],
            kwargs.get("partial_mode_override") or False,
            user_input=kwargs.get("user_input", ""),
            handoff_policy=kwargs.get("handoff_policy", "forced"),
            handoff_context=kwargs.get("handoff_context", ""),
        )
        if self.mutate_intro_on_continue and result[0].handoff == "continue":
            ckpt.session.visual_introductions[kwargs["character_id"]] = list(
                _AUTHORED
            )
        return result


def _checkpoint():
    return gatehouse_checkpoint(
        bindings={"alice": "u1"},
        player_character_id="alice",
    )


def _spawn_requests() -> list[SpawnRequest]:
    return [
        SpawnRequest(
            character_id=character_id,
            seed={
                "role": role,
                "reason": "the first summon wave arrives",
                "location": "summoning_hall",
                "objectives": [f"act as the {role}"],
                "knowledge_tier": 1,
            },
        )
        for character_id, (_name, role, _loadout) in _AUTHORED.items()
    ]


def _spawn_event(*, next_output: str = "") -> EventRouterOutput:
    ids = list(_AUTHORED)
    return router_output(
        event_id="evt_three_arrive",
        event_kind="beat_continues" if next_output else "cascade_exhausted",
        agent_ids=[next_output] if next_output else [],
        observer_ids=["alice", *ids],
        facts=[ObservableFact.all(
            f"{ids[0]}, {ids[1]}, and {ids[2]} step out of the summon light."
        )],
        spawn=_spawn_requests(),
    )


def _install_runtime(ckpt, manager, *, transaction_id: str):
    coordinator = SpawnAuthoringCoordinator(manager)
    runtime = ClosedEventRuntime(
        transaction_id=transaction_id,
        source_turn_index=1,
        spawn_authoring=coordinator,
        record_applier=Orchestrator._apply_authored_spawn_records,
    )
    install_closed_event_runtime(ckpt, runtime)
    return coordinator, runtime


def _narrator_prompt_text(dispatcher: TransactionDispatcher, index: int) -> str:
    messages = dispatcher.narrator_client.calls[index]["messages"]
    return "\n".join(str(message.get("content", "")) for message in messages)


@pytest.mark.asyncio
async def test_heterogeneous_spawns_are_in_first_render_and_accept_once():
    ckpt = _checkpoint()
    manager = RecordingCharacterManager()
    coordinator, runtime = _install_runtime(
        ckpt,
        manager,
        transaction_id="tx_first_render",
    )
    event = _spawn_event()
    dispatcher = TransactionDispatcher(routes=[event])

    result = await run_beat(
        ckpt=ckpt,
        dispatcher=dispatcher,
        actor_id="alice",
        intention="I confirm the summon.",
    )
    records = await coordinator.result(runtime.spawn_keys_by_event_id[event.event_id])

    assert manager.calls == 1
    assert result.renders == {"alice": "RENDERED"}
    for record in records:
        assert dispatcher.narrator_rosters[0][record.character_id] is record
        assert next(
            character
            for character in ckpt.characters
            if character.character_id == record.character_id
        ) is record

    prompt = _narrator_prompt_text(dispatcher, 0)
    for character_id, (name, _role, loadout) in _AUTHORED.items():
        assert name in prompt
        assert loadout in prompt
        assert character_id not in prompt

    orchestrator = Orchestrator(
        client=SimpleNamespace(),
        checkpoint_mgr=SimpleNamespace(),
        prompt_mgr=PromptManager("app/prompts"),
        spawn_authoring=coordinator,
    )
    await orchestrator._apply_beat_roster_side_effects(
        ckpt,
        result,
        log_label="accepted spawn beat",
    )
    await orchestrator._apply_beat_roster_side_effects(
        ckpt,
        result,
        log_label="idempotent replay",
    )
    for character_id in _AUTHORED:
        assert [
            character.character_id for character in ckpt.characters
        ].count(character_id) == 1
    compact = "\n".join(
        str(message.content) for message in ckpt.session_conversation
    )
    assert "spawn summon_sword name=Davan Corse" in compact
    assert "spawn summon_staff name=Petra Vale" in compact
    assert "spawn summon_knife name=Tam Rill" in compact


@pytest.mark.asyncio
async def test_immediate_spawn_responder_reuses_shared_authored_record():
    ckpt = _checkpoint()
    manager = RecordingCharacterManager()
    coordinator, runtime = _install_runtime(
        ckpt,
        manager,
        transaction_id="tx_immediate",
    )
    event = _spawn_event(next_output="summon_staff")
    dispatcher = TransactionDispatcher(
        routes=[
            event,
            router_output(
                event_id="evt_staff_answers",
                event_kind="cascade_exhausted",
                observer_ids=["alice", *list(_AUTHORED)],
                facts=[ObservableFact.all(
                    "summon_staff plants the ash staff and answers Alice."
                )],
            ),
        ],
        agent_outputs=["I ask where we have been summoned."],
    )

    await run_beat(
        ckpt=ckpt,
        dispatcher=dispatcher,
        actor_id="alice",
        intention="I wait for one of them to speak.",
    )
    records = await coordinator.result(runtime.spawn_keys_by_event_id[event.event_id])
    by_id = {record.character_id: record for record in records}

    assert manager.calls == 1
    assert dispatcher.agent_records == [by_id["summon_staff"]]
    assert dispatcher.agent_records[0] is by_id["summon_staff"]
    assert dispatcher.agent_records[0].name == "Petra Vale"
    assert "ash staff" in dispatcher.agent_records[0].visuals.default_loadout
    assert dispatcher.narrator_rosters[0]["summon_staff"] is by_id["summon_staff"]
    for record in records:
        assert any(
            "step out of the summon light" in observation
            for observation in record.pending_observations
        )


@pytest.mark.asyncio
async def test_narrator_continue_rolls_back_then_restores_same_overlay():
    ckpt = _checkpoint()
    manager = RecordingCharacterManager()
    coordinator, runtime = _install_runtime(
        ckpt,
        manager,
        transaction_id="tx_continue",
    )
    event = _spawn_event()
    dispatcher = TransactionDispatcher(
        routes=[
            event,
            router_output(
                event_id="evt_light_fades",
                event_kind="cascade_exhausted",
                observer_ids=["alice", *list(_AUTHORED)],
                facts=[ObservableFact.all(
                    "The summon light fades around summon_sword, "
                    "summon_staff, and summon_knife."
                )],
            ),
        ],
        narrator_outcomes=[
            ("continue", "the arrival is still resolving", "DISCARDED"),
            ("render", "the arrival is complete", "ACCEPTED"),
        ],
        mutate_intro_on_continue=True,
    )

    result = await run_beat(
        ckpt=ckpt,
        dispatcher=dispatcher,
        actor_id="alice",
        intention="I watch the summon finish.",
    )
    records = await coordinator.result(runtime.spawn_keys_by_event_id[event.event_id])
    by_id = {record.character_id: record for record in records}

    assert manager.calls == 1
    assert result.renders == {"alice": "ACCEPTED"}
    assert set(dispatcher.continuation_rosters[0]) >= set(_AUTHORED)
    for character_id, record in by_id.items():
        assert dispatcher.continuation_rosters[0][character_id] is record
        assert dispatcher.narrator_rosters[0][character_id] is record
        assert dispatcher.narrator_rosters[1][character_id] is record
    assert not (
        set(dispatcher.continuation_introductions[0].get("alice", []))
        & set(_AUTHORED)
    )
    assert set(ckpt.session.visual_introductions["alice"]) >= set(_AUTHORED)
    assert "DISCARDED" not in str(ckpt.narrator_conversations["alice"])
    assert "ACCEPTED" in str(ckpt.narrator_conversations["alice"])


@pytest.mark.asyncio
async def test_failed_render_persists_inactive_records_and_restart_reuses_them(
    tmp_path,
):
    ckpt = _checkpoint()
    manager = RecordingCharacterManager()
    coordinator, runtime = _install_runtime(
        ckpt,
        manager,
        transaction_id="tx_failed",
    )
    alice = next(
        character
        for character in ckpt.characters
        if character.character_id == "alice"
    )
    alice.descriptions.public = "The summoner who called the trio here."
    alice.visuals.default_loadout = (
        "A weathered blue coat hangs over a silver summoning sash."
    )
    event = _spawn_event(next_output="summon_staff")
    event.canonical_event.observable_facts = [ObservableFact.all(
        'Alice says, "Welcome," as summon_sword, summon_staff, and '
        "summon_knife step out of the summon light."
    )]
    dispatcher = TransactionDispatcher(
        routes=[
            event,
            router_output(
                event_id="evt_staff_answers_before_failure",
                event_kind="cascade_exhausted",
                observer_ids=["alice", *list(_AUTHORED)],
                facts=[ObservableFact.all(
                    "summon_staff plants the ash staff and answers Alice."
                )],
            ),
        ],
        narrator_outcomes=[RuntimeError("narrator offline")],
        agent_outputs=["I ask where we have been summoned."],
    )
    checkpoint_manager = CheckpointManager(str(tmp_path / "sessions"))
    dispatcher.persist_pending_narrator_render = checkpoint_manager.save

    with pytest.raises(RuntimeError, match="narrator offline"):
        await run_beat(
            ckpt=ckpt,
            dispatcher=dispatcher,
            actor_id="alice",
            intention="I confirm the summon.",
        )

    assert manager.calls == 1
    assert [record.character_id for record in dispatcher.agent_records] == [
        "summon_staff"
    ]
    expected_introductions = {
        character_id: ["alice"] for character_id in _AUTHORED
    }
    assert {
        character_id: dispatcher.narrator_introductions[0][character_id]
        for character_id in _AUTHORED
    } == expected_introductions
    assert not set(_AUTHORED) & {
        character.character_id for character in ckpt.characters
    }
    assert not set(_AUTHORED) & set(
        ckpt.session.visual_introductions.get("alice", [])
    )
    pending = ckpt.session.pending_narrator_render
    assert pending is not None
    expected_records = {
        record.character_id: record.model_dump()
        for record in pending.pending_spawn_records
    }
    assert set(expected_records) == set(_AUTHORED)
    assert pending.pending_spawn_introductions == expected_introductions

    coordinator.discard_transaction(
        runtime.transaction_id,
        cancel_running=False,
    )
    reloaded = checkpoint_manager.load_latest(ckpt.session.session_id)
    assert not set(_AUTHORED) & {
        character.character_id for character in reloaded.characters
    }
    reloaded_pending = reloaded.session.pending_narrator_render
    assert reloaded_pending is not None
    assert {
        record.character_id: record.model_dump()
        for record in reloaded_pending.pending_spawn_records
    } == expected_records
    assert (
        reloaded_pending.pending_spawn_introductions
        == expected_introductions
    )
    assert not set(_AUTHORED) & set(
        reloaded.session.visual_introductions
    )

    restart_manager = RecordingCharacterManager(fail_if_called=True)
    _restart_coordinator, _restart_runtime = _install_runtime(
        reloaded,
        restart_manager,
        transaction_id="tx_retry_after_restart",
    )
    retry_dispatcher = TransactionDispatcher()
    retry_result = await _end_beat(
        reloaded,
        retry_dispatcher,
        ended_reason=reloaded_pending.ended_reason,
        events_closed=reloaded_pending.events_closed,
        event_actor_ids=list(reloaded_pending.event_actor_ids),
        release_slots=reloaded_pending.release_slots,
        force_partial=reloaded_pending.force_partial,
        acting_player_id=reloaded_pending.acting_player_id,
        acting_player_input=reloaded_pending.acting_player_input,
        suppress_reaction_prompts=reloaded_pending.suppress_reaction_prompts,
        soft_handoff_candidate=reloaded_pending.soft_handoff_candidate,
    )

    assert restart_manager.calls == 0
    assert retry_result.renders == {"alice": "RENDERED"}
    assert reloaded.session.pending_narrator_render is None
    assert not set(_AUTHORED) & set(
        retry_dispatcher.narrator_introductions[0]
    )
    for character_id, expected in expected_records.items():
        character = next(
            character
            for character in reloaded.characters
            if character.character_id == character_id
        )
        assert character.model_dump() == expected
        assert reloaded.session.visual_introductions[character_id] == [
            "alice"
        ]
    assert set(reloaded.session.visual_introductions["alice"]) >= set(_AUTHORED)
    retry_prompt = _narrator_prompt_text(retry_dispatcher, 0)
    for character_id, (name, _role, loadout) in _AUTHORED.items():
        assert name in retry_prompt
        assert loadout in retry_prompt
        assert character_id not in retry_prompt
