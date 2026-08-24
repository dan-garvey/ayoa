"""Orchestrator integration tests for the v11 turn path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine import dice, dnd_experience, dnd_monsters
from app.engine.action_rejection import PlayerActionRejected
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.dnd_combat import apply_damage, current_combatant
from app.engine.imported_statblocks import ImportedStatBlockNotFoundError
from app.engine.orchestrator import (
    Orchestrator,
    _turn_response_from_beat_results,
    _with_pre_turn_resolutions,
)
from app.engine.turn_loop import BeatResult, broadcast_event
from app.schemas.checkpoint import CheckpointFile
from app.schemas.characters import CharacterStatus
from app.schemas.conversation import ConversationMessage
from app.schemas.content import ContentPackState
from app.schemas.content_pack import SafeAssetRevealPayload
from app.schemas.event_router import (
    DndEventRouterOutput,
    EventRouterOutput,
    ObserverEntry,
    empty_commitment_open_signal,
)
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    WorldAdjudication,
)
from app.schemas.requests import TurnRequest
from app.schemas.one_star import OneStarEventRouterOutput
from app.schemas.state import (
    CatIIRollRecord,
    CatIIRollTransaction,
    CommitmentRevisionPrompt,
    DndCombatantState,
    DndCombatState,
    DndExperienceAwardDisplay,
    OpenCatIIEvent,
    OpenCommitment,
    PendingNarratorRender,
    RenderBufferEntry,
    SlotEntry,
)
from app.schemas.responses import TurnResponse
from tests.support.factories import (
    ClassFakeDispatcher as FakeDispatcher,
    character_record,
    dnd5e_mechanics as _dnd_mechanics,
    dnd_router_output as _dnd_router_out,
    gatehouse_checkpoint,
    router_output as _router_out,
)


def _ckpt(bindings: dict[str, str] | None = None) -> CheckpointFile:
    return gatehouse_checkpoint(
        bindings=bindings or {"alice": "u1"},
        player_character_id="alice",
        pip_role="guard",
    )


def _rat_combatant_spawn() -> dict:
    return {
        "character_id": "rat_1",
        "monster_key": "rat",
        "name": "Rat",
        "location": "",
        "description": "A small rat snaps at exposed ankles.",
        "statblock": {
            "size": "Tiny",
            "creature_type": "beast",
            "alignment": "unaligned",
            "armor_class": 10,
            "hit_points": 1,
            "hit_dice": "1d4 - 1",
            "speed": "20 ft.",
            "ability_scores": {
                "strength": 2,
                "dexterity": 11,
                "constitution": 9,
                "intelligence": 2,
                "wisdom": 10,
                "charisma": 4,
            },
            "proficiency_bonus": 2,
            "skills": [],
            "senses": ["darkvision 30 ft."],
            "passive_perception": 10,
            "languages": [],
            "challenge_rating": "0",
            "xp": 10,
            "traits": [],
            "actions": [
                {
                    "action_id": "bite",
                    "name": "Bite",
                    "attack_bonus": 0,
                    "reach_ft": 5,
                    "range_normal_ft": 0,
                    "range_long_ft": 0,
                    "target": "one target",
                    "damage": "1 piercing",
                    "damage_type": "piercing",
                    "description": (
                        "Melee Weapon Attack: +0 to hit, reach 5 ft., one "
                        "target. Hit: 1 piercing damage."
                    ),
                }
            ],
        },
    }


def _guardian_statblock() -> dict:
    return {
        "ref": "stat.guardian",
        "content_hash": "sha256:stat-guardian",
        "title": "Synthetic Guardian",
        "summary": "A combat-ready synthetic guardian.",
        "body": "Private source notes at /private/source.pdf must not be copied.",
        "confidence": 0.98,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "automation_scope": "combat",
        "size": "Medium",
        "creature_type": "Construct",
        "alignment": "Unaligned",
        "armor_class": 15,
        "hit_points": 33,
        "hit_dice": "6d8+6",
        "speed_ft_by_mode": {"walk": 30},
        "ability_scores": {
            "strength": 16,
            "dexterity": 14,
            "constitution": 13,
            "intelligence": 8,
            "wisdom": 12,
            "charisma": 7,
        },
        "proficiency_bonus": 2,
        "senses": ["darkvision 60 ft."],
        "passive_perception": 15,
        "languages": ["understands its creator"],
        "challenge_rating": "2",
        "xp": 450,
        "actions": [
            {
                "feature_id": "slam",
                "name": "Slam",
                "economy": "action",
                "attack_bonus": 5,
                "reach_ft": 5,
                "target": "one target",
                "damage": [
                    {"expression": "1d8+3", "damage_type": "bludgeoning"}
                ],
                "description": "Melee Weapon Attack.",
            }
        ],
    }


def _encounter_template() -> dict:
    return {
        "pack_id": "synthetic-pack",
        "ref": "enc.entry",
        "content_hash": "sha256:enc-entry",
        "title": "Synthetic Entry Encounter",
        "summary": "A reviewed encounter seed.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "trigger": "The party enters the watched area.",
        "location_refs": ["gatehouse"],
        "participants": [
            {
                "participant_id": "guardian",
                "statblock_ref": "stat.guardian",
                "count": 1,
                "role": "sentinel",
                "starting_anchor_ref": "spawn.enemies",
                "tactics": "Hold the gatehouse threshold.",
            }
        ],
        "map_template_refs": ["map.entry"],
        "trap_refs": ["trap.floor"],
        "treasure_refs": ["treasure.cache"],
    }


def _encounter_map_template() -> dict:
    return {
        "pack_id": "synthetic-pack",
        "ref": "map.entry",
        "content_hash": "sha256:map-entry",
        "title": "Entry Map",
        "summary": "Reviewed entry map.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "derived_from_map_asset_id": "asset.map.entry.player",
        "grid_width": 8,
        "grid_height": 6,
        "spawn_anchors": [
            {
                "anchor_id": "spawn.enemies",
                "anchor_kind": "enemies",
                "cells": [{"x": 5, "y": 3}],
                "label": "Enemy start",
            }
        ],
    }


def _encounter_trap() -> dict:
    return {
        "pack_id": "synthetic-pack",
        "ref": "trap.floor",
        "content_hash": "sha256:trap-floor",
        "title": "Entry Floor Trap",
        "summary": "A reviewed floor trap.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "trigger": "A creature crosses the marked stones.",
        "detection": "A seam crosses the floor.",
        "countermeasures": ["Jam the floor plate"],
        "linked_location_refs": ["gatehouse"],
        "placements": [
            {
                "placement_id": "place.floor",
                "location_ref": "gatehouse",
                "map_template_ref": "map.entry",
                "map_feature_ref": "feature.floor",
                "bounds": {"x": 2, "y": 2, "width": 1, "height": 1},
            }
        ],
        "mechanics": {
            "detection_dc": 13,
            "disarm_dc": 14,
            "save_dc": 12,
            "save_ability": "dexterity",
            "damage": [{"expression": "2d6", "damage_type": "piercing"}],
            "depletion_ref": "depleted.trap.floor",
        },
    }


def _encounter_treasure() -> dict:
    return {
        "pack_id": "synthetic-pack",
        "ref": "treasure.cache",
        "content_hash": "sha256:treasure-cache",
        "title": "Entry Cache",
        "summary": "A reviewed cache.",
        "confidence": 0.95,
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "treasure_kind": "container",
        "container_ref": "container.cache",
        "currency": [{"denomination": "gp", "amount": 10}],
    }


def _safe_asset_payload(asset_id: str) -> SafeAssetRevealPayload:
    return SafeAssetRevealPayload(
        pack_id="synthetic",
        asset_id=asset_id,
        kind="handout",
        title=asset_id.replace("-", " ").title(),
        mime_type="image/png",
        width=320,
        height=180,
        sha256=f"hash-{asset_id}",
        delivery_ref=f"asset://synthetic/{asset_id}",
        presentation="attachment",
        caption=f"{asset_id} caption",
        alt_text=f"{asset_id} alt text",
    )


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeDispatcher.reset()
    yield
    FakeDispatcher.reset()


def test_with_pre_turn_resolutions_preserves_asset_payloads_separately():
    pre_payload = _safe_asset_payload("pre-turn-handout")
    main_payload = _safe_asset_payload("main-turn-map")
    pre_turn = TurnResponse(
        session_id="s",
        checkpoint_id="ckpt_0001",
        turn_index=1,
        output_text="The room shifts before Alice acts.",
        asset_reveals=[pre_payload],
        per_player_asset_reveals={"bob": [pre_payload]},
    )
    main = TurnResponse(
        session_id="s",
        checkpoint_id="ckpt_0002",
        turn_index=2,
        output_text="Alice studies the map.",
        asset_reveals=[main_payload],
        per_player_asset_reveals={"alice": [main_payload]},
    )

    response = _with_pre_turn_resolutions(main, [pre_turn])

    assert response.asset_reveals == [main_payload]
    assert response.per_player_asset_reveals == {"alice": [main_payload]}
    assert response.pre_turn_resolutions == [pre_turn]
    assert response.pre_turn_resolutions[0].asset_reveals == [pre_payload]
    assert response.pre_turn_resolutions[0].per_player_asset_reveals == {
        "bob": [pre_payload]
    }


def test_turn_response_xp_award_drain_is_persisted(patched_orchestrator):
    ckpt = _ckpt(bindings={"alice": "u1"})
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
    award = DndExperienceAwardDisplay(
        character_id="alice",
        character_name="Alice",
        amount=25,
        source="Rat",
        experience_points=25,
        total_level=1,
        eligible_level=1,
        next_level=2,
        xp_to_next_level=275,
    )
    ckpt.session.active_combat = DndCombatState(
        combat_id="combat-xp",
        combatants=[
            DndCombatantState(
                combatant_id="alice",
                character_id="alice",
                name="Alice",
                player_controlled=True,
                armor_class=12,
                hit_points_current=13,
                hit_points_max=13,
            )
        ],
        pending_experience_awards=[award],
    )
    orch, mgr = patched_orchestrator(ckpt)

    response = _turn_response_from_beat_results(
        session_id="s",
        ckpt=ckpt,
        acting_id="alice",
        beat_results=[
            BeatResult(
                renders={"alice": "The combat quiets."},
                events_closed=0,
                ended_reason="cascade_exhausted",
                transcript_entries={},
                event_actor_ids=[],
            )
        ],
        roll_keys_before=set(),
    )
    assert response is not None
    assert response.experience_awards == [award]
    assert ckpt.session.active_combat.pending_experience_awards == []

    orch._save_if_response_drained_runtime_state(ckpt, response)

    mgr.save.assert_called_once_with(ckpt)


@pytest.fixture
def patched_orchestrator(monkeypatch):
    monkeypatch.setattr("app.engine.orchestrator.LLMDispatcher", FakeDispatcher)
    client = MagicMock()
    client.config = MagicMock()
    prompt_mgr = MagicMock()

    def _factory(ckpt: CheckpointFile):
        mgr = MagicMock()
        mgr.load_latest.return_value = ckpt
        mgr.save = MagicMock()
        return Orchestrator(client, mgr, prompt_mgr), mgr

    return _factory


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_real_infeasible_router_result_uses_rejection_rollback(
        self,
        patched_orchestrator,
        monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        orch, mgr = patched_orchestrator(ckpt)
        from app.engine.model_config_sync import sync_checkpoint_runtime_models

        sync_checkpoint_runtime_models(ckpt, orch.client.config)
        before = ckpt.model_dump_json()
        rejected_data = _router_out(observer_ids=[], facts=[]).model_dump()
        rejected_data["state_updates"] = []
        rejected = OneStarEventRouterOutput.model_validate(rejected_data)
        rejected.canonical_event.world_adjudication.feasible = False
        FakeDispatcher.queue_route(rejected)
        original_route = FakeDispatcher.route_intention

        async def route_with_compact_history(self, **kwargs):
            kwargs["ckpt"].session_conversation.append(ConversationMessage(
                role="assistant",
                content="prior_event rejected-but-not-canonical",
            ))
            return await original_route(self, **kwargs)

        monkeypatch.setattr(
            FakeDispatcher,
            "route_intention",
            route_with_compact_history,
        )

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I buy an unavailable quantity of Gems.",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "player_action_infeasible"
        assert "Nothing changed" in response.output_text
        assert response.per_player_renders == {}
        assert ckpt.model_dump_json() == before
        assert ckpt.canonical_events == []
        assert ckpt.session_conversation == []
        assert FakeDispatcher.narrator_calls == []
        mgr.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_player_action_rejection_returns_message_without_saving_or_narrating(
        self, patched_orchestrator, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        orch, mgr = patched_orchestrator(ckpt)
        from app.engine.model_config_sync import sync_checkpoint_runtime_models

        sync_checkpoint_runtime_models(ckpt, orch.client.config)
        before = ckpt.model_dump_json()
        rejected_run = AsyncMock(side_effect=PlayerActionRejected(
            "Premium summon rejected: 5 pulls cost 25 Gems, but only 5 "
            "Gems are available. Nothing was spent and no Heroes were "
            "summoned.",
            reason="one_star_summon_rejected",
        ))
        monkeypatch.setattr(
            "app.engine.orchestrator.run_beat",
            rejected_run,
        )

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I perform five premium summons.",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "one_star_summon_rejected"
        assert response.per_player_renders == {}
        assert "5 pulls cost 25 Gems" in response.output_text
        assert "Nothing was spent" in response.output_text
        assert ckpt.model_dump_json() == before
        assert FakeDispatcher.narrator_calls == []
        mgr.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_cat_i_close_populates_renders_and_saves(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(event_kind="cascade_exhausted"))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I look around",
            acting_character_id="alice",
        ))

        assert response.per_player_renders["alice"] == "POV_RENDER"
        assert response.output_text == "POV_RENDER"
        assert response.beat_ended_reason == "cascade_exhausted"
        assert response.turn_index == 1
        assert mgr.save.call_count == 1
        saved = mgr.save.call_args[0][0]
        assert len(saved.canonical_events) == 1
        assert saved.session.active_act_slots == {}
        assert not hasattr(saved, "transcript")

    @pytest.mark.asyncio
    async def test_defer_resumes_latest_autonomous_handoff(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        prior = _router_out(
            event_id="evt_waiting_on_pip",
            event_kind="response_requested",
            agent_ids=["pip"],
            observer_ids=["alice", "pip"],
            facts=[ObservableFact.all("Alice waits for Pip's answer.")],
        )
        ckpt.canonical_events.append(prior)
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_agent("Pip gives Alice a direct answer.")
        FakeDispatcher.queue_route(_router_out(
            event_id="evt_pip_answers",
            event_kind="cascade_exhausted",
            observer_ids=["alice", "pip"],
            facts=[ObservableFact.all("Pip answers Alice plainly.")],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="(defer)",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "cascade_exhausted"
        assert [call["character_id"] for call in FakeDispatcher.agent_calls] == [
            "pip"
        ]
        assert len(FakeDispatcher.route_calls) == 1
        assert FakeDispatcher.route_calls[0]["actor_id"] == "pip"
        assert FakeDispatcher.route_calls[0]["intention"] == (
            "Pip gives Alice a direct answer."
        )
        assert [event.event_id for event in ckpt.canonical_events] == [
            "evt_waiting_on_pip",
            "evt_pip_answers",
        ]
        assert FakeDispatcher.narrator_calls[-1]["user_input"] == "(defer)"
        assert "(defer)" in str(ckpt.narrator_conversations["alice"])
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_defer_does_not_resume_a_bound_handoff(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.canonical_events.append(_router_out(
            event_id="evt_waiting_on_bob",
            event_kind="response_requested",
            agent_ids=["bob"],
            observer_ids=["alice", "bob"],
            facts=[ObservableFact.all("Alice waits for Bob's answer.")],
        ))
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(
            event_id="evt_alice_defers",
            event_kind="cascade_exhausted",
            observer_ids=["alice", "bob"],
        ))

        await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="(defer)",
            acting_character_id="alice",
        ))

        assert FakeDispatcher.agent_calls == []
        assert len(FakeDispatcher.route_calls) == 1
        assert FakeDispatcher.route_calls[0]["actor_id"] == "alice"
        assert FakeDispatcher.route_calls[0]["intention"] == "(defer)"

    @pytest.mark.asyncio
    async def test_narrator_failure_preserves_beat_for_render_retry(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(event_kind="cascade_exhausted"))
        FakeDispatcher.queue_narrator_error(RuntimeError("narrator offline"))

        with pytest.raises(RuntimeError, match="narrator offline"):
            await orch.process_turn(TurnRequest(
                session_id="s",
                user_input="I look around",
                acting_character_id="alice",
            ))

        assert mgr.save.call_count == 1
        assert ckpt.session.turn_index == 1
        assert ckpt.session.pending_narrator_render is not None
        assert ckpt.session.pending_narrator_render.acting_player_input == (
            "I look around"
        )
        assert len(ckpt.canonical_events) == 1
        assert ckpt.session.render_buffers["alice"]
        assert ckpt.session.active_act_slots["alice"].reason == "initiator"
        assert not hasattr(ckpt, "transcript")
        assert len(FakeDispatcher.route_calls) == 1
        assert len(FakeDispatcher.narrator_calls) == 1

        response = await orch.retry_pending_narrator_render("s")

        assert response.output_text == "POV_RENDER"
        assert response.turn_index == 1
        assert ckpt.session.pending_narrator_render is None
        assert ckpt.session.render_buffers["alice"] == []
        assert ckpt.session.active_act_slots == {}
        assert not hasattr(ckpt, "transcript")
        assert len(FakeDispatcher.route_calls) == 1
        assert len(FakeDispatcher.narrator_calls) == 2
        assert mgr.save.call_count == 2

    @pytest.mark.asyncio
    async def test_soft_handoff_retry_can_continue_without_replaying_action(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(
            event_id="evt_waiting",
            event_kind="cascade_exhausted",
            facts=[ObservableFact.all("The lift starts to descend.")],
        ))
        FakeDispatcher.queue_route(_router_out(
            event_id="evt_arrived",
            event_kind="cascade_exhausted",
            facts=[ObservableFact.all("The lift reaches the lower hall.")],
        ))
        FakeDispatcher.queue_narrator_error(RuntimeError("narrator offline"))

        with pytest.raises(RuntimeError, match="narrator offline"):
            await orch.process_turn(TurnRequest(
                session_id="s",
                user_input="I ride the lift to the lower hall.",
                acting_character_id="alice",
            ))

        pending = ckpt.session.pending_narrator_render
        assert pending is not None
        assert pending.soft_handoff_candidate is True
        assert pending.handoff_event_id == "evt_waiting"
        FakeDispatcher.queue_narrator(
            handoff="continue",
            reason="The lift is still moving.",
            text="DISCARDED",
        )
        FakeDispatcher.queue_narrator(
            handoff="render",
            reason="The lift has arrived.",
            text="ARRIVED",
        )

        response = await orch.retry_pending_narrator_render("s")

        assert response.output_text == "ARRIVED"
        assert len(FakeDispatcher.route_calls) == 2
        assert FakeDispatcher.route_calls[1]["original_action"] == (
            "I ride the lift to the lower hall."
        )
        assert "handoff_reason" not in FakeDispatcher.route_calls[1]
        assert [
            len(call["buffered_events"])
            for call in FakeDispatcher.narrator_calls
        ] == [1, 1, 2]
        assert "DISCARDED" not in str(ckpt.narrator_conversations["alice"])
        assert ckpt.session.pending_narrator_render is None
        assert mgr.save.call_count == 2

    @pytest.mark.asyncio
    async def test_continue_then_failure_rebuilds_the_pending_retry_snapshot(
        self, patched_orchestrator, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(
            event_id="evt_waiting",
            event_kind="cascade_exhausted",
            dormant=["pip"],
            facts=[ObservableFact.all("The lift starts to descend.")],
        ))
        FakeDispatcher.queue_route(_router_out(
            event_id="evt_arrived",
            event_kind="cascade_exhausted",
            facts=[ObservableFact.all("The lift reaches the lower hall.")],
        ))
        FakeDispatcher.queue_narrator(
            handoff="continue",
            reason="The lift is still moving.",
            text="DISCARDED",
        )
        original_narrator_compose = FakeDispatcher.narrator_compose
        narrator_attempt = 0

        async def _continue_then_fail(self, **kwargs):
            nonlocal narrator_attempt
            narrator_attempt += 1
            if narrator_attempt == 2:
                type(self).narrator_calls.append(kwargs)
                raise RuntimeError("narrator offline after continuation")
            return await original_narrator_compose(self, **kwargs)

        monkeypatch.setattr(
            FakeDispatcher,
            "narrator_compose",
            _continue_then_fail,
        )

        with pytest.raises(
            RuntimeError,
            match="narrator offline after continuation",
        ):
            await orch.process_turn(TurnRequest(
                session_id="s",
                user_input="I ride the lift to the lower hall.",
                acting_character_id="alice",
            ))

        pending = ckpt.session.pending_narrator_render
        assert pending is not None
        assert pending.events_closed == 2
        assert pending.event_actor_ids == ["alice", "alice"]
        assert pending.handoff_event_id == "evt_arrived"
        assert [
            entry.event_id for entry in ckpt.session.render_buffers["alice"]
        ] == ["evt_waiting", "evt_arrived"]
        pip = next(char for char in ckpt.characters if char.character_id == "pip")
        assert pip.status == CharacterStatus.active

        FakeDispatcher.queue_narrator(
            handoff="render",
            reason="The lift has arrived.",
            text="ARRIVED",
        )
        response = await orch.retry_pending_narrator_render("s")

        assert response.output_text == "ARRIVED"
        assert len(FakeDispatcher.route_calls) == 2
        assert pip.status == CharacterStatus.dormant
        assert ckpt.session.pending_narrator_render is None

    @pytest.mark.asyncio
    async def test_retry_image_lineage_stays_on_the_persisted_failed_turn(
        self, tmp_path, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.turn_index = 4
        event = _router_out(
            event_id="evt_failed_render",
            event_kind="cascade_exhausted",
            observer_ids=["alice"],
        )
        ckpt.canonical_events.append(event)
        ckpt.session.render_buffers["alice"] = [RenderBufferEntry(
            event_id=event.event_id,
            visible_at_s=0,
            event_sequence=0,
        )]
        ckpt.session.active_act_slots["alice"] = SlotEntry(reason="initiator")
        ckpt.session.pending_narrator_render = PendingNarratorRender(
            ended_reason="cascade_exhausted",
            events_closed=1,
            event_actor_ids=["alice"],
            acting_player_id="alice",
            acting_player_input="I look around.",
        )
        manager = CheckpointManager(str(tmp_path / "sessions"))
        manager.save(ckpt)

        class RecordingImageSink:
            config = SimpleNamespace(director_enabled=True)

            def __init__(self):
                self.source_turn_indexes = []
                self.committed = []

            async def start_render_candidate(self, **kwargs):
                self.source_turn_indexes.append(kwargs["source_turn_index"])
                return "imgtx_retry"

            async def cancel_transaction(self, transaction_id, **_kwargs):
                raise AssertionError(
                    f"accepted retry transaction was cancelled: {transaction_id}"
                )

            async def commit_transaction(self, transaction_id, **_kwargs):
                self.committed.append(transaction_id)

        sink = RecordingImageSink()
        generation = MagicMock()
        generation.reconcile_lineage = MagicMock()
        client = MagicMock()
        client.config.provider_for_role.return_value = "test"
        client.config.model_for_role.return_value = "test-model"
        monkeypatch.setattr(
            "app.engine.orchestrator.LLMDispatcher",
            FakeDispatcher,
        )
        orch = Orchestrator(
            client,
            manager,
            MagicMock(),
            image_sink=sink,
            image_generation=generation,
        )

        response = await orch.retry_pending_narrator_render("s")

        assert response.turn_index == 4
        assert sink.source_turn_indexes == [4]
        assert sink.committed == ["imgtx_retry"]
        reloaded = manager.load_latest("s")
        assert reloaded.session.turn_index == 4
        assert reloaded.session.pending_narrator_render is None

    @pytest.mark.asyncio
    async def test_retry_pending_narrator_render_noops_without_pending_state(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.turn_index = 2
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.retry_pending_narrator_render("s")

        assert response.turn_index == 2
        assert response.beat_ended_reason == "no_pending_render"
        assert "No failed narrator render" in response.output_text
        assert FakeDispatcher.route_calls == []
        assert FakeDispatcher.narrator_calls == []
        assert mgr.save.call_count == 0

    @pytest.mark.asyncio
    async def test_dnd_loot_offer_becomes_turn_prompt(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        orch, mgr = patched_orchestrator(ckpt)
        data = _dnd_router_out(interaction_mode="narrative").model_dump()
        data["loot_offer"] = {
            "present": True,
            "source_kind": "container",
            "source_label": "iron chest",
            "visibility": "table",
            "eligible_character_ids": ["alice"],
            "items": [
                {
                    "item_id": "healing_potion",
                    "name": "Potion of Healing",
                    "kind": "consumable",
                    "quantity": 1,
                    "identified": True,
                    "requires_identification": False,
                    "requires_attunement": False,
                    "consumable": True,
                    "value_gp": 50,
                    "weight": 0.5,
                    "notes": "",
                }
            ],
            "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 12, "pp": 0},
            "notes": "",
        }
        FakeDispatcher.queue_route(DndEventRouterOutput(**data))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I open the chest",
            acting_character_id="alice",
        ))

        offer_id = response.loot_prompts["alice"][0]
        assert offer_id.startswith("loot_evt_")
        saved = mgr.save.call_args[0][0]
        assert len(saved.session.dnd_inventory_offers) == 1
        offer = saved.session.dnd_inventory_offers[0]
        assert offer.offer_id == offer_id
        assert offer.source_label == "iron chest"
        assert offer.items[0].name == "Potion of Healing"


class TestSlotRejection:
    @pytest.mark.asyncio
    async def test_missing_acting_character_returns_friendly_response(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={})
        ckpt.session.player_character_id = ""
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I look around",
        ))

        assert response.beat_ended_reason == "acting_character_required"
        assert "Choose a character before acting" in response.output_text
        assert mgr.save.call_count == 0
        assert FakeDispatcher.route_calls == []

    @pytest.mark.asyncio
    async def test_second_act_against_held_slot_rejected_without_save(
        self, patched_orchestrator,
    ):
        from app.engine.turn_loop import claim_initiator_slot

        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        claim_initiator_slot(ckpt, "alice")
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I speak up",
            acting_character_id="bob",
        ))

        assert "didn't go through" in response.output_text
        assert response.per_player_renders == {}
        assert response.beat_ended_reason == "slot_rejected"
        assert mgr.save.call_count == 0
        assert FakeDispatcher.route_calls == []


class TestPendingCombatRolls:
    @pytest.mark.asyncio
    async def test_continue_roll_without_open_cat_ii_finalizes_combat(
        self,
        patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.turn_index = 4
        ckpt.session.active_combat = DndCombatState(
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="cat_ii_roll",
            cat_ii_event_id="cmb_1",
        )
        ckpt.session.cat_ii_roll_transactions.append(
            CatIIRollTransaction(
                transaction_id="rolltxn_1",
                event_id="cmb_1",
                source="combat",
                actor_id="alice",
                status="ready_to_finalize",
                plan={"needs_rolls": True, "roll_requests": []},
                ledger_lines=["attack_alice: alice attack_roll rolled 18"],
            )
        )
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(
            _router_out(event_kind="ruleset_resolution")
        )

        response = await orch.continue_cat_ii_after_roll(
            session_id="s",
            event_id="cmb_1",
            actor_id="alice",
        )

        assert response.beat_ended_reason == "ruleset_resolution"
        assert response.per_player_renders["alice"] == "POV_RENDER"
        assert response.turn_index == 5
        assert FakeDispatcher.route_calls[0]["event_id"] == "cmb_1"
        assert ckpt.session.active_act_slots == {}
        assert len(ckpt.canonical_events) == 1
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_continue_combat_response_includes_new_automatic_roll(
        self,
        patched_orchestrator,
        monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.turn_index = 4
        ckpt.session.active_combat = DndCombatState(
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="rat",
                    character_id="rat",
                    name="Rat",
                    armor_class=12,
                ),
            ],
        )
        record = CatIIRollRecord(
            roll_id="attack_rat",
            actor_id="alice",
            actor_control="agent",
            status="pending",
            request={
                "roll_id": "attack_rat",
                "actor_id": "alice",
                "kind": "attack_roll",
                "ability": "str",
                "skill": "",
                "dc": 0,
                "opposed_by": "",
                "advantage_state": "normal",
                "reason": "Alice attacks the rat.",
                "action_id": "shortsword",
                "target_id": "rat",
                "damage_adjustments": [],
            },
            modifier=4,
            label="Attack (Shortsword)",
            reason="Alice attacks the rat.",
        )
        ckpt.session.cat_ii_roll_transactions.append(
            CatIIRollTransaction(
                transaction_id="rolltxn_2",
                event_id="cmb_2",
                source="combat",
                actor_id="alice",
                status="ready_to_finalize",
                rolls=[record],
            )
        )

        async def _fake_continue(self, **kw):
            txn = kw["ckpt"].session.cat_ii_roll_transactions[0]
            roll = txn.rolls[0]
            roll.status = "completed"
            roll.completed_by_user_id = "engine"
            roll.result = {
                "roll_id": "attack_rat",
                "expression": "1d20+4",
                "total": 17,
                "detail": "1d20 (13) + 4 = `17`",
                "crit": "none",
                "dice": [
                    {"size": 20, "values": [13], "kept": True, "total": 13},
                ],
            }
            return _router_out(event_kind="ruleset_resolution")

        monkeypatch.setattr(
            FakeDispatcher,
            "continue_combat_transaction",
            _fake_continue,
        )
        orch, _mgr = patched_orchestrator(ckpt)

        response = await orch.continue_cat_ii_after_roll(
            session_id="s",
            event_id="cmb_2",
            actor_id="alice",
        )

        assert response.dice_rolls
        assert response.dice_rolls[0].roll_id == "attack_rat"
        assert response.dice_rolls[0].actor_name == "Alice"
        assert response.dice_rolls[0].target_name == "Rat"
        assert response.dice_rolls[0].dc == 12
        assert response.dice_rolls[0].outcome == "hit"

    @pytest.mark.asyncio
    async def test_cancelled_combat_roll_clears_slot_without_dispatch(
        self,
        patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="cat_ii_roll",
            cat_ii_event_id="cmb_cancelled",
        )
        ckpt.session.cat_ii_roll_transactions.append(
            CatIIRollTransaction(
                transaction_id="rolltxn_cancelled",
                event_id="cmb_cancelled",
                source="combat",
                actor_id="alice",
                status="cancelled",
            )
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.continue_cat_ii_after_roll(
            session_id="s",
            event_id="cmb_cancelled",
            actor_id="alice",
        )

        assert response.beat_ended_reason == "cat_ii_stale"
        assert response.output_text == (
            "That combat roll is no longer active. Use /combat status "
            "to see the current state."
        )
        assert ckpt.session.active_act_slots == {}
        assert FakeDispatcher.route_calls == []
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_submit_cancelled_combat_roll_returns_stale_without_dispatch(
        self,
        patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="cat_ii_roll",
            cat_ii_event_id="cmb_cancelled",
        )
        ckpt.session.cat_ii_roll_transactions.append(
            CatIIRollTransaction(
                transaction_id="rolltxn_cancelled",
                event_id="cmb_cancelled",
                source="combat",
                actor_id="alice",
                status="cancelled",
            )
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.submit_cat_ii_roll(
            session_id="s",
            event_id="cmb_cancelled",
            roll_id="attack_alice",
            actor_id="alice",
            user_id="u1",
        )

        assert response.beat_ended_reason == "cat_ii_stale"
        assert response.output_text == (
            "That combat roll is no longer active. Use /combat status "
            "to see the current state."
        )
        assert ckpt.session.active_act_slots == {}
        assert FakeDispatcher.route_calls == []
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_submit_non_pending_cat_ii_roll_returns_friendly_stale(
        self,
        patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.open_cat_ii_events.append(OpenCatIIEvent(
            event_id="evt_open",
            initiator_id="bob",
            initiator_intention="Bob trips Alice.",
            required_responders=["alice"],
            collected_intentions={"alice": "Alice keeps her feet."},
        ))
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.submit_cat_ii_roll(
            session_id="s",
            event_id="evt_open",
            roll_id="roll_alice",
            actor_id="alice",
            user_id="u1",
        )

        assert response.beat_ended_reason == "cat_ii_stale"
        assert response.output_text == (
            "That roll is no longer pending for your character. "
            "Use /combat status to see the current state."
        )
        assert "alice" not in response.output_text
        assert "roll_alice" not in response.output_text
        assert FakeDispatcher.route_calls == []
        assert mgr.save.call_count == 0


class TestCombatTurnGating:
    @pytest.mark.asyncio
    async def test_dnd_combat_start_signal_starts_initiative_without_advancing(
        self, patched_orchestrator, monkeypatch,
    ):
        values = iter([19, 0, 0])
        monkeypatch.setattr(
            dice.d20.expression.random,
            "randrange",
            lambda _: next(values),
        )
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice", "bob"],
            facts=[ObservableFact.all("Alice commits to an attack against Bob.")],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I attack Bob",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_started"
        assert ckpt.session.active_combat is not None
        assert current_combatant(ckpt.session).character_id == "alice"
        assert ckpt.session.open_cat_ii_events == []
        assert ckpt.session.active_act_slots == {}
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_dnd_combat_start_materializes_spawned_monster_with_override(
        self, patched_orchestrator, monkeypatch,
    ):
        values = iter([19, 0, 10, 10])
        monkeypatch.setattr(
            dice.d20.expression.random,
            "randrange",
            lambda _: next(values),
        )
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.characters[0].mechanics = _dnd_mechanics()
        spawn = _rat_combatant_spawn()

        def corrected(candidate):
            assert candidate.monster_key == "rat"
            return {
                **candidate.statblock.model_dump(),
                "armor_class": 13,
                "hit_points": 2,
                "challenge_rating": "1/8",
                "xp": 25,
            }

        try:
            dnd_monsters.clear_statblock_override_providers()
            dnd_monsters.register_statblock_override_provider(corrected)
            orch, _mgr = patched_orchestrator(ckpt)
            FakeDispatcher.queue_route(_dnd_router_out(
                interaction_mode="dnd_combat_start",
                combatant_ids=["alice"],
                combatant_spawns=[spawn],
                facts=[ObservableFact.all("Alice kicks at the rat under the table.")],
            ))

            response = await orch.process_turn(TurnRequest(
                session_id="s",
                user_input="I kick the rat",
                acting_character_id="alice",
            ))

            assert response.beat_ended_reason == "combat_started"
            rat = next(c for c in ckpt.characters if c.character_id == "rat_1")
            assert rat.name == "Rat"
            assert rat.location == "gatehouse"
            assert rat.mechanics["armor_class"] == 13
            assert rat.mechanics["hit_points"]["max"] == 2
            assert rat.mechanics["challenge_rating"] == "1/8"
            combat = ckpt.session.active_combat
            assert combat is not None
            rat_combatant = next(
                c for c in combat.combatants if c.character_id == "rat_1"
            )
            assert rat_combatant.armor_class == 13
            assert rat_combatant.hit_points_max == 2

            apply_damage(ckpt.session, "rat_1", 99, characters=ckpt.characters)

            assert dnd_experience.experience_points(ckpt.characters[0]) == 25
            assert combat.xp_awarded_combatant_ids == ["rat_1"]
        finally:
            dnd_monsters.clear_statblock_override_providers()

    @pytest.mark.asyncio
    async def test_dnd_combat_start_drops_duplicate_generic_spawn(
        self, patched_orchestrator, monkeypatch,
    ):
        values = iter([19, 0, 10, 10])
        monkeypatch.setattr(
            dice.d20.expression.random,
            "randrange",
            lambda _: next(values),
        )
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.characters[0].mechanics = _dnd_mechanics()
        combatant_spawn = _rat_combatant_spawn()
        generic_spawn = {
            "character_id": combatant_spawn["character_id"],
            "seed": {
                "role": "hostile combatant",
                "reason": (
                    "duplicate generic spawn emitted alongside combatant_spawns"
                ),
                "location": "gatehouse",
                "objectives": ["fight the party"],
                "knowledge_tier": 0,
            },
        }
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice"],
            combatant_spawns=[combatant_spawn],
            spawn=[generic_spawn],
            facts=[ObservableFact.all("Alice kicks at the rat under the table.")],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I kick the rat",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_started"
        assert [c.character_id for c in ckpt.characters].count("rat_1") == 1
        event = ckpt.canonical_events[-1]
        assert event.spawn == []
        assert event.combatant_spawns[0].character_id == "rat_1"

    @pytest.mark.asyncio
    async def test_dnd_combat_start_defaults_existing_story_npc_stats(
        self, patched_orchestrator, monkeypatch,
    ):
        values = iter([19, 0, 10])
        monkeypatch.setattr(
            dice.d20.expression.random,
            "randrange",
            lambda _: next(values),
        )
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.characters[0].mechanics = _dnd_mechanics()
        ckpt.characters.append(
            character_record(
                "npc_meris",
                name="Meris Venn",
                role="custodian",
                location="gatehouse",
            )
        )
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice", "npc_meris"],
            facts=[ObservableFact.all("Alice draws Meris into the fight.")],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I protect Meris as the fight starts",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_started"
        meris = next(c for c in ckpt.characters if c.character_id == "npc_meris")
        assert meris.mechanics["source"] == "dnd_default_combatant_profile"
        assert meris.mechanics["hit_points"]["max"] == 4
        combat = ckpt.session.active_combat
        assert combat is not None
        meris_combatant = next(
            c for c in combat.combatants if c.character_id == "npc_meris"
        )
        assert meris_combatant.hit_points_current == 4
        assert meris_combatant.hit_points_max == 4

    @pytest.mark.asyncio
    async def test_dnd_combat_start_reactivates_selected_dormant_combatants(
        self, patched_orchestrator, monkeypatch,
    ):
        values = iter([19, 0, 10])
        monkeypatch.setattr(
            dice.d20.expression.random,
            "randrange",
            lambda _: next(values),
        )
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.characters[0].mechanics = _dnd_mechanics()
        bandit = character_record(
            "bandit_01",
            name="Bandit",
            role="ambusher",
            location="gatehouse",
        )
        bandit.status = CharacterStatus.dormant
        bandit.mechanics = _dnd_mechanics(hp=11, ac=12)
        ckpt.characters.append(bandit)
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice", "bandit_01"],
            facts=[ObservableFact.all("The bandit steps back into the fight.")],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I keep moving while the bandit reengages.",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_started"
        assert bandit.status == CharacterStatus.active
        combat = ckpt.session.active_combat
        assert combat is not None
        assert {
            combatant.character_id for combatant in combat.combatants
        } == {"alice", "bandit_01"}

    @pytest.mark.asyncio
    async def test_dnd_combat_start_culls_selected_defeated_dormant_spawn(
        self, patched_orchestrator, monkeypatch,
    ):
        values = iter([19, 0, 10, 5, 7])
        monkeypatch.setattr(
            dice.d20.expression.random,
            "randrange",
            lambda _: next(values),
        )
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.characters[0].mechanics = _dnd_mechanics()
        defeated = character_record(
            "bandit_defeated",
            name="Defeated Bandit",
            role="ambusher",
            location="gatehouse",
        )
        defeated.status = CharacterStatus.dormant
        defeated.mechanics = _dnd_mechanics(hp=11, ac=12)
        defeated.mechanics["hit_points"]["current"] = 0
        defeated.mechanics["combat_spawn"] = {
            "spawned": True,
            "source_event_id": "evt_prior_fight",
            "monster_key": "bandit",
        }
        active = character_record(
            "bandit_active",
            name="Bandit",
            role="ambusher",
            location="gatehouse",
        )
        active.status = CharacterStatus.dormant
        active.mechanics = _dnd_mechanics(hp=11, ac=12)
        active.mechanics["combat_spawn"] = {
            "spawned": True,
            "source_event_id": "evt_prior_fight",
            "monster_key": "bandit",
        }
        ckpt.characters.extend([defeated, active])
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice", "bandit_defeated", "bandit_active"],
            facts=[ObservableFact.all("The surviving bandit reenters combat.")],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I move under renewed pressure from the bandits.",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_started"
        assert defeated.status == CharacterStatus.culled
        assert active.status == CharacterStatus.active
        combat = ckpt.session.active_combat
        assert combat is not None
        assert {
            combatant.character_id for combatant in combat.combatants
        } == {"alice", "bandit_active"}

    @pytest.mark.asyncio
    async def test_dnd_combat_start_materializes_ref_only_imported_statblock(
        self, patched_orchestrator, monkeypatch,
    ):
        values = iter([19, 0, 10, 10])
        monkeypatch.setattr(
            dice.d20.expression.random,
            "randrange",
            lambda _: next(values),
        )
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.session.content_state = {
            "synthetic-pack": ContentPackState(
                pack_id="synthetic-pack",
                metadata={"statblocks": [_guardian_statblock()]},
            )
        }
        ckpt.characters[0].mechanics = _dnd_mechanics()
        spawn = {
            "character_id": "guardian_1",
            "monster_key": "",
            "statblock_ref": "stat.guardian",
            "name": "",
            "location": "",
            "description": "",
            "statblock": None,
        }
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice"],
            combatant_spawns=[spawn],
            facts=[ObservableFact.all("Alice strikes the guardian.")],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I strike the guardian",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_started"
        guardian = next(c for c in ckpt.characters if c.character_id == "guardian_1")
        assert guardian.name == "Synthetic Guardian"
        assert guardian.location == "gatehouse"
        assert guardian.mechanics["source"] == "imported_statblock_catalog"
        assert guardian.mechanics["armor_class"] == 15
        assert guardian.mechanics["hit_points"]["max"] == 33
        assert guardian.mechanics["imported_statblock"]["ref"] == "stat.guardian"
        combat = ckpt.session.active_combat
        assert combat is not None
        guardian_combatant = next(
            c for c in combat.combatants if c.character_id == "guardian_1"
        )
        assert guardian_combatant.armor_class == 15
        assert guardian_combatant.hit_points_max == 33
        event = ckpt.canonical_events[-1]
        assert event.combatant_spawns[0].statblock_ref == "stat.guardian"
        assert event.combatant_spawns[0].statblock is None
        dumped = json.dumps(guardian.model_dump(mode="json"), sort_keys=True)
        assert "/private/source.pdf" not in dumped

    @pytest.mark.asyncio
    async def test_dnd_combat_start_resolves_imported_encounter_template(
        self, patched_orchestrator, monkeypatch,
    ):
        values = iter([19, 0, 10, 10])
        monkeypatch.setattr(
            dice.d20.expression.random,
            "randrange",
            lambda _: next(values),
        )
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.session.content_state = {
            "synthetic-pack": ContentPackState(
                pack_id="synthetic-pack",
                metadata={
                    "locations": [{"ref": "gatehouse"}],
                    "encounter_templates": [_encounter_template()],
                    "statblocks": [_guardian_statblock()],
                    "tactical_map_templates": [_encounter_map_template()],
                    "trap_hazards": [_encounter_trap()],
                    "treasures": [_encounter_treasure()],
                },
            )
        }
        ckpt.characters[0].mechanics = _dnd_mechanics()
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice"],
            combatant_spawns=[],
            facts=[ObservableFact.all("Alice challenges the guardian.")],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I challenge the guardian",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_started"
        guardian = next(c for c in ckpt.characters if c.character_id == "guardian")
        assert guardian.mechanics["imported_statblock"]["ref"] == "stat.guardian"
        combat = ckpt.session.active_combat
        assert combat is not None
        assert {c.character_id for c in combat.combatants} == {"alice", "guardian"}
        assert combat.battle_map is not None
        assert combat.battle_map.source_template_ref == "map.entry"
        assert any(
            "Imported encounter template applied: enc.entry." in line
            for line in combat.audit_lines
        )
        event = ckpt.canonical_events[-1]
        assert event.combatant_spawns[0].statblock_ref == "stat.guardian"
        assert event.battle_map_seed.present is True

    @pytest.mark.asyncio
    async def test_dnd_combat_start_missing_imported_statblock_ref_errors(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.session.content_state = {
            "synthetic-pack": ContentPackState(
                pack_id="synthetic-pack",
                metadata={"statblocks": [_guardian_statblock()]},
            )
        }
        ckpt.characters[0].mechanics = _dnd_mechanics()
        spawn = {
            "character_id": "guardian_1",
            "statblock_ref": "stat.missing",
            "statblock": None,
        }
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice"],
            combatant_spawns=[spawn],
            facts=[ObservableFact.all("Alice strikes the guardian.")],
        ))

        with pytest.raises(ImportedStatBlockNotFoundError, match="stat.missing"):
            await orch.process_turn(TurnRequest(
                session_id="s",
                user_input="I strike the guardian",
                acting_character_id="alice",
            ))

        assert ckpt.session.active_combat is None
        assert all(c.character_id != "guardian_1" for c in ckpt.characters)

    @pytest.mark.asyncio
    async def test_failed_dnd_combat_start_rolls_back_spawned_monsters(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        spawn = _rat_combatant_spawn()
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["missing_actor"],
            combatant_spawns=[spawn],
            facts=[ObservableFact.all("Something skitters under the table.")],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I attack",
            acting_character_id="missing_actor",
        ))

        assert response.beat_ended_reason == "state_change"
        assert ckpt.session.active_combat is None
        assert all(c.character_id != "rat_1" for c in ckpt.characters)
        event = ckpt.canonical_events[-1]
        assert event.combatant_spawns == []
        assert event.combatant_ids == ["missing_actor"]

    @pytest.mark.asyncio
    async def test_dnd_combat_spawn_id_collision_mints_new_monster_id(
        self, patched_orchestrator, monkeypatch,
    ):
        values = iter([19, 0, 10, 10])
        monkeypatch.setattr(
            dice.d20.expression.random,
            "randrange",
            lambda _: next(values),
        )
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.characters[0].mechanics = _dnd_mechanics()
        spawn = {**_rat_combatant_spawn(), "character_id": "pip"}
        orch, _mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_dnd_router_out(
            interaction_mode="dnd_combat_start",
            combatant_ids=["alice"],
            combatant_spawns=[spawn],
            facts=[ObservableFact.all("Alice kicks at the rat under the table.")],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I kick the rat",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_started"
        assert any(c.character_id == "rat" for c in ckpt.characters)
        existing_pip = next(c for c in ckpt.characters if c.character_id == "pip")
        assert existing_pip.name == "Pip"
        combat_ids = {
            combatant.character_id
            for combatant in ckpt.session.active_combat.combatants
        }
        assert "rat" in combat_ids
        assert "pip" not in combat_ids

    @pytest.mark.asyncio
    async def test_non_current_human_combatant_is_rejected_without_save(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = {
            "status": "active",
            "round_number": 1,
            "turn_index": 0,
            "combatants": [
                {
                    "character_id": "alice",
                    "name": "Alice",
                    "player_controlled": True,
                },
                {
                    "character_id": "bob",
                    "name": "Bob",
                    "player_controlled": True,
                },
            ],
        }
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I rush\x1b[2J in anyway",
            acting_character_id="bob",
        ))

        assert response.beat_ended_reason == "combat_turn_rejected"
        assert "Alice" in response.output_text
        assert "initiative turn" in response.output_text
        assert "can't /act" not in response.output_text
        assert "Wait for" in response.output_text
        assert "\x1b" not in response.output_text
        assert response.per_player_renders == {}
        assert mgr.save.call_count == 0
        assert FakeDispatcher.route_calls == []

    @pytest.mark.asyncio
    async def test_npc_current_after_rewind_resumes_before_player_act(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            actor_id = kw["actor_id"]
            return BeatResult(
                renders={"alice": f"{actor_id} acts."},
                events_closed=0,
                ended_reason="response_requested",
                transcript_entries={},
                event_actor_ids=[actor_id],
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        FakeDispatcher.queue_agent("Rat bites.")
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=3,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="rat",
                    character_id="rat",
                    name="Rat",
                    player_controlled=False,
                ),
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I strike back",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "pre_turn_resolution"
        assert "scene changed" in response.output_text
        assert response.per_player_renders == {}
        assert len(response.pre_turn_resolutions) == 1
        assert response.pre_turn_resolutions[0].output_text == "rat acts."
        assert response.pre_turn_resolutions[0].turn_index == 1
        assert FakeDispatcher.agent_calls[0]["character_id"] == "rat"
        assert FakeDispatcher.route_calls == []
        assert ckpt.session.active_combat.turn_index == 1
        assert response.turn_index == 1
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_failed_automated_npc_turn_rolls_back_partial_state_only(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fail_combat_action(self, *, ckpt, actor_id, intention):
            assert actor_id == "rat"
            partial = _router_out(
                event_kind="ruleset_resolution",
                facts=[ObservableFact.all("Rat snaps at Alice but resolution fails.")],
            )
            partial.observers = [
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    routing_role="observe_only",
                ),
                ObserverEntry(
                    character_id="rat",
                    observation_level="d",
                    routing_role="observe_only",
                ),
            ]
            broadcast_event(ckpt, partial, actor_id=actor_id)
            ckpt.session.active_act_slots["rat_extra"] = SlotEntry(
                reason="initiator",
            )
            raise RuntimeError("forced automated failure")

        monkeypatch.setattr(
            FakeDispatcher,
            "route_combat_action",
            fail_combat_action,
        )
        FakeDispatcher.queue_agent("Rat bites.")
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.session.turn_index = 7
        prior = _router_out(facts=[ObservableFact.all("Prior visible event.")])
        prior.event_id = "evt_prior"
        ckpt.canonical_events.append(prior)
        ckpt.session.render_buffers["alice"] = [
            RenderBufferEntry(event_id="evt_prior")
        ]
        ckpt.session.active_combat = DndCombatState(
            round_number=3,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="rat",
                    character_id="rat",
                    name="Rat",
                    player_controlled=False,
                ),
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        results = await orch._run_automated_combat_turns_locked(
            ckpt=ckpt,
            dispatcher=FakeDispatcher(),
        )

        assert results == []
        assert [event.event_id for event in ckpt.canonical_events] == [
            "evt_prior"
        ]
        assert [
            entry.event_id for entry in ckpt.session.render_buffers["alice"]
        ] == ["evt_prior"]
        assert ckpt.session.active_act_slots == {}
        assert ckpt.session.active_combat.turn_index == 1
        assert ckpt.session.turn_index == 8
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_bound_human_outside_combat_can_act_while_combat_exists(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(event_kind="cascade_exhausted"))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I search the library",
            acting_character_id="bob",
        ))

        assert response.beat_ended_reason == "cascade_exhausted"
        assert FakeDispatcher.route_calls[0]["actor_id"] == "bob"
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_npc_initiative_turn_runs_agent_then_advances_to_human(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            actor_id = kw["actor_id"]
            return BeatResult(
                renders={"alice": f"{actor_id} acts."},
                events_closed=0,
                ended_reason="response_requested",
                transcript_entries={},
                event_actor_ids=[actor_id],
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        FakeDispatcher.queue_agent("Pip attacks.")
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I strike the goblin",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "response_requested"
        assert ckpt.session.active_combat.turn_index == 2
        assert ckpt.session.active_combat.round_number == 1
        assert response.output_text == "alice acts. pip acts."
        assert FakeDispatcher.agent_calls[0]["character_id"] == "pip"
        assert any(
            "Initiative advanced to Pip" in line
            for line in ckpt.session.active_combat.audit_lines
        )
        assert any(
            "Initiative advanced to Bob" in line
            for line in ckpt.session.active_combat.audit_lines
        )
        assert mgr.save.call_count == 2

    @pytest.mark.asyncio
    async def test_npc_automation_agent_failure_skips_to_next_turn(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            return BeatResult(
                renders={"alice": "Alice acts."},
                events_closed=0,
                ended_reason="response_requested",
                transcript_entries={},
                event_actor_ids=[kw["actor_id"]],
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I strike",
            acting_character_id="alice",
        ))

        assert response.output_text == "Alice acts."
        assert ckpt.session.active_combat.turn_index == 2
        assert any(
            "failed before an intention was produced" in line
            for line in ckpt.session.active_combat.audit_lines
        )
        assert mgr.save.call_count == 2

    @pytest.mark.asyncio
    async def test_npc_automation_run_beat_failure_aborts_and_advances(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            if kw["actor_id"] == "pip":
                from app.engine.turn_loop import claim_initiator_slot

                claim_initiator_slot(kw["ckpt"], "pip")
                raise RuntimeError("simulated route outage")
            return BeatResult(
                renders={"alice": "Alice acts."},
                events_closed=0,
                ended_reason="response_requested",
                transcript_entries={},
                event_actor_ids=[kw["actor_id"]],
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        FakeDispatcher.queue_agent("Pip attacks.")
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I strike",
            acting_character_id="alice",
        ))

        assert response.output_text == "Alice acts."
        assert ckpt.session.active_combat.turn_index == 2
        assert ckpt.session.active_act_slots == {}
        assert any(
            "failed during resolution" in line
            for line in ckpt.session.active_combat.audit_lines
        )
        assert mgr.save.call_count == 2

    @pytest.mark.asyncio
    async def test_pending_cat_ii_render_does_not_advance_combat(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            return BeatResult(
                renders={"alice": "The exchange hangs unresolved."},
                events_closed=0,
                ended_reason="cat_ii_pending",
                transcript_entries={},
                event_actor_ids=["alice"],
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = {
            "status": "active",
            "round_number": 1,
            "turn_index": 0,
            "combatants": [
                {
                    "character_id": "alice",
                    "name": "Alice",
                    "player_controlled": True,
                },
                {
                    "character_id": "bob",
                    "name": "Bob",
                    "player_controlled": True,
                },
            ],
        }
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I shove Bob",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "cat_ii_pending"
        assert ckpt.session.active_combat["turn_index"] == 0
        assert "audit_lines" not in ckpt.session.active_combat
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_reaction_prompt_delays_initiative_advance(
        self, patched_orchestrator, monkeypatch,
    ):
        async def fake_run_beat(**kw):
            return BeatResult(
                renders={"alice": "Alice moves.", "bob": "Bob can react."},
                events_closed=1,
                ended_reason="combat_reaction_pending",
                transcript_entries={},
                event_actor_ids=["alice"],
                reaction_prompts={"bob": "evt_react"},
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I move away",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_reaction_pending"
        assert response.reaction_prompts == {"bob": "evt_react"}
        assert ckpt.session.active_combat.turn_index == 0
        assert ckpt.session.active_combat.pending_advance_actor_id == "alice"
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_ruleset_resolution_advances_initiative(
        self,
        patched_orchestrator,
        monkeypatch,
    ):
        async def fake_run_beat(**kw):
            return BeatResult(
                renders={"alice": "Alice resolves a combat action."},
                events_closed=1,
                ended_reason="ruleset_resolution",
                transcript_entries={},
                event_actor_ids=["alice"],
                reaction_prompts={},
            )

        monkeypatch.setattr("app.engine.orchestrator.run_beat", fake_run_beat)
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
            ],
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I attack",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "ruleset_resolution"
        assert ckpt.session.active_combat.turn_index == 1
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_non_current_reaction_slot_bypasses_combat_turn_rejection(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                    reaction_available=True,
                ),
            ],
            pending_advance_actor_id="alice",
        )
        ckpt.session.active_act_slots["bob"] = SlotEntry(
            reason="combat_reaction",
            trigger_event_id="evt_react",
        )
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(EventRouterOutput(
            event_id="",
            effective_at_s=0,
            duration_s=0,
            decision_rationale="reaction act",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[ObservableFact.all("Bob reacts.")],
            ),
            event_kind="cascade_exhausted",
            observers=[
                ObserverEntry(
                    character_id="bob",
                    observation_level="d",
                    routing_role="observe_only",
                )
            ],
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I make an opportunity attack",
            acting_character_id="bob",
        ))

        assert response.beat_ended_reason == "cascade_exhausted"
        assert FakeDispatcher.route_calls[0]["actor_id"] == "bob"
        assert "Combat reaction" in FakeDispatcher.route_calls[0]["intention"]
        assert ckpt.session.active_act_slots == {}
        assert ckpt.session.active_combat.turn_index == 1
        assert ckpt.session.active_combat.pending_advance_actor_id == ""
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_defer_clears_blocked_combat_start_without_llm(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
            ],
        )
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="combat_blocked",
            trigger_event_id="evt_blocked",
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="(defer)",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "combat_start_blocked_deferred"
        assert "dropped" in response.output_text
        assert ckpt.session.active_act_slots == {}
        assert FakeDispatcher.route_calls == []
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_normal_act_abandons_blocked_combat_start(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                    player_controlled=False,
                ),
            ],
        )
        ckpt.session.active_act_slots["alice"] = SlotEntry(
            reason="combat_blocked",
            trigger_event_id="evt_blocked",
        )
        FakeDispatcher.queue_route(_router_out(
            event_kind="cascade_exhausted",
            facts=[ObservableFact.only("Alice steps back.", ["alice"])],
        ))
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I step back and watch.",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "cascade_exhausted"
        assert ckpt.session.active_act_slots == {}
        assert FakeDispatcher.route_calls[0]["actor_id"] == "alice"
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_defer_combat_reaction_clears_slot_without_llm_and_advances(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                    reaction_available=False,
                ),
            ],
            pending_advance_actor_id="alice",
        )
        ckpt.session.active_act_slots["bob"] = SlotEntry(
            reason="combat_reaction",
            trigger_event_id="evt_react",
        )
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.defer_combat_reaction(
            session_id="s",
            character_id="bob",
            event_id="evt_react",
        )

        assert response.beat_ended_reason == "combat_reaction_deferred"
        assert "Initiative advances to **Bob**" in response.output_text
        assert ckpt.session.active_act_slots == {}
        assert ckpt.session.active_combat.turn_index == 1
        assert ckpt.session.active_combat.combatants[1].reaction_available is True
        assert FakeDispatcher.route_calls == []
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_cat_ii_resolution_resumes_pending_combat_advance(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.active_combat = DndCombatState(
            round_number=1,
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="bob",
                    character_id="bob",
                    name="Bob",
                    player_controlled=True,
                    reaction_available=False,
                ),
            ],
            pending_advance_actor_id="alice",
        )
        from app.engine.turn_loop import open_cat_ii, pin_cat_ii_responder

        opened = open_cat_ii(
            ckpt,
            initiator_id="alice",
            initiator_intention="Alice's interrupted action",
            required_responders=["bob"],
        )
        pin_cat_ii_responder(ckpt, "bob", opened.event_id)
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(EventRouterOutput(
            event_id="",
            effective_at_s=0,
            duration_s=0,
            decision_rationale="cat ii closes after reaction",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[ObservableFact.all("The exchange resolves.")],
            ),
            event_kind="cat_ii_resolution",
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    routing_role="observe_only",
                ),
                ObserverEntry(
                    character_id="bob",
                    observation_level="d",
                    routing_role="observe_only",
                ),
            ],
            requires_responders=False,
            required_responders=[],
            spawn=[],
            dormant=[],
            cull=[],
            commitment_open=empty_commitment_open_signal(),
            commitment_resolutions=[],
            commitment_interrupts=[],
            location_updates=[],
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I block",
            acting_character_id="bob",
        ))

        assert response.beat_ended_reason == "cat_ii_resolution"
        assert ckpt.session.active_act_slots == {}
        assert ckpt.session.active_combat.turn_index == 1
        assert ckpt.session.active_combat.pending_advance_actor_id == ""
        assert ckpt.session.active_combat.combatants[1].reaction_available is True
        assert mgr.save.call_count == 1


class TestCatIIPending:
    @pytest.mark.asyncio
    async def test_cat_ii_response_does_not_clear_commitment_revision(
        self, patched_orchestrator,
    ):
        from app.engine.turn_loop import open_cat_ii, pin_cat_ii_responder

        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        ckpt.session.open_commitments = [
            OpenCommitment(
                commitment_id="commit_bob_watch",
                actor_ids=["bob"],
                description="Bob keeps watch.",
            )
        ]
        ckpt.session.pending_commitment_revisions["bob"] = (
            CommitmentRevisionPrompt(
                character_id="bob",
                commitment_id="commit_bob_watch",
                trigger_event_id="evt_scene_changed",
                observed_at_s=10,
                reason="the scene changed",
                previous_description="Bob keeps watch.",
            )
        )
        opened = open_cat_ii(
            ckpt,
            initiator_id="alice",
            initiator_intention="Alice presses Bob",
            required_responders=["bob"],
        )
        pin_cat_ii_responder(ckpt, "bob", opened.event_id)
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(
            event_kind="cat_ii_resolution",
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I answer Alice",
            acting_character_id="bob",
        ))

        assert response.beat_ended_reason == "cat_ii_resolution"
        assert ckpt.session.pending_commitment_revisions["bob"].commitment_id == (
            "commit_bob_watch"
        )
        assert response.commitment_revision_prompts == {
            "bob": ["commit_bob_watch"]
        }
        assert mgr.save.call_count == 1

    @pytest.mark.asyncio
    async def test_cat_ii_against_human_pauses_and_persists_open_event(
        self, patched_orchestrator,
    ):
        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        orch, mgr = patched_orchestrator(ckpt)
        FakeDispatcher.queue_route(_router_out(
            requires_responders=True,
            required_responders=["bob"],
            event_kind="cat_ii_open",
        ))

        response = await orch.process_turn(TurnRequest(
            session_id="s",
            user_input="I attack Bob",
            acting_character_id="alice",
        ))

        assert response.beat_ended_reason == "cat_ii_pending"
        assert "alice" in response.per_player_renders
        assert "bob" in response.per_player_renders
        saved = mgr.save.call_args[0][0]
        assert len(saved.session.open_cat_ii_events) == 1
        assert saved.session.active_act_slots["bob"].reason == "cat_ii_responder"


class TestResolveCatII:
    @pytest.mark.asyncio
    async def test_ready_event_without_semantic_next_output_ends_after_resolution(
        self, patched_orchestrator,
    ):
        from app.engine.turn_loop import open_cat_ii

        ckpt = _ckpt(bindings={"alice": "u1"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="pip",
            initiator_intention="pip swings at alice",
            required_responders=["alice"],
        )
        evt.collected_intentions["alice"] = "[AFK-swept: no player intention]"
        evt.swept_responders.append("alice")
        orch, mgr = patched_orchestrator(ckpt)

        resolution = _router_out(
            event_kind="cascade_exhausted",
            facts=[
                ObservableFact.all("Alice keeps her guard up."),
                ObservableFact.all("Pip ends the exchange checked."),
            ],
        )
        resolution.observers = [
            ObserverEntry(
                character_id="alice",
                observation_level="d",
                routing_role="observe_only",
            ),
            ObserverEntry(
                character_id="pip",
                observation_level="d",
                routing_role="observe_only",
            ),
        ]
        FakeDispatcher.queue_route(resolution)

        response = await orch.resolve_cat_ii("s", evt.event_id)

        assert response.beat_ended_reason == "cat_ii_resolution"
        assert response.per_player_renders["alice"] == "POV_RENDER"
        saved = mgr.save.call_args[0][0]
        assert all(e.event_id != evt.event_id for e in saved.session.open_cat_ii_events)
        assert len(saved.canonical_events) == 1
        assert FakeDispatcher.agent_calls == []
        assert len(FakeDispatcher.route_calls) == 1

    @pytest.mark.asyncio
    async def test_ready_event_routes_resolution_next_output_before_initiator(
        self, patched_orchestrator,
    ):
        from app.engine.turn_loop import open_cat_ii

        ckpt = _ckpt(bindings={"alice": "u1"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="alice",
            initiator_intention="alice warns pip away",
            required_responders=["pip"],
        )
        evt.collected_intentions["pip"] = "Pip tries not to flinch."
        orch, mgr = patched_orchestrator(ckpt)

        resolution = _router_out(
            event_kind="beat_continues",
            agent_ids=["pip"],
            facts=[
                ObservableFact.all("Alice's warning hangs in the doorway."),
                ObservableFact.only(
                    "The warning feels immediate to Pip.",
                    ["pip"],
                ),
            ],
        )
        FakeDispatcher.queue_route(resolution)
        FakeDispatcher.queue_agent("Pip answers the warning.")
        FakeDispatcher.queue_route(_router_out(event_kind="cascade_exhausted", agent_ids=[]))

        response = await orch.resolve_cat_ii("s", evt.event_id)

        assert response.beat_ended_reason == "cascade_exhausted"
        saved = mgr.save.call_args[0][0]
        assert len(saved.canonical_events) == 2
        assert FakeDispatcher.agent_calls[0]["character_id"] == "pip"
        assert FakeDispatcher.route_calls[1]["actor_id"] == "pip"

    @pytest.mark.asyncio
    async def test_ready_event_yields_to_bound_semantic_next_output(
        self, patched_orchestrator,
    ):
        from app.engine.turn_loop import open_cat_ii

        ckpt = _ckpt(bindings={"alice": "u1", "bob": "u2"})
        evt = open_cat_ii(
            ckpt,
            initiator_id="pip",
            initiator_intention="pip pressures alice",
            required_responders=["alice"],
        )
        evt.collected_intentions["alice"] = "Alice holds her ground."
        orch, mgr = patched_orchestrator(ckpt)

        FakeDispatcher.queue_route(_router_out(
            event_kind="beat_continues",
            agent_ids=["bob"],
            observer_ids=["alice", "bob", "pip"],
            facts=[ObservableFact.all("Bob now has the next choice.")],
        ))

        response = await orch.resolve_cat_ii("s", evt.event_id)

        assert response.beat_ended_reason == "awaiting_player_turn"
        saved = mgr.save.call_args[0][0]
        assert len(saved.canonical_events) == 1
        assert saved.canonical_events[0].next_output_character_ids == ["bob"]
        assert FakeDispatcher.agent_calls == []
        assert len(FakeDispatcher.route_calls) == 1

    @pytest.mark.asyncio
    async def test_ready_cat_ii_flushes_pending_combat_visible_facts(
        self, patched_orchestrator,
    ):
        from app.engine.turn_loop import open_cat_ii

        ckpt = _ckpt(bindings={"alice": "u1"})
        ckpt.session.active_combat = DndCombatState(
            combat_id="combat",
            turn_index=0,
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="pip",
                    character_id="pip",
                    name="Pip",
                ),
            ],
            pending_visible_facts=["Pip's burning effect ends."],
        )
        evt = open_cat_ii(
            ckpt,
            initiator_id="alice",
            initiator_intention="alice shoves pip",
            required_responders=["pip"],
        )
        evt.collected_intentions["pip"] = "I step back"
        orch, mgr = patched_orchestrator(ckpt)
        resolution = _router_out(
            event_kind="cat_ii_resolution",
            facts=[ObservableFact.all("Pip steps back from Alice.")],
        )
        resolution.observers = [
            ObserverEntry(
                character_id="alice",
                observation_level="d",
                routing_role="observe_only",
            ),
            ObserverEntry(
                character_id="pip",
                observation_level="d",
                routing_role="observe_only",
            ),
        ]
        FakeDispatcher.queue_route(resolution)

        response = await orch.resolve_cat_ii("s", evt.event_id)

        assert response.beat_ended_reason == "cat_ii_resolution"
        saved = mgr.save.call_args[0][0]
        assert saved.session.active_combat.pending_visible_facts == []
        assert any(
            fact.text == "Pip's burning effect ends."
            for event in saved.canonical_events
            for fact in event.canonical_event.observable_facts
        )

    @pytest.mark.asyncio
    async def test_stale_event_returns_noop(self, patched_orchestrator):
        ckpt = _ckpt(bindings={"alice": "u1"})
        orch, mgr = patched_orchestrator(ckpt)

        response = await orch.resolve_cat_ii("s", "missing_evt")

        assert response.beat_ended_reason == "cat_ii_stale"
        assert response.per_player_renders == {}
        assert mgr.save.call_count == 0
