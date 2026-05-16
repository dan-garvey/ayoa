"""Background tick scheduler coverage.

The scheduler is now session-wide: it fires only on stagnation, filters
characters by player/pin/eligibility state, and routes successful off-stage
agent prose through the unified router.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.engine.orchestrator import Orchestrator, TICK_CONCURRENCY_HARD_CAP
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterRecord,
    CharacterStatus,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication
from app.schemas.state import (
    DndCombatantState,
    DndCombatState,
    OpenCatIIEvent,
    SessionState,
    SlotEntry,
    WorldState,
)


def _npc(
    char_id: str,
    *,
    intentions_enabled: bool = True,
    status: CharacterStatus = CharacterStatus.active,
    location: str = "hall",
    is_playable: bool = False,
    agent_tier: CharacterAgentTier = CharacterAgentTier.standard,
    tick_cues: list[str] | None = None,
    current_objectives: list[str] | None = None,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=char_id,
        name=char_id.title(),
        status=status,
        location=location,
        is_playable=is_playable,
        agent_tier=agent_tier,
        public_sheet=PublicSheet(role="npc"),
        private_state=PrivateState(
            intentions_enabled=intentions_enabled,
            tick_cues=list(tick_cues or []),
            current_objectives=list(current_objectives or []),
        ),
    )


def _ckpt(
    *,
    bindings: dict[str, str] | None = None,
    player_character_id: str = "alice",
    characters: list[CharacterRecord] | None = None,
    turns_since_last_tick: int = 0,
    stagnation: int = 15,
    tick_concurrency: int = 4,
    tick_batch_size: int = 4,
    ticks_enabled: bool = True,
) -> CheckpointFile:
    sess = SessionState(
        session_id="s",
        player_character_id=player_character_id,
        character_bindings=bindings or {player_character_id: "u1"},
        turns_since_last_tick=turns_since_last_tick,
    )
    sess.config.settings.tick_stagnation_max = stagnation
    sess.config.settings.tick_concurrency = tick_concurrency
    sess.config.settings.tick_batch_size = tick_batch_size
    sess.config.settings.ticks_enabled = ticks_enabled
    return CheckpointFile(
        session=sess,
        world_state=WorldState(),
        characters=characters or [
            CharacterRecord(
                character_id="alice",
                name="Alice",
                is_playable=True,
                location="pod_a",
                public_sheet=PublicSheet(role="player"),
            ),
            _npc("regent"),
            _npc("scribe"),
        ],
    )


def _orchestrator() -> Orchestrator:
    return Orchestrator(
        client=MagicMock(),
        checkpoint_mgr=MagicMock(),
        prompt_mgr=MagicMock(),
    )


def _draft_for(
    character: CharacterRecord,
    *,
    public_text: str | None = None,
    intent: str = "watch the gate",
):
    from app.engine.character_agent import CharacterAgentTurnDraft

    output = CharacterAgentOutput(
        character_id=character.character_id,
        public_text=public_text or f"{character.name} paces.",
        intent=intent,
    )
    return CharacterAgentTurnDraft(
        output=output,
        user_message=ConversationMessage(
            role="user",
            content=f"tick user {character.character_id}",
        ),
        assistant_message=ConversationMessage(
            role="assistant",
            content=f"{output.public_text} ({intent})",
        ),
    )


def _stub_character_agent(monkeypatch, recorder: list[str] | None = None):
    class _StubAgent:
        def __init__(self, client, prompt_mgr):
            self.last_usage = {"input_tokens": 1, "output_tokens": 2}

        async def draft_tick(self, *, character, checkpoint, acting_character_id):
            if recorder is not None:
                recorder.append(character.character_id)
            return _draft_for(character)

    monkeypatch.setattr("app.engine.orchestrator.CharacterAgent", _StubAgent)
    return _StubAgent


def _patch_semaphore_recorder(monkeypatch) -> list[int]:
    real_semaphore = asyncio.Semaphore
    sizes: list[int] = []

    class _ProbeSemaphore:
        def __init__(self, value):
            sizes.append(value)
            self._inner = real_semaphore(value)

        async def __aenter__(self):
            await self._inner.acquire()
            return self

        async def __aexit__(self, *exc):
            self._inner.release()

    monkeypatch.setattr(
        "app.engine.orchestrator.asyncio.Semaphore", _ProbeSemaphore,
    )
    return sizes


def _tick_router_output(
    observer_ids: tuple[str, ...] = ("regent", "scribe"),
) -> EventRouterOutput:
    return EventRouterOutput(
        event_id="",
        decision_rationale="tick fan-in fixture",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(feasible=True),
            observable_facts=[],
        ),
        observers=[
            ObserverEntry(
                character_id=character_id,
                observation_level="d",
                response_priority=1,
            )
            for character_id in observer_ids
        ],
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="off_stage_tick",
        spawn=[],
        dormant=[],
        cull=[],
    )


def _event_with_fact(text: str) -> EventRouterOutput:
    routed = _tick_router_output(())
    routed.canonical_event.observable_facts = [ObservableFact.all(text)]
    return routed


def _stub_dispatcher(
    monkeypatch,
    routed: EventRouterOutput | None,
    recorder: list[dict] | None = None,
    *,
    raise_on_call: Exception | None = None,
):
    class _StubDispatcher:
        def __init__(self, client, prompt_mgr):
            pass

        async def route_tick_intentions(
            self, *, ckpt, tick_outputs, acting_character_id="",
        ):
            if recorder is not None:
                recorder.append({
                    "tick_outputs": list(tick_outputs),
                    "acting_character_id": acting_character_id,
                })
            if raise_on_call is not None:
                raise raise_on_call
            return routed

    monkeypatch.setattr("app.engine.orchestrator.LLMDispatcher", _StubDispatcher)
    return _StubDispatcher


class TestEligibility:
    def test_filters_non_tickable_characters(self):
        ckpt = _ckpt(characters=[
            _npc("alice", is_playable=True, location="pod_a"),
            _npc("dormant", status=CharacterStatus.dormant),
            _npc("culled", status=CharacterStatus.culled),
            _npc("quiet", intentions_enabled=False),
            _npc("acted"),
            _npc("eligible"),
        ])
        orch = _orchestrator()
        eligible = orch._eligible_for_tick(ckpt, acted_this_turn={"acted"})
        assert [c.character_id for c in eligible] == ["eligible"]

    def test_bound_npc_excluded(self):
        ckpt = _ckpt(
            bindings={"alice": "u1", "regent": "u2"},
            characters=[
                _npc("alice", is_playable=True, location="pod_a"),
                _npc("regent", location="pod_b"),
                _npc("scribe"),
            ],
        )
        eligible = _orchestrator()._eligible_for_tick(ckpt, acted_this_turn=set())
        assert [c.character_id for c in eligible] == ["scribe"]

    def test_pinned_slots_and_cat_ii_responders_excluded(self):
        ckpt = _ckpt(characters=[
            _npc("alice", is_playable=True, location="pod_a"),
            _npc("slotted"),
            _npc("required"),
            _npc("eligible"),
        ])
        ckpt.session.active_act_slots["slotted"] = SlotEntry(reason="initiator")
        ckpt.session.open_cat_ii_events.append(OpenCatIIEvent(
            event_id="evt1",
            initiator_id="alice",
            initiator_intention="(test)",
            required_responders=["required"],
        ))
        eligible = _orchestrator()._eligible_for_tick(ckpt, acted_this_turn=set())
        assert [c.character_id for c in eligible] == ["eligible"]

    def test_active_combatants_excluded_from_offstage_ticks(self):
        ckpt = _ckpt(characters=[
            _npc("alice", is_playable=True, location="pod_a"),
            _npc("regent"),
            _npc("scribe"),
        ])
        ckpt.session.active_combat = DndCombatState(
            combatants=[
                DndCombatantState(
                    combatant_id="alice",
                    character_id="alice",
                    name="Alice",
                    player_controlled=True,
                ),
                DndCombatantState(
                    combatant_id="regent",
                    character_id="regent",
                    name="Regent",
                    player_controlled=False,
                ),
            ]
        )

        eligible = _orchestrator()._eligible_for_tick(ckpt, acted_this_turn=set())

        assert [c.character_id for c in eligible] == ["scribe"]

    def test_current_player_location_excluded_from_offstage_ticks(self):
        ckpt = _ckpt(characters=[
            _npc("alice", is_playable=True, location="hall"),
            _npc("present", location="hall"),
            _npc("remote", location="garden"),
        ])

        eligible = _orchestrator()._eligible_for_tick(ckpt, acted_this_turn=set())

        assert [c.character_id for c in eligible] == ["remote"]

    def test_active_unseen_antagonist_is_eligible(self):
        ckpt = _ckpt(characters=[
            _npc("alice", is_playable=True, location="hall"),
            _npc("lord_verantus", location="celestial_keep"),
        ])

        eligible = _orchestrator()._eligible_for_tick(ckpt, acted_this_turn=set())

        assert [c.character_id for c in eligible] == ["lord_verantus"]


class TestTriggerLogic:
    @pytest.mark.asyncio
    async def test_disabled_freezes_trigger_state(self, monkeypatch):
        ckpt = _ckpt(turns_since_last_tick=14, ticks_enabled=False)
        _stub_character_agent(monkeypatch)

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert result == []
        assert ckpt.session.turns_since_last_tick == 14

    @pytest.mark.asyncio
    async def test_below_stagnation_increments_without_firing(self, monkeypatch):
        ckpt = _ckpt(turns_since_last_tick=2, stagnation=5)
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert result == []
        assert recorder == []
        assert ckpt.session.turns_since_last_tick == 3

    @pytest.mark.asyncio
    async def test_stagnation_fires_and_resets_counter(self, monkeypatch):
        ckpt = _ckpt(turns_since_last_tick=4, stagnation=5)
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)
        _stub_dispatcher(monkeypatch, _tick_router_output())

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert sorted(c.character_id for c, _ in result) == ["regent", "scribe"]
        assert sorted(recorder) == ["regent", "scribe"]
        assert ckpt.session.turns_since_last_tick == 0

    @pytest.mark.asyncio
    async def test_fire_with_no_eligible_resets_counter(self, monkeypatch):
        ckpt = _ckpt(
            turns_since_last_tick=14,
            stagnation=15,
            characters=[
                _npc("alice", is_playable=True),
                _npc("regent", status=CharacterStatus.dormant),
            ],
        )
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert result == []
        assert recorder == []
        assert ckpt.session.turns_since_last_tick == 0

    @pytest.mark.asyncio
    async def test_cue_match_fires_before_stagnation(self, monkeypatch):
        ckpt = _ckpt(
            turns_since_last_tick=0,
            stagnation=15,
            characters=[
                _npc("alice", is_playable=True, location="pod_a"),
                _npc("regent", tick_cues=["failed candle"]),
                _npc("scribe", tick_cues=["missing ledger"]),
            ],
        )
        ckpt.canonical_events.append(
            _event_with_fact("The failed candle gutters out on the altar.")
        )
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)
        _stub_dispatcher(monkeypatch, _tick_router_output(("regent",)))

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert [c.character_id for c, _ in result] == ["regent"]
        assert recorder == ["regent"]
        assert ckpt.session.turns_since_last_tick == 0

    @pytest.mark.asyncio
    async def test_no_cue_match_waits_for_stagnation(self, monkeypatch):
        ckpt = _ckpt(turns_since_last_tick=0, stagnation=15)
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert result == []
        assert recorder == []
        assert ckpt.session.turns_since_last_tick == 1

    @pytest.mark.asyncio
    async def test_stagnation_uses_bounded_tick_batch(self, monkeypatch):
        ckpt = _ckpt(
            turns_since_last_tick=14,
            stagnation=15,
            tick_batch_size=3,
            characters=[
                _npc("alice", is_playable=True, location="pod_a"),
                *[_npc(f"npc_{idx}") for idx in range(6)],
            ],
        )
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)
        _stub_dispatcher(
            monkeypatch,
            _tick_router_output(("npc_0", "npc_1", "npc_2")),
        )

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert [c.character_id for c, _ in result] == [
            "npc_0",
            "npc_1",
            "npc_2",
        ]
        assert recorder == ["npc_0", "npc_1", "npc_2"]

    @pytest.mark.asyncio
    async def test_premium_tiers_capped_when_lower_tiers_available(
        self, monkeypatch,
    ):
        ckpt = _ckpt(
            turns_since_last_tick=14,
            stagnation=15,
            tick_batch_size=4,
            characters=[
                _npc("alice", is_playable=True, location="pod_a"),
                _npc("premium_0", agent_tier=CharacterAgentTier.premium),
                _npc("premium_1", agent_tier=CharacterAgentTier.premium),
                _npc("premium_2", agent_tier=CharacterAgentTier.premium),
                _npc("standard_0", agent_tier=CharacterAgentTier.standard),
                _npc("standard_1", agent_tier=CharacterAgentTier.standard),
                _npc("utility_0", agent_tier=CharacterAgentTier.utility),
            ],
        )
        recorder: list[str] = []
        _stub_character_agent(monkeypatch, recorder)
        _stub_dispatcher(
            monkeypatch,
            _tick_router_output(
                ("premium_0", "standard_0", "standard_1", "utility_0")
            ),
        )

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        selected_ids = [c.character_id for c, _ in result]
        assert selected_ids == [
            "premium_0",
            "standard_0",
            "standard_1",
            "utility_0",
        ]
        assert recorder == selected_ids


class TestFanOut:
    @pytest.mark.asyncio
    async def test_concurrency_cap_respects_settings(self, monkeypatch):
        sizes = _patch_semaphore_recorder(monkeypatch)
        ckpt = _ckpt(
            turns_since_last_tick=14,
            stagnation=15,
            tick_concurrency=2,
        )
        _stub_character_agent(monkeypatch, recorder=[])
        _stub_dispatcher(monkeypatch, _tick_router_output())

        await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert sizes == [2]

    @pytest.mark.asyncio
    async def test_concurrency_clamped_by_hard_cap(self, monkeypatch):
        sizes = _patch_semaphore_recorder(monkeypatch)
        ckpt = _ckpt(
            turns_since_last_tick=14,
            stagnation=15,
            tick_concurrency=999,
        )
        _stub_character_agent(monkeypatch, recorder=[])
        _stub_dispatcher(monkeypatch, _tick_router_output())

        await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert sizes == [TICK_CONCURRENCY_HARD_CAP]

    @pytest.mark.asyncio
    async def test_first_tick_per_model_role_warms_before_parallelizing(
        self, monkeypatch,
    ):
        order: list[str] = []

        class _TracingAgent:
            def __init__(self, client, prompt_mgr):
                self.last_usage = {}

            async def draft_tick(self, *, character, checkpoint, acting_character_id):
                order.append(f"start:{character.character_id}")
                await asyncio.sleep(0)
                order.append(f"finish:{character.character_id}")
                return _draft_for(character)

        monkeypatch.setattr("app.engine.orchestrator.CharacterAgent", _TracingAgent)
        _stub_dispatcher(
            monkeypatch,
            _tick_router_output(
                ("premium_0", "standard_0", "standard_1", "utility_0")
            ),
        )
        ckpt = _ckpt(
            turns_since_last_tick=14,
            stagnation=15,
            tick_batch_size=4,
            characters=[
                _npc("alice", is_playable=True, location="pod_a"),
                _npc("premium_0", agent_tier=CharacterAgentTier.premium),
                _npc("standard_0", agent_tier=CharacterAgentTier.standard),
                _npc("standard_1", agent_tier=CharacterAgentTier.standard),
                _npc("utility_0", agent_tier=CharacterAgentTier.utility),
            ],
        )

        await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert order.index("finish:premium_0") < order.index("start:standard_0")
        assert order.index("finish:standard_0") < order.index("start:standard_1")
        assert order.index("finish:standard_0") < order.index("start:utility_0")

    @pytest.mark.asyncio
    async def test_per_tick_failure_does_not_drop_others(self, monkeypatch):
        class _PartialFailureAgent:
            def __init__(self, client, prompt_mgr):
                self.last_usage = {}

            async def draft_tick(self, *, character, checkpoint, acting_character_id):
                if character.character_id == "regent":
                    raise RuntimeError("simulated API hiccup")
                return _draft_for(
                    character,
                    public_text=f"{character.name} watches.",
                    intent="stay quiet",
                )

        monkeypatch.setattr(
            "app.engine.orchestrator.CharacterAgent", _PartialFailureAgent,
        )
        _stub_dispatcher(monkeypatch, _tick_router_output())
        ckpt = _ckpt(turns_since_last_tick=14, stagnation=15)

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert [c.character_id for c, _ in result] == ["scribe"]
        assert ckpt.session.turns_since_last_tick == 0


class TestTickFanIn:
    @pytest.mark.asyncio
    async def test_fan_in_called_with_public_text_and_locations(
        self, monkeypatch,
    ):
        ckpt = _ckpt(turns_since_last_tick=14, stagnation=15)
        _stub_character_agent(monkeypatch, recorder=[])
        recorded: list[dict] = []
        _stub_dispatcher(monkeypatch, _tick_router_output(), recorded)

        await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert len(recorded) == 1
        call = recorded[0]
        assert call["acting_character_id"] == "alice"
        assert len(call["tick_outputs"]) == 2
        for name, char_id, location, public_text in call["tick_outputs"]:
            assert "paces" in public_text
            assert "watch the gate" not in public_text
            assert location == "hall"
            assert name in ("Regent", "Scribe")
            assert char_id in ("regent", "scribe")

    @pytest.mark.asyncio
    async def test_fan_in_appends_event(
        self, monkeypatch,
    ):
        ckpt = _ckpt(turns_since_last_tick=14, stagnation=15)
        ckpt.session.leading_at_s = 50
        routed = EventRouterOutput.model_validate({
            **_tick_router_output().model_dump(),
            "effective_at_s": 10,
        })
        _stub_character_agent(monkeypatch, recorder=[])
        _stub_dispatcher(monkeypatch, routed)
        before = len(ckpt.canonical_events)

        await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert len(ckpt.canonical_events) == before + 1
        assert ckpt.canonical_events[-1].effective_at_s == 50

    @pytest.mark.asyncio
    async def test_only_canonized_tick_drafts_commit_history_and_inbox(
        self, monkeypatch,
    ):
        ckpt = _ckpt(turns_since_last_tick=14, stagnation=15)
        by_id = {char.character_id: char for char in ckpt.characters}
        by_id["regent"].pending_observations = ["Regent saw a bell."]
        by_id["scribe"].pending_observations = ["Scribe saw a door."]
        routed = EventRouterOutput.model_validate({
            **_tick_router_output().model_dump(),
            "effective_at_s": 30,
            "duration_s": 5,
            "observers": [
                {
                    "character_id": "regent",
                    "observation_level": "d",
                    "response_priority": 1,
                },
            ],
        })
        _stub_character_agent(monkeypatch, recorder=[])
        _stub_dispatcher(monkeypatch, routed)

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert [char.character_id for char, _ in result] == ["regent"]
        assert "regent" in ckpt.character_conversations
        assert "scribe" not in ckpt.character_conversations
        assert by_id["regent"].pending_observations == []
        assert by_id["scribe"].pending_observations == ["Scribe saw a door."]
        assert by_id["regent"].last_agent_turn_at_s == 35
        assert by_id["scribe"].last_agent_turn_at_s is None

    @pytest.mark.asyncio
    async def test_fan_in_skipped_when_no_ticks_succeed(self, monkeypatch):
        class _AlwaysFailAgent:
            def __init__(self, client, prompt_mgr):
                self.last_usage = {}

            async def draft_tick(self, *, character, checkpoint, acting_character_id):
                raise RuntimeError("every tick fails this beat")

        monkeypatch.setattr(
            "app.engine.orchestrator.CharacterAgent", _AlwaysFailAgent,
        )
        recorded: list[dict] = []
        _stub_dispatcher(monkeypatch, _tick_router_output(), recorded)
        ckpt = _ckpt(turns_since_last_tick=14, stagnation=15)

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert result == []
        assert recorded == []
        assert ckpt.session.turns_since_last_tick == 0

    @pytest.mark.asyncio
    async def test_fan_in_router_failure_swallowed(self, monkeypatch):
        ckpt = _ckpt(turns_since_last_tick=14, stagnation=15)
        before = len(ckpt.canonical_events)
        _stub_character_agent(monkeypatch, recorder=[])
        _stub_dispatcher(
            monkeypatch,
            None,
            raise_on_call=RuntimeError("simulated fan-in API hiccup"),
        )

        result = await _orchestrator()._run_ticks(
            ckpt, acted_this_turn=set(), acting_id="alice",
        )

        assert result == []
        assert len(ckpt.canonical_events) == before
        assert ckpt.session.turns_since_last_tick == 0
