"""Tests for LLMDispatcher framing contracts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine import narrator as narrator_module
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop import pin_cat_ii_responder
from app.engine.turn_loop_contracts import ROUTER_CONTINUATION_HEADER
from app.engine.turn_loop_dispatcher import LLMDispatcher, _build_router_context
from app.llm.client import LLMClient, LLMResponse
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import (
    DndEventRouterOutput,
    EventRouterOutput,
    ObserverEntry,
)
from app.schemas.events import CanonicalEvent, ObservableFact, WorldAdjudication
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.dnd_cat_ii import RollPlan, RulesAdjudication
from app.schemas.state import (
    CommitmentRevisionPrompt,
    OpenCatIIEvent,
    OpenCommitment,
    RenderBufferEntry,
    SessionState,
    StorySetting,
    WorldState,
)


def _ckpt(*, bindings: dict[str, str] | None = None) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(session_id="s", character_bindings=bindings or {}),
        world_state=WorldState(
            setting=StorySetting(genre="fantasy", tone="grim"),
        ),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                public_sheet=PublicSheet(role="player"),
                location="gatehouse",
                is_playable=True,
            ),
            CharacterRecord(
                character_id="pip",
                name="Pip",
                public_sheet=PublicSheet(role="npc"),
                location="gatehouse",
                is_playable=False,
            ),
        ],
    )


def _router_output() -> EventRouterOutput:
    return EventRouterOutput(
        event_id="",
        decision_rationale="test fixture",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(feasible=True),
            observable_facts=[],
        ),
        observers=[],
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="directed_at_player",
        spawn=[],
        dormant=[],
        cull=[],
    )


def _dnd_router_output() -> DndEventRouterOutput:
    data = _router_output().model_dump()
    data["interaction_mode"] = "cat_i"
    data["combatant_ids"] = []
    return DndEventRouterOutput(**data)


def _llm_response(parsed) -> LLMResponse:
    raw = MagicMock()
    raw.content = []
    raw.model = "gpt-5.2"
    return LLMResponse(
        parsed=parsed,
        raw_response=raw,
        content="{}",
        model="gpt-5.2",
    )


@pytest.fixture
def prompt_mgr():
    return PromptManager("app/prompts")


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock()
    return client


def _last_user_content(messages: list[dict]) -> str:
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert user_msgs
    content = user_msgs[-1]["content"]
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content)
    return content


class TestRouterContext:
    def test_context_has_no_scene_graph_or_scene_context_block(self):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ctx = _build_router_context(ckpt, "alice")
        assert "scene_graph" not in ctx
        assert "scene_context_block" not in ctx

    def test_pending_observations_surface_once_then_drain(self):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        alice.pending_observations = ["A bell rings."]

        ctx = _build_router_context(ckpt, "alice")

        assert "A bell rings." in ctx["since_last_turn_block"]
        assert alice.pending_observations == []

    def test_relative_time_and_commitments_are_user_tail_context(self):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        alice.clock_at_s = 12
        ckpt.session.leading_at_s = 30
        ckpt.session.open_commitments = [
            OpenCommitment(
                commitment_id="commit_search",
                actor_ids=["alice"],
                description="Alice searches the cabinet.",
                started_at_s=12,
                expected_end_s=72,
                max_end_s=192,
                location_label="gatehouse",
            )
        ]
        ckpt.session.pending_commitment_revisions["alice"] = (
            CommitmentRevisionPrompt(
                character_id="alice",
                commitment_id="commit_search",
                trigger_event_id="evt_noise",
                observed_at_s=20,
                reason="the cabinet changed",
                previous_description="Alice searches the cabinet.",
            )
        )

        ctx = _build_router_context(ckpt, "alice")

        assert "Session leading time: 30s" in ctx["relative_time_block"]
        assert "alice at 12s" in ctx["relative_time_block"]
        assert "Alice (id: alice)" not in ctx["relative_time_block"]
        assert "commit_search" in ctx["open_commitments_block"]
        assert "Alice searches the cabinet." in ctx["open_commitments_block"]
        assert "evt_noise" in ctx["commitment_revision_block"]


class TestRouteIntention:
    def test_human_initiator_emits_attempts_framing(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="examine the lock",
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        assert "## Acting Character\nalice" in user_content
        assert "examine the lock" in user_content
        assert "Alice attempts:" not in user_content
        assert "alice attempts:" not in user_content
        assert "Alice intends:" not in user_content
        assert "## Intention" not in user_content

    def test_pending_inventory_update_precedes_next_intention(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.pending_router_state_changes = [
            "Inventory update before the next action: alice took 8 sp.",
        ]
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I leave the shop.",
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        update_index = user_content.index(
            "Inventory update before the next action",
        )
        intention_index = user_content.index("I leave the shop.")
        assert update_index < intention_index
        assert ckpt.session.pending_router_state_changes == []

    def test_npc_cascade_emits_intends_framing(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="pip",
            intention="polishes the bell",
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        assert "## Acting Character\npip" in user_content
        assert "polishes the bell" in user_content
        assert "Pip intends:" not in user_content
        assert "pip intends:" not in user_content
        assert "Pip attempts:" not in user_content
        assert "## Intention" not in user_content

    def test_router_history_omits_volatile_commitment_context(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.open_commitments = [
            OpenCommitment(
                commitment_id="commit_search",
                actor_ids=["alice"],
                description="Alice searches the cabinet.",
            )
        ]
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="check the hinges",
        ))

        live_user = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        stored = ckpt.session_conversation[-1]
        stored_text = stored.content
        assert "## Open Commitments" in live_user
        assert "commit_search" in live_user
        assert stored.role == "assistant"
        assert isinstance(stored_text, str)
        assert stored_text.startswith("prior_event evt_")
        assert "## Open Commitments" not in stored_text
        assert "commit_search" not in stored_text
        assert "[beat:" not in stored_text
        assert "effective_at_s" not in stored_text
        assert "decision_rationale" not in stored_text
        assert "world_adjudication" not in stored_text
        assert "## Intention" not in stored_text
        assert "check the hinges" not in stored_text

    def test_router_history_stores_compact_facts_without_user_message(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        result = _router_output()
        result.canonical_event.observable_facts = [
            ObservableFact.only(
                "Alice whispers, 'The hinge is loose.'",
                ["alice", "pip"],
                duration_s=2,
            )
        ]
        result.observers = [
            ObserverEntry(
                character_id="alice",
                observation_level="d",
                response_priority=1,
            ),
            ObserverEntry(
                character_id="pip",
                observation_level="i",
                response_priority=3,
            ),
        ]
        mock_client.complete.return_value = _llm_response(result)

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I check whether the hinge is loose.",
        ))

        assert len(ckpt.session_conversation) == 1
        record = ckpt.session_conversation[0]
        assert record.role == "assistant"
        assert isinstance(record.content, str)
        assert "prior_event" in record.content
        assert "source=alice mode=intention" in record.content
        assert "fact only[alice,pip] @0+2" in record.content
        assert "Alice whispers, 'The hinge is loose.'" in record.content
        assert "obs alice:d1 pip:i3" in record.content
        assert "I check whether the hinge is loose" not in record.content
        assert "decision_rationale" not in record.content
        assert '"canonical_event"' not in record.content

    def test_router_history_preserves_defer_user_prompt(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        result = _router_output()
        result.event_id = "evt_defer_continue"
        mock_client.complete.return_value = _llm_response(result)

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="  (defer)  ",
        ))

        assert [m.role for m in ckpt.session_conversation] == [
            "user",
            "assistant",
        ]
        assert ckpt.session_conversation[0].content == "(defer)"
        assert ckpt.session_conversation[1].content.startswith(
            "prior_event evt_defer_continue "
        )

    def test_router_history_replays_prior_defer_to_next_call(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        first = _router_output()
        first.event_id = "evt_defer_continue"
        second = _router_output()
        second.event_id = "evt_after_defer"
        mock_client.complete.side_effect = [
            _llm_response(first),
            _llm_response(second),
        ]

        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        asyncio.run(dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="(defer)",
        ))
        asyncio.run(dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="look around",
        ))

        second_messages = mock_client.complete.await_args_list[1].kwargs["messages"]
        prior_user_messages = [
            m for m in second_messages[:-1]
            if m.get("role") == "user"
        ]
        assert any(m["content"] == "(defer)" for m in prior_user_messages)
        assert [m.role for m in ckpt.session_conversation] == [
            "user",
            "assistant",
            "assistant",
        ]

    def test_dnd_fresh_intention_uses_dnd_router_contract(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        mock_client.complete.return_value = _llm_response(_dnd_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="draw steel",
        ))

        call = mock_client.complete.await_args.kwargs
        assert call["response_model"] is DndEventRouterOutput
        system_content = call["messages"][0]["content"]
        assert "D&D Interaction Mode" in system_content
        assert '"interaction_mode"' in system_content
        assert '"combatant_spawns"' in system_content

    def test_dnd_loot_offer_is_not_replayed_in_router_history(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        data = _dnd_router_output().model_dump()
        data["event_id"] = "evt_loot"
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
                    "notes": "red liquid",
                }
            ],
            "currency": {"gp": 5},
            "notes": "under the false bottom",
        }
        mock_client.complete.return_value = _llm_response(
            DndEventRouterOutput(**data)
        )

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="open the chest",
        ))

        stored = ckpt.session_conversation[-1].content
        assert "loot_offer" not in stored
        assert "healing_potion" not in stored
        assert "Potion of Healing" not in stored
        assert "5gp" not in stored
        assert "iron chest" not in stored
        assert "under the false bottom" not in stored

    def test_narrative_fresh_intention_keeps_generic_router_contract(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="draw steel",
        ))

        call = mock_client.complete.await_args.kwargs
        assert call["response_model"] is EventRouterOutput
        system_content = call["messages"][0]["content"]
        assert "D&D Interaction Mode" not in system_content
        assert '"interaction_mode"' not in system_content

    def test_cat_ii_resolution_formats_collected_intentions(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())
        evt = OpenCatIIEvent(
            event_id="evt_abc123",
            initiator_id="pip",
            initiator_intention="throws a punch at Alice",
            required_responders=["alice", "bob"],
            collected_intentions={
                "alice": "I duck",
                "bob": "[AFK-swept: no player intention]",
            },
            swept_responders=["bob"],
        )

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="pip",
            intention="throws a punch at Alice",
            cat_ii_event=evt,
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        assert "## Cat II Resolution" in user_content
        assert "Initiator (pip): throws a punch at Alice" in user_content
        assert "alice: I duck" in user_content
        assert "## Swept Responders (AFK)" in user_content
        assert "AFK-swept" not in user_content
        assert "attempts:" not in user_content

    def test_cat_ii_dnd_mode_uses_event_router_without_history_append(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.config.settings.ruleset_id = "dnd5e_basic"
        mock_client.complete.side_effect = [
            _llm_response(RollPlan(
                needs_rolls=False,
                roll_requests=[],
                no_roll_reason="Pip yields.",
            )),
            _llm_response(RulesAdjudication(
                feasible=True,
                mechanical_summary="Pip yields before contact.",
                visible_outcome_facts=["Pip steps aside before Alice hits him."],
                state_deltas=[],
                rules_notes=[],
                fallback_reason="",
            )),
        ]
        evt = OpenCatIIEvent(
            event_id="evt_abc123",
            initiator_id="alice",
            initiator_intention="I shove Pip",
            required_responders=["pip"],
            collected_intentions={"pip": "I yield"},
            opening_observer_ids=["alice", "pip"],
            opening_observable_facts=["Alice drives toward Pip."],
        )

        out = asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I shove Pip",
            cat_ii_event=evt,
        ))

        assert out.ends_beat_reason == "cat_ii_resolution"
        assert out.canonical_event.observable_facts[0].text == (
            "Pip steps aside before Alice hits him."
        )
        assert [
            call.kwargs["role"] for call in mock_client.complete.await_args_list
        ] == ["event_router", "event_router"]
        assert "## Cat II Resolution" not in _last_user_content(
            mock_client.complete.await_args_list[0].kwargs["messages"]
        )
        assert ckpt.session_conversation == []

    def test_session_conversation_passed_as_history(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        from app.schemas.conversation import ConversationMessage

        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session_conversation = [
            ConversationMessage(role="user", content="PRIOR_USER"),
            ConversationMessage(role="assistant", content="PRIOR_ASSISTANT"),
        ]
        mock_client.complete.return_value = _llm_response(_router_output())

        captured: dict = {}
        original = prompt_mgr.render_conversation

        def _spy(template_name, history, **variables):
            captured["history"] = history
            captured["template"] = template_name
            return original(template_name, history, **variables)

        monkeypatch.setattr(prompt_mgr, "render_conversation", _spy)

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="examine the lock",
        ))

        assert captured["template"] == "event_router"
        assert captured["history"] is ckpt.session_conversation
        assert len(ckpt.session_conversation) == 3
        assert ckpt.session_conversation[-1].role == "assistant"
        assert "prior_event" in ckpt.session_conversation[-1].content

    def test_route_continuation_uses_recovery_block_not_intention(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        prior = _router_output()
        prior.ends_beat = False
        prior.ends_beat_reason = ""
        prior.decision_rationale = "The beat stayed open without a pick."
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_continuation(
            ckpt=ckpt,
            actor_id="alice",
            prior_result=prior,
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        assert ROUTER_CONTINUATION_HEADER in user_content
        assert "The beat stayed open without a pick." in user_content
        assert "Alice attempts:" not in user_content
        assert "Alice intends:" not in user_content

    def test_failed_router_call_restores_drained_queues(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.pending_router_state_changes = [
            "Spawned: sera_01",
        ]
        ckpt.world_state.facts = ["The keep predates the road."]
        ckpt.session.surfaced_world_facts = []
        before_state_changes = list(ckpt.session.pending_router_state_changes)
        before_surfaced = list(ckpt.session.surfaced_world_facts)
        mock_client.complete.side_effect = RuntimeError("transient API failure")

        with pytest.raises(RuntimeError):
            asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
                ckpt=ckpt,
                actor_id="alice",
                intention="examine the lock",
            ))

        assert ckpt.session.pending_router_state_changes == before_state_changes
        assert ckpt.session.surfaced_world_facts == before_surfaced
        assert ckpt.session_conversation == []


class TestRouteTickIntentions:
    def test_empty_tick_outputs_returns_none_without_llm_call(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        result = asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_tick_intentions(
            ckpt=ckpt,
            tick_outputs=[],
        ))
        assert result is None
        mock_client.complete.assert_not_called()
        assert ckpt.session_conversation == []

    def test_bundles_per_ticker_prose_in_user_message(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_tick_intentions(
            ckpt=ckpt,
            tick_outputs=[
                ("Pip", "pip", "gatehouse", "He paces the threshold."),
                ("Wraith", "wraith_42", "tower", "It descends the stair."),
            ],
            acting_character_id="alice",
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        assert "## Off-Stage Tick" in user_content
        assert "pip" in user_content
        assert "Pip" not in user_content
        assert "He paces the threshold" in user_content
        assert "gatehouse" in user_content
        assert "wraith_42" in user_content
        assert "Wraith" not in user_content
        assert "It descends the stair" in user_content
        assert "tower" in user_content
        assert "## Intention" not in user_content
        assert "## Cat II Resolution" not in user_content

    def test_tick_router_failure_restores_queues(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.pending_router_state_changes = ["Spawned: Sera"]
        ckpt.world_state.facts = ["The keep predates the road."]
        before_state_changes = list(ckpt.session.pending_router_state_changes)
        before_surfaced = list(ckpt.session.surfaced_world_facts)
        mock_client.complete.side_effect = RuntimeError("tick API hiccup")

        with pytest.raises(RuntimeError):
            asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_tick_intentions(
                ckpt=ckpt,
                tick_outputs=[("Pip", "pip", "gatehouse", "He paces.")],
                acting_character_id="alice",
            ))

        assert ckpt.session.pending_router_state_changes == before_state_changes
        assert ckpt.session.surfaced_world_facts == before_surfaced
        assert ckpt.session_conversation == []


class TestAgentIntend:
    def test_returns_public_text_only(self, prompt_mgr, mock_client, monkeypatch):
        ckpt = _ckpt(bindings={"alice": "discord_1"})

        async def _fake_respond(self, *, character, checkpoint,
                                acting_character_id=""):
            return CharacterAgentOutput(
                character_id=character.character_id,
                public_text='He plants himself in the doorway. "Hold there."',
                intent="Cover the threshold.",
            )

        monkeypatch.setattr(
            "app.engine.character_agent.CharacterAgent.respond",
            _fake_respond,
        )

        out = asyncio.run(LLMDispatcher(mock_client, prompt_mgr).agent_intend(
            ckpt=ckpt,
            character_id="pip",
        ))
        assert "Hold there." in out
        assert "Cover the threshold" not in out

    def test_silent_beat_returns_sentinel_when_intent_present(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})

        async def _silent_respond(self, *, character, checkpoint,
                                  acting_character_id=""):
            return CharacterAgentOutput(
                character_id=character.character_id,
                public_text="",
                intent="Watching to see who notices my silence.",
            )

        monkeypatch.setattr(
            "app.engine.character_agent.CharacterAgent.respond",
            _silent_respond,
        )

        out = asyncio.run(LLMDispatcher(mock_client, prompt_mgr).agent_intend(
            ckpt=ckpt,
            character_id="pip",
        ))
        assert out == "(remains silent)"
        assert "notices" not in out


class TestHarvestPerceptions:
    def test_returns_fragments_in_input_order(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.characters.append(CharacterRecord(
            character_id="vex",
            name="Vex",
            public_sheet=PublicSheet(role="npc"),
            location="gatehouse",
        ))
        loadouts = {
            "pip": "Pip in patched leather.",
            "vex": "Vex in midnight silk.",
        }

        async def _fake_perceive(self, character, checkpoint,
                                 acting_character_id=""):
            return loadouts[character.character_id]

        monkeypatch.setattr(
            "app.engine.character_agent.CharacterAgent.perceive",
            _fake_perceive,
        )

        out = asyncio.run(LLMDispatcher(mock_client, prompt_mgr).harvest_perceptions(
            ckpt=ckpt,
            character_ids=["vex", "pip"],
            acting_character_id="alice",
        ))
        assert out == [loadouts["vex"], loadouts["pip"]]

    def test_unknown_id_returns_empty_without_crash(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        out = asyncio.run(LLMDispatcher(mock_client, prompt_mgr).harvest_perceptions(
            ckpt=ckpt,
            character_ids=["never_existed"],
            acting_character_id="alice",
        ))
        assert out == [""]

    def test_per_character_exception_absorbed_into_empty(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.characters.append(CharacterRecord(
            character_id="vex",
            name="Vex",
            public_sheet=PublicSheet(role="npc"),
            location="gatehouse",
        ))

        async def _flaky_perceive(self, character, checkpoint,
                                  acting_character_id=""):
            if character.character_id == "vex":
                raise RuntimeError("model timeout")
            return "Pip's loadout"

        monkeypatch.setattr(
            "app.engine.character_agent.CharacterAgent.perceive",
            _flaky_perceive,
        )

        out = asyncio.run(LLMDispatcher(mock_client, prompt_mgr).harvest_perceptions(
            ckpt=ckpt,
            character_ids=["pip", "vex"],
            acting_character_id="alice",
        ))
        assert out == ["Pip's loadout", ""]


class TestNarratorCompose:
    def test_partial_mode_true_when_pinned_as_cat_ii_responder(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        pin_cat_ii_responder(ckpt, "alice", "evt_abc")
        recorded: dict = {}

        async def _fake_compose_pov_render(
            *, client, prompt_mgr, ckpt, pov_character_id,
            buffered_events, partial_mode, user_input="",
        ):
            recorded["partial_mode"] = partial_mode
            recorded["pov"] = pov_character_id
            return (
                NarratorFinalOutput(final_text="RENDERED"),
                TranscriptEntry(user=user_input, assistant="RENDERED"),
            )

        monkeypatch.setattr(
            narrator_module,
            "compose_pov_render",
            _fake_compose_pov_render,
            raising=False,
        )

        out, _entry = asyncio.run(LLMDispatcher(mock_client, prompt_mgr).narrator_compose(
            ckpt=ckpt,
            character_id="alice",
            buffered_events=[RenderBufferEntry(event_id="e1")],
        ))
        assert out.final_text == "RENDERED"
        assert recorded["partial_mode"] is True
        assert recorded["pov"] == "alice"

    def test_partial_mode_override_wins(self, prompt_mgr, mock_client, monkeypatch):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        recorded: dict = {}

        async def _fake_compose_pov_render(
            *, client, prompt_mgr, ckpt, pov_character_id,
            buffered_events, partial_mode, user_input="",
        ):
            recorded["partial_mode"] = partial_mode
            return (
                NarratorFinalOutput(final_text="RENDERED"),
                TranscriptEntry(user=user_input, assistant="RENDERED"),
            )

        monkeypatch.setattr(
            narrator_module,
            "compose_pov_render",
            _fake_compose_pov_render,
            raising=False,
        )

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).narrator_compose(
            ckpt=ckpt,
            character_id="alice",
            buffered_events=[RenderBufferEntry(event_id="e1")],
            partial_mode_override=True,
        ))
        assert recorded["partial_mode"] is True
