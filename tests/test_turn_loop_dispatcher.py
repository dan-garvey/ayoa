"""Tests for LLMDispatcher framing contracts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine import narrator as narrator_module
from app.engine.content_pack_compiler import (
    CompiledContentPackWriter,
    SCHEMA_VERSION as CONTENT_PACK_SCHEMA_VERSION,
)
from app.engine.content_lookup import MissingContentError
from app.engine.content_lookup import EventRouterContentLookupOutput
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop import pin_cat_ii_responder
from app.engine.turn_loop_contracts import (
    ROUTER_CONTINUATION_HEADER,
)
from app.engine.turn_loop_dispatcher import (
    EVENT_ROUTER_MAX_TOKENS,
    LLMDispatcher,
    _build_router_context,
)
from app.llm.client import LLMClient
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.content import (
    ContentKnowledgeEntityState,
    ContentPackState,
    PendingContentSignal,
)
from app.schemas.content_manager import ContentManagerOutput
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import (
    DndEventRouterOutput,
    EventRouterOutput,
    ObserverEntry,
)
from app.schemas.events import ObservableFact
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.dnd_cat_ii import DndCombatTurnPlan, RollPlan, RulesAdjudication
from app.schemas.state import (
    CatIIRollTransaction,
    DndCombatantState,
    DndCombatState,
    OpenCatIIEvent,
    OpenCommitment,
    RenderBufferEntry,
    StorySetting,
    WorldState,
)
from tests.support.factories import (
    character_record,
    checkpoint,
    dnd_router_output,
    llm_response,
    router_output,
)


def _ckpt(*, bindings: dict[str, str] | None = None):
    return checkpoint(
        bindings=bindings,
        world_state=WorldState(
            setting=StorySetting(genre="fantasy", tone="grim"),
        ),
        characters=[
            character_record(
                "alice",
                name="Alice",
                role="player",
                is_playable=True,
            ),
            character_record("pip", name="Pip", role="npc"),
        ],
    )


def _router_output() -> EventRouterOutput:
    return router_output(facts=[], observer_ids=[])


def _dnd_router_output() -> DndEventRouterOutput:
    return dnd_router_output(facts=[], observer_ids=[])


def _llm_response(parsed):
    return llm_response(parsed, content="{}", model="gpt-5.2")


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


def _queue_content_signal(
    ckpt,
    *,
    signal_id: str = "sig-1",
    ref_id: str = "room/entry",
    content_hash: str = "hash-1",
    kind: str = "location_card",
    metadata: dict | None = None,
) -> None:
    pack_state = ckpt.session.content_state.setdefault(
        "pack",
        ContentPackState(pack_id="pack"),
    )
    pack_state.pending_signals[signal_id] = PendingContentSignal(
        signal_id=signal_id,
        pack_id="pack",
        ref_id=ref_id,
        content_hash=content_hash,
        metadata=metadata or {
            "kind": kind,
            "visibility": "hidden",
            "summary": "Entry chamber context.",
            "exits": ["north"],
            "hazards": ["loose floor"],
            "clues": ["old crest"],
        },
    )


PACK_VERSION = "1.0.0"
SOURCE_FINGERPRINT = "sha256:test-source"


def _content_pack_db(
    tmp_path,
    rows: list[tuple[str, str, str, str, str, str]],
    *,
    aliases: list[tuple[str, str]] | None = None,
):
    db_path = tmp_path / "pack.sqlite"
    writer = CompiledContentPackWriter(
        db_path,
        pack_id="pack",
        pack_version=PACK_VERSION,
        source_fingerprint=SOURCE_FINGERPRINT,
    )
    writer.write_pack(
        pages=[],
        cards=[
            {
                "pack_id": pack_id,
                "ref": ref,
                "content_hash": content_hash,
                "card_kind": kind,
                "visibility": visibility,
                "summary": summary,
                "review_status": "approved",
                "confidence": 1.0,
            }
            for pack_id, ref, content_hash, kind, visibility, summary in rows
        ],
        aliases=[
            {
                "alias": alias,
                "ref": ref,
                "review_status": "approved",
                "confidence": 1.0,
            }
            for alias, ref in aliases or []
        ],
    )
    return db_path


def _content_pack_metadata(db_path, **overrides):
    metadata = {
        "db_path": str(db_path),
        "pack_version": PACK_VERSION,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "schema_version": CONTENT_PACK_SCHEMA_VERSION,
    }
    metadata.update(overrides)
    return metadata


def _enable_dnd(ckpt) -> None:
    ckpt.session.config.settings.ruleset_id = "dnd5e_basic"


def _open_cat_ii_event() -> OpenCatIIEvent:
    return OpenCatIIEvent(
        event_id="evt_open",
        initiator_id="alice",
        initiator_intention="I shove Pip away from the door.",
        required_responders=["pip"],
        collected_intentions={"pip": "I twist aside."},
        opening_observer_ids=["alice", "pip"],
        opening_observable_facts=["Alice lunges toward Pip at the doorway."],
    )


def _rules_adjudication(fact: str = "The action resolves.") -> RulesAdjudication:
    return RulesAdjudication(
        feasible=True,
        combat_status="ongoing",
        mechanical_summary=fact,
        visible_outcome_facts=[fact],
        state_deltas=[],
        combat_state_deltas=[],
        effect_deltas=[],
        spatial_deltas=[],
        rules_notes=[],
        fallback_reason="",
    )


def _activate_dnd_combat(ckpt) -> None:
    ckpt.session.active_combat = DndCombatState(
        combatants=[
            DndCombatantState(
                combatant_id="alice",
                character_id="alice",
                name="Alice",
                player_controlled=True,
                hit_points_current=12,
                hit_points_max=12,
            ),
            DndCombatantState(
                combatant_id="pip",
                character_id="pip",
                name="Pip",
                hit_points_current=7,
                hit_points_max=7,
                initiative_order=1,
            ),
        ],
    )


def _no_action_plan() -> DndCombatTurnPlan:
    return DndCombatTurnPlan(
        feasible=True,
        actions=[],
        no_action_reason="No roll is needed.",
    )


def _pending_content_record_count(ckpt, marker: str) -> int:
    return sum(
        1
        for message in ckpt.session_conversation
        if message.role == "assistant" and marker in str(message.content)
    )


class TestRouterContext:
    def test_context_has_no_scene_graph_or_scene_context_block(self):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ctx = _build_router_context(ckpt, "alice")
        assert "scene_graph" not in ctx
        assert "scene_context_block" not in ctx

    def test_router_context_does_not_consume_pending_observations(self):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        alice.pending_observations = ["A bell rings."]

        ctx = _build_router_context(ckpt, "alice")

        assert "since_last_turn_block" not in ctx
        assert alice.pending_observations == ["A bell rings."]

    def test_router_context_omits_derived_router_state(self):
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

        ctx = _build_router_context(ckpt, "alice")

        assert "relative_time_block" not in ctx
        assert "open_commitments_block" not in ctx
        assert "commitment_revision_block" not in ctx

    def test_first_turn_context_surfaces_current_locations(self):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        pip.location = "archive"

        ctx = _build_router_context(ckpt, "alice")

        assert "Location: gatehouse" in ctx["player_characters_block"]
        assert "Location: archive" in ctx["initial_roster_block"]

    def test_initial_roster_ignores_opening_content_history_only(self):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session_conversation = [
            ConversationMessage(
                role="assistant",
                content="location_card ref=room/entry hash=hash-1",
            )
        ]

        ctx = _build_router_context(ckpt, "alice")

        assert "## Initial Roster" in ctx["initial_roster_block"]
        assert "- pip" in ctx["initial_roster_block"]

    def test_initial_roster_still_omits_after_non_content_history(self):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session_conversation = [
            ConversationMessage(
                role="assistant",
                content="prior_event evt_1 @0+1 source=alice mode=intention",
            )
        ]

        ctx = _build_router_context(ckpt, "alice")

        assert ctx["initial_roster_block"] == ""

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
        assert (
            mock_client.complete.await_args.kwargs["max_tokens"]
            == EVENT_ROUTER_MAX_TOKENS
        )

    def test_pending_inventory_update_precedes_next_intention(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.pending_engine_state_updates = [
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
        assert ckpt.session.pending_engine_state_updates == []

    def test_fresh_intention_omits_private_liveness_tick_cues(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        pip.location = "archive"
        pip.private_state.tick_cues = ["the bell tolls twice"]
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="examine the lock",
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        assert "the bell tolls twice" not in user_content

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

    def test_router_input_omits_derived_commitment_context(
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
        assert "## Open Commitments" not in live_user
        assert "commit_search" not in live_user
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
                routing_role="observe_only",
            ),
            ObserverEntry(
                character_id="pip",
                observation_level="i",
                routing_role="observe_only",
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
        assert "obs alice:d:observe_only pip:i:observe_only" in record.content
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

    def test_route_intention_adds_pending_content_as_prior_history_only(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        _queue_content_signal(ckpt)
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I inspect the threshold.",
        ))

        messages = mock_client.complete.await_args.kwargs["messages"]
        system_content = messages[0]["content"]
        user_content = _last_user_content(messages)

        assert "location_card ref=room/entry" not in system_content
        assert "location_card ref=room/entry" not in user_content
        assert "## Initial Roster" in user_content
        assert "- pip" in user_content
        assert any(
            message.get("role") == "assistant"
            and "location_card ref=room/entry" in message.get("content", "")
            for message in messages
        )
        assert [message.role for message in ckpt.session_conversation] == [
            "assistant",
            "assistant",
        ]
        assert ckpt.session_conversation[0].content.startswith(
            "location_card ref=room/entry "
        )
        assert ckpt.session_conversation[1].content.startswith("prior_event ")
        assert ckpt.session.content_state["pack"].pending_signals == {}
        assert sorted(ckpt.session.content_state["pack"].introduced_refs) == [
            "pack::room/entry::hash-1"
        ]

    def test_route_intention_runs_lookup_preflight_before_router(
        self, prompt_mgr, mock_client, tmp_path,
    ):
        db_path = _content_pack_db(
            tmp_path,
            [
                (
                    "pack",
                    "room/entry",
                    "hash-entry",
                    "location_card",
                    "hidden",
                    "Entry chamber context.",
                )
            ],
        )
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.content_state = {
            "pack": ContentPackState(
                pack_id="pack",
                metadata=_content_pack_metadata(
                    db_path,
                    aliases={"threshold": "room/entry"},
                ),
            )
        }
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I inspect the threshold.",
        ))

        messages = mock_client.complete.await_args.kwargs["messages"]
        system_content = messages[0]["content"]
        user_content = _last_user_content(messages)
        assert "location_card ref=room/entry" not in system_content
        assert "location_card ref=room/entry" not in user_content
        assert any(
            message.get("role") == "assistant"
            and "location_card ref=room/entry" in message.get("content", "")
            for message in messages
        )
        assert ckpt.session_conversation[0].content.startswith(
            "location_card ref=room/entry "
        )
        assert ckpt.session_conversation[-1].content.startswith("prior_event ")

    def test_lookup_preflight_missing_content_aborts_before_router_call(
        self, prompt_mgr, mock_client, tmp_path,
    ):
        db_path = _content_pack_db(tmp_path, [])
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.content_state = {
            "pack": ContentPackState(
                pack_id="pack",
                metadata=_content_pack_metadata(
                    db_path,
                    aliases={"secret door": "room/secret"},
                ),
            )
        }
        before_content = ckpt.session.content_state["pack"].model_dump()

        with pytest.raises(MissingContentError):
            asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
                ckpt=ckpt,
                actor_id="alice",
                intention="I search for the secret door.",
            ))

        mock_client.complete.assert_not_awaited()
        assert ckpt.session.content_state["pack"].model_dump() == before_content
        assert ckpt.session_conversation == []

    def test_route_intention_runs_bounded_lookup_model_before_router(
        self, prompt_mgr, mock_client, tmp_path,
    ):
        db_path = _content_pack_db(
            tmp_path,
            [
                (
                    "pack",
                    "room/secret",
                    "hash-secret",
                    "location_card",
                    "hidden",
                    "Reviewed secret latch context.",
                )
            ],
        )
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.content_state = {
            "pack": ContentPackState(
                pack_id="pack",
                metadata=_content_pack_metadata(db_path),
            )
        }
        mock_client.complete.side_effect = [
            _llm_response(
                EventRouterContentLookupOutput(
                    requests=[{"pack_id": "pack", "ref": "room/secret"}]
                )
            ),
            _llm_response(_router_output()),
        ]

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I investigate the odd draft in the wall.",
        ))

        assert mock_client.complete.await_count == 2
        lookup_call = mock_client.complete.await_args_list[0].kwargs
        router_call = mock_client.complete.await_args_list[1].kwargs
        assert lookup_call["response_model"] is EventRouterContentLookupOutput
        assert router_call["response_model"] is EventRouterOutput
        lookup_text = "\n".join(
            message["content"] for message in lookup_call["messages"]
        )
        assert "room/secret" in lookup_text
        assert str(db_path) not in lookup_text

        router_messages = router_call["messages"]
        assert any(
            message.get("role") == "assistant"
            and "location_card ref=room/secret" in message.get("content", "")
            for message in router_messages
        )
        assert "location_card ref=room/secret" not in router_messages[0]["content"]
        assert "location_card ref=room/secret" not in _last_user_content(router_messages)
        assert ckpt.session_conversation[0].content.startswith(
            "location_card ref=room/secret "
        )
        assert ckpt.session_conversation[-1].content.startswith("prior_event ")

    def test_content_manager_preflight_projects_only_router_packet(
        self, prompt_mgr, mock_client, tmp_path,
    ):
        db_path = _content_pack_db(
            tmp_path,
            [
                (
                    "pack",
                    "front/strahd",
                    "hash-front",
                    "front_signal",
                    "hidden",
                    "The antagonist tracks public trouble.",
                ),
                (
                    "pack",
                    "room/secret",
                    "hash-secret",
                    "location_card",
                    "hidden",
                    "Reviewed secret latch context.",
                ),
            ],
        )
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.content_state = {
            "pack": ContentPackState(
                pack_id="pack",
                knowledge_map={
                    "pip": ContentKnowledgeEntityState(
                        entity_id="pip",
                        suspected_refs=["pack:front/old@hash-old"],
                        notes="full private knowledge",
                    )
                },
                metadata=_content_pack_metadata(
                    db_path,
                    router_knowledge_index=[
                        {
                            "key": "pack.room.secret",
                            "kind": "location",
                            "label": "Secret room",
                            "summary": "Reviewed secret latch context.",
                            "scope_facets": {"phase_tags": ["draft", "wall"]},
                            "activation_hints": ["draft", "wall", "secret"],
                            "priority": 100,
                            "packet_count": 1,
                            "packet_kinds": ["location_card"],
                        }
                    ],
                    router_knowledge_packets=[
                        {
                            "key": "pack.room.secret",
                            "packet_refs": [
                                {
                                    "pack_id": "pack",
                                    "ref": "room/secret",
                                    "content_hash": "hash-secret",
                                    "kind": "location_card",
                                    "visibility": "hidden",
                                    "summary": "Reviewed secret latch context.",
                                }
                            ],
                        }
                    ],
                ),
            )
        }
        ckpt.canonical_events = [
            router_output(
                facts=[ObservableFact.all("Alice notices a cold draft.")]
            )
        ]
        mock_client.complete.side_effect = [
            _llm_response(
                ContentManagerOutput(
                    knowledge_updates=[
                        {
                            "entity_id": "pip",
                            "pack_id": "pack",
                            "ref": "front/strahd",
                            "operation": "mark_known",
                            "reason": "f01 makes Pip relevant.",
                            "source_fact_ids": ["f01"],
                        }
                    ],
                    router_required_keys=[
                        {
                            "key": "pack.room.secret",
                            "reason": "The router needs the secret room context.",
                        }
                    ],
                    router_turn_candidates=[
                        {
                            "character_id": "pip",
                            "priority": "medium",
                            "reason": "Pip may react to the discovery.",
                            "source_fact_ids": ["f01"],
                            "related_keys": ["pack.room.secret"],
                        }
                    ],
                )
            ),
            _llm_response(_router_output()),
        ]

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I investigate the odd draft in the wall.",
        ))

        assert mock_client.complete.await_count == 2
        content_call = mock_client.complete.await_args_list[0].kwargs
        router_call = mock_client.complete.await_args_list[1].kwargs
        assert content_call["role"] == "content_manager"
        assert content_call["response_model"] is ContentManagerOutput
        assert router_call["role"] == "event_router"
        assert router_call["response_model"] is EventRouterOutput

        content_text = "\n".join(
            message["content"] for message in content_call["messages"]
        )
        assert "router_knowledge_state" in content_text
        assert "router_knowledge_dispatch_index" in content_text
        assert "engine_knowledge_map" not in content_text
        assert "pack=pack entity=pip" not in content_text

        router_text = "\n".join(
            str(message.get("content", ""))
            for message in router_call["messages"]
        )
        assert "location_card ref=room/secret" in router_text
        assert "turn_hint scope=attention_hint character=pip priority=medium" in router_text
        assert "engine_knowledge_map" not in router_text
        assert "pack=pack entity=pip" not in router_text
        assert "suspected=pack:front/old@hash-old" not in router_text
        assert "full private knowledge" not in router_text
        assert ckpt.session.content_state["pack"].knowledge_map["pip"].known_refs == [
            "pack:front/strahd@hash-front"
        ]
        assert ckpt.session_conversation[-1].content.startswith("prior_event ")

    def test_content_manager_preflight_throttles_between_cycles(
        self, prompt_mgr, mock_client, tmp_path,
    ):
        db_path = _content_pack_db(
            tmp_path,
            [
                (
                    "pack",
                    "front/strahd",
                    "hash-front",
                    "front_signal",
                    "hidden",
                    "The antagonist tracks public trouble.",
                ),
            ],
        )
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.config.settings.content_manager_refresh_interval = 3
        ckpt.session.content_state = {
            "pack": ContentPackState(
                pack_id="pack",
                knowledge_map={
                    "pip": ContentKnowledgeEntityState(entity_id="pip")
                },
                metadata=_content_pack_metadata(
                    db_path,
                    router_knowledge_index=[
                        {
                            "key": "pack.front.strahd",
                            "kind": "front",
                            "label": "Strahd front",
                            "summary": "The antagonist tracks public trouble.",
                            "scope_facets": {"character_ids": ["pip"]},
                            "activation_hints": ["wolf tracks"],
                            "priority": 100,
                            "packet_count": 1,
                            "packet_kinds": ["front_signal"],
                        }
                    ],
                    router_knowledge_packets=[
                        {
                            "key": "pack.front.strahd",
                            "packet_refs": [
                                {
                                    "pack_id": "pack",
                                    "ref": "front/strahd",
                                    "content_hash": "hash-front",
                                    "kind": "front_signal",
                                    "visibility": "hidden",
                                    "summary": "The antagonist tracks public trouble.",
                                }
                            ],
                        }
                    ],
                ),
            )
        }
        ckpt.canonical_events = [
            router_output(
                facts=[ObservableFact.all("Alice spots old wolf tracks.")]
            )
        ]
        mock_client.complete.side_effect = [
            _llm_response(
                ContentManagerOutput(
                    router_turn_candidates=[
                        {
                            "character_id": "pip",
                            "priority": "medium",
                            "reason": "Pip may react.",
                            "source_fact_ids": ["f01"],
                            "related_content_refs": ["pack:front/strahd"],
                        }
                    ],
                )
            ),
            _llm_response(_router_output()),
            _llm_response(_router_output()),
        ]

        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        asyncio.run(dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I check the tracks.",
        ))
        asyncio.run(dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I keep moving carefully.",
        ))

        roles = [
            call.kwargs["role"]
            for call in mock_client.complete.await_args_list
        ]
        assert roles == ["content_manager", "event_router", "event_router"]
        assert ckpt.session.content_manager_preflight_cycle == 2
        assert ckpt.session.content_manager_last_run_cycle == 0

    def test_content_manager_throttle_still_drains_pending_content(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.config.settings.content_manager_refresh_interval = 3
        ckpt.session.content_manager_preflight_cycle = 1
        ckpt.session.content_manager_last_run_cycle = 0
        ckpt.canonical_events = [
            router_output(
                facts=[ObservableFact.all("Alice hears stone scrape.")]
            )
        ]
        _queue_content_signal(ckpt)
        ckpt.session.content_state["pack"].metadata = {
            "router_knowledge_index": [
                {
                    "key": "pack.room.entry",
                    "kind": "location",
                    "label": "Entry room",
                    "summary": "Entry chamber context.",
                    "scope_facets": {"phase_tags": ["entry"]},
                    "activation_hints": ["entry", "door"],
                    "priority": 100,
                    "packet_count": 1,
                    "packet_kinds": ["location_card"],
                }
            ],
            "router_knowledge_packets": [
                {
                    "key": "pack.room.entry",
                    "packet_refs": [
                        {
                            "pack_id": "pack",
                            "ref": "room/entry",
                            "content_hash": "hash-1",
                            "kind": "location_card",
                            "visibility": "hidden",
                            "summary": "Entry chamber context.",
                        }
                    ],
                }
            ],
        }
        ckpt.session.content_state["pack"].knowledge_map = {
            "pip": ContentKnowledgeEntityState(entity_id="pip")
        }
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I push the entry door open.",
        ))

        assert mock_client.complete.await_count == 1
        router_call = mock_client.complete.await_args_list[0].kwargs
        assert router_call["role"] == "event_router"
        router_text = "\n".join(
            str(message.get("content", ""))
            for message in router_call["messages"]
        )
        assert "location_card ref=room/entry" in router_text
        assert ckpt.session.content_state["pack"].pending_signals == {}
        assert ckpt.session.content_manager_preflight_cycle == 2
        assert ckpt.session.content_manager_last_run_cycle == 0

    def test_content_manager_throttle_counter_rolls_back_on_failure(
        self, prompt_mgr, mock_client, tmp_path,
    ):
        db_path = _content_pack_db(
            tmp_path,
            [
                (
                    "pack",
                    "front/strahd",
                    "hash-front",
                    "front_signal",
                    "hidden",
                    "The antagonist tracks public trouble.",
                ),
            ],
        )
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.content_state = {
            "pack": ContentPackState(
                pack_id="pack",
                knowledge_map={
                    "pip": ContentKnowledgeEntityState(entity_id="pip")
                },
                metadata=_content_pack_metadata(db_path),
            )
        }
        ckpt.canonical_events = [
            router_output(
                facts=[ObservableFact.all("Alice spots old wolf tracks.")]
            )
        ]
        mock_client.complete.side_effect = RuntimeError("content lookup failed")

        with pytest.raises(RuntimeError, match="content lookup failed"):
            asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
                ckpt=ckpt,
                actor_id="alice",
                intention="I check the tracks.",
            ))

        assert ckpt.session.content_manager_preflight_cycle == 0
        assert ckpt.session.content_manager_last_run_cycle == -1

    def test_dnd_cat_ii_packet_receives_pending_content_context(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        _enable_dnd(ckpt)
        _queue_content_signal(ckpt)
        _queue_content_signal(
            ckpt,
            signal_id="sig-trap",
            ref_id="trap/needle",
            content_hash="hash-trap",
            metadata={
                "kind": "trap_card",
                "visibility": "hidden",
                "summary": "Needle trap under the door latch.",
            },
        )
        mock_client.complete.side_effect = [
            _llm_response(RollPlan(
                needs_rolls=False,
                roll_requests=[],
                no_roll_reason="The contest resolves without dice.",
            )),
            _llm_response(_rules_adjudication("Pip gives ground.")),
        ]

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I press Pip back.",
            cat_ii_event=_open_cat_ii_event(),
        ))

        plan_messages = mock_client.complete.await_args_list[0].kwargs["messages"]
        plan_user_content = _last_user_content(plan_messages)
        assert '"content_context": [' in plan_user_content
        assert "location_card ref=room/entry" in plan_user_content
        assert "content_known ref=trap/needle" in plan_user_content
        assert ckpt.session.content_state["pack"].pending_signals == {}
        assert _pending_content_record_count(ckpt, "location_card ref=room/entry") == 1
        assert _pending_content_record_count(ckpt, "content_known ref=trap/needle") == 1

        mock_client.complete.reset_mock()
        mock_client.complete.side_effect = None
        mock_client.complete.return_value = _llm_response(_dnd_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I inspect the latch again.",
        ))

        assert _pending_content_record_count(ckpt, "location_card ref=room/entry") == 1
        assert _pending_content_record_count(ckpt, "content_known ref=trap/needle") == 1

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
        assert "`interaction_mode`" in system_content
        assert "`combatant_spawns`" in system_content

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

    def test_cat_ii_dnd_mode_appends_compact_router_history(
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

        assert out.event_kind == "cat_ii_resolution"
        assert out.canonical_event.observable_facts[0].text == (
            "Pip steps aside before Alice hits him."
        )
        assert [
            call.kwargs["role"] for call in mock_client.complete.await_args_list
        ] == ["event_router", "event_router"]
        assert "## Cat II Resolution" not in _last_user_content(
            mock_client.complete.await_args_list[0].kwargs["messages"]
        )
        assert len(ckpt.session_conversation) == 1
        assert "mode=cat_ii_resolution" in ckpt.session_conversation[0].content
        assert "Pip steps aside before Alice hits him." in (
            ckpt.session_conversation[0].content
        )

    def test_dnd_combat_action_skips_router_history_while_active(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.active_combat = DndCombatState(combat_id="combat")
        result = _router_output()
        result.canonical_event.observable_facts = [
            ObservableFact.all("Alice cuts across Pip's guard."),
        ]

        class FakeCombatResolver:
            async def resolve_combat_action(self, **kwargs):
                assert kwargs["actor_id"] == "alice"
                return result

        monkeypatch.setattr(
            "app.engine.turn_loop_dispatcher.dnd_combat_manager_enabled",
            lambda checkpoint: True,
        )
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        dispatcher._dnd_combat = FakeCombatResolver()

        out = asyncio.run(dispatcher.route_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I strike Pip.",
        ))

        assert out is result
        assert ckpt.session_conversation == []

    def test_dnd_combat_end_appends_compact_router_history(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.active_combat = DndCombatState(combat_id="combat")
        result = _router_output()
        result.canonical_event.observable_facts = [
            ObservableFact.all("Alice accepts Pip's surrender."),
            ObservableFact.all("D&D combat ends."),
        ]

        class FakeCombatResolver:
            async def resolve_combat_action(self, **kwargs):
                assert kwargs["actor_id"] == "alice"
                kwargs["ckpt"].session.active_combat = None
                return result

        monkeypatch.setattr(
            "app.engine.turn_loop_dispatcher.dnd_combat_manager_enabled",
            lambda checkpoint: True,
        )
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        dispatcher._dnd_combat = FakeCombatResolver()

        out = asyncio.run(dispatcher.route_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I accept Pip's surrender.",
        ))

        assert out is result
        assert len(ckpt.session_conversation) == 1
        assert "mode=dnd_combat_end" in ckpt.session_conversation[0].content
        assert "Alice accepts Pip's surrender." in (
            ckpt.session_conversation[0].content
        )

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
        prior.event_kind = "beat_continues"
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

    def test_route_continuation_adds_pending_content_before_recovery_call(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        _queue_content_signal(ckpt, ref_id="front/villain", kind="front_signal")
        prior = _router_output()
        prior.event_kind = "beat_continues"
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_continuation(
            ckpt=ckpt,
            actor_id="alice",
            prior_result=prior,
        ))

        messages = mock_client.complete.await_args.kwargs["messages"]
        user_content = _last_user_content(messages)
        assert ROUTER_CONTINUATION_HEADER in user_content
        assert "front_signal ref=front/villain" not in user_content
        assert any(
            message.get("role") == "assistant"
            and "front_signal ref=front/villain" in message.get("content", "")
            for message in messages
        )

    def test_failed_router_call_restores_engine_state_updates(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.pending_engine_state_updates = [
            "Inventory update before the next action: alice took 8 sp.",
        ]
        before_updates = list(ckpt.session.pending_engine_state_updates)
        mock_client.complete.side_effect = RuntimeError("transient API failure")

        with pytest.raises(RuntimeError):
            asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
                ckpt=ckpt,
                actor_id="alice",
                intention="examine the lock",
            ))

        assert ckpt.session.pending_engine_state_updates == before_updates
        assert ckpt.session_conversation == []

    def test_failed_router_call_restores_pending_content_state(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        _queue_content_signal(ckpt)
        before_content = ckpt.session.content_state["pack"].model_dump()
        mock_client.complete.side_effect = RuntimeError("transient API failure")

        with pytest.raises(RuntimeError):
            asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
                ckpt=ckpt,
                actor_id="alice",
                intention="examine the threshold",
            ))

        assert ckpt.session.content_state["pack"].model_dump() == before_content
        assert ckpt.session_conversation == []


class TestDndCombatContentContext:
    def test_active_combat_packet_receives_pending_front_context(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        _enable_dnd(ckpt)
        _activate_dnd_combat(ckpt)
        _queue_content_signal(
            ckpt,
            ref_id="front/vampire",
            kind="front_signal",
            metadata={
                "kind": "front_signal",
                "visibility": "hidden",
                "summary": "The vampire presses the ambush.",
                "actor": "pip",
                "knows": "the east door is barred",
                "pressure": "draw Alice toward the trapped latch",
            },
        )
        mock_client.complete.side_effect = [
            _llm_response(_no_action_plan()),
            _llm_response(_rules_adjudication("Alice holds position.")),
        ]

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_combat_action(
            ckpt=ckpt,
            actor_id="alice",
            intention="I hold the doorway.",
        ))

        plan_messages = mock_client.complete.await_args_list[0].kwargs["messages"]
        plan_user_content = _last_user_content(plan_messages)
        assert '"content_context": [' in plan_user_content
        assert "front_signal ref=front/vampire" in plan_user_content
        assert ckpt.session.content_state["pack"].pending_signals == {}
        assert _pending_content_record_count(ckpt, "front_signal ref=front/vampire") == 1

        mock_client.complete.reset_mock()
        mock_client.complete.side_effect = None
        mock_client.complete.return_value = _llm_response(_dnd_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="I glance back at Pip.",
        ))

        assert _pending_content_record_count(ckpt, "front_signal ref=front/vampire") == 1

    def test_combat_continuation_merges_pending_content_into_transaction_packet(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        _enable_dnd(ckpt)
        _activate_dnd_combat(ckpt)
        transaction = CatIIRollTransaction(
            transaction_id="rolltxn_existing",
            event_id="cmb_existing",
            source="combat",
            actor_id="alice",
            intention="I wait for the opening.",
            ruleset_id="dnd5e_basic",
            status="ready_to_finalize",
            plan=_no_action_plan().model_dump(),
            context={
                "ruleset_id": "dnd5e_basic",
                "current_turn": {"actor_id": "alice"},
                "intention": "I wait for the opening.",
                "combatants": [],
            },
            no_roll_reason="No roll is needed.",
            ledger_lines=["No rolls were made."],
        )
        ckpt.session.cat_ii_roll_transactions.append(transaction)
        _queue_content_signal(
            ckpt,
            ref_id="trap/needle",
            content_hash="hash-trap",
            metadata={
                "kind": "trap_card",
                "visibility": "hidden",
                "summary": "Needle trap under the door latch.",
            },
        )
        mock_client.complete.return_value = _llm_response(
            _rules_adjudication("Alice keeps the opening covered.")
        )

        asyncio.run(
            LLMDispatcher(mock_client, prompt_mgr).continue_combat_transaction(
                ckpt=ckpt,
                event_id="cmb_existing",
            )
        )

        final_messages = mock_client.complete.await_args.kwargs["messages"]
        final_user_content = _last_user_content(final_messages)
        assert "content_known ref=trap/needle" in final_user_content
        assert transaction.context["content_context"] == [
            'content_known ref=trap/needle scope=router visibility=hidden '
            'hash=hash-trap kind=trap_card pack=pack '
            'summary="Needle trap under the door latch."'
        ]
        assert ckpt.session.content_state["pack"].pending_signals == {}


class TestRouteAgentOutput:
    def test_routes_agent_output_in_bare_user_message(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_agent_output(
            ckpt=ckpt,
            character_id="pip",
            public_text="He paces the threshold.",
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        assert user_content == "pip: He paces the threshold."
        assert "pip: He paces the threshold." in user_content
        assert "Pip" not in user_content
        assert "He paces the threshold" in user_content
        assert "foreground" not in user_content
        assert "background" not in user_content
        assert "evt_prior" not in user_content
        assert "agent_turn" not in user_content
        assert "## Frontier Results" not in user_content
        assert "<turn_context>" not in user_content
        assert "## Acting Character" not in user_content
        assert "## Player Characters" not in user_content
        assert "<input>" not in user_content
        assert "**alice** (acting this turn)" not in user_content
        assert "## Intention" not in user_content
        assert "## Cat II Resolution" not in user_content
        assert "source=pip mode=agent_output" in ckpt.session_conversation[-1].content

    def test_agent_output_keeps_content_delta_out_of_bare_user_message(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        _queue_content_signal(ckpt)
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_agent_output(
            ckpt=ckpt,
            character_id="pip",
            public_text="He paces the threshold.",
        ))

        messages = mock_client.complete.await_args.kwargs["messages"]
        user_content = _last_user_content(messages)
        assert user_content == "pip: He paces the threshold."
        assert "location_card ref=room/entry" not in user_content
        assert any(
            message.get("role") == "assistant"
            and "location_card ref=room/entry" in message.get("content", "")
            for message in messages
        )
        assert ckpt.session_conversation[0].content.startswith(
            "location_card ref=room/entry "
        )
        assert ckpt.session_conversation[-1].content.startswith("prior_event ")

    def test_agent_output_render_failure_restores_content_delta(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        _queue_content_signal(ckpt)
        before_content = ckpt.session.content_state["pack"].model_dump()

        def _boom(*args, **kwargs):
            raise RuntimeError("render failed")

        monkeypatch.setattr(prompt_mgr, "render_system_message", _boom)

        with pytest.raises(RuntimeError, match="render failed"):
            asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_agent_output(
                ckpt=ckpt,
                character_id="pip",
                public_text="He paces the threshold.",
            ))

        assert ckpt.session.content_state["pack"].model_dump() == before_content
        assert ckpt.session_conversation == []

    def test_agent_output_does_not_render_discarded_user_tail(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())

        def _boom(*args, **kwargs):
            raise AssertionError("route_agent_output should not render user tail")

        monkeypatch.setattr(prompt_mgr, "render_messages", _boom)

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_agent_output(
            ckpt=ckpt,
            character_id="pip",
            public_text="He paces the threshold.",
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        assert user_content == "pip: He paces the threshold."

    def test_agent_output_time_is_floored_to_session_edge(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        pip = next(c for c in ckpt.characters if c.character_id == "pip")
        ckpt.session.leading_at_s = 30
        pip.clock_at_s = 90
        routed = _router_output()
        routed.effective_at_s = 0
        mock_client.complete.return_value = _llm_response(routed)

        result = asyncio.run(
            LLMDispatcher(mock_client, prompt_mgr).route_agent_output(
                ckpt=ckpt,
                character_id="pip",
                public_text="He paces the threshold.",
            )
        )

        assert result.effective_at_s == 30
        assert (
            "source=pip mode=agent_output"
            in ckpt.session_conversation[-1].content
        )

    def test_agent_output_does_not_drain_original_actor_context(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        alice = next(c for c in ckpt.characters if c.character_id == "alice")
        alice.pending_observations = ["A bell rings for Alice."]
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_agent_output(
            ckpt=ckpt,
            character_id="pip",
            public_text="He paces the threshold.",
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        assert "A bell rings for Alice." not in user_content
        assert alice.pending_observations == ["A bell rings for Alice."]

    def test_agent_output_does_not_drain_next_fresh_engine_updates(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.pending_engine_state_updates = [
            "Inventory update before the next action: alice took 8 sp.",
        ]
        before_updates = list(ckpt.session.pending_engine_state_updates)
        mock_client.complete.return_value = _llm_response(_router_output())

        asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_agent_output(
            ckpt=ckpt,
            character_id="pip",
            public_text="He paces the threshold.",
        ))

        user_content = _last_user_content(
            mock_client.complete.await_args.kwargs["messages"]
        )
        assert user_content == "pip: He paces the threshold."
        assert "Inventory update before the next action" not in user_content
        assert ckpt.session.pending_engine_state_updates == before_updates

    def test_agent_output_failure_leaves_engine_updates_untouched(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.pending_engine_state_updates = [
            "Inventory update before the next action: alice took 8 sp.",
        ]
        before_updates = list(ckpt.session.pending_engine_state_updates)
        mock_client.complete.side_effect = RuntimeError("agent output API hiccup")

        with pytest.raises(RuntimeError):
            asyncio.run(LLMDispatcher(mock_client, prompt_mgr).route_agent_output(
                ckpt=ckpt,
                character_id="pip",
                public_text="He paces.",
            ))

        assert ckpt.session.pending_engine_state_updates == before_updates
        assert ckpt.session_conversation == []


class TestAgentIntend:
    def test_returns_public_text_only(self, prompt_mgr, mock_client, monkeypatch):
        ckpt = _ckpt(bindings={"alice": "discord_1"})

        async def _fake_turn(
            self, *, character, checkpoint,
            frame="foreground", local_context="",
        ):
            return CharacterAgentOutput(
                character_id=character.character_id,
                public_text='He plants himself in the doorway. "Hold there."',
                intent="Cover the threshold.",
            )

        monkeypatch.setattr(
            "app.engine.character_agent.CharacterAgent.turn",
            _fake_turn,
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

        async def _silent_turn(
            self, *, character, checkpoint,
            frame="foreground", local_context="",
        ):
            return CharacterAgentOutput(
                character_id=character.character_id,
                public_text="",
                intent="Watching to see who notices my silence.",
            )

        monkeypatch.setattr(
            "app.engine.character_agent.CharacterAgent.turn",
            _silent_turn,
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

        async def _fake_perceive(self, character, checkpoint):
            return loadouts[character.character_id]

        monkeypatch.setattr(
            "app.engine.character_agent.CharacterAgent.perceive",
            _fake_perceive,
        )

        out = asyncio.run(LLMDispatcher(mock_client, prompt_mgr).harvest_perceptions(
            ckpt=ckpt,
            character_ids=["vex", "pip"],
        ))
        assert out == [loadouts["vex"], loadouts["pip"]]

    def test_unknown_id_returns_empty_without_crash(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        out = asyncio.run(LLMDispatcher(mock_client, prompt_mgr).harvest_perceptions(
            ckpt=ckpt,
            character_ids=["never_existed"],
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

        async def _flaky_perceive(self, character, checkpoint):
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
