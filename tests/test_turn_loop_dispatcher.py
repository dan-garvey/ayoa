"""Tests for LLMDispatcher — verifies the concrete Dispatcher binding
frames router / agent / narrator calls correctly per v11 contracts.

The LLMClient is mocked throughout; when `narrator.compose_pov_render`
doesn't exist yet in the narrator module (sibling agent still wiring),
it's monkeypatched in per-test.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine import narrator as narrator_module
from app.engine import turn_loop_dispatcher
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop import pin_cat_ii_responder
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.llm.client import LLMClient, LLMResponse
from app.schemas.agents import CharacterAgentOutput
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.events import CanonicalEvent, SceneDelta, WorldAdjudication
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import (
    LocationState,
    OpenCatIIEvent,
    RenderBufferEntry,
    SessionState,
    StorySetting,
    WorldState,
)


# ---- helpers ---------------------------------------------------------------


def _ckpt(*, bindings: dict[str, str] | None = None) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id="s",
            character_bindings=bindings or {},
        ),
        world_state=WorldState(
            locations=LocationState(
                scene_graph={
                    "gatehouse": {
                        "name": "The Gatehouse",
                        "description": "A stone gatehouse.",
                        "connected_to": [],
                    },
                },
            ),
            setting=StorySetting(genre="fantasy", tone="grim"),
        ),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                public_sheet=PublicSheet(role="player"),
                location="gatehouse",
                is_player=True,
            ),
            CharacterRecord(
                character_id="pip",
                name="Pip",
                public_sheet=PublicSheet(role="npc"),
                location="gatehouse",
                is_player=False,
            ),
        ],
    )


def _router_output() -> EventRouterOutput:
    """A minimal valid EventRouterOutput for the mocked LLM response."""
    return EventRouterOutput(
        event_id="",
        decision_rationale="(test fixture)",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                attempted_action="x",
                feasible=True,
                resolved_outcome="y",
            ),
            scene_delta=SceneDelta(time_advanced_seconds=0),
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
        roster_moves=[],
        scenes_created=[],
    )


def _llm_response(parsed) -> LLMResponse:
    raw = MagicMock()
    raw.content = []
    raw.model = "claude-haiku-4-5"
    return LLMResponse(
        parsed=parsed, raw_response=raw, content="{}", model="claude-haiku-4-5",
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
    """Pull the user-message content from the list `render_messages` returns."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert user_msgs, "expected at least one user message"
    content = user_msgs[-1]["content"]
    if isinstance(content, list):
        # Defensive — LLMClient adds cache_control blocks, but the
        # Dispatcher hands plain text to client.complete.
        return "".join(p.get("text", "") for p in content)
    return content


# ---- 1. route_intention framing for humans ---------------------------------


class TestRouteIntention:
    def test_human_initiator_emits_attempts_framing(
        self, prompt_mgr, mock_client,
    ):
        """Human-bound actor → `{name} attempts: {input}` framing."""
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)

        asyncio.run(dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="examine the lock",
            scene_id="gatehouse",
        ))

        assert mock_client.complete.await_count == 1
        kwargs = mock_client.complete.await_args.kwargs
        user_content = _last_user_content(kwargs["messages"])
        assert "Alice attempts: examine the lock" in user_content
        # NPC framing must NOT be present.
        assert "Alice intends:" not in user_content

    def test_npc_cascade_emits_intends_framing(
        self, prompt_mgr, mock_client,
    ):
        """Non-human actor → `{name} intends: {intention}` framing."""
        # No bindings — pip is an NPC.
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)

        asyncio.run(dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="pip",
            intention="polishes the bell",
            scene_id="gatehouse",
        ))

        kwargs = mock_client.complete.await_args.kwargs
        user_content = _last_user_content(kwargs["messages"])
        assert "Pip intends: polishes the bell" in user_content
        # Crucial — NPC cascades MUST NOT use "attempts:" framing.
        assert "Pip attempts:" not in user_content

    def test_cat_ii_resolution_emits_swept_subheader_when_present(
        self, prompt_mgr, mock_client,
    ):
        """Cat II adjudication with swept responders → swept sub-header fires."""
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)

        evt = OpenCatIIEvent(
            event_id="evt_abc123",
            scene_id="gatehouse",
            initiator_id="pip",
            initiator_intention="throws a punch at Alice",
            required_responders=["alice", "bob"],
            collected_intentions={
                "alice": "I duck",
                "bob": "[AFK-swept: no player intention]",
            },
            swept_responders=["bob"],
        )

        asyncio.run(dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="pip",
            intention="throws a punch at Alice",
            scene_id="gatehouse",
            cat_ii_event=evt,
        ))

        kwargs = mock_client.complete.await_args.kwargs
        user_content = _last_user_content(kwargs["messages"])
        assert "## Cat II Resolution" in user_content
        assert "Initiator (pip): throws a punch at Alice" in user_content
        assert "## Swept Responders (AFK)" in user_content
        # Live responder is present, swept sentinel text is NOT.
        assert "alice: I duck" in user_content
        assert "AFK-swept" not in user_content
        # Intention block must be empty in Part C call.
        assert "attempts:" not in user_content

    def test_cat_ii_resolution_no_swept_subheader_when_empty(
        self, prompt_mgr, mock_client,
    ):
        """Cat II adjudication with no swept responders → subheader absent."""
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)

        evt = OpenCatIIEvent(
            event_id="evt_abc123",
            scene_id="gatehouse",
            initiator_id="pip",
            initiator_intention="swings a fist",
            required_responders=["alice"],
            collected_intentions={"alice": "dodges"},
            swept_responders=[],
        )

        asyncio.run(dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="pip",
            intention="swings a fist",
            scene_id="gatehouse",
            cat_ii_event=evt,
        ))

        kwargs = mock_client.complete.await_args.kwargs
        user_content = _last_user_content(kwargs["messages"])
        assert "## Cat II Resolution" in user_content
        assert "alice: dodges" in user_content
        assert "## Swept Responders (AFK)" not in user_content

    def test_session_conversation_passed_as_history(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        """v11-r6c: route_intention uses `render_conversation`, passing
        `ckpt.session_conversation` as history so the event_router_v9
        prompt sees the full rolling router history (per its own line 8
        "The prior messages in this conversation are the full session
        history"). After the call, the user/assistant pair is appended
        so continuity compounds across turns."""
        from app.schemas.conversation import ConversationMessage

        ckpt = _ckpt(bindings={"alice": "discord_1"})
        # Pre-populate session_conversation with one prior exchange so
        # we can verify it's what gets forwarded verbatim as history.
        ckpt.session_conversation = [
            ConversationMessage(role="user", content="PRIOR_USER"),
            ConversationMessage(role="assistant", content="PRIOR_ASSISTANT"),
        ]
        mock_client.complete.return_value = _llm_response(_router_output())

        # Intercept render_conversation to capture the history kwarg.
        captured: dict = {}
        original = prompt_mgr.render_conversation

        def _spy(template_name, history, **variables):
            captured["history"] = history
            captured["template"] = template_name
            return original(template_name, history, **variables)

        monkeypatch.setattr(prompt_mgr, "render_conversation", _spy)

        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        asyncio.run(dispatcher.route_intention(
            ckpt=ckpt,
            actor_id="alice",
            intention="examine the lock",
            scene_id="gatehouse",
        ))

        assert captured["template"] == "event_router"
        # History kwarg IS the checkpoint's session_conversation list —
        # we pass the list itself, not a copy.
        assert captured["history"] is ckpt.session_conversation
        # And the prior messages are still the first two entries (post-
        # append of this turn's pair).
        assert captured["history"][0].content == "PRIOR_USER"
        assert captured["history"][1].content == "PRIOR_ASSISTANT"
        # After the call, session_conversation has grown by 2 entries
        # (this turn's user + assistant).
        assert len(ckpt.session_conversation) == 4
        assert ckpt.session_conversation[-2].role == "user"
        assert ckpt.session_conversation[-1].role == "assistant"

    def test_failed_router_call_restores_drained_session_queues(
        self, prompt_mgr, mock_client,
    ):
        """Commit-4b P0: if `client.complete` raises after the
        delta-builders have already mutated session state, the
        dispatcher must restore both `pending_router_state_changes`
        and `surfaced_world_facts` so the queued lines re-render on
        retry instead of being silently lost.
        """
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.pending_router_state_changes = [
            "Spawned: Sera (id: sera_01) — fresh arrival",
            "Player binding: Alice (id: alice) is now driven by a human player.",
        ]
        ckpt.world_state.facts = ["The keep predates the road."]
        ckpt.session.surfaced_world_facts = []
        before_state_changes = list(
            ckpt.session.pending_router_state_changes
        )
        before_surfaced = list(ckpt.session.surfaced_world_facts)

        boom = RuntimeError("simulated transient API failure")
        mock_client.complete.side_effect = boom

        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        with pytest.raises(RuntimeError):
            asyncio.run(dispatcher.route_intention(
                ckpt=ckpt,
                actor_id="alice",
                intention="examine the lock",
                scene_id="gatehouse",
            ))

        assert (
            ckpt.session.pending_router_state_changes == before_state_changes
        ), "pending_router_state_changes drained but not restored"
        assert (
            ckpt.session.surfaced_world_facts == before_surfaced
        ), "surfaced_world_facts mutated but not restored"


# ---- 4b. route_tick_intentions (Commit 6 fan-in) ---------------------------


class TestRouteTickIntentions:
    """Unit coverage of the off-stage tick fan-in router call.

    The dispatcher bundles N off-stage agents' PUBLIC prose into a
    single router call in tick mode. Two contracts the tests pin:

      - Empty `tick_outputs` → no LLM call, returns None. The
        orchestrator can compose a tick block from zero successful
        ticks (every per-tick attempt failed) and we must NOT spend
        a router call on an empty payload.
      - The bundled user message carries TICK_FAN_IN_HEADER + per-
        ticker prose, NEVER the parenthetical the agent originally
        emitted. The information-asymmetry guard is the caller's
        responsibility (orchestrator passes `output.public_text`),
        but the dispatcher must not synthesize anything that could
        leak interior either.
    """

    def test_empty_tick_outputs_returns_none_without_llm_call(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)

        result = asyncio.run(dispatcher.route_tick_intentions(
            ckpt=ckpt, tick_outputs=[],
        ))

        assert result is None
        mock_client.complete.assert_not_called()
        # Conversation history untouched — empty fan-in must not
        # smear a stub user/assistant pair onto the rolling router
        # context.
        assert ckpt.session_conversation == []

    def test_bundles_per_ticker_prose_in_user_message(
        self, prompt_mgr, mock_client,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)

        asyncio.run(dispatcher.route_tick_intentions(
            ckpt=ckpt,
            tick_outputs=[
                ("Pip", "pip", "gatehouse",
                 "He paces the threshold, watching the road."),
                ("Wraith", "wraith_42", "tower",
                 "It descends the stair, soundless."),
            ],
            acting_character_id="alice",
        ))

        assert mock_client.complete.await_count == 1
        kwargs = mock_client.complete.await_args.kwargs
        user_content = _last_user_content(kwargs["messages"])
        # Mode header so the prompt's tick-mode branch fires.
        assert "## Off-Stage Tick" in user_content
        # Per-ticker prose surfaces verbatim with the location annotation.
        assert "Pip" in user_content
        assert "He paces the threshold" in user_content
        assert "gatehouse" in user_content
        assert "Wraith" in user_content
        assert "It descends the stair" in user_content
        assert "tower" in user_content
        # The on-stage intention block + Cat II resolution block must
        # both be empty in tick mode (mutually exclusive user-message
        # slots per the USER-TEMPLATE CONTRACT in event_router_v9).
        assert "## Intention" not in user_content
        assert "## Cat II Resolution" not in user_content

    def test_user_message_contains_no_parenthetical_intent_text(
        self, prompt_mgr, mock_client,
    ):
        """Caller-side asymmetry guard: the dispatcher's signature
        only takes `public_text`, so even if a sloppy caller passed
        a string that happens to contain a trailing parenthetical,
        the dispatcher renders it as-is and does not invent a
        separate intent channel. This test pins that the helper has
        no hidden code path that resurrects intent text from
        somewhere.
        """
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)

        asyncio.run(dispatcher.route_tick_intentions(
            ckpt=ckpt,
            tick_outputs=[
                ("Pip", "pip", "gatehouse",
                 "He polishes the bell. PRIVATE_INTENT_SENTINEL"),
            ],
            acting_character_id="alice",
        ))

        kwargs = mock_client.complete.await_args.kwargs
        user_content = _last_user_content(kwargs["messages"])
        # The string we passed IS in user_content (we asked the
        # dispatcher to forward it verbatim). The asymmetry contract
        # is enforced upstream; here we just confirm the dispatcher
        # adds no SECOND surface beyond what we handed it — there's
        # no field labeled "intent" or "interior" in the bundle.
        assert "PRIVATE_INTENT_SENTINEL" in user_content
        assert "intent:" not in user_content.lower()
        assert "interior:" not in user_content.lower()

    def test_appends_to_session_conversation_after_call(
        self, prompt_mgr, mock_client,
    ):
        """Tick fan-in shares the same `session_conversation` history
        as on-stage router calls — single cache lineage, single
        source of routing truth. The user/assistant pair must land
        in that history after the LLM responds so the next router
        call (whether tick or on-stage) picks up the off-stage
        canonical event as context.
        """
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        mock_client.complete.return_value = _llm_response(_router_output())
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)

        before_len = len(ckpt.session_conversation)
        asyncio.run(dispatcher.route_tick_intentions(
            ckpt=ckpt,
            tick_outputs=[
                ("Pip", "pip", "gatehouse", "He paces."),
            ],
            acting_character_id="alice",
        ))

        # Exactly one user + one assistant appended.
        assert len(ckpt.session_conversation) == before_len + 2
        assert ckpt.session_conversation[-2].role == "user"
        assert ckpt.session_conversation[-1].role == "assistant"

    def test_failed_tick_router_call_restores_session_queues(
        self, prompt_mgr, mock_client,
    ):
        """Same snapshot/restore contract as `route_intention`: a
        failed tick fan-in must not silently drain queued state-
        changes / world-fact entries. The orchestrator swallows the
        exception, but the queues must survive into the next on-
        stage router call so spawn lines etc. still reach the
        router.
        """
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        ckpt.session.pending_router_state_changes = [
            "Spawned: Sera (id: sera_01) — fresh arrival",
        ]
        ckpt.world_state.facts = ["The keep predates the road."]
        ckpt.session.surfaced_world_facts = []
        before_state_changes = list(
            ckpt.session.pending_router_state_changes
        )
        before_surfaced = list(ckpt.session.surfaced_world_facts)

        mock_client.complete.side_effect = RuntimeError("tick API hiccup")
        dispatcher = LLMDispatcher(mock_client, prompt_mgr)

        with pytest.raises(RuntimeError):
            asyncio.run(dispatcher.route_tick_intentions(
                ckpt=ckpt,
                tick_outputs=[
                    ("Pip", "pip", "gatehouse", "He paces."),
                ],
                acting_character_id="alice",
            ))

        assert (
            ckpt.session.pending_router_state_changes == before_state_changes
        )
        assert ckpt.session.surfaced_world_facts == before_surfaced
        # And the failed call did NOT pollute conversation history —
        # snapshot/restore covers session_conversation by virtue of
        # the append-after-success ordering in the dispatcher.
        assert ckpt.session_conversation == []


# ---- 5. agent_intend returns a compact string ------------------------------


class TestAgentIntend:
    def test_returns_non_empty_for_successful_agent(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})

        # Stub CharacterAgent.respond to a fixed output, bypassing
        # prompt rendering + LLM plumbing. Commit 1: the agent emits
        # prose directly; `agent_intend` returns `public_text.strip()`,
        # so this fixture must populate `public_text` accordingly. The
        # parenthetical (intent) is the engine-private surface and MUST
        # NOT appear in the dispatched intention.
        async def _fake_respond(self, *, character, observed_facts, checkpoint,
                                prior_responses=None, acting_character_id=""):
            return CharacterAgentOutput(
                character_id=character.character_id,
                public_text=(
                    'He steps forward, wary, and plants himself in the '
                    'doorway. "Hold there."'
                ),
                intent="Cover the threshold; do not let the visitor pass.",
            )

        monkeypatch.setattr(
            "app.engine.character_agent.CharacterAgent.respond",
            _fake_respond,
        )

        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        out = asyncio.run(dispatcher.agent_intend(
            ckpt=ckpt, character_id="pip", scene_id="gatehouse",
        ))
        assert out
        # Prose surface present; no JSON braces or structured field names.
        assert "{" not in out
        assert '"dialogue":' not in out
        assert "Hold there." in out
        assert "steps forward" in out
        # Intent leak guard: the parenthetical's contents MUST NOT reach
        # the router. The dispatcher hands `public_text` only.
        assert "Cover the threshold" not in out

    def test_returns_empty_when_agent_output_empty(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})

        async def _empty_respond(self, *, character, observed_facts, checkpoint,
                                 prior_responses=None, acting_character_id=""):
            return CharacterAgentOutput(
                character_id=character.character_id,
                public_text="",
                intent="",
            )

        monkeypatch.setattr(
            "app.engine.character_agent.CharacterAgent.respond",
            _empty_respond,
        )

        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        out = asyncio.run(dispatcher.agent_intend(
            ckpt=ckpt, character_id="pip", scene_id="gatehouse",
        ))
        assert out == ""

    def test_silent_beat_returns_sentinel_when_intent_present(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        """The agent prompt's "Sparse is valid" shared rule says
        paren-only output (silent beat) is a valid in-character
        choice. The dispatcher must surface it as a recognizable
        sentinel so the cascade routes the beat instead of collapsing
        to `cascade_exhausted`. The sentinel must NOT leak the
        agent's private intent."""
        ckpt = _ckpt(bindings={"alice": "discord_1"})

        async def _silent_respond(self, *, character, observed_facts, checkpoint,
                                  prior_responses=None, acting_character_id=""):
            return CharacterAgentOutput(
                character_id=character.character_id,
                public_text="",
                intent=(
                    "He spoke for me. I let it land. "
                    "Watching to see who notices my silence."
                ),
            )

        monkeypatch.setattr(
            "app.engine.character_agent.CharacterAgent.respond",
            _silent_respond,
        )

        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        out = asyncio.run(dispatcher.agent_intend(
            ckpt=ckpt, character_id="pip", scene_id="gatehouse",
        ))
        assert out == "(remains silent)"
        assert "spoke for me" not in out


# ---- 6. narrator_compose passes partial_mode correctly ---------------------


class TestNarratorCompose:
    def test_partial_mode_true_when_pinned_as_cat_ii_responder(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        # Pin Alice as a Cat II responder in some scene.
        pin_cat_ii_responder(ckpt, "gatehouse", "alice", "evt_abc")

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
            narrator_module, "compose_pov_render",
            _fake_compose_pov_render, raising=False,
        )

        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        out, _entry = asyncio.run(dispatcher.narrator_compose(
            ckpt=ckpt, character_id="alice",
            buffered_events=[RenderBufferEntry(event_id="e1")],
        ))
        assert out.final_text == "RENDERED"
        assert recorded["partial_mode"] is True
        assert recorded["pov"] == "alice"

    def test_partial_mode_false_when_not_pinned(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        # Alice is NOT pinned anywhere.

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
            narrator_module, "compose_pov_render",
            _fake_compose_pov_render, raising=False,
        )

        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        asyncio.run(dispatcher.narrator_compose(
            ckpt=ckpt, character_id="alice",
            buffered_events=[RenderBufferEntry(event_id="e1")],
        ))
        assert recorded["partial_mode"] is False

    def test_partial_mode_override_wins_over_autodetect(
        self, prompt_mgr, mock_client, monkeypatch,
    ):
        """v11-r6a: callers can force partial_mode=True for POVs that
        aren't pinned (e.g. the initiator of a Cat II open beat)."""
        ckpt = _ckpt(bindings={"alice": "discord_1"})
        # Alice is NOT pinned anywhere — autodetect would return False.

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
            narrator_module, "compose_pov_render",
            _fake_compose_pov_render, raising=False,
        )

        dispatcher = LLMDispatcher(mock_client, prompt_mgr)
        asyncio.run(dispatcher.narrator_compose(
            ckpt=ckpt, character_id="alice",
            buffered_events=[RenderBufferEntry(event_id="e1")],
            partial_mode_override=True,
        ))
        # Override wins even though Alice isn't pinned.
        assert recorded["partial_mode"] is True
