"""Tests for the v11 per-POV narrator entry point (`compose_pov_render`).

Exercises the new function against a mocked LLMClient so we can verify:
- Buffered events resolve against ckpt.canonical_events by event_id and
  the resolved prose is returned unchanged from the LLM parsed output.
- Per-POV rolling history grows by one user + one assistant message.
- partial_mode toggles the PARTIAL_MODE_MARKER into the user payload.
- A stale buffer entry (event_id missing from canonical_events) is
  logged and skipped without aborting the render.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.narrator import compose_pov_render
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import PARTIAL_MODE_MARKER
from app.llm.client import LLMClient, LLMResponse
from app.schemas.characters import CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    SceneDelta,
    WorldAdjudication,
)
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.state import (
    LocationState,
    RenderBufferEntry,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)


# ---- helpers --------------------------------------------------------------


def _ckpt() -> CheckpointFile:
    """Minimal checkpoint with one scene, two canonical events, and a
    single human-bound character."""
    ckpt = CheckpointFile(
        session=SessionState(
            session_id="s",
            character_bindings={"alice": "1"},
            player_character_id="alice",
        ),
        world_state=WorldState(
            locations=LocationState(
                scene_graph={
                    "gatehouse": {
                        "name": "Gatehouse",
                        "description": "Weathered stone arch.",
                        "connected_to": [],
                    },
                },
            ),
            setting=StorySetting(genre="fantasy", tone="quiet"),
        ),
        characters=[
            CharacterRecord(
                character_id="alice", name="Alice",
                public_sheet=PublicSheet(role="player", appearance="dark-haired"),
                location="gatehouse",
                is_playable=True,
            ),
            CharacterRecord(
                character_id="pip", name="Pip",
                public_sheet=PublicSheet(role="npc"),
                location="gatehouse",
                is_playable=False,
            ),
        ],
        config=SessionConfig(narrative_rules="Concise prose."),
    )
    # Seed two canonical events into the log.
    ev1 = EventRouterOutput(
        event_id="evt_alpha",
        decision_rationale="(test fixture)",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                feasible=True,
                resolved_outcome="Alice sees the arch.",
            ),
            scene_delta=SceneDelta(time_advanced_seconds=0),
            observable_facts=["The arch is weathered."],
        ),
        observers=[
            ObserverEntry(character_id="alice", observation_level="d", response_priority=3),
        ],
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="",
        spawn=[],
        dormant=[],
        cull=[],
        roster_moves=[],
        scenes_created=[],
    )
    ev2 = EventRouterOutput(
        event_id="evt_beta",
        decision_rationale="(test fixture)",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                feasible=True,
                resolved_outcome="Pip dips his chin.",
            ),
            scene_delta=SceneDelta(time_advanced_seconds=0),
            observable_facts=["Pip nods."],
        ),
        observers=[
            ObserverEntry(character_id="alice", observation_level="d", response_priority=3),
        ],
        requires_responders=False,
        required_responders=[],
        agent_responder_picks=[],
        ends_beat=True,
        ends_beat_reason="",
        spawn=[],
        dormant=[],
        cull=[],
        roster_moves=[],
        scenes_created=[],
    )
    ckpt.canonical_events.extend([ev1, ev2])
    return ckpt


def _llm_response(final_text: str = "RENDERED") -> LLMResponse:
    """Minimal LLMResponse that can pass through
    `serialize_assistant_content` when the narrator appends history."""
    parsed = NarratorFinalOutput(final_text=final_text)
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "{}"
    # model_dump is called in serialize_assistant_content's fallback path;
    # for type=="text" it takes the `text` attr directly, so either works.
    text_block.model_dump = lambda: {"type": "text", "text": "{}"}
    raw.content = [text_block]
    raw.model = "claude-sonnet-4-6"
    return LLMResponse(
        parsed=parsed,
        raw_response=raw,
        content="{}",
        model="claude-sonnet-4-6",
    )


@pytest.fixture
def prompt_manager() -> PromptManager:
    return PromptManager("app/prompts")


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=_llm_response("RENDERED"))
    return client


# ---- tests ----------------------------------------------------------------


class TestComposePovRender:
    @pytest.mark.asyncio
    async def test_basic_render_appends_history(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        buffered = [
            RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
            RenderBufferEntry(event_id="evt_beta", observation_level="indirect"),
        ]

        result, entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=False,
            user_input="I look around.",
        )

        # v11-r7j: compose_pov_render returns (envelope, transcript_entry).
        # The narrator only emits final_text; the engine builds the entry
        # from the real player input (passed in) and the rendered prose.
        assert isinstance(result, NarratorFinalOutput)
        assert result.final_text == "RENDERED"
        assert entry.user == "I look around."
        assert entry.assistant == "RENDERED"
        # Per-POV history grew by exactly one exchange (user + assistant).
        alice_hist = ckpt.narrator_conversations["alice"]
        assert len(alice_hist) == 2
        assert alice_hist[0].role == "user"
        assert alice_hist[1].role == "assistant"

        # Canonical event details made it into the rendered prompt.
        call_kwargs = mock_client.complete.call_args.kwargs
        flat = "\n".join(
            m["content"] for m in call_kwargs["messages"]
            if isinstance(m.get("content"), str)
        )
        assert "evt_alpha" in flat
        assert "evt_beta" in flat
        assert "The arch is weathered" in flat  # observable_fact for ev1
        assert "Pip nods" in flat  # observable_fact for ev2
        # Audit/framing fields are dropped from the narrator input.
        assert "Alice sees the arch" not in flat
        assert "Pip dips his chin" not in flat
        # When partial_mode=False the user message should NOT start with
        # the PARTIAL marker. (The marker string itself is documented in
        # the prompt template, so we can't check a blanket "not in" —
        # check the user message head specifically.)
        user_msg = call_kwargs["messages"][-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], str)
        assert not user_msg["content"].startswith(PARTIAL_MODE_MARKER)

    @pytest.mark.asyncio
    async def test_partial_mode_prepends_marker(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        buffered = [
            RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
        ]

        await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=True,
        )

        call_kwargs = mock_client.complete.call_args.kwargs
        messages = call_kwargs["messages"]
        # Marker lives on the last (user) message in the sequence.
        last = messages[-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], str)
        assert last["content"].startswith(PARTIAL_MODE_MARKER)

        # And the stored history captures the marker — so the next
        # per-POV call will still see that PARTIAL was the framing for
        # this exchange.
        alice_hist = ckpt.narrator_conversations["alice"]
        assert alice_hist[0].role == "user"
        assert PARTIAL_MODE_MARKER in alice_hist[0].content

    @pytest.mark.asyncio
    async def test_no_partial_marker_when_not_partial(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        buffered = [
            RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
        ]

        await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=buffered,
            partial_mode=False,
        )

        call_kwargs = mock_client.complete.call_args.kwargs
        # The user message (last in the sequence) must NOT lead with the
        # marker. We check the user message head specifically because
        # the marker STRING appears inside the system prompt's rule-15
        # documentation.
        user_msg = call_kwargs["messages"][-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], str)
        assert not user_msg["content"].startswith(PARTIAL_MODE_MARKER)
        # And the POV's stored history records a user message that also
        # doesn't lead with the marker.
        alice_hist = ckpt.narrator_conversations["alice"]
        assert alice_hist[0].role == "user"
        assert not alice_hist[0].content.startswith(PARTIAL_MODE_MARKER)

    @pytest.mark.asyncio
    async def test_missing_event_id_is_warned_and_skipped(
        self, mock_client, prompt_manager, caplog,
    ):
        ckpt = _ckpt()
        # Two entries: one stale (not in canonical_events), one valid.
        buffered = [
            RenderBufferEntry(event_id="evt_ghost", observation_level="direct"),
            RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
        ]

        with caplog.at_level(logging.WARNING, logger="app.engine.narrator"):
            result, _entry = await compose_pov_render(
                client=mock_client,
                prompt_mgr=prompt_manager,
                ckpt=ckpt,
                pov_character_id="alice",
                buffered_events=buffered,
                partial_mode=False,
            )

        assert result.final_text == "RENDERED"
        # The missing id should appear in a warn log.
        assert any("evt_ghost" in rec.message for rec in caplog.records)
        # And the real event should still have been rendered.
        call_kwargs = mock_client.complete.call_args.kwargs
        flat = "\n".join(
            m["content"] for m in call_kwargs["messages"]
            if isinstance(m.get("content"), str)
        )
        assert "evt_alpha" in flat
        assert "evt_ghost" not in flat


class TestFormatCanonicalEventsBlock:
    """The narrator reads only visible `observable_facts` from each
    canonical event. Audit/framing fields are not part of the render
    input."""

    def _resolved(
        self, *, event_id: str, outcome: str, facts: list[object],
        level: str = "direct", observers: list[str] | None = None,
        fact_visibility: str = "all",
    ):
        ev = EventRouterOutput(
            event_id=event_id,
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True,
                    resolved_outcome=outcome,
                ),
                scene_delta=SceneDelta(time_advanced_seconds=0),
                observable_facts=list(facts),
            ),
            observers=[
                ObserverEntry(
                    character_id=cid,
                    observation_level="d",
                    response_priority=3,
                )
                for cid in (observers or [])
            ],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="",
            spawn=[], dormant=[], cull=[], roster_moves=[], scenes_created=[],
        )
        entry = RenderBufferEntry(
            event_id=event_id,
            observation_level=level,
            fact_visibility=fact_visibility,
        )
        return [(entry, ev)]

    def test_facts_surface_audit_fields_do_not(self):
        from app.engine.narrator import _format_canonical_events_block

        resolved = self._resolved(
            event_id="evt_x",
            outcome=(
                "Seraphel recites a fractured plague verse, the strain "
                "of speaking close to the edge of what she is permitted "
                "showing in her wings."
            ),
            facts=[
                "Seraphel recites: 'The plague that fell on human ground'",
                "her wings draw tight against her back",
            ],
        )
        out = _format_canonical_events_block(resolved)
        assert "observable_facts:" in out
        assert "wings draw tight" in out
        assert "The plague that fell" in out
        # Audit line must NOT appear — that's the whole point.
        assert "resolved_outcome:" not in out
        assert "what she is permitted" not in out
        assert "the strain of speaking" not in out

    def test_empty_facts_renders_none_marker(self):
        from app.engine.narrator import _format_canonical_events_block

        resolved = self._resolved(
            event_id="evt_y",
            outcome="(audit-only)",
            facts=[],
        )
        out = _format_canonical_events_block(resolved)
        assert "observable_facts: (none)" in out
        assert "(audit-only)" not in out

    def test_scoped_facts_filter_by_pov_before_narrator_sees_them(self):
        from app.engine.narrator import _format_canonical_events_block
        resolved = self._resolved(
            event_id="evt_private",
            outcome="Dan questions Thessaly and signals Ashara.",
            facts=[
                ObservableFact.only(
                    "Dan's foot touches Ashara's boot under the table.",
                    ["ashara"],
                ),
                ObservableFact.all(
                    "Dan asks Thessaly whether she knows curses.",
                ),
            ],
            observers=["ashara", "aldric"],
        )

        as_ashara = _format_canonical_events_block(resolved, "ashara")
        as_aldric = _format_canonical_events_block(resolved, "aldric")

        assert "foot touches Ashara's boot" in as_ashara
        assert "knows curses" in as_ashara
        assert "foot touches Ashara's boot" not in as_aldric
        assert "knows curses" in as_aldric

    def test_remote_explicit_only_buffer_drops_public_room_facts(self):
        from app.engine.narrator import _format_canonical_events_block

        resolved = self._resolved(
            event_id="evt_monitor",
            outcome="A monitored room event.",
            facts=[
                ObservableFact.all("The room lights flicker."),
                ObservableFact.only(
                    "A live monitor carries Alice's voice into the booth.",
                    ["remote_watcher"],
                ),
            ],
            observers=["alice", "remote_watcher"],
            fact_visibility="explicit_only",
        )

        out = _format_canonical_events_block(resolved, "remote_watcher")

        assert "live monitor carries Alice's voice" in out
        assert "room lights flicker" not in out


# NOTE: The `TestOpeningVerbatimRender` class lived here in v8 and earlier.
# It exercised a now-removed verbatim shortcut that rendered the importer's
# `opening_narrative` byte-for-byte on the first `(begin)` turn. The whole
# `opening_narrative` field was removed in v9 — the router now composes the
# opening dynamically from world_state and the narrator renders it like any
# other turn — so the verbatim path, its gates, and these tests are obsolete.
# The dynamic opening flow is covered by router/narrator turn-loop tests.
