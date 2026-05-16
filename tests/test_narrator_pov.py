"""Tests for the per-POV narrator entry point (`compose_pov_render`).

Exercises the new function against a mocked LLMClient so we can verify:
- Buffered events resolve against ckpt.canonical_events by event_id and
  the resolved prose is returned unchanged from the LLM parsed output.
- Per-POV rolling history stores assistant messages only.
- partial_mode puts the stop-before-resolution instruction in the user payload.
- A stale buffer entry (event_id missing from canonical_events) is
  logged and skipped without aborting the render.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.context_builder import build_narrator_public_character_context_block
from app.engine.narrator import compose_pov_render
from app.engine.prompt_manager import PromptManager
from app.engine.turn_loop_contracts import PARTIAL_MODE_MARKER
from app.llm.client import LLMClient, LLMResponse
from app.schemas.characters import CharacterDescriptions, CharacterRecord, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput, ObserverEntry
from app.schemas.events import (
    CanonicalEvent,
    ObservableFact,
    WorldAdjudication,
)
from app.schemas.narrator import NarratorFinalOutput
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
    """Minimal checkpoint with two canonical events and a
    single human-bound character."""
    ckpt = CheckpointFile(
        session=SessionState(
            session_id="s",
            character_bindings={"alice": "1"},
            player_character_id="alice",
        ),
        world_state=WorldState(
            locations=LocationState(),
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
    )
    ev2 = EventRouterOutput(
        event_id="evt_beta",
        decision_rationale="(test fixture)",
        canonical_event=CanonicalEvent(
            world_adjudication=WorldAdjudication(
                feasible=True,
                resolved_outcome="Pip dips his chin.",
            ),
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
    )
    ckpt.canonical_events.extend([ev1, ev2])
    return ckpt


def test_public_character_context_skips_nameless_player_slot():
    ckpt = _ckpt()
    ckpt.characters.append(CharacterRecord(
        character_id="player_protagonist",
        name="",
        descriptions=CharacterDescriptions(
            public="The nameless player slot should not render as a blank label.",
            private="Private player-slot context.",
        ),
        is_playable=True,
    ))

    block = build_narrator_public_character_context_block(ckpt)

    assert "blank label" not in block
    assert "- :" not in block


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

        # The narrator only emits final_text; the engine builds the entry
        # from the real player input (passed in) and the rendered prose.
        assert isinstance(result, NarratorFinalOutput)
        assert result.final_text == "RENDERED"
        assert entry.user == "I look around."
        assert entry.assistant == "RENDERED"
        # Per-POV history stores only the assistant output.
        alice_hist = ckpt.narrator_conversations["alice"]
        assert len(alice_hist) == 1
        assert alice_hist[0].role == "assistant"

        # Visible details made it into the rendered prompt.
        call_kwargs = mock_client.complete.call_args.kwargs
        flat = "\n".join(
            m["content"] for m in call_kwargs["messages"]
            if isinstance(m.get("content"), str)
        )
        assert "evt_alpha" not in flat
        assert "evt_beta" not in flat
        assert "The arch is weathered" in flat
        assert "Pip nods" in flat
        # Audit/framing fields are dropped from the narrator input.
        assert "Alice sees the arch" not in flat
        assert "Pip dips his chin" not in flat
        user_msg = call_kwargs["messages"][-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], str)
        assert PARTIAL_MODE_MARKER not in user_msg["content"]

    @pytest.mark.asyncio
    async def test_render_strips_unmatched_trailing_brace_from_final_text(
        self, mock_client, prompt_manager,
    ):
        mock_client.complete = AsyncMock(return_value=_llm_response(
            "She says, 'entirely human?'}",
        ))
        ckpt = _ckpt()

        result, entry = await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(event_id="evt_alpha", observation_level="direct"),
            ],
            partial_mode=False,
            user_input="I listen.",
        )

        assert result.final_text == "She says, 'entirely human?'"
        assert entry.assistant == "She says, 'entirely human?'"
        assistant = ckpt.narrator_conversations["alice"][-1]
        assert assistant.role == "assistant"
        assert isinstance(assistant.content, list)
        stored = json.loads(assistant.content[0]["text"])
        assert stored["final_text"] == "She says, 'entirely human?'"

    @pytest.mark.asyncio
    async def test_player_legible_public_context_is_in_system_prefix(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        ckpt.session.character_bindings["sora_kageyama"] = "2"
        ckpt.characters.append(CharacterRecord(
            character_id="sora_kageyama",
            name="Sora Kageyama",
            public_sheet=PublicSheet(
                role="Hero of the Realm; private authorial role should not leak",
                appearance=(
                    "Japanese, tall, quick posture. Wears the Crown's "
                    "blue Hero livery over a close-fitting white shirt."
                ),
            ),
            descriptions=CharacterDescriptions(
                public=(
                    "Sora is the cohort's informal leader; his blue "
                    "sun-crest tabard marks Crown Hero livery."
                ),
                private="Sora is also quietly watching the defective summon.",
            ),
            location="gatehouse",
        ))
        ckpt.canonical_events.append(EventRouterOutput(
            event_id="evt_sora",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[
                    ObservableFact.all(
                        "sora_kageyama adjusts the blue tabard."
                    ),
                ],
            ),
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    response_priority=3,
                ),
            ],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="",
            spawn=[],
            dormant=[],
            cull=[],
        ))

        await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(
                    event_id="evt_sora", observation_level="direct",
                ),
            ],
            partial_mode=False,
        )

        messages = mock_client.complete.call_args.kwargs["messages"]
        system_content = messages[0]["content"]
        user_content = messages[-1]["content"]
        flat = "\n".join(
            m["content"] for m in messages
            if isinstance(m.get("content"), str)
        )

        assert "Public character context for brief glosses" in system_content
        assert "Sora Kageyama: Sora is the cohort's informal leader" in system_content
        assert "blue sun-crest tabard" in system_content
        assert "quietly watching" not in system_content
        assert "private authorial role" not in system_content
        assert "Crown's blue Hero livery" not in flat

        assert "Gloss candidates for this passage" in user_content
        assert "- Sora Kageyama" in user_content
        assert "cohort's informal leader" not in user_content
        assert "blue sun-crest tabard" not in user_content
        assert "private authorial role" not in user_content

    @pytest.mark.asyncio
    async def test_public_context_does_not_use_raw_sheet_or_private_description(
        self, mock_client, prompt_manager,
    ):
        ckpt = _ckpt()
        ckpt.characters.append(CharacterRecord(
            character_id="korva_sahl",
            name="Korva Sahl",
            public_sheet=PublicSheet(
                role="quartermaster; privately the demon heir",
                appearance=(
                    "Plain travel leathers. Hidden horns are tucked under "
                    "her hair."
                ),
                faction="Public Guild. Private demonic court.",
            ),
            descriptions=CharacterDescriptions(
                public="Korva is an S-rank Guild adventurer usually found near the contract board.",
                private="Korva is the Demon Lord's daughter with hidden horns.",
            ),
            location="gatehouse",
        ))
        ckpt.canonical_events.append(EventRouterOutput(
            event_id="evt_korva",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[
                    ObservableFact.all(
                        "korva_sahl stands near the notice board."
                    ),
                ],
            ),
            observers=[
                ObserverEntry(
                    character_id="alice",
                    observation_level="d",
                    response_priority=3,
                ),
            ],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="",
            spawn=[],
            dormant=[],
            cull=[],
        ))

        await compose_pov_render(
            client=mock_client,
            prompt_mgr=prompt_manager,
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(
                    event_id="evt_korva", observation_level="direct",
                ),
            ],
            partial_mode=False,
        )

        messages = mock_client.complete.call_args.kwargs["messages"]
        flat = "\n".join(
            m["content"] for m in messages
            if isinstance(m.get("content"), str)
        )
        assert "Korva is an S-rank Guild adventurer" in flat
        assert "quartermaster" not in flat
        assert "privately the demon heir" not in flat
        assert "Plain travel leathers" not in flat
        assert "Hidden horns" not in flat
        assert "demonic court" not in flat
        assert "Demon Lord's daughter" not in flat

    @pytest.mark.asyncio
    async def test_partial_mode_includes_stop_instruction(
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
        # The stop-before-resolution instruction lives in the volatile
        # user message, not the cached system prefix.
        last = messages[-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], str)
        assert PARTIAL_MODE_MARKER in last["content"]
        assert PARTIAL_MODE_MARKER not in messages[0]["content"]

        # The stored history does not replay redundant user packets.
        alice_hist = ckpt.narrator_conversations["alice"]
        assert len(alice_hist) == 1
        assert alice_hist[0].role == "assistant"
        assert PARTIAL_MODE_MARKER not in json.dumps(alice_hist[0].content)

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
        user_msg = call_kwargs["messages"][-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], str)
        assert PARTIAL_MODE_MARKER not in user_msg["content"]
        # And the POV's stored history records only assistant output.
        alice_hist = ckpt.narrator_conversations["alice"]
        assert len(alice_hist) == 1
        assert alice_hist[0].role == "assistant"
        assert PARTIAL_MODE_MARKER not in json.dumps(alice_hist[0].content)

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
        # And the real event's visible details should still have been rendered.
        call_kwargs = mock_client.complete.call_args.kwargs
        flat = "\n".join(
            m["content"] for m in call_kwargs["messages"]
            if isinstance(m.get("content"), str)
        )
        assert "The arch is weathered" in flat
        assert "evt_alpha" not in flat
        assert "evt_ghost" not in flat


class TestFormatVisibleEventsBlock:
    """The narrator reads only visible surface details from each event.
    Audit/framing fields are not part of the render input."""

    def _resolved(
        self, *, event_id: str, outcome: str, facts: list[object],
        level: str = "direct", observers: list[str] | None = None,
        duration_s: int = 0,
    ):
        ev = EventRouterOutput(
            event_id=event_id,
            effective_at_s=0,
            duration_s=duration_s,
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True,
                    resolved_outcome=outcome,
                ),
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
            spawn=[], dormant=[], cull=[],
        )
        entry = RenderBufferEntry(
            event_id=event_id,
            observation_level=level,
        )
        return [(entry, ev)]

    def test_facts_surface_audit_fields_do_not(self):
        from app.engine.narrator import _format_visible_events_block

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
        out = _format_visible_events_block(resolved)
        assert "Seen directly:" in out
        assert "wings draw tight" in out
        assert "The plague that fell" in out
        # Audit line must NOT appear — that's the whole point.
        assert "resolved_outcome:" not in out
        assert "what she is permitted" not in out
        assert "the strain of speaking" not in out

    def test_empty_facts_renders_none_marker(self):
        from app.engine.narrator import _format_visible_events_block

        resolved = self._resolved(
            event_id="evt_y",
            outcome="(audit-only)",
            facts=[],
        )
        out = _format_visible_events_block(resolved)
        assert "Nothing concrete is visible." in out
        assert "(audit-only)" not in out

    def test_loadout_tags_are_removed_before_narrator_sees_them(self):
        from app.engine.narrator import _format_visible_events_block

        resolved = self._resolved(
            event_id="evt_loadout",
            outcome="Dan looks over the room.",
            facts=[
                "[loadout — Pip] Pip wears a red coat.",
                "[loadout - Vex] Vex keeps a hand on the doorframe.",
            ],
        )
        out = _format_visible_events_block(resolved)
        assert "Pip wears a red coat." in out
        assert "Vex keeps a hand on the doorframe." in out
        assert "[loadout" not in out

    def test_router_ids_render_as_names_for_narrator(self):
        from app.engine.narrator import _format_visible_events_block

        ckpt = _ckpt()
        resolved = self._resolved(
            event_id="evt_ids",
            outcome="alice passes pip.",
            facts=["alice sets pip's ledger on the table."],
        )

        out = _format_visible_events_block(resolved, ckpt=ckpt)

        assert "Alice sets Pip's ledger on the table." in out
        assert "alice sets pip" not in out

    def test_scoped_facts_filter_by_pov_before_narrator_sees_them(self):
        from app.engine.narrator import _format_visible_events_block
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

        as_ashara = _format_visible_events_block(resolved, "ashara")
        as_aldric = _format_visible_events_block(resolved, "aldric")

        assert "foot touches Ashara's boot" in as_ashara
        assert "knows curses" in as_ashara
        assert "foot touches Ashara's boot" not in as_aldric
        assert "knows curses" in as_aldric

    def test_resolved_buffers_sort_by_visible_time(self):
        from app.engine.narrator import _resolve_buffered_events

        ckpt = _ckpt()
        buffered = [
            RenderBufferEntry(
                event_id="evt_alpha",
                observation_level="direct",
                visible_at_s=20,
                event_sequence=0,
            ),
            RenderBufferEntry(
                event_id="evt_beta",
                observation_level="direct",
                visible_at_s=10,
                event_sequence=1,
            ),
        ]

        resolved = _resolve_buffered_events(ckpt, buffered)

        assert [event.event_id for _, event in resolved] == [
            "evt_beta",
            "evt_alpha",
        ]

    def test_visible_facts_sort_by_fact_time(self):
        from app.engine.narrator import _format_visible_events_block

        resolved = self._resolved(
            event_id="evt_timed",
            outcome="(audit-only)",
            facts=[
                ObservableFact.all("Second visible beat.", at_offset_s=5),
                ObservableFact.all("First visible beat.", at_offset_s=1),
            ],
            duration_s=10,
        )

        out = _format_visible_events_block(resolved)

        assert out.index("First visible beat.") < out.index("Second visible beat.")

# NOTE: The `TestOpeningVerbatimRender` class lived here in v8 and earlier.
# It exercised a now-removed verbatim shortcut that rendered the importer's
# `opening_narrative` byte-for-byte on the first `(begin)` turn. The whole
# `opening_narrative` field was removed in v9 — the router now composes the
# opening dynamically from world_state and the narrator renders it like any
# other turn — so the verbatim path, its gates, and these tests are obsolete.
# The dynamic opening flow is covered by router/narrator turn-loop tests.
