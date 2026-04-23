"""Tests for the turn-recap summarizer and pending_recap buffer flow."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.prompt_manager import PromptManager
from app.engine.turn_recap import _TurnRecapOutput, summarize_turn
from app.schemas.events import (
    CanonicalEvent,
    SceneDelta,
    WorldAdjudication,
)

def _canonical() -> CanonicalEvent:
    return CanonicalEvent(
        world_adjudication=WorldAdjudication(
            attempted_action="survey the room",
            feasible=True,
            resolved_outcome="The player surveys the room.",
        ),
        scene_delta=SceneDelta(time_advanced_seconds=0),
        observable_facts=[],
    )

def _llm_response(parsed):
    from app.llm.client import LLMResponse
    raw = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "{}"
    text_block.model_dump = lambda: {"type": "text", "text": "{}"}
    raw.content = [text_block]
    raw.model = "claude-haiku-4-5"
    return LLMResponse(parsed=parsed, raw_response=raw, content="{}", model="claude-haiku-4-5")

class TestSummarizeTurn:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_happy_path_returns_note(self):
        client = MagicMock()
        client.config = MagicMock()
        client.config.model_for_role = MagicMock(return_value="claude-haiku-4-5")
        client.complete = AsyncMock(return_value=_llm_response(
            _TurnRecapOutput(note="Ashara stepped into the doorway."),
        ))
        pm = PromptManager("app/prompts")

        note = self._run(summarize_turn(
            client, pm, _canonical(),
            narrator_final_text="...scene prose...",
        ))
        assert note == "Ashara stepped into the doorway."
        client.complete.assert_awaited_once()

    def test_empty_note_returned_empty(self):
        """Summarizer may decide the narrator added nothing state-level.
        Callers get an empty string; that's a legitimate outcome."""
        client = MagicMock()
        client.config = MagicMock()
        client.config.model_for_role = MagicMock(return_value="claude-haiku-4-5")
        client.complete = AsyncMock(return_value=_llm_response(
            _TurnRecapOutput(note=""),
        ))
        pm = PromptManager("app/prompts")
        note = self._run(summarize_turn(client, pm, _canonical(), "prose"))
        assert note == ""

    def test_whitespace_stripped(self):
        client = MagicMock()
        client.config = MagicMock()
        client.config.model_for_role = MagicMock(return_value="claude-haiku-4-5")
        client.complete = AsyncMock(return_value=_llm_response(
            _TurnRecapOutput(note="  Note with padding.  \n"),
        ))
        pm = PromptManager("app/prompts")
        note = self._run(summarize_turn(client, pm, _canonical(), "prose"))
        assert note == "Note with padding."

    def test_uses_summarizer_role(self):
        """The summarizer should dispatch under role='summarizer' so the
        LLMClient picks Haiku via config.model_for_role."""
        client = MagicMock()
        client.config = MagicMock()
        client.config.model_for_role = MagicMock(return_value="claude-haiku-4-5")
        client.complete = AsyncMock(return_value=_llm_response(
            _TurnRecapOutput(note="ok"),
        ))
        pm = PromptManager("app/prompts")

        self._run(summarize_turn(client, pm, _canonical(), "prose"))
        call_kwargs = client.complete.await_args.kwargs
        assert call_kwargs["role"] == "summarizer"

@pytest.mark.skip(reason="v11: legacy v8 pipeline path; re-port against run_beat.")
class TestRouterConsumesPendingRecap:
    """event_router builds the recent_turn_recap block from
    checkpoint.session.pending_recap and clears the buffer after
    appending to session_conversation."""

    def test_populated_recap_renders_in_block(self):
        """When pending_recap is set, the router's user message contains
        the recap text under a clear header."""
        from app.engine.prompt_manager import PromptManager
        pm = PromptManager("app/prompts")
        rendered = pm.render(
            "event_router",
            setting_summary="x", world_lore="x", world_rules="x",
            scene_graph="x", current_scene="x", characters_present="x",
            hidden_lore="x", hidden_facts="x",
            user_input="next",
            acting_character_name="Aldric",
            acting_character_id="aldric",
            player_characters_block="x",
            since_last_turn_block="",
            opening_directive="",
            world_facts_delta_block="",
            initial_roster_block="",
            state_changes_block="",
            cat_ii_resolution_block="",
            intention_block="## Intention\nAldric attempts: next",
            recent_turn_recap=(
                "## Previous Turn — Narrator Delta Note\n"
                "- Ashara stepped into the doorway.\n\n"
            ),
        )
        assert "Previous Turn — Narrator Delta Note" in rendered
        assert "Ashara stepped into the doorway" in rendered

    def test_empty_recap_omits_block(self):
        from app.engine.prompt_manager import PromptManager
        pm = PromptManager("app/prompts")
        rendered = pm.render(
            "event_router",
            setting_summary="x", world_lore="x", world_rules="x",
            scene_graph="x", current_scene="x", characters_present="x",
            hidden_lore="x", hidden_facts="x",
            user_input="next",
            acting_character_name="Aldric",
            acting_character_id="aldric",
            player_characters_block="x",
            since_last_turn_block="",
            opening_directive="",
            world_facts_delta_block="",
            initial_roster_block="",
            state_changes_block="",
            cat_ii_resolution_block="",
            intention_block="## Intention\nAldric attempts: next",
            recent_turn_recap="",
        )
        assert "Previous Turn — Narrator Delta Note" not in rendered

class TestPendingRecapField:
    def test_defaults_empty(self):
        from app.schemas.state import SessionState
        s = SessionState(session_id="x")
        assert s.pending_recap == ""

    def test_can_be_set_and_cleared(self):
        from app.schemas.state import SessionState
        s = SessionState(session_id="x")
        s.pending_recap = "note"
        assert s.pending_recap == "note"
        s.pending_recap = ""
        assert s.pending_recap == ""
