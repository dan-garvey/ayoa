"""Tests for streaming and transcript compression (Stage 13)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.engine.narrator import (
    Narrator,
    TRANSCRIPT_COMPRESS_THRESHOLD,
    TRANSCRIPT_KEEP_RECENT,
    TranscriptSummary,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient, LLMResponse
from app.schemas.narrator import NarratorFinalOutput, TranscriptEntry
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import LocationState, SessionState, WorldState


# --- Fixtures ---

@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock()
    return client


@pytest.fixture
def prompt_manager():
    return PromptManager("app/prompts")


@pytest.fixture
def checkpoint_with_long_transcript():
    entries = [
        TranscriptEntry(
            user=f"Turn {i}: I do something.",
            assistant=f"Turn {i}: Something happens.",
        )
        for i in range(25)
    ]
    return CheckpointFile(
        session=SessionState(session_id="test", turn_index=25),
        world_state=WorldState(
            locations=LocationState(current_scene_id="courtyard"),
        ),
        transcript=entries,
    )


@pytest.fixture
def checkpoint_with_short_transcript():
    entries = [
        TranscriptEntry(
            user=f"Turn {i}: I do something.",
            assistant=f"Turn {i}: Something happens.",
        )
        for i in range(5)
    ]
    return CheckpointFile(
        session=SessionState(session_id="test", turn_index=5),
        world_state=WorldState(
            locations=LocationState(current_scene_id="courtyard"),
        ),
        transcript=entries,
    )


# --- Transcript Compression Tests ---

class TestTranscriptCompression:
    @pytest.mark.asyncio
    async def test_no_compression_below_threshold(
        self, mock_client, prompt_manager, checkpoint_with_short_transcript
    ):
        narrator = Narrator(mock_client, prompt_manager)
        result = await narrator.compress_transcript(checkpoint_with_short_transcript)
        assert result is False
        assert len(checkpoint_with_short_transcript.transcript) == 5
        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_compression_above_threshold(
        self, mock_client, prompt_manager, checkpoint_with_long_transcript
    ):
        summary = TranscriptSummary(
            summary="The player explored the courtyard over many turns.",
            key_facts=["Player discovered the hidden path."],
        )
        mock_client.complete.return_value = LLMResponse(parsed=summary)

        narrator = Narrator(mock_client, prompt_manager)
        result = await narrator.compress_transcript(checkpoint_with_long_transcript)

        assert result is True
        # Should have 1 summary + KEEP_RECENT entries
        assert len(checkpoint_with_long_transcript.transcript) == 1 + TRANSCRIPT_KEEP_RECENT
        # First entry should be the summary
        assert checkpoint_with_long_transcript.transcript[0].user == "[TRANSCRIPT SUMMARY]"
        assert "explored" in checkpoint_with_long_transcript.transcript[0].assistant
        # Key facts should be added to world state
        assert "Player discovered the hidden path." in checkpoint_with_long_transcript.world_state.facts

    @pytest.mark.asyncio
    async def test_compression_preserves_recent_entries(
        self, mock_client, prompt_manager, checkpoint_with_long_transcript
    ):
        summary = TranscriptSummary(summary="Summary.")
        mock_client.complete.return_value = LLMResponse(parsed=summary)

        narrator = Narrator(mock_client, prompt_manager)
        # Store the last KEEP_RECENT entries for comparison
        expected_recent = checkpoint_with_long_transcript.transcript[-TRANSCRIPT_KEEP_RECENT:]

        await narrator.compress_transcript(checkpoint_with_long_transcript)

        # Recent entries should be preserved exactly
        actual_recent = checkpoint_with_long_transcript.transcript[1:]
        assert len(actual_recent) == TRANSCRIPT_KEEP_RECENT
        for expected, actual in zip(expected_recent, actual_recent):
            assert expected.user == actual.user
            assert expected.assistant == actual.assistant

    @pytest.mark.asyncio
    async def test_compression_threshold_value(self):
        """Verify the threshold is reasonable."""
        assert TRANSCRIPT_COMPRESS_THRESHOLD == 20
        assert TRANSCRIPT_KEEP_RECENT == 5
        assert TRANSCRIPT_KEEP_RECENT < TRANSCRIPT_COMPRESS_THRESHOLD


# --- SSE Event Format Tests ---

class TestSSEFormat:
    def test_sse_event_format(self):
        from app.api.story_routes import _sse_event
        result = _sse_event("status", {"phase": "started"})
        assert result.startswith("event: status\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_sse_event_json(self):
        import json
        from app.api.story_routes import _sse_event
        result = _sse_event("chunk", {"text": "hello"})
        # Extract data line
        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        data = json.loads(data_line[6:])
        assert data["text"] == "hello"
