from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.characters import CharacterRecord
from app.schemas.conversation import ConversationMessage
from app.schemas.event_router import EventRouterOutput
from app.schemas.narrator import TranscriptEntry
from app.schemas.state import SessionConfig, SessionState, WorldState


class ImportAnalysis(BaseModel):
    """Post-import preservation report — how much of the source master
    prompt actually survived into the checkpoint. Runs as a separate LLM
    pass (background on the bot path, inline on the CLI path) and gets
    patched into the checkpoint when it completes.
    """
    # Deterministic metrics
    source_chars: int = 0
    source_words: int = 0
    output_chars: int = 0       # sum across lore, facts, narrative_rules,
                                # per-character sheets, opening, envelopes
    output_words: int = 0
    duration_s: float = 0.0
    model: str = ""

    # LLM assessment
    coverage_rating: Literal["high", "medium", "low", "unknown"] = "unknown"
    dropped_topics: list[str] = Field(default_factory=list)
    compressed_topics: list[str] = Field(default_factory=list)
    preservation_notes: str = ""


CURRENT_SCHEMA_VERSION = "3.0"


class CheckpointFile(BaseModel):
    # v11 schema is 3.0 (turn pipeline rewrite). v2.0 checkpoints are
    # HARD-BREAK on load — the loader in checkpoint_manager raises with a
    # message pointing the user at /story start. No migration shim.
    schema_version: str = CURRENT_SCHEMA_VERSION
    # Stamped by the importer at build time (see IMPORTER_VERSION in
    # story_importer.py). Empty on checkpoints produced before versioning
    # was introduced — treat those as "v0" for diagnostic purposes.
    importer_version: str = ""
    # Preservation audit of the import. Populated asynchronously on the
    # bot path (fire-and-forget background task patches the checkpoint
    # when done), or inline on the CLI path. None until the analysis
    # completes successfully.
    import_analysis: ImportAnalysis | None = None
    session: SessionState
    opening_narrative: str = ""
    world_state: WorldState = Field(default_factory=WorldState)
    characters: list[CharacterRecord] = Field(default_factory=list)
    # Rolling conversation histories: each role sees the full prior exchange
    # on every call, so continuity and caching both work.
    session_conversation: list[ConversationMessage] = Field(default_factory=list)
    # v11: per-character narrator rolling history. Each human (by their
    # bound character_id) has their own stream; the narrator_phase2 call
    # reads character_id-keyed history. The old session-wide
    # narrator_conversation is gone in the v11 pipeline.
    narrator_conversations: dict[str, list[ConversationMessage]] = Field(default_factory=dict)
    character_conversations: dict[str, list[ConversationMessage]] = Field(default_factory=dict)
    # v11: the canonical event log. Every closed canonical event appended
    # here. Source of truth for rendering, recap, replay, and debug. The
    # router's recap pass (Haiku summarizer) produces compact summaries
    # over a tail of this log.
    canonical_events: list[EventRouterOutput] = Field(default_factory=list)
    # Display/audit log only — no longer fed into prompts.
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    visibility_log: list[dict[str, Any]] = Field(default_factory=list)
    config: SessionConfig = Field(default_factory=SessionConfig)
    prompt_versions: dict[str, str] = Field(default_factory=lambda: {
        "event_router": "v9",
        "agent": "v8",
        "narrator_phase2": "v8",
    })
