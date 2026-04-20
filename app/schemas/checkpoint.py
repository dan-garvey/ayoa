from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.characters import CharacterRecord
from app.schemas.conversation import ConversationMessage
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


class CheckpointFile(BaseModel):
    schema_version: str = "2.0"
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
    narrator_conversation: list[ConversationMessage] = Field(default_factory=list)
    character_conversations: dict[str, list[ConversationMessage]] = Field(default_factory=dict)
    # Display/audit log only — no longer fed into prompts.
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    visibility_log: list[dict[str, Any]] = Field(default_factory=list)
    config: SessionConfig = Field(default_factory=SessionConfig)
    prompt_versions: dict[str, str] = Field(default_factory=lambda: {
        "event_router": "v5",
        "agent": "v7",
        "narrator_phase2": "v5",
    })
