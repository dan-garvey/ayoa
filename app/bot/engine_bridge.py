"""Thin async wrapper around the engine for the Discord bot.

Responsibilities:
- Holds the shared Orchestrator, LLMClient, CheckpointManager, PromptManager.
- Creates a fresh session from an imported story (copies ckpt_0000 into the
  session dir; no auto-binding — players pick a character via /join).
- Runs turns behind a per-session asyncio.Lock so concurrent /act commands
  on the same channel serialize cleanly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from app.engine.character_agent import _extract_parenthetical
from app.engine.character_manager import _normalize_router_summary
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.model_config_sync import sync_checkpoint_runtime_models
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.engine.settings import (
    SETTINGS_BY_KEY,
    get_setting,
    list_settings_view,
    set_setting,
)
from app.engine.story_importer import (
    run_import_two_call,
    run_preservation_analysis_continuation,
)
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.characters import CharacterRecord, CharacterStatus, PublicSheet
from app.schemas.checkpoint import CheckpointFile, ImportAnalysis
from app.schemas.requests import TurnRequest
from app.schemas.responses import TurnResponse

if TYPE_CHECKING:
    from app.schemas.query import QueryResponse


@dataclass(frozen=True)
class CharacterSummary:
    """Spoiler-free summary of a character for /story characters and /join.

    Public-sheet fields plus status / name / binding — nothing from
    private_state, backstory, personality, or hidden lore. `bound_user_id`
    is populated when reading a session checkpoint; empty on pristine
    story-level lookups. `is_playable` mirrors the runtime
    `CharacterRecord.is_playable` flag (renamed from `is_player` in the
    playable-2 commit; see schema docstring).
    """
    character_id: str
    name: str
    role: str
    faction: str
    appearance: str
    status: str  # "active" | "dormant" | "culled"
    is_playable: bool
    bound_user_id: str = ""


@dataclass(frozen=True)
class RewindResult:
    """Result of an EngineBridge.rewind_session call.

    `target_turn` is the checkpoint id the session is now anchored to
    (`ckpt_<target_turn>.json` is the new latest). `previous_latest` and
    `new_latest` describe the before/after state for the confirmation
    embed. `deleted_turns` is the list of turn indices whose checkpoints
    were culled — usually contiguous from target_turn+1 to previous_latest
    inclusive, but the rewind is robust to gaps (an old corrupted save
    that was already missing simply doesn't appear in the list).

    `scene_id` and `actor_character_id` are recovered from the loaded
    checkpoint to help the frontend render a "you are now at <scene>"
    line. Both may be empty if the session has no bound actor (e.g. a
    fresh session before any /join).
    """
    session_id: str
    target_turn: int
    previous_latest: int
    new_latest: int
    deleted_turns: list[int]
    scene_id: str = ""
    actor_character_id: str = ""


logger = logging.getLogger(__name__)


class EngineBridge:
    """Shared engine state for all Discord interactions.

    Stories (imported master prompts, pristine ckpt_0000 only) live under
    `stories_dir`. Player sessions (one dir per session_id, ckpt_NNNN
    grows with turn_index) live under `sessions_dir`. Keeping them in
    separate namespaces means creating a new session from a story
    doesn't drag the story's `import_analysis` along — that stays on
    the story's canonical source.

    `saves_dir` is accepted for backward compatibility; when set, it
    becomes the parent of both stories_dir and sessions_dir unless those
    are specified explicitly. Legacy flat layouts at
    `app/storage/saves/` are auto-migrated on construction.
    """

    def __init__(
        self,
        *,
        stories_dir: str | None = None,
        sessions_dir: str | None = None,
        saves_dir: str | None = None,
        prompts_dir: str = "app/prompts",
        llm_config: LLMConfig | None = None,
    ):
        # `saves_dir` is a convenience for tests and backward-compat: when
        # provided, stories and sessions live under subdirs of it unless
        # overridden. New code should pass stories_dir / sessions_dir
        # directly.
        if saves_dir is not None:
            base = Path(saves_dir)
            self.stories_dir = Path(stories_dir) if stories_dir else base / "stories"
            self.sessions_dir = Path(sessions_dir) if sessions_dir else base / "sessions"
        else:
            self.stories_dir = Path(stories_dir or "app/storage/stories")
            self.sessions_dir = Path(sessions_dir or "app/storage/sessions")
        self.stories_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.client = LLMClient(config=llm_config or LLMConfig.from_env())
        self.checkpoint_mgr = CheckpointManager(save_dir=str(self.sessions_dir))
        self.prompt_mgr = PromptManager(prompts_dir=prompts_dir)
        self.orchestrator = Orchestrator(
            self.client, self.checkpoint_mgr, self.prompt_mgr
        )
        # One lock per session_id; created lazily.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_mutex = asyncio.Lock()

    async def close(self) -> None:
        await self.client.close()

    # ---- session lifecycle ---------------------------------------------------

    def list_story_ids(self) -> list[str]:
        """Return available story IDs — directories under stories_dir that
        contain a ckpt_0000.json."""
        if not self.stories_dir.exists():
            return []
        return sorted(
            child.name for child in self.stories_dir.iterdir()
            if child.is_dir() and (child / "ckpt_0000.json").exists()
        )

    def load_story_ckpt(self, story_id: str) -> CheckpointFile:
        """Load a story's source ckpt_0000 (pristine, not a session checkpoint)."""
        path = self.stories_dir / story_id / "ckpt_0000.json"
        if not path.exists():
            raise FileNotFoundError(f"Story '{story_id}' not found at {path}")
        return CheckpointFile.model_validate_json(path.read_text())

    async def import_story(
        self,
        source_text: str,
        story_id: str,
        on_analysis_complete: Callable[
            [ImportAnalysis | None, Exception | None], Awaitable[None]
        ] | None = None,
    ) -> CheckpointFile:
        """Run the import pipeline and save the resulting ckpt_0000.json
        under saves/<story_id>/. Fires preservation analysis as a
        background task that patches the checkpoint when it completes.

        `on_analysis_complete`, if provided, is awaited after the analysis
        pass finishes (or fails). The callback receives `(analysis, error)`
        where exactly one is non-None. The bot uses this to DM the user
        who ran /story import with the coverage outcome — close the loop
        so the user knows analysis finished without polling the file.

        Refuses to overwrite an existing story — caller should delete first
        or pick a different story_id. Raises FileExistsError in that case.
        """
        dst_dir = self.stories_dir / story_id
        dst_ckpt = dst_dir / "ckpt_0000.json"
        if dst_ckpt.exists():
            raise FileExistsError(
                f"Story '{story_id}' already exists at {dst_ckpt}. "
                f"Delete it first or pick a different story_id."
            )

        result = await run_import_two_call(self.client, source_text, story_id)
        checkpoint = result.checkpoint
        sync_checkpoint_runtime_models(checkpoint, self.client.config)

        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_ckpt.write_text(checkpoint.model_dump_json(indent=2))
        logger.info("Imported story %s → %s", story_id, dst_ckpt)

        # Fire preservation analysis in the background as a CONTINUATION
        # of the two-call import: the analysis call replays the full
        # Call-1 + Call-2 conversation as cached prefix (cache_user_tail
        # on each call's user turn), so the analysis only pays fresh
        # tokens for the Call-2 assistant echo + the analysis question.
        asyncio.create_task(
            self._background_preservation_analysis(
                source_text, story_id, dst_ckpt,
                priming_messages=result.priming_messages,
                assistant_text=result.assistant_text,
                on_complete=on_analysis_complete,
            )
        )
        return checkpoint

    async def _background_preservation_analysis(
        self,
        source_text: str,
        story_id: str,
        dst_ckpt: Path,
        priming_messages: list[dict[str, str]],
        assistant_text: str,
        on_complete: Callable[
            [ImportAnalysis | None, Exception | None], Awaitable[None]
        ] | None = None,
    ) -> None:
        """Run preservation analysis as a continuation of the combined-
        import call and patch the checkpoint on disk. Any analysis error
        is caught and surfaced to `on_complete` so the frontend can notify
        the user — best-effort metadata should not crash the task loop."""
        analysis: ImportAnalysis | None = None
        err: Exception | None = None
        try:
            checkpoint = CheckpointFile.model_validate_json(dst_ckpt.read_text())
            analysis = await run_preservation_analysis_continuation(
                self.client,
                priming_messages=priming_messages,
                assistant_text=assistant_text,
                source_text=source_text,
                checkpoint=checkpoint,
            )
            # Re-read before patching in case other writes happened — ckpt_0000
            # is pristine source, typically untouched after the initial write,
            # but be safe anyway.
            checkpoint = CheckpointFile.model_validate_json(dst_ckpt.read_text())
            checkpoint.import_analysis = analysis
            dst_ckpt.write_text(checkpoint.model_dump_json(indent=2))
            logger.info(
                "Preservation analysis for %s patched to disk: coverage=%s, "
                "source=%dw, output=%dw, dropped=%d, compressed=%d",
                story_id, analysis.coverage_rating,
                analysis.source_words, analysis.output_words,
                len(analysis.dropped_topics), len(analysis.compressed_topics),
            )
        except Exception as e:
            err = e
            logger.exception(
                "Preservation analysis failed for %s (non-fatal)", story_id,
            )

        if on_complete is not None:
            try:
                await on_complete(analysis, err)
            except Exception:
                logger.exception(
                    "on_analysis_complete callback raised for %s", story_id,
                )

    # Story imports are treated as permanent: delete_story is gone. Each
    # import costs real dollars and produces an artifact that's cheaper to
    # keep around than to recreate. Operators who genuinely need to remove
    # a story should rm the directory by hand.

    # ---- session primitives --------------------------------------------------

    def list_session_ids(self) -> list[str]:
        """Return directory names under sessions_dir. Each entry is a
        named save the user can resume."""
        if not self.sessions_dir.exists():
            return []
        return sorted(
            child.name for child in self.sessions_dir.iterdir()
            if child.is_dir()
        )

    def create_empty_session(self, session_id: str) -> None:
        """Create a session directory with no checkpoint. The caller
        runs /story start to load content into it afterward.
        Raises FileExistsError if the session already exists with ckpts."""
        dst = self.sessions_dir / session_id
        if dst.exists() and any(dst.glob("ckpt_*.json")):
            raise FileExistsError(
                f"Session '{session_id}' already exists with checkpoints. "
                f"Run /session resume or pick a different name."
            )
        dst.mkdir(parents=True, exist_ok=True)
        logger.info("Created empty session %s at %s", session_id, dst)

    def load_story_into_session(
        self,
        session_id: str,
        story_id: str,
    ) -> CheckpointFile:
        """Copy a story's pristine ckpt_0000 into the named session dir,
        rewriting session_id and stripping import_analysis. No
        personalize, no auto-bind — the player picks characters via
        /join after. Refuses if the session already has
        a story loaded (run /story delete first)."""
        src = self.stories_dir / story_id / "ckpt_0000.json"
        if not src.exists():
            raise FileNotFoundError(f"Story '{story_id}' not found at {src}")

        dst_dir = self.sessions_dir / session_id
        if not dst_dir.exists():
            raise FileNotFoundError(
                f"Session '{session_id}' does not exist. "
                f"Run /session start first."
            )
        if any(dst_dir.glob("ckpt_*.json")):
            raise FileExistsError(
                f"Session '{session_id}' already has a story loaded. "
                f"Run /story delete first to unload it."
            )

        data = json.loads(src.read_text())
        data["session"]["session_id"] = session_id
        data.pop("import_analysis", None)
        ckpt = CheckpointFile.model_validate(data)
        sync_checkpoint_runtime_models(ckpt, self.client.config)
        (dst_dir / "ckpt_0000.json").write_text(ckpt.model_dump_json(indent=2))
        ckpt = self.checkpoint_mgr.load(session_id, "ckpt_0000")
        if ckpt.session.turn_index == 0:
            ckpt.session.turn_index = 1
        sync_checkpoint_runtime_models(ckpt, self.client.config)
        self.checkpoint_mgr.save(ckpt)
        logger.info("Loaded story %s into session %s", story_id, session_id)
        return ckpt

    def unload_story_from_session(self, session_id: str) -> int:
        """Wipe all checkpoints from a session dir. Returns files removed.
        After this, /story start may be run again to load a different
        story. Bindings and character state are gone — the session is
        an empty container."""
        dst = self.sessions_dir / session_id
        if not dst.exists():
            raise FileNotFoundError(f"Session '{session_id}' does not exist.")
        removed = 0
        for ckpt in dst.glob("ckpt_*.json"):
            ckpt.unlink()
            removed += 1
        logger.info("Unloaded story from session %s (%d files)", session_id, removed)
        return removed

    def load_latest(self, session_id: str) -> CheckpointFile:
        return self.checkpoint_mgr.load_latest(session_id)

    def list_checkpoint_turns(self, session_id: str) -> list[int]:
        """Return the integer turn indices of every saved checkpoint for
        this session, sorted ascending. Used by the /rewind command to
        validate the user's target and show the playable range."""
        return self.checkpoint_mgr.list_turn_indices(session_id)

    def preview_rewind(
        self, session_id: str, target_turn: int,
    ) -> RewindResult:
        """Same validation as `rewind_session`, without mutating disk.
        Used by the Discord confirmation flow: show the user exactly
        what would be deleted before they click Confirm. The
        `deleted_turns` list in the returned result is the *would-be*
        deletion list; `new_latest` is what the latest WOULD be after
        the cull (i.e. == target_turn). On confirm, the frontend calls
        `rewind_session` and the actual cull happens.

        Raises the same exceptions as `rewind_session` for invalid
        inputs, so the frontend's confirmation prompt never advances
        past validation if the target is bogus.
        """
        turns = self.list_checkpoint_turns(session_id)
        if not turns:
            raise FileNotFoundError(
                f"Session '{session_id}' has no checkpoints to rewind."
            )
        latest = turns[-1]
        if target_turn < 0:
            raise ValueError(
                f"Cannot rewind to turn {target_turn}: must be >= 0."
            )
        if target_turn not in turns:
            raise ValueError(
                f"Turn {target_turn} has no checkpoint for session "
                f"'{session_id}'. Available: {turns[0]}..{latest}."
            )
        if target_turn >= latest:
            raise ValueError(
                f"Cannot rewind to turn {target_turn}: that's already the "
                f"current state (latest is turn {latest}). Pick an "
                f"earlier turn."
            )
        would_delete = [t for t in turns if t > target_turn]
        ckpt = self.checkpoint_mgr.load(session_id, f"ckpt_{target_turn:04d}")
        return RewindResult(
            session_id=session_id,
            target_turn=target_turn,
            previous_latest=latest,
            new_latest=target_turn,
            deleted_turns=would_delete,
            scene_id=self._actor_scene_after_rewind(ckpt),
            actor_character_id=ckpt.session.player_character_id,
        )

    async def rewind_session(
        self,
        session_id: str,
        target_turn: int,
    ) -> RewindResult:
        """Rewind a session to the state captured by `ckpt_<target_turn>.json`.

        Mechanism: cull every ckpt_NNNN.json with NNNN > target_turn from
        the session directory. The next call to load_latest then returns
        the target checkpoint, and the orchestrator resumes from there
        as if the deleted turns never happened. No engine-level state
        replay needed — checkpoints are atomic per-turn snapshots
        capturing canonical_events, all rolling conversations,
        world_state, every CharacterRecord (with its pending_observations
        queue), render_buffers, slot/Cat-II state, recap, and bindings.

        What this does NOT touch:

        - SessionMap (Discord channel→session, pov_threads cache). Those
          are session-stable and orthogonal to per-turn state. A user
          who joined between target_turn and now will lose their
          binding (the loaded checkpoint predates their /join), but
          their cached pov_thread row is fine — they just need to
          /join again.
        - The per-session asyncio Lock. We acquire it for the cull so
          a concurrent /act cannot save into the gap, but the lock
          object itself is ephemeral and survives.
        - LLM/Anthropic prompt cache. The first post-rewind LLM call
          will miss cache (different conversation prefix); subsequent
          calls re-warm normally. Cost, not correctness.

        Validation:
        - Target must be >= 0 and present in `list_checkpoint_turns`.
        - Target must be < the current latest (refusing to "rewind to
          the present" prevents a noisy no-op and a stray confirmation
          on the user side).
        - If the session has no checkpoints at all, raise FileNotFoundError.

        Returns a `RewindResult` with the deleted turn list and a
        human-readable summary suitable for confirmation embeds.
        """
        turns = self.list_checkpoint_turns(session_id)
        if not turns:
            raise FileNotFoundError(
                f"Session '{session_id}' has no checkpoints to rewind."
            )
        latest = turns[-1]
        if target_turn < 0:
            raise ValueError(
                f"Cannot rewind to turn {target_turn}: must be >= 0."
            )
        if target_turn not in turns:
            raise ValueError(
                f"Turn {target_turn} has no checkpoint for session "
                f"'{session_id}'. Available: {turns[0]}..{latest}."
            )
        if target_turn >= latest:
            raise ValueError(
                f"Cannot rewind to turn {target_turn}: that's already the "
                f"current state (latest is turn {latest}). Pick an "
                f"earlier turn."
            )

        lock = await self._lock_for(session_id)
        async with lock:
            # Re-check inside the lock — a concurrent /act may have
            # advanced past our snapshot's `latest` between read and cull.
            # Re-validating against the current latest catches the
            # rare race where rewind and /act arrive simultaneously.
            current_turns = self.list_checkpoint_turns(session_id)
            if target_turn not in current_turns:
                raise ValueError(
                    f"Turn {target_turn} is no longer available "
                    f"(checkpoint disappeared between validation and cull)."
                )
            current_latest = current_turns[-1]
            if target_turn >= current_latest:
                raise ValueError(
                    f"Cannot rewind to turn {target_turn}: latest is now "
                    f"{current_latest}."
                )
            deleted = self.checkpoint_mgr.delete_checkpoints_after(
                session_id, target_turn,
            )
            new_latest = self.list_checkpoint_turns(session_id)[-1]

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        return RewindResult(
            session_id=session_id,
            target_turn=target_turn,
            previous_latest=current_latest,
            new_latest=new_latest,
            deleted_turns=deleted,
            scene_id=self._actor_scene_after_rewind(ckpt),
            actor_character_id=ckpt.session.player_character_id,
        )

    def _actor_scene_after_rewind(self, ckpt: CheckpointFile) -> str:
        """Best-effort: return the scene id of the bound actor after
        rewind, so the confirmation embed can tell the user where they
        landed. Empty string if we can't resolve it (no bound character,
        unknown character_id) — the embed just omits the location line."""
        cid = ckpt.session.player_character_id
        if not cid:
            return ""
        for c in ckpt.characters:
            if c.character_id == cid:
                return c.location or ""
        return ""

    def set_character_identity(
        self,
        session_id: str,
        character_id: str,
        *,
        name: str | None = None,
        appearance: str | None = None,
    ) -> CheckpointFile:
        """Update a character's name and/or appearance. Used by /describe
        after takeover so the player's name and look land on the record
        without touching personality (which they'll fill through play,
        or leave blank for agent handoff)."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        target = next(
            (c for c in ckpt.characters if c.character_id == character_id), None
        )
        if target is None:
            raise ValueError(
                f"Character '{character_id}' not found in session '{session_id}'"
            )
        if name is not None and name.strip():
            target.name = name.strip()
        if appearance is not None:
            target.public_sheet.appearance = appearance.strip()
        self.checkpoint_mgr.save(ckpt)
        return ckpt

    async def leave_character(
        self,
        session_id: str,
        user_id: int,
    ) -> str | None:
        """Unified leave endpoint shared by CLI and Discord.

        Synthesizes `personality` from the rolling conversation if the
        character's own personality field is empty (fresh takeover by
        the player who never wrote one), then unbinds the user. The
        synthesize step lets an agent pick the character up with
        voice intact; no-op when personality is already set.

        Returns the freed character_id, or None if the user had no
        binding. Synthesis errors are logged and swallowed — don't
        block the unbind on a synthesis failure.
        """
        binding = self.get_user_binding(session_id, user_id)
        if binding:
            try:
                await self.synthesize_personality(session_id, binding)
            except Exception:
                logger.exception(
                    "personality synthesis failed for %s; unbinding anyway",
                    binding,
                )
        return self.unbind_user(session_id, user_id)

    async def synthesize_personality(
        self,
        session_id: str,
        character_id: str,
    ) -> CheckpointFile:
        """Fill a character's `personality` field by asking the narrator to
        synthesize one from the rolling conversation history. Called on
        /leave when a player hands a character back to the agent and
        personality is still empty (player never wrote it, or played
        entirely through the rolling conversation).

        No-op if personality is already non-empty. Uses the narrator
        role (Sonnet) because it has the best grasp of prose voice and
        character continuity across the transcript.
        """
        from pydantic import BaseModel

        class _PersonalityOutput(BaseModel):
            personality: str

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        target = next(
            (c for c in ckpt.characters if c.character_id == character_id), None,
        )
        if target is None:
            raise ValueError(f"No character '{character_id}' in session.")
        if target.personality and target.personality.strip():
            logger.info(
                "Personality already set on %s; skipping synthesis", character_id,
            )
            return ckpt

        # Pull this character's rolling conversation history. On a fresh
        # custom character who never had an agent turn, the history
        # is empty — fall back to synthesizing from authored fields only.
        #
        # Leak guard: assistant turns in the rolling conversation include
        # the agent's trailing "(intent)" parenthetical, which is private
        # to the agent and the engine. Personality synthesis output is
        # rendered to the player on /leave, so we strip the parenthetical
        # from every assistant snippet before handing the block to the
        # synthesizer LLM. User-role turns (router framings) don't carry
        # intent in the same shape, but they still go through the same
        # strip for safety — a trailing balanced parenthetical at the
        # very end of the snippet is dropped regardless of role.
        history = ckpt.character_conversations.get(character_id, [])
        convo_snippets = []
        for msg in history[-20:]:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if content:
                role = msg.role if hasattr(msg, "role") else msg.get("role")
                public, _intent = _extract_parenthetical(content)
                snippet = (public or "").strip()
                if snippet:
                    convo_snippets.append(f"[{role}] {snippet[:500]}")
        history_block = "\n".join(convo_snippets) or "(no rolling conversation yet)"

        messages = [
            {"role": "system", "content": (
                "<role>\n"
                "You are a characterization editor for an interactive fiction "
                "engine.\n"
                "</role>\n\n"
                "<instructions>\n"
                "Distill a character's personality into a single prose block "
                "for engine-side use. Cover three things in one paragraph (or a "
                "few): how they speak, how they carry themselves, and how to "
                "play them under pressure. Base your write-up on the character's "
                "authored identity and their prior rolling conversation if any. "
                "No bullet points. No commentary outside the JSON.\n"
                "</instructions>\n\n"
                "<output_schema>\n"
                'Respond with ONLY valid JSON: {"personality": "<prose>"}\n'
                "</output_schema>"
            )},
            {"role": "user", "content": (
                "<character_context>\n"
                f"Character: {target.name} ({character_id})\n"
                f"Role: {target.public_sheet.role}\n"
                f"Appearance: {target.public_sheet.appearance}\n"
                f"Faction: {target.public_sheet.faction}\n"
                f"Backstory: {target.backstory}\n"
                f"Known context: {target.known_context}\n"
                f"Goals: {', '.join(target.private_state.goals)}\n"
                "</character_context>\n\n"
                "<recent_conversation>\n"
                f"{history_block}\n"
                "</recent_conversation>\n\n"
                "<task>\n"
                "Write the personality JSON now.\n"
                "</task>"
            )},
        ]
        response = await self.client.complete(
            role="narrator",
            messages=messages,
            temperature=0.5,
            max_tokens=1500,
        )
        out = _parse_model_json(_PersonalityOutput, response.content)
        target.personality = out.personality.strip()
        self.checkpoint_mgr.save(ckpt)
        logger.info(
            "Synthesized personality for %s (%d chars)",
            character_id, len(target.personality),
        )
        return ckpt

    # ---- character catalog (spoiler-free) ------------------------------------

    def list_session_characters(self, session_id: str) -> list[CharacterSummary]:
        """Spoiler-free roster for the session's current checkpoint, annotated
        with binding state. Used by /story characters when a session exists."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        return _summaries_from_checkpoint(ckpt)

    def list_story_characters(self, story_id: str) -> list[CharacterSummary]:
        """Spoiler-free roster from the source ckpt_0000 (no session needed)."""
        ckpt = self.load_story_ckpt(story_id)
        return _summaries_from_checkpoint(ckpt)

    # ---- bindings ------------------------------------------------------------

    def get_user_binding(self, session_id: str, user_id: int) -> str | None:
        """Return the character_id bound to this Discord user, or None."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        uid = str(user_id)
        for char_id, bound in ckpt.session.character_bindings.items():
            if bound == uid:
                return char_id
        return None

    def bind_user(
        self,
        session_id: str,
        user_id: int,
        character_id: str,
    ) -> CheckpointFile:
        """Bind a Discord user to a roster character.

        Refuses if the user already has a different binding, if the character
        doesn't exist, if the character is culled, or if another user is
        already bound to it. Dormant characters are allowed — binding doesn't
        wake them; the fiction decides reactivation.
        """
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        uid = str(user_id)

        target = next(
            (c for c in ckpt.characters if c.character_id == character_id), None
        )
        if target is None:
            raise ValueError(f"No character '{character_id}' in this session.")
        if target.status.value == "culled":
            raise ValueError(
                f"Character '{target.name}' is no longer in the story (culled)."
            )

        existing_for_char = ckpt.session.character_bindings.get(character_id)
        if existing_for_char and existing_for_char != uid:
            raise ValueError(
                f"Character '{target.name}' is already bound to another player."
            )

        existing_for_user = next(
            (cid for cid, bound in ckpt.session.character_bindings.items()
             if bound == uid and cid != character_id),
            None,
        )
        if existing_for_user:
            raise ValueError(
                f"You are already bound to '{existing_for_user}'. "
                f"Run /leave first if you want to switch."
            )

        ckpt.session.character_bindings[character_id] = uid
        target_name = next(
            (c.name for c in ckpt.characters if c.character_id == character_id),
            character_id,
        )
        ckpt.session.pending_router_state_changes.append(
            f"Player binding: {target_name} (id: {character_id}) is now "
            f"driven by a human player. Treat them as a protagonist; "
            f"the narrator may pivot POV to them."
        )
        self.checkpoint_mgr.save(ckpt)
        return ckpt

    def unbind_user(self, session_id: str, user_id: int) -> str | None:
        """Remove this user's binding. Returns the freed character_id, or None
        if they had no binding.

        v11-A5: purges any v11 state (active_act_slots entries, open Cat II
        event responder lists/collected intentions, render buffers) the
        character held before removing the binding. Prevents stranded pins
        from freezing the scene when a player /leave's mid-beat.
        """
        from app.engine.turn_loop import purge_character_state

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        uid = str(user_id)
        freed = None
        for char_id, bound in list(ckpt.session.character_bindings.items()):
            if bound == uid:
                freed = char_id
        if freed is not None:
            # v11-r5: purge v11 state BEFORE removing the binding so no
            # reader can observe an unbound character with a stale pin.
            # Purge is a pure in-memory mutation; del + save follow
            # atomically on the same checkpoint write.
            purge_character_state(ckpt, freed)
            del ckpt.session.character_bindings[freed]
            freed_name = next(
                (c.name for c in ckpt.characters if c.character_id == freed),
                freed,
            )
            ckpt.session.pending_router_state_changes.append(
                f"Player binding: {freed_name} (id: {freed}) returned to "
                f"AI control. Their character agent will resume producing "
                f"intentions on cascade."
            )
            self.checkpoint_mgr.save(ckpt)
        return freed

    # ---- takeover -----------------------------------------------------------

    def takeover(
        self,
        session_id: str,
        character_id: str,
        user_id: int,
    ) -> CheckpointFile:
        """Plain takeover: bind the user to an existing character.

        Name, appearance, identity, everything else stays as-authored.
        This is the default `/join` path — the user becomes the
        character as the importer wrote them. Refuses if already
        claimed, culled, or nonexistent (via `bind_user`).

        Under playable-2 semantics the binding IS the takeover —
        `is_playable` is an authoring-time flag describing which slots
        a human can step into, not a runtime flag toggled by binding.
        We log a warning if the importer didn't mark this character
        playable but a player is taking them over anyway (most likely
        a custom override via /character path); the binding still
        applies because explicit user intent wins.
        """
        ckpt = self.bind_user(session_id, user_id, character_id)
        target = next(
            (c for c in ckpt.characters if c.character_id == character_id), None,
        )
        if target is None:
            raise ValueError(f"No character '{character_id}' in this session.")
        if not target.is_playable:
            logger.warning(
                "takeover: %s (%s) was not marked is_playable=true at import "
                "time, but user %s bound to them anyway. Binding stands; the "
                "importer probably should have flagged them playable.",
                target.name, character_id, user_id,
            )
        return ckpt

    async def create_custom_character(
        self,
        session_id: str,
        user_id: int,
        description: str,
    ) -> CharacterRecord:
        """Mode='describe': router authors a full new character from the
        player's concept, lands them in the world, binds to the user. The
        returned record has its engine-assigned character_id,
        is_playable=True, and is already written to the checkpoint."""
        from app.schemas.takeover import TakeoverAuthoredOutput

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        out: TakeoverAuthoredOutput = await self._call_takeover(
            ckpt,
            mode="describe",
            description=description,
            invoking_user_id=str(user_id),
        )
        new_id = _pick_unused_character_id(ckpt, out.character.name)
        new_char = out.character.to_record(character_id=new_id)
        new_char.is_playable = True
        ckpt.characters.append(new_char)
        ckpt.session.character_bindings[new_id] = str(user_id)
        # Prefer the LLM-authored router_summary (it has full omniscient
        # context). Fall back to a mechanical line if missing — the
        # describe prompt is supposed to always emit a non-empty
        # summary, but the engine should never silently swallow a
        # spawn just because the LLM regressed. The summary is
        # normalized before interpolation (whitespace collapsed,
        # over-long entries truncated) using the same helper the spawn
        # path uses, so an LLM that drops a multi-paragraph backstory
        # into the field can't shatter the next router prompt.
        summary = _normalize_router_summary(out.character.router_summary or "")
        if summary:
            # Engine-side player-bound tag is a tight, repeatable
            # signal; the takeover prompt's "describe" mode does NOT
            # need to embed protagonist framing in the LLM-authored
            # summary anymore (it owns the in-fiction line; we own the
            # binding metadata). Tag stays compact to minimize echo
            # surface in router short-circuit prose.
            ckpt.session.pending_router_state_changes.append(
                f"Custom player character created: {new_char.name} "
                f"(id: {new_id}) — {summary} [player-bound]"
            )
        else:
            role = new_char.public_sheet.role or "unknown role"
            loc = new_char.location or "unknown"
            ckpt.session.pending_router_state_changes.append(
                f"Custom player character created: {new_char.name} "
                f"(id: {new_id}), role={role}, location={loc}, bound to a "
                f"human player."
            )
            logger.warning(
                "Custom-character spawn for %s landed without "
                "router_summary; surfaced mechanical fallback line.",
                new_id,
            )

        if out.session_note:
            _append_session_note(ckpt, out.session_note)

        self.checkpoint_mgr.save(ckpt)
        logger.info(
            "Custom character spawned in %s: %s (%s)",
            session_id, new_char.name, new_char.character_id,
        )
        return new_char

    def create_player_character_simple(
        self,
        session_id: str,
        user_id: int,
        *,
        name: str,
        appearance: str,
        backstory: str = "",
    ) -> CharacterRecord:
        """LLM-free player-character spawn from raw user inputs.

        This is the fast path behind /join's "Create your own character"
        option. Unlike `create_custom_character`, it does not call the
        takeover prompt to author personality/goals/secrets/location —
        it just builds a `CharacterRecord` directly from the player's
        own words, marks them `is_playable=True`, binds the user, and
        leaves the rest empty. The router places them via the
        `(arrive)` directive that fires immediately after; the agent
        only takes over voice when the player eventually `/leave`s
        (the personality-synthesis step on /leave fills the gap from
        the rolling conversation).
        - `name` and `appearance` are required.
        - `backstory` is optional; when provided it goes verbatim into
          `CharacterRecord.backstory` so the agent can reference it
          on /leave handoff.

        Returns the freshly-created record (already saved to the
        checkpoint).
        """
        name = name.strip()
        appearance = appearance.strip()
        backstory = backstory.strip()
        if not name:
            raise ValueError("Character name cannot be empty.")
        if not appearance:
            raise ValueError("Character appearance cannot be empty.")

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        new_id = _pick_unused_character_id(ckpt, name)
        new_char = CharacterRecord(
            character_id=new_id,
            name=name,
            status=CharacterStatus.active,
            location="",  # router places via (arrive) directive
            is_playable=True,
            public_sheet=PublicSheet(appearance=appearance),
            backstory=backstory,
        )
        ckpt.characters.append(new_char)
        ckpt.session.character_bindings[new_id] = str(user_id)

        # Surface the spawn to the router. The (arrive) turn that fires
        # right after this will pick the line up via the standard
        # state-changes block and decide where to drop them in. Keep
        # the line tight — long backstories belong on the record, not
        # in the per-turn router context.
        bits = [f"appearance: {appearance[:200]}"]
        if backstory:
            bits.append(f"player-supplied backstory: {backstory[:300]}")
        bits.append(
            "sparse player-authored arrival: infer a concrete story role "
            "and immediate on-ramp from the premise; surface that on-ramp "
            "as in-fiction observable_facts to the NPCs who would know"
        )
        ckpt.session.pending_router_state_changes.append(
            f"Custom player character created: {new_char.name} "
            f"(id: {new_id}) — {'; '.join(bits)}. [player-bound]"
        )

        self.checkpoint_mgr.save(ckpt)
        logger.info(
            "Custom player character (LLM-free) spawned in %s by user %s: "
            "%s (%s)", session_id, user_id, name, new_id,
        )
        return new_char

    async def suggest_replacement_targets(
        self,
        session_id: str,
        description: str,
        *,
        invoking_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Mode='suggest': router surveys the roster for NPCs worth
        replacing with the player's concept. Returns the candidate list
        and an optional preamble. No mutation.

        `invoking_user_id` is forwarded to the prompt's POV-scene
        resolver so multi-player sessions don't see another player's
        scene as "the action" in the suggest context."""
        from app.schemas.takeover import TakeoverSuggestOutput

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        out: TakeoverSuggestOutput = await self._call_takeover(
            ckpt,
            mode="suggest",
            description=description,
            invoking_user_id=invoking_user_id,
        )
        return {
            "candidates": [c.model_dump() for c in out.candidates],
            "preamble": out.preamble,
        }

    async def replace_with_custom(
        self,
        session_id: str,
        user_id: int,
        target_character_id: str,
        description: str,
    ) -> CharacterRecord:
        """Mode='replace': graft a player-authored character onto an
        existing NPC's slot. Preserves circumstances (location, status,
        pending_observations, current_objectives) and overwrites
        identity (name, sheet, backstory, personality, goals,
        known_context, secrets, narrative_notes). Clears the target's
        rolling character_conversation so the new self starts fresh.

        Binds user, marks the slot is_playable=true, saves."""
        from app.schemas.takeover import TakeoverAuthoredOutput

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        target = next(
            (c for c in ckpt.characters if c.character_id == target_character_id),
            None,
        )
        if target is None:
            raise ValueError(
                f"No character '{target_character_id}' in this session."
            )
        claimed_by = ckpt.session.character_bindings.get(target_character_id)
        if claimed_by and claimed_by != str(user_id):
            raise ValueError(
                f"'{target.name}' is already bound to another player."
            )

        out: TakeoverAuthoredOutput = await self._call_takeover(
            ckpt,
            mode="replace",
            description=description,
            picked_target=target,
            invoking_user_id=str(user_id),
        )
        authored = out.character

        # Identity overwrite from authored (flat shape); circumstances
        # preserved from target.
        target.name = authored.name
        target.public_sheet = PublicSheet(
            role=authored.role,
            appearance=authored.appearance,
            faction=authored.faction,
        )
        target.backstory = authored.backstory
        target.personality = authored.personality
        target.known_context = authored.known_context
        target.private_state.goals = list(authored.goals)
        target.private_state.secrets = list(authored.secrets)
        target.private_state.intentions_enabled = authored.intentions_enabled
        # Keep target's location, status, pending_observations, and
        # current_objectives as-is — those are the "circumstances" the
        # player inherits.
        target.is_playable = True

        # Drop rolling character conversation — the voice has changed.
        # The agent's prior parentheticals (their interior continuity)
        # lived in this conversation; popping it gives the new authored
        # character a clean slate.
        ckpt.character_conversations.pop(target_character_id, None)

        ckpt.session.character_bindings[target_character_id] = str(user_id)
        # Same router_summary preference as create_custom_character.
        # For replace, the takeover prompt DOES instruct the LLM to
        # acknowledge the graft (same body / new actor / new
        # motivation) in the summary itself — that's substantive
        # in-fiction content the LLM is best positioned to phrase.
        # The engine adds a compact `[player-bound, replaced X]` tag
        # so the router has unambiguous metadata even if the LLM
        # phrasing is loose; replaced-id tag also double-keys the
        # ghost-spawn cleanup helper since the new line carries the
        # SAME id as the prior NPC.
        summary = _normalize_router_summary(out.character.router_summary or "")
        if summary:
            ckpt.session.pending_router_state_changes.append(
                f"Character replacement: id {target_character_id} "
                f"is now '{target.name}' — {summary} "
                f"[player-bound, replaced prior occupant of this id]"
            )
        else:
            ckpt.session.pending_router_state_changes.append(
                f"Character replacement: identity of {target_character_id} "
                f"has been overwritten — they are now '{target.name}', "
                f"role={target.public_sheet.role or 'unknown role'}, bound "
                f"to a human player. Goals and personality are different "
                f"from the prior version; treat as a new actor with the "
                f"same body."
            )
            logger.warning(
                "Character replacement for %s landed without "
                "router_summary; surfaced mechanical fallback line.",
                target_character_id,
            )

        if out.session_note:
            _append_session_note(ckpt, out.session_note)

        self.checkpoint_mgr.save(ckpt)
        logger.info(
            "Character replaced in %s: %s grafted onto %s",
            session_id, target.name, target_character_id,
        )
        return target

    async def _call_takeover(
        self,
        ckpt: CheckpointFile,
        *,
        mode: str,
        description: str,
        picked_target: CharacterRecord | None = None,
        invoking_user_id: str | None = None,
    ):
        """Render the takeover prompt, dispatch to the LLM with the
        mode-appropriate response model, and return the parsed output.

        mode: 'describe' | 'suggest' | 'replace'
        picked_target: required only for mode='replace'.
        invoking_user_id: the Discord/CLI user running this takeover.
            Forwarded to `_build_takeover_context` so the prompt's
            `current_scene` block tracks the invoking user's scene
            instead of falling through to the creator binding (which
            in multi-player would land on the wrong scene).
        """
        from app.schemas.takeover import (
            TakeoverAuthoredOutput,
            TakeoverSuggestOutput,
        )

        if mode not in {"describe", "suggest", "replace"}:
            raise ValueError(f"Unknown takeover mode: {mode}")
        if mode == "replace" and picked_target is None:
            raise ValueError("mode='replace' requires picked_target")

        context = _build_takeover_context(
            ckpt, description, picked_target,
            invoking_user_id=invoking_user_id,
        )

        response_model = (
            TakeoverSuggestOutput if mode == "suggest" else TakeoverAuthoredOutput
        )
        messages = self.prompt_mgr.render_messages(
            "takeover",
            mode=mode,
            player_description=description,
            **context,
        )
        response = await self.client.complete(
            role="event_router",
            messages=messages,
            response_model=response_model,
            temperature=0.7,
            max_tokens=4000,
        )
        return response.parsed

    def build_character_dossier(
        self,
        session_id: str,
        character_id: str,
    ) -> str:
        """Build the private DM a joining player needs to play this character.

        Strictly character-interior: who they are, what they want, what they
        know, what they're keeping to themselves. The player should learn
        the WORLD through play, not through the dossier.

        Deliberately excludes:
        - `personality` — now absorbs what used to be narrative_notes
          (portrayal direction). That's authorial direction for the AI
          agent; collapses discovery if the player reads it upfront.
          Kept on the record for agent use only.
        - `world_state.hidden_lore` / `hidden_facts` — engine-wide secrets.
          Most characters don't know most of these; dumping them spoils
          the plot. If a specific character genuinely knows a specific
          secret, it belongs in `private_state.secrets` at import time.
        """
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        char = next(
            (c for c in ckpt.characters if c.character_id == character_id), None
        )
        if char is None:
            raise ValueError(f"No character '{character_id}' in this session.")

        lines: list[str] = [
            f"# Dossier · {char.name}",
            "_This is what your character knows about themselves, wants, and "
            "keeps private. The world beyond this page is for you to discover "
            "through play._",
        ]

        sheet = char.public_sheet
        sheet_bits: list[str] = []
        if sheet.role:
            sheet_bits.append(f"**Role** — {sheet.role}")
        if sheet.faction:
            sheet_bits.append(f"**Faction** — {sheet.faction}")
        if sheet.appearance:
            sheet_bits.append(f"**Appearance** — {sheet.appearance}")
        if sheet_bits:
            lines.append("\n".join(sheet_bits))

        if char.backstory:
            lines.append(f"## Your Backstory\n{char.backstory}")
        if char.personality:
            lines.append(f"## How You Think & Feel\n{char.personality}")
        if char.known_context:
            lines.append(f"## The World As You Know It\n{char.known_context}")

        ps = char.private_state
        if ps.goals:
            lines.append(
                "## What Drives You\n" + "\n".join(f"- {g}" for g in ps.goals)
            )
        if ps.current_objectives:
            lines.append(
                "## What You're Working On\n"
                + "\n".join(f"- {o}" for o in ps.current_objectives)
            )
        if ps.secrets:
            lines.append(
                "## What You Keep To Yourself\n"
                + "\n".join(f"- {s}" for s in ps.secrets)
            )

        return "\n\n".join(lines)

    # ---- settings -----------------------------------------------------------

    def list_settings(self, session_id: str) -> list[dict[str, Any]]:
        """Return the full settings view (keys, current + default values,
        descriptions) for the /settings list command."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        return list_settings_view(ckpt)

    def get_setting(self, session_id: str, key: str) -> Any:
        """Return the current value of a single setting by key.
        Raises UnknownSettingError if the key isn't registered."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        return get_setting(ckpt, key)

    def set_setting(self, session_id: str, key: str, raw_value: str) -> Any:
        """Update a setting from its string representation and persist.
        Returns the parsed new value. Raises UnknownSettingError for an
        unregistered key, or ValueError if the raw value can't be parsed."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        new_value = set_setting(ckpt, key, raw_value)
        self.checkpoint_mgr.save(ckpt)
        logger.info(
            "Setting updated in %s: %s = %r", session_id, key, new_value,
        )
        return new_value

    def known_setting_keys(self) -> list[str]:
        """Exposed so bot command autocomplete and CLI can surface the
        valid keys without each frontend importing the registry."""
        return list(SETTINGS_BY_KEY.keys())

    # ---- turn execution ------------------------------------------------------

    async def _lock_for(self, session_id: str) -> asyncio.Lock:
        async with self._locks_mutex:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    # v11-A5 lazy sweep hook.
    # Invoked from the /act hot path (run_turn) so stale Cat II pins are
    # surfaced without a background scheduler. The sweep itself is pure
    # state mutation — synthesizing AFK intentions for humans whose pin
    # has outlived `cat_ii_human_timeout_seconds`. Re-running run_beat on
    # the affected event IDs to drive adjudication is wired later in the
    # full orchestrator-bind commit; for now, logging the IDs at INFO
    # makes the sweep observable in production.
    def sweep_stale_pins(self, session_id: str) -> list[str]:
        """Load the checkpoint, sweep stale Cat II pins, save iff any were
        swept, and return the list of event IDs that now need
        re-adjudication. Safe to call with no open events (returns []).
        """
        from app.engine.turn_loop import sweep_stale_cat_ii_pins

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        swept = sweep_stale_cat_ii_pins(ckpt)
        if swept:
            self.checkpoint_mgr.save(ckpt)
            logger.info(
                "v11 sweep: auto-resolved %d Cat II event(s) pre-turn: %s",
                len(swept), swept,
            )
        return swept

    async def run_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        acting_character_id: str = "",
    ) -> TurnResponse:
        """Process one turn under a per-session lock. Subsequent concurrent calls
        for the same session_id queue and run in order.

        v11-A5 / r5: `sweep_stale_pins` runs INSIDE the per-session lock
        so it cannot race with a concurrent orchestrator turn. Two /acts
        arriving at the same moment serialize: the lock-holder sweeps
        (no-op if nothing stale) then runs the orchestrator; the second
        /act then sees the already-swept state and its own sweep is a
        no-op. This is a narrow path — sweeps usually don't mutate —
        but moving it inside the lock eliminates the clobber race
        flagged by the round-4 edge review.
        """
        lock = await self._lock_for(session_id)
        async with lock:
            return await self._run_turn_locked(
                session_id=session_id,
                user_input=user_input,
                acting_character_id=acting_character_id,
            )

    async def run_arrival_turn(
        self,
        *,
        session_id: str,
        acting_character_id: str,
    ) -> TurnResponse:
        """Run an `(arrive)` turn for a player joining a story that's
        already underway. Always fires `(arrive)` — the canonical
        opening (`(begin)`) is now driven exclusively by `/begin`
        through `run_begin_turn`, which the caller invokes BEFORE this
        method becomes the right tool.

        Pre-r9d this method picked `(begin)` vs `(arrive)` itself by
        inspecting `narrator_conversations`. That branch went away
        with the lobby split: `/join` now binds-only when the story
        hasn't begun, and the dedicated `/begin` command opens the
        story. By the time anyone calls `run_arrival_turn`, the
        opening has already happened and `(arrive)` is unambiguous.
        """
        lock = await self._lock_for(session_id)
        async with lock:
            logger.info(
                "run_arrival_turn: session=%s actor=%s directive=(arrive)",
                session_id, acting_character_id,
            )
            return await self._run_turn_locked(
                session_id=session_id,
                user_input="(arrive)",
                acting_character_id=acting_character_id,
            )

    async def run_begin_turn(
        self,
        *,
        session_id: str,
        triggering_character_id: str = "",
    ) -> TurnResponse:
        """Fire the canonical opening turn (`(begin)`) for the session.

        Driven by `/begin` after one or more players have `/join`'d
        into the lobby (pre-play). The router sees `(begin)` and
        composes the opening scene from `world_state` + the
        `## Initial Roster`, placing EVERY bound player at the chosen
        starting scene so each gets their own POV render through the
        normal `_end_beat` per-POV fan-out.

        Args:
            session_id: the session to open.
            triggering_character_id: the player who typed `/begin`.
                Used as the `acting_character_id` so the router's
                framing names them as the protagonist. May be empty
                — the helper falls back to a deterministic pick from
                the bound roster (sorted by id) so two simultaneous
                `/begin`s converge on the same actor.

        Raises:
            ValueError: if no players are bound, or if the story has
                already begun (any narrator history present). Both
                are pre-checked under the per-session lock so two
                racing `/begin`s can't both fire the opener.

        TODO(multi-scene-opening): this function currently asks the
        router to converge all bound players on one shared starting
        scene (see event_router.txt OOC `(begin)` rules). When we
        want distinct starting scenes per player, run the opening as
        N parallel `(begin)` calls — one per bound player, each
        seeded with their own intended location — and merge the
        resulting per-POV renders. The current single-scene path is
        the simplest correct first step; the multi-scene shape can
        be additive on top of it.
        """
        lock = await self._lock_for(session_id)
        async with lock:
            ckpt = self.checkpoint_mgr.load_latest(session_id)

            # Bound players by id, sorted for deterministic
            # actor selection when triggering_character_id is unset
            # or has been kicked between dispatch and lock acquisition.
            bound_ids = sorted((ckpt.session.character_bindings or {}).keys())
            if not bound_ids:
                raise ValueError(
                    "Cannot /begin: no players are bound to this session. "
                    "Have at least one player run /join first."
                )

            if any(ckpt.narrator_conversations.values()):
                raise ValueError(
                    "Cannot /begin: this story has already started. "
                    "Late joiners use /join — it now fires (arrive) "
                    "for any player binding after the opening."
                )

            actor_id = (
                triggering_character_id
                if triggering_character_id in bound_ids
                else bound_ids[0]
            )
            logger.info(
                "run_begin_turn: session=%s actor=%s bound=%s",
                session_id, actor_id, bound_ids,
            )
            return await self._run_turn_locked(
                session_id=session_id,
                user_input="(begin)",
                acting_character_id=actor_id,
            )

    async def _run_turn_locked(
        self,
        *,
        session_id: str,
        user_input: str,
        acting_character_id: str,
    ) -> TurnResponse:
        """Body of `run_turn` — caller MUST already hold the per-session
        lock. Extracted so `run_arrival_turn` can interleave its
        directive choice with the same sweep+orchestrate flow."""
        try:
            event_ids = self.sweep_stale_pins(session_id)
        except Exception:
            # Best-effort — never let an AFK sweep error crash a turn.
            logger.exception(
                "v11 sweep_stale_pins failed for %s", session_id,
            )
            event_ids = []

        # v11-r6b: drive adjudication of any events the sweep filled.
        # Without this, a scene pinned on an AFK human sits open
        # indefinitely — the next /act would bounce off the pin. By
        # closing the sweep-populated events first, the hot path
        # clears their state before the player's /act runs.
        #
        # v11-r7a: capture each resolution's TurnResponse so the
        # frontend can fan its per-POV renders out. Previously the
        # responses were dropped and pinned humans never saw their
        # AFK-resolved beats.
        pre_turn: list[TurnResponse] = []
        for event_id in event_ids:
            try:
                resp = await self.orchestrator.resolve_cat_ii(
                    session_id, event_id,
                )
                if resp.per_player_renders:
                    pre_turn.append(resp)
            except Exception:
                logger.exception(
                    "resolve_cat_ii failed for session=%s event=%s",
                    session_id, event_id,
                )

        response = await self.orchestrator.process_turn(TurnRequest(
            session_id=session_id,
            user_input=user_input,
            acting_character_id=acting_character_id,
        ))
        response.pre_turn_resolutions = pre_turn
        return response

    async def run_query(
        self,
        *,
        session_id: str,
        character_id: str,
        question: str,
    ) -> "QueryResponse":
        """Answer an out-of-character /query for `character_id` in
        `session_id`. Read-only — no checkpoint mutation, no
        per-session lock, no turn advancement, no broadcast. The
        asking character's POV envelope bounds the answer; questions
        outside that envelope come back with `knowledge_gated=True`
        and an in-fiction refusal.

        Deliberately bypasses the per-session lock that wraps
        `run_turn`. The query is a pure read of the latest
        checkpoint snapshot on disk — concurrent /query and /act
        on the same session are safe because /act writes a brand-new
        checkpoint file rather than mutating in place. A query
        running while a turn is mid-flight may read the
        pre-turn snapshot; that's an acceptable staleness window
        (the alternative is making the user wait on the lock for an
        OOC question, which would be worse).
        """
        # Local import to keep the engine layer's full dependency
        # graph out of EngineBridge's module load — query_handler
        # transitively imports the schema + prompt manager + LLM
        # client, all of which are fine but the lazy import keeps
        # the symbol surface small at import time.
        from app.engine.query_handler import answer_query
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        return await answer_query(
            self.client, self.prompt_mgr, ckpt, character_id, question,
        )


# ---- player-flow helpers ----------------------------------------------------
# Helpers that support the bot's player-facing flows. Under playable-2
# semantics, every player→character lookup routes through
# `session.character_bindings` (the canonical live state). `is_playable`
# is now an authoring-time flag describing which slots a human MAY step
# into via /join — not a runtime flag toggled by binding.
#
# * `build_character_dossier` / `_summaries_from_checkpoint` — bindings-
#   driven; used by takeover prompts and roster summaries.
# * `_pick_unused_character_id` — pure slug helper.


def _pick_unused_character_id(
    ckpt: CheckpointFile, name: str,
) -> str:
    """Slugify `name` into a snake_case character_id; disambiguate with
    a numeric suffix if the slug is already in use."""
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "character"
    taken = {c.character_id for c in ckpt.characters}
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def _build_takeover_context(
    ckpt: CheckpointFile,
    description: str,
    picked_target: "CharacterRecord | None",
    invoking_user_id: str | None = None,
) -> dict[str, str]:
    """Render the context blocks the takeover prompt expects. Stays
    router-style: omniscient world state, full roster (spoiler-safe is
    NOT a constraint here — the router is authoring for itself).

    `invoking_user_id`: the Discord user (or CLI session) running the
    takeover. Threaded into `pov_scene_for_user` so the prompt's
    `current_scene` block reflects WHERE THIS USER IS, not the
    creator binding or "first is_playable" fallback. In multi-player
    sessions those fallbacks would otherwise hand the takeover LLM
    another player's scene as "the action," which is the wrong frame.
    """
    from app.engine.context_builder import (
        build_setting_summary,
        pov_scene_for_user,
    )
    setting_summary = build_setting_summary(ckpt)
    world_lore = ckpt.world_state.lore or "No detailed lore."
    hidden_lore = ckpt.world_state.hidden_lore or "(none)"
    hidden_facts = (
        "\n".join(f"- {f}" for f in ckpt.world_state.hidden_facts)
        or "(none)"
    )
    physics = ckpt.world_state.physics_ruleset
    world_rules = (
        f"Strength limits: {physics.strength_limits}\n"
        f"Magic: {'enabled' if physics.magic_enabled else 'disabled'}"
    )

    locations = ckpt.world_state.locations
    # The "current scene" the takeover prompt cares about is "where THE
    # INVOKING USER is right now." Multi-player sessions can have
    # players in different scenes, and "where the action is" must be
    # keyed on whoever invoked takeover, not on a session-global
    # default. `pov_scene_for_user` resolves: bound character → creator
    # binding → first is_playable → "". For takeover the invoking user
    # almost always has a binding (they took over from somewhere); if
    # they don't (rare: brand-new join replacing their first NPC),
    # the creator-binding fallback is the next-best frame.
    scene_id = pov_scene_for_user(ckpt, user_id=invoking_user_id)
    scene = locations.scene_graph.get(scene_id, {}) if scene_id else {}
    scene_graph_lines = []
    for sid, sdata in locations.scene_graph.items():
        if isinstance(sdata, dict):
            sname = sdata.get("name", sid)
            conn = sdata.get("connected_to", []) or []
            scene_graph_lines.append(
                f"- {sname} (id: {sid})"
                + (f"; connected to {', '.join(conn)}" if conn else "")
            )
    scene_graph = "\n".join(scene_graph_lines) or "(empty)"
    current_scene = (
        f"{scene.get('name', scene_id)} (id: {scene_id})\n"
        f"{scene.get('description', '')}".strip()
        if scene_id else "(no active scene)"
    )

    registry_lines = []
    bindings = ckpt.session.character_bindings or {}
    for c in ckpt.characters:
        if c.status.value == "culled":
            continue
        if c.character_id in bindings:
            marker = " [player-bound]"
        elif c.is_playable:
            marker = " [playable]"
        else:
            marker = ""
        role = f" — {c.public_sheet.role}" if c.public_sheet.role else ""
        fac = f" ({c.public_sheet.faction})" if c.public_sheet.faction else ""
        loc = f" @ {c.location}" if c.location else ""
        registry_lines.append(
            f"- {c.name} ({c.character_id}){role}{fac}{loc}{marker}"
        )
    character_registry = "\n".join(registry_lines) or "(empty)"

    transcript = ckpt.transcript[-6:] if ckpt.transcript else []
    if transcript:
        recent_bits = []
        for entry in transcript:
            recent_bits.append(f"> {entry.user}\n{entry.assistant}")
        recent_session_summary = "\n\n".join(recent_bits)
    else:
        recent_session_summary = "(no turns played yet)"

    if picked_target is not None:
        picked_target_block = (
            "## Picked Target\n"
            f"character_id: {picked_target.character_id}\n"
            f"name: {picked_target.name}\n"
            f"role: {picked_target.public_sheet.role}\n"
            f"faction: {picked_target.public_sheet.faction}\n"
            f"location: {picked_target.location}\n"
            f"backstory: {picked_target.backstory}\n\n"
        )
    else:
        picked_target_block = ""

    return {
        "setting_summary": setting_summary,
        "world_lore": world_lore,
        "hidden_lore": hidden_lore,
        "hidden_facts": hidden_facts,
        "world_rules": world_rules,
        "scene_graph": scene_graph,
        "current_scene": current_scene,
        "character_registry": character_registry,
        "recent_session_summary": recent_session_summary,
        "picked_target_block": picked_target_block,
    }


def _append_session_note(ckpt: CheckpointFile, note: str) -> None:
    """Post a router-authored in-fiction note into the session_conversation
    so the next turn's router sees the new character's arrival. Stored
    as an assistant-like entry; non-authoritative — purely context."""
    from app.schemas.conversation import ConversationMessage

    if not note.strip():
        return
    ckpt.session_conversation.append(ConversationMessage(
        role="assistant",
        content=f'{{"takeover_note": {json.dumps(note)}}}',
    ))


def _parse_model_json(model_cls, content: str):
    """Parse a Pydantic model from the LLM's free-form JSON output.

    Used when we can't enforce structured output via output_format. A
    live benchmark (tests/test_structured_output_benchmark.py) showed
    the enforced path 400s with "Schema is too complex" 0/3 times for
    AuthoredCharacter after ~180s of hanging, while raw JSON + this
    helper succeeded 3/3 times in ~29s. output_format is also marked
    deprecated by the SDK, so we're on the sunset path using it.

    Strips common LLM envelope noise (markdown fences) and parses with
    Pydantic. Raises ValueError on unparseable output with a 500-char
    snippet so call sites surface something diagnostic.
    """
    import re

    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return model_cls.model_validate_json(text)
    except Exception as e:
        snippet = text[:500]
        raise ValueError(
            f"Failed to parse {model_cls.__name__} from LLM output: {e} "
            f"— first 500 chars: {snippet!r}"
        ) from e


def migrate_legacy_saves(
    legacy_dir: Path,
    stories_dir: Path,
    sessions_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Move a legacy flat `saves/` layout into the split stories/ + sessions/
    layout. Returns `(stories_moved, sessions_moved)`.

    Safety:
    - All three paths are explicit — no hidden defaults that could target
      production state by accident.
    - Refuses to run if `legacy_dir` resolves to a parent/equal of either
      target dir, or vice-versa — prevents moving a dir into its own
      subtree.
    - Refuses to run if either target already contains subdirs — prior
      migration present.
    - `dry_run=True` logs the classification without moving anything.

    Classification: a legacy dir is treated as a session if it has more
    than one ckpt, a non-zero turn_index, or any transcript entries.
    Otherwise it's treated as a story (pristine ckpt_0000).

    Not called automatically. Invoke via scripts/migrate_storage.py with
    explicit --legacy-dir / --stories-dir / --sessions-dir arguments.
    """
    legacy_dir = Path(legacy_dir).resolve()
    stories_dir = Path(stories_dir).resolve()
    sessions_dir = Path(sessions_dir).resolve()

    if not legacy_dir.exists() or not legacy_dir.is_dir():
        raise FileNotFoundError(f"legacy_dir does not exist: {legacy_dir}")

    for target in (stories_dir, sessions_dir):
        # Target cannot be inside legacy and legacy cannot be inside target.
        if target == legacy_dir or target in legacy_dir.parents:
            raise ValueError(
                f"target {target} is an ancestor of legacy_dir {legacy_dir}"
            )
        if legacy_dir in target.parents:
            raise ValueError(
                f"legacy_dir {legacy_dir} is an ancestor of target {target}"
            )

    for target in (stories_dir, sessions_dir):
        if target.exists() and any(c.is_dir() for c in target.iterdir()):
            raise RuntimeError(
                f"target already has content: {target} "
                f"— refusing to migrate into a populated directory"
            )

    if not dry_run:
        stories_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir.mkdir(parents=True, exist_ok=True)

    stories_moved = 0
    sessions_moved = 0
    for child in legacy_dir.iterdir():
        if not child.is_dir():
            continue
        ckpts = sorted(child.glob("ckpt_*.json"))
        if not ckpts:
            continue
        kind = "story"
        try:
            latest = json.loads(ckpts[-1].read_text())
            turn = latest.get("session", {}).get("turn_index", 0) or 0
            has_transcript = bool(latest.get("transcript"))
            if len(ckpts) > 1 or turn > 0 or has_transcript:
                kind = "session"
        except Exception:
            kind = "session"  # safer to treat unknown as session (don't overwrite a story import)
        target = (stories_dir if kind == "story" else sessions_dir) / child.name
        logger.info(
            "%s %s → %s/%s",
            "[dry-run] would move" if dry_run else "Moving",
            child.name, kind, child.name,
        )
        if not dry_run:
            if target.exists():
                logger.warning(
                    "Target %s already exists, skipping", target
                )
                continue
            shutil.move(str(child), str(target))
        if kind == "story":
            stories_moved += 1
        else:
            sessions_moved += 1
    return stories_moved, sessions_moved


def _summaries_from_checkpoint(ckpt: CheckpointFile) -> list[CharacterSummary]:
    """Render a roster's spoiler-free summaries from any checkpoint."""
    bindings = ckpt.session.character_bindings or {}
    summaries: list[CharacterSummary] = []
    for char in ckpt.characters:
        summaries.append(CharacterSummary(
            character_id=char.character_id,
            name=char.name,
            role=char.public_sheet.role or "",
            faction=char.public_sheet.faction or "",
            appearance=char.public_sheet.appearance or "",
            status=char.status.value,
            is_playable=char.is_playable,
            bound_user_id=bindings.get(char.character_id, ""),
        ))
    return summaries
