"""Thin async wrapper around the engine for the Discord bot.

Responsibilities:
- Holds the shared Orchestrator, LLMClient, CheckpointManager, PromptManager.
- Creates a fresh session from an imported story (copy ckpt_0000 + personalize).
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
from typing import Any, Awaitable, Callable

from app.engine.checkpoint_manager import CheckpointManager
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.engine.settings import (
    SETTINGS_BY_KEY,
    get_setting,
    list_settings_view,
    set_setting,
)
from app.engine.story_importer import run_import, run_preservation_analysis
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile, ImportAnalysis
from app.schemas.requests import TurnRequest
from app.schemas.responses import TurnResponse


@dataclass(frozen=True)
class CharacterSummary:
    """Spoiler-free summary of a character for /story characters and /join.

    Only fields from `public_sheet` plus status/name/binding — nothing from
    private_state, backstory, personality, narrative_notes, or hidden lore.
    `bound_user_id` is populated when reading a session checkpoint; empty on
    pristine story-level lookups.
    """
    character_id: str
    name: str
    role: str
    faction: str
    traits: list[str]
    appearance: str
    status: str  # "active" | "dormant" | "culled"
    is_player: bool
    bound_user_id: str = ""

logger = logging.getLogger(__name__)


class EngineBridge:
    """Shared engine state for all Discord interactions."""

    def __init__(
        self,
        *,
        saves_dir: str = "app/storage/saves",
        prompts_dir: str = "app/prompts",
        llm_config: LLMConfig | None = None,
    ):
        self.saves_dir = Path(saves_dir)
        self.client = LLMClient(config=llm_config or LLMConfig.from_env())
        self.checkpoint_mgr = CheckpointManager(save_dir=str(self.saves_dir))
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
        """Return available story IDs — directories under saves/ that contain
        a ckpt_0000.json. Excludes any session directory we created previously
        (those have names prefixed with 'discord_')."""
        if not self.saves_dir.exists():
            return []
        ids = []
        for child in sorted(self.saves_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("discord_"):
                continue
            if (child / "ckpt_0000.json").exists():
                ids.append(child.name)
        return ids

    def load_story_ckpt(self, story_id: str) -> CheckpointFile:
        """Load a story's source ckpt_0000 (pristine, not a session checkpoint)."""
        path = self.saves_dir / story_id / "ckpt_0000.json"
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
        dst_dir = self.saves_dir / story_id
        dst_ckpt = dst_dir / "ckpt_0000.json"
        if dst_ckpt.exists():
            raise FileExistsError(
                f"Story '{story_id}' already exists at {dst_ckpt}. "
                f"Delete it first or pick a different story_id."
            )

        checkpoint = await run_import(self.client, source_text, story_id)

        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_ckpt.write_text(checkpoint.model_dump_json(indent=2))
        logger.info("Imported story %s → %s", story_id, dst_ckpt)

        # Fire preservation analysis in the background. The bot keeps an
        # event loop running for the duration of the process, so the task
        # has a home. It reloads the saved checkpoint, runs the audit,
        # patches import_analysis back to disk. We don't await — the
        # import command returns to the user immediately.
        asyncio.create_task(
            self._background_preservation_analysis(
                source_text, story_id, dst_ckpt,
                on_complete=on_analysis_complete,
            )
        )
        return checkpoint

    async def _background_preservation_analysis(
        self,
        source_text: str,
        story_id: str,
        dst_ckpt: Path,
        on_complete: Callable[
            [ImportAnalysis | None, Exception | None], Awaitable[None]
        ] | None = None,
    ) -> None:
        """Run preservation analysis and patch the checkpoint on disk.
        Any analysis error is caught and surfaced to `on_complete` so the
        frontend can notify the user — best-effort metadata should not
        crash the task loop."""
        analysis: ImportAnalysis | None = None
        err: Exception | None = None
        try:
            checkpoint = CheckpointFile.model_validate_json(dst_ckpt.read_text())
            analysis = await run_preservation_analysis(
                self.client, source_text, checkpoint,
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

    def delete_story(self, story_id: str) -> tuple[int, int]:
        """Delete a story's source dir AND any discord_* session dirs derived
        from it. Returns (session_dirs_removed, total_files_removed).

        Raises FileNotFoundError if the source story does not exist.
        """
        source_dir = self.saves_dir / story_id
        if not source_dir.exists():
            raise FileNotFoundError(f"Story '{story_id}' not found at {source_dir}")

        sessions_removed = 0
        files_removed = 0

        # Derived discord sessions have name pattern
        # discord_<channel_id>_<story_slug> where story_slug is the sluggified
        # story_id. Match by suffix for safety.
        slug = re.sub(r"[^a-z0-9_]+", "_", story_id.lower()).strip("_")
        suffix = f"_{slug}"
        for child in self.saves_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith("discord_") and child.name.endswith(suffix):
                file_count = sum(1 for _ in child.iterdir())
                shutil.rmtree(child)
                sessions_removed += 1
                files_removed += file_count

        file_count = sum(1 for _ in source_dir.iterdir())
        shutil.rmtree(source_dir)
        files_removed += file_count

        logger.info(
            "Deleted story %s and %d derived session dir(s), %d files total",
            story_id, sessions_removed, files_removed,
        )
        return sessions_removed, files_removed

    def session_id_for_channel(self, channel_id: int, story_id: str) -> str:
        """Deterministic session id for (channel, story). Stable across /resume."""
        # Short-hash-free: channel IDs are unique and story id is stable.
        slug = re.sub(r"[^a-z0-9_]+", "_", story_id.lower()).strip("_")
        return f"discord_{channel_id}_{slug}"

    async def create_session(
        self,
        *,
        story_id: str,
        session_id: str,
        player_display_name: str,
        creator_user_id: int | None = None,
    ) -> CheckpointFile:
        """Copy the story's ckpt_0000 into a new session dir, personalize it,
        and return the resulting checkpoint.

        Since session_id is deterministic from (channel, story), re-running
        /story start on the same channel/story would land in a directory that
        may contain stale ckpt_NNNN.json files from a prior play-through.
        We wipe those so the new session starts clean — otherwise load_latest
        picks up an old pre-schema-migration checkpoint and personalize fails.
        """
        src = self.saves_dir / story_id / "ckpt_0000.json"
        if not src.exists():
            raise FileNotFoundError(f"Story '{story_id}' not found at {src}")

        dst_dir = self.saves_dir / session_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Clear any stale checkpoint files in this session dir — user called
        # /story start expecting a fresh start (they'd have used /story resume
        # to continue). Only touches ckpt_*.json, leaves any other files alone
        # (e.g. .pre_conv_migration backups).
        for old in dst_dir.glob("ckpt_*.json"):
            old.unlink()

        dst = dst_dir / "ckpt_0000.json"
        shutil.copy2(src, dst)

        # Rewrite session_id inside the JSON before personalizing.
        raw = dst.read_text()
        data = json.loads(raw)
        data["session"]["session_id"] = session_id
        dst.write_text(json.dumps(data, indent=2))

        # Personalize the pristine ckpt_0000, save as ckpt_0001.
        ckpt = self.checkpoint_mgr.load(session_id, "ckpt_0000")
        personalized = _personalize(ckpt, player_display_name)
        if personalized.session.turn_index == 0:
            personalized.session.turn_index = 1

        # Auto-bind the creator to the is_player character so /act works
        # without requiring a separate /join from them.
        if creator_user_id is not None and personalized.session.player_character_id:
            personalized.session.character_bindings[
                personalized.session.player_character_id
            ] = str(creator_user_id)

        self.checkpoint_mgr.save(personalized)
        return personalized

    def load_latest(self, session_id: str) -> CheckpointFile:
        return self.checkpoint_mgr.load_latest(session_id)

    def set_character_appearance(
        self,
        session_id: str,
        character_id: str,
        description: str,
    ) -> CheckpointFile:
        """Update a single character's public appearance and save.

        This replaces the old single-player set_player_character_description.
        For multi-player sessions the caller passes the character_id of
        whichever player character owns the description (their own binding).
        The appearance string is used by the Player Characters block in all
        engine prompts, and by the agent context builder's Characters Present
        summary that NPCs see.
        """
        description = description.strip()
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        target = next(
            (c for c in ckpt.characters if c.character_id == character_id), None
        )
        if target is None:
            raise ValueError(
                f"Character '{character_id}' not found in session '{session_id}'"
            )
        target.public_sheet.appearance = description
        self.checkpoint_mgr.save(ckpt)
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
        self.checkpoint_mgr.save(ckpt)
        return ckpt

    def unbind_user(self, session_id: str, user_id: int) -> str | None:
        """Remove this user's binding. Returns the freed character_id, or None
        if they had no binding."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        uid = str(user_id)
        freed = None
        for char_id, bound in list(ckpt.session.character_bindings.items()):
            if bound == uid:
                freed = char_id
                del ckpt.session.character_bindings[char_id]
        if freed is not None:
            self.checkpoint_mgr.save(ckpt)
        return freed

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
        - `narrative_notes` — authorial portrayal direction ("reached through
          sustained non-demanding sincerity") collapses discovery if the
          player reads it upfront. Kept on the character for AI agent use.
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
        if sheet.traits:
            sheet_bits.append(f"**Traits** — {', '.join(sheet.traits)}")
        if sheet.voice:
            sheet_bits.append(f"**Voice** — {sheet.voice}")
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
        if ps.attitudes:
            att_lines = [
                f"- {k}: {v:+.1f}" for k, v in ps.attitudes.items()
            ]
            lines.append("## How You See Others\n" + "\n".join(att_lines))

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

    async def run_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        acting_character_id: str = "",
        debug: bool = False,
    ) -> TurnResponse:
        """Process one turn under a per-session lock. Subsequent concurrent calls
        for the same session_id queue and run in order."""
        lock = await self._lock_for(session_id)
        async with lock:
            return await self.orchestrator.process_turn(TurnRequest(
                session_id=session_id,
                user_input=user_input,
                acting_character_id=acting_character_id,
                debug=debug,
            ))


# ---- personalize helper -----------------------------------------------------
# Finds the player character via the roster's is_player flag (not a
# hard-coded id suffix), rewrites character_id/name/PLAYER_NAME, and keeps
# the runtime checkpoint schema consistent. Mirrors
# app/api/story_routes.py::_apply_personalize so the bot does not depend on
# the FastAPI route.


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
            traits=list(char.public_sheet.traits),
            appearance=char.public_sheet.appearance or "",
            status=char.status.value,
            is_player=char.is_player,
            bound_user_id=bindings.get(char.character_id, ""),
        ))
    return summaries


def _personalize(checkpoint: CheckpointFile, player_name: str) -> CheckpointFile:
    name = player_name.strip()
    if not name:
        raise ValueError("player_name cannot be empty")

    player_chars = [c for c in checkpoint.characters if c.is_player]
    if not player_chars:
        raise ValueError(
            "Roster has no is_player=true character. Re-import the master "
            "prompt with a protagonist explicitly designated."
        )
    if len(player_chars) > 1:
        logger.warning(
            "Roster has %d is_player characters; personalize binds only the first "
            "(%s). Multi-player support lands in Phase 3.",
            len(player_chars), player_chars[0].name,
        )

    pc = player_chars[0]
    old_char_id = pc.character_id
    new_char_id = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "player"

    raw = checkpoint.model_dump_json()
    raw = raw.replace("PLAYER_NAME", name)
    raw = raw.replace(f'"{old_char_id}"', f'"{new_char_id}"')
    data: dict[str, Any] = json.loads(raw)

    data["session"]["player_name"] = name
    data["session"]["player_character_id"] = new_char_id

    known = data.get("world_state", {}).get("known_characters", [])
    if new_char_id not in known:
        known.append(new_char_id)
    data.setdefault("world_state", {})["known_characters"] = known

    return CheckpointFile.model_validate(data)
