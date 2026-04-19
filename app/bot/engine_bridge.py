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
from pathlib import Path
from typing import Any

from app.engine.checkpoint_manager import CheckpointManager
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.engine.story_importer import run_import
from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile
from app.schemas.requests import TurnRequest
from app.schemas.responses import TurnResponse

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
    ) -> CheckpointFile:
        """Run the three-stage import pipeline and save the resulting
        ckpt_0000.json under saves/<story_id>/.

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
        return checkpoint

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
        self.checkpoint_mgr.save(personalized)
        return personalized

    def load_latest(self, session_id: str) -> CheckpointFile:
        return self.checkpoint_mgr.load_latest(session_id)

    def set_player_character_description(
        self,
        session_id: str,
        description: str,
    ) -> CheckpointFile:
        """Update the player character's appearance on the checkpoint and save.

        Writes to both `session.player_character_description` (used by the
        narrator/router/agent system prompts' Player Character block) AND
        the player character's `public_sheet.appearance` (used by the
        agent context builder's 'Characters Present' summary that NPCs see).
        Master prompts often leave appearance as a placeholder like
        'Defined by player input' which would otherwise leak into NPC context.
        """
        description = description.strip()
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        ckpt.session.player_character_description = description

        pc_id = ckpt.session.player_character_id
        for char in ckpt.characters:
            if char.character_id == pc_id or char.is_player:
                char.public_sheet.appearance = description

        self.checkpoint_mgr.save(ckpt)
        return ckpt

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
        debug: bool = False,
    ) -> TurnResponse:
        """Process one turn under a per-session lock. Subsequent concurrent calls
        for the same session_id queue and run in order."""
        lock = await self._lock_for(session_id)
        async with lock:
            return await self.orchestrator.process_turn(TurnRequest(
                session_id=session_id,
                user_input=user_input,
                debug=debug,
            ))


# ---- personalize helper -----------------------------------------------------
# Finds the player character via the roster's is_player flag (not a
# hard-coded id suffix), rewrites character_id/name/PLAYER_NAME, and keeps
# the runtime checkpoint schema consistent. Mirrors
# app/api/story_routes.py::_apply_personalize so the bot does not depend on
# the FastAPI route.


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
