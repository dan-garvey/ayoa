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
        and return the resulting checkpoint. Safe to call again (overwrites)."""
        src = self.saves_dir / story_id / "ckpt_0000.json"
        if not src.exists():
            raise FileNotFoundError(f"Story '{story_id}' not found at {src}")

        dst_dir = self.saves_dir / session_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "ckpt_0000.json"
        shutil.copy2(src, dst)

        # Rewrite session_id inside the JSON before personalizing.
        raw = dst.read_text()
        data = json.loads(raw)
        data["session"]["session_id"] = session_id
        dst.write_text(json.dumps(data, indent=2))

        # Personalize in-place, save as ckpt_0001 (keeps ckpt_0000 pristine).
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        personalized = _personalize(ckpt, player_display_name)
        # Bump the turn index so save() lands at ckpt_0001.
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
        """Update session.player_character_description in-place and save."""
        ckpt = self.checkpoint_mgr.load_latest(session_id)
        ckpt.session.player_character_description = description.strip()
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
# Mirrors the logic in app/api/story_routes.py::personalize_story so the bot
# does not depend on the FastAPI route.

_PLACEHOLDER_CHAR_ID_SUFFIX = "_garvey"
_PLACEHOLDER_CHAR_ID = "player_name" + _PLACEHOLDER_CHAR_ID_SUFFIX


def _personalize(checkpoint: CheckpointFile, player_name: str) -> CheckpointFile:
    """Replace PLAYER_NAME placeholder with the given display name and set
    player_character_id from the slugged name."""
    name = player_name.strip()
    if not name:
        raise ValueError("player_name cannot be empty")

    new_char_id = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + _PLACEHOLDER_CHAR_ID_SUFFIX

    data: dict[str, Any] = json.loads(checkpoint.model_dump_json())
    raw = json.dumps(data)
    raw = raw.replace("PLAYER_NAME", name)
    raw = raw.replace(_PLACEHOLDER_CHAR_ID, new_char_id)
    data = json.loads(raw)

    data["session"]["player_name"] = name
    data["session"]["player_character_id"] = new_char_id

    known = data.get("world_state", {}).get("known_characters", [])
    if new_char_id not in known:
        known.append(new_char_id)
    data.setdefault("world_state", {})["known_characters"] = known

    return CheckpointFile(**data)
