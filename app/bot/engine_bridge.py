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
from app.schemas.characters import CharacterRecord
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
        source_dir = self.stories_dir / story_id
        if not source_dir.exists():
            raise FileNotFoundError(f"Story '{story_id}' not found at {source_dir}")

        sessions_removed = 0
        files_removed = 0

        # Derived discord sessions have name pattern
        # discord_<channel_id>_<story_slug> where story_slug is the sluggified
        # story_id. Match by suffix for safety.
        slug = re.sub(r"[^a-z0-9_]+", "_", story_id.lower()).strip("_")
        suffix = f"_{slug}"
        if self.sessions_dir.exists():
            for child in self.sessions_dir.iterdir():
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
        src = self.stories_dir / story_id / "ckpt_0000.json"
        if not src.exists():
            raise FileNotFoundError(f"Story '{story_id}' not found at {src}")

        dst_dir = self.sessions_dir / session_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Clear any stale checkpoint files in this session dir — user called
        # /story start expecting a fresh start (they'd have used /story resume
        # to continue). Only touches ckpt_*.json, leaves any other files alone
        # (e.g. .pre_conv_migration backups).
        for old in dst_dir.glob("ckpt_*.json"):
            old.unlink()

        # Copy the story's pristine ckpt_0000 into the new session dir,
        # rewriting session_id and dropping import_analysis — the analysis
        # describes the one-time import pass and belongs on the story's
        # canonical source, not on every session derived from it.
        dst = dst_dir / "ckpt_0000.json"
        data = json.loads(src.read_text())
        data["session"]["session_id"] = session_id
        data.pop("import_analysis", None)
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

    # ---- takeover -----------------------------------------------------------

    def takeover(
        self,
        session_id: str,
        character_id: str,
        user_id: int,
    ) -> CheckpointFile:
        """Plain takeover: bind the user to an existing character and flip
        is_player=True. Name, appearance, everything else unchanged.

        This is the default `/join` path — the user becomes the character
        as-authored. Refuses if already claimed, culled, or nonexistent
        (via bind_user's checks) and on already-is_player+already-bound."""
        ckpt = self.bind_user(session_id, user_id, character_id)
        target = next(
            (c for c in ckpt.characters if c.character_id == character_id), None,
        )
        if target is None:
            raise ValueError(f"No character '{character_id}' in this session.")
        if not target.is_player:
            target.is_player = True
            self.checkpoint_mgr.save(ckpt)
        return ckpt

    async def create_custom_character(
        self,
        session_id: str,
        user_id: int,
        description: str,
    ) -> CharacterRecord:
        """Mode='describe': router authors a full new character from the
        player's concept, lands them in the world, binds to the user. The
        returned record has its engine-assigned character_id, is_player=True,
        and is already written to the checkpoint."""
        from app.schemas.takeover import TakeoverAuthoredOutput

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        out: TakeoverAuthoredOutput = await self._call_takeover(
            ckpt,
            mode="describe",
            description=description,
        )
        new_char = out.character
        new_char.character_id = _pick_unused_character_id(ckpt, new_char.name)
        new_char.is_player = True
        ckpt.characters.append(new_char)
        ckpt.session.character_bindings[new_char.character_id] = str(user_id)

        if out.session_note:
            _append_session_note(ckpt, out.session_note)

        self.checkpoint_mgr.save(ckpt)
        logger.info(
            "Custom character spawned in %s: %s (%s)",
            session_id, new_char.name, new_char.character_id,
        )
        return new_char

    async def suggest_replacement_targets(
        self,
        session_id: str,
        description: str,
    ) -> dict[str, Any]:
        """Mode='suggest': router surveys the roster for NPCs worth
        replacing with the player's concept. Returns the candidate list
        and an optional preamble. No mutation."""
        from app.schemas.takeover import TakeoverSuggestOutput

        ckpt = self.checkpoint_mgr.load_latest(session_id)
        out: TakeoverSuggestOutput = await self._call_takeover(
            ckpt,
            mode="suggest",
            description=description,
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
        incoming_directives, pending_observations, current_objectives)
        and overwrites identity (name, sheet, backstory, personality,
        goals, known_context, secrets, narrative_notes). Clears the
        target's rolling character_conversation so the new self starts
        fresh.

        Binds user, flips is_player, saves."""
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
        if target.is_player:
            raise ValueError(
                f"'{target.name}' is already a player character — pick another target."
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
        )
        authored = out.character

        # Identity overwrite from authored; circumstances preserved from target.
        target.name = authored.name
        target.public_sheet = authored.public_sheet
        target.backstory = authored.backstory
        target.personality = authored.personality
        target.narrative_notes = authored.narrative_notes
        target.known_context = authored.known_context
        target.private_state.goals = list(authored.private_state.goals)
        target.private_state.secrets = list(authored.private_state.secrets)
        target.private_state.intentions_enabled = (
            authored.private_state.intentions_enabled
        )
        # Keep target's location, status, incoming_directives,
        # pending_observations, current_objectives as-is — those are the
        # "circumstances" the player inherits.
        target.is_player = True

        # Drop rolling character conversation — the voice has changed.
        ckpt.character_conversations.pop(target_character_id, None)

        ckpt.session.character_bindings[target_character_id] = str(user_id)

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
    ):
        """Render the takeover prompt, dispatch to the LLM with the
        mode-appropriate response model, and return the parsed output.

        mode: 'describe' | 'suggest' | 'replace'
        picked_target: required only for mode='replace'.
        """
        from app.schemas.takeover import (
            TakeoverAuthoredOutput,
            TakeoverSuggestOutput,
        )

        if mode not in {"describe", "suggest", "replace"}:
            raise ValueError(f"Unknown takeover mode: {mode}")
        if mode == "replace" and picked_target is None:
            raise ValueError("mode='replace' requires picked_target")

        context = _build_takeover_context(ckpt, description, picked_target)

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
) -> dict[str, str]:
    """Render the context blocks the takeover prompt expects. Stays
    router-style: omniscient world state, full roster (spoiler-safe is
    NOT a constraint here — the router is authoring for itself)."""
    setting = ckpt.world_state.setting
    setting_summary = (
        f"Genre: {setting.genre}\nEra: {setting.era}\n"
        f"Tone: {setting.tone}\nPremise: {setting.premise}"
    )
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
    scene_id = locations.current_scene_id
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
    for c in ckpt.characters:
        if c.status.value == "culled":
            continue
        marker = " [player]" if c.is_player else ""
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
