"""Slash commands for the narrative-engine Discord bot.

Commands:
    /session start <name>                 — create + bind an empty session to this channel
    /session end                          — detach this channel's session
    /session resume                       — show the last scene
    /session list                         — list existing sessions
    /story list                           — list available stories
    /story start <story_id>               — load a story into the current session
    /story info <story_id>                — show briefing for a source story
    /story import <attachment> [id]       — import a master prompt
    /story delete                         — unload the story from this session
    /describe <traits>                    — set player appearance; opens scene
    /act <action>                         — submit a turn
    /status                               — summarize current state

The bot calls the engine in-process (no HTTP). Each turn runs under a
per-session lock so concurrent /act commands on the same channel serialize.
"""

from __future__ import annotations

import logging
import os
import re
import time

import discord
from discord import app_commands

from app.bot.embed import render_briefing, render_error, render_info, render_turn
from app.bot.engine_bridge import EngineBridge
from app.bot.session_map import SessionMap
from app.schemas.checkpoint import CheckpointFile

logger = logging.getLogger(__name__)

# Attachments above this are rejected in /story import to bound token cost.
# Real master prompts are ~100KB; this leaves headroom.
MAX_IMPORT_BYTES = 500_000


# Cache the parsed admin set per env-value so we only log about invalid
# entries once per unique DISCORD_ADMIN_USER_IDS string. Tests can
# mutate the env between calls; the cache key is the raw string so
# changes cause a re-parse automatically.
_ADMIN_CACHE: tuple[str, frozenset[int]] | None = None


def _is_admin(user_id: int) -> bool:
    """True if the Discord user ID is in DISCORD_ADMIN_USER_IDS (comma-sep env).

    Parses best-effort: non-integer entries are skipped with a one-time
    warning, and the remaining valid IDs still grant admin access. A typo
    in the env no longer nukes every admin's access.
    """
    global _ADMIN_CACHE
    raw = os.environ.get("DISCORD_ADMIN_USER_IDS", "").strip()
    if _ADMIN_CACHE is None or _ADMIN_CACHE[0] != raw:
        admin_ids: set[int] = set()
        bad: list[str] = []
        for part in raw.split(","):
            s = part.strip()
            if not s:
                continue
            try:
                admin_ids.add(int(s))
            except ValueError:
                bad.append(s)
        if bad:
            logger.warning(
                "DISCORD_ADMIN_USER_IDS skipped %d non-integer entr%s (%s); "
                "the %d valid ID(s) still grant admin access.",
                len(bad), "y" if len(bad) == 1 else "ies",
                ", ".join(repr(b) for b in bad),
                len(admin_ids),
            )
        _ADMIN_CACHE = (raw, frozenset(admin_ids))
    return user_id in _ADMIN_CACHE[1]


def _sanitize_story_id(raw: str) -> str:
    """Lowercase + replace non-alnum with underscores; strip leading/trailing _."""
    slug = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")
    return slug


async def _send_private(inter: discord.Interaction, text: str) -> None:
    """Post a private block of text back to the invoker as ephemeral
    followups. Chunks at Discord's 2000-char message cap, breaking on
    paragraph/newline boundaries when possible. Requires the caller to
    have already deferred (ephemeral=True) the interaction, or otherwise
    be in a state where `followup.send` is valid.
    """
    for chunk in _chunks(text, 1900):
        await inter.followup.send(chunk, ephemeral=True)


def _chunks(text: str, size: int) -> list[str]:
    """Split text into chunks of at most `size` chars, breaking on paragraph
    or newline boundaries when possible. Used for ephemeral dossier sends
    that exceed Discord's 2000-char message cap."""
    if len(text) <= size:
        return [text]
    out: list[str] = []
    remaining = text
    while len(remaining) > size:
        window = remaining[:size]
        cut = window.rfind("\n\n")
        if cut == -1 or cut < size // 2:
            cut = window.rfind("\n")
        if cut == -1:
            cut = size
        out.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        out.append(remaining)
    return out


def register(
    tree: app_commands.CommandTree,
    engine: EngineBridge,
    smap: SessionMap,
    guild: discord.Object | None,
) -> None:
    """Attach the full command set to `tree`.

    If `guild` is provided, commands are registered guild-scoped (propagate
    instantly). Without it, commands are global (up to 1h to propagate).
    """
    story_group = app_commands.Group(
        name="story",
        description="Manage the story loaded into this channel's session.",
    )

    session_group = app_commands.Group(
        name="session",
        description="Manage named saves (sessions) bound to this channel.",
    )

    # ---- /session start / end / resume / list -------------------------------

    @session_group.command(
        name="start", description="Create a named save and bind this channel.",
    )
    @app_commands.describe(name="Save name (directory under sessions/).")
    async def _session_start(inter: discord.Interaction, name: str):
        name = name.strip()
        if not name:
            await inter.response.send_message(
                "Session name cannot be empty.", ephemeral=True,
            )
            return
        existing = await smap.get(inter.channel_id)
        if existing is not None:
            await inter.response.send_message(
                f"This channel is already bound to session `{existing.session_id}`. "
                f"Run `/session end` first.",
                ephemeral=True,
            )
            return
        try:
            engine.create_empty_session(name)
        except FileExistsError as e:
            await inter.response.send_message(str(e), ephemeral=True)
            return
        await smap.upsert(
            channel_id=inter.channel_id,
            guild_id=inter.guild_id,
            session_id=name,
            owner_user_id=inter.user.id,
            story_id="",  # no story loaded yet
        )
        await inter.response.send_message(
            f"Session `{name}` created and bound. "
            f"Run `/story list` then `/story start story_id:<id>`.",
            ephemeral=True,
        )

    @session_group.command(
        name="end", description="Detach this channel from its session (files stay).",
    )
    async def _session_end(inter: discord.Interaction):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message("No session here.", ephemeral=True)
            return
        await smap.delete(inter.channel_id)
        await inter.response.send_message(
            f"Detached from `{row.session_id}`. Files kept on disk; "
            f"run `/session resume name:{row.session_id}` to rejoin.",
            ephemeral=True,
        )

    @session_group.command(
        name="resume", description="Rebind this channel to an existing session.",
    )
    @app_commands.describe(name="Saved session name.")
    async def _session_resume(inter: discord.Interaction, name: str):
        name = name.strip()
        if not name:
            await inter.response.send_message(
                "Session name cannot be empty.", ephemeral=True,
            )
            return
        existing = await smap.get(inter.channel_id)
        if existing is not None:
            await inter.response.send_message(
                f"Channel already bound to `{existing.session_id}`. "
                f"Run `/session end` first.",
                ephemeral=True,
            )
            return
        if name not in engine.list_session_ids():
            await inter.response.send_message(
                f"Unknown session `{name}`. `/session list` to see saves.",
                ephemeral=True,
            )
            return

        story_id = ""
        last_text = "(empty session — /story start to load content)"
        try:
            ckpt = engine.load_latest(name)
            story_id = ckpt.session.story_id
            if ckpt.transcript:
                last_text = ckpt.transcript[-1].assistant
            elif ckpt.opening_narrative:
                last_text = ckpt.opening_narrative
        except FileNotFoundError:
            pass  # empty session, no ckpt yet

        await smap.upsert(
            channel_id=inter.channel_id,
            guild_id=inter.guild_id,
            session_id=name,
            owner_user_id=inter.user.id,
            story_id=story_id,
        )
        await inter.response.send_message(
            f"Resumed session `{name}`"
            + (f" · story **{story_id}**" if story_id else "")
            + f".\n\n{last_text[:1800]}",
            ephemeral=True,
        )

    @session_group.command(
        name="list", description="List saved sessions.",
    )
    async def _session_list(inter: discord.Interaction):
        ids = engine.list_session_ids()
        if not ids:
            await inter.response.send_message(
                "No sessions yet. Run `/session start name:<save>`.",
                ephemeral=True,
            )
            return
        body = "\n".join(f"- `{i}`" for i in ids)
        await inter.response.send_message(
            embed=render_info("Saved sessions", body),
            ephemeral=True,
        )

    # ---- /story list --------------------------------------------------------

    @story_group.command(name="list", description="List available stories.")
    async def _list(inter: discord.Interaction):
        ids = engine.list_story_ids()
        if not ids:
            await inter.response.send_message(
                "No stories imported. Run `scripts/import_story.py <prompt.txt>` first.",
                ephemeral=True,
            )
            return
        body = "\n".join(f"- `{i}`" for i in ids)
        await inter.response.send_message(
            embed=render_info("Available stories", body),
            ephemeral=True,
        )

    # ---- /story start -------------------------------------------------------

    async def _execute_story_start(
        inter: discord.Interaction,
        row,
        story_id: str,
    ) -> None:
        """Shared body for /story start — invoked from the slash command
        directly when story_id is passed, or from the picker select
        callback when the user picks from the dropdown."""
        try:
            ckpt = engine.load_story_into_session(row.session_id, story_id)
        except FileExistsError as e:
            await inter.followup.send(embed=render_error(str(e)))
            return
        except Exception as e:
            logger.exception("load_story_into_session failed")
            await inter.followup.send(embed=render_error(
                f"`{type(e).__name__}: {e}`"
            ))
            return

        await smap.upsert(
            channel_id=inter.channel_id,
            guild_id=inter.guild_id,
            session_id=row.session_id,
            owner_user_id=row.owner_user_id,
            story_id=story_id,
        )

        briefing = render_briefing(ckpt, story_id)
        intro = (
            f"Loaded **{story_id}** into session `{row.session_id}`. "
            f"Run `/story characters` then `/join <character_id>` to claim one."
        )
        await inter.followup.send(content=intro, embed=briefing)

    class _StoryPickerView(discord.ui.View):
        """Ephemeral dropdown shown when /story start is called without a
        story_id. The invoker picks, the callback loads + briefs. Limited
        to 25 stories because that's Discord's Select cap."""
        def __init__(self, *, session_row, invoker_id: int, story_ids: list[str]):
            super().__init__(timeout=120)
            self.session_row = session_row
            self.invoker_id = invoker_id
            options = [
                discord.SelectOption(label=sid[:100], value=sid[:100])
                for sid in story_ids[:25]
            ]
            self._select = discord.ui.Select(
                placeholder="Pick a story to load...",
                options=options,
                min_values=1, max_values=1,
            )
            self._select.callback = self._on_pick
            self.add_item(self._select)

        async def interaction_check(self, check_inter: discord.Interaction) -> bool:
            if check_inter.user.id != self.invoker_id:
                await check_inter.response.send_message(
                    "This picker isn't yours.", ephemeral=True,
                )
                return False
            return True

        async def _on_pick(self, pick_inter: discord.Interaction):
            picked = self._select.values[0]
            self._select.disabled = True
            await pick_inter.response.defer(thinking=True)
            await _execute_story_start(pick_inter, self.session_row, picked)
            self.stop()

    @story_group.command(
        name="start",
        description="Load a story into the current session (pick from a list if omitted).",
    )
    @app_commands.describe(
        story_id="Optional story ID (see /story list). If omitted, shows a picker.",
    )
    async def _start(
        inter: discord.Interaction,
        story_id: str = "",
    ):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start name:<save>` first.",
                ephemeral=True,
            )
            return

        story_ids = engine.list_story_ids()
        story_id = (story_id or "").strip()

        if not story_id:
            if not story_ids:
                await inter.response.send_message(
                    "No stories imported. Run `/story import` first.",
                    ephemeral=True,
                )
                return
            view = _StoryPickerView(
                session_row=row,
                invoker_id=inter.user.id,
                story_ids=story_ids,
            )
            await inter.response.send_message(
                embed=render_info(
                    "Pick a story",
                    f"{len(story_ids)} available — pick one from the dropdown. "
                    "The briefing will post publicly once loaded.",
                ),
                view=view,
                ephemeral=True,
            )
            return

        if story_id not in story_ids:
            await inter.response.send_message(
                f"Unknown story `{story_id}`. Try `/story list`.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True)
        await _execute_story_start(inter, row, story_id)

    # ---- /story resume ------------------------------------------------------

    # /story resume and /story end are gone; see /session resume and
    # /session end below.

    # ---- /story info --------------------------------------------------------

    @story_group.command(name="info", description="Show details for a story.")
    @app_commands.describe(story_id="A story ID from /story list.")
    async def _info(inter: discord.Interaction, story_id: str):
        story_id = story_id.strip()
        if story_id not in engine.list_story_ids():
            await inter.response.send_message(
                f"Unknown story `{story_id}`. Try `/story list`.",
                ephemeral=True,
            )
            return
        try:
            ckpt = engine.load_story_ckpt(story_id)
        except Exception as e:
            logger.exception("load_story_ckpt failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        await inter.response.send_message(
            embed=render_briefing(ckpt, story_id),
            ephemeral=True,
        )

    # ---- /story characters --------------------------------------------------

    @story_group.command(
        name="characters",
        description="List characters in this channel's story (grouped, terse).",
    )
    @app_commands.describe(
        story_id="Optional story_id to inspect before starting a session.",
    )
    async def _characters(
        inter: discord.Interaction,
        story_id: str | None = None,
    ):
        # Pre-session browsing path: pristine roster from ckpt_0000.
        row = await smap.get(inter.channel_id)
        if story_id:
            story_id = story_id.strip()
            if story_id not in engine.list_story_ids():
                await inter.response.send_message(
                    f"Unknown story `{story_id}`. Try `/story list`.",
                    ephemeral=True,
                )
                return
            try:
                summaries = engine.list_story_characters(story_id)
            except Exception as e:
                logger.exception("list_story_characters failed")
                await inter.response.send_message(
                    embed=render_error(f"`{type(e).__name__}: {e}`"),
                    ephemeral=True,
                )
                return
            if not summaries:
                await inter.response.send_message(
                    "No characters in the roster.", ephemeral=True,
                )
                return
            lines = [f"**{story_id}** (source roster):"]
            for s in summaries:
                tag = f"  [{s.status}]" if s.status != "active" else ""
                lines.append(f"  {s.character_id}{tag}")
            await inter.response.send_message(
                embed=render_info("Characters", "\n".join(lines)),
                ephemeral=True,
            )
            return

        if row is None:
            await inter.response.send_message(
                "No session here and no `story_id` given. "
                "Pass one, or run `/story start` first.",
                ephemeral=True,
            )
            return

        # Live-session path: group by location vs. current scene, and
        # highlight the invoker's own binding.
        try:
            ckpt = engine.checkpoint_mgr.load_latest(row.session_id)
        except Exception as e:
            logger.exception("load_latest failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        scene_id = ckpt.world_state.locations.current_scene_id
        uid = str(inter.user.id)
        claimed_by_me = {
            cid for cid, bound in ckpt.session.character_bindings.items()
            if bound == uid
        }

        here: list[str] = []
        elsewhere: list[str] = []
        dormant: list[str] = []
        culled: list[str] = []
        my_ids: list[str] = []

        for c in ckpt.characters:
            cid = c.character_id
            status = c.status.value
            if cid in claimed_by_me:
                my_ids.append(cid)
            if status == "culled":
                culled.append(cid)
            elif status == "dormant":
                dormant.append(cid)
            else:  # active
                if c.location == scene_id:
                    here.append(cid)
                else:
                    elsewhere.append(cid)

        def _block(title: str, ids: list[str]) -> str:
            if not ids:
                return f"**{title}**:\n  (none)"
            lines = [f"**{title}**:"]
            for i in ids:
                suffix = " ← you" if i in claimed_by_me else ""
                lines.append(f"  {i}{suffix}")
            return "\n".join(lines)

        body_parts = [
            _block(f"Here ({scene_id or 'no scene'})", here),
            _block("Claimed by you", my_ids),
            _block("Active (elsewhere)", elsewhere),
            _block("Dormant", dormant),
            _block("Culled", culled),
        ]
        await inter.response.send_message(
            embed=render_info(
                f"Characters · `{row.story_id}`",
                "\n\n".join(body_parts),
            ),
            ephemeral=True,
        )

    # ---- /story import ------------------------------------------------------

    @story_group.command(
        name="import",
        description="Import a master prompt attachment as a new story.",
    )
    @app_commands.describe(
        attachment="A .txt, .md, or .markdown file containing the master prompt.",
        story_id="Optional override for the story id (default: derived from filename).",
    )
    async def _import(
        inter: discord.Interaction,
        attachment: discord.Attachment,
        story_id: str | None = None,
    ):
        # Size + type guardrails.
        if attachment.size > MAX_IMPORT_BYTES:
            await inter.response.send_message(
                f"Attachment too large ({attachment.size} bytes). "
                f"Max is {MAX_IMPORT_BYTES} bytes (~500KB).",
                ephemeral=True,
            )
            return
        fname = (attachment.filename or "").lower()
        if not fname.endswith((".txt", ".md", ".markdown")):
            await inter.response.send_message(
                "Only `.txt`, `.md`, or `.markdown` attachments are supported.",
                ephemeral=True,
            )
            return
        # Content-type check when Discord populates it — catches binaries
        # renamed to .txt. Some clients omit content_type entirely, so
        # missing is allowed; the extension check is the primary guard
        # and the UTF-8 decode below is the final defense.
        content_type = (attachment.content_type or "").lower().split(";")[0].strip()
        if content_type and not content_type.startswith("text/"):
            await inter.response.send_message(
                f"Attachment content_type `{content_type}` isn't a text type. "
                f"Upload the raw master prompt as a text file, not a rendered "
                f"document or archive.",
                ephemeral=True,
            )
            return

        # Derive story_id if not given.
        if not story_id:
            base = os.path.splitext(attachment.filename)[0]
            story_id = _sanitize_story_id(base)
        else:
            story_id = _sanitize_story_id(story_id)
        if not story_id:
            await inter.response.send_message(
                "Could not derive a story_id. Pass `story_id` explicitly.",
                ephemeral=True,
            )
            return

        if story_id in engine.list_story_ids():
            await inter.response.send_message(
                f"A story with id `{story_id}` already exists. "
                f"Pick a different id or have an admin run `/story delete {story_id}` first.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True)

        # Download attachment and import. 3 sequential LLM calls take 3–8
        # minutes; Discord's followup window is 15 min so we're comfortable.
        try:
            raw = await attachment.read()
            source_text = raw.decode("utf-8")
        except UnicodeDecodeError:
            await inter.followup.send(
                embed=render_error(
                    "Attachment is not valid UTF-8. Save the file as plain text."
                )
            )
            return
        except Exception as e:
            logger.exception("attachment download failed")
            await inter.followup.send(
                embed=render_error(f"Attachment download failed: `{e}`")
            )
            return

        logger.info(
            "Importing story %s from attachment %r (%d bytes) by %s",
            story_id, attachment.filename, attachment.size, inter.user.display_name,
        )
        start = time.monotonic()

        # Notify the invoker when the background preservation analysis
        # finishes so they don't have to poll the file to know it's
        # done. Posted as an ephemeral followup on the original import
        # interaction — the 15-minute followup window covers typical
        # analysis latency. Closures capture `inter` + story_id by
        # reference; Discord keeps the webhook alive for the window.
        async def _notify(analysis, err):
            try:
                if err is not None:
                    await inter.followup.send(
                        f"Preservation analysis for `{story_id}` failed: "
                        f"`{type(err).__name__}: {err}` — the import itself "
                        f"succeeded, only the coverage audit is missing.",
                        ephemeral=True,
                    )
                    return
                if analysis is None:
                    return
                summary = (
                    f"Preservation analysis for `{story_id}` is complete — "
                    f"coverage **{analysis.coverage_rating}**, "
                    f"{len(analysis.dropped_topics)} dropped topic(s), "
                    f"{len(analysis.compressed_topics)} compressed. "
                    f"Run `/story info {story_id}` for the full picture."
                )
                await inter.followup.send(summary, ephemeral=True)
            except Exception:
                logger.exception("analysis-complete notify failed")

        try:
            ckpt = await engine.import_story(
                source_text, story_id,
                on_analysis_complete=_notify,
            )
        except FileExistsError as e:
            await inter.followup.send(embed=render_error(str(e)))
            return
        except Exception as e:
            logger.exception("import_story failed")
            await inter.followup.send(
                embed=render_error(f"Import failed: `{type(e).__name__}: {e}`")
            )
            return

        elapsed = time.monotonic() - start
        logger.info(
            "Imported %s in %.1fs: %d chars, %d scenes, %d characters",
            story_id, elapsed,
            len(ckpt.opening_narrative),
            len(ckpt.world_state.locations.scene_graph),
            len(ckpt.characters),
        )

        # No briefing here — the checkpoint still has PLAYER_NAME placeholders
        # that only get substituted at /story start time.
        intro = (
            f"Imported **{story_id}** in {int(elapsed)}s — "
            f"{len(ckpt.characters)} characters, "
            f"{len(ckpt.world_state.locations.scene_graph)} scenes. "
            f"Start with `/session start name:<save>` then `/story start story_id:{story_id}`."
        )
        await inter.followup.send(content=intro)

    # ---- /story delete -----------------------------------------------------

    @story_group.command(
        name="delete",
        description="Unload the current story from this session, leaving it empty.",
    )
    async def _story_delete(inter: discord.Interaction):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here.", ephemeral=True,
            )
            return
        try:
            removed = engine.unload_story_from_session(row.session_id)
        except FileNotFoundError as e:
            await inter.response.send_message(str(e), ephemeral=True)
            return
        except Exception as e:
            logger.exception("unload_story_from_session failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        # Drop story_id from the channel mapping but KEEP the binding —
        # the user can /story start a different story without /session
        # end/start. Character bindings and CLI claims reset because
        # the checkpoint state is gone.
        await smap.upsert(
            channel_id=inter.channel_id,
            guild_id=inter.guild_id,
            session_id=row.session_id,
            owner_user_id=row.owner_user_id,
            story_id="",
        )
        await inter.response.send_message(
            f"Unloaded story from `{row.session_id}` ({removed} files removed). "
            f"Run `/story start story_id:<id>` to load a new one.",
            ephemeral=True,
        )

    # ---- /join --------------------------------------------------------------

    async def _join_custom_describe_submit(
        submit_inter: discord.Interaction,
        session_id: str,
        description: str,
    ) -> None:
        """Shared post-submit path for the custom-describe modal fired from
        /join with no args. Defers ephemeral, spawns a custom character via
        the engine, and sends the dossier + prompt to /describe."""
        await submit_inter.response.defer(thinking=True, ephemeral=True)
        try:
            new_char = await engine.create_custom_character(
                session_id, submit_inter.user.id, description,
            )
        except ValueError as e:
            await submit_inter.followup.send(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("modal create_custom_character failed")
            await submit_inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        try:
            dossier = engine.build_character_dossier(
                session_id, new_char.character_id,
            )
        except Exception as e:
            logger.exception("build_character_dossier failed")
            dossier = f"(dossier unavailable: `{e}`)"

        msg = (
            f"You are now **{new_char.name}** (`{new_char.character_id}`). "
            "Run `/describe` with a name and appearance to set how the "
            "world sees you, then `/act` to play. Dossier below."
        )
        await submit_inter.followup.send(
            embed=render_info("Joined", msg), ephemeral=True,
        )
        await _send_private(submit_inter, dossier)

    class _JoinCustomModal(discord.ui.Modal, title="Create a custom character"):
        description = discord.ui.TextInput(
            label="Character concept",
            style=discord.TextStyle.paragraph,
            placeholder="Role, background, vibe — freeform.",
            max_length=1000,
            required=True,
        )

        def __init__(self, session_id: str):
            super().__init__()
            self._session_id = session_id

        async def on_submit(self, modal_inter: discord.Interaction):
            desc = (self.description.value or "").strip()
            if not desc:
                await modal_inter.response.send_message(
                    "Description cannot be empty.", ephemeral=True,
                )
                return
            await _join_custom_describe_submit(
                modal_inter, self._session_id, desc,
            )

    @tree.command(
        name="join",
        description="Claim a character, or open a custom-character prompt if no id is given.",
        guild=guild,
    )
    @app_commands.describe(
        character_id="Optional character_id (see /story characters). Omit to create a custom character.",
    )
    async def _join(inter: discord.Interaction, character_id: str = ""):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. `/session start` then `/story start` first.", ephemeral=True,
            )
            return

        character_id = (character_id or "").strip()
        if not character_id:
            # No id → open the custom-describe modal (shortcut for
            # /join_custom mode:describe). Modal must be the direct
            # response — cannot be shown after defer.
            await inter.response.send_modal(_JoinCustomModal(row.session_id))
            return

        # Defer up front — binding + dossier-building both do checkpoint
        # I/O plus a chunked ephemeral send, any of which could blow past
        # the 3s interaction window on a large checkpoint or a slow network.
        await inter.response.defer(thinking=True, ephemeral=True)

        try:
            ckpt = engine.takeover(row.session_id, character_id, inter.user.id)
        except ValueError as e:
            await inter.followup.send(embed=render_error(str(e)), ephemeral=True)
            return
        except Exception as e:
            logger.exception("takeover failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        char = next(
            (c for c in ckpt.characters if c.character_id == character_id), None
        )
        char_name = char.name if char else character_id

        try:
            dossier = engine.build_character_dossier(
                row.session_id, character_id
            )
        except Exception as e:
            logger.exception("build_character_dossier failed")
            dossier = f"(dossier unavailable: `{e}`)"

        logger.info(
            "Bound %s to %s in %s",
            inter.user.display_name, character_id, row.session_id,
        )

        lines = [f"You are now playing **{char_name}** (`{character_id}`)."]
        if char and char.status.value == "dormant":
            lines.append(
                "⚠️ This character is currently **dormant** — they're off-screen "
                "in the fiction and can't `/act` until the story brings them "
                "back. Your binding is saved and you'll be ready when they return."
            )
        lines.append("Your private dossier is below. Read it, then `/describe` to set your appearance and `/act` to play.")
        await inter.followup.send(
            embed=render_info("Joined", "\n".join(lines)),
            ephemeral=True,
        )
        await _send_private(inter, dossier)

    # ---- /leave -------------------------------------------------------------

    @tree.command(
        name="leave",
        description="Release your character binding in this channel's story.",
        guild=guild,
    )
    async def _leave(inter: discord.Interaction):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here.", ephemeral=True,
            )
            return

        # Shared endpoint: synthesize personality (if empty) then unbind.
        # Matches CLI /leave behavior — the agent inherits voice.
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            freed = await engine.leave_character(row.session_id, inter.user.id)
        except Exception as e:
            logger.exception("leave_character failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        if freed is None:
            await inter.followup.send(
                "You weren't bound to a character.", ephemeral=True,
            )
            return
        await inter.followup.send(
            f"Released `{freed}`. Other players can now `/join` them.",
            ephemeral=True,
        )

    # ---- /join_custom -------------------------------------------------------

    @tree.command(
        name="join_custom",
        description="Join with a player-authored character (describe or replace).",
        guild=guild,
    )
    @app_commands.describe(
        mode="'describe' to spawn a new character; 'replace' to get candidate NPCs to graft onto.",
        description="Your character concept (free-form).",
    )
    @app_commands.choices(mode=[
        app_commands.Choice(name="describe", value="describe"),
        app_commands.Choice(name="replace", value="replace"),
    ])
    async def _join_custom(
        inter: discord.Interaction,
        mode: app_commands.Choice[str],
        description: str,
    ):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. `/session start` then `/story start` first.", ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True, ephemeral=True)

        description = description.strip()
        if not description:
            await inter.followup.send(
                embed=render_error("Description cannot be empty."),
                ephemeral=True,
            )
            return

        if mode.value == "describe":
            try:
                new_char = await engine.create_custom_character(
                    row.session_id, inter.user.id, description,
                )
            except ValueError as e:
                await inter.followup.send(
                    embed=render_error(str(e)), ephemeral=True,
                )
                return
            except Exception as e:
                logger.exception("create_custom_character failed")
                await inter.followup.send(
                    embed=render_error(f"`{type(e).__name__}: {e}`"),
                    ephemeral=True,
                )
                return

            try:
                dossier = engine.build_character_dossier(
                    row.session_id, new_char.character_id,
                )
            except Exception as e:
                logger.exception("build_character_dossier failed")
                dossier = f"(dossier unavailable: `{e}`)"

            msg = (
                f"You are now **{new_char.name}** (`{new_char.character_id}`). "
                "Run `/describe` with a name and appearance when you're ready. "
                "Dossier below."
            )
            await inter.followup.send(
                embed=render_info("Joined", msg), ephemeral=True,
            )
            await _send_private(inter, dossier)
            return

        # mode == "replace"
        try:
            result = await engine.suggest_replacement_targets(
                row.session_id, description,
            )
        except ValueError as e:
            await inter.followup.send(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("suggest_replacement_targets failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        candidates = result.get("candidates") or []
        preamble = (result.get("preamble") or "").strip()

        if not candidates:
            await inter.followup.send(
                embed=render_info(
                    "No candidates",
                    (preamble + "\n\n" if preamble else "")
                    + "The router found no replaceable NPCs for that concept. "
                    "Try `/join_custom mode:describe` instead.",
                ),
                ephemeral=True,
            )
            return

        lines: list[str] = []
        if preamble:
            lines.append(preamble)
            lines.append("")
        lines.append(
            "**Candidates** — run `/pick_replacement` with one of these character_ids "
            "(re-paste your description):"
        )
        for c in candidates:
            cid = c.get("character_id", "?")
            name = c.get("name", cid)
            rationale = c.get("fit_rationale", "")
            lines.append(f"- `{cid}` — {name} — {rationale}")

        await inter.followup.send(
            embed=render_info("Replacement candidates", "\n".join(lines)),
            ephemeral=True,
        )

    # ---- /pick_replacement --------------------------------------------------

    @tree.command(
        name="pick_replacement",
        description="Graft your custom character onto one of the candidates from /join_custom.",
        guild=guild,
    )
    @app_commands.describe(
        character_id="Target character_id from /join_custom candidate list.",
        description="Same character concept you passed to /join_custom.",
    )
    async def _pick_replacement(
        inter: discord.Interaction,
        character_id: str,
        description: str,
    ):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. `/session start` then `/story start` first.", ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True, ephemeral=True)

        character_id = character_id.strip()
        description = description.strip()
        if not character_id or not description:
            await inter.followup.send(
                embed=render_error(
                    "Both `character_id` and `description` are required."
                ),
                ephemeral=True,
            )
            return

        try:
            new_char = await engine.replace_with_custom(
                row.session_id, inter.user.id, character_id, description,
            )
        except ValueError as e:
            await inter.followup.send(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("replace_with_custom failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        try:
            dossier = engine.build_character_dossier(
                row.session_id, new_char.character_id,
            )
        except Exception as e:
            logger.exception("build_character_dossier failed")
            dossier = f"(dossier unavailable: `{e}`)"

        msg = (
            f"You are now **{new_char.name}** "
            f"(replaced `{character_id}`). Dossier below."
        )
        await inter.followup.send(
            embed=render_info("Joined", msg), ephemeral=True,
        )
        await _send_private(inter, dossier)

    # ---- /character ---------------------------------------------------------

    @tree.command(
        name="character",
        description="Show the private dossier for a character in this channel's story.",
        guild=guild,
    )
    @app_commands.describe(
        character_id="The character_id (see /story characters).",
    )
    async def _character(inter: discord.Interaction, character_id: str):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. `/session start` then `/story start` first.", ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True, ephemeral=True)

        character_id = character_id.strip()
        try:
            dossier = engine.build_character_dossier(
                row.session_id, character_id,
            )
        except ValueError as e:
            await inter.followup.send(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("build_character_dossier failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        await _send_private(inter, dossier)

    # ---- /describe ----------------------------------------------------------

    @tree.command(
        name="describe",
        description="Set your character's name and/or appearance. Opens the story on first use.",
        guild=guild,
    )
    @app_commands.describe(
        name="The name your character will use in-story (optional).",
        appearance="Height, build, clothing, notable features — freeform (optional).",
    )
    async def _describe(
        inter: discord.Interaction,
        name: str = "",
        appearance: str = "",
    ):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start` then `/story start`.",
                ephemeral=True,
            )
            return

        name = name.strip()
        appearance = appearance.strip()
        if not name and not appearance:
            await inter.response.send_message(
                "Provide at least one of `name` or `appearance`.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True)

        # Each player describes their own bound character. /describe without
        # a binding is ambiguous in multi-player — refuse.
        binding = engine.get_user_binding(row.session_id, inter.user.id)
        if binding is None:
            await inter.followup.send(
                embed=render_error(
                    "You aren't bound to a character. Run `/story characters` "
                    "and then `/join <character_id>` first."
                )
            )
            return

        try:
            ckpt = engine.set_character_identity(
                row.session_id, binding,
                name=name or None,
                appearance=appearance or None,
            )
        except Exception as e:
            logger.exception("set_character_identity failed")
            await inter.followup.send(embed=render_error(
                f"`{type(e).__name__}: {e}`"
            ))
            return

        # If this is the first describe AND no turns have happened yet, fire a
        # synthetic arrival action so the narrator renders the opening scene
        # with the description in hand. Otherwise this is just a mid-story
        # update and we don't trigger anything.
        is_pre_play = not ckpt.narrator_conversation

        changed_bits = []
        if name:
            changed_bits.append(f"name: **{name}**")
        if appearance:
            changed_bits.append(f"appearance: *{appearance}*")
        changed = " · ".join(changed_bits)

        if not is_pre_play:
            await inter.followup.send(
                embed=render_info(
                    "Character updated",
                    f"{changed}\n\nThis takes effect on your next `/act`.",
                )
            )
            return

        # Pre-play: fire the opening turn using the OOC meta-channel. The
        # EventRouter prompt recognizes fully-parenthesized input as an
        # author's directive rather than an in-character attempt, so
        # `(begin)` cleanly maps to "open the story from the scene's actual
        # start per the opening_directive" without the narrator compressing
        # arrival into hallucinated subtext.
        arrival_action = "(begin)"
        logger.info(
            "Describe+open for %s by %s: %s",
            row.session_id, inter.user.display_name, changed,
        )

        try:
            response = await engine.run_turn(
                session_id=row.session_id,
                user_input=arrival_action,
                acting_character_id=binding,
                debug=True,
            )
        except Exception as e:
            logger.exception("opening run_turn failed")
            await inter.followup.send(embed=render_error(
                f"`{type(e).__name__}: {e}`"
            ))
            return

        if response.debug is not None:
            for lat in response.debug.latencies:
                logger.info(
                    "  phase=%-18s %5.0fms (render %4.0fms)  "
                    "in=%4d out=%4d cache_read=%5d cache_write=%5d  (%s)",
                    lat.phase, lat.duration_ms, lat.prompt_render_ms,
                    lat.input_tokens, lat.output_tokens,
                    lat.cache_read_input_tokens, lat.cache_creation_input_tokens,
                    lat.model,
                )

        embeds = render_turn(
            output_text=response.output_text,
            turn_index=response.turn_index,
            story_id=row.story_id,
        )
        intro = (
            f"*{traits}*\n\n"
            f"The scene opens. Use `/act <action>` from here on."
        )
        await inter.followup.send(content=intro, embeds=embeds)

    # ---- /act ---------------------------------------------------------------

    @tree.command(
        name="act",
        description="Take a turn in the current story.",
        guild=guild,
    )
    @app_commands.describe(action="What your character does or says.")
    async def _act(inter: discord.Interaction, action: str):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start` then `/story start`.",
                ephemeral=True,
            )
            return

        binding = engine.get_user_binding(row.session_id, inter.user.id)
        if binding is None:
            await inter.response.send_message(
                "You're not bound to a character in this story. "
                "Run `/story characters` then `/join <character_id>`.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True)
        start = time.monotonic()

        try:
            response = await engine.run_turn(
                session_id=row.session_id,
                user_input=action,
                acting_character_id=binding,
                debug=True,   # keeps per-phase cache/latency metrics server-side
            )
        except Exception as e:
            logger.exception("run_turn failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`")
            )
            return

        elapsed = time.monotonic() - start
        logger.info(
            "Turn %d completed for %s by %s in %.1fs (%d chars): %r",
            response.turn_index, row.session_id, inter.user.display_name,
            elapsed, len(response.output_text), action[:120],
        )
        if response.debug is not None:
            for lat in response.debug.latencies:
                logger.info(
                    "  phase=%-18s %5.0fms  in=%4d out=%4d cache_read=%5d cache_write=%5d  (%s)",
                    lat.phase,
                    lat.duration_ms,
                    lat.input_tokens,
                    lat.output_tokens,
                    lat.cache_read_input_tokens,
                    lat.cache_creation_input_tokens,
                    lat.model,
                )

        embeds = render_turn(
            output_text=response.output_text,
            turn_index=response.turn_index,
            story_id=row.story_id,
        )
        await inter.followup.send(embeds=embeds)

    # ---- /status ------------------------------------------------------------

    @tree.command(
        name="status",
        description="Show the current state of this channel's story.",
        guild=guild,
    )
    async def _status(inter: discord.Interaction):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No story here.", ephemeral=True,
            )
            return

        try:
            ckpt: CheckpointFile = engine.load_latest(row.session_id)
        except FileNotFoundError:
            # Orphaned mapping — purge and tell the user so they can
            # start fresh without bumping into a stale binding.
            await smap.delete(inter.channel_id)
            await inter.response.send_message(
                f"Session `{row.session_id}` had no checkpoint on disk; "
                f"the channel mapping has been purged. Run `/session start` "
                f"then `/story start` to begin fresh.",
                ephemeral=True,
            )
            return

        scene_id = ckpt.world_state.locations.current_scene_id
        scene_name = scene_id
        scene = ckpt.world_state.locations.scene_graph.get(scene_id, {})
        if isinstance(scene, dict) and scene.get("name"):
            scene_name = scene["name"]

        bound_ids = set(ckpt.session.character_bindings.keys())
        present = [
            c.name for c in ckpt.characters
            if c.location == scene_id and c.status.value == "active"
            and c.character_id not in bound_ids
        ]

        bindings_lines: list[str] = []
        for char_id, user_id in ckpt.session.character_bindings.items():
            char = next(
                (c for c in ckpt.characters if c.character_id == char_id), None
            )
            char_name = char.name if char else char_id
            bindings_lines.append(f"• <@{user_id}> — **{char_name}** (`{char_id}`)")
        if not bindings_lines:
            bindings_lines.append("• (no players bound — `/join` to claim one)")

        body_lines = [
            f"**Story:** `{row.story_id}`",
            f"**Turn:** {ckpt.session.turn_index}",
            f"**Scene:** {scene_name}",
            f"**Present (NPCs):** {', '.join(present) if present else 'no one else'}",
            "",
            "**Players:**",
            *bindings_lines,
        ]
        await inter.response.send_message(
            embed=render_info("Session status", "\n".join(body_lines)),
            ephemeral=True,
        )

    # ---- /clear (admin-gated) -----------------------------------------------

    @tree.command(
        name="clear",
        description="Delete recent messages in this channel. Admin-only.",
        guild=guild,
    )
    @app_commands.describe(
        count="How many messages to delete (1-1000, default 100).",
    )
    async def _clear(
        inter: discord.Interaction,
        count: app_commands.Range[int, 1, 1000] = 100,
    ):
        if not _is_admin(inter.user.id):
            await inter.response.send_message(
                "Only admins can clear the channel.", ephemeral=True,
            )
            return

        # DM channels don't allow bulk-delete; TextChannel does.
        channel = inter.channel
        if not isinstance(channel, discord.TextChannel):
            await inter.response.send_message(
                "`/clear` only works in text channels, not DMs or threads.",
                ephemeral=True,
            )
            return

        # Check bot permissions before attempting — prevents a half-baked error.
        me = channel.guild.me
        perms = channel.permissions_for(me)
        if not perms.manage_messages:
            await inter.response.send_message(
                "I need the **Manage Messages** permission in this channel to "
                "delete messages. Ask an admin to grant it.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True, ephemeral=True)

        try:
            # purge() uses bulk_delete for messages < 14 days and falls back
            # to per-message delete for older ones (slow, rate-limited).
            deleted = await channel.purge(limit=count)
        except discord.Forbidden:
            await inter.followup.send(
                "Forbidden — Manage Messages permission got revoked mid-call.",
                ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("channel.purge failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        logger.info(
            "Cleared %d messages in #%s by %s",
            len(deleted), channel.name, inter.user.display_name,
        )
        await inter.followup.send(
            f"Deleted {len(deleted)} message(s).",
            ephemeral=True,
        )

    # ---- /settings ----------------------------------------------------------

    settings_group = app_commands.Group(
        name="settings",
        description="View or change experimental per-session settings.",
    )

    @settings_group.command(
        name="list",
        description="Show all experimental settings for this session.",
    )
    async def _settings_list(inter: discord.Interaction):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start` first.", ephemeral=True,
            )
            return
        try:
            view = engine.list_settings(row.session_id)
        except Exception as e:
            logger.exception("list_settings failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        if not view:
            await inter.response.send_message(
                "No tunable settings registered.", ephemeral=True,
            )
            return

        lines: list[str] = []
        for s in view:
            marker = "" if s["value"] == s["default"] else "  *(modified)*"
            lines.append(
                f"**`{s['key']}`** · `{s['rendered_value']}`  "
                f"(default `{s['rendered_default']}`){marker}\n"
                f"  {s['description']}"
            )
        await inter.response.send_message(
            embed=render_info("Session settings", "\n\n".join(lines)),
            ephemeral=True,
        )

    @settings_group.command(
        name="set",
        description="Change an experimental setting for this session.",
    )
    @app_commands.describe(
        key="The setting key (see /settings list).",
        value="New value. Booleans accept true/false, on/off, yes/no, 1/0.",
    )
    async def _settings_set(
        inter: discord.Interaction,
        key: str,
        value: str,
    ):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start` first.", ephemeral=True,
            )
            return
        try:
            new_value = engine.set_setting(row.session_id, key, value)
        except KeyError:
            valid = ", ".join(engine.known_setting_keys()) or "(none)"
            await inter.response.send_message(
                embed=render_error(
                    f"Unknown setting `{key}`. Valid keys: {valid}"
                ),
                ephemeral=True,
            )
            return
        except ValueError as e:
            await inter.response.send_message(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("set_setting failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        await inter.response.send_message(
            f"Setting `{key}` → `{new_value}`.",
            ephemeral=True,
        )

    # v11-r3d: /abort_beat admin recovery command. A wedged Cat II event
    # (responder disconnected, session frozen) can only be cleared by an
    # admin invoking this command; it force-releases the scene's slot
    # state and abandons any open Cat II events in that scene. The
    # event LOG is preserved — this does not rewrite history, just
    # unblocks the next /act.
    @tree.command(
        name="abort_beat",
        description="[Admin] Force-unwedge the current scene's beat.",
        guild=guild,
    )
    async def _abort_beat(inter: discord.Interaction):
        if not _is_admin(inter.user.id):
            await inter.response.send_message(
                "Admin-only command.", ephemeral=True,
            )
            return
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session bound to this channel.", ephemeral=True,
            )
            return
        try:
            from app.engine.turn_loop import abort_scene
            ckpt = engine.load_latest(row.session_id)
            scene_id = ckpt.world_state.locations.current_scene_id
            dropped = abort_scene(ckpt, scene_id)
            engine.checkpoint_mgr.save(ckpt)
        except Exception as e:
            logger.exception("abort_beat failed")
            await inter.response.send_message(
                f"abort failed: `{type(e).__name__}: {e}`", ephemeral=True,
            )
            return
        await inter.response.send_message(
            f"Scene `{scene_id}` released. {dropped} open Cat II event(s) "
            f"abandoned. Next /act can claim the slot.",
            ephemeral=True,
        )

    # Attach the /story and /settings groups to the tree last, once
    # their subcommands are defined.
    if guild is not None:
        tree.add_command(session_group, guild=guild)
        tree.add_command(story_group, guild=guild)
        tree.add_command(settings_group, guild=guild)
    else:
        tree.add_command(session_group)
        tree.add_command(story_group)
        tree.add_command(settings_group)
