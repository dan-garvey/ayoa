"""Slash commands for the narrative-engine Discord bot.

Commands:
    /story list                           — list available stories
    /story start <story_id> <name>        — create a session in this channel
    /story resume                         — show the last scene
    /story end                            — detach this channel's session
    /story info <story_id>                — show briefing for a source story
    /story import <attachment> [id]       — import a master prompt
    /story delete <story_id>              — admin-only, delete a story
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


def _chunks(text: str, size: int) -> list[str]:
    """Split text into chunks of at most `size` chars, breaking on paragraph
    or newline boundaries when possible. Used for DM'ing dossiers that blow
    through Discord's 2000-char message cap."""
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
        description="Manage the interactive fiction session in this channel.",
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

    @story_group.command(name="start", description="Begin a story in this channel.")
    @app_commands.describe(
        story_id="The story ID (see /story list).",
    )
    async def _start(
        inter: discord.Interaction,
        story_id: str,
    ):
        existing = await smap.get(inter.channel_id)
        if existing is not None:
            await inter.response.send_message(
                f"This channel is already bound to session `{existing.session_id}`. "
                f"Run `/story end` first if you want to switch.",
                ephemeral=True,
            )
            return

        if story_id not in engine.list_story_ids():
            await inter.response.send_message(
                f"Unknown story `{story_id}`. Try `/story list`.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True)

        session_id = engine.session_id_for_channel(inter.channel_id, story_id)

        # Mirror the CLI: creating a session does NOT assign a character
        # or run the personalize rename. Single-protagonist stories have
        # one authored is_player slot the caller can /join. Multi-slot
        # stories have several; the creator chooses via /join or
        # /join_custom like any other player. No character_name arg.
        try:
            ckpt = await engine.create_session(
                story_id=story_id,
                session_id=session_id,
                player_display_name="",
                creator_user_id=None,
            )
        except Exception as e:
            logger.exception("create_session failed")
            await inter.followup.send(embed=render_error(
                f"`{type(e).__name__}: {e}`"
            ))
            return

        await smap.upsert(
            channel_id=inter.channel_id,
            guild_id=inter.guild_id,
            session_id=session_id,
            owner_user_id=inter.user.id,
            story_id=story_id,
        )

        briefing = render_briefing(ckpt, story_id)
        intro = (
            f"Session started for **{story_id}**. "
            f"Run `/story characters` to see who's available, "
            f"then `/join <character_id>` to claim one."
        )
        await inter.followup.send(content=intro, embed=briefing)

    # ---- /story resume ------------------------------------------------------

    @story_group.command(name="resume", description="Re-anchor to the existing session.")
    async def _resume(inter: discord.Interaction):
        existing = await smap.get(inter.channel_id)
        if existing is None:
            await inter.response.send_message(
                "No session here. Try `/story start <id>`.",
                ephemeral=True,
            )
            return

        try:
            ckpt = engine.load_latest(existing.session_id)
        except FileNotFoundError:
            # Orphaned mapping — the on-disk checkpoint was deleted out
            # from under us. Purge the row so the next /story start on
            # this channel doesn't hit "session already bound" on the
            # way in.
            await smap.delete(inter.channel_id)
            await inter.response.send_message(
                f"Session `{existing.session_id}` had no checkpoint on disk; "
                f"the channel mapping has been purged. Run `/story start "
                f"<story_id> \"<Name>\"` to begin fresh.",
                ephemeral=True,
            )
            return

        last_text = ""
        if ckpt.transcript:
            last_text = ckpt.transcript[-1].assistant
        if not last_text:
            last_text = ckpt.opening_narrative or "(nothing to show)"

        embeds = render_turn(
            output_text=last_text,
            turn_index=ckpt.session.turn_index,
            story_id=existing.story_id,
        )
        await inter.response.send_message(
            content=f"Resuming **{existing.story_id}**. Last scene:",
            embeds=embeds,
        )

    # ---- /story end ---------------------------------------------------------

    @story_group.command(name="end", description="Detach this channel from its session.")
    async def _end(inter: discord.Interaction):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here.", ephemeral=True,
            )
            return
        await smap.delete(inter.channel_id)
        await inter.response.send_message(
            f"Detached from `{row.session_id}`. Checkpoint files are kept on disk.",
            ephemeral=True,
        )

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

        # DM the invoker when the background preservation analysis
        # finishes so they don't have to poll the file to know it's
        # done. Closures capture `inter.user` + story_id by reference,
        # which is fine — the task fires with whatever those refs point
        # at the moment analysis completes.
        async def _notify(analysis, err):
            try:
                if err is not None:
                    await inter.user.send(
                        f"Preservation analysis for `{story_id}` failed: "
                        f"`{type(err).__name__}: {err}` — the import itself "
                        f"succeeded, only the coverage audit is missing."
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
                await inter.user.send(summary)
            except discord.Forbidden:
                logger.warning(
                    "Could not DM %s about analysis completion (server DMs off)",
                    inter.user.display_name,
                )
            except Exception:
                logger.exception("analysis-complete DM failed")

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
            f"Start a session with `/story start {story_id} \"<Your Name>\"`."
        )
        await inter.followup.send(content=intro)

    # ---- /story delete (admin-gated) ---------------------------------------

    @story_group.command(
        name="delete",
        description="Delete a story and all sessions derived from it. Admin-only.",
    )
    @app_commands.describe(story_id="The story ID to delete.")
    async def _delete(inter: discord.Interaction, story_id: str):
        if not _is_admin(inter.user.id):
            await inter.response.send_message(
                "Only admins can delete stories. Set `DISCORD_ADMIN_USER_IDS` "
                "in the bot's env to grant access.",
                ephemeral=True,
            )
            return
        story_id = story_id.strip()
        if story_id not in engine.list_story_ids():
            await inter.response.send_message(
                f"Unknown story `{story_id}`.", ephemeral=True,
            )
            return
        try:
            sessions_removed, files_removed = engine.delete_story(story_id)
        except Exception as e:
            logger.exception("delete_story failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        # Any channel-mapping rows pointing at derived sessions become stale;
        # since session_id patterns are deterministic, just leave them — a
        # future /story start or /story resume on those channels will error
        # cleanly with "no checkpoint found" and the user can /story end them.
        await inter.response.send_message(
            f"Deleted `{story_id}` and {sessions_removed} derived session dir(s) "
            f"({files_removed} files).",
            ephemeral=True,
        )

    # ---- /join --------------------------------------------------------------

    @tree.command(
        name="join",
        description="Claim an unbound character so you can /act as them.",
        guild=guild,
    )
    @app_commands.describe(
        character_id="The character_id to claim (see /story characters).",
    )
    async def _join(inter: discord.Interaction, character_id: str):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. `/story start` first.", ephemeral=True,
            )
            return

        # Defer up front — binding + dossier-building both do checkpoint
        # I/O plus a chunked DM, any of which could blow past the 3s
        # interaction window on a large checkpoint or a slow network.
        await inter.response.defer(thinking=True, ephemeral=True)

        character_id = character_id.strip()
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

        # Build dossier. Send as DM; if DM fails, tell them in-channel.
        try:
            dossier = engine.build_character_dossier(
                row.session_id, character_id
            )
        except Exception as e:
            logger.exception("build_character_dossier failed")
            dossier = f"(dossier unavailable: `{e}`)"

        dm_ok = True
        try:
            # Discord DMs cap at 2000 chars per message; chunk the dossier.
            for chunk in _chunks(dossier, 1900):
                await inter.user.send(chunk)
        except discord.Forbidden:
            dm_ok = False
        except Exception:
            logger.exception("DM send failed")
            dm_ok = False

        logger.info(
            "Bound %s to %s in %s (DM ok=%s)",
            inter.user.display_name, character_id, row.session_id, dm_ok,
        )

        lines = [f"You are now playing **{char_name}** (`{character_id}`)."]
        if char and char.status.value == "dormant":
            lines.append(
                "⚠️ This character is currently **dormant** — they're off-screen "
                "in the fiction and can't `/act` until the story brings them "
                "back. Your binding is saved and you'll be ready when they return."
            )
        if dm_ok:
            lines.append("I DM'd you your private dossier — read it before you /act.")
        else:
            lines.append(
                "⚠️ I couldn't DM your dossier (server DMs disabled). "
                "Enable DMs from this server and run `/join` again to receive it."
            )
        lines.append("Use `/describe` to set your appearance, then `/act` to play.")
        await inter.followup.send(
            embed=render_info("Joined", "\n".join(lines)),
            ephemeral=True,
        )

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
                "No session here. `/story start` first.", ephemeral=True,
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

            dm_ok = True
            try:
                dossier = engine.build_character_dossier(
                    row.session_id, new_char.character_id,
                )
                for chunk in _chunks(dossier, 1900):
                    await inter.user.send(chunk)
            except discord.Forbidden:
                dm_ok = False
            except Exception:
                logger.exception("dossier DM failed")
                dm_ok = False

            msg = (
                f"You are now **{new_char.name}** (`{new_char.character_id}`). "
                + ("Dossier DM'd." if dm_ok
                   else "⚠️ Couldn't DM dossier (server DMs disabled).")
            )
            await inter.followup.send(
                embed=render_info("Joined", msg), ephemeral=True,
            )
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
                "No session here. `/story start` first.", ephemeral=True,
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

        dm_ok = True
        try:
            dossier = engine.build_character_dossier(
                row.session_id, new_char.character_id,
            )
            for chunk in _chunks(dossier, 1900):
                await inter.user.send(chunk)
        except discord.Forbidden:
            dm_ok = False
        except Exception:
            logger.exception("dossier DM failed")
            dm_ok = False

        msg = (
            f"You are now **{new_char.name}** "
            f"(replaced `{character_id}`). "
            + ("Dossier DM'd." if dm_ok
               else "⚠️ Couldn't DM dossier (server DMs disabled).")
        )
        await inter.followup.send(
            embed=render_info("Joined", msg), ephemeral=True,
        )

    # ---- /character ---------------------------------------------------------

    @tree.command(
        name="character",
        description="DM the dossier for a character in this channel's story.",
        guild=guild,
    )
    @app_commands.describe(
        character_id="The character_id (see /story characters).",
    )
    async def _character(inter: discord.Interaction, character_id: str):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No session here. `/story start` first.", ephemeral=True,
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

        dm_ok = True
        try:
            for chunk in _chunks(dossier, 1900):
                await inter.user.send(chunk)
        except discord.Forbidden:
            dm_ok = False
        except Exception:
            logger.exception("dossier DM failed")
            dm_ok = False

        if dm_ok:
            await inter.followup.send(
                f"Dossier for `{character_id}` DM'd.", ephemeral=True,
            )
        else:
            await inter.followup.send(
                embed=render_error(
                    "Couldn't DM the dossier (server DMs disabled). "
                    "Enable DMs from this server and try again."
                ),
                ephemeral=True,
            )

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
                "No story here yet. Try `/story start <id>`.",
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
                "No story here yet. Try `/story start <id>`.",
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
                f"the channel mapping has been purged. Run `/story start` "
                f"to begin fresh.",
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
                "No session here. Run `/story start` first.", ephemeral=True,
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
                "No session here. Run `/story start` first.", ephemeral=True,
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

    # Attach the /story and /settings groups to the tree last, once
    # their subcommands are defined.
    if guild is not None:
        tree.add_command(story_group, guild=guild)
        tree.add_command(settings_group, guild=guild)
    else:
        tree.add_command(story_group)
        tree.add_command(settings_group)
