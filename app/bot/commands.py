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


def _is_admin(user_id: int) -> bool:
    """True if the Discord user ID is in DISCORD_ADMIN_USER_IDS (comma-sep env)."""
    raw = os.environ.get("DISCORD_ADMIN_USER_IDS", "").strip()
    if not raw:
        return False
    try:
        admin_ids = {int(x) for x in raw.split(",") if x.strip()}
    except ValueError:
        logger.warning("DISCORD_ADMIN_USER_IDS contains a non-integer; ignoring all")
        return False
    return user_id in admin_ids


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


def _format_character_list(summaries) -> str:
    """Render a CharacterSummary list as a compact markdown block for an
    info embed. Active characters first, then dormant, then culled. Bound
    characters get a human-visible marker so joiners know who's claimed."""
    order = {"active": 0, "dormant": 1, "culled": 2}
    sorted_summaries = sorted(summaries, key=lambda s: (order.get(s.status, 3), s.name))

    lines: list[str] = []
    for s in sorted_summaries:
        bits: list[str] = [f"**{s.name}** · `{s.character_id}`"]
        tags: list[str] = []
        if s.status != "active":
            tags.append(s.status)
        if s.is_player:
            tags.append("player slot")
        if s.bound_user_id:
            tags.append(f"bound: <@{s.bound_user_id}>")
        if tags:
            bits.append(f"*[{' · '.join(tags)}]*")
        line = " ".join(bits)
        if s.role:
            line += f" — {s.role}"
        if s.faction:
            line += f" ({s.faction})"
        if s.appearance:
            line += f"\n  {s.appearance}"
        lines.append(line)
    return "\n".join(lines)


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
        character_name="The name your character will use in-story (e.g. 'Marcus Hale').",
    )
    async def _start(
        inter: discord.Interaction,
        story_id: str,
        character_name: str,
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

        character_name = character_name.strip()
        if not character_name:
            await inter.response.send_message(
                "Character name cannot be empty.", ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True)

        session_id = engine.session_id_for_channel(inter.channel_id, story_id)

        try:
            ckpt = await engine.create_session(
                story_id=story_id,
                session_id=session_id,
                player_display_name=character_name,
                creator_user_id=inter.user.id,
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

        # The briefing embed carries the full first-time-user onboarding
        # (setting, role, facts, and a "How to play" block). Keep the
        # content message terse so they don't fight for attention.
        briefing = render_briefing(ckpt, story_id)
        intro = f"**{character_name}** begins **{story_id}**."
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
            await inter.response.send_message(
                f"Session `{existing.session_id}` mapping exists but no checkpoint "
                f"found. Run `/story end` to clean up.",
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
        description="List characters in this channel's story (spoiler-free).",
    )
    @app_commands.describe(
        story_id="Optional story_id to inspect before starting a session.",
    )
    async def _characters(
        inter: discord.Interaction,
        story_id: str | None = None,
    ):
        # Prefer the live session so bindings/statuses reflect actual play;
        # fall back to story_id for pre-session browsing.
        summaries = None
        source_label = ""
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
                source_label = f"`{story_id}` (source roster)"
            except Exception as e:
                logger.exception("list_story_characters failed")
                await inter.response.send_message(
                    embed=render_error(f"`{type(e).__name__}: {e}`"),
                    ephemeral=True,
                )
                return
        elif row is not None:
            try:
                summaries = engine.list_session_characters(row.session_id)
                source_label = f"`{row.story_id}` (this session)"
            except Exception as e:
                logger.exception("list_session_characters failed")
                await inter.response.send_message(
                    embed=render_error(f"`{type(e).__name__}: {e}`"),
                    ephemeral=True,
                )
                return
        else:
            await inter.response.send_message(
                "No session here and no `story_id` given. "
                "Pass one, or run `/story start` first.",
                ephemeral=True,
            )
            return

        if not summaries:
            await inter.response.send_message(
                "No characters in the roster.", ephemeral=True,
            )
            return

        body = _format_character_list(summaries)
        await inter.response.send_message(
            embed=render_info(f"Characters · {source_label}", body),
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
        try:
            ckpt = await engine.import_story(source_text, story_id)
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

        character_id = character_id.strip()
        try:
            ckpt = engine.bind_user(row.session_id, inter.user.id, character_id)
        except ValueError as e:
            await inter.response.send_message(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("bind_user failed")
            await inter.response.send_message(
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
        await inter.response.send_message(
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

        try:
            freed = engine.unbind_user(row.session_id, inter.user.id)
        except Exception as e:
            logger.exception("unbind_user failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        if freed is None:
            await inter.response.send_message(
                "You weren't bound to a character.", ephemeral=True,
            )
            return
        await inter.response.send_message(
            f"Released `{freed}`. Other players can now `/join` them.",
            ephemeral=True,
        )

    # ---- /describe ----------------------------------------------------------

    @tree.command(
        name="describe",
        description="Describe your character's physical presence. Opens the story on first use.",
        guild=guild,
    )
    @app_commands.describe(
        traits="Height, build, clothing, voice, notable features — freeform.",
    )
    async def _describe(inter: discord.Interaction, traits: str):
        row = await smap.get(inter.channel_id)
        if row is None:
            await inter.response.send_message(
                "No story here yet. Try `/story start <id> <name>`.",
                ephemeral=True,
            )
            return

        traits = traits.strip()
        if not traits:
            await inter.response.send_message(
                "Description cannot be empty.", ephemeral=True,
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
            ckpt = engine.set_character_appearance(
                row.session_id, binding, traits,
            )
        except Exception as e:
            logger.exception("set_character_appearance failed")
            await inter.followup.send(embed=render_error(
                f"`{type(e).__name__}: {e}`"
            ))
            return

        # If this is the first describe AND no turns have happened yet, fire a
        # synthetic arrival action so the narrator renders the opening scene
        # with the description in hand. Otherwise this is just a mid-story
        # update and we don't trigger anything.
        is_pre_play = not ckpt.narrator_conversation

        if not is_pre_play:
            await inter.followup.send(
                embed=render_info(
                    "Description updated",
                    f"New description: *{traits}*\n\n"
                    f"This takes effect on your next `/act`.",
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
            "Describe+open for %s by %s: traits=%r",
            row.session_id, inter.user.display_name, traits[:200],
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
                    "  phase=%-18s %5.0fms  in=%4d out=%4d cache_read=%5d cache_write=%5d  (%s)",
                    lat.phase, lat.duration_ms,
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
            await inter.response.send_message(
                "Session mapping exists but no checkpoint found.",
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
