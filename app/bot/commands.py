"""Slash commands for the narrative-engine Discord bot.

Commands:
    /story list                — list available stories
    /story start <story_id>    — create a session in this channel, post opening
    /story resume              — show the last scene from an existing session
    /story end                 — detach this channel from its session
    /act <action>              — submit a turn (defer + followup embed)
    /status                    — summarize current state

The bot calls the engine in-process (no HTTP). Each turn runs under a
per-session lock so concurrent /act commands on the same channel serialize.
"""

from __future__ import annotations

import logging
import time

import discord
from discord import app_commands

from app.bot.embed import render_error, render_info, render_turn
from app.bot.engine_bridge import EngineBridge
from app.bot.session_map import SessionMap
from app.schemas.checkpoint import CheckpointFile

logger = logging.getLogger(__name__)


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

        # Post a factual briefing so the player has world/role/stakes context.
        # The player now runs `/describe <traits>` to set their character's
        # physical presence, which triggers the narrator-rendered opening scene.
        from app.bot.embed import render_briefing
        briefing = render_briefing(ckpt, story_id)
        intro = (
            f"**{character_name}** begins **{story_id}**. "
            f"Next: `/describe <traits>` — describe your character's physical "
            f"presence (height, build, clothing, bearing) so the world can "
            f"react to them. The opening scene will render from that."
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

        # Save the description first.
        try:
            ckpt = engine.set_player_character_description(row.session_id, traits)
        except Exception as e:
            logger.exception("set_player_character_description failed")
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

        # Pre-play: fire the opening turn using a neutral arrival prompt.
        arrival_action = (
            "I take a steadying breath and step fully into the space, "
            "letting my eyes move across the room to see what is actually here."
        )
        logger.info(
            "Describe+open for %s by %s: traits=%r",
            row.session_id, inter.user.display_name, traits[:200],
        )

        try:
            response = await engine.run_turn(
                session_id=row.session_id,
                user_input=arrival_action,
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

        await inter.response.defer(thinking=True)
        start = time.monotonic()

        try:
            response = await engine.run_turn(
                session_id=row.session_id,
                user_input=action,
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

        present = [
            c.name for c in ckpt.characters
            if c.location == scene_id and c.status.value == "active"
            and c.character_id != ckpt.session.player_character_id
        ]

        body_lines = [
            f"**Story:** `{row.story_id}`",
            f"**Player:** {ckpt.session.player_name or '(unknown)'}",
            f"**Turn:** {ckpt.session.turn_index}",
            f"**Scene:** {scene_name}",
            f"**Present:** {', '.join(present) if present else 'no one else'}",
        ]
        await inter.response.send_message(
            embed=render_info("Session status", "\n".join(body_lines)),
            ephemeral=True,
        )

    # Attach the /story group to the tree last, once its subcommands are defined.
    if guild is not None:
        tree.add_command(story_group, guild=guild)
    else:
        tree.add_command(story_group)
