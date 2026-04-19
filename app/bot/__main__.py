"""Entrypoint for the narrative-engine Discord bot.

Run with: `.venv/bin/python -m app.bot`

Reads DISCORD_BOT_TOKEN (required) and DISCORD_GUILD_ID (optional but
recommended for dev — guild-scoped command syncs are instant) from the
environment. Also picks up ANTHROPIC_API_KEY for the engine.

Loads `.env` automatically via python-dotenv if present.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import discord
from discord import app_commands
from dotenv import load_dotenv

from app.bot.commands import register
from app.bot.engine_bridge import EngineBridge
from app.bot.session_map import SessionMap

_LOG_FILE = Path(".bot.log")
_LOG_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FMT,
    datefmt=_LOG_DATEFMT,
    handlers=[
        logging.StreamHandler(),                 # stdout for interactive runs
        logging.FileHandler(_LOG_FILE, mode="a"),  # persistent tail-able log
    ],
)
logger = logging.getLogger("app.bot")


async def _run() -> int:
    load_dotenv(Path(".env"), override=False)

    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("ERROR: DISCORD_BOT_TOKEN not set", file=sys.stderr)
        return 1
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set (engine needs it)", file=sys.stderr)
        return 1

    guild_id_raw = os.environ.get("DISCORD_GUILD_ID")
    guild = discord.Object(id=int(guild_id_raw)) if guild_id_raw else None

    intents = discord.Intents.default()
    # Slash commands don't need message_content; leave it off for now.
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    engine = EngineBridge()
    smap = SessionMap()
    await smap.init()

    register(tree, engine, smap, guild)

    @client.event
    async def on_ready():
        logger.info(
            "Bot connected as %s (id=%s). Syncing commands%s …",
            client.user,
            client.user.id if client.user else "?",
            f" to guild {guild.id}" if guild else " globally",
        )
        try:
            if guild is not None:
                # Copy global-only command definitions to guild for instant sync.
                tree.copy_global_to(guild=guild)
                synced = await tree.sync(guild=guild)
            else:
                synced = await tree.sync()
            logger.info("Synced %d command(s).", len(synced))
        except Exception:
            logger.exception("Command sync failed")

    @client.event
    async def on_disconnect():
        logger.warning("Disconnected from Discord gateway.")

    try:
        await client.start(token)
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt).")
    finally:
        await client.close()
        await engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
