"""Slash commands for the narrative-engine Discord bot.

Commands:
    /session start <name>                 — create + bind an empty session to this channel
    /session end                          — detach this channel's session
    /session resume                       — show the last turn
    /session list                         — list existing sessions
    /story list                           — list available stories
    /story start <story_id>               — load a story into the current session
    /story info <story_id>                — show briefing for a source story
    /story import <attachment> [id]       — import a master prompt
    /story delete                         — unload the story from this session
    /join                                 — pick a character via interactive menu
                                            (pre-play: enters the lobby; post-play: arrives mid-story)
    /begin                                — open the story for everyone in the lobby
                                            (any bound player or admin can fire)
    /leave                                — release your character
    /describe [name] [appearance]         — set/update name and appearance
    /act <action>                         — submit a turn
    /defer                                — submit no action and let the scene continue
    /query <question>                     — ask an out-of-character question
    /status                               — summarize current state

The bot calls the engine in-process (no HTTP). Each turn runs under a
per-session lock so concurrent /act commands on the same channel serialize.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
import time
from typing import Optional

import discord
from discord import app_commands

from app.bot.embed import render_briefing, render_error, render_info, render_turn
from app.bot.engine_bridge import EngineBridge
from app.bot.session_map import SessionMap, TurnMessageRef
from app.llm.client import TransientLLMError
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


def _session_channel_id(inter: discord.Interaction) -> int:
    """Return the snowflake of the channel that owns the engine session
    for `inter`.

    Slash commands invoked from inside a thread (a player typing
    `/act` in their POV thread, for example) carry the **thread** as
    `inter.channel` and its id as `inter.channel_id`. The engine
    session, however, was bound to the parent text channel by
    `/story start`. Without this resolution every `smap.get(...)`
    misses, every `pov_threads` lookup keys to the wrong row, and
    `_ensure_pov_thread` would try to spawn threads-inside-threads
    (which Discord rejects). Returning the parent id here lets a
    user run any session command from inside their POV thread and
    get the same routing they would from the main channel.
    """
    chan = inter.channel
    if isinstance(chan, discord.Thread) and chan.parent_id is not None:
        return chan.parent_id
    return inter.channel_id


def _session_text_channel(
    inter: discord.Interaction,
) -> Optional[discord.TextChannel]:
    """Return the parent `TextChannel` that hosts this interaction's
    engine session, or None if no parent text channel is reachable
    (DMs, voice, uncached parent in a thread).

    `_ensure_pov_thread` needs a real `TextChannel` to spawn private
    sub-threads on. When `inter.channel` is already a `TextChannel`
    we hand it back; when it's a `Thread` we resolve the parent
    (preferring the cached `.parent`, falling back to a guild
    lookup); otherwise None and the caller takes the DM path.
    """
    chan = inter.channel
    if isinstance(chan, discord.TextChannel):
        return chan
    if isinstance(chan, discord.Thread):
        parent = chan.parent
        if parent is None and inter.guild is not None and chan.parent_id:
            fetched = inter.guild.get_channel(chan.parent_id)
            if isinstance(fetched, discord.TextChannel):
                parent = fetched
        return parent if isinstance(parent, discord.TextChannel) else None
    return None


async def _send_private(inter: discord.Interaction, text: str) -> None:
    """Post a private block of text back to the invoker as ephemeral
    followups. Chunks at Discord's 2000-char message cap, breaking on
    paragraph/newline boundaries when possible. Requires the caller to
    have already deferred (ephemeral=True) the interaction, or otherwise
    be in a state where `followup.send` is valid.
    """
    for chunk in _chunks(text, 1900):
        await inter.followup.send(chunk, ephemeral=True)


async def _clear_interaction_response(inter: discord.Interaction) -> None:
    """Best-effort cleanup for a deferred slash-command response.

    Successful `/act` renders land in the player's POV thread/DM. The
    interaction response is only a temporary "thinking" placeholder, so
    remove it instead of leaving "Posted to ..." acknowledgement clutter
    in the channel/client.
    """
    try:
        await inter.delete_original_response()
    except discord.NotFound:
        return
    except Exception:
        logger.debug("interaction response cleanup failed", exc_info=True)


@dataclass
class TurnMessageCleanup:
    tracked: int = 0
    deleted: int = 0
    hidden: int = 0
    missing: int = 0
    failed: int = 0


async def _record_turn_message(
    *,
    smap: SessionMap,
    session_channel_id: int,
    session_id: str,
    turn_index: Optional[int],
    message: object,
    delivery: str,
    discord_channel_id: Optional[int] = None,
    recipient_user_id: Optional[int] = None,
) -> None:
    if not session_id or turn_index is None:
        return
    message_id = getattr(message, "id", None)
    if message_id is None:
        return
    message_channel = getattr(message, "channel", None)
    channel_id = discord_channel_id or getattr(message_channel, "id", None)
    if channel_id is None:
        return
    try:
        await smap.record_turn_message(
            channel_id=session_channel_id,
            session_id=session_id,
            turn_index=int(turn_index),
            discord_channel_id=int(channel_id),
            message_id=int(message_id),
            delivery=delivery,
            recipient_user_id=recipient_user_id,
        )
    except Exception:
        logger.exception(
            "turn-message tracking failed for session=%s turn=%s message=%s",
            session_id, turn_index, message_id,
        )


async def _ensure_pov_thread(
    *,
    channel: discord.abc.Messageable,
    user: discord.abc.User,
    smap: SessionMap,
    character_id: str,
    char_name: str,
) -> Optional[discord.Thread]:
    """Get or create the private POV thread for `user` in `channel`.

    Returns None if thread creation isn't possible (channel doesn't
    support threads, missing CREATE_PRIVATE_THREADS perm, network
    failure). Callers fall back to DM in that case so the player still
    receives their narrative beat.

    Threads persist across re-binds: a player who /leave's and /joins
    a different character keeps the same thread. `character_id` is
    stored for diagnostic purposes only.
    """
    if not isinstance(channel, discord.TextChannel):
        return None  # DM channels and voice channels can't host threads
    cached = await smap.get_pov_thread_id(channel.id, user.id)
    if cached is not None:
        thread = channel.guild.get_thread(cached)
        if thread is None:
            try:
                thread = await channel.guild.fetch_channel(cached)
            except discord.NotFound:
                thread = None
            except Exception:
                logger.exception(
                    "ensure_pov_thread: fetch_channel(%s) failed", cached,
                )
                thread = None
        if isinstance(thread, discord.Thread) and not thread.archived:
            return thread
        # Cached id is stale or archived — drop it and recreate below.
        await smap.clear_pov_thread(channel.id, user.id)

    suffix = (char_name or character_id or "pov").strip()[:60] or "pov"
    thread_name = f"{user.display_name} · {suffix}"
    try:
        thread = await channel.create_thread(
            name=thread_name[:100],
            type=discord.ChannelType.private_thread,
            invitable=False,
            auto_archive_duration=10080,  # 7 days; max allowed without boost
            reason=f"POV thread for {user.display_name} ({character_id})",
        )
    except discord.Forbidden:
        logger.warning(
            "ensure_pov_thread: missing CREATE_PRIVATE_THREADS in #%s "
            "(channel %s); falling back to DM for user %s",
            channel.name, channel.id, user.id,
        )
        return None
    except Exception:
        logger.exception(
            "ensure_pov_thread: create_thread failed in #%s for user %s",
            channel.name, user.id,
        )
        return None

    try:
        await thread.add_user(user)
    except Exception:
        logger.exception(
            "ensure_pov_thread: add_user(%s) failed on thread %s; "
            "abandoning the thread and falling back to DM",
            user.id, thread.id,
        )
        # Don't cache: a private thread the user was never added to is
        # invisible to them, and `thread.send` won't raise — so without
        # bailing out we'd silently drop every POV beat on the floor
        # (the comment used to claim DM fallback would kick in; it
        # would not, because send-success masked the underlying
        # add_user failure). Returning None here forces the caller into
        # the explicit DM path. The orphan thread is left alive in
        # Discord; an operator can clean up if needed.
        return None

    await smap.set_pov_thread(
        channel_id=channel.id,
        user_id=user.id,
        thread_id=thread.id,
        character_id=character_id,
    )
    return thread


async def _post_actor_render(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    user: discord.abc.User,
    character_id: str,
    char_name: str,
    embeds: list[discord.Embed],
    intro_content: Optional[str] = None,
    session_id: str = "",
    turn_index: Optional[int] = None,
) -> tuple[str, Optional[discord.Thread]]:
    """Post the ACTOR's own beat privately: POV thread first, DM fallback,
    `("none", None)` if both fail (caller should then post publicly to
    the channel via `inter.followup` so the user still sees their
    narrative).

    Returns one of:
      - `("thread", thread)` — landed in the actor's private POV thread.
      - `("dm", None)`        — thread unavailable; landed in DMs.
      - `("none", None)`      — both private paths failed; caller must
                                fall back to a public render.

    The deferred slash-command interaction is NOT closed by this helper
    (private posts go to thread/DM, not `inter.followup`). The caller
    is responsible for an ephemeral `inter.followup.send` ack on
    success, or a non-ephemeral public render on failure.

    `embeds` should already be `render_turn(...)`-shaped. `intro_content`
    is a short prefix line (e.g. "**Jimbo Higgins** joined.") that goes
    in the same message as the embed.
    """
    channel = _session_text_channel(inter)
    thread: Optional[discord.Thread] = None
    if channel is not None:
        thread = await _ensure_pov_thread(
            channel=channel,
            user=user,
            smap=smap,
            character_id=character_id,
            char_name=char_name,
        )

    if thread is not None:
        try:
            msg = await thread.send(content=intro_content, embeds=embeds)
            await _record_turn_message(
                smap=smap,
                session_channel_id=_session_channel_id(inter),
                session_id=session_id,
                turn_index=turn_index,
                message=msg,
                delivery="thread",
                discord_channel_id=thread.id,
                recipient_user_id=user.id,
            )
            return ("thread", thread)
        except Exception:
            logger.exception(
                "post_actor_render: thread.send to %s failed; "
                "falling back to DM", thread.id,
            )
            await smap.clear_pov_thread(
                _session_channel_id(inter), user.id,
            )

    try:
        msg = await user.send(content=intro_content, embeds=embeds)
        await _record_turn_message(
            smap=smap,
            session_channel_id=_session_channel_id(inter),
            session_id=session_id,
            turn_index=turn_index,
            message=msg,
            delivery="dm",
            recipient_user_id=user.id,
        )
        return ("dm", None)
    except Exception:
        logger.exception(
            "post_actor_render: DM fallback to user %s failed", user.id,
        )

    return ("none", None)


async def _post_to_pov(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    user_id: int,
    character_id: str,
    char_name: str,
    text: str,
    bot: "discord.Client",
    session_id: str = "",
    turn_index: Optional[int] = None,
) -> bool:
    """Post `text` to the user's POV thread in the channel where this
    interaction lives. Falls back to DM on any thread-related failure.

    Returns True if the message was delivered, False if both thread
    AND DM failed (caller may want to surface that).
    """
    chunks = _chunks(text, 1900)
    # Resolve the parent text channel — if `inter` came in via a
    # thread, `inter.channel` is the thread itself and we'd otherwise
    # try to nest a thread inside a thread.
    channel = _session_text_channel(inter)
    session_chan_id = _session_channel_id(inter)
    user = bot.get_user(user_id)
    if user is None:
        try:
            user = await bot.fetch_user(user_id)
        except Exception:
            logger.exception(
                "post_to_pov: fetch_user(%s) failed; can't deliver", user_id,
            )
            return False

    thread: Optional[discord.Thread] = None
    if channel is not None:
        thread = await _ensure_pov_thread(
            channel=channel,
            user=user,
            smap=smap,
            character_id=character_id,
            char_name=char_name,
        )

    if thread is not None:
        try:
            for chunk in chunks:
                msg = await thread.send(chunk)
                await _record_turn_message(
                    smap=smap,
                    session_channel_id=session_chan_id,
                    session_id=session_id,
                    turn_index=turn_index,
                    message=msg,
                    delivery="thread",
                    discord_channel_id=thread.id,
                    recipient_user_id=user_id,
                )
            return True
        except Exception:
            logger.exception(
                "post_to_pov: thread.send to thread %s failed; "
                "falling back to DM", thread.id,
            )
            await smap.clear_pov_thread(session_chan_id, user_id)

    try:
        for chunk in chunks:
            msg = await user.send(chunk)
            await _record_turn_message(
                smap=smap,
                session_channel_id=session_chan_id,
                session_id=session_id,
                turn_index=turn_index,
                message=msg,
                delivery="dm",
                recipient_user_id=user_id,
            )
        return True
    except Exception:
        logger.exception(
            "post_to_pov: DM fallback to user %s failed", user_id,
        )
        return False


async def _send_public_turn_render(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    session_id: str,
    turn_index: int,
    content: Optional[str] = None,
    embeds: Optional[list[discord.Embed]] = None,
) -> None:
    channel = _session_text_channel(inter)
    if channel is not None:
        msg = await channel.send(content=content, embeds=embeds)
    else:
        msg = await inter.followup.send(
            content=content,
            embeds=embeds,
            wait=True,
        )
    await _record_turn_message(
        smap=smap,
        session_channel_id=_session_channel_id(inter),
        session_id=session_id,
        turn_index=turn_index,
        message=msg,
        delivery="public",
    )


async def _message_channel_for_ref(
    client: discord.Client,
    ref: TurnMessageRef,
):
    if ref.delivery == "dm" and ref.recipient_user_id is not None:
        user = client.get_user(ref.recipient_user_id)
        if user is None:
            try:
                user = await client.fetch_user(ref.recipient_user_id)
            except Exception:
                logger.exception(
                    "rewind cleanup: fetch_user(%s) failed",
                    ref.recipient_user_id,
                )
                return None
        dm_channel = getattr(user, "dm_channel", None)
        if dm_channel is None:
            try:
                dm_channel = await user.create_dm()
            except Exception:
                logger.exception(
                    "rewind cleanup: create_dm(%s) failed",
                    ref.recipient_user_id,
                )
                return None
        return dm_channel

    channel = client.get_channel(ref.discord_channel_id)
    if channel is not None:
        return channel
    try:
        return await client.fetch_channel(ref.discord_channel_id)
    except discord.NotFound:
        return None
    except Exception:
        logger.exception(
            "rewind cleanup: fetch_channel(%s) failed",
            ref.discord_channel_id,
        )
        return None


async def _delete_rewound_turn_messages(
    *,
    client: discord.Client,
    smap: SessionMap,
    channel_id: int,
    session_id: str,
    deleted_turns: list[int],
) -> TurnMessageCleanup:
    refs = await smap.list_turn_messages(
        channel_id=channel_id,
        session_id=session_id,
        turns=deleted_turns,
    )
    cleanup = TurnMessageCleanup(tracked=len(refs))
    handled: list[TurnMessageRef] = []

    for ref in refs:
        channel = await _message_channel_for_ref(client, ref)
        if channel is None:
            cleanup.failed += 1
            continue

        try:
            msg = await channel.fetch_message(ref.message_id)
        except discord.NotFound:
            cleanup.missing += 1
            handled.append(ref)
            continue
        except Exception:
            cleanup.failed += 1
            logger.exception(
                "rewind cleanup: fetch_message(%s) failed in channel %s",
                ref.message_id, ref.discord_channel_id,
            )
            continue

        try:
            await msg.delete()
            cleanup.deleted += 1
            handled.append(ref)
            continue
        except discord.NotFound:
            cleanup.missing += 1
            handled.append(ref)
            continue
        except Exception:
            logger.warning(
                "rewind cleanup: delete failed for message %s in channel %s; "
                "trying edit fallback",
                ref.message_id, ref.discord_channel_id,
                exc_info=True,
            )

        try:
            await msg.edit(content="_Rewound turn hidden._", embeds=[])
            cleanup.hidden += 1
            handled.append(ref)
        except discord.NotFound:
            cleanup.missing += 1
            handled.append(ref)
        except Exception:
            cleanup.failed += 1
            logger.exception(
                "rewind cleanup: edit fallback failed for message %s "
                "in channel %s",
                ref.message_id, ref.discord_channel_id,
            )

    if handled:
        await smap.forget_turn_messages(handled)
    return cleanup


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
        existing = await smap.get(_session_channel_id(inter))
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
            channel_id=_session_channel_id(inter),
            guild_id=inter.guild_id,
            session_id=name,
            owner_user_id=inter.user.id,
            story_id="",  # no story loaded yet
        )
        await inter.response.send_message(
            f"Session `{name}` created. Run `/story start` to load a story.",
            ephemeral=True,
        )

    @session_group.command(
        name="end", description="Detach this channel from its session (files stay).",
    )
    async def _session_end(inter: discord.Interaction):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message("No session here.", ephemeral=True)
            return
        await smap.delete(_session_channel_id(inter))
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
        existing = await smap.get(_session_channel_id(inter))
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
        except FileNotFoundError:
            pass  # empty session, no ckpt yet

        await smap.upsert(
            channel_id=_session_channel_id(inter),
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
            channel_id=_session_channel_id(inter),
            guild_id=inter.guild_id,
            session_id=row.session_id,
            owner_user_id=row.owner_user_id,
            story_id=story_id,
        )

        briefing = render_briefing(ckpt, story_id)
        intro = (
            f"Loaded **{story_id}** into session `{row.session_id}`. "
            f"Run `/join` when you're ready to step in."
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
        row = await smap.get(_session_channel_id(inter))
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
        row = await smap.get(_session_channel_id(inter))
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

        # Live-session path: group by location and
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

        # v11: location = the invoker's POV location label.
        from app.engine.context_builder import pov_location_for_user
        uid = str(inter.user.id)
        location = pov_location_for_user(ckpt, user_id=uid)
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
                if c.location == location:
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
            _block(f"Here ({location or 'no active location'})", here),
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

        await inter.response.defer(thinking=True, ephemeral=True)

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
            "Imported %s in %.1fs: %d characters",
            story_id, elapsed,
            len(ckpt.characters),
        )

        # No briefing here — the player_primer is shown on /story start
        # and the actual opening prose is rendered when the first player
        # types /begin (composed by the router, voiced by the narrator).
        intro = (
            f"Imported **{story_id}** in {int(elapsed)}s — "
            f"{len(ckpt.characters)} characters, "
            f"Start with `/session start name:<save>` then `/story start story_id:{story_id}`."
        )
        await inter.followup.send(content=intro)

    # ---- /story delete -----------------------------------------------------

    @story_group.command(
        name="delete",
        description="Unload the current story from this session, leaving it empty.",
    )
    async def _story_delete(inter: discord.Interaction):
        row = await smap.get(_session_channel_id(inter))
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
            channel_id=_session_channel_id(inter),
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

    # ---- /join (interactive roster picker) ----------------------------------
    #
    # No-arg slash command. Presents a SelectMenu of `is_playable=true`
    # characters that aren't already claimed. After the player picks,
    # an optional name+appearance modal pops; on submit we bind and
    # then either:
    #
    #   - **Pre-play (no narrator history yet):** post a lobby
    #     acknowledgement listing every bound player and reminding the
    #     room to type `/begin` when everyone is ready. NO turn runs.
    #     This decouples binding from opening so multiple humans can
    #     `/join` in any order before the story starts.
    #
    #   - **Mid-play (story already opened by `/begin`):** fire
    #     `(arrive)` via `engine.run_arrival_turn` so the router
    #     places the character into a sensible live situation and the
    #     narrator renders the arrival.
    #
    # The split came in r9d alongside the `/begin` command. Pre-r9d
    # `/join` always fired an arrival turn and let the engine decide
    # `(begin)` vs `(arrive)` from `narrator_conversations`; that
    # flow couldn't gracefully handle two players who wanted to start
    # together (the first one's `/join` would unilaterally open the
    # story before the second got their character in).
    #
    # The legacy /join_custom and /pick_replacement commands (player-
    # authored character with invented backstory) were removed as part
    # of the playable-2 UX overhaul — they generated content the
    # player never asked for. The engine_bridge methods that backed
    # them (create_custom_character, suggest_replacement_targets,
    # replace_with_custom) are kept on EngineBridge for the play CLI.

    async def _post_lobby_message(
        inter: discord.Interaction,
        session_id: str,
        joined_name: str,
    ) -> None:
        """Pre-play: ack the freshly-bound player and remind everyone
        to `/begin` when ready.

        Posts a public message in the session channel listing every
        bound player so newcomers can see who's in the lobby. The
        ephemeral followup tells the joining player how to act next.

        Failures during checkpoint load fall back to a single-player
        ack — the binding itself succeeded, only the lobby roster
        listing is best-effort."""
        bound_lines: list[str] = []
        try:
            ckpt = engine.load_latest(session_id)
            bindings = ckpt.session.character_bindings or {}
            roster = list(ckpt.characters or [])
            by_id = {c.character_id: c for c in roster}
            for cid in sorted(bindings.keys()):
                ch = by_id.get(cid)
                bound_lines.append(f"• **{ch.name if ch else cid}**")
        except Exception:
            logger.exception(
                "lobby message: load_latest(%s) failed; "
                "falling back to single-player ack", session_id,
            )

        if not bound_lines:
            bound_lines = [f"• **{joined_name}**"]

        body = (
            f"**{joined_name}** stepped into the lobby.\n\n"
            f"In the lobby:\n" + "\n".join(bound_lines) + "\n\n"
            "Type `/begin` when everyone's ready and the story will open."
        )
        try:
            await inter.followup.send(
                embed=render_info("Story lobby", body),
            )
        except Exception:
            logger.exception("lobby message: public followup failed")
            try:
                await inter.followup.send(body, ephemeral=True)
            except Exception:
                logger.exception(
                    "lobby message: ephemeral fallback also failed",
                )

    async def _handle_post_join(
        inter: discord.Interaction,
        session_id: str,
        story_id: str,
        binding_cid: str,
        char_name: str,
    ) -> None:
        """Post-bind dispatcher: send the freshly-bound player to the
        lobby (pre-play) or fire `(arrive)` (mid-play). Caller must
        have deferred non-ephemerally; this helper writes the user-
        facing followup either way.

        The pre-play check inspects `narrator_conversations` directly
        on the latest checkpoint. We cannot fully race-proof the
        pre-play branch from here (two players hitting the modal
        submit in the same tick before either /begin's would both
        post a lobby message), but the actual opening turn is
        race-proofed inside `engine.run_begin_turn` under the per-
        session lock — at most one /begin will succeed regardless
        of how many lobby messages got posted."""
        try:
            ckpt = engine.load_latest(session_id)
            is_pre_play = not any(ckpt.narrator_conversations.values())
        except Exception:
            logger.exception(
                "post-join dispatch: load_latest(%s) failed; "
                "defaulting to lobby path", session_id,
            )
            is_pre_play = True

        if is_pre_play:
            await _post_lobby_message(inter, session_id, char_name)
        else:
            await _fire_arrival_turn(
                inter,
                session_id=session_id,
                story_id=story_id,
                binding_cid=binding_cid,
                char_name=char_name,
            )

    async def _fire_arrival_turn(
        inter: discord.Interaction,
        session_id: str,
        story_id: str,
        binding_cid: str,
        char_name: str,
    ) -> None:
        """Mid-play `/join` path: fire `(arrive)` for a freshly-bound
        player and render the result. Caller must have deferred
        non-ephemerally.

        Pre-r9d this helper also handled the canonical opening turn
        via `engine.run_arrival_turn`'s `(begin)` branch. The opening
        moved to the dedicated `/begin` command, so the only directive
        this helper ever sends now is `(arrive)`."""
        try:
            response = await engine.run_arrival_turn(
                session_id=session_id,
                acting_character_id=binding_cid,
            )
        except TransientLLMError as e:
            logger.warning(
                "join arrival turn hit transient LLM error after %d "
                "attempts: %s", e.attempts, e.last_error,
            )
            await inter.followup.send(embed=render_error(str(e)))
            return
        except Exception as e:
            logger.exception("join arrival run_turn failed")
            await inter.followup.send(embed=render_error(
                f"`{type(e).__name__}: {e}`"
            ))
            return

        embeds = render_turn(
            output_text=response.output_text,
            turn_index=response.turn_index,
            story_id=story_id,
        )
        intro = f"**{char_name}** joined. You step into the moment."

        venue, thread = await _post_actor_render(
            inter=inter,
            smap=smap,
            user=inter.user,
            character_id=binding_cid,
            char_name=char_name,
            embeds=embeds,
            intro_content=intro,
            session_id=session_id,
            turn_index=response.turn_index,
        )
        if venue == "thread" and thread is not None:
            await inter.followup.send(
                f"**{char_name}** joined. Your story opens in "
                f"{thread.mention}.",
                ephemeral=True,
            )
        elif venue == "dm":
            await inter.followup.send(
                f"**{char_name}** joined. Your story opens in your DMs "
                "(POV thread unavailable here).",
                ephemeral=True,
            )
        else:
            await _send_public_turn_render(
                inter=inter,
                smap=smap,
                session_id=session_id,
                turn_index=response.turn_index,
                content=intro,
                embeds=embeds,
            )

        await _fan_out_per_player_renders(
            inter=inter,
            session_id=session_id,
            actor_cid=binding_cid,
            per_player=response.per_player_renders or {},
            turn_index=response.turn_index,
        )

    async def _fan_out_per_player_renders(
        *,
        inter: discord.Interaction,
        session_id: str,
        actor_cid: str,
        per_player: dict[str, str],
        turn_index: int,
    ) -> None:
        """DM each non-acting bound human their POV render for the
        last beat. Mirrors the /act fan-out path so multi-POV beats
        (joins, opening turns) don't drop bystanders.

        Logs and bails on each per-POV failure rather than aborting
        the loop — the actor's render already shipped, and one bad
        thread shouldn't silence the rest. Sends a single ephemeral
        ack to the actor afterwards listing who got notified."""
        if not per_player:
            return
        try:
            ckpt_after = engine.load_latest(session_id)
            bindings = ckpt_after.session.character_bindings or {}
            roster = list(ckpt_after.characters or [])
        except Exception:
            logger.exception(
                "per-player fan-out: load_latest failed; skipping DMs",
            )
            return
        notified: list[str] = []
        for cid, prose in per_player.items():
            if cid == actor_cid or not prose:
                continue
            uid_str = bindings.get(cid, "")
            if not uid_str:
                continue
            try:
                uid = int(uid_str)
            except ValueError:
                continue
            char = next(
                (c for c in roster if c.character_id == cid), None,
            )
            other_name = char.name if char else cid
            ok = await _post_to_pov(
                inter=inter,
                smap=smap,
                user_id=uid,
                character_id=cid,
                char_name=other_name,
                text=prose,
                bot=inter.client,
                session_id=session_id,
                turn_index=turn_index,
            )
            if ok:
                notified.append(other_name)
        if notified:
            try:
                phrase = ", ".join(f"**{n}**" for n in notified)
                await inter.followup.send(
                    f"({phrase} notified.)", ephemeral=True,
                )
            except Exception:
                logger.exception(
                    "per-player fan-out: ephemeral ack failed",
                )

    class _JoinIdentityModal(discord.ui.Modal, title="Step into the role"):
        """Optional name+appearance prompt fired after the user picks
        from the /join SelectMenu. Both fields are optional — leaving
        them blank keeps the authored defaults. On submit we bind,
        apply identity (if provided), and fire arrival."""

        name_in = discord.ui.TextInput(
            label="Display name (optional)",
            placeholder="Leave blank to keep the character's authored name.",
            required=False,
            max_length=80,
        )
        appearance_in = discord.ui.TextInput(
            label="Appearance (optional)",
            style=discord.TextStyle.paragraph,
            placeholder=(
                "Height, build, clothing, notable features. Leave blank "
                "to keep the authored appearance."
            ),
            required=False,
            max_length=600,
        )

        def __init__(
            self,
            *,
            session_id: str,
            story_id: str,
            character_id: str,
            character_name: str,
        ):
            super().__init__()
            self._session_id = session_id
            self._story_id = story_id
            self._character_id = character_id
            self._character_name = character_name

        async def on_submit(self, modal_inter: discord.Interaction):
            await modal_inter.response.defer(thinking=True)

            try:
                engine.takeover(
                    self._session_id, self._character_id, modal_inter.user.id,
                )
            except ValueError as e:
                await modal_inter.followup.send(
                    embed=render_error(str(e)), ephemeral=True,
                )
                return
            except Exception as e:
                logger.exception("/join takeover failed")
                await modal_inter.followup.send(
                    embed=render_error(f"`{type(e).__name__}: {e}`"),
                    ephemeral=True,
                )
                return

            chosen_name = (self.name_in.value or "").strip()
            chosen_appearance = (self.appearance_in.value or "").strip()
            if chosen_name or chosen_appearance:
                try:
                    engine.set_character_identity(
                        self._session_id, self._character_id,
                        name=chosen_name or None,
                        appearance=chosen_appearance or None,
                    )
                except Exception as e:
                    logger.exception("/join set_character_identity failed")
                    await modal_inter.followup.send(
                        embed=render_error(
                            f"Bound, but couldn't apply identity: "
                            f"`{type(e).__name__}: {e}`"
                        ),
                        ephemeral=True,
                    )
                    return

            display_name = chosen_name or self._character_name
            await _handle_post_join(
                modal_inter,
                session_id=self._session_id,
                story_id=self._story_id,
                binding_cid=self._character_id,
                char_name=display_name,
            )

    # Sentinel value used in the /join SelectMenu for the
    # "Create your own character" row. Real character_ids are
    # snake_case slugs (`_pick_unused_character_id`), so this
    # double-underscore form is unambiguous.
    JOIN_CUSTOM_SENTINEL = "__custom__"

    class _JoinCustomCreateModal(
        discord.ui.Modal, title="Create your character",
    ):
        """Modal fired when the user picks the 'Create your own character'
        row in /join. Three text inputs (name, appearance, backstory) feed
        `EngineBridge.create_player_character_simple` directly — no LLM
        round-trip. Backstory is optional; the agent can synthesize voice
        from rolling conversation later when the player /leaves."""

        name_in = discord.ui.TextInput(
            label="Character name",
            placeholder="e.g. Akari Tanaka",
            required=True,
            max_length=80,
        )
        appearance_in = discord.ui.TextInput(
            label="Appearance",
            style=discord.TextStyle.paragraph,
            placeholder=(
                "Height, build, clothing, notable features — whatever the "
                "world should see at a glance."
            ),
            required=True,
            max_length=600,
        )
        backstory_in = discord.ui.TextInput(
            label="Backstory (optional)",
            style=discord.TextStyle.paragraph,
            placeholder=(
                "Where you're from, what you do, what drives you. Leave "
                "blank to discover it through play."
            ),
            required=False,
            max_length=1000,
        )

        def __init__(self, *, session_id: str, story_id: str):
            super().__init__()
            self._session_id = session_id
            self._story_id = story_id

        async def on_submit(self, modal_inter: discord.Interaction):
            await modal_inter.response.defer(thinking=True)

            try:
                new_char = engine.create_player_character_simple(
                    self._session_id,
                    modal_inter.user.id,
                    name=self.name_in.value,
                    appearance=self.appearance_in.value,
                    backstory=self.backstory_in.value or "",
                )
            except ValueError as e:
                await modal_inter.followup.send(
                    embed=render_error(str(e)), ephemeral=True,
                )
                return
            except Exception as e:
                logger.exception("/join custom-create failed")
                await modal_inter.followup.send(
                    embed=render_error(f"`{type(e).__name__}: {e}`"),
                    ephemeral=True,
                )
                return

            await _handle_post_join(
                modal_inter,
                session_id=self._session_id,
                story_id=self._story_id,
                binding_cid=new_char.character_id,
                char_name=new_char.name,
            )

    class _JoinPickerView(discord.ui.View):
        """Ephemeral SelectMenu showing claimable playable characters and
        a 'Create your own character' row. Restricted to the invoker.
        Picking a row pops a modal as the picker callback's direct
        response (the 3s interaction window is respected)."""

        def __init__(
            self,
            *,
            session_id: str,
            story_id: str,
            invoker_id: int,
            options: list[discord.SelectOption],
            char_lookup: dict[str, str],
        ):
            super().__init__(timeout=180)
            self._session_id = session_id
            self._story_id = story_id
            self._invoker_id = invoker_id
            self._char_lookup = char_lookup
            self._select = discord.ui.Select(
                placeholder="Pick a character to play…",
                options=options,
                min_values=1, max_values=1,
            )
            self._select.callback = self._on_pick
            self.add_item(self._select)

        async def interaction_check(
            self, check_inter: discord.Interaction,
        ) -> bool:
            if check_inter.user.id != self._invoker_id:
                await check_inter.response.send_message(
                    "This picker isn't yours.", ephemeral=True,
                )
                return False
            return True

        async def _on_pick(self, pick_inter: discord.Interaction):
            picked_value = self._select.values[0]
            self._select.disabled = True
            if picked_value == JOIN_CUSTOM_SENTINEL:
                await pick_inter.response.send_modal(
                    _JoinCustomCreateModal(
                        session_id=self._session_id,
                        story_id=self._story_id,
                    )
                )
            else:
                picked_name = self._char_lookup.get(
                    picked_value, picked_value,
                )
                await pick_inter.response.send_modal(
                    _JoinIdentityModal(
                        session_id=self._session_id,
                        story_id=self._story_id,
                        character_id=picked_value,
                        character_name=picked_name,
                    )
                )
            self.stop()

    @tree.command(
        name="join",
        description="Step into the story — claim an existing character or create your own.",
        guild=guild,
    )
    async def _join(inter: discord.Interaction):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message(
                "No session here. `/session start` then `/story start` first.",
                ephemeral=True,
            )
            return
        if not row.story_id:
            await inter.response.send_message(
                "This session has no story loaded yet. Run `/story start` "
                "first.",
                ephemeral=True,
            )
            return

        existing = engine.get_user_binding(row.session_id, inter.user.id)
        if existing is not None:
            await inter.response.send_message(
                f"You're already bound to `{existing}`. Run `/leave` first "
                f"if you want to switch characters.",
                ephemeral=True,
            )
            return

        try:
            roster = engine.list_session_characters(row.session_id)
        except Exception as e:
            logger.exception("/join: list_session_characters failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        candidates = [
            c for c in roster
            if c.is_playable
            and c.status != "culled"
            and not c.bound_user_id
        ]

        # SelectMenu cap is 25 total options. Reserve one slot for the
        # custom-create row, leaving 24 for pre-authored playables. If a
        # roster ever exceeds 24 playable slots we'll truncate; paginated
        # pickers are deferred until we actually need them.
        truncated = len(candidates) > 24
        candidates = candidates[:24]

        def _trim(text: str, limit: int) -> str:
            text = (text or "").strip().replace("\n", " ")
            return text if len(text) <= limit else text[: limit - 1] + "…"

        # Custom-create option always sits at the top. Stories with zero
        # pre-authored playable slots (sengoku-style "humans bring their
        # own outsider character" designs) still get a usable picker —
        # just with one option.
        options: list[discord.SelectOption] = [
            discord.SelectOption(
                label="Create your own character",
                value=JOIN_CUSTOM_SENTINEL,
                description=(
                    "Pick a name, appearance, and optional backstory."
                ),
            ),
        ]
        char_lookup: dict[str, str] = {}
        for c in candidates:
            char_lookup[c.character_id] = c.name
            descr_bits = [b for b in (c.role, c.faction) if b]
            description = _trim(" · ".join(descr_bits) or c.appearance, 100)
            options.append(discord.SelectOption(
                label=_trim(c.name, 100),
                value=c.character_id,
                description=description or None,
            ))

        view = _JoinPickerView(
            session_id=row.session_id,
            story_id=row.story_id,
            invoker_id=inter.user.id,
            options=options,
            char_lookup=char_lookup,
        )
        body_lines: list[str] = []
        if candidates:
            n = len(candidates)
            body_lines.append(
                f"{n} pre-authored character{'s' if n != 1 else ''} open, "
                "or create your own. Pick a row to begin."
            )
        else:
            body_lines.append(
                "This story has no pre-authored playable slots — pick "
                "**Create your own character** to step in."
            )
        if truncated:
            body_lines.append(
                "_(Roster has more than 24 playable slots; only the first "
                "24 are listed. Tell an admin if you need a different one.)_"
            )
        body_lines.append(
            "Picking a pre-authored character lets you optionally override "
            "their name and appearance. Picking **Create your own character** "
            "asks you for a name, appearance, and backstory directly."
        )
        await inter.response.send_message(
            embed=render_info("Join the story", "\n\n".join(body_lines)),
            view=view,
            ephemeral=True,
        )

    # ---- /begin -------------------------------------------------------------
    #
    # No-arg slash command. Opens the canonical story for everyone in
    # the lobby — fires `(begin)` through `engine.run_begin_turn`,
    # which composes the opening beat from world_state and places
    # every bound player at the chosen starting location. Each bound
    # player gets their own POV render fanned out via the same per-
    # player path that /act uses.
    #
    # Permission shape: any bound player can fire it (it's the natural
    # follow-up to their own /join), and admins can fire it without
    # being bound (so a host can kick the story off for a table of
    # newer players who haven't claimed characters yet — though if no
    # one has /joined, the engine raises and we surface the error).
    #
    # The race / no-op behaviors live inside `engine.run_begin_turn`:
    # the per-session lock guarantees at most one (begin) ever runs,
    # and the "story already started" guard turns later /begins into
    # a friendly error after the opener landed.

    @tree.command(
        name="begin",
        description="Open the story for everyone in the lobby.",
        guild=guild,
    )
    async def _begin(inter: discord.Interaction):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message(
                "No session here. `/session start` then `/story start` first.",
                ephemeral=True,
            )
            return
        if not row.story_id:
            await inter.response.send_message(
                "This session has no story loaded yet. Run `/story start` "
                "first.",
                ephemeral=True,
            )
            return

        triggering_cid = engine.get_user_binding(
            row.session_id, inter.user.id,
        ) or ""
        if not triggering_cid and not _is_admin(inter.user.id):
            await inter.response.send_message(
                "You aren't bound to a character. `/join` first, then "
                "`/begin` will open the story.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True)

        try:
            response = await engine.run_begin_turn(
                session_id=row.session_id,
                triggering_character_id=triggering_cid,
            )
        except ValueError as e:
            await inter.followup.send(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except TransientLLMError as e:
            logger.warning(
                "/begin hit transient LLM error after %d attempts: %s",
                e.attempts, e.last_error,
            )
            await inter.followup.send(embed=render_error(str(e)))
            return
        except Exception as e:
            logger.exception("/begin run_begin_turn failed")
            await inter.followup.send(embed=render_error(
                f"`{type(e).__name__}: {e}`"
            ))
            return

        # The router placed every bound player at the opening location
        # (see event_router.txt OOC `(begin)` rules), so per_player
        # carries one render per bound POV. Pick the triggering
        # player's render for the actor path; fan the rest out via
        # the standard per-POV helper.
        per_player = response.per_player_renders or {}

        # Triggering character is either /begin'd by a bound player
        # or by an admin who isn't bound. Bound case: route to their
        # POV thread. Admin-no-binding case: announce publicly that
        # the story opened and skip the actor-thread post (admin
        # gets the public ack and any real bound POVs land in the
        # fan-out below).
        if triggering_cid:
            actor_text = per_player.get(triggering_cid, response.output_text)
            try:
                ckpt = engine.load_latest(row.session_id)
                actor_char = next(
                    (c for c in ckpt.characters
                     if c.character_id == triggering_cid),
                    None,
                )
                actor_name = actor_char.name if actor_char else triggering_cid
            except Exception:
                logger.exception(
                    "/begin: load_latest failed for actor name lookup",
                )
                actor_name = triggering_cid

            embeds = render_turn(
                output_text=actor_text,
                turn_index=response.turn_index,
                story_id=row.story_id,
            )
            intro = "**The story opens.**"
            venue, thread = await _post_actor_render(
                inter=inter,
                smap=smap,
                user=inter.user,
                character_id=triggering_cid,
                char_name=actor_name,
                embeds=embeds,
                intro_content=intro,
                session_id=row.session_id,
                turn_index=response.turn_index,
            )
            if venue == "thread" and thread is not None:
                await inter.followup.send(
                    f"The story opens in {thread.mention}.",
                    ephemeral=True,
                )
            elif venue == "dm":
                await inter.followup.send(
                    "The story opens in your DMs (POV thread "
                    "unavailable here).",
                    ephemeral=True,
                )
            else:
                await _send_public_turn_render(
                    inter=inter,
                    smap=smap,
                    session_id=row.session_id,
                    turn_index=response.turn_index,
                    content=intro,
                    embeds=embeds,
                )
        else:
            await inter.followup.send(
                embed=render_info(
                    "The story opens",
                    "Each bound player's opening render is being delivered "
                    "to their POV thread.",
                ),
            )

        await _fan_out_per_player_renders(
            inter=inter,
            session_id=row.session_id,
            actor_cid=triggering_cid,
            per_player=per_player,
            turn_index=response.turn_index,
        )

    # ---- /leave -------------------------------------------------------------

    @tree.command(
        name="leave",
        description="Release your character binding in this channel's story.",
        guild=guild,
    )
    async def _leave(inter: discord.Interaction):
        row = await smap.get(_session_channel_id(inter))
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
        row = await smap.get(_session_channel_id(inter))
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
        description="(advanced) Update your character's name or appearance mid-story.",
        guild=guild,
    )
    @app_commands.describe(
        name="New in-story name (optional).",
        appearance="New height, build, clothing, notable features — freeform (optional).",
    )
    async def _describe(
        inter: discord.Interaction,
        name: str = "",
        appearance: str = "",
    ):
        row = await smap.get(_session_channel_id(inter))
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
                    "You aren't bound to a character. Run `/join` first."
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

        # Defensive: if we somehow got here pre-play (no narrator turns
        # yet), fire `(begin)`. The canonical opener is /join's arrival
        # turn now — but the play CLI and any non-/join binding path
        # (manual test setup, future programmatic frontends) can still
        # land here, and we shouldn't silently swallow the opener.
        # v11: narrator_conversations is per-POV dict keyed by character_id.
        # "no turns yet" = no POV has any narrator history.
        is_pre_play = not any(ckpt.narrator_conversations.values())

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
        # event_router prompt recognizes fully-parenthesized input as an
        # author's directive rather than an in-character attempt, so
        # `(begin)` cleanly maps to "compose the opening beat from
        # world_state and place this character in it" — see the
        # `(begin)` instructions in event_router.txt for the full
        # contract. (Pre-v9 this leaned on an authored opening passage
        # extracted at import time; that's gone now and the router
        # composes the opening from world + roster state alone.)
        #
        # v11-r6c note: we pass `(begin)` as a plain `user_input` through
        # /act's normal path. The dispatcher's `format_human_initiator_
        # intention` wraps this as "Alice attempts: (begin)" — visually a
        # bit odd, but the event_router prompt's Author-directive OOC
        # rule (search for "Author-directive OOC" in event_router.txt)
        # fires on the shape of the content itself (fully parenthesized
        # input), NOT on the surrounding "attempts:" framing. So OOC
        # routing is correct here and no code change is needed. If the
        # router prompt's OOC detection ever narrows, revisit the
        # intention framing here.
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
            )
        except TransientLLMError as e:
            logger.warning(
                "opening run_turn hit transient LLM error after %d attempts: %s",
                e.attempts, e.last_error,
            )
            await inter.followup.send(embed=render_error(str(e)))
            return
        except Exception as e:
            logger.exception("opening run_turn failed")
            await inter.followup.send(embed=render_error(
                f"`{type(e).__name__}: {e}`"
            ))
            return

        # Per-phase latency / cache metrics now flow through
        # `app.engine.turn_loop`'s logger ("turn_loop.router[route] …"
        # lines and the per-phase records the orchestrator emits) so
        # they show up in the same place for CLI, Discord, and the
        # programmatic playtest harness. The TurnResponse.debug
        # payload was murdered in v11-r7j.

        embeds = render_turn(
            output_text=response.output_text,
            turn_index=response.turn_index,
            story_id=row.story_id,
        )
        intro_bits: list[str] = []
        if name:
            intro_bits.append(f"**{name}**")
        if appearance:
            intro_bits.append(f"*{appearance}*")
        intro_line = " — ".join(intro_bits)
        intro = (
            (f"{intro_line}\n\n" if intro_line else "")
            + "The story opens. Use `/act <action>` or `/defer` from here on."
        )

        # /describe's defensive pre-play opener follows the same
        # POV-thread-first delivery as /act and /join arrival.
        bound_char = next(
            (c for c in ckpt.characters if c.character_id == binding), None,
        )
        bound_name = bound_char.name if bound_char else binding
        venue, thread = await _post_actor_render(
            inter=inter,
            smap=smap,
            user=inter.user,
            character_id=binding,
            char_name=bound_name,
            embeds=embeds,
            intro_content=intro,
            session_id=row.session_id,
            turn_index=response.turn_index,
        )
        if venue == "thread" and thread is not None:
            await inter.followup.send(
                f"Character updated. Scene opens in {thread.mention}.",
                ephemeral=True,
            )
        elif venue == "dm":
            await inter.followup.send(
                "Character updated. Scene opens in your DMs "
                "(POV thread unavailable here).",
                ephemeral=True,
            )
        else:
            await _send_public_turn_render(
                inter=inter,
                smap=smap,
                session_id=row.session_id,
                turn_index=response.turn_index,
                content=intro,
                embeds=embeds,
            )

    # ---- /act ---------------------------------------------------------------

    @tree.command(
        name="act",
        description="Take a turn in the current story.",
        guild=guild,
    )
    @app_commands.describe(action="What your character does or says.")
    async def _act(inter: discord.Interaction, action: str):
        row = await smap.get(_session_channel_id(inter))
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
                "Run `/join` to pick one.",
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
            )
        except TransientLLMError as e:
            logger.warning(
                "run_turn hit transient LLM error after %d attempts: %s",
                e.attempts, e.last_error,
            )
            await inter.followup.send(embed=render_error(str(e)), ephemeral=True)
            return
        except Exception as e:
            logger.exception("run_turn failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        elapsed = time.monotonic() - start
        logger.info(
            "Turn %d completed for %s by %s in %.1fs (%d chars): %r",
            response.turn_index, row.session_id, inter.user.display_name,
            elapsed, len(response.output_text), action[:120],
        )
        # Per-phase latency / cache metrics now flow through the
        # engine logger (see opening-turn note above). v11-r7j murdered
        # `TurnResponse.debug`; this used to render its `latencies`
        # list here and silently no-op'd because the orchestrator
        # never populated it.

        # v11-r6b/r7a: branch on beat_ended_reason.
        #   - slot_rejected: the orchestrator rejected the /act on slot
        #     grounds; response.output_text carries the user-facing
        #     explanation (with the attempted_text echoed for copy-
        #     paste). Send plain ephemeral — no embed, no fan-out.
        #   - cat_ii_pending: r6a renders a PARTIAL cliffhanger to the
        #     initiator (if human) and pinned responders. Show the
        #     initiator their cliffhanger if present, prefix with a
        #     pause notice, then fan-out the pinned humans below.
        #   - other: normal render to actor + fan-out to non-actors.
        if response.beat_ended_reason == "slot_rejected":
            await inter.followup.send(
                response.output_text or "Your /act could not be accepted.",
                ephemeral=True,
            )
            return

        # Load bindings + roster ONCE for both fan-outs (pre-turn
        # resolutions + the actor's beat). Failure here only kills DMs,
        # not the actor's followup.
        try:
            ckpt_for_fanout = engine.load_latest(row.session_id)
            bindings = ckpt_for_fanout.session.character_bindings or {}
            roster = list(ckpt_for_fanout.characters or [])
        except Exception:
            logger.exception(
                "per-POV fan-out: load_latest failed; skipping DMs",
            )
            bindings = {}
            roster = []

        async def _dm_per_pov(
            renders: dict[str, str],
            *,
            skip_cid: str | None,
            turn_index: int,
            note_prefix: str = "",
        ) -> list[str]:
            """Post each (cid, prose) in `renders` to that user's POV
            thread (DM fallback). Returns the list of character display
            names that were successfully notified. `skip_cid` skips the
            actor (who already saw their render in-channel)."""
            notified: list[str] = []
            for cid, prose in renders.items():
                if cid == skip_cid or not prose:
                    continue
                uid_str = bindings.get(cid, "")
                if not uid_str:
                    continue
                try:
                    uid = int(uid_str)
                except ValueError:
                    continue
                payload = (
                    f"{note_prefix}\n\n{prose}" if note_prefix else prose
                )
                char = next(
                    (c for c in roster if c.character_id == cid), None,
                )
                char_name = char.name if char else cid
                ok = await _post_to_pov(
                    inter=inter,
                    smap=smap,
                    user_id=uid,
                    character_id=cid,
                    char_name=char_name,
                    text=payload,
                    bot=inter.client,
                    session_id=row.session_id,
                    turn_index=turn_index,
                )
                if ok:
                    notified.append(char_name)
            return notified

        # Step 1: fan out pre-turn AFK-sweep resolutions BEFORE the
        # actor's /act render so the in-DM order matches story time.
        # The acting user gets their own DM here too (they may have
        # been the AFK pin holder).
        for pre_resp in (response.pre_turn_resolutions or []):
            await _dm_per_pov(
                pre_resp.per_player_renders or {},
                skip_cid=None,
                turn_index=pre_resp.turn_index,
                note_prefix=(
                    "_(Auto-resolved while you were away — your prior "
                    "beat closed out.)_"
                ),
            )

        # Step 2: actor's /act result. Always routed through
        # `_post_actor_render` so the narrative lands in the actor's
        # private POV thread (DM fallback, public-channel fallback if
        # both private paths fail). Public channel sees an ephemeral
        # pointer at most.
        actor_char = next(
            (c for c in roster if c.character_id == binding), None,
        )
        actor_name = actor_char.name if actor_char else binding
        per_player = response.per_player_renders or {}
        if response.beat_ended_reason == "cat_ii_pending":
            actor_render = per_player.get(binding) or ""
            pause_note = (
                "_Scene paused — waiting on another player's response. "
                "You'll see the beat continue when they /act._"
            )
            if actor_render:
                embeds = render_turn(
                    output_text=actor_render,
                    turn_index=response.turn_index,
                    story_id=row.story_id,
                )
                venue, thread = await _post_actor_render(
                    inter=inter,
                    smap=smap,
                    user=inter.user,
                    character_id=binding,
                    char_name=actor_name,
                    embeds=embeds,
                    intro_content=pause_note,
                    session_id=row.session_id,
                    turn_index=response.turn_index,
                )
                if venue == "thread" and thread is not None:
                    await _clear_interaction_response(inter)
                elif venue == "dm":
                    await _clear_interaction_response(inter)
                else:
                    await _send_public_turn_render(
                        inter=inter,
                        smap=smap,
                        session_id=row.session_id,
                        turn_index=response.turn_index,
                        content=pause_note,
                        embeds=embeds,
                    )
            else:
                await inter.followup.send(content=pause_note, ephemeral=True)
        else:
            embeds = render_turn(
                output_text=response.output_text,
                turn_index=response.turn_index,
                story_id=row.story_id,
            )
            venue, thread = await _post_actor_render(
                inter=inter,
                smap=smap,
                user=inter.user,
                character_id=binding,
                char_name=actor_name,
                embeds=embeds,
                session_id=row.session_id,
                turn_index=response.turn_index,
            )
            if venue == "thread" and thread is not None:
                await _clear_interaction_response(inter)
            elif venue == "dm":
                await _clear_interaction_response(inter)
            else:
                # Both private paths failed — public fallback so the
                # actor still sees their beat.
                await _send_public_turn_render(
                    inter=inter,
                    smap=smap,
                    session_id=row.session_id,
                    turn_index=response.turn_index,
                    embeds=embeds,
                )

        # Step 3: DM the actor's beat to the OTHER bound humans with
        # queued POV renders. Acting user already saw it in-channel above, so
        # skip_cid=binding.
        if per_player:
            notified_names = await _dm_per_pov(
                per_player, skip_cid=binding,
                turn_index=response.turn_index,
            )
            if notified_names:
                try:
                    notified_phrase = ", ".join(
                        f"**{n}**" for n in notified_names
                    )
                    await inter.followup.send(
                        f"({notified_phrase} notified via DM.)",
                        ephemeral=True,
                    )
                except Exception:
                    logger.exception(
                        "per-POV fan-out: ephemeral ack failed",
                    )

    # ---- /defer -------------------------------------------------------------

    @tree.command(
        name="defer",
        description="Take no action and let the scene continue.",
        guild=guild,
    )
    async def _defer(inter: discord.Interaction):
        # Reuse /act so null turns get identical binding, locking, render,
        # and fan-out behavior.
        await _act.callback(inter, "(defer)")

    # ---- /query -------------------------------------------------------------
    # Out-of-character consultation. `/query` now enters the router as
    # a private OOC clarification so the answer is canonically grounded
    # as an observable fact for the asking POV.

    @tree.command(
        name="query",
        description=(
            "Ask an out-of-character question (what do I see, who is X, "
            "what day is it). Ephemeral."
        ),
        guild=guild,
    )
    @app_commands.describe(
        question=(
            "Out-of-character question. Answered from your character's POV "
            "or refused in-fiction if they couldn't know."
        ),
    )
    async def _query(inter: discord.Interaction, question: str):
        if not question.strip():
            await inter.response.send_message(
                "Ask a question — what do you see, who is around, what day "
                "is it, did you meet someone, etc.",
                ephemeral=True,
            )
            return

        row = await smap.get(_session_channel_id(inter))
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
                "Run `/join` to pick one before /query.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True, ephemeral=True)

        try:
            result = await engine.run_query(
                session_id=row.session_id,
                character_id=binding,
                question=question,
            )
        except TransientLLMError as e:
            logger.warning(
                "run_query hit transient LLM error after %d attempts: %s",
                e.attempts, e.last_error,
            )
            await inter.followup.send(str(e), ephemeral=True)
            return
        except Exception as e:
            logger.exception("run_query failed")
            await inter.followup.send(
                f"`{type(e).__name__}: {e}`", ephemeral=True,
            )
            return

        answer = result.answer or "(no response)"
        # Echo the question back italicized so the player has the
        # context inside the ephemeral (Discord ephemerals don't
        # appear in normal scrollback, so the question text isn't
        # visible without this).
        body = f"_> {question.strip()}_\n\n{answer}"
        # Discord caps message bodies at 2000 chars; keep some
        # headroom for the italic prefix.
        if len(body) > 1900:
            body = body[:1900] + "\n…(truncated)"
        await inter.followup.send(body, ephemeral=True)

    # ---- /rewind ------------------------------------------------------------
    # Destructive: deletes ckpt_>target.json from disk. Owner-only by
    # default (admins also allowed) — multi-player sessions need a single
    # voice for "everyone agrees we go back." The flow is two-step:
    # /rewind <N> shows an ephemeral preview with a Confirm button, and
    # only the actual click triggers the cull. This keeps a careless
    # mistype from nuking an arc.

    class _RewindConfirmView(discord.ui.View):
        """Ephemeral confirm/cancel pair shown after /rewind <N> validates.

        Single-use: any button click disables both and stops the view.
        Restricted to the invoker — other players poking the confirm
        on someone else's preview would be a footgun. Times out at 60s
        (the rewind is meant to be deliberate but not laborious).
        """

        def __init__(
            self,
            *,
            session_id: str,
            target_turn: int,
            invoker_id: int,
            preview: "RewindResult",  # forward-ref string to avoid import cycle
        ):
            super().__init__(timeout=60)
            self._session_id = session_id
            self._target_turn = target_turn
            self._invoker_id = invoker_id
            self._preview = preview

        async def interaction_check(
            self, check_inter: discord.Interaction,
        ) -> bool:
            if check_inter.user.id != self._invoker_id:
                await check_inter.response.send_message(
                    "This rewind preview isn't yours.", ephemeral=True,
                )
                return False
            return True

        @discord.ui.button(
            label="Confirm rewind",
            style=discord.ButtonStyle.danger,
        )
        async def _confirm(
            self,
            click_inter: discord.Interaction,
            _button: discord.ui.Button,
        ):
            for child in self.children:
                if isinstance(child, discord.ui.Button):
                    child.disabled = True
            self.stop()

            await click_inter.response.edit_message(
                embed=render_info(
                    "Rewinding...",
                    f"Rewinding to **Turn {self._target_turn}** and "
                    "cleaning up tracked Discord messages.",
                ),
                view=None,
            )

            try:
                result = await engine.rewind_session(
                    self._session_id, self._target_turn,
                )
            except (ValueError, FileNotFoundError) as e:
                await click_inter.edit_original_response(
                    embed=render_error(str(e)), view=None,
                )
                return
            except Exception as e:
                logger.exception("rewind_session failed")
                await click_inter.edit_original_response(
                    embed=render_error(f"`{type(e).__name__}: {e}`"),
                    view=None,
                )
                return

            logger.info(
                "Rewound session %s by %s: %d → %d (deleted %d ckpt(s))",
                self._session_id, click_inter.user.display_name,
                result.previous_latest, result.new_latest,
                len(result.deleted_turns),
            )

            try:
                cleanup = await _delete_rewound_turn_messages(
                    client=click_inter.client,
                    smap=smap,
                    channel_id=_session_channel_id(click_inter),
                    session_id=self._session_id,
                    deleted_turns=result.deleted_turns,
                )
            except Exception:
                logger.exception(
                    "rewind: Discord turn-message cleanup failed",
                )
                cleanup = TurnMessageCleanup(
                    tracked=0,
                    failed=len(result.deleted_turns),
                )

            # Public announcement so co-players know the world
            # backed up. Includes who triggered it for accountability.
            location_line = (
                f"\nResume location: `{result.location}`"
                if result.location else ""
            )
            cleanup_line = ""
            if cleanup.tracked:
                bits = [f"{cleanup.deleted} deleted"]
                if cleanup.hidden:
                    bits.append(f"{cleanup.hidden} hidden")
                if cleanup.missing:
                    bits.append(f"{cleanup.missing} already gone")
                if cleanup.failed:
                    bits.append(f"{cleanup.failed} failed")
                cleanup_line = (
                    "\nDiscord messages: " + ", ".join(bits) + "."
                )
            announce = render_info(
                "Session rewound",
                f"**{click_inter.user.display_name}** rewound the story to "
                f"**Turn {result.new_latest}**.\n"
                f"Turns {result.deleted_turns[0]}–{result.deleted_turns[-1]} "
                f"({len(result.deleted_turns)} checkpoint(s)) erased."
                f"{cleanup_line}{location_line}\n\n"
                f"Use `/act` to continue from here.",
            )
            try:
                channel = _session_text_channel(click_inter)
                if channel is None:
                    raise RuntimeError("no text channel for rewind announcement")
                await channel.send(embed=announce)
                await _clear_interaction_response(click_inter)
            except Exception:
                # Public post or ephemeral cleanup failed. Keep the
                # edited preview as a private confirmation so the
                # invoker still sees the outcome.
                logger.exception(
                    "rewind: public announcement/preview cleanup failed",
                )
                await click_inter.edit_original_response(
                    embed=announce, view=None,
                )

        @discord.ui.button(
            label="Cancel",
            style=discord.ButtonStyle.secondary,
        )
        async def _cancel(
            self,
            click_inter: discord.Interaction,
            _button: discord.ui.Button,
        ):
            for child in self.children:
                if isinstance(child, discord.ui.Button):
                    child.disabled = True
            self.stop()
            await click_inter.response.edit_message(
                embed=render_info(
                    "Rewind cancelled",
                    "Nothing was deleted.",
                ),
                view=None,
            )

    @tree.command(
        name="rewind",
        description=(
            "Rewind the session to an earlier turn. Owner-only. "
            "Permanently deletes later checkpoints."
        ),
        guild=guild,
    )
    @app_commands.describe(
        turn=(
            "The turn number to rewind to (matches embed footer). "
            "Omit to list available turns."
        ),
    )
    async def _rewind(
        inter: discord.Interaction,
        turn: Optional[int] = None,
    ):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start` then `/story start`.",
                ephemeral=True,
            )
            return

        # Owner-or-admin gate. Multi-player sessions could end up with
        # a tug-of-war if every player could rewind; centralize the
        # capability on whoever set up the session. Admins override
        # for ops/recovery scenarios.
        if inter.user.id != row.owner_user_id and not _is_admin(inter.user.id):
            await inter.response.send_message(
                "Only the session owner can `/rewind`. Ask whoever ran "
                "`/session start` to do it.",
                ephemeral=True,
            )
            return

        try:
            turns = engine.list_checkpoint_turns(row.session_id)
        except Exception as e:
            logger.exception("list_checkpoint_turns failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        if not turns:
            await inter.response.send_message(
                "No checkpoints on disk for this session yet.",
                ephemeral=True,
            )
            return

        # Bare /rewind: list available turns so the user can pick.
        if turn is None:
            current = turns[-1]
            playable = [t for t in turns if t < current]
            if not playable:
                body = (
                    f"Current turn: **{current}**\n"
                    f"No earlier turns to rewind to."
                )
            else:
                body = (
                    f"Current turn: **{current}**\n"
                    f"Rewindable range: **{playable[0]}** to **{playable[-1]}** "
                    f"({len(playable)} checkpoint(s)).\n\n"
                    f"Run `/rewind <turn>` with a number in that range."
                )
            await inter.response.send_message(
                embed=render_info("Available turns", body), ephemeral=True,
            )
            return

        try:
            preview = engine.preview_rewind(row.session_id, turn)
        except (ValueError, FileNotFoundError) as e:
            await inter.response.send_message(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("preview_rewind failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        location_line = (
            f"\n**Resume location:** `{preview.location}`"
            if preview.location else ""
        )
        actor_line = (
            f"\n**Resume actor:** `{preview.actor_character_id}`"
            if preview.actor_character_id else ""
        )
        body = (
            f"**Current turn:** {preview.previous_latest}\n"
            f"**Rewind target:** {preview.target_turn}\n"
            f"**Will delete:** turns "
            f"{preview.deleted_turns[0]}–{preview.deleted_turns[-1]} "
            f"({len(preview.deleted_turns)} checkpoint(s))"
            f"{actor_line}{location_line}\n\n"
            f"⚠️ This permanently erases the deleted turns from disk and "
            f"deletes or hides tracked Discord messages for those turns. "
            f"Players who joined after turn {preview.target_turn} will "
            f"need to `/join` again."
        )
        view = _RewindConfirmView(
            session_id=row.session_id,
            target_turn=turn,
            invoker_id=inter.user.id,
            preview=preview,
        )
        await inter.response.send_message(
            embed=render_info("Confirm rewind?", body),
            view=view,
            ephemeral=True,
        )

    # ---- /status ------------------------------------------------------------

    @tree.command(
        name="status",
        description="Show the current state of this channel's story.",
        guild=guild,
    )
    async def _status(inter: discord.Interaction):
        row = await smap.get(_session_channel_id(inter))
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
            await smap.delete(_session_channel_id(inter))
            await inter.response.send_message(
                f"Session `{row.session_id}` had no checkpoint on disk; "
                f"the channel mapping has been purged. Run `/session start` "
                f"then `/story start` to begin fresh.",
                ephemeral=True,
            )
            return

        # v11: pick the invoker's POV location label.
        from app.engine.context_builder import pov_location_for_user
        location = pov_location_for_user(ckpt, user_id=str(inter.user.id))
        location_name = location or "(no active location)"

        bound_ids = set(ckpt.session.character_bindings.keys())
        present = [
            c.name for c in ckpt.characters
            if c.location == location and c.status.value == "active"
            and c.character_id not in bound_ids
        ] if location else []

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
            f"**Location:** {location_name}",
            f"**Same-location NPCs:** {', '.join(present) if present else 'no one else'}",
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
        description="Delete recent messages AND every thread in this channel. Admin-only.",
        guild=guild,
    )
    @app_commands.describe(
        count="How many channel messages to delete (1-1000, default 100). Threads are always nuked entirely.",
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
        # Manage Messages handles channel.purge; Manage Threads handles deleting
        # threads the bot did not author (POV threads are bot-owned and don't
        # need it, but any user-spawned thread does).
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

        # 1) Delete every thread on this channel. This is the half of "clear"
        # that the channel-only purge was missing — POV threads (and any
        # other public/private threads attached to the channel) live in
        # their own message stores and survive channel.purge() entirely.
        # We also sweep archived threads so the sidebar/archived view ends
        # up empty too. Best-effort per thread: a single Forbidden on a
        # user-owned thread shouldn't abort the whole sweep.
        threads_deleted = 0
        thread_failures = 0

        async def _try_delete(thread: discord.Thread) -> None:
            nonlocal threads_deleted, thread_failures
            try:
                await thread.delete()
                threads_deleted += 1
            except discord.Forbidden:
                # Likely missing Manage Threads on a user-owned thread.
                # Fall back to purging messages so at least the content
                # is gone even if the thread shell stays.
                thread_failures += 1
                logger.warning(
                    "clear: cannot delete thread %s (Forbidden); "
                    "purging messages instead", thread.id,
                )
                try:
                    await thread.purge(limit=1000)
                except Exception:
                    logger.exception(
                        "clear: thread.purge fallback also failed for %s",
                        thread.id,
                    )
            except Exception:
                thread_failures += 1
                logger.exception("clear: thread.delete failed for %s", thread.id)

        # Active threads (visible in the sidebar). Snapshot the list
        # because deleting mutates channel.threads.
        for t in list(channel.threads):
            await _try_delete(t)

        # Archived public threads.
        try:
            async for t in channel.archived_threads(limit=None):
                await _try_delete(t)
        except discord.Forbidden:
            logger.warning(
                "clear: cannot enumerate archived public threads in #%s "
                "(missing Read Message History?); skipping",
                channel.name,
            )
        except Exception:
            logger.exception(
                "clear: archived_threads(public) iteration failed in #%s",
                channel.name,
            )

        # Archived private threads (where bot-created POV threads land
        # after the auto-archive timer fires). joined=False asks Discord
        # for *all* private archived threads on the channel, not just
        # the ones the bot has joined — needs Manage Threads.
        if perms.manage_threads:
            try:
                async for t in channel.archived_threads(
                    private=True, joined=False, limit=None,
                ):
                    await _try_delete(t)
            except discord.Forbidden:
                logger.warning(
                    "clear: cannot enumerate archived private threads in #%s",
                    channel.name,
                )
            except Exception:
                logger.exception(
                    "clear: archived_threads(private) iteration failed in #%s",
                    channel.name,
                )
        else:
            # Fall back to the joined=True variant, which doesn't need
            # Manage Threads but only sees threads the bot is a member
            # of. Bot-created POV threads qualify, so this still catches
            # the common case.
            try:
                async for t in channel.archived_threads(
                    private=True, joined=True, limit=None,
                ):
                    await _try_delete(t)
            except Exception:
                logger.exception(
                    "clear: archived_threads(private,joined) iteration "
                    "failed in #%s", channel.name,
                )

        # 2) Drop the POV-thread SQL cache for this channel so the next
        # /join doesn't try to send to a thread id we just deleted.
        # Keep the session row itself intact — /clear is not /session end.
        try:
            cache_dropped = await smap.clear_all_pov_threads(channel.id)
        except Exception:
            logger.exception(
                "clear: clear_all_pov_threads failed for channel %s",
                channel.id,
            )
            cache_dropped = 0
        try:
            turn_refs_dropped = await smap.clear_all_turn_messages(channel.id)
        except Exception:
            logger.exception(
                "clear: clear_all_turn_messages failed for channel %s",
                channel.id,
            )
            turn_refs_dropped = 0

        # 3) Purge the channel itself. purge() uses bulk_delete for
        # messages < 14 days and falls back to per-message delete for
        # older ones (slow, rate-limited).
        try:
            deleted = await channel.purge(limit=count)
        except discord.Forbidden:
            await inter.followup.send(
                "Forbidden — Manage Messages permission got revoked mid-call. "
                f"(Threads cleared: {threads_deleted}.)",
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
            "Cleared %d messages and %d threads (%d pov cache rows, "
            "%d turn-message rows) "
            "in #%s by %s; %d thread failures",
            len(deleted), threads_deleted, cache_dropped,
            turn_refs_dropped,
            channel.name, inter.user.display_name, thread_failures,
        )

        parts = [f"Deleted {len(deleted)} message(s) and {threads_deleted} thread(s)."]
        if thread_failures:
            parts.append(
                f"{thread_failures} thread(s) could not be deleted "
                "(see logs — likely missing **Manage Threads**)."
            )
        if not perms.manage_threads:
            parts.append(
                "Tip: grant me **Manage Threads** so I can also nuke "
                "user-spawned and archived private threads."
            )
        await inter.followup.send("\n".join(parts), ephemeral=True)

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
        row = await smap.get(_session_channel_id(inter))
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
        row = await smap.get(_session_channel_id(inter))
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
    # admin invoking this command; it force-releases the beat slot
    # state and abandons any open Cat II events. The
    # event LOG is preserved — this does not rewrite history, just
    # unblocks the next /act.
    @tree.command(
        name="abort_beat",
        description="[Admin] Force-unwedge the current beat.",
        guild=guild,
    )
    async def _abort_beat(inter: discord.Interaction):
        # v11-A5: audit-log entry fires before any work so even a failure
        # path leaves a trail of who ran it, where, and when.
        logger.info(
            "abort_beat invoked: admin_id=%s channel_id=%s",
            inter.user.id, inter.channel_id,
        )
        if not _is_admin(inter.user.id):
            await inter.response.send_message(
                "Admin-only command.", ephemeral=True,
            )
            return
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message(
                "No session bound to this channel.", ephemeral=True,
            )
            return
        try:
            from app.engine.turn_loop import abort_beat
            ckpt = engine.load_latest(row.session_id)
            dropped = abort_beat(ckpt)
            engine.checkpoint_mgr.save(ckpt)
        except Exception as e:
            logger.exception("abort_beat failed")
            await inter.response.send_message(
                f"abort failed: `{type(e).__name__}: {e}`", ephemeral=True,
            )
            return
        # v11-A5: audit-log the outcome so ops can correlate an admin
        # identity with the events that got abandoned.
        logger.info(
            "abort_beat completed: admin_id=%s channel_id=%s session_id=%s "
            "dropped=%d",
            inter.user.id, inter.channel_id, row.session_id, dropped,
        )
        await inter.response.send_message(
            f"Beat released. {dropped} pending reaction(s) "
            f"cancelled. Next /act can claim the slot.",
            ephemeral=True,
        )
        # v11-A5: thread-visible notification so players see that the
        # beat was released without needing an admin to manually re-echo
        # the state. Skipped for no-op aborts (nothing to announce when
        # zero events were dropped).
        if dropped > 0:
            try:
                await inter.followup.send(
                    f"Beat released by admin. {dropped} pending "
                    f"reaction(s) cancelled — the next /act reopens the "
                    f"beat.",
                    ephemeral=False,
                )
            except Exception:
                # Follow-up posts can fail if the channel permissions
                # shifted or the interaction expired; swallow so the
                # admin still sees their ephemeral confirmation.
                logger.exception(
                    "abort_beat: thread-visible notification failed",
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
