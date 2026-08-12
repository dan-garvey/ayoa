from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from io import BytesIO
from typing import Optional

import discord

from app.bot.session_map import SessionMap, TurnMessageRef
from app.engine.player_media import PlayerMediaBytes


logger = logging.getLogger(__name__)


async def record_turn_message(
    *,
    smap: SessionMap,
    session_channel_id: int,
    session_id: str,
    turn_index: Optional[int],
    message: object,
    delivery: str,
    discord_channel_id: Optional[int] = None,
    recipient_user_id: Optional[int] = None,
) -> TurnMessageRef | None:
    if not session_id or turn_index is None:
        return None
    message_id = getattr(message, "id", None)
    if message_id is None:
        return None
    message_channel = getattr(message, "channel", None)
    channel_id = discord_channel_id or getattr(message_channel, "id", None)
    if channel_id is None:
        return None
    ref = TurnMessageRef(
        channel_id=session_channel_id,
        session_id=session_id,
        turn_index=int(turn_index),
        discord_channel_id=int(channel_id),
        message_id=int(message_id),
        delivery=delivery,
        recipient_user_id=recipient_user_id,
        created_at=0,
    )
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
        return ref
    except Exception:
        logger.exception(
            "turn-message tracking failed for session=%s turn=%s message=%s",
            session_id,
            turn_index,
            message_id,
        )
        return None


async def ensure_pov_thread(
    *,
    channel: discord.abc.Messageable,
    user: discord.abc.User,
    smap: SessionMap,
    character_id: str,
    char_name: str,
) -> Optional[discord.Thread]:
    """Get or create a private POV thread, returning None for DM fallback."""

    if not isinstance(channel, discord.TextChannel):
        return None
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
                    "ensure_pov_thread: fetch_channel(%s) failed",
                    cached,
                )
                thread = None
        if isinstance(thread, discord.Thread) and not thread.archived:
            return thread
        await smap.clear_pov_thread(channel.id, user.id)

    suffix = (char_name or character_id or "pov").strip()[:60] or "pov"
    thread_name = f"{user.display_name} · {suffix}"
    try:
        thread = await channel.create_thread(
            name=thread_name[:100],
            type=discord.ChannelType.private_thread,
            invitable=False,
            auto_archive_duration=10080,
            reason=f"POV thread for {user.display_name} ({character_id})",
        )
    except discord.Forbidden:
        logger.warning(
            "ensure_pov_thread: missing CREATE_PRIVATE_THREADS in #%s "
            "(channel %s); falling back to DM for user %s",
            channel.name,
            channel.id,
            user.id,
        )
        return None
    except Exception:
        logger.exception(
            "ensure_pov_thread: create_thread failed in #%s for user %s",
            channel.name,
            user.id,
        )
        return None

    try:
        await thread.add_user(user)
    except Exception:
        logger.exception(
            "ensure_pov_thread: add_user(%s) failed on thread %s; "
            "falling back to DM",
            user.id,
            thread.id,
        )
        return None

    await smap.set_pov_thread(
        channel_id=channel.id,
        user_id=user.id,
        thread_id=thread.id,
        character_id=character_id,
    )
    return thread


async def deliver_player_media(
    *,
    client: discord.Client,
    smap: SessionMap,
    session_channel_id: int,
    user_id: int,
    character_id: str,
    char_name: str,
    media: PlayerMediaBytes,
    caption: str | None,
    alt_text: str | None,
    session_id: str,
    turn_index: Optional[int],
    parent_channel: discord.TextChannel | None = None,
    delivery_label: str = "asset",
    preferred_thread: discord.Thread | None = None,
    resolve_thread: bool = True,
    delivery_is_current: Callable[[], Awaitable[bool]] | None = None,
) -> bool:
    """Deliver validated bytes privately: POV thread, then DM, never public."""

    if delivery_is_current is not None and not await delivery_is_current():
        return False

    user = client.get_user(user_id)
    if user is None:
        try:
            user = await client.fetch_user(user_id)
        except Exception:
            logger.exception(
                "player media delivery: fetch_user(%s) failed",
                user_id,
            )
            return False

    channel = parent_channel
    if channel is None:
        cached = client.get_channel(session_channel_id)
        if isinstance(cached, discord.TextChannel):
            channel = cached
        else:
            try:
                fetched = await client.fetch_channel(session_channel_id)
            except Exception:
                fetched = None
            if isinstance(fetched, discord.TextChannel):
                channel = fetched

    thread = preferred_thread
    if thread is None and resolve_thread and channel is not None:
        thread = await ensure_pov_thread(
            channel=channel,
            user=user,
            smap=smap,
            character_id=character_id,
            char_name=char_name,
        )

    def _file() -> discord.File:
        return discord.File(
            BytesIO(media.data),
            filename=media.filename,
            description=alt_text or None,
        )

    if thread is not None:
        if delivery_is_current is not None and not await delivery_is_current():
            return False
        try:
            message = await thread.send(content=caption, file=_file())
        except Exception:
            logger.warning(
                "player media delivery: thread.send to %s failed; "
                "falling back to DM",
                thread.id,
            )
            await smap.clear_pov_thread(session_channel_id, user_id)
        else:
            tracked_ref = await record_turn_message(
                smap=smap,
                session_channel_id=session_channel_id,
                session_id=session_id,
                turn_index=turn_index,
                message=message,
                delivery=f"thread_{delivery_label}",
                discord_channel_id=thread.id,
                recipient_user_id=user_id,
            )
            if session_id and turn_index is not None and tracked_ref is None:
                fallback_ref = _message_ref(
                    message=message,
                    session_channel_id=session_channel_id,
                    session_id=session_id,
                    turn_index=turn_index,
                    delivery=f"thread_{delivery_label}",
                    discord_channel_id=thread.id,
                    recipient_user_id=user_id,
                )
                removed = await _remove_stale_media_message(message)
                if not removed and fallback_ref is not None:
                    await _queue_media_cleanup(smap, fallback_ref)
                return False
            if (
                delivery_is_current is not None
                and not await delivery_is_current()
            ):
                removed = await _remove_stale_media_message(message)
                if removed and tracked_ref is not None:
                    await smap.forget_turn_messages([tracked_ref])
                elif tracked_ref is not None:
                    await _queue_media_cleanup(smap, tracked_ref)
                return False
            return True

    if delivery_is_current is not None and not await delivery_is_current():
        return False
    try:
        message = await user.send(content=caption, file=_file())
    except Exception:
        logger.warning(
            "player media delivery: DM fallback to user %s failed",
            user_id,
        )
        return False
    tracked_ref = await record_turn_message(
        smap=smap,
        session_channel_id=session_channel_id,
        session_id=session_id,
        turn_index=turn_index,
        message=message,
        delivery=f"dm_{delivery_label}",
        recipient_user_id=user_id,
    )
    if session_id and turn_index is not None and tracked_ref is None:
        fallback_ref = _message_ref(
            message=message,
            session_channel_id=session_channel_id,
            session_id=session_id,
            turn_index=turn_index,
            delivery=f"dm_{delivery_label}",
            recipient_user_id=user_id,
        )
        removed = await _remove_stale_media_message(message)
        if not removed and fallback_ref is not None:
            await _queue_media_cleanup(smap, fallback_ref)
        return False
    if delivery_is_current is not None and not await delivery_is_current():
        removed = await _remove_stale_media_message(message)
        if removed and tracked_ref is not None:
            await smap.forget_turn_messages([tracked_ref])
        elif tracked_ref is not None:
            await _queue_media_cleanup(smap, tracked_ref)
        return False
    return True


def _message_ref(
    *,
    message: object,
    session_channel_id: int,
    session_id: str,
    turn_index: int,
    delivery: str,
    discord_channel_id: int | None = None,
    recipient_user_id: int | None = None,
) -> TurnMessageRef | None:
    message_id = getattr(message, "id", None)
    message_channel = getattr(message, "channel", None)
    channel_id = discord_channel_id or getattr(message_channel, "id", None)
    if message_id is None or channel_id is None:
        return None
    return TurnMessageRef(
        channel_id=session_channel_id,
        session_id=session_id,
        turn_index=int(turn_index),
        discord_channel_id=int(channel_id),
        message_id=int(message_id),
        delivery=delivery,
        recipient_user_id=recipient_user_id,
        created_at=0,
    )


async def _queue_media_cleanup(
    smap: SessionMap,
    ref: TurnMessageRef,
) -> None:
    try:
        await smap.record_media_cleanup(ref)
    except Exception:
        logger.exception("failed to persist stale media cleanup retry")


async def retry_media_cleanup_outbox(
    *,
    client: discord.Client,
    smap: SessionMap,
) -> int:
    """Retry durable cleanup for media that could not be removed inline."""

    try:
        refs = await smap.list_media_cleanup()
    except Exception:
        logger.exception("failed to load media cleanup outbox")
        return 0
    handled: list[TurnMessageRef] = []
    for ref in refs:
        channel = await _cleanup_channel(client, ref)
        if channel is None:
            continue
        try:
            message = await channel.fetch_message(ref.message_id)
        except discord.NotFound:
            handled.append(ref)
            continue
        except Exception:
            logger.warning(
                "media cleanup retry could not fetch message %s",
                ref.message_id,
            )
            continue
        if await _remove_stale_media_message(message):
            handled.append(ref)
    if handled:
        await smap.forget_turn_messages(handled)
        await smap.forget_media_cleanup(handled)
    return len(handled)


async def _cleanup_channel(
    client: discord.Client,
    ref: TurnMessageRef,
) -> object | None:
    if ref.delivery.startswith("dm") and ref.recipient_user_id is not None:
        user = client.get_user(ref.recipient_user_id)
        if user is None:
            try:
                user = await client.fetch_user(ref.recipient_user_id)
            except Exception:
                return None
        channel = getattr(user, "dm_channel", None)
        if channel is None:
            try:
                channel = await user.create_dm()
            except Exception:
                return None
        return channel
    channel = client.get_channel(ref.discord_channel_id)
    if channel is not None:
        return channel
    try:
        return await client.fetch_channel(ref.discord_channel_id)
    except Exception:
        return None


async def _remove_stale_media_message(message: object) -> bool:
    try:
        await message.delete()
        return True
    except discord.NotFound:
        return True
    except Exception:
        logger.warning(
            "stale generated media delete failed; trying attachment removal"
        )
    try:
        await message.edit(
            content="_Rewound illustration hidden._",
            embeds=[],
            attachments=[],
        )
        return True
    except Exception:
        logger.exception("stale generated media attachment removal failed")
        return False
