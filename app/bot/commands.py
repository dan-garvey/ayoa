"""Slash commands for the narrative-engine Discord bot.

Commands:
    /session start <name>                 — create + bind an empty session to this channel
    /session end                          — detach this channel's session
    /session resume                       — show the last turn
    /session list                         — list existing sessions
    /story list                           — list available stories
    /story start <story_id>               — load a story into the current session
    /story info <story_id>                — show briefing for a source story
    /story delete                         — unload the story from this session
    /join                                 — pick a character via interactive menu
                                            (pre-play: enters the lobby; post-play: arrives mid-story)
    /attach <json> [name_override]        — attach a D&D Beyond JSON sheet to your character
    /sheet [page]                         — show your attached D&D character sheet
    /inventory                            — show your current D&D inventory
    /loot list/take/take_all/split/decline — inspect and claim D&D loot offers
    /xp award                             — admin-only D&D experience award
    /begin                                — open the story for everyone in the lobby
                                            (any bound player or admin can fire)
    /leave                                — release your character
    /describe [name] [appearance]         — set/update name and appearance
    /act <action>                         — submit a turn
    /retry                                — retry a failed narrator render
    /defer                                — submit no action and let the scene continue
    /query <question>                     — ask an out-of-character question
    /status                               — summarize current state

The bot calls the engine in-process (no HTTP). Each turn runs under a
per-session lock so concurrent /act commands on the same channel serialize.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Optional

import discord
from discord import app_commands

from app.bot.embed import (
    MAX_DESCRIPTION,
    render_briefing,
    render_error,
    render_info,
    render_turn,
)
from app.bot.engine_bridge import (
    EngineBridge,
)
from app.bot.player_errors import player_safe_error_message
from app.engine import dnd_experience, dnd_inventory, dnd_presentation
from app.engine.content_asset_bytes import (
    AssetByteResolutionError,
    ResolvedAssetBytes,
    resolve_asset_bytes,
)
from app.engine.content_assets import load_asset_catalog
from app.engine.frontend_views import (
    CompletedPendingRoll,
    DndCombatView,
    DndExperienceAwardResult,
    DndInventoryView,
    DndLootClaimResult,
    DndSheetAttachmentSummary,
    PendingRollPrompt,
    RewindResult,
)
from app.schemas.dnd_inventory import DndLootOffer
from app.bot.session_map import SessionMap, TurnMessageRef
from app.llm.client import TransientLLMError
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import redact_imported_asset_text
from app.schemas.content_pack import ContentImageAsset, SafeAssetRevealPayload
from app.schemas.responses import DiceRollDisplay, TurnResponse

logger = logging.getLogger(__name__)

# D&D Beyond browser exports are larger than story prompts because they
# include source data for spells, actions, and inventory. The attachment
# still stays out of prompts; it is stored in checkpoints for rewind and
# projected down before any rules packet reaches the router.
MAX_DND_CHARACTER_BYTES = 5_000_000

DND_SHEET_PAGES = (
    "overview",
    "abilities",
    "actions",
    "spells",
    "inventory",
    "features",
)
DICE_ROLL_ANIMATION_DELAY_S = 0.35

DISCORD_SELECT_OPTION_TEXT_MAX = 100


def _discord_select_text(
    text: str,
    limit: int = DISCORD_SELECT_OPTION_TEXT_MAX,
) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _join_select_label(
    name: str,
    character_id: str,
    *,
    role: str = "",
    appearance: str = "",
) -> str:
    for candidate in (name, role, appearance, character_id, "Character"):
        label = _discord_select_text(candidate)
        if label:
            return label
    return "Character"


def _render_combat_status(
    view: DndCombatView,
    *,
    include_map: bool = False,
) -> discord.Embed:
    if not view.active:
        return render_info("Combat", view.message or "No active combat.")
    lines = dnd_presentation.combat_status_lines(
        view,
        markdown=True,
        include_map=include_map,
        max_chars=MAX_DESCRIPTION if include_map else None,
    )
    return render_info("Combat", "\n".join(lines))


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


def _safe_cleanup_error(exc: BaseException) -> str:
    text = redact_imported_asset_text(f"{type(exc).__name__}: {exc}")
    return text or type(exc).__name__


def _roll_result_line(result: CompletedPendingRoll) -> str:
    crit_note = ""
    if result.crit == "crit":
        crit_note = " Critical success."
    elif result.crit == "fail":
        crit_note = " Natural 1."
    detail = result.detail or f"{result.expression} = `{result.total}`"
    return f"**Rolled {result.label}:** {detail}.{crit_note}"


def _roll_prompt_content(
    *,
    prompt: PendingRollPrompt,
    char_name: str,
    result: CompletedPendingRoll | None = None,
    interpreting: bool = False,
) -> str:
    reason = f"\n{prompt.reason}" if prompt.reason else ""
    lines = [
        f"**{char_name}** needs to roll **{prompt.label}**.{reason}",
    ]
    if result is not None:
        lines.append("")
        lines.append(_roll_result_line(result))
    if interpreting:
        lines.append("")
        lines.append("_Resolving..._")
    return "\n".join(lines)


def _signed_modifier(value: int) -> str:
    if value > 0:
        return f" + {value}"
    if value < 0:
        return f" - {abs(value)}"
    return " + 0"


def _dice_roll_heading(roll: DiceRollDisplay | CompletedPendingRoll) -> str:
    actor = (
        getattr(roll, "actor_name", "")
        or getattr(roll, "actor_id", "")
        or "Unknown"
    )
    label = getattr(roll, "label", "") or "Roll"
    target = (
        getattr(roll, "target_name", "")
        or getattr(roll, "target_id", "")
        or ""
    )
    heading = f"{actor} - {label}" if actor else label
    return f"{heading} vs {target}" if target else heading


def _dice_d20_text(roll: DiceRollDisplay | CompletedPendingRoll) -> str:
    values = list(getattr(roll, "die_values", ()) or ())
    kept = list(getattr(roll, "kept_die_values", ()) or values)
    if not values:
        match = re.search(
            r"1d20\s*\((?:\*\*)?(\d+)",
            str(getattr(roll, "detail", "") or ""),
        )
        if match:
            values = [int(match.group(1))]
            kept = values
    if not values:
        return "?"
    if len(values) == 1:
        return str(values[0])
    kept_text = "/".join(str(v) for v in kept) if kept else str(values[-1])
    return f"{'/'.join(str(v) for v in values)} -> {kept_text}"


def _dice_roll_formula(
    roll: DiceRollDisplay | CompletedPendingRoll,
) -> str:
    total = int(getattr(roll, "total", 0) or 0)
    modifier = int(getattr(roll, "modifier", 0) or 0)
    if modifier == 0:
        match = re.fullmatch(
            r"1d20(?P<modifier>[+-]\d+)?",
            str(getattr(roll, "expression", "") or "").strip(),
        )
        if match and match.group("modifier"):
            modifier = int(match.group("modifier"))
    dc = int(getattr(roll, "dc", 0) or 0)
    line = f"d20 {_dice_d20_text(roll)}{_signed_modifier(modifier)} = {total}"
    if dc:
        line = f"{line} vs DC {dc}"
    return line


def _dice_has_d20_roll_evidence(
    roll: DiceRollDisplay | CompletedPendingRoll,
) -> bool:
    if list(getattr(roll, "die_values", ()) or ()):
        return True
    expression = str(getattr(roll, "expression", "") or "").strip()
    if expression.startswith("1d20"):
        return True
    detail = str(getattr(roll, "detail", "") or "")
    return "1d20" in detail


def _dice_is_damage_only_roll(
    roll: DiceRollDisplay | CompletedPendingRoll,
) -> bool:
    kind = str(getattr(roll, "kind", "") or "")
    if kind == "damage_roll":
        return True
    return bool(getattr(roll, "damage_total", 0) or 0) and not (
        _dice_has_d20_roll_evidence(roll)
    )


def _dice_is_healing_roll(
    roll: DiceRollDisplay | CompletedPendingRoll,
) -> bool:
    return str(getattr(roll, "kind", "") or "") == "healing_roll"


def _plain_dice_detail(text: str) -> str:
    return " ".join(str(text or "").replace("`", "").replace("**", "").split())


def _dice_damage_formula(roll: DiceRollDisplay | CompletedPendingRoll) -> str:
    total = int(
        getattr(roll, "damage_total", 0)
        or getattr(roll, "total", 0)
        or 0
    )
    raw_total = int(getattr(roll, "damage_raw_total", 0) or 0)
    damage_type = str(getattr(roll, "damage_type", "") or "")
    suffix = f" {damage_type}" if damage_type else ""
    detail = _plain_dice_detail(str(getattr(roll, "damage_detail", "") or ""))
    expression = str(getattr(roll, "damage_expression", "") or "").strip()
    if detail:
        line = f"Damage: {detail}{suffix}"
    elif expression:
        line = f"Damage: {expression} = {total}{suffix}"
    else:
        line = f"Damage: {total}{suffix}"
    if raw_total and raw_total != total:
        line = f"{line}; applied {total}{suffix} after adjustments"
    return line


def _dice_hp_text(current: int, maximum: int, temporary: int = 0) -> str:
    text = f"{current}/{maximum}" if maximum else str(current)
    if temporary:
        text = f"{text} (+{temporary} temp)"
    return text


def _dice_damage_target_status(
    roll: DiceRollDisplay | CompletedPendingRoll,
) -> str:
    maximum = int(getattr(roll, "target_hp_max", 0) or 0)
    before = int(getattr(roll, "target_hp_before", 0) or 0)
    after = int(getattr(roll, "target_hp_after", 0) or 0)
    temp_before = int(getattr(roll, "target_temp_hp_before", 0) or 0)
    temp_after = int(getattr(roll, "target_temp_hp_after", 0) or 0)
    if not any((maximum, before, after, temp_before, temp_after)):
        return ""
    target = (
        getattr(roll, "target_name", "")
        or getattr(roll, "target_id", "")
        or "Target"
    )
    before_text = _dice_hp_text(before, maximum, temp_before)
    after_text = _dice_hp_text(after, maximum, temp_after)
    state = str(getattr(roll, "target_defeat_state", "") or "")
    state_text = (
        f"; {state}" if state and state not in {"active", "unknown"} else ""
    )
    return f"Target HP: {target} {before_text} -> {after_text}{state_text}"


def _dice_healing_formula(roll: DiceRollDisplay | CompletedPendingRoll) -> str:
    total = int(getattr(roll, "total", 0) or 0)
    detail = _plain_dice_detail(str(getattr(roll, "detail", "") or ""))
    expression = str(getattr(roll, "expression", "") or "").strip()
    if detail:
        return f"Healing: {detail}"
    if expression:
        return f"Healing: {expression} = {total}"
    return f"Healing: {total}"


def _dice_roll_content(
    roll: DiceRollDisplay | CompletedPendingRoll,
    *,
    stage: str = "final",
    interpreting: bool = False,
) -> str:
    healing_roll = _dice_is_healing_roll(roll)
    damage_only = _dice_is_damage_only_roll(roll)
    heading_kind = (
        "D&D Healing" if healing_roll
        else "D&D Damage" if damage_only
        else "D&D Roll"
    )
    heading = f"**{heading_kind}: {_dice_roll_heading(roll)}**"
    if stage == "rolling":
        rolling = (
            "`healing rolling...`" if healing_roll
            else "`damage rolling...`" if damage_only
            else "`d20 rolling...`"
        )
        lines = [heading, rolling]
    elif stage == "settled":
        settled = (
            _dice_healing_formula(roll) if healing_roll
            else _dice_damage_formula(roll) if damage_only
            else f"d20 {_dice_d20_text(roll)}"
        )
        lines = [heading, f"`{settled}`"]
    elif healing_roll:
        lines = [heading, f"`{_dice_healing_formula(roll)}`"]
        status = _dice_damage_target_status(roll)
        if status:
            lines.append(status)
    elif damage_only:
        lines = [heading, f"`{_dice_damage_formula(roll)}`"]
        status = _dice_damage_target_status(roll)
        if status:
            lines.append(status)
    else:
        lines = [heading, f"`{_dice_roll_formula(roll)}`"]
        outcome = str(getattr(roll, "outcome", "") or "")
        if outcome:
            lines.append(f"**{outcome.title()}**")
        crit = str(getattr(roll, "crit", "") or "")
        if crit == "crit":
            lines.append("_Critical success._")
        elif crit == "fail":
            lines.append("_Natural 1._")
        damage_total = int(getattr(roll, "damage_total", 0) or 0)
        if damage_total > 0:
            lines.append(f"`{_dice_damage_formula(roll)}`")
            status = _dice_damage_target_status(roll)
            if status:
                lines.append(status)
    if interpreting:
        lines.append("")
        lines.append("_Resolving..._")
    return "\n".join(lines)


def _experience_award_content(award: Any) -> str:
    name = (
        str(getattr(award, "character_name", "") or "")
        or str(getattr(award, "character_id", "") or "Character")
    )
    amount = int(getattr(award, "amount", 0) or 0)
    source = str(getattr(award, "source", "") or "").strip()
    progress = dnd_experience.format_experience_progress({
        "experience_points": getattr(award, "experience_points", 0),
        "total_level": getattr(award, "total_level", 0),
        "eligible_level": getattr(award, "eligible_level", 0),
        "level_available": bool(getattr(award, "eligible_level", 0)),
        "next_level": getattr(award, "next_level", 0),
        "xp_to_next_level": getattr(award, "xp_to_next_level", 0),
    })
    source_text = f"\n{source}" if source else ""
    return f"**XP Gained:** {name} gains **{amount:,} XP**.{source_text}\n{progress}"


async def _animate_interaction_roll_result(
    inter: discord.Interaction,
    result: CompletedPendingRoll,
    *,
    initial_response_sent: bool,
    view: Optional[discord.ui.View] = None,
    interpreting: bool = True,
) -> None:
    rolling = _dice_roll_content(result, stage="rolling")
    final = _dice_roll_content(
        result, stage="final", interpreting=interpreting,
    )
    try:
        if initial_response_sent:
            await inter.edit_original_response(content=rolling, view=view)
        elif inter.response.is_done():
            await inter.followup.send(rolling, ephemeral=True)
            initial_response_sent = True
        else:
            await inter.response.send_message(
                rolling,
                ephemeral=True,
                view=view,
            )
            initial_response_sent = True
        await asyncio.sleep(DICE_ROLL_ANIMATION_DELAY_S)
        await inter.edit_original_response(
            content=_dice_roll_content(result, stage="settled"),
            view=view,
        )
        await asyncio.sleep(DICE_ROLL_ANIMATION_DELAY_S)
        await inter.edit_original_response(content=final, view=view)
    except Exception:
        logger.debug("dice roll animation failed", exc_info=True)
        fallback = "\n".join([
            _roll_result_line(result),
            "_Resolving..._" if interpreting else "",
        ]).strip()
        if inter.response.is_done():
            await inter.followup.send(fallback, ephemeral=True)
        else:
            await inter.response.send_message(
                fallback,
                ephemeral=True,
                view=view,
            )


async def _send_roll_animation_to_messageable(
    *,
    target,
    roll: DiceRollDisplay,
    smap: SessionMap,
    session_channel_id: int,
    session_id: str,
    turn_index: Optional[int],
    delivery: str,
    discord_channel_id: Optional[int] = None,
    recipient_user_id: Optional[int] = None,
) -> None:
    msg = await target.send(_dice_roll_content(roll, stage="rolling"))
    await _record_turn_message(
        smap=smap,
        session_channel_id=session_channel_id,
        session_id=session_id,
        turn_index=turn_index,
        message=msg,
        delivery=delivery,
        discord_channel_id=discord_channel_id,
        recipient_user_id=recipient_user_id,
    )
    try:
        await asyncio.sleep(DICE_ROLL_ANIMATION_DELAY_S)
        await msg.edit(content=_dice_roll_content(roll, stage="settled"))
        await asyncio.sleep(DICE_ROLL_ANIMATION_DELAY_S)
        await msg.edit(content=_dice_roll_content(roll, stage="final"))
    except Exception:
        logger.debug("dice roll message edit failed", exc_info=True)


async def _post_roll_displays_to_pov(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    user_id: int,
    character_id: str,
    char_name: str,
    rolls: list[DiceRollDisplay],
    bot: "discord.Client",
    session_id: str,
    turn_index: Optional[int],
) -> bool:
    if not rolls:
        return True
    channel = _session_text_channel(inter)
    session_chan_id = _session_channel_id(inter)
    user = bot.get_user(user_id)
    if user is None:
        try:
            user = await bot.fetch_user(user_id)
        except Exception:
            logger.exception(
                "dice roll POV: fetch_user(%s) failed", user_id,
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
            for roll in rolls:
                await _send_roll_animation_to_messageable(
                    target=thread,
                    roll=roll,
                    smap=smap,
                    session_channel_id=session_chan_id,
                    session_id=session_id,
                    turn_index=turn_index,
                    delivery="thread",
                    discord_channel_id=thread.id,
                    recipient_user_id=user_id,
                )
            return True
        except Exception:
            logger.exception(
                "dice roll POV: thread.send to %s failed; falling back to DM",
                thread.id,
            )
            await smap.clear_pov_thread(session_chan_id, user_id)

    try:
        for roll in rolls:
            await _send_roll_animation_to_messageable(
                target=user,
                roll=roll,
                smap=smap,
                session_channel_id=session_chan_id,
                session_id=session_id,
                turn_index=turn_index,
                delivery="dm",
                recipient_user_id=user_id,
            )
        return True
    except Exception:
        logger.exception("dice roll POV: DM fallback to %s failed", user_id)
        return False


async def _send_public_roll_displays(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    session_id: str,
    turn_index: Optional[int],
    rolls: list[DiceRollDisplay],
) -> None:
    if not rolls:
        return
    channel = _session_text_channel(inter)
    session_chan_id = _session_channel_id(inter)
    for roll in rolls:
        if channel is not None:
            await _send_roll_animation_to_messageable(
                target=channel,
                roll=roll,
                smap=smap,
                session_channel_id=session_chan_id,
                session_id=session_id,
                turn_index=turn_index,
                delivery="public",
            )
        else:
            msg = await inter.followup.send(
                _dice_roll_content(roll, stage="rolling"),
                wait=True,
            )
            await _record_turn_message(
                smap=smap,
                session_channel_id=session_chan_id,
                session_id=session_id,
                turn_index=turn_index,
                message=msg,
                delivery="public",
            )
            try:
                await asyncio.sleep(DICE_ROLL_ANIMATION_DELAY_S)
                await msg.edit(content=_dice_roll_content(roll, stage="final"))
            except Exception:
                logger.debug("public dice roll edit failed", exc_info=True)


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
    view: Optional[discord.ui.View] = None,
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
            msg = await thread.send(
                content=intro_content,
                embeds=embeds,
                view=view,
            )
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
        msg = await user.send(
            content=intro_content,
            embeds=embeds,
            view=view,
        )
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
    view: Optional[discord.ui.View] = None,
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
            for idx, chunk in enumerate(chunks):
                msg = await thread.send(
                    chunk,
                    view=view if idx == len(chunks) - 1 else None,
                )
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
        for idx, chunk in enumerate(chunks):
            msg = await user.send(
                chunk,
                view=view if idx == len(chunks) - 1 else None,
            )
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


async def _resolve_user_or_fallback(
    *,
    bot: discord.Client,
    user_id: str,
    fallback: discord.abc.User,
) -> discord.abc.User:
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return fallback
    user = bot.get_user(uid)
    if user is not None:
        return user
    try:
        fetched = await bot.fetch_user(uid)
    except Exception:
        logger.exception("resolve_user_or_fallback: fetch_user(%s) failed", uid)
        return fallback
    return fetched or fallback


async def _send_public_turn_render(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    session_id: str,
    turn_index: int,
    content: Optional[str] = None,
    embeds: Optional[list[discord.Embed]] = None,
    view: Optional[discord.ui.View] = None,
) -> None:
    channel = _session_text_channel(inter)
    if channel is not None:
        msg = await channel.send(content=content, embeds=embeds, view=view)
    else:
        msg = await inter.followup.send(
            content=content,
            embeds=embeds,
            view=view,
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


def _asset_key(pack_id: str, asset_id: str) -> str:
    pack = pack_id.strip()
    asset = asset_id.strip()
    return f"{pack}::{asset}" if pack else asset


def _content_asset_catalog_paths(engine: EngineBridge) -> list[Path]:
    configured = getattr(engine, "content_asset_catalog_paths", None)
    if configured:
        if isinstance(configured, str | Path):
            return [Path(configured)]
        return [Path(value) for value in configured]

    env_value = os.getenv("AYOA_CONTENT_ASSET_CATALOGS", "").strip()
    if env_value:
        return [
            Path(value)
            for value in env_value.split(os.pathsep)
            if value.strip()
        ]

    roots: list[Path] = []
    stories_dir = getattr(engine, "stories_dir", None)
    if stories_dir is not None:
        roots.append(Path(stories_dir).parent / "content_assets.sqlite")
    roots.extend(
        [
            Path("app/storage/content_assets.sqlite"),
            Path("private_extractions/content_assets.sqlite"),
            Path("private_extractions/compiled/content_assets.sqlite"),
        ]
    )
    return list(dict.fromkeys(roots))


def _load_content_asset_catalog(
    engine: EngineBridge,
) -> dict[str, ContentImageAsset]:
    assets: dict[str, ContentImageAsset] = {}
    for path in _content_asset_catalog_paths(engine):
        try:
            if not path.exists():
                continue
            assets.update(load_asset_catalog(path))
        except Exception:
            logger.exception("content asset catalog load failed")
    return assets


def _content_asset_roots(
    engine: EngineBridge,
    payload: SafeAssetRevealPayload,
    *,
    attr_name: str,
    env_name: str,
    defaults: Sequence[Path],
) -> dict[str, list[Path]]:
    configured = getattr(engine, attr_name, None)
    pack_id = payload.pack_id.strip()
    roots: list[Path] = []

    if isinstance(configured, Mapping):
        raw = configured.get(pack_id)
        if raw is not None:
            if isinstance(raw, str | Path):
                roots.append(Path(raw))
            else:
                roots.extend(Path(value) for value in raw)
    elif configured:
        raw_values = [configured] if isinstance(configured, str | Path) else configured
        roots.extend(Path(value) for value in raw_values)

    env_value = os.getenv(env_name, "").strip()
    if env_value:
        roots.extend(
            Path(value)
            for value in env_value.split(os.pathsep)
            if value.strip()
        )

    roots.extend(defaults)
    expanded: list[Path] = []
    for root in roots:
        expanded.append(root / pack_id)
        expanded.append(root)
    return {pack_id: list(dict.fromkeys(expanded))}


def _safe_asset_caption(payload: SafeAssetRevealPayload) -> str | None:
    caption = " ".join((payload.caption or "").split())
    return caption or None


def _safe_asset_alt_text(payload: SafeAssetRevealPayload) -> str | None:
    alt_text = " ".join((payload.alt_text or "").split())
    if not alt_text:
        return None
    return alt_text[:1024]


def _resolve_safe_discord_asset(
    payload: SafeAssetRevealPayload,
    *,
    engine: EngineBridge,
    catalog: Mapping[str, ContentImageAsset],
) -> ResolvedAssetBytes:
    asset = catalog.get(_asset_key(payload.pack_id, payload.asset_id))
    if asset is None:
        asset = catalog.get(payload.asset_id.strip())
    if asset is None:
        raise AssetByteResolutionError(
            "missing_asset_catalog_row",
            pack_id=payload.pack_id,
            asset_id=payload.asset_id,
        )

    media_roots = _content_asset_roots(
        engine,
        payload,
        attr_name="content_asset_media_roots",
        env_name="AYOA_CONTENT_ASSET_MEDIA_ROOTS",
        defaults=(
            Path("private_extractions/media"),
            Path("private_extractions/compiled/media"),
            Path("app/storage/content_assets/media"),
        ),
    )
    cache_roots = _content_asset_roots(
        engine,
        payload,
        attr_name="content_asset_cache_roots",
        env_name="AYOA_CONTENT_ASSET_CACHE_ROOTS",
        defaults=(Path("app/storage/content_assets/cache"),),
    )
    return resolve_asset_bytes(
        payload,
        asset,
        media_roots=media_roots,
        cache_roots=cache_roots,
    )


async def _send_private_asset_failure_notice(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    user: discord.abc.User | None,
    recipient_user_id: int,
    session_id: str,
    turn_index: Optional[int],
) -> None:
    notice = (
        "A private image for this turn could not be delivered. "
        "The asset was withheld."
    )
    invoker_notice = (
        "A private image for this turn could not be delivered to one "
        "intended recipient. The asset was withheld."
    )

    if getattr(inter.user, "id", None) == recipient_user_id:
        try:
            await inter.followup.send(
                notice,
                ephemeral=True,
            )
        except Exception:
            logger.debug("asset failure ephemeral notice failed", exc_info=True)
        return

    if user is not None:
        try:
            msg = await user.send(notice)
            await _record_turn_message(
                smap=smap,
                session_channel_id=_session_channel_id(inter),
                session_id=session_id,
                turn_index=turn_index,
                message=msg,
                delivery="dm",
                recipient_user_id=recipient_user_id,
            )
            return
        except Exception:
            logger.debug("asset failure DM notice failed", exc_info=True)

    try:
        await inter.followup.send(
            invoker_notice,
            ephemeral=True,
        )
    except Exception:
        logger.debug("asset failure invoker notice failed", exc_info=True)


async def _post_assets_to_pov(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    user_id: int,
    character_id: str,
    char_name: str,
    asset_reveals: Sequence[SafeAssetRevealPayload],
    bot: discord.Client,
    engine: EngineBridge,
    session_id: str = "",
    turn_index: Optional[int] = None,
    catalog: Mapping[str, ContentImageAsset] | None = None,
) -> bool:
    """Privately send safe asset reveals to one POV, with no public fallback."""
    if not asset_reveals:
        return True

    user = bot.get_user(user_id)
    if user is None:
        try:
            user = await bot.fetch_user(user_id)
        except Exception:
            logger.exception(
                "post_assets_to_pov: fetch_user(%s) failed; withholding assets",
                user_id,
            )
            await _send_private_asset_failure_notice(
                inter=inter,
                smap=smap,
                user=None,
                recipient_user_id=user_id,
                session_id=session_id,
                turn_index=turn_index,
            )
            return False

    asset_catalog = (
        catalog if catalog is not None else _load_content_asset_catalog(engine)
    )
    channel = _session_text_channel(inter)
    session_chan_id = _session_channel_id(inter)
    delivered_all = True
    thread: Optional[discord.Thread] = None
    if channel is not None:
        thread = await _ensure_pov_thread(
            channel=channel,
            user=user,
            smap=smap,
            character_id=character_id,
            char_name=char_name,
        )

    for payload in asset_reveals:
        try:
            resolved = _resolve_safe_discord_asset(
                payload,
                engine=engine,
                catalog=asset_catalog,
            )
        except AssetByteResolutionError:
            logger.warning("post_assets_to_pov: asset resolution failed; withholding")
            delivered_all = False
            await _send_private_asset_failure_notice(
                inter=inter,
                smap=smap,
                user=user,
                recipient_user_id=user_id,
                session_id=session_id,
                turn_index=turn_index,
            )
            continue
        except Exception:
            logger.warning(
                "post_assets_to_pov: unexpected asset resolution failure; "
                "withholding"
            )
            delivered_all = False
            await _send_private_asset_failure_notice(
                inter=inter,
                smap=smap,
                user=user,
                recipient_user_id=user_id,
                session_id=session_id,
                turn_index=turn_index,
            )
            continue

        def _file() -> discord.File:
            return discord.File(
                BytesIO(resolved.data),
                filename=resolved.filename,
                description=_safe_asset_alt_text(payload),
            )

        caption = _safe_asset_caption(payload)
        sent = False
        if thread is not None:
            try:
                msg = await thread.send(content=caption, file=_file())
                await _record_turn_message(
                    smap=smap,
                    session_channel_id=session_chan_id,
                    session_id=session_id,
                    turn_index=turn_index,
                    message=msg,
                    delivery="thread_asset",
                    discord_channel_id=thread.id,
                    recipient_user_id=user_id,
                )
                sent = True
            except Exception:
                logger.warning(
                    "post_assets_to_pov: thread.send to %s failed; "
                    "falling back to DM",
                    thread.id,
                )
                await smap.clear_pov_thread(session_chan_id, user_id)
                thread = None

        if not sent:
            try:
                msg = await user.send(content=caption, file=_file())
                await _record_turn_message(
                    smap=smap,
                    session_channel_id=session_chan_id,
                    session_id=session_id,
                    turn_index=turn_index,
                    message=msg,
                    delivery="dm_asset",
                    recipient_user_id=user_id,
                )
                sent = True
            except Exception:
                logger.warning(
                    "post_assets_to_pov: DM fallback to user %s failed; "
                    "withholding asset",
                    user_id,
                )

        if not sent:
            delivered_all = False
            await _send_private_asset_failure_notice(
                inter=inter,
                smap=smap,
                user=user,
                recipient_user_id=user_id,
                session_id=session_id,
                turn_index=turn_index,
            )

    return delivered_all


class _PendingRollView(discord.ui.View):
    def __init__(
        self,
        *,
        engine: EngineBridge,
        smap: SessionMap,
        prompt: PendingRollPrompt,
        story_id: str,
        char_name: str,
    ):
        super().__init__(timeout=24 * 60 * 60)
        self.engine = engine
        self.smap = smap
        self.prompt = prompt
        self.story_id = story_id
        self.char_name = char_name
        button = discord.ui.Button(
            label=f"Roll {prompt.label}",
            style=discord.ButtonStyle.primary,
            custom_id=(
                f"ayoa:roll:{prompt.session_id}:"
                f"{prompt.event_id}:{prompt.roll_id}"
            )[:100],
        )
        button.callback = self._roll
        self.add_item(button)

    async def _roll(self, roll_inter: discord.Interaction) -> None:
        if str(roll_inter.user.id) != self.prompt.user_id:
            await roll_inter.response.send_message(
                "That roll belongs to another character.",
                ephemeral=True,
            )
            return

        try:
            result = await self.engine.complete_pending_roll(
                session_id=self.prompt.session_id,
                event_id=self.prompt.event_id,
                roll_id=self.prompt.roll_id,
                user_id=roll_inter.user.id,
            )
        except Exception as e:
            logger.exception("pending roll submission failed")
            message = (
                str(e)
                if isinstance(e, ValueError)
                else f"`{type(e).__name__}: {e}`"
            )
            if roll_inter.response.is_done():
                await roll_inter.followup.send(
                    message,
                    ephemeral=True,
                )
            else:
                await roll_inter.response.send_message(
                    message,
                    ephemeral=True,
                )
            return

        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
                child.style = discord.ButtonStyle.success
                child.label = f"Rolled {result.total}"
        try:
            await roll_inter.response.edit_message(
                content=_dice_roll_content(result, stage="rolling"),
                view=self,
            )
            await asyncio.sleep(DICE_ROLL_ANIMATION_DELAY_S)
            await roll_inter.edit_original_response(
                content=_dice_roll_content(result, stage="settled"),
                view=self,
            )
            await asyncio.sleep(DICE_ROLL_ANIMATION_DELAY_S)
            await roll_inter.edit_original_response(
                content=_dice_roll_content(
                    result, stage="final", interpreting=True,
                ),
                view=self,
            )
        except Exception:
            logger.debug("pending roll prompt result edit failed", exc_info=True)
            if not roll_inter.response.is_done():
                try:
                    await roll_inter.response.send_message(
                        "\n".join([
                            _roll_result_line(result),
                            "_Resolving..._",
                        ]),
                        ephemeral=True,
                    )
                except Exception:
                    logger.debug(
                        "pending roll result fallback response failed",
                        exc_info=True,
                    )

        try:
            response = await self.engine.continue_pending_roll(
                session_id=self.prompt.session_id,
                event_id=self.prompt.event_id,
                actor_id=self.prompt.actor_id,
            )
        except Exception as e:
            logger.exception("pending roll continuation failed")
            await roll_inter.followup.send(
                f"`{type(e).__name__}: {e}`",
                ephemeral=True,
            )
            return

        await _deliver_turn_response_to_povs(
            inter=roll_inter,
            smap=self.smap,
            engine=self.engine,
            session_id=self.prompt.session_id,
            story_id=self.story_id,
            actor_character_id=self.prompt.actor_id,
            actor_user=roll_inter.user,
            response=response,
            clear_interaction_response=False,
        )


class _CombatReactionView(discord.ui.View):
    def __init__(
        self,
        *,
        engine: EngineBridge,
        smap: SessionMap,
        session_id: str,
        character_id: str,
        event_id: str,
        user_id: int,
        turn_index: int,
    ):
        super().__init__(timeout=24 * 60 * 60)
        self.engine = engine
        self.smap = smap
        self.session_id = session_id
        self.character_id = character_id
        self.event_id = event_id
        self.user_id = user_id
        self.turn_index = turn_index
        button = discord.ui.Button(
            label="No reaction",
            style=discord.ButtonStyle.secondary,
            custom_id=(
                f"ayoa:reaction:no:{session_id}:{character_id}:{event_id}"
            )[:100],
        )
        button.callback = self._defer
        self.add_item(button)

    async def _defer(self, inter: discord.Interaction) -> None:
        if inter.user.id != self.user_id:
            await inter.response.send_message(
                "That reaction belongs to another character.",
                ephemeral=True,
            )
            return

        try:
            response = await self.engine.defer_combat_reaction(
                session_id=self.session_id,
                character_id=self.character_id,
                event_id=self.event_id,
                user_id=inter.user.id,
            )
        except Exception as e:
            logger.exception("combat reaction defer failed")
            await inter.response.send_message(
                f"`{type(e).__name__}: {e}`",
                ephemeral=True,
            )
            return

        is_stale = response.beat_ended_reason == "combat_reaction_stale"
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
                child.label = (
                    "Reaction closed" if is_stale else "No reaction selected"
                )
        try:
            await inter.response.edit_message(view=self)
        except Exception:
            logger.debug("combat reaction button edit failed", exc_info=True)
            if not inter.response.is_done():
                await inter.response.send_message(
                    response.output_text or "No reaction recorded.",
                    ephemeral=True,
                )
                return
        if response.output_text and "Initiative advances" in response.output_text:
            try:
                await _send_public_turn_render(
                    inter=inter,
                    smap=self.smap,
                    session_id=self.session_id,
                    turn_index=response.turn_index or self.turn_index,
                    content=response.output_text,
                )
                return
            except Exception:
                logger.debug(
                    "combat reaction public handoff failed",
                    exc_info=True,
                )
        if response.output_text:
            await inter.followup.send(response.output_text, ephemeral=True)


class _LootOfferView(discord.ui.View):
    def __init__(
        self,
        *,
        engine: EngineBridge,
        session_id: str,
        user_id: int,
        character_id: str,
        offer: DndLootOffer,
    ):
        super().__init__(timeout=24 * 60 * 60)
        self.engine = engine
        self.session_id = session_id
        self.user_id = user_id
        self.character_id = character_id
        self.offer_id = offer.offer_id
        self.selected_item_ids: list[str] = []

        available_ids = set(dnd_inventory.available_item_ids(offer))
        available = [item for item in offer.items if item.item_id in available_ids]
        if available and len(available) <= 25 and all(
            len(item.item_id) <= 100 for item in available
        ):
            select = discord.ui.Select(
                placeholder="Choose item(s)",
                min_values=0,
                max_values=len(available),
                options=[
                    discord.SelectOption(
                        label=_loot_item_option_label(item),
                        value=item.item_id,
                    )
                    for item in available
                ],
            )
            select.callback = self._select_items
            self.add_item(select)

        take_selected = discord.ui.Button(
            label="Take selected",
            style=discord.ButtonStyle.primary,
            custom_id=f"ayoa:loot:take:{session_id}:{offer.offer_id}"[:100],
        )
        take_selected.callback = self._take_selected
        self.add_item(take_selected)

        take_all = discord.ui.Button(
            label="Take all",
            style=discord.ButtonStyle.secondary,
            custom_id=f"ayoa:loot:all:{session_id}:{offer.offer_id}"[:100],
        )
        take_all.callback = self._take_all
        self.add_item(take_all)

        split = discord.ui.Button(
            label="Split coins",
            style=discord.ButtonStyle.secondary,
            custom_id=f"ayoa:loot:split:{session_id}:{offer.offer_id}"[:100],
            disabled=not offer.has_available_currency(),
        )
        split.callback = self._split
        self.add_item(split)

        decline = discord.ui.Button(
            label="Decline",
            style=discord.ButtonStyle.secondary,
            custom_id=f"ayoa:loot:decline:{session_id}:{offer.offer_id}"[:100],
        )
        decline.callback = self._decline
        self.add_item(decline)

    async def interaction_check(self, inter: discord.Interaction) -> bool:
        if inter.user.id != self.user_id:
            await inter.response.send_message(
                "That loot offer belongs to another character.",
                ephemeral=True,
            )
            return False
        return True

    async def _select_items(self, inter: discord.Interaction) -> None:
        values = []
        data = getattr(inter, "data", {}) or {}
        if isinstance(data, dict):
            values = [str(value) for value in (data.get("values") or [])]
        self.selected_item_ids = values
        await inter.response.defer()

    async def _take_selected(self, inter: discord.Interaction) -> None:
        if not self.selected_item_ids:
            await inter.response.send_message(
                "Select at least one item, or use Take all.",
                ephemeral=True,
            )
            return
        await self._claim(
            inter,
            item_ids=list(self.selected_item_ids),
            take_currency=False,
        )

    async def _take_all(self, inter: discord.Interaction) -> None:
        await self._claim(
            inter,
            item_ids=[],
            take_currency=True,
            take_all_available=True,
        )

    async def _claim(
        self,
        inter: discord.Interaction,
        *,
        item_ids: list[str],
        take_currency: bool,
        take_all_available: bool = False,
    ) -> None:
        try:
            result = await self.engine.claim_loot(
                session_id=self.session_id,
                user_id=self.user_id,
                character_id=self.character_id,
                offer_id=self.offer_id,
                item_ids=item_ids,
                take_currency=take_currency,
                take_all_available=take_all_available,
            )
        except ValueError as e:
            await inter.response.send_message(
                _loot_player_error(e),
                ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("loot claim failed")
            await inter.response.send_message(
                _loot_player_error(e),
                ephemeral=True,
            )
            return
        await self._finish(inter, result)

    async def _split(self, inter: discord.Interaction) -> None:
        try:
            result = await self.engine.split_loot_currency(
                session_id=self.session_id,
                user_id=self.user_id,
                offer_id=self.offer_id,
                character_id=self.character_id,
            )
        except ValueError as e:
            await inter.response.send_message(
                _loot_player_error(e),
                ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("loot split failed")
            await inter.response.send_message(
                _loot_player_error(e),
                ephemeral=True,
            )
            return
        await self._finish(inter, result)

    async def _decline(self, inter: discord.Interaction) -> None:
        try:
            result = await self.engine.decline_loot(
                session_id=self.session_id,
                user_id=self.user_id,
                offer_id=self.offer_id,
                character_id=self.character_id,
            )
        except ValueError as e:
            await inter.response.send_message(
                _loot_player_error(e),
                ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("loot decline failed")
            await inter.response.send_message(
                _loot_player_error(e),
                ephemeral=True,
            )
            return
        await self._finish(inter, result)

    async def _finish(
        self,
        inter: discord.Interaction,
        result: DndLootClaimResult,
    ) -> None:
        if result.offer_closed:
            for child in self.children:
                child.disabled = True
        await inter.response.edit_message(content=result.message, view=self)


async def _post_roll_prompt_to_pov(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    engine: EngineBridge,
    prompt: PendingRollPrompt,
    story_id: str,
    roster: list,
    turn_index: int,
) -> bool:
    try:
        uid = int(prompt.user_id)
    except ValueError:
        return False

    user = inter.client.get_user(uid)
    if user is None:
        try:
            user = await inter.client.fetch_user(uid)
        except Exception:
            logger.exception("roll prompt: fetch_user(%s) failed", uid)
            return False

    char = next((c for c in roster if c.character_id == prompt.actor_id), None)
    char_name = char.name if char else prompt.actor_id
    content = _roll_prompt_content(prompt=prompt, char_name=char_name)
    view = _PendingRollView(
        engine=engine,
        smap=smap,
        prompt=prompt,
        story_id=story_id,
        char_name=char_name,
    )

    channel = _session_text_channel(inter)
    session_chan_id = _session_channel_id(inter)
    thread: Optional[discord.Thread] = None
    if channel is not None:
        thread = await _ensure_pov_thread(
            channel=channel,
            user=user,
            smap=smap,
            character_id=prompt.actor_id,
            char_name=char_name,
        )

    if thread is not None:
        try:
            msg = await thread.send(content, view=view)
            await _record_turn_message(
                smap=smap,
                session_channel_id=session_chan_id,
                session_id=prompt.session_id,
                turn_index=turn_index,
                message=msg,
                delivery="thread",
                discord_channel_id=thread.id,
                recipient_user_id=uid,
            )
            return True
        except Exception:
            logger.exception("roll prompt: thread send failed")
            await smap.clear_pov_thread(session_chan_id, uid)

    try:
        msg = await user.send(content, view=view)
        await _record_turn_message(
            smap=smap,
            session_channel_id=session_chan_id,
            session_id=prompt.session_id,
            turn_index=turn_index,
            message=msg,
            delivery="dm",
            recipient_user_id=uid,
        )
        return True
    except Exception:
        logger.exception("roll prompt: DM fallback failed")

    if uid == inter.user.id:
        await inter.followup.send(content, view=view, ephemeral=True)
        return True
    return False


async def _deliver_pending_roll_prompts(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    engine: EngineBridge,
    session_id: str,
    story_id: str,
    roster: list,
    turn_index: int,
) -> None:
    try:
        prompts = engine.pending_roll_prompts(session_id)
    except Exception:
        logger.exception("pending roll prompt lookup failed")
        return
    for prompt in prompts:
        await _post_roll_prompt_to_pov(
            inter=inter,
            smap=smap,
            engine=engine,
            prompt=prompt,
            story_id=story_id,
            roster=roster,
            turn_index=turn_index,
        )


async def _deliver_loot_prompts(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    engine: EngineBridge,
    session_id: str,
    response: TurnResponse,
) -> None:
    prompts = response.loot_prompts or {}
    if not prompts:
        return
    try:
        ckpt_for_fanout = engine.load_latest(session_id)
        bindings = ckpt_for_fanout.session.character_bindings or {}
        roster = list(ckpt_for_fanout.characters or [])
    except Exception:
        logger.exception("loot prompt fan-out: load_latest failed")
        return

    for cid, offer_ids in prompts.items():
        uid_raw = bindings.get(cid, "")
        if not uid_raw:
            continue
        try:
            uid = int(uid_raw)
        except ValueError:
            continue
        try:
            offers = engine.list_loot_offers(
                session_id,
                uid,
                character_id=cid,
            )
        except Exception:
            logger.exception("loot prompt lookup failed for %s", cid)
            continue
        wanted = set(offer_ids)
        char = next((c for c in roster if c.character_id == cid), None)
        char_name = char.name if char else cid
        for offer in offers:
            if offer.offer_id not in wanted:
                continue
            await _post_to_pov(
                inter=inter,
                smap=smap,
                user_id=uid,
                character_id=cid,
                char_name=char_name,
                text=_loot_offer_content(offer),
                bot=inter.client,
                session_id=session_id,
                turn_index=response.turn_index,
                view=_LootOfferView(
                    engine=engine,
                    session_id=session_id,
                    user_id=uid,
                    character_id=cid,
                    offer=offer,
                ),
            )


async def _deliver_turn_response_to_povs(
    *,
    inter: discord.Interaction,
    smap: SessionMap,
    engine: EngineBridge,
    session_id: str,
    story_id: str,
    actor_character_id: str,
    actor_user: discord.abc.User,
    response: TurnResponse,
    clear_interaction_response: bool = True,
) -> None:
    """Deliver a TurnResponse using the standard /act POV format.

    Actor render goes to the actor's POV thread first, then DM, then public
    fallback. Other human POV renders fan out privately. This is shared by
    `/act` and router-backed private directives such as `/query`.
    """
    if response.beat_ended_reason in {
        "slot_rejected",
        "combat_start_blocked_deferred",
    }:
        await inter.followup.send(
            response.output_text or "Your /act could not be accepted.",
            ephemeral=True,
        )
        return

    # Load bindings + roster ONCE for both fan-outs (pre-turn resolutions +
    # the actor's beat). Failure here only kills DMs, not the actor's render.
    try:
        ckpt_for_fanout = engine.load_latest(session_id)
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
        reaction_prompts: dict[str, str] | None = None,
        commitment_revision_prompts: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """Post each (cid, prose) to that user's POV thread or DM."""
        notified: list[str] = []
        reaction_prompts = reaction_prompts or {}
        commitment_revision_prompts = commitment_revision_prompts or {}
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
            event_id = reaction_prompts.get(cid, "")
            reaction_note = (
                "_You may use `/act` to spend your reaction now, or press "
                "**No reaction** to pass._"
                if event_id else ""
            )
            revision_note = (
                "_Your ongoing activity was interrupted. Use `/act` to revise "
                "it, or `/act (continue)` to keep going if the new situation "
                "still permits it._"
                if commitment_revision_prompts.get(cid) else ""
            )
            prefixes = [
                p for p in (note_prefix, reaction_note, revision_note) if p
            ]
            payload = "\n\n".join([*prefixes, prose]) if prefixes else prose
            char = next((c for c in roster if c.character_id == cid), None)
            char_name = char.name if char else cid
            view = (
                _CombatReactionView(
                    engine=engine,
                    smap=smap,
                    session_id=session_id,
                    character_id=cid,
                    event_id=event_id,
                    user_id=uid,
                    turn_index=turn_index,
                )
                if event_id else None
            )
            ok = await _post_to_pov(
                inter=inter,
                smap=smap,
                user_id=uid,
                character_id=cid,
                char_name=char_name,
                text=payload,
                bot=inter.client,
                session_id=session_id,
                turn_index=turn_index,
                view=view,
            )
            if ok:
                notified.append(char_name)
        return notified

    async def _deliver_rolls_to_povs(
        rolls: list[DiceRollDisplay],
        renders: dict[str, str],
        *,
        skip_cid: str | None,
        turn_index: int,
    ) -> None:
        if not rolls:
            return
        for cid in renders:
            if cid == skip_cid:
                continue
            uid_str = bindings.get(cid, "")
            if not uid_str:
                continue
            try:
                uid = int(uid_str)
            except ValueError:
                continue
            char = next((c for c in roster if c.character_id == cid), None)
            char_name = char.name if char else cid
            await _post_roll_displays_to_pov(
                inter=inter,
                smap=smap,
                user_id=uid,
                character_id=cid,
                char_name=char_name,
                rolls=rolls,
                bot=inter.client,
                session_id=session_id,
                turn_index=turn_index,
            )

    def _asset_reveals_by_pov(
        turn_response: TurnResponse,
        *,
        fallback_character_id: str,
    ) -> dict[str, list[SafeAssetRevealPayload]]:
        per_player_assets = {
            cid: list(payloads or [])
            for cid, payloads in (
                turn_response.per_player_asset_reveals or {}
            ).items()
            if cid and payloads
        }
        if per_player_assets:
            return per_player_assets
        if turn_response.asset_reveals:
            return {fallback_character_id: list(turn_response.asset_reveals)}
        return {}

    async def _deliver_assets_to_povs(
        turn_response: TurnResponse,
        *,
        fallback_character_id: str,
    ) -> None:
        assets_by_pov = _asset_reveals_by_pov(
            turn_response,
            fallback_character_id=fallback_character_id,
        )
        if not assets_by_pov:
            return
        catalog = _load_content_asset_catalog(engine)
        for cid, asset_reveals in assets_by_pov.items():
            if not asset_reveals:
                continue
            if cid == actor_character_id:
                uid = int(actor_user.id)
            else:
                uid_str = bindings.get(cid, "")
                if not uid_str:
                    continue
                try:
                    uid = int(uid_str)
                except ValueError:
                    continue
            char = next((c for c in roster if c.character_id == cid), None)
            char_name = char.name if char else cid
            await _post_assets_to_pov(
                inter=inter,
                smap=smap,
                user_id=uid,
                character_id=cid,
                char_name=char_name,
                asset_reveals=asset_reveals,
                bot=inter.client,
                engine=engine,
                session_id=session_id,
                turn_index=turn_response.turn_index,
                catalog=catalog,
            )

    async def _deliver_experience_awards(
        awards: list[Any],
        *,
        turn_index: int,
    ) -> None:
        if not awards:
            return
        for award in awards:
            cid = str(getattr(award, "character_id", "") or "")
            if not cid:
                continue
            content = _experience_award_content(award)
            if cid == actor_character_id:
                if actor_user.id == inter.user.id:
                    await inter.followup.send(content, ephemeral=True)
                else:
                    char = next(
                        (c for c in roster if c.character_id == cid), None,
                    )
                    char_name = char.name if char else cid
                    await _post_to_pov(
                        inter=inter,
                        smap=smap,
                        user_id=actor_user.id,
                        character_id=cid,
                        char_name=char_name,
                        text=content,
                        bot=inter.client,
                        session_id=session_id,
                        turn_index=turn_index,
                    )
                continue
            uid_str = bindings.get(cid, "")
            if not uid_str:
                continue
            try:
                uid = int(uid_str)
            except ValueError:
                continue
            char = next((c for c in roster if c.character_id == cid), None)
            char_name = char.name if char else cid
            await _post_to_pov(
                inter=inter,
                smap=smap,
                user_id=uid,
                character_id=cid,
                char_name=char_name,
                text=content,
                bot=inter.client,
                session_id=session_id,
                turn_index=turn_index,
            )

    # Fan out pre-turn resolutions before the actor's render so private POV
    # order matches story time. These can come from stale Cat II closure or
    # resumed automated combat after a rewind.
    for pre_resp in (response.pre_turn_resolutions or []):
        await _deliver_rolls_to_povs(
            pre_resp.dice_rolls or [],
            pre_resp.per_player_renders or {},
            skip_cid=None,
            turn_index=pre_resp.turn_index,
        )
        await _deliver_experience_awards(
            pre_resp.experience_awards or [],
            turn_index=pre_resp.turn_index,
        )
        await _dm_per_pov(
            pre_resp.per_player_renders or {},
            skip_cid=None,
            turn_index=pre_resp.turn_index,
            note_prefix="_(Resolved before your action.)_",
            commitment_revision_prompts=(
                pre_resp.commitment_revision_prompts or {}
            ),
        )
        await _deliver_assets_to_povs(
            pre_resp,
            fallback_character_id=actor_character_id,
        )
        await _deliver_loot_prompts(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id=session_id,
            response=pre_resp,
        )

    if response.beat_ended_reason == "pre_turn_resolution":
        await inter.followup.send(
            response.output_text or (
                "The scene changed before your submitted action could be "
                "applied. Submit your next action from the updated state."
            ),
            ephemeral=True,
        )
        return

    actor_char = next(
        (c for c in roster if c.character_id == actor_character_id), None,
    )
    actor_name = actor_char.name if actor_char else actor_character_id
    per_player = response.per_player_renders or {}
    reaction_prompts = response.reaction_prompts or {}
    commitment_revision_prompts = response.commitment_revision_prompts or {}
    actor_rolls_delivered = False
    if response.dice_rolls:
        actor_rolls_delivered = await _post_roll_displays_to_pov(
            inter=inter,
            smap=smap,
            user_id=actor_user.id,
            character_id=actor_character_id,
            char_name=actor_name,
            rolls=response.dice_rolls,
            bot=inter.client,
            session_id=session_id,
            turn_index=response.turn_index,
        )
    await _deliver_experience_awards(
        response.experience_awards or [],
        turn_index=response.turn_index,
    )
    actor_revision_note = (
        "_Your ongoing activity was interrupted. Use `/act` to revise it, "
        "or `/act (continue)` to keep going if the new situation still "
        "permits it._"
        if commitment_revision_prompts.get(actor_character_id) else ""
    )

    combat_start_blocked = response.beat_ended_reason == "combat_start_blocked"
    pending_rolls = response.beat_ended_reason == "cat_ii_pending_rolls"
    reaction_pending = response.beat_ended_reason == "combat_reaction_pending"
    if (
        response.beat_ended_reason == "cat_ii_pending"
        or pending_rolls
        or reaction_pending
        or combat_start_blocked
    ):
        actor_render = per_player.get(actor_character_id) or ""
        if pending_rolls:
            pause_note = (
                "_Scene paused — waiting on D&D roll prompt(s). "
                "The beat will continue after the required roll(s)._"
            )
        elif combat_start_blocked:
            pause_note = (
                "_That hostile action could not start a second D&D combat "
                "while another combat is already in initiative. You can "
                "revise and act normally._"
            )
        elif reaction_pending:
            pause_note = (
                "_Scene paused — waiting on possible reaction(s). "
                "Prompted players can `/act` to react or press "
                "**No reaction**._"
            )
        else:
            pause_note = (
                "_Scene paused — waiting on another player's response. "
                "You'll see the beat continue when they /act._"
            )
        if actor_render:
            embeds = render_turn(
                output_text=actor_render,
                turn_index=response.turn_index,
                story_id=story_id,
            )
            venue, thread = await _post_actor_render(
                inter=inter,
                smap=smap,
                user=actor_user,
                character_id=actor_character_id,
                char_name=actor_name,
                embeds=embeds,
                intro_content="\n\n".join(
                    p for p in (pause_note, actor_revision_note) if p
                ),
                session_id=session_id,
                turn_index=response.turn_index,
                view=(
                    _CombatReactionView(
                        engine=engine,
                        smap=smap,
                        session_id=session_id,
                        character_id=actor_character_id,
                        event_id=reaction_prompts[actor_character_id],
                        user_id=actor_user.id,
                        turn_index=response.turn_index,
                    )
                    if actor_character_id in reaction_prompts else None
                ),
            )
            if venue == "thread" and thread is not None:
                if clear_interaction_response:
                    await _clear_interaction_response(inter)
            elif venue == "dm":
                if clear_interaction_response:
                    await _clear_interaction_response(inter)
            else:
                if response.dice_rolls and not actor_rolls_delivered:
                    await _send_public_roll_displays(
                        inter=inter,
                        smap=smap,
                        session_id=session_id,
                        turn_index=response.turn_index,
                        rolls=response.dice_rolls,
                    )
                await _send_public_turn_render(
                    inter=inter,
                    smap=smap,
                    session_id=session_id,
                    turn_index=response.turn_index,
                    content="\n\n".join(
                        p for p in (pause_note, actor_revision_note) if p
                    ),
                    embeds=embeds,
                    view=(
                        _CombatReactionView(
                            engine=engine,
                            smap=smap,
                            session_id=session_id,
                            character_id=actor_character_id,
                            event_id=reaction_prompts[actor_character_id],
                            user_id=actor_user.id,
                            turn_index=response.turn_index,
                        )
                        if actor_character_id in reaction_prompts else None
                    ),
                )
        else:
            if response.dice_rolls and not actor_rolls_delivered:
                for roll in response.dice_rolls:
                    await inter.followup.send(
                        _dice_roll_content(roll, stage="final"),
                        ephemeral=True,
                    )
            await inter.followup.send(
                content="\n\n".join(
                    p for p in (pause_note, actor_revision_note) if p
                ),
                ephemeral=True,
            )
    else:
        actor_render = (
            response.output_text
            or per_player.get(actor_character_id, "")
            or "(no response)"
        )
        embeds = render_turn(
            output_text=actor_render,
            turn_index=response.turn_index,
            story_id=story_id,
        )
        venue, thread = await _post_actor_render(
            inter=inter,
            smap=smap,
            user=actor_user,
            character_id=actor_character_id,
            char_name=actor_name,
            embeds=embeds,
            intro_content=actor_revision_note or None,
            session_id=session_id,
            turn_index=response.turn_index,
            view=(
                _CombatReactionView(
                    engine=engine,
                    smap=smap,
                    session_id=session_id,
                    character_id=actor_character_id,
                    event_id=reaction_prompts[actor_character_id],
                    user_id=actor_user.id,
                    turn_index=response.turn_index,
                )
                if actor_character_id in reaction_prompts else None
            ),
        )
        if venue == "thread" and thread is not None:
            if clear_interaction_response:
                await _clear_interaction_response(inter)
        elif venue == "dm":
            if clear_interaction_response:
                await _clear_interaction_response(inter)
        else:
            if response.dice_rolls and not actor_rolls_delivered:
                await _send_public_roll_displays(
                    inter=inter,
                    smap=smap,
                    session_id=session_id,
                    turn_index=response.turn_index,
                    rolls=response.dice_rolls,
                )
            await _send_public_turn_render(
                inter=inter,
                smap=smap,
                session_id=session_id,
                turn_index=response.turn_index,
                content=actor_revision_note or None,
                embeds=embeds,
                view=(
                    _CombatReactionView(
                        engine=engine,
                        smap=smap,
                        session_id=session_id,
                        character_id=actor_character_id,
                        event_id=reaction_prompts[actor_character_id],
                        user_id=actor_user.id,
                        turn_index=response.turn_index,
                    )
                    if actor_character_id in reaction_prompts else None
                ),
            )

    if per_player:
        await _deliver_rolls_to_povs(
            response.dice_rolls or [],
            per_player,
            skip_cid=actor_character_id,
            turn_index=response.turn_index,
        )
        notified_names = await _dm_per_pov(
            per_player, skip_cid=actor_character_id,
            turn_index=response.turn_index,
            reaction_prompts=reaction_prompts,
            commitment_revision_prompts=commitment_revision_prompts,
        )
        if notified_names:
            try:
                notified_phrase = ", ".join(f"**{n}**" for n in notified_names)
                await inter.followup.send(
                    f"({notified_phrase} notified via DM.)",
                    ephemeral=True,
                )
            except Exception:
                logger.exception("per-POV fan-out: ephemeral ack failed")

    await _deliver_assets_to_povs(
        response,
        fallback_character_id=actor_character_id,
    )

    if pending_rolls:
        await _deliver_pending_roll_prompts(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id=session_id,
            story_id=story_id,
            roster=roster,
            turn_index=response.turn_index,
        )

    await _deliver_loot_prompts(
        inter=inter,
        smap=smap,
        engine=engine,
        session_id=session_id,
        response=response,
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
            except Exception as e:
                logger.warning(
                    "rewind cleanup: fetch_user(%s) failed (%s)",
                    ref.recipient_user_id, _safe_cleanup_error(e),
                )
                return None
        dm_channel = getattr(user, "dm_channel", None)
        if dm_channel is None:
            try:
                dm_channel = await user.create_dm()
            except Exception as e:
                logger.warning(
                    "rewind cleanup: create_dm(%s) failed (%s)",
                    ref.recipient_user_id, _safe_cleanup_error(e),
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
    except Exception as e:
        logger.warning(
            "rewind cleanup: fetch_channel(%s) failed (%s)",
            ref.discord_channel_id, _safe_cleanup_error(e),
        )
        return None


async def _hide_rewound_turn_message(msg: object) -> None:
    try:
        await msg.edit(
            content="_Rewound turn hidden._",
            embeds=[],
            attachments=[],
        )
    except TypeError as e:
        if "attachments" not in str(e):
            raise
        await msg.edit(content="_Rewound turn hidden._", embeds=[])


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
        except Exception as e:
            cleanup.failed += 1
            logger.warning(
                "rewind cleanup: fetch_message(%s) failed in channel %s (%s)",
                ref.message_id, ref.discord_channel_id,
                _safe_cleanup_error(e),
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
        except Exception as e:
            logger.warning(
                "rewind cleanup: delete failed for message %s in channel %s; "
                "trying edit fallback (%s)",
                ref.message_id, ref.discord_channel_id,
                _safe_cleanup_error(e),
            )

        try:
            await _hide_rewound_turn_message(msg)
            cleanup.hidden += 1
            handled.append(ref)
        except discord.NotFound:
            cleanup.missing += 1
            handled.append(ref)
        except Exception as e:
            cleanup.failed += 1
            logger.warning(
                "rewind cleanup: edit fallback failed for message %s "
                "in channel %s (%s)",
                ref.message_id, ref.discord_channel_id,
                _safe_cleanup_error(e),
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


ABILITY_ORDER = ("str", "dex", "con", "int", "wis", "cha")
ABILITY_LABELS = {
    "str": "STR",
    "dex": "DEX",
    "con": "CON",
    "int": "INT",
    "wis": "WIS",
    "cha": "CHA",
}


def _render_dnd_sheet_page(
    character: CharacterRecord,
    page: str,
) -> discord.Embed:
    """Render a compact Avrae-style sheet page from the imported snapshot.

    This intentionally reads `mechanics.dnd5e_sheet`, not `raw_source`.
    The raw DDB export is stored for future import fidelity and rewind,
    but the user-facing sheet should stay structured and bounded.
    """
    page = (page or "overview").strip().lower()
    page_index = _sheet_page_index(page)
    if page_index is None:
        raise ValueError(
            f"Unknown sheet page '{page}'. Use one of: "
            f"{', '.join(DND_SHEET_PAGES)}."
        )
    sheet = _dnd_sheet_for(character)
    identity = sheet.get("identity") or {}
    statblock = sheet.get("statblock") or {}

    embed = render_info(
        f"Sheet · {character.name}",
        _sheet_identity_line(character, identity),
    )
    embed.set_footer(
        text=(
            f"{character.character_id} · {_sheet_page_label(page)} "
            f"{page_index + 1}/{len(DND_SHEET_PAGES)}"
        )
    )

    if page == "overview":
        _sheet_overview(embed, character, sheet, statblock)
    elif page == "abilities":
        _sheet_abilities(embed, statblock)
    elif page == "actions":
        _sheet_actions(embed, statblock)
    elif page == "spells":
        _sheet_spells(embed, statblock)
    elif page == "inventory":
        _sheet_inventory(embed, dnd_inventory.inventory_view(character))
    elif page == "features":
        _sheet_features(embed, statblock)
    return embed


def _sheet_page_index(page: str) -> int | None:
    normalized = (page or "overview").strip().lower()
    try:
        return DND_SHEET_PAGES.index(normalized)
    except ValueError:
        return None


def _sheet_page_label(page: str) -> str:
    return (page or "overview").replace("_", " ").title()


def _dnd_sheet_for(character: CharacterRecord) -> dict[str, Any]:
    mechanics = character.mechanics or {}
    sheet = mechanics.get("dnd5e_sheet") or {}
    if not isinstance(sheet, dict) or not sheet:
        raise ValueError(
            f"`{character.character_id}` does not have an attached D&D sheet. "
            "Use `/attach` with a D&D Beyond JSON export first."
        )
    return sheet


def _sheet_identity_line(
    character: CharacterRecord,
    identity: dict[str, Any],
) -> str:
    bits: list[str] = []
    class_line = _class_line(identity)
    if class_line:
        bits.append(class_line)
    species = str(identity.get("species") or "").strip()
    background = str(identity.get("background") or "").strip()
    if species:
        bits.append(species)
    if background:
        bits.append(background)
    imported = str(identity.get("name") or "").strip()
    if imported and imported != character.name:
        bits.append(f"Imported name: {imported}")
    return " · ".join(bits) or "D&D character sheet"


def _class_line(identity: dict[str, Any]) -> str:
    labels = []
    for entry in identity.get("classes") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        subclass = str(entry.get("subclass") or "").strip()
        level = _int_or(entry.get("level"), 0)
        label = name
        if subclass and subclass.lower() != name.lower():
            label = f"{subclass} {name}" if name else subclass
        if label:
            labels.append(f"{label} {level}".strip())
    return ", ".join(labels)


def _sheet_overview(
    embed: discord.Embed,
    character: CharacterRecord,
    sheet: dict[str, Any],
    statblock: dict[str, Any],
) -> None:
    defenses = statblock.get("defenses") or {}
    hp = defenses.get("hit_points") or {}
    ac = defenses.get("armor_class") or {}
    initiative = defenses.get("initiative") or {}
    skills = statblock.get("skills") or {}
    perception = skills.get("perception") or {}
    combat = [
        f"AC {_int_or(ac.get('value'), 0)}",
        (
            f"HP {_int_or(hp.get('current'), 0)}/"
            f"{_int_or(hp.get('max'), 0)}"
            + (
                f" (+{_int_or(hp.get('temporary'), 0)} temp)"
                if _int_or(hp.get("temporary"), 0)
                else ""
            )
        ),
        f"PB {_signed(statblock.get('proficiency_bonus') or 0)}",
        f"Initiative {_signed(initiative.get('value') or 0)}",
    ]
    if perception:
        passive_perception = perception.get("passive")
        if passive_perception is None:
            passive_perception = 10 + _int_or(perception.get("value"), 0)
        combat.append(
            f"Passive Perception {_int_or(passive_perception, 10)}"
        )
    movement = _movement_line(defenses.get("movement") or {})
    if movement:
        combat.append(f"Speed {movement}")
    _add_sheet_field(embed, "Combat", " · ".join(combat), inline=False)

    experience = dnd_experience.format_experience_progress(
        dnd_experience.experience_view(character)
    )
    if experience:
        _add_sheet_field(embed, "Progression", experience, inline=False)

    resources = _resource_lines(statblock.get("resources") or [], limit=10)
    if resources:
        _add_sheet_field(embed, "Resources", "\n".join(resources), inline=False)

    defenses_lines = _defense_lines(defenses)
    if defenses_lines:
        _add_sheet_field(
            embed,
            "Defenses",
            "\n".join(defenses_lines),
            inline=False,
        )

    source = sheet.get("source") or {}
    source_bits = [
        str(sheet.get("ruleset_id") or ""),
        str(source.get("type") or ""),
    ]
    source_line = " · ".join(bit for bit in source_bits if bit)
    if source_line:
        _add_sheet_field(embed, "Source", source_line, inline=False)


def _sheet_abilities(embed: discord.Embed, statblock: dict[str, Any]) -> None:
    scores = statblock.get("ability_scores") or {}
    saves = statblock.get("saves") or {}
    lines = []
    for ability in ABILITY_ORDER:
        score = scores.get(ability) or {}
        save = saves.get(ability) or {}
        prof = " prof" if (save.get("proficiency_multiplier") or 0) > 0 else ""
        lines.append(
            f"`{ABILITY_LABELS[ability]}` "
            f"{_int_or(score.get('score'), 10):>2} "
            f"({_signed(score.get('modifier') or 0)}) · "
            f"save {_signed(save.get('value') or score.get('modifier') or 0)}"
            f"{prof}"
        )
    _add_sheet_field(embed, "Abilities", "\n".join(lines), inline=False)

    skill_lines = []
    for name, skill in sorted((statblock.get("skills") or {}).items()):
        if not isinstance(skill, dict):
            continue
        prof = " prof" if (skill.get("proficiency_multiplier") or 0) > 0 else ""
        adv = _advantage_suffix(skill.get("advantage_state"))
        skill_lines.append(
            f"{_title_skill(name)} {_signed(skill.get('value') or 0)}"
            f"{prof}{adv}"
        )
    midpoint = (len(skill_lines) + 1) // 2
    _add_sheet_field(
        embed,
        "Skills",
        "\n".join(skill_lines[:midpoint]) or "No skills imported.",
        inline=True,
    )
    if skill_lines[midpoint:]:
        _add_sheet_field(
            embed,
            "Skills",
            "\n".join(skill_lines[midpoint:]),
            inline=True,
        )


def _sheet_actions(embed: discord.Embed, statblock: dict[str, Any]) -> None:
    actions = [
        action for action in (statblock.get("actions") or [])
        if isinstance(action, dict)
    ]
    if not actions:
        _add_sheet_field(embed, "Actions", "No actions imported.", inline=False)
        return
    lines = [_action_line(action) for action in actions[:18]]
    if len(actions) > 18:
        lines.append(f"... {len(actions) - 18} more")
    _add_sheet_field(embed, "Actions", "\n".join(lines), inline=False)


def _sheet_spells(embed: discord.Embed, statblock: dict[str, Any]) -> None:
    spellcasting = statblock.get("spellcasting") or {}
    profiles = []
    for profile in spellcasting.get("profiles") or []:
        if not isinstance(profile, dict):
            continue
        profiles.append(
            f"{profile.get('name') or 'Spellcasting'} "
            f"({_upper(profile.get('ability'))}) · "
            f"ATK {_signed(profile.get('spell_attack_bonus') or 0)} · "
            f"DC {_int_or(profile.get('spell_save_dc'), 0)}"
        )
    if profiles:
        _add_sheet_field(embed, "Spellcasting", "\n".join(profiles), inline=False)

    slots = _slots_line(spellcasting.get("slots") or {})
    pact = spellcasting.get("pact_slots")
    if isinstance(pact, dict):
        slots = (
            f"{slots}\nPact L{_int_or(pact.get('level'), 1)} "
            f"{_int_or(pact.get('current'), 0)}/{_int_or(pact.get('max'), 0)}"
            if slots else
            f"Pact L{_int_or(pact.get('level'), 1)} "
            f"{_int_or(pact.get('current'), 0)}/{_int_or(pact.get('max'), 0)}"
        )
    if slots:
        _add_sheet_field(embed, "Slots", slots, inline=False)

    spells = [
        spell for spell in (spellcasting.get("spells") or [])
        if isinstance(spell, dict)
    ]
    if not spells:
        _add_sheet_field(embed, "Spells", "No spells imported.", inline=False)
        return
    grouped: dict[int, list[str]] = {}
    for spell in spells:
        grouped.setdefault(_int_or(spell.get("level"), 0), []).append(
            str(spell.get("name") or "Spell")
        )
    lines = []
    for level in sorted(grouped):
        label = "Cantrips" if level == 0 else f"Level {level}"
        names = ", ".join(sorted(grouped[level])[:18])
        extra = len(grouped[level]) - min(len(grouped[level]), 18)
        if extra:
            names += f", ... {extra} more"
        lines.append(f"**{label}:** {names}")
    _add_sheet_field(embed, "Prepared/Known", "\n".join(lines), inline=False)


def _sheet_inventory(embed: discord.Embed, inventory: dict[str, Any]) -> None:
    items = [
        item for item in (inventory.get("items") or [])
        if isinstance(item, dict)
    ]
    equipped = [item for item in items if item.get("equipped")]
    carried = [item for item in items if not item.get("equipped")]
    currency = inventory.get("currency") or {}
    currency_line = _currency_line(currency)
    if currency_line:
        _add_sheet_field(embed, "Currency", currency_line, inline=False)

    if equipped:
        _add_sheet_field(
            embed,
            "Equipped",
            "\n".join(_item_line(item) for item in equipped[:15]),
            inline=False,
        )
    if carried:
        lines = [_item_line(item) for item in carried[:15]]
        if len(carried) > 15:
            lines.append(f"... {len(carried) - 15} more")
        _add_sheet_field(embed, "Carried", "\n".join(lines), inline=False)
    if not items:
        _add_sheet_field(embed, "Inventory", "No items imported.", inline=False)


def _render_inventory_view(view: DndInventoryView) -> discord.Embed:
    embed = render_info(
        f"Inventory · {view.character_name}",
        f"`{view.character_id}`",
    )
    currency_line = _currency_line(view.currency)
    if currency_line:
        _add_sheet_field(embed, "Currency", currency_line, inline=False)
    equipped = [item for item in view.items if item.get("equipped")]
    carried = [item for item in view.items if not item.get("equipped")]
    if equipped:
        _add_sheet_field(
            embed,
            "Equipped",
            "\n".join(_item_line(item) for item in equipped[:15]),
            inline=False,
        )
    if carried:
        lines = [_item_line(item) for item in carried[:20]]
        if len(carried) > 20:
            lines.append(f"... {len(carried) - 20} more")
        _add_sheet_field(embed, "Carried", "\n".join(lines), inline=False)
    if not view.items and not currency_line:
        _add_sheet_field(embed, "Inventory", "No items or coins.", inline=False)
    return embed


def _render_loot_list(offers: list[DndLootOffer]) -> discord.Embed:
    if not offers:
        return render_info("Loot", "No open loot offers for this character.")
    lines = [_loot_offer_summary(offer) for offer in offers[:12]]
    if len(offers) > 12:
        lines.append(f"... {len(offers) - 12} more")
    return render_info("Loot", "\n\n".join(lines))


def _loot_offer_content(offer: DndLootOffer) -> str:
    return "\n".join([
        "--- Loot Offer ---",
        _loot_offer_summary(offer),
        "",
        (
            "Use the buttons below, `/loot take_all`, `/loot split_coins`, "
            "or `/loot decline`."
        ),
        "Decline is final for your character while this offer remains open.",
    ]).strip()


def _loot_offer_summary(offer: DndLootOffer) -> str:
    label = offer.source_label or offer.source_kind
    lines = [f"**{label}** (`{offer.offer_id}`)"]
    available = [
        item for item in offer.items
        if item.item_id in set(dnd_inventory.available_item_ids(offer))
    ]
    for item in available[:10]:
        item_line = _loot_item_line(item)
        lines.append(f"- `{item.item_id}` {item_line}")
    if len(available) > 10:
        lines.append(f"- ... {len(available) - 10} more item(s)")
    currency = _currency_line(dnd_inventory.available_currency_dict(offer))
    if currency:
        lines.append(f"- coins: {currency}")
    if offer.notes:
        lines.append(offer.notes)
    return "\n".join(lines)


def _loot_item_line(item: Any) -> str:
    return dnd_presentation.loot_item_line(item)


def _loot_item_option_label(item: Any) -> str:
    label = _loot_item_line(item)
    return label[:100]


def _loot_player_error(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return str(exc)
    return "Loot action failed. Use `/loot list` to refresh open offers."


def _sheet_features(embed: discord.Embed, statblock: dict[str, Any]) -> None:
    features = [
        feature for feature in (statblock.get("features") or [])
        if isinstance(feature, dict)
    ]
    proficiencies = statblock.get("proficiencies") or {}
    languages = statblock.get("languages") or []

    if features:
        lines = [_feature_line(feature) for feature in features[:20]]
        if len(features) > 20:
            lines.append(f"... {len(features) - 20} more")
        _add_sheet_field(embed, "Features", "\n".join(lines), inline=False)
    else:
        _add_sheet_field(embed, "Features", "No features imported.", inline=False)

    prof_lines = []
    if isinstance(proficiencies, dict):
        for key in ("armor", "weapons", "tools", "other"):
            values = [str(v) for v in proficiencies.get(key) or [] if v]
            if values:
                prof_lines.append(f"**{key.title()}:** {', '.join(values[:12])}")
    if prof_lines:
        _add_sheet_field(embed, "Proficiencies", "\n".join(prof_lines), inline=False)
    if languages:
        _add_sheet_field(
            embed,
            "Languages",
            ", ".join(str(lang) for lang in languages[:20]),
            inline=False,
        )


def _add_sheet_field(
    embed: discord.Embed,
    name: str,
    value: str,
    *,
    inline: bool,
) -> None:
    text = (value or "").strip() or "None."
    if len(text) > 1024:
        text = text[:1021].rstrip() + "..."
    embed.add_field(name=name, value=text, inline=inline)


def _movement_line(movement: dict[str, Any]) -> str:
    bits = []
    for key in ("walk", "fly", "swim", "climb", "burrow"):
        item = movement.get(key)
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value is not None:
            bits.append(f"{key} {value} {item.get('unit') or 'ft'}")
    return ", ".join(bits)


def _defense_lines(defenses: dict[str, Any]) -> list[str]:
    lines = []
    for label, key in (
        ("Resist", "damage_resistances"),
        ("Immune", "damage_immunities"),
        ("Vulnerable", "damage_vulnerabilities"),
        ("Condition immune", "condition_immunities"),
    ):
        names = [
            str(item.get("name") or item.get("id") or "")
            for item in defenses.get(key) or []
            if isinstance(item, dict)
        ]
        if names:
            lines.append(f"**{label}:** {', '.join(names)}")
    conditions = [
        str(item.get("name") or item.get("id") or "")
        for item in defenses.get("conditions") or []
        if isinstance(item, dict)
    ]
    if conditions:
        lines.append(f"**Conditions:** {', '.join(conditions)}")
    exhaustion = _int_or(defenses.get("exhaustion_level"), 0)
    if exhaustion:
        lines.append(f"**Exhaustion:** {exhaustion}")
    return lines


def _resource_lines(resources: list[dict[str, Any]], *, limit: int) -> list[str]:
    lines = []
    for res in resources[:limit]:
        if not isinstance(res, dict):
            continue
        name = str(res.get("name") or res.get("id") or "Resource")
        current = res.get("current")
        maximum = res.get("max")
        if current is not None and maximum is not None:
            value = f"{_int_or(current, 0)}/{_int_or(maximum, 0)}"
        else:
            value = str(current if current is not None else maximum or "")
        reset = ((res.get("reset") or {}).get("type") or "").replace("_", " ")
        suffix = f" ({reset})" if reset else ""
        lines.append(f"- {name}: {value}{suffix}".rstrip())
    if len(resources) > limit:
        lines.append(f"... {len(resources) - limit} more")
    return lines


def _action_line(action: dict[str, Any]) -> str:
    parts = [str(action.get("kind") or "").replace("_", " ")]
    attack = action.get("attack") or {}
    if isinstance(attack, dict) and attack.get("bonus") is not None:
        parts.append(f"hit {_signed(attack.get('bonus'))}")
    save = action.get("save") or {}
    if isinstance(save, dict) and (save.get("dc") or save.get("ability")):
        dc = f" DC {_int_or(save.get('dc'), 0)}" if save.get("dc") else ""
        parts.append(f"{_upper(save.get('ability'))}{dc} save".strip())
    damage = _damage_line(action.get("damage") or [])
    if damage:
        parts.append(damage)
    text = ", ".join(part for part in parts if part)
    return f"- **{action.get('name') or 'Action'}**" + (f" - {text}" if text else "")


def _damage_line(items: list[dict[str, Any]]) -> str:
    bits = []
    for item in items:
        if not isinstance(item, dict):
            continue
        formula = str(item.get("formula") or "").strip()
        damage_type = str(item.get("damage_type") or "").strip()
        if formula:
            bits.append(f"{formula} {damage_type}".strip())
    return ", ".join(bits)


def _slots_line(slots: dict[str, dict[str, Any]]) -> str:
    lines = []
    for level in sorted(slots, key=lambda value: _int_or(value, 0)):
        slot = slots[level]
        if not isinstance(slot, dict):
            continue
        lines.append(
            f"L{level}: {_int_or(slot.get('current'), 0)}/"
            f"{_int_or(slot.get('max'), 0)}"
        )
    return " · ".join(lines)


def _item_line(item: dict[str, Any]) -> str:
    return dnd_presentation.inventory_item_line(
        item,
        bullet=True,
        include_id=False,
        include_attuned=True,
    )


def _feature_line(feature: dict[str, Any]) -> str:
    kind = str(feature.get("kind") or "").replace("_", " ")
    level = _int_or(feature.get("level"), 0)
    suffix_bits = [bit for bit in (kind, f"L{level}" if level else "") if bit]
    suffix = f" ({', '.join(suffix_bits)})" if suffix_bits else ""
    return f"- {feature.get('name') or 'Feature'}{suffix}"


def _currency_line(currency: dict[str, Any]) -> str:
    return dnd_presentation.currency_line(currency, separator=" · ")


def _advantage_suffix(value: Any) -> str:
    text = str(value or "normal")
    if text == "normal":
        return ""
    return f" {text.replace('_', ' ')}"


def _title_skill(name: str) -> str:
    return str(name).replace("-", " ").title()


def _upper(value: Any) -> str:
    text = str(value or "").strip()
    return text.upper() if text else "?"


def _signed(value: Any) -> str:
    number = _int_or(value, 0)
    return f"+{number}" if number >= 0 else str(number)


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dnd_attachment_body(summary: DndSheetAttachmentSummary) -> str:
    classes = ", ".join(summary.classes) or "unknown class"
    name_line = (
        f"Story name changed to **{summary.character_name}**."
        if summary.name_overridden
        else f"Story name preserved as **{summary.character_name}**."
    )
    hp = f"{summary.hit_points_current}/{summary.hit_points_max}"
    if summary.hit_points_temporary:
        hp += f" (+{summary.hit_points_temporary} temp)"
    return "\n".join([
        f"Attached **{summary.imported_name or 'D&D character'}** to "
        f"`{summary.character_id}`.",
        name_line,
        "",
        f"**Classes:** {classes}",
        f"**Level:** {summary.total_level}",
        f"**AC / HP:** {summary.armor_class} / {hp}",
        (
            f"**D&D mode:** `{summary.session_ruleset_id}` · "
            f"player rolls `{summary.player_roll_mode}`"
        ),
        (
            f"**Imported:** {summary.skills_count} skills, "
            f"{summary.actions_count} actions, {summary.spells_count} spells, "
            f"{summary.resources_count} resources"
        ),
        "",
        "Use `/sheet` to view it.",
    ])


def _render_xp_award(results: list[DndExperienceAwardResult]) -> discord.Embed:
    if not results:
        return render_info("D&D XP", "No experience was awarded.")
    lines = []
    for result in results:
        progress = f"{result.before:,} -> {result.after:,} XP"
        if result.level_available and result.eligible_level:
            progress += f"; eligible for level {result.eligible_level}"
        elif result.next_level:
            progress += (
                f"; {result.xp_to_next_level:,} XP to level "
                f"{result.next_level}"
            )
        lines.append(
            f"**{result.character_name}** (`{result.character_id}`): "
            f"+{result.amount:,} XP ({progress})"
        )
    return render_info("D&D XP Awarded", "\n".join(lines))


class _DndSheetView(discord.ui.View):
    """Ephemeral button pager for `/sheet`.

    The view stores only the session/user/character keys and reloads the
    sheet from the checkpoint on each click. That keeps the display tied
    to current mechanical state instead of freezing whatever HP/resources
    were visible when `/sheet` was first submitted.
    """

    def __init__(
        self,
        *,
        engine: EngineBridge,
        session_id: str,
        user_id: int,
        character_id: str,
        page: str = "overview",
    ):
        super().__init__(timeout=10 * 60)
        self.engine = engine
        self.session_id = session_id
        self.user_id = user_id
        self.character_id = character_id
        self.page_index = _sheet_page_index(page) or 0

        self.previous_button = discord.ui.Button(
            label="Previous",
            emoji="◀️",
            style=discord.ButtonStyle.secondary,
        )
        self.page_button = discord.ui.Button(
            label="",
            style=discord.ButtonStyle.secondary,
            disabled=True,
        )
        self.next_button = discord.ui.Button(
            label="Next",
            emoji="▶️",
            style=discord.ButtonStyle.secondary,
        )
        self.previous_button.callback = self._previous
        self.next_button.callback = self._next
        self.add_item(self.previous_button)
        self.add_item(self.page_button)
        self.add_item(self.next_button)
        self._sync_buttons()

    @property
    def page(self) -> str:
        return DND_SHEET_PAGES[self.page_index]

    def _sync_buttons(self) -> None:
        self.page_button.label = (
            f"{_sheet_page_label(self.page)} "
            f"{self.page_index + 1}/{len(DND_SHEET_PAGES)}"
        )

    def _advance(self, delta: int) -> None:
        self.page_index = (
            self.page_index + delta
        ) % len(DND_SHEET_PAGES)
        self._sync_buttons()

    def _render_current(self) -> discord.Embed:
        character = self.engine.get_bound_character_record(
            self.session_id,
            self.user_id,
            character_id=self.character_id,
        )
        return _render_dnd_sheet_page(character, self.page)

    async def interaction_check(
        self,
        check_inter: discord.Interaction,
    ) -> bool:
        if check_inter.user.id != self.user_id:
            await check_inter.response.send_message(
                "This sheet view isn't yours.",
                ephemeral=True,
            )
            return False
        return True

    async def _previous(self, click_inter: discord.Interaction) -> None:
        await self._turn_page(click_inter, -1)

    async def _next(self, click_inter: discord.Interaction) -> None:
        await self._turn_page(click_inter, 1)

    async def _turn_page(
        self,
        click_inter: discord.Interaction,
        delta: int,
    ) -> None:
        self._advance(delta)
        try:
            embed = self._render_current()
        except ValueError as e:
            await click_inter.response.edit_message(
                embed=render_error(str(e)),
                view=None,
            )
            return
        except Exception as e:
            logger.exception("D&D sheet pagination failed")
            await click_inter.response.edit_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                view=None,
            )
            return
        await click_inter.response.edit_message(embed=embed, view=self)


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

    combat_group = app_commands.Group(
        name="combat",
        description="Track D&D combat for this channel's session.",
    )

    loot_group = app_commands.Group(
        name="loot",
        description="Inspect and claim D&D loot offers.",
    )

    xp_group = app_commands.Group(
        name="xp",
        description="Admin D&D experience tools.",
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
                "No stories available. Add a synthetic checkpoint under "
                "`app/storage/stories/<story_id>/ckpt_0000.json` first.",
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
                    "No stories available. Add a synthetic checkpoint under "
                    "`app/storage/stories/<story_id>/ckpt_0000.json` first.",
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
        # Pre-session browsing path: roster from the story seed ckpt_0000.
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
                display_name = (
                    ch.name if ch and (ch.name or "").strip() else cid
                )
                bound_lines.append(f"• **{display_name}**")
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

    async def _send_dnd_attach_hint(
        inter: discord.Interaction,
        *,
        character_id: str,
        char_name: str,
    ) -> None:
        """Private post-join hint for optional D&D sheet attachment.

        Discord select menus and modals cannot accept file uploads, so
        the actual JSON upload lives on `/attach`. Keeping the hint
        after `/join` preserves the existing dropdown flow while giving
        D&D players the next action at the point it becomes meaningful:
        after a story character exists and is bound.
        """
        body = (
            f"Optional: attach a D&D Beyond JSON sheet to **{char_name}** "
            f"(`{character_id}`) with `/attach attachment:<json>`.\n\n"
            "Leave `name_override` blank to keep the current story name. "
            "Fill it only if the character's in-story name should change."
        )
        try:
            await inter.followup.send(
                embed=render_info("D&D sheet", body),
                ephemeral=True,
            )
        except Exception:
            logger.exception("post-join D&D attach hint failed")

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
                player_safe_error_message(e)
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
            await _send_dnd_attach_hint(
                modal_inter,
                character_id=self._character_id,
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
            await _send_dnd_attach_hint(
                modal_inter,
                character_id=new_char.character_id,
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
            candidates = engine.list_joinable_characters(row.session_id)
        except Exception as e:
            logger.exception("/join: list_joinable_characters failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        # SelectMenu cap is 25 total options. Reserve one slot for the
        # custom-create row, leaving 24 for pre-authored playables. If a
        # roster ever exceeds 24 playable slots we'll truncate; paginated
        # pickers are deferred until we actually need them.
        truncated = len(candidates) > 24
        candidates = candidates[:24]

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
            label = _join_select_label(
                c.name,
                c.character_id,
                role=c.role,
                appearance=c.appearance,
            )
            char_lookup[c.character_id] = (c.name or "").strip() or label
            descr_bits = [b for b in (c.role, c.faction) if b]
            description = _discord_select_text(
                " · ".join(descr_bits) or c.appearance,
            )
            if description == label:
                description = _discord_select_text(c.character_id)
            options.append(discord.SelectOption(
                label=label,
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

    # ---- /attach -----------------------------------------------------------

    @tree.command(
        name="attach",
        description="Attach a D&D Beyond JSON character sheet to your current character.",
        guild=guild,
    )
    @app_commands.describe(
        attachment="A D&D Beyond character JSON export.",
        character_id=(
            "Optional character_id. Defaults to your current bound character."
        ),
        name_override=(
            "Optional in-story name override. Leave blank to preserve the "
            "current story name."
        ),
    )
    async def _attach(
        inter: discord.Interaction,
        attachment: discord.Attachment,
        character_id: str = "",
        name_override: str = "",
    ):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start` then `/story start`.",
                ephemeral=True,
            )
            return

        if engine.get_user_binding(row.session_id, inter.user.id) is None:
            await inter.response.send_message(
                "You're not bound to a character in this story. Run `/join` "
                "first.",
                ephemeral=True,
            )
            return

        filename = (attachment.filename or "").lower()
        content_type = (attachment.content_type or "").lower()
        if not filename.endswith(".json") and "json" not in content_type:
            await inter.response.send_message(
                "Attach a `.json` export from D&D Beyond.",
                ephemeral=True,
            )
            return

        if attachment.size > MAX_DND_CHARACTER_BYTES:
            await inter.response.send_message(
                f"Sheet export is too large ({attachment.size} bytes). "
                f"Limit is {MAX_DND_CHARACTER_BYTES} bytes.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True, ephemeral=True)

        try:
            raw = await attachment.read()
        except Exception:
            logger.exception("/attach attachment download failed")
            await inter.followup.send(
                embed=render_error("Could not download the attachment."),
                ephemeral=True,
            )
            return

        try:
            export = json.loads(raw.decode("utf-8-sig"))
        except UnicodeDecodeError:
            await inter.followup.send(
                embed=render_error("Attachment is not valid UTF-8 JSON."),
                ephemeral=True,
            )
            return
        except json.JSONDecodeError as e:
            await inter.followup.send(
                embed=render_error(f"Attachment is not valid JSON: {e}"),
                ephemeral=True,
            )
            return

        if not isinstance(export, dict):
            await inter.followup.send(
                embed=render_error("D&D Beyond export must be a JSON object."),
                ephemeral=True,
            )
            return

        try:
            summary = await engine.attach_dndbeyond_character_export(
                row.session_id,
                inter.user.id,
                export,
                character_id=character_id or None,
                name_override=name_override or None,
            )
        except ValueError as e:
            await inter.followup.send(
                embed=render_error(str(e)),
                ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("/attach failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        await inter.followup.send(
            embed=render_info("D&D sheet attached", _dnd_attachment_body(summary)),
            ephemeral=True,
        )

    # ---- /sheet ------------------------------------------------------------

    @tree.command(
        name="sheet",
        description="Show your attached D&D character sheet.",
        guild=guild,
    )
    @app_commands.describe(
        page="Which sheet page to show.",
        character_id=(
            "Optional character_id. Defaults to your current bound character."
        ),
    )
    @app_commands.choices(
        page=[
            app_commands.Choice(name="Overview", value="overview"),
            app_commands.Choice(name="Abilities", value="abilities"),
            app_commands.Choice(name="Actions", value="actions"),
            app_commands.Choice(name="Spells", value="spells"),
            app_commands.Choice(name="Inventory", value="inventory"),
            app_commands.Choice(name="Features", value="features"),
        ]
    )
    async def _sheet(
        inter: discord.Interaction,
        page: str = "overview",
        character_id: str = "",
    ):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start` then `/story start`.",
                ephemeral=True,
            )
            return

        try:
            character = engine.get_bound_character_record(
                row.session_id,
                inter.user.id,
                character_id=character_id or None,
            )
            embed = _render_dnd_sheet_page(character, page)
            view = _DndSheetView(
                engine=engine,
                session_id=row.session_id,
                user_id=inter.user.id,
                character_id=character.character_id,
                page=page,
            )
        except ValueError as e:
            await inter.response.send_message(
                embed=render_error(str(e)),
                ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("/sheet failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return

        await inter.response.send_message(
            embed=embed,
            view=view,
            ephemeral=True,
        )

    # ---- /inventory --------------------------------------------------------

    @tree.command(
        name="inventory",
        description="Show your current D&D inventory.",
        guild=guild,
    )
    @app_commands.describe(
        character_id=(
            "Optional character_id. Defaults to your current bound character."
        ),
    )
    async def _inventory(
        inter: discord.Interaction,
        character_id: str = "",
    ):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start` then `/story start`.",
                ephemeral=True,
            )
            return
        try:
            view = engine.list_inventory(
                row.session_id,
                inter.user.id,
                character_id=character_id or None,
            )
        except ValueError as e:
            await inter.response.send_message(
                embed=render_error(str(e)),
                ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("/inventory failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        await inter.response.send_message(
            embed=_render_inventory_view(view),
            ephemeral=True,
        )

    # ---- /loot -------------------------------------------------------------

    @loot_group.command(
        name="list",
        description="List open loot offers for your character.",
    )
    @app_commands.describe(character_id="Optional character_id.")
    async def _loot_list(
        inter: discord.Interaction,
        character_id: str = "",
    ):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message("No session here.", ephemeral=True)
            return
        try:
            offers = engine.list_loot_offers(
                row.session_id,
                inter.user.id,
                character_id=character_id or None,
            )
        except ValueError as e:
            await inter.response.send_message(
                embed=render_error(str(e)),
                ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("/loot list failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        await inter.response.send_message(
            embed=_render_loot_list(offers),
            ephemeral=True,
        )

    @loot_group.command(
        name="take",
        description="Take selected item ids from a loot offer.",
    )
    @app_commands.describe(
        offer_id="Loot offer id.",
        item_ids="Comma-separated item ids from /loot list.",
        character_id="Optional character_id.",
    )
    async def _loot_take(
        inter: discord.Interaction,
        offer_id: str,
        item_ids: str,
        character_id: str = "",
    ):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message("No session here.", ephemeral=True)
            return
        selected = [part.strip() for part in item_ids.split(",") if part.strip()]
        try:
            result = await engine.claim_loot(
                session_id=row.session_id,
                user_id=inter.user.id,
                character_id=character_id or None,
                offer_id=offer_id,
                item_ids=selected,
                take_currency=False,
            )
        except ValueError as e:
            await inter.response.send_message(_loot_player_error(e), ephemeral=True)
            return
        except Exception as e:
            logger.exception("/loot take failed")
            await inter.response.send_message(
                _loot_player_error(e),
                ephemeral=True,
            )
            return
        await inter.response.send_message(result.message, ephemeral=True)

    @loot_group.command(
        name="take_all",
        description="Take every remaining item and coin from a loot offer.",
    )
    @app_commands.describe(
        offer_id="Loot offer id.",
        character_id="Optional character_id.",
    )
    async def _loot_take_all(
        inter: discord.Interaction,
        offer_id: str,
        character_id: str = "",
    ):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message("No session here.", ephemeral=True)
            return
        try:
            result = await engine.claim_loot(
                session_id=row.session_id,
                user_id=inter.user.id,
                character_id=character_id or None,
                offer_id=offer_id,
                item_ids=[],
                take_currency=True,
                take_all_available=True,
            )
        except ValueError as e:
            await inter.response.send_message(_loot_player_error(e), ephemeral=True)
            return
        except Exception as e:
            logger.exception("/loot take_all failed")
            await inter.response.send_message(
                _loot_player_error(e),
                ephemeral=True,
            )
            return
        await inter.response.send_message(result.message, ephemeral=True)

    @loot_group.command(
        name="split_coins",
        description="Split remaining coins in a loot offer among eligible players.",
    )
    @app_commands.describe(
        offer_id="Loot offer id.",
        character_id="Optional character_id.",
    )
    async def _loot_split_coins(
        inter: discord.Interaction,
        offer_id: str,
        character_id: str = "",
    ):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message("No session here.", ephemeral=True)
            return
        try:
            result = await engine.split_loot_currency(
                session_id=row.session_id,
                user_id=inter.user.id,
                offer_id=offer_id,
                character_id=character_id or None,
            )
        except ValueError as e:
            await inter.response.send_message(_loot_player_error(e), ephemeral=True)
            return
        except Exception as e:
            logger.exception("/loot split_coins failed")
            await inter.response.send_message(
                _loot_player_error(e),
                ephemeral=True,
            )
            return
        await inter.response.send_message(result.message, ephemeral=True)

    @loot_group.command(
        name="decline",
        description="Decline an open loot offer for your character.",
    )
    @app_commands.describe(
        offer_id="Loot offer id.",
        character_id="Optional character_id.",
    )
    async def _loot_decline(
        inter: discord.Interaction,
        offer_id: str,
        character_id: str = "",
    ):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message("No session here.", ephemeral=True)
            return
        try:
            result = await engine.decline_loot(
                session_id=row.session_id,
                user_id=inter.user.id,
                offer_id=offer_id,
                character_id=character_id or None,
            )
        except ValueError as e:
            await inter.response.send_message(_loot_player_error(e), ephemeral=True)
            return
        except Exception as e:
            logger.exception("/loot decline failed")
            await inter.response.send_message(
                _loot_player_error(e),
                ephemeral=True,
            )
            return
        await inter.response.send_message(result.message, ephemeral=True)

    # ---- /xp ---------------------------------------------------------------

    @xp_group.command(
        name="award",
        description="[Admin] Award D&D experience to one character or all bound players.",
    )
    @app_commands.describe(
        target="Character id, or `all` for all bound player characters.",
        amount="Experience points to add.",
        note="Optional source note for the checkpoint audit log.",
    )
    async def _xp_award(
        inter: discord.Interaction,
        target: str,
        amount: app_commands.Range[int, 1, 1_000_000],
        note: str = "",
    ):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message("No session here.", ephemeral=True)
            return
        if not _is_admin(inter.user.id):
            await inter.response.send_message(
                "Admin-only command.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True)
        try:
            results = await engine.award_dnd_experience_locked(
                row.session_id,
                target.strip(),
                int(amount),
                source=note.strip(),
            )
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            await inter.followup.send(
                embed=render_error(str(e)),
                ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("/xp award failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        await inter.followup.send(embed=_render_xp_award(results))

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
                player_safe_error_message(e, operation="the opening")
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
        # yet), fire the canonical opening. /begin and run_begin_turn are
        # the source of truth, but manual test setup and future frontends
        # can still land here, and we shouldn't silently swallow the opener.
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

        logger.info(
            "Describe+open for %s by %s: %s",
            row.session_id, inter.user.display_name, changed,
        )

        try:
            response = await engine.run_begin_turn(
                session_id=row.session_id,
                triggering_character_id=binding,
            )
        except TransientLLMError as e:
            logger.warning(
                "opening run_begin_turn hit transient LLM error after %d "
                "attempts: %s",
                e.attempts, e.last_error,
            )
            await inter.followup.send(embed=render_error(str(e)))
            return
        except Exception as e:
            logger.exception("opening run_begin_turn failed")
            await inter.followup.send(embed=render_error(
                player_safe_error_message(e)
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
                embed=render_error(player_safe_error_message(e)),
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

        await _deliver_turn_response_to_povs(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id=row.session_id,
            story_id=row.story_id,
            actor_character_id=binding,
            actor_user=inter.user,
            response=response,
        )

    # ---- /retry ------------------------------------------------------------

    @tree.command(
        name="retry",
        description="Retry a failed narrator render without a new action.",
        guild=guild,
    )
    async def _retry(inter: discord.Interaction):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start` then `/story start`.",
                ephemeral=True,
            )
            return

        await inter.response.defer(thinking=True)
        start = time.monotonic()

        try:
            result = await engine.retry_failed_render(session_id=row.session_id)
        except TransientLLMError as e:
            logger.warning(
                "retry_failed_render hit transient LLM error after %d "
                "attempts: %s",
                e.attempts, e.last_error,
            )
            await inter.followup.send(embed=render_error(str(e)), ephemeral=True)
            return
        except Exception as e:
            logger.exception("retry_failed_render failed")
            await inter.followup.send(
                embed=render_error(player_safe_error_message(e)),
                ephemeral=True,
            )
            return

        response = result.response
        if (
            response.beat_ended_reason == "no_pending_render"
            or not result.actor_character_id
        ):
            await inter.followup.send(
                response.output_text
                or "No failed narrator render is pending for this session.",
                ephemeral=True,
            )
            return

        actor_user = await _resolve_user_or_fallback(
            bot=inter.client,
            user_id=result.actor_user_id,
            fallback=inter.user,
        )
        elapsed = time.monotonic() - start
        logger.info(
            "Retried render for turn %d in %s actor=%s by %s in %.1fs",
            response.turn_index,
            row.session_id,
            result.actor_character_id,
            inter.user.display_name,
            elapsed,
        )
        await _deliver_turn_response_to_povs(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id=row.session_id,
            story_id=row.story_id,
            actor_character_id=result.actor_character_id,
            actor_user=actor_user,
            response=response,
        )

    # ---- /roll --------------------------------------------------------------

    @tree.command(
        name="roll",
        description="Roll a pending D&D check for the current contested action.",
        guild=guild,
    )
    @app_commands.describe(
        roll_id="Optional roll id if your character has multiple pending rolls.",
    )
    async def _roll(inter: discord.Interaction, roll_id: str = ""):
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

        prompts = engine.pending_roll_prompts(
            row.session_id, user_id=inter.user.id
        )
        if not prompts:
            await inter.response.send_message(
                "You do not have a pending D&D roll right now.",
                ephemeral=True,
            )
            return

        chosen: PendingRollPrompt | None = None
        requested = roll_id.strip()
        if requested:
            chosen = next((p for p in prompts if p.roll_id == requested), None)
            if chosen is None:
                await inter.response.send_message(
                    "That roll id is not pending for your character.",
                    ephemeral=True,
                )
                return
        elif len(prompts) == 1:
            chosen = prompts[0]
        else:
            lines = [
                "Multiple rolls are pending. Run `/roll roll_id:<id>` "
                "with one of these:"
            ]
            for prompt in prompts:
                lines.append(f"- `{prompt.roll_id}`: {prompt.label}")
            await inter.response.send_message("\n".join(lines), ephemeral=True)
            return

        try:
            result = await engine.complete_pending_roll(
                session_id=row.session_id,
                event_id=chosen.event_id,
                roll_id=chosen.roll_id,
                user_id=inter.user.id,
            )
        except Exception as e:
            logger.exception("/roll complete_pending_roll failed")
            message = (
                str(e)
                if isinstance(e, ValueError)
                else f"`{type(e).__name__}: {e}`"
            )
            await inter.response.send_message(
                message,
                ephemeral=True,
            )
            return

        await _animate_interaction_roll_result(
            inter,
            result,
            initial_response_sent=False,
            interpreting=True,
        )

        try:
            response = await engine.continue_pending_roll(
                session_id=row.session_id,
                event_id=chosen.event_id,
                actor_id=chosen.actor_id,
            )
        except Exception as e:
            logger.exception("/roll continue_pending_roll failed")
            await inter.followup.send(
                f"`{type(e).__name__}: {e}`",
                ephemeral=True,
            )
            return

        await _deliver_turn_response_to_povs(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id=row.session_id,
            story_id=row.story_id,
            actor_character_id=binding,
            actor_user=inter.user,
            response=response,
            clear_interaction_response=False,
        )

    # ---- /defer -------------------------------------------------------------

    @tree.command(
        name="defer",
        description="Take no action and let the scene continue.",
        guild=guild,
    )
    async def _defer(inter: discord.Interaction):
        row = await smap.get(_session_channel_id(inter))
        if row is not None:
            binding = engine.get_user_binding(row.session_id, inter.user.id)
            if binding is not None:
                try:
                    event_id = engine.combat_reaction_prompt_event(
                        row.session_id, binding,
                    )
                except Exception:
                    logger.exception("/defer reaction lookup failed")
                    event_id = ""
                if event_id:
                    await inter.response.defer(thinking=True, ephemeral=True)
                    try:
                        response = await engine.defer_combat_reaction(
                            session_id=row.session_id,
                            character_id=binding,
                            event_id=event_id,
                            user_id=inter.user.id,
                        )
                    except Exception as e:
                        logger.exception("/defer combat reaction failed")
                        await inter.followup.send(
                            embed=render_error(f"`{type(e).__name__}: {e}`"),
                            ephemeral=True,
                        )
                        return
                    if (
                        response.output_text
                        and "Initiative advances" in response.output_text
                    ):
                        try:
                            await _send_public_turn_render(
                                inter=inter,
                                smap=smap,
                                session_id=row.session_id,
                                turn_index=response.turn_index,
                                content=response.output_text,
                            )
                            return
                        except Exception:
                            logger.debug(
                                "/defer public combat handoff failed",
                                exc_info=True,
                            )
                    await inter.followup.send(
                        response.output_text or "No reaction recorded.",
                        ephemeral=True,
                    )
                    return
        # Reuse /act for ordinary null turns so they get identical binding,
        # locking, render, and fan-out behavior.
        await _act.callback(inter, "(defer)")

    # ---- /query -------------------------------------------------------------
    # Out-of-character consultation. `/query` now enters the router as
    # a private OOC clarification so the answer is canonically grounded
    # as an observable fact for the asking POV.

    @tree.command(
        name="query",
        description=(
            "Ask an out-of-character question (what do I see, who is X, "
            "what day is it)."
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

        await inter.response.defer(thinking=True)

        try:
            response = await engine.run_query(
                session_id=row.session_id,
                character_id=binding,
                question=question.strip(),
            )
        except TransientLLMError as e:
            logger.warning(
                "/query run_query hit transient LLM error after %d attempts: %s",
                e.attempts, e.last_error,
            )
            await inter.followup.send(str(e), ephemeral=True)
            return
        except Exception as e:
            logger.exception("/query run_query failed")
            await inter.followup.send(
                player_safe_error_message(e, operation="that query"),
                ephemeral=True,
            )
            return

        await _deliver_turn_response_to_povs(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id=row.session_id,
            story_id=row.story_id,
            actor_character_id=binding,
            actor_user=inter.user,
            response=response,
        )

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
            preview: RewindResult,
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

    # ---- /combat ------------------------------------------------------------

    async def _combat_row(inter: discord.Interaction):
        row = await smap.get(_session_channel_id(inter))
        if row is None:
            await inter.response.send_message(
                "No session here. Run `/session start` then `/story start`.",
                ephemeral=True,
            )
            return None
        return row

    @combat_group.command(
        name="begin",
        description="Begin D&D combat with bound characters and nearby NPCs.",
    )
    @app_commands.describe(
        participants=(
            "Optional comma-separated character IDs. Defaults to party plus "
            "same-location NPCs."
        ),
    )
    async def _combat_begin(
        inter: discord.Interaction,
        participants: str = "",
    ):
        row = await _combat_row(inter)
        if row is None:
            return
        participant_ids = [
            part.strip() for part in participants.split(",") if part.strip()
        ] or None
        await inter.response.defer(thinking=True)
        try:
            view = await engine.begin_combat_locked(
                row.session_id, participant_ids
            )
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            await inter.followup.send(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("combat begin failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        await inter.followup.send(embed=_render_combat_status(view))

    @combat_group.command(
        name="status",
        description="Show the current D&D combat order and HP.",
    )
    async def _combat_status(inter: discord.Interaction):
        row = await _combat_row(inter)
        if row is None:
            return
        try:
            view = engine.combat_status(row.session_id, private=True)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            await inter.response.send_message(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("combat status failed")
            await inter.response.send_message(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        await inter.response.send_message(
            embed=_render_combat_status(view, include_map=True),
            ephemeral=True,
        )

    @combat_group.command(
        name="next",
        description="Advance D&D combat to the next turn.",
    )
    async def _combat_next(inter: discord.Interaction):
        row = await _combat_row(inter)
        if row is None:
            return
        await inter.response.defer(thinking=True)
        try:
            view = await engine.combat_next_locked(row.session_id)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            await inter.followup.send(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("combat next failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        await inter.followup.send(embed=_render_combat_status(view))

    @combat_group.command(
        name="end",
        description="End the active D&D combat.",
    )
    async def _combat_end(inter: discord.Interaction):
        row = await _combat_row(inter)
        if row is None:
            return
        await inter.response.defer(thinking=True)
        try:
            view = await engine.combat_end_locked(row.session_id)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            await inter.followup.send(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("combat end failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        await inter.followup.send(embed=_render_combat_status(view))

    @combat_group.command(
        name="damage",
        description="Apply raw override damage to a D&D combat participant.",
    )
    @app_commands.describe(
        target="Character ID to damage.",
        amount="Damage amount.",
    )
    async def _combat_damage(
        inter: discord.Interaction,
        target: str,
        amount: app_commands.Range[int, 1, 999],
    ):
        row = await _combat_row(inter)
        if row is None:
            return
        await inter.response.defer(thinking=True)
        try:
            view = await engine.combat_damage_locked(
                row.session_id, target.strip(), int(amount)
            )
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            await inter.followup.send(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("combat damage failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        await inter.followup.send(embed=_render_combat_status(view))

    @combat_group.command(
        name="heal",
        description="Apply healing to a D&D combat participant.",
    )
    @app_commands.describe(
        target="Character ID to heal.",
        amount="Healing amount.",
    )
    async def _combat_heal(
        inter: discord.Interaction,
        target: str,
        amount: app_commands.Range[int, 1, 999],
    ):
        row = await _combat_row(inter)
        if row is None:
            return
        await inter.response.defer(thinking=True)
        try:
            view = await engine.combat_heal_locked(
                row.session_id, target.strip(), int(amount)
            )
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            await inter.followup.send(
                embed=render_error(str(e)), ephemeral=True,
            )
            return
        except Exception as e:
            logger.exception("combat heal failed")
            await inter.followup.send(
                embed=render_error(f"`{type(e).__name__}: {e}`"),
                ephemeral=True,
            )
            return
        await inter.followup.send(embed=_render_combat_status(view))

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
        tree.add_command(combat_group, guild=guild)
        tree.add_command(loot_group, guild=guild)
        tree.add_command(xp_group, guild=guild)
        tree.add_command(settings_group, guild=guild)
    else:
        tree.add_command(session_group)
        tree.add_command(story_group)
        tree.add_command(combat_group)
        tree.add_command(loot_group)
        tree.add_command(xp_group)
        tree.add_command(settings_group)
