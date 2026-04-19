"""Render engine outputs as Discord embeds.

Discord limits an embed description to 4096 chars, and 10 embeds per message
with a total of 6000 chars across them. Narrator output is usually 1-3K chars
so one embed is enough; we split on paragraph boundaries when it overflows.
"""

from __future__ import annotations

import discord

from app.schemas.checkpoint import CheckpointFile

MAX_DESCRIPTION = 4096
MAX_TOTAL = 6000
EMBED_COLOR_STORY = 0x5865F2  # Discord blurple
EMBED_COLOR_ERROR = 0xED4245  # Discord red
EMBED_COLOR_INFO = 0x57F287   # Discord green


def _split_at_paragraph(text: str, limit: int) -> tuple[str, str]:
    """Split `text` into (first_chunk, rest) so that first_chunk <= limit and
    the cut lands on a paragraph boundary when possible."""
    if len(text) <= limit:
        return text, ""
    # Prefer the last double-newline within the first `limit` chars.
    window = text[:limit]
    cut = window.rfind("\n\n")
    if cut == -1 or cut < limit // 2:
        # Fall back to a single newline, then to a hard cut.
        cut = window.rfind("\n")
        if cut == -1:
            cut = limit
    return text[:cut].rstrip(), text[cut:].lstrip()


def render_turn(
    *,
    output_text: str,
    turn_index: int,
    story_id: str,
) -> list[discord.Embed]:
    """Render a narrator turn output to one or two embeds."""
    embeds: list[discord.Embed] = []

    first, rest = _split_at_paragraph(output_text, MAX_DESCRIPTION)
    primary = discord.Embed(description=first, color=EMBED_COLOR_STORY)
    primary.set_footer(text=f"Turn {turn_index} · {story_id}")
    embeds.append(primary)

    if rest:
        # Keep the second embed's description under the remaining budget.
        remaining_budget = MAX_TOTAL - len(first) - len(primary.footer.text or "")
        if remaining_budget <= 0:
            remaining_budget = MAX_DESCRIPTION // 2  # sanity fallback
        second_text, leftover = _split_at_paragraph(rest, min(MAX_DESCRIPTION, remaining_budget))
        if leftover:
            second_text = second_text.rstrip() + "\n\n…"
        embeds.append(discord.Embed(description=second_text, color=EMBED_COLOR_STORY))

    return embeds


def render_info(title: str, body: str) -> discord.Embed:
    """Generic informational embed (status, story listings, etc.)."""
    return discord.Embed(
        title=title,
        description=body[:MAX_DESCRIPTION],
        color=EMBED_COLOR_INFO,
    )


def render_briefing(ckpt: CheckpointFile, story_id: str) -> discord.Embed:
    """Factual briefing shown on /story start — world + role + stakes.

    Pulled straight from the checkpoint: setting, premise, player character
    identity and backstory, plus a handful of common-knowledge facts. Not
    prose — structured context so the narrator's first-turn rendered opening
    can land with the player already oriented.
    """
    setting = ckpt.world_state.setting
    player_char = _find_player_character(ckpt)

    parts: list[str] = []

    # World framing
    setting_bits = []
    if setting.genre:
        setting_bits.append(f"**Genre** · {setting.genre}")
    if setting.era:
        setting_bits.append(f"**Era** · {setting.era}")
    if setting.tone:
        setting_bits.append(f"**Tone** · {setting.tone}")
    if setting_bits:
        parts.append("\n".join(setting_bits))

    if setting.premise:
        parts.append(f"**Premise**\n{setting.premise}")

    # Player role
    if player_char is not None:
        role = player_char.public_sheet.role or "(role unspecified)"
        backstory = (player_char.backstory or "").strip()
        role_block = f"**You are {player_char.name}** — {role}"
        if backstory:
            # Cap backstory to keep the briefing readable
            if len(backstory) > 900:
                backstory = backstory[:900].rstrip() + "…"
            role_block += f"\n\n{backstory}"
        parts.append(role_block)

    # A handful of grounding facts (common knowledge the PC would have)
    facts = ckpt.world_state.facts[:6]
    if facts:
        facts_block = "**What you know**\n" + "\n".join(f"• {f}" for f in facts)
        parts.append(facts_block)

    description = "\n\n".join(parts)
    # Keep under Discord's embed description limit.
    if len(description) > MAX_DESCRIPTION:
        description = description[: MAX_DESCRIPTION - 1] + "…"

    story_title = setting.genre.split("—")[0].strip() if setting.genre else story_id
    embed = discord.Embed(
        title=f"Briefing · {story_title}",
        description=description,
        color=EMBED_COLOR_STORY,
    )
    embed.set_footer(text=f"{story_id} · `/act <your first move>` to begin")
    return embed


def _find_player_character(ckpt: CheckpointFile):
    """Locate the player's CharacterRecord, if present."""
    pcid = ckpt.session.player_character_id
    if not pcid:
        return None
    for c in ckpt.characters:
        if c.character_id == pcid:
            return c
    return None


def render_error(message: str) -> discord.Embed:
    """Error embed for followup responses."""
    return discord.Embed(
        title="Turn failed",
        description=message[:MAX_DESCRIPTION],
        color=EMBED_COLOR_ERROR,
    )
