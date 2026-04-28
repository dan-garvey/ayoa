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
_FIELD_VALUE_MAX = 1024  # Discord's per-field value limit
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
    """Player-facing briefing on /story start: short truck-kun primer +
    a single nudge to /join.

    The primer (1–2 paragraphs, second-person, spoiler-free) is generated
    by the importer's Call 6 and stored on `CheckpointFile.player_primer`.
    Pre-v8 checkpoints have an empty primer; we fall back to a stub
    composed from the public setting fields so older saves still render
    something usable.

    No facts list, no genre/era/tone breakdown, no roster preview — those
    used to leak proper nouns the player hadn't earned and turned the
    briefing into a wall of dossier text. The /join command surfaces
    playable characters interactively now, so the briefing can stay
    minimal.
    """
    setting = ckpt.world_state.setting

    story_title = setting.genre.split("—")[0].strip() if setting.genre else story_id
    title = f"Welcome · {story_title}"
    importer_tag = ckpt.importer_version or "v0"
    coverage = ckpt.import_analysis.coverage_rating if ckpt.import_analysis else ""
    coverage_tag = f" · coverage {coverage}" if coverage and coverage != "unknown" else ""
    footer = f"{story_id} · importer {importer_tag}{coverage_tag} · /join to claim a character"

    primer = (ckpt.player_primer or "").strip()
    if not primer:
        # Pre-v8 checkpoint, no primer baked in. Compose a minimal stub
        # from setting fields. Never reach into facts/lore — those are
        # the spoilers the v8 primer was introduced to filter out.
        bits: list[str] = []
        if setting.tone:
            bits.append(f"_{setting.tone}_")
        if setting.premise:
            bits.append(setting.premise)
        primer = "\n\n".join(bits) or (
            "You wake up somewhere unfamiliar. You don't remember how you "
            "got here, and no one's bothered to fill you in yet."
        )

    if len(primer) > MAX_DESCRIPTION:
        primer = primer[: MAX_DESCRIPTION - 1].rstrip() + "…"

    embed = discord.Embed(
        title=title,
        description=primer,
        color=EMBED_COLOR_STORY,
    )
    embed.add_field(
        name="What now?",
        value=(
            "Run `/join` to step into the story — claim an existing "
            "character or create your own. Then `/act <what you do>` "
            "plays a turn; `/defer` lets the story continue without "
            "your character acting."
        ),
        inline=False,
    )
    embed.set_footer(text=footer)
    return embed


def render_error(message: str) -> discord.Embed:
    """Error embed for followup responses."""
    return discord.Embed(
        title="Turn failed",
        description=message[:MAX_DESCRIPTION],
        color=EMBED_COLOR_ERROR,
    )
