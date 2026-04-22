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
    """Factual briefing shown on /story start — world + stakes + next steps.

    Pulled straight from the checkpoint: setting, premise, and a handful of
    common-knowledge facts. No player character is assigned at /story start —
    the user picks one with /join or /join_custom after this briefing.
    """
    setting = ckpt.world_state.setting

    story_title = setting.genre.split("—")[0].strip() if setting.genre else story_id
    title = f"Briefing · {story_title}"
    importer_tag = ckpt.importer_version or "v0"
    coverage = ckpt.import_analysis.coverage_rating if ckpt.import_analysis else ""
    coverage_tag = f" · coverage {coverage}" if coverage and coverage != "unknown" else ""
    footer = f"{story_id} · importer {importer_tag}{coverage_tag} · /join or /join_custom to claim a character"

    # Build the fields first so we can reserve the description budget based on
    # their actual size. The "How to play" fields MUST always render — that's
    # the whole reason for this briefing — so they get priority over the
    # premise when the budget gets tight.
    fields: list[tuple[str, str]] = []  # (name, value) pairs, in render order

    facts = ckpt.world_state.facts[:6]
    if facts:
        facts_value = "\n".join(f"• {f}" for f in facts)
        if len(facts_value) > _FIELD_VALUE_MAX:
            facts_value = facts_value[: _FIELD_VALUE_MAX - 1] + "…"
        fields.append(("What's known", facts_value))

    fields.append((
        "1. Pick a character",
        "Run `/story characters` to see the roster, then `/join <character_id>` "
        "to play as an existing one. Prefer a custom concept? "
        "`/join_custom mode:describe description:<your concept>` spawns a new "
        "character, or `mode:replace` grafts your concept onto an existing NPC."
    ))
    fields.append((
        "2. Describe yourself",
        "Once claimed, `/describe` prompts you for a name and appearance — the "
        "physical presence the world will react to. Include whatever detail "
        "you want seen, heard, or remembered."
    ))
    fields.append((
        "3. Open the scene, then /act",
        "Your first `/describe` opens the scene in-fiction. From then on, one "
        "`/act <what you do>` per turn — speak, move, observe, improvise. "
        "`/status` shows the current scene; `/session end` detaches the save."
    ))

    fields_total = sum(len(n) + len(v) for n, v in fields)

    # Build description (genre/era/tone + premise) within the remaining budget.
    # MAX_TOTAL - title - footer - fields - 50 safety → description cap.
    description_budget = min(
        MAX_DESCRIPTION,
        MAX_TOTAL - len(title) - len(footer) - fields_total - 50,
    )

    description_parts: list[str] = []
    setting_bits = []
    if setting.genre:
        setting_bits.append(f"**Genre** · {setting.genre}")
    if setting.era:
        setting_bits.append(f"**Era** · {setting.era}")
    if setting.tone:
        setting_bits.append(f"**Tone** · {setting.tone}")
    if setting_bits:
        description_parts.append("\n".join(setting_bits))
    if setting.premise:
        description_parts.append(f"**Premise**\n{setting.premise}")

    description = "\n\n".join(description_parts)
    if len(description) > description_budget:
        description = description[: description_budget - 1].rstrip() + "…"

    embed = discord.Embed(
        title=title,
        description=description,
        color=EMBED_COLOR_STORY,
    )
    for name, value in fields:
        embed.add_field(name=name, value=value, inline=False)
    embed.set_footer(text=footer)
    return embed


def render_error(message: str) -> discord.Embed:
    """Error embed for followup responses."""
    return discord.Embed(
        title="Turn failed",
        description=message[:MAX_DESCRIPTION],
        color=EMBED_COLOR_ERROR,
    )
