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
    """Factual briefing shown on /story start — world + role + stakes.

    Pulled straight from the checkpoint: setting, premise, player character
    identity and backstory, plus a handful of common-knowledge facts. Not
    prose — structured context so the narrator's first-turn rendered opening
    can land with the player already oriented.
    """
    setting = ckpt.world_state.setting
    player_char = _find_player_character(ckpt)

    story_title = setting.genre.split("—")[0].strip() if setting.genre else story_id
    title = f"Briefing · {story_title}"
    footer = f"{story_id} · /describe <traits> to open the scene"

    # Build the fields first so we can reserve the description budget based on
    # their actual size. The "How to play" fields MUST always render — that's
    # the whole reason for this briefing — so they get priority over the
    # premise when the budget gets tight.
    fields: list[tuple[str, str]] = []  # (name, value) pairs, in render order

    if player_char is not None:
        role = player_char.public_sheet.role or "(role unspecified)"
        backstory = (player_char.backstory or "").strip()
        role_value = f"**{player_char.name}** — {role}"
        if backstory:
            remaining = _FIELD_VALUE_MAX - len(role_value) - 4
            if len(backstory) > remaining:
                backstory = backstory[:remaining].rstrip() + "…"
            role_value += f"\n\n{backstory}"
        fields.append(("Your role", role_value))

    facts = ckpt.world_state.facts[:6]
    if facts:
        facts_value = "\n".join(f"• {f}" for f in facts)
        if len(facts_value) > _FIELD_VALUE_MAX:
            facts_value = facts_value[: _FIELD_VALUE_MAX - 1] + "…"
        fields.append(("What you know", facts_value))

    fields.append((
        "1. Describe your character",
        "Run `/describe <traits>` to set the physical presence the world will "
        "react to. Include whatever detail you want seen, heard, or remembered "
        "— height, build, age, clothing, bearing, voice, distinctive features. "
        "Be as specific as you like; the narrator uses this verbatim.\n"
        "_Examples_\n"
        "> `/describe tall, early forties, grey at the temples, travel-worn "
        "wool coat, ink-stained fingers, a quiet voice that rarely rises`\n"
        "> `/describe short and wiry, close-cropped dark hair, formal black "
        "robe with silver clasps, walks with a slight limp`"
    ))
    fields.append((
        "2. Your first /describe opens the scene",
        "The narrator renders your arrival in-fiction, incorporating "
        "everything you wrote. You'll see the opening as a reply."
    ))
    fields.append((
        "3. From then on, /act <what you do>",
        "One `/act` per turn. Speak, move, observe, improvise — first person "
        "is natural. Turns take ~20–40 seconds; the bot shows a thinking "
        "indicator while composing. `/status` shows the current scene and "
        "roster; `/story end` closes the session."
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


def _find_player_character(ckpt: CheckpointFile):
    """Locate the player's CharacterRecord.

    Prefers the character whose id matches `session.player_character_id`
    (post-personalize), falling back to the first `is_player=True` character
    (for pristine / pre-personalize checkpoints).
    """
    pcid = ckpt.session.player_character_id
    if pcid:
        for c in ckpt.characters:
            if c.character_id == pcid:
                return c
    for c in ckpt.characters:
        if c.is_player:
            return c
    return None


def render_error(message: str) -> discord.Embed:
    """Error embed for followup responses."""
    return discord.Embed(
        title="Turn failed",
        description=message[:MAX_DESCRIPTION],
        color=EMBED_COLOR_ERROR,
    )
