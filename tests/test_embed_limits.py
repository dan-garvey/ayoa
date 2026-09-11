from __future__ import annotations

import pytest

from app.bot.embed import MAX_DESCRIPTION, MAX_FOOTER, MAX_TOTAL, render_turn


@pytest.mark.parametrize("story_id", ["one_star_ascension_s1", "s" * 1800, "s" * 8000])
@pytest.mark.parametrize(
    "size", [0, 1, 3952, 4095, 4096, 4097, 5969, 5970, 5971, 6000, 8048, 16000]
)
@pytest.mark.parametrize("unit", ["x", "word\n\n", "𝄞🙂"])
def test_turn_embeds_fit_individual_and_aggregate_limits(story_id, size, unit):
    text = (unit * (size // len(unit) + 1))[:size]
    embeds = render_turn(output_text=text, turn_index=4, story_id=story_id)
    assert 1 <= len(embeds) <= 2
    assert sum(map(len, embeds)) <= MAX_TOTAL
    assert all(len(embed.description or "") <= MAX_DESCRIPTION for embed in embeds)
    assert all(len(embed.footer.text or "") <= MAX_FOOTER for embed in embeds)


def test_discord_panes_exact_8048_shape_reserves_truncation_marker():
    embeds = render_turn(
        output_text="x" * 8048, turn_index=3, story_id="one_star_ascension_s1"
    )
    assert [len(embed) for embed in embeds] == [4126, 1874]
    assert embeds[1].description.endswith("\n\n…")


def test_complete_second_chunk_at_exact_limit_is_not_truncated():
    text = "x" * 5970
    embeds = render_turn(
        output_text=text, turn_index=3, story_id="one_star_ascension_s1"
    )
    assert "".join(embed.description for embed in embeds) == text
    assert sum(map(len, embeds)) == MAX_TOTAL
