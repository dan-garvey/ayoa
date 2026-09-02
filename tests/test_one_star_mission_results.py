"""Single-path terminal mission state and presentation contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.engine_bridge import EngineBridge
from app.engine.one_star_adapter import (
    load_one_star_account,
    load_one_star_hero,
    one_star_terminal_system_recipient_ids,
)
from app.schemas.events import ObservableFact
from app.schemas.narrator import VisualNovelPage
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    OneStarEventRouterOutput,
    OneStarMissionEndOperation,
    OneStarMissionUpdateOperation,
)
from app.schemas.responses import VisualNovelRender, VisualNovelRenderSegment
from tests.support.factories import character_record, router_output
from tests.test_one_star_atomicity import _checkpoint, _hero


def test_retired_report_fields_are_not_valid_operations() -> None:
    with pytest.raises(ValueError):
        OneStarMissionUpdateOperation.model_validate({
            "operation": "mission_update",
            "mission_id": "mission_1",
            "counters": [{"counter_id": "clear", "current": 1, "target": 1}],
            "report_kind": "critical",
            "report_credit": ["hero"],
        })
    with pytest.raises(ValueError):
        OneStarMissionEndOperation.model_validate({
            "operation": "mission_end",
            "mission_id": "mission_1",
            "outcome": "completed",
            "return_destination": "lobby",
            "escape_authority_id": "",
            "mvp_character_id": "hero",
            "mvp_evidence_event_id": "evt_final",
        })


def test_terminal_system_recipients_follow_existing_system_visibility() -> None:
    innate = _hero(location="lobby")
    innate.character_id = "innate"
    innate.name = "Innate"
    innate_state = load_one_star_hero(innate)
    assert innate_state is not None
    innate_state.innate_system_sight = True
    innate.mechanics[ONE_STAR_HERO_KEY] = innate_state.model_dump(mode="json")
    ordinary = _hero(location="lobby")
    ordinary.character_id = "ordinary"
    checkpoint = _checkpoint(heroes=[_hero(), innate, ordinary])
    checkpoint.characters.append(character_record("guide", location="lobby"))
    owner, account = load_one_star_account(checkpoint)
    account.state.guide_character_ids = ["guide"]
    account.state.system_observer_ids = ["guide"]
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")

    assert one_star_terminal_system_recipient_ids(checkpoint) == (
        "account_owner",
        "guide",
        "innate",
    )


@pytest.mark.asyncio
async def test_vn_deck_uses_only_narrator_result_despite_raw_canonical_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint()
    checkpoint.session.character_bindings["account_owner"] = "master-player"
    payload = router_output(
        event_id="evt_floor_complete",
        event_kind="state_change",
        observer_ids=["account_owner", "hero"],
        facts=[ObservableFact.all(
            "The survivors return to niflheim_lobby after the floor clears."
        )],
    ).model_dump(mode="json")
    payload["state_updates"] = [{
        "kind": "mission_end",
        "target_id": "mission_1",
        "value": "completed",
        "details": ["return_destination=lobby"],
    }]
    checkpoint.canonical_events.append(
        OneStarEventRouterOutput.model_validate(payload)
    )
    render = VisualNovelRender(segments=[VisualNovelRenderSegment(
        pages=[VisualNovelPage(
            kind="narration",
            text="Floor One is clear. The survivors return to Niflheim.",
        )],
        rendered_event_ids=["evt_floor_complete"],
    )])

    bridge = EngineBridge.__new__(EngineBridge)
    bridge.load_checkpoint = MagicMock(return_value=checkpoint)  # type: ignore[method-assign]
    bridge._previous_visual_novel_checkpoint = MagicMock(  # type: ignore[method-assign]
        return_value=None
    )
    bridge._prewarm_visual_novel_sprites = AsyncMock()  # type: ignore[method-assign]
    bridge.wait_for_visual_novel_stage_work = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    bridge.image_generation = MagicMock()
    bridge.image_generation.resolve_visual_novel_stage.return_value = (
        SimpleNamespace(fallback_reason=""),
        None,
    )
    bridge.visual_novel_renderer = MagicMock()
    bridge.visual_novel_renderer.render_deck.side_effect = tuple
    monkeypatch.setattr(
        "app.bot.engine_bridge.one_star_hero_card_events_for_render",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        "app.bot.engine_bridge.generated_portrait_prewarm_character_ids",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        "app.bot.engine_bridge.resolve_visual_novel_sprite_placements",
        lambda **_kwargs: (),
    )

    sections = await bridge.prepare_visual_novel_deck(
        session_id=checkpoint.session.session_id,
        checkpoint_id="ckpt_0001",
        pov_character_id="account_owner",
        render=render,
    )

    assert len(sections) == 1
    assert sections[0].card_style == "adv"
    assert sections[0].pages[0].text == (
        "Floor One is clear. The survivors return to Niflheim."
    )
    assert "niflheim_lobby" not in sections[0].pages[0].text
