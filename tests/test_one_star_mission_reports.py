"""Mission-report projection and presentation contracts for One-Star."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.engine_bridge import EngineBridge
from app.engine.one_star_adapter import (
    load_one_star_account,
    load_one_star_hero,
    one_star_mission_report_recipient_ids,
)
from app.engine.one_star_mission_reports import (
    new_one_star_mission_reports,
    one_star_mission_reports_for_render,
    render_one_star_mission_report_accessibility,
    render_one_star_mission_report_boards,
)
from app.engine.visual_novel_presentation import (
    VisualNovelCardRenderer,
    VisualNovelDeckSection,
)
from app.schemas.events import ObservableFact
from app.schemas.narrator import VisualNovelPage
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
    OneStarEventRouterOutput,
)
from app.schemas.responses import VisualNovelRender, VisualNovelRenderSegment
from app.schemas.responses import TurnResponse
from tests.support.factories import character_record, router_output
from tests.test_one_star_atomicity import _checkpoint, _hero


def _named_hero(character_id: str, name: str):
    character = _hero()
    character.character_id = character_id
    character.name = name
    return character


def _event(
    event_id: str,
    *,
    state_updates: list[dict[str, object]],
    facts: list[ObservableFact],
) -> OneStarEventRouterOutput:
    payload = router_output(
        event_id=event_id,
        event_kind="state_change",
        observer_ids=[
            "account_owner",
            "hero",
            "fallen",
            "newcomer",
            "ordinary",
        ],
        facts=facts,
    ).model_dump(mode="json")
    payload["state_updates"] = state_updates
    return OneStarEventRouterOutput.model_validate(payload)


def _completed_report_checkpoints():
    hero = _named_hero("hero", "Arden Vale")
    fallen = _named_hero("fallen", "Bryn Ash")
    newcomer = _named_hero("newcomer", "Newcomer")
    newcomer_state = load_one_star_hero(newcomer)
    assert newcomer_state is not None
    newcomer_state.innate_system_sight = True
    newcomer.mechanics[ONE_STAR_HERO_KEY] = newcomer_state.model_dump(mode="json")
    ordinary = _named_hero("ordinary", "Ordinary Hero")
    checkpoint = _checkpoint(heroes=[hero, fallen, newcomer, ordinary])
    checkpoint.session.character_bindings["newcomer"] = "player-newcomer"

    owner, account = load_one_star_account(checkpoint)
    first_reward = account.config.floor_rewards[1]
    first_scenario = account.config.floor_scenarios[1]
    account.config.floor_rewards[2] = first_reward.model_copy(
        update={"gold": 8}
    )
    account.config.floor_scenarios[2] = first_scenario.model_copy(update={
        "mission_id": "mission_2",
        "destination": "tower_floor_2",
        "premise": "Clear the second floor.",
        "completion_declaration": "the second floor is cleared",
    })
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    previous = deepcopy(checkpoint)

    start = _event(
        "evt_floor_start",
        state_updates=[{
            "kind": "mission_start",
            "target_id": "mission_1",
            "value": "1",
            "details": [
                "pending_operation_id=deployment_1",
                "party=hero",
                "party=fallen",
                "destination=tower_floor_1",
                "completion=the floor is cleared",
                "failure=the party is broken",
                "counter.clear=0/1",
            ],
        }],
        facts=[ObservableFact.all("The party enters the first floor.")],
    )
    highlight = _event(
        "evt_boss_falls",
        state_updates=[{
            "kind": "mission_update",
            "target_id": "mission_1",
            "value": "",
            "details": [
                "counter.clear=1/1",
                "report_kind=boss_kill",
                "report_credit=hero",
            ],
        }],
        facts=[
            ObservableFact.all(
                "Arden splits the floor tyrant's crown with one final blow."
            ),
            ObservableFact.only(
                "Bryn privately admits that the tyrant terrified him.",
                ["fallen"],
            ),
        ],
    )
    death = _event(
        "evt_bryn_falls",
        state_updates=[{
            "kind": "hero_delta",
            "target_id": "fallen",
            "value": "",
            "details": [
                "terminal_action=death",
                "death_cause=holding the collapsing bridge",
            ],
        }],
        facts=[ObservableFact.all("Bryn falls holding the bridge long enough.")],
    )
    end = _event(
        "evt_floor_complete",
        state_updates=[{
            "kind": "mission_end",
            "target_id": "mission_1",
            "value": "completed",
            "details": [
                "return_destination=lobby",
                "escape_authority_id=",
                "mvp_character_id=hero",
                "mvp_evidence_event_id=evt_boss_falls",
            ],
        }],
        facts=[ObservableFact.all("The System declares the floor cleared.")],
    )
    checkpoint.canonical_events.extend([start, highlight, death, end])
    owner, account = load_one_star_account(checkpoint)
    account.state.applied_event_fingerprints[end.event_id] = "committed"
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    return previous, checkpoint, end


def _render(event_id: str) -> VisualNovelRender:
    return VisualNovelRender(segments=[VisualNovelRenderSegment(
        pages=[VisualNovelPage(kind="narration", text="The party returns.")],
        rendered_event_ids=["evt_floor_start", event_id],
    )])


def test_report_projects_marked_public_facts_without_private_leakage() -> None:
    previous, checkpoint, _end = _completed_report_checkpoints()

    reports = new_one_star_mission_reports(checkpoint, previous)

    assert len(reports) == 1
    report = reports[0]
    assert report.outcome == "completed"
    assert [entry.text for entry in report.boss_kills] == [
        "Arden splits the floor tyrant's crown with one final blow."
    ]
    assert "privately admits" not in report.boss_kills[0].text
    assert [death.character_name for death in report.deaths] == ["Bryn Ash"]
    assert report.deaths[0].cause == "holding the collapsing bridge"
    assert report.reward is not None
    assert report.reward.first_clear is True
    assert report.reward.resources.gold == 4
    assert report.reward.unlocked_floor == 2
    assert report.mvp_character_id == "hero"
    assert report.mvp_evidence_event_id == "evt_boss_falls"


def test_report_projects_character_ids_before_vn_board_render() -> None:
    previous, checkpoint, _end = _completed_report_checkpoints()
    checkpoint.characters.append(_named_hero("renna_holt", "Renna Holt"))
    highlight_event = next(
        event
        for event in checkpoint.canonical_events
        if event.event_id == "evt_boss_falls"
    )
    highlight_event.canonical_event.observable_facts[0].text = (
        "renna_holt opens the tyrant's guard for Arden's final blow."
    )

    report = new_one_star_mission_reports(checkpoint, previous)[0]
    accessible = render_one_star_mission_report_accessibility(
        checkpoint=checkpoint,
        report=report,
    )
    boards = render_one_star_mission_report_boards(
        checkpoint=checkpoint,
        report=report,
    )

    assert report.boss_kills[0].text.startswith("Renna Holt opens")
    assert "renna_holt" not in accessible
    assert all("renna_holt" not in board.accessible_text for board in boards)


def test_report_recipient_selection_matches_system_sight() -> None:
    previous, checkpoint, end = _completed_report_checkpoints()
    render = _render(end.event_id)

    assert one_star_mission_report_recipient_ids(checkpoint) == (
        "account_owner",
        "newcomer",
    )
    assert len(one_star_mission_reports_for_render(
        checkpoint=checkpoint,
        previous_checkpoint=previous,
        viewer_character_id="account_owner",
        render=render,
    )) == 1
    assert len(one_star_mission_reports_for_render(
        checkpoint=checkpoint,
        previous_checkpoint=previous,
        viewer_character_id="newcomer",
        render=render,
    )) == 1
    assert one_star_mission_reports_for_render(
        checkpoint=checkpoint,
        previous_checkpoint=previous,
        viewer_character_id="ordinary",
        render=render,
    ) == ()

    owner, account = load_one_star_account(checkpoint)
    checkpoint.characters.append(character_record("iselle", location="lobby"))
    account.state.guide_character_ids = ["iselle"]
    account.state.system_observer_ids = ["iselle"]
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    assert one_star_mission_report_recipient_ids(checkpoint) == (
        "account_owner",
        "iselle",
        "newcomer",
    )

    account.config.hero_system_visibility_research_key = "system_sight"
    account.state.research_levels["system_sight"] = 1
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    assert "ordinary" in one_star_mission_report_recipient_ids(checkpoint)


def test_report_board_is_player_safe_and_renders_as_real_system_panel(
    tmp_path,
) -> None:
    previous, checkpoint, _end = _completed_report_checkpoints()
    report = new_one_star_mission_reports(checkpoint, previous)[0]

    accessible = render_one_star_mission_report_accessibility(
        checkpoint=checkpoint,
        report=report,
    )
    boards = render_one_star_mission_report_boards(
        checkpoint=checkpoint,
        report=report,
    )

    assert "MVP: Arden Vale" in accessible
    assert report.mvp_evidence_event_id not in accessible
    assert report.mission_id not in accessible
    assert all(board.media.width == 1024 for board in boards)
    assert all(board.media.height == 576 for board in boards)
    sections = [
        VisualNovelDeckSection(
            pages=(VisualNovelPage(
                kind="narration",
                text=board.accessible_text,
            ),),
            stage_media=board.media,
            card_style="system_panel",
        )
        for board in boards
    ]
    deck = VisualNovelCardRenderer(tmp_path / "presentations").render_deck(
        sections
    )
    assert [card.image_bytes for card in deck.cards] == [
        board.media.data for board in boards
    ]
    assert report.mvp_evidence_event_id not in deck.transcript
    assert report.mission_id not in deck.transcript


def test_bridge_appends_report_prose_to_bound_system_povs() -> None:
    previous, checkpoint, _end = _completed_report_checkpoints()
    checkpoint.session.character_bindings["account_owner"] = "player-owner"
    bridge = EngineBridge.__new__(EngineBridge)
    bridge.load_checkpoint = MagicMock(return_value=checkpoint)  # type: ignore[method-assign]
    bridge._previous_visual_novel_checkpoint = MagicMock(  # type: ignore[method-assign]
        return_value=previous
    )
    response = TurnResponse(
        session_id=checkpoint.session.session_id,
        checkpoint_id="ckpt_0001",
        output_text="The party returns.",
        per_player_renders={
            "account_owner": "The party returns.",
            "newcomer": "The party returns around you.",
            "ordinary": "The party returns around you.",
        },
    )

    bridge._append_one_star_mission_report_prose(
        response,
        acting_character_id="account_owner",
    )

    assert "System mission report" in response.output_text
    assert "MVP: Arden Vale" in response.output_text
    assert "System mission report" in response.per_player_renders["newcomer"]
    assert "System mission report" not in response.per_player_renders["ordinary"]


@pytest.mark.asyncio
async def test_bridge_inserts_report_board_after_matching_vn_segment(
    monkeypatch,
) -> None:
    previous, checkpoint, end = _completed_report_checkpoints()
    checkpoint.session.character_bindings["account_owner"] = "player-owner"
    report = new_one_star_mission_reports(checkpoint, previous)[0]
    report_boards = render_one_star_mission_report_boards(
        checkpoint=checkpoint,
        report=report,
    )
    bridge = EngineBridge.__new__(EngineBridge)
    bridge.load_checkpoint = MagicMock(return_value=checkpoint)  # type: ignore[method-assign]
    bridge._previous_visual_novel_checkpoint = MagicMock(  # type: ignore[method-assign]
        return_value=previous
    )
    bridge._prewarm_visual_novel_sprites = AsyncMock()  # type: ignore[method-assign]
    bridge.wait_for_visual_novel_stage_work = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    bridge.image_generation = MagicMock()
    bridge.image_generation.resolve_visual_novel_stage.return_value = (
        SimpleNamespace(fallback_reason=""),
        report_boards[0].media,
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
        render=_render(end.event_id),
    )

    assert [section.card_style for section in sections] == [
        "adv",
        *("system_panel" for _board in report_boards),
    ]
    assert [
        section.pages[0].text for section in sections[1:]
    ] == [board.accessible_text for board in report_boards]


def test_bridge_requires_each_bound_system_pov_vn_end_segment() -> None:
    from app.engine.one_star_mission_reports import OneStarMissionReportError

    previous, checkpoint, end = _completed_report_checkpoints()
    checkpoint.session.character_bindings["account_owner"] = "player-owner"
    bridge = EngineBridge.__new__(EngineBridge)
    bridge.load_checkpoint = MagicMock(return_value=checkpoint)  # type: ignore[method-assign]
    bridge._previous_visual_novel_checkpoint = MagicMock(  # type: ignore[method-assign]
        return_value=previous
    )
    checkpoint.session.config.settings.presentation_mode = "visual_novel"
    response = TurnResponse(
        session_id=checkpoint.session.session_id,
        checkpoint_id="ckpt_0001",
        per_player_visual_novel_renders={
            "account_owner": _render(end.event_id),
        },
    )

    with pytest.raises(
        OneStarMissionReportError,
        match="recipient_render_missing_mission_end",
    ):
        bridge._validate_one_star_mission_report_routing(response)

    response.per_player_visual_novel_renders["newcomer"] = _render(
        end.event_id
    )
    bridge._validate_one_star_mission_report_routing(response)
