"""Focused unit tests for bot-layer internals that the code review
flagged as having no coverage. Covers:

- F3.9: best-effort parsing of DISCORD_ADMIN_USER_IDS (_is_admin).
- briefing copy: render_briefing must not mention `/describe` (legacy
  command renamed in the join-overhaul) and must point at `/join`.
- POV-thread cascade: `_post_actor_render` falls thread → DM → none.

Heavier discord-interaction paths (full /act and /join harness, orphan
thread purge) still aren't covered — they'd need a discord.py mock
infrastructure we don't have.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.bot import commands as bot_commands
from app.bot.engine_bridge import EngineBridge, _narrator_history_message_text
from app.engine.frontend_views import (
    CharacterSummary,
    OpeningLobbyView,
    PlayerJoinResult,
    RetryRenderResult,
)
from app.engine.image_job_store import VisualNovelStageResolution
from app.engine.player_media import ResolvedPlayerMedia
from app.engine.visual_novel_presentation import (
    VisualNovelCard,
    VisualNovelCardRenderer,
    VisualNovelDeckSection,
)
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_pack import SafeAssetRevealPayload
from app.schemas.dnd_inventory import DndLootOffer
from app.schemas.narrator import VisualNovelPage
from app.schemas.responses import (
    DiceRollDisplay,
    TurnResponse,
    VisualNovelRender,
    VisualNovelRenderSegment,
)
from app.schemas.state import (
    PendingNarratorRender,
    SessionState,
    StorySetting,
    WorldState,
)


# ---- F3.9: admin env parsing ------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_admin_cache():
    """_is_admin memoizes per env-value. Clear between tests so each test
    sees a fresh parse."""
    bot_commands._ADMIN_CACHE = None
    yield
    bot_commands._ADMIN_CACHE = None


class TestAdminEnvParsing:
    def test_unset_env_denies_everyone(self, monkeypatch):
        monkeypatch.delenv("DISCORD_ADMIN_USER_IDS", raising=False)
        assert bot_commands._is_admin(12345) is False

    def test_empty_env_denies_everyone(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "")
        assert bot_commands._is_admin(12345) is False

    def test_single_valid_id(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "12345")
        assert bot_commands._is_admin(12345) is True
        assert bot_commands._is_admin(99999) is False

    def test_comma_list(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "111, 222 , 333")
        for uid in (111, 222, 333):
            assert bot_commands._is_admin(uid) is True
        assert bot_commands._is_admin(444) is False

    def test_bad_entries_skipped_not_rejecting_whole_list(self, monkeypatch, caplog):
        """Core F3.9 fix: one bad entry doesn't nuke all admin access."""
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "111,notanumber,222")
        with caplog.at_level(logging.WARNING):
            assert bot_commands._is_admin(111) is True
            assert bot_commands._is_admin(222) is True
        assert any("notanumber" in r.message for r in caplog.records)

    def test_warning_logged_once_per_unique_env_value(self, monkeypatch, caplog):
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "111,nope")
        with caplog.at_level(logging.WARNING):
            bot_commands._is_admin(111)
            bot_commands._is_admin(111)
            bot_commands._is_admin(111)
        bad_warnings = [r for r in caplog.records if "nope" in r.message]
        assert len(bad_warnings) == 1

    def test_warning_re_fires_when_env_changes(self, monkeypatch, caplog):
        monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "111,nope")
        with caplog.at_level(logging.WARNING):
            bot_commands._is_admin(111)
            monkeypatch.setenv("DISCORD_ADMIN_USER_IDS", "222,stillbad")
            bot_commands._is_admin(222)
        bad_warnings = [r for r in caplog.records if "stillbad" in r.message]
        assert len(bad_warnings) == 1


class TestNumberedReferenceHelpers:
    def test_numbered_ref_lines_are_one_based(self):
        assert bot_commands._numbered_ref_lines(["alpha", "beta"]) == (
            "1: `alpha`\n2: `beta`"
        )

    def test_numbered_ref_resolves_index_or_raw_id(self):
        choices = ["alpha", "beta"]

        assert bot_commands._resolve_numbered_ref(
            "2", choices, label="story",
        ) == "beta"
        assert bot_commands._resolve_numbered_ref(
            "alpha", choices, label="story",
        ) == "alpha"
        with pytest.raises(ValueError, match="numbered 3"):
            bot_commands._resolve_numbered_ref("3", choices, label="story")


class TestSettingsMutation:
    def test_discord_defers_privately_before_awaiting_setting_write(self):
        class FakeTree:
            def __init__(self):
                self.commands = {}
                self.groups = {}

            def command(self, *, name, **_kwargs):
                def _decorator(fn):
                    self.commands[name] = fn
                    return fn

                return _decorator

            def add_command(self, group, **_kwargs):
                self.groups[group.name] = group

        engine = MagicMock()
        engine.set_setting = AsyncMock(return_value="visual_novel")
        smap = MagicMock()
        smap.get = AsyncMock(return_value=SimpleNamespace(session_id="s"))
        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)
        command = tree.groups["settings"].get_command("set")
        inter = MagicMock()
        inter.channel_id = 123
        inter.response.defer = AsyncMock()
        inter.response.send_message = AsyncMock()
        inter.followup.send = AsyncMock()

        asyncio.run(command.callback(inter, "presentation_mode", "vn"))

        inter.response.defer.assert_awaited_once_with(
            thinking=True,
            ephemeral=True,
        )
        engine.set_setting.assert_awaited_once_with(
            "s",
            "presentation_mode",
            "vn",
        )
        inter.response.send_message.assert_not_awaited()
        inter.followup.send.assert_awaited_once_with(
            "Setting `presentation_mode` → `visual_novel`.",
            ephemeral=True,
        )


@pytest.fixture
def mock_bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )


def _visual_novel_stage_media(
    color: tuple[int, int, int],
) -> ResolvedPlayerMedia:
    image = Image.new("RGB", (1024, 576), (19, 31, 47))
    image.putpixel((5, 5), color)
    stream = BytesIO()
    image.save(stream, format="PNG")
    data = stream.getvalue()
    return ResolvedPlayerMedia(
        filename="visual-novel-stage.png",
        mime_type="image/png",
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        width=1024,
        height=576,
    )


class TestEngineBridgeVisualNovelPresentation:
    def test_segments_resolve_ordered_stages_and_wait_for_stable_union(
        self,
        mock_bridge: EngineBridge,
    ):
        render = VisualNovelRender(segments=[
            VisualNovelRenderSegment(
                pages=[VisualNovelPage(
                    kind="dialogue",
                    speaker="Iselle",
                    text=" ".join(["Stay on the first stage."] * 45),
                )],
                rendered_event_ids=["evt_first", "evt_shared"],
            ),
            VisualNovelRenderSegment(
                pages=[VisualNovelPage(
                    kind="dialogue",
                    speaker="Wren",
                    text="Now the scene has changed.",
                )],
                rendered_event_ids=["evt_shared", "evt_second"],
            ),
        ])
        mock_bridge.image_sidecar.wait_for_stage_discovery = AsyncMock()
        mock_bridge.image_generation.wait_for_render_images = AsyncMock(
            return_value=True
        )
        first_media = _visual_novel_stage_media((221, 37, 73))
        second_media = _visual_novel_stage_media((17, 199, 101))
        mock_bridge.image_generation.resolve_visual_novel_stage = MagicMock(
            side_effect=[
                (
                    VisualNovelStageResolution(
                        action="replace",
                        artifact=None,
                    ),
                    first_media,
                ),
                (
                    VisualNovelStageResolution(
                        action="replace",
                        artifact=None,
                    ),
                    second_media,
                ),
            ]
        )

        deck = asyncio.run(mock_bridge.prepare_visual_novel_deck(
            session_id="session",
            pov_character_id="alice",
            render=render,
        ))

        mock_bridge.image_sidecar.wait_for_stage_discovery.assert_awaited_once_with(
            "session"
        )
        mock_bridge.image_generation.wait_for_render_images.assert_awaited_once_with(
            session_id="session",
            rendered_event_ids_by_pov={
                "alice": ["evt_first", "evt_shared", "evt_second"]
            },
        )
        assert [
            invocation.kwargs["rendered_event_ids"]
            for invocation in (
                mock_bridge.image_generation.resolve_visual_novel_stage
                .call_args_list
            )
        ] == [
            ["evt_first", "evt_shared"],
            ["evt_shared", "evt_second"],
        ]
        iselle_cards = [
            card for card in deck.cards if card.speaker == "Iselle"
        ]
        wren_cards = [card for card in deck.cards if card.speaker == "Wren"]
        assert len(iselle_cards) > 1
        assert len(wren_cards) == 1
        for card in iselle_cards:
            with Image.open(card.image_path) as image:
                assert image.convert("RGB").getpixel((5, 5)) == (
                    221, 37, 73
                )
        with Image.open(wren_cards[0].image_path) as image:
            assert image.convert("RGB").getpixel((5, 5)) == (17, 199, 101)


class TestEngineBridgeQuery:
    def test_run_query_routes_through_turn_loop(self, mock_bridge: EngineBridge):
        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0001",
            turn_index=1,
            output_text="You can see Pip's red coat.",
            per_player_renders={"alice": "You can see Pip's red coat."},
            beat_ended_reason="query_response",
        )
        mock_bridge.run_turn = AsyncMock(return_value=response)

        result = asyncio.run(mock_bridge.run_query(
            session_id="s",
            character_id="alice",
            question=" what does Pip look like? ",
        ))

        mock_bridge.run_turn.assert_awaited_once_with(
            session_id="s",
            user_input="(query: what does Pip look like?)",
            acting_character_id="alice",
        )
        assert result is response


class TestEngineBridgeRetry:
    def test_retry_failed_render_resumes_without_submitting_new_act(
        self, mock_bridge: EngineBridge,
    ):
        ckpt = CheckpointFile(
            session=SessionState(
                session_id="session",
                character_bindings={"alice": "42"},
                pending_narrator_render=PendingNarratorRender(
                    ended_reason="response_requested",
                    events_closed=1,
                    event_actor_ids=["alice"],
                    acting_player_id="alice",
                    acting_player_input="I look around",
                ),
            ),
            world_state=WorldState(),
            characters=[CharacterRecord(character_id="alice", name="Alice")],
        )
        mock_bridge.checkpoint_mgr.save(ckpt)
        response = TurnResponse(
            session_id="session",
            checkpoint_id="ckpt_0001",
            turn_index=1,
            output_text="Recovered POV",
            per_player_renders={"alice": "Recovered POV"},
            beat_ended_reason="response_requested",
        )
        mock_bridge.sweep_stale_pins = MagicMock()
        mock_bridge.orchestrator.retry_pending_narrator_render = AsyncMock(
            return_value=response
        )
        mock_bridge.orchestrator.process_turn = AsyncMock()

        result = asyncio.run(
            mock_bridge.retry_failed_render(session_id="session")
        )

        assert result.response is response
        assert result.actor_character_id == "alice"
        assert result.actor_user_id == "42"
        mock_bridge.sweep_stale_pins.assert_not_called()
        mock_bridge.orchestrator.process_turn.assert_not_called()
        retry = mock_bridge.orchestrator.retry_pending_narrator_render
        retry.assert_awaited_once_with("session")

    def test_retry_failed_render_noops_when_no_pending_render(
        self, mock_bridge: EngineBridge,
    ):
        ckpt = CheckpointFile(
            session=SessionState(session_id="session", turn_index=3),
            world_state=WorldState(),
            characters=[],
        )
        mock_bridge.checkpoint_mgr.save(ckpt)
        mock_bridge.orchestrator.retry_pending_narrator_render = AsyncMock()
        mock_bridge.orchestrator.process_turn = AsyncMock()

        result = asyncio.run(
            mock_bridge.retry_failed_render(session_id="session")
        )

        assert result.actor_character_id == ""
        assert result.actor_user_id == ""
        assert result.response.turn_index == 3
        assert result.response.beat_ended_reason == "no_pending_render"
        assert "No failed narrator render" in result.response.output_text
        mock_bridge.orchestrator.retry_pending_narrator_render.assert_not_called()
        mock_bridge.orchestrator.process_turn.assert_not_called()


class TestRetryCommandDelivery:
    def test_retry_delivers_recovered_render_without_new_act(self, monkeypatch):
        class FakeTree:
            def __init__(self):
                self.commands = {}

            def command(self, *, name, **_kwargs):
                def _decorator(fn):
                    self.commands[name] = fn
                    return fn

                return _decorator

            def add_command(self, *_args, **_kwargs):
                return None

        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0002",
            turn_index=2,
            output_text="Recovered POV",
            per_player_renders={"alice": "Recovered POV"},
            beat_ended_reason="response_requested",
        )
        result = RetryRenderResult(
            response=response,
            actor_character_id="alice",
            actor_user_id="42",
        )

        engine = MagicMock()
        engine.retry_failed_render = AsyncMock(return_value=result)
        engine.run_turn = AsyncMock()

        smap = MagicMock()
        smap.get = AsyncMock(
            return_value=SimpleNamespace(session_id="s", story_id="story"),
        )

        captured: dict = {}

        async def _fake_deliver(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(
            bot_commands, "_deliver_turn_response_to_povs", _fake_deliver,
        )

        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)

        actor_user = MagicMock()
        actor_user.id = 42
        inter = MagicMock()
        inter.channel_id = 123
        inter.channel = object()
        inter.user = MagicMock()
        inter.user.id = 7
        inter.user.display_name = "Retry User"
        inter.client.get_user.return_value = actor_user
        inter.client.fetch_user = AsyncMock()
        inter.response.defer = AsyncMock()
        inter.response.send_message = AsyncMock()
        inter.followup.send = AsyncMock()

        asyncio.run(tree.commands["retry"](inter))

        inter.response.defer.assert_awaited_once_with(thinking=True)
        engine.retry_failed_render.assert_awaited_once_with(session_id="s")
        engine.run_turn.assert_not_awaited()
        assert captured["response"] is response
        assert captured["actor_character_id"] == "alice"
        assert captured["actor_user"] is actor_user
        assert captured["story_id"] == "story"
        inter.client.fetch_user.assert_not_awaited()

    def test_retry_no_pending_render_reports_ephemeral(self, monkeypatch):
        class FakeTree:
            def __init__(self):
                self.commands = {}

            def command(self, *, name, **_kwargs):
                def _decorator(fn):
                    self.commands[name] = fn
                    return fn

                return _decorator

            def add_command(self, *_args, **_kwargs):
                return None

        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0002",
            turn_index=2,
            output_text="No failed narrator render is pending for this session.",
            per_player_renders={},
            beat_ended_reason="no_pending_render",
        )
        engine = MagicMock()
        engine.retry_failed_render = AsyncMock(
            return_value=RetryRenderResult(response=response)
        )
        engine.run_turn = AsyncMock()
        smap = MagicMock()
        smap.get = AsyncMock(
            return_value=SimpleNamespace(session_id="s", story_id="story"),
        )
        deliver = AsyncMock()
        monkeypatch.setattr(
            bot_commands, "_deliver_turn_response_to_povs", deliver,
        )

        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)

        inter = MagicMock()
        inter.channel_id = 123
        inter.user = MagicMock()
        inter.user.display_name = "Retry User"
        inter.response.defer = AsyncMock()
        inter.response.send_message = AsyncMock()
        inter.followup.send = AsyncMock()

        asyncio.run(tree.commands["retry"](inter))

        inter.response.defer.assert_awaited_once_with(thinking=True)
        engine.retry_failed_render.assert_awaited_once_with(session_id="s")
        deliver.assert_not_awaited()
        inter.followup.send.assert_awaited_once_with(
            "No failed narrator render is pending for this session.",
            ephemeral=True,
        )


class TestLootRouterSync:
    def test_claim_loot_queues_next_router_update(
        self,
        mock_bridge: EngineBridge,
    ):
        ckpt = CheckpointFile(
            session=SessionState(
                session_id="loot_sync",
                character_bindings={"alice": "42"},
            ),
            world_state=WorldState(),
            characters=[
                CharacterRecord(
                    character_id="alice",
                    name="Alice",
                    mechanics={
                        "ruleset_id": "dnd5e_basic",
                        "dnd5e_sheet": {
                            "statblock": {
                                "inventory": {
                                    "items": [],
                                    "currency": {"gp": 2},
                                },
                            },
                        },
                    },
                ),
            ],
        )
        ckpt.session.dnd_inventory_offers.append(DndLootOffer(
            offer_id="loot_evt_chest",
            source_event_id="evt_chest",
            source_kind="container",
            source_label="iron chest",
            eligible_character_ids=["alice"],
            items=[
                {
                    "item_id": "healing_potion",
                    "name": "Potion of Healing",
                    "kind": "consumable",
                    "quantity": 1,
                    "identified": True,
                    "requires_identification": False,
                    "requires_attunement": False,
                    "consumable": True,
                    "value_gp": 50,
                    "weight": 0.5,
                    "notes": "",
                },
            ],
            currency={"sp": 8},
        ))
        mock_bridge.checkpoint_mgr.save(ckpt)

        result = asyncio.run(mock_bridge.claim_loot(
            session_id="loot_sync",
            user_id=42,
            character_id="alice",
            offer_id="loot_evt_chest",
            item_ids=[],
            take_currency=True,
            take_all_available=True,
        ))

        assert result.message.startswith("Claimed Potion of Healing.")
        reloaded = mock_bridge.load_latest("loot_sync")
        assert len(reloaded.session.pending_engine_state_updates) == 1
        update = reloaded.session.pending_engine_state_updates[0]
        assert "Inventory update before the next action" in update
        assert "alice took Potion of Healing and 8 sp from iron chest" in update
        assert "established inventory continuity" in update
        assert "player" not in update.lower()

    def test_decline_loot_does_not_queue_router_update(
        self,
        mock_bridge: EngineBridge,
    ):
        ckpt = CheckpointFile(
            session=SessionState(
                session_id="loot_decline",
                character_bindings={"alice": "42"},
            ),
            world_state=WorldState(),
            characters=[
                CharacterRecord(
                    character_id="alice",
                    name="Alice",
                    mechanics={"ruleset_id": "dnd5e_basic"},
                ),
            ],
        )
        ckpt.session.dnd_inventory_offers.append(DndLootOffer(
            offer_id="loot_evt_chest",
            source_event_id="evt_chest",
            source_kind="container",
            source_label="iron chest",
            eligible_character_ids=["alice"],
            currency={"sp": 8},
        ))
        mock_bridge.checkpoint_mgr.save(ckpt)

        result = asyncio.run(mock_bridge.decline_loot(
            session_id="loot_decline",
            user_id=42,
            character_id="alice",
            offer_id="loot_evt_chest",
        ))

        assert result.message.startswith("Declined the loot offer")
        reloaded = mock_bridge.load_latest("loot_decline")
        assert reloaded.session.pending_engine_state_updates == []


class TestRewindMetadata:
    def test_single_binding_supplies_actor_when_player_id_is_empty(
        self,
        mock_bridge: EngineBridge,
    ):
        ckpt = CheckpointFile(
            session=SessionState(
                session_id="rewind_meta",
                turn_index=0,
                character_bindings={"alice": "42"},
            ),
            world_state=WorldState(),
            characters=[
                CharacterRecord(
                    character_id="alice",
                    name="Alice",
                    location="cellar",
                    is_playable=True,
                ),
            ],
        )
        mock_bridge.checkpoint_mgr.save(ckpt)
        later = ckpt.model_copy(deep=True)
        later.session.turn_index = 1
        mock_bridge.checkpoint_mgr.save(later)

        preview = mock_bridge.preview_rewind("rewind_meta", 0)

        assert preview.actor_character_id == "alice"
        assert preview.location == "cellar"


class TestQueryCommandDelivery:
    def test_query_uses_turn_response_delivery_path(self, monkeypatch):
        """`/query` should render like `/act`, not as a disappearing
        ephemeral plaintext answer."""

        class FakeTree:
            def __init__(self):
                self.commands = {}

            def command(self, *, name, **_kwargs):
                def _decorator(fn):
                    self.commands[name] = fn
                    return fn

                return _decorator

            def add_command(self, *_args, **_kwargs):
                return None

        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0002",
            turn_index=2,
            output_text="You can see Pip's red coat.",
            per_player_renders={"alice": "You can see Pip's red coat."},
            beat_ended_reason="query_response",
        )

        engine = MagicMock()
        engine.get_user_binding.return_value = "alice"
        engine.run_turn = AsyncMock()
        engine.run_query = AsyncMock(return_value=response)

        smap = MagicMock()
        smap.get = AsyncMock(
            return_value=SimpleNamespace(session_id="s", story_id="story"),
        )

        captured: dict = {}

        async def _fake_deliver(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(
            bot_commands, "_deliver_turn_response_to_povs", _fake_deliver,
        )

        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)

        inter = MagicMock()
        inter.channel_id = 123
        inter.channel = object()
        inter.user = MagicMock()
        inter.user.id = 42
        inter.response.defer = AsyncMock()
        inter.response.send_message = AsyncMock()
        inter.followup.send = AsyncMock()

        asyncio.run(tree.commands["query"](inter, " what does Pip look like? "))

        inter.response.defer.assert_awaited_once_with(thinking=True)
        engine.run_query.assert_awaited_once_with(
            session_id="s",
            character_id="alice",
            question="what does Pip look like?",
        )
        engine.run_turn.assert_not_awaited()
        assert captured["response"] is response
        assert captured["actor_character_id"] == "alice"
        assert captured["story_id"] == "story"

    def test_describe_preplay_updates_identity_without_opening(self):
        class FakeTree:
            def __init__(self):
                self.commands = {}

            def command(self, *, name, **_kwargs):
                def _decorator(fn):
                    self.commands[name] = fn
                    return fn

                return _decorator

            def add_command(self, *_args, **_kwargs):
                return None

        engine = MagicMock()
        engine.get_user_binding.return_value = "alice"
        engine.set_character_identity = AsyncMock()
        engine.run_begin_turn = AsyncMock()
        engine.run_turn = AsyncMock()

        smap = MagicMock()
        smap.get = AsyncMock(
            return_value=SimpleNamespace(session_id="s", story_id="story"),
        )

        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)

        inter = MagicMock()
        inter.channel_id = 123
        inter.channel = object()
        inter.user = MagicMock()
        inter.user.id = 42
        inter.user.display_name = "Alice User"
        inter.response.defer = AsyncMock()
        inter.response.send_message = AsyncMock()
        inter.followup.send = AsyncMock()

        asyncio.run(tree.commands["describe"](inter, "Alice", "red coat"))

        engine.set_character_identity.assert_awaited_once_with(
            "s",
            "alice",
            name="Alice",
            appearance="red coat",
        )
        engine.run_begin_turn.assert_not_awaited()
        engine.run_turn.assert_not_awaited()
        inter.followup.send.assert_awaited_once()
        assert inter.followup.send.await_args.kwargs["ephemeral"] is True

    def test_session_resume_requests_only_invokers_bound_history(self):
        class FakeTree:
            def __init__(self):
                self.commands = {}
                self.groups = {}

            def command(self, *, name, **_kwargs):
                def _decorator(fn):
                    self.commands[name] = fn
                    return fn

                return _decorator

            def add_command(self, group, **_kwargs):
                self.groups[group.name] = group

        ckpt = CheckpointFile(
            session=SessionState(
                session_id="s",
                story_id="story",
                character_bindings={"alice": "42", "bob": "99"},
            ),
            characters=[
                CharacterRecord(character_id="alice", name="Alice"),
                CharacterRecord(character_id="bob", name="Bob"),
            ],
        )
        engine = MagicMock()
        engine.list_session_ids.return_value = ["s"]
        engine.load_latest.return_value = ckpt
        engine.turn_history.return_value = [SimpleNamespace(
            entry=SimpleNamespace(assistant="Alice's private history."),
        )]
        smap = MagicMock()
        smap.get = AsyncMock(return_value=None)
        smap.upsert = AsyncMock()
        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)
        resume = tree.groups["session"].get_command("resume")
        inter = MagicMock()
        inter.channel_id = 123
        inter.guild_id = 7
        inter.user = MagicMock(id=42)
        inter.response.send_message = AsyncMock()

        asyncio.run(resume.callback(inter, "s"))

        engine.turn_history.assert_called_once_with("s", "alice")
        content = inter.response.send_message.await_args.args[0]
        assert "Alice's private history" in content
        assert inter.response.send_message.await_args.kwargs["ephemeral"] is True

class TestXpAwardCommandPermissions:
    def test_session_owner_cannot_award_xp_without_admin_env(self, monkeypatch):
        class FakeTree:
            def __init__(self):
                self.commands = {}
                self.groups = {}

            def command(self, *, name, **_kwargs):
                def _decorator(fn):
                    self.commands[name] = fn
                    return fn

                return _decorator

            def add_command(self, group, **_kwargs):
                self.groups[group.name] = group
                return None

        monkeypatch.delenv("DISCORD_ADMIN_USER_IDS", raising=False)
        engine = MagicMock()
        engine.award_dnd_experience_locked = AsyncMock(return_value=[])
        smap = MagicMock()
        smap.get = AsyncMock(
            return_value=SimpleNamespace(
                session_id="s",
                story_id="story",
                owner_user_id=42,
            ),
        )

        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)
        xp_award = tree.groups["xp"].get_command("award")

        inter = MagicMock()
        inter.channel_id = 123
        inter.user = MagicMock()
        inter.user.id = 42
        inter.response.send_message = AsyncMock()
        inter.response.defer = AsyncMock()
        inter.followup.send = AsyncMock()

        asyncio.run(xp_award.callback(inter, "all", 100, "session reward"))

        inter.response.send_message.assert_awaited_once_with(
            "Admin-only command.",
            ephemeral=True,
        )
        inter.response.defer.assert_not_awaited()
        engine.award_dnd_experience_locked.assert_not_awaited()
        inter.followup.send.assert_not_awaited()


class TestOneStarMasterCommands:
    def test_discord_master_group_uses_bound_viewpoint_and_shared_projection(
        self,
        monkeypatch,
    ):
        class FakeTree:
            def __init__(self):
                self.commands = {}
                self.groups = {}

            def command(self, *, name, **_kwargs):
                def _decorator(fn):
                    self.commands[name] = fn
                    return fn

                return _decorator

            def add_command(self, group, **_kwargs):
                self.groups[group.name] = group

        engine = MagicMock()
        engine.get_user_binding.return_value = "owner"
        engine.one_star_master_command.return_value = (
            "Tired Baker [hero]",
            "HP 17/53",
        )
        smap = MagicMock()
        smap.get = AsyncMock(return_value=SimpleNamespace(
            session_id="s",
            story_id="one_star",
        ))
        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)
        master = tree.groups["master"]
        assert master.get_command("status") is not None
        assert master.get_command("heroes") is not None
        assert master.get_command("synthesis") is not None
        hero_command = master.get_command("hero")

        inter = MagicMock()
        inter.channel_id = 123
        inter.channel = object()
        inter.user = MagicMock(id=42)
        inter.response.send_message = AsyncMock()
        inter.followup.send = AsyncMock()

        asyncio.run(hero_command.callback(inter, "Tired Baker"))

        engine.get_user_binding.assert_called_once_with("s", 42)
        engine.one_star_master_command.assert_called_once_with(
            "s",
            "owner",
            "hero",
            hero_ref="Tired Baker",
        )
        sent = inter.response.send_message.await_args
        assert sent.kwargs["ephemeral"] is True
        assert "Tired Baker [hero]" in sent.kwargs["embed"].description
        assert "HP 17/53" in sent.kwargs["embed"].description
        inter.followup.send.assert_not_awaited()

        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0002",
            turn_index=2,
            output_text="The selection opens.",
            per_player_renders={"owner": "The selection opens."},
            beat_ended_reason="cat_ii_pending",
        )
        engine.run_one_star_synthesis_command = AsyncMock(
            return_value=response
        )
        deliver = AsyncMock()
        monkeypatch.setattr(
            bot_commands,
            "_deliver_turn_response_to_povs",
            deliver,
        )
        synthesis_command = master.get_command("synthesis")
        synthesis_inter = MagicMock()
        synthesis_inter.channel_id = 123
        synthesis_inter.channel = object()
        synthesis_inter.user = MagicMock(id=42, display_name="Master")
        synthesis_inter.response.send_message = AsyncMock()
        synthesis_inter.response.defer = AsyncMock()
        synthesis_inter.followup.send = AsyncMock()

        asyncio.run(synthesis_command.callback(
            synthesis_inter,
            "Tired Baker",
            "Edric, Pip the Younger",
        ))

        engine.run_one_star_synthesis_command.assert_awaited_once_with(
            "s",
            "owner",
            target_ref="Tired Baker",
            source_refs=("Edric", "Pip the Younger"),
        )
        synthesis_inter.response.defer.assert_awaited_once_with(thinking=True)
        deliver.assert_awaited_once()
        assert deliver.await_args.kwargs["response"] is response


class TestTurnResponseDelivery:
    def test_actor_render_goes_through_embed_thread_delivery(self, monkeypatch):
        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0003",
            turn_index=3,
            output_text="The answer lands as narrator prose.",
            per_player_renders={"alice": "The answer lands as narrator prose."},
            beat_ended_reason="query_response",
        )

        engine = MagicMock()
        engine.load_latest.return_value = SimpleNamespace(
            session=SimpleNamespace(character_bindings={"alice": "42"}),
            characters=[SimpleNamespace(character_id="alice", name="Alice")],
        )

        inter = MagicMock()
        inter.channel_id = 123
        inter.channel = object()
        inter.user = MagicMock()
        inter.user.id = 42
        inter.client = MagicMock()
        inter.followup.send = AsyncMock()

        smap = MagicMock()
        captured = {}
        thread = MagicMock()
        thread.id = 999
        thread.mention = "<#999>"

        async def _fake_post_actor_render(**kwargs):
            captured.update(kwargs)
            return ("thread", thread)

        clear = AsyncMock()
        public_fallback = AsyncMock()
        monkeypatch.setattr(
            bot_commands, "_post_actor_render", _fake_post_actor_render,
        )
        monkeypatch.setattr(bot_commands, "_clear_interaction_response", clear)
        monkeypatch.setattr(
            bot_commands, "_send_public_turn_render", public_fallback,
        )

        asyncio.run(bot_commands._deliver_turn_response_to_povs(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id="s",
            story_id="story",
            actor_character_id="alice",
            actor_user=inter.user,
            response=response,
        ))

        assert captured["character_id"] == "alice"
        assert captured["char_name"] == "Alice"
        assert captured["session_id"] == "s"
        assert captured["turn_index"] == 3
        assert captured["embeds"]
        clear.assert_awaited_once_with(inter)
        public_fallback.assert_not_awaited()
        inter.followup.send.assert_not_awaited()

    def test_actor_private_delivery_failure_never_publishes_prose(
        self, monkeypatch,
    ):
        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0003",
            turn_index=3,
            output_text="Secret actor-only result.",
            per_player_renders={"alice": "Secret actor-only result."},
            beat_ended_reason="query_response",
        )
        engine = MagicMock()
        engine.load_latest.return_value = SimpleNamespace(
            session=SimpleNamespace(character_bindings={"alice": "42"}),
            characters=[SimpleNamespace(character_id="alice", name="Alice")],
        )
        inter = MagicMock()
        inter.user = MagicMock(id=42)
        inter.client = MagicMock()
        inter.followup.send = AsyncMock()
        public_render = AsyncMock()
        monkeypatch.setattr(
            bot_commands,
            "_post_actor_render",
            AsyncMock(return_value=("none", None)),
        )
        monkeypatch.setattr(
            bot_commands,
            "_send_public_turn_render",
            public_render,
        )

        asyncio.run(bot_commands._deliver_turn_response_to_povs(
            inter=inter,
            smap=MagicMock(),
            engine=engine,
            session_id="s",
            story_id="story",
            actor_character_id="alice",
            actor_user=inter.user,
            response=response,
        ))

        public_render.assert_not_awaited()
        inter.followup.send.assert_awaited_once()
        args, kwargs = inter.followup.send.await_args
        assert "Nothing was posted publicly" in args[0]
        assert "Secret actor-only result" not in args[0]
        assert kwargs["ephemeral"] is True

    def test_begin_private_delivery_failure_remains_fail_closed(
        self, monkeypatch,
    ):
        class FakeTree:
            def __init__(self):
                self.commands = {}

            def command(self, *, name, **_kwargs):
                def _decorator(fn):
                    self.commands[name] = fn
                    return fn

                return _decorator

            def add_command(self, *_args, **_kwargs):
                return None

        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0001",
            turn_index=1,
            output_text="Alice's private opening.",
            per_player_renders={"alice": "Alice's private opening."},
            beat_ended_reason="state_change",
        )
        engine = MagicMock()
        engine.get_user_binding.return_value = "alice"
        engine.opening_lobby.return_value = OpeningLobbyView(
            requires_confirmation=True,
            claimed_seat_names=("Alice",),
            open_seat_names=(),
        )
        engine.run_begin_turn = AsyncMock(return_value=response)
        engine.load_latest.return_value = SimpleNamespace(
            characters=[SimpleNamespace(character_id="alice", name="Alice")],
            session=SimpleNamespace(character_bindings={"alice": "42"}),
        )
        smap = MagicMock()
        smap.get = AsyncMock(return_value=SimpleNamespace(
            session_id="s",
            story_id="story",
        ))
        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)
        inter = MagicMock()
        inter.channel_id = 123
        inter.user = MagicMock(id=42)
        inter.client = MagicMock()
        inter.response.defer = AsyncMock()
        inter.response.send_message = AsyncMock()
        inter.followup.send = AsyncMock()
        public_render = AsyncMock()
        monkeypatch.setattr(
            bot_commands,
            "_post_actor_render",
            AsyncMock(return_value=("none", None)),
        )
        monkeypatch.setattr(
            bot_commands,
            "_send_public_turn_render",
            public_render,
        )

        asyncio.run(tree.commands["begin"](inter, True))

        public_render.assert_not_awaited()
        followups = [str(call.args[0]) for call in inter.followup.send.await_args_list]
        assert any("Nothing was posted publicly" in text for text in followups)
        assert all("Alice's private opening" not in text for text in followups)

    def test_actor_asset_failure_does_not_trigger_public_render(
        self, monkeypatch,
    ):
        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0003",
            turn_index=3,
            output_text="The answer lands as narrator prose.",
            per_player_renders={"alice": "The answer lands as narrator prose."},
            per_player_asset_reveals={"alice": [_asset_payload()]},
            beat_ended_reason="query_response",
        )

        engine = MagicMock()
        engine.load_latest.return_value = SimpleNamespace(
            session=SimpleNamespace(character_bindings={"alice": "42"}),
            characters=[SimpleNamespace(character_id="alice", name="Alice")],
        )

        inter = MagicMock()
        inter.channel_id = 123
        inter.channel = object()
        inter.user = MagicMock()
        inter.user.id = 42
        inter.client = MagicMock()
        inter.followup.send = AsyncMock()

        smap = MagicMock()
        thread = MagicMock()
        thread.id = 999
        events: list[str] = []

        async def _fake_post_actor_render(**_kwargs):
            events.append("render")
            return ("thread", thread)

        async def _fake_post_assets_to_pov(**kwargs):
            events.append("assets")
            assert kwargs["user_id"] == 42
            assert kwargs["character_id"] == "alice"
            assert kwargs["asset_reveals"] == response.per_player_asset_reveals[
                "alice"
            ]
            return False

        clear = AsyncMock()
        public_fallback = AsyncMock()
        monkeypatch.setattr(
            bot_commands, "_post_actor_render", _fake_post_actor_render,
        )
        monkeypatch.setattr(
            bot_commands, "_post_assets_to_pov", _fake_post_assets_to_pov,
        )
        monkeypatch.setattr(bot_commands, "_clear_interaction_response", clear)
        monkeypatch.setattr(
            bot_commands, "_send_public_turn_render", public_fallback,
        )

        asyncio.run(bot_commands._deliver_turn_response_to_povs(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id="s",
            story_id="story",
            actor_character_id="alice",
            actor_user=inter.user,
            response=response,
        ))

        assert events == ["render", "assets"]
        clear.assert_awaited_once_with(inter)
        public_fallback.assert_not_awaited()

    def test_actor_auto_rolls_deliver_before_render(self, monkeypatch):
        roll = DiceRollDisplay(
            event_id="evt_roll",
            roll_id="attack_rat",
            actor_id="alice",
            actor_name="Alice",
            target_id="rat",
            target_name="Rat",
            label="Attack",
            die_values=[13],
            kept_die_values=[13],
            modifier=4,
            total=17,
            dc=12,
            outcome="hit",
        )
        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0006",
            turn_index=6,
            output_text="Alice's strike lands.",
            per_player_renders={"alice": "Alice's strike lands."},
            beat_ended_reason="ruleset_resolution",
            dice_rolls=[roll],
        )

        engine = MagicMock()
        engine.load_latest.return_value = SimpleNamespace(
            session=SimpleNamespace(character_bindings={"alice": "42"}),
            characters=[SimpleNamespace(character_id="alice", name="Alice")],
        )

        inter = MagicMock()
        inter.channel_id = 123
        inter.channel = object()
        inter.user = MagicMock()
        inter.user.id = 42
        inter.client = MagicMock()
        inter.followup.send = AsyncMock()

        smap = MagicMock()
        thread = MagicMock()
        thread.id = 999
        events: list[str] = []

        async def _fake_rolls(**kwargs):
            events.append("rolls")
            assert kwargs["rolls"] == [roll]
            assert kwargs["character_id"] == "alice"
            return True

        async def _fake_post_actor_render(**kwargs):
            events.append("render")
            return ("thread", thread)

        monkeypatch.setattr(
            bot_commands, "_post_roll_displays_to_pov", _fake_rolls,
        )
        monkeypatch.setattr(
            bot_commands, "_post_actor_render", _fake_post_actor_render,
        )
        monkeypatch.setattr(
            bot_commands, "_clear_interaction_response", AsyncMock(),
        )

        asyncio.run(bot_commands._deliver_turn_response_to_povs(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id="s",
            story_id="story",
            actor_character_id="alice",
            actor_user=inter.user,
            response=response,
        ))

        assert events == ["rolls", "render"]

    def test_reaction_prompt_attaches_no_reaction_view(self, monkeypatch):
        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0004",
            turn_index=4,
            output_text="Bob can react.",
            per_player_renders={"bob": "Bob can react."},
            beat_ended_reason="combat_reaction_pending",
            reaction_prompts={"bob": "evt_react"},
        )

        engine = MagicMock()
        engine.load_latest.return_value = SimpleNamespace(
            session=SimpleNamespace(character_bindings={"bob": "42"}),
            characters=[SimpleNamespace(character_id="bob", name="Bob")],
        )

        inter = MagicMock()
        inter.channel_id = 123
        inter.channel = object()
        inter.user = MagicMock()
        inter.user.id = 42
        inter.client = MagicMock()
        inter.followup.send = AsyncMock()

        smap = MagicMock()
        captured = {}
        thread = MagicMock()
        thread.id = 999

        async def _fake_post_actor_render(**kwargs):
            captured.update(kwargs)
            return ("thread", thread)

        monkeypatch.setattr(
            bot_commands, "_post_actor_render", _fake_post_actor_render,
        )
        monkeypatch.setattr(
            bot_commands, "_clear_interaction_response", AsyncMock(),
        )

        asyncio.run(bot_commands._deliver_turn_response_to_povs(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id="s",
            story_id="story",
            actor_character_id="bob",
            actor_user=inter.user,
            response=response,
        ))

        assert isinstance(captured["view"], bot_commands._CombatReactionView)
        assert captured["view"].character_id == "bob"
        assert captured["view"].event_id == "evt_react"

    def test_actor_commitment_revision_prompt_adds_intro_note(self, monkeypatch):
        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0005",
            turn_index=5,
            output_text="The room changes around Alice.",
            per_player_renders={"alice": "The room changes around Alice."},
            beat_ended_reason="state_change",
            commitment_revision_prompts={"alice": ["commit_watch"]},
        )

        engine = MagicMock()
        engine.load_latest.return_value = SimpleNamespace(
            session=SimpleNamespace(character_bindings={"alice": "42"}),
            characters=[SimpleNamespace(character_id="alice", name="Alice")],
        )

        inter = MagicMock()
        inter.channel_id = 123
        inter.channel = object()
        inter.user = MagicMock()
        inter.user.id = 42
        inter.client = MagicMock()
        inter.followup.send = AsyncMock()

        smap = MagicMock()
        captured = {}
        thread = MagicMock()
        thread.id = 999

        async def _fake_post_actor_render(**kwargs):
            captured.update(kwargs)
            return ("thread", thread)

        monkeypatch.setattr(
            bot_commands, "_post_actor_render", _fake_post_actor_render,
        )
        monkeypatch.setattr(
            bot_commands, "_clear_interaction_response", AsyncMock(),
        )

        asyncio.run(bot_commands._deliver_turn_response_to_povs(
            inter=inter,
            smap=smap,
            engine=engine,
            session_id="s",
            story_id="story",
            actor_character_id="alice",
            actor_user=inter.user,
            response=response,
        ))

        assert "ongoing activity was interrupted" in captured["intro_content"]
        assert "/act (continue)" in captured["intro_content"]


# ---- v11-A5: sweep hook + purge wire-up --------------------------------------


class TestEngineBridgeSweepHook:
    """EngineBridge exposes a sweep-stale-pins hook the /act hot path can
    call before running the orchestrator. Invocation happens inside
    run_turn; tests here drive the primitive directly.
    """

    def test_sweep_resolves_expired_pins(self, mock_bridge):
        """When a session has a Cat II event older than the timeout, the
        sweep marks its stale responders as swept (via structured list)."""
        from datetime import datetime, timedelta, timezone

        from app.schemas.checkpoint import CheckpointFile
        from app.schemas.state import OpenCatIIEvent, SessionState, WorldState

        session_id = "swtest"
        ckpt = CheckpointFile(
            session=SessionState(
                session_id=session_id,
                character_bindings={"alice": "1"},
            ),
            world_state=WorldState(),
            characters=[],
        )
        ckpt.session.config.settings.cat_ii_human_timeout_seconds = 1
        evt = OpenCatIIEvent(
            event_id="evt_a",
            initiator_id="pip",
            initiator_intention="punch",
            required_responders=["alice"],
        )
        evt.opened_at = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).isoformat()
        ckpt.session.open_cat_ii_events.append(evt)

        mock_bridge.checkpoint_mgr.save(ckpt)

        swept = mock_bridge.sweep_stale_pins(session_id)
        assert "evt_a" in swept

        reloaded = mock_bridge.load_latest(session_id)
        evt_live = next(
            e for e in reloaded.session.open_cat_ii_events
            if e.event_id == "evt_a"
        )
        assert "alice" in evt_live.swept_responders

    def test_sweep_noop_when_nothing_stale(self, mock_bridge):
        """Sweep on a session with no open events returns [] and doesn't
        touch the checkpoint."""
        from app.schemas.checkpoint import CheckpointFile
        from app.schemas.state import SessionState, WorldState

        ckpt = CheckpointFile(
            session=SessionState(session_id="noop_sw"),
            world_state=WorldState(),
            characters=[],
        )
        mock_bridge.checkpoint_mgr.save(ckpt)
        assert mock_bridge.sweep_stale_pins("noop_sw") == []


class TestPurgeOnUnbind:
    """unbind_user must call purge_character_state so a /leave mid-beat
    doesn't strand slot pins, responder entries, or render buffers."""

    async def test_unbind_user_calls_purge(self, mock_bridge):
        from app.schemas.characters import CharacterRecord, PublicSheet
        from app.schemas.checkpoint import CheckpointFile
        from app.schemas.state import (
            OpenCatIIEvent,
            RenderBufferEntry,
            SessionState,
            SlotEntry,
            WorldState,
        )

        session_id = "purgetest"
        ckpt = CheckpointFile(
            session=SessionState(
                session_id=session_id,
                character_bindings={"bob": "77"},
            ),
            world_state=WorldState(),
            characters=[
                CharacterRecord(
                    character_id="bob",
                    name="Bob",
                    public_sheet=PublicSheet(role="player"),
                    location="gate",
                    is_playable=True,
                ),
            ],
        )
        evt = OpenCatIIEvent(
            event_id="evt_b",
            initiator_id="pip",
            initiator_intention="punch",
            required_responders=["bob"],
        )
        ckpt.session.open_cat_ii_events.append(evt)
        ckpt.session.active_act_slots["bob"] = SlotEntry(
            reason="cat_ii_responder",
            cat_ii_event_id="evt_b",
            claimed_at="",
        )
        ckpt.session.render_buffers["bob"] = [
            RenderBufferEntry(event_id="evt_prior", observation_level="direct"),
        ]
        mock_bridge.checkpoint_mgr.save(ckpt)

        freed = await mock_bridge.unbind_user(session_id, 77)
        assert freed == "bob"

        reloaded = mock_bridge.load_latest(session_id)
        # Binding gone.
        assert "bob" not in reloaded.session.character_bindings
        # Slot pin gone.
        assert "bob" not in reloaded.session.active_act_slots
        # Bob removed from the open event's required_responders.
        assert not any(
            "bob" in e.required_responders
            for e in reloaded.session.open_cat_ii_events
        )
        # Render buffer swept.
        assert "bob" not in reloaded.session.render_buffers


class TestApplyRosterUpdatesPurgesCulled:
    """Culling a character should purge their v11 slot/event state too."""

    def test_cull_triggers_purge(self):
        from app.engine.character_manager import CharacterManager
        from app.schemas.characters import CharacterRecord, PublicSheet
        from app.schemas.checkpoint import CheckpointFile
        from tests.support.factories import router_output
        from app.schemas.state import OpenCatIIEvent, SessionState, WorldState

        ckpt = CheckpointFile(
            session=SessionState(session_id="culltest"),
            world_state=WorldState(),
            characters=[
                CharacterRecord(
                    character_id="villain",
                    name="Villain",
                    public_sheet=PublicSheet(role="npc"),
                    location="gate",
                    is_playable=False,
                ),
            ],
        )
        ckpt.session.open_cat_ii_events.append(
            OpenCatIIEvent(
                event_id="evt_c",
                initiator_id="villain",
                initiator_intention="swing",
                required_responders=["hero"],
            )
        )

        routed = router_output(facts=[], cull=["villain"])
        mgr = CharacterManager()
        mgr.apply_roster_updates(ckpt, routed)
        # Event initiated by the culled character is abandoned.
        assert not any(
            e.initiator_id == "villain"
            for e in ckpt.session.open_cat_ii_events
        )


# ---- v11-r6b: sweep drives re-adjudication -----------------------------------


class TestSweepDrivesReadjudication:
    """EngineBridge.run_turn must close out Cat II events that sweep_
    stale_pins populated with AFK intentions BEFORE running the
    player's /act. Without this, a scene pinned on an AFK human sits
    open and every subsequent /act bounces off the pin."""

    def test_sweep_returns_event_ids_triggers_resolve_cat_ii(
        self, mock_bridge,
    ):
        """When sweep_stale_pins returns event ids, run_turn awaits
        orchestrator.resolve_cat_ii(session_id, event_id) for each
        before invoking process_turn."""
        from app.schemas.responses import TurnResponse

        # Stub sweep_stale_pins to return a single stale event id.
        mock_bridge.sweep_stale_pins = MagicMock(return_value=["evt_x"])
        # Mock both orchestrator entry points as AsyncMocks; the test
        # only cares about the call order + arguments.
        mock_bridge.orchestrator.resolve_cat_ii = AsyncMock(
            return_value=TurnResponse(
                session_id="session",
                beat_ended_reason="cat_ii_resolution",
            )
        )
        mock_bridge.orchestrator.process_turn = AsyncMock(
            return_value=TurnResponse(
                session_id="session",
                beat_ended_reason="response_requested",
            )
        )

        async def run():
            return await mock_bridge.run_turn(
                session_id="session",
                user_input="I look around",
                acting_character_id="alice",
            )

        result = asyncio.run(run())

        # resolve_cat_ii was awaited exactly once with the swept event id.
        mock_bridge.orchestrator.resolve_cat_ii.assert_awaited_once_with(
            "session", "evt_x",
        )
        # process_turn still ran after the re-adjudication completed;
        # the caller's /act should never be silently dropped.
        assert mock_bridge.orchestrator.process_turn.await_count == 1
        # run_turn returns the process_turn result, not resolve_cat_ii's.
        assert result.beat_ended_reason == "response_requested"

    def test_resolve_cat_ii_failure_does_not_block_current_act(
        self, mock_bridge,
    ):
        """If resolve_cat_ii raises, the /act still proceeds — one
        wedged stale event should never permanently block the session."""
        from app.schemas.responses import TurnResponse

        mock_bridge.sweep_stale_pins = MagicMock(return_value=["evt_bad"])
        mock_bridge.orchestrator.resolve_cat_ii = AsyncMock(
            side_effect=RuntimeError("boom"),
        )
        mock_bridge.orchestrator.process_turn = AsyncMock(
            return_value=TurnResponse(
                session_id="session",
                beat_ended_reason="response_requested",
            )
        )

        async def run():
            return await mock_bridge.run_turn(
                session_id="session",
                user_input="hi",
                acting_character_id="alice",
            )

        result = asyncio.run(run())
        mock_bridge.orchestrator.resolve_cat_ii.assert_awaited_once()
        assert mock_bridge.orchestrator.process_turn.await_count == 1
        assert result.beat_ended_reason == "response_requested"

    def test_process_turn_pre_resolutions_are_preserved(
        self, mock_bridge,
    ):
        """Orchestrator-level pre-turn work, such as resumed NPC combat,
        must survive the bridge's stale-pin pre-turn handling."""
        from app.schemas.responses import TurnResponse

        mock_bridge.sweep_stale_pins = MagicMock(return_value=[])
        npc_response = TurnResponse(
            session_id="session",
            checkpoint_id="ckpt_0005",
            turn_index=5,
            output_text="The rat lunges.",
            per_player_renders={"alice": "The rat lunges."},
            beat_ended_reason="response_requested",
        )
        mock_bridge.orchestrator.process_turn = AsyncMock(
            return_value=TurnResponse(
                session_id="session",
                beat_ended_reason="response_requested",
                pre_turn_resolutions=[npc_response],
            )
        )

        async def run():
            return await mock_bridge.run_turn(
                session_id="session",
                user_input="I strike back",
                acting_character_id="alice",
            )

        result = asyncio.run(run())

        assert result.pre_turn_resolutions == [npc_response]


# ---- /join directive choice happens INSIDE the per-session lock --------------


class TestRunArrivalTurnDirective:
    """`run_arrival_turn` is now `(arrive)`-only — the canonical opener
    moved to `/begin` / `run_begin_turn` in r9d. This class confirms
    `run_arrival_turn` always sends `(arrive)` regardless of session
    history, so callers don't have to remember which directive applies
    when."""

    def _seed_session(self, mock_bridge, narrator_conversations):
        """Build a checkpoint with `narrator_conversations` and stub
        `checkpoint_mgr.load_latest` to return it."""
        ckpt = CheckpointFile(
            session=SessionState(session_id="session"),
            world_state=WorldState(),
            narrator_conversations=narrator_conversations,
        )
        mock_bridge.checkpoint_mgr.load_latest = MagicMock(return_value=ckpt)
        return ckpt

    def _stub_orchestrator(self, mock_bridge):
        """Set up minimal stubs so `_run_turn_locked` can run end-to-
        end: a no-op sweep and a process_turn that returns a bare
        TurnResponse."""
        from app.schemas.responses import TurnResponse

        mock_bridge.sweep_stale_pins = MagicMock(return_value=[])
        mock_bridge.orchestrator.process_turn = AsyncMock(
            return_value=TurnResponse(
                session_id="session", beat_ended_reason="response_requested",
            )
        )

    def test_pristine_session_still_fires_arrive(self, mock_bridge):
        """`run_arrival_turn` is `(arrive)`-only by design — even
        on a pristine session it does NOT pick `(begin)`. The
        canonical opener now lives behind `run_begin_turn` /
        `/begin`; `run_arrival_turn` exists strictly for late
        joins. This test guards against the pre-r9d auto-`(begin)`
        regression."""
        self._seed_session(mock_bridge, narrator_conversations={})
        self._stub_orchestrator(mock_bridge)

        async def run():
            return await mock_bridge.run_arrival_turn(
                session_id="session", acting_character_id="alice",
            )

        response = asyncio.run(run())
        call_kwargs = mock_bridge.orchestrator.process_turn.call_args.args[0]
        assert call_kwargs.user_input == "(arrive)"
        assert call_kwargs.acting_character_id == "alice"
        assert response.beat_ended_reason == "response_requested"

    def test_session_with_prior_narrator_history_fires_arrive(
        self, mock_bridge,
    ):
        """A session whose narrator history is populated (story
        already opened) fires `(arrive)` — same as the empty case.
        The directive no longer depends on session state."""
        from app.schemas.conversation import ConversationMessage

        self._seed_session(
            mock_bridge,
            narrator_conversations={
                "first_player": [
                    ConversationMessage(role="user", content="(begin)"),
                    ConversationMessage(
                        role="assistant", content="The story opens…",
                    ),
                ],
            },
        )
        self._stub_orchestrator(mock_bridge)

        async def run():
            return await mock_bridge.run_arrival_turn(
                session_id="session", acting_character_id="alice",
            )

        asyncio.run(run())
        call_kwargs = mock_bridge.orchestrator.process_turn.call_args.args[0]
        assert call_kwargs.user_input == "(arrive)"

    def test_concurrent_arrivals_each_fire_arrive(self, mock_bridge):
        """Two `run_arrival_turn` calls in the same tick BOTH fire
        `(arrive)`. The pre-r9d race fix (one wins `(begin)`, the
        other gets `(arrive)`) is no longer relevant here because
        neither path ever touches `(begin)` — the per-session lock
        still serializes them, but only for orchestrator
        contention, not for directive selection."""
        from app.schemas.responses import TurnResponse

        ckpt = CheckpointFile(
            session=SessionState(session_id="session"),
            world_state=WorldState(),
            narrator_conversations={},
        )
        mock_bridge.checkpoint_mgr.load_latest = MagicMock(return_value=ckpt)
        mock_bridge.sweep_stale_pins = MagicMock(return_value=[])

        recorded_directives: list[str] = []

        async def fake_process_turn(req):
            recorded_directives.append(req.user_input)
            return TurnResponse(
                session_id="session",
                beat_ended_reason="response_requested",
            )

        mock_bridge.orchestrator.process_turn = AsyncMock(
            side_effect=fake_process_turn,
        )

        async def run():
            return await asyncio.gather(
                mock_bridge.run_arrival_turn(
                    session_id="session", acting_character_id="alice",
                ),
                mock_bridge.run_arrival_turn(
                    session_id="session", acting_character_id="bob",
                ),
            )

        asyncio.run(run())
        assert recorded_directives == ["(arrive)", "(arrive)"]


class TestRunBeginTurn:
    """`run_begin_turn` is the canonical opener: fires `(begin)` once,
    refuses to re-fire after the story has started, refuses if no
    players are bound, and picks the actor deterministically when the
    triggering binding is ambiguous (so two racing /begins converge
    on the same actor before the lock decides which one wins)."""

    def _seed_session(
        self, mock_bridge, *, bindings: dict[str, str],
        narrator_conversations: dict | None = None,
    ):
        """Build a checkpoint with the given bindings + narrator
        history and stub load_latest to return it."""
        ckpt = CheckpointFile(
            session=SessionState(
                session_id="session", character_bindings=bindings,
            ),
            world_state=WorldState(),
            narrator_conversations=narrator_conversations or {},
        )
        mock_bridge.checkpoint_mgr.load_latest = MagicMock(return_value=ckpt)
        return ckpt

    def _stub_orchestrator(self, mock_bridge):
        from app.schemas.responses import TurnResponse

        mock_bridge.sweep_stale_pins = MagicMock(return_value=[])
        mock_bridge.orchestrator.process_turn = AsyncMock(
            return_value=TurnResponse(
                session_id="session", beat_ended_reason="state_change",
            )
        )

    def test_pristine_with_bound_player_fires_begin(self, mock_bridge):
        """One bound player + empty narrator history = the canonical
        first-call shape. `(begin)` lands at the orchestrator with
        the triggering player as the actor."""
        self._seed_session(mock_bridge, bindings={"alice": "100"})
        self._stub_orchestrator(mock_bridge)

        async def run():
            return await mock_bridge.run_begin_turn(
                session_id="session",
                triggering_character_id="alice",
            )

        response = asyncio.run(run())
        call_args = mock_bridge.orchestrator.process_turn.call_args.args[0]
        assert call_args.user_input == "(begin)"
        assert call_args.acting_character_id == "alice"
        assert response.beat_ended_reason == "state_change"

    def test_no_bound_players_raises(self, mock_bridge):
        """`(begin)` without any bound players is meaningless — the
        router has no human POV to render for. Surface a ValueError
        so the bot command can give a friendly error instead of
        firing a ghost opening."""
        self._seed_session(mock_bridge, bindings={})
        self._stub_orchestrator(mock_bridge)

        async def run():
            return await mock_bridge.run_begin_turn(session_id="session")

        with pytest.raises(ValueError, match="no players are bound"):
            asyncio.run(run())
        mock_bridge.orchestrator.process_turn.assert_not_called()

    def test_already_started_raises(self, mock_bridge):
        """Once narrator history exists the story has already opened.
        A late `/begin` must NOT re-fire `(begin)` — that would
        clobber the prior opening prose. ValueError surfaces the
        misuse to the bot command."""
        from app.schemas.conversation import ConversationMessage

        self._seed_session(
            mock_bridge,
            bindings={"alice": "100"},
            narrator_conversations={
                "alice": [
                    ConversationMessage(role="user", content="(begin)"),
                    ConversationMessage(
                        role="assistant", content="The story opens…",
                    ),
                ],
            },
        )
        self._stub_orchestrator(mock_bridge)

        async def run():
            return await mock_bridge.run_begin_turn(
                session_id="session", triggering_character_id="alice",
            )

        with pytest.raises(ValueError, match="already started"):
            asyncio.run(run())
        mock_bridge.orchestrator.process_turn.assert_not_called()

    def test_unbound_triggering_id_falls_back_deterministically(
        self, mock_bridge,
    ):
        """If the triggering character_id isn't actually bound (admin
        firing /begin without a binding, or the player /leave'd
        between dispatch and lock), pick the lexicographically-first
        bound id. Deterministic so two racing calls converge on the
        same actor regardless of who reaches the lock first."""
        self._seed_session(
            mock_bridge,
            bindings={"pip": "200", "alice": "100", "rashid": "300"},
        )
        self._stub_orchestrator(mock_bridge)

        async def run():
            return await mock_bridge.run_begin_turn(
                session_id="session",
                triggering_character_id="ghost_admin_no_binding",
            )

        asyncio.run(run())
        call_args = mock_bridge.orchestrator.process_turn.call_args.args[0]
        assert call_args.acting_character_id == "alice"

    def test_empty_history_lists_do_not_count_as_started(self, mock_bridge):
        """`narrator_conversations` may contain empty lists left
        behind by `setdefault`. Those are NOT prior history; the
        story should still be openable."""
        self._seed_session(
            mock_bridge,
            bindings={"alice": "100"},
            narrator_conversations={"alice": [], "pip": []},
        )
        self._stub_orchestrator(mock_bridge)

        async def run():
            return await mock_bridge.run_begin_turn(
                session_id="session", triggering_character_id="alice",
            )

        asyncio.run(run())
        call_args = mock_bridge.orchestrator.process_turn.call_args.args[0]
        assert call_args.user_input == "(begin)"

    def test_concurrent_begins_one_wins_one_errors(self, mock_bridge):
        """Two `/begin`s racing through the lock: exactly ONE fires
        `(begin)`, the other observes the post-opening checkpoint
        and raises 'already started'. The lock + ValueError pair
        is the source-of-truth race fix."""
        from app.schemas.conversation import ConversationMessage
        from app.schemas.responses import TurnResponse

        ckpt = CheckpointFile(
            session=SessionState(
                session_id="session",
                character_bindings={"alice": "100", "bob": "200"},
            ),
            world_state=WorldState(),
            narrator_conversations={},
        )
        mock_bridge.checkpoint_mgr.load_latest = MagicMock(return_value=ckpt)
        mock_bridge.sweep_stale_pins = MagicMock(return_value=[])

        recorded_directives: list[str] = []

        async def fake_process_turn(req):
            recorded_directives.append(req.user_input)
            ckpt.narrator_conversations[req.acting_character_id] = [
                ConversationMessage(role="user", content="(begin)"),
                ConversationMessage(role="assistant", content="opening…"),
            ]
            return TurnResponse(
                session_id="session",
                beat_ended_reason="state_change",
            )

        mock_bridge.orchestrator.process_turn = AsyncMock(
            side_effect=fake_process_turn,
        )

        async def run():
            return await asyncio.gather(
                mock_bridge.run_begin_turn(
                    session_id="session",
                    triggering_character_id="alice",
                ),
                mock_bridge.run_begin_turn(
                    session_id="session",
                    triggering_character_id="bob",
                ),
                return_exceptions=True,
            )

        results = asyncio.run(run())

        # Exactly one (begin) hit the orchestrator.
        assert recorded_directives == ["(begin)"]
        # Exactly one of the gather results is a ValueError; the
        # other is a TurnResponse.
        errors = [r for r in results if isinstance(r, ValueError)]
        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(errors) == 1
        assert len(successes) == 1
        assert "already started" in str(errors[0])


# ---- /join Discord option text ----------------------------------------------


class TestJoinOptionText:
    def test_join_select_label_prefers_role_for_unnamed_character(self):
        label = bot_commands._join_select_label(
            "",
            "player_protagonist",
            role="the kingdom's defective summon",
        )

        assert label == "the kingdom's defective summon"

    def test_join_select_label_falls_back_to_character_id(self):
        label = bot_commands._join_select_label("   ", "player_protagonist")

        assert label == "player_protagonist"

    def test_join_select_label_stays_within_discord_limit(self):
        label = bot_commands._join_select_label("x" * 150, "hero")

        assert 1 <= len(label) <= bot_commands.DISCORD_SELECT_OPTION_TEXT_MAX


# ---- briefing copy: /describe demoted, /join is the canonical opener ---


class TestBriefingCopy:
    """The /story start briefing used to point players at /describe as the
    next-step command. Under the new /join flow `/describe` is an
    advanced-only mid-game tweak, and the briefing should funnel
    everyone through /join (which also exposes custom-create)."""

    def _ckpt_with_primer(self, primer: str) -> CheckpointFile:
        return CheckpointFile(
            session=SessionState(session_id="briefing_test"),
            world_state=WorldState(),
            player_primer=primer,
        )

    def _embed_text(self, embed) -> str:
        """Concatenate description + every field value so the assertion
        catches mentions wherever the renderer puts them."""
        parts = [embed.description or ""]
        for field in embed.fields:
            parts.append(field.value or "")
        return "\n".join(parts)

    def test_briefing_does_not_mention_describe(self):
        from app.bot.embed import render_briefing
        ckpt = self._ckpt_with_primer(
            "You wake up in a sun-drenched villa. You don't remember "
            "the cameras or the roses. You suspect both are imminent."
        )
        embed = render_briefing(ckpt, story_id="dating_villa_s1")

        text = self._embed_text(embed)
        assert "/describe" not in text, text
        # Sanity: still funnels players to the canonical entry command.
        assert "/join" in text, text

    def test_briefing_falls_back_to_stub_without_describe(self):
        """Pre-v8 / hand-built checkpoints with no primer also must not
        leak a /describe mention via the fallback copy."""
        from app.bot.embed import render_briefing
        ckpt = CheckpointFile(
            session=SessionState(session_id="briefing_fallback"),
            world_state=WorldState(),
            player_primer="",
        )
        embed = render_briefing(ckpt, story_id="legacy_story")
        text = self._embed_text(embed)
        assert "/describe" not in text, text
        assert "/join" in text, text

    def test_briefing_title_is_discord_safe_for_long_genre(self):
        from app.bot.embed import MAX_TITLE, render_briefing
        ckpt = CheckpointFile(
            session=SessionState(session_id="briefing_long_title"),
            world_state=WorldState(
                setting=StorySetting(
                    genre="Mature isekai dark fantasy with romance, "
                    "political intrigue, and a three-layer hidden conspiracy. "
                    * 8,
                ),
            ),
            player_primer="You wake up somewhere strange after the truck.",
        )

        embed = render_briefing(ckpt, story_id="long_story")

        assert embed.title is not None
        assert len(embed.title) <= MAX_TITLE


# ---- _post_actor_render: thread → DM → public cascade -----------------


class TestPostActorRenderCascade:
    """The actor's narrative is delivered POV-thread-first, with DM and
    public-channel fallbacks. This unifies solo and multi-player UX —
    every bound human reads their beat in a private thread; the public
    channel is lobby/acks only. Tests stub the discord layer at the
    helper boundary (`_session_text_channel`, `_ensure_pov_thread`)
    so we can drive every venue branch deterministically without a
    full discord.py harness."""

    def _make_env(
        self, monkeypatch, tmp_path: Path,
        *, thread_send_behavior, dm_succeeds: bool,
    ):
        """Patch the two discord-touching helpers used by
        `_post_actor_render` and return (inter, user, smap, embeds,
        captured) so the test can drive every venue branch.

        `thread_send_behavior` is one of:
          * `None` — `_ensure_pov_thread` returns None (no thread).
          * `"ok"` — thread exists; `.send` succeeds and captures.
          * `"raise"` — thread exists; `.send` raises RuntimeError.
        """
        from app.bot import commands as bot_commands

        monkeypatch.setattr(
            bot_commands, "_session_text_channel",
            lambda inter: object(),  # non-None sentinel
        )

        captured: dict = {"thread_sends": [], "dm_sends": []}
        thread_obj = None
        if thread_send_behavior is not None:
            thread_obj = MagicMock()
            thread_obj.id = 999
            thread_obj.mention = "<#999>"
            if thread_send_behavior == "ok":
                async def _thread_send(*args, **kwargs):
                    captured["thread_sends"].append((args, kwargs))
                thread_obj.send = AsyncMock(side_effect=_thread_send)
            elif thread_send_behavior == "raise":
                thread_obj.send = AsyncMock(
                    side_effect=RuntimeError("server hates us"),
                )
            else:
                raise ValueError(
                    f"unknown thread_send_behavior: {thread_send_behavior!r}"
                )

        async def _ensure(**kwargs):
            return thread_obj

        monkeypatch.setattr(
            bot_commands, "_ensure_pov_thread", _ensure,
        )

        async def _user_send(*args, **kwargs):
            captured["dm_sends"].append((args, kwargs))
            if not dm_succeeds:
                raise RuntimeError("simulated DM failure")

        user = MagicMock()
        user.id = 42
        user.send = AsyncMock(side_effect=_user_send)

        inter = MagicMock()
        inter.channel = MagicMock()
        inter.channel.id = 777
        inter.channel_id = 777
        inter.user = user

        smap = MagicMock()
        smap.clear_pov_thread = AsyncMock()
        smap.record_turn_message = AsyncMock()

        import discord as _discord
        embeds = [MagicMock(spec=_discord.Embed)]
        return inter, user, smap, embeds, captured, thread_obj

    def test_thread_success_returns_thread_and_skips_dm(
        self, monkeypatch, tmp_path: Path,
    ):
        """Happy path — thread.send works; DM is never attempted."""
        from app.bot.commands import _post_actor_render

        inter, user, smap, embeds, captured, thread = self._make_env(
            monkeypatch, tmp_path,
            thread_send_behavior="ok", dm_succeeds=True,
        )

        venue, returned_thread = asyncio.run(_post_actor_render(
            inter=inter, smap=smap, user=user,
            character_id="alice", char_name="Alice",
            embeds=embeds, intro_content="**Alice** acted.",
        ))
        assert venue == "thread"
        assert returned_thread is thread
        assert len(captured["thread_sends"]) == 1
        assert captured["dm_sends"] == []
        _, kwargs = captured["thread_sends"][0]
        assert kwargs.get("content") == "**Alice** acted."
        assert kwargs.get("embeds") is embeds

    def test_thread_send_failure_falls_back_to_dm(
        self, monkeypatch, tmp_path: Path,
    ):
        """thread.send raising → cached id is cleared and DM is tried."""
        from app.bot.commands import _post_actor_render

        inter, user, smap, embeds, captured, _ = self._make_env(
            monkeypatch, tmp_path,
            thread_send_behavior="raise", dm_succeeds=True,
        )

        venue, returned_thread = asyncio.run(_post_actor_render(
            inter=inter, smap=smap, user=user,
            character_id="alice", char_name="Alice",
            embeds=embeds, intro_content="x",
        ))
        assert venue == "dm"
        assert returned_thread is None
        assert len(captured["dm_sends"]) == 1

    def test_no_thread_available_uses_dm(
        self, monkeypatch, tmp_path: Path,
    ):
        """`_ensure_pov_thread` returning None (no perms etc.) → DM only."""
        from app.bot.commands import _post_actor_render

        inter, user, smap, embeds, captured, _ = self._make_env(
            monkeypatch, tmp_path,
            thread_send_behavior=None, dm_succeeds=True,
        )
        venue, returned_thread = asyncio.run(_post_actor_render(
            inter=inter, smap=smap, user=user,
            character_id="alice", char_name="Alice",
            embeds=embeds,
        ))
        assert venue == "dm"
        assert returned_thread is None
        assert len(captured["dm_sends"]) == 1

    def test_both_paths_fail_returns_none(
        self, monkeypatch, tmp_path: Path,
    ):
        """Neither thread nor DM works → caller must fall back to public."""
        from app.bot.commands import _post_actor_render

        inter, user, smap, embeds, captured, _ = self._make_env(
            monkeypatch, tmp_path,
            thread_send_behavior=None, dm_succeeds=False,
        )
        venue, returned_thread = asyncio.run(_post_actor_render(
            inter=inter, smap=smap, user=user,
            character_id="alice", char_name="Alice",
            embeds=embeds,
        ))
        assert venue == "none"
        assert returned_thread is None


class TestVisualNovelDiscordDeck:
    def _deck(self, tmp_path: Path):
        renderer = VisualNovelCardRenderer(tmp_path / "vn")
        return renderer.render_deck(
            [VisualNovelDeckSection(pages=(
                VisualNovelPage(
                    kind="narration",
                    text="The terrace opens beneath a clear turquoise sky.",
                ),
                VisualNovelPage(
                    kind="dialogue",
                    speaker="Iselle",
                    text="You made it. Wren was beginning to worry.",
                ),
            ))],
        )

    def test_restart_safe_controls_encode_complete_navigation_state(
        self, tmp_path: Path,
    ):
        deck = self._deck(tmp_path)
        view = bot_commands._VisualNovelView(
            deck_id=deck.deck_id,
            user_id=42,
            index=0,
            count=len(deck.cards),
        )

        assert view.timeout is None
        assert len(view.children) == 3
        controls = {
            child.action: child
            for child in view.children
            if isinstance(child, bot_commands._VisualNovelControl)
        }
        assert set(controls) == {"p", "n", "t"}
        assert controls["p"].item.disabled is True
        assert controls["n"].item.disabled is False
        for control in controls.values():
            custom_id = control.item.custom_id
            assert custom_id is not None
            assert len(custom_id) <= 100
            assert custom_id.startswith(f"avn:{deck.deck_id}:42:0:")

    def test_next_control_reloads_deck_and_edits_one_attachment(
        self, tmp_path: Path,
    ):
        deck = self._deck(tmp_path)
        engine = SimpleNamespace(
            load_visual_novel_deck=MagicMock(return_value=deck),
        )
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=42),
            client=SimpleNamespace(_ayoa_visual_novel_engine=engine),
            response=SimpleNamespace(
                edit_message=AsyncMock(),
                send_message=AsyncMock(),
            ),
            followup=SimpleNamespace(send=AsyncMock()),
        )
        control = bot_commands._VisualNovelControl(
            deck_id=deck.deck_id,
            user_id=42,
            index=0,
            action="n",
        )

        asyncio.run(control.callback(interaction))

        engine.load_visual_novel_deck.assert_called_once_with(deck.deck_id)
        interaction.response.edit_message.assert_awaited_once()
        kwargs = interaction.response.edit_message.await_args.kwargs
        assert len(kwargs["attachments"]) == 1
        assert kwargs["attachments"][0].filename.endswith("-002.png")
        assert kwargs["attachments"][0].description == (
            "Visual novel story page 2 of 2. "
            "Iselle: You made it. Wren was beginning to worry."
        )
        assert isinstance(kwargs["view"], bot_commands._VisualNovelView)
        assert any(
            isinstance(child, bot_commands._VisualNovelControl)
            and child.index == 1
            for child in kwargs["view"].children
        )
        kwargs["attachments"][0].close()

    def test_private_card_posts_to_thread_and_records_restart_safe_delivery(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        deck = self._deck(tmp_path)
        engine = SimpleNamespace(
            prepare_visual_novel_deck=AsyncMock(return_value=deck),
        )
        sent_message = SimpleNamespace(id=1234)
        thread = SimpleNamespace(
            id=999,
            send=AsyncMock(return_value=sent_message),
        )
        monkeypatch.setattr(
            bot_commands,
            "_session_text_channel",
            lambda _interaction: object(),
        )
        monkeypatch.setattr(
            bot_commands,
            "_ensure_pov_thread",
            AsyncMock(return_value=thread),
        )
        record = AsyncMock()
        monkeypatch.setattr(bot_commands, "_record_turn_message", record)
        user = SimpleNamespace(id=42, send=AsyncMock())
        interaction = SimpleNamespace(
            channel=object(),
            channel_id=777,
        )
        render = VisualNovelRender(segments=[VisualNovelRenderSegment(
            pages=[VisualNovelPage(
                kind="dialogue", speaker="Iselle", text="Hello."
            )],
            rendered_event_ids=["evt_1"],
        )])

        venue, returned_thread = asyncio.run(
            bot_commands._post_visual_novel_render(
                inter=interaction,
                smap=MagicMock(),
                engine=engine,
                user=user,
                character_id="iselle",
                char_name="Iselle",
                render=render,
                intro_content=None,
                session_id="session",
                turn_index=3,
            )
        )

        assert venue == "thread"
        assert returned_thread is thread
        engine.prepare_visual_novel_deck.assert_awaited_once_with(
            session_id="session",
            pov_character_id="iselle",
            render=render,
        )
        thread.send.assert_awaited_once()
        send_kwargs = thread.send.await_args.kwargs
        assert isinstance(send_kwargs["file"], bot_commands.discord.File)
        assert send_kwargs["file"].description == (
            "Visual novel story page 1 of 2. Narration: "
            "The terrace opens beneath a clear turquoise sky."
        )
        assert isinstance(send_kwargs["view"], bot_commands._VisualNovelView)
        user.send.assert_not_awaited()
        record.assert_awaited_once()
        assert record.await_args.kwargs["delivery"] == "thread_visual_novel"
        send_kwargs["file"].close()

    def test_attachment_description_truncates_current_page_at_discord_limit(
        self,
        tmp_path: Path,
    ):
        deck = self._deck(tmp_path)
        card = VisualNovelCard(
            index=1,
            count=1,
            kind="narration",
            speaker="",
            text="  ".join(["wind moves over the terrace"] * 100),
            image_path=deck.cards[0].image_path,
        )

        description = bot_commands._visual_novel_discord_description(card)

        assert len(description) <= 1024
        assert description.startswith(
            "Visual novel story page 1 of 1. Narration: wind moves"
        )
        assert description.endswith("…")
        assert "  " not in description

    def test_thread_tracking_exception_does_not_retry_delivery(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        deck = self._deck(tmp_path)
        engine = SimpleNamespace(
            prepare_visual_novel_deck=AsyncMock(return_value=deck),
        )
        sent_message = SimpleNamespace(
            id=1234,
            channel=SimpleNamespace(id=999),
        )
        thread = SimpleNamespace(
            id=999,
            send=AsyncMock(return_value=sent_message),
        )
        monkeypatch.setattr(
            bot_commands,
            "_session_text_channel",
            lambda _interaction: object(),
        )
        monkeypatch.setattr(
            bot_commands,
            "_ensure_pov_thread",
            AsyncMock(return_value=thread),
        )
        monkeypatch.setattr(
            bot_commands,
            "_record_turn_message",
            AsyncMock(side_effect=RuntimeError("tracking unavailable")),
        )
        user = SimpleNamespace(id=42, send=AsyncMock())
        interaction = SimpleNamespace(channel_id=777, channel=object())
        render = VisualNovelRender(segments=[VisualNovelRenderSegment(
            pages=[VisualNovelPage(
                kind="dialogue", speaker="Iselle", text="Hello."
            )],
            rendered_event_ids=["evt_1"],
        )])

        with caplog.at_level(logging.ERROR, logger="app.bot.commands"):
            venue, returned_thread = asyncio.run(
                bot_commands._post_visual_novel_render(
                    inter=interaction,
                    smap=MagicMock(),
                    engine=engine,
                    user=user,
                    character_id="iselle",
                    char_name="Iselle",
                    render=render,
                    session_id="session",
                    turn_index=3,
                )
            )

        assert (venue, returned_thread) == ("thread", thread)
        thread.send.assert_awaited_once()
        user.send.assert_not_awaited()
        assert "message sent but tracking raised" in caplog.text
        assert "session=session" in caplog.text
        assert "Hello" not in caplog.text
        thread.send.await_args.kwargs["file"].close()

    def test_dm_tracking_exception_does_not_trigger_prose_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        deck = self._deck(tmp_path)
        engine = SimpleNamespace(
            prepare_visual_novel_deck=AsyncMock(return_value=deck),
        )
        monkeypatch.setattr(
            bot_commands,
            "_session_text_channel",
            lambda _interaction: None,
        )
        monkeypatch.setattr(
            bot_commands,
            "_record_turn_message",
            AsyncMock(side_effect=RuntimeError("tracking unavailable")),
        )
        sent_message = SimpleNamespace(
            id=4321,
            channel=SimpleNamespace(id=555),
        )
        user = SimpleNamespace(
            id=42,
            send=AsyncMock(return_value=sent_message),
        )
        interaction = SimpleNamespace(channel_id=777, channel=object())
        render = VisualNovelRender(segments=[VisualNovelRenderSegment(
            pages=[VisualNovelPage(
                kind="dialogue", speaker="Iselle", text="Hello."
            )],
            rendered_event_ids=["evt_1"],
        )])

        with caplog.at_level(logging.ERROR, logger="app.bot.commands"):
            venue, returned_thread = asyncio.run(
                bot_commands._post_visual_novel_render(
                    inter=interaction,
                    smap=MagicMock(),
                    engine=engine,
                    user=user,
                    character_id="iselle",
                    char_name="Iselle",
                    render=render,
                    session_id="session",
                    turn_index=3,
                )
            )

        assert (venue, returned_thread) == ("dm", None)
        user.send.assert_awaited_once()
        assert "message sent but tracking raised" in caplog.text
        assert "session=session" in caplog.text
        assert "Hello" not in caplog.text
        user.send.await_args.kwargs["file"].close()

    def test_history_projection_reads_structured_pages(self):
        content = json.dumps({
            "pages": [
                {"kind": "narration", "speaker": "", "text": "Wind stirs."},
                {"kind": "dialogue", "speaker": "Wren", "text": "Ready?"},
            ],
        })

        assert _narrator_history_message_text(content) == (
            "Wind stirs.\n\nWren: Ready?"
        )


class TestVisualNovelJoinArrival:
    def test_failed_visual_arrival_uses_prose_and_fans_out_bystander(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        class FakeTree:
            def __init__(self):
                self.commands = {}

            def command(self, *, name, **_kwargs):
                def _decorator(fn):
                    self.commands[name] = fn
                    return fn

                return _decorator

            def add_command(self, *_args, **_kwargs):
                return None

        actor_render = VisualNovelRender(segments=[VisualNovelRenderSegment(
            pages=[VisualNovelPage(
                kind="narration",
                text="Alice steps into the lantern light.",
            )],
            rendered_event_ids=["evt_arrive"],
        )])
        bystander_render = VisualNovelRender(segments=[VisualNovelRenderSegment(
            pages=[VisualNovelPage(
                kind="narration",
                text="Bob sees Alice arrive.",
            )],
            rendered_event_ids=["evt_arrive"],
        )])
        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0004",
            turn_index=4,
            output_text="Alice steps into the lantern light.",
            per_player_renders={
                "alice": "Alice steps into the lantern light.",
                "bob": "Bob sees Alice arrive.",
            },
            per_player_visual_novel_renders={
                "alice": actor_render,
                "bob": bystander_render,
            },
        )
        engine = MagicMock()
        engine.get_user_binding.return_value = None
        engine.list_joinable_characters.return_value = [CharacterSummary(
            character_id="alice",
            name="Alice",
            role="traveler",
            faction="",
            appearance="dusty coat",
            status="active",
            is_playable=True,
        )]
        engine.join_player_character = AsyncMock(return_value=PlayerJoinResult(
            character_id="alice",
            character_name="Alice",
            pre_play=False,
            response=response,
        ))
        engine.load_latest.return_value = SimpleNamespace(
            session=SimpleNamespace(
                character_bindings={"alice": "42", "bob": "99"},
            ),
            characters=[
                SimpleNamespace(character_id="alice", name="Alice"),
                SimpleNamespace(character_id="bob", name="Bob"),
            ],
        )
        smap = MagicMock()
        smap.get = AsyncMock(return_value=SimpleNamespace(
            session_id="s",
            story_id="story",
        ))
        visual_post = AsyncMock(side_effect=RuntimeError("card failed"))
        actor_post = AsyncMock(return_value=(
            "thread",
            SimpleNamespace(mention="#alice-pov"),
        ))
        prose_fanout = AsyncMock(return_value=True)
        monkeypatch.setattr(
            bot_commands,
            "_post_visual_novel_render",
            visual_post,
        )
        monkeypatch.setattr(bot_commands, "_post_actor_render", actor_post)
        monkeypatch.setattr(bot_commands, "_post_to_pov", prose_fanout)

        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)
        join_inter = MagicMock()
        join_inter.channel_id = 777
        join_inter.user = SimpleNamespace(id=42)
        join_inter.response.send_message = AsyncMock()
        asyncio.run(tree.commands["join"](join_inter))
        picker = join_inter.response.send_message.await_args.kwargs["view"]

        picker._select._values = ["alice"]
        pick_inter = MagicMock()
        pick_inter.user = SimpleNamespace(id=42)
        pick_inter.response.send_modal = AsyncMock()
        asyncio.run(picker._on_pick(pick_inter))
        modal = pick_inter.response.send_modal.await_args.args[0]
        modal.name_in._value = ""
        modal.appearance_in._value = ""

        bob_user = SimpleNamespace(id=99, send=AsyncMock())
        modal_inter = MagicMock()
        modal_inter.channel_id = 777
        modal_inter.channel = MagicMock()
        modal_inter.user = SimpleNamespace(id=42)
        modal_inter.client.get_user.return_value = bob_user
        modal_inter.client.fetch_user = AsyncMock()
        modal_inter.response.defer = AsyncMock()
        modal_inter.followup.send = AsyncMock()

        asyncio.run(modal.on_submit(modal_inter))

        engine.join_player_character.assert_awaited_once_with(
            "s",
            "alice",
            42,
            name="",
            appearance="",
        )
        assert visual_post.await_count == 2
        actor_post.assert_awaited_once()
        assert actor_post.await_args.kwargs["embeds"][0].description == (
            "Alice steps into the lantern light."
        )
        prose_fanout.assert_awaited_once()
        assert prose_fanout.await_args.kwargs["user_id"] == 99
        assert prose_fanout.await_args.kwargs["text"] == (
            "Bob sees Alice arrive."
        )
        modal_inter.channel.send.assert_not_called()


def _asset_payload(**overrides) -> SafeAssetRevealPayload:
    values = {
        "pack_id": "synthetic",
        "asset_id": "map-room",
        "kind": "map",
        "title": "Safe Map Title",
        "mime_type": "image/png",
        "width": 64,
        "height": 64,
        "sha256": "a" * 64,
        "delivery_ref": "asset://synthetic/map-room",
        "presentation": "attachment",
        "caption": "A safe caption.",
        "alt_text": "Safe alt text.",
    }
    values.update(overrides)
    return SafeAssetRevealPayload(**values)


class TestPostAssetsToPov:
    def _resolved_asset(self):
        return bot_commands.ResolvedAssetBytes(
            pack_id="synthetic",
            asset_id="map-room",
            delivery_ref="asset://synthetic/map-room",
            filename="asset-aaaaaaaaaaaaaaaa.png",
            mime_type="image/png",
            data=b"\x89PNG\r\n",
            sha256="a" * 64,
            byte_count=6,
            width=64,
            height=64,
        )

    def _env(self, monkeypatch):
        channel = MagicMock()
        channel.id = 777
        channel.send = AsyncMock()
        monkeypatch.setattr(
            bot_commands,
            "_session_text_channel",
            lambda inter: channel,
        )

        inter = MagicMock()
        inter.channel = channel
        inter.channel_id = 777
        inter.user = MagicMock()
        inter.user.id = 42
        inter.followup.send = AsyncMock()

        user = MagicMock()
        user.id = 42
        user.send = AsyncMock()

        bot = MagicMock()
        bot.get_user.return_value = user
        bot.fetch_user = AsyncMock()

        smap = MagicMock()
        smap.record_turn_message = AsyncMock()
        smap.forget_turn_messages = AsyncMock()
        smap.clear_pov_thread = AsyncMock()

        engine = MagicMock()
        return inter, channel, user, bot, smap, engine

    def test_thread_asset_success_records_rewind_message(self, monkeypatch):
        inter, _channel, user, bot, smap, engine = self._env(monkeypatch)
        thread = MagicMock()
        thread.id = 999
        thread_msg = SimpleNamespace(id=555, channel=SimpleNamespace(id=999))
        thread.send = AsyncMock(return_value=thread_msg)

        async def _ensure(**_kwargs):
            return thread

        monkeypatch.setattr(bot_commands, "_ensure_pov_thread", _ensure)
        monkeypatch.setattr(
            bot_commands,
            "_resolve_safe_discord_asset",
            lambda *_args, **_kwargs: self._resolved_asset(),
        )

        ok = asyncio.run(bot_commands._post_assets_to_pov(
            inter=inter,
            smap=smap,
            user_id=42,
            character_id="alice",
            char_name="Alice",
            asset_reveals=[_asset_payload()],
            bot=bot,
            engine=engine,
            session_id="s",
            turn_index=7,
            catalog={},
        ))

        assert ok is True
        thread.send.assert_awaited_once()
        _, kwargs = thread.send.await_args
        assert kwargs["content"] == "A safe caption."
        assert kwargs["file"].filename == "asset-aaaaaaaaaaaaaaaa.png"
        assert kwargs["file"].description == "Safe alt text."
        user.send.assert_not_awaited()
        smap.record_turn_message.assert_awaited_once()
        record_kwargs = smap.record_turn_message.await_args.kwargs
        assert record_kwargs["channel_id"] == 777
        assert record_kwargs["session_id"] == "s"
        assert record_kwargs["turn_index"] == 7
        assert record_kwargs["discord_channel_id"] == 999
        assert record_kwargs["message_id"] == 555
        assert record_kwargs["delivery"] == "thread_asset"
        assert record_kwargs["recipient_user_id"] == 42

    def test_private_asset_failure_logs_no_asset_source_sentinels(
        self, monkeypatch, caplog,
    ):
        inter, channel, user, bot, smap, engine = self._env(monkeypatch)
        thread = MagicMock()
        thread.id = 999
        thread.send = AsyncMock(side_effect=RuntimeError(
            "upload failed /private/table/source-map.png "
            "delivery_ref=asset://synthetic/hidden-map"
        ))
        user.send = AsyncMock(side_effect=RuntimeError(
            "dm failed raw_ocr=PROTECTED_SOURCE_EXCERPT"
        ))

        async def _ensure(**_kwargs):
            return thread

        monkeypatch.setattr(bot_commands, "_ensure_pov_thread", _ensure)
        monkeypatch.setattr(
            bot_commands,
            "_resolve_safe_discord_asset",
            lambda *_args, **_kwargs: self._resolved_asset(),
        )

        with caplog.at_level("WARNING", logger="app.bot.commands"):
            ok = asyncio.run(bot_commands._post_assets_to_pov(
                inter=inter,
                smap=smap,
                user_id=42,
                character_id="alice",
                char_name="Alice",
                asset_reveals=[_asset_payload(
                    caption="Spoiler-safe but still private.",
                    title="Private title",
                    asset_id="hidden-map",
                    delivery_ref="asset://synthetic/hidden-map",
                )],
                bot=bot,
                engine=engine,
                session_id="s",
                turn_index=8,
                catalog={},
            ))

        assert ok is False
        for sentinel in (
            "/private/table/source-map.png",
            "delivery_ref=asset://synthetic/hidden-map",
            "raw_ocr=PROTECTED_SOURCE_EXCERPT",
            "Private title",
            "Spoiler-safe",
        ):
            assert sentinel not in caplog.text
        channel.send.assert_not_awaited()
        inter.followup.send.assert_awaited_once()
        _, notice_kwargs = inter.followup.send.await_args
        notice_text = inter.followup.send.await_args.args[0]
        assert notice_kwargs["ephemeral"] is True
        assert "withheld" in notice_text
        assert "hidden-map" not in notice_text
        assert "Private title" not in notice_text
        assert "Spoiler-safe" not in notice_text

    def test_generated_media_uses_private_delivery_and_rewind_tracking(
        self, monkeypatch,
    ):
        from app.bot.media_delivery import deliver_player_media
        from app.engine.player_media import ResolvedPlayerMedia

        _inter, channel, user, bot, smap, _engine = self._env(monkeypatch)
        thread = MagicMock()
        thread.id = 999
        message = SimpleNamespace(id=556, channel=SimpleNamespace(id=999))
        thread.send = AsyncMock(return_value=message)
        media = ResolvedPlayerMedia(
            filename="illustration-aaaaaaaaaaaaaaaa.webp",
            mime_type="image/webp",
            data=b"generated",
            sha256="a" * 64,
            byte_count=9,
            width=1024,
            height=1024,
        )

        delivered = asyncio.run(deliver_player_media(
            client=bot,
            smap=smap,
            session_channel_id=777,
            user_id=42,
            character_id="alice",
            char_name="Alice",
            media=media,
            caption="AI-generated, noncanonical illustration · Turn 8",
            alt_text="A noncanonical illustration.",
            session_id="s",
            turn_index=8,
            parent_channel=channel,
            delivery_label="generated_image",
            preferred_thread=thread,
            resolve_thread=False,
        ))

        assert delivered is True
        thread.send.assert_awaited_once()
        user.send.assert_not_awaited()
        record = smap.record_turn_message.await_args.kwargs
        assert record["delivery"] == "thread_generated_image"
        assert record["turn_index"] == 8
        channel.send.assert_not_awaited()

    def test_generated_media_removes_attachment_if_rewind_wins_send_race(
        self, monkeypatch,
    ):
        from app.bot.media_delivery import deliver_player_media
        from app.engine.player_media import ResolvedPlayerMedia

        _inter, channel, _user, bot, smap, _engine = self._env(monkeypatch)
        thread = MagicMock()
        thread.id = 999
        message = MagicMock()
        message.id = 557
        message.channel.id = 999
        message.delete = AsyncMock()
        thread.send = AsyncMock(return_value=message)
        checks = iter((True, True, False))

        async def _is_current():
            return next(checks)

        delivered = asyncio.run(deliver_player_media(
            client=bot,
            smap=smap,
            session_channel_id=777,
            user_id=42,
            character_id="alice",
            char_name="Alice",
            media=ResolvedPlayerMedia(
                filename="illustration.webp",
                mime_type="image/webp",
                data=b"generated",
                sha256="a" * 64,
                byte_count=9,
                width=1024,
                height=1024,
            ),
            caption="Noncanonical illustration.",
            alt_text="Noncanonical illustration.",
            session_id="s",
            turn_index=8,
            parent_channel=channel,
            delivery_label="generated_image",
            preferred_thread=thread,
            resolve_thread=False,
            delivery_is_current=_is_current,
        ))

        assert delivered is False
        message.delete.assert_awaited_once()
        smap.record_turn_message.assert_awaited_once()
        smap.forget_turn_messages.assert_awaited_once()

    def test_durable_media_cleanup_outbox_retries_and_forgets_ref(self, tmp_path):
        from app.bot.media_delivery import retry_media_cleanup_outbox
        from app.bot.session_map import SessionMap, TurnMessageRef

        smap = SessionMap(tmp_path / "sessionmap.db")
        ref = TurnMessageRef(
            channel_id=777,
            session_id="s",
            turn_index=8,
            discord_channel_id=999,
            message_id=558,
            delivery="thread_generated_image_img_test",
            recipient_user_id=42,
            created_at=0,
        )
        message = MagicMock()
        message.delete = AsyncMock()
        channel = MagicMock()
        channel.fetch_message = AsyncMock(return_value=message)
        client = MagicMock()
        client.get_channel.return_value = channel

        async def _run():
            await smap.init()
            await smap.record_turn_message(
                channel_id=ref.channel_id,
                session_id=ref.session_id,
                turn_index=ref.turn_index,
                discord_channel_id=ref.discord_channel_id,
                message_id=ref.message_id,
                delivery=ref.delivery,
                recipient_user_id=ref.recipient_user_id,
            )
            await smap.record_media_cleanup(ref)
            handled = await retry_media_cleanup_outbox(
                client=client,
                smap=smap,
            )
            return (
                handled,
                await smap.list_media_cleanup(),
                await smap.list_turn_messages(
                    channel_id=777,
                    session_id="s",
                    turns=[8],
                ),
            )

        handled, outbox, tracked = asyncio.run(_run())
        assert handled == 1
        assert outbox == []
        assert tracked == []
        message.delete.assert_awaited_once()


# ---- rewind Discord message cleanup ---------------------------------------


class _FakeDiscordMessage:
    def __init__(
        self,
        message_id: int,
        *,
        delete_raises: bool = False,
        delete_error: Exception | None = None,
        supports_attachments: bool = True,
    ):
        self.id = message_id
        self.delete_raises = delete_raises
        self.delete_error = delete_error
        self.supports_attachments = supports_attachments
        self.deleted = False
        self.edited_content = None
        self.edited_embeds = None
        self.edited_attachments = None

    async def delete(self):
        if self.delete_error is not None:
            raise self.delete_error
        if self.delete_raises:
            raise RuntimeError("delete denied")
        self.deleted = True

    async def edit(self, *, content=None, embeds=None, attachments=None):
        if attachments is not None and not self.supports_attachments:
            raise TypeError("unexpected keyword argument 'attachments'")
        self.edited_content = content
        self.edited_embeds = embeds
        self.edited_attachments = attachments


class _FakeDiscordChannel:
    def __init__(self, channel_id: int, messages):
        self.id = channel_id
        self._messages = messages

    async def fetch_message(self, message_id: int):
        return self._messages[message_id]


class _FakeDiscordUser:
    def __init__(self, user_id: int, dm_channel):
        self.id = user_id
        self.dm_channel = dm_channel

    async def create_dm(self):
        return self.dm_channel


class _FakeDiscordClient:
    def __init__(self, *, channels=None, users=None):
        self._channels = channels or {}
        self._users = users or {}

    def get_channel(self, channel_id: int):
        return self._channels.get(channel_id)

    async def fetch_channel(self, channel_id: int):
        return self._channels.get(channel_id)

    def get_user(self, user_id: int):
        return self._users.get(user_id)

    async def fetch_user(self, user_id: int):
        return self._users.get(user_id)


class _FakeTurnMessageStore:
    def __init__(self):
        self._refs = []
        self._created_at = 0

    async def record_turn_message(
        self,
        *,
        channel_id: int,
        session_id: str,
        turn_index: int,
        discord_channel_id: int,
        message_id: int,
        delivery: str,
        recipient_user_id=None,
    ):
        from app.bot.session_map import TurnMessageRef

        self._created_at += 1
        self._refs.append(
            TurnMessageRef(
                channel_id=channel_id,
                session_id=session_id,
                turn_index=turn_index,
                discord_channel_id=discord_channel_id,
                message_id=message_id,
                delivery=delivery,
                recipient_user_id=recipient_user_id,
                created_at=self._created_at,
            )
        )

    async def list_turn_messages(
        self,
        *,
        channel_id: int,
        session_id: str,
        turns,
    ):
        turn_set = {int(t) for t in turns}
        refs = [
            ref for ref in self._refs
            if ref.channel_id == channel_id
            and ref.session_id == session_id
            and ref.turn_index in turn_set
        ]
        return sorted(
            refs,
            key=lambda r: (
                r.turn_index, r.created_at,
                r.discord_channel_id, r.message_id,
            ),
        )

    async def forget_turn_messages(self, refs):
        targets = {
            (
                ref.channel_id, ref.session_id, ref.turn_index,
                ref.discord_channel_id, ref.message_id,
            )
            for ref in refs
        }
        before = len(self._refs)
        self._refs = [
            ref for ref in self._refs
            if (
                ref.channel_id, ref.session_id, ref.turn_index,
                ref.discord_channel_id, ref.message_id,
            ) not in targets
        ]
        return before - len(self._refs)


class TestRewindDiscordCleanup:
    def _smap(self) -> _FakeTurnMessageStore:
        return _FakeTurnMessageStore()

    def test_deletes_only_messages_from_rewound_turns(
        self, tmp_path: Path,
    ):
        smap = self._smap()
        turn_3_msg = _FakeDiscordMessage(3000)
        turn_4_msg = _FakeDiscordMessage(4000)
        turn_5_msg = _FakeDiscordMessage(5000)
        channel = _FakeDiscordChannel(
            900,
            {
                3000: turn_3_msg,
                4000: turn_4_msg,
                5000: turn_5_msg,
            },
        )
        client = _FakeDiscordClient(channels={900: channel})

        async def _run():
            for turn, message_id in ((3, 3000), (4, 4000), (5, 5000)):
                await smap.record_turn_message(
                    channel_id=10,
                    session_id="sess",
                    turn_index=turn,
                    discord_channel_id=900,
                    message_id=message_id,
                    delivery="thread",
                )
            return await bot_commands._delete_rewound_turn_messages(
                client=client,
                smap=smap,
                channel_id=10,
                session_id="sess",
                deleted_turns=[4, 5],
            )

        cleanup = asyncio.run(_run())

        assert cleanup.tracked == 2
        assert cleanup.deleted == 2
        assert turn_3_msg.deleted is False
        assert turn_4_msg.deleted is True
        assert turn_5_msg.deleted is True
        remaining = asyncio.run(smap.list_turn_messages(
            channel_id=10, session_id="sess", turns=[3, 4, 5],
        ))
        assert [r.message_id for r in remaining] == [3000]

    def test_edits_message_when_delete_fails(self, tmp_path: Path):
        smap = self._smap()
        msg = _FakeDiscordMessage(4000, delete_raises=True)
        channel = _FakeDiscordChannel(900, {4000: msg})
        client = _FakeDiscordClient(channels={900: channel})

        async def _run():
            await smap.record_turn_message(
                channel_id=10,
                session_id="sess",
                turn_index=4,
                discord_channel_id=900,
                message_id=4000,
                delivery="thread",
            )
            return await bot_commands._delete_rewound_turn_messages(
                client=client,
                smap=smap,
                channel_id=10,
                session_id="sess",
                deleted_turns=[4],
            )

        cleanup = asyncio.run(_run())

        assert cleanup.deleted == 0
        assert cleanup.hidden == 1
        assert msg.edited_content == "_Rewound turn hidden._"
        assert msg.edited_embeds == []
        assert msg.edited_attachments == []
        remaining = asyncio.run(smap.list_turn_messages(
            channel_id=10, session_id="sess", turns=[4],
        ))
        assert remaining == []

    def test_asset_edit_fallback_clears_attachment_without_private_logs(
        self, tmp_path: Path, caplog,
    ):
        smap = self._smap()
        msg = _FakeDiscordMessage(
            4000,
            delete_error=RuntimeError(
                "delete denied for /private/table/source-map.png "
                "delivery_ref=asset://synthetic/hidden-map"
            ),
        )
        channel = _FakeDiscordChannel(900, {4000: msg})
        client = _FakeDiscordClient(channels={900: channel})

        async def _run():
            await smap.record_turn_message(
                channel_id=10,
                session_id="sess",
                turn_index=4,
                discord_channel_id=900,
                message_id=4000,
                delivery="thread_asset",
                recipient_user_id=42,
            )
            return await bot_commands._delete_rewound_turn_messages(
                client=client,
                smap=smap,
                channel_id=10,
                session_id="sess",
                deleted_turns=[4],
            )

        with caplog.at_level("WARNING", logger="app.bot.commands"):
            cleanup = asyncio.run(_run())

        assert cleanup.deleted == 0
        assert cleanup.hidden == 1
        assert msg.edited_content == "_Rewound turn hidden._"
        assert msg.edited_embeds == []
        assert msg.edited_attachments == []
        assert "/private/table/source-map.png" not in caplog.text
        assert "asset://synthetic/hidden-map" not in caplog.text

    def test_edit_fallback_hides_when_attachment_clear_is_unsupported(
        self, tmp_path: Path,
    ):
        smap = self._smap()
        msg = _FakeDiscordMessage(
            4000,
            delete_raises=True,
            supports_attachments=False,
        )
        channel = _FakeDiscordChannel(900, {4000: msg})
        client = _FakeDiscordClient(channels={900: channel})

        async def _run():
            await smap.record_turn_message(
                channel_id=10,
                session_id="sess",
                turn_index=4,
                discord_channel_id=900,
                message_id=4000,
                delivery="thread_asset",
                recipient_user_id=42,
            )
            return await bot_commands._delete_rewound_turn_messages(
                client=client,
                smap=smap,
                channel_id=10,
                session_id="sess",
                deleted_turns=[4],
            )

        cleanup = asyncio.run(_run())

        assert cleanup.hidden == 1
        assert msg.edited_content == "_Rewound turn hidden._"
        assert msg.edited_embeds == []
        assert msg.edited_attachments is None

    def test_dm_refs_resolve_through_recipient_user(self, tmp_path: Path):
        smap = self._smap()
        msg = _FakeDiscordMessage(4000)
        dm_channel = _FakeDiscordChannel(901, {4000: msg})
        user = _FakeDiscordUser(42, dm_channel)
        client = _FakeDiscordClient(users={42: user})

        async def _run():
            await smap.record_turn_message(
                channel_id=10,
                session_id="sess",
                turn_index=4,
                discord_channel_id=901,
                message_id=4000,
                delivery="dm",
                recipient_user_id=42,
            )
            return await bot_commands._delete_rewound_turn_messages(
                client=client,
                smap=smap,
                channel_id=10,
                session_id="sess",
                deleted_turns=[4],
            )

        cleanup = asyncio.run(_run())

        assert cleanup.deleted == 1
        assert msg.deleted is True
