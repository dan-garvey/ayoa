"""Focused unit tests for bot-layer internals that the code review
flagged as having no coverage. Covers:

- F3.9: best-effort parsing of DISCORD_ADMIN_USER_IDS (_is_admin).
- F3.8: EngineBridge.import_story invokes on_analysis_complete with the
  right (analysis, error) tuple in both success and failure paths.
- briefing copy: render_briefing must not mention `/describe` (legacy
  command renamed in the join-overhaul) and must point at `/join`.
- POV-thread cascade: `_post_actor_render` falls thread → DM → none.

Heavier discord-interaction paths (full /act and /join harness, orphan
thread purge) still aren't covered — they'd need a discord.py mock
infrastructure we don't have.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.bot import commands as bot_commands
from app.bot.engine_bridge import EngineBridge
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile, ImportAnalysis
from app.schemas.dnd_inventory import DndLootOffer
from app.schemas.responses import DiceRollDisplay, TurnResponse
from app.schemas.state import SessionState, StorySetting, WorldState


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


# ---- F3.8: import_story callback --------------------------------------------


@pytest.fixture
def mock_bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return EngineBridge(saves_dir=str(tmp_path), prompts_dir="app/prompts")


def _minimal_ckpt(story_id: str) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(session_id=story_id, story_id=story_id),
        world_state=WorldState(),
        characters=[],
    )


def _minimal_combined_result(story_id: str):
    """Match the shape `run_import_combined` returns: checkpoint +
    priming_messages + assistant_text, so the test can mock the new
    combined-import path without importing the private result class."""
    from app.engine.story_importer import _CombinedImportResult
    return _CombinedImportResult(
        checkpoint=_minimal_ckpt(story_id),
        priming_messages=[
            {"role": "system", "content": "fake system"},
            {"role": "user", "content": "fake user"},
        ],
        assistant_text='{"fake": "assistant echo"}',
    )


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
        assert len(reloaded.session.pending_router_state_changes) == 1
        update = reloaded.session.pending_router_state_changes[0]
        assert "Inventory update before the next action" in update
        assert "alice took Potion of Healing and 8 sp from iron chest" in update
        assert "explicit player continuity" in update

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
        assert reloaded.session.pending_router_state_changes == []


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

    def test_describe_preplay_opens_with_run_begin_turn(self, monkeypatch):
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

        ckpt = CheckpointFile(
            session=SessionState(
                session_id="s",
                character_bindings={"alice": "42"},
            ),
            world_state=WorldState(),
            characters=[CharacterRecord(character_id="alice", name="Alice")],
            narrator_conversations={},
        )
        response = TurnResponse(
            session_id="s",
            checkpoint_id="ckpt_0001",
            turn_index=1,
            output_text="The story opens.",
            per_player_renders={"alice": "The story opens."},
            beat_ended_reason="state_change",
        )

        engine = MagicMock()
        engine.get_user_binding.return_value = "alice"
        engine.set_character_identity.return_value = ckpt
        engine.run_begin_turn = AsyncMock(return_value=response)
        engine.run_turn = AsyncMock()

        smap = MagicMock()
        smap.get = AsyncMock(
            return_value=SimpleNamespace(session_id="s", story_id="story"),
        )

        captured: dict = {}

        async def _fake_post_actor_render(**kwargs):
            captured.update(kwargs)
            return "thread", SimpleNamespace(mention="#alice-thread")

        monkeypatch.setattr(
            bot_commands, "_post_actor_render", _fake_post_actor_render,
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

        engine.run_begin_turn.assert_awaited_once_with(
            session_id="s",
            triggering_character_id="alice",
        )
        engine.run_turn.assert_not_awaited()
        assert captured["turn_index"] == 1
        assert captured["character_id"] == "alice"

    def test_duplicate_story_import_guidance_matches_delete_contract(self):
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
        engine.list_story_ids.return_value = ["dupe_story"]
        smap = MagicMock()
        tree = FakeTree()
        bot_commands.register(tree, engine, smap, None)

        inter = MagicMock()
        inter.response.send_message = AsyncMock()
        attachment = MagicMock()
        attachment.size = 10
        attachment.filename = "dupe_story.txt"
        attachment.content_type = "text/plain"

        story_import = tree.groups["story"].get_command("import")
        asyncio.run(story_import.callback(inter, attachment, "dupe_story"))

        message = inter.response.send_message.await_args.args[0]
        assert "/story delete`" in message
        assert "/story delete dupe_story" not in message


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


class TestImportAnalysisCallback:
    """EngineBridge.import_story fires on_analysis_complete with
    (analysis, None) on success and (None, exception) on failure.

    Both paths mock out run_import and run_preservation_analysis so the
    test doesn't touch the LLM. The callback is awaited by the
    background task that the bridge schedules; we use asyncio.Event
    plumbing to wait for it deterministically."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_callback_fires_with_analysis_on_success(self, mock_bridge):
        story_id = "cb_success"
        analysis = ImportAnalysis(
            source_chars=100,
            source_words=20,
            output_chars=80,
            output_words=15,
            coverage_rating="high",
            dropped_topics=[],
            compressed_topics=[],
            preservation_notes="fine",
            duration_s=1.0,
            model="claude-sonnet-4-6",
        )

        received: dict = {}
        done = asyncio.Event()

        async def _cb(a, e):
            received["analysis"] = a
            received["err"] = e
            done.set()

        async def run():
            with patch(
                "app.bot.engine_bridge.run_import_two_call",
                new=AsyncMock(return_value=_minimal_combined_result(story_id)),
            ), patch(
                "app.bot.engine_bridge.run_preservation_analysis_continuation",
                new=AsyncMock(return_value=analysis),
            ):
                await mock_bridge.import_story(
                    "source text here", story_id,
                    on_analysis_complete=_cb,
                )
                # Background task is scheduled but not awaited — wait for
                # the callback to fire.
                await asyncio.wait_for(done.wait(), timeout=2.0)

        self._run(run())
        assert received["err"] is None
        assert received["analysis"] is analysis

    def test_callback_fires_with_error_on_failure(self, mock_bridge):
        story_id = "cb_failure"
        boom = RuntimeError("analysis exploded")

        received: dict = {}
        done = asyncio.Event()

        async def _cb(a, e):
            received["analysis"] = a
            received["err"] = e
            done.set()

        async def run():
            with patch(
                "app.bot.engine_bridge.run_import_two_call",
                new=AsyncMock(return_value=_minimal_combined_result(story_id)),
            ), patch(
                "app.bot.engine_bridge.run_preservation_analysis_continuation",
                new=AsyncMock(side_effect=boom),
            ):
                await mock_bridge.import_story(
                    "source text here", story_id,
                    on_analysis_complete=_cb,
                )
                await asyncio.wait_for(done.wait(), timeout=2.0)

        self._run(run())
        assert received["analysis"] is None
        assert received["err"] is boom

    def test_no_callback_is_fine(self, mock_bridge):
        """Omitting the callback is the CLI/legacy path — shouldn't error."""
        story_id = "no_cb"
        analysis = ImportAnalysis(
            coverage_rating="high", duration_s=0.0,
        )

        async def run():
            with patch(
                "app.bot.engine_bridge.run_import_two_call",
                new=AsyncMock(return_value=_minimal_combined_result(story_id)),
            ), patch(
                "app.bot.engine_bridge.run_preservation_analysis_continuation",
                new=AsyncMock(return_value=analysis),
            ):
                ckpt = await mock_bridge.import_story("src", story_id)
                # Give the background task a moment to complete; success
                # path just logs.
                await asyncio.sleep(0.05)
                assert ckpt.session.story_id == story_id

        self._run(run())


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

    def test_unbind_user_calls_purge(self, mock_bridge):
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

        freed = mock_bridge.unbind_user(session_id, 77)
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
        from app.schemas.event_router import EventRouterOutput
        from app.schemas.events import (
            CanonicalEvent,
            WorldAdjudication,
        )
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

        routed = EventRouterOutput(
            event_id="",
            decision_rationale="(test fixture)",
            canonical_event=CanonicalEvent(
                world_adjudication=WorldAdjudication(feasible=True),
                observable_facts=[],
            ),
            observers=[],
            requires_responders=False,
            required_responders=[],
            agent_responder_picks=[],
            ends_beat=True,
            ends_beat_reason="",
            spawn=[],
            dormant=[],
            cull=["villain"],
        )
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
                beat_ended_reason="directed_at_player",
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
        assert result.beat_ended_reason == "directed_at_player"

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
                beat_ended_reason="directed_at_player",
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
        assert result.beat_ended_reason == "directed_at_player"

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
            beat_ended_reason="directed_at_player",
        )
        mock_bridge.orchestrator.process_turn = AsyncMock(
            return_value=TurnResponse(
                session_id="session",
                beat_ended_reason="directed_at_player",
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
                session_id="session", beat_ended_reason="directed_at_player",
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
        assert response.beat_ended_reason == "directed_at_player"

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
                beat_ended_reason="directed_at_player",
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


# ---- rewind Discord message cleanup ---------------------------------------


class _FakeDiscordMessage:
    def __init__(self, message_id: int, *, delete_raises: bool = False):
        self.id = message_id
        self.delete_raises = delete_raises
        self.deleted = False
        self.edited_content = None
        self.edited_embeds = None

    async def delete(self):
        if self.delete_raises:
            raise RuntimeError("delete denied")
        self.deleted = True

    async def edit(self, *, content=None, embeds=None):
        self.edited_content = content
        self.edited_embeds = embeds


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
        remaining = asyncio.run(smap.list_turn_messages(
            channel_id=10, session_id="sess", turns=[4],
        ))
        assert remaining == []

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
