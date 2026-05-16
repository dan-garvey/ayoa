"""Smoke tests for the interactive CLI (scripts/play.py).

Exercises CLIState command dispatch with a mocked EngineBridge — the goal
is to catch regressions in how commands map to bridge calls, claim state,
and current-actor updates. End-to-end engine behavior is covered by the
regular test suite."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from scripts.play import CLIState, _cli_log_level
from app.bot.engine_bridge import (
    CharacterSummary,
    CompletedPendingRoll,
    DndCombatParticipantView,
    DndCombatView,
    DndSheetAttachmentSummary,
    PendingRollPrompt,
    TurnHistoryEntry,
)
from app.schemas.characters import CharacterRecord, CharacterStatus, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.narrator import TranscriptEntry
from app.schemas.state import SessionState, SlotEntry, WorldState


SESSION_ID = "cli_test"
STORY_ID = "test_story"


def test_cli_suppresses_warning_logs_by_default():
    assert _cli_log_level(verbose=False) == logging.ERROR
    assert _cli_log_level(verbose=True) == logging.INFO


def _empty_ckpt(bindings: dict[str, str] | None = None) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id=SESSION_ID,
            player_character_id="aldric",
            character_bindings=bindings or {},
        ),
        world_state=WorldState(),
        characters=[],
    )


def _narrated_ckpt(bindings: dict[str, str] | None = None) -> CheckpointFile:
    ckpt = _empty_ckpt(bindings)
    ckpt.narrator_conversations["aldric"] = [
        ConversationMessage(role="assistant", content="Previously...")
    ]
    return ckpt


def _turn_response(**overrides):
    base = {
        "beat_ended_reason": "cascade_exhausted",
        "turn_index": 2,
        "output_text": "narration",
        "pre_turn_resolutions": [],
        "per_player_renders": {},
        "reaction_prompts": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _attachment_summary(**overrides) -> DndSheetAttachmentSummary:
    base = {
        "character_id": "aldric",
        "character_name": "Aldric",
        "imported_name": "DDB Aldric",
        "ruleset_id": "dnd5e_basic",
        "session_ruleset_id": "dnd5e_basic",
        "player_roll_mode": "auto",
        "source_type": "dndbeyond_browser_export",
        "total_level": 3,
        "classes": ["Fighter 3"],
        "armor_class": 16,
        "hit_points_current": 19,
        "hit_points_max": 22,
        "hit_points_temporary": 0,
        "skills_count": 1,
        "actions_count": 2,
        "spells_count": 0,
        "resources_count": 1,
        "name_overridden": False,
    }
    base.update(overrides)
    return DndSheetAttachmentSummary(**base)


def _mock_engine(bindings: dict[str, str] | None = None) -> MagicMock:
    engine = MagicMock()
    engine.load_latest.return_value = _empty_ckpt(bindings)
    engine.list_session_characters.return_value = [
        CharacterSummary(
            character_id="aldric", name="Aldric", role="wanderer",
            faction="", appearance="tall",
            status="active", is_playable=True,
            bound_user_id=(bindings or {}).get("aldric", ""),
        ),
        CharacterSummary(
            character_id="sera", name="Sera Vance", role="thief",
            faction="", appearance="wiry",
            status="active", is_playable=False,
            bound_user_id=(bindings or {}).get("sera", ""),
        ),
    ]
    engine.turn_history.return_value = []
    engine.takeover = MagicMock()
    engine.unbind_user = MagicMock()
    engine.build_character_dossier = MagicMock(return_value="# Dossier · Sera")
    engine.set_character_identity = MagicMock(return_value=_empty_ckpt(bindings))
    engine.get_bound_character_record = MagicMock()
    engine.create_player_character_simple = MagicMock()
    engine.attach_dndbeyond_character_export = AsyncMock(
        return_value=_attachment_summary(),
    )
    engine.run_turn = AsyncMock()
    engine.run_begin_turn = AsyncMock(return_value=_turn_response(
        beat_ended_reason="state_change",
        turn_index=1,
        output_text="opening narration",
        per_player_renders={"aldric": "Aldric wakes."},
    ))
    engine.combat_reaction_prompt_event = MagicMock(return_value="")
    engine.pending_roll_prompts = MagicMock(return_value=[])
    engine.complete_pending_roll = AsyncMock()
    engine.continue_pending_roll = AsyncMock()
    engine.list_loot_offers = MagicMock(return_value=[])
    engine.claim_loot = AsyncMock(return_value=SimpleNamespace(
        message="Claimed loot.",
    ))
    engine.split_loot_currency = AsyncMock(return_value=SimpleNamespace(
        message="Split coins.",
    ))
    engine.decline_loot = AsyncMock(return_value=SimpleNamespace(
        message="Declined loot.",
    ))
    engine.begin_combat = MagicMock(return_value=DndCombatView(
        session_id=SESSION_ID,
        active=False,
        message="No active combat.",
    ))
    engine.combat_status = MagicMock(return_value=DndCombatView(
        session_id=SESSION_ID,
        active=False,
        message="No active combat.",
    ))
    engine.combat_next = MagicMock(return_value=DndCombatView(
        session_id=SESSION_ID,
        active=False,
        message="No active combat.",
    ))
    engine.combat_end = MagicMock(return_value=DndCombatView(
        session_id=SESSION_ID,
        active=False,
        message="No active combat.",
    ))
    engine.combat_damage = MagicMock(return_value=DndCombatView(
        session_id=SESSION_ID,
        active=False,
        message="No active combat.",
    ))
    engine.combat_heal = MagicMock(return_value=DndCombatView(
        session_id=SESSION_ID,
        active=False,
        message="No active combat.",
    ))
    engine.combat_add = MagicMock(return_value=DndCombatView(
        session_id=SESSION_ID,
        active=False,
        message="No active combat.",
    ))
    engine.combat_remove = MagicMock(return_value=DndCombatView(
        session_id=SESSION_ID,
        active=False,
        message="No active combat.",
    ))
    return engine


def _character(
    character_id: str,
    name: str,
    *,
    role: str = "",
    location: str = "hall",
    status: CharacterStatus = CharacterStatus.active,
    is_playable: bool = False,
) -> CharacterRecord:
    return CharacterRecord(
        character_id=character_id,
        name=name,
        location=location,
        status=status,
        is_playable=is_playable,
        public_sheet=PublicSheet(role=role),
    )


def _sheet_character(character_id: str = "aldric") -> CharacterRecord:
    char = _character(character_id, "Aldric", role="fighter", is_playable=True)
    char.mechanics["dnd5e_sheet"] = {
        "ruleset_id": "dnd5e_basic",
        "identity": {
            "name": "DDB Aldric",
            "species": "Human",
            "background": "Soldier",
            "classes": [{"name": "Fighter", "level": 3}],
        },
        "source": {"type": "dndbeyond_browser_export"},
        "statblock": {
            "proficiency_bonus": 2,
            "ability_scores": {
                "str": {"score": 16, "modifier": 3},
                "dex": {"score": 12, "modifier": 1},
                "con": {"score": 14, "modifier": 2},
                "int": {"score": 10, "modifier": 0},
                "wis": {"score": 11, "modifier": 0},
                "cha": {"score": 8, "modifier": -1},
            },
            "saves": {"str": {"value": 5, "proficiency_multiplier": 1}},
            "skills": {
                "athletics": {"value": 5, "proficiency_multiplier": 1},
                "perception": {"value": 2, "passive": 12},
            },
            "defenses": {
                "armor_class": {"value": 16},
                "hit_points": {"current": 19, "max": 22, "temporary": 4},
                "initiative": {"value": 1},
                "movement": {"walk": {"value": 30, "unit": "ft"}},
            },
            "actions": [
                {
                    "name": "Longsword",
                    "kind": "weapon",
                    "attack": {"bonus": 5},
                    "damage": [
                        {"formula": "1d8+3", "damage_type": "slashing"},
                    ],
                },
            ],
            "spellcasting": {
                "slots": {"1": {"current": 2, "max": 2}},
                "spells": [{"name": "Light", "level": 0}],
            },
            "features": [{"name": "Second Wind", "kind": "class", "level": 1}],
        },
    }
    return char


@pytest.fixture
def run():
    """Run an async coroutine in a sync test."""
    return asyncio.run


class TestInitialization:
    def test_adopts_existing_bindings_on_resume(self):
        engine = _mock_engine(bindings={"aldric": "1", "sera": "2"})
        state = CLIState(engine, SESSION_ID, STORY_ID)
        assert state.claims == {"aldric": 1, "sera": 2}
        # Current actor auto-picks the first adopted claim.
        assert state.current_actor == "aldric"
        # Next uid is one past the max seen so we don't collide.
        assert state._next_user_id == 3

    def test_fresh_session_has_no_claims(self):
        engine = _mock_engine(bindings=None)
        state = CLIState(engine, SESSION_ID, STORY_ID)
        assert state.claims == {}
        assert state.current_actor is None

    def test_skips_non_integer_bindings(self):
        # Discord user_ids fit in int, but corrupted data shouldn't crash.
        engine = _mock_engine(bindings={"aldric": "not-a-number"})
        state = CLIState(engine, SESSION_ID, STORY_ID)
        assert state.claims == {}

    def test_resume_prefers_claimed_current_combatant(self):
        engine = _mock_engine(bindings={"aldric": "1", "sera": "2"})
        engine.combat_status.return_value = DndCombatView(
            session_id=SESSION_ID,
            active=True,
            current_participant_id="sera",
            participants=(
                DndCombatParticipantView(character_id="aldric", name="Aldric"),
                DndCombatParticipantView(
                    character_id="sera",
                    name="Sera",
                    current=True,
                ),
            ),
        )

        state = CLIState(engine, SESSION_ID, STORY_ID)

        assert state.current_actor == "sera"


class TestHistoryCommand:
    def test_history_uses_checkpoint_turn_ids(self, run, capsys):
        engine = _mock_engine()
        engine.turn_history.return_value = [
            TurnHistoryEntry(
                turn_index=44,
                entry=TranscriptEntry(user="", assistant="Rat runs."),
            ),
            TurnHistoryEntry(
                turn_index=47,
                entry=TranscriptEntry(user="I swing.", assistant="Club lands."),
            ),
        ]
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/history"))

        out = capsys.readouterr().out
        assert "--- Turn 44 ---" in out
        assert "--- Turn 47 ---" in out
        assert "--- Turn 1 ---" not in out
        assert "> I swing." in out

    def test_history_limit_preserves_checkpoint_turn_id(self, run, capsys):
        engine = _mock_engine()
        engine.turn_history.return_value = [
            TurnHistoryEntry(
                turn_index=44,
                entry=TranscriptEntry(user="", assistant="Rat runs."),
            ),
            TurnHistoryEntry(
                turn_index=47,
                entry=TranscriptEntry(user="I swing.", assistant="Club lands."),
            ),
        ]
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/history 1"))

        out = capsys.readouterr().out
        assert "--- Turn 44 ---" not in out
        assert "--- Turn 47 ---" in out


class TestJoinLeave:
    def test_join_without_target_lists_open_playables(self, run, capsys):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join"))

        engine.takeover.assert_not_called()
        out = capsys.readouterr().out
        assert "## Joinable" in out
        assert "aldric — Aldric · wanderer" in out
        assert "sera —" not in out
        assert "Custom character: /join_custom" in out

    def test_join_binds_and_sets_current(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join sera"))
        engine.takeover.assert_called_once_with(SESSION_ID, "sera", 1)
        assert state.claims == {"sera": 1}
        assert state.current_actor == "sera"

    def test_second_join_does_not_steal_current_actor(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        assert state.current_actor == "aldric"
        assert state.claims == {"aldric": 1, "sera": 2}

    def test_join_refuses_duplicate_claim(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join sera"))
        engine.takeover.reset_mock()
        run(state.handle_line("/join sera"))
        engine.takeover.assert_not_called()

    def test_leave_default_is_current_actor(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join sera"))
        run(state.handle_line("/leave"))
        engine.unbind_user.assert_called_once_with(SESSION_ID, 1)
        assert state.claims == {}
        assert state.current_actor is None

    def test_leave_switches_current_to_remaining_claim(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        run(state.handle_line("/leave aldric"))
        assert state.current_actor == "sera"

    def test_join_custom_uses_discord_style_create_flow(
        self, run, monkeypatch, capsys,
    ):
        engine = _mock_engine()
        new_char = _character(
            "akari_tanaka",
            "Akari Tanaka",
            is_playable=True,
        )
        engine.create_player_character_simple.return_value = new_char
        state = CLIState(engine, SESSION_ID, STORY_ID)
        inputs = iter([
            "Akari Tanaka",
            "blue cloak, short sword, travel-stained boots",
            "Former shrine guard.",
        ])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        run(state.handle_line("/join_custom"))

        engine.create_player_character_simple.assert_called_once_with(
            SESSION_ID,
            1,
            name="Akari Tanaka",
            appearance="blue cloak, short sword, travel-stained boots",
            backstory="Former shrine guard.",
        )
        assert state.claims == {"akari_tanaka": 1}
        assert state.current_actor == "akari_tanaka"
        assert "created akari_tanaka" in capsys.readouterr().out


class TestAttachCommand:
    def test_attach_defaults_to_current_actor(self, run, capsys, tmp_path):
        payload = {"raw": {"data": {"name": "DDB Aldric"}}}
        path = tmp_path / "aldric.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line(f"/attach {path}"))

        engine.attach_dndbeyond_character_export.assert_awaited_once_with(
            SESSION_ID,
            1,
            payload,
            character_id="aldric",
            name_override=None,
        )
        out = capsys.readouterr().out
        assert "--- D&D Sheet Attached ---" in out
        assert "Attached DDB Aldric to aldric." in out
        assert "Use /inventory to view carried items." in out

    def test_attach_can_target_claimed_character_with_name_override(
        self, run, capsys, tmp_path,
    ):
        payload = {"raw": {"data": {"name": "DDB Sera"}}}
        path = tmp_path / "sera.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        engine = _mock_engine()
        engine.attach_dndbeyond_character_export.return_value = (
            _attachment_summary(
                character_id="sera",
                character_name="Seren Swift",
                imported_name="DDB Sera",
                name_overridden=True,
            )
        )
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        capsys.readouterr()
        run(state.handle_line(f"/attach {path} sera --name \"Seren Swift\""))

        engine.attach_dndbeyond_character_export.assert_awaited_once_with(
            SESSION_ID,
            2,
            payload,
            character_id="sera",
            name_override="Seren Swift",
        )
        out = capsys.readouterr().out
        assert "Story name changed to Seren Swift." in out

    def test_attach_rejects_invalid_json(self, run, capsys, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{", encoding="utf-8")
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line(f"/attach {path}"))

        engine.attach_dndbeyond_character_export.assert_not_awaited()
        assert "not valid JSON" in capsys.readouterr().out


class TestSheetCommand:
    def test_sheet_defaults_to_current_actor_overview(self, run, capsys):
        engine = _mock_engine()
        engine.get_bound_character_record.return_value = _sheet_character()
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("/sheet"))

        engine.get_bound_character_record.assert_called_once_with(
            SESSION_ID,
            1,
            character_id="aldric",
        )
        out = capsys.readouterr().out
        assert "--- Sheet · Aldric (aldric) ---" in out
        assert "Fighter 3 · Human · Soldier · Imported name: DDB Aldric" in out
        assert "## Overview" in out
        assert "AC 16" in out
        assert "HP 19/22 (+4 temp)" in out

    def test_sheet_page_and_character_id(self, run, capsys):
        engine = _mock_engine()
        engine.get_bound_character_record.return_value = _sheet_character("sera")
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        capsys.readouterr()
        run(state.handle_line("/sheet abilities sera"))

        engine.get_bound_character_record.assert_called_once_with(
            SESSION_ID,
            2,
            character_id="sera",
        )
        out = capsys.readouterr().out
        assert "## Abilities" in out
        assert "STR 16 (+3) · save +5 prof" in out
        assert "Athletics +5 prof" in out

    def test_sheet_all_prints_multiple_pages(self, run, capsys):
        engine = _mock_engine()
        engine.get_bound_character_record.return_value = _sheet_character()
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("/sheet all"))

        out = capsys.readouterr().out
        assert "## Overview" in out
        assert "## Actions" in out
        assert "Longsword - weapon, hit +5, 1d8+3 slashing" in out
        assert "## Features" in out
        assert "Second Wind" in out


class TestCharactersCommand:
    def test_characters_lists_discord_joinable_slots_first(
        self, run, capsys,
    ):
        engine = _mock_engine()
        ckpt = _empty_ckpt()
        ckpt.characters = [
            _character(
                "player_protagonist",
                "Intended Protagonist",
                role="defective hero",
                is_playable=True,
            ),
            _character("guild_master", "Guild Master", role="guild master"),
            _character(
                "claimed_slot",
                "Claimed Slot",
                role="claimed hero",
                is_playable=True,
            ),
            _character(
                "dead_queen",
                "Dead Queen",
                status=CharacterStatus.dormant,
                is_playable=True,
            ),
        ]
        engine.load_latest.return_value = ckpt
        engine.list_session_characters.return_value = [
            CharacterSummary(
                character_id="player_protagonist",
                name="Intended Protagonist",
                role="defective hero",
                faction="",
                appearance="",
                status="active",
                is_playable=True,
                bound_user_id="",
            ),
            CharacterSummary(
                character_id="guild_master",
                name="Guild Master",
                role="guild master",
                faction="",
                appearance="",
                status="active",
                is_playable=False,
                bound_user_id="",
            ),
            CharacterSummary(
                character_id="claimed_slot",
                name="Claimed Slot",
                role="claimed hero",
                faction="",
                appearance="",
                status="active",
                is_playable=True,
                bound_user_id="2",
            ),
            CharacterSummary(
                character_id="dead_queen",
                name="Dead Queen",
                role="",
                faction="",
                appearance="",
                status="dormant",
                is_playable=True,
                bound_user_id="",
            ),
        ]
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/characters"))

        out = capsys.readouterr().out
        joinable_block = out.split("## Here", 1)[0]
        assert "## Joinable" in joinable_block
        assert (
            "player_protagonist — Intended Protagonist · defective hero"
            in joinable_block
        )
        assert "dead_queen — Dead Queen" in joinable_block
        assert "guild_master" not in joinable_block
        assert "claimed_slot" not in joinable_block
        assert "player_protagonist  [joinable]" in out


class TestAs:
    def test_switches_current_when_claimed(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        run(state.handle_line("/as sera"))
        assert state.current_actor == "sera"

    def test_refuses_unclaimed_target(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/as sera"))
        assert state.current_actor == "aldric"  # unchanged


class TestDeferCommand:
    def test_defer_submits_null_turn_for_current_actor(self, run):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(return_value=_turn_response(
            beat_ended_reason="state_change",
            turn_index=3,
            output_text="",
        ))

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/defer"))

        engine.run_turn.assert_awaited_once_with(
            session_id=SESSION_ID,
            user_input="(defer)",
            acting_character_id="aldric",
        )

    def test_defer_uses_reaction_endpoint_when_actor_has_reaction(self, run):
        engine = _mock_engine()
        engine.combat_reaction_prompt_event.return_value = "evt_react"
        engine.defer_combat_reaction = AsyncMock(return_value=_turn_response(
            output_text="Seren passes on the reaction.",
        ))

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/defer"))

        engine.defer_combat_reaction.assert_awaited_once_with(
            session_id=SESSION_ID,
            character_id="aldric",
            event_id="evt_react",
            user_id=1,
        )
        engine.run_turn.assert_not_awaited()


class TestLootCommand:
    def test_loot_all_claims_every_open_offer(self, run, capsys):
        engine = _mock_engine()
        engine.list_loot_offers.return_value = [
            SimpleNamespace(offer_id="offer_a"),
            SimpleNamespace(offer_id="offer_b"),
        ]
        engine.claim_loot.side_effect = [
            SimpleNamespace(message="Claimed from offer_a."),
            SimpleNamespace(message="Claimed from offer_b."),
        ]
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("/loot all"))

        engine.list_loot_offers.assert_called_once_with(
            SESSION_ID,
            1,
            character_id="aldric",
        )
        engine.claim_loot.assert_has_awaits([
            call(
                session_id=SESSION_ID,
                user_id=1,
                character_id="aldric",
                offer_id="offer_a",
                item_ids=[],
                take_currency=True,
                take_all_available=True,
            ),
            call(
                session_id=SESSION_ID,
                user_id=1,
                character_id="aldric",
                offer_id="offer_b",
                item_ids=[],
                take_currency=True,
                take_all_available=True,
            ),
        ])
        out = capsys.readouterr().out
        assert "Claimed from offer_a." in out
        assert "Claimed from offer_b." in out

    def test_loot_all_reports_empty_offers(self, run, capsys):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("/loot all"))

        engine.claim_loot.assert_not_awaited()
        assert "no open loot offers" in capsys.readouterr().out


class TestBeginCommand:
    def test_begin_uses_current_actor_and_prints_opening(self, run, capsys):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("/begin"))

        engine.run_begin_turn.assert_awaited_once_with(
            session_id=SESSION_ID,
            triggering_character_id="aldric",
        )
        out = capsys.readouterr().out
        assert "--- Turn 1 · aldric ---" in out
        assert "opening narration" in out


class TestCombatCommand:
    def test_combat_status_renders_order(self, run, capsys):
        engine = _mock_engine()
        engine.combat_status.return_value = DndCombatView(
            session_id=SESSION_ID,
            active=True,
            round_number=2,
            turn_number=3,
            current_participant_id="sera",
            participants=(
                DndCombatParticipantView(
                    character_id="aldric",
                    name="Aldric",
                    initiative=15,
                    hp_current=12,
                    hp_max=20,
                    armor_class=14,
                ),
                DndCombatParticipantView(
                    character_id="sera",
                    name="Sera",
                    current=True,
                    initiative=18,
                    hp_current=7,
                    hp_max=9,
                    hp_temporary=2,
                    armor_class=16,
                    active_effects=("Bless (concentration; 8 rounds)",),
                ),
            ),
        )

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/combat status"))

        engine.combat_status.assert_called_once_with(SESSION_ID, private=True)
        out = capsys.readouterr().out
        assert "Round 2 · Turn 3" in out
        assert (
            "> Sera (sera) - HP 7/9 (+2); AC 16; Init 18; "
            "Effects: Bless (concentration; 8 rounds)"
        ) in out

    def test_combat_status_renders_map_lines(self, run, capsys):
        engine = _mock_engine()
        engine.combat_status.return_value = DndCombatView(
            session_id=SESSION_ID,
            active=True,
            participants=(
                DndCombatParticipantView(character_id="aldric", name="Aldric"),
            ),
            map_lines=(
                "Battle map: Bridge (8x5, 5 ft squares).",
                "A1: Aldric; B2: Sera.",
            ),
        )
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/combat status"))

        out = capsys.readouterr().out
        assert "Battle map: Bridge (8x5, 5 ft squares)." in out
        assert "A1: Aldric; B2: Sera." in out

    def test_combat_damage_parses_amount(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/combat damage sera 5"))

        engine.combat_damage.assert_called_once_with(SESSION_ID, "sera", 5)

    def test_turn_result_syncs_prompt_to_claimed_current_initiative(
        self, run, capsys,
    ):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(return_value=_turn_response(
            output_text="Aldric acts.",
        ))
        engine.combat_status.return_value = DndCombatView(
            session_id=SESSION_ID,
            active=True,
            round_number=1,
            turn_number=2,
            current_participant_id="sera",
            participants=(
                DndCombatParticipantView(
                    character_id="aldric",
                    name="Aldric",
                ),
                DndCombatParticipantView(
                    character_id="sera",
                    name="Sera",
                    current=True,
                ),
            ),
        )

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        run(state.handle_line("I attack."))

        assert state.current_actor == "sera"
        assert "now acting as sera (current initiative)" in capsys.readouterr().out

    def test_combat_started_result_explains_initiative_handoff(
        self, run, capsys,
    ):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(return_value=_turn_response(
            beat_ended_reason="combat_started",
            turn_index=5,
            output_text="Seren reaches for steel.",
        ))
        engine.combat_status.return_value = DndCombatView(
            session_id=SESSION_ID,
            active=True,
            round_number=1,
            turn_number=1,
            current_participant_id="sera",
            participants=(
                DndCombatParticipantView(
                    character_id="sera",
                    name="Sera",
                    current=True,
                    initiative=18,
                ),
                DndCombatParticipantView(
                    character_id="aldric",
                    name="Aldric",
                    initiative=13,
                ),
            ),
        )

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        run(state.handle_line("I attack."))

        out = capsys.readouterr().out
        assert "=== COMBAT BEGINS ===" in out
        assert "Initiative: Sera (sera) 18, Aldric (aldric) 13." in out
        assert "Current turn: Sera (sera)." in out
        assert "initiating action has not resolved before initiative" in out

    def test_cat_ii_pending_switches_to_claimed_responder(
        self, run, capsys,
    ):
        engine = _mock_engine()
        ckpt = _empty_ckpt({"aldric": "1", "sera": "2"})
        ckpt.session.active_act_slots["sera"] = SlotEntry(
            reason="cat_ii_responder",
            cat_ii_event_id="evt_cat",
        )
        engine.load_latest.return_value = ckpt
        engine.run_turn = AsyncMock(return_value=_turn_response(
            beat_ended_reason="cat_ii_pending",
            turn_index=6,
            output_text="",
            per_player_renders={"aldric": "Aldric waits for Sera's answer."},
        ))

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("I reach for Sera's hand."))

        out = capsys.readouterr().out
        assert state.current_actor == "sera"
        assert "beat paused" in out
        assert "waiting on sera" in out
        assert "Switched to sera" in out
        assert "type their response to continue" in out
        assert "Aldric waits for Sera's answer." in out

    def test_commitment_revision_prompt_prints_for_claimed_character(
        self, run, capsys,
    ):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(return_value=_turn_response(
            output_text="The hall changes.",
            commitment_revision_prompts={"aldric": ["commit_watch"]},
        ))

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("I wait."))

        out = capsys.readouterr().out
        assert "--- Commitment Interrupted · aldric ---" in out
        assert "commitment commit_watch" in out
        assert "type a revised action or (continue)" in out


class TestRollCommand:
    def test_act_surfaces_pending_rolls(self, run, capsys):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(return_value=_turn_response(
            beat_ended_reason="cat_ii_pending_rolls",
            output_text="Ash's blade flashes toward the captain.",
        ))
        engine.pending_roll_prompts.return_value = [
            PendingRollPrompt(
                session_id=SESSION_ID,
                event_id="evt_1",
                roll_id="roll_attack",
                actor_id="aldric",
                user_id="1",
                label="Attack",
                reason="A blade strike.",
            ),
        ]

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("I attack."))

        out = capsys.readouterr().out
        assert "--- Pending D&D Rolls ---" in out
        assert "roll_attack: Attack" in out
        assert "Use /roll" in out

    def test_roll_all_completes_rolls_then_continues(self, run, capsys):
        engine = _mock_engine()
        prompts = [
            PendingRollPrompt(
                session_id=SESSION_ID,
                event_id="evt_1",
                roll_id="roll_attack",
                actor_id="aldric",
                user_id="1",
                label="Attack",
                reason="A blade strike.",
            ),
            PendingRollPrompt(
                session_id=SESSION_ID,
                event_id="evt_1",
                roll_id="roll_acrobatics",
                actor_id="aldric",
                user_id="1",
                label="Acrobatics",
                reason="Keep footing.",
            ),
        ]
        engine.pending_roll_prompts.side_effect = [prompts, []]
        engine.complete_pending_roll.side_effect = [
            CompletedPendingRoll(
                session_id=SESSION_ID,
                event_id="evt_1",
                roll_id="roll_attack",
                actor_id="aldric",
                user_id="1",
                label="Attack",
                reason="A blade strike.",
                expression="1d20+5",
                total=12,
                detail="1d20 (7) + 5 = `12`",
                crit="none",
                remaining_pending_rolls=1,
            ),
            CompletedPendingRoll(
                session_id=SESSION_ID,
                event_id="evt_1",
                roll_id="roll_acrobatics",
                actor_id="aldric",
                user_id="1",
                label="Acrobatics",
                reason="Keep footing.",
                expression="1d20+7",
                total=23,
                detail="1d20 (16) + 7 = `23`",
                crit="none",
                remaining_pending_rolls=0,
            ),
        ]
        engine.continue_pending_roll.return_value = _turn_response(
            turn_index=4,
            output_text="Ash keeps his footing.",
        )

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/roll all"))

        assert engine.complete_pending_roll.await_count == 2
        engine.continue_pending_roll.assert_awaited_once_with(
            session_id=SESSION_ID,
            event_id="evt_1",
            actor_id="aldric",
        )
        out = capsys.readouterr().out
        assert "Rolled Attack:" in out
        assert "Rolled Acrobatics:" in out
        assert "Ash keeps his footing." in out


class TestActingDescribe:
    def test_plain_text_acts_as_current(self, run):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(return_value=_turn_response())

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("I look around"))
        engine.run_turn.assert_awaited_once()
        call_kwargs = engine.run_turn.await_args.kwargs
        assert call_kwargs["user_input"] == "I look around"
        assert call_kwargs["acting_character_id"] == "aldric"

    def test_act_refused_without_actor(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("I look around"))
        engine.run_turn.assert_not_called()

    def test_describe_writes_to_current_actor(self, run, monkeypatch):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        # /describe is now interactive (two prompts). Stub the input
        # path so the test doesn't hang on stdin.
        engine.set_character_identity.return_value = _narrated_ckpt(
            {"aldric": "1"}
        )
        inputs = iter(["Aldric", "tall and weary"])
        monkeypatch.setattr(
            "builtins.input", lambda prompt="": next(inputs),
        )
        run(state.handle_line("/describe"))
        engine.set_character_identity.assert_called_once_with(
            SESSION_ID, "aldric", name="Aldric", appearance="tall and weary",
        )

    def test_describe_preplay_auto_begins(self, run, monkeypatch):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(return_value=_turn_response(
            turn_index=1,
            output_text="opening narration",
        ))

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        inputs = iter(["", "tall and weary"])
        monkeypatch.setattr(
            "builtins.input", lambda prompt="": next(inputs),
        )
        run(state.handle_line("/describe"))
        engine.run_turn.assert_awaited_once()
        assert engine.run_turn.await_args.kwargs["user_input"] == "(begin)"


class TestQuit:
    def test_quit_clears_running(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/quit"))
        assert state.running is False

    def test_unknown_command_does_not_exit(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/nope"))
        assert state.running is True


class TestRewindCommand:
    """CLI dispatch tests for /rewind. The engine layer is stubbed to
    isolate the command-handler logic — actual rewind behavior is
    covered by tests/test_rewind.py."""

    def _make_engine_with_rewind(
        self, *, turns: list[int],
    ) -> MagicMock:
        engine = _mock_engine()
        engine.list_checkpoint_turns = MagicMock(return_value=turns)

        # `preview_rewind` and `rewind_session` mirror each other's
        # validation. The mocks here re-implement the bare-minimum
        # validation so the CLI handler exercises the branches it
        # needs to handle (unknown turn, target == latest, etc.) the
        # same way the real bridge would.
        from app.bot.engine_bridge import RewindResult

        def _preview(_session_id: str, target: int) -> RewindResult:
            if not turns:
                raise FileNotFoundError("no checkpoints")
            if target < 0:
                raise ValueError("must be >= 0")
            if target not in turns:
                raise ValueError(
                    f"Turn {target} has no checkpoint."
                )
            if target >= turns[-1]:
                raise ValueError(
                    f"already the current state (latest is turn {turns[-1]})"
                )
            return RewindResult(
                session_id=SESSION_ID,
                target_turn=target,
                previous_latest=turns[-1],
                new_latest=target,
                deleted_turns=[t for t in turns if t > target],
                location="hall",
                actor_character_id="aldric",
            )

        async def _rewind(_session_id: str, target: int) -> RewindResult:
            return _preview(_session_id, target)

        engine.preview_rewind = MagicMock(side_effect=_preview)
        engine.rewind_session = AsyncMock(side_effect=_rewind)
        return engine

    def test_rewind_no_arg_lists_range(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2, 3])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind"))

        out = capsys.readouterr().out
        assert "available turns: 0..3" in out
        assert "current latest: turn 3" in out
        # Bare /rewind should NOT mutate.
        engine.rewind_session.assert_not_called()

    def test_rewind_with_target_invokes_engine(
        self, run, capsys, monkeypatch,
    ):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2, 3, 4])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        prompts: list[str] = []
        monkeypatch.setattr(
            "builtins.input",
            lambda prompt="": prompts.append(prompt) or "rewind 2",
        )
        run(state.handle_line("/rewind 2"))

        engine.preview_rewind.assert_called_once_with(SESSION_ID, 2)
        engine.rewind_session.assert_awaited_once_with(SESSION_ID, 2)
        assert prompts == ["Type 'rewind 2' to confirm: "]
        out = capsys.readouterr().out
        assert "rewinding" in out
        assert "turn 4 → turn 2" in out
        assert "permanently deletes" in out
        assert "rewound to turn 2" in out
        assert "deleted 2 checkpoint(s)" in out

    def test_rewind_cancel_does_not_commit(
        self, run, capsys, monkeypatch,
    ):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2, 3, 4])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        prompts: list[str] = []
        monkeypatch.setattr(
            "builtins.input",
            lambda prompt="": prompts.append(prompt) or "no",
        )
        run(state.handle_line("/rewind 2"))

        engine.preview_rewind.assert_called_once_with(SESSION_ID, 2)
        engine.rewind_session.assert_not_called()
        assert prompts == ["Type 'rewind 2' to confirm: "]
        out = capsys.readouterr().out
        assert "rewinding" in out
        assert "cancelled" in out
        assert "nothing was deleted" in out

    def test_rewind_invalid_target_does_not_commit(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind 5"))

        engine.preview_rewind.assert_called_once_with(SESSION_ID, 5)
        engine.rewind_session.assert_not_called()
        out = capsys.readouterr().out
        assert "error" in out.lower()

    def test_rewind_to_latest_rejected_before_commit(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind 2"))

        engine.preview_rewind.assert_called_once_with(SESSION_ID, 2)
        engine.rewind_session.assert_not_called()
        out = capsys.readouterr().out
        assert "already the current state" in out

    def test_rewind_non_integer_arg_complains(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[0, 1, 2])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind hello"))

        engine.preview_rewind.assert_not_called()
        engine.rewind_session.assert_not_called()
        out = capsys.readouterr().out
        assert "integer" in out.lower()

    def test_rewind_empty_session_says_so(self, run, capsys):
        engine = self._make_engine_with_rewind(turns=[])
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/rewind"))

        out = capsys.readouterr().out
        assert "no checkpoints" in out
