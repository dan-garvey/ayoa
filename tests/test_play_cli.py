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

from scripts.play import (
    CLIState,
    _ConsoleInput,
    _cli_log_level,
    _default_history_path,
    _format_missing_llm_credentials,
    _prepare_session_story,
    _print_dice_roll_displays,
    _session_command_lock,
    _split_combat_ids,
    run_oneshot_commands,
)
from app.engine.cli_image_display import CliImageDisplayResult
from app.engine.frontend_views import (
    CharacterSummary,
    CompletedPendingRoll,
    DndCombatParticipantView,
    DndCombatView,
    DndSheetAttachmentSummary,
    OpeningLobbyView,
    PendingRollPrompt,
    PlayerJoinResult,
    SessionActivityView,
    StorySummary,
    TurnHistoryEntry,
)
from app.llm.config import LIVE_PLAY_REQUIRED_ROLES, LLMConfig, MissingLLMCredential
from app.schemas.characters import CharacterRecord, CharacterStatus, PublicSheet
from app.schemas.checkpoint import CheckpointFile
from app.schemas.conversation import ConversationMessage
from app.schemas.dnd_inventory import DndLootOffer, DndLootOfferItem
from app.schemas.narrator import TranscriptEntry
from app.schemas.responses import DiceRollDisplay
from app.schemas.state import SessionState, SlotEntry, WorldState


SESSION_ID = "cli_test"
STORY_ID = "test_story"


def test_cli_suppresses_warning_logs_by_default():
    assert _cli_log_level(verbose=False) == logging.ERROR
    assert _cli_log_level(verbose=True) == logging.INFO


class _FakeReadline:
    def __init__(self) -> None:
        self.history: list[str] = []
        self.bindings: list[str] = []
        self.history_length = 0
        self.completer = None
        self.completer_delims = ""
        self.read_history_path = ""
        self.written_history_path = ""

    def read_history_file(self, path: str) -> None:
        self.read_history_path = path
        raise FileNotFoundError(path)

    def write_history_file(self, path: str) -> None:
        self.written_history_path = path

    def parse_and_bind(self, binding: str) -> None:
        self.bindings.append(binding)

    def set_history_length(self, length: int) -> None:
        self.history_length = length

    def set_completer(self, completer) -> None:
        self.completer = completer

    def set_completer_delims(self, delims: str) -> None:
        self.completer_delims = delims

    def get_current_history_length(self) -> int:
        return len(self.history)

    def get_history_item(self, index: int) -> str:
        return self.history[index - 1]

    def add_history(self, line: str) -> None:
        self.history.append(line)


def test_console_input_installs_readline_history_and_completion(tmp_path):
    fake_readline = _FakeReadline()
    history_path = tmp_path / "state" / "play_history"
    console = _ConsoleInput(
        history_path=history_path,
        readline_module=fake_readline,
        interactive=True,
    )

    assert console.install() is True
    console.add_history("I look around")
    console.add_history("I look around")
    console.add_history("")
    console.save_history()

    assert fake_readline.read_history_path == str(history_path)
    assert fake_readline.written_history_path == str(history_path)
    assert fake_readline.history == ["I look around"]
    assert fake_readline.history_length == 1000
    assert "tab: complete" in fake_readline.bindings
    assert fake_readline.completer_delims == "\t\n"
    assert fake_readline.completer("/he", 0) == "/help"
    assert fake_readline.completer("/loot t", 0) == "/loot take"


def test_default_history_path_uses_xdg_state_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    assert _default_history_path() == tmp_path / "ayoa" / "play_history"


def test_format_missing_llm_credentials_names_roles_and_envs():
    text = _format_missing_llm_credentials((
        MissingLLMCredential(
            role="event_router",
            provider="openai",
            env_names=("OPEN_AI_ROUTER", "OPENAI_API_KEY"),
        ),
    ))

    assert "event_router (openai)" in text
    assert "OPEN_AI_ROUTER" in text
    assert "OPENAI_API_KEY" in text
    assert "LLM_ROLE_MODELS" not in text
    assert "dnd_combat_manager=anthropic" not in text


def test_live_play_preflight_does_not_require_content_manager_key():
    config = LLMConfig(
        api_key="anthropic-key",
        openai_role_api_keys={
            "event_router": "router-key",
            "narrator": "narrator-key",
            "dnd_combat_manager": "combat-key",
        },
    )

    missing = config.missing_credentials(LIVE_PLAY_REQUIRED_ROLES)

    assert missing == ()
    assert "content_manager" not in LIVE_PLAY_REQUIRED_ROLES


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


def _loot_offer(
    offer_id: str = "loot_evt_chest",
    *,
    source_label: str = "iron chest",
    item_id: str = "old_sword",
    item_name: str = "Old sword",
) -> DndLootOffer:
    return DndLootOffer(
        offer_id=offer_id,
        source_event_id="evt_chest",
        source_kind="container",
        source_label=source_label,
        eligible_character_ids=["aldric"],
        items=[
            DndLootOfferItem(
                item_id=item_id,
                name=item_name,
                kind="weapon",
                quantity=1,
                identified=True,
                requires_identification=False,
                requires_attunement=False,
                consumable=False,
                value_gp=0,
                weight=1,
                notes="",
            )
        ],
    )


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
            status="active", is_playable=True,
            bound_user_id=(bindings or {}).get("sera", ""),
        ),
    ]
    engine.list_story_ids.return_value = [STORY_ID]
    engine.list_story_characters.side_effect = (
        lambda _story_id: list(engine.list_session_characters.return_value)
    )
    engine.story_summary.side_effect = lambda story_id: StorySummary(
        story_id=story_id,
        title=story_id.replace("_", " ").title(),
        genre="test genre",
        premise="Test premise.",
        player_primer="Test briefing.",
        recommended_players="1-2 players",
        play_guidance="Claim a seat and play from its viewpoint.",
        playable_seat_count=2,
    )
    engine.list_story_summaries.side_effect = lambda: [
        engine.story_summary(story_id)
        for story_id in engine.list_story_ids.return_value
    ]
    engine.turn_history.return_value = []
    engine.takeover = AsyncMock()
    engine.join_player_character = AsyncMock(side_effect=lambda *args, **kwargs: (
        PlayerJoinResult(
            character_id=args[1],
            character_name=next(
                summary.name
                for summary in engine.list_session_characters.return_value
                if summary.character_id == args[1]
            ),
            pre_play=True,
        )
    ))
    engine.leave_character = AsyncMock(return_value="")
    engine.unbind_user = AsyncMock()
    engine.build_character_dossier = MagicMock(return_value="# Dossier · Sera")
    engine.set_character_identity = AsyncMock(return_value=_empty_ckpt(bindings))
    engine.get_bound_character_record = MagicMock()
    engine.create_player_character_simple = AsyncMock()
    engine.attach_dndbeyond_character_export = AsyncMock(
        return_value=_attachment_summary(),
    )
    engine.run_turn = AsyncMock()
    engine.run_query = AsyncMock(return_value=_turn_response(
        beat_ended_reason="query_response",
        turn_index=2,
        output_text="query narration",
        per_player_renders={"aldric": "query narration"},
    ))
    engine.run_begin_turn = AsyncMock(return_value=_turn_response(
        beat_ended_reason="state_change",
        turn_index=1,
        output_text="opening narration",
        per_player_renders={"aldric": "Aldric wakes."},
    ))
    engine.image_sidecar.config.diffusion_enabled = False
    engine.image_generation.wait_for_render_images = AsyncMock(
        return_value=True,
    )
    engine.opening_lobby.return_value = OpeningLobbyView(
        requires_confirmation=False,
        claimed_seat_names=(),
        open_seat_names=("Aldric", "Sera Vance"),
    )
    engine.session_activity.return_value = SessionActivityView(
        session_id=SESSION_ID,
        story_id=STORY_ID,
        turn_index=0,
        state="Open table: any joined player may act.",
    )
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


class _FakeAssetImageRenderer:
    def __init__(self, *, displayed: bool = True) -> None:
        self.displayed = displayed
        self.prepare_calls = []
        self.rendered_items = []

    def prepare_reveals(self, response, *, ckpt, session_id, character_ids):
        self.prepare_calls.append({
            "response": response,
            "ckpt": ckpt,
            "session_id": session_id,
            "character_ids": set(character_ids),
        })
        return {
            cid: [
                CliImageDisplayResult(
                    pov_character_id=cid,
                    displayed=self.displayed,
                    degraded=not self.displayed,
                    error_code="" if self.displayed else "unsupported_terminal",
                )
            ]
            for cid in sorted(character_ids)
        }

    def render_prepared(self, item):
        self.rendered_items.append(item)
        return item

    def prepare_generated(
        self,
        media,
        *,
        session_id: str,
        pov_character_id: str,
        cache_root=None,
    ):
        self.prepare_calls.append({
            "media": media,
            "session_id": session_id,
            "pov_character_id": pov_character_id,
            "cache_root": cache_root,
        })
        return CliImageDisplayResult(
            pov_character_id=pov_character_id,
            displayed=self.displayed,
            degraded=not self.displayed,
            error_code="" if self.displayed else "unsupported_terminal",
        )


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
        state.current_actor = "aldric"

        run(state.handle_line("/history"))

        out = capsys.readouterr().out
        assert "--- Turn 44 ---" in out
        assert "--- Turn 47 ---" in out
        assert "--- Turn 1 ---" not in out
        assert "> I swing." in out
        engine.turn_history.assert_called_once_with(SESSION_ID, "aldric")

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
        state.current_actor = "aldric"

        run(state.handle_line("/history 1"))

        out = capsys.readouterr().out
        assert "--- Turn 44 ---" not in out
        assert "--- Turn 47 ---" in out


class TestNumberedSelectionRefs:
    def test_story_list_numbers_and_start_accepts_number(self, run, capsys):
        engine = _mock_engine()
        engine.list_story_ids.return_value = ["spring_rain", "starfall"]
        state = CLIState(engine, SESSION_ID, "")

        run(state.handle_line("/story list"))
        out = capsys.readouterr().out

        assert "1: Spring Rain (`spring_rain`)" in out
        assert "2: Starfall (`starfall`)" in out

        run(state.handle_line("/story start 2"))

        engine.load_story_into_session.assert_called_once_with(
            SESSION_ID,
            "starfall",
        )
        assert state.story_id == "starfall"

    def test_story_info_accepts_number(self, run):
        engine = _mock_engine()
        engine.list_story_ids.return_value = ["spring_rain", "starfall"]
        state = CLIState(engine, SESSION_ID, "")

        run(state.handle_line("/story info 1"))

        engine.story_summary.assert_called_with("spring_rain")
        engine.list_story_characters.assert_called_once_with("spring_rain")

    def test_character_list_alias_and_consumers_accept_numbers(
        self, run, capsys,
    ):
        engine = _mock_engine()
        ckpt = _empty_ckpt()
        ckpt.characters = [
            _character("aldric", "Aldric", role="wanderer", is_playable=True),
            _character("sera", "Sera Vance", role="thief"),
        ]
        engine.load_latest.return_value = ckpt
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/character list"))
        out = capsys.readouterr().out

        assert "1: Aldric - wanderer" in out
        assert "2: Sera Vance - thief" in out
        assert "`aldric`" not in out
        assert "`sera`" not in out

        run(state.handle_line("/join 2"))
        engine.join_player_character.assert_awaited_with(
            SESSION_ID,
            "sera",
            1,
            name="",
            appearance="",
        )
        capsys.readouterr()

        run(state.handle_line("/character 2"))
        engine.build_character_dossier.assert_called_with(SESSION_ID, "sera")

        run(state.handle_line("/as 2"))
        assert state.current_actor == "sera"


class TestJoinLeave:
    def test_join_without_target_lists_open_playables(self, run, capsys):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join"))

        engine.join_player_character.assert_not_awaited()
        out = capsys.readouterr().out
        assert "## Joinable" in out
        assert "1: Aldric — wanderer" in out
        assert "2: Sera Vance — thief" in out
        assert "aldric —" not in out
        assert "sera —" not in out
        assert "Custom character: /join_custom" in out

    def test_join_binds_and_sets_current(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join sera"))
        engine.join_player_character.assert_awaited_once_with(
            SESSION_ID,
            "sera",
            1,
            name="",
            appearance="",
        )
        assert state.claims == {"sera": 1}
        assert state.current_actor == "sera"

    def test_oneshot_player_authored_join_requires_complete_identity(
        self, run, capsys,
    ):
        engine = _mock_engine()
        engine.list_session_characters.return_value = [CharacterSummary(
            character_id="blank_arrival",
            name="the Newcomer",
            role="new arrival",
            faction="",
            appearance="",
            status="dormant",
            is_playable=True,
            bound_user_id="",
            player_slot_kind="player_authored",
        )]
        state = CLIState(engine, SESSION_ID, STORY_ID)
        state.one_shot_mode = True

        run(state.handle_line("/join blank_arrival --name Mara"))

        engine.join_player_character.assert_not_awaited()
        out = capsys.readouterr().out
        assert "requires both identity fields" in out
        assert "--appearance" in out

    def test_oneshot_player_authored_join_submits_identity_atomically(
        self, run,
    ):
        engine = _mock_engine()
        engine.list_session_characters.return_value = [CharacterSummary(
            character_id="blank_arrival",
            name="the Newcomer",
            role="new arrival",
            faction="",
            appearance="",
            status="dormant",
            is_playable=True,
            bound_user_id="",
            player_slot_kind="player_authored",
        )]
        engine.join_player_character.return_value = PlayerJoinResult(
            character_id="blank_arrival",
            character_name="Mara Vale",
            pre_play=True,
        )
        state = CLIState(engine, SESSION_ID, STORY_ID)
        state.one_shot_mode = True

        run(state.handle_line(
            '/join blank_arrival --name "Mara Vale" '
            '--appearance "scarlet coat and iron-gray braid"'
        ))

        engine.join_player_character.assert_awaited_once_with(
            SESSION_ID,
            "blank_arrival",
            1,
            name="Mara Vale",
            appearance="scarlet coat and iron-gray braid",
        )
        assert state.claims == {"blank_arrival": 1}

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
        engine.join_player_character.reset_mock()
        run(state.handle_line("/join sera"))
        engine.join_player_character.assert_not_awaited()

    def test_leave_default_is_current_actor(self, run):
        engine = _mock_engine()
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join sera"))
        run(state.handle_line("/leave"))
        engine.leave_character.assert_awaited_once_with(SESSION_ID, 1)
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
        assert "## Available" in out
        assert "1: Intended Protagonist - defective hero" in out
        assert "3: Dead Queen" in out
        assert "## Yours\n  (none)" in out
        assert "## Claimed by another player" in out
        assert "2: Claimed Slot - claimed hero" in out
        assert "Guild Master" not in out
        assert "player_protagonist" not in out
        assert "guild_master" not in out
        assert "claimed_slot" not in out
        assert "dead_queen" not in out


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
    def test_loot_help_and_list_use_numbered_offers(self, run, capsys):
        engine = _mock_engine()
        engine.list_loot_offers.return_value = [
            _loot_offer(source_label="iron chest"),
        ]
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("/loot --help"))
        run(state.handle_line("/loot"))

        out = capsys.readouterr().out
        assert "Loot commands:" in out
        assert "1. iron chest" in out
        assert "loot_evt_chest" not in out
        assert "/loot take <offer>" in out

    def test_loot_take_accepts_number_and_defaults_all(self, run, capsys):
        engine = _mock_engine()
        engine.list_loot_offers.return_value = [_loot_offer()]
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("/loot take 1"))

        engine.claim_loot.assert_awaited_once_with(
            session_id=SESSION_ID,
            user_id=1,
            character_id="aldric",
            offer_id="loot_evt_chest",
            item_ids=[],
            take_currency=True,
            take_all_available=True,
        )

    def test_loot_item_ref_inspects_offer(self, run, capsys):
        engine = _mock_engine()
        engine.list_loot_offers.return_value = [
            _loot_offer(item_id="split_arch_hostel_voucher"),
        ]
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("/loot split_arch_hostel_voucher"))

        out = capsys.readouterr().out
        assert "--- Loot Offer ---" in out
        assert "split_arch_hostel_voucher" in out
        assert "unknown loot subcommand" not in out

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
        assert "--- Story update 1 · viewed as Aldric ---" in out
        assert "opening narration" in out

    def test_begin_error_is_player_safe(self, run, capsys):
        engine = _mock_engine()
        engine.run_begin_turn = AsyncMock(side_effect=RuntimeError(
            "openai.BadRequestError: /home/dan/ayoa/app/llm/client.py",
        ))
        state = CLIState(engine, SESSION_ID, STORY_ID)

        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("/begin"))

        out = capsys.readouterr().out
        assert "internal error" in out
        assert "BadRequestError" not in out
        assert "/home/dan" not in out

    def test_claim_sensitive_opening_lists_lobby_before_confirmation(
        self, run, capsys,
    ):
        engine = _mock_engine()
        engine.opening_lobby.return_value = OpeningLobbyView(
            requires_confirmation=True,
            claimed_seat_names=("Aldric",),
            open_seat_names=("Sera Vance",),
        )
        state = CLIState(engine, SESSION_ID, STORY_ID)
        state.one_shot_mode = True

        run(state.handle_line("/begin"))

        engine.run_begin_turn.assert_not_awaited()
        out = capsys.readouterr().out
        assert "claimed seats: Aldric" in out
        assert "still open: Sera Vance" in out
        assert "/begin --confirm" in out

    def test_explicit_begin_confirmation_opens_claim_sensitive_story(
        self, run,
    ):
        engine = _mock_engine()
        engine.opening_lobby.return_value = OpeningLobbyView(
            requires_confirmation=True,
            claimed_seat_names=("Aldric",),
            open_seat_names=("Sera Vance",),
        )
        state = CLIState(engine, SESSION_ID, STORY_ID)
        state.claims = {"aldric": 1}
        state.current_actor = "aldric"
        state.one_shot_mode = True

        run(state.handle_line("/begin --confirm"))

        engine.run_begin_turn.assert_awaited_once_with(
            session_id=SESSION_ID,
            triggering_character_id="aldric",
        )


class TestStatusCommand:
    def test_status_uses_shared_activity_without_internal_ids(
        self, run, capsys,
    ):
        engine = _mock_engine(bindings={"aldric": "1", "sera": "2"})
        engine.session_activity.return_value = SessionActivityView(
            session_id=SESSION_ID,
            story_id=STORY_ID,
            turn_index=8,
            state="requested next: Sera Vance (advisory)",
            viewpoint_name="Aldric",
            location="gatehouse",
            joined_seat_names=("Aldric", "Sera Vance"),
            nearby_character_names=("Pip",),
            requested_next_names=("Sera Vance",),
            last_visible_update="A bell rings.",
        )
        state = CLIState(engine, SESSION_ID, STORY_ID)
        state.current_actor = "aldric"

        run(state.handle_line("/status"))

        engine.session_activity.assert_called_once_with(SESSION_ID, "aldric")
        out = capsys.readouterr().out
        assert "viewpoint: Aldric" in out
        assert "joined: Aldric, Sera Vance" in out
        assert "activity: requested next: Sera Vance (advisory)" in out
        assert "discord" not in out.lower()
        assert "user id" not in out.lower()


class TestCombatCommand:
    def test_combat_begin_parser_preserves_comma_or_quoted_names(self):
        assert _split_combat_ids("alice guard") == ["alice", "guard"]
        assert _split_combat_ids("alice, Herrik Voss") == [
            "alice",
            "Herrik Voss",
        ]
        assert _split_combat_ids('alice "Herrik Voss"') == [
            "alice",
            "Herrik Voss",
        ]

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
        assert "Initiative: Sera 18, Aldric 13." in out
        assert "Current turn: Sera." in out
        assert "initiating action has not resolved before initiative" not in out
        assert "(sera)" not in out

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
    def test_cli_damage_roll_display_uses_damage_formula_not_d20(
        self, capsys,
    ):
        _print_dice_roll_displays([
            DiceRollDisplay(
                actor_id="mon_mountain_lion_1",
                actor_name="Mountain Lion",
                target_id="pc_expedition_leader",
                target_name="Demo Expedition Leader",
                label="Damage (Claw)",
                kind="damage_roll",
                modifier=2,
                total=4,
                damage_raw_total=4,
                damage_total=4,
                damage_type="slashing",
                damage_expression="1d4+2",
                damage_detail="1d4 (2) + 2 = `4`",
                target_hp_before=33,
                target_hp_after=29,
                target_hp_max=38,
                target_defeat_state="active",
            )
        ])

        out = capsys.readouterr().out
        assert "--- D&D Damage · Mountain Lion: Damage (Claw)" in out
        assert "d20 ?" not in out
        assert "Damage: 1d4 (2) + 2 = 4 slashing" in out
        assert "Target HP: Demo Expedition Leader 33/38 -> 29/38" in out

    def test_cli_damage_roll_display_has_total_only_fallback(self, capsys):
        _print_dice_roll_displays([
            DiceRollDisplay(
                actor_id="trap",
                actor_name="Trap",
                target_id="aldric",
                target_name="Aldric",
                label="Damage",
                kind="damage_roll",
                total=6,
                damage_total=6,
                damage_type="fire",
            )
        ])

        out = capsys.readouterr().out
        assert "d20 ?" not in out
        assert "Damage: 6 fire" in out

    def test_cli_healing_roll_display_uses_healing_formula(self, capsys):
        _print_dice_roll_displays([
            DiceRollDisplay(
                actor_id="ilyra",
                actor_name="Ilyra",
                target_id="marlowe",
                target_name="Marlowe Vex",
                label="Healing Word",
                kind="healing_roll",
                total=6,
                expression="1d4+3",
                detail="1d4 (3) + 3 = `6`",
                target_hp_before=9,
                target_hp_after=15,
                target_hp_max=24,
                target_defeat_state="active",
            )
        ])

        out = capsys.readouterr().out
        assert "--- D&D Healing · Ilyra: Healing Word" in out
        assert "d20 ?" not in out
        assert "Healing: 1d4 (3) + 3 = 6" in out
        assert "Target HP: Marlowe Vex 9/24 -> 15/24" in out

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

    def test_roll_all_uses_joined_rolls_when_current_actor_has_none(
        self, run, capsys,
    ):
        engine = _mock_engine(bindings={"aldric": "1", "sera": "2"})
        prompt = PendingRollPrompt(
            session_id=SESSION_ID,
            event_id="evt_sera",
            roll_id="roll_stealth",
            actor_id="sera",
            user_id="2",
            label="Stealth",
            reason="Slip behind the pillar.",
        )
        completed = False

        def pending_rolls(session_id, *, user_id=None):
            assert session_id == SESSION_ID
            if completed:
                return []
            return [prompt] if user_id == 2 else []

        async def complete_roll(**kwargs):
            nonlocal completed
            completed = True
            assert kwargs == {
                "session_id": SESSION_ID,
                "event_id": "evt_sera",
                "roll_id": "roll_stealth",
                "user_id": 2,
            }
            return CompletedPendingRoll(
                session_id=SESSION_ID,
                event_id="evt_sera",
                roll_id="roll_stealth",
                actor_id="sera",
                user_id="2",
                label="Stealth",
                reason="Slip behind the pillar.",
                expression="1d20+6",
                total=18,
                detail="1d20 (12) + 6 = `18`",
                crit="none",
                remaining_pending_rolls=0,
            )

        engine.pending_roll_prompts.side_effect = pending_rolls
        engine.complete_pending_roll.side_effect = complete_roll
        engine.continue_pending_roll.return_value = _turn_response(
            turn_index=5,
            output_text="Sera ducks back into shadow.",
        )

        state = CLIState(engine, SESSION_ID, STORY_ID)
        assert state.current_actor == "aldric"
        run(state.handle_line("/roll all"))

        engine.continue_pending_roll.assert_awaited_once_with(
            session_id=SESSION_ID,
            event_id="evt_sera",
            actor_id="sera",
        )
        out = capsys.readouterr().out
        assert "Rolled Stealth:" in out
        assert "Sera ducks back into shadow." in out


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

    def test_plain_text_strips_terminal_control_bytes(self, run):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(return_value=_turn_response())

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        run(state.handle_line("I look\x1b[2J around\x07"))

        assert engine.run_turn.await_args.kwargs["user_input"] == "I look around"

    def test_query_uses_bridge_and_prints_turn_response(self, run, capsys):
        engine = _mock_engine()
        engine.run_query = AsyncMock(return_value=_turn_response(
            beat_ended_reason="query_response",
            turn_index=4,
            output_text="The crest is weathered silver.",
            per_player_renders={"aldric": "The crest is weathered silver."},
        ))

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("/query what does the crest look like?"))

        engine.run_query.assert_awaited_once_with(
            session_id=SESSION_ID,
            character_id="aldric",
            question="what does the crest look like?",
        )
        engine.run_turn.assert_not_awaited()
        out = capsys.readouterr().out
        assert "--- Story update 4 · viewed as Aldric ---" in out
        assert "The crest is weathered silver." in out

    def test_disabled_diffusion_does_not_announce_or_wait_for_illustration(
        self, run, capsys,
    ):
        engine = _mock_engine()
        engine.run_query = AsyncMock(return_value=_turn_response(
            beat_ended_reason="query_response",
            turn_index=4,
            output_text="The crest is weathered silver.",
            per_player_renders={"aldric": "The crest is weathered silver."},
            rendered_event_ids_by_pov={"aldric": ["evt_crest"]},
        ))
        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        capsys.readouterr()

        run(state.handle_line("/query what does the crest look like?"))

        engine.image_generation.wait_for_render_images.assert_not_awaited()
        assert "illustrating" not in capsys.readouterr().out

    def test_unsupported_terminal_does_not_wait_for_illustration(
        self, run, capsys,
    ):
        engine = _mock_engine()
        engine.image_sidecar.config.diffusion_enabled = True
        engine.run_query = AsyncMock(return_value=_turn_response(
            beat_ended_reason="query_response",
            turn_index=4,
            output_text="The crest is weathered silver.",
            per_player_renders={"aldric": "The crest is weathered silver."},
            rendered_event_ids_by_pov={"aldric": ["evt_crest"]},
        ))
        state = CLIState(engine, SESSION_ID, STORY_ID)
        engine.image_generation.can_accept_render.return_value = False
        run(state.handle_line("/join aldric"))
        capsys.readouterr()

        run(state.handle_line("/query what does the crest look like?"))

        engine.image_generation.wait_for_render_images.assert_not_awaited()
        assert "illustrating" not in capsys.readouterr().out

    def test_turn_response_prints_per_pov_asset_reveals_for_claimed_characters(
        self, run, capsys,
    ):
        engine = _mock_engine()
        image_renderer = _FakeAssetImageRenderer()
        engine.run_turn = AsyncMock(return_value=_turn_response(
            turn_index=4,
            output_text="Aldric sees the western door.",
            per_player_renders={
                "aldric": "Aldric sees the western door.",
                "sera": "Sera sees the eastern alcove.",
            },
            asset_reveals=[SimpleNamespace(delivery_ref="asset://leaked/merged")],
            per_player_asset_reveals={
                "aldric": [SimpleNamespace(delivery_ref="asset://pack/aldric")],
                "sera": [SimpleNamespace(delivery_ref="asset://pack/sera")],
                "unclaimed": [SimpleNamespace(delivery_ref="asset://pack/hidden")],
            },
        ))

        state = CLIState(
            engine,
            SESSION_ID,
            STORY_ID,
            asset_image_renderer=image_renderer,
        )
        run(state.handle_line("/join aldric"))
        run(state.handle_line("/join sera"))
        capsys.readouterr()
        run(state.handle_line("I look around."))

        assert image_renderer.prepare_calls[0]["character_ids"] == {
            "aldric",
            "sera",
        }
        out = capsys.readouterr().out
        assert "--- Image Reveal · aldric ---" in out
        assert "--- Image Reveal · sera ---" in out
        assert "--- Image Reveal · unclaimed ---" not in out
        assert out.count("Displayed revealed image.") == 2
        assert "asset://pack" not in out
        assert "asset://leaked" not in out

    def test_unsupported_image_backend_is_degraded_not_success(
        self, run, capsys,
    ):
        engine = _mock_engine()
        image_renderer = _FakeAssetImageRenderer(displayed=False)
        engine.run_turn = AsyncMock(return_value=_turn_response(
            turn_index=4,
            output_text="Aldric sees a handout.",
            per_player_asset_reveals={
                "aldric": [SimpleNamespace(delivery_ref="asset://pack/handout")],
            },
        ))
        state = CLIState(
            engine,
            SESSION_ID,
            STORY_ID,
            asset_image_renderer=image_renderer,
        )
        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("I inspect it."))

        out = capsys.readouterr().out
        assert "Image reveal could not be displayed in this terminal." in out
        assert "Displayed revealed image." not in out
        assert "asset://pack" not in out

    def test_generated_image_delivery_ignores_other_sessions(self, run):
        engine = _mock_engine()
        image_renderer = _FakeAssetImageRenderer()
        state = CLIState(
            engine,
            SESSION_ID,
            STORY_ID,
            asset_image_renderer=image_renderer,
        )
        state.claims = {"aldric": 1}
        state.current_actor = "aldric"
        job = SimpleNamespace(
            request=SimpleNamespace(
                session_id="other_session",
                title="Stale Image",
            ),
        )
        delivery = SimpleNamespace(
            session_id="other_session",
            pov_character_id="aldric",
        )

        delivered = run(state._deliver_cli_image(
            job,
            delivery,
            SimpleNamespace(),
            "",
        ))

        assert delivered is False
        assert image_renderer.prepare_calls == []
        assert image_renderer.rendered_items == []

    def test_run_turn_error_is_player_safe(self, run, capsys):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(side_effect=RuntimeError(
            "openai.BadRequestError: /home/dan/ayoa/app/llm/client.py",
        ))

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("I look around"))

        out = capsys.readouterr().out
        assert "internal error" in out
        assert "BadRequestError" not in out
        assert "/home/dan" not in out

    def test_pre_turn_resolution_header_is_player_facing(self, run, capsys):
        engine = _mock_engine()
        engine.run_turn = AsyncMock(return_value=_turn_response(
            beat_ended_reason="pre_turn_resolution",
            output_text="Submit again.",
            pre_turn_resolutions=[
                _turn_response(
                    per_player_renders={"aldric": "The rat darts away."},
                )
            ],
        ))

        state = CLIState(engine, SESSION_ID, STORY_ID)
        run(state.handle_line("/join aldric"))
        capsys.readouterr()
        run(state.handle_line("I swing."))

        out = capsys.readouterr().out
        assert "--- Earlier story update · viewed as Aldric ---" in out
        assert "POV aldric" not in out

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

    def test_describe_preplay_does_not_begin(self, run, monkeypatch):
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
        engine.set_character_identity.assert_awaited_once_with(
            SESSION_ID,
            "aldric",
            name=None,
            appearance="tall and weary",
        )
        engine.run_begin_turn.assert_not_awaited()
        engine.run_turn.assert_not_awaited()


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
        from app.engine.frontend_views import RewindResult

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


class TestOneShotAndSessionLock:
    """Non-interactive one-shot mode + cross-process session lock: the
    enabling surface for separate-terminal multiplayer."""

    def test_session_lock_creates_lockfile_and_reenters(self, tmp_path):
        # Sequential (re)entry must not error and should leave a lockfile in
        # the session dir. Creating the dir lazily is part of the contract.
        for _ in range(2):
            with _session_command_lock(tmp_path, "sess"):
                pass
        assert (tmp_path / "sess" / ".session.lock").exists()

    def test_oneshot_rejects_unbound_actor(self, run, tmp_path):
        engine = _mock_engine(bindings=None)
        state = CLIState(engine, SESSION_ID, STORY_ID)
        state.handle_line = AsyncMock()

        code = run(run_oneshot_commands(
            state,
            sessions_dir=tmp_path,
            session_id=SESSION_ID,
            commands=["I look around"],
            act_as="ren_sato",
        ))

        assert code == 2
        state.handle_line.assert_not_awaited()

    def test_oneshot_runs_commands_as_bound_actor(self, run, tmp_path):
        engine = _mock_engine(bindings={"aldric": "1"})
        state = CLIState(engine, SESSION_ID, STORY_ID)
        assert "aldric" in state.claims
        state.handle_line = AsyncMock()

        code = run(run_oneshot_commands(
            state,
            sessions_dir=tmp_path,
            session_id=SESSION_ID,
            commands=["I steady my hands", "/status"],
            act_as="Aldric",
        ))

        assert code == 0
        assert state.current_actor == "aldric"
        assert state.pov_filter == "aldric"
        assert [c.args[0] for c in state.handle_line.await_args_list] == [
            "I steady my hands",
            "/status",
        ]

    def test_oneshot_requires_startup_actor_for_multiple_claims(
        self, run, tmp_path, capsys,
    ):
        engine = _mock_engine(bindings={"aldric": "1", "sera": "2"})
        state = CLIState(engine, SESSION_ID, STORY_ID)
        state.handle_line = AsyncMock()

        code = run(run_oneshot_commands(
            state,
            sessions_dir=tmp_path,
            session_id=SESSION_ID,
            commands=["I look around"],
        ))

        assert code == 2
        state.handle_line.assert_not_awaited()
        assert "--as is required" in capsys.readouterr().err

    def test_oneshot_rejects_in_batch_actor_switch(
        self, run, tmp_path, capsys,
    ):
        engine = _mock_engine(bindings={"aldric": "1", "sera": "2"})
        state = CLIState(engine, SESSION_ID, STORY_ID)
        state.handle_line = AsyncMock()

        code = run(run_oneshot_commands(
            state,
            sessions_dir=tmp_path,
            session_id=SESSION_ID,
            commands=["/as sera", "I look around"],
            act_as="aldric",
        ))

        assert code == 2
        state.handle_line.assert_not_awaited()
        assert "startup --as" in capsys.readouterr().err

    def test_oneshot_refreshes_bindings_after_lock(
        self, run, tmp_path,
    ):
        engine = _mock_engine(bindings={"aldric": "1"})
        state = CLIState(engine, SESSION_ID, STORY_ID)
        engine.load_latest.return_value = _empty_ckpt({"sera": "9"})
        state.handle_line = AsyncMock()

        code = run(run_oneshot_commands(
            state,
            sessions_dir=tmp_path,
            session_id=SESSION_ID,
            commands=["I look around"],
            act_as="Sera Vance",
        ))

        assert code == 0
        assert state.claims == {"sera": 9}
        assert state.current_actor == "sera"
        state.handle_line.assert_awaited_once_with("I look around")

    def test_pov_filter_scopes_printed_claims(self):
        engine = _mock_engine(bindings={"aldric": "1", "sera": "2"})
        state = CLIState(engine, SESSION_ID, STORY_ID)
        assert state._pov_claims() == {"aldric", "sera"}
        state.pov_filter = "aldric"
        assert state._pov_claims() == {"aldric"}


class TestCommandLineStorySelection:
    def _engine(self, tmp_path):
        engine = _mock_engine()
        engine.sessions_dir = tmp_path
        engine.list_story_ids.return_value = ["spring_rain", "starfall"]
        return engine

    def test_loads_numbered_story_into_empty_session(self, tmp_path):
        engine = self._engine(tmp_path)
        engine.load_latest.side_effect = FileNotFoundError

        story_id, resumed = _prepare_session_story(
            engine,
            session_id=SESSION_ID,
            requested_story="2",
            announce=False,
        )

        assert story_id == "starfall"
        assert resumed is False
        engine.create_empty_session.assert_called_once_with(SESSION_ID)
        engine.load_story_into_session.assert_called_once_with(
            SESSION_ID,
            "starfall",
        )

    def test_same_story_resume_does_not_reload(self, tmp_path):
        engine = self._engine(tmp_path)
        ckpt = _empty_ckpt()
        ckpt.session.story_id = "spring_rain"
        engine.load_latest.return_value = ckpt

        story_id, resumed = _prepare_session_story(
            engine,
            session_id=SESSION_ID,
            requested_story="spring_rain",
            announce=False,
        )

        assert story_id == "spring_rain"
        assert resumed is True
        engine.create_empty_session.assert_not_called()
        engine.load_story_into_session.assert_not_called()

    def test_conflicting_story_is_rejected_without_mutation(self, tmp_path):
        engine = self._engine(tmp_path)
        ckpt = _empty_ckpt()
        ckpt.session.story_id = "spring_rain"
        engine.load_latest.return_value = ckpt

        with pytest.raises(ValueError, match="refusing to replace"):
            _prepare_session_story(
                engine,
                session_id=SESSION_ID,
                requested_story="starfall",
                announce=False,
            )

        engine.create_empty_session.assert_not_called()
        engine.load_story_into_session.assert_not_called()
