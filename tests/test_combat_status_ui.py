import pytest

from app.bot.commands import _render_combat_status
from app.bot.embed import MAX_DESCRIPTION
from app.bot.engine_bridge import EngineBridge
from app.engine.frontend_views import (
    DndCombatParticipantView,
    DndCombatView,
)
from app.schemas.characters import CharacterRecord
from app.schemas.checkpoint import CheckpointFile
from app.schemas.state import (
    DndCombatantState,
    DndCombatState,
    DndRuntimeEffect,
    SessionState,
    WorldState,
)


@pytest.fixture
def bridge(tmp_path, monkeypatch) -> EngineBridge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return EngineBridge(
        stories_dir=str(tmp_path / "stories"),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )


def test_bridge_combat_status_labels_active_effects_without_roll_details(
    bridge: EngineBridge,
):
    ckpt = CheckpointFile(
        session=SessionState(
            session_id="s",
            active_combat=DndCombatState(
                combatants=[
                    DndCombatantState(
                        combatant_id="alice",
                        character_id="alice",
                        name="Alice",
                        active_effects=[
                            DndRuntimeEffect(
                                effect_id="eff_bless",
                                name="Bless",
                                target_id="alice",
                                originator_id="alice",
                                concentration=True,
                                remaining_rounds=8,
                                recurring_save={
                                    "ability": "wis",
                                    "dc": 13,
                                    "timing": "end_of_turn",
                                    "ends_on": "success",
                                    "repeat": True,
                                },
                            ),
                        ],
                    ),
                ],
            ),
        ),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                mechanics={"hit_points": {"current": 10, "max": 12}},
            ),
        ],
        world_state=WorldState(),
    )
    bridge.checkpoint_mgr.save(ckpt)

    status = bridge.combat_status("s", private=True)

    alice = status.participants[0]
    assert alice.active_effects == ("Bless (concentration; 8 rounds)",)
    assert "eff_bless" not in alice.active_effects[0]
    assert "13" not in alice.active_effects[0]


def test_discord_combat_status_renders_active_effects():
    embed = _render_combat_status(
        DndCombatView(
            session_id="s",
            active=True,
            round_number=1,
            turn_number=1,
            current_participant_id="alice",
            participants=(
                DndCombatParticipantView(
                    character_id="alice",
                    name="Alice",
                    current=True,
                    hp_current=10,
                    hp_max=12,
                    armor_class=15,
                    active_effects=("Bless (concentration; 8 rounds)",),
                ),
            ),
        )
    )

    assert "Effects: Bless (concentration; 8 rounds)" in (embed.description or "")


def test_discord_combat_status_hides_battle_map_lines_by_default():
    embed = _render_combat_status(
        DndCombatView(
            session_id="s",
            active=True,
            participants=(
                DndCombatParticipantView(character_id="alice", name="Alice"),
            ),
            map_lines=("Battle map: Bridge (8x5, 5 ft squares).",),
        )
    )

    assert "Battle map: Bridge" not in (embed.description or "")


def test_discord_combat_status_renders_battle_map_lines_when_requested():
    embed = _render_combat_status(
        DndCombatView(
            session_id="s",
            active=True,
            participants=(
                DndCombatParticipantView(character_id="alice", name="Alice"),
            ),
            map_lines=("Battle map: Bridge (8x5, 5 ft squares).",),
        ),
        include_map=True,
    )

    assert "Battle map: Bridge" in (embed.description or "")


def test_discord_combat_status_truncates_large_battle_map_lines():
    embed = _render_combat_status(
        DndCombatView(
            session_id="s",
            active=True,
            participants=(
                DndCombatParticipantView(character_id="alice", name="Alice"),
            ),
            map_lines=tuple(
                f"Map row {index}: " + ("x" * 500)
                for index in range(20)
            ),
        ),
        include_map=True,
    )

    description = embed.description or ""
    assert len(description) <= MAX_DESCRIPTION
    assert "... map truncated." in description
