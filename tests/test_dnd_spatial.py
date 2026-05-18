from app.engine import dnd_combat, dnd_spatial
from app.schemas.dnd_spatial import (
    DndAreaTemplate,
    DndBattleMapState,
    DndBattleMapToken,
    DndSpatialDelta,
    MAX_BATTLE_MAP_HEIGHT,
    MAX_BATTLE_MAP_WIDTH,
)
from app.schemas.state import DndCombatantState, DndCombatState, SessionState


def _combatant(
    combatant_id: str,
    *,
    character_id: str | None = None,
    name: str | None = None,
) -> DndCombatantState:
    return DndCombatantState(
        combatant_id=combatant_id,
        character_id=character_id or combatant_id,
        name=name or combatant_id,
    )


def test_normalize_caps_oversized_seed_and_fills_missing_tokens():
    combatants = [_combatant("alice"), _combatant("bob")]
    seed = DndBattleMapState(
        present=True,
        map_name="Huge Hall",
        width=1_000_000,
        height=1_000_000,
        tokens=[],
    )

    battle_map = dnd_spatial.normalize_battle_map_seed(seed, combatants)

    assert battle_map is not None
    assert battle_map.width == MAX_BATTLE_MAP_WIDTH
    assert battle_map.height == MAX_BATTLE_MAP_HEIGHT
    assert {
        (token.token_id, token.character_id)
        for token in battle_map.tokens
    } == {("alice", "alice"), ("bob", "bob")}


def test_duplicate_character_combatants_keep_distinct_tokens():
    combatants = [
        _combatant("wolf_a", character_id="wolf", name="Wolf A"),
        _combatant("wolf_b", character_id="wolf", name="Wolf B"),
    ]
    seed = DndBattleMapState(
        present=True,
        map_name="Kennel",
        width=8,
        height=8,
        tokens=[
            DndBattleMapToken(
                token_id="wolf_a",
                character_id="wolf",
                label="Wolf A",
                x=1,
                y=1,
            ),
            DndBattleMapToken(
                token_id="wolf_b",
                character_id="wolf",
                label="Wolf B",
                x=2,
                y=1,
            ),
        ],
    )
    battle_map = dnd_spatial.normalize_battle_map_seed(seed, combatants)
    combat = DndCombatState(combatants=combatants, battle_map=battle_map)

    dnd_spatial.apply_spatial_deltas(combat, [
        DndSpatialDelta(
            kind="move_token",
            target_id="wolf_b",
            character_id="wolf",
            x=5,
            y=5,
        )
    ])

    positions = {
        token.token_id: (token.character_id, token.x, token.y)
        for token in combat.battle_map.tokens
    }
    assert positions == {
        "wolf_a": ("wolf", 1, 1),
        "wolf_b": ("wolf", 5, 5),
    }


def test_remove_token_uses_character_id_when_unambiguous():
    combat = DndCombatState(
        combatants=[_combatant("alice"), _combatant("bob")],
        battle_map=DndBattleMapState(
            present=True,
            map_name="Bridge",
            width=8,
            height=5,
            tokens=[
                DndBattleMapToken(
                    token_id="alice",
                    character_id="alice",
                    label="Alice",
                    x=0,
                    y=0,
                ),
                DndBattleMapToken(
                    token_id="bob",
                    character_id="bob",
                    label="Bob",
                    x=3,
                    y=0,
                ),
            ],
        ),
    )

    dnd_spatial.apply_spatial_deltas(combat, [
        DndSpatialDelta(
            kind="remove_token",
            character_id="alice",
        )
    ])

    assert [token.token_id for token in combat.battle_map.tokens] == ["bob"]


def test_large_token_distance_uses_occupied_footprints():
    combat = DndCombatState(
        combatants=[_combatant("ogre"), _combatant("bob")],
        battle_map=DndBattleMapState(
            present=True,
            map_name="Yard",
            width=6,
            height=4,
            tokens=[
                DndBattleMapToken(
                    token_id="ogre",
                    character_id="ogre",
                    label="Ogre",
                    x=0,
                    y=0,
                    size_squares=2,
                ),
                DndBattleMapToken(
                    token_id="bob",
                    character_id="bob",
                    label="Bob",
                    x=2,
                    y=0,
                    size_squares=1,
                ),
            ],
        ),
    )

    [advisory] = dnd_spatial.spatial_advisories(combat, "ogre")

    assert advisory["to"] == "bob"
    assert advisory["distance_ft"] == 5
    assert advisory["within_5_ft"] is True


def test_area_duration_expires_on_turn_advance():
    combat = DndCombatState(
        combatants=[_combatant("alice"), _combatant("bob")],
        battle_map=DndBattleMapState(
            present=True,
            map_name="Bridge",
            width=8,
            height=5,
            tokens=[],
            areas=[
                DndAreaTemplate(
                    template_id="fog",
                    label="Fog Cloud",
                    shape="circle",
                    x=2,
                    y=2,
                    duration_rounds=1,
                )
            ],
        ),
    )
    session = SessionState(session_id="s", active_combat=combat)

    dnd_combat.advance_turn(session)

    assert combat.battle_map.areas == []
    assert "Area expired: Fog Cloud." in combat.audit_lines


def test_area_targeting_advisory_ranks_cone_by_enemy_targets():
    combat = DndCombatState(
        combatants=[
            _combatant("caster"),
            _combatant("enemy_a"),
            _combatant("enemy_b"),
            _combatant("enemy_c"),
            _combatant("ally"),
        ],
        battle_map=DndBattleMapState(
            present=True,
            map_name="Hall",
            width=8,
            height=5,
            tokens=[
                DndBattleMapToken(
                    token_id="caster",
                    character_id="caster",
                    label="Caster",
                    x=1,
                    y=2,
                ),
                DndBattleMapToken(
                    token_id="enemy_a",
                    character_id="enemy_a",
                    label="Enemy A",
                    x=2,
                    y=2,
                ),
                DndBattleMapToken(
                    token_id="enemy_b",
                    character_id="enemy_b",
                    label="Enemy B",
                    x=3,
                    y=2,
                ),
                DndBattleMapToken(
                    token_id="enemy_c",
                    character_id="enemy_c",
                    label="Enemy C",
                    x=3,
                    y=3,
                ),
                DndBattleMapToken(
                    token_id="ally",
                    character_id="ally",
                    label="Ally",
                    x=0,
                    y=2,
                ),
            ],
        ),
    )

    [advisory] = dnd_spatial.area_targeting_advisories(
        combat,
        "caster",
        [{
            "action_id": "cone_of_cold",
            "name": "Cone of Cold",
            "shape": "cone",
            "length_ft": 15,
        }],
        relationships_by_id={
            "enemy_a": "enemy",
            "enemy_b": "enemy",
            "enemy_c": "enemy",
            "ally": "ally",
            "caster": "self",
        },
    )

    top = advisory["candidates"][0]
    assert top["direction"] == "E"
    assert set(top["enemy_targets"]) == {"enemy_a", "enemy_b", "enemy_c"}
    assert top["ally_targets"] == []
