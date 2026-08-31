"""Smoke tests for the hand-authored One-Star Ascension (Pick Me Up) story seed.

Guards the runtime contracts the engine depends on for this floor-zero seed
rather than freezing prose: schema/ruleset shape, the router/Master/units role
split, a blank user-created player-character, the bounded-cast discipline, the
dormant summon pool of pre-authored candidates, and the router-facing reveals.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.engine.character_manager import _assemble_knowledge_grant
from app.engine.context_builder import (
    build_character_self_packet,
)
from app.engine.one_star_progression import (
    birth_hp_mean,
    birth_stat_total_mean,
    derive_progression_seed,
    experience_to_reach_level,
    rebalance_hero,
)
from app.engine.one_star_adapter import (
    load_one_star_account,
    one_star_opening_roster_preview,
)
from app.engine.reviewed_visual_references import validate_story_visual_references
from app.schemas.characters import CharacterAgentTier, PlayerSlotKind
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_COMBATANT_KEY,
    ONE_STAR_HERO_KEY,
    ONE_STAR_RULESET_ID,
    OneStarAccountEnvelope,
    OneStarCombatantState,
    OneStarHeroState,
)


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "storage"
    / "stories"
    / "one_star_ascension_s1"
    / "ckpt_0000.json"
)

EXPECTED_CHARACTER_IDS = {
    "one_star_newcomer",
    "renna_holt",
    "edren_marr",
    "iselle_the_guide",
    "the_master",
    "halcyon_of_the_gilded_march",
    "soren_ironvow",
    "castor_valebrand",
    "wren_thelantern",
    "rowan_kest",
    "liora_fen",
    "mirelle_voss",
    "seris_nightglass",
    "aveline_morcant",
    "veil_the_unnumbered",
    "warden_of_the_eighth",
}

# Only genuine off-stage movers may drive background fan-out; everyone else acts
# on-stage when observed (and the pool/Warden are dormant and out of play).
EXPECTED_INTENTION_MOVERS = {
    "the_master",
    "halcyon_of_the_gilded_march",
}

# Pre-authored characters held in reserve as a dormant, quarantined summon pool.
SUMMON_POOL_IDS = {
    "renna_holt",
    "edren_marr",
    "soren_ironvow",
    "castor_valebrand",
    "wren_thelantern",
    "rowan_kest",
    "liora_fen",
    "mirelle_voss",
    "seris_nightglass",
    "aveline_morcant",
    "veil_the_unnumbered",
}

# The blank, user-created player slot is intentionally empty (filled in play).
BLANK_PLAYER_ID = "one_star_newcomer"


def _load_checkpoint() -> CheckpointFile:
    raw = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return CheckpointFile.model_validate(raw)


def _actor_text(character: object) -> str:
    actor = getattr(character, "actor", None)
    if actor is None:
        return ""
    return "\n".join(fact.text for fact in actor.facts)


def _may_act_offstage(character: object) -> bool:
    actor = getattr(character, "actor", None)
    return bool(actor is not None and actor.may_act_offstage)


def test_checkpoint_loads_as_typed_one_star_story() -> None:
    checkpoint = _load_checkpoint()

    assert checkpoint.schema_version == "5.0"
    assert checkpoint.session.story_id == "one_star_ascension_s1"
    assert checkpoint.session.session_id == "one_star_ascension_s1"
    assert checkpoint.session.config.settings.ruleset_id == ONE_STAR_RULESET_ID
    assert checkpoint.session.config.settings.max_agent_cascades_per_beat == 5
    assert checkpoint.world_state.physics_ruleset.magic_enabled is True
    assert checkpoint.session.active_combat is None
    owners = [
        character for character in checkpoint.characters
        if ONE_STAR_ACCOUNT_KEY in character.mechanics
    ]
    assert [character.character_id for character in owners] == ["the_master"]
    account = OneStarAccountEnvelope.model_validate(
        owners[0].mechanics[ONE_STAR_ACCOUNT_KEY]
    )
    assert account.config.lobby_id == "niflheim"
    assert account.state.resources.gold == 40
    assert account.state.capacity == 20

    expected_hero_ids = EXPECTED_CHARACTER_IDS - {
        "iselle_the_guide",
        "the_master",
        "warden_of_the_eighth",
    }
    for character in checkpoint.characters:
        if character.character_id in expected_hero_ids:
            assert ONE_STAR_HERO_KEY in character.mechanics, character.character_id
            OneStarHeroState.model_validate(character.mechanics[ONE_STAR_HERO_KEY])
        else:
            assert ONE_STAR_HERO_KEY not in character.mechanics, character.character_id

    iselle = next(
        character
        for character in checkpoint.characters
        if character.character_id == "iselle_the_guide"
    )
    combatant = OneStarCombatantState.model_validate(
        iselle.mechanics[ONE_STAR_COMBATANT_KEY]
    )
    assert combatant.hp_current == combatant.hp_max == 2_000
    assert combatant.stats == {
        stat_id: 200 for stat_id in account.config.progression.stat_ids
    }


def test_opening_roster_guide_handoff_is_branch_specific() -> None:
    """The mixed opening follows the same guide gate as the Master trio."""
    expected = {
        "master_opening_roster": (
            True,
            ["renna_holt", "mirelle_voss", "edren_marr"],
            {},
        ),
        "master_newcomer_opening_roster": (
            True,
            [
                "renna_holt",
                "mirelle_voss",
                "edren_marr",
                BLANK_PLAYER_ID,
            ],
            {BLANK_PLAYER_ID: "player-1"},
        ),
        "newcomer_opening_roster": (
            False,
            [BLANK_PLAYER_ID],
            {BLANK_PLAYER_ID: "player-1"},
        ),
    }
    for pool_id, (requires_handoff, character_ids, bindings) in expected.items():
        branch = _load_checkpoint()
        branch.session.character_bindings = bindings
        _owner, branch_account = load_one_star_account(branch)
        pool = branch_account.config.summon_pools[pool_id]
        assert pool.initial_deployment_requires_guide_handoff is requires_handoff
        draws = one_star_opening_roster_preview(branch, pool_id)
        assert [draw.existing_character_id for draw in draws] == character_ids


def test_one_star_opens_with_authored_visual_novel_onboarding() -> None:
    checkpoint = _load_checkpoint()
    onboarding = checkpoint.visual_novel_onboarding

    assert checkpoint.session.config.settings.presentation_mode == "visual_novel"
    assert onboarding is not None
    assert onboarding.stage_reference_id == "osa_loc_1f_courtyard_v1"
    assert len(onboarding.pages) == 3
    assert [page.page.sprites for page in onboarding.pages] == [
        ["Iselle"],
        ["Iselle"],
        ["Iselle"],
    ]
    assert [
        page.sprite_variant_keys_by_label["Iselle"]
        for page in onboarding.pages
    ] == ["happy", "neutral", "happy"]
    assert {
        (choice.label, choice.character_id)
        for choice in onboarding.join_choices
    } == {
        ("Join as Master", "the_master"),
        ("Join as Newcomer", "one_star_newcomer"),
    }

    public_payload = json.loads(checkpoint.model_dump_json())
    private_payload = json.loads(checkpoint.model_dump_json(
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
    ))
    assert public_payload["visual_novel_onboarding"] is None
    assert private_payload["visual_novel_onboarding"]["stage_reference_id"] == (
        onboarding.stage_reference_id
    )
    assert {
        choice["character_id"]
        for choice in private_payload["visual_novel_onboarding"]["join_choices"]
    } == {"the_master", "one_star_newcomer"}


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload["visual_novel_onboarding"]["pages"][0][
                "page"
            ].__setitem__("text", "one_star_newcomer waits."),
            "pages expose a source identifier",
        ),
        (
            lambda payload: payload["visual_novel_onboarding"]["pages"][0][
                "sprite_variant_keys_by_label"
            ].__setitem__("Iselle", "missing"),
            "selects an unavailable sprite variant",
        ),
        (
            lambda payload: payload["visual_novel_onboarding"]["join_choices"][
                0
            ].__setitem__("character_id", "iselle_the_guide"),
            "choices require playable seats",
        ),
    ),
)
def test_one_star_onboarding_contract_rejects_unsafe_authored_data(
    mutate,
    message: str,
) -> None:
    checkpoint = _load_checkpoint()
    payload = checkpoint.model_dump(
        mode="json",
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
    )
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        CheckpointFile.model_validate(payload)


def test_one_star_ledger_matches_approved_seed_authority() -> None:
    checkpoint = _load_checkpoint()
    owner = next(c for c in checkpoint.characters if c.character_id == "the_master")
    account = OneStarAccountEnvelope.model_validate(owner.mechanics[ONE_STAR_ACCOUNT_KEY])
    config = account.config
    state = account.state

    assert config.starting_resources.model_dump() == {
        "gold": 40,
        "gems": 5,
        "building_resources": 3,
        "materials": {},
    }
    assert config.lobby_id == "niflheim"
    assert config.lobby_location_label == "niflheim_lobby"
    assert config.max_summon_batch == 5
    assert config.maximum_stamina == 5
    assert config.stamina_recovery_seconds == 1800
    assert config.deployment_stamina_cost == 1
    assert config.gem_purchase is not None
    assert config.gem_purchase.model_dump() == {
        "funds_label": "$",
        "starting_funds": 200,
        "periodic_income": 100,
        "income_interval_seconds": 604_800,
        "funds_cost": 100,
        "gems_granted": 20,
    }
    assert state.discretionary_funds == 200
    assert state.funds_accrual_anchor_s == 0
    assert config.summon_pools["basic"].minimum_birth_stars == 1
    assert config.summon_pools["basic"].maximum_birth_stars == 3
    assert config.summon_pools["premium"].minimum_birth_stars == 2
    assert config.summon_pools["premium"].maximum_birth_stars == 5
    assert config.summon_pools["basic"].star_weights == {
        1: 8000,
        2: 1800,
        3: 200,
    }
    assert config.summon_pools["premium"].star_weights == {
        2: 7500,
        3: 2300,
        4: 175,
        5: 25,
    }
    assert config.summon_pools["basic"].usage == "standard"
    assert config.summon_pools["premium"].usage == "standard"
    assert config.summon_pools["basic"].cost.gold == 2
    assert config.summon_pools["premium"].cost.gems == 5
    master_opening = config.summon_pools["master_opening_roster"]
    assert master_opening.model_dump(mode="json") == {
        "usage": "opening_roster",
        "slots": [
            {"kind": "fixed", "character_id": "renna_holt"},
            {"kind": "fixed", "character_id": "mirelle_voss"},
            {"kind": "fixed", "character_id": "edren_marr"},
        ],
        "initial_deployment_requires_guide_handoff": True,
    }
    assert config.summon_pools["master_newcomer_opening_roster"].model_dump(
        mode="json"
    ) == {
        "usage": "opening_roster",
        "slots": [
            {"kind": "fixed", "character_id": "renna_holt"},
            {"kind": "fixed", "character_id": "mirelle_voss"},
            {"kind": "fixed", "character_id": "edren_marr"},
            {"kind": "bound_player_actor", "character_id": BLANK_PLAYER_ID},
        ],
        "initial_deployment_requires_guide_handoff": True,
    }
    assert config.summon_pools["newcomer_opening_roster"].model_dump(
        mode="json"
    ) == {
        "usage": "opening_roster",
        "slots": [{"kind": "bound_player_actor", "character_id": BLANK_PLAYER_ID}],
        "initial_deployment_requires_guide_handoff": False,
    }
    assert "basic_summon" not in config.catalogue
    assert "premium_summon" not in config.catalogue
    assert set(config.summon_pools["basic"].eligible_existing_ids) == {
        "renna_holt",
        "wren_thelantern",
        "rowan_kest",
        "liora_fen",
        "mirelle_voss",
    }
    assert set(config.summon_pools["premium"].eligible_existing_ids) == {
        "soren_ironvow",
        "castor_valebrand",
        "wren_thelantern",
        "rowan_kest",
        "liora_fen",
        "mirelle_voss",
        "seris_nightglass",
        "aveline_morcant",
        "veil_the_unnumbered",
    }
    assert BLANK_PLAYER_ID not in {
        *config.summon_pools["basic"].eligible_existing_ids,
        *config.summon_pools["premium"].eligible_existing_ids,
    }
    immutable_birth_stars = {
        character_id: OneStarHeroState.model_validate(
            next(c for c in checkpoint.characters if c.character_id == character_id)
            .mechanics[ONE_STAR_HERO_KEY]
        ).birth_stars
        for character_id in SUMMON_POOL_IDS
    }
    for pool in (
        config.summon_pools["basic"],
        config.summon_pools["premium"],
    ):
        assert all(
            pool.minimum_birth_stars <= immutable_birth_stars[character_id]
            <= pool.maximum_birth_stars
            for character_id in pool.eligible_existing_ids
        )
        assert all(
            next(
                c for c in checkpoint.characters if c.character_id == character_id
            ).status.value == "dormant"
            and OneStarHeroState.model_validate(
                next(
                    c for c in checkpoint.characters
                    if c.character_id == character_id
                ).mechanics[ONE_STAR_HERO_KEY]
            ).owner_lobby_id == ""
            for character_id in pool.eligible_existing_ids
        )
    assert config.star_level_caps == {
        1: 10,
        2: 20,
        3: 40,
        4: 60,
        5: 80,
        6: 99,
        7: 999999,
    }
    assert config.progression.model_dump() == {
        "stat_ids": ["power", "agility", "resilience"],
        "grade_multiplier_milli": 1250,
        "birth_stat_total": 15,
        "birth_hp_max": 8,
        "variance_basis_points": 500,
        "stat_growth_per_level_milli": 1000,
        "hp_growth_per_level_milli": 500,
        "xp_threshold_factor": 50,
        "floor_xp_per_floor": 100,
        "overlevel_xp_percentages": [100, 75, 50, 25, 10, 5, 0],
        "cap_bank_extra_levels": 1,
        "synthesis_source_base_xp": 100,
        "synthesis_skill_chance_basis_points": 500,
    }
    assert [
        birth_stat_total_mean(stars, config) for stars in range(1, 8)
    ] == [15, 19, 23, 29, 37, 46, 57]
    assert [birth_hp_mean(stars, config) for stars in range(1, 8)] == [
        8,
        10,
        13,
        16,
        20,
        24,
        31,
    ]
    assert config.floor_rewards[5].model_dump() == {
        "gold": 8,
        "gems": 5,
        "building_resources": 2,
        "materials": {},
    }
    assert config.floor_rewards[10].model_dump() == {
        "gold": 16,
        "gems": 15,
        "building_resources": 5,
        "materials": {"lesser_promotion_stone": 1},
    }
    assert set(config.floor_scenarios) == {1, 2, 3, 4, 5}
    floor_one = config.floor_scenarios[1]
    assert floor_one.model_dump(exclude={"pressure_beats"}) == {
        "mission_id": "floor_1_toll_bell",
        "destination": "tower_floor_1_toll_bell",
        "premise": (
            "Secure the goblins' gate crank and reach the exit; "
            "killing everyone is optional."
        ),
        "completion_declaration": (
            "The party has secured the goblins' gate crank and reached the exit."
        ),
        "failure_declaration": (
            "The party can no longer secure the goblins' gate crank or reach the exit."
        ),
        "counters": [
            {"counter_id": "gate_crank_secured", "current": 0, "target": 1},
            {"counter_id": "exit_reached", "current": 0, "target": 1},
        ],
    }
    assert len(floor_one.pressure_beats) == 1
    floor_one_pressure = floor_one.pressure_beats[0]
    assert "frightened deserter" in floor_one_pressure
    assert "shortcut" in floor_one_pressure
    assert "separate drainage culvert" in floor_one_pressure
    assert "away from the party's route" in floor_one_pressure
    assert config.floor_scenarios[2].counters[0].current == 0
    assert config.floor_scenarios[2].counters[0].target == 1
    assert "trapped scavenger" in config.floor_scenarios[2].pressure_beats[0]
    assert config.floor_scenarios[3].counters[0].target == 3
    assert "Echoes copy" in config.floor_scenarios[3].pressure_beats[0]
    assert config.floor_scenarios[4].counters[0].target == 2
    assert "safe exit" in config.floor_scenarios[4].pressure_beats[0]
    assert config.floor_scenarios[5].counters[0].target == 300
    assert "morally revealing" in config.floor_scenarios[5].pressure_beats[0]
    assert config.repeat_gold_numerator == 1
    assert config.repeat_gold_denominator == 4
    assert config.repeat_gold_minimum == 1
    assert config.catalogue["common_weapon"].cost.gold == 1
    assert config.catalogue["common_armor"].cost.gold == 2
    assert config.catalogue["common_field_item"].cost.gold == 1
    assert config.catalogue["armory_1"].cost.building_resources == 2
    assert config.catalogue["armory_1"].facility_id == "armory"
    assert "synthesis_chamber_1" not in config.catalogue
    assert config.catalogue["hero_reaction_research_1"].required_cleared_floor == 5
    assert config.catalogue["daily_dungeon_gate_1"].required_cleared_floor == 5
    assert config.catalogue["lobby_floor_2"].required_cleared_floor == 10
    assert config.catalogue["lobby_floor_2"].resulting_capacity == 40
    assert config.catalogue["promotion_chamber_1"].required_lobby_floor == 2
    for key in (
        "summoning_hall_2",
        "tower_gate_2",
        "accommodation_2",
        "warehouse_2",
        "armory_2",
        "training_camp_2",
    ):
        assert config.catalogue[key].cost.gold == 6
        assert config.catalogue[key].cost.building_resources == 1
        assert config.catalogue[key].required_lobby_floor == 2
    assert {
        key: requirement.model_dump()
        for key, requirement in config.operation_requirements.items()
    } == {
        "deployment": {
            "facility_id": "tower_gate",
            "required_location": "",
        },
        "synthesis": {
            "facility_id": "synthesis_chamber",
            "required_location": "niflheim_synthesis_chamber",
        },
        "promotion": {
            "facility_id": "promotion_chamber",
            "required_location": "niflheim_promotion_chamber",
        },
    }
    assert state.facilities == {
        "summoning_hall": 1,
        "tower_gate": 1,
        "accommodation": 1,
        "warehouse": 1,
        "training_camp": 1,
        "synthesis_chamber": 1,
    }
    assert state.highest_unlocked_floor == 1
    assert state.highest_cleared_floor == 0
    assert state.active_mission is None
    assert state.pending_operation is None
    assert state.research_levels == {}
    assert state.tutorial_deliveries == {}
    assert state.synthesis_resolution_count == 0
    assert state.applied_event_fingerprints == {}
    assert state.stored_equipment == []
    assert state.guide_character_ids == ["iselle_the_guide"]
    assert state.system_observer_ids == ["iselle_the_guide"]


def test_player_primer_covers_each_playable_perspective() -> None:
    checkpoint = _load_checkpoint()
    primer = checkpoint.player_primer

    for perspective in ("Newcomer", "Master", "Halcyon"):
        assert perspective in primer
    assert "Moebius" not in primer
    assert "stolen" not in primer.lower()


def test_player_setup_metadata_is_structured_for_both_frontends() -> None:
    checkpoint = _load_checkpoint()
    setting = checkpoint.world_state.setting
    playable = {
        character.character_id: character
        for character in checkpoint.characters
        if character.is_playable
    }

    assert setting.title == "One-Star Ascension"
    assert setting.recommended_players
    assert setting.play_guidance
    assert checkpoint.world_state.opening is not None
    assert set(playable) == {
        "one_star_newcomer",
        "the_master",
        "halcyon_of_the_gilded_march",
    }
    assert all(character.player_guidance for character in playable.values())


def test_roster_shape_and_player_binding() -> None:
    checkpoint = _load_checkpoint()
    by_id = {c.character_id: c for c in checkpoint.characters}

    assert set(by_id) == EXPECTED_CHARACTER_IDS

    # Keep the blank slot first for existing character-selection surfaces.
    # Whether it appears in the opening is claim-driven, not implied by order.
    first = checkpoint.characters[0]
    assert first.character_id == BLANK_PLAYER_ID
    assert first.is_playable is True
    assert first.agent_tier == CharacterAgentTier.premium

    # The Master is claimable-but-AI-by-default and must NOT be the auto-bind target.
    master_index = next(
        i for i, c in enumerate(checkpoint.characters)
        if c.character_id == "the_master"
    )
    assert master_index != 0
    assert by_id["the_master"].is_playable is True

    # The tower boss is a dormant future hazard, not a claimable person.
    assert by_id["warden_of_the_eighth"].is_playable is False
    assert by_id["warden_of_the_eighth"].status.value == "dormant"


BOUND_IDENTITY_REFERENCE_IDS = {
    "iselle_the_guide": "osa_iselle_the_guide_locked_active_profile_v1",
    "renna_holt": "osa_renna_holt_locked_active_profile_v1",
    "rowan_kest": "osa_rowan_kest_locked_active_profile_v1",
    "liora_fen": "osa_liora_fen_locked_active_profile_v1",
    "wren_thelantern": "osa_wren_thelantern_locked_active_profile_v1",
    "mirelle_voss": "osa_mirelle_voss_locked_active_profile_v1",
    "seris_nightglass": "osa_seris_nightglass_locked_active_profile_v1",
    "castor_valebrand": "osa_castor_valebrand_locked_active_profile_v1",
    "soren_ironvow": "osa_soren_ironvow_locked_identity_base_v1",
    "aveline_morcant": "osa_aveline_morcant_locked_active_profile_v1",
    "halcyon_of_the_gilded_march": "osa_halcyon_of_the_gilded_march_locked_active_profile_v1",
    "veil_the_unnumbered": "osa_veil_the_unnumbered_locked_active_profile_v1",
    "warden_of_the_eighth": "osa_warden_of_the_eighth_locked_v1",
}

UNBOUND_IDENTITY_IDS = {
    "one_star_newcomer",
    "edren_marr",
    "the_master",
}


def test_reviewed_identity_bindings_match_human_selection() -> None:
    checkpoint = _load_checkpoint()
    by_id = {character.character_id: character for character in checkpoint.characters}
    references = {
        reference.reference_id: reference
        for reference in checkpoint.reviewed_visual_references
    }

    assert "mother_yset" not in by_id
    assert "cael_avoran" not in by_id
    assert set(by_id) == EXPECTED_CHARACTER_IDS
    assert set(BOUND_IDENTITY_REFERENCE_IDS) | UNBOUND_IDENTITY_IDS == set(by_id)

    for character_id, reference_id in BOUND_IDENTITY_REFERENCE_IDS.items():
        assert by_id[character_id].visuals.identity_reference_id == reference_id
        reference = references[reference_id]
        assert reference.purpose == "identity"
        assert reference.scope == "character"
        assert reference.scope_id == character_id
        assert reference.selection_hint
        assert reference.diffusion_authorized is True

    grouped = {
        character_id: [
            reference
            for reference in checkpoint.reviewed_visual_references
            if reference.scope_id == character_id
            and reference.purpose == "identity"
        ]
        for character_id in BOUND_IDENTITY_REFERENCE_IDS
    }
    assert sum(
        reference.scope == "character" and reference.purpose == "identity"
        for reference in checkpoint.reviewed_visual_references
    ) == 61
    assert len(grouped["soren_ironvow"]) == 5
    assert len(grouped["warden_of_the_eighth"]) == 1
    assert all(
        len(references) == 5
        for character_id, references in grouped.items()
        if character_id not in {"soren_ironvow", "warden_of_the_eighth"}
    )
    expected_originals = {
        "aveline_morcant": "aveline_morcant.webp",
        "castor_valebrand": "castor_valebrand.webp",
        "halcyon_of_the_gilded_march": "halcyon_of_the_gilded_march.webp",
        "iselle_the_guide": "iselle_source.png",
        "liora_fen": "liora_fen.webp",
        "mirelle_voss": "mirelle_voss.webp",
        "renna_holt": "renna_holt.webp",
        "rowan_kest": "rowan_kest.webp",
        "seris_nightglass": "seris_nightglass.webp",
        "veil_the_unnumbered": "veil_the_unnumbered.webp",
        "warden_of_the_eighth": "warden_of_the_eighth.webp",
        "wren_thelantern": "wren_thelantern.png",
    }
    assert {
        reference.scope_id: reference.storage_ref
        for reference in checkpoint.reviewed_visual_references
        if reference.storage_ref in set(expected_originals.values())
    } == expected_originals
    assert all(
        reference.storage_ref != "soren_ironvow.webp"
        for reference in checkpoint.reviewed_visual_references
    )

    for character_id in UNBOUND_IDENTITY_IDS:
        assert by_id[character_id].visuals.identity_reference_id == ""

    validate_story_visual_references(
        checkpoint,
        story_dir=CHECKPOINT_PATH.parent,
    )

    veil = by_id["veil_the_unnumbered"]
    assert "androgynous" not in veil.public_sheet.appearance.lower()
    assert "half-veil" in veil.public_sheet.appearance.lower()
    renna = by_id["renna_holt"]
    assert "child" not in renna.public_sheet.appearance.lower()
    assert "copper-red" in renna.public_sheet.appearance.lower()


def test_reviewed_lobby_backgrounds_bind_only_current_1f_scenes() -> None:
    checkpoint = _load_checkpoint()
    references = {
        reference.reference_id: reference
        for reference in checkpoint.reviewed_visual_references
        if reference.scope == "location"
    }

    assert checkpoint.location_visual_reference_ids == {
        "niflheim_lobby": [
            "osa_loc_1f_courtyard_v1",
            "osa_loc_1f_pavilion_v1",
            "osa_loc_1f_crack_lobby_v1",
        ],
        "niflheim_synthesis_chamber": ["osa_loc_1f_synthesis_v1"],
        "niflheim_promotion_chamber": ["osa_loc_1f_promotion_v1"],
        "niflheim_crack_of_space_and_time": [
            "osa_loc_1f_crack_facility_v1"
        ],
    }
    active_ids = {
        reference_id
        for reference_ids in checkpoint.location_visual_reference_ids.values()
        for reference_id in reference_ids
    }
    assert {
        "osa_loc_2f3f_courtyard_v1",
        "osa_loc_2f3f_pavilion_v1",
    }.isdisjoint(active_ids)
    assert set(references) == active_ids | {
        "osa_loc_2f3f_courtyard_v1",
        "osa_loc_2f3f_pavilion_v1",
    }
    for label, reference_ids in checkpoint.location_visual_reference_ids.items():
        for reference_id in reference_ids:
            reference = references[reference_id]
            assert reference.scope_id == label
            assert reference.purpose == "environment"
            assert reference.diffusion_authorized is True
    assert {
        reference_id
        for reference_id, reference in references.items()
        if reference.fixed_stage
    } == {
        "osa_loc_1f_synthesis_v1",
        "osa_loc_1f_promotion_v1",
    }
    assert all(
        not reference.fixed_stage
        for reference_id, reference in references.items()
        if reference_id
        not in {"osa_loc_1f_synthesis_v1", "osa_loc_1f_promotion_v1"}
    )


def test_player_character_is_a_blank_user_created_slot() -> None:
    checkpoint = _load_checkpoint()
    pc = next(c for c in checkpoint.characters if c.character_id == BLANK_PLAYER_ID)

    # Intentionally empty: the human creates this character in play.
    assert pc.actor is None
    assert pc.public_sheet.public_context == ""
    assert ONE_STAR_HERO_KEY in pc.mechanics
    newcomer_hero = OneStarHeroState.model_validate(pc.mechanics[ONE_STAR_HERO_KEY])
    assert newcomer_hero.birth_stars == 1
    assert newcomer_hero.innate_system_sight is True
    assert pc.player_slot_kind == PlayerSlotKind.player_authored
    assert pc.player_guidance

    # The record itself carries no authored arrival, location narration, or
    # fallback actor material. Claim-aware opening policy decides whether it appears.
    assert pc.pending_observations == []
    assert pc.status.value == "dormant"
    assert pc.location == "not_yet_fictional"


def test_unclaimed_newcomer_seed_is_an_unbound_claim_aware_seat() -> None:
    checkpoint = _load_checkpoint()
    newcomer = next(
        character
        for character in checkpoint.characters
        if character.character_id == BLANK_PLAYER_ID
    )
    opening = checkpoint.world_state.opening

    assert checkpoint.session.player_character_id == ""
    assert BLANK_PLAYER_ID not in checkpoint.session.character_bindings
    assert newcomer.player_slot_kind == PlayerSlotKind.player_authored
    assert newcomer.status.value == "dormant"
    assert opening is not None


def test_model_visible_seed_surfaces_exclude_live_controller_metadata() -> None:
    checkpoint = _load_checkpoint()
    world = checkpoint.world_state
    surfaces = [
        checkpoint.session.config.narrative_rules,
        *world.facts,
        world.lore,
        world.hidden_lore,
        *world.hidden_facts,
        world.opening.context if world.opening is not None else "",
    ]
    for character in checkpoint.characters:
        surfaces.extend([
            character.name,
            character.public_sheet.role,
            character.public_sheet.appearance,
            character.public_sheet.faction,
            character.public_sheet.public_context,
            *(
                fact.text
                for fact in (character.actor.facts if character.actor else [])
            ),
        ])
    model_visible_seed = "\n".join(text for text in surfaces if text)
    forbidden = (
        r"\bhuman[-_ ](?:bound|controlled|played|player)\b",
        r"\bplayer[-_ ](?:owned|controlled|bound|characters?)\b",
        r"\bnpcs?\b",
        r"\bagent[-_ ]output\b",
        r"\bcharacter bindings?\b",
        r"\bai[-_ ]control(?:led)?\b",
        r"\bwho or what is directing\b",
        r"\brun by the game(?:'s)? own logic\b",
        r"\bprotagonist\b",
    )

    for pattern in forbidden:
        assert re.search(
            pattern,
            model_visible_seed,
            flags=re.IGNORECASE,
        ) is None, pattern


def test_brutality_contract_has_no_blanket_low_rank_safety() -> None:
    """Guard the softening clauses found in repeated casualty-free replays."""
    checkpoint = _load_checkpoint()
    current_contract = "\n".join(
        (
            checkpoint.session.config.narrative_rules,
            *checkpoint.world_state.facts,
            checkpoint.world_state.lore,
            checkpoint.world_state.hidden_lore,
            *checkpoint.world_state.hidden_facts,
            checkpoint.world_state.opening.context,
        )
    ).lower()

    for obsolete_softener in (
        "ration loss",
        "light, learnable",
        "safe for now",
        "never a flat instant death",
        "never an arbitrary early dice-kill",
        "hard lethal outcome requires accumulated setup",
        "without hesitation or error",
    ):
        assert obsolete_softener not in current_contract


def test_master_is_offstage_unreachable_actor() -> None:
    checkpoint = _load_checkpoint()
    master = next(c for c in checkpoint.characters if c.character_id == "the_master")

    assert master.location == "the_masters_screen"
    assert master.actor is not None
    assert master.actor.may_act_offstage is True
    assert master.actor.facts


def test_cast_is_bounded_so_the_lobby_never_becomes_thousands_of_agents() -> None:
    checkpoint = _load_checkpoint()

    movers = {
        c.character_id
        for c in checkpoint.characters
        if _may_act_offstage(c)
    }
    assert movers == EXPECTED_INTENTION_MOVERS
    assert len(movers) <= 3
    assert "the_master" in movers

    non_movers = [
        c for c in checkpoint.characters
        if not _may_act_offstage(c)
    ]
    assert len(non_movers) > len(movers)

    # A small authored roster; any lobby crowd lives as ambient world text.
    assert len(checkpoint.characters) <= 18

def test_floor_zero_start_and_summon_pool() -> None:
    checkpoint = _load_checkpoint()
    by_id = {c.character_id: c for c in checkpoint.characters}

    # Floor-zero: a brand-new lobby / tutorial start, not the old mid-climb stall.
    assert checkpoint.world_state.global_flags["floor"] == 0
    assert checkpoint.world_state.global_flags["phase"] == "floor_zero_tutorial"
    assert checkpoint.world_state.global_flags["lobby_freshly_instanced"] is True

    # The pre-authored characters are preserved as a dormant, quarantined pool:
    # not in play until the router summons/introduces one.
    for pool_id in SUMMON_POOL_IDS:
        pooled = by_id[pool_id]
        assert pooled.status.value == "dormant", pool_id
        assert pooled.location == "unsummoned_pool", pool_id
        assert pooled.is_playable is False, pool_id
        assert _may_act_offstage(pooled) is False, pool_id

    # Ordinary one-stars are generated for paid summons. The free authored
    # starter roster carries two fixed, dormant one-star exceptions.
    assert "bex_greenpull" not in by_id
    assert "dala_greenpull" not in by_id
    seeded_tier_ones = {
        character.character_id
        for character in checkpoint.characters
        if character.knowledge_tier == 1
    }
    assert seeded_tier_ones == {"renna_holt", "edren_marr"}
    renna = by_id["renna_holt"]
    assert renna.name == "Renna Holt"
    assert renna.status.value == "dormant"
    assert renna.location == "unsummoned_pool"
    assert renna.is_playable is False
    assert renna.actor is not None
    assert renna.actor.facts
    edren = by_id["edren_marr"]
    assert edren.name == "Edren Marr"
    assert edren.status.value == "dormant"
    assert edren.location == "unsummoned_pool"
    assert edren.is_playable is False
    assert edren.actor is not None
    assert edren.actor.may_act_offstage is False

    for character in checkpoint.characters:
        role = character.public_sheet.role.lower()
        assert "summon pool" not in role
        assert "not yet in play" not in role

    lifecycle_leaks = (
        "dormant reserve",
        "absent until acquired",
        "until acquired and activated",
        "summon pool candidate",
        "future hazard, held dormant",
    )
    for pool_id in SUMMON_POOL_IDS | {"warden_of_the_eighth"}:
        actor_text = _actor_text(by_id[pool_id]).lower()
        assert not any(leak in actor_text for leak in lifecycle_leaks), pool_id


def test_birth_one_star_self_packet_is_sparse_and_fiction_bound() -> None:
    checkpoint = _load_checkpoint()
    renna = next(
        character
        for character in checkpoint.characters
        if character.character_id == "renna_holt"
    )
    edren = next(
        character
        for character in checkpoint.characters
        if character.character_id == "edren_marr"
    )

    rendered_facts = build_character_self_packet(edren, checkpoint)

    assert edren.actor is not None
    assert all(fact.text in rendered_facts for fact in edren.actor.facts)
    assert "master" not in rendered_facts.lower()
    assert "tower" not in rendered_facts.lower()
    assert "deployment" not in rendered_facts.lower()
    assert renna.actor is not None


def test_iselle_self_packet_preserves_her_sparse_facts() -> None:
    checkpoint = _load_checkpoint()
    iselle = next(
        character
        for character in checkpoint.characters
        if character.character_id == "iselle_the_guide"
    )

    assert iselle.actor is not None
    rendered_facts = build_character_self_packet(iselle, checkpoint)
    assert all(fact.text in rendered_facts for fact in iselle.actor.facts)
    assert "character_current_objectives" not in rendered_facts


def test_opening_seed_has_no_stale_slime_guidance() -> None:
    checkpoint = _load_checkpoint()
    by_id = {character.character_id: character for character in checkpoint.characters}

    public_facts = "\n".join(checkpoint.world_state.facts).lower()
    router_only = (
        checkpoint.world_state.hidden_lore
        + "\n"
        + "\n".join(checkpoint.world_state.hidden_facts)
    ).lower()
    guide_context = _actor_text(by_id["iselle_the_guide"]).lower()

    assert guide_context
    assert "floor scenarios" in public_facts
    assert "acid slime" not in public_facts
    assert "acid slime" not in guide_context

    assert "acid slime" not in router_only


def test_expanded_summon_pool_spans_two_through_five_stars() -> None:
    checkpoint = _load_checkpoint()
    by_id = {character.character_id: character for character in checkpoint.characters}
    expected_tiers = {
        "rowan_kest": 2,
        "liora_fen": 2,
        "mirelle_voss": 3,
        "seris_nightglass": 4,
        "aveline_morcant": 5,
    }

    for character_id, tier in expected_tiers.items():
        character = by_id[character_id]
        assert character.knowledge_tier == tier
        assert character.status.value == "dormant"
        assert character.location == "unsummoned_pool"
        assert character.is_playable is False
        assert _may_act_offstage(character) is False


def test_grade_memory_status_and_reserve_authority_are_coherent() -> None:
    checkpoint = _load_checkpoint()
    by_id = {character.character_id: character for character in checkpoint.characters}

    for character_id in SUMMON_POOL_IDS:
        character = by_id[character_id]
        assert character.status.value == "dormant"
        assert character.location == "unsummoned_pool"
        assert character.knowledge_tier is not None

    veil_public = (
        by_id["veil_the_unnumbered"].public_sheet.role
        + " "
        + by_id["veil_the_unnumbered"].public_sheet.appearance
    ).lower()
    assert "awakened" not in veil_public
    assert "status window" not in veil_public


def test_hero_mechanics_follow_authored_tiers_without_public_hidden_potential() -> None:
    checkpoint = _load_checkpoint()
    by_id = {character.character_id: character for character in checkpoint.characters}
    owner = by_id["the_master"]
    config = OneStarAccountEnvelope.model_validate(
        owner.mechanics[ONE_STAR_ACCOUNT_KEY]
    ).config
    expected_birth_stars = {
        "one_star_newcomer": 1,
        "renna_holt": 1,
        "edren_marr": 1,
        "halcyon_of_the_gilded_march": 6,
        "soren_ironvow": 5,
        "castor_valebrand": 4,
        "wren_thelantern": 3,
        "rowan_kest": 2,
        "liora_fen": 2,
        "mirelle_voss": 3,
        "seris_nightglass": 4,
        "aveline_morcant": 5,
        # Veil's authored grade is unreadable; the durable bookkeeping keeps
        # her five-star mechanics private without changing her public identity.
        "veil_the_unnumbered": 5,
    }
    expected_affinities = {
        "one_star_newcomer": ("agility", "power"),
        "renna_holt": ("agility", "power"),
        "edren_marr": ("resilience", "agility"),
        "halcyon_of_the_gilded_march": ("power", "agility"),
        "soren_ironvow": ("resilience", "agility"),
        "castor_valebrand": ("agility", "resilience"),
        "wren_thelantern": ("resilience", "power"),
        "rowan_kest": ("agility", "resilience"),
        "liora_fen": ("agility", "resilience"),
        "mirelle_voss": ("power", "resilience"),
        "seris_nightglass": ("power", "resilience"),
        "aveline_morcant": ("resilience", "agility"),
        "veil_the_unnumbered": ("agility", "resilience"),
    }
    for character_id, birth_stars in expected_birth_stars.items():
        hero = OneStarHeroState.model_validate(
            by_id[character_id].mechanics[ONE_STAR_HERO_KEY]
        )
        assert hero.birth_stars == birth_stars
        assert hero.current_stars == birth_stars
        assert hero.hp_current == hero.hp_max
        assert hero.stats
        assert hero.experience_points == experience_to_reach_level(hero.level, config)
        assert hero.progression_seed == derive_progression_seed(
            character_id=character_id,
            birth_stars=birth_stars,
        )
        assert (hero.strong_stat_id, hero.weak_stat_id) == expected_affinities[
            character_id
        ]
        assert max(hero.stats, key=hero.stats.get) == hero.strong_stat_id
        assert min(hero.stats, key=hero.stats.get) == hero.weak_stat_id
        assert hero.potential_grade == (
            4 if character_id == "renna_holt" else birth_stars
        )
        expected = hero.model_copy(deep=True)
        rebalance_hero(hero=expected, config=config, restore_full_hp=True)
        assert hero.stats == expected.stats
        assert hero.hp_max == expected.hp_max
        assert hero.hp_current == expected.hp_current
        assert hero.equipment
        assert hero.skills
        assert all(
            item.quantity >= 1
            and isinstance(item.tags, list)
            and isinstance(item.visible, bool)
            for item in hero.equipment
        )
        assert all(
            skill.capability
            and isinstance(skill.tags, list)
            and isinstance(skill.visible, bool)
            for skill in hero.skills
        )
        assert hero.owner_lobby_id == (
            "gilded_march"
            if character_id == "halcyon_of_the_gilded_march"
            else ""
        )
    newcomer = OneStarHeroState.model_validate(
        by_id[BLANK_PLAYER_ID].mechanics[ONE_STAR_HERO_KEY]
    )
    assert newcomer.innate_system_sight is True
    assert newcomer.potential_grade == newcomer.birth_stars
    halcyon = OneStarHeroState.model_validate(
        by_id["halcyon_of_the_gilded_march"].mechanics[ONE_STAR_HERO_KEY]
    )
    assert halcyon.level == 99
    assert halcyon.experience_points == experience_to_reach_level(99, config)
    assert halcyon.hp_current == halcyon.hp_max
    veil = OneStarHeroState.model_validate(
        by_id["veil_the_unnumbered"].mechanics[ONE_STAR_HERO_KEY]
    )
    assert veil.potential_grade == veil.birth_stars
    assert veil.hidden_capabilities
    public_identity = (
        by_id["veil_the_unnumbered"].public_sheet.role
        + " "
        + by_id["veil_the_unnumbered"].public_sheet.appearance
    ).lower()
    assert "schema-valid" not in public_identity


def test_seed_rules_text_matches_typed_economy_and_retreat_law() -> None:
    checkpoint = _load_checkpoint()
    searchable = "\n".join(
        [
            checkpoint.session.config.narrative_rules,
            *checkpoint.world_state.facts,
            checkpoint.world_state.lore,
            checkpoint.world_state.hidden_lore,
            *checkpoint.world_state.hidden_facts,
        ]
    ).lower()
    assert "premium gem summons improve odds but can still produce an ordinary one-star" not in searchable
    assert "multiple distinct input heroes" not in searchable
    assert "retreat can return the party to the lobby" not in searchable
    assert "retreat" in searchable
    assert "very rare escape item or very powerful magic" in searchable


def test_common_world_authority_does_not_leak_hidden_origin() -> None:
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state

    common = ("\n".join(ws.facts) + "\n" + ws.lore).lower()
    for hidden_origin in ("moebius", "abduct", "stolen people"):
        assert hidden_origin not in common


def test_hidden_reveals_are_router_only() -> None:
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state

    hidden = (ws.hidden_lore + "\n" + "\n".join(ws.hidden_facts)).lower()
    for token in [
        "moebius",
        "permadeath",
        "synthes",
        "memor",
        "intervention",
        "fade",
        "wailing wall",
    ]:
        assert token in hidden, token
    primer = checkpoint.player_primer.lower()
    assert "moebius" not in primer
    assert "stolen" not in primer


def test_promotion_star_up_and_memory_spine() -> None:
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state

    facts = "\n".join(ws.facts).lower()
    assert "promotion" in facts
    assert "leveling" in facts
    assert "promotion chamber" in facts

    hidden = (ws.hidden_lore + "\n" + "\n".join(ws.hidden_facts)).lower()
    assert "four stars" in hidden or "four-star" in hidden
    for grade in ("three-stars", "four-stars", "five-stars"):
        assert grade in hidden

    master = next(c for c in checkpoint.characters if c.character_id == "the_master")
    assert master.actor is not None

    halcyon = next(
        c for c in checkpoint.characters
        if c.character_id == "halcyon_of_the_gilded_march"
    )
    assert halcyon.actor is not None
    assert halcyon.actor.facts


def test_system_sight_is_master_view_with_protagonist_exception() -> None:
    """Canon fix: the crisp System UI (status/stat windows, mission text, and
    dialogue boxes) is the Master's view. Ordinary Heroes are System-blind until
    the lobby buys Hero Reaction Research; the out-of-world protagonist is the
    sole authored exception who reads the System from the moment of summoning."""
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state

    facts = "\n".join(ws.facts).lower()
    hidden = (ws.hidden_lore + "\n" + "\n".join(ws.hidden_facts)).lower()
    rules = checkpoint.session.config.narrative_rules.lower()

    # The System's crisp readouts belong to the Master, not to an ordinary Hero,
    # and the research upgrade is the diegetic gate that shares them with a party.
    assert "hero reaction research" in facts
    assert hidden.count("hero reaction research") == 0
    # Guard against regressing to the old (wrong) "shared reality" framing.
    assert "the heroes' shared reality" not in facts

    # The player-character is the authored exception: an out-of-world summon who
    # reads the System from birth (both the POV necessity and a live anomaly).
    assert BLANK_PLAYER_ID in hidden
    assert "system windows" in hidden

    # The narrator must gate dialogue boxes by who can actually see the System,
    # and the protagonist's sight is not tied to the Master being logged in.
    assert "unresearched" in rules
    assert "system-sight" in rules

    # The exception lives in world-truth/router context, NOT on the PC record,
    # so the blank user-created slot stays blank.
    pc = next(c for c in checkpoint.characters if c.character_id == BLANK_PLAYER_ID)
    assert pc.actor is None


def test_lobby_master_and_guide_framing() -> None:
    """Playtest fixes: the party is formed by the Master (the guide only
    assists), Niflheim is a home hub distinct from the Tower, and the
    tutorial-guide's unit-management coaching is aimed at the Master, never at
    a pre-tutorial arrival; plus newcomer POV/jargon discipline."""
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state
    by_id = {c.character_id: c for c in checkpoint.characters}

    # Concern 1: no character carries the static "the Master's party" tag; a
    # party is formed later, by the Master.
    for character in checkpoint.characters:
        assert "the Master's party" not in character.public_sheet.faction, (
            character.character_id
        )
    assert by_id["one_star_newcomer"].public_sheet.faction == "Niflheim lobby"
    assert by_id["renna_holt"].public_sheet.faction == ""

    rules = checkpoint.session.config.narrative_rules
    rules_lower = rules.lower()

    # Concern 4: the story begins in the lobby, before any floor deployment.
    assert ws.global_flags["floor"] == 0
    assert by_id["one_star_newcomer"].location == "not_yet_fictional"
    assert by_id["iselle_the_guide"].location == "niflheim_lobby"
    assert "lobby vs. tower" in rules_lower

    iselle = by_id["iselle_the_guide"]
    # The public role is reused as visible local-cast context during character
    # generation, so it cannot preload the unseen authority into a birth
    # one-star. Iselle's actor record still carries her full function.
    assert "master" not in iselle.public_sheet.role.lower()
    assert "master" in _actor_text(iselle).lower()
    assert iselle.actor is not None

    # Concern 5: the narrator is told to aim the guide's coaching at the Master.
    assert "tutorial-guide's audience" in rules
    assert "iselle serves the master" in rules_lower

    # Concern 3: newcomer POV / jargon discipline for the blank Hero, with the
    # gacha terms named as things the narrator must NOT speak in its own voice.
    assert "newcomer pov and jargon" in rules_lower
    assert "one-star" in rules


def test_tower_floor_exit_requires_extraordinary_means() -> None:
    """Every model that can decide or render a deployment receives the same
    floor-exit law, while a fresh one-star learns it only in fiction."""
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state
    by_id = {c.character_id: c for c in checkpoint.characters}

    router_contract = "\n".join((*ws.facts, ws.lore)).lower()
    narrator_contract = checkpoint.session.config.narrative_rules.lower()
    tier_one_grant, _ = _assemble_knowledge_grant(checkpoint, 1)
    tier_one_contract = tier_one_grant.lower()
    master_contract = "\n".join(
        (
            ws.setting.play_guidance,
            by_id["the_master"].player_guidance,
            _actor_text(by_id["the_master"]),
        )
    ).lower()

    for contract_name, contract in (
        ("router", router_contract),
        ("narrator", narrator_contract),
        ("Master", master_contract),
    ):
        assert "very rare escape item" in contract, contract_name
        assert "very powerful magic" in contract, contract_name
        assert "lobby" in contract, contract_name

    assert "tactical withdrawal" in router_contract
    assert "tactical withdrawal" in narrator_contract
    assert by_id["iselle_the_guide"].actor is not None
    # Birth one-stars begin before Iselle's orientation. Neither generated
    # dossiers nor the authored reserve may preload the tutorial's premise or
    # floor law; it reaches them later as witnessed canonical dialogue.
    for forbidden in (
        "master",
        "tower",
        "deployment",
        "very rare escape item",
        "very powerful magic",
    ):
        assert forbidden not in tier_one_contract
        assert forbidden not in _actor_text(by_id["renna_holt"]).lower()

    # Authored Heroes born at tier two or above have actor-owned knowledge;
    # the birth tier itself remains clear of the tutorial until it is witnessed.
    for character in checkpoint.characters:
        if character.knowledge_tier is None or character.knowledge_tier < 2:
            continue
        assert character.actor is not None, character.character_id
        assert character.actor.facts, character.character_id

    # Preserve the authored-Newcomer contract and the inert floor boss.
    assert by_id[BLANK_PLAYER_ID].actor is None
    warden_context = _actor_text(by_id["warden_of_the_eighth"]).lower()
    assert "tower-floor law" not in warden_context

    all_model_contracts = "\n".join(
        (
            router_contract,
            narrator_contract,
            master_contract,
            _actor_text(by_id["iselle_the_guide"]).lower(),
        )
    )
    for obsolete_exit_rule in (
        "when it is cleared or they retreat",
        "left again when a floor is cleared or a party retreats",
        "a floor's end flashes them back",
        "whether to press on or retreat",
        "sound tactics, timely retreat",
        "fails on a party wipe or exhausted attempts",
    ):
        assert obsolete_exit_rule not in all_model_contracts


def test_lobby_facilities_healing_and_enforcement() -> None:
    """Source-fidelity pass: the lobby is a build/upgrade economy of named
    chambers (incl. a Synthesis Chamber), it restores Heroes between missions
    (death/synthesis/old-amputation excepted), and the guide is also a warden
    who compels refusers and is defended by a genuinely lethal protocol."""
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state
    by_id = {c.character_id: c for c in checkpoint.characters}

    facts = "\n".join(ws.facts).lower()
    lore = ws.lore.lower()
    rules = checkpoint.session.config.narrative_rules.lower()
    hidden = (ws.hidden_lore + "\n" + "\n".join(ws.hidden_facts)).lower()

    # Facilities exist with gacha purposes and are built/upgraded; the shrine
    # is folded into summoning rather than kept as a purposeless building.
    facts_lore = facts + "\n" + lore
    for facility in (
        "summoning hall",
        "training hall",
        "synthesis chamber",
        "transformation chamber",
        "blacksmith",
        "crack of space and time",
    ):
        assert facility in facts_lore, facility
    assert "waiting room" in facts
    assert "shrine" not in facts_lore  # repurposed, not a purposeless building

    # Lobby restoration with the permadeath / old-amputation exceptions.
    assert "restorative" in facts
    assert "limb lost long ago" in facts
    assert "do not carry wounds" in rules

    # The guide is also the warden: compels refusers, lethal defense protocol.
    assert "warden" in facts
    assert "lethal defense protocol" in facts
    iselle = by_id["iselle_the_guide"]
    assert "warden" in iselle.public_sheet.role.lower()
    guide_ctx = _actor_text(iselle).lower()
    assert guide_ctx
    assert iselle.actor is not None
    assert iselle.actor.facts

    # Old softening clauses must not silently reintroduce immunity.
    assert "treat it exactly like synthesis" not in hidden
    assert "never an arbitrary instant kill" not in hidden

    # The human-facing Master toolkit includes facilities and transformation;
    # these controls remain in the human-facing control contract.
    master = by_id["the_master"]
    player_contract = (
        master.player_guidance + "\n" + _actor_text(master)
    ).lower()
    assert "facilit" in player_contract
    assert "transform" in player_contract


def test_master_commits_deployment_then_watches_autonomous_mission() -> None:
    """The Master owns lobby and pre-deployment choices, not Hero tactics."""
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state
    by_id = {c.character_id: c for c in checkpoint.characters}
    master = by_id["the_master"]
    iselle = by_id["iselle_the_guide"]

    player_contract = "\n".join(
        (
            checkpoint.player_primer,
            ws.setting.play_guidance,
            master.player_guidance,
            _actor_text(master),
        )
    ).lower()
    for management_choice in (
        "summon",
        "facilities",
        "choose the floor",
        "roster",
        "loadout",
    ):
        assert management_choice in player_contract

    # Positioning belongs to the embodied Heroes rather than a second tactical
    # control surface for the disembodied Master.
    opening_contract = "\n".join(
        (
            segment.text
            for beat in ws.opening.authored_character_beats
            for segment in beat.segments
        )
    )
    master_management_surface = "\n".join(
        (
            checkpoint.session.config.narrative_rules,
            "\n".join(ws.facts),
            "\n".join(ws.hidden_facts),
            ws.lore,
            player_contract,
            iselle.public_sheet.role,
            _actor_text(iselle),
            master.public_sheet.appearance,
            master.visuals.default_loadout,
            opening_contract,
        )
    ).lower()
    for retired_formation_surface in (
        "opening formation",
        "formation lattice",
        "set that formation",
        "formation positions",
        "pre-mission formation",
        "pre-deployment formation",
        "pre-deployment arrangement",
    ):
        assert retired_formation_surface not in master_management_surface

    # The human-facing Master contract permits narrative viewpoint shifts and
    # one-way audio while explicitly assigning mission choices to Heroes.
    assert "during a mission" in master.player_guidance.lower()
    assert "mostly watch" in player_contract
    assert "advances through the heroes' next actions" in player_contract
    assert "hear" in master.player_guidance.lower()
    assert "cannot speak or type" in master.player_guidance.lower()
    assert "heroes choose targets" in player_contract
    assert master.actor is not None
    assert "active feed" not in player_contract
    assert "live hero feed" not in player_contract

    # The tutorial guide must not recreate the removed tactical control loop.
    guide_contract = "\n".join(
        (iselle.public_sheet.role, _actor_text(iselle))
    ).lower()
    for obsolete_tactical_cue in (
        "target priority",
        "priority directives",
        "item and consumable use",
        "advance or retreat",
    ):
        assert obsolete_tactical_cue not in player_contract
        assert obsolete_tactical_cue not in guide_contract


def test_knowledge_tier_ladder_gradient() -> None:
    """Knowledge, sparse actor-fact guidance, and presentation scale by tier."""
    checkpoint = _load_checkpoint()
    tiers = {t.tier: t for t in checkpoint.world_state.knowledge_tiers}

    assert set(tiers) == {1, 2, 3, 4, 5, 6}

    # A birth one-star has no pre-tutorial world knowledge; high rungs do.
    assert tiers[1].world_knowledge == ""
    t5_world = tiers[5].world_knowledge.lower()
    assert "moebius" in t5_world
    assert "fade" in t5_world

    guidance_fields = {
        "actor_fact_guidance",
        "public_visual_detail",
        "loadout_detail",
        "visual_salience",
        "presentation_guidance",
    }
    for tier in tiers.values():
        assert tier.generation_guidance is not None
        guidance = tier.generation_guidance.model_dump()
        assert set(guidance) == guidance_fields
        assert all(value.strip() for value in guidance.values())

    # The sparse actor-fact instruction changes at the first rare rung without
    # restoring paired dossier-depth fields.
    low = tiers[2].generation_guidance.model_dump()
    rich = tiers[3].generation_guidance.model_dump()
    assert low["actor_fact_guidance"] != rich["actor_fact_guidance"]
    for field in guidance_fields - {"presentation_guidance", "actor_fact_guidance"}:
        assert len(rich[field].split()) > len(low[field].split()), field

    # Presentation is story-local and optimizes rare summons for immediately
    # legible popular-fantasy appeal instead of a demographic coverage matrix.
    for tier_number in (3, 4, 5, 6):
        presentation = (
            tiers[tier_number].generation_guidance.presentation_guidance.lower()
        )
        assert "appeal" in presentation
        assert "adult" in presentation
        assert "gender" in presentation
    all_presentation = " ".join(
        tier.generation_guidance.presentation_guidance for tier in tiers.values()
    ).lower()
    assert "demographic coverage" in all_presentation
    assert "all genders" not in all_presentation

    # Agent tier escalates with knowledge tier: fodder cheap, plot-bearing strong.
    assert tiers[1].agent_tier == CharacterAgentTier.utility
    assert tiers[2].agent_tier == CharacterAgentTier.utility
    assert tiers[3].agent_tier == CharacterAgentTier.standard
    assert tiers[4].agent_tier == CharacterAgentTier.premium
    assert tiers[5].agent_tier == CharacterAgentTier.premium
    assert tiers[6].agent_tier == CharacterAgentTier.premium


def test_assemble_knowledge_grant_is_cumulative_and_tier_gated() -> None:
    """The char-gen budget is cumulative (tier N covers 1..N), gates plot
    knowledge to high tiers, carries the rung's agent tier, and gives tier zero
    an explicit no-knowledge contract when the story authors a ladder."""
    checkpoint = _load_checkpoint()

    grant0, agent0 = _assemble_knowledge_grant(checkpoint, 0)
    assert grant0
    assert all(f"Tier {n}" not in grant0 for n in range(1, 7))
    assert agent0 is None

    tiers = {t.tier: t for t in checkpoint.world_state.knowledge_tiers}
    tiers[1].generation_guidance.visual_salience = "TARGET_ONE_MARKER"
    tiers[5].generation_guidance.visual_salience = "TARGET_FIVE_MARKER"

    grant1, agent1 = _assemble_knowledge_grant(checkpoint, 1)
    assert "Tier 1" in grant1
    assert "TARGET_ONE_MARKER" in grant1
    for pre_tutorial_leak in ("master", "tower", "climb", "deployment"):
        assert pre_tutorial_leak not in grant1.lower()
    assert agent1 == CharacterAgentTier.utility

    grant5, agent5 = _assemble_knowledge_grant(checkpoint, 5)
    assert all(f"Tier {n}" in grant5 for n in (1, 2, 3, 4, 5))  # cumulative
    assert "TARGET_FIVE_MARKER" in grant5
    assert "TARGET_ONE_MARKER" not in grant5  # generation target is not cumulative
    assert "moebius" in grant5.lower()
    assert "fade" in grant5.lower()
    assert agent5 == CharacterAgentTier.premium

    # A knowledge ladder without generation guidance keeps the original
    # knowledge-only contract.
    for rung in checkpoint.world_state.knowledge_tiers:
        rung.generation_guidance = None
    knowledge_only, _ = _assemble_knowledge_grant(checkpoint, 3)
    assert knowledge_only
    assert "TARGET_ONE_MARKER" not in knowledge_only

    # A story with no ladder is unaffected: no budget block, default agent tier.
    checkpoint.world_state.knowledge_tiers = []
    assert _assemble_knowledge_grant(checkpoint, 3) == ("", None)


def test_seeded_rare_characters_keep_sparse_actor_records_and_public_visual_identity() -> None:
    checkpoint = _load_checkpoint()
    by_id = {character.character_id: character for character in checkpoint.characters}
    scaled = [
        (3, by_id["wren_thelantern"]),
        (4, by_id["castor_valebrand"]),
        (5, by_id["soren_ironvow"]),
        (6, by_id["halcyon_of_the_gilded_march"]),
    ]

    assert [character.knowledge_tier for _tier, character in scaled] == [
        tier for tier, _character in scaled
    ]
    visual_depths = [
        len(
            (
                character.public_sheet.appearance
                + " "
                + character.visuals.default_loadout
            ).split()
        )
        for _tier, character in scaled
    ]
    assert visual_depths == sorted(visual_depths)
    assert len(set(visual_depths)) == len(visual_depths)

    material_terms = {
        "wool",
        "linen",
        "leather",
        "iron",
        "glass",
        "bone",
        "silk",
        "steel",
        "silver",
        "gold",
        "plate",
        "star-metal",
        "oxhide",
        "kidskin",
    }
    build_terms = {
        "athletic",
        "long-limbed",
        "broad",
        "powerful",
        "build",
        "frame",
        "shoulders",
    }
    for tier, character in scaled:
        assert character.actor is not None, tier
        assert character.actor.facts, tier
        appearance = character.public_sheet.appearance.lower()
        loadout = character.visuals.default_loadout.lower()
        assert any(
            age in appearance
            for age in ("twenties", "thirties", "forties", "adult")
        ), tier
        assert any(term in appearance for term in build_terms), tier
        for anchor in (
            "human",
            "complexion",
            "hair",
            "face",
            "eyes",
            "silhouette",
            "palette",
        ):
            assert anchor in appearance, (tier, anchor)
        assert len({term for term in material_terms if term in loadout}) >= 2, tier
        assert "signature" in loadout, tier

        public_visuals = appearance + " " + loadout
        for private_token in (
            "moebius",
            "abduct",
            "memory-wipe",
            "stolen life",
            "pre-tower name",
        ):
            assert private_token not in public_visuals, (tier, private_token)

    # The authored one-star exception remains visually shared even when its
    # wiped-life interior is deeper than a generated tier-one profile. Rank
    # constrains world knowledge and mechanics, not an authored prose budget.
    renna = by_id["renna_holt"]
    assert len(
        (
            renna.public_sheet.appearance
            + " "
            + renna.visuals.default_loadout
        ).split()
    ) < visual_depths[0] / 2


def test_seed_has_actor_records_without_dnd_mechanics() -> None:
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state

    assert len(checkpoint.player_primer) > 500
    assert ws.lore.strip()
    assert ws.hidden_lore.strip()
    assert ws.facts
    assert ws.hidden_facts

    for character in checkpoint.characters:
        # The blank user-created player slot is intentionally empty.
        if character.character_id == BLANK_PLAYER_ID:
            continue
        if character.entity_kind.value == "hazard":
            assert character.actor is None
        else:
            assert character.actor is not None, character.character_id
            assert character.actor.facts, character.character_id
            assert character.public_sheet.public_context.strip(), character.character_id
        assert character.visuals.default_loadout.strip(), character.character_id
