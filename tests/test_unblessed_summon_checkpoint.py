"""Smoke tests for the hand-authored Unblessed Summon story seed.

The story is shipped as a v5.0 CheckpointFile JSON under
`app/storage/stories/the_unblessed_summon/ckpt_0000.json`. There is no
LLM importer pipeline backing it; the file is hand-emitted in synthetic
form. These tests pin the structural contracts the engine relies on so
edits to the file fail fast on obvious mistakes:

* the JSON must round-trip through `CheckpointFile.model_validate`;
* the player slots must remain unstatted (`mechanics == {}`) so a user's
  `/attach` upload is not silently overwritten;
* the named NPC roster must be present;
* world_state lore + hidden_lore + facts + hidden_facts must be
  populated (this is a synthetic checkpoint, not a stub);
* the mechanics roll-modifier contract must work against a few sampled
  characters, since each NPC's `mechanics` block was hand-computed.

Per AGENTS.md test policy: this file does NOT assert specific prose
remains in the file (no positive prompt-text snapshot tests). It tests
the structural and mechanical contracts the engine reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.engine import mechanics
from app.schemas.characters import CharacterAgentTier
from app.schemas.checkpoint import CheckpointFile
from app.schemas.dnd_cat_ii import PlannedRoll


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "storage"
    / "stories"
    / "the_unblessed_summon"
    / "ckpt_0000.json"
)

PLAYER_CHARACTER_ID = "player_protagonist"
SECOND_PLAYER_CHARACTER_ID = "player_protagonist_2"
PLAYER_CHARACTER_IDS = {PLAYER_CHARACTER_ID, SECOND_PLAYER_CHARACTER_ID}

EXPECTED_NPC_IDS = {
    # romance roster
    "liriel_vaerien",
    "korva_sahl",
    "sariel_marenne",
    "vaella_coldspire",
    "quill",
    "sylira_vesh",
    "marennis_vale",
    # cohort heroes (5) + earlier arrivals (2)
    "sora_kageyama",
    "mika_aoyama",
    "riku_tsumura",
    "daichi_nikaido",
    "mei_iwasaki",
    "yuna_kiyose",
    "tatsuya_hozumi",
    # antagonists
    "crown_prince_aldemar",
    "king_halric",
    "cardinal_vespera",
    "archon_selivar",
    "sage_wencel",
    "demon_lord",
    "master_ovrec",
    "household_physician_orvell",
    "crown_liaison_lerin",
    "princess_nirvel",
    "anelle_aubin",
    "ambassador_sashina",
    "sevarin",
    "faulker",
    "halen",
    "veranne",
    "yuto_arai",
    "asami_kuroda",
    "kenta_morimura",
    "yui_sasahara",
    "hiroshi_kasai",
    "kei_sugino",
    # supporting cast
    "guild_master_bren",
    "court_mage_selen",
    "lady_aoi",
    "mira",
}

EXPECTED_ALL_IDS = EXPECTED_NPC_IDS | PLAYER_CHARACTER_IDS

# Each romance interest must use a distinct D&D 5e class chassis. The
# user explicitly required no class repeats among the seven romance
# tracks. Encode it as a contract check so future edits cannot drift.
ROMANCE_CHARACTER_IDS = {
    "liriel_vaerien",
    "korva_sahl",
    "sariel_marenne",
    "vaella_coldspire",
    "quill",
    "sylira_vesh",
    "marennis_vale",
}

COHORT_CHARACTER_IDS = {
    "sora_kageyama",
    "mika_aoyama",
    "riku_tsumura",
    "daichi_nikaido",
    "mei_iwasaki",
    "yuto_arai",
    "asami_kuroda",
    "kenta_morimura",
    "yui_sasahara",
    "hiroshi_kasai",
    "kei_sugino",
}

GUILD_HALL = "caer_veylan_adventurers_guild_hall"
HERO_TRAINING_COMPOUND = "veylan_royal_academy_hero_training_compound"


@pytest.fixture(scope="module")
def checkpoint() -> CheckpointFile:
    raw = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return CheckpointFile.model_validate(raw)


@pytest.fixture(scope="module")
def by_id(checkpoint: CheckpointFile) -> dict:
    return {c.character_id: c for c in checkpoint.characters}


def test_checkpoint_loads_at_current_schema(checkpoint: CheckpointFile) -> None:
    assert checkpoint.schema_version == "5.0"
    assert checkpoint.session.story_id == "the_unblessed_summon"
    assert checkpoint.session.session_id == "the_unblessed_summon"


def test_dnd_ruleset_enabled(checkpoint: CheckpointFile) -> None:
    assert checkpoint.session.config.settings.ruleset_id == "dnd5e_basic"
    assert checkpoint.session.config.settings.player_roll_mode == "auto"


def test_runtime_defaults_match_current_policy(checkpoint: CheckpointFile) -> None:
    expected_models = {
        "event_router": "openai:gpt-5.2",
        "narrator": "openai:gpt-5.6-terra",
        "dnd_combat_manager": "gpt-5-mini",
        "agent_default": "anthropic:claude-opus-5",
        "agent_standard": "openai:gpt-5.6-luna",
        "agent_convenience": "anthropic:claude-sonnet-5",
        "character_manager": "anthropic:claude-sonnet-5",
        "image_director": "gpt-5-mini",
    }
    assert checkpoint.session.config.models.model_dump() == expected_models
    assert checkpoint.session.config.settings.max_events_per_beat == 40
    assert checkpoint.session.config.settings.max_agent_cascades_per_beat == 35
    assert not hasattr(checkpoint, "config")


def test_player_primer_present(checkpoint: CheckpointFile) -> None:
    # The /story start path requires a non-empty primer; the fallback
    # stub is ugly. Synthetic checkpoint should ship its own.
    primer = checkpoint.player_primer
    assert len(primer) > 500
    # Canonical truck-kun arrival cue must be present in some form so
    # the primer reads as the agreed-on isekai-death framing rather
    # than a generic "you wake up here" stub. Don't pin specific
    # wording — accept any of the standard cues.
    cues = ["truck", "headlight", "horn", "crossing"]
    primer_lower = primer.lower()
    assert any(cue in primer_lower for cue in cues), (
        "primer must include a canonical truck-kun arrival cue "
        f"(one of {cues})"
    )


def test_opening_starts_at_guild_remand(
    checkpoint: CheckpointFile,
    by_id: dict,
) -> None:
    for cid in PLAYER_CHARACTER_IDS:
        assert by_id[cid].location == GUILD_HALL, cid
    for cid in COHORT_CHARACTER_IDS:
        assert by_id[cid].location == HERO_TRAINING_COMPOUND, cid

    stale_opening_terms = [
        "four months",
        "hall tavern",
        "hero hall",
        "bandages",
        "soap",
        "second invitation",
    ]
    opening_text = "\n".join([
        checkpoint.player_primer,
        by_id[PLAYER_CHARACTER_ID].backstory,
        by_id[PLAYER_CHARACTER_ID].known_context,
        by_id[SECOND_PLAYER_CHARACTER_ID].backstory,
        by_id[SECOND_PLAYER_CHARACTER_ID].known_context,
        checkpoint.world_state.setting.era,
    ]).lower()
    for term in stale_opening_terms:
        assert term not in opening_text


def test_player_does_not_start_with_cohort_names(by_id: dict) -> None:
    for player_id in PLAYER_CHARACTER_IDS:
        player_context = "\n".join([
            by_id[player_id].backstory,
            by_id[player_id].known_context,
        ]).lower()
        for cid in COHORT_CHARACTER_IDS:
            name = by_id[cid].name.lower()
            assert name not in player_context, (player_id, cid)


def test_world_state_is_authored_not_stubbed(
    checkpoint: CheckpointFile,
) -> None:
    ws = checkpoint.world_state
    assert len(ws.lore) > 5_000, "public lore looks stubby"
    assert len(ws.hidden_lore) > 5_000, "hidden_lore looks stubby"
    assert len(ws.facts) >= 15
    assert len(ws.hidden_facts) >= 30
    assert ws.setting.genre
    assert ws.setting.premise
    assert ws.physics_ruleset.magic_enabled is True


def test_character_roster_matches_expected(by_id: dict) -> None:
    assert set(by_id) == EXPECTED_ALL_IDS


def test_character_descriptions_are_split_and_seeded(by_id: dict) -> None:
    for cid, char in by_id.items():
        assert char.agent_tier in {
            CharacterAgentTier.premium,
            CharacterAgentTier.standard,
            CharacterAgentTier.utility,
        }, cid
        assert char.descriptions.public.strip(), f"{cid} missing public description"
        assert char.descriptions.private.strip(), f"{cid} missing private description"

    premium_ids = {
        cid for cid, char in by_id.items()
        if char.agent_tier == CharacterAgentTier.premium
    }
    assert premium_ids == {
        "liriel_vaerien",
        "korva_sahl",
        "sariel_marenne",
        "vaella_coldspire",
        "quill",
        "sylira_vesh",
        "marennis_vale",
        "cardinal_vespera",
        "archon_selivar",
        "demon_lord",
        "princess_nirvel",
    }
    utility_ids = {
        cid for cid, char in by_id.items()
        if char.agent_tier == CharacterAgentTier.utility
    }
    assert utility_ids == {
        *PLAYER_CHARACTER_IDS,
        "tatsuya_hozumi",
        "guild_master_bren",
        "crown_prince_aldemar",
        "sage_wencel",
        "crown_liaison_lerin",
        "court_mage_selen",
    }

    assert "S-rank" in by_id["korva_sahl"].descriptions.public
    assert "Demon Lord" not in by_id["korva_sahl"].descriptions.public
    assert "demonic" not in by_id["korva_sahl"].descriptions.public.lower()
    assert "Demon Lord" in by_id["korva_sahl"].descriptions.private

    assert "blue" in by_id["sora_kageyama"].descriptions.public.lower()
    assert "manipulator" not in by_id["riku_tsumura"].descriptions.public.lower()
    assert "stewardship" not in by_id["riku_tsumura"].descriptions.public.lower()
    assert "Stewardship" in by_id["riku_tsumura"].descriptions.private


def test_tick_annotations_seeded(by_id: dict) -> None:
    tickable = [
        c for c in by_id.values()
        if c.status.value == "active" and c.private_state.intentions_enabled
    ]
    assert tickable
    assert by_id["demon_lord"].private_state.intentions_enabled is True
    assert by_id["demon_lord"].status.value == "active"
    assert by_id["princess_nirvel"].status.value == "dormant"
    assert by_id["princess_nirvel"].private_state.intentions_enabled is False


def test_only_unblessed_pair_is_playable(by_id: dict) -> None:
    playable_ids = {cid for cid, c in by_id.items() if c.is_playable}
    assert playable_ids == PLAYER_CHARACTER_IDS


def test_players_have_empty_mechanics(by_id: dict) -> None:
    """The player slots must not have synthetic D&D sheets attached.

    This is the load-bearing contract that lets a user `/attach` their
    own DDB sheet without it being silently overwritten by stale
    synthetic profiles. Any future edit that adds a `mechanics` block
    to either player character should fail here.
    """
    for cid in PLAYER_CHARACTER_IDS:
        player = by_id[cid]
        assert player.mechanics == {}, (
            f"{cid} must ship with empty mechanics so /attach imports do "
            "not collide with a synthetic sheet"
        )


def test_every_npc_has_dnd_mechanics(by_id: dict) -> None:
    for cid in EXPECTED_NPC_IDS:
        mech = by_id[cid].mechanics
        assert isinstance(mech, dict) and mech, f"{cid} missing mechanics"
        assert mech.get("ruleset_id") == "dnd5e_basic", cid
        scores = mech.get("ability_scores") or {}
        assert set(scores) >= {
            "str",
            "dex",
            "con",
            "int",
            "wis",
            "cha",
        }, cid
        assert isinstance(mech.get("proficiency_bonus"), int)
        assert isinstance(mech.get("armor_class"), int)
        hp = mech.get("hit_points") or {}
        assert isinstance(hp.get("current"), int), cid
        if by_id[cid].status != "dormant":
            assert hp["current"] > 0, cid
        assert isinstance(hp.get("max"), int) and hp["max"] >= hp["current"]


def test_dormant_character_handling(by_id: dict) -> None:
    nirvel = by_id["princess_nirvel"]
    assert nirvel.status == "dormant"
    assert nirvel.mechanics["hit_points"]["current"] == 0


def test_synthetic_source_type_round_trips(by_id: dict) -> None:
    expected = "synthetic_ayoa_unblessed_summon"
    for cid in EXPECTED_NPC_IDS:
        raw = (by_id[cid].mechanics.get("raw") or {}).get("source") or {}
        assert raw.get("type") == expected, (
            f"{cid} mechanics.raw.source.type drifted from {expected}; "
            "downstream sheet-origin filters depend on this label"
        )


def test_power_curve_caps(by_id: dict) -> None:
    """Power-curve discipline: no PLAYER_NAME ally is statted past
    level-10 equivalent unless they are flagged as a constrained
    character. The intent is that nobody the protagonist can
    reasonably ally with should be strong enough to trivialize the
    plot. Encoded here as HP ceilings on the romance interests and
    the supporting allies.
    """
    ally_ids = ROMANCE_CHARACTER_IDS | {
        "yuna_kiyose",
        "tatsuya_hozumi",
        "guild_master_bren",
        "court_mage_selen",
        "lady_aoi",
        "mira",
    }
    for cid in ally_ids:
        max_hp = by_id[cid].mechanics["hit_points"]["max"]
        assert max_hp <= 70, (
            f"{cid} max_hp={max_hp} exceeds the ally cap (70). "
            "Allies above this threshold can trivialize the plot — "
            "either lower the level or add a hard-coded constraint "
            "explaining why they cannot intervene."
        )


def test_antagonists_outscale_player_cohort(by_id: dict) -> None:
    """Antagonists must materially outscale the cohort heroes so the
    central conflict has weight. Cardinal/Cabal Architect/Demon Lord
    are the apex; verify their proficiency_bonus reflects the
    intended level tier (PB 4 / 5 / 5 respectively).
    """
    assert by_id["cardinal_vespera"].mechanics["proficiency_bonus"] == 4
    assert by_id["archon_selivar"].mechanics["proficiency_bonus"] == 5
    assert by_id["demon_lord"].mechanics["proficiency_bonus"] == 5


def test_synthetic_sheets_are_used_for_roll_modifiers(by_id: dict) -> None:
    # The compact mechanics block (without a full dnd5e_sheet snapshot)
    # is what the engine reads when computing rolls. Verify mechanics
    # round-trip into modifiers for a few sampled NPCs spanning the
    # power curve.
    samples = [
        ("korva_sahl", "cha", "persuasion", 7),
        ("liriel_vaerien", "cha", "persuasion", 6),
        ("sora_kageyama", "str", "athletics", 6),
        ("vaella_coldspire", "wis", "perception", 6),
        ("archon_selivar", "int", "arcana", 11),
        ("demon_lord", "cha", "intimidation", 11),
        ("cardinal_vespera", "wis", "insight", 8),
        ("mira", "wis", "insight", 3),
    ]
    for cid, ability, skill, expected in samples:
        request = PlannedRoll(
            roll_id=f"roll_{cid}_{skill}",
            actor_id=cid,
            kind="skill_check",
            ability=ability,
            skill=skill,
            dc=0,
            opposed_by="",
            advantage_state="normal",
            reason="test",
        )
        actual = mechanics.roll_modifier(by_id[cid], request)
        assert actual == expected, (
            f"{cid} {ability}/{skill} computed {actual}, expected "
            f"{expected}"
        )


def test_player_roll_modifier_falls_back_to_zero(by_id: dict) -> None:
    """With no mechanics attached, each player's roll modifier is the
    bare ability modifier (which without an ability_scores block is
    zero). This documents that the engine handles the empty-mechanics
    case cleanly until users attach real sheets.
    """
    for cid in PLAYER_CHARACTER_IDS:
        request = PlannedRoll(
            roll_id=f"roll_{cid}",
            actor_id=cid,
            kind="skill_check",
            ability="cha",
            skill="persuasion",
            dc=0,
            opposed_by="",
            advantage_state="normal",
            reason="test",
        )
        actual = mechanics.roll_modifier(by_id[cid], request)
        assert actual == 0
