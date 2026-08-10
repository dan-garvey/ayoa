"""Smoke tests for the hand-authored One-Star Ascension (Pick Me Up) story seed.

Guards the runtime contracts the engine depends on for this floor-zero seed
rather than freezing prose: schema/ruleset shape, the router/Master/units role
split, a blank user-created player-character, the bounded-cast discipline, the
dormant summon pool of pre-authored candidates, and the router-facing reveals.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.characters import CharacterAgentTier
from app.schemas.checkpoint import CheckpointFile


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
    "pip_secondlight",
    "bex_greenpull",
    "dala_greenpull",
    "iselle_the_guide",
    "mother_yset",
    "the_master",
    "halcyon_of_the_gilded_march",
    "soren_ironvow",
    "castor_valebrand",
    "wren_thelantern",
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
    "soren_ironvow",
    "castor_valebrand",
    "wren_thelantern",
    "veil_the_unnumbered",
}

# The blank, user-created player slot is intentionally empty (filled in play).
BLANK_PLAYER_ID = "one_star_newcomer"


def _load_checkpoint() -> CheckpointFile:
    raw = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return CheckpointFile.model_validate(raw)


def test_checkpoint_loads_as_rules_neutral_magic_story() -> None:
    checkpoint = _load_checkpoint()

    assert checkpoint.schema_version == "4.0"
    assert checkpoint.session.story_id == "one_star_ascension_s1"
    assert checkpoint.session.session_id == "one_star_ascension_s1"
    assert checkpoint.session.config.settings.ruleset_id == "narrative"
    assert checkpoint.world_state.physics_ruleset.magic_enabled is True
    assert checkpoint.session.active_combat is None
    for character in checkpoint.characters:
        assert character.mechanics == {}, character.character_id


def test_roster_shape_and_player_binding() -> None:
    checkpoint = _load_checkpoint()
    by_id = {c.character_id: c for c in checkpoint.characters}

    assert set(by_id) == EXPECTED_CHARACTER_IDS

    # /story start auto-binds the creator to the FIRST is_playable character,
    # so the blank user-created player slot must lead the roster.
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


def test_player_character_is_a_blank_user_created_slot() -> None:
    checkpoint = _load_checkpoint()
    pc = next(c for c in checkpoint.characters if c.character_id == BLANK_PLAYER_ID)

    # Intentionally empty: the human creates this character in play.
    assert pc.backstory == ""
    assert pc.personality == ""
    assert pc.known_context == ""
    assert pc.descriptions.public == ""
    assert pc.descriptions.private == ""
    assert pc.private_state.secrets == []
    assert pc.private_state.intentions_enabled is False
    assert pc.mechanics == {}


def test_master_is_offstage_unreachable_actor() -> None:
    checkpoint = _load_checkpoint()
    master = next(c for c in checkpoint.characters if c.character_id == "the_master")

    assert master.location == "the_masters_screen"
    assert master.private_state.intentions_enabled is True
    private = master.descriptions.private.lower()
    assert "novice" in private
    assert "router" in private  # explicitly adjudicated by the router, not it


def test_cast_is_bounded_so_the_lobby_never_becomes_thousands_of_agents() -> None:
    checkpoint = _load_checkpoint()

    movers = {
        c.character_id
        for c in checkpoint.characters
        if c.private_state.intentions_enabled
    }
    assert movers == EXPECTED_INTENTION_MOVERS
    assert len(movers) <= 3
    assert "the_master" in movers

    non_movers = [
        c for c in checkpoint.characters
        if not c.private_state.intentions_enabled
    ]
    assert len(non_movers) > len(movers)

    # A small authored roster; any lobby crowd lives as ambient world text.
    assert len(checkpoint.characters) <= 14

    facts = "\n".join(checkpoint.world_state.facts).lower()
    assert "only a small active party" in facts
    assert "dormant" in facts


def test_floor_zero_start_and_summon_pool() -> None:
    checkpoint = _load_checkpoint()
    by_id = {c.character_id: c for c in checkpoint.characters}

    # Floor-zero: a brand-new lobby / tutorial start, not the old mid-climb stall.
    facts = "\n".join(checkpoint.world_state.facts).lower()
    assert "the very start of a climb" in facts or "brand-new master" in facts
    assert "tutorial" in facts
    assert "stalled on the eighth floor" not in facts

    # The pre-authored characters are preserved as a dormant, quarantined pool:
    # not in play until the router summons/introduces one.
    for pool_id in SUMMON_POOL_IDS:
        pooled = by_id[pool_id]
        assert pooled.status.value == "dormant", pool_id
        assert pooled.location == "unsummoned_pool", pool_id
        assert pooled.is_playable is False, pool_id
        assert pooled.private_state.intentions_enabled is False, pool_id

    # Router-only guidance describes the pool + floor-zero framing.
    hidden_lore = checkpoint.world_state.hidden_lore.lower()
    assert "summon pool" in hidden_lore
    assert "unsummoned_pool" in hidden_lore
    assert "floor-zero framing" in hidden_lore

    # A fresh floor-zero starter party exists (light NPCs) alongside the blank PC.
    for starter in ("pip_secondlight", "bex_greenpull", "dala_greenpull"):
        assert by_id[starter].is_playable is True
        assert by_id[starter].status.value == "active"


def test_router_facing_game_world_manual_is_present() -> None:
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state

    facts = "\n".join(ws.facts).lower()
    lore = ws.lore.lower()

    assert "of their own will" in facts
    assert "does not puppet" in facts
    assert "synthes" in facts
    assert "gems" in facts and "stamina" in facts
    assert "one-star" in facts or "one star" in facts
    assert "dialogue box" in facts  # System pop-up notices are a core texture
    for token in ["interface", "niflheim", "synthesis menu", "intervention"]:
        assert token in lore, token


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
    assert "arbitrary early" in hidden

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
    assert "blank fodder" in facts
    assert "realized person" in facts

    facts_lore = ("\n".join(ws.facts) + "\n" + ws.lore).lower()
    assert "1-star to level 10" in facts_lore
    assert "6-star to 99" in facts_lore

    hidden_lore = ws.hidden_lore.lower()
    hidden = (ws.hidden_lore + "\n" + "\n".join(ws.hidden_facts)).lower()
    assert "promotion and memory" in hidden_lore
    assert "new summons" in hidden_lore
    assert "four stars" in hidden or "four-star" in hidden
    assert "cannot see inside the promotion chamber" in hidden
    assert "three- to five-star" in hidden

    master = next(c for c in checkpoint.characters if c.character_id == "the_master")
    msecrets = " ".join(master.private_state.secrets).lower()
    assert "promotion chamber" in msecrets and "memories" in msecrets

    halcyon = next(
        c for c in checkpoint.characters
        if c.character_id == "halcyon_of_the_gilded_march"
    )
    assert "memory-wiped" not in halcyon.descriptions.private.lower()


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
    assert "hero reaction research" in hidden
    assert "cannot see them" in facts
    assert "system-blind" in hidden
    # Guard against regressing to the old (wrong) "shared reality" framing.
    assert "the heroes' shared reality" not in facts

    # The player-character is the authored exception: an out-of-world summon who
    # reads the System from birth (both the POV necessity and a live anomaly).
    assert "another world" in facts
    assert "exception" in facts
    assert "one authored anomaly" in hidden

    # The narrator must gate dialogue boxes by who can actually see the System,
    # and the protagonist's sight is not tied to the Master being logged in.
    assert "unresearched" in rules
    assert "system-sight" in rules

    # The exception lives in world-truth/router context, NOT on the PC record,
    # so the blank user-created slot stays blank.
    pc = next(c for c in checkpoint.characters if c.character_id == BLANK_PLAYER_ID)
    assert pc.known_context == ""


def test_seed_has_depth_without_dnd_mechanics() -> None:
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state

    assert len(checkpoint.player_primer) > 500
    assert len(ws.lore) > 3000
    assert len(ws.hidden_lore) > 2000
    assert len(ws.facts) >= 16
    assert len(ws.hidden_facts) >= 20

    for character in checkpoint.characters:
        # The blank user-created player slot is intentionally empty.
        if character.character_id == BLANK_PLAYER_ID:
            continue
        assert character.backstory.strip(), character.character_id
        assert character.personality.strip(), character.character_id
        assert character.known_context.strip(), character.character_id
        assert character.descriptions.public.strip(), character.character_id
        assert character.descriptions.private.strip(), character.character_id
        assert character.visuals.default_loadout.strip(), character.character_id
