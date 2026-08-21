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

from app.engine.character_manager import _assemble_knowledge_grant
from app.engine.reviewed_visual_references import validate_story_visual_references
from app.schemas.characters import CharacterAgentTier, PlayerSlotKind
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
    "renna_holt",
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


def test_checkpoint_loads_as_rules_neutral_magic_story() -> None:
    checkpoint = _load_checkpoint()

    assert checkpoint.schema_version == "5.0"
    assert checkpoint.session.story_id == "one_star_ascension_s1"
    assert checkpoint.session.session_id == "one_star_ascension_s1"
    assert checkpoint.session.config.settings.ruleset_id == "narrative"
    assert checkpoint.world_state.physics_ruleset.magic_enabled is True
    assert checkpoint.session.active_combat is None
    for character in checkpoint.characters:
        assert character.mechanics == {}, character.character_id


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
    assert checkpoint.world_state.opening.requires_claim_confirmation is True
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
        ]
        for character_id in BOUND_IDENTITY_REFERENCE_IDS
    }
    assert len(checkpoint.reviewed_visual_references) == 61
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
    assert pc.player_slot_kind == PlayerSlotKind.player_authored
    assert pc.player_guidance

    # The record itself carries no authored arrival, location narration, or
    # fallback personality. Claim-aware opening policy decides whether it appears.
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
    assert opening.requires_claim_confirmation is True


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
            character.backstory,
            character.personality,
            character.known_context,
            character.descriptions.public,
            character.descriptions.private,
            *character.private_state.goals,
            *character.private_state.current_objectives,
            *character.private_state.secrets,
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
    assert master.private_state.intentions_enabled is True
    private = master.descriptions.private.lower()
    assert "novice" in private
    assert "fallible" in private
    assert "uncertain outcomes" in private
    assert "router" not in private


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
    assert len(checkpoint.characters) <= 18

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

    # Ordinary one-stars are generated for an actual summon instead of living as
    # fixed active starters. Renna is the only authored one-star exception:
    # dormant, optional, publicly ordinary, and privately high-potential.
    assert "bex_greenpull" not in by_id
    assert "dala_greenpull" not in by_id
    seeded_tier_ones = {
        character.character_id
        for character in checkpoint.characters
        if character.knowledge_tier == 1
    }
    assert seeded_tier_ones == {"renna_holt"}
    renna = by_id["renna_holt"]
    assert renna.name == "Renna Holt"
    assert renna.status.value == "dormant"
    assert renna.location == "unsummoned_pool"
    assert renna.is_playable is False
    assert len(renna.backstory.split()) < 40
    assert len(renna.personality.split()) < 55
    renna_private = (
        renna.descriptions.private
        + " "
        + " ".join(renna.private_state.secrets)
    ).lower()
    assert "retain" in renna_private
    assert "pattern" in renna_private
    assert "no visible marker" in renna_private

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
        private = by_id[pool_id].descriptions.private.lower()
        assert not any(leak in private for leak in lifecycle_leaks), pool_id


def test_opening_goblin_chapter_is_authored_without_stale_slime_guidance() -> None:
    checkpoint = _load_checkpoint()
    by_id = {character.character_id: character for character in checkpoint.characters}

    public_facts = "\n".join(checkpoint.world_state.facts).lower()
    router_only = (
        checkpoint.world_state.hidden_lore
        + "\n"
        + "\n".join(checkpoint.world_state.hidden_facts)
    ).lower()
    guide_context = by_id["iselle_the_guide"].known_context.lower()

    assert "goblin" in public_facts
    assert "goblin" in guide_context
    assert "acid slime" not in public_facts
    assert "acid slime" not in guide_context

    for floor in range(1, 6):
        assert f"floor {floor}" in router_only
    assert "five minutes" in router_only
    assert "hundreds of goblins" in router_only
    assert "five-floor cadence" in router_only
    assert "materially different motif" in router_only


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
        assert character.private_state.intentions_enabled is False


def test_grade_memory_status_and_reserve_authority_are_coherent() -> None:
    checkpoint = _load_checkpoint()
    facts = "\n".join(checkpoint.world_state.facts).lower()
    hidden = (
        checkpoint.world_state.hidden_lore
        + "\n"
        + "\n".join(checkpoint.world_state.hidden_facts)
    ).lower()
    by_id = {character.character_id: character for character in checkpoint.characters}

    assert "two-stars retain thin" in facts
    assert "four-stars cross the memory threshold" in facts
    assert "status mapping" in hidden
    assert "remains active" in hidden
    assert "not lower-stage templates" in hidden
    assert "trade, poaching, rescue, transfer" in hidden
    assert "through the interface" not in facts

    veil_public = (
        by_id["veil_the_unnumbered"].public_sheet.role
        + " "
        + by_id["veil_the_unnumbered"].public_sheet.appearance
    ).lower()
    assert "awakened" not in veil_public
    assert "status window" not in veil_public


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
    assert "one-star summons arrive blank" in facts
    assert "two-stars retain thin" in facts
    assert "four-stars cross the memory threshold" in facts
    assert "five-stars retain" in facts

    facts_lore = ("\n".join(ws.facts) + "\n" + ws.lore).lower()
    assert "1-star to level 10" in facts_lore
    assert "6-star to 99" in facts_lore

    hidden_lore = ws.hidden_lore.lower()
    hidden = (ws.hidden_lore + "\n" + "\n".join(ws.hidden_facts)).lower()
    assert "promotion and memory" in hidden_lore
    assert "new summons" in hidden_lore
    assert "four stars" in hidden or "four-star" in hidden
    assert "cannot see inside the promotion chamber" in hidden
    for grade in ("three-stars", "four-stars", "five-stars"):
        assert grade in hidden

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


def test_lobby_master_and_guide_framing() -> None:
    """Playtest fixes: the party is formed by the Master (the guide only
    assists), Niflheim is a home hub distinct from the Tower, and the
    tutorial-guide's unit-management coaching is aimed at the Master, never at
    the player-Hero; plus newcomer POV/jargon discipline for the blank Hero."""
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
    assert by_id["renna_holt"].public_sheet.faction == "Hero"

    facts = "\n".join(ws.facts).lower()
    lore = ws.lore.lower()
    rules = checkpoint.session.config.narrative_rules
    rules_lower = rules.lower()

    # Concern 4: the lobby is a home hub, distinct from the Tower's floors.
    assert "not part of the tower" in facts
    assert "not a part of the tower" in lore
    assert "lobby vs. tower" in rules_lower

    # Concern 1/5: the Master prepares the party before deployment; the guide
    # only assists and the deployed Heroes command themselves.
    assert "the master prepares the party before a mission" in facts
    assert "after deployment, the heroes command themselves" in facts
    iselle = by_id["iselle_the_guide"]
    assert "master" in iselle.public_sheet.role.lower()
    guide_ctx = iselle.known_context.lower()
    assert "the master chooses the floor and prepares the party" in guide_ctx
    assert "does not treat a hero" in guide_ctx

    # Concern 5: the narrator is told to aim the guide's coaching at the Master.
    assert "tutorial-guide's audience" in rules
    assert "iselle serves the master" in rules_lower

    # Concern 3: newcomer POV / jargon discipline for the blank Hero, with the
    # gacha terms named as things the narrator must NOT speak in its own voice.
    assert "newcomer pov and jargon" in rules_lower
    assert "one-star" in rules


def test_tower_floor_exit_requires_extraordinary_means() -> None:
    """Every model that can decide or render a deployment receives the same
    floor-exit law, while tactical withdrawal within the floor remains valid."""
    checkpoint = _load_checkpoint()
    ws = checkpoint.world_state
    by_id = {c.character_id: c for c in checkpoint.characters}

    router_contract = "\n".join((*ws.facts, ws.lore)).lower()
    narrator_contract = checkpoint.session.config.narrative_rules.lower()
    adjudication_contract = (
        ws.hidden_lore + "\n" + "\n".join(ws.hidden_facts)
    ).lower()
    tier_one_grant, _ = _assemble_knowledge_grant(checkpoint, 1)
    tier_one_contract = tier_one_grant.lower()
    master_contract = "\n".join(
        (
            ws.setting.play_guidance,
            by_id["the_master"].player_guidance,
            by_id["the_master"].personality,
            by_id["the_master"].known_context,
        )
    ).lower()

    for contract_name, contract in (
        ("router", router_contract),
        ("narrator", narrator_contract),
        ("adjudication", adjudication_contract),
        ("generated one-star", tier_one_contract),
        ("Master", master_contract),
    ):
        assert "very rare escape item" in contract, contract_name
        assert "very powerful magic" in contract, contract_name
        assert "lobby" in contract, contract_name

    assert "tactical withdrawal" in router_contract
    assert "tactical withdrawal" in narrator_contract
    assert "another position on the same floor" in adjudication_contract
    assert "ordinary retreat never changes" in adjudication_contract

    # Existing seeded agents receive authoritative known_context instead of
    # the generated-character tier grant, so each social character needs the
    # same common floor law on its own record.
    for character in checkpoint.characters:
        if (
            not character.known_context
            or character.character_id == "warden_of_the_eighth"
        ):
            continue
        known = character.known_context.lower()
        assert "very rare escape item" in known, character.character_id
        assert "very powerful magic" in known, character.character_id
        assert "current floor" in known, character.character_id

    # Preserve the authored-Newcomer contract and the inert floor boss.
    assert by_id[BLANK_PLAYER_ID].known_context == ""
    warden_context = by_id["warden_of_the_eighth"].known_context.lower()
    assert "tower-floor law" not in warden_context

    all_model_contracts = "\n".join(
        (
            router_contract,
            narrator_contract,
            adjudication_contract,
            tier_one_contract,
            master_contract,
            by_id["iselle_the_guide"].known_context.lower(),
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

    # Facilities exist with gacha purposes and are built/upgraded (the missing
    # synthesis chamber is now present; the shrine is folded into summoning).
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
    assert "facilities and lobby management" in lore
    assert "builds and upgrades" in facts
    assert "waiting room" in facts
    assert "shrine" not in facts_lore  # repurposed, not a purposeless building

    # Lobby restoration with the permadeath / old-amputation exceptions.
    assert "mends its own" in facts
    assert "cannot regrow a limb" in facts
    assert "do not carry wounds" in rules

    # The guide is also the warden: compels refusers, lethal defense protocol.
    assert "warden" in facts
    assert "lethal defense protocol" in facts
    iselle = by_id["iselle_the_guide"]
    assert "warden" in iselle.public_sheet.role.lower()
    guide_ctx = iselle.known_context.lower()
    assert "lethal defense protocol" in guide_ctx
    assert "compel a hero who refuses to deploy" in guide_ctx

    # A durable name must not silently reintroduce immunity to the protocol.
    assert "warden" in hidden
    assert "treat it exactly like synthesis" not in hidden
    assert "never an arbitrary instant kill" not in hidden

    # The Master's lobby toolkit includes facilities and transformation.
    master = by_id["the_master"]
    persona = master.personality.lower()
    assert "facilities and upgrades" in persona
    assert "transformation" in persona


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
            master.personality,
            master.known_context,
        )
    ).lower()
    for management_choice in (
        "summon",
        "facilities",
        "choose the floor",
        "roster",
        "loadout",
        "opening formation",
    ):
        assert management_choice in player_contract

    # The human-facing Master contract permits observation, including the
    # one-way audio feed, while explicitly assigning mission choices to Heroes.
    assert "during a mission" in master.player_guidance.lower()
    assert "mostly watch" in player_contract
    assert "switch the available live hero feed" in player_contract
    assert "advances through the heroes' next actions" in player_contract
    assert "hear" in master.player_guidance.lower()
    assert "audible hero dialogue" in master.player_guidance.lower()
    assert "cannot speak or type" in master.player_guidance.lower()
    assert "heroes choose targets" in player_contract
    assert "cannot choose targets" in master.known_context.lower()

    adjudication = (
        ws.hidden_lore + "\n" + "\n".join(ws.hidden_facts)
    ).lower()
    assert "master mission boundary (authoritative)" in adjudication
    assert "crossing the deployment gate commits" in adjudication
    assert "mission advances without routine master input" in adjudication
    assert "next deployed hero under pressure acts" in adjudication
    assert "after at most one brief coordination exchange" in adjudication
    assert (
        "concrete movement, scouting, attack, or tactical withdrawal "
        "to another position on the same floor"
    ) in adjudication
    assert "never stop on a warning or approaching threat" in adjudication
    assert "none exists at the start" in adjudication

    # The tutorial guide must not recreate the removed tactical control loop.
    guide_contract = "\n".join(
        (iselle.public_sheet.role, iselle.personality, iselle.known_context)
    ).lower()
    assert "heroes act for themselves" in guide_contract
    assert "does not ask the master to choose targets" in guide_contract
    for obsolete_tactical_cue in (
        "target priority",
        "priority directives",
        "item and consumable use",
        "advance or retreat",
    ):
        assert obsolete_tactical_cue not in player_contract
        assert obsolete_tactical_cue not in guide_contract


def test_knowledge_tier_ladder_gradient() -> None:
    """Knowledge, authoring depth, and presentation all scale by story tier."""
    checkpoint = _load_checkpoint()
    tiers = {t.tier: t for t in checkpoint.world_state.knowledge_tiers}

    assert set(tiers) == {1, 2, 3, 4, 5, 6}

    # One-star is near-blank and has no real plot knowledge; high rungs do.
    assert "moebius" not in tiers[1].world_knowledge.lower()
    assert "fade" not in tiers[1].world_knowledge.lower()
    t5_world = tiers[5].world_knowledge.lower()
    assert "moebius" in t5_world
    assert "fade" in t5_world

    guidance_fields = {
        "backstory_depth",
        "personality_depth",
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

    # Every generative dimension is materially richer by the first rare rung.
    low = tiers[2].generation_guidance.model_dump()
    rich = tiers[3].generation_guidance.model_dump()
    for field in guidance_fields - {"presentation_guidance"}:
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
    knowledge to high tiers, carries the rung's agent tier, and is inert
    (no block, default agent tier) when the story authors no ladder."""
    checkpoint = _load_checkpoint()

    assert _assemble_knowledge_grant(checkpoint, 0) == ("", None)

    tiers = {t.tier: t for t in checkpoint.world_state.knowledge_tiers}
    tiers[1].generation_guidance.visual_salience = "TARGET_ONE_MARKER"
    tiers[5].generation_guidance.visual_salience = "TARGET_FIVE_MARKER"

    grant1, agent1 = _assemble_knowledge_grant(checkpoint, 1)
    assert "Tier 1" in grant1
    assert "## Authored Generation Budget (authoritative)" in grant1
    assert "TARGET_ONE_MARKER" in grant1
    assert "moebius" not in grant1.lower()
    assert "fade" not in grant1.lower()
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
    assert "## Knowledge Budget (authoritative)" in knowledge_only
    assert "## Authored Generation Budget" not in knowledge_only

    # A story with no ladder is unaffected: no budget block, default agent tier.
    checkpoint.world_state.knowledge_tiers = []
    assert _assemble_knowledge_grant(checkpoint, 3) == ("", None)


def test_seeded_rare_characters_scale_depth_and_public_visual_identity() -> None:
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
    backstory_depths = [len(character.backstory.split()) for _tier, character in scaled]
    personality_depths = [
        len(character.personality.split()) for _tier, character in scaled
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
    assert backstory_depths == sorted(backstory_depths)
    assert len(set(backstory_depths)) == len(backstory_depths)
    assert personality_depths == sorted(personality_depths)
    assert len(set(personality_depths)) == len(personality_depths)
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

    # The authored one-star exception remains intentionally sparse and visually
    # shared even though its private growth ceiling is unusual.
    renna = by_id["renna_holt"]
    assert len(renna.backstory.split()) < backstory_depths[0] / 4
    assert len(renna.personality.split()) < personality_depths[0] / 3
    assert len(
        (
            renna.public_sheet.appearance
            + " "
            + renna.visuals.default_loadout
        ).split()
    ) < visual_depths[0] / 2


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
