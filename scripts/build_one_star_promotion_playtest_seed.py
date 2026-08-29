"""Build the shipped One-Star promotion/sprite playtest story.

The fixture is a later-state copy of ``one_star_ascension_s1``. It preserves
the reviewed art registry byte-for-byte while changing only authored story
state needed to exercise seeded reveal, generated-sprite prewarm, synthesis,
and generated reveal without replaying the opening campaign.
"""

from __future__ import annotations

# ruff: noqa: E402 -- executable scripts add the repository root to sys.path.

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.engine.one_star_adapter import (
    load_one_star_account,
    load_one_star_hero,
)
from app.engine.one_star_progression import (
    apply_experience,
    build_generated_hero,
    experience_to_reach_level,
    rebalance_hero,
)
from app.schemas.characters import (
    CharacterAgentTier,
    CharacterDescriptions,
    CharacterRecord,
    CharacterStatus,
    CharacterVisuals,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.one_star import (
    ONE_STAR_ACCOUNT_KEY,
    ONE_STAR_HERO_KEY,
)
from app.schemas.one_star_character_gen import AuthoredOneStarHeroMechanics


SOURCE_STORY_ID = "one_star_ascension_s1"
TARGET_STORY_ID = "one_star_ascension_s1_promotion_playtest"
SOURCE_STORY_DIR = REPO_ROOT / "app/storage/stories" / SOURCE_STORY_ID
TARGET_STORY_DIR = REPO_ROOT / "app/storage/stories" / TARGET_STORY_ID
FACELESS_CHARACTER_ID = "promotion_playtest_faceless"


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one seed phrase: {old!r}")
    return text.replace(old, new, 1)


def _character(checkpoint: CheckpointFile, character_id: str) -> CharacterRecord:
    return next(
        character
        for character in checkpoint.characters
        if character.character_id == character_id
    )


def _set_seeded_hero_level(
    checkpoint: CheckpointFile,
    *,
    character_id: str,
    current_stars: int,
    level: int,
) -> None:
    character = _character(checkpoint, character_id)
    hero = load_one_star_hero(character)
    if hero is None:
        raise RuntimeError(f"{character_id} is not a One-Star Hero")
    _owner, account = load_one_star_account(checkpoint)
    hero.current_stars = current_stars
    hero.level = 1
    hero.experience_points = 0
    hero.owner_lobby_id = account.config.lobby_id
    hero.acquisition_event_id = "promotion_playtest_seed"
    hero.terminal_cause = ""
    hero.terminal_event_id = ""
    rebalance_hero(hero=hero, config=account.config, restore_full_hp=True)
    apply_experience(
        hero=hero,
        experience_delta=experience_to_reach_level(level, account.config),
        config=account.config,
    )
    character.mechanics[ONE_STAR_HERO_KEY] = hero.model_dump(mode="json")
    character.status = CharacterStatus.active
    character.location = account.config.lobby_location_label
    character.clock_at_s = 0
    character.last_agent_turn_at_s = None
    character.pending_observations = []
    character.private_state.intentions_enabled = True


def _faceless_character(checkpoint: CheckpointFile) -> CharacterRecord:
    _owner, account = load_one_star_account(checkpoint)
    authored = AuthoredOneStarHeroMechanics.model_validate(
        {
            "strong_stat_id": "agility",
            "weak_stat_id": "resilience",
            "equipment": [
                {
                    "item_id": "worn_wooden_cargo_hook",
                    "name": "Worn Wooden Cargo Hook",
                    "slot": "belt",
                    "quantity": 1,
                    "durability_current": 12,
                    "durability_max": 20,
                    "tags": ["tool", "hauling", "wood", "worn"],
                    "visible": True,
                }
            ],
            "skills": [
                {
                    "skill_id": "load_balancing",
                    "name": "Load Balancing",
                    "rank": 1,
                    "capability": (
                        "Steady, shift, and carry ordinary loads without losing "
                        "balance."
                    ),
                    "tags": ["porter", "hauling", "practical"],
                    "visible": True,
                }
            ],
            "conditions": [],
            "persistent_injuries": [],
            "innate_system_sight": False,
            "hidden_capabilities": [],
        }
    )
    hero = build_generated_hero(
        character_id=FACELESS_CHARACTER_ID,
        generated=authored,
        birth_stars=1,
        config=account.config,
    )
    hero.owner_lobby_id = account.config.lobby_id
    hero.acquisition_event_id = "promotion_playtest_seed"
    apply_experience(
        hero=hero,
        experience_delta=experience_to_reach_level(10, account.config),
        config=account.config,
    )
    return CharacterRecord(
        character_id=FACELESS_CHARACTER_ID,
        name="Mara Venn",
        status=CharacterStatus.active,
        location=account.config.lobby_location_label,
        is_playable=False,
        agent_tier=CharacterAgentTier.standard,
        knowledge_tier=1,
        public_sheet=PublicSheet(
            role="a wary young woman and former dock porter",
            appearance=(
                "A young adult woman with a wiry build, blunt dark-brown "
                "hair cut close around the ears, a broad tired face, and "
                "work-callused hands."
            ),
            faction="Hero",
        ),
        descriptions=CharacterDescriptions(
            public=(
                "A practical one-star porter who watches exits and keeps "
                "her old cargo hook close."
            ),
            private=(
                "A generated one-star whose Master-facing identity remains "
                "System-veiled until her third star."
            ),
        ),
        visuals=CharacterVisuals(
            default_loadout=(
                "Rough gray cloth shirt and trousers tied with plain cords, "
                "a short worn wooden cargo hook tucked through a belt strap, "
                "and scuffed cloth shoes."
            ),
        ),
        private_state=PrivateState(
            goals=[
                "stay alive without surrendering control to the System",
                "recover a sense of self through choices she can still make",
            ],
            current_objectives=[
                "complete both freely chosen promotions after Iselle confirms the sanctioned cost and effect",
                "honor Castor's informed voluntary transfer without pretending his death is an abstraction",
            ],
            secrets=[],
            intentions_enabled=True,
        ),
        backstory=(
            "Cold summon-light left Mara with no coherent history. Her hands "
            "still know how to balance a load and test a working knot."
        ),
        personality=(
            "Mara is wary, practical, and terse. She asks direct questions, "
            "keeps near an exit, and becomes more deliberate rather than more "
            "obedient when frightened."
        ),
        known_context=(
            "Mara knows the lobby's basic sanctioned rules and that she has "
            "reached the limit of one-star training. Iselle has already "
            "briefed her that completing promotion consumes one lesser stone, "
            "advances her exactly one star, applies retained experience, and "
            "restores any authored memory tier for the new rank. Mara has "
            "freely decided to accept both available promotions in this "
            "playtest. She and Castor have already reviewed the exact "
            "synthesis result together: synthesis permanently kills its "
            "source, Castor has freely volunteered for this transfer without "
            "coercion, and Mara has chosen to receive it and finish the second "
            "promotion afterward."
        ),
        mechanics={ONE_STAR_HERO_KEY: hero.model_dump(mode="json")},
    )


def build_checkpoint() -> CheckpointFile:
    checkpoint = CheckpointFile.model_validate_json(
        (SOURCE_STORY_DIR / "ckpt_0000.json").read_text(encoding="utf-8")
    )
    checkpoint.session.session_id = TARGET_STORY_ID
    checkpoint.session.story_id = TARGET_STORY_ID
    checkpoint.session.turn_index = 0
    checkpoint.session.leading_at_s = 0
    checkpoint.session.player_name = ""
    checkpoint.session.player_character_id = ""
    checkpoint.session.character_bindings = {}
    checkpoint.session.visual_introductions = {}
    checkpoint.session.pending_engine_state_updates = []
    checkpoint.session.active_act_slots = {}
    checkpoint.session.open_cat_ii_events = []
    checkpoint.session.render_buffers = {}
    checkpoint.session.pending_narrator_render = None
    checkpoint.session.config.settings.presentation_mode = "visual_novel"

    checkpoint.session_conversation = []
    checkpoint.narrator_conversations = {}
    checkpoint.character_conversations = {}
    checkpoint.canonical_events = []
    checkpoint.visibility_log = []

    setting = checkpoint.world_state.setting
    setting.title = "One-Star Ascension — Promotion Sprite Playtest"
    setting.recommended_players = "One player, claiming the Master"
    setting.play_guidance = (
        "Claim the Master, promote Renna Holt, promote Mara Venn to two stars, "
        "synthesize level-45 Castor into Mara, then promote Mara to three "
        "stars. This copy accelerates only cap-banked XP so one donor reaches "
        "the intended image-reveal checkpoint."
    )
    checkpoint.player_primer = (
        "This is a focused visual-novel promotion fixture, copied from the "
        "One-Star Ascension seed after its opening campaign state. Claim the "
        "Master. Renna Holt and Mara Venn are level-10 one-stars, three lesser "
        "promotion stones are available, and both the Promotion and Synthesis "
        "Chambers are operational.\n\n"
        "Promote Renna once to verify her reviewed identity replaces the "
        "one-star veil. Promote Mara once to start her generated neutral and "
        "pose-expression sweep while she remains veiled. Synthesize level-45 "
        "four-star Castor into Mara, then promote Mara again. This fixture "
        "retains XP through the level-30 threshold so the second promotion "
        "lands Mara at three stars and reveals the generated sprite pack."
    )

    if checkpoint.world_state.opening is None:
        raise RuntimeError("source story has no opening policy")
    checkpoint.world_state.opening.allow_spawns = False
    checkpoint.world_state.opening.context = (
        "This is a later-state promotion playtest. The authored participants "
        "already exist and no summon or spawn occurs. Place the claimed Master "
        "at the_masters_screen and show the operational lobby management view. "
        "Renna Holt, Mara Venn, Castor, and Iselle are already present in "
        "niflheim_lobby. The immediate choices are promotion and synthesis; do "
        "not replay the tutorial, first summon, or opening arrival."
    )

    rules = checkpoint.session.config.narrative_rules
    rules = _replace_once(
        rules,
        (
            "Starting point: this is the very beginning of the climb. A "
            "brand-new Master, a freshly instanced and nearly empty lobby, "
            "the tutorial, and the first five authored floors ahead. The "
            "opening branches on the live seats: Master-only receives Renna "
            "Holt, Mirelle Voss as the fixed authored three-star, and Edren "
            "Marr; Master plus a player-chosen Newcomer receives that exact "
            "trio plus the player-chosen Newcomer; Newcomer-only receives the "
            "player-chosen Newcomer through the same opening-roster authority. "
            "Renna and Edren are freshly wiped one-stars, while Mirelle retains "
            "the deeper history appropriate to her grade. Let new relationships "
            "and the deeper questions arrive through play, later summons, and "
            "promotion rather than making any opening cast instantly loyal."
        ),
        (
            "Starting point: this focused playtest begins after initial lobby "
            "training. Renna Holt and Mara Venn are established level-10 "
            "one-stars, Castor is an established four-star veteran, and the "
            "Master is choosing promotion and synthesis. Do not replay the "
            "tutorial or first summon."
        ),
    )
    checkpoint.session.config.narrative_rules = rules

    checkpoint.world_state.facts[0] = (
        "The Hundred-Floor Tower is a summon-RPG made real. This focused "
        "playtest begins in Niflheim after initial training, with promotion "
        "and synthesis ready for the Master to test."
    )
    facility_fact_index = next(
        index
        for index, fact in enumerate(checkpoint.world_state.facts)
        if fact.startswith("Lobby facilities are physical rooms built")
    )
    checkpoint.world_state.facts[facility_fact_index] = (
        "Lobby facilities are physical rooms. In this playtest copy, the "
        "one-floor Niflheim lobby has operational Promotion and Synthesis "
        "Chambers so their embodied scenes can run immediately; the Armory "
        "and later facilities remain unbuilt."
    )

    owner, account = load_one_star_account(checkpoint)
    account.config.progression.cap_bank_extra_levels = 10
    account.state.resources.materials = {"lesser_promotion_stone": 3}
    account.state.facilities["promotion_chamber"] = 1
    account.state.pending_operation = None
    account.state.active_mission = None
    account.state.synthesis_resolution_count = 0
    account.state.applied_event_fingerprints = {}
    account.state.stored_equipment = []
    owner.mechanics[ONE_STAR_ACCOUNT_KEY] = account.model_dump(mode="json")
    owner.private_state.current_objectives = [
        "promote Renna Holt and verify her authored identity reveal",
        "promote Mara Venn once, synthesize Castor into her, then promote her again",
        "verify Mara's generated sprite set appears after the three-star reveal",
    ]

    for character in checkpoint.characters:
        if character.character_id == "halcyon_of_the_gilded_march":
            character.status = CharacterStatus.dormant

    _set_seeded_hero_level(
        checkpoint,
        character_id="renna_holt",
        current_stars=1,
        level=10,
    )
    renna = _character(checkpoint, "renna_holt")
    renna.known_context = (
        "Renna knows Niflheim's basic rules and has reached level ten, the "
        "one-star limit. Iselle has already briefed her that completing "
        "promotion consumes one lesser stone, advances her exactly one star, "
        "applies retained experience, and restores any authored memory tier "
        "for the new rank. Renna has freely decided that she wants this "
        "promotion and will enter once its sanctioned cost and effect are "
        "confirmed."
    )
    renna.private_state.current_objectives = [
        "enter the Promotion Chamber once its sanctioned cost and effect are confirmed",
        "watch how the Master treats Mara and Castor",
    ]

    _set_seeded_hero_level(
        checkpoint,
        character_id="castor_valebrand",
        current_stars=4,
        level=45,
    )
    castor = _character(checkpoint, "castor_valebrand")
    castor.private_state.goals = [
        "exercise agency through the informed final transfer he chose for Mara",
        "make the record acknowledge that his death is real and voluntary",
    ]
    castor.private_state.current_objectives = [
        "enter the Synthesis Chamber as Mara's freely consenting source",
        "state his choice plainly before the already-reviewed transfer resolves",
    ]
    castor.known_context = (
        castor.known_context
        + "\n\nFor this focused playtest, Castor and Mara have already reviewed "
        "the exact synthesis result and its finality. Castor has freely "
        "volunteered to become Mara's synthesis source without coercion. He "
        "has decided to follow through even though death frightens him, and "
        "his immediate objective is to make that choice legible rather than "
        "resist or seek cancellation."
    )

    if any(
        character.character_id == FACELESS_CHARACTER_ID
        for character in checkpoint.characters
    ):
        raise RuntimeError("playtest faceless character already exists")
    checkpoint.characters.append(_faceless_character(checkpoint))

    iselle = _character(checkpoint, "iselle_the_guide")
    iselle.private_state.current_objectives = [
        "present Renna and Mara's available promotions to the Master",
        "honor the exact synthesis terms Castor and Mara already reviewed",
        "witness Castor's voluntary response before the selected transfer resolves",
        "release Mara back to the lobby for her freely chosen second promotion",
    ]

    return CheckpointFile.model_validate_json(
        checkpoint.model_dump_json(
            context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
        )
    )


def main() -> None:
    checkpoint = build_checkpoint()
    TARGET_STORY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        SOURCE_STORY_DIR / "visual-references",
        TARGET_STORY_DIR / "visual-references",
        dirs_exist_ok=True,
    )
    destination = TARGET_STORY_DIR / "ckpt_0000.json"
    destination.write_text(
        checkpoint.model_dump_json(
            indent=2,
            context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
        )
        + "\n",
        encoding="utf-8",
    )
    print(destination.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
