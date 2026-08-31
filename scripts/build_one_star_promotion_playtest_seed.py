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
    ActorFact,
    ActorFactOrigin,
    ActorRecord,
    CharacterAgentTier,
    CharacterRecord,
    CharacterStatus,
    CharacterVisuals,
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


def _append_actor_fact(
    character: CharacterRecord,
    *,
    origin: ActorFactOrigin,
    text: str,
) -> None:
    """Add one distinct playtest fact without replacing actor continuity."""
    fact_text = text.strip()
    if not fact_text:
        return
    if character.actor is None:
        character.actor = ActorRecord()
    if fact_text.casefold() not in {
        fact.text.casefold() for fact in character.actor.facts
    }:
        character.actor.facts.append(ActorFact(origin=origin, text=fact_text))


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
    if character.actor is None:
        character.actor = ActorRecord(may_act_offstage=True)
    else:
        character.actor.may_act_offstage = True


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
            public_context=(
                "A practical one-star porter who watches exits and keeps her "
                "old cargo hook close."
            ),
        ),
        visuals=CharacterVisuals(
            default_loadout=(
                "Rough gray cloth shirt and trousers tied with plain cords, "
                "a short worn wooden cargo hook tucked through a belt strap, "
                "and scuffed cloth shoes."
            ),
        ),
        actor=ActorRecord(
            may_act_offstage=True,
            facts=[
                ActorFact(
                    origin=ActorFactOrigin.lived,
                    text=(
                        "Cold summon-light left you without coherent history, "
                        "but your hands still know how to balance a load and "
                        "test a working knot."
                    ),
                ),
                ActorFact(
                    origin=ActorFactOrigin.lived,
                    text=(
                        "You keep near an exit and become more deliberate, not "
                        "more obedient, when frightened."
                    ),
                ),
                ActorFact(
                    origin=ActorFactOrigin.told,
                    text=(
                        "Iselle briefed you that promotion costs one lesser "
                        "stone, advances one star, applies retained experience, "
                        "and may return an authored memory tier."
                    ),
                ),
                ActorFact(
                    origin=ActorFactOrigin.lived,
                    text=(
                        "You and Castor reviewed the exact synthesis result: it "
                        "permanently kills its source, he volunteered without "
                        "coercion, and you chose to receive the transfer before "
                        "your second promotion."
                    ),
                ),
                ActorFact(
                    origin=ActorFactOrigin.told,
                    text=(
                        "Your supplied identity remains System-veiled until "
                        "your third star."
                    ),
                ),
            ],
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
    _append_actor_fact(
        owner,
        origin=ActorFactOrigin.told,
        text=(
            "This focused playtest is arranged to show Renna's authored reveal, "
            "Mara's voluntary synthesis with Castor, and Mara's generated "
            "three-star reveal."
        ),
    )

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
    _append_actor_fact(
        renna,
        origin=ActorFactOrigin.told,
        text=(
            "Iselle briefed you that you have reached the one-star level limit "
            "and that promotion costs one lesser stone, advances one star, "
            "applies retained experience, and may return an authored memory "
            "tier. You decided to enter after its sanctioned cost and effect "
            "are confirmed."
        ),
    )

    _set_seeded_hero_level(
        checkpoint,
        character_id="castor_valebrand",
        current_stars=4,
        level=45,
    )
    castor = _character(checkpoint, "castor_valebrand")
    _append_actor_fact(
        castor,
        origin=ActorFactOrigin.lived,
        text=(
            "You and Mara reviewed the exact, final synthesis result. You "
            "freely volunteered to become her source without coercion and chose "
            "to make that choice legible even though death frightens you."
        ),
    )

    if any(
        character.character_id == FACELESS_CHARACTER_ID
        for character in checkpoint.characters
    ):
        raise RuntimeError("playtest faceless character already exists")
    checkpoint.characters.append(_faceless_character(checkpoint))

    iselle = _character(checkpoint, "iselle_the_guide")
    _append_actor_fact(
        iselle,
        origin=ActorFactOrigin.told,
        text=(
            "For this focused playtest, Renna and Mara's available promotions, "
            "Castor's already-reviewed voluntary synthesis response, and Mara's "
            "return for a second promotion are the scheduled lobby sequence."
        ),
    )

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
