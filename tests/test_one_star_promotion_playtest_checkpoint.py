"""Contracts for the focused One-Star promotion/sprite playtest seed."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from PIL import Image, ImageDraw
import pytest

from app.engine.prompt_manager import PromptManager
from app.engine.one_star_adapter import (
    OneStarTransactionError,
    load_one_star_account,
    load_one_star_hero,
    prepare_one_star_transaction,
)
from app.engine.one_star_visuals import (
    characters_needing_generated_sprite_prewarm,
    generated_sprite_pack_id,
    sprite_set_id_for_viewer,
)
from app.engine.player_media import ResolvedPlayerMedia
from app.engine.reviewed_visual_references import (
    validate_story_visual_references,
)
from app.engine.visual_novel_sprites import (
    resolve_visual_novel_sprite_placements,
)
from app.engine.turn_loop_dispatcher import LLMDispatcher
from app.llm.client import LLMClient
from app.schemas.checkpoint import CheckpointFile
from app.schemas.events import ObservableFact
from app.schemas.narrator import VisualNovelPage
from app.schemas.one_star import (
    OneStarEventRouterOutput,
    OneStarStateUpdateList,
    OneStarTransaction,
)
from scripts.build_one_star_promotion_playtest_seed import build_checkpoint
from scripts.run_one_star_promotion_sprite_playtest import (
    _promotion_comparison_pages,
)
from scripts.run_one_star_promotion_vn_live_playtest import (
    _deck_has_committed_identity_reveal,
    _deck_uses_only_stage,
)
from tests.support.factories import llm_response, router_output


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_STORY_DIR = REPO_ROOT / "app/storage/stories/one_star_ascension_s1"
PLAYTEST_STORY_DIR = (
    REPO_ROOT / "app/storage/stories/one_star_ascension_s1_promotion_playtest"
)
FACELESS_ID = "promotion_playtest_faceless"


def _load(story_dir: Path = PLAYTEST_STORY_DIR) -> CheckpointFile:
    return CheckpointFile.model_validate_json(
        (story_dir / "ckpt_0000.json").read_text(encoding="utf-8")
    )


def _character(checkpoint: CheckpointFile, character_id: str):
    return next(
        character
        for character in checkpoint.characters
        if character.character_id == character_id
    )


def _transaction(*operations: dict[str, object]) -> OneStarTransaction:
    return OneStarTransaction.model_validate(
        {
            "present": True,
            "operations": list(operations),
        }
    )


def _promote(
    checkpoint: CheckpointFile,
    *,
    character_id: str,
    operation_id: str,
) -> CheckpointFile:
    opened = prepare_one_star_transaction(
        checkpoint,
        event_id=f"{operation_id}_open",
        transaction=_transaction(
            {
                "operation": "pending_open",
                "pending": {
                    "operation_id": operation_id,
                    "kind": "promotion",
                    "participant_ids": [character_id],
                    "target_id": character_id,
                    "destination": "niflheim_promotion_chamber",
                    "opened_at_s": 0,
                },
            }
        ),
        canonical_at_s=0,
        initiating_actor_id="the_master",
    )
    resolved = prepare_one_star_transaction(
        opened.after_checkpoint,
        event_id=f"{operation_id}_resolve",
        transaction=_transaction(
            {
                "operation": "pending_resolve",
                "operation_id": operation_id,
            }
        ),
        location_updates={character_id: "niflheim_promotion_chamber"},
        canonical_at_s=0,
        initiating_actor_id="the_master",
    )
    return resolved.after_checkpoint


def _open_synthesis(checkpoint: CheckpointFile) -> CheckpointFile:
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="mara_castor_synthesis_open",
        transaction=_transaction(
            {
                "operation": "pending_open",
                "pending": {
                    "operation_id": "mara_castor_synthesis",
                    "kind": "synthesis",
                    "participant_ids": ["castor_valebrand"],
                    "target_id": FACELESS_ID,
                    "destination": "niflheim_synthesis_chamber",
                    "opened_at_s": 0,
                },
            }
        ),
        canonical_at_s=0,
        initiating_actor_id="the_master",
    )
    return prepared.after_checkpoint


def _resolve_synthesis(checkpoint: CheckpointFile) -> CheckpointFile:
    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="mara_castor_synthesis_resolve",
        transaction=_transaction(
            {
                "operation": "pending_resolve",
                "operation_id": "mara_castor_synthesis",
            }
        ),
        location_updates={
            FACELESS_ID: "niflheim_synthesis_chamber",
            "castor_valebrand": "niflheim_synthesis_chamber",
        },
        canonical_at_s=0,
        initiating_actor_id="the_master",
    )
    return prepared.after_checkpoint


def _generated_media() -> ResolvedPlayerMedia:
    image = Image.new("RGBA", (1100, 1500), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(
        (330, 80, 770, 1479),
        fill=(73, 94, 128, 255),
    )
    output = BytesIO()
    image.save(output, format="PNG")
    data = output.getvalue()
    return ResolvedPlayerMedia(
        filename="generated-mara-neutral.png",
        mime_type="image/png",
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        width=1100,
        height=1500,
    )


def test_live_playtest_reveal_check_requires_old_flash_new_in_fixed_chamber() -> None:
    stage_sha256 = "a" * 64
    old_identity = "sprite.veiled"
    new_identity = "sprite.revealed"
    deck = {
        "render": {
            "segments": [{
                "pages": [
                    {
                        "kind": "narration",
                        "speaker": "",
                        "text": "Mara enters the chamber.",
                            "sprites": ["Mara Venn"],
                    },
                    {
                        "kind": "narration",
                        "speaker": "",
                        "text": "Promotion complete.",
                        "sprites": [],
                    },
                ]
            }]
        },
        "manifest": {
            "identity": {
                "sections": [
                    {
                        "card_style": "adv",
                        "stage_sha256": stage_sha256,
                        "sprites": [{"identity_handle": old_identity}],
                    },
                    {
                        "card_style": "adv",
                        "stage_sha256": stage_sha256,
                        "sprites": [],
                    },
                    {
                        "card_style": "identity_flash",
                        "stage_sha256": stage_sha256,
                        "sprites": [{"identity_handle": old_identity}],
                    },
                    {
                        "card_style": "identity_reveal",
                        "stage_sha256": stage_sha256,
                        "sprites": [{"identity_handle": new_identity}],
                    },
                ]
            }
        },
    }

    assert _deck_has_committed_identity_reveal(
        deck,
        character_name="Mara Venn",
        before_identity_handle=old_identity,
        after_identity_handle=new_identity,
        expected_stage_sha256=stage_sha256,
    )
    assert _deck_uses_only_stage(deck, stage_sha256=stage_sha256)

    event_aligned_entry = deepcopy(deck)
    event_aligned_entry["manifest"]["identity"]["sections"][0][
        "stage_sha256"
    ] = "b" * 64
    assert _deck_has_committed_identity_reveal(
        event_aligned_entry,
        character_name="Mara Venn",
        before_identity_handle=old_identity,
        after_identity_handle=new_identity,
        expected_stage_sha256=stage_sha256,
    )
    assert not _deck_uses_only_stage(
        event_aligned_entry,
        stage_sha256=stage_sha256,
    )

    early_reveal = deepcopy(deck)
    early_reveal["manifest"]["identity"]["sections"][0]["sprites"] = [
        {"identity_handle": new_identity}
    ]
    assert not _deck_has_committed_identity_reveal(
        early_reveal,
        character_name="Mara Venn",
        before_identity_handle=old_identity,
        after_identity_handle=new_identity,
        expected_stage_sha256=stage_sha256,
    )

    drifting_stage = deepcopy(deck)
    drifting_stage["manifest"]["identity"]["sections"][2][
        "stage_sha256"
    ] = "b" * 64
    assert not _deck_has_committed_identity_reveal(
        drifting_stage,
        character_name="Mara Venn",
        before_identity_handle=old_identity,
        after_identity_handle=new_identity,
        expected_stage_sha256=stage_sha256,
    )
    assert not _deck_uses_only_stage(
        drifting_stage,
        stage_sha256=stage_sha256,
    )


@pytest.mark.asyncio
async def test_fixed_promotion_resolution_cannot_be_erased_by_state_repair():
    checkpoint = _load()
    opened = prepare_one_star_transaction(
        checkpoint,
        event_id="promotion_open",
        transaction=_transaction(
            {
                "operation": "pending_open",
                "pending": {
                    "operation_id": "promotion_renna",
                    "kind": "promotion",
                    "participant_ids": ["renna_holt"],
                    "target_id": "renna_holt",
                    "destination": "niflheim_promotion_chamber",
                    "opened_at_s": 0,
                },
            }
        ),
        canonical_at_s=0,
        initiating_actor_id="the_master",
    ).after_checkpoint
    data = router_output(
        event_id="promotion_resolution",
        observer_ids=["the_master", "renna_holt", "iselle_the_guide"],
        event_kind="cat_ii_resolution",
        facts=[
            ObservableFact.all("Renna enters and the promotion completes."),
            ObservableFact.only(
                "The System reports the completed promotion.",
                ["iselle_the_guide"],
            ),
        ],
        location_updates=[
            {
                "character_id": "renna_holt",
                "location_label": "niflheim_lobby",
            }
        ],
    ).model_dump(mode="json")
    data["state_updates"] = [
        {
            "kind": "pending_resolve",
            "target_id": "promotion_renna",
            "value": "",
            "details": [],
        }
    ]
    event = OneStarEventRouterOutput.model_validate(data)
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(
        return_value=llm_response(
            OneStarStateUpdateList(state_updates=[]),
        )
    )
    dispatcher = LLMDispatcher(client, PromptManager("app/prompts"))

    with pytest.raises(
        OneStarTransactionError,
        match="repair cannot erase a pending resolution",
    ):
        await dispatcher.prepare_ruleset_event(
            ckpt=opened,
            result=event,
            actor_id="the_master",
        )

    assert client.complete.await_count == 1
    assert load_one_star_hero(_character(opened, "renna_holt")).current_stars == 1


class _GeneratedSpriteResolver:
    def __init__(self, pack_id: str) -> None:
        self.pack_id = pack_id

    def resolve_visual_novel_sprite_variant(
        self,
        *,
        sprite_pack_id: str,
        variant_key: str,
        **_kwargs: object,
    ):
        assert sprite_pack_id == self.pack_id
        assert variant_key == "neutral"
        return "imgsprite_mara_neutral", _generated_media(), "right"


def test_playtest_seed_is_a_valid_visual_novel_story_copy() -> None:
    checkpoint = _load()
    source = _load(SOURCE_STORY_DIR)

    assert checkpoint.session.story_id == ("one_star_ascension_s1_promotion_playtest")
    assert checkpoint.session.session_id == checkpoint.session.story_id
    assert checkpoint.session.config.settings.presentation_mode == "visual_novel"
    assert checkpoint.world_state.opening is not None
    assert checkpoint.world_state.opening.allow_spawns is False
    assert checkpoint.world_state.global_flags == {}
    assert "sealed promotion" not in (checkpoint.session.config.narrative_rules.lower())
    assert source.session.config.settings.presentation_mode == "visual_novel"
    assert source.visual_novel_onboarding is not None

    _source_owner, source_account = load_one_star_account(source)
    _owner, account = load_one_star_account(checkpoint)
    assert source_account.config.progression.cap_bank_extra_levels == 1
    assert account.config.progression.cap_bank_extra_levels == 10
    assert account.state.facilities["promotion_chamber"] == 1
    assert account.state.facilities["synthesis_chamber"] == 1
    assert account.state.resources.materials == {
        "lesser_promotion_stone": 3,
    }
    promotion_reference = next(
        reference
        for reference in checkpoint.reviewed_visual_references
        if reference.reference_id == "osa_loc_1f_promotion_v1"
    )
    assert checkpoint.location_visual_reference_ids[
        "niflheim_promotion_chamber"
    ] == [promotion_reference.reference_id]
    assert promotion_reference.fixed_stage is True
    assert promotion_reference.sha256 == (
        "c73621400a8d9c960a38391816c3fe16f57d5e04fb00e4bc2968d7fdeb07512a"
    )

    expected = {
        "renna_holt": (1, 10, 4_500, False),
        FACELESS_ID: (1, 10, 4_500, True),
        "castor_valebrand": (4, 45, 99_000, False),
    }
    for character_id, state in expected.items():
        character = _character(checkpoint, character_id)
        hero = load_one_star_hero(character)
        assert hero is not None
        assert character.status.value == "active"
        assert character.location == "niflheim_lobby"
        assert (
            hero.current_stars,
            hero.level,
            hero.experience_points,
            hero.generated_for_summon,
        ) == state
        assert hero.owner_lobby_id == "niflheim"

    validate_story_visual_references(
        checkpoint,
        story_dir=PLAYTEST_STORY_DIR,
    )


def test_playtest_builder_does_not_restore_one_star_global_flags() -> None:
    assert build_checkpoint().world_state.global_flags == {}


def test_playtest_seed_inherits_current_one_star_motivation_policy() -> None:
    checkpoint = _load()
    source = _load(SOURCE_STORY_DIR)

    def rule_section(seed: CheckpointFile, heading: str) -> str:
        return next(
            paragraph
            for paragraph in seed.session.config.narrative_rules.split("\n\n")
            if paragraph.startswith(f"{heading}:")
        )

    source_tier_one = next(
        tier for tier in source.world_state.knowledge_tiers if tier.tier == 1
    )
    playtest_tier_one = next(
        tier for tier in checkpoint.world_state.knowledge_tiers if tier.tier == 1
    )

    # The focused later-state fixture may change economy and live character
    # state, but it must not fork the story's durable autonomy/generation policy.
    assert rule_section(checkpoint, "Autonomy") == rule_section(source, "Autonomy")
    assert (
        playtest_tier_one.generation_guidance
        == source_tier_one.generation_guidance
    )

    source_renna = _character(source, "renna_holt")
    playtest_renna = _character(checkpoint, "renna_holt")
    source_hero = load_one_star_hero(source_renna)
    playtest_hero = load_one_star_hero(playtest_renna)
    assert source_hero is not None
    assert playtest_hero is not None

    # The focused fixture keeps each Hero's durable actor record and may append
    # scenario-specific facts that are actually learned or chosen in play.
    for character_id in ("renna_holt", "mirelle_voss"):
        source_character = _character(source, character_id)
        playtest_character = _character(checkpoint, character_id)
        assert (
            playtest_character.public_sheet.role
            == source_character.public_sheet.role
        )
        assert source_character.actor is not None
        assert playtest_character.actor is not None
        source_facts = source_character.actor.facts
        playtest_facts = playtest_character.actor.facts
        assert playtest_facts[: len(source_facts)] == source_facts
    assert source_renna.actor is not None
    assert playtest_renna.actor is not None
    renna_grants = playtest_renna.actor.facts[len(source_renna.actor.facts) :]
    assert renna_grants
    assert all(fact.origin.value == "told" for fact in renna_grants)
    assert playtest_hero.hidden_capabilities == source_hero.hidden_capabilities
    assert set(playtest_hero.hidden_capabilities) == {"pattern_retention"}


def test_one_star_seeds_have_no_forced_resistance_quota() -> None:
    obsolete_resistance_cues = (
        "fear may surface as refusal, freezing",
        "turn every fear into dutiful compliance",
        "at least one person's strategy should create material friction",
        "eighty-year-old baker",
        "helpful adventurer waiting for an assignment",
        "if pressed toward danger without answers, she refuses",
        "no explanation has yet been given",
    )

    for checkpoint in (_load(SOURCE_STORY_DIR), _load()):
        tier_one = next(
            tier for tier in checkpoint.world_state.knowledge_tiers if tier.tier == 1
        )
        renna = _character(checkpoint, "renna_holt")
        model_visible_bias_surfaces = "\n".join((
            checkpoint.session.config.narrative_rules,
            tier_one.generation_guidance.actor_fact_guidance,
            renna.public_sheet.public_context,
            *(fact.text for fact in (renna.actor.facts if renna.actor else [])),
        )).lower()
        for obsolete_cue in obsolete_resistance_cues:
            assert obsolete_cue not in model_visible_bias_surfaces


def test_fixture_reaches_authored_and_generated_reveal_contracts() -> None:
    checkpoint = _load()
    master_id = "the_master"
    renna = _character(checkpoint, "renna_holt")
    mara = _character(checkpoint, FACELESS_ID)

    renna_before_sprite_set = sprite_set_id_for_viewer(
        checkpoint,
        viewer_character_id=master_id,
        character=renna,
    )
    mara_before_sprite_set = sprite_set_id_for_viewer(
        checkpoint,
        viewer_character_id=master_id,
        character=mara,
    )
    assert renna_before_sprite_set == "osa_vnset_veiled_feminine_v1"
    assert mara_before_sprite_set == "osa_vnset_veiled_feminine_v1"

    checkpoint = _promote(
        checkpoint,
        character_id="renna_holt",
        operation_id="renna_two_star",
    )
    renna = _character(checkpoint, "renna_holt")
    renna_after_sprite_set = sprite_set_id_for_viewer(
        checkpoint,
        viewer_character_id=master_id,
        character=renna,
    )
    assert renna_after_sprite_set == "osa_vnset_renna_holt_v1"

    checkpoint = _promote(
        checkpoint,
        character_id=FACELESS_ID,
        operation_id="mara_two_star",
    )
    mara = _character(checkpoint, FACELESS_ID)
    mara_hero = load_one_star_hero(mara)
    assert mara_hero is not None
    assert (mara_hero.current_stars, mara_hero.level) == (2, 10)
    assert (
        sprite_set_id_for_viewer(
            checkpoint,
            viewer_character_id=master_id,
            character=mara,
        )
        == "osa_vnset_veiled_feminine_v1"
    )
    assert [
        character.character_id
        for character in characters_needing_generated_sprite_prewarm(checkpoint)
    ] == [FACELESS_ID]

    checkpoint = _open_synthesis(checkpoint)
    _owner, account = load_one_star_account(checkpoint)
    preview = account.state.pending_operation
    assert preview is not None
    assert preview.synthesis_preview is not None
    assert (
        preview.synthesis_preview.offered_xp,
        preview.synthesis_preview.applied_xp,
        preview.synthesis_preview.wasted_xp,
    ) == (49_695, 39_000, 10_695)

    checkpoint = _resolve_synthesis(checkpoint)
    mara = _character(checkpoint, FACELESS_ID)
    mara_hero = load_one_star_hero(mara)
    assert mara_hero is not None
    assert (
        mara_hero.current_stars,
        mara_hero.level,
        mara_hero.experience_points,
    ) == (2, 20, 43_500)
    assert (
        _character(
            checkpoint,
            "castor_valebrand",
        ).status.value
        == "culled"
    )

    checkpoint = _promote(
        checkpoint,
        character_id=FACELESS_ID,
        operation_id="mara_three_star",
    )
    mara = _character(checkpoint, FACELESS_ID)
    mara_hero = load_one_star_hero(mara)
    assert mara_hero is not None
    assert (
        mara_hero.current_stars,
        mara_hero.level,
        mara_hero.experience_points,
    ) == (3, 30, 43_500)

    pack_id = generated_sprite_pack_id(checkpoint, mara)
    mara_after_sprite_set = sprite_set_id_for_viewer(
        checkpoint,
        viewer_character_id=master_id,
        character=mara,
    )
    assert mara_after_sprite_set == pack_id
    assert (
        renna_before_sprite_set,
        renna_after_sprite_set,
        mara_before_sprite_set,
        mara_after_sprite_set,
    ) == (
        "osa_vnset_veiled_feminine_v1",
        "osa_vnset_renna_holt_v1",
        "osa_vnset_veiled_feminine_v1",
        pack_id,
    )
    placements = resolve_visual_novel_sprite_placements(
        checkpoint=checkpoint,
        viewer_character_id=master_id,
        page=VisualNovelPage(
            kind="dialogue",
            speaker="Mara Venn",
            text="I remember enough to know this face is mine.",
            sprites=["Mara Venn"],
        ),
        generation=_GeneratedSpriteResolver(pack_id),  # type: ignore[arg-type]
    )
    assert len(placements) == 1
    assert placements[0].identity_handle == pack_id
    assert placements[0].variant_handle == "imgsprite_mara_neutral"

    _owner, account = load_one_star_account(checkpoint)
    assert account.state.resources.materials == {
        "lesser_promotion_stone": 0,
    }


def test_promotion_comparison_pages_name_and_order_both_transitions() -> None:
    pages = _promotion_comparison_pages()

    assert tuple(page.speaker for page in pages) == (
        "Renna Holt",
        "Renna Holt",
        "Mara Venn",
        "Mara Venn",
    )
    assert tuple(page.text.split(" · ", 1)[0] for page in pages) == (
        "Before ascent",
        "After ascent",
        "Before ascent",
        "After ascent",
    )
    assert tuple(page.sprites[0] for page in pages) == (
        "Renna Holt",
        "Renna Holt",
        "Mara Venn",
        "Mara Venn",
    )
