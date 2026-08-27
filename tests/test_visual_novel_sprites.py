from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.engine.image_generation import (
    ImageGenerationConfig,
    ImageGenerationCoordinator,
)
from app.engine.one_star_adapter import prepare_one_star_transaction
from app.engine.one_star_visuals import (
    characters_needing_generated_sprite_prewarm,
    generated_sprite_pack_id,
)
from app.engine.player_media import ResolvedPlayerMedia
from app.engine.visual_novel_sprite_processing import (
    materialize_visual_novel_sprite,
)
from app.engine.visual_novel_sprites import (
    resolve_visual_novel_sprite_placements,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.image_generation import FrozenReferenceInput, ImageWorkerResult
from app.schemas.narrator import VisualNovelPage, VisualNovelSpriteCue
from app.schemas.one_star import OneStarTransaction


STORY_CHECKPOINT = Path(
    "app/storage/stories/one_star_ascension_s1/ckpt_0000.json"
)


def _sprite_source_bytes(width: int = 1024, height: int = 1536) -> bytes:
    image = Image.new("RGB", (width, height), (250, 4, 248))
    draw = ImageDraw.Draw(image)
    draw.ellipse((365, 110, 659, 404), fill=(198, 132, 101))
    draw.polygon(
        ((312, 385), (712, 385), (824, 1325), (200, 1325)),
        fill=(170, 34, 49),
    )
    draw.rectangle((260, 1310, 440, 1500), fill=(35, 41, 55))
    draw.rectangle((584, 1310, 764, 1500), fill=(35, 41, 55))
    # A closed prop/body loop leaves physical screen that is not connected to
    # any border. The runtime matte must remove this island as well.
    draw.rectangle((735, 535, 930, 800), fill=(70, 78, 91))
    draw.rectangle((775, 575, 890, 760), fill=(250, 4, 248))
    encoded = BytesIO()
    image.save(encoded, format="WEBP", lossless=True)
    return encoded.getvalue()


def test_one_star_prepared_mutation_preserves_private_visual_bindings() -> None:
    checkpoint = CheckpointFile.model_validate_json(STORY_CHECKPOINT.read_text())
    renna = next(
        character
        for character in checkpoint.characters
        if character.character_id == "renna_holt"
    )
    identity_reference_id = renna.visuals.identity_reference_id
    sprite_set_id = renna.visuals.sprite_set_id
    reference_count = len(checkpoint.reviewed_visual_references)
    sprite_set_count = len(checkpoint.reviewed_visual_novel_sprite_sets)

    prepared = prepare_one_star_transaction(
        checkpoint,
        event_id="visual-binding-round-trip",
        transaction=OneStarTransaction(present=False, operations=[]),
    )
    prepared_renna = next(
        character
        for character in prepared.after_checkpoint.characters
        if character.character_id == "renna_holt"
    )

    assert prepared_renna.visuals.identity_reference_id == identity_reference_id
    assert prepared_renna.visuals.sprite_set_id == sprite_set_id
    assert len(prepared.after_checkpoint.reviewed_visual_references) == reference_count
    assert (
        len(prepared.after_checkpoint.reviewed_visual_novel_sprite_sets)
        == sprite_set_count
    )


def test_seeded_hero_without_binding_never_enters_generated_prewarm() -> None:
    checkpoint = CheckpointFile.model_validate_json(STORY_CHECKPOINT.read_text())
    checkpoint.session.config.settings.presentation_mode = "visual_novel"
    renna = next(
        character
        for character in checkpoint.characters
        if character.character_id == "renna_holt"
    )
    renna.status = "active"
    renna.visuals.sprite_set_id = ""
    renna.mechanics["one_star_hero"]["current_stars"] = 3
    assert renna.mechanics["one_star_hero"]["generated_for_summon"] is False

    assert renna not in characters_needing_generated_sprite_prewarm(checkpoint)


class _SpriteWorker:
    available = True

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_id="fake/compose",
            model_revision="sprite-test",
            style_trigger="",
        )
        self.requests = []

    async def generate(self, request, *, output_path):
        self.requests.append(request)
        data = _sprite_source_bytes(request.width, request.height)
        Path(output_path).write_bytes(data)
        return ImageWorkerResult(
            ok=True,
            sha256=hashlib.sha256(data).hexdigest(),
            mime_type="image/webp",
            width=request.width,
            height=request.height,
            byte_count=len(data),
            generation_seconds=0.01,
        )

    async def abort_current(self) -> None:
        return None

    async def close(self) -> None:
        return None


def test_connected_magenta_matte_preserves_red_and_skin(tmp_path: Path) -> None:
    data = _sprite_source_bytes()
    media = ResolvedPlayerMedia(
        filename="candidate.webp",
        mime_type="image/webp",
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        width=1024,
        height=1536,
    )

    frozen = materialize_visual_novel_sprite(
        media,
        runtime_root=tmp_path,
    )

    output = tmp_path / frozen.relative_path
    with Image.open(output) as image:
        rgba = image.convert("RGBA")
        assert rgba.size == (1100, 1500)
        assert rgba.getpixel((0, 0))[3] == 0
        opaque = [
            pixel for pixel in rgba.getdata() if pixel[3] >= 250
        ]
        assert any(red > 140 and green < 70 for red, green, _blue, _a in opaque)
        assert any(
            red > 170 and 80 < green < 170 and blue < 140
            for red, green, blue, _a in opaque
        )
        pixels = np.asarray(rgba)
        visible_hot_key = (
            (pixels[:, :, 3] >= 32)
            & (pixels[:, :, 0] >= 180)
            & (pixels[:, :, 2] >= 140)
            & (pixels[:, :, 1] <= 90)
        )
        assert not visible_hot_key.any()


class _ReviewedStore:
    def __init__(self, frozen: FrozenReferenceInput) -> None:
        self.frozen = frozen

    def reviewed_reference(self, *, session_id: str, reference_id: str):
        del session_id
        return self.frozen if "_neutral_" in reference_id else None


class _ReviewedGeneration:
    def __init__(self, runtime_root: Path, frozen: FrozenReferenceInput) -> None:
        self.config = SimpleNamespace(runtime_root=runtime_root)
        self.store = _ReviewedStore(frozen)

    def resolve_visual_novel_sprite_variant(self, **_kwargs):
        return None


def test_master_uses_veil_until_seeded_reveal_and_missing_mood_uses_neutral(
    tmp_path: Path,
) -> None:
    checkpoint = CheckpointFile.model_validate_json(
        STORY_CHECKPOINT.read_text()
    )
    checkpoint.session.session_id = "sprite-resolver-test"
    sprite_path = tmp_path / "artifacts" / "reviewed.png"
    sprite_path.parent.mkdir(parents=True)
    image = Image.new("RGBA", (1100, 1500), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(
        (350, 80, 750, 1479),
        fill=(50, 65, 80, 255),
    )
    image.save(sprite_path)
    data = sprite_path.read_bytes()
    frozen = FrozenReferenceInput(
        reference_id="test-reviewed-sprite",
        sha256=hashlib.sha256(data).hexdigest(),
        mime_type="image/png",
        width=1100,
        height=1500,
        byte_count=len(data),
        relative_path="artifacts/reviewed.png",
        allowed_root="artifacts",
    )
    generation = _ReviewedGeneration(tmp_path, frozen)
    page = VisualNovelPage(
        kind="dialogue",
        speaker="Renna Holt",
        text="I am listening.",
        sprites=[VisualNovelSpriteCue(
            character="Renna Holt",
            expression="concerned",
        )],
    )

    veiled = resolve_visual_novel_sprite_placements(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        page=page,
        generation=generation,
    )
    assert len(veiled) == 1
    assert veiled[0].identity_handle == "osa_vnset_veiled_feminine_v1"
    assert "_neutral_" in veiled[0].variant_handle

    renna = next(
        character
        for character in checkpoint.characters
        if character.character_id == "renna_holt"
    )
    renna.mechanics["one_star_hero"]["current_stars"] = 2
    revealed = resolve_visual_novel_sprite_placements(
        checkpoint=checkpoint,
        viewer_character_id="the_master",
        page=page,
        generation=generation,
    )
    assert len(revealed) == 1
    assert revealed[0].identity_handle == "osa_vnset_renna_holt_v1"


@pytest.mark.asyncio
async def test_generated_pack_builds_neutral_then_parallelizable_sweep(
    tmp_path: Path,
) -> None:
    checkpoint = CheckpointFile.model_validate_json(
        STORY_CHECKPOINT.read_text()
    )
    checkpoint.session.session_id = "sprite-prewarm-test"
    checkpoint.session.turn_index = 4
    checkpoint.session.config.settings.presentation_mode = "visual_novel"
    renna = next(
        character
        for character in checkpoint.characters
        if character.character_id == "renna_holt"
    )
    renna.status = "active"
    renna.visuals.sprite_set_id = ""
    renna.mechanics["one_star_hero"]["generated_for_summon"] = True
    renna.mechanics["one_star_hero"]["current_stars"] = 2
    worker = _SpriteWorker()
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=ImageGenerationConfig(
            runtime_root=tmp_path / "runtime",
            queue_limit=16,
            per_session_queue_limit=16,
            steps=4,
            guidance=1.0,
        ),
        worker=worker,
    )
    await coordinator.start()
    try:
        admitted = await coordinator.ensure_visual_novel_sprite_prewarm(
            checkpoint
        )
        assert len(admitted) == 1
        await coordinator.wait_for_terminal(admitted[0], timeout=20)

        deadline = asyncio.get_running_loop().time() + 30
        while True:
            jobs = [
                job
                for job in coordinator.store.all_jobs()
                if job.request.sprite_pack_id
            ]
            if len(jobs) == 8 and all(job.status.value in {
                "succeeded", "failed", "cancelled"
            } for job in jobs):
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("generated sprite sweep did not finish")
            await asyncio.sleep(0.05)

        assert len(jobs) == 8
        assert all(job.status.value == "succeeded" for job in jobs)
        neutral_request = next(
            job.request for job in jobs if job.request.sprite_expression == "neutral"
        )
        assert neutral_request.reference_inputs == []
        assert "pack's neutral baseline" in neutral_request.prompt
        assert "distinct from the neutral baseline" not in neutral_request.prompt
        assert all(
            len(job.request.reference_inputs) == 1
            for job in jobs
            if job.request.sprite_expression != "neutral"
        )
        assert all(
            "distinct from the neutral baseline" in job.request.prompt
            for job in jobs
            if job.request.sprite_expression != "neutral"
        )
        assert all(
            request.sprite_pack_id and "/home/" not in request.prompt
            for request in worker.requests
        )

        pack_id = generated_sprite_pack_id(checkpoint, renna)
        resolved = coordinator.resolve_visual_novel_sprite_variant(
            session_id=checkpoint.session.session_id,
            character_id=renna.character_id,
            sprite_pack_id=pack_id,
            expression="happy",
        )
        assert resolved is not None
        _handle, media, facing = resolved
        assert facing == "right"
        assert (media.width, media.height, media.mime_type) == (
            1100,
            1500,
            "image/png",
        )
    finally:
        await coordinator.close()
