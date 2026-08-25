from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from experiments.staged_image.pipeline import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_STYLE_PATH,
    LEGACY_CINEMATIC_CORPUS_PATH,
    CaseState,
    CanvasSpec,
    CameraSpec,
    CorpusSpec,
    ReferenceResolver,
    ResolvedReference,
    SceneSpec,
    SpriteVariantSpec,
    StagedImageRunner,
    StylePack,
    SubjectSpec,
    UnitBox,
    UnitPoint,
    VisualNovelFrameSpec,
    blend_masked_result,
    build_report,
    composite_subjects,
    derived_seed,
    load_experiment,
    parse_seeds,
    prepare_component_reference_inputs,
    record_review,
    validate_matte,
)


def _write_png(path: Path, color: tuple[int, ...], mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (64, 96), color).save(path)


def _artifact_payload(path: Path, runtime_root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "relative_path": path.relative_to(runtime_root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mime_type": "image/png",
        "width": 64,
        "height": 96,
        "byte_count": len(data),
    }


def _reference_database(runtime_root: Path) -> sqlite3.Connection:
    database_path = runtime_root / "jobs.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE image_reviewed_references "
        "(session_id TEXT, reference_id TEXT, frozen_json TEXT)"
    )
    connection.execute(
        "CREATE TABLE image_identity_candidates "
        "(session_id TEXT, candidate_id TEXT, artifact_json TEXT)"
    )
    return connection


def _visual_novel_scene() -> SceneSpec:
    left = SubjectSpec(
        character_id="left",
        display_name="Left",
        reference_ids=["left-reference"],
        identity_prompt="Stable left identity and clothing.",
        pose_prompt="Neutral three-quarter conversational anchor.",
        screen_side="left",
        facing="right",
        variants=[
            SpriteVariantSpec(
                variant_id="concerned",
                expression_prompt="quiet concern",
                pose_adjustment_prompt="raise the inner hand slightly",
            )
        ],
    )
    right = SubjectSpec(
        character_id="right",
        display_name="Right",
        reference_ids=["right-reference"],
        identity_prompt="Stable right identity and clothing.",
        pose_prompt="Neutral three-quarter conversational anchor.",
        screen_side="right",
        facing="left",
        variants=[
            SpriteVariantSpec(
                variant_id="reassuring",
                expression_prompt="restrained reassurance",
                pose_adjustment_prompt="open the inner hand slightly",
            )
        ],
    )
    return SceneSpec(
        scene_id="visual_novel_scene",
        title="Visual novel scene",
        split="tuning",
        style_pack_id="visual-novel-cel",
        presentation="visual_novel",
        canvas=CanvasSpec(width=1024, height=576),
        camera=CameraSpec(description="Locked eye-level widescreen room view."),
        background_prompt="An empty quiet room with a clear central gap.",
        subjects=[left, right],
        frames=[
            VisualNovelFrameSpec(frame_id="neutral", title="Neutral anchors"),
            VisualNovelFrameSpec(
                frame_id="left-concerned",
                title="Left is concerned",
                active_character_id="left",
                subject_variants={"left": "concerned"},
            ),
            VisualNovelFrameSpec(
                frame_id="right-reassuring",
                title="Right reassures",
                active_character_id="right",
                subject_variants={
                    "left": "concerned",
                    "right": "reassuring",
                },
            ),
        ],
    )


def test_checked_in_default_is_small_visual_novel_corpus_and_legacy_is_preserved():
    corpus, styles = load_experiment(DEFAULT_CORPUS_PATH, DEFAULT_STYLE_PATH)

    assert isinstance(corpus, CorpusSpec)
    assert sum(scene.split == "tuning" for scene in corpus.scenes) == 3
    assert sum(scene.split == "holdout" for scene in corpus.scenes) == 1
    assert all(scene.presentation == "visual_novel" for scene in corpus.scenes)
    assert all(len(scene.subjects) == 2 for scene in corpus.scenes)
    assert all(len(scene.frames) >= 3 for scene in corpus.scenes)
    assert len(styles) == 4
    assert all(
        1 <= len(subject.reference_ids) <= 2
        for scene in corpus.scenes
        for subject in scene.subjects
    )
    lobby = next(
        scene
        for scene in corpus.scenes
        if scene.scene_id == "vn03_beginner_lobby_iselle_wren"
    )
    assert len(lobby.frames) == 7
    assert all(len(subject.variants) == 3 for subject in lobby.subjects)
    iselle = next(
        subject
        for subject in lobby.subjects
        if subject.character_id == "iselle_the_guide"
    )
    assert iselle.stage_height_fraction == 0.62
    assert iselle.grounded is False
    assert iselle.component_mode == "reference_cutout"
    assert iselle.facing_control == "mirror_reference"
    wren = next(
        subject
        for subject in lobby.subjects
        if subject.character_id == "wren_thelantern"
    )
    assert wren.facing_control == "preserve_reference"

    legacy, _ = load_experiment(
        LEGACY_CINEMATIC_CORPUS_PATH,
        DEFAULT_STYLE_PATH,
    )
    assert sum(scene.split == "tuning" for scene in legacy.scenes) == 6
    assert sum(scene.split == "holdout" for scene in legacy.scenes) == 2
    arrival_iselle = next(
        subject
        for scene in legacy.scenes
        if scene.scene_id == "t01_failed_arrival_hall"
        for subject in scene.subjects
        if subject.character_id == "iselle_the_guide"
    )
    assert arrival_iselle.component_mode == "reference_cutout"


def test_holdouts_require_an_explicit_release_flag():
    runner = StagedImageRunner()

    assert len(runner.select_scenes([])) == 3
    with pytest.raises(ValueError, match="--include-holdout"):
        runner.select_scenes(["vh01_champion_anteroom"])
    assert (
        runner.select_scenes(
            ["vh01_champion_anteroom"],
            include_holdout=True,
        )[0].split
        == "holdout"
    )


def test_visual_novel_schema_requires_inward_facing_derived_stage_geometry():
    scene = _visual_novel_scene()

    assert {subject.screen_side for subject in scene.subjects} == {"left", "right"}
    assert all(subject.target_box is None for subject in scene.subjects)

    invalid = scene.model_dump(mode="json")
    invalid["subjects"][0]["facing"] = "left"
    with pytest.raises(ValidationError, match="must face inward"):
        SceneSpec.model_validate(invalid)

    duplicate_layout = scene.model_dump(mode="json")
    duplicate_layout["subjects"][0]["target_box"] = {
        "x": 0.0,
        "y": 0.0,
        "width": 0.5,
        "height": 1.0,
    }
    with pytest.raises(ValidationError, match="derived from screen_side"):
        SceneSpec.model_validate(duplicate_layout)


def test_cinematic_scene_rejects_visual_novel_stage_height():
    corpus, _ = load_experiment(
        LEGACY_CINEMATIC_CORPUS_PATH,
        DEFAULT_STYLE_PATH,
    )
    invalid = corpus.scenes[0].model_dump(mode="json")
    invalid["subjects"][0]["stage_height_fraction"] = 0.5

    with pytest.raises(ValidationError, match="cannot declare visual-novel stage"):
        SceneSpec.model_validate(invalid)

    invalid = corpus.scenes[0].model_dump(mode="json")
    invalid["subjects"][0]["facing_control"] = "preserve_reference"
    with pytest.raises(ValidationError, match="cannot declare visual-novel facing"):
        SceneSpec.model_validate(invalid)


def test_visual_novel_run_rejects_whole_scene_model_tracks_before_generation():
    runner = StagedImageRunner()

    with pytest.raises(ValueError, match="only deterministic pixel"):
        runner.run(
            scene_ids=["vn01_arrival_exchange"],
            seeds=(1, 2, 3, 4),
            tracks=("pixel", "masked"),
            dry_run=True,
        )


def test_reference_resolver_verifies_reviewed_and_candidate_bytes(tmp_path):
    runtime_root = tmp_path / "runtime"
    reviewed_path = runtime_root / "artifacts/references/reviewed.png"
    candidate_path = runtime_root / "artifacts/candidate.png"
    _write_png(reviewed_path, (30, 60, 90))
    _write_png(candidate_path, (90, 60, 30))
    connection = _reference_database(runtime_root)
    connection.execute(
        "INSERT INTO image_reviewed_references VALUES (?, ?, ?)",
        (
            "session",
            "reviewed",
            json.dumps(_artifact_payload(reviewed_path, runtime_root)),
        ),
    )
    connection.execute(
        "INSERT INTO image_identity_candidates VALUES (?, ?, ?)",
        (
            "session",
            "candidate",
            json.dumps(_artifact_payload(candidate_path, runtime_root)),
        ),
    )
    connection.commit()
    connection.close()

    resolver = ReferenceResolver(runtime_root)

    assert resolver.resolve("session", "reviewed").source == "reviewed"
    assert resolver.resolve("session", "candidate").source == "candidate"


def test_reference_resolver_rejects_paths_outside_artifact_root(tmp_path):
    runtime_root = tmp_path / "runtime"
    outside_path = runtime_root / "outside.png"
    _write_png(outside_path, (30, 60, 90))
    connection = _reference_database(runtime_root)
    connection.execute(
        "INSERT INTO image_reviewed_references VALUES (?, ?, ?)",
        (
            "session",
            "escape",
            json.dumps(_artifact_payload(outside_path, runtime_root)),
        ),
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="escapes artifact root"):
        ReferenceResolver(runtime_root).resolve("session", "escape")


def test_matte_validation_rejects_empty_and_accepts_one_clean_subject():
    subject = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(subject).rounded_rectangle((70, 30, 185, 235), radius=25, fill=255)
    policy = SubjectSpec(
        character_id="subject",
        display_name="Subject",
        reference_ids=["reference"],
        identity_prompt="Stable reference identity.",
        pose_prompt="A complete standing pose.",
        target_box=UnitBox(x=0.1, y=0.1, width=0.4, height=0.8),
        foot_anchor=UnitPoint(x=0.3, y=0.9),
        z_index=1,
    ).mask_policy

    metrics = validate_matte(subject, policy)

    assert metrics["significant_components"] == 1
    assert 0.2 < metrics["coverage"] < 0.5
    with pytest.raises(ValueError, match="coverage"):
        validate_matte(Image.new("L", (256, 256), 0), policy)

    duplicate = Image.new("L", (256, 256), 0)
    duplicate_draw = ImageDraw.Draw(duplicate)
    duplicate_draw.ellipse((20, 40, 105, 225), fill=255)
    duplicate_draw.ellipse((150, 40, 235, 225), fill=255)
    with pytest.raises(ValueError, match="significant components"):
        validate_matte(duplicate, policy)


def test_pixel_composite_respects_authored_depth_and_builds_repair_mask(tmp_path):
    background = tmp_path / "background.png"
    _write_png(background, (40, 50, 70))
    rgba_paths: dict[str, Path] = {}
    for character_id, color in (
        ("rear", (220, 30, 30, 255)),
        ("front", (30, 220, 30, 255)),
    ):
        path = tmp_path / f"{character_id}.png"
        image = Image.new("RGBA", (64, 96), (0, 0, 0, 0))
        ImageDraw.Draw(image).rounded_rectangle((10, 5, 54, 92), radius=8, fill=color)
        image.save(path)
        rgba_paths[character_id] = path
    scene = SceneSpec(
        scene_id="test_scene",
        title="Test scene",
        split="tuning",
        style_pack_id="current-action-fantasy",
        canvas=CanvasSpec(width=512, height=512),
        camera=CameraSpec(description="Eye-level camera."),
        background_prompt="A neutral test background.",
        harmonize_prompt="Repair only contacts.",
        diagnostic_prompt="Render one coherent scene.",
        subjects=[
            SubjectSpec(
                character_id="rear",
                display_name="Rear",
                reference_ids=["rear-reference"],
                identity_prompt="Rear identity.",
                pose_prompt="Complete rear pose.",
                target_box=UnitBox(x=0.2, y=0.15, width=0.35, height=0.7),
                foot_anchor=UnitPoint(x=0.4, y=0.85),
                z_index=1,
            ),
            SubjectSpec(
                character_id="front",
                display_name="Front",
                reference_ids=["front-reference"],
                identity_prompt="Front identity.",
                pose_prompt="Complete front pose.",
                target_box=UnitBox(x=0.4, y=0.2, width=0.35, height=0.7),
                foot_anchor=UnitPoint(x=0.58, y=0.9),
                z_index=2,
            ),
        ],
    )
    output = tmp_path / "pixel.png"
    subject_mask = tmp_path / "subject-mask.png"
    repair_mask = tmp_path / "repair-mask.png"

    placements = composite_subjects(
        scene,
        background,
        rgba_paths,
        output,
        subject_mask,
        repair_mask,
    )

    assert [placement["character_id"] for placement in placements] == ["rear", "front"]
    assert output.is_file()
    with Image.open(subject_mask) as mask:
        assert mask.getbbox() is not None
    with Image.open(repair_mask) as mask:
        assert mask.getbbox() is not None


def test_visual_novel_frames_reuse_one_plate_and_fixed_height_stage_slots(tmp_path):
    scene = _visual_novel_scene()
    scene.subjects[0].stage_height_fraction = 0.62
    style = StylePack(
        style_pack_id="visual-novel-cel",
        title="Visual novel",
        visual_language="Stable cel-shaded visual-novel illustration.",
        component_direction="Preserve identity and stable framing.",
        harmonization_direction="Use deterministic sprite layering.",
    )
    case_root = tmp_path / "case"
    case_root.mkdir()
    Image.new("RGB", (1024, 576), (35, 45, 65)).save(case_root / "background.png")
    matte_root = case_root / "mattes"
    matte_root.mkdir()
    sprite_specs = (
        ("left-rgba.png", (180, 300), (220, 60, 60, 255)),
        ("left--concerned-rgba.png", (150, 250), (190, 45, 45, 255)),
        ("right-rgba.png", (160, 280), (60, 180, 220, 255)),
        ("right--reassuring-rgba.png", (120, 210), (45, 150, 190, 255)),
    )
    for filename, size, color in sprite_specs:
        Image.new("RGBA", size, color).save(matte_root / filename)
    case = CaseState(
        scene=scene,
        style=style,
        seed=9,
        directory=case_root,
        manifest_path=case_root / "manifest.json",
        manifest={"stages": {}},
        references={subject.character_id: [] for subject in scene.subjects},
    )

    StagedImageRunner._visual_novel_pixel_case(case, tmp_path)

    stage = case.manifest["stages"]["pixel"]
    assert stage["background_reused_for_every_frame"] is True
    assert stage["whole_scene_model_edit"] is False
    assert len(stage["frames"]) == 3
    assert len({frame["background_sha256"] for frame in stage["frames"]}) == 1
    left_heights = []
    for frame in stage["frames"]:
        placement = next(
            item for item in frame["placements"] if item["character_id"] == "left"
        )
        assert placement["scale_mode"] == "fixed_height"
        box = placement["paste_box_pixels"]
        left_heights.append(box[3] - box[1])
    assert len(set(left_heights)) == 1
    assert left_heights == [round(0.62 * scene.canvas.height)] * 3
    right_placement = next(
        item
        for item in stage["frames"][0]["placements"]
        if item["character_id"] == "right"
    )
    right_box = right_placement["paste_box_pixels"]
    assert right_box[3] - right_box[1] == round(0.94 * scene.canvas.height)
    assert (case_root / "final/pixel.png").is_file()
    frame_paths = [
        tmp_path / frame["artifact"]["relative_path"] for frame in stage["frames"]
    ]
    with Image.open(frame_paths[0]) as first, Image.open(frame_paths[-1]) as last:
        assert first.getpixel((512, 5)) == last.getpixel((512, 5))


def test_declared_reference_cutout_never_calls_a_generation_client(tmp_path):
    reference_path = tmp_path / "artifacts/reference.png"
    _write_png(reference_path, (40, 80, 120))
    reference = ResolvedReference(
        reference_id="reference",
        source="reviewed",
        path=reference_path,
        sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        mime_type="image/png",
        width=64,
        height=96,
        byte_count=reference_path.stat().st_size,
    )
    subject = SubjectSpec(
        character_id="subject",
        display_name="Subject",
        reference_ids=["reference"],
        component_mode="reference_cutout",
        identity_prompt="Stable reference identity.",
        pose_prompt="Keep the approved complete pose.",
        target_box=UnitBox(x=0.1, y=0.1, width=0.4, height=0.8),
        foot_anchor=UnitPoint(x=0.3, y=0.9),
        z_index=1,
    )
    scene = SceneSpec(
        scene_id="cutout_scene",
        title="Cutout scene",
        split="tuning",
        style_pack_id="style",
        camera=CameraSpec(description="Eye-level camera."),
        background_prompt="A neutral test background.",
        harmonize_prompt="Repair contacts.",
        diagnostic_prompt="Render one coherent scene.",
        subjects=[subject],
    )
    style = StylePack(
        style_pack_id="style",
        title="Style",
        visual_language="A coherent illustration style.",
        component_direction="Preserve the component.",
        harmonization_direction="Repair only seams.",
    )
    case_root = tmp_path / "case"
    case_root.mkdir()
    case = CaseState(
        scene=scene,
        style=style,
        seed=1,
        directory=case_root,
        manifest_path=case_root / "manifest.json",
        manifest={"stages": {}},
        references={"subject": [reference]},
    )

    StagedImageRunner._component_stage(
        object.__new__(StagedImageRunner), [case], tmp_path
    )

    stage = case.manifest["stages"]["components"]["subject"]
    assert stage["component_mode"] == "reference_cutout"
    assert stage["prompt"] is None
    assert (case_root / "components/subject.png").is_file()


def test_reference_facing_mirror_is_explicit_and_non_destructive(tmp_path):
    reference_path = tmp_path / "artifacts/reference.png"
    reference_path.parent.mkdir()
    source = Image.new("RGB", (4, 2), (0, 0, 0))
    source.putpixel((0, 0), (255, 0, 0))
    source.putpixel((3, 0), (0, 0, 255))
    source.save(reference_path)
    reference = ResolvedReference(
        reference_id="reference",
        source="reviewed",
        path=reference_path,
        sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        mime_type="image/png",
        width=4,
        height=2,
        byte_count=reference_path.stat().st_size,
    )
    scene = _visual_novel_scene()
    subject = scene.subjects[0].model_copy(
        update={"facing_control": "mirror_reference"}
    )
    case_root = tmp_path / "case"
    case_root.mkdir()
    case = CaseState(
        scene=scene,
        style=StylePack(
            style_pack_id="visual-novel-cel",
            title="Visual novel",
            visual_language="Stable illustration.",
            component_direction="Preserve identity.",
            harmonization_direction="Use deterministic layers.",
        ),
        seed=1,
        directory=case_root,
        manifest_path=case_root / "manifest.json",
        manifest={"stages": {}},
        references={"left": [reference], "right": []},
    )

    paths, records = prepare_component_reference_inputs(
        case,
        subject,
        tmp_path,
    )

    with Image.open(reference_path) as original, Image.open(paths[0]) as mirrored:
        assert original.getpixel((0, 0)) == (255, 0, 0)
        assert original.getpixel((3, 0)) == (0, 0, 255)
        assert mirrored.getpixel((0, 0)) == (0, 0, 255)
        assert mirrored.getpixel((3, 0)) == (255, 0, 0)
    assert records[0]["operation"] == "mirror_horizontal"
    assert records[0]["source_sha256"] == reference.sha256
    assert records[0]["artifact"]["relative_path"].startswith("case/inputs/")


def test_resume_skips_hash_valid_background_and_matte_artifacts(tmp_path):
    subject = SubjectSpec(
        character_id="subject",
        display_name="Subject",
        reference_ids=["reference"],
        identity_prompt="Stable identity.",
        pose_prompt="Complete standing pose.",
        target_box=UnitBox(x=0.1, y=0.1, width=0.4, height=0.8),
        foot_anchor=UnitPoint(x=0.3, y=0.9),
        z_index=1,
    )
    scene = SceneSpec(
        scene_id="resume_scene",
        title="Resume scene",
        split="tuning",
        style_pack_id="style",
        camera=CameraSpec(description="Eye-level camera."),
        background_prompt="An empty room.",
        harmonize_prompt="Repair contacts.",
        diagnostic_prompt="Render one coherent scene.",
        subjects=[subject],
    )
    style = StylePack(
        style_pack_id="style",
        title="Style",
        visual_language="A coherent illustration style.",
        component_direction="Preserve the component.",
        harmonization_direction="Repair only seams.",
    )
    case_root = tmp_path / "case"
    background_path = case_root / "background.png"
    component_path = case_root / "components/subject.png"
    mask_path = case_root / "mattes/subject-mask.png"
    rgba_path = case_root / "mattes/subject-rgba.png"
    _write_png(background_path, (10, 20, 30))
    _write_png(component_path, (40, 50, 60))
    _write_png(mask_path, (255,), mode="L")
    _write_png(rgba_path, (40, 50, 60, 255), mode="RGBA")
    case = CaseState(
        scene=scene,
        style=style,
        seed=5,
        directory=case_root,
        manifest_path=case_root / "manifest.json",
        manifest={"stages": {}},
        references={"subject": []},
    )
    background_prompt = StagedImageRunner._background_prompt(case)
    case.manifest["stages"] = {
        "background": {
            "prompt_sha256": hashlib.sha256(
                background_prompt.encode("utf-8")
            ).hexdigest(),
            "seed": derived_seed(5, "resume_scene", "background"),
            "artifact": _artifact_payload(background_path, tmp_path),
        },
        "mattes": {
            "subject": {
                "component_artifact_sha256": hashlib.sha256(
                    component_path.read_bytes()
                ).hexdigest(),
                "mask_artifact": _artifact_payload(mask_path, tmp_path),
                "rgba_artifact": _artifact_payload(rgba_path, tmp_path),
            }
        },
    }

    class ExplodingClient:
        def post_image(self, *_args, **_kwargs):
            raise AssertionError("a completed stage called the gateway")

    runner = object.__new__(StagedImageRunner)
    runner.client = ExplodingClient()

    runner._background_stage([case], tmp_path)
    runner._matte_stage([case], tmp_path)


def test_background_reuse_copies_only_a_matching_frozen_plate(tmp_path):
    scene = _visual_novel_scene()
    style = StylePack(
        style_pack_id="visual-novel-cel",
        title="Visual novel",
        visual_language="Stable cel-shaded visual-novel illustration.",
        component_direction="Preserve identity and stable framing.",
        harmonization_direction="Use deterministic sprite layering.",
    )
    source_root = tmp_path / "source"
    source_case_root = source_root / scene.scene_id / "seed-5"
    source_background = source_case_root / "background.png"
    _write_png(source_background, (12, 34, 56))
    destination_root = tmp_path / "destination"
    destination_case_root = destination_root / scene.scene_id / "seed-5"
    destination_case_root.mkdir(parents=True)
    case = CaseState(
        scene=scene,
        style=style,
        seed=5,
        directory=destination_case_root,
        manifest_path=destination_case_root / "manifest.json",
        manifest={"stages": {}},
        references={subject.character_id: [] for subject in scene.subjects},
    )
    prompt = StagedImageRunner._background_prompt(case)
    source_manifest = {
        "scene_id": scene.scene_id,
        "seed": 5,
        "stages": {
            "background": {
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "seed": derived_seed(5, scene.scene_id, "background"),
                "artifact": _artifact_payload(source_background, source_root),
            }
        },
    }
    (source_root / "run.json").write_text("{}", encoding="utf-8")
    (source_case_root / "manifest.json").write_text(
        json.dumps(source_manifest),
        encoding="utf-8",
    )

    runner = object.__new__(StagedImageRunner)
    runner._reuse_backgrounds(
        [case],
        destination_root,
        source_root,
        "source",
    )

    reused = destination_case_root / "background.png"
    assert reused.read_bytes() == source_background.read_bytes()
    stage = case.manifest["stages"]["background"]
    assert stage["operation"] == "frozen background reuse"
    assert stage["source_run_id"] == "source"
    assert (
        stage["source_artifact_sha256"]
        == hashlib.sha256(source_background.read_bytes()).hexdigest()
    )


def test_rendered_background_prompt_contains_only_positive_environment_context():
    subject = SubjectSpec(
        character_id="subject",
        display_name="Unique Person",
        reference_ids=["reference"],
        identity_prompt="Unique identity details that belong only to the subject.",
        pose_prompt="A unique full-body subject pose.",
        target_box=UnitBox(x=0.1, y=0.1, width=0.4, height=0.8),
        foot_anchor=UnitPoint(x=0.3, y=0.9),
        z_index=1,
    )
    scene = SceneSpec(
        scene_id="environment_only",
        title="Environment only",
        split="tuning",
        style_pack_id="style",
        camera=CameraSpec(description="Eye-level view across an empty stone room."),
        background_prompt="An unoccupied stone room with a broad open floor.",
        harmonize_prompt="Repair contacts.",
        diagnostic_prompt="Render one coherent scene.",
        subjects=[subject],
    )
    style = StylePack(
        style_pack_id="style",
        title="Style",
        visual_language="Full-color cinematic illustration.",
        component_direction="Preserve the component.",
        harmonization_direction="Repair only seams.",
    )
    case = CaseState(
        scene=scene,
        style=style,
        seed=1,
        directory=Path("unused"),
        manifest_path=Path("unused/manifest.json"),
        manifest={"stages": {}},
        references={"subject": []},
    )

    prompt = StagedImageRunner._background_prompt(case)

    assert prompt.startswith("Empty environment plate.")
    assert "Unique Person" not in prompt
    assert subject.identity_prompt not in prompt
    assert subject.pose_prompt not in prompt
    assert "No people" not in prompt


def test_checked_in_plate_descriptions_do_not_request_future_subjects():
    corpora = [
        load_experiment(path, DEFAULT_STYLE_PATH)[0]
        for path in (DEFAULT_CORPUS_PATH, LEGACY_CINEMATIC_CORPUS_PATH)
    ]
    forbidden = (
        " figure",
        " scout",
        " adult",
        " runner",
        " duel",
        " weapon",
        " silhouette",
        " pursuer",
        " no ",
    )

    for corpus in corpora:
        for scene in corpus.scenes:
            plate_text = (
                f" {scene.camera.description} {scene.background_prompt} ".lower()
            )
            assert not any(term in plate_text for term in forbidden), scene.scene_id
            assert all(
                re.search(
                    rf"\b{re.escape(subject.display_name.lower())}\b",
                    plate_text,
                )
                is None
                for subject in scene.subjects
            )


def test_qwen_component_receives_identity_views_but_not_environment_plate(tmp_path):
    reference_paths = [
        tmp_path / "artifacts/reference-1.png",
        tmp_path / "artifacts/reference-2.png",
    ]
    for index, path in enumerate(reference_paths, start=1):
        _write_png(path, (index * 30, 60, 90))
    references = [
        ResolvedReference(
            reference_id=f"reference-{index}",
            source="reviewed",
            path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            mime_type="image/png",
            width=64,
            height=96,
            byte_count=path.stat().st_size,
        )
        for index, path in enumerate(reference_paths, start=1)
    ]
    subject = SubjectSpec(
        character_id="subject",
        display_name="Subject",
        reference_ids=[item.reference_id for item in references],
        identity_prompt="Stable identity.",
        pose_prompt="Complete standing pose.",
        target_box=UnitBox(x=0.1, y=0.1, width=0.4, height=0.8),
        foot_anchor=UnitPoint(x=0.3, y=0.9),
        z_index=1,
    )
    scene = SceneSpec(
        scene_id="component_scene",
        title="Component scene",
        split="tuning",
        style_pack_id="style",
        camera=CameraSpec(description="Eye-level camera."),
        background_prompt="An empty room.",
        harmonize_prompt="Repair contacts.",
        diagnostic_prompt="Render one coherent scene.",
        subjects=[subject],
    )
    style = StylePack(
        style_pack_id="style",
        title="Style",
        visual_language="A coherent illustration style.",
        component_direction="Preserve the component.",
        harmonization_direction="Repair only seams.",
    )
    case_root = tmp_path / "case"
    case_root.mkdir()
    _write_png(case_root / "background.png", (220, 10, 10))
    case = CaseState(
        scene=scene,
        style=style,
        seed=1,
        directory=case_root,
        manifest_path=case_root / "manifest.json",
        manifest={"stages": {}},
        references={"subject": references},
    )

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def post_image(
            self, endpoint: str, payload: dict[str, object]
        ) -> tuple[bytes, dict[str, str]]:
            self.calls.append((endpoint, payload))
            return reference_paths[0].read_bytes(), {}

    client = RecordingClient()
    runner = object.__new__(StagedImageRunner)
    runner.client = client
    runner.max_workers = 2
    runner.worker_targets = ("worker-0", "worker-1", "worker-2", "worker-3")

    runner._component_stage([case], tmp_path)
    runner._component_stage([case], tmp_path)

    endpoint, payload = client.calls[0]
    assert len(client.calls) == 1
    assert endpoint == "/edit/qwen"
    assert "image2_base64" in payload
    assert "image3_base64" not in payload
    assert "environment plate" not in str(payload["prompt"]).lower()
    assert case.manifest["stages"]["components"]["subject"]["reference_order"] == [
        "reference-1",
        "reference-2",
    ]


def test_visual_novel_variants_edit_only_the_neutral_sprite_anchor(tmp_path):
    scene = _visual_novel_scene()
    style = StylePack(
        style_pack_id="visual-novel-cel",
        title="Visual novel",
        visual_language="Stable cel-shaded visual-novel illustration.",
        component_direction="Preserve identity and stable framing.",
        harmonization_direction="Use deterministic sprite layering.",
    )
    case_root = tmp_path / "case"
    for subject, color in zip(
        scene.subjects,
        ((40, 80, 120), (120, 80, 40)),
        strict=True,
    ):
        _write_png(case_root / "components" / f"{subject.character_id}.png", color)
    case = CaseState(
        scene=scene,
        style=style,
        seed=7,
        directory=case_root,
        manifest_path=case_root / "manifest.json",
        manifest={"stages": {}},
        references={subject.character_id: [] for subject in scene.subjects},
    )

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def post_image(
            self, endpoint: str, payload: dict[str, object]
        ) -> tuple[bytes, dict[str, str]]:
            self.calls.append((endpoint, payload))
            return (case_root / "components/left.png").read_bytes(), {}

    client = RecordingClient()
    runner = object.__new__(StagedImageRunner)
    runner.client = client
    runner.max_workers = 2
    runner.worker_targets = ("worker-0", "worker-1", "worker-2", "worker-3")

    runner._variant_stage([case], tmp_path)
    runner._variant_stage([case], tmp_path)

    assert len(client.calls) == 2
    assert all(endpoint == "/edit/qwen" for endpoint, _ in client.calls)
    assert all("image_base64" in payload for _, payload in client.calls)
    assert all("image2_base64" not in payload for _, payload in client.calls)
    assert all("image3_base64" not in payload for _, payload in client.calls)
    assert {str(payload["worker"]) for _, payload in client.calls} == {
        "worker-0",
        "worker-1",
    }
    assert (case_root / "components/left--concerned.png").is_file()
    assert (case_root / "components/right--reassuring.png").is_file()
    left_stage = case.manifest["stages"]["variants"]["left"]["concerned"]
    assert left_stage["anchor_variant_id"] == "base"
    assert left_stage["input_count"] == 1
    prompt = left_stage["prompt"]
    assert "only edit target" in prompt
    assert "Preserve identity" in prompt
    assert "environment" not in prompt.lower()


def test_visual_novel_anchor_prompt_is_single_subject_and_inward_facing():
    scene = _visual_novel_scene()
    style = StylePack(
        style_pack_id="visual-novel-cel",
        title="Visual novel",
        visual_language="Stable cel-shaded visual-novel illustration.",
        component_direction="Preserve identity and stable framing.",
        harmonization_direction="Use deterministic sprite layering.",
    )
    case = CaseState(
        scene=scene,
        style=style,
        seed=7,
        directory=Path("unused"),
        manifest_path=Path("unused/manifest.json"),
        manifest={"stages": {}},
        references={"left": [object()], "right": [object()]},
    )

    prompt = StagedImageRunner._component_prompt(case, scene.subjects[0])

    assert "neutral reusable visual-novel sprite anchor" in prompt
    assert "left side of the screen" in prompt
    assert "toward screen right" in prompt
    assert "Follow the framing specified by the neutral pose" in prompt
    assert scene.subjects[1].display_name not in prompt
    assert scene.background_prompt not in prompt

    prepared = scene.subjects[0].model_copy(
        update={"facing_control": "preserve_reference"}
    )
    prepared_prompt = StagedImageRunner._component_prompt(case, prepared)
    assert "already been prepared to face screen right" in prepared_prompt
    assert "do not turn or mirror the character" in prepared_prompt


def test_qwen_final_edits_receive_only_the_assembled_scene(tmp_path):
    subject = SubjectSpec(
        character_id="subject",
        display_name="Subject",
        reference_ids=["reference"],
        identity_prompt="Stable identity.",
        pose_prompt="Complete standing pose.",
        target_box=UnitBox(x=0.1, y=0.1, width=0.4, height=0.8),
        foot_anchor=UnitPoint(x=0.3, y=0.9),
        z_index=1,
    )
    scene = SceneSpec(
        scene_id="final_scene",
        title="Final scene",
        split="tuning",
        style_pack_id="style",
        camera=CameraSpec(description="Eye-level camera."),
        background_prompt="An empty room.",
        harmonize_prompt="Repair contacts.",
        diagnostic_prompt="Render one coherent scene.",
        subjects=[subject],
    )
    style = StylePack(
        style_pack_id="style",
        title="Style",
        visual_language="A coherent illustration style.",
        component_direction="Preserve the component.",
        harmonization_direction="Repair only seams.",
    )
    case_root = tmp_path / "case"
    final_root = case_root / "final"
    final_root.mkdir(parents=True)
    _write_png(final_root / "pixel.png", (10, 20, 30))
    Image.new("L", (64, 96), 255).save(final_root / "repair-mask.png")
    case = CaseState(
        scene=scene,
        style=style,
        seed=1,
        directory=case_root,
        manifest_path=case_root / "manifest.json",
        manifest={"stages": {}},
        references={"subject": []},
    )

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def post_image(
            self, endpoint: str, payload: dict[str, object]
        ) -> tuple[bytes, dict[str, str]]:
            self.calls.append((endpoint, payload))
            return (final_root / "pixel.png").read_bytes(), {}

    client = RecordingClient()
    runner = object.__new__(StagedImageRunner)
    runner.client = client

    runner._final_stage([case], tmp_path, ("pixel", "masked", "global"))

    assert {endpoint for endpoint, _ in client.calls} == {
        "/prototype/edit/qwen/masked",
        "/edit/qwen",
    }
    assert all("image2_base64" not in payload for _, payload in client.calls)
    assert all("image3_base64" not in payload for _, payload in client.calls)


def test_masked_blend_freezes_every_pixel_outside_mask(tmp_path):
    original = tmp_path / "original.png"
    generated = tmp_path / "generated.png"
    mask = tmp_path / "mask.png"
    output = tmp_path / "output.png"
    _write_png(original, (10, 20, 30))
    _write_png(generated, (240, 230, 220))
    mask_image = Image.new("L", (64, 96), 0)
    ImageDraw.Draw(mask_image).rectangle((20, 30, 40, 60), fill=255)
    mask_image.save(mask)

    changed_outside = blend_masked_result(original, generated, mask, output)

    assert changed_outside == 0
    with Image.open(original) as before, Image.open(output) as after:
        assert before.getpixel((0, 0)) == after.getpixel((0, 0))
        assert before.getpixel((30, 45)) != after.getpixel((30, 45))


def test_review_requires_loser_reason_and_report_keeps_all_tracks(tmp_path):
    run_root = tmp_path / "run"
    case_root = run_root / "scene" / "seed-1"
    artifact_path = case_root / "final/pixel.png"
    _write_png(artifact_path, (10, 20, 30))
    artifact = {
        "relative_path": artifact_path.relative_to(run_root).as_posix(),
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "byte_count": artifact_path.stat().st_size,
        "width": 64,
        "height": 96,
        "mode": "RGB",
    }
    frame_path = case_root / "final/frames/neutral.png"
    _write_png(frame_path, (30, 20, 10))
    frame_artifact = {
        "relative_path": frame_path.relative_to(run_root).as_posix(),
        "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        "byte_count": frame_path.stat().st_size,
        "width": 64,
        "height": 96,
        "mode": "RGB",
    }
    case_manifest = {
        "scene_id": "scene",
        "scene_title": "Scene",
        "split": "tuning",
        "seed": 1,
        "tracks": ["pixel"],
        "stages": {
            "pixel": {
                "artifact": artifact,
                "frames": [
                    {
                        "title": "Neutral anchors",
                        "artifact": frame_artifact,
                    }
                ],
            }
        },
    }
    (case_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (case_root / "manifest.json").write_text(json.dumps(case_manifest))
    (run_root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "status": "complete",
                "cases": ["scene/seed-1/manifest.json"],
            }
        )
    )

    with pytest.raises(ValidationError, match="requires at least one reason"):
        record_review(
            run_root=run_root,
            scene_id="scene",
            seed=1,
            track="pixel",
            verdict="loser",
            reasons=[],
            note="",
            reviewer="tester",
        )
    record_review(
        run_root=run_root,
        scene_id="scene",
        seed=1,
        track="pixel",
        verdict="loser",
        reasons=["identity"],
        note="Identity drifted.",
        reviewer="tester",
    )

    html_path, json_path = build_report(run_root)

    assert html_path.is_file()
    report = json.loads(json_path.read_text())
    assert report["counts"] == {
        "pass": 0,
        "loser": 1,
        "unreviewed": 0,
        "stale": 0,
    }
    assert report["quality_gate_passed"] is False
    assert report["loser_reasons"]["identity"] == 1
    assert report["records"][0]["frames"][0]["title"] == "Neutral anchors"
    assert "Neutral anchors" in html_path.read_text(encoding="utf-8")


def test_seed_parser_forbids_cherry_pick_batches():
    assert parse_seeds("1,2,3,4") == (1, 2, 3, 4)
    with pytest.raises(Exception, match="exactly four"):
        parse_seeds("1")


def test_worker_concurrency_is_bounded_and_round_robin():
    with pytest.raises(ValueError, match="between 1 and 4"):
        StagedImageRunner(max_workers=0)
    with pytest.raises(ValueError, match="unique values"):
        StagedImageRunner(worker_indices=[1, 1])

    runner = StagedImageRunner(max_workers=1)
    runner.worker_targets = ("worker-0", "worker-1", "worker-2", "worker-3")
    assert [runner._worker_target(index) for index in range(6)] == [
        "worker-0",
        "worker-1",
        "worker-2",
        "worker-3",
        "worker-0",
        "worker-1",
    ]


def test_resume_manifest_must_match_the_frozen_batch_contract():
    runner = StagedImageRunner()
    scenes = runner.select_scenes(["vn03_beginner_lobby_iselle_wren"])
    seeds = (1, 2, 3, 4)
    tracks = ("pixel",)
    protocol_versions = {"visual_novel": "visual-novel-anchors-v2"}
    manifest = {
        "run_id": "run",
        "status": "interrupted",
        "scene_ids": ["vn03_beginner_lobby_iselle_wren"],
        "seeds": list(seeds),
        "tracks": list(tracks),
        "style_pack_override": None,
        "background_source_run_id": None,
        "holdouts_included": False,
        "reference_session_id": runner.corpus.reference_session_id,
        "prompt_protocol_versions": protocol_versions,
        "corpus": {
            "sha256": hashlib.sha256(runner.corpus_path.read_bytes()).hexdigest()
        },
        "styles": {
            "sha256": hashlib.sha256(runner.style_path.read_bytes()).hexdigest()
        },
        "cases": [
            f"vn03_beginner_lobby_iselle_wren/seed-{seed}/manifest.json"
            for seed in seeds
        ],
    }

    runner._validate_resume_manifest(
        run_manifest=manifest,
        run_id="run",
        scenes=scenes,
        seeds=seeds,
        tracks=tracks,
        include_holdout=False,
        style_pack_id=None,
        protocol_versions=protocol_versions,
        reuse_backgrounds_from=None,
    )

    manifest["seeds"] = [4, 3, 2, 1]
    with pytest.raises(ValueError, match="frozen run: seeds"):
        runner._validate_resume_manifest(
            run_manifest=manifest,
            run_id="run",
            scenes=scenes,
            seeds=seeds,
            tracks=tracks,
            include_holdout=False,
            style_pack_id=None,
            protocol_versions=protocol_versions,
            reuse_backgrounds_from=None,
        )


def test_derived_stage_seeds_fit_every_deployed_backend():
    seed = derived_seed(26082301, "scene", "background")

    assert 0 <= seed <= (1 << 63) - 1
    assert seed == derived_seed(26082301, "scene", "background")


def test_subject_schema_caps_identity_inputs_to_the_reviewed_two_view_contract():
    with pytest.raises(ValidationError):
        SubjectSpec(
            character_id="subject",
            display_name="Subject",
            reference_ids=["one", "two", "three"],
            identity_prompt="Stable reference identity.",
            pose_prompt="A complete standing pose.",
            target_box=UnitBox(x=0.1, y=0.1, width=0.4, height=0.8),
            foot_anchor=UnitPoint(x=0.3, y=0.9),
            z_index=1,
        )
