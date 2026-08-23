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
    CaseState,
    CanvasSpec,
    CameraSpec,
    CorpusSpec,
    ReferenceResolver,
    ResolvedReference,
    SceneSpec,
    StagedImageRunner,
    StylePack,
    SubjectSpec,
    UnitBox,
    UnitPoint,
    blend_masked_result,
    build_report,
    composite_subjects,
    derived_seed,
    load_experiment,
    parse_seeds,
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


def test_checked_in_corpus_has_six_tuning_scenes_and_two_untouched_holdouts():
    corpus, styles = load_experiment(DEFAULT_CORPUS_PATH, DEFAULT_STYLE_PATH)

    assert isinstance(corpus, CorpusSpec)
    assert sum(scene.split == "tuning" for scene in corpus.scenes) == 6
    assert sum(scene.split == "holdout" for scene in corpus.scenes) == 2
    assert len(styles) == 3
    assert all(
        1 <= len(subject.reference_ids) <= 2
        for scene in corpus.scenes
        for subject in scene.subjects
    )
    arrival_iselle = next(
        subject
        for scene in corpus.scenes
        if scene.scene_id == "t01_failed_arrival_hall"
        for subject in scene.subjects
        if subject.character_id == "iselle_the_guide"
    )
    assert arrival_iselle.component_mode == "reference_cutout"


def test_holdouts_require_an_explicit_release_flag():
    runner = StagedImageRunner()

    assert len(runner.select_scenes([])) == 6
    with pytest.raises(ValueError, match="--include-holdout"):
        runner.select_scenes(["h01_frost_courtyard"])
    assert (
        runner.select_scenes(
            ["h01_frost_courtyard"],
            include_holdout=True,
        )[0].split
        == "holdout"
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
    corpus, _ = load_experiment(DEFAULT_CORPUS_PATH, DEFAULT_STYLE_PATH)
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

    for scene in corpus.scenes:
        plate_text = f" {scene.camera.description} {scene.background_prompt} ".lower()
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

    runner._component_stage([case], tmp_path)

    endpoint, payload = client.calls[0]
    assert endpoint == "/edit/qwen"
    assert "image2_base64" in payload
    assert "image3_base64" not in payload
    assert "environment plate" not in str(payload["prompt"]).lower()
    assert case.manifest["stages"]["components"]["subject"]["reference_order"] == [
        "reference-1",
        "reference-2",
    ]


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
    case_manifest = {
        "scene_id": "scene",
        "scene_title": "Scene",
        "split": "tuning",
        "seed": 1,
        "tracks": ["pixel"],
        "stages": {"pixel": {"artifact": artifact}},
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


def test_seed_parser_forbids_cherry_pick_batches():
    assert parse_seeds("1,2,3,4") == (1, 2, 3, 4)
    with pytest.raises(Exception, match="exactly four"):
        parse_seeds("1")


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
