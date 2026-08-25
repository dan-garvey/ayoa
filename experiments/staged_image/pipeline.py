"""Quality-first staged image composition experiment.

This module is intentionally separate from Ayoa's runtime image director.  It
uses reviewed runtime artifacts as immutable inputs, but it does not enqueue,
deliver, promote, or otherwise mutate runtime image-generation state.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
from pydantic import BaseModel, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CINEMATIC_CORPUS_PATH = REPO_ROOT / "experiments/staged_image/corpus.json"
DEFAULT_VISUAL_NOVEL_CORPUS_PATH = (
    REPO_ROOT / "experiments/staged_image/visual_novel_corpus.json"
)
DEFAULT_CORPUS_PATH = DEFAULT_VISUAL_NOVEL_CORPUS_PATH
DEFAULT_STYLE_PATH = REPO_ROOT / "experiments/staged_image/style_packs.json"
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "app/storage/runtime/image_generation"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "app/storage/runtime/image_prototypes"
DEFAULT_GATEWAY_URL = "http://127.0.0.1:8199"
DEFAULT_SEEDS = (26082301, 26082302, 26082303, 26082304)
TRACKS = ("pixel", "masked", "global")
DEFAULT_TRACKS = ("pixel",)
PROMPT_PROTOCOL_VERSIONS = {
    "cinematic": "official-minimal-v2",
    "visual_novel": "visual-novel-anchors-v2",
}
BASE_VARIANT_ID = "base"
REVIEW_REASONS = (
    "identity",
    "subject_count",
    "scale",
    "pose",
    "anatomy",
    "prop",
    "matte",
    "lighting",
    "composition",
    "style",
    "expression",
    "continuity",
    "text_artifact",
    "other",
)


class UnitPoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class UnitBox(BaseModel):
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def stays_on_canvas(self) -> "UnitBox":
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("normalized box extends beyond the canvas")
        return self


class CanvasSpec(BaseModel):
    width: int = Field(default=768, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)


class CameraSpec(BaseModel):
    description: str = Field(min_length=1, max_length=1000)
    horizon_y: float = Field(default=0.42, ge=0, le=1)


class ContactRegion(BaseModel):
    kind: Literal["ellipse", "polygon"]
    box: UnitBox | None = None
    points: list[UnitPoint] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def has_geometry(self) -> "ContactRegion":
        if self.kind == "ellipse" and self.box is None:
            raise ValueError("ellipse contact region requires box")
        if self.kind == "polygon" and len(self.points) < 3:
            raise ValueError("polygon contact region requires at least 3 points")
        return self


class MaskPolicy(BaseModel):
    min_coverage: float = Field(default=0.025, gt=0, lt=1)
    max_coverage: float = Field(default=0.82, gt=0, le=1)
    max_edge_fraction: float = Field(default=0.04, ge=0, le=1)
    max_significant_components: int = Field(default=1, ge=1, le=8)


class SpriteVariantSpec(BaseModel):
    variant_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9_-]+$",
    )
    expression_prompt: str = Field(min_length=1, max_length=1000)
    pose_adjustment_prompt: str = Field(min_length=1, max_length=1000)


class VisualNovelFrameSpec(BaseModel):
    frame_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9_-]+$",
    )
    title: str = Field(min_length=1, max_length=160)
    active_character_id: str | None = Field(default=None, max_length=200)
    subject_variants: dict[str, str] = Field(default_factory=dict, max_length=2)


class SubjectSpec(BaseModel):
    character_id: str = Field(min_length=1, max_length=200)
    display_name: str = Field(min_length=1, max_length=100)
    reference_ids: list[str] = Field(min_length=1, max_length=2)
    component_mode: Literal["qwen_edit", "reference_cutout"] = "qwen_edit"
    identity_prompt: str = Field(min_length=1, max_length=2000)
    pose_prompt: str = Field(min_length=1, max_length=2000)
    target_box: UnitBox | None = None
    foot_anchor: UnitPoint | None = None
    z_index: int | None = Field(default=None, ge=0, le=100)
    screen_side: Literal["left", "right"] | None = None
    facing: Literal["left", "right"] | None = None
    facing_control: Literal[
        "prompt",
        "preserve_reference",
        "mirror_reference",
    ] = "prompt"
    stage_height_fraction: float | None = Field(default=None, ge=0.25, le=0.94)
    variants: list[SpriteVariantSpec] = Field(default_factory=list, max_length=6)
    grounded: bool = True
    mask_policy: MaskPolicy = Field(default_factory=MaskPolicy)

    @model_validator(mode="after")
    def cutout_has_one_authoritative_source(self) -> "SubjectSpec":
        if self.component_mode == "reference_cutout" and len(self.reference_ids) != 1:
            raise ValueError("reference cutout requires exactly one reference")
        if self.facing_control != "prompt" and len(self.reference_ids) != 1:
            raise ValueError("reference-facing control requires exactly one reference")
        variant_ids = [variant.variant_id for variant in self.variants]
        if BASE_VARIANT_ID in variant_ids:
            raise ValueError(f"{BASE_VARIANT_ID!r} is reserved for the anchor sprite")
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("sprite variant ids must be unique per subject")
        return self


class SceneSpec(BaseModel):
    scene_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_-]+$")
    title: str = Field(min_length=1, max_length=200)
    split: Literal["tuning", "holdout"]
    style_pack_id: str = Field(min_length=1, max_length=100)
    presentation: Literal["cinematic", "visual_novel"] = "cinematic"
    canvas: CanvasSpec = Field(default_factory=CanvasSpec)
    camera: CameraSpec
    background_prompt: str = Field(min_length=1, max_length=4000)
    harmonize_prompt: str | None = Field(default=None, min_length=1, max_length=4000)
    diagnostic_prompt: str | None = Field(default=None, min_length=1, max_length=4000)
    subjects: list[SubjectSpec] = Field(min_length=1, max_length=4)
    contact_regions: list[ContactRegion] = Field(default_factory=list, max_length=12)
    frames: list[VisualNovelFrameSpec] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def has_unique_subjects(self) -> "SceneSpec":
        ids = [subject.character_id for subject in self.subjects]
        if len(ids) != len(set(ids)):
            raise ValueError("scene character ids must be unique")
        if self.presentation == "cinematic":
            if self.harmonize_prompt is None or self.diagnostic_prompt is None:
                raise ValueError(
                    "cinematic scenes require harmonize and diagnostic prompts"
                )
            if self.frames:
                raise ValueError("cinematic scenes cannot declare visual-novel frames")
            for subject in self.subjects:
                if (
                    subject.target_box is None
                    or subject.foot_anchor is None
                    or subject.z_index is None
                ):
                    raise ValueError(
                        "cinematic subjects require target_box, foot_anchor, and z_index"
                    )
                if subject.screen_side is not None or subject.facing is not None:
                    raise ValueError(
                        "cinematic subjects cannot declare visual-novel screen geometry"
                    )
                if subject.stage_height_fraction is not None:
                    raise ValueError(
                        "cinematic subjects cannot declare visual-novel stage height"
                    )
                if subject.facing_control != "prompt":
                    raise ValueError(
                        "cinematic subjects cannot declare visual-novel facing control"
                    )
                if subject.variants:
                    raise ValueError(
                        "cinematic subjects cannot declare visual-novel variants"
                    )
            return self

        if self.harmonize_prompt is not None or self.diagnostic_prompt is not None:
            raise ValueError("visual-novel scenes do not use whole-scene edit prompts")
        if self.contact_regions:
            raise ValueError("visual-novel scenes do not use contact repair regions")
        if len(self.subjects) != 2:
            raise ValueError("visual-novel scenes require exactly two subjects")
        by_side = {subject.screen_side: subject for subject in self.subjects}
        if set(by_side) != {"left", "right"}:
            raise ValueError(
                "visual-novel scenes require one left and one right subject"
            )
        if by_side["left"].facing != "right" or by_side["right"].facing != "left":
            raise ValueError("visual-novel subjects must face inward")
        for subject in self.subjects:
            if (
                subject.target_box is not None
                or subject.foot_anchor is not None
                or subject.z_index is not None
            ):
                raise ValueError(
                    "visual-novel placement is derived from screen_side, not authored boxes"
                )
        if len(self.frames) < 2:
            raise ValueError("visual-novel scenes require at least two frames")
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("visual-novel frame ids must be unique")
        valid_variants = {
            subject.character_id: {
                BASE_VARIANT_ID,
                *(variant.variant_id for variant in subject.variants),
            }
            for subject in self.subjects
        }
        used_variants = {subject.character_id: set() for subject in self.subjects}
        has_anchor_frame = False
        for frame in self.frames:
            if (
                frame.active_character_id is not None
                and frame.active_character_id not in ids
            ):
                raise ValueError(f"frame {frame.frame_id} has unknown active character")
            unknown_subjects = set(frame.subject_variants).difference(ids)
            if unknown_subjects:
                raise ValueError(
                    f"frame {frame.frame_id} has unknown subjects: "
                    + ", ".join(sorted(unknown_subjects))
                )
            resolved = {
                character_id: frame.subject_variants.get(
                    character_id,
                    BASE_VARIANT_ID,
                )
                for character_id in ids
            }
            has_anchor_frame = has_anchor_frame or all(
                variant_id == BASE_VARIANT_ID for variant_id in resolved.values()
            )
            for character_id, variant_id in resolved.items():
                if variant_id not in valid_variants[character_id]:
                    raise ValueError(
                        f"frame {frame.frame_id} uses unknown variant "
                        f"{character_id}/{variant_id}"
                    )
                if variant_id != BASE_VARIANT_ID:
                    used_variants[character_id].add(variant_id)
        if not has_anchor_frame:
            raise ValueError("visual-novel scenes require one all-anchor frame")
        for subject in self.subjects:
            declared = {variant.variant_id for variant in subject.variants}
            unused = declared.difference(used_variants[subject.character_id])
            if unused:
                raise ValueError(
                    f"unused variants for {subject.character_id}: "
                    + ", ".join(sorted(unused))
                )
        return self


class CorpusSpec(BaseModel):
    version: int = Field(ge=1)
    reference_session_id: str = Field(min_length=1)
    scenes: list[SceneSpec] = Field(min_length=3)

    @model_validator(mode="after")
    def has_required_split(self) -> "CorpusSpec":
        tuning = sum(scene.split == "tuning" for scene in self.scenes)
        holdout = sum(scene.split == "holdout" for scene in self.scenes)
        if tuning < 2 or holdout < 1:
            raise ValueError("corpus requires at least 2 tuning and 1 holdout scene")
        ids = [scene.scene_id for scene in self.scenes]
        if len(ids) != len(set(ids)):
            raise ValueError("scene ids must be unique")
        return self


class StylePack(BaseModel):
    style_pack_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    visual_language: str = Field(min_length=1, max_length=3000)
    component_direction: str = Field(min_length=1, max_length=2000)
    harmonization_direction: str = Field(min_length=1, max_length=2000)


class StyleLibrary(BaseModel):
    version: int = Field(ge=1)
    packs: list[StylePack] = Field(min_length=3)

    @model_validator(mode="after")
    def has_unique_ids(self) -> "StyleLibrary":
        ids = [pack.style_pack_id for pack in self.packs]
        if len(ids) != len(set(ids)):
            raise ValueError("style pack ids must be unique")
        return self


class ReviewRecord(BaseModel):
    scene_id: str
    seed: int = Field(ge=0)
    track: Literal["pixel", "masked", "global"]
    verdict: Literal["pass", "loser"]
    reasons: list[
        Literal[
            "identity",
            "subject_count",
            "scale",
            "pose",
            "anatomy",
            "prop",
            "matte",
            "lighting",
            "composition",
            "style",
            "expression",
            "continuity",
            "text_artifact",
            "other",
        ]
    ] = Field(default_factory=list)
    note: str = Field(default="", max_length=4000)
    reviewer: str = Field(default="human", min_length=1, max_length=100)
    artifact_sha256: str = Field(min_length=64, max_length=64)
    reviewed_at: str

    @model_validator(mode="after")
    def loser_has_reason(self) -> "ReviewRecord":
        if self.verdict == "loser" and not self.reasons:
            raise ValueError("loser verdict requires at least one reason")
        return self


@dataclass(frozen=True)
class ResolvedReference:
    reference_id: str
    source: Literal["reviewed", "candidate"]
    path: Path
    sha256: str
    mime_type: str
    width: int
    height: int
    byte_count: int

    def manifest_value(self, runtime_root: Path) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "source": self.source,
            "relative_path": self.path.relative_to(runtime_root).as_posix(),
            "sha256": self.sha256,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "byte_count": self.byte_count,
        }


class ReferenceResolver:
    """Read and verify immutable reviewed/candidate inputs without mutating DB state."""

    def __init__(self, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> None:
        self.runtime_root = runtime_root.resolve()
        self.artifacts_root = (self.runtime_root / "artifacts").resolve()
        self.database_path = (self.runtime_root / "jobs.sqlite").resolve()
        self._cache: dict[tuple[str, str], ResolvedReference] = {}

    def resolve(self, session_id: str, reference_id: str) -> ResolvedReference:
        cache_key = (session_id, reference_id)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not self.database_path.is_file():
            raise FileNotFoundError(
                f"image reference database missing: {self.database_path}"
            )

        uri = f"file:{self.database_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            row = connection.execute(
                "SELECT frozen_json FROM image_reviewed_references "
                "WHERE session_id = ? AND reference_id = ?",
                (session_id, reference_id),
            ).fetchone()
            source: Literal["reviewed", "candidate"] = "reviewed"
            if row is None:
                row = connection.execute(
                    "SELECT artifact_json FROM image_identity_candidates "
                    "WHERE session_id = ? AND candidate_id = ?",
                    (session_id, reference_id),
                ).fetchone()
                source = "candidate"
        if row is None:
            raise KeyError(
                f"reference {reference_id!r} is not frozen in session {session_id!r}"
            )

        payload = json.loads(row[0])
        relative_path = Path(str(payload["relative_path"]))
        if relative_path.is_absolute():
            raise ValueError(f"absolute artifact path rejected for {reference_id}")
        artifact_path = (self.runtime_root / relative_path).resolve()
        try:
            artifact_path.relative_to(self.artifacts_root)
        except ValueError as exc:
            raise ValueError(
                f"reference escapes artifact root: {reference_id}"
            ) from exc
        if not artifact_path.is_file():
            raise FileNotFoundError(f"frozen artifact missing: {artifact_path}")

        actual_sha = sha256_path(artifact_path)
        if actual_sha != payload["sha256"]:
            raise ValueError(f"frozen artifact hash mismatch: {reference_id}")
        actual_size = artifact_path.stat().st_size
        if actual_size != int(payload["byte_count"]):
            raise ValueError(f"frozen artifact byte count mismatch: {reference_id}")
        with Image.open(artifact_path) as image:
            dimensions = image.size
        if dimensions != (int(payload["width"]), int(payload["height"])):
            raise ValueError(f"frozen artifact dimensions mismatch: {reference_id}")

        resolved = ResolvedReference(
            reference_id=reference_id,
            source=source,
            path=artifact_path,
            sha256=actual_sha,
            mime_type=str(payload["mime_type"]),
            width=dimensions[0],
            height=dimensions[1],
            byte_count=actual_size,
        )
        self._cache[cache_key] = resolved
        return resolved


class GatewayError(RuntimeError):
    pass


class GatewayClient:
    def __init__(self, base_url: str, timeout_seconds: int = 1800) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise GatewayError(
                f"gateway health request failed: {type(exc).__name__}"
            ) from exc

    def post_image(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> tuple[bytes, dict[str, str]]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                content = response.read()
                headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                    if key.lower().startswith("x-ayoa-")
                }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise GatewayError(
                f"{endpoint} returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise GatewayError(
                f"{endpoint} request failed: {type(exc.reason).__name__}"
            ) from exc
        if not content:
            raise GatewayError(f"{endpoint} returned an empty image")
        return content, headers


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derived_seed(base_seed: int, *parts: str) -> int:
    material = ":".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def slug(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "_.-" else "-"
        for character in value
    ).strip("-.")
    return normalized[:90] or "image"


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def save_response_image(data: bytes, path: Path, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(io.BytesIO(data)) as received:
            received.load()
            image = received.convert(mode)
    except Exception as exc:
        raise GatewayError("gateway response was not a readable image") from exc
    image.save(path, format="PNG", optimize=True)


def artifact_metadata(path: Path, run_root: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return {
        "relative_path": path.relative_to(run_root).as_posix(),
        "sha256": sha256_path(path),
        "byte_count": path.stat().st_size,
        "width": width,
        "height": height,
        "mode": mode,
    }


def artifact_matches(record: Any, run_root: Path) -> bool:
    if not isinstance(record, dict):
        return False
    relative_path = record.get("relative_path")
    expected_sha256 = record.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        return False
    root = run_root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return False
    return sha256_path(path) == expected_sha256


def _significant_component_count(mask: Image.Image) -> int:
    sample = mask.convert("L")
    sample.thumbnail((256, 256), Image.Resampling.NEAREST)
    width, height = sample.size
    foreground = [value >= 32 for value in sample.getdata()]
    total = sum(foreground)
    minimum = max(8, int(total * 0.008))
    seen = bytearray(width * height)
    significant = 0
    for start, is_foreground in enumerate(foreground):
        if not is_foreground or seen[start]:
            continue
        stack = [start]
        seen[start] = 1
        size = 0
        while stack:
            index = stack.pop()
            size += 1
            x = index % width
            y = index // width
            for neighbor in (
                index - 1 if x else -1,
                index + 1 if x + 1 < width else -1,
                index - width if y else -1,
                index + width if y + 1 < height else -1,
            ):
                if neighbor >= 0 and foreground[neighbor] and not seen[neighbor]:
                    seen[neighbor] = 1
                    stack.append(neighbor)
        if size >= minimum:
            significant += 1
    return significant


def validate_matte(mask: Image.Image, policy: MaskPolicy) -> dict[str, Any]:
    grayscale = mask.convert("L")
    width, height = grayscale.size
    binary = grayscale.point(lambda value: 255 if value >= 32 else 0)
    histogram = binary.histogram()
    foreground = histogram[255]
    coverage = foreground / (width * height)
    if not policy.min_coverage <= coverage <= policy.max_coverage:
        raise ValueError(
            f"matte coverage {coverage:.4f} outside "
            f"[{policy.min_coverage:.4f}, {policy.max_coverage:.4f}]"
        )
    bbox = binary.getbbox()
    if bbox is None:
        raise ValueError("matte has no foreground")
    border = Image.new("L", (width, height), 0)
    border_draw = ImageDraw.Draw(border)
    border_draw.rectangle((0, 0, width - 1, height - 1), outline=255, width=2)
    edge_pixels = ImageChops.multiply(binary, border).histogram()[255]
    edge_fraction = edge_pixels / max(1, foreground)
    if edge_fraction > policy.max_edge_fraction:
        raise ValueError(
            f"matte edge fraction {edge_fraction:.4f} exceeds "
            f"{policy.max_edge_fraction:.4f}"
        )
    component_count = _significant_component_count(binary)
    if component_count > policy.max_significant_components:
        raise ValueError(
            f"matte has {component_count} significant components; "
            f"maximum is {policy.max_significant_components}"
        )
    return {
        "coverage": round(coverage, 6),
        "edge_fraction": round(edge_fraction, 6),
        "significant_components": component_count,
        "content_bbox": list(bbox),
    }


def create_rgba(component_path: Path, mask_path: Path, output_path: Path) -> None:
    with Image.open(component_path) as source_image:
        source = source_image.convert("RGB")
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
    if mask.size != source.size:
        mask = mask.resize(source.size, Image.Resampling.LANCZOS)
    rgba = source.convert("RGBA")
    rgba.putalpha(mask)
    rgba.save(output_path, format="PNG", optimize=True)


def _pixel_box(box: UnitBox, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(box.x * width),
        round(box.y * height),
        round((box.x + box.width) * width),
        round((box.y + box.height) * height),
    )


def _odd_filter_size(radius: int) -> int:
    return max(3, radius * 2 + 1)


def _subject_placement(
    scene: SceneSpec,
    subject: SubjectSpec,
) -> tuple[UnitBox, UnitPoint, int, Literal["contain", "fixed_height"]]:
    if scene.presentation == "visual_novel":
        stage_top = 0.06
        stage_height = subject.stage_height_fraction or 0.94
        stage_bottom = stage_top + stage_height
        if subject.screen_side == "left":
            return (
                UnitBox(x=0.0, y=stage_top, width=0.5, height=stage_height),
                UnitPoint(x=0.25, y=stage_bottom),
                2,
                "fixed_height",
            )
        if subject.screen_side == "right":
            return (
                UnitBox(x=0.5, y=stage_top, width=0.5, height=stage_height),
                UnitPoint(x=0.75, y=stage_bottom),
                2,
                "fixed_height",
            )
        raise ValueError(
            f"visual-novel subject has no screen side: {subject.character_id}"
        )
    if (
        subject.target_box is None
        or subject.foot_anchor is None
        or subject.z_index is None
    ):
        raise ValueError(
            f"cinematic subject has incomplete placement: {subject.character_id}"
        )
    return subject.target_box, subject.foot_anchor, subject.z_index, "contain"


def _sprite_component_path(
    case_directory: Path,
    character_id: str,
    variant_id: str = BASE_VARIANT_ID,
) -> Path:
    suffix = "" if variant_id == BASE_VARIANT_ID else f"--{slug(variant_id)}"
    return case_directory / "components" / f"{slug(character_id)}{suffix}.png"


def _sprite_matte_paths(
    case_directory: Path,
    character_id: str,
    variant_id: str = BASE_VARIANT_ID,
) -> tuple[Path, Path]:
    suffix = "" if variant_id == BASE_VARIANT_ID else f"--{slug(variant_id)}"
    stem = f"{slug(character_id)}{suffix}"
    matte_directory = case_directory / "mattes"
    return matte_directory / f"{stem}-mask.png", matte_directory / f"{stem}-rgba.png"


def _case_sprite_assets(
    case: CaseState,
) -> list[tuple[SubjectSpec, str, Path]]:
    assets: list[tuple[SubjectSpec, str, Path]] = []
    for subject in case.scene.subjects:
        assets.append(
            (
                subject,
                BASE_VARIANT_ID,
                _sprite_component_path(case.directory, subject.character_id),
            )
        )
        for variant in subject.variants:
            assets.append(
                (
                    subject,
                    variant.variant_id,
                    _sprite_component_path(
                        case.directory,
                        subject.character_id,
                        variant.variant_id,
                    ),
                )
            )
    return assets


def composite_subjects(
    scene: SceneSpec,
    background_path: Path,
    rgba_paths: dict[str, Path],
    output_path: Path,
    subject_mask_path: Path,
    repair_mask_path: Path,
) -> list[dict[str, Any]]:
    canvas_size = (scene.canvas.width, scene.canvas.height)
    with Image.open(background_path) as background_image:
        background = ImageOps.fit(
            background_image.convert("RGB"),
            canvas_size,
            method=Image.Resampling.LANCZOS,
        ).convert("RGBA")
    if scene.presentation == "visual_novel":
        background = background.filter(ImageFilter.GaussianBlur(1.25))
        background = Image.alpha_composite(
            background,
            Image.new("RGBA", canvas_size, (10, 12, 20, 28)),
        )
    combined_mask = Image.new("L", canvas_size, 0)
    placements: list[dict[str, Any]] = []

    for subject in sorted(
        scene.subjects,
        key=lambda item: _subject_placement(scene, item)[2],
    ):
        target_box, anchor, z_index, scale_mode = _subject_placement(scene, subject)
        with Image.open(rgba_paths[subject.character_id]) as component_image:
            component = component_image.convert("RGBA")
        alpha = component.getchannel("A")
        content_bbox = alpha.point(lambda value: 255 if value >= 16 else 0).getbbox()
        if content_bbox is None:
            raise ValueError(f"empty RGBA component for {subject.character_id}")
        component = component.crop(content_bbox)
        left, top, right, bottom = _pixel_box(
            target_box,
            scene.canvas.width,
            scene.canvas.height,
        )
        available_width = max(1, right - left)
        available_height = max(1, bottom - top)
        if scale_mode == "fixed_height":
            scale = available_height / component.height
            if round(component.width * scale) > available_width:
                raise ValueError(
                    f"visual-novel sprite is too wide for its fixed stage slot: "
                    f"{subject.character_id}"
                )
        else:
            scale = min(
                available_width / component.width,
                available_height / component.height,
            )
        resized_size = (
            max(1, round(component.width * scale)),
            max(1, round(component.height * scale)),
        )
        component = component.resize(resized_size, Image.Resampling.LANCZOS)
        anchor_x = round(anchor.x * scene.canvas.width)
        anchor_y = round(anchor.y * scene.canvas.height)
        paste_x = max(
            0,
            min(scene.canvas.width - component.width, anchor_x - component.width // 2),
        )
        paste_y = max(
            0, min(scene.canvas.height - component.height, anchor_y - component.height)
        )

        if scene.presentation == "visual_novel" and (
            paste_x < left or paste_x + component.width > right
        ):
            raise ValueError(
                f"visual-novel sprite escaped its fixed stage slot: "
                f"{subject.character_id}"
            )

        placed_alpha = Image.new("L", canvas_size, 0)
        placed_alpha.paste(component.getchannel("A"), (paste_x, paste_y))

        if scene.presentation == "visual_novel":
            shadow_alpha = Image.new("L", canvas_size, 0)
            shadow_alpha.paste(
                component.getchannel("A"),
                (paste_x + 6, paste_y + 8),
            )
            shadow_alpha = shadow_alpha.filter(ImageFilter.MaxFilter(7))
            shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(7))
            shadow_alpha = shadow_alpha.point(lambda value: round(value * 0.42))
            shadow = Image.new("RGBA", canvas_size, (8, 9, 16, 0))
            shadow.putalpha(shadow_alpha)
            background = Image.alpha_composite(background, shadow)
        elif subject.grounded:
            shadow = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(shadow)
            shadow_width = max(12, round(available_width * 0.55))
            shadow_height = max(5, round(scene.canvas.height * 0.018))
            draw.ellipse(
                (
                    anchor_x - shadow_width // 2,
                    anchor_y - shadow_height // 2,
                    anchor_x + shadow_width // 2,
                    anchor_y + shadow_height // 2,
                ),
                fill=(15, 12, 18, 90),
            )
            shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, shadow_height // 2)))
            background = Image.alpha_composite(background, shadow)

        if scene.presentation == "cinematic":
            wrap_radius = max(1, round(min(canvas_size) * 0.003))
            outer = placed_alpha.filter(
                ImageFilter.MaxFilter(_odd_filter_size(wrap_radius))
            )
            outer = ImageChops.subtract(outer, placed_alpha).point(
                lambda value: value // 3
            )
            light_wrap = background.filter(ImageFilter.GaussianBlur(5))
            background = Image.composite(light_wrap, background, outer)

        layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        layer.alpha_composite(component, (paste_x, paste_y))
        background = Image.alpha_composite(background, layer)
        combined_mask = ImageChops.lighter(combined_mask, placed_alpha)
        placements.append(
            {
                "character_id": subject.character_id,
                "source_content_bbox": list(content_bbox),
                "target_box_pixels": [left, top, right, bottom],
                "paste_box_pixels": [
                    paste_x,
                    paste_y,
                    paste_x + component.width,
                    paste_y + component.height,
                ],
                "scale": round(scale, 6),
                "scale_mode": scale_mode,
                "screen_side": subject.screen_side,
                "z_index": z_index,
            }
        )

    seam_radius = max(2, round(min(canvas_size) * 0.006))
    dilated = combined_mask.filter(ImageFilter.MaxFilter(_odd_filter_size(seam_radius)))
    eroded = combined_mask.filter(ImageFilter.MinFilter(_odd_filter_size(seam_radius)))
    seam_mask = ImageChops.subtract(dilated, eroded)
    repair_mask = seam_mask.copy()
    contact = Image.new("L", canvas_size, 0)
    contact_draw = ImageDraw.Draw(contact)
    for region in scene.contact_regions:
        if region.kind == "ellipse" and region.box is not None:
            contact_draw.ellipse(
                _pixel_box(region.box, *canvas_size),
                fill=255,
            )
        elif region.kind == "polygon":
            contact_draw.polygon(
                [
                    (round(point.x * canvas_size[0]), round(point.y * canvas_size[1]))
                    for point in region.points
                ],
                fill=255,
            )
    repair_mask = ImageChops.lighter(repair_mask, contact)
    repair_mask = repair_mask.filter(ImageFilter.GaussianBlur(max(1, seam_radius // 2)))

    interior = combined_mask.filter(
        ImageFilter.MinFilter(_odd_filter_size(seam_radius * 2))
    )
    interior_overlap = ImageChops.multiply(seam_mask, interior)
    overlap_histogram = interior_overlap.histogram()
    overlap_pixels = sum(overlap_histogram[1:])
    interior_pixels = sum(interior.histogram()[1:])
    if interior_pixels and overlap_pixels / interior_pixels > 0.12:
        raise ValueError("repair mask covers too much of subject interiors")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.convert("RGB").save(output_path, format="PNG", optimize=True)
    combined_mask.save(subject_mask_path, format="PNG", optimize=True)
    repair_mask.save(repair_mask_path, format="PNG", optimize=True)
    return placements


def build_storyboard(
    frames: Sequence[tuple[str, Path]],
    output_path: Path,
) -> None:
    if not frames:
        raise ValueError("storyboard requires at least one frame")
    thumbnail_size = (512, 288)
    label_height = 34
    gutter = 12
    columns = min(3, len(frames))
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * thumbnail_size[0] + (columns + 1) * gutter,
            rows * (thumbnail_size[1] + label_height) + (rows + 1) * gutter,
        ),
        (19, 19, 25),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (title, path) in enumerate(frames):
        column = index % columns
        row = index // columns
        x = gutter + column * (thumbnail_size[0] + gutter)
        y = gutter + row * (thumbnail_size[1] + label_height + gutter)
        with Image.open(path) as frame_image:
            thumbnail = ImageOps.fit(
                frame_image.convert("RGB"),
                thumbnail_size,
                method=Image.Resampling.LANCZOS,
            )
        sheet.paste(thumbnail, (x, y))
        draw.text(
            (x + 8, y + thumbnail_size[1] + 8),
            title,
            fill=(232, 232, 238),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)


def blend_masked_result(
    original_path: Path,
    generated_path: Path,
    mask_path: Path,
    output_path: Path,
) -> int:
    with Image.open(original_path) as original_image:
        original = original_image.convert("RGB")
    with Image.open(generated_path) as generated_image:
        generated = generated_image.convert("RGB").resize(
            original.size,
            Image.Resampling.LANCZOS,
        )
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L").resize(original.size, Image.Resampling.LANCZOS)
    blended = Image.composite(generated, original, mask)
    blended.save(output_path, format="PNG", optimize=True)

    original_pixels = original.load()
    blended_pixels = blended.load()
    mask_pixels = mask.load()
    changed_outside = 0
    for y in range(original.height):
        for x in range(original.width):
            if mask_pixels[x, y] == 0 and original_pixels[x, y] != blended_pixels[x, y]:
                changed_outside += 1
    if changed_outside:
        raise ValueError(
            f"masked blend changed {changed_outside} pixels outside repair mask"
        )
    return changed_outside


def load_experiment(
    corpus_path: Path,
    style_path: Path,
) -> tuple[CorpusSpec, dict[str, StylePack]]:
    corpus = CorpusSpec.model_validate(json_load(corpus_path))
    library = StyleLibrary.model_validate(json_load(style_path))
    styles = {pack.style_pack_id: pack for pack in library.packs}
    missing = sorted(
        {scene.style_pack_id for scene in corpus.scenes}.difference(styles)
    )
    if missing:
        raise ValueError(f"corpus uses unknown style packs: {', '.join(missing)}")
    return corpus, styles


@dataclass
class CaseState:
    scene: SceneSpec
    style: StylePack
    seed: int
    directory: Path
    manifest_path: Path
    manifest: dict[str, Any]
    references: dict[str, list[ResolvedReference]]


def prepare_component_reference_inputs(
    case: CaseState,
    subject: SubjectSpec,
    run_root: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for index, reference in enumerate(case.references[subject.character_id], start=1):
        operation = "none"
        path = reference.path
        record: dict[str, Any] = {
            "reference_id": reference.reference_id,
            "operation": operation,
            "source_sha256": reference.sha256,
        }
        if subject.facing_control == "mirror_reference":
            operation = "mirror_horizontal"
            path = (
                case.directory
                / "inputs"
                / f"{slug(subject.character_id)}-reference-{index}-mirrored.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(reference.path) as source:
                ImageOps.mirror(source.convert("RGB")).save(
                    path,
                    format="PNG",
                    optimize=True,
                )
            record["operation"] = operation
            record["artifact"] = artifact_metadata(path, run_root)
        paths.append(path)
        records.append(record)
    return paths, records


class StagedImageRunner:
    def __init__(
        self,
        *,
        corpus_path: Path = DEFAULT_CORPUS_PATH,
        style_path: Path = DEFAULT_STYLE_PATH,
        runtime_root: Path = DEFAULT_RUNTIME_ROOT,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        gateway_url: str = DEFAULT_GATEWAY_URL,
        max_workers: int = 4,
        worker_indices: Sequence[int] | None = None,
    ) -> None:
        if not 1 <= max_workers <= 4:
            raise ValueError("max_workers must be between 1 and 4")
        if worker_indices is not None and (
            not worker_indices
            or len(set(worker_indices)) != len(worker_indices)
            or any(index < 0 or index > 3 for index in worker_indices)
        ):
            raise ValueError("worker_indices must be unique values between 0 and 3")
        self.corpus_path = corpus_path.resolve()
        self.style_path = style_path.resolve()
        self.output_root = output_root.resolve()
        self.corpus, self.styles = load_experiment(
            self.corpus_path,
            self.style_path,
        )
        self.resolver = ReferenceResolver(runtime_root)
        self.client = GatewayClient(gateway_url)
        self.max_workers = max_workers
        self.worker_indices = (
            tuple(worker_indices) if worker_indices is not None else None
        )
        self.worker_targets: tuple[str, ...] = ()

    def _worker_target(self, job_index: int) -> str:
        targets = getattr(self, "worker_targets", ())
        if not targets:
            return ""
        return targets[job_index % len(targets)]

    def select_scenes(
        self,
        scene_ids: Sequence[str],
        *,
        include_holdout: bool = False,
    ) -> list[SceneSpec]:
        if not scene_ids or scene_ids == ["all"]:
            return [
                scene
                for scene in self.corpus.scenes
                if include_holdout or scene.split != "holdout"
            ]
        requested = set(scene_ids)
        selected = [
            scene for scene in self.corpus.scenes if scene.scene_id in requested
        ]
        missing = requested.difference(scene.scene_id for scene in selected)
        if missing:
            raise KeyError(f"unknown scenes: {', '.join(sorted(missing))}")
        forbidden = [scene.scene_id for scene in selected if scene.split == "holdout"]
        if forbidden and not include_holdout:
            raise ValueError(
                "holdout scenes require --include-holdout: "
                + ", ".join(sorted(forbidden))
            )
        return selected

    def validate_inputs(self, scenes: Sequence[SceneSpec]) -> dict[str, Any]:
        references: dict[str, dict[str, Any]] = {}
        for scene in scenes:
            for subject in scene.subjects:
                for reference_id in subject.reference_ids:
                    resolved = self.resolver.resolve(
                        self.corpus.reference_session_id,
                        reference_id,
                    )
                    references[reference_id] = resolved.manifest_value(
                        self.resolver.runtime_root
                    )
        return {
            "ok": True,
            "scene_ids": [scene.scene_id for scene in scenes],
            "reference_session_id": self.corpus.reference_session_id,
            "references": references,
        }

    def run(
        self,
        *,
        scene_ids: Sequence[str],
        seeds: Sequence[int],
        tracks: Sequence[str],
        run_id: str | None = None,
        dry_run: bool = False,
        resume: bool = False,
        reuse_backgrounds_from: str | None = None,
        include_holdout: bool = False,
        style_pack_id: str | None = None,
    ) -> Path | dict[str, Any]:
        scenes = self.select_scenes(
            scene_ids,
            include_holdout=include_holdout,
        )
        if len(seeds) != 4 or len(set(seeds)) != 4:
            raise ValueError("a run requires exactly four distinct seeds")
        if any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative")
        selected_tracks = tuple(dict.fromkeys(tracks))
        if "pixel" not in selected_tracks:
            raise ValueError("the deterministic pixel track is mandatory")
        unknown_tracks = set(selected_tracks).difference(TRACKS)
        if unknown_tracks:
            raise ValueError(f"unknown tracks: {', '.join(sorted(unknown_tracks))}")
        if any(scene.presentation == "visual_novel" for scene in scenes) and set(
            selected_tracks
        ) != {"pixel"}:
            raise ValueError(
                "visual-novel scenes support only deterministic pixel frames"
            )
        if style_pack_id is not None and style_pack_id not in self.styles:
            raise KeyError(f"unknown style pack: {style_pack_id}")
        validation = self.validate_inputs(scenes)
        if dry_run:
            if resume:
                raise ValueError("--resume cannot be combined with --dry-run")
            return {
                **validation,
                "seeds": list(seeds),
                "tracks": list(selected_tracks),
                "style_pack_override": style_pack_id,
                "background_source_run_id": reuse_backgrounds_from,
            }

        gateway_health = self.client.health()
        pipelines = gateway_health.get("pipelines", {})
        if not gateway_health.get("ok") or not pipelines.get("compose", {}).get(
            "available"
        ):
            raise GatewayError("gateway compose pipeline is unavailable")
        if not pipelines.get("edit", {}).get("available"):
            raise GatewayError("gateway edit pipeline is unavailable")
        healthy_workers = [
            worker
            for worker in gateway_health.get("workers", [])
            if worker.get("ok") and worker.get("base")
        ]
        if self.worker_indices is None:
            selected_workers = healthy_workers
        else:
            advertised_workers = list(gateway_health.get("workers", []))
            selected_workers = []
            for index in self.worker_indices:
                if index >= len(advertised_workers) or not advertised_workers[
                    index
                ].get("ok"):
                    raise GatewayError(f"requested image worker {index} is unavailable")
                selected_workers.append(advertised_workers[index])
        self.worker_targets = tuple(str(worker["base"]) for worker in selected_workers)
        if self.max_workers > len(self.worker_targets):
            raise GatewayError(
                "execution concurrency exceeds the selected healthy worker count"
            )
        execution_attempt = {
            "started_at": utc_now(),
            "max_workers": self.max_workers,
            "worker_targets": list(self.worker_targets),
        }

        if resume and run_id is None:
            raise ValueError("--resume requires --run-id")
        run_id = run_id or datetime.now(UTC).strftime("staged-%Y%m%dT%H%M%SZ")
        if slug(run_id) != run_id:
            raise ValueError(
                "run id may contain only letters, digits, dot, dash, underscore"
            )
        run_root = self.output_root / run_id
        run_manifest_path = run_root / "run.json"
        protocol_versions = {
            scene.presentation: PROMPT_PROTOCOL_VERSIONS[scene.presentation]
            for scene in scenes
        }
        run_protocol_version = (
            next(iter(protocol_versions.values()))
            if len(protocol_versions) == 1
            else "mixed"
        )
        if resume:
            if not run_root.is_dir() or not run_manifest_path.is_file():
                raise FileNotFoundError(f"run directory is not resumable: {run_root}")
            run_manifest = json_load(run_manifest_path)
            self._validate_resume_manifest(
                run_manifest=run_manifest,
                run_id=run_id,
                scenes=scenes,
                seeds=seeds,
                tracks=selected_tracks,
                include_holdout=include_holdout,
                style_pack_id=style_pack_id,
                protocol_versions=protocol_versions,
                reuse_backgrounds_from=reuse_backgrounds_from,
            )
            cases = self._load_cases(
                run_root,
                scenes,
                seeds,
                selected_tracks,
                style_pack_id,
            )
            resumed_at = utc_now()
            run_manifest.setdefault("resume_events", []).append(
                {
                    "resumed_at": resumed_at,
                    "previous_status": run_manifest.get("status"),
                    "previous_error": run_manifest.get("error"),
                }
            )
            run_manifest["status"] = "running"
            run_manifest["gateway_health"] = gateway_health
            run_manifest.setdefault("execution_attempts", []).append(execution_attempt)
            run_manifest.pop("failed_at", None)
            run_manifest.pop("error", None)
            for case in cases:
                case.manifest.setdefault("resume_events", []).append(
                    {
                        "resumed_at": resumed_at,
                        "previous_status": case.manifest.get("status"),
                        "previous_error": case.manifest.get("run_error"),
                    }
                )
                case.manifest["status"] = "running"
                case.manifest.pop("failed_at", None)
                case.manifest.pop("run_error", None)
                write_json(case.manifest_path, case.manifest)
            write_json(run_manifest_path, run_manifest)
        else:
            if run_root.exists():
                raise FileExistsError(f"run directory already exists: {run_root}")
            run_root.mkdir(parents=True)
            run_manifest = {
                "schema_version": 2,
                "run_id": run_id,
                "status": "running",
                "started_at": utc_now(),
                "corpus": {
                    "path": self.corpus_path.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_path(self.corpus_path),
                },
                "styles": {
                    "path": self.style_path.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_path(self.style_path),
                },
                "reference_session_id": self.corpus.reference_session_id,
                "scene_ids": [scene.scene_id for scene in scenes],
                "seeds": list(seeds),
                "tracks": list(selected_tracks),
                "style_pack_override": style_pack_id,
                "background_source_run_id": reuse_backgrounds_from,
                "holdouts_included": include_holdout,
                "batch_contract": {
                    "exact_seed_count": 4,
                    "winner_selection": False,
                    "fallback_generation": False,
                },
                "prompt_protocol_version": run_protocol_version,
                "prompt_protocol_versions": protocol_versions,
                "gateway_health": gateway_health,
                "execution_attempts": [execution_attempt],
                "cases": [],
            }
            write_json(run_manifest_path, run_manifest)

            cases = self._initialize_cases(
                run_root,
                scenes,
                seeds,
                selected_tracks,
                style_pack_id,
            )
            run_manifest["cases"] = [
                case.manifest_path.relative_to(run_root).as_posix() for case in cases
            ]
            write_json(run_manifest_path, run_manifest)

        try:
            if reuse_backgrounds_from is not None:
                source_root = self.output_root / reuse_backgrounds_from
                if source_root.resolve() == run_root.resolve():
                    raise ValueError("a run cannot reuse its own backgrounds")
                self._reuse_backgrounds(
                    cases,
                    run_root,
                    source_root,
                    reuse_backgrounds_from,
                )
            self._background_stage(cases, run_root)
            self._component_stage(cases, run_root)
            self._variant_stage(cases, run_root)
            self._matte_stage(cases, run_root)
            self._pixel_stage(cases, run_root)
            self._final_stage(cases, run_root, selected_tracks)
        except BaseException as exc:
            for case in cases:
                case.manifest["status"] = (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                )
                case.manifest["failed_at"] = utc_now()
                case.manifest["run_error"] = f"{type(exc).__name__}: {exc}"
                write_json(case.manifest_path, case.manifest)
            run_manifest["status"] = (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
            run_manifest["failed_at"] = utc_now()
            run_manifest["error"] = f"{type(exc).__name__}: {exc}"
            write_json(run_manifest_path, run_manifest)
            raise

        for case in cases:
            case.manifest["status"] = "complete"
            case.manifest["completed_at"] = utc_now()
            write_json(case.manifest_path, case.manifest)
        run_manifest["status"] = "complete"
        run_manifest["completed_at"] = utc_now()
        write_json(run_manifest_path, run_manifest)
        return run_root

    def _validate_resume_manifest(
        self,
        *,
        run_manifest: dict[str, Any],
        run_id: str,
        scenes: Sequence[SceneSpec],
        seeds: Sequence[int],
        tracks: Sequence[str],
        include_holdout: bool,
        style_pack_id: str | None,
        protocol_versions: dict[str, str],
        reuse_backgrounds_from: str | None,
    ) -> None:
        if run_manifest.get("status") == "complete":
            raise ValueError("a complete run cannot be resumed")
        expected: dict[str, Any] = {
            "run_id": run_id,
            "scene_ids": [scene.scene_id for scene in scenes],
            "seeds": list(seeds),
            "tracks": list(tracks),
            "style_pack_override": style_pack_id,
            "background_source_run_id": reuse_backgrounds_from,
            "holdouts_included": include_holdout,
            "reference_session_id": self.corpus.reference_session_id,
            "prompt_protocol_versions": protocol_versions,
        }
        mismatches = [
            key for key, value in expected.items() if run_manifest.get(key) != value
        ]
        if run_manifest.get("corpus", {}).get("sha256") != sha256_path(
            self.corpus_path
        ):
            mismatches.append("corpus.sha256")
        if run_manifest.get("styles", {}).get("sha256") != sha256_path(self.style_path):
            mismatches.append("styles.sha256")
        expected_cases = [
            f"{scene.scene_id}/seed-{seed}/manifest.json"
            for scene in scenes
            for seed in seeds
        ]
        if run_manifest.get("cases") != expected_cases:
            mismatches.append("cases")
        if mismatches:
            raise ValueError(
                "resume arguments or source artifacts differ from the frozen run: "
                + ", ".join(sorted(set(mismatches)))
            )

    def _load_cases(
        self,
        run_root: Path,
        scenes: Sequence[SceneSpec],
        seeds: Sequence[int],
        tracks: Sequence[str],
        style_pack_id: str | None,
    ) -> list[CaseState]:
        cases: list[CaseState] = []
        for scene in scenes:
            style = self.styles[style_pack_id or scene.style_pack_id]
            for seed in seeds:
                directory = run_root / scene.scene_id / f"seed-{seed}"
                manifest_path = directory / "manifest.json"
                if not manifest_path.is_file():
                    raise FileNotFoundError(
                        f"resume case manifest is missing: {manifest_path}"
                    )
                manifest = json_load(manifest_path)
                resolved_by_subject: dict[str, list[ResolvedReference]] = {}
                reference_manifest: dict[str, list[dict[str, Any]]] = {}
                for subject in scene.subjects:
                    resolved = [
                        self.resolver.resolve(
                            self.corpus.reference_session_id,
                            reference_id,
                        )
                        for reference_id in subject.reference_ids
                    ]
                    resolved_by_subject[subject.character_id] = resolved
                    reference_manifest[subject.character_id] = [
                        item.manifest_value(self.resolver.runtime_root)
                        for item in resolved
                    ]
                expected_values = {
                    "scene_id": scene.scene_id,
                    "seed": seed,
                    "tracks": list(tracks),
                    "scene_spec_sha256": sha256_text(
                        scene.model_dump_json(exclude_none=True)
                    ),
                    "style_pack_sha256": sha256_text(style.model_dump_json()),
                    "prompt_protocol_version": PROMPT_PROTOCOL_VERSIONS[
                        scene.presentation
                    ],
                    "references": reference_manifest,
                }
                mismatches = [
                    key
                    for key, value in expected_values.items()
                    if manifest.get(key) != value
                ]
                if mismatches:
                    raise ValueError(
                        f"resume case differs from frozen inputs ({scene.scene_id}/"
                        f"{seed}): {', '.join(sorted(mismatches))}"
                    )
                cases.append(
                    CaseState(
                        scene=scene,
                        style=style,
                        seed=seed,
                        directory=directory,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        references=resolved_by_subject,
                    )
                )
        return cases

    def _initialize_cases(
        self,
        run_root: Path,
        scenes: Sequence[SceneSpec],
        seeds: Sequence[int],
        tracks: Sequence[str],
        style_pack_id: str | None,
    ) -> list[CaseState]:
        cases: list[CaseState] = []
        for scene in scenes:
            style = self.styles[style_pack_id or scene.style_pack_id]
            for seed in seeds:
                directory = run_root / scene.scene_id / f"seed-{seed}"
                directory.mkdir(parents=True)
                resolved_by_subject: dict[str, list[ResolvedReference]] = {}
                reference_manifest: dict[str, list[dict[str, Any]]] = {}
                for subject in scene.subjects:
                    resolved = [
                        self.resolver.resolve(
                            self.corpus.reference_session_id,
                            reference_id,
                        )
                        for reference_id in subject.reference_ids
                    ]
                    resolved_by_subject[subject.character_id] = resolved
                    reference_manifest[subject.character_id] = [
                        item.manifest_value(self.resolver.runtime_root)
                        for item in resolved
                    ]
                manifest_path = directory / "manifest.json"
                manifest: dict[str, Any] = {
                    "schema_version": 2,
                    "status": "running",
                    "scene_id": scene.scene_id,
                    "scene_title": scene.title,
                    "split": scene.split,
                    "seed": seed,
                    "tracks": list(tracks),
                    "scene_spec": scene.model_dump(mode="json"),
                    "scene_spec_sha256": sha256_text(
                        scene.model_dump_json(exclude_none=True)
                    ),
                    "style_pack": style.model_dump(mode="json"),
                    "style_pack_sha256": sha256_text(style.model_dump_json()),
                    "prompt_protocol_version": PROMPT_PROTOCOL_VERSIONS[
                        scene.presentation
                    ],
                    "references": reference_manifest,
                    "stages": {},
                    "started_at": utc_now(),
                }
                write_json(manifest_path, manifest)
                cases.append(
                    CaseState(
                        scene=scene,
                        style=style,
                        seed=seed,
                        directory=directory,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        references=resolved_by_subject,
                    )
                )
        return cases

    @staticmethod
    def _flush_cases(cases: Sequence[CaseState]) -> None:
        for case in cases:
            write_json(case.manifest_path, case.manifest)

    def _reuse_backgrounds(
        self,
        cases: Sequence[CaseState],
        run_root: Path,
        source_root: Path,
        source_run_id: str,
    ) -> None:
        if not source_root.is_dir() or not (source_root / "run.json").is_file():
            raise FileNotFoundError(
                f"background source run does not exist: {source_root}"
            )
        for case in cases:
            prompt = self._background_prompt(case)
            stage_seed = derived_seed(case.seed, case.scene.scene_id, "background")
            existing = case.manifest.get("stages", {}).get("background", {})
            if (
                existing.get("prompt_sha256") == sha256_text(prompt)
                and existing.get("seed") == stage_seed
                and artifact_matches(existing.get("artifact"), run_root)
            ):
                continue
            source_manifest_path = (
                source_root
                / case.scene.scene_id
                / f"seed-{case.seed}"
                / "manifest.json"
            )
            if not source_manifest_path.is_file():
                raise FileNotFoundError(
                    f"source background case is missing: {source_manifest_path}"
                )
            source_manifest = json_load(source_manifest_path)
            source_stage = source_manifest.get("stages", {}).get("background", {})
            if (
                source_manifest.get("scene_id") != case.scene.scene_id
                or source_manifest.get("seed") != case.seed
            ):
                raise ValueError(
                    f"source background case identity mismatch: {source_manifest_path}"
                )
            if (
                source_stage.get("prompt_sha256") != sha256_text(prompt)
                or source_stage.get("seed") != stage_seed
                or not artifact_matches(source_stage.get("artifact"), source_root)
            ):
                raise ValueError(
                    "source background does not match the current plate contract: "
                    f"{source_manifest_path}"
                )
            source_artifact = source_stage["artifact"]
            source_path = (
                source_root / str(source_artifact["relative_path"])
            ).resolve()
            destination = case.directory / "background.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            shutil.copy2(source_path, destination)
            destination_artifact = artifact_metadata(destination, run_root)
            if destination_artifact["sha256"] != source_artifact["sha256"]:
                raise ValueError(
                    f"reused background hash changed while copying: {destination}"
                )
            case.manifest["stages"]["background"] = {
                "operation": "frozen background reuse",
                "source_run_id": source_run_id,
                "source_case_manifest": source_manifest_path.relative_to(
                    source_root
                ).as_posix(),
                "source_artifact_sha256": source_artifact["sha256"],
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "seed": stage_seed,
                "duration_seconds": round(time.monotonic() - started, 3),
                "response_headers": {},
                "artifact": destination_artifact,
            }
            write_json(case.manifest_path, case.manifest)

    def _background_stage(self, cases: Sequence[CaseState], run_root: Path) -> None:
        print(f"background stage: {len(cases)} jobs", file=sys.stderr, flush=True)
        for index, case in enumerate(cases, start=1):
            prompt = self._background_prompt(case)
            stage_seed = derived_seed(case.seed, case.scene.scene_id, "background")
            existing = case.manifest.get("stages", {}).get("background", {})
            if (
                existing.get("prompt_sha256") == sha256_text(prompt)
                and existing.get("seed") == stage_seed
                and artifact_matches(existing.get("artifact"), run_root)
            ):
                print(
                    f"background {index}/{len(cases)} reused: "
                    f"{case.scene.scene_id} {case.seed}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            started = time.monotonic()
            data, headers = self.client.post_image(
                "/generate",
                {
                    "prompt": prompt,
                    "width": case.scene.canvas.width,
                    "height": case.scene.canvas.height,
                    "steps": 28,
                    "guidance": 4.0,
                    "seed": stage_seed,
                    "reference_images": [],
                },
            )
            path = case.directory / "background.png"
            save_response_image(data, path)
            case.manifest["stages"]["background"] = {
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "seed": stage_seed,
                "duration_seconds": round(time.monotonic() - started, 3),
                "response_headers": headers,
                "artifact": artifact_metadata(path, run_root),
            }
            write_json(case.manifest_path, case.manifest)
            print(
                f"background {index}/{len(cases)}: {case.scene.scene_id} {case.seed}",
                file=sys.stderr,
                flush=True,
            )

    @staticmethod
    def _background_prompt(case: CaseState) -> str:
        if case.scene.presentation == "visual_novel":
            composition = (
                "Composition: stable widescreen visual-novel stage, quiet lateral "
                "balance, uncluttered lower frame, and a readable center gap."
            )
        else:
            composition = (
                "Composition: full-bleed setting with an open, unoccupied "
                "foreground and middle distance."
            )
        return "\n".join(
            (
                "Empty environment plate.",
                f"Scene: {case.scene.background_prompt}",
                f"Camera: {case.scene.camera.description}",
                f"Style: {case.style.visual_language}",
                composition,
            )
        )

    def _component_stage(self, cases: Sequence[CaseState], run_root: Path) -> None:
        concurrency = getattr(self, "max_workers", 4)

        def is_complete(case: CaseState, subject: SubjectSpec) -> bool:
            record = (
                case.manifest.get("stages", {})
                .get("components", {})
                .get(subject.character_id, {})
            )
            if record.get("component_mode") != subject.component_mode:
                return False
            if record.get("reference_order") != subject.reference_ids:
                return False
            if subject.component_mode == "qwen_edit":
                prompt = self._component_prompt(case, subject)
                expected_seed = derived_seed(
                    case.seed,
                    case.scene.scene_id,
                    "component",
                    subject.character_id,
                )
                if record.get("prompt_sha256") != sha256_text(prompt):
                    return False
                if record.get("seed") != expected_seed:
                    return False
            return artifact_matches(record.get("artifact"), run_root)

        job_count = sum(
            subject.component_mode == "qwen_edit" and not is_complete(case, subject)
            for case in cases
            for subject in case.scene.subjects
        )
        cutout_count = sum(
            subject.component_mode == "reference_cutout"
            and not is_complete(case, subject)
            for case in cases
            for subject in case.scene.subjects
        )
        print(
            f"component stage: {job_count} Qwen jobs at "
            f"concurrency {concurrency}, "
            f"{cutout_count} reviewed cutouts",
            file=sys.stderr,
            flush=True,
        )
        futures: dict[
            Future[Any],
            tuple[CaseState, SubjectSpec, str, int, float, list[dict[str, Any]]],
        ] = {}
        request_index = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for case in cases:
                for subject in case.scene.subjects:
                    reference_input_paths, reference_input_records = (
                        prepare_component_reference_inputs(
                            case,
                            subject,
                            run_root,
                        )
                    )
                    if is_complete(case, subject):
                        continue
                    if subject.component_mode == "reference_cutout":
                        started = time.monotonic()
                        reference = case.references[subject.character_id][0]
                        path = _sprite_component_path(
                            case.directory,
                            subject.character_id,
                        )
                        save_response_image(reference_input_paths[0].read_bytes(), path)
                        component_stages = case.manifest["stages"].setdefault(
                            "components", {}
                        )
                        component_stages[subject.character_id] = {
                            "component_mode": "reference_cutout",
                            "prompt": None,
                            "prompt_sha256": None,
                            "seed": None,
                            "reference_order": [reference.reference_id],
                            "reference_inputs": reference_input_records,
                            "duration_seconds": round(time.monotonic() - started, 3),
                            "response_headers": {},
                            "artifact": artifact_metadata(path, run_root),
                        }
                        write_json(case.manifest_path, case.manifest)
                        continue
                    prompt = self._component_prompt(case, subject)
                    stage_seed = derived_seed(
                        case.seed,
                        case.scene.scene_id,
                        "component",
                        subject.character_id,
                    )
                    references = case.references[subject.character_id]
                    payload: dict[str, Any] = {
                        "prompt": prompt,
                        "image_base64": encode_image(reference_input_paths[0]),
                        "seed": stage_seed,
                        "steps": 28,
                        "cfg": 4.0,
                        "filename_prefix": slug(
                            f"staged_{case.scene.scene_id}_{subject.character_id}"
                        ),
                    }
                    if len(references) == 2:
                        payload["image2_base64"] = encode_image(
                            reference_input_paths[1]
                        )
                    worker = self._worker_target(request_index)
                    request_index += 1
                    if worker:
                        payload["worker"] = worker
                    started = time.monotonic()
                    future = executor.submit(
                        self.client.post_image,
                        "/edit/qwen",
                        payload,
                    )
                    futures[future] = (
                        case,
                        subject,
                        prompt,
                        stage_seed,
                        started,
                        reference_input_records,
                    )
            failures: list[str] = []
            for future in as_completed(futures):
                (
                    case,
                    subject,
                    prompt,
                    stage_seed,
                    started,
                    reference_input_records,
                ) = futures[future]
                try:
                    data, headers = future.result()
                    path = _sprite_component_path(
                        case.directory,
                        subject.character_id,
                    )
                    save_response_image(data, path)
                    component_stages = case.manifest["stages"].setdefault(
                        "components", {}
                    )
                    component_stages[subject.character_id] = {
                        "component_mode": "qwen_edit",
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "seed": stage_seed,
                        "reference_order": [
                            item.reference_id
                            for item in case.references[subject.character_id]
                        ],
                        "reference_inputs": reference_input_records,
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "response_headers": headers,
                        "artifact": artifact_metadata(path, run_root),
                    }
                    write_json(case.manifest_path, case.manifest)
                except Exception as exc:
                    failures.append(
                        f"{case.scene.scene_id}/{case.seed}/{subject.character_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if failures:
            raise RuntimeError("component stage failed:\n" + "\n".join(failures))

    @staticmethod
    def _component_prompt(case: CaseState, subject: SubjectSpec) -> str:
        reference_count = len(case.references[subject.character_id])
        if reference_count == 2:
            mapping = (
                f"Input images: Images 1 and 2 are two identity views of the same "
                f"character, {subject.display_name}."
            )
        else:
            mapping = (
                f"Input image: Image 1 is the identity reference for "
                f"{subject.display_name}."
            )
        if case.scene.presentation == "visual_novel":
            if subject.facing_control == "prompt":
                orientation = (
                    f"Orientation: Place the sprite for the {subject.screen_side} "
                    f"side of the screen, with body and face turned slightly toward "
                    f"screen {subject.facing}; keep the face readable to the viewer."
                )
            else:
                orientation = (
                    f"Orientation: Image 1 has already been prepared to face screen "
                    f"{subject.facing} from the {subject.screen_side} stage slot. "
                    "Preserve its body and face direction exactly; do not turn or "
                    "mirror the character."
                )
            return "\n".join(
                (
                    mapping,
                    f"Primary edit: Create the neutral reusable visual-novel sprite "
                    f"anchor for {subject.display_name} on a plain neutral gray "
                    "studio backdrop.",
                    orientation,
                    f"Neutral pose: {subject.pose_prompt}",
                    f"Identity to preserve: {subject.identity_prompt}",
                    f"Style: {case.style.visual_language} {case.style.component_direction}",
                    f"Composition: Only {subject.display_name} appears. Follow the "
                    "framing specified by the neutral pose, with comfortable neutral "
                    "margin around the complete visible silhouette and any equipment.",
                    "Continuity: Neutral conversational expression and restrained "
                    "gesture; avoid dramatic action, foreshortening, or camera tilt.",
                )
            )
        return "\n".join(
            (
                mapping,
                f"Primary edit: Create a single isolated full-body asset of "
                f"{subject.display_name} on a plain neutral gray studio backdrop.",
                f"Pose: {subject.pose_prompt}",
                f"Identity to preserve: {subject.identity_prompt}",
                f"Camera: {case.scene.camera.description}",
                f"Style: {case.style.visual_language} {case.style.component_direction}",
                f"Composition: Only {subject.display_name} appears; show the complete "
                "head-to-feet silhouette and every held prop with clear margin.",
            )
        )

    def _variant_stage(self, cases: Sequence[CaseState], run_root: Path) -> None:
        concurrency = getattr(self, "max_workers", 4)

        def is_complete(
            case: CaseState,
            subject: SubjectSpec,
            variant: SpriteVariantSpec,
        ) -> bool:
            record = (
                case.manifest.get("stages", {})
                .get("variants", {})
                .get(subject.character_id, {})
                .get(variant.variant_id, {})
            )
            anchor_path = _sprite_component_path(
                case.directory,
                subject.character_id,
            )
            if not anchor_path.is_file():
                return False
            prompt = self._variant_prompt(case, subject, variant)
            expected_seed = derived_seed(
                case.seed,
                case.scene.scene_id,
                "variant",
                subject.character_id,
                variant.variant_id,
            )
            return (
                record.get("anchor_artifact_sha256") == sha256_path(anchor_path)
                and record.get("prompt_sha256") == sha256_text(prompt)
                and record.get("seed") == expected_seed
                and artifact_matches(record.get("artifact"), run_root)
            )

        jobs = [
            (case, subject, variant)
            for case in cases
            for subject in case.scene.subjects
            for variant in subject.variants
            if not is_complete(case, subject, variant)
        ]
        if not jobs:
            return
        print(
            f"sprite variant stage: {len(jobs)} Qwen jobs at concurrency {concurrency}",
            file=sys.stderr,
            flush=True,
        )
        futures: dict[
            Future[Any],
            tuple[CaseState, SubjectSpec, SpriteVariantSpec, str, int, float],
        ] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for job_index, (case, subject, variant) in enumerate(jobs):
                prompt = self._variant_prompt(case, subject, variant)
                stage_seed = derived_seed(
                    case.seed,
                    case.scene.scene_id,
                    "variant",
                    subject.character_id,
                    variant.variant_id,
                )
                anchor_path = _sprite_component_path(
                    case.directory,
                    subject.character_id,
                )
                payload = {
                    "prompt": prompt,
                    "image_base64": encode_image(anchor_path),
                    "seed": stage_seed,
                    "steps": 28,
                    "cfg": 4.0,
                    "filename_prefix": slug(
                        f"staged_{case.scene.scene_id}_{subject.character_id}_"
                        f"{variant.variant_id}"
                    ),
                }
                worker = self._worker_target(job_index)
                if worker:
                    payload["worker"] = worker
                started = time.monotonic()
                future = executor.submit(
                    self.client.post_image,
                    "/edit/qwen",
                    payload,
                )
                futures[future] = (
                    case,
                    subject,
                    variant,
                    prompt,
                    stage_seed,
                    started,
                )
            failures: list[str] = []
            for future in as_completed(futures):
                case, subject, variant, prompt, stage_seed, started = futures[future]
                try:
                    data, headers = future.result()
                    output_path = _sprite_component_path(
                        case.directory,
                        subject.character_id,
                        variant.variant_id,
                    )
                    save_response_image(data, output_path)
                    variant_stages = (
                        case.manifest["stages"]
                        .setdefault("variants", {})
                        .setdefault(subject.character_id, {})
                    )
                    variant_stages[variant.variant_id] = {
                        "component_mode": "anchor_edit",
                        "anchor_variant_id": BASE_VARIANT_ID,
                        "anchor_artifact_sha256": sha256_path(
                            _sprite_component_path(
                                case.directory,
                                subject.character_id,
                            )
                        ),
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "seed": stage_seed,
                        "input_count": 1,
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "response_headers": headers,
                        "artifact": artifact_metadata(output_path, run_root),
                    }
                    write_json(case.manifest_path, case.manifest)
                except Exception as exc:
                    failures.append(
                        f"{case.scene.scene_id}/{case.seed}/"
                        f"{subject.character_id}/{variant.variant_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if failures:
            raise RuntimeError("sprite variant stage failed:\n" + "\n".join(failures))

    @staticmethod
    def _variant_prompt(
        case: CaseState,
        subject: SubjectSpec,
        variant: SpriteVariantSpec,
    ) -> str:
        if case.scene.presentation != "visual_novel":
            raise ValueError("sprite variants are only valid for visual-novel scenes")
        return "\n".join(
            (
                f"Input image: Image 1 is the approved neutral visual-novel sprite "
                f"anchor for {subject.display_name} and is the only edit target.",
                f"Primary edit: Change only the facial expression to "
                f"{variant.expression_prompt}",
                f"Small pose adjustment: {variant.pose_adjustment_prompt}",
                "Continuity: Preserve identity, age, body type, facial structure, "
                "hair, clothing, equipment, screen-facing direction, three-quarter "
                "or complete-body anchor crop, camera, scale, linework, lighting, "
                "and neutral backdrop.",
                "Scope: Keep the adjustment restrained and conversational. Do not "
                "add or remove a character, prop, limb, costume element, or text.",
            )
        )

    def _matte_stage(self, cases: Sequence[CaseState], run_root: Path) -> None:
        concurrency = getattr(self, "max_workers", 4)

        def stage_record(
            case: CaseState,
            subject: SubjectSpec,
            variant_id: str,
        ) -> dict[str, Any]:
            if variant_id == BASE_VARIANT_ID:
                return (
                    case.manifest.get("stages", {})
                    .get("mattes", {})
                    .get(subject.character_id, {})
                )
            return (
                case.manifest.get("stages", {})
                .get("variant_mattes", {})
                .get(subject.character_id, {})
                .get(variant_id, {})
            )

        jobs = [
            (case, subject, variant_id, component_path)
            for case in cases
            for subject, variant_id, component_path in _case_sprite_assets(case)
            if not (
                component_path.is_file()
                and stage_record(case, subject, variant_id).get(
                    "component_artifact_sha256"
                )
                == sha256_path(component_path)
                and artifact_matches(
                    stage_record(case, subject, variant_id).get("mask_artifact"),
                    run_root,
                )
                and artifact_matches(
                    stage_record(case, subject, variant_id).get("rgba_artifact"),
                    run_root,
                )
            )
        ]
        job_count = len(jobs)
        print(f"matte stage: {job_count} BiRefNet jobs", file=sys.stderr, flush=True)
        futures: dict[
            Future[Any],
            tuple[CaseState, SubjectSpec, str, Path, float],
        ] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for job_index, (case, subject, variant_id, component_path) in enumerate(
                jobs
            ):
                payload = {
                    "image_base64": encode_image(component_path),
                    "filename_prefix": slug(
                        f"matte_{case.scene.scene_id}_{subject.character_id}_"
                        f"{variant_id}"
                    ),
                    "model_name": "birefnet.safetensors",
                }
                worker = self._worker_target(job_index)
                if worker:
                    payload["worker"] = worker
                started = time.monotonic()
                future = executor.submit(
                    self.client.post_image,
                    "/prototype/matte/birefnet",
                    payload,
                )
                futures[future] = (
                    case,
                    subject,
                    variant_id,
                    component_path,
                    started,
                )
            failures: list[str] = []
            for future in as_completed(futures):
                case, subject, variant_id, component_path, started = futures[future]
                try:
                    data, headers = future.result()
                    matte_dir = case.directory / "mattes"
                    matte_dir.mkdir(parents=True, exist_ok=True)
                    mask_path, rgba_path = _sprite_matte_paths(
                        case.directory,
                        subject.character_id,
                        variant_id,
                    )
                    save_response_image(data, mask_path, mode="L")
                    with Image.open(mask_path) as mask_image:
                        metrics = validate_matte(mask_image, subject.mask_policy)
                    create_rgba(component_path, mask_path, rgba_path)
                    if variant_id == BASE_VARIANT_ID:
                        matte_stages = case.manifest["stages"].setdefault("mattes", {})
                        stage_container = matte_stages
                        stage_key = subject.character_id
                    else:
                        stage_container = (
                            case.manifest["stages"]
                            .setdefault("variant_mattes", {})
                            .setdefault(subject.character_id, {})
                        )
                        stage_key = variant_id
                    stage_container[stage_key] = {
                        "model": "BiRefNet",
                        "model_file": "birefnet.safetensors",
                        "variant_id": variant_id,
                        "component_artifact_sha256": sha256_path(component_path),
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "response_headers": headers,
                        "validation": metrics,
                        "mask_artifact": artifact_metadata(mask_path, run_root),
                        "rgba_artifact": artifact_metadata(rgba_path, run_root),
                    }
                    write_json(case.manifest_path, case.manifest)
                except Exception as exc:
                    failures.append(
                        f"{case.scene.scene_id}/{case.seed}/{subject.character_id}: "
                        f"{variant_id}: {type(exc).__name__}: {exc}"
                    )
        if failures:
            raise RuntimeError("matte stage failed:\n" + "\n".join(failures))

    def _pixel_stage(self, cases: Sequence[CaseState], run_root: Path) -> None:
        print(f"pixel composite stage: {len(cases)} cases", file=sys.stderr, flush=True)
        for case in cases:
            if case.scene.presentation == "visual_novel":
                self._visual_novel_pixel_case(case, run_root)
                continue
            rgba_paths = {
                subject.character_id: _sprite_matte_paths(
                    case.directory,
                    subject.character_id,
                )[1]
                for subject in case.scene.subjects
            }
            output_path = case.directory / "final" / "pixel.png"
            subject_mask_path = case.directory / "final" / "subject-mask.png"
            repair_mask_path = case.directory / "final" / "repair-mask.png"
            started = time.monotonic()
            placements = composite_subjects(
                case.scene,
                case.directory / "background.png",
                rgba_paths,
                output_path,
                subject_mask_path,
                repair_mask_path,
            )
            case.manifest["stages"]["pixel"] = {
                "operation": "deterministic RGBA source-over composite",
                "duration_seconds": round(time.monotonic() - started, 3),
                "placements": placements,
                "artifact": artifact_metadata(output_path, run_root),
                "subject_mask": artifact_metadata(subject_mask_path, run_root),
                "repair_mask": artifact_metadata(repair_mask_path, run_root),
            }
            write_json(case.manifest_path, case.manifest)

    @staticmethod
    def _visual_novel_pixel_case(case: CaseState, run_root: Path) -> None:
        started = time.monotonic()
        frame_records: list[dict[str, Any]] = []
        storyboard_inputs: list[tuple[str, Path]] = []
        background_path = case.directory / "background.png"
        background_sha256 = sha256_path(background_path)
        for frame in case.scene.frames:
            rgba_paths = {}
            resolved_variants: dict[str, str] = {}
            for subject in case.scene.subjects:
                variant_id = frame.subject_variants.get(
                    subject.character_id,
                    BASE_VARIANT_ID,
                )
                resolved_variants[subject.character_id] = variant_id
                rgba_paths[subject.character_id] = _sprite_matte_paths(
                    case.directory,
                    subject.character_id,
                    variant_id,
                )[1]
            frame_directory = case.directory / "final" / "frames"
            output_path = frame_directory / f"{slug(frame.frame_id)}.png"
            subject_mask_path = frame_directory / f"{slug(frame.frame_id)}-subjects.png"
            repair_mask_path = frame_directory / f"{slug(frame.frame_id)}-seams.png"
            placements = composite_subjects(
                case.scene,
                background_path,
                rgba_paths,
                output_path,
                subject_mask_path,
                repair_mask_path,
            )
            frame_records.append(
                {
                    "frame_id": frame.frame_id,
                    "title": frame.title,
                    "active_character_id": frame.active_character_id,
                    "subject_variants": resolved_variants,
                    "background_sha256": background_sha256,
                    "placements": placements,
                    "artifact": artifact_metadata(output_path, run_root),
                    "subject_mask": artifact_metadata(subject_mask_path, run_root),
                    "seam_mask": artifact_metadata(repair_mask_path, run_root),
                }
            )
            storyboard_inputs.append((frame.title, output_path))
        output_path = case.directory / "final" / "pixel.png"
        build_storyboard(storyboard_inputs, output_path)
        case.manifest["stages"]["pixel"] = {
            "operation": "fixed visual-novel stage with anchor-derived sprites",
            "duration_seconds": round(time.monotonic() - started, 3),
            "background_sha256": background_sha256,
            "background_reused_for_every_frame": True,
            "whole_scene_model_edit": False,
            "frames": frame_records,
            "artifact": artifact_metadata(output_path, run_root),
        }
        write_json(case.manifest_path, case.manifest)

    def _final_stage(
        self,
        cases: Sequence[CaseState],
        run_root: Path,
        tracks: Sequence[str],
    ) -> None:
        concurrency = getattr(self, "max_workers", 4)
        requested = [track for track in tracks if track != "pixel"]
        if not requested:
            return
        if any(case.scene.presentation == "visual_novel" for case in cases):
            raise ValueError("visual-novel scenes cannot use whole-scene edit tracks")
        print(
            f"final stage: {len(cases) * len(requested)} Qwen jobs at "
            f"concurrency {concurrency}",
            file=sys.stderr,
            flush=True,
        )
        futures: dict[Future[Any], tuple[CaseState, str, str, int, float]] = {}
        job_index = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for case in cases:
                pixel_path = case.directory / "final" / "pixel.png"
                for track in requested:
                    stage_seed = derived_seed(
                        case.seed,
                        case.scene.scene_id,
                        track,
                    )
                    if track == "masked":
                        prompt = self._masked_prompt(case)
                        endpoint = "/prototype/edit/qwen/masked"
                        payload = {
                            "prompt": prompt,
                            "image_base64": encode_image(pixel_path),
                            "mask_base64": encode_image(
                                case.directory / "final" / "repair-mask.png"
                            ),
                            "seed": stage_seed,
                            "steps": 26,
                            "cfg": 4.0,
                            "denoise": 0.38,
                            "filename_prefix": slug(
                                f"staged_masked_{case.scene.scene_id}_{case.seed}"
                            ),
                        }
                    else:
                        prompt = self._global_prompt(case)
                        endpoint = "/edit/qwen"
                        payload = {
                            "prompt": prompt,
                            "image_base64": encode_image(pixel_path),
                            "seed": stage_seed,
                            "steps": 28,
                            "cfg": 4.0,
                            "filename_prefix": slug(
                                f"staged_global_{case.scene.scene_id}_{case.seed}"
                            ),
                        }
                    worker = self._worker_target(job_index)
                    job_index += 1
                    if worker:
                        payload["worker"] = worker
                    started = time.monotonic()
                    future = executor.submit(self.client.post_image, endpoint, payload)
                    futures[future] = (case, track, prompt, stage_seed, started)

            failures: list[str] = []
            for future in as_completed(futures):
                case, track, prompt, stage_seed, started = futures[future]
                try:
                    data, headers = future.result()
                    final_dir = case.directory / "final"
                    if track == "masked":
                        model_output = final_dir / "masked-model-output.png"
                        save_response_image(data, model_output)
                        output_path = final_dir / "masked.png"
                        changed_outside = blend_masked_result(
                            final_dir / "pixel.png",
                            model_output,
                            final_dir / "repair-mask.png",
                            output_path,
                        )
                        extra = {
                            "model_output": artifact_metadata(model_output, run_root),
                            "changed_pixels_outside_mask": changed_outside,
                        }
                    else:
                        output_path = final_dir / "global.png"
                        save_response_image(data, output_path)
                        extra = {"diagnostic_only": True}
                    case.manifest["stages"][track] = {
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "seed": stage_seed,
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "response_headers": headers,
                        "artifact": artifact_metadata(output_path, run_root),
                        **extra,
                    }
                    write_json(case.manifest_path, case.manifest)
                except Exception as exc:
                    failures.append(
                        f"{case.scene.scene_id}/{case.seed}/{track}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if failures:
            raise RuntimeError("final stage failed:\n" + "\n".join(failures))

    @staticmethod
    def _masked_prompt(case: CaseState) -> str:
        return "\n".join(
            (
                "Input image: Image 1 is the assembled scene and edit target.",
                f"Primary edit: {case.scene.harmonize_prompt}",
                f"Integration style: {case.style.harmonization_direction}",
                "Scope: Change only assembly seams, edge light, cast shadows, and "
                "the specified physical-contact regions. Keep faces, bodies, "
                "clothing, equipment, poses, subject count, and environment unchanged.",
            )
        )

    @staticmethod
    def _global_prompt(case: CaseState) -> str:
        return "\n".join(
            (
                "Input image: Image 1 is the assembled scene and edit target.",
                f"Primary edit: {case.scene.diagnostic_prompt}",
                f"Style: {case.style.visual_language}",
                "Preserve every subject's identity, age, body type, clothing, "
                "equipment, count, scale, and pose while integrating light and grounding.",
            )
        )


def _resolve_run_root(output_root: Path, run_id_or_path: str) -> Path:
    candidate = Path(run_id_or_path)
    run_root = candidate if candidate.is_absolute() else output_root / candidate
    run_root = run_root.resolve()
    if not (run_root / "run.json").is_file():
        raise FileNotFoundError(f"run manifest missing: {run_root / 'run.json'}")
    return run_root


def record_review(
    *,
    run_root: Path,
    scene_id: str,
    seed: int,
    track: str,
    verdict: str,
    reasons: Sequence[str],
    note: str,
    reviewer: str,
) -> ReviewRecord:
    manifest_path = run_root / scene_id / f"seed-{seed}" / "manifest.json"
    manifest = json_load(manifest_path)
    stage = manifest.get("stages", {}).get(track)
    if not stage or "artifact" not in stage:
        raise KeyError(f"track artifact does not exist: {scene_id}/{seed}/{track}")
    artifact_path = run_root / stage["artifact"]["relative_path"]
    if not artifact_path.is_file():
        raise FileNotFoundError(f"track artifact is missing: {artifact_path}")
    actual_sha = sha256_path(artifact_path)
    if actual_sha != stage["artifact"]["sha256"]:
        raise ValueError(f"track artifact hash mismatch: {artifact_path}")
    record = ReviewRecord.model_validate(
        {
            "scene_id": scene_id,
            "seed": seed,
            "track": track,
            "verdict": verdict,
            "reasons": list(dict.fromkeys(reasons)),
            "note": note,
            "reviewer": reviewer,
            "artifact_sha256": actual_sha,
            "reviewed_at": utc_now(),
        }
    )
    reviews_path = run_root / "reviews.jsonl"
    with reviews_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
    return record


def load_reviews(run_root: Path) -> dict[tuple[str, int, str], ReviewRecord]:
    path = run_root / "reviews.jsonl"
    latest: dict[tuple[str, int, str], ReviewRecord] = {}
    if not path.is_file():
        return latest
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = ReviewRecord.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"invalid review at {path}:{line_number}") from exc
        latest[(record.scene_id, record.seed, record.track)] = record
    return latest


def build_report(run_root: Path) -> tuple[Path, Path]:
    run_manifest = json_load(run_root / "run.json")
    reviews = load_reviews(run_root)
    case_manifests = [
        json_load(run_root / relative) for relative in run_manifest["cases"]
    ]
    records: list[dict[str, Any]] = []
    counts = {"pass": 0, "loser": 0, "unreviewed": 0, "stale": 0}
    reason_counts = {reason: 0 for reason in REVIEW_REASONS}
    for manifest in case_manifests:
        for track in manifest["tracks"]:
            stage = manifest.get("stages", {}).get(track)
            if not stage or "artifact" not in stage:
                continue
            key = (manifest["scene_id"], manifest["seed"], track)
            review = reviews.get(key)
            if review and review.artifact_sha256 != stage["artifact"]["sha256"]:
                verdict = "stale"
                review = None
            else:
                verdict = review.verdict if review else "unreviewed"
            counts[verdict] += 1
            if review:
                for reason in review.reasons:
                    reason_counts[reason] += 1
            records.append(
                {
                    "scene_id": manifest["scene_id"],
                    "scene_title": manifest["scene_title"],
                    "split": manifest["split"],
                    "seed": manifest["seed"],
                    "track": track,
                    "verdict": verdict,
                    "artifact": stage["artifact"],
                    "frames": stage.get("frames", []),
                    "review": review.model_dump(mode="json") if review else None,
                }
            )
    scene_seed_counts: dict[str, set[int]] = {}
    for manifest in case_manifests:
        scene_seed_counts.setdefault(manifest["scene_id"], set()).add(manifest["seed"])
    batch_complete = all(len(seeds) == 4 for seeds in scene_seed_counts.values())
    expected_outputs = sum(len(manifest["tracks"]) for manifest in case_manifests)
    quality_gate_passed = (
        run_manifest["status"] == "complete"
        and batch_complete
        and len(records) == expected_outputs
        and counts["pass"] == expected_outputs
        and counts["loser"] == 0
        and counts["unreviewed"] == 0
        and counts["stale"] == 0
    )
    summary = {
        "run_id": run_manifest["run_id"],
        "run_status": run_manifest["status"],
        "generated_at": utc_now(),
        "batch_complete": batch_complete,
        "quality_gate_passed": quality_gate_passed,
        "counts": counts,
        "loser_reasons": reason_counts,
        "records": records,
    }
    json_path = run_root / "report.json"
    write_json(json_path, summary)

    cards: list[str] = []
    for record in records:
        artifact_path = run_root / record["artifact"]["relative_path"]
        relative_image = artifact_path.relative_to(run_root).as_posix()
        review = record["review"]
        verdict = record["verdict"]
        reasons = ", ".join(review["reasons"]) if review else ""
        note = review["note"] if review else ""
        frame_links = "".join(
            f"<a href='{html.escape(frame['artifact']['relative_path'])}'>"
            f"{html.escape(frame['title'])}</a>"
            for frame in record.get("frames", [])
        )
        if frame_links:
            frame_links = f"<nav class='frames'>{frame_links}</nav>"
        cards.append(
            "<article class='card'>"
            f"<img loading='lazy' src='{html.escape(relative_image)}' alt='generated track'>"
            f"<h2>{html.escape(record['scene_title'])}</h2>"
            f"<p>{html.escape(record['split'])} · seed {record['seed']} · "
            f"{html.escape(record['track'])}</p>"
            f"<p class='verdict {html.escape(verdict)}'>{html.escape(verdict)}</p>"
            f"<p>{html.escape(reasons)}</p><p>{html.escape(note)}</p>{frame_links}"
            "</article>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Staged image report: {html.escape(run_manifest["run_id"])}</title>
<style>
body {{ margin: 1.5rem; color: #ddd; background: #17171b; font: 15px system-ui; }}
.summary {{ position: sticky; top: 0; z-index: 2; padding: 1rem; background: #25252c; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill,minmax(280px,1fr)); gap: 1rem; }}
.card {{ padding: .8rem; border: 1px solid #454550; background: #202027; }}
.card img {{ width: 100%; height: 360px; object-fit: contain; background: #0d0d10; }}
.card h2 {{ margin: .6rem 0 .2rem; font-size: 1rem; }}
.card p {{ margin: .25rem 0; }}
.frames {{ display: flex; flex-wrap: wrap; gap: .6rem; margin-top: .65rem; }}
.frames a {{ color: #8dc7ff; }}
.verdict {{ font-weight: 700; text-transform: uppercase; }}
.pass {{ color: #66d58b; }} .loser {{ color: #ff7979; }}
.unreviewed {{ color: #e6ca6d; }} .stale {{ color: #e6a66d; }}
</style></head><body>
<section class="summary"><h1>{html.escape(run_manifest["run_id"])}</h1>
<p>Status: {html.escape(run_manifest["status"])}; four-seed batch complete: {str(batch_complete).lower()};
quality gate passed: {str(quality_gate_passed).lower()}; pass {counts["pass"]};
loser {counts["loser"]}; unreviewed {counts["unreviewed"]}; stale {counts["stale"]}.</p></section>
<main class="grid">{"".join(cards)}</main></body></html>
"""
    html_path = run_root / "report.html"
    html_path.write_text(document, encoding="utf-8")
    return html_path, json_path


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc
    if len(seeds) != 4 or len(set(seeds)) != 4 or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError(
            "exactly four distinct non-negative seeds required"
        )
    return seeds


def parse_tracks(value: str) -> tuple[str, ...]:
    tracks = tuple(item.strip() for item in value.split(",") if item.strip())
    if "pixel" not in tracks or set(tracks).difference(TRACKS):
        raise argparse.ArgumentTypeError(
            "tracks must be a comma-separated subset of pixel,masked,global and include pixel"
        )
    return tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and review the isolated staged-image prototype."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="execute a four-seed staged batch")
    run.add_argument("--scene", action="append", dest="scenes", default=[])
    run.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    run.add_argument("--tracks", type=parse_tracks, default=DEFAULT_TRACKS)
    run.add_argument("--run-id")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--reuse-backgrounds-from")
    run.add_argument("--max-workers", type=int, choices=range(1, 5), default=4)
    run.add_argument(
        "--worker",
        action="append",
        dest="workers",
        type=int,
        choices=range(4),
    )
    run.add_argument("--include-holdout", action="store_true")
    run.add_argument("--style-pack")
    run.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    run.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    run.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    run.add_argument("--styles", type=Path, default=DEFAULT_STYLE_PATH)

    review = subparsers.add_parser("review", help="append a human QA verdict")
    review.add_argument("run")
    review.add_argument("--scene", required=True)
    review.add_argument("--seed", required=True, type=int)
    review.add_argument("--track", required=True, choices=TRACKS)
    review.add_argument("--verdict", required=True, choices=("pass", "loser"))
    review.add_argument("--reason", action="append", choices=REVIEW_REASONS, default=[])
    review.add_argument("--note", default="")
    review.add_argument("--reviewer", default="human")

    report = subparsers.add_parser("report", help="build an HTML and JSON QA report")
    report.add_argument("run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run":
        runner = StagedImageRunner(
            corpus_path=arguments.corpus,
            style_path=arguments.styles,
            runtime_root=arguments.runtime_root,
            output_root=arguments.output_root,
            gateway_url=arguments.gateway_url,
            max_workers=arguments.max_workers,
            worker_indices=arguments.workers,
        )
        result = runner.run(
            scene_ids=arguments.scenes,
            seeds=arguments.seeds,
            tracks=arguments.tracks,
            run_id=arguments.run_id,
            dry_run=arguments.dry_run,
            resume=arguments.resume,
            reuse_backgrounds_from=arguments.reuse_backgrounds_from,
            include_holdout=arguments.include_holdout,
            style_pack_id=arguments.style_pack,
        )
        if isinstance(result, dict):
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(result)
        return 0
    run_root = _resolve_run_root(arguments.output_root, arguments.run)
    if arguments.command == "review":
        record = record_review(
            run_root=run_root,
            scene_id=arguments.scene,
            seed=arguments.seed,
            track=arguments.track,
            verdict=arguments.verdict,
            reasons=arguments.reason,
            note=arguments.note,
            reviewer=arguments.reviewer,
        )
        print(record.model_dump_json(indent=2))
        return 0
    html_path, json_path = build_report(run_root)
    print(html_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
