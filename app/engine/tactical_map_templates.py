from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from app.schemas.content_pack import (
    ContentImageAsset,
    TacticalMapAreaLink,
    TacticalMapFeature,
    TacticalMapSpawnAnchor,
    TacticalMapTemplateRecord,
)
from app.schemas.dnd_spatial import (
    DndBattleMapAreaLink as RuntimeBattleMapAreaLink,
    DndBattleMapFeature as RuntimeBattleMapFeature,
    DndBattleMapRuntimeState,
    DndBattleMapSpawnAnchor as RuntimeBattleMapSpawnAnchor,
    DndMapPoint,
    DndMapRect,
    DndTerrainZone,
    MAX_BATTLE_MAP_HEIGHT,
    MAX_BATTLE_MAP_WIDTH,
)


REVIEW_READY_STATUSES = {"reviewed", "approved"}
_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TERRAIN_COVER = {"none", "half", "three_quarters", "total"}
_VERTICAL_FEATURE_KINDS = {"stairs"}
_SCHEMA_GAP_LAYERS = {
    "floors_submaps": (
        "TacticalMapTemplateRecord does not represent floors/submaps; "
        "strict multi-plane automation needs a successor schema."
    ),
    "fog_reveal_regions": (
        "TacticalMapTemplateRecord does not represent fog/reveal regions; "
        "strict fog/reveal automation needs a successor schema."
    ),
}

RequiredTacticalMapLayer = Literal[
    "map_ref",
    "spawn_anchors",
    "terrain",
    "areas",
    "secrets",
    "vertical_links",
    "floors_submaps",
    "fog_reveal_regions",
]


class TacticalMapTemplateCompileError(ValueError):
    """Raised when a reviewed map template cannot become runtime geometry."""

    def __init__(self, message: str, *, reasons: Sequence[str] = ()):
        self.reasons = tuple(reason for reason in reasons if reason)
        detail = "; ".join(self.reasons)
        super().__init__(f"{message}: {detail}" if detail else message)


@dataclass(frozen=True, slots=True)
class CompiledGridPoint:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class CompiledGridRect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CompiledSpawnAnchor:
    anchor_id: str
    anchor_kind: str
    cells: tuple[CompiledGridPoint, ...]
    label: str
    linked_ref: str


@dataclass(frozen=True, slots=True)
class CompiledMapFeature:
    feature_id: str
    feature_kind: str
    cells: tuple[CompiledGridPoint, ...]
    bounds: CompiledGridRect | None
    label: str
    blocks_movement: bool
    blocks_line_of_sight: bool
    difficult_terrain: bool
    cover: str
    secret: bool
    reveal_trigger: str
    linked_refs: tuple[str, ...]

    @property
    def is_vertical_link(self) -> bool:
        return self.feature_kind in _VERTICAL_FEATURE_KINDS


@dataclass(frozen=True, slots=True)
class CompiledAreaLink:
    area_id: str
    location_ref: str
    cells: tuple[CompiledGridPoint, ...]
    bounds: CompiledGridRect | None


@dataclass(frozen=True, slots=True)
class CompiledTacticalMapTemplate:
    pack_id: str
    ref: str
    content_hash: str
    title: str
    map_kind: str
    source_map_asset_id: str
    grid_width: int
    grid_height: int
    square_size_ft: int
    orientation: str
    spawn_anchors: tuple[CompiledSpawnAnchor, ...]
    terrain_features: tuple[CompiledMapFeature, ...]
    area_links: tuple[CompiledAreaLink, ...]

    @property
    def secret_features(self) -> tuple[CompiledMapFeature, ...]:
        return tuple(feature for feature in self.terrain_features if feature.secret)

    @property
    def vertical_links(self) -> tuple[CompiledMapFeature, ...]:
        return tuple(
            feature for feature in self.terrain_features if feature.is_vertical_link
        )

    def to_battle_map_state(self) -> DndBattleMapRuntimeState:
        """Compile reviewed imported geometry into runtime combat-map state."""
        terrain: list[DndTerrainZone] = []
        for feature in self.terrain_features:
            terrain.extend(_terrain_zones_for_feature(feature))

        return DndBattleMapRuntimeState(
            present=True,
            map_name=self.title or self.ref or "Battle map",
            width=self.grid_width,
            height=self.grid_height,
            square_size_ft=self.square_size_ft,
            tokens=[],
            terrain=terrain,
            areas=[],
            notes="",
            source_template_ref=self.ref,
            source_content_hash=self.content_hash,
            orientation=self.orientation,
            spawn_anchors=[
                RuntimeBattleMapSpawnAnchor(
                    anchor_id=anchor.anchor_id,
                    anchor_kind=anchor.anchor_kind,
                    cells=[DndMapPoint(x=cell.x, y=cell.y) for cell in anchor.cells],
                    label=anchor.label,
                    linked_ref=anchor.linked_ref,
                )
                for anchor in self.spawn_anchors
            ],
            features=[_runtime_feature(feature) for feature in self.terrain_features],
            area_links=[
                RuntimeBattleMapAreaLink(
                    area_id=link.area_id,
                    location_ref=link.location_ref,
                    cells=[DndMapPoint(x=cell.x, y=cell.y) for cell in link.cells],
                    bounds=_runtime_rect(link.bounds),
                )
                for link in self.area_links
            ],
        )


def compile_tactical_map_template(
    template: TacticalMapTemplateRecord | Mapping[str, Any],
    *,
    map_assets: (
        Iterable[ContentImageAsset | Mapping[str, Any]]
        | Mapping[str, ContentImageAsset | Mapping[str, Any]]
        | None
    ) = None,
    authored_refs: Iterable[str] | None = None,
    required_layers: Iterable[RequiredTacticalMapLayer] = ("map_ref",),
) -> CompiledTacticalMapTemplate:
    """Validate and freeze a reviewed content-pack tactical map template.

    The compiler consumes only typed, reviewed records. It never reads source
    pages, images, OCR output, local paths, or asset bytes.
    """
    record = _coerce_template(template)
    authored_ref_set = (
        {ref.strip() for ref in authored_refs if ref.strip()}
        if authored_refs is not None
        else None
    )
    asset_lookup = _asset_lookup(map_assets)

    blockers: list[str] = []
    _validate_runtime_ready(record, blockers)
    _validate_required_layers(record, required_layers, blockers)
    _validate_grid(record, blockers)
    _validate_map_asset(record, asset_lookup, blockers)

    spawn_anchors = _compile_spawn_anchors(
        record.spawn_anchors,
        record.grid_width,
        record.grid_height,
        authored_ref_set,
        blockers,
    )
    terrain_features = _compile_features(
        record.terrain_features,
        record.grid_width,
        record.grid_height,
        authored_ref_set,
        blockers,
    )
    area_links = _compile_area_links(
        record.area_links,
        record.grid_width,
        record.grid_height,
        authored_ref_set,
        blockers,
    )

    if blockers:
        raise TacticalMapTemplateCompileError(
            f"tactical map template {record.ref!r} is not runtime-ready",
            reasons=blockers,
        )

    return CompiledTacticalMapTemplate(
        pack_id=record.pack_id,
        ref=record.ref,
        content_hash=record.content_hash,
        title=record.title,
        map_kind=record.map_kind,
        source_map_asset_id=record.derived_from_map_asset_id,
        grid_width=record.grid_width,
        grid_height=record.grid_height,
        square_size_ft=record.square_size_ft,
        orientation=record.orientation,
        spawn_anchors=spawn_anchors,
        terrain_features=terrain_features,
        area_links=area_links,
    )


def compile_tactical_map_template_battle_map_state(
    template: TacticalMapTemplateRecord | Mapping[str, Any],
    **kwargs: Any,
) -> DndBattleMapRuntimeState:
    compiled = compile_tactical_map_template(template, **kwargs)
    return compiled.to_battle_map_state()


def _coerce_template(
    template: TacticalMapTemplateRecord | Mapping[str, Any],
) -> TacticalMapTemplateRecord:
    if isinstance(template, TacticalMapTemplateRecord):
        return template
    return TacticalMapTemplateRecord.model_validate(template)


def _validate_runtime_ready(
    record: TacticalMapTemplateRecord,
    blockers: list[str],
) -> None:
    if record.target_runtime_schema != "DndBattleMapRuntimeState":
        blockers.append(
            f"target_runtime_schema {record.target_runtime_schema!r} is unsupported"
        )
    if record.map_kind != "battle_map":
        blockers.append(
            f"map_kind {record.map_kind!r} cannot compile to D&D combat map state"
        )
    if not record.content_hash:
        blockers.append("runtime-ready tactical maps need content_hash")
    if record.review_status not in REVIEW_READY_STATUSES:
        blockers.append("runtime-ready tactical maps must be reviewed or approved")
    if record.gate_status != "runtime_ready":
        blockers.append(f"gate_status {record.gate_status!r} is not runtime_ready")
    if record.gate_reasons:
        blockers.append("runtime-ready tactical maps must not carry gate_reasons")


def _validate_required_layers(
    record: TacticalMapTemplateRecord,
    required_layers: Iterable[RequiredTacticalMapLayer],
    blockers: list[str],
) -> None:
    required = tuple(dict.fromkeys(required_layers))
    for layer in required:
        if layer in _SCHEMA_GAP_LAYERS:
            blockers.append(_SCHEMA_GAP_LAYERS[layer])
        elif layer == "map_ref" and not record.derived_from_map_asset_id:
            blockers.append("required map_ref layer is missing")
        elif layer == "spawn_anchors" and not record.spawn_anchors:
            blockers.append("required spawn_anchors layer is missing")
        elif layer == "terrain" and not record.terrain_features:
            blockers.append("required terrain layer is missing")
        elif layer == "areas" and not record.area_links:
            blockers.append("required keyed-area layer is missing")
        elif layer == "secrets" and not any(
            feature.secret for feature in record.terrain_features
        ):
            blockers.append("required secret-feature layer is missing")
        elif layer == "vertical_links" and not any(
            feature.feature_kind in _VERTICAL_FEATURE_KINDS
            for feature in record.terrain_features
        ):
            blockers.append("required vertical-link layer is missing")


def _validate_grid(
    record: TacticalMapTemplateRecord,
    blockers: list[str],
) -> None:
    if record.grid_width > MAX_BATTLE_MAP_WIDTH:
        blockers.append(
            f"grid_width {record.grid_width} exceeds DndBattleMapRuntimeState cap "
            f"{MAX_BATTLE_MAP_WIDTH}"
        )
    if record.grid_height > MAX_BATTLE_MAP_HEIGHT:
        blockers.append(
            f"grid_height {record.grid_height} exceeds DndBattleMapRuntimeState cap "
            f"{MAX_BATTLE_MAP_HEIGHT}"
        )


def _validate_map_asset(
    record: TacticalMapTemplateRecord,
    asset_lookup: dict[str, ContentImageAsset] | None,
    blockers: list[str],
) -> None:
    asset_id = record.derived_from_map_asset_id
    if not asset_id:
        return
    if not _ASSET_ID_RE.fullmatch(asset_id):
        blockers.append(
            f"derived_from_map_asset_id {asset_id!r} must be a logical asset id, "
            "not a path, URL, source page, OCR block, or image payload"
        )
        return
    if asset_lookup is None:
        return

    asset = asset_lookup.get(_asset_key(record.pack_id, asset_id)) or asset_lookup.get(
        asset_id
    )
    if asset is None:
        blockers.append(f"map asset {asset_id!r} is missing from the asset catalog")
        return
    if record.pack_id and asset.pack_id and asset.pack_id != record.pack_id:
        blockers.append(
            f"map asset {asset_id!r} belongs to pack {asset.pack_id!r}, "
            f"not {record.pack_id!r}"
        )
    if asset.review_status not in REVIEW_READY_STATUSES:
        blockers.append(
            f"map asset {asset_id!r} review_status {asset.review_status!r} "
            "is not reviewed or approved"
        )


def _compile_spawn_anchors(
    anchors: Iterable[TacticalMapSpawnAnchor],
    width: int,
    height: int,
    authored_refs: set[str] | None,
    blockers: list[str],
) -> tuple[CompiledSpawnAnchor, ...]:
    compiled: list[CompiledSpawnAnchor] = []
    seen: set[str] = set()
    for anchor in anchors:
        if not anchor.anchor_id:
            blockers.append("spawn anchor id is empty")
            continue
        if anchor.anchor_id in seen:
            blockers.append(f"duplicate spawn anchor id {anchor.anchor_id!r}")
        seen.add(anchor.anchor_id)
        if not anchor.cells:
            blockers.append(f"spawn anchor {anchor.anchor_id!r} has no cells")
        cells = _compile_points(
            anchor.cells,
            width,
            height,
            f"spawn anchor {anchor.anchor_id!r}",
            blockers,
        )
        if anchor.linked_ref:
            _validate_authored_ref(
                anchor.linked_ref,
                authored_refs,
                f"spawn anchor {anchor.anchor_id!r}",
                blockers,
            )
        compiled.append(
            CompiledSpawnAnchor(
                anchor_id=anchor.anchor_id,
                anchor_kind=anchor.anchor_kind,
                cells=cells,
                label=anchor.label,
                linked_ref=anchor.linked_ref,
            )
        )
    return tuple(compiled)


def _compile_features(
    features: Iterable[TacticalMapFeature],
    width: int,
    height: int,
    authored_refs: set[str] | None,
    blockers: list[str],
) -> tuple[CompiledMapFeature, ...]:
    compiled: list[CompiledMapFeature] = []
    seen: set[str] = set()
    for feature in features:
        if not feature.feature_id:
            blockers.append("terrain feature id is empty")
            continue
        if feature.feature_id in seen:
            blockers.append(f"duplicate terrain feature id {feature.feature_id!r}")
        seen.add(feature.feature_id)
        if not feature.cells and feature.bounds is None:
            blockers.append(f"terrain feature {feature.feature_id!r} has no geometry")
        if feature.cells and feature.bounds is not None:
            blockers.append(
                f"terrain feature {feature.feature_id!r} must use cells or bounds, "
                "not both"
            )
        if feature.feature_kind == "secret_feature" and not feature.secret:
            blockers.append(
                f"secret_feature {feature.feature_id!r} must set secret=true"
            )
        cover = feature.cover or "none"
        if cover not in _TERRAIN_COVER:
            blockers.append(
                f"terrain feature {feature.feature_id!r} has unsupported cover "
                f"{feature.cover!r}"
            )
        if feature.feature_kind in _VERTICAL_FEATURE_KINDS and not feature.linked_refs:
            blockers.append(
                f"vertical feature {feature.feature_id!r} needs linked_refs"
            )
        cells = _compile_points(
            feature.cells,
            width,
            height,
            f"terrain feature {feature.feature_id!r}",
            blockers,
        )
        bounds = _compile_rect(
            feature.bounds,
            width,
            height,
            f"terrain feature {feature.feature_id!r}",
            blockers,
        )
        for linked_ref in feature.linked_refs:
            _validate_authored_ref(
                linked_ref,
                authored_refs,
                f"terrain feature {feature.feature_id!r}",
                blockers,
            )
        compiled.append(
            CompiledMapFeature(
                feature_id=feature.feature_id,
                feature_kind=feature.feature_kind,
                cells=cells,
                bounds=bounds,
                label=feature.label,
                blocks_movement=feature.blocks_movement,
                blocks_line_of_sight=feature.blocks_line_of_sight,
                difficult_terrain=feature.difficult_terrain,
                cover=cover,
                secret=feature.secret,
                reveal_trigger=feature.reveal_trigger,
                linked_refs=tuple(feature.linked_refs),
            )
        )
    return tuple(compiled)


def _compile_area_links(
    links: Iterable[TacticalMapAreaLink],
    width: int,
    height: int,
    authored_refs: set[str] | None,
    blockers: list[str],
) -> tuple[CompiledAreaLink, ...]:
    compiled: list[CompiledAreaLink] = []
    seen: set[str] = set()
    for link in links:
        if not link.area_id:
            blockers.append("keyed-area link id is empty")
            continue
        if link.area_id in seen:
            blockers.append(f"duplicate keyed-area link id {link.area_id!r}")
        seen.add(link.area_id)
        if not link.location_ref:
            blockers.append(f"keyed-area link {link.area_id!r} has no location_ref")
        if not link.cells and link.bounds is None:
            blockers.append(f"keyed-area link {link.area_id!r} has no geometry")
        if link.cells and link.bounds is not None:
            blockers.append(
                f"keyed-area link {link.area_id!r} must use cells or bounds, not both"
            )
        cells = _compile_points(
            link.cells,
            width,
            height,
            f"keyed-area link {link.area_id!r}",
            blockers,
        )
        bounds = _compile_rect(
            link.bounds,
            width,
            height,
            f"keyed-area link {link.area_id!r}",
            blockers,
        )
        if link.location_ref:
            _validate_authored_ref(
                link.location_ref,
                authored_refs,
                f"keyed-area link {link.area_id!r}",
                blockers,
            )
        compiled.append(
            CompiledAreaLink(
                area_id=link.area_id,
                location_ref=link.location_ref,
                cells=cells,
                bounds=bounds,
            )
        )
    return tuple(compiled)


def _compile_points(
    points: Iterable[Any],
    width: int,
    height: int,
    label: str,
    blockers: list[str],
) -> tuple[CompiledGridPoint, ...]:
    compiled: list[CompiledGridPoint] = []
    seen: set[tuple[int, int]] = set()
    for point in points:
        x = point.x
        y = point.y
        if x >= width or y >= height:
            blockers.append(f"{label} cell ({x}, {y}) is outside map bounds")
            continue
        key = (x, y)
        if key in seen:
            blockers.append(f"{label} repeats cell ({x}, {y})")
            continue
        seen.add(key)
        compiled.append(CompiledGridPoint(x=x, y=y))
    return tuple(compiled)


def _compile_rect(
    rect: Any | None,
    width: int,
    height: int,
    label: str,
    blockers: list[str],
) -> CompiledGridRect | None:
    if rect is None:
        return None
    if rect.x >= width or rect.y >= height:
        blockers.append(f"{label} bounds origin is outside map bounds")
        return None
    if rect.x + rect.width > width or rect.y + rect.height > height:
        blockers.append(f"{label} bounds extend outside map bounds")
    return CompiledGridRect(
        x=rect.x,
        y=rect.y,
        width=rect.width,
        height=rect.height,
    )


def _validate_authored_ref(
    ref: str,
    authored_refs: set[str] | None,
    context: str,
    blockers: list[str],
) -> None:
    if authored_refs is None:
        return
    if ref not in authored_refs:
        blockers.append(f"{context} references missing content ref {ref!r}")


def _asset_lookup(
    assets: (
        Iterable[ContentImageAsset | Mapping[str, Any]]
        | Mapping[str, ContentImageAsset | Mapping[str, Any]]
        | None
    ),
) -> dict[str, ContentImageAsset] | None:
    if assets is None:
        return None
    values: Iterable[ContentImageAsset | Mapping[str, Any]]
    values = assets.values() if isinstance(assets, Mapping) else assets
    lookup: dict[str, ContentImageAsset] = {}
    for raw in values:
        asset = (
            raw
            if isinstance(raw, ContentImageAsset)
            else ContentImageAsset.model_validate(raw)
        )
        lookup[asset.asset_id] = asset
        lookup[_asset_key(asset.pack_id, asset.asset_id)] = asset
    return lookup


def _asset_key(pack_id: str, asset_id: str) -> str:
    return f"{pack_id.strip()}::{asset_id.strip()}"


def _terrain_zones_for_feature(
    feature: CompiledMapFeature,
) -> list[DndTerrainZone]:
    if feature.secret:
        return []
    if (
        not feature.blocks_movement
        and not feature.blocks_line_of_sight
        and not feature.difficult_terrain
        and feature.cover == "none"
    ):
        return []
    if feature.bounds is not None:
        return [
            _terrain_zone(
                feature,
                feature.feature_id,
                feature.bounds.x,
                feature.bounds.y,
                feature.bounds.width,
                feature.bounds.height,
            )
        ]
    return [
        _terrain_zone(feature, f"{feature.feature_id}.{index}", cell.x, cell.y, 1, 1)
        for index, cell in enumerate(feature.cells, start=1)
    ]


def _terrain_zone(
    feature: CompiledMapFeature,
    zone_id: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> DndTerrainZone:
    return DndTerrainZone(
        zone_id=zone_id,
        label=feature.label or feature.feature_id,
        x=x,
        y=y,
        width=width,
        height=height,
        blocks_movement=feature.blocks_movement,
        blocks_line_of_sight=feature.blocks_line_of_sight,
        cover=feature.cover or "none",
        notes="",
    )


def _runtime_feature(feature: CompiledMapFeature) -> RuntimeBattleMapFeature:
    return RuntimeBattleMapFeature(
        feature_id=feature.feature_id,
        feature_kind=feature.feature_kind,
        cells=[DndMapPoint(x=cell.x, y=cell.y) for cell in feature.cells],
        bounds=_runtime_rect(feature.bounds),
        label=feature.label,
        blocks_movement=feature.blocks_movement,
        blocks_line_of_sight=feature.blocks_line_of_sight,
        difficult_terrain=feature.difficult_terrain,
        cover=feature.cover,
        secret=feature.secret,
        reveal_trigger=feature.reveal_trigger,
        linked_refs=list(feature.linked_refs),
    )


def _runtime_rect(rect: CompiledGridRect | None) -> DndMapRect | None:
    if rect is None:
        return None
    return DndMapRect(
        x=rect.x,
        y=rect.y,
        width=rect.width,
        height=rect.height,
    )
