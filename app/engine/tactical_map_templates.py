from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from app.schemas.content_pack import (
    ContentImageAsset,
    TacticalMapAreaLink,
    TacticalMapFeature,
    TacticalMapFogMask,
    TacticalMapFloor,
    TacticalMapRevealRegion,
    TacticalMapSpawnAnchor,
    TacticalMapTemplateRecord,
)
from app.schemas.content_privacy import contains_imported_asset_sentinel
from app.schemas.dnd_spatial import (
    DndBattleMapAreaLink,
    DndBattleMapFeature,
    DndBattleMapFloor,
    DndBattleMapFogMask,
    DndBattleMapRevealRegion,
    DndBattleMapState,
    DndBattleMapSpawnAnchor,
    DndMapPoint,
    DndMapRect,
    DndTerrainZone,
    MAX_BATTLE_MAP_HEIGHT,
    MAX_BATTLE_MAP_WIDTH,
)


REVIEW_READY_STATUSES = {"reviewed", "approved"}
_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]*$")
_TERRAIN_COVER = {"none", "half", "three_quarters", "total"}
_VERTICAL_FEATURE_KINDS = {"stairs"}

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
    floor_id: str
    cells: tuple[CompiledGridPoint, ...]
    label: str
    linked_ref: str


@dataclass(frozen=True, slots=True)
class CompiledMapFeature:
    feature_id: str
    feature_kind: str
    floor_id: str
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
    floor_id: str
    cells: tuple[CompiledGridPoint, ...]
    bounds: CompiledGridRect | None


@dataclass(frozen=True, slots=True)
class CompiledMapFloor:
    floor_id: str
    floor_kind: str
    label: str
    grid_width: int
    grid_height: int
    square_size_ft: int
    level_index: int
    elevation_ft: int
    parent_floor_id: str
    parent_bounds: CompiledGridRect | None
    map_asset_id: str
    area_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledFogMask:
    mask_id: str
    floor_id: str
    cells: tuple[CompiledGridPoint, ...]
    bounds: CompiledGridRect | None
    label: str
    area_refs: tuple[str, ...]
    revealed_by_region_refs: tuple[str, ...]
    initially_hidden: bool


@dataclass(frozen=True, slots=True)
class CompiledRevealRegion:
    reveal_id: str
    floor_id: str
    cells: tuple[CompiledGridPoint, ...]
    bounds: CompiledGridRect | None
    label: str
    reveal_trigger: str
    audience: str
    initially_revealed: bool
    pov_character_ids: tuple[str, ...]
    pov_area_refs: tuple[str, ...]
    revealed_area_refs: tuple[str, ...]
    fog_mask_refs: tuple[str, ...]


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
    floors: tuple[CompiledMapFloor, ...]
    spawn_anchors: tuple[CompiledSpawnAnchor, ...]
    terrain_features: tuple[CompiledMapFeature, ...]
    area_links: tuple[CompiledAreaLink, ...]
    fog_masks: tuple[CompiledFogMask, ...]
    reveal_regions: tuple[CompiledRevealRegion, ...]

    @property
    def secret_features(self) -> tuple[CompiledMapFeature, ...]:
        return tuple(feature for feature in self.terrain_features if feature.secret)

    @property
    def vertical_links(self) -> tuple[CompiledMapFeature, ...]:
        return tuple(
            feature for feature in self.terrain_features if feature.is_vertical_link
        )

    def to_battle_map_state(self) -> DndBattleMapState:
        """Compile reviewed imported geometry into runtime combat-map state."""
        terrain: list[DndTerrainZone] = []
        for feature in self.terrain_features:
            terrain.extend(_terrain_zones_for_feature(feature))

        return DndBattleMapState(
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
                DndBattleMapSpawnAnchor(
                    anchor_id=anchor.anchor_id,
                    anchor_kind=anchor.anchor_kind,
                    floor_id=anchor.floor_id,
                    cells=[DndMapPoint(x=cell.x, y=cell.y) for cell in anchor.cells],
                    label=anchor.label,
                    linked_ref=anchor.linked_ref,
                )
                for anchor in self.spawn_anchors
            ],
            features=[_runtime_feature(feature) for feature in self.terrain_features],
            area_links=[
                DndBattleMapAreaLink(
                    area_id=link.area_id,
                    location_ref=link.location_ref,
                    floor_id=link.floor_id,
                    cells=[DndMapPoint(x=cell.x, y=cell.y) for cell in link.cells],
                    bounds=_runtime_rect(link.bounds),
                )
                for link in self.area_links
            ],
            floors=[
                DndBattleMapFloor(
                    floor_id=floor.floor_id,
                    floor_kind=floor.floor_kind,
                    label=floor.label,
                    grid_width=floor.grid_width,
                    grid_height=floor.grid_height,
                    square_size_ft=floor.square_size_ft,
                    level_index=floor.level_index,
                    elevation_ft=floor.elevation_ft,
                    parent_floor_id=floor.parent_floor_id,
                    parent_bounds=_runtime_rect(floor.parent_bounds),
                    map_asset_id=floor.map_asset_id,
                    area_refs=list(floor.area_refs),
                )
                for floor in self.floors
            ],
            fog_masks=[
                DndBattleMapFogMask(
                    mask_id=mask.mask_id,
                    floor_id=mask.floor_id,
                    cells=[DndMapPoint(x=cell.x, y=cell.y) for cell in mask.cells],
                    bounds=_runtime_rect(mask.bounds),
                    label=mask.label,
                    area_refs=list(mask.area_refs),
                    revealed_by_region_refs=list(mask.revealed_by_region_refs),
                    initially_hidden=mask.initially_hidden,
                )
                for mask in self.fog_masks
            ],
            reveal_regions=[
                DndBattleMapRevealRegion(
                    reveal_id=region.reveal_id,
                    floor_id=region.floor_id,
                    cells=[
                        DndMapPoint(x=cell.x, y=cell.y) for cell in region.cells
                    ],
                    bounds=_runtime_rect(region.bounds),
                    label=region.label,
                    reveal_trigger=region.reveal_trigger,
                    audience=region.audience,
                    initially_revealed=region.initially_revealed,
                    pov_character_ids=list(region.pov_character_ids),
                    pov_area_refs=list(region.pov_area_refs),
                    revealed_area_refs=list(region.revealed_area_refs),
                    fog_mask_refs=list(region.fog_mask_refs),
                )
                for region in self.reveal_regions
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

    floors = _compile_floors(
        record.floors,
        record.grid_width,
        record.grid_height,
        record.square_size_ft,
        record.pack_id,
        asset_lookup,
        authored_ref_set,
        blockers,
    )
    floor_lookup = {floor.floor_id: floor for floor in floors}
    spawn_anchors = _compile_spawn_anchors(
        record.spawn_anchors,
        record.grid_width,
        record.grid_height,
        floor_lookup,
        authored_ref_set,
        blockers,
    )
    terrain_features = _compile_features(
        record.terrain_features,
        record.grid_width,
        record.grid_height,
        floor_lookup,
        authored_ref_set,
        blockers,
    )
    area_links = _compile_area_links(
        record.area_links,
        record.grid_width,
        record.grid_height,
        floor_lookup,
        authored_ref_set,
        blockers,
    )
    fog_masks = _compile_fog_masks(
        record.fog_masks,
        record.grid_width,
        record.grid_height,
        floor_lookup,
        authored_ref_set,
        blockers,
    )
    reveal_regions = _compile_reveal_regions(
        record.reveal_regions,
        record.grid_width,
        record.grid_height,
        floor_lookup,
        authored_ref_set,
        blockers,
    )
    _validate_reveal_cross_refs(record.fog_masks, record.reveal_regions, blockers)

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
        floors=floors,
        spawn_anchors=spawn_anchors,
        terrain_features=terrain_features,
        area_links=area_links,
        fog_masks=fog_masks,
        reveal_regions=reveal_regions,
    )


def compile_tactical_map_template_battle_map_state(
    template: TacticalMapTemplateRecord | Mapping[str, Any],
    **kwargs: Any,
) -> DndBattleMapState:
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
    if record.target_runtime_schema != "DndBattleMapState":
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
        if layer == "map_ref" and not record.derived_from_map_asset_id:
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
        elif layer == "floors_submaps" and not record.floors:
            blockers.append("required floors/submaps layer is missing")
        elif layer == "fog_reveal_regions":
            if not record.fog_masks:
                blockers.append("required fog-mask layer is missing")
            if not record.reveal_regions:
                blockers.append("required reveal-region layer is missing")


def _validate_grid(
    record: TacticalMapTemplateRecord,
    blockers: list[str],
) -> None:
    if record.grid_width > MAX_BATTLE_MAP_WIDTH:
        blockers.append(
            f"grid_width {record.grid_width} exceeds DndBattleMapState cap "
            f"{MAX_BATTLE_MAP_WIDTH}"
        )
    if record.grid_height > MAX_BATTLE_MAP_HEIGHT:
        blockers.append(
            f"grid_height {record.grid_height} exceeds DndBattleMapState cap "
            f"{MAX_BATTLE_MAP_HEIGHT}"
        )


def _validate_map_asset(
    record: TacticalMapTemplateRecord,
    asset_lookup: dict[str, ContentImageAsset] | None,
    blockers: list[str],
) -> None:
    _validate_map_asset_id(
        record.pack_id,
        record.derived_from_map_asset_id,
        asset_lookup,
        "derived_from_map_asset_id",
        blockers,
    )


def _validate_map_asset_id(
    pack_id: str,
    asset_id: str,
    asset_lookup: dict[str, ContentImageAsset] | None,
    context: str,
    blockers: list[str],
) -> None:
    if not asset_id:
        return
    if not _ASSET_ID_RE.fullmatch(asset_id):
        blockers.append(
            f"{context} {asset_id!r} must be a logical asset id, "
            "not a path, URL, source page, OCR block, or image payload"
        )
        return
    if asset_lookup is None:
        return

    asset = asset_lookup.get(_asset_key(pack_id, asset_id)) or asset_lookup.get(
        asset_id
    )
    if asset is None:
        blockers.append(f"map asset {asset_id!r} is missing from the asset catalog")
        return
    if pack_id and asset.pack_id and asset.pack_id != pack_id:
        blockers.append(
            f"map asset {asset_id!r} belongs to pack {asset.pack_id!r}, "
            f"not {pack_id!r}"
        )
    if asset.review_status not in REVIEW_READY_STATUSES:
        blockers.append(
            f"map asset {asset_id!r} review_status {asset.review_status!r} "
            "is not reviewed or approved"
        )


def _compile_floors(
    floors: Iterable[TacticalMapFloor],
    default_width: int,
    default_height: int,
    default_square_size_ft: int,
    pack_id: str,
    asset_lookup: dict[str, ContentImageAsset] | None,
    authored_refs: set[str] | None,
    blockers: list[str],
) -> tuple[CompiledMapFloor, ...]:
    floor_records = tuple(floors)
    floor_by_id = {
        floor.floor_id: floor for floor in floor_records if floor.floor_id
    }
    compiled: list[CompiledMapFloor] = []
    seen: set[str] = set()
    for floor in floor_records:
        context = f"map floor {floor.floor_id!r}" if floor.floor_id else "map floor"
        if not floor.floor_id:
            blockers.append("map floor id is empty")
            continue
        _validate_runtime_id(floor.floor_id, "map floor id", blockers)
        if floor.floor_id in seen:
            blockers.append(f"duplicate map floor id {floor.floor_id!r}")
        seen.add(floor.floor_id)
        _validate_runtime_text(floor.label, f"{context} label", blockers)
        if floor.grid_width > MAX_BATTLE_MAP_WIDTH:
            blockers.append(
                f"{context} grid_width {floor.grid_width} exceeds "
                f"DndBattleMapState cap {MAX_BATTLE_MAP_WIDTH}"
            )
        if floor.grid_height > MAX_BATTLE_MAP_HEIGHT:
            blockers.append(
                f"{context} grid_height {floor.grid_height} exceeds "
                f"DndBattleMapState cap {MAX_BATTLE_MAP_HEIGHT}"
            )
        if floor.map_asset_id:
            _validate_map_asset_id(
                pack_id,
                floor.map_asset_id,
                asset_lookup,
                f"{context} map_asset_id",
                blockers,
            )
        for area_ref in floor.area_refs:
            _validate_authored_ref(area_ref, authored_refs, context, blockers)
        if floor.parent_floor_id:
            _validate_runtime_id(
                floor.parent_floor_id,
                f"{context} parent_floor_id",
                blockers,
            )
            if floor.parent_floor_id == floor.floor_id:
                blockers.append(f"{context} cannot parent itself")
            elif floor.parent_floor_id not in floor_by_id:
                blockers.append(
                    f"{context} references missing parent floor "
                    f"{floor.parent_floor_id!r}"
                )
        if floor.floor_kind == "submap":
            if not floor.parent_floor_id:
                blockers.append(f"submap {floor.floor_id!r} needs parent_floor_id")
            if floor.parent_bounds is None:
                blockers.append(f"submap {floor.floor_id!r} needs parent_bounds")
        parent_width, parent_height = _floor_record_dimensions(
            floor.parent_floor_id,
            floor_by_id,
            default_width,
            default_height,
        )
        parent_bounds = _compile_rect(
            floor.parent_bounds,
            parent_width,
            parent_height,
            f"{context} parent_bounds",
            blockers,
        )
        compiled.append(
            CompiledMapFloor(
                floor_id=floor.floor_id,
                floor_kind=floor.floor_kind,
                label=floor.label,
                grid_width=floor.grid_width,
                grid_height=floor.grid_height,
                square_size_ft=floor.square_size_ft or default_square_size_ft,
                level_index=floor.level_index,
                elevation_ft=floor.elevation_ft,
                parent_floor_id=floor.parent_floor_id,
                parent_bounds=parent_bounds,
                map_asset_id=floor.map_asset_id,
                area_refs=tuple(floor.area_refs),
            )
        )
    return tuple(compiled)


def _compile_spawn_anchors(
    anchors: Iterable[TacticalMapSpawnAnchor],
    width: int,
    height: int,
    floors: Mapping[str, CompiledMapFloor],
    authored_refs: set[str] | None,
    blockers: list[str],
) -> tuple[CompiledSpawnAnchor, ...]:
    compiled: list[CompiledSpawnAnchor] = []
    seen: set[str] = set()
    for anchor in anchors:
        if not anchor.anchor_id:
            blockers.append("spawn anchor id is empty")
            continue
        _validate_runtime_id(anchor.anchor_id, "spawn anchor id", blockers)
        if anchor.anchor_id in seen:
            blockers.append(f"duplicate spawn anchor id {anchor.anchor_id!r}")
        seen.add(anchor.anchor_id)
        _validate_floor_id(
            anchor.floor_id,
            floors,
            f"spawn anchor {anchor.anchor_id!r}",
            blockers,
        )
        anchor_width, anchor_height = _geometry_dimensions(
            anchor.floor_id,
            floors,
            width,
            height,
        )
        _validate_runtime_text(
            anchor.label,
            f"spawn anchor {anchor.anchor_id!r} label",
            blockers,
        )
        if not anchor.cells:
            blockers.append(f"spawn anchor {anchor.anchor_id!r} has no cells")
        cells = _compile_points(
            anchor.cells,
            anchor_width,
            anchor_height,
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
                floor_id=anchor.floor_id,
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
    floors: Mapping[str, CompiledMapFloor],
    authored_refs: set[str] | None,
    blockers: list[str],
) -> tuple[CompiledMapFeature, ...]:
    compiled: list[CompiledMapFeature] = []
    seen: set[str] = set()
    for feature in features:
        if not feature.feature_id:
            blockers.append("terrain feature id is empty")
            continue
        _validate_runtime_id(feature.feature_id, "terrain feature id", blockers)
        if feature.feature_id in seen:
            blockers.append(f"duplicate terrain feature id {feature.feature_id!r}")
        seen.add(feature.feature_id)
        _validate_floor_id(
            feature.floor_id,
            floors,
            f"terrain feature {feature.feature_id!r}",
            blockers,
        )
        feature_width, feature_height = _geometry_dimensions(
            feature.floor_id,
            floors,
            width,
            height,
        )
        _validate_runtime_text(
            feature.label,
            f"terrain feature {feature.feature_id!r} label",
            blockers,
        )
        _validate_runtime_text(
            feature.reveal_trigger,
            f"terrain feature {feature.feature_id!r} reveal_trigger",
            blockers,
        )
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
            feature_width,
            feature_height,
            f"terrain feature {feature.feature_id!r}",
            blockers,
        )
        bounds = _compile_rect(
            feature.bounds,
            feature_width,
            feature_height,
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
                floor_id=feature.floor_id,
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
    floors: Mapping[str, CompiledMapFloor],
    authored_refs: set[str] | None,
    blockers: list[str],
) -> tuple[CompiledAreaLink, ...]:
    compiled: list[CompiledAreaLink] = []
    seen: set[str] = set()
    for link in links:
        if not link.area_id:
            blockers.append("keyed-area link id is empty")
            continue
        _validate_runtime_id(link.area_id, "keyed-area link id", blockers)
        if link.area_id in seen:
            blockers.append(f"duplicate keyed-area link id {link.area_id!r}")
        seen.add(link.area_id)
        _validate_floor_id(
            link.floor_id,
            floors,
            f"keyed-area link {link.area_id!r}",
            blockers,
        )
        link_width, link_height = _geometry_dimensions(
            link.floor_id,
            floors,
            width,
            height,
        )
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
            link_width,
            link_height,
            f"keyed-area link {link.area_id!r}",
            blockers,
        )
        bounds = _compile_rect(
            link.bounds,
            link_width,
            link_height,
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
                floor_id=link.floor_id,
                cells=cells,
                bounds=bounds,
            )
        )
    return tuple(compiled)


def _compile_fog_masks(
    masks: Iterable[TacticalMapFogMask],
    width: int,
    height: int,
    floors: Mapping[str, CompiledMapFloor],
    authored_refs: set[str] | None,
    blockers: list[str],
) -> tuple[CompiledFogMask, ...]:
    compiled: list[CompiledFogMask] = []
    seen: set[str] = set()
    for mask in masks:
        if not mask.mask_id:
            blockers.append("fog mask id is empty")
            continue
        context = f"fog mask {mask.mask_id!r}"
        _validate_runtime_id(mask.mask_id, "fog mask id", blockers)
        if mask.mask_id in seen:
            blockers.append(f"duplicate fog mask id {mask.mask_id!r}")
        seen.add(mask.mask_id)
        _validate_floor_id(mask.floor_id, floors, context, blockers)
        mask_width, mask_height = _geometry_dimensions(
            mask.floor_id,
            floors,
            width,
            height,
        )
        _validate_runtime_text(mask.label, f"{context} label", blockers)
        _validate_region_shape(mask.cells, mask.bounds, context, blockers)
        cells = _compile_points(mask.cells, mask_width, mask_height, context, blockers)
        bounds = _compile_rect(mask.bounds, mask_width, mask_height, context, blockers)
        for area_ref in mask.area_refs:
            _validate_authored_ref(area_ref, authored_refs, context, blockers)
        for reveal_ref in mask.revealed_by_region_refs:
            _validate_runtime_id(
                reveal_ref,
                f"{context} revealed_by_region_refs",
                blockers,
            )
        compiled.append(
            CompiledFogMask(
                mask_id=mask.mask_id,
                floor_id=mask.floor_id,
                cells=cells,
                bounds=bounds,
                label=mask.label,
                area_refs=tuple(mask.area_refs),
                revealed_by_region_refs=tuple(mask.revealed_by_region_refs),
                initially_hidden=mask.initially_hidden,
            )
        )
    return tuple(compiled)


def _compile_reveal_regions(
    regions: Iterable[TacticalMapRevealRegion],
    width: int,
    height: int,
    floors: Mapping[str, CompiledMapFloor],
    authored_refs: set[str] | None,
    blockers: list[str],
) -> tuple[CompiledRevealRegion, ...]:
    compiled: list[CompiledRevealRegion] = []
    seen: set[str] = set()
    for region in regions:
        if not region.reveal_id:
            blockers.append("reveal region id is empty")
            continue
        context = f"reveal region {region.reveal_id!r}"
        _validate_runtime_id(region.reveal_id, "reveal region id", blockers)
        if region.reveal_id in seen:
            blockers.append(f"duplicate reveal region id {region.reveal_id!r}")
        seen.add(region.reveal_id)
        _validate_floor_id(region.floor_id, floors, context, blockers)
        region_width, region_height = _geometry_dimensions(
            region.floor_id,
            floors,
            width,
            height,
        )
        _validate_runtime_text(region.label, f"{context} label", blockers)
        _validate_runtime_text(
            region.reveal_trigger,
            f"{context} reveal_trigger",
            blockers,
        )
        if not region.initially_revealed and not region.reveal_trigger:
            blockers.append(f"{context} needs reveal_trigger")
        _validate_region_shape(region.cells, region.bounds, context, blockers)
        cells = _compile_points(
            region.cells,
            region_width,
            region_height,
            context,
            blockers,
        )
        bounds = _compile_rect(
            region.bounds,
            region_width,
            region_height,
            context,
            blockers,
        )
        for character_id in region.pov_character_ids:
            _validate_runtime_id(
                character_id,
                f"{context} pov_character_ids",
                blockers,
            )
        for area_ref in [*region.pov_area_refs, *region.revealed_area_refs]:
            _validate_authored_ref(area_ref, authored_refs, context, blockers)
        for mask_ref in region.fog_mask_refs:
            _validate_runtime_id(mask_ref, f"{context} fog_mask_refs", blockers)
        compiled.append(
            CompiledRevealRegion(
                reveal_id=region.reveal_id,
                floor_id=region.floor_id,
                cells=cells,
                bounds=bounds,
                label=region.label,
                reveal_trigger=region.reveal_trigger,
                audience=region.audience,
                initially_revealed=region.initially_revealed,
                pov_character_ids=tuple(region.pov_character_ids),
                pov_area_refs=tuple(region.pov_area_refs),
                revealed_area_refs=tuple(region.revealed_area_refs),
                fog_mask_refs=tuple(region.fog_mask_refs),
            )
        )
    return tuple(compiled)


def _validate_reveal_cross_refs(
    fog_masks: Iterable[TacticalMapFogMask],
    reveal_regions: Iterable[TacticalMapRevealRegion],
    blockers: list[str],
) -> None:
    fog_mask_ids = {mask.mask_id for mask in fog_masks if mask.mask_id}
    reveal_ids = {region.reveal_id for region in reveal_regions if region.reveal_id}
    for mask in fog_masks:
        for reveal_ref in mask.revealed_by_region_refs:
            if reveal_ref not in reveal_ids:
                blockers.append(
                    f"fog mask {mask.mask_id!r} references missing reveal region "
                    f"{reveal_ref!r}"
                )
    for region in reveal_regions:
        for mask_ref in region.fog_mask_refs:
            if mask_ref not in fog_mask_ids:
                blockers.append(
                    f"reveal region {region.reveal_id!r} references missing "
                    f"fog mask {mask_ref!r}"
                )


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


def _validate_region_shape(
    cells: Sequence[Any],
    bounds: Any | None,
    context: str,
    blockers: list[str],
) -> None:
    if not cells and bounds is None:
        blockers.append(f"{context} has no geometry")
    if cells and bounds is not None:
        blockers.append(f"{context} must use cells or bounds, not both")


def _validate_floor_id(
    floor_id: str,
    floors: Mapping[str, CompiledMapFloor],
    context: str,
    blockers: list[str],
) -> None:
    if not floor_id:
        return
    _validate_runtime_id(floor_id, f"{context} floor_id", blockers)
    if floor_id not in floors:
        blockers.append(f"{context} references missing map floor {floor_id!r}")


def _geometry_dimensions(
    floor_id: str,
    floors: Mapping[str, CompiledMapFloor],
    width: int,
    height: int,
) -> tuple[int, int]:
    if floor_id and floor_id in floors:
        floor = floors[floor_id]
        return floor.grid_width, floor.grid_height
    return width, height


def _floor_record_dimensions(
    floor_id: str,
    floors: Mapping[str, TacticalMapFloor],
    width: int,
    height: int,
) -> tuple[int, int]:
    if floor_id and floor_id in floors:
        floor = floors[floor_id]
        return floor.grid_width, floor.grid_height
    return width, height


def _validate_runtime_id(
    value: str,
    context: str,
    blockers: list[str],
) -> None:
    if not value:
        return
    if not _RUNTIME_ID_RE.fullmatch(value):
        blockers.append(
            f"{context} {value!r} must be a runtime-safe id, not a path, URL, "
            "source page, OCR block, or image payload"
        )


def _validate_runtime_text(
    value: str,
    context: str,
    blockers: list[str],
) -> None:
    if value and contains_imported_asset_sentinel(value):
        blockers.append(
            f"{context} must not contain raw source paths, image payloads, "
            "asset URLs, or source metadata"
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
        floor_id=feature.floor_id,
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


def _runtime_feature(feature: CompiledMapFeature) -> DndBattleMapFeature:
    return DndBattleMapFeature(
        feature_id=feature.feature_id,
        feature_kind=feature.feature_kind,
        floor_id=feature.floor_id,
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
