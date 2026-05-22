from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


MAX_BATTLE_MAP_WIDTH = 80
MAX_BATTLE_MAP_HEIGHT = 80
MAX_BATTLE_MAP_TOKENS = 80
MAX_BATTLE_MAP_TERRAIN = 80
MAX_BATTLE_MAP_AREAS = 40
MAX_BATTLE_MAP_ANCHORS = 40
MAX_BATTLE_MAP_FEATURES = 120
MAX_BATTLE_MAP_AREA_LINKS = 80

TerrainCover = Literal["none", "half", "three_quarters", "total"]
AreaShape = Literal["square", "circle", "cone", "line"]
DndSpawnAnchorKind = Literal[
    "players",
    "enemies",
    "reinforcements",
    "exit",
    "retreat",
    "fallback",
    "other",
]
DndMapFeatureKind = Literal[
    "wall",
    "door",
    "window",
    "pit",
    "cliff",
    "stairs",
    "balcony",
    "furniture",
    "water",
    "cover",
    "difficult_ground",
    "blocked_movement",
    "line_of_sight_blocker",
    "secret_feature",
    "other",
]
SpatialDeltaKind = Literal[
    "move_token",
    "place_token",
    "remove_token",
    "add_area",
    "remove_area",
]
SpatialDeltaShape = Literal["", "square", "circle", "cone", "line"]


class DndBattleMapToken(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_id: str
    character_id: str
    label: str
    x: int
    y: int
    size_squares: int

    @model_validator(mode="before")
    @classmethod
    def _coerce_missing_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("token_id", "")
            data.setdefault("character_id", "")
            data.setdefault("label", "")
            data.setdefault("x", 0)
            data.setdefault("y", 0)
            data.setdefault("size_squares", 1)
        return data

    @model_validator(mode="after")
    def _clean(self) -> "DndBattleMapToken":
        self.token_id = self.token_id.strip()
        self.character_id = self.character_id.strip()
        self.label = self.label.strip()
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.size_squares < 1:
            self.size_squares = 1
        return self


class DndTerrainZone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zone_id: str
    label: str
    x: int
    y: int
    width: int
    height: int
    blocks_movement: bool
    blocks_line_of_sight: bool
    cover: TerrainCover
    notes: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.setdefault("zone_id", "")
        data.setdefault("label", "")
        data.setdefault("x", 0)
        data.setdefault("y", 0)
        data.setdefault("width", 1)
        data.setdefault("height", 1)
        data.setdefault("blocks_movement", False)
        data.setdefault("blocks_line_of_sight", False)
        data.setdefault("cover", "none")
        data.setdefault("notes", "")
        if data.get("cover") not in {"none", "half", "three_quarters", "total"}:
            data["cover"] = "none"
        return data

    @model_validator(mode="after")
    def _clean(self) -> "DndTerrainZone":
        self.zone_id = self.zone_id.strip()
        self.label = self.label.strip()
        self.notes = self.notes.strip()
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.width < 1:
            self.width = 1
        if self.height < 1:
            self.height = 1
        return self


class DndAreaTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str
    label: str
    shape: AreaShape
    x: int
    y: int
    radius_squares: int
    width: int
    height: int
    duration_rounds: int
    notes: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.setdefault("template_id", "")
        data.setdefault("label", "")
        data.setdefault("shape", "square")
        data.setdefault("x", 0)
        data.setdefault("y", 0)
        data.setdefault("radius_squares", 0)
        data.setdefault("width", 1)
        data.setdefault("height", 1)
        data.setdefault("duration_rounds", 0)
        data.setdefault("notes", "")
        if data.get("shape") not in {"square", "circle", "cone", "line"}:
            data["shape"] = "square"
        return data

    @model_validator(mode="after")
    def _clean(self) -> "DndAreaTemplate":
        self.template_id = self.template_id.strip()
        self.label = self.label.strip()
        self.notes = self.notes.strip()
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.radius_squares < 0:
            self.radius_squares = 0
        if self.width < 1:
            self.width = 1
        if self.height < 1:
            self.height = 1
        if self.duration_rounds < 0:
            self.duration_rounds = 0
        return self


class DndMapPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int

    @model_validator(mode="after")
    def _clean(self) -> "DndMapPoint":
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        return self


class DndMapRect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int
    width: int
    height: int

    @model_validator(mode="after")
    def _clean(self) -> "DndMapRect":
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.width < 1:
            self.width = 1
        if self.height < 1:
            self.height = 1
        return self


class DndBattleMapSpawnAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_id: str
    anchor_kind: DndSpawnAnchorKind
    cells: list[DndMapPoint]
    label: str
    linked_ref: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.setdefault("anchor_id", "")
        data.setdefault("anchor_kind", "other")
        data.setdefault("cells", [])
        data.setdefault("label", "")
        data.setdefault("linked_ref", "")
        if data.get("anchor_kind") not in {
            "players",
            "enemies",
            "reinforcements",
            "exit",
            "retreat",
            "fallback",
            "other",
        }:
            data["anchor_kind"] = "other"
        return data

    @model_validator(mode="after")
    def _clean(self) -> "DndBattleMapSpawnAnchor":
        self.anchor_id = self.anchor_id.strip()
        self.label = self.label.strip()
        self.linked_ref = self.linked_ref.strip()
        return self


class DndBattleMapFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_id: str
    feature_kind: DndMapFeatureKind
    cells: list[DndMapPoint]
    bounds: DndMapRect | None
    label: str
    blocks_movement: bool
    blocks_line_of_sight: bool
    difficult_terrain: bool
    cover: TerrainCover
    secret: bool
    reveal_trigger: str
    linked_refs: list[str]

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.setdefault("feature_id", "")
        data.setdefault("feature_kind", "other")
        data.setdefault("cells", [])
        data.setdefault("bounds", None)
        data.setdefault("label", "")
        data.setdefault("blocks_movement", False)
        data.setdefault("blocks_line_of_sight", False)
        data.setdefault("difficult_terrain", False)
        data.setdefault("cover", "none")
        data.setdefault("secret", False)
        data.setdefault("reveal_trigger", "")
        data.setdefault("linked_refs", [])
        if data.get("cover") not in {"none", "half", "three_quarters", "total"}:
            data["cover"] = "none"
        return data

    @model_validator(mode="after")
    def _clean(self) -> "DndBattleMapFeature":
        self.feature_id = self.feature_id.strip()
        self.label = self.label.strip()
        self.reveal_trigger = self.reveal_trigger.strip()
        self.linked_refs = _clean_unique_strings(self.linked_refs)
        if self.secret and not self.reveal_trigger:
            raise ValueError("secret battle-map features need reveal_trigger")
        return self


class DndBattleMapAreaLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area_id: str
    location_ref: str
    cells: list[DndMapPoint]
    bounds: DndMapRect | None

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.setdefault("area_id", "")
        data.setdefault("location_ref", "")
        data.setdefault("cells", [])
        data.setdefault("bounds", None)
        return data

    @model_validator(mode="after")
    def _clean(self) -> "DndBattleMapAreaLink":
        self.area_id = self.area_id.strip()
        self.location_ref = self.location_ref.strip()
        return self


class DndBattleMapState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    present: bool
    map_name: str
    width: int
    height: int
    square_size_ft: int
    tokens: list[DndBattleMapToken]
    terrain: list[DndTerrainZone]
    areas: list[DndAreaTemplate]
    notes: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_missing_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("present", False)
            data.setdefault("map_name", "")
            data.setdefault("width", 0)
            data.setdefault("height", 0)
            data.setdefault("square_size_ft", 5)
            data.setdefault("tokens", [])
            data.setdefault("terrain", [])
            data.setdefault("areas", [])
            data.setdefault("notes", "")
        return data

    @model_validator(mode="after")
    def _clean(self) -> "DndBattleMapState":
        self.map_name = self.map_name.strip()
        self.notes = self.notes.strip()
        if self.width < 0:
            self.width = 0
        if self.width > MAX_BATTLE_MAP_WIDTH:
            self.width = MAX_BATTLE_MAP_WIDTH
        if self.height < 0:
            self.height = 0
        if self.height > MAX_BATTLE_MAP_HEIGHT:
            self.height = MAX_BATTLE_MAP_HEIGHT
        if self.square_size_ft <= 0:
            self.square_size_ft = 5
        self.tokens = self.tokens[:MAX_BATTLE_MAP_TOKENS]
        self.terrain = self.terrain[:MAX_BATTLE_MAP_TERRAIN]
        self.areas = self.areas[:MAX_BATTLE_MAP_AREAS]
        if self.width > 0 and self.height > 0:
            max_size = max(1, min(self.width, self.height))
            for token in self.tokens:
                if token.size_squares > max_size:
                    token.size_squares = max_size
                token.x = min(token.x, max(0, self.width - token.size_squares))
                token.y = min(token.y, max(0, self.height - token.size_squares))
            for zone in self.terrain:
                zone.x = min(zone.x, max(0, self.width - 1))
                zone.y = min(zone.y, max(0, self.height - 1))
                zone.width = max(1, min(zone.width, self.width - zone.x))
                zone.height = max(1, min(zone.height, self.height - zone.y))
            for area in self.areas:
                area.x = min(area.x, max(0, self.width - 1))
                area.y = min(area.y, max(0, self.height - 1))
                area.width = max(1, min(area.width, self.width - area.x))
                area.height = max(1, min(area.height, self.height - area.y))
        return self


class DndBattleMapRuntimeState(DndBattleMapState):
    """Adapter-owned runtime map state for imported D&D tactical geometry."""

    source_template_ref: str
    source_content_hash: str
    orientation: str
    spawn_anchors: list[DndBattleMapSpawnAnchor]
    features: list[DndBattleMapFeature]
    area_links: list[DndBattleMapAreaLink]

    @model_validator(mode="before")
    @classmethod
    def _coerce_runtime_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("source_template_ref", "")
            data.setdefault("source_content_hash", "")
            data.setdefault("orientation", "")
            data.setdefault("spawn_anchors", [])
            data.setdefault("features", [])
            data.setdefault("area_links", [])
        return data

    @model_validator(mode="after")
    def _clean_runtime(self) -> "DndBattleMapRuntimeState":
        self.source_template_ref = self.source_template_ref.strip()
        self.source_content_hash = self.source_content_hash.strip()
        self.orientation = self.orientation.strip()
        self.spawn_anchors = self.spawn_anchors[:MAX_BATTLE_MAP_ANCHORS]
        self.features = self.features[:MAX_BATTLE_MAP_FEATURES]
        self.area_links = self.area_links[:MAX_BATTLE_MAP_AREA_LINKS]

        if self.width > 0 and self.height > 0:
            for anchor in self.spawn_anchors:
                _clamp_points(anchor.cells, self.width, self.height)
            for feature in self.features:
                _clamp_points(feature.cells, self.width, self.height)
                _clamp_rect(feature.bounds, self.width, self.height)
            for link in self.area_links:
                _clamp_points(link.cells, self.width, self.height)
                _clamp_rect(link.bounds, self.width, self.height)
        return self


class DndSpatialDelta(BaseModel):
    """Router-authored tactical map mutation for one combat turn.

    The fixed field set keeps structured output schemas simple. Fields that do
    not apply to an operation are emitted as empty strings or zeroes.
    """

    model_config = ConfigDict(extra="forbid")

    kind: SpatialDeltaKind
    target_id: str
    character_id: str
    x: int
    y: int
    size_squares: int
    label: str
    shape: SpatialDeltaShape
    radius_squares: int
    width: int
    height: int
    duration_rounds: int
    reason: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_missing_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("target_id", "")
            data.setdefault("character_id", "")
            data.setdefault("x", 0)
            data.setdefault("y", 0)
            data.setdefault("size_squares", 1)
            data.setdefault("label", "")
            data.setdefault("shape", "")
            data.setdefault("radius_squares", 0)
            data.setdefault("width", 1)
            data.setdefault("height", 1)
            data.setdefault("duration_rounds", 0)
            data.setdefault("reason", "")
            if data.get("shape") not in {"", "square", "circle", "cone", "line"}:
                data["shape"] = ""
        return data

    @model_validator(mode="after")
    def _clean(self) -> "DndSpatialDelta":
        self.target_id = self.target_id.strip()
        self.character_id = self.character_id.strip()
        self.label = self.label.strip()
        self.reason = self.reason.strip()
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.size_squares < 1:
            self.size_squares = 1
        if self.radius_squares < 0:
            self.radius_squares = 0
        if self.width < 1:
            self.width = 1
        if self.height < 1:
            self.height = 1
        if self.duration_rounds < 0:
            self.duration_rounds = 0
        return self


def empty_battle_map_state() -> dict[str, Any]:
    return {
        "present": False,
        "map_name": "",
        "width": 0,
        "height": 0,
        "square_size_ft": 5,
        "tokens": [],
        "terrain": [],
        "areas": [],
        "notes": "",
    }


def empty_battle_map_runtime_state() -> dict[str, Any]:
    data = empty_battle_map_state()
    data.update({
        "source_template_ref": "",
        "source_content_hash": "",
        "orientation": "",
        "spawn_anchors": [],
        "features": [],
        "area_links": [],
    })
    return data


def _clean_unique_strings(values: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        )
    )


def _clamp_points(points: list[DndMapPoint], width: int, height: int) -> None:
    for point in points:
        point.x = min(point.x, max(0, width - 1))
        point.y = min(point.y, max(0, height - 1))


def _clamp_rect(rect: DndMapRect | None, width: int, height: int) -> None:
    if rect is None:
        return
    rect.x = min(rect.x, max(0, width - 1))
    rect.y = min(rect.y, max(0, height - 1))
    rect.width = max(1, min(rect.width, width - rect.x))
    rect.height = max(1, min(rect.height, height - rect.y))
