from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


TerrainCover = Literal["none", "half", "three_quarters", "total"]
AreaShape = Literal["square", "circle", "cone", "line"]
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
        if self.height < 0:
            self.height = 0
        if self.square_size_ft <= 0:
            self.square_size_ft = 5
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
