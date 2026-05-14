from __future__ import annotations

from typing import Any, Iterable

from app.schemas.dnd_spatial import (
    DndAreaTemplate,
    DndBattleMapState,
    DndBattleMapToken,
    DndSpatialDelta,
    DndTerrainZone,
)


def normalize_battle_map_seed(
    seed: DndBattleMapState | dict[str, Any] | None,
    combatants: Iterable[Any],
) -> DndBattleMapState | None:
    """Validate a router-seeded map and fill missing participant tokens."""
    if seed is None:
        return None
    battle_map = (
        seed if isinstance(seed, DndBattleMapState)
        else DndBattleMapState.model_validate(seed)
    )
    if not battle_map.present:
        return None

    participants = _participant_records(combatants)
    if not participants:
        return None

    width = battle_map.width if battle_map.width > 0 else 12
    height = battle_map.height if battle_map.height > 0 else 12
    participant_by_id = {
        value: participant
        for participant in participants
        for value in (participant["combatant_id"], participant["character_id"])
        if value
    }

    tokens_by_character: dict[str, DndBattleMapToken] = {}
    for raw in battle_map.tokens:
        token_id = raw.token_id or raw.character_id
        participant = participant_by_id.get(raw.character_id) or participant_by_id.get(
            token_id
        )
        if participant is None:
            continue
        character_id = participant["character_id"] or participant["combatant_id"]
        token = DndBattleMapToken(
            token_id=token_id or participant["combatant_id"] or character_id,
            character_id=character_id,
            label=raw.label or participant["name"] or character_id,
            x=_clamp(raw.x, 0, max(0, width - 1)),
            y=_clamp(raw.y, 0, max(0, height - 1)),
            size_squares=max(1, raw.size_squares),
        )
        tokens_by_character[character_id] = token

    occupied = {(token.x, token.y) for token in tokens_by_character.values()}
    for index, participant in enumerate(participants):
        character_id = participant["character_id"] or participant["combatant_id"]
        if not character_id or character_id in tokens_by_character:
            continue
        x, y = _fallback_position(index, width, height, occupied)
        occupied.add((x, y))
        tokens_by_character[character_id] = DndBattleMapToken(
            token_id=participant["combatant_id"] or character_id,
            character_id=character_id,
            label=participant["name"] or character_id,
            x=x,
            y=y,
            size_squares=1,
        )

    terrain = [
        _clamp_terrain(zone, width, height)
        for zone in battle_map.terrain
        if zone.zone_id or zone.label
    ]
    areas = [
        _clamp_area(area, width, height)
        for area in battle_map.areas
        if area.template_id or area.label
    ]

    return DndBattleMapState(
        present=True,
        map_name=battle_map.map_name or "Battle map",
        width=width,
        height=height,
        square_size_ft=battle_map.square_size_ft or 5,
        tokens=list(tokens_by_character.values()),
        terrain=terrain,
        areas=areas,
        notes=battle_map.notes,
    )


def combat_packet_context(combat: Any, actor_id: str) -> dict[str, Any]:
    battle_map = _battle_map(combat)
    if battle_map is None:
        return {
            "battle_map": {},
            "spatial_advisories": [],
        }
    return {
        "battle_map": battle_map.model_dump(),
        "spatial_advisories": spatial_advisories(combat, actor_id),
    }


def spatial_advisories(combat: Any, actor_id: str) -> list[dict[str, Any]]:
    battle_map = _battle_map(combat)
    if battle_map is None:
        return []
    actor_token = _token_for(battle_map, actor_id)
    if actor_token is None:
        return []

    advisories: list[dict[str, Any]] = []
    for token in battle_map.tokens:
        if token is actor_token:
            continue
        distance_ft = _distance_ft(battle_map, actor_token, token)
        line_of_sight = _line_of_sight_clear(battle_map, actor_token, token)
        cover = _cover_between(battle_map, actor_token, token)
        advisories.append({
            "from": actor_token.character_id or actor_token.token_id,
            "to": token.character_id or token.token_id,
            "distance_ft": distance_ft,
            "within_5_ft": distance_ft <= 5,
            "within_30_ft": distance_ft <= 30,
            "line_of_sight": "clear" if line_of_sight else "blocked",
            "cover": cover,
        })
    return advisories


def apply_spatial_deltas(
    combat: Any,
    deltas: Iterable[DndSpatialDelta],
) -> list[str]:
    deltas = list(deltas)
    battle_map = _battle_map(combat)
    if battle_map is None:
        return (
            ["Spatial deltas skipped: no active battle map."]
            if deltas else []
        )

    notes: list[str] = []
    for delta in deltas:
        if delta.kind in {"move_token", "place_token"}:
            token = (
                _token_for(battle_map, delta.target_id)
                if delta.target_id else None
            )
            if token is None and delta.character_id:
                token = _token_for(battle_map, delta.character_id)
            if token is None and delta.kind == "place_token":
                token = DndBattleMapToken(
                    token_id=delta.target_id or delta.character_id,
                    character_id=delta.character_id or delta.target_id,
                    label=delta.label or delta.character_id or delta.target_id,
                    x=0,
                    y=0,
                    size_squares=delta.size_squares,
                )
                battle_map.tokens.append(token)
            if token is None:
                notes.append(
                    f"Spatial delta skipped: token {delta.target_id!r} not found."
                )
                continue
            token.x = _clamp(delta.x, 0, max(0, battle_map.width - 1))
            token.y = _clamp(delta.y, 0, max(0, battle_map.height - 1))
            token.size_squares = max(1, delta.size_squares)
            if delta.label:
                token.label = delta.label
            notes.append(
                f"Moved {token.character_id or token.token_id} to "
                f"({token.x}, {token.y})."
            )
        elif delta.kind == "remove_token":
            before = len(battle_map.tokens)
            battle_map.tokens = [
                token for token in battle_map.tokens
                if delta.target_id not in {
                    token.token_id,
                    token.character_id,
                    token.label,
                }
            ]
            if len(battle_map.tokens) == before:
                notes.append(
                    f"Spatial delta skipped: token {delta.target_id!r} not found."
                )
        elif delta.kind == "add_area":
            template_id = delta.target_id or _slug(delta.label) or "area"
            battle_map.areas = [
                area for area in battle_map.areas
                if area.template_id != template_id
            ]
            battle_map.areas.append(_clamp_area(
                DndAreaTemplate(
                    template_id=template_id,
                    label=delta.label or template_id,
                    shape=delta.shape or "square",
                    x=delta.x,
                    y=delta.y,
                    radius_squares=delta.radius_squares,
                    width=delta.width,
                    height=delta.height,
                    duration_rounds=delta.duration_rounds,
                    notes=delta.reason,
                ),
                battle_map.width,
                battle_map.height,
            ))
        elif delta.kind == "remove_area":
            battle_map.areas = [
                area for area in battle_map.areas
                if delta.target_id not in {area.template_id, area.label}
            ]
    return notes


def battle_map_status(combat: Any) -> dict[str, Any]:
    battle_map = _battle_map(combat)
    return battle_map.model_dump() if battle_map is not None else {}


def render_battle_map_summary(
    combat_or_map: Any,
    *,
    actor_id: str = "",
    max_lines: int = 8,
) -> list[str]:
    battle_map = (
        combat_or_map
        if isinstance(combat_or_map, DndBattleMapState)
        else _battle_map(combat_or_map)
    )
    if battle_map is None:
        return []

    lines = [
        "Battle map: "
        f"{battle_map.map_name or 'Battle map'} "
        f"({battle_map.width}x{battle_map.height}, "
        f"{battle_map.square_size_ft} ft squares)."
    ]
    if battle_map.tokens:
        positions = [
            f"{token.label or token.character_id or token.token_id} "
            f"({token.character_id or token.token_id}) at "
            f"({token.x}, {token.y})"
            for token in battle_map.tokens
        ]
        lines.append("Positions: " + "; ".join(positions) + ".")
    if battle_map.terrain:
        terrain = [
            f"{zone.label or zone.zone_id} at ({zone.x}, {zone.y}) "
            f"{zone.width}x{zone.height}"
            + (f", cover {zone.cover}" if zone.cover != "none" else "")
            + (", blocks sight" if zone.blocks_line_of_sight else "")
            for zone in battle_map.terrain[:4]
        ]
        lines.append("Terrain: " + "; ".join(terrain) + ".")
    if battle_map.areas:
        areas = [
            f"{area.label or area.template_id} {area.shape} at "
            f"({area.x}, {area.y})"
            for area in battle_map.areas[:4]
        ]
        lines.append("Areas: " + "; ".join(areas) + ".")
    if actor_id:
        advisory_text = [
            f"{item['to']} {item['distance_ft']} ft, "
            f"{item['line_of_sight']} sight"
            + (f", cover {item['cover']}" if item["cover"] != "none" else "")
            for item in spatial_advisories(combat_or_map, actor_id)[:4]
        ]
        if advisory_text:
            lines.append("From you: " + "; ".join(advisory_text) + ".")
    return lines[:max_lines]


def _battle_map(combat: Any) -> DndBattleMapState | None:
    if combat is None:
        return None
    if isinstance(combat, DndBattleMapState):
        return combat if combat.present else None
    if isinstance(combat, dict):
        raw = combat if "present" in combat and "tokens" in combat else combat.get(
            "battle_map"
        )
    else:
        raw = getattr(combat, "battle_map", None)
    if not raw:
        return None
    battle_map = (
        raw if isinstance(raw, DndBattleMapState)
        else DndBattleMapState.model_validate(raw)
    )
    return battle_map if battle_map.present else None


def _participant_records(combatants: Iterable[Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for combatant in combatants:
        combatant_id = str(_obj_get(combatant, "combatant_id", "") or "")
        character_id = str(_obj_get(combatant, "character_id", "") or combatant_id)
        name = str(_obj_get(combatant, "name", "") or character_id or combatant_id)
        if character_id or combatant_id:
            records.append({
                "combatant_id": combatant_id,
                "character_id": character_id,
                "name": name,
            })
    return records


def _token_for(
    battle_map: DndBattleMapState,
    target_id: str,
) -> DndBattleMapToken | None:
    target = str(target_id or "").strip()
    if not target:
        return None
    for token in battle_map.tokens:
        if target in {token.token_id, token.character_id, token.label}:
            return token
    return None


def _distance_ft(
    battle_map: DndBattleMapState,
    a: DndBattleMapToken,
    b: DndBattleMapToken,
) -> int:
    squares = max(abs(a.x - b.x), abs(a.y - b.y))
    return squares * battle_map.square_size_ft


def _line_of_sight_clear(
    battle_map: DndBattleMapState,
    a: DndBattleMapToken,
    b: DndBattleMapToken,
) -> bool:
    blockers = [zone for zone in battle_map.terrain if zone.blocks_line_of_sight]
    if not blockers:
        return True
    for x, y in _line_points(a.x, a.y, b.x, b.y):
        if (x, y) in {(a.x, a.y), (b.x, b.y)}:
            continue
        if any(_point_in_zone(x, y, zone) for zone in blockers):
            return False
    return True


def _cover_between(
    battle_map: DndBattleMapState,
    a: DndBattleMapToken,
    b: DndBattleMapToken,
) -> str:
    cover_rank = {"none": 0, "half": 1, "three_quarters": 2, "total": 3}
    best = "none"
    for zone in battle_map.terrain:
        if zone.cover == "none":
            continue
        if _point_in_zone(b.x, b.y, zone) or any(
            _point_in_zone(x, y, zone)
            for x, y in _line_points(a.x, a.y, b.x, b.y)
        ):
            if cover_rank[zone.cover] > cover_rank[best]:
                best = zone.cover
    return best


def _line_points(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x = x0
    y = y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def _point_in_zone(x: int, y: int, zone: DndTerrainZone) -> bool:
    return (
        zone.x <= x < zone.x + zone.width
        and zone.y <= y < zone.y + zone.height
    )


def _fallback_position(
    index: int,
    width: int,
    height: int,
    occupied: set[tuple[int, int]],
) -> tuple[int, int]:
    left_x = 1 if width > 2 else 0
    right_x = width - 2 if width > 2 else max(0, width - 1)
    base_y = 1 if height > 2 else 0
    span = max(1, height - 2)
    candidates: list[tuple[int, int]] = []
    for offset in range(max(width, height, 1) * 2):
        y = min(height - 1, base_y + ((index // 2 + offset) % span))
        x = left_x if (index + offset) % 2 == 0 else right_x
        candidates.append((x, y))
    for y in range(height):
        for x in range(width):
            candidates.append((x, y))
    for candidate in candidates:
        if candidate not in occupied:
            return candidate
    return (0, 0)


def _clamp_terrain(zone: DndTerrainZone, width: int, height: int) -> DndTerrainZone:
    x = _clamp(zone.x, 0, max(0, width - 1))
    y = _clamp(zone.y, 0, max(0, height - 1))
    return DndTerrainZone(
        zone_id=zone.zone_id,
        label=zone.label,
        x=x,
        y=y,
        width=max(1, min(zone.width, max(1, width - x))),
        height=max(1, min(zone.height, max(1, height - y))),
        blocks_movement=zone.blocks_movement,
        blocks_line_of_sight=zone.blocks_line_of_sight,
        cover=zone.cover,
        notes=zone.notes,
    )


def _clamp_area(area: DndAreaTemplate, width: int, height: int) -> DndAreaTemplate:
    x = _clamp(area.x, 0, max(0, width - 1))
    y = _clamp(area.y, 0, max(0, height - 1))
    return DndAreaTemplate(
        template_id=area.template_id,
        label=area.label,
        shape=area.shape,
        x=x,
        y=y,
        radius_squares=area.radius_squares,
        width=max(1, min(area.width, max(1, width - x))),
        height=max(1, min(area.height, max(1, height - y))),
        duration_rounds=area.duration_rounds,
        notes=area.notes,
    )


def _clamp(value: int, low: int, high: int) -> int:
    if high < low:
        return low
    return max(low, min(value, high))


def _obj_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _slug(text: str) -> str:
    return "_".join(str(text or "").strip().lower().split())
