from __future__ import annotations

from typing import Any, Iterable

from app.engine.dnd_combat_access import obj_get as _obj_get
from app.schemas.dnd_spatial import (
    DndAreaTemplate,
    DndBattleMapState,
    DndBattleMapToken,
    DndSpatialDelta,
    DndTerrainZone,
    MAX_BATTLE_MAP_HEIGHT,
    MAX_BATTLE_MAP_TOKENS,
    MAX_BATTLE_MAP_WIDTH,
)


SUMMARY_TOKEN_LIMIT = 12
SUMMARY_TERRAIN_LIMIT = 4
SUMMARY_AREA_LIMIT = 4
SUMMARY_LABEL_MAX = 32


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

    participants = _participant_records(combatants)[:MAX_BATTLE_MAP_TOKENS]
    if not participants:
        return None

    width = _clamp(
        battle_map.width if battle_map.width > 0 else 12,
        1,
        MAX_BATTLE_MAP_WIDTH,
    )
    height = _clamp(
        battle_map.height if battle_map.height > 0 else 12,
        1,
        MAX_BATTLE_MAP_HEIGHT,
    )

    participants_by_combatant = {
        participant["combatant_id"]: participant
        for participant in participants
        if participant["combatant_id"]
    }
    character_counts: dict[str, int] = {}
    for participant in participants:
        character_id = participant["character_id"]
        if character_id:
            character_counts[character_id] = (
                character_counts.get(character_id, 0) + 1
            )
    participants_by_unique_character = {
        participant["character_id"]: participant
        for participant in participants
        if participant["character_id"]
        and character_counts.get(participant["character_id"]) == 1
    }

    tokens_by_id: dict[str, DndBattleMapToken] = {}
    for raw in battle_map.tokens:
        raw_token_id = raw.token_id.strip()
        raw_character_id = raw.character_id.strip()
        participant = (
            participants_by_combatant.get(raw_token_id)
            if raw_token_id else None
        )
        if participant is None and raw_character_id:
            participant = participants_by_unique_character.get(raw_character_id)
        if participant is None and raw_token_id:
            participant = participants_by_unique_character.get(raw_token_id)
        if participant is None:
            continue

        identity = _participant_identity(participant)
        if not identity:
            continue
        character_id = participant["character_id"] or identity
        max_size = max(1, min(width, height))
        size = _clamp(raw.size_squares, 1, max_size)
        tokens_by_id[identity] = DndBattleMapToken(
            token_id=identity,
            character_id=character_id,
            label=raw.label or participant["name"] or character_id,
            x=_clamp(raw.x, 0, max(0, width - size)),
            y=_clamp(raw.y, 0, max(0, height - size)),
            size_squares=size,
        )

    occupied = {(token.x, token.y) for token in tokens_by_id.values()}
    for index, participant in enumerate(participants):
        identity = _participant_identity(participant)
        if not identity or identity in tokens_by_id:
            continue
        character_id = participant["character_id"] or identity
        x, y = _fallback_position(index, width, height, occupied)
        occupied.add((x, y))
        tokens_by_id[identity] = DndBattleMapToken(
            token_id=identity,
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
        tokens=list(tokens_by_id.values()),
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
            "from": _token_identity(actor_token),
            "to": _token_identity(token),
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
            selector = delta.target_id.strip()
            token = _token_for(battle_map, selector) if selector else None
            if token is None and not selector and delta.character_id:
                token = _token_for(battle_map, delta.character_id)
            created = False
            if token is None and delta.kind == "place_token":
                max_size = max(1, min(battle_map.width, battle_map.height))
                size = _clamp(delta.size_squares, 1, max_size)
                token_id = (
                    selector
                    or delta.character_id
                    or _slug(delta.label)
                    or "token"
                )
                token = DndBattleMapToken(
                    token_id=token_id,
                    character_id=delta.character_id or token_id,
                    label=delta.label or delta.character_id or token_id,
                    x=0,
                    y=0,
                    size_squares=size,
                )
                battle_map.tokens.append(token)
                created = True
            if token is None:
                missing = selector or delta.character_id
                notes.append(
                    f"Spatial delta skipped: token {missing!r} not found."
                )
                continue
            max_size = max(1, min(battle_map.width, battle_map.height))
            token.size_squares = _clamp(delta.size_squares, 1, max_size)
            token.x = _clamp(
                delta.x, 0, max(0, battle_map.width - token.size_squares)
            )
            token.y = _clamp(
                delta.y, 0, max(0, battle_map.height - token.size_squares)
            )
            if delta.label:
                token.label = delta.label
            verb = "Placed" if created else "Moved"
            notes.append(
                f"{verb} {_token_identity(token)} to ({token.x}, {token.y})."
            )
        elif delta.kind == "remove_token":
            selector = delta.target_id.strip() or delta.character_id.strip()
            token = _token_for(battle_map, selector)
            if token is None:
                notes.append(
                    f"Spatial delta skipped: token {selector!r} not found."
                )
                continue
            battle_map.tokens = [
                candidate for candidate in battle_map.tokens
                if candidate is not token
            ]
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
            selector = delta.target_id.strip() or delta.label.strip()
            if not selector:
                notes.append("Spatial delta skipped: area selector is empty.")
                continue
            battle_map.areas = [
                area for area in battle_map.areas
                if selector not in {area.template_id, area.label}
            ]
    return notes


def tick_area_durations(combat: Any) -> list[str]:
    battle_map = _battle_map(combat)
    if battle_map is None:
        return []

    remaining: list[DndAreaTemplate] = []
    expired: list[DndAreaTemplate] = []
    for area in battle_map.areas:
        if area.duration_rounds <= 0:
            remaining.append(area)
            continue
        area.duration_rounds -= 1
        if area.duration_rounds <= 0:
            expired.append(area)
        else:
            remaining.append(area)
    if not expired:
        return []
    battle_map.areas = remaining
    return [
        f"Area expired: {area.label or area.template_id}."
        for area in expired
    ]


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
        shown = battle_map.tokens[:SUMMARY_TOKEN_LIMIT]
        positions = [
            f"{_short_text(token.label or _token_identity(token))} "
            f"({_short_text(_token_identity(token))}) at "
            f"({token.x}, {token.y})"
            for token in shown
        ]
        omitted = len(battle_map.tokens) - len(shown)
        if omitted > 0:
            positions.append(f"... {omitted} more")
        lines.append("Positions: " + "; ".join(positions) + ".")
    if battle_map.terrain:
        terrain = [
            f"{_short_text(zone.label or zone.zone_id)} "
            f"at ({zone.x}, {zone.y}) {zone.width}x{zone.height}"
            + (f", cover {zone.cover}" if zone.cover != "none" else "")
            + (", blocks sight" if zone.blocks_line_of_sight else "")
            for zone in battle_map.terrain[:SUMMARY_TERRAIN_LIMIT]
        ]
        omitted = len(battle_map.terrain) - SUMMARY_TERRAIN_LIMIT
        if omitted > 0:
            terrain.append(f"... {omitted} more")
        lines.append("Terrain: " + "; ".join(terrain) + ".")
    if battle_map.areas:
        areas = [
            f"{_short_text(area.label or area.template_id)} "
            f"{area.shape} at ({area.x}, {area.y})"
            + (
                f", {area.duration_rounds} rounds"
                if area.duration_rounds > 0 else ""
            )
            for area in battle_map.areas[:SUMMARY_AREA_LIMIT]
        ]
        omitted = len(battle_map.areas) - SUMMARY_AREA_LIMIT
        if omitted > 0:
            areas.append(f"... {omitted} more")
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


def _participant_identity(participant: dict[str, str]) -> str:
    return participant["combatant_id"] or participant["character_id"]


def _token_for(
    battle_map: DndBattleMapState,
    target_id: str,
) -> DndBattleMapToken | None:
    target = str(target_id or "").strip()
    if not target:
        return None
    for token in battle_map.tokens:
        if target == token.token_id:
            return token
    character_matches = [
        token for token in battle_map.tokens
        if target == token.character_id
    ]
    if len(character_matches) == 1:
        return character_matches[0]
    label_matches = [
        token for token in battle_map.tokens
        if target == token.label
    ]
    if len(label_matches) == 1:
        return label_matches[0]
    return None


def _token_identity(token: DndBattleMapToken) -> str:
    return token.token_id or token.character_id or token.label


def _token_extent(token: DndBattleMapToken) -> tuple[int, int, int, int]:
    size = max(1, token.size_squares)
    return token.x, token.y, token.x + size - 1, token.y + size - 1


def _point_in_token(x: int, y: int, token: DndBattleMapToken) -> bool:
    left, top, right, bottom = _token_extent(token)
    return left <= x <= right and top <= y <= bottom


def _closest_square(
    origin: DndBattleMapToken,
    target: DndBattleMapToken,
) -> tuple[int, int]:
    left, top, right, bottom = _token_extent(origin)
    target_left, target_top, target_right, target_bottom = _token_extent(target)
    target_x = (target_left + target_right) // 2
    target_y = (target_top + target_bottom) // 2
    return _clamp(target_x, left, right), _clamp(target_y, top, bottom)


def _line_between_tokens(
    a: DndBattleMapToken,
    b: DndBattleMapToken,
) -> list[tuple[int, int]]:
    ax, ay = _closest_square(a, b)
    bx, by = _closest_square(b, a)
    return _line_points(ax, ay, bx, by)


def _short_text(text: str, limit: int = SUMMARY_LABEL_MAX) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _token_gap_squares(
    a: DndBattleMapToken,
    b: DndBattleMapToken,
) -> tuple[int, int]:
    a_left, a_top, a_right, a_bottom = _token_extent(a)
    b_left, b_top, b_right, b_bottom = _token_extent(b)
    dx = max(0, b_left - a_right, a_left - b_right)
    dy = max(0, b_top - a_bottom, a_top - b_bottom)
    return dx, dy


def _distance_ft(
    battle_map: DndBattleMapState,
    a: DndBattleMapToken,
    b: DndBattleMapToken,
) -> int:
    dx, dy = _token_gap_squares(a, b)
    squares = max(dx, dy)
    return squares * battle_map.square_size_ft


def _line_of_sight_clear(
    battle_map: DndBattleMapState,
    a: DndBattleMapToken,
    b: DndBattleMapToken,
) -> bool:
    blockers = [zone for zone in battle_map.terrain if zone.blocks_line_of_sight]
    if not blockers:
        return True
    for x, y in _line_between_tokens(a, b):
        if _point_in_token(x, y, a) or _point_in_token(x, y, b):
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
        if any(
            _point_in_zone(x, y, zone)
            for x in range(b.x, b.x + max(1, b.size_squares))
            for y in range(b.y, b.y + max(1, b.size_squares))
        ) or any(
            _point_in_zone(x, y, zone)
            for x, y in _line_between_tokens(a, b)
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
    for offset in range(max(width, height, 1) * 2):
        y = min(height - 1, base_y + ((index // 2 + offset) % span))
        x = left_x if (index + offset) % 2 == 0 else right_x
        candidate = (x, y)
        if candidate not in occupied:
            return candidate
    for y in range(height):
        for x in range(width):
            candidate = (x, y)
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


def _slug(text: str) -> str:
    return "_".join(str(text or "").strip().lower().split())
