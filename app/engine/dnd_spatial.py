from __future__ import annotations

import math
from typing import Any, Iterable

from app.engine.dnd_combat_access import obj_get as _obj_get
from app.schemas.dnd_spatial import (
    DndAreaTemplate,
    DndBattleMapFeature,
    DndBattleMapSeed,
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
CONTEXT_GEOMETRY_LIMIT = 40
CONTEXT_AREA_LIMIT = 20
CONTEXT_TARGET_LIMIT = 80


def normalize_battle_map_seed(
    seed: DndBattleMapSeed | DndBattleMapState | dict[str, Any] | None,
    combatants: Iterable[Any],
) -> DndBattleMapState | None:
    """Validate a router-seeded map and fill missing participant tokens."""
    if seed is None:
        return None
    battle_map = _coerce_battle_map_state(seed)
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
        source_template_ref=battle_map.source_template_ref,
        source_content_hash=battle_map.source_content_hash,
        orientation=battle_map.orientation,
        spawn_anchors=battle_map.spawn_anchors,
        features=battle_map.features,
        area_links=battle_map.area_links,
        floors=battle_map.floors,
        fog_masks=battle_map.fog_masks,
        reveal_regions=battle_map.reveal_regions,
    )


def combat_packet_context(
    combat: Any,
    actor_id: str,
    *,
    area_templates: Iterable[dict[str, Any]] | None = None,
    relationships_by_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    battle_map = _battle_map(combat)
    if battle_map is None:
        return {
            "battle_map": {},
            "tactical_map_enforcement": {},
            "spatial_advisories": [],
            "area_targeting_advisories": [],
        }
    return {
        "battle_map": battle_map_status(battle_map),
        "tactical_map_enforcement": tactical_map_enforcement_context(
            combat,
            actor_id,
            relationships_by_id=relationships_by_id or {},
        ),
        "spatial_advisories": spatial_advisories(combat, actor_id),
        "area_targeting_advisories": area_targeting_advisories(
            combat,
            actor_id,
            area_templates or [],
            relationships_by_id=relationships_by_id or {},
        ),
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


def tactical_map_enforcement_context(
    combat: Any,
    actor_id: str,
    *,
    relationships_by_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute structured tactical geometry for LLM adjudication.

    The context is deliberately advisory except for hard gates that protect the
    runtime map from impossible geometry such as overlapping tokens or blocked
    destinations.
    """
    battle_map = _battle_map(combat)
    if battle_map is None:
        return {}

    relationships = relationships_by_id or {}
    actor_token = _token_for(battle_map, actor_id)
    hard_gates: list[dict[str, Any]] = []
    if actor_token is None:
        hard_gates.append({
            "kind": "actor_token_missing",
            "reason": f"No tactical-map token found for {actor_id}.",
        })

    context: dict[str, Any] = {
        "authority": {
            "geometry": (
                "reviewed_tactical_map"
                if battle_map.source_template_ref
                and battle_map.source_content_hash
                else "runtime_battle_map"
            ),
            "reviewed_geometry": bool(
                battle_map.source_template_ref
                and battle_map.source_content_hash
            ),
        },
        "grid": {
            "width": battle_map.width,
            "height": battle_map.height,
            "square_size_ft": battle_map.square_size_ft,
        },
        "actor": _token_payload(actor_token) if actor_token is not None else {},
        "occupancy": [
            _token_payload(token)
            for token in battle_map.tokens[:MAX_BATTLE_MAP_TOKENS]
        ],
        "terrain": _terrain_enforcement_payload(battle_map),
        "active_areas": _active_area_payloads(battle_map),
        "verticality": _verticality_payload(battle_map),
        "targets": _target_geometry_payloads(
            battle_map,
            actor_token,
            relationships,
        ) if actor_token is not None else [],
        "movement": _movement_payload(battle_map, actor_token)
        if actor_token is not None else {},
        "hard_gates": hard_gates,
        "advisories": [],
    }

    if actor_token is not None:
        current_gates = _placement_hard_gates(
            battle_map,
            actor_token,
            actor_token.x,
            actor_token.y,
            actor_token.size_squares,
        )
        if current_gates:
            context["hard_gates"].extend({
                **gate,
                "reason": f"Current actor position is invalid: {gate['reason']}",
            } for gate in current_gates)
    if battle_map.floors and not _tokens_have_floor_state(battle_map):
        context["advisories"].append({
            "kind": "floor_assignment_untracked",
            "reason": (
                "Map has floors or submaps, but combatant tokens are tracked "
                "on the primary runtime grid only."
            ),
        })
    return context


def area_targeting_advisories(
    combat: Any,
    actor_id: str,
    templates: Iterable[dict[str, Any]],
    *,
    relationships_by_id: dict[str, str] | None = None,
    max_templates: int = 6,
    max_candidates: int = 5,
) -> list[dict[str, Any]]:
    battle_map = _battle_map(combat)
    if battle_map is None:
        return []
    actor_token = _token_for(battle_map, actor_id)
    if actor_token is None:
        return []
    relationships = relationships_by_id or {}
    advisories: list[dict[str, Any]] = []
    for template in list(templates)[:max_templates]:
        shape = str(template.get("shape") or "").strip().lower()
        if shape not in {"cone", "line", "circle", "sphere", "square", "cube"}:
            continue
        normalized = _normalize_area_template(template, battle_map.square_size_ft)
        candidates = _area_candidates(
            battle_map,
            actor_token,
            normalized,
            relationships,
        )
        if not candidates:
            continue
        advisories.append({
            "action_id": str(template.get("action_id") or ""),
            "name": str(template.get("name") or ""),
            "shape": normalized["shape"],
            "size": normalized["size"],
            "candidates": candidates[:max_candidates],
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
            target_size = _clamp(delta.size_squares, 1, max_size)
            placement_gates = _placement_hard_gates(
                battle_map,
                token,
                delta.x,
                delta.y,
                target_size,
            )
            if placement_gates:
                notes.append(_blocked_delta_note(token, delta, placement_gates))
                if created:
                    battle_map.tokens = [
                        candidate for candidate in battle_map.tokens
                        if candidate is not token
                    ]
                continue
            token.size_squares = target_size
            token.x = delta.x
            token.y = delta.y
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


def battle_map_status(combat: Any, *, include_hidden: bool = False) -> dict[str, Any]:
    battle_map = _battle_map(combat)
    if battle_map is None:
        return {}
    if include_hidden:
        return battle_map.model_dump()
    return _player_safe_battle_map(battle_map).model_dump()


def render_battle_map_summary(
    combat_or_map: Any,
    *,
    actor_id: str = "",
    max_lines: int = 8,
) -> list[str]:
    battle_map = (
        _coerce_battle_map_state(combat_or_map)
        if isinstance(combat_or_map, DndBattleMapSeed)
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
    visible_features = [
        feature for feature in battle_map.features if not feature.secret
    ]
    if visible_features:
        features = [
            _feature_summary(feature)
            for feature in visible_features[:SUMMARY_AREA_LIMIT]
        ]
        omitted = len(visible_features) - SUMMARY_AREA_LIMIT
        if omitted > 0:
            features.append(f"... {omitted} more")
        lines.append("Features: " + "; ".join(features) + ".")
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


def _token_payload(token: DndBattleMapToken | None) -> dict[str, Any]:
    if token is None:
        return {}
    return {
        "token_id": token.token_id,
        "character_id": token.character_id,
        "label": token.label,
        "x": token.x,
        "y": token.y,
        "size_squares": token.size_squares,
        "rect": _rect_payload(
            token.x,
            token.y,
            token.size_squares,
            token.size_squares,
        ),
    }


def _terrain_enforcement_payload(
    battle_map: DndBattleMapState,
) -> dict[str, list[dict[str, Any]]]:
    zones = _mechanical_zone_payloads(battle_map)
    return {
        "movement_blockers": [
            zone for zone in zones if zone["blocks_movement"]
        ][:CONTEXT_GEOMETRY_LIMIT],
        "line_of_sight_blockers": [
            zone for zone in zones if zone["blocks_line_of_sight"]
        ][:CONTEXT_GEOMETRY_LIMIT],
        "difficult_terrain": [
            zone for zone in zones if zone["difficult_terrain"]
        ][:CONTEXT_GEOMETRY_LIMIT],
        "cover": [
            zone for zone in zones if zone["cover"] != "none"
        ][:CONTEXT_GEOMETRY_LIMIT],
    }


def _mechanical_zone_payloads(
    battle_map: DndBattleMapState,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for zone in battle_map.terrain:
        payload = {
            "id": zone.zone_id or zone.label,
            "label": zone.label or zone.zone_id,
            "kind": "terrain",
            "floor_id": zone.floor_id,
            "rect": _rect_payload(zone.x, zone.y, zone.width, zone.height),
            "cells": [],
            "blocks_movement": zone.blocks_movement,
            "blocks_line_of_sight": zone.blocks_line_of_sight,
            "difficult_terrain": False,
            "cover": zone.cover,
        }
        payloads.append(payload)
        if payload["id"]:
            by_id[payload["id"]] = payload

    for feature in _visible_features(battle_map):
        if not _feature_has_mechanical_geometry(feature):
            continue
        payload = _feature_zone_payload(feature)
        existing = by_id.get(feature.feature_id)
        if existing is not None:
            existing["kind"] = feature.feature_kind
            existing["difficult_terrain"] = (
                existing["difficult_terrain"] or feature.difficult_terrain
            )
            existing["blocks_movement"] = (
                existing["blocks_movement"] or feature.blocks_movement
            )
            existing["blocks_line_of_sight"] = (
                existing["blocks_line_of_sight"]
                or feature.blocks_line_of_sight
            )
            if existing["cover"] == "none":
                existing["cover"] = feature.cover
            continue
        payloads.append(payload)
        if payload["id"]:
            by_id[payload["id"]] = payload
    return payloads


def _feature_zone_payload(feature: DndBattleMapFeature) -> dict[str, Any]:
    bounds = feature.bounds
    cells = []
    rect: dict[str, int] | None = None
    if bounds is not None:
        rect = _rect_payload(bounds.x, bounds.y, bounds.width, bounds.height)
    elif feature.cells:
        cells = [
            {"x": point.x, "y": point.y}
            for point in feature.cells[:CONTEXT_GEOMETRY_LIMIT]
        ]
    return {
        "id": feature.feature_id or feature.label,
        "label": feature.label or feature.feature_id,
        "kind": feature.feature_kind,
        "floor_id": feature.floor_id,
        "rect": rect,
        "cells": cells,
        "blocks_movement": feature.blocks_movement,
        "blocks_line_of_sight": feature.blocks_line_of_sight,
        "difficult_terrain": feature.difficult_terrain,
        "cover": feature.cover,
    }


def _active_area_payloads(
    battle_map: DndBattleMapState,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for area in battle_map.areas[:CONTEXT_AREA_LIMIT]:
        affected = [
            _token_identity(token)
            for token in battle_map.tokens
            if _area_contains_token(area, token)
        ]
        payloads.append({
            "template_id": area.template_id,
            "label": area.label,
            "shape": area.shape,
            "x": area.x,
            "y": area.y,
            "radius_squares": area.radius_squares,
            "width": area.width,
            "height": area.height,
            "duration_rounds": area.duration_rounds,
            "affected_tokens": affected,
        })
    return payloads


def _verticality_payload(battle_map: DndBattleMapState) -> dict[str, Any]:
    floors = [
        {
            "floor_id": floor.floor_id,
            "label": floor.label,
            "kind": floor.floor_kind,
            "level_index": floor.level_index,
            "elevation_ft": floor.elevation_ft,
            "grid_width": floor.grid_width,
            "grid_height": floor.grid_height,
            "parent_floor_id": floor.parent_floor_id,
            "parent_bounds": (
                _rect_payload(
                    floor.parent_bounds.x,
                    floor.parent_bounds.y,
                    floor.parent_bounds.width,
                    floor.parent_bounds.height,
                )
                if floor.parent_bounds is not None else None
            ),
        }
        for floor in battle_map.floors[:CONTEXT_GEOMETRY_LIMIT]
    ]
    vertical_links = []
    for feature in _visible_features(battle_map):
        if not _feature_is_vertical_link(feature):
            continue
        vertical_links.append({
            "feature_id": feature.feature_id,
            "label": feature.label or feature.feature_id,
            "kind": feature.feature_kind,
            "floor_id": feature.floor_id,
            "rect": (
                _rect_payload(
                    feature.bounds.x,
                    feature.bounds.y,
                    feature.bounds.width,
                    feature.bounds.height,
                )
                if feature.bounds is not None else None
            ),
            "cells": [
                {"x": point.x, "y": point.y}
                for point in feature.cells[:CONTEXT_GEOMETRY_LIMIT]
            ],
        })
    return {
        "floors": floors,
        "vertical_links": vertical_links[:CONTEXT_GEOMETRY_LIMIT],
    }


def _target_geometry_payloads(
    battle_map: DndBattleMapState,
    actor_token: DndBattleMapToken,
    relationships: dict[str, str],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for token in battle_map.tokens[:CONTEXT_TARGET_LIMIT]:
        if token is actor_token:
            continue
        distance_ft = _distance_ft(battle_map, actor_token, token)
        line_of_sight = _line_of_sight_clear(battle_map, actor_token, token)
        cover = _cover_between(battle_map, actor_token, token)
        payloads.append({
            "token_id": token.token_id,
            "character_id": token.character_id,
            "label": token.label,
            "relationship": _token_relationship(token, relationships),
            "distance_ft": distance_ft,
            "adjacent": distance_ft <= battle_map.square_size_ft,
            "line_of_sight": "clear" if line_of_sight else "blocked",
            "cover": cover,
        })
    return payloads


def _movement_payload(
    battle_map: DndBattleMapState,
    actor_token: DndBattleMapToken,
) -> dict[str, Any]:
    adjacent = []
    for label, dx, dy in (
        ("N", 0, -1),
        ("NE", 1, -1),
        ("E", 1, 0),
        ("SE", 1, 1),
        ("S", 0, 1),
        ("SW", -1, 1),
        ("W", -1, 0),
        ("NW", -1, -1),
    ):
        x = actor_token.x + dx
        y = actor_token.y + dy
        gates = _placement_hard_gates(
            battle_map,
            actor_token,
            x,
            y,
            actor_token.size_squares,
        )
        adjacent.append({
            "direction": label,
            "x": x,
            "y": y,
            "legal_destination": not gates,
            "hard_gates": gates,
            "terrain_cost": (
                "difficult"
                if not gates
                and _placement_has_difficult_terrain(
                    battle_map,
                    x,
                    y,
                    actor_token.size_squares,
                )
                else "normal"
            ),
        })
    return {
        "from": {
            "x": actor_token.x,
            "y": actor_token.y,
            "size_squares": actor_token.size_squares,
        },
        "adjacent_destinations": adjacent,
    }


def _normalize_area_template(
    template: dict[str, Any],
    square_size_ft: int,
) -> dict[str, Any]:
    square = max(1, int(square_size_ft or 5))
    shape = str(template.get("shape") or "").strip().lower()
    if shape == "sphere":
        shape = "circle"
    if shape == "cube":
        shape = "square"
    return {
        "shape": shape,
        "length_squares": _ft_to_squares(template.get("length_ft"), square),
        "radius_squares": _ft_to_squares(template.get("radius_ft"), square),
        "width_squares": max(1, _ft_to_squares(template.get("width_ft"), square)),
        "height_squares": max(1, _ft_to_squares(template.get("height_ft"), square)),
        "size": {
            key: value
            for key, value in {
                "length_ft": template.get("length_ft"),
                "radius_ft": template.get("radius_ft"),
                "width_ft": template.get("width_ft"),
                "height_ft": template.get("height_ft"),
            }.items()
            if value
        },
    }


def _ft_to_squares(value: Any, square_size_ft: int) -> int:
    try:
        amount = int(value or 0)
    except (TypeError, ValueError):
        amount = 0
    if amount <= 0:
        return 0
    return max(1, math.ceil(amount / max(1, square_size_ft)))


def _area_candidates(
    battle_map: DndBattleMapState,
    actor_token: DndBattleMapToken,
    template: dict[str, Any],
    relationships: dict[str, str],
) -> list[dict[str, Any]]:
    shape = template["shape"]
    candidates: list[dict[str, Any]] = []
    if shape in {"cone", "line"}:
        length = int(template.get("length_squares") or 0)
        if length <= 0:
            return []
        width = int(template.get("width_squares") or 1)
        for label, dx, dy in _DIRECTIONS:
            affected, blocked = _directional_area_targets(
                battle_map,
                actor_token,
                shape=shape,
                direction=(dx, dy),
                length_squares=length,
                width_squares=width,
            )
            candidates.append(_ranked_area_candidate(
                affected,
                blocked,
                relationships,
                origin={"token_id": _token_identity(actor_token)},
                direction=label,
            ))
    elif shape == "circle":
        radius = int(template.get("radius_squares") or 0)
        if radius <= 0:
            return []
        for x, y in _candidate_centers(battle_map):
            affected, blocked = _centered_area_targets(
                battle_map,
                actor_token,
                center=(x, y),
                radius_squares=radius,
            )
            candidates.append(_ranked_area_candidate(
                affected,
                blocked,
                relationships,
                origin={"x": x, "y": y},
                direction="",
            ))
    elif shape == "square":
        width = int(template.get("width_squares") or 1)
        height = int(template.get("height_squares") or width)
        for x, y in _candidate_square_origins(battle_map, width, height):
            affected, blocked = _square_area_targets(
                battle_map,
                actor_token,
                origin=(x, y),
                width=width,
                height=height,
            )
            candidates.append(_ranked_area_candidate(
                affected,
                blocked,
                relationships,
                origin={"x": x, "y": y},
                direction="",
            ))
    candidates = [candidate for candidate in candidates if candidate["affected"]]
    candidates.sort(
        key=lambda item: (
            item["score"],
            len(item["enemy_targets"]),
            -len(item["ally_targets"]),
            -len(item["unknown_targets"]),
        ),
        reverse=True,
    )
    return _dedupe_candidates(candidates)


_DIRECTIONS = (
    ("N", 0.0, -1.0),
    ("NE", 1.0, -1.0),
    ("E", 1.0, 0.0),
    ("SE", 1.0, 1.0),
    ("S", 0.0, 1.0),
    ("SW", -1.0, 1.0),
    ("W", -1.0, 0.0),
    ("NW", -1.0, -1.0),
)


def _directional_area_targets(
    battle_map: DndBattleMapState,
    actor_token: DndBattleMapToken,
    *,
    shape: str,
    direction: tuple[float, float],
    length_squares: int,
    width_squares: int,
) -> tuple[list[DndBattleMapToken], list[DndBattleMapToken]]:
    dx, dy = _unit(direction)
    ax, ay = _token_center(actor_token)
    affected: list[DndBattleMapToken] = []
    blocked: list[DndBattleMapToken] = []
    for token in battle_map.tokens:
        if token is actor_token:
            continue
        tx, ty = _token_center(token)
        vx = tx - ax
        vy = ty - ay
        projection = vx * dx + vy * dy
        if projection <= 0 or projection > length_squares:
            continue
        perpendicular = abs(vx * dy - vy * dx)
        if shape == "line":
            if perpendicular > max(0.5, width_squares / 2):
                continue
        elif perpendicular > max(0.5, projection):
            continue
        if _cover_between(battle_map, actor_token, token) == "total":
            blocked.append(token)
            continue
        affected.append(token)
    return affected, blocked


def _centered_area_targets(
    battle_map: DndBattleMapState,
    actor_token: DndBattleMapToken,
    *,
    center: tuple[int, int],
    radius_squares: int,
) -> tuple[list[DndBattleMapToken], list[DndBattleMapToken]]:
    cx, cy = center
    affected: list[DndBattleMapToken] = []
    blocked: list[DndBattleMapToken] = []
    for token in battle_map.tokens:
        if token is actor_token:
            continue
        tx, ty = _token_center(token)
        if max(abs(tx - cx), abs(ty - cy)) > radius_squares:
            continue
        if _cover_between(battle_map, actor_token, token) == "total":
            blocked.append(token)
            continue
        affected.append(token)
    return affected, blocked


def _square_area_targets(
    battle_map: DndBattleMapState,
    actor_token: DndBattleMapToken,
    *,
    origin: tuple[int, int],
    width: int,
    height: int,
) -> tuple[list[DndBattleMapToken], list[DndBattleMapToken]]:
    ox, oy = origin
    affected: list[DndBattleMapToken] = []
    blocked: list[DndBattleMapToken] = []
    for token in battle_map.tokens:
        if token is actor_token:
            continue
        tx, ty = _token_center(token)
        if not (ox <= tx < ox + width and oy <= ty < oy + height):
            continue
        if _cover_between(battle_map, actor_token, token) == "total":
            blocked.append(token)
            continue
        affected.append(token)
    return affected, blocked


def _ranked_area_candidate(
    affected: list[DndBattleMapToken],
    blocked: list[DndBattleMapToken],
    relationships: dict[str, str],
    *,
    origin: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    enemy_targets: list[str] = []
    ally_targets: list[str] = []
    unknown_targets: list[str] = []
    for token in affected:
        identity = _token_identity(token)
        relationship = _token_relationship(token, relationships)
        if relationship == "enemy":
            enemy_targets.append(identity)
        elif relationship in {"ally", "self"}:
            ally_targets.append(identity)
        else:
            unknown_targets.append(identity)
    blocked_targets = [_token_identity(token) for token in blocked]
    score = (
        100 * len(enemy_targets)
        - 80 * len(ally_targets)
        - 10 * len(unknown_targets)
        - 20 * len(blocked_targets)
    )
    return {
        "origin": origin,
        "direction": direction,
        "affected": [_token_identity(token) for token in affected],
        "enemy_targets": enemy_targets,
        "ally_targets": ally_targets,
        "unknown_targets": unknown_targets,
        "blocked_by_total_cover": blocked_targets,
        "score": score,
    }


def _token_relationship(
    token: DndBattleMapToken,
    relationships: dict[str, str],
) -> str:
    for key in (token.token_id, token.character_id, token.label):
        relationship = relationships.get(str(key or ""))
        if relationship:
            return relationship
    return "unknown"


def _rect_payload(x: int, y: int, width: int, height: int) -> dict[str, int]:
    return {
        "x": int(x),
        "y": int(y),
        "width": max(1, int(width)),
        "height": max(1, int(height)),
    }


def _visible_features(
    battle_map: DndBattleMapState,
) -> list[DndBattleMapFeature]:
    return [feature for feature in battle_map.features if not feature.secret]


def _feature_has_mechanical_geometry(feature: DndBattleMapFeature) -> bool:
    return (
        feature.blocks_movement
        or feature.blocks_line_of_sight
        or feature.difficult_terrain
        or feature.cover != "none"
    )


def _feature_is_vertical_link(feature: DndBattleMapFeature) -> bool:
    return feature.feature_kind in {
        "stairs",
        "pit",
        "cliff",
        "balcony",
    }


def _tokens_have_floor_state(battle_map: DndBattleMapState) -> bool:
    return any(
        str(getattr(token, "floor_id", "") or "").strip()
        for token in battle_map.tokens
    )


def _area_contains_token(
    area: DndAreaTemplate,
    token: DndBattleMapToken,
) -> bool:
    tx, ty = _token_center(token)
    if area.shape == "circle":
        radius = max(0, area.radius_squares)
        return max(abs(tx - area.x), abs(ty - area.y)) <= radius
    return (
        area.x <= tx < area.x + max(1, area.width)
        and area.y <= ty < area.y + max(1, area.height)
    )


def _placement_hard_gates(
    battle_map: DndBattleMapState,
    moving_token: DndBattleMapToken | None,
    x: int,
    y: int,
    size_squares: int,
) -> list[dict[str, Any]]:
    size = max(1, size_squares)
    if x < 0 or y < 0 or x + size > battle_map.width or y + size > battle_map.height:
        return [{
            "kind": "out_of_bounds",
            "reason": (
                f"Destination ({x}, {y}) size {size} is outside "
                f"the {battle_map.width}x{battle_map.height} map."
            ),
        }]

    cells = _rect_cells(x, y, size, size)
    gates: list[dict[str, Any]] = []
    blockers = _movement_blockers_for_cells(battle_map, cells)
    if blockers:
        gates.append({
            "kind": "movement_blocked",
            "reason": (
                "Destination includes movement blocker"
                f"{'s' if len(blockers) != 1 else ''}: "
                f"{', '.join(blockers)}."
            ),
            "blockers": blockers,
        })

    occupants = _occupying_tokens_for_cells(
        battle_map,
        cells,
        ignore=moving_token,
    )
    if occupants:
        gates.append({
            "kind": "occupied",
            "reason": (
                "Destination is occupied by "
                f"{', '.join(occupants)}."
            ),
            "occupants": occupants,
        })
    return gates


def _placement_has_difficult_terrain(
    battle_map: DndBattleMapState,
    x: int,
    y: int,
    size_squares: int,
) -> bool:
    if x < 0 or y < 0:
        return False
    return any(
        _cell_has_difficult_terrain(battle_map, cell_x, cell_y)
        for cell_x, cell_y in _rect_cells(x, y, size_squares, size_squares)
        if 0 <= cell_x < battle_map.width and 0 <= cell_y < battle_map.height
    )


def _movement_blockers_for_cells(
    battle_map: DndBattleMapState,
    cells: Iterable[tuple[int, int]],
) -> list[str]:
    blockers: list[str] = []
    seen: set[str] = set()
    for x, y in cells:
        for label in _cell_movement_blockers(battle_map, x, y):
            if label in seen:
                continue
            blockers.append(label)
            seen.add(label)
            if len(blockers) >= CONTEXT_GEOMETRY_LIMIT:
                return blockers
    return blockers


def _cell_movement_blockers(
    battle_map: DndBattleMapState,
    x: int,
    y: int,
) -> list[str]:
    labels: list[str] = []
    for zone in battle_map.terrain:
        if zone.blocks_movement and _point_in_zone(x, y, zone):
            labels.append(zone.label or zone.zone_id or "terrain")
    for feature in _visible_features(battle_map):
        if feature.blocks_movement and _point_in_feature(x, y, feature):
            labels.append(feature.label or feature.feature_id or "feature")
    return list(dict.fromkeys(labels))


def _cell_has_difficult_terrain(
    battle_map: DndBattleMapState,
    x: int,
    y: int,
) -> bool:
    for zone in battle_map.terrain:
        if getattr(zone, "difficult_terrain", False) and _point_in_zone(x, y, zone):
            return True
    for feature in _visible_features(battle_map):
        if feature.difficult_terrain and _point_in_feature(x, y, feature):
            return True
    return False


def _occupying_tokens_for_cells(
    battle_map: DndBattleMapState,
    cells: Iterable[tuple[int, int]],
    *,
    ignore: DndBattleMapToken | None,
) -> list[str]:
    occupied = set(cells)
    occupants: list[str] = []
    for token in battle_map.tokens:
        if token is ignore:
            continue
        if occupied.intersection(_token_cells(token)):
            occupants.append(_token_identity(token))
    return occupants


def _token_cells(token: DndBattleMapToken) -> set[tuple[int, int]]:
    return set(_rect_cells(token.x, token.y, token.size_squares, token.size_squares))


def _rect_cells(
    x: int,
    y: int,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    return [
        (cell_x, cell_y)
        for cell_x in range(x, x + max(1, width))
        for cell_y in range(y, y + max(1, height))
    ]


def _blocked_delta_note(
    token: DndBattleMapToken,
    delta: DndSpatialDelta,
    gates: list[dict[str, Any]],
) -> str:
    reasons = "; ".join(str(gate.get("reason") or "") for gate in gates)
    return (
        f"Spatial delta blocked: {_token_identity(token)} cannot occupy "
        f"({delta.x}, {delta.y}). {reasons}"
    )


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        key = tuple(candidate["affected"])
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _candidate_centers(battle_map: DndBattleMapState) -> list[tuple[int, int]]:
    centers = {
        (int(round(x)), int(round(y)))
        for x, y in (_token_center(token) for token in battle_map.tokens)
    }
    return [
        (x, y)
        for x, y in sorted(centers)
        if 0 <= x < battle_map.width and 0 <= y < battle_map.height
    ]


def _candidate_square_origins(
    battle_map: DndBattleMapState,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    origins: set[tuple[int, int]] = set()
    for token in battle_map.tokens:
        cx, cy = _token_center(token)
        for x in range(math.floor(cx) - width + 1, math.floor(cx) + 1):
            for y in range(math.floor(cy) - height + 1, math.floor(cy) + 1):
                if 0 <= x <= max(0, battle_map.width - width) and 0 <= y <= max(0, battle_map.height - height):
                    origins.add((x, y))
    return sorted(origins)


def _unit(direction: tuple[float, float]) -> tuple[float, float]:
    dx, dy = direction
    magnitude = math.hypot(dx, dy)
    if magnitude <= 0:
        return 0.0, 0.0
    return dx / magnitude, dy / magnitude


def _token_center(token: DndBattleMapToken) -> tuple[float, float]:
    size = max(1, token.size_squares)
    offset = (size - 1) / 2
    return token.x + offset, token.y + offset


def _battle_map(combat: Any) -> DndBattleMapState | None:
    if combat is None:
        return None
    if isinstance(combat, DndBattleMapSeed):
        battle_map = _coerce_battle_map_state(combat)
        return battle_map if battle_map.present else None
    if isinstance(combat, dict):
        raw = combat if "present" in combat and "tokens" in combat else combat.get(
            "battle_map"
        )
    else:
        raw = getattr(combat, "battle_map", None)
    if not raw:
        return None
    battle_map = _coerce_battle_map_state(raw)
    if battle_map is not raw and not isinstance(combat, dict):
        try:
            setattr(combat, "battle_map", battle_map)
        except (AttributeError, TypeError, ValueError):
            pass
    return battle_map if battle_map.present else None


def _coerce_battle_map_state(
    raw: DndBattleMapSeed | DndBattleMapState | dict[str, Any],
) -> DndBattleMapState:
    if isinstance(raw, DndBattleMapState):
        return raw
    if isinstance(raw, DndBattleMapSeed):
        return DndBattleMapState.model_validate(raw.model_dump())
    return DndBattleMapState.model_validate(raw)


def _player_safe_battle_map(
    battle_map: DndBattleMapState,
) -> DndBattleMapState:
    return battle_map.model_copy(
        deep=True,
        update={
            "source_template_ref": "",
            "source_content_hash": "",
            "spawn_anchors": [],
            "features": [
                _player_safe_feature(feature)
                for feature in battle_map.features
                if not feature.secret
            ],
            "area_links": [],
            "floors": [
                _player_safe_floor(floor)
                for floor in battle_map.floors
            ],
            "fog_masks": [],
            "reveal_regions": [],
        },
    )


def _player_safe_floor(floor: Any) -> Any:
    return floor.model_copy(
        deep=True,
        update={
            "map_asset_id": "",
            "area_refs": [],
        },
    )


def _player_safe_feature(feature: DndBattleMapFeature) -> DndBattleMapFeature:
    return feature.model_copy(
        deep=True,
        update={
            "reveal_trigger": "",
            "linked_refs": [],
        },
    )


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


def _feature_summary(feature: DndBattleMapFeature) -> str:
    label = _short_text(feature.label or feature.feature_id)
    bits = [label, feature.feature_kind.replace("_", " ")]
    if feature.bounds is not None:
        bits.append(
            f"at ({feature.bounds.x}, {feature.bounds.y}) "
            f"{feature.bounds.width}x{feature.bounds.height}"
        )
    elif feature.cells:
        first = feature.cells[0]
        bits.append(f"at ({first.x}, {first.y})")
    if feature.difficult_terrain:
        bits.append("difficult terrain")
    if feature.cover != "none":
        bits.append(f"cover {feature.cover}")
    if feature.blocks_line_of_sight:
        bits.append("blocks sight")
    return ", ".join(bits)


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
    feature_blockers = [
        feature
        for feature in _visible_features(battle_map)
        if feature.blocks_line_of_sight
    ]
    if not blockers and not feature_blockers:
        return True
    for x, y in _line_between_tokens(a, b):
        if _point_in_token(x, y, a) or _point_in_token(x, y, b):
            continue
        if any(_point_in_zone(x, y, zone) for zone in blockers):
            return False
        if any(_point_in_feature(x, y, feature) for feature in feature_blockers):
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
    for feature in _visible_features(battle_map):
        if feature.cover == "none":
            continue
        if any(
            _point_in_feature(x, y, feature)
            for x in range(b.x, b.x + max(1, b.size_squares))
            for y in range(b.y, b.y + max(1, b.size_squares))
        ) or any(
            _point_in_feature(x, y, feature)
            for x, y in _line_between_tokens(a, b)
        ):
            if cover_rank[feature.cover] > cover_rank[best]:
                best = feature.cover
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


def _point_in_feature(x: int, y: int, feature: DndBattleMapFeature) -> bool:
    if feature.bounds is not None:
        return (
            feature.bounds.x <= x < feature.bounds.x + feature.bounds.width
            and feature.bounds.y <= y < feature.bounds.y + feature.bounds.height
        )
    return any(point.x == x and point.y == y for point in feature.cells)


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
        floor_id=zone.floor_id,
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
