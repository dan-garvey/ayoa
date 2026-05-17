from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.schemas.event_router import EventKind, EventRouterOutput

RouterTargetKind = Literal[
    "agent_turn",
    "player_render",
    "perception_harvest",
]

RouterTargetFrame = Literal[
    "foreground",
    "private",
    "background",
]


class RouterOutputTarget(BaseModel):
    """One engine task selected from a router output.

    This is not an LLM output model yet. It is the runtime contract that
    lets the turn loop project the router's routing roles into dispatchable
    work. Same-scene agent turns are consumed sequentially: one public result
    is sent back to the router, canonicalized, and only then can another
    routed character respond with updated context.
    """

    model_config = ConfigDict(extra="forbid")

    target_kind: RouterTargetKind
    character_id: str
    frame: RouterTargetFrame
    source_event_id: str


class RouterTargetProjection(BaseModel):
    """Runtime projection of the current router output into dispatch work."""

    model_config = ConfigDict(extra="forbid")

    events: list[EventRouterOutput]
    event_kind: EventKind
    target_audience: list[str]
    perception_harvest_targets: list[str]
    targets: list[RouterOutputTarget]


def event_kind_from_router_output(result: EventRouterOutput) -> EventKind:
    return result.event_kind


def router_frame_for_pick(
    result: EventRouterOutput,
    *,
    player_ids: set[str],
    character_id: str,
) -> RouterTargetFrame:
    if result.event_kind == "public_fact":
        return "background"
    if any(
        update.character_id == character_id
        for update in result.location_updates
    ):
        return "background"
    observer_ids = {observer.character_id for observer in result.observers}
    if character_id not in observer_ids:
        return "background"
    if observer_ids.intersection(player_ids):
        return "foreground"
    return "private"


def targets_from_router_output(
    result: EventRouterOutput,
    *,
    player_ids: set[str],
    agent_ids: list[str],
    perception_targets: list[str] | None = None,
) -> RouterTargetProjection:
    """Build a target projection from the router's routing roles."""

    event_kind = event_kind_from_router_output(result)
    perception_targets = list(perception_targets or [])
    targets: list[RouterOutputTarget] = []
    if perception_targets:
        targets.extend(
            RouterOutputTarget(
                target_kind="perception_harvest",
                character_id=cid,
                frame="foreground",
                source_event_id=result.event_id,
            )
            for cid in perception_targets
        )
    elif not result.ends_beat or result.event_kind == "public_fact":
        targets.extend(
            RouterOutputTarget(
                target_kind="agent_turn",
                character_id=cid,
                frame=router_frame_for_pick(
                    result,
                    player_ids=player_ids,
                    character_id=cid,
                ),
                source_event_id=result.event_id,
            )
            for cid in agent_ids
        )

    target_audience = [
        observer.character_id
        for observer in result.observers
        if result.ends_beat and observer.character_id in player_ids
    ]

    return RouterTargetProjection(
        events=[result],
        event_kind=event_kind,
        target_audience=list(dict.fromkeys(target_audience)),
        perception_harvest_targets=list(dict.fromkeys(perception_targets)),
        targets=targets,
    )
