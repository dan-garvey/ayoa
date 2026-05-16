from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.schemas.event_router import EventKind, EventRouterOutput

FrontierTargetKind = Literal[
    "agent_turn",
    "player_render",
    "perception_harvest",
]

FrontierFrame = Literal[
    "foreground",
    "private",
    "background",
]

FrontierResultKind = Literal[
    "agent_turn",
    "player_render",
    "perception_harvest",
]


class RouterFrontierTarget(BaseModel):
    """One engine task selected from a router output.

    This is not an LLM output model yet. It is the runtime contract that
    lets the turn loop wait for every selected target in a frontier group
    before submitting the group's public results back to the router.
    """

    model_config = ConfigDict(extra="forbid")

    target_kind: FrontierTargetKind
    character_id: str
    frame: FrontierFrame
    source_event_id: str


class RouterFrontierResult(BaseModel):
    """Sanitized completion payload for one frontier target."""

    model_config = ConfigDict(extra="forbid")

    result_kind: FrontierResultKind
    character_id: str
    frame: FrontierFrame
    public_text: str
    source_event_id: str


class RouterFrontierOutput(BaseModel):
    """Runtime projection of the current router output into frontier work."""

    model_config = ConfigDict(extra="forbid")

    events: list[EventRouterOutput]
    event_kind: EventKind
    target_audience: list[str]
    agent_responder_picks: list[str]
    perception_harvest_targets: list[str]
    frontier_targets: list[RouterFrontierTarget]


def event_kind_from_router_output(result: EventRouterOutput) -> EventKind:
    return result.event_kind


def frontier_from_router_output(
    result: EventRouterOutput,
    *,
    player_ids: set[str],
    agent_picks: list[str],
    perception_targets: list[str] | None = None,
) -> RouterFrontierOutput:
    """Build a frontier projection from the current legacy router shape."""

    event_kind = event_kind_from_router_output(result)
    perception_targets = list(perception_targets or [])
    targets: list[RouterFrontierTarget] = []
    if perception_targets:
        targets.extend(
            RouterFrontierTarget(
                target_kind="perception_harvest",
                character_id=cid,
                frame="foreground",
                source_event_id=result.event_id,
            )
            for cid in perception_targets
        )
    elif not result.ends_beat:
        observer_ids = {observer.character_id for observer in result.observers}
        targets.extend(
            RouterFrontierTarget(
                target_kind="agent_turn",
                character_id=cid,
                frame="foreground" if cid in observer_ids else "background",
                source_event_id=result.event_id,
            )
            for cid in agent_picks
        )

    target_audience = [
        observer.character_id
        for observer in result.observers
        if result.ends_beat and observer.character_id in player_ids
    ]

    return RouterFrontierOutput(
        events=[result],
        event_kind=event_kind,
        target_audience=list(dict.fromkeys(target_audience)),
        agent_responder_picks=list(dict.fromkeys(agent_picks)),
        perception_harvest_targets=list(dict.fromkeys(perception_targets)),
        frontier_targets=targets,
    )
