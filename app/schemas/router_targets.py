from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.schemas.event_router import EventRouterOutput

RouterTargetFrame = Literal[
    "foreground",
    "private",
    "background",
]


class RouterOutputTarget(BaseModel):
    """One agent turn selected from a router output.

    This is not an LLM output model yet. It is the runtime contract that
    lets the turn loop project the router's routing roles into dispatchable
    work. Same-scene agent turns are consumed sequentially: one public result
    is sent back to the router, canonicalized, and only then can another
    routed character respond with updated context.
    """

    model_config = ConfigDict(extra="forbid")

    character_id: str
    frame: RouterTargetFrame
    source_event_id: str


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
) -> list[RouterOutputTarget]:
    """Build agent-turn targets from semantic routing roles.

    Event kind does not decide whether an actor is autonomous. The binding-aware
    turn loop resolves each ordered `next_output` id before considering a soft
    render candidate.
    """
    return [
        RouterOutputTarget(
            character_id=cid,
            frame=router_frame_for_pick(
                result,
                player_ids=player_ids,
                character_id=cid,
            ),
            source_event_id=result.event_id,
        )
        for cid in agent_ids
    ]
