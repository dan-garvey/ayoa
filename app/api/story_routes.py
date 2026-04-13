"""Story API routes."""

from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.engine.checkpoint_manager import CheckpointManager
from app.engine.orchestrator import Orchestrator
from app.schemas.requests import PersonalizeRequest, TurnRequest
from app.schemas.responses import TurnResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/story", tags=["story"])

# Set during app startup
_orchestrator: Orchestrator | None = None
_checkpoint_mgr: CheckpointManager | None = None


def set_orchestrator(orchestrator: Orchestrator) -> None:
    """Set the orchestrator instance (called during app startup)."""
    global _orchestrator, _checkpoint_mgr
    _orchestrator = orchestrator
    _checkpoint_mgr = orchestrator.checkpoint_mgr


@router.post("/turn", response_model=TurnResponse)
async def process_turn(request: TurnRequest) -> TurnResponse:
    """Process a single turn in the story.

    If request.stream is True, returns an SSE stream with incremental
    progress events followed by the final response.
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if request.stream:
        return StreamingResponse(
            _stream_turn(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        return await _orchestrator.process_turn(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Turn processing failed")
        raise HTTPException(status_code=500, detail=f"Turn processing failed: {e}")


@router.post("/personalize")
async def personalize_story(request: PersonalizeRequest):
    """Replace PLAYER_NAME placeholder with the actual player name.

    Call this once before the first turn to personalize the story.
    """
    if _checkpoint_mgr is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        checkpoint = _checkpoint_mgr.load_latest(request.session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    name = request.player_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Player name cannot be empty")

    # Build new character_id from name
    old_char_id = "player_name_garvey"
    new_char_id = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + "_garvey"

    # Serialize to JSON, do the replacement, deserialize back
    data = json.loads(checkpoint.model_dump_json())
    raw = json.dumps(data)
    raw = raw.replace("PLAYER_NAME", name)
    raw = raw.replace(old_char_id, new_char_id)
    data = json.loads(raw)

    # Set player identity on session
    data["session"]["player_name"] = name
    data["session"]["player_character_id"] = new_char_id

    # Ensure the player character is in known_characters (player always knows their own name)
    known = data.get("world_state", {}).get("known_characters", [])
    if new_char_id not in known:
        known.append(new_char_id)
        data["world_state"]["known_characters"] = known

    # Bump turn_index so we save as ckpt_0001, keeping ckpt_0000 pristine
    # for --new resets
    if data["session"].get("turn_index", 0) == 0:
        data["session"]["turn_index"] = 1

    # Reload as checkpoint and save
    from app.schemas.checkpoint import CheckpointFile
    checkpoint = CheckpointFile(**data)
    _checkpoint_mgr.save(checkpoint)

    logger.info("Personalized story %s: player=%s", request.session_id, name)
    return {"status": "ok", "player_name": name, "player_character_id": new_char_id}


async def _stream_turn(request: TurnRequest):
    """Generator that yields SSE events during turn processing."""
    try:
        yield _sse_event("status", {"phase": "started"})
        response = await _orchestrator.process_turn(request)
        # Stream the final text in chunks
        text = response.output_text
        chunk_size = 80
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            yield _sse_event("chunk", {"text": chunk})
        # Final event with full response
        yield _sse_event("done", response.model_dump())
    except FileNotFoundError as e:
        yield _sse_event("error", {"detail": str(e)})
    except Exception as e:
        logger.exception("Streaming turn failed")
        yield _sse_event("error", {"detail": f"Turn processing failed: {e}"})


def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
