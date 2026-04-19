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
    """Replace PLAYER_NAME placeholder with the actual player name and bind the
    player character. Finds the player character by its `is_player=True` flag
    in the roster — the master prompt must designate one.

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

    try:
        personalized, new_char_id = _apply_personalize(checkpoint, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Bump turn_index so we save as ckpt_0001, keeping ckpt_0000 pristine
    # for --new resets.
    if personalized.session.turn_index == 0:
        personalized.session.turn_index = 1

    _checkpoint_mgr.save(personalized)

    logger.info("Personalized story %s: player=%s (%s)", request.session_id, name, new_char_id)
    return {"status": "ok", "player_name": name, "player_character_id": new_char_id}


def _apply_personalize(checkpoint, name: str):
    """Rewrite a checkpoint so it's bound to `name` as the player character.

    Returns (new_checkpoint, new_character_id). Raises ValueError if the roster
    has no is_player character to personalize.
    """
    from app.schemas.checkpoint import CheckpointFile

    player_chars = [c for c in checkpoint.characters if c.is_player]
    if not player_chars:
        raise ValueError(
            "Roster has no is_player=true character. Re-import the master "
            "prompt with a protagonist explicitly designated."
        )
    if len(player_chars) > 1:
        logger.warning(
            "Roster has %d is_player characters; personalize binds only the first "
            "(%s). Multi-player support lands in Phase 3.",
            len(player_chars), player_chars[0].name,
        )

    pc = player_chars[0]
    old_char_id = pc.character_id
    new_char_id = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "player"

    # JSON round-trip with string replace: handles the character_id in its own
    # record, in any other character's attitudes dict keys, and in
    # known_characters list entries simultaneously. Quoting the search string
    # avoids matching mid-substring in prose fields.
    raw = checkpoint.model_dump_json()
    raw = raw.replace("PLAYER_NAME", name)
    raw = raw.replace(f'"{old_char_id}"', f'"{new_char_id}"')
    data = json.loads(raw)

    data["session"]["player_name"] = name
    data["session"]["player_character_id"] = new_char_id

    known = data.get("world_state", {}).get("known_characters", [])
    if new_char_id not in known:
        known.append(new_char_id)
    data.setdefault("world_state", {})["known_characters"] = known

    return CheckpointFile.model_validate(data), new_char_id


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
