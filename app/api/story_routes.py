"""Story API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.engine.orchestrator import Orchestrator
from app.schemas.requests import TurnRequest
from app.schemas.responses import TurnResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/story", tags=["story"])

# Orchestrator is set during app startup
_orchestrator: Orchestrator | None = None


def set_orchestrator(orchestrator: Orchestrator) -> None:
    """Set the orchestrator instance (called during app startup)."""
    global _orchestrator
    _orchestrator = orchestrator


@router.post("/turn", response_model=TurnResponse)
async def process_turn(request: TurnRequest) -> TurnResponse:
    """Process a single turn in the story."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        return await _orchestrator.process_turn(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Turn processing failed")
        raise HTTPException(status_code=500, detail=f"Turn processing failed: {e}")
