"""FastAPI server for the game engine."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.engine.orchestrator import orchestrator
from core.models.schemas import (
    AgentState,
    PlayerCharacter,
    StoryConfig,
    StoryOutput,
    StoryPreferences,
)

app = FastAPI(title="Multi-Agent Narrative Game API", version="0.1.0")


# Request/Response models
class CreateStoryRequest(BaseModel):
    """Request to create a new story."""

    player_character: PlayerCharacter
    preferences: StoryPreferences


class CreateStoryResponse(BaseModel):
    """Response from creating a story."""

    story_id: str
    character_sheet: dict
    outline: dict


class StartStoryRequest(BaseModel):
    """Request to start a story."""

    story_id: str


class StartStoryResponse(BaseModel):
    """Response from starting a story."""

    story_id: str
    opening: StoryOutput


class TurnRequest(BaseModel):
    """Request to process a turn."""

    story_id: str
    user_input: str


class SaveRequest(BaseModel):
    """Request to save a story."""

    story_id: str


class SaveResponse(BaseModel):
    """Response from saving."""

    saved: bool


class LoadRequest(BaseModel):
    """Request to load a story."""

    story_id: str


class LoadResponse(BaseModel):
    """Response from loading."""

    loaded: bool


class InspectResponse(BaseModel):
    """Response from inspect command."""

    data: dict


# Endpoints
@app.post("/story/create", response_model=CreateStoryResponse)
async def create_story(request: CreateStoryRequest):
    """Create a new story with character and preferences."""
    try:
        config = StoryConfig(
            player_character=request.player_character, preferences=request.preferences
        )

        story_id, outline = await orchestrator.create_story(config)

        return CreateStoryResponse(
            story_id=story_id,
            character_sheet=request.player_character.model_dump(),
            outline=outline.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/story/start", response_model=StartStoryResponse)
async def start_story(request: StartStoryRequest):
    """Start a created story - spawns agents and generates opening."""
    try:
        opening = await orchestrator.start_story(request.story_id)

        return StartStoryResponse(story_id=request.story_id, opening=opening)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/story/turn", response_model=StoryOutput)
async def process_turn(request: TurnRequest):
    """Process a turn of gameplay."""
    try:
        output = await orchestrator.process_turn(request.story_id, request.user_input)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/story/save", response_model=SaveResponse)
async def save_story(request: SaveRequest):
    """Save current story state."""
    try:
        orchestrator._save_story_state(request.story_id)
        return SaveResponse(saved=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/story/load", response_model=LoadResponse)
async def load_story(request: LoadRequest):
    """Load story state."""
    try:
        orchestrator._load_story_state(request.story_id)
        return LoadResponse(loaded=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/story/agents/{story_id}", response_model=list[AgentState])
async def get_agents(story_id: str):
    """Get all agents for a story."""
    try:
        # Load story if not current
        if orchestrator.current_story_id != story_id:
            orchestrator._load_story_state(story_id)

        agents = orchestrator.agent_manager.list_agents()
        return list(agents.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/story/inspect/{story_id}", response_model=InspectResponse)
async def inspect_story(story_id: str, what: str = "scene"):
    """Inspect various aspects of the story."""
    try:
        # Load story if not current
        if orchestrator.current_story_id != story_id:
            orchestrator._load_story_state(story_id)

        if what == "scene":
            data = orchestrator.current_scene.model_dump() if orchestrator.current_scene else {}
        elif what == "cast":
            agents = orchestrator.agent_manager.list_agents()
            data = {
                agent_id: {"name": state.dossier.name, "role": state.dossier.character_concept.role}
                for agent_id, state in agents.items()
            }
        elif what == "outline":
            data = orchestrator.current_outline.model_dump() if orchestrator.current_outline else {}
        elif what == "eventlog":
            data = {"events": orchestrator.turn_history}
        elif what == "dossiers":
            agents = orchestrator.agent_manager.list_agents()
            data = {agent_id: state.dossier.model_dump() for agent_id, state in agents.items()}
        else:
            data = {"error": f"Unknown inspection type: {what}"}

        return InspectResponse(data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/state/{agent_id}", response_model=AgentState)
async def get_agent_state(agent_id: str):
    """Get state for a specific agent."""
    try:
        agent = orchestrator.agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        state = AgentState(
            agent_id=agent.agent_id,
            dossier=agent.dossier,
            turn_memory=agent.memory_buffer.copy(),
            active=True,
            last_action_turn=len(agent.memory_buffer),
        )
        return state
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
