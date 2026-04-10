from contextlib import asynccontextmanager
import logging

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.story_routes import router as story_router, set_orchestrator
from app.engine.checkpoint_manager import CheckpointManager
from app.engine.orchestrator import Orchestrator
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.llm.config import LLMConfig

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_client: LLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    config = LLMConfig.from_env()
    _client = LLMClient(config=config)
    checkpoint_mgr = CheckpointManager()
    prompt_mgr = PromptManager()
    orchestrator = Orchestrator(_client, checkpoint_mgr, prompt_mgr)
    set_orchestrator(orchestrator)
    logging.getLogger(__name__).info("Engine initialized")
    yield
    if _client:
        await _client.close()


app = FastAPI(title="Narrative Engine", version="0.1.0", lifespan=lifespan)
app.include_router(story_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
