"""CLI wrapper around the import pipeline.

Usage:
    .venv/bin/python scripts/import_story.py story_prompt.txt [--output path] [--story-id id]

The pipeline itself lives in `app/engine/story_importer.py` so the Discord
bot can call the same code path without going through the shell.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from app.engine.story_importer import run_import
from app.llm.client import LLMClient
from app.llm.config import LLMConfig

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main_async(source_path: str, output_path: str, story_id: str) -> None:
    with open(source_path, "r") as f:
        source = f.read()

    client = LLMClient(config=LLMConfig.from_env())
    try:
        checkpoint = await run_import(client, source, story_id)
    finally:
        await client.close()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(checkpoint.model_dump_json(indent=2))

    logger.info("\nCheckpoint written to %s", output_path)
    logger.info("  Session ID: %s", checkpoint.session.session_id)
    logger.info("  Characters: %d", len(checkpoint.characters))
    logger.info("  Genre: %s", checkpoint.world_state.setting.genre)
    logger.info("  Lore length: %d chars", len(checkpoint.world_state.lore))
    logger.info("  Hidden lore length: %d chars", len(checkpoint.world_state.hidden_lore))
    logger.info("  Narrative rules length: %d chars", len(checkpoint.config.narrative_rules))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a master IF prompt into engine format")
    parser.add_argument("source", help="Path to the source prompt text file")
    parser.add_argument("--output", "-o", default=None, help="Output checkpoint JSON path")
    parser.add_argument("--story-id", default=None, help="Story ID for the checkpoint")
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.source))[0]
    story_id = args.story_id or base
    output_path = args.output or f"app/storage/saves/{story_id}/ckpt_0000.json"

    asyncio.run(main_async(args.source, output_path, story_id))


if __name__ == "__main__":
    main()
