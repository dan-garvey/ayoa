"""CLI wrapper around the import pipeline.

Usage:
    .venv/bin/python scripts/import_story.py story_prompt.txt [--output path] [--story-id id] [--no-analysis]

The pipeline itself lives in `app/engine/story_importer.py` so the Discord
bot can call the same code path without going through the shell.

Preservation analysis runs inline by default — the CLI event loop exits
when main_async returns, so a background task would be cancelled before
completing. Pass --no-analysis to skip it.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from app.engine.story_importer import run_import, run_preservation_analysis
from app.llm.client import LLMClient
from app.llm.config import LLMConfig

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main_async(
    source_path: str,
    output_path: str,
    story_id: str,
    run_analysis: bool,
) -> None:
    with open(source_path, "r") as f:
        source = f.read()

    client = LLMClient(config=LLMConfig.from_env())
    try:
        checkpoint = await run_import(client, source, story_id)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(checkpoint.model_dump_json(indent=2))

        logger.info("\nCheckpoint written to %s", output_path)
        logger.info("  Importer:   %s", checkpoint.importer_version)
        logger.info("  Characters: %d", len(checkpoint.characters))
        logger.info("  Genre:      %s", checkpoint.world_state.setting.genre)
        logger.info("  Lore:       %d chars", len(checkpoint.world_state.lore))
        logger.info("  Hidden:     %d chars", len(checkpoint.world_state.hidden_lore))
        logger.info("  Rules:      %d chars", len(checkpoint.config.narrative_rules))

        if run_analysis:
            logger.info("\nRunning preservation analysis (inline — this adds a minute or two)...")
            analysis = await run_preservation_analysis(client, source, checkpoint)
            checkpoint.import_analysis = analysis
            with open(output_path, "w") as f:
                f.write(checkpoint.model_dump_json(indent=2))
            logger.info(
                "  Coverage: %s  (source %dw → output %dw)",
                analysis.coverage_rating,
                analysis.source_words, analysis.output_words,
            )
            if analysis.dropped_topics:
                logger.info("  Dropped topics (%d):", len(analysis.dropped_topics))
                for t in analysis.dropped_topics:
                    logger.info("    - %s", t)
            if analysis.compressed_topics:
                logger.info("  Compressed topics (%d):", len(analysis.compressed_topics))
                for t in analysis.compressed_topics:
                    logger.info("    - %s", t)
        else:
            logger.info("\n(Skipping preservation analysis; pass --analysis to run.)")
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a master IF prompt into engine format")
    parser.add_argument("source", help="Path to the source prompt text file")
    parser.add_argument("--output", "-o", default=None, help="Output checkpoint JSON path")
    parser.add_argument("--story-id", default=None, help="Story ID for the checkpoint")
    parser.add_argument(
        "--no-analysis", action="store_true",
        help="Skip the preservation-analysis pass (faster, but leaves "
             "import_analysis=null on the checkpoint).",
    )
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.source))[0]
    story_id = args.story_id or base
    output_path = args.output or f"app/storage/stories/{story_id}/ckpt_0000.json"

    asyncio.run(main_async(
        args.source, output_path, story_id,
        run_analysis=not args.no_analysis,
    ))


if __name__ == "__main__":
    main()
