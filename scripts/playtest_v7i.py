#!/usr/bin/env python3
"""Scripted 30-turn playtest for v11-r7i prompt audit follow-up.

Uses the same EngineBridge code path as the Discord bot and the CLI REPL,
but drives a fixed 30-turn script so the playtest is reproducible. Each
turn we capture: input, narrator render, beat-end reason, router rationale
(via INFO log), errors, and warnings. At the end we emit a structured
report grouped by category.

Usage:
    .venv/bin/python scripts/playtest_v7i.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from app.bot.engine_bridge import EngineBridge
from app.llm.config import LLMConfig

load_dotenv()


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SESSION_ID = "playtest_v7i"
STORY_ID = "hollowstone_cleaned_v5"
PLAYER_CHAR_ID = "mira_calder"  # opening protagonist


# Script — 30 distinct turns, designed to exercise:
#   • opening (begin) flow
#   • simple movement through observable facts
#   • dialogue with NPCs
#   • multi-NPC scenes (cascade depth)
#   • environmental/observational beats
#   • a Cat II contested action (touch/grab/push)
#   • time skip OOC
#   • re-traversal of explored scenes
SCRIPT: list[str] = [
    # T1: open the scene
    "(begin)",
    # T2: orient — observation at start location (archive main hall)
    "I take stock of the hall: the shelves, the reading tables, anyone working here.",
    # T3: greet the mentor in the room
    "I find Callum and lower my voice. \"Callum. I want to ask you something private.\"",
    # T4: probing dialogue — push him on a deflection
    "\"Three intake records from last month are missing. I checked the indexes twice. Have you seen anything like that before?\"",
    # T5: follow-up — listen for what he won't say
    "I watch his face for a moment before I speak again. \"You don't have to answer. I just want to know whether to keep asking.\"",
    # T6: pivot — environmental action
    "I walk to the indexes and pull the ledger for last month, looking for the gap.",
    # T7: try to reach the sealed sections (move to adjacent scene)
    "I leave the main hall and head toward the sealed sections. I want to find Warden Maren.",
    # T8: observation in the new scene
    "I take in the sealed sections — the locks, the wards, who is on duty.",
    # T9: address the warden directly
    "If Maren Holt is here, I find her. \"Warden. I have a question about reclassified intake records.\"",
    # T10: wait for the answer, pressing if she deflects
    "I wait for her to speak, and if she gives me nothing, I do not break eye contact.",
    # T11: try to inspect a sealed shelf
    "I move toward the nearest sealed cabinet and read the seal aloud — slowly, in full.",
    # T12: physical interaction — Cat II candidate (touch the seal)
    "I rest my fingertips on the seal, just to feel whether it answers a touch.",
    # T13: defuse — back away
    "I lift my hand and step back. \"Apologies, Warden. Force of habit.\"",
    # T14: leave the sealed sections, return to main hall
    "I leave the sealed sections and walk back to the main hall.",
    # T15: speak to Callum again with what I learned
    "I find Callum. \"Maren wouldn't tell me anything. That itself is an answer.\"",
    # T16: leave the archive entirely — to archive_quarter
    "I walk out of the archive proper, into the archive quarter.",
    # T17: travel to threshold plaza
    "I cross the archive quarter and head to the threshold plaza. I want to see the bell.",
    # T18: observation in plaza
    "I scan the plaza — the bell, the new arrivals, anyone watching me.",
    # T19: speak to anyone present
    "If anyone in plaza colors notices me, I address them. \"I'm Mira Calder, Remembrance. Has anyone seen Theron Vasik today?\"",
    # T20: time skip
    "(wait twenty minutes)",
    # T21: observation after the wait
    "I look around the plaza again. What has changed?",
    # T22: travel further — to gatehouse district
    "I walk to the gatehouse district. If Theron is reaching, that's where he'll be.",
    # T23: dialogue
    "I find a return-court face and ask. \"Where is Theron Vasik?\"",
    # T24: travel deeper — into the reaching halls
    "I head into the reaching halls themselves.",
    # T25: address Theron specifically if present
    "If Theron is in the reaching halls, I approach him. \"Theron. I need to ask you what return looks like from your side.\"",
    # T26: substantive question
    "\"Is reaching the living possible — really possible? Has anyone done it with evidence, not just memory?\"",
    # T27: observe his answer
    "I listen, watching his hands.",
    # T28: change tack — leave
    "I thank him and step away. I want to think.",
    # T29: travel back toward the archive
    "I walk back toward the archive quarter. I want to be alone with the records.",
    # T30: closing reflective beat in the main hall
    "I sit at a reading table in the main hall and write down everything I learned today.",
]


# -----------------------------------------------------------------------------
# Logging — capture INFO so we can scrape decision_rationale and warnings
# -----------------------------------------------------------------------------

class _CaptureHandler(logging.Handler):
    """Captures formatted log records into a buffer for post-run scraping."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.buffer = StringIO()
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        self.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.write(self.format(record))
            self.buffer.write("\n")
        except Exception:
            pass


def _setup_logging() -> _CaptureHandler:
    capture = _CaptureHandler()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    # Quiet down anthropic SDK chatter (lots of HTTP logs).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    return capture


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def _missing_llm_keys(config: LLMConfig) -> list[str]:
    missing: list[str] = []
    if (
        "anthropic" in config.providers_in_use()
        and not config.api_key_for_provider("anthropic")
    ):
        missing.append("ANTHROPIC_API_KEY")
    if "openai" in config.providers_in_use():
        openai_roles = config.roles_for_provider("openai")
        if openai_roles:
            for role in sorted(openai_roles):
                if config.api_key_for_provider("openai", role=role):
                    continue
                role_env = config.openai_role_api_key_env_names(role)[0]
                missing.append(f"{role_env} or OPENAI_API_KEY for {role}")
        elif not config.api_key_for_provider("openai"):
            missing.append("OPENAI_API_KEY")
    return missing


async def _run() -> None:
    missing_llm_keys = _missing_llm_keys(LLMConfig.from_env())
    if missing_llm_keys:
        missing = ", ".join(missing_llm_keys)
        print(f"ERROR: {missing} not set", file=sys.stderr)
        sys.exit(1)

    capture = _setup_logging()
    bridge = EngineBridge()

    # Wipe any prior playtest_v7i session so we start fresh.
    sessions_dir = Path("app/storage/sessions") / SESSION_ID
    if sessions_dir.exists():
        for f in sessions_dir.iterdir():
            f.unlink()
        sessions_dir.rmdir()

    bridge.create_empty_session(SESSION_ID)
    bridge.load_story_into_session(SESSION_ID, STORY_ID)

    # Bind the protagonist to a synthetic Discord user_id (1).
    bridge.bind_user(
        session_id=SESSION_ID, user_id="1", character_id=PLAYER_CHAR_ID,
    )

    print(f"=== playtest v11-r7i — {SESSION_ID} / {STORY_ID} ===\n")

    turns_log: list[dict] = []

    for i, user_input in enumerate(SCRIPT, start=1):
        print(f"\n--- T{i:02d} > {user_input}")
        try:
            response = await bridge.run_turn(
                session_id=SESSION_ID,
                user_input=user_input,
                acting_character_id=PLAYER_CHAR_ID,
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            turns_log.append({
                "turn": i,
                "input": user_input,
                "error": f"{type(e).__name__}: {e}",
                "render": "",
                "ended_reason": "",
            })
            continue

        per_pov = response.per_player_renders or {}
        actor_render = per_pov.get(PLAYER_CHAR_ID, response.output_text or "")
        ended_reason = response.beat_ended_reason or ""

        if actor_render:
            preview = actor_render.strip()
            print(preview)
        print(f"[ended_reason={ended_reason!r}]")

        turns_log.append({
            "turn": i,
            "input": user_input,
            "render": actor_render,
            "ended_reason": ended_reason,
        })

    # Save full log buffer to a file for post-run analysis.
    log_path = Path("playtest_v7i_logs.txt")
    log_path.write_text(capture.buffer.getvalue())
    print(f"\nFull logs written to {log_path} ({log_path.stat().st_size} bytes)")

    # Summary report.
    print("\n=== SUMMARY ===")
    print(f"Turns completed: {sum(1 for t in turns_log if not t.get('error'))}/{len(SCRIPT)}")
    errors = [t for t in turns_log if t.get("error")]
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  T{e['turn']}: {e['error']}")

    # Bucket ended_reasons.
    bucket: dict[str, int] = defaultdict(int)
    for t in turns_log:
        bucket[t.get("ended_reason", "")] += 1
    print("\nended_reason distribution:")
    for r, c in sorted(bucket.items(), key=lambda kv: -kv[1]):
        print(f"  {r!r}: {c}")

    # Final character state.
    ckpt = bridge.load_latest(SESSION_ID)
    mira = next(
        (c for c in ckpt.characters if c.character_id == PLAYER_CHAR_ID),
        None,
    )
    print(f"\nFinal Mira location: {mira.location if mira else '?'}")
    print(f"Total characters in roster: {len(ckpt.characters)}")
    distinct_locations = {c.location for c in ckpt.characters if c.location}
    print(f"Total occupied location labels: {len(distinct_locations)}")

    print("\nDone. Inspect playtest_v7i_logs.txt for full log + decision rationales.")


if __name__ == "__main__":
    asyncio.run(_run())
