#!/usr/bin/env python3

import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bot.engine_bridge import EngineBridge

SESSION_ID = "playtest_escape_room_less_handholding_10"
STORY_ID = "escape_room_s1"
USER_ID = 1001
REPORT_DIR = Path("app/storage/playtest_reports")
REPORT_PATH = REPORT_DIR / f"{SESSION_ID}.md"
JSON_PATH = REPORT_DIR / f"{SESSION_ID}.json"

ACTIONS = [
    "I stay still for a breath, then take a slow inventory of the room: door, keypad, table, clock, sink, cabinet, vent, and speaker.",
    "I examine the wall clock closely, checking whether it is running and whether anything about the hands looks deliberate.",
    "I crouch beside the bolted table and run my fingers along the underside and edges, looking for marks, seams, or anything taped there.",
    "I try the cabinet handle once, then inspect the cabinet door and frame for labels, scratches, gaps, or anything hidden in the finish.",
    "I turn on the sink and splash water over my hands, then let the room's moisture and my wet fingers fog or streak the cabinet door.",
    "I study whatever mark appears on the cabinet, then check around the sink and cabinet for a switch, breaker, loose panel, or power lead.",
    "I go to the keypad and enter 731, then stop before the final digit to see whether the keypad reacts to partial input.",
    "I enter 7319 on the keypad and listen carefully to the door, keypad, cabinet, vent, and speaker for any response.",
    "If the door is still locked, I pry at the keypad cover or any loosened panel with whatever edge or tool I can find in the room.",
    "I check the vent grille, remove or loosen it if I can, and search inside for anything that could work with the door or keypad.",
]


async def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    session_dir = Path("app/storage/sessions") / SESSION_ID
    if session_dir.exists():
        shutil.rmtree(session_dir)

    engine = EngineBridge()
    records = []
    char = None
    try:
        engine.create_empty_session(SESSION_ID)
        engine.load_story_into_session(SESSION_ID, STORY_ID)
        char = engine.create_player_character_simple(
            SESSION_ID,
            USER_ID,
            name="Morgan Vale",
            appearance=(
                "A practical, tired-looking escape-room hobbyist in a gray "
                "hoodie, jeans, and scuffed sneakers."
            ),
            backstory=(
                "Morgan signs up for strange puzzle rooms whenever work gets "
                "too quiet."
            ),
        )

        begin = await engine.run_begin_turn(
            session_id=SESSION_ID,
            triggering_character_id=char.character_id,
        )
        records.append({
            "turn": "begin",
            "input": "(begin)",
            "reason": begin.beat_ended_reason,
            "output": begin.output_text,
        })

        for idx, action in enumerate(ACTIONS, 1):
            resp = await engine.run_turn(
                session_id=SESSION_ID,
                user_input=action,
                acting_character_id=char.character_id,
            )
            records.append({
                "turn": idx,
                "input": action,
                "reason": resp.beat_ended_reason,
                "output": resp.output_text,
            })
    finally:
        await engine.close()

    ckpt_files = sorted(session_dir.glob("ckpt_*.json"))
    latest = json.loads(ckpt_files[-1].read_text()) if ckpt_files else {}
    router_events = []
    for ev in latest.get("canonical_events", []):
        router_events.append({
            "event_id": ev.get("event_id"),
            "decision_rationale": ev.get("decision_rationale"),
            "event_kind": ev.get("event_kind"),
            "observers": ev.get("observers"),
            "facts": ev.get("canonical_event", {}).get("observable_facts", []),
        })

    JSON_PATH.write_text(json.dumps({
        "session_id": SESSION_ID,
        "story_id": STORY_ID,
        "character_id": char.character_id if char else "",
        "records": records,
        "router_events": router_events,
    }, indent=2))

    lines = [
        "# Escape Room 10-Turn Playtest",
        "",
        f"Session: `{SESSION_ID}`",
        f"Story: `{STORY_ID}`",
        f"Character: `{char.character_id if char else ''}`",
        "",
    ]
    for rec in records:
        lines.extend([
            f"## Turn {rec['turn']}",
            "",
            f"Input: {rec['input']}",
            f"Beat reason: `{rec['reason']}`",
            "",
            rec["output"].strip() or "(no render)",
            "",
        ])
    lines.extend(["## Router Events", ""])
    for i, ev in enumerate(router_events, 1):
        lines.append(f"### Event {i}: {ev.get('event_id')}")
        lines.append(f"event_kind={ev.get('event_kind')}")
        lines.append(f"rationale: {ev.get('decision_rationale')}")
        facts = ev.get("facts") or []
        for fact in facts:
            if isinstance(fact, dict):
                lines.append(f"- [{fact.get('audience')}] {fact.get('text')}")
            else:
                lines.append(f"- {fact}")
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines))

    print(REPORT_PATH)
    print(JSON_PATH)
    latest_name = ckpt_files[-1].name if ckpt_files else "none"
    print(f"records={len(records)} events={len(router_events)} latest={latest_name}")
    for rec in records:
        print(
            f"TURN {rec['turn']} reason={rec['reason']} "
            f"chars={len(rec['output'])}"
        )


asyncio.run(main())
