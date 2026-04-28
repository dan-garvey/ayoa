#!/usr/bin/env python3
"""Scripted 30+ turn playtest of dating_villa_s1 with ticks disabled.

Drives `EngineBridge.run_turn` directly so each render + diagnostic is
captured to a structured log file rather than a TTY. Action plan is
designed to surface the things the v11 reviewer agents flagged for
playtest, restricted to the scenarios that don't require off-stage
ticks (since ticks_enabled=False for this run):

  - Unified agent prompt mode-routing (only ON-STAGE fires; TICK never).
  - Cat II pin-and-resolve, including all-NPC pins.
  - Multi-NPC cascade in one beat (cap behavior).
  - Cross-character communication via couriers / overheard speech.
  - Silent-beat sentinel (paren-only output → "(remains silent)").
  - Multi-paragraph restraint (agents shouldn't wall-of-text).
  - Parenthetical-as-name misfire (agent names a person in the trailing
    parenthetical instead of stating intent).
  - Save/reload mid-session: snapshot turn 15, fresh bridge, continue.
  - Ticks_enabled=False kill switch verification (no tick fan-out).

Outputs a JSONL log at logs/playtest_villa_no_ticks_<ts>.jsonl with one
record per turn (action, render, latencies, beat_ended_reason, errors)
plus a summary.txt with aggregate counters and headline findings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from app.bot.engine_bridge import EngineBridge

load_dotenv()

SESSION = "playtest_v11_villa_no_ticks"
STORY = "dating_villa_s1"
PLAYER_USER_ID = 1
PLAYER_CHAR = "jordan_reeves"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
LOG_PATH = LOG_DIR / f"playtest_villa_no_ticks_{TS}.jsonl"
SUMMARY_PATH = LOG_DIR / f"playtest_villa_no_ticks_{TS}_summary.txt"
ENGINE_LOG = LOG_DIR / f"playtest_villa_no_ticks_{TS}_engine.log"

# Engine logger → file (kept separate from per-turn JSONL so I can grep
# tick activity, cache stats, retries without blowing up the turn log).
file_handler = logging.FileHandler(ENGINE_LOG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s",
))
root = logging.getLogger()
root.addHandler(file_handler)
root.setLevel(logging.INFO)


# Action script. Each entry is (label, user_input, intent_for_log).
# Designed to hit the reviewer-flagged scenarios in order so I can
# correlate findings to specific turns when reading the log.
ACTIONS: list[tuple[str, str, str]] = [
    # 1: open the scene at the reveal threshold; tests opening directive
    # honoring + first-token unified-prompt routing for everyone present.
    ("open_scene", "(begin)",
     "Opening beat — Marcus reveal moment, no prior history."),

    # 2: low-key player line, deliberately short. Tests sparseness / silent-
    # beat sentinel for any NPC the router pulls in.
    ("light_dialogue", "I take a half-step forward and meet Marcus's eyes, "
     "saying nothing, just waiting for him to choose what to say first.",
     "Tests sparse output + agent restraint."),

    # 3: Cat II initiation directed at Marcus. Player intends an
    # interpersonal action that requires Marcus's response to land.
    ("cat_ii_to_npc", "I extend my hand, palm up, and ask quietly: "
     "'do you want to do the awkward producer-handshake or the version "
     "where we figure it out ourselves?'",
     "Cat II pin onto Marcus."),

    # 4: continued Cat II — let Marcus's response close the pin via
    # cascade adjudication.
    ("cat_ii_followup", "I let his answer set the tempo and follow his lead "
     "into the next beat — body language only, no fresh question.",
     "Tests cascade closure of the pinned event."),

    # 5: movement via natural language.
    ("scene_move_request",
     "I gesture toward the citrus garden. 'walk with me a minute? "
     "the great hall feeds on first impressions.' I head toward the "
     "private citrus garden, expecting him to follow.",
     "Movement via NL."),

    # 6: confessional dialogue in a private space. Tests asymmetry —
    # nothing in this prose should leak as parenthetical to other NPCs.
    ("confessional", "Once we're past the arch, I tell him the thing I'd "
     "decided to keep back: the old exposé brushed his ex-sponsor. I "
     "say it flat, no apology in it, and watch his face.",
     "Asymmetry — confidential disclosure."),

    # 7: deliberate silent beat by player. Want to see whether NPC
    # agent reciprocates with restraint or walls of text.
    ("player_silence", "I don't speak. I let the silence sit and look at "
     "the citrus tree behind his shoulder.",
     "Tests NPC restraint when player gives no opening."),

    # 8: navigate back through villa. Multiple connected scenes — should
    # exercise the new unified-router move handling.
    ("move_to_pool", "I step back through the arch and head toward the "
     "pool deck — I want to see who else is around before the group toast.",
     "Move test, leaving Marcus behind in citrus garden."),

    # 9: enter multi-NPC space. Should pull several NPCs into cascade,
    # router uses ends_beat to do pacing.
    ("multi_npc_intro", "At the pool deck I drift toward whichever group "
     "has the loudest laugh and introduce myself: 'jordan, pod nine, "
     "cameras keep telling me to smile so I'm trying to do it on accident.'",
     "Multi-NPC cascade test."),

    # 10: ask a direct question of one NPC in a group. Tests whether the
    # router pins a Cat II just on them or lets others chime in.
    ("targeted_question", "I turn directly to whoever introduced themselves "
     "first and ask: 'who here is pretending the hardest right now? "
     "everyone or just me?'",
     "Tests Cat II vs cascade — directed question in a group."),

    # 11: short refusal-style player beat. Want NPC reactions to be
    # equally short — restraint test.
    ("brief_refusal", "I shake my head once. 'no — different question. "
     "what do you actually want from this?'",
     "Tests reciprocal sparseness."),

    # 12: cross-scene reference. The other NPCs at the pool weren't in
    # the citrus garden when Jordan disclosed the exposé — none of them
    # should know about it. Tests information asymmetry across scenes.
    ("asymmetry_probe",
     "I deflect the answering question with a half-shrug — 'i told marcus "
     "something earlier, that's all.' I leave it ambiguous and watch which "
     "of them perks up like they already heard.",
     "Information asymmetry — only Marcus should know about the exposé."),

    # 13: courier-style message. Tests router rule 13 (cross-scene
    # communication must materialize as engine state).
    ("courier", "I flag a producer-runner over and ask them to take a note "
     "to Marcus in the citrus garden: 'tell him i'll find him after the "
     "toast — i didn't mean it as ammunition.'",
     "Tests router rule 13 — explicit courier."),

    # 14: announce a movement change. Saves capture happens after this.
    ("plan_move", "I peel off from the pool group with a 'see you at the "
     "toast' and head toward the great hall — i want to see who's already "
     "claimed the room.",
     "Move 2 + scene-state ahead of save/reload."),

    # 15: SAVE-RELOAD CHECKPOINT. The driver tears down the bridge here.
    ("scene_in_great_hall",
     "At the great hall I take a glass off a passing tray, turn so a "
     "camera can see my back and not my face, and just listen to the "
     "room for a beat.",
     "Last action before save/reload split."),

    # 16: post-reload. Driver restarts the bridge, reloads from disk,
    # continues. Tests checkpoint round-trip + rolling history fidelity.
    ("post_reload_dialogue",
     "I find priya in the room and ask, low: 'how loaded is vik tonight, "
     "honestly?' — I want a read before he gets to me.",
     "Post-reload continuity test."),

    # 17: deliberate paren-as-name bait. The player's prose is sparse
    # and sets up an NPC to potentially mis-format. Watch the agent
    # parenthetical extraction.
    ("paren_bait", "I nod once. nothing else.",
     "Parenthetical-as-name misfire bait — minimal prose, "
     "lots of silence room."),

    # 18: provoke a tense exchange — Cat II onto Vik when he arrives.
    ("provoke_vik",
     "When vik makes his entrance i wait until he's three steps in and "
     "say, conversational, loud enough for the nearer table: 'vik. you "
     "still going by 'serial founder' or have we settled on something else.'",
     "Tense Cat II to Vik in front of an audience."),

    # 19: let the moment continue. Tests cascade post-Cat-II closure
    # with audience NPCs reacting on their own initiative.
    ("audience_reaction", "I hold my drink at chest height and don't move. "
     "I let the room react first.",
     "Audience cascade after a charged beat."),

    # 20: short multi-target line. Tests whether router fans to the right
    # set of responders, not all of them.
    ("multi_target",
     "I tilt my glass toward priya and noah and say: 'someone change the "
     "subject — i don't want this to be the moment of the night.'",
     "Multi-target dialogue — router responder selection."),

    # 21: emotional self-reveal in a quiet aside. Tests that the
    # parenthetical (private intent) doesn't leak to anyone hearing only
    # the prose.
    ("self_reveal",
     "I drift to the rail and stand next to noah and say, off-mic-low: "
     "'i'm bad at this part. the part where i pretend i don't already "
     "know who i'm choosing.'",
     "Asymmetry — quiet line, intent stays private."),

    # 22: mid-game scene move back to citrus garden — testing whether
    # Marcus is still there or has moved on his own (he shouldn't have,
    # since ticks are off).
    ("return_to_marcus",
     "I excuse myself from noah and head back to the citrus garden — "
     "i need to close the loop with marcus.",
     "Tick kill-switch verification — Marcus should be where I left him."),

    # 23: conversational repair. Tests a cooperative exchange (no Cat II).
    ("repair_dialogue",
     "If marcus is there i say plain: 'the message lands flat in writing. "
     "what i meant is — the exposé sat on me. it didn't sit on you. "
     "i'd rather you have it from me than the producers.'",
     "Cooperative dialogue — extended exchange."),

    # 24: explicit two-character intimate beat. Cat II ('do you forgive me')
    # should pin Marcus.
    ("intimate_cat_ii",
     "I take half a step closer. 'are you still in this — pod-jordan and "
     "stone-jordan, both?'",
     "Intimate Cat II."),

    # 25: response-driven follow. Pin closure.
    ("intimate_close",
     "Whatever he says i answer with one sentence and let it land.",
     "Pin closure — agent restraint."),

    # 26: scene change to public again. Group dynamics return.
    ("return_public",
     "We walk back together to the great hall when the bell rings for "
     "the toast.",
     "Movement with NPC in tow."),

    # 27: communal beat. Many NPCs present.
    ("toast",
     "At the toast I raise my glass when prompted and say only: "
     "'to the part of this that's real.' nothing more.",
     "Single-line group beat — restraint at scale."),

    # 28: probe whether off-stage NPCs have inherited any state. Ticks
    # off — they shouldn't have done anything since their last on-stage
    # appearance.
    ("offstage_state_probe",
     "I scan the room for elena and marisol — i haven't seen either of "
     "them in two beats. where are they.",
     "Asks engine to surface off-stage state — should be static."),

    # 29: end-of-night confessional. Tests narrator restraint on long
    # emotional beats.
    ("nightcap",
     "Late, on a couch in the lobby atrium, i tell whoever's still up: "
     "'i think i embarrassed myself twice tonight and meant once. that "
     "feels close to honest.'",
     "Reflective nightcap — narrator pacing test."),

    # 30: deliberate end-state. Player retires.
    ("retire",
     "I rinse my glass at the bar and head toward my room — done for the "
     "night.",
     "Closing move."),

    # 31: bonus — re-engage one more turn after retiring to see how
    # narrator handles a 'next morning' jump or a single-character scene.
    ("morning",
     "Morning. I'm up before the rest, on the pool deck with coffee and "
     "no makeup. I want to see who else couldn't sleep.",
     "Time skip / single-occupancy scene."),
]


def _record(rec: dict) -> None:
    """Append a single JSON record to the playtest log."""
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


async def _run_one(
    bridge: EngineBridge,
    label: str,
    user_input: str,
    intent: str,
    turn_no: int,
) -> dict:
    """Run a single turn and return a structured record."""
    t0 = time.monotonic()
    error = None
    response = None
    try:
        response = await bridge.run_turn(
            session_id=SESSION,
            user_input=user_input,
            acting_character_id=PLAYER_CHAR,
        )
    except Exception:
        error = traceback.format_exc()

    wall_ms = (time.monotonic() - t0) * 1000.0

    rec: dict = {
        "turn_no": turn_no,
        "label": label,
        "user_input": user_input,
        "intent": intent,
        "wall_ms": round(wall_ms, 1),
    }
    if error:
        rec["error"] = error
        return rec

    rec["session_id"] = response.session_id
    rec["checkpoint_id"] = response.checkpoint_id
    rec["turn_index"] = response.turn_index
    rec["beat_ended_reason"] = response.beat_ended_reason
    rec["output_text"] = response.output_text
    rec["per_player_renders"] = response.per_player_renders
    rec["pre_turn_resolutions"] = [
        {
            "checkpoint_id": pre.checkpoint_id,
            "beat_ended_reason": pre.beat_ended_reason,
            "per_player_renders": pre.per_player_renders,
        }
        for pre in response.pre_turn_resolutions
    ]
    # Per-turn router rationale, agent outputs, and phase latencies
    # used to be folded into `response.debug` here. v11-r7j murdered
    # that field; if you want those signals back, source them from the
    # engine logger (turn_loop.router[route] lines) and the per-turn
    # checkpoint files instead.
    return rec


def _setup_bridge() -> EngineBridge:
    """Build an EngineBridge with the same defaults as play.py."""
    return EngineBridge()


async def _ensure_session(bridge: EngineBridge) -> None:
    """Idempotently create the session, load the story, set ticks off,
    bind the player. Safe to call on a pre-existing session: skips
    re-loading the story and re-binding if state is already there."""

    sessions = bridge.list_session_ids()
    if SESSION not in sessions:
        bridge.create_empty_session(SESSION)
        bridge.load_story_into_session(SESSION, STORY)
        _record({"event": "session_created", "session": SESSION,
                 "story": STORY})
    else:
        # If the session exists but is empty, still need a story load.
        try:
            bridge.load_latest(SESSION)
        except Exception:
            bridge.load_story_into_session(SESSION, STORY)
            _record({"event": "session_existed_loaded_story",
                     "session": SESSION, "story": STORY})

    # Disable ticks. The whole point of this playtest is on-stage only.
    bridge.set_setting(SESSION, "ticks_enabled", "false")
    after = bridge.get_setting(SESSION, "ticks_enabled")
    _record({"event": "settings", "ticks_enabled": after})

    # Bind the player to Jordan Reeves. Idempotent — bind_user raises if
    # already bound to someone else; we treat the same-binding case as ok.
    ckpt = bridge.load_latest(SESSION)
    cur = ckpt.session.character_bindings.get(PLAYER_CHAR)
    if cur != str(PLAYER_USER_ID):
        bridge.takeover(SESSION, PLAYER_CHAR, PLAYER_USER_ID)
        _record({"event": "bound_player", "char": PLAYER_CHAR,
                 "user": PLAYER_USER_ID})


async def main() -> None:
    print(f"Playtest log: {LOG_PATH}")
    print(f"Engine log:   {ENGINE_LOG}")
    print(f"Summary:      {SUMMARY_PATH}")
    _record({"event": "playtest_start", "story": STORY,
             "session": SESSION, "ts": TS})

    bridge = _setup_bridge()
    try:
        await _ensure_session(bridge)

        SAVE_RELOAD_AFTER_TURN = 15  # halfway through the script
        for i, (label, user_input, intent) in enumerate(ACTIONS, start=1):
            print(f"\n--- turn {i} [{label}] ---")
            rec = await _run_one(bridge, label, user_input, intent, i)
            _record(rec)
            if rec.get("error"):
                print(f"  ERROR: {rec['error'].splitlines()[-1][:180]}")
            else:
                pov = (
                    rec.get("per_player_renders", {}).get(PLAYER_CHAR)
                    or rec.get("output_text", "")
                )
                print(f"  end={rec['beat_ended_reason']!r} "
                      f"wall_ms={rec['wall_ms']}")
                print(f"  pov: {pov[:240].replace(chr(10), ' / ')}")

            if i == SAVE_RELOAD_AFTER_TURN:
                print("\n*** save/reload checkpoint — fresh bridge ***\n")
                await bridge.close()
                _record({"event": "save_reload_split"})
                bridge = _setup_bridge()
                # Re-ensure but DO NOT re-load the story or re-bind; both
                # are persisted on disk. _ensure_session is idempotent.
                await _ensure_session(bridge)
                # Verify the rolling history survived.
                ckpt = bridge.load_latest(SESSION)
                # Report the player character's opaque location label
                # after reload.
                player_loc = ""
                pcid = ckpt.session.player_character_id
                if pcid:
                    for c in ckpt.characters:
                        if c.character_id == pcid:
                            player_loc = c.location
                            break
                _record({
                    "event": "post_reload_state",
                    "turn_index": ckpt.session.turn_index,
                    "ticks_enabled": ckpt.session.config.settings.ticks_enabled,
                    "char_conv_lengths": {
                        cid: len(conv)
                        for cid, conv in ckpt.character_conversations.items()
                    },
                    "session_conv_len": len(ckpt.session_conversation),
                    "player_location": player_loc,
                })
    finally:
        try:
            await bridge.close()
        except Exception:
            pass

    _summarize()
    print("\n--- done ---")


def _summarize() -> None:
    """Walk the JSONL and write a human-readable summary."""
    records = []
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    turn_recs = [r for r in records if "turn_no" in r]
    n_turns = len(turn_recs)
    n_errors = sum(1 for r in turn_recs if r.get("error"))
    end_reasons: dict[str, int] = {}
    wall_ms_total = 0.0
    pre_turn_resolutions = 0
    cat_ii_pending = 0
    cascade_exhausted = 0
    silent_sentinel = 0
    paren_misfires = 0
    multi_paragraph_agents = 0
    cache_reads = 0
    cache_creates = 0

    # Tick-fire detection from engine log.
    tick_fires = 0
    if ENGINE_LOG.exists():
        with ENGINE_LOG.open() as ef:
            for line in ef:
                if "tick scheduler:" in line.lower() and "fire" in line.lower() \
                   and "no fire" not in line.lower():
                    tick_fires += 1

    for r in turn_recs:
        wall_ms_total += r.get("wall_ms", 0.0)
        end_reasons[r.get("beat_ended_reason", "")] = (
            end_reasons.get(r.get("beat_ended_reason", ""), 0) + 1
        )
        if r.get("beat_ended_reason") == "cat_ii_pending":
            cat_ii_pending += 1
        if r.get("beat_ended_reason") == "cascade_exhausted":
            cascade_exhausted += 1
        pre_turn_resolutions += len(r.get("pre_turn_resolutions", []))

        # Walk agent outputs to detect things the reviewers asked us to look
        # for.
        ao = (r.get("debug", {}) or {}).get("agent_outputs", []) or []
        for a in ao:
            txt = a.get("public_text", "") or ""
            if txt.strip() == "(remains silent)":
                silent_sentinel += 1
            if txt.count("\n\n") >= 2:
                multi_paragraph_agents += 1
            intent_paren = a.get("intent", "") or ""
            # Heuristic: a single-word or two-word parenthetical that
            # looks like a name (capitalized, no verb) is the misfire
            # the reviewer flagged.
            stripped = intent_paren.strip().strip(".").strip()
            if stripped and len(stripped.split()) <= 2 and stripped[:1].isupper():
                paren_misfires += 1
        for lat in (r.get("debug", {}) or {}).get("latencies", []) or []:
            cache_reads += lat.get("cache_read_input_tokens", 0) or 0
            cache_creates += (
                lat.get("cache_creation_input_tokens", 0) or 0
            )

    lines = []
    lines.append(f"PLAYTEST SUMMARY — {TS}")
    lines.append(f"Story: {STORY}   Session: {SESSION}   Player: {PLAYER_CHAR}")
    lines.append(f"Log:   {LOG_PATH}")
    lines.append(f"Engine log: {ENGINE_LOG}")
    lines.append("")
    lines.append(f"Turns attempted:           {n_turns}")
    lines.append(f"Turns errored:             {n_errors}")
    lines.append(f"Total wall time (s):       {wall_ms_total / 1000:.1f}")
    lines.append(f"Mean wall per turn (ms):   "
                 f"{(wall_ms_total / n_turns) if n_turns else 0:.0f}")
    lines.append("")
    lines.append("Beat-end reasons:")
    for k, v in sorted(end_reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k or '(none)':30}  {v}")
    lines.append("")
    lines.append(f"Cat II pending beats:      {cat_ii_pending}")
    lines.append(f"Cascade exhausted beats:   {cascade_exhausted}")
    lines.append(f"Pre-turn resolutions:      {pre_turn_resolutions}")
    lines.append(f"Silent-beat sentinels:     {silent_sentinel}")
    lines.append(f"Multi-paragraph agent emit:{multi_paragraph_agents}")
    lines.append(f"Suspicious paren misfires: {paren_misfires}")
    lines.append("")
    lines.append("Cache (Anthropic):")
    lines.append(f"  cache reads (input toks):    {cache_reads}")
    lines.append(f"  cache creations (input toks):{cache_creates}")
    lines.append("")
    lines.append(f"TICK FIRES (engine log scan): {tick_fires}")
    lines.append("  Expected with ticks_enabled=False: 0")
    lines.append("")
    lines.append("Save/reload events:")
    for r in records:
        if r.get("event") == "save_reload_split":
            lines.append("  - save_reload_split occurred")
        if r.get("event") == "post_reload_state":
            lines.append(
                f"  - post_reload turn_index={r.get('turn_index')} "
                f"player_location={r.get('player_location')!r} "
                f"session_conv_len={r.get('session_conv_len')}"
            )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
