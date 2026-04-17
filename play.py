#!/usr/bin/env python3
"""Interactive fiction CLI — single command to play.

Usage:
    .venv/bin/python play.py          # resume latest save
    .venv/bin/python play.py --new    # start fresh from beginning

Starts the engine server in the background, auto-detects available
stories, and drops you into a chat-like REPL.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAVES_DIR = Path("app/storage/saves")
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8811
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
HEALTH_URL = f"{BASE_URL}/health"
TURN_URL = f"{BASE_URL}/v1/story/turn"
PERSONALIZE_URL = f"{BASE_URL}/v1/story/personalize"
SERVER_LOG = Path(".server.log")
STARTUP_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

class Spinner:
    """Animated spinner that runs in a background thread."""

    FRAMES = ["   ", ".  ", ".. ", "..."]

    def __init__(self, message: str = "Thinking"):
        self._message = message
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self._message) + 10) + "\r")
        sys.stdout.flush()

    def _spin(self):
        i = 0
        while self._running:
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r  {self._message}{frame}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.4)


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def wrap_print(text: str, indent: int = 0):
    """Word-wrap and print text to terminal width."""
    width = term_width() - indent
    prefix = " " * indent
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            print()
            continue
        for line in textwrap.wrap(paragraph, width=width):
            print(f"{prefix}{line}")


def separator():
    print(f"\n{'~' * min(term_width(), 72)}\n")


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


# ---------------------------------------------------------------------------
# HTTP helpers (using urllib — no extra deps needed)
# ---------------------------------------------------------------------------

import urllib.request
import urllib.error


def http_get(url: str, timeout: float = 5.0) -> dict | None:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def http_post(url: str, body: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

_server_proc: subprocess.Popen | None = None


def start_server() -> subprocess.Popen:
    """Start uvicorn as a background subprocess."""
    global _server_proc

    log_fh = open(SERVER_LOG, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--log-level", "warning",
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    _server_proc = proc

    # Register cleanup
    atexit.register(stop_server)

    return proc


def stop_server():
    """Terminate the server subprocess."""
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        try:
            os.killpg(os.getpgid(_server_proc.pid), signal.SIGTERM)
            _server_proc.wait(timeout=5)
        except Exception:
            try:
                _server_proc.kill()
            except Exception:
                pass
    _server_proc = None


def wait_for_server(proc: subprocess.Popen, timeout: float = STARTUP_TIMEOUT) -> bool:
    """Poll the health endpoint until the server is ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False  # process died
        result = http_get(HEALTH_URL, timeout=1.0)
        if result and result.get("status") == "ok":
            return True
        time.sleep(0.3)
    return False


def server_alive() -> bool:
    return _server_proc is not None and _server_proc.poll() is None


# ---------------------------------------------------------------------------
# Story selection
# ---------------------------------------------------------------------------

def find_stories() -> list[dict]:
    """Scan saves directory for playable stories."""
    stories = []
    if not SAVES_DIR.exists():
        return stories

    for d in sorted(SAVES_DIR.iterdir()):
        if not d.is_dir():
            continue
        checkpoints = sorted(d.glob("ckpt_*.json"))
        if not checkpoints:
            continue

        # Read the latest checkpoint for metadata
        latest = checkpoints[-1]
        try:
            with open(latest) as f:
                data = json.load(f)
            setting = data.get("world_state", {}).get("setting", {})
            session = data.get("session", {})
            stories.append({
                "session_id": d.name,
                "turn_index": session.get("turn_index", 0),
                "genre": setting.get("genre", ""),
                "premise": setting.get("premise", ""),
                "tone": setting.get("tone", ""),
                "characters": len(data.get("characters", [])),
                "checkpoint_path": str(latest),
            })
        except Exception:
            stories.append({
                "session_id": d.name,
                "turn_index": 0,
                "genre": "?",
                "premise": "",
                "tone": "",
                "characters": 0,
                "checkpoint_path": str(latest),
            })

    return stories


def select_story(stories: list[dict]) -> dict:
    """Auto-select or prompt for story choice."""
    if len(stories) == 1:
        s = stories[0]
        print(f"  Loading {bold(s['session_id'])} ({s['genre']}, turn {s['turn_index']})")
        return s

    print("\n  Available stories:\n")
    for i, s in enumerate(stories, 1):
        label = s['session_id'].replace('_', ' ').title()
        print(f"    {i}. {bold(label)}")
        if s['genre']:
            print(f"       {dim(s['genre'])} | Turn {s['turn_index']} | {s['characters']} characters")
    print()

    while True:
        try:
            choice = input("  Choose a story [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not choice:
            return stories[0]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(stories):
                return stories[idx]
        except ValueError:
            pass
        print(f"  Please enter a number 1-{len(stories)}")


def load_story_context(story: dict) -> dict:
    """Load checkpoint data for display."""
    with open(story["checkpoint_path"]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Chat REPL
# ---------------------------------------------------------------------------

def print_story_header(data: dict):
    """Print story context when resuming a saved game."""
    setting = data.get("world_state", {}).get("setting", {})
    locations = data.get("world_state", {}).get("locations", {})
    scene_id = locations.get("current_scene_id", "")
    scene = locations.get("scene_graph", {}).get(scene_id, {})
    transcript = data.get("transcript", [])
    player_name = data.get("session", {}).get("player_name", "")

    print()
    separator()

    if setting.get("premise"):
        premise = setting["premise"]
        if player_name:
            premise = premise.replace("PLAYER_NAME", player_name)
        wrap_print(premise, indent=2)
        print()

    if setting.get("genre") or setting.get("tone"):
        parts = []
        if setting.get("genre"):
            parts.append(setting["genre"])
        if setting.get("tone"):
            parts.append(setting["tone"])
        print(f"  {dim(' | '.join(parts))}")

    if scene.get("name"):
        print(f"\n  {bold('Scene:')} {scene['name']}")
    if scene.get("description"):
        wrap_print(scene["description"], indent=4)

    if transcript:
        last = transcript[-1]
        print(f"\n  {dim('Last turn:')}")
        wrap_print(last.get("assistant", ""), indent=4)

    separator()
    print(f"  {dim('Type your actions. /help for commands.')}\n")


def run_opening_turn(session_id: str) -> str | None:
    """Auto-fire the opening turn through the NP1/NP2 pipeline.

    Returns the opening narrative text, or None on failure.
    """
    spinner = Spinner("Setting the scene")
    spinner.start()

    try:
        body = {
            "session_id": session_id,
            "user_input": "I look around and take in my surroundings.",
            "debug": False,
        }
        response = http_post(TURN_URL, body, timeout=300.0)
        spinner.stop()
        return response.get("output_text", "")
    except Exception as e:
        spinner.stop()
        print(f"\n  {dim(f'Opening scene generation failed: {e}')}")
        return None


def print_help():
    print(f"""
  {bold('Commands:')}
    /help, /h       Show this message
    /quit, /q       Exit the game
    /debug          Cycle debug: off → basic → verbose
    /save           Show current save info
""")


DEBUG_OFF = 0
DEBUG_BASIC = 1
DEBUG_VERBOSE = 2
DEBUG_LABELS = {DEBUG_OFF: "off", DEBUG_BASIC: "basic", DEBUG_VERBOSE: "verbose"}


def print_debug_basic(debug: dict):
    """Print basic debug info: latency + leakage flags."""
    print()
    latencies = debug.get("latencies", [])
    if latencies:
        parts = [f"{lat['phase']}={lat['duration_ms']:.0f}ms" for lat in latencies]
        print(f"  {dim('Latency: ' + ', '.join(parts))}")
    total = debug.get("total_duration_ms", 0)
    if total:
        print(f"  {dim(f'Total: {total:.0f}ms')}")
    validations = debug.get("validations", [])
    flagged = [v for v in validations if not v.get("passed")]
    if flagged:
        print(f"  {dim(f'Leakage flags: {len(flagged)} characters flagged')}")


def print_debug_verbose(debug: dict):
    """Print verbose debug: discriminator graph + agent details."""
    print_debug_basic(debug)

    # NP1 adjudication summary
    event = debug.get("canonical_event", {})
    adj = event.get("world_adjudication", {})
    if adj:
        feasible = adj.get("feasible", "?")
        phase_label = "EventRouter" if "event_router" in debug.get("models_used", {}) else "NP1"
        print(f"\n  {dim(f'{phase_label}: feasible={feasible}')}")
        outcome = adj.get("resolved_outcome", "")
        if outcome:
            print(f"  {dim(f'  outcome: {outcome[:120]}...' if len(outcome) > 120 else f'  outcome: {outcome}')}")

    obs_facts = event.get("observable_facts", [])
    if obs_facts:
        print(f"  {dim(f'  observable_facts ({len(obs_facts)}):')}")
        for fact in obs_facts:
            print(f"    {dim(f'- {fact}')}")

    # Discriminator graph
    disc = debug.get("discriminator", {})
    observers = disc.get("observers", [])
    spawn = disc.get("spawn", [])
    dormant = disc.get("dormant", [])

    if observers:
        print(f"\n  {dim(f'Discriminator: {len(observers)} observers')}")
        for obs in observers:
            cid = obs.get("character_id", "?")
            level = obs.get("observation_level", "?")
            priority = obs.get("response_priority", 0)
            facts = obs.get("facts", [])
            tag = f"P{priority}" + (" RESPOND" if priority >= 3 else "")
            print(f"    {dim(f'{cid} [{level}] [{tag}]')}")
            for fact in facts:
                print(f"      {dim(f'- {fact}')}")

    if spawn:
        print(f"  {dim(f'Spawn: {len(spawn)} new characters')}")
        for s in spawn:
            cid = s.get("character_id", "?")
            seed = s.get("seed", {})
            print(f"    {dim(f'{cid}: {seed}')}")
    if dormant:
        print(f"  {dim(f'Dormant: {dormant}')}")

    # Agent output summaries
    agent_outputs = debug.get("agent_outputs", [])
    if agent_outputs:
        print(f"\n  {dim(f'Agents: {len(agent_outputs)} responded')}")
        for ao in agent_outputs:
            cid = ao.get("character_id", "?")
            pub = ao.get("public_response", {})
            dialogue = pub.get("dialogue", [])
            actions = pub.get("actions", [])
            expr = pub.get("expression", "")
            parts = []
            if dialogue:
                parts.append(f'{len(dialogue)} lines')
            if actions:
                parts.append(f'{len(actions)} actions')
            if expr:
                parts.append(f'expr: {expr[:40]}')
            summary = ", ".join(parts) if parts else "(empty)"
            print(f"    {dim(f'{cid}: {summary}')}")

    # Leakage details
    validations = debug.get("validations", [])
    flagged = [v for v in validations if not v.get("passed")]
    if flagged:
        print(f"\n  {dim(f'Leakage details:')}")
        for v in flagged:
            cid = v.get("character_id", "?")
            for flag in v.get("flags", []):
                text = flag.get("text", "?")
                reason = flag.get("reason", "?")
                snippet = text[:60]
                print(f"    {dim(f'{cid}: [{snippet}] — {reason}')}")


def run_repl(session_id: str, data: dict):
    """Main chat loop."""
    debug_level = DEBUG_OFF
    turn_index = data.get("session", {}).get("turn_index", 0)

    while True:
        # Check server is alive
        if not server_alive():
            print(f"\n  {bold('Server process died.')} Check {SERVER_LOG} for details.")
            break

        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {dim('Farewell.')}\n")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd in ("/quit", "/q", "/exit"):
                print(f"\n  {dim('Farewell.')}\n")
                break
            elif cmd in ("/help", "/h"):
                print_help()
                continue
            elif cmd == "/debug":
                debug_level = (debug_level + 1) % 3
                label = DEBUG_LABELS[debug_level]
                print(f"  {dim(f'Debug mode: {label}')}\n")
                continue
            elif cmd == "/save":
                print(f"  {dim(f'Session: {session_id} | Turn: {turn_index}')}\n")
                continue
            else:
                print(f"  {dim('Unknown command. /help for options.')}\n")
                continue

        # Send turn request
        spinner = Spinner()
        spinner.start()

        try:
            body = {
                "session_id": session_id,
                "user_input": user_input,
                "debug": debug_level > DEBUG_OFF,
            }
            response = http_post(TURN_URL, body, timeout=180.0)
            spinner.stop()

            turn_index = response.get("turn_index", turn_index + 1)
            output = response.get("output_text", "")

            print()
            wrap_print(output, indent=2)

            # Show debug info if enabled
            debug = response.get("debug")
            if debug and debug_level == DEBUG_BASIC:
                print_debug_basic(debug)
            elif debug and debug_level == DEBUG_VERBOSE:
                print_debug_verbose(debug)

            print()

        except urllib.error.HTTPError as e:
            spinner.stop()
            try:
                err = json.loads(e.read())
                detail = err.get("detail", str(e))
            except Exception:
                detail = str(e)
            print(f"\n  Error: {detail}\n")

        except urllib.error.URLError as e:
            spinner.stop()
            print(f"\n  Connection error: {e.reason}")
            print(f"  Server may have crashed. Check {SERVER_LOG}\n")

        except Exception as e:
            spinner.stop()
            print(f"\n  Error: {e}\n")


# ---------------------------------------------------------------------------
# Character creation
# ---------------------------------------------------------------------------

def character_creation(session_id: str, data: dict) -> dict:
    """Prompt for player name and personalize the story. Returns updated data."""
    # Check if already personalized
    player_name = data.get("session", {}).get("player_name", "")
    if player_name:
        return data  # Already personalized

    # Find the player character to show context
    characters = data.get("characters", [])
    player_char = None
    for c in characters:
        if "player_name" in c.get("character_id", "").lower():
            player_char = c
            break

    separator()
    print(f"  {bold('Character Creation')}\n")

    while True:
        try:
            name = input(f"  What is your character's first name? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if name and len(name) <= 30 and name.replace(" ", "").isalpha():
            break
        if not name:
            print(f"  {dim('Please enter a name.')}")
        else:
            print(f"  {dim('Letters only, 30 characters max.')}")

    # Show backstory with the chosen name substituted in
    if player_char:
        role = player_char.get("public_sheet", {}).get("role", "")
        if role:
            role = role.replace("PLAYER_NAME", name)
            print(f"\n  {dim(f'Role: {role}')}")
        backstory = player_char.get("backstory", "")
        if backstory:
            wrap_print(backstory.replace("PLAYER_NAME", name), indent=2)
            print()

    print(f"\n  {dim(f'Personalizing story for {name}...')}")

    try:
        result = http_post(PERSONALIZE_URL, {
            "session_id": session_id,
            "player_name": name,
        }, timeout=10.0)
        # Reload the checkpoint data
        return load_story_context({"checkpoint_path": _find_latest_checkpoint(session_id)})
    except Exception as e:
        print(f"\n  Warning: personalization failed ({e}), continuing with placeholder name.\n")
        return data


def _find_latest_checkpoint(session_id: str) -> str:
    """Find the latest checkpoint file for a session."""
    session_dir = SAVES_DIR / session_id
    checkpoints = sorted(session_dir.glob("ckpt_*.json"))
    return str(checkpoints[-1]) if checkpoints else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reset_story(session_id: str):
    """Remove all checkpoints except ckpt_0000, resetting the story."""
    session_dir = SAVES_DIR / session_id
    for ckpt in sorted(session_dir.glob("ckpt_*.json")):
        if ckpt.name != "ckpt_0000.json":
            ckpt.unlink()
            print(f"  {dim(f'Removed {ckpt.name}')}")


def main():
    parser = argparse.ArgumentParser(description="Interactive Fiction Engine")
    parser.add_argument("--new", action="store_true", help="Start fresh from the beginning")
    args = parser.parse_args()

    print(f"\n  {bold('Interactive Fiction Engine')}")
    print(f"  {dim('Multi-agent narrative engine')}\n")

    # Find stories
    stories = find_stories()
    if not stories:
        print("  No stories found.\n")
        print("  Import one first:")
        print(f"    {dim('.venv/bin/python scripts/import_story.py <story_file.txt>')}\n")
        sys.exit(1)

    # Select story
    story = select_story(stories)

    # Reset if --new
    if args.new:
        print(f"  {dim('Resetting to beginning...')}")
        reset_story(story["session_id"])
        # Reload story metadata from ckpt_0000
        story["turn_index"] = 0
        story["checkpoint_path"] = str(SAVES_DIR / story["session_id"] / "ckpt_0000.json")

    # Start server
    print(f"\n  Starting engine", end="", flush=True)
    proc = start_server()

    if not wait_for_server(proc):
        print(" FAILED\n")
        if proc.poll() is not None:
            print(f"  Server exited with code {proc.returncode}.")
        else:
            print("  Server did not respond in time.")
        print(f"  Check {SERVER_LOG} for details.\n")
        stop_server()
        sys.exit(1)

    print(f" {dim('ready')}")

    # Load story context
    data = load_story_context(story)

    # Character creation on first play
    is_new_game = data.get("session", {}).get("turn_index", 0) == 0
    if is_new_game:
        data = character_creation(story["session_id"], data)

    # Fresh game: auto-fire the opening turn through NP1/NP2
    if is_new_game:
        opening_text = run_opening_turn(story["session_id"])
        if opening_text:
            print()
            separator()
            wrap_print(opening_text, indent=2)
            separator()
            print(f"  {dim('Type your actions. /help for commands.')}\n")
            # Reload data with the new transcript
            data = load_story_context({
                "checkpoint_path": _find_latest_checkpoint(story["session_id"])
            })
        else:
            # Fallback to static header if opening turn failed
            print_story_header(data)
    else:
        print_story_header(data)

    # Run the REPL
    try:
        run_repl(story["session_id"], data)
    finally:
        stop_server()


if __name__ == "__main__":
    main()
