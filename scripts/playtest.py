#!/usr/bin/env python3
"""Automated playtesting — runs scripted or agent-driven sessions through the engine.

Usage:
    .venv/bin/python scripts/playtest.py --story covenant_of_thrones --strategy scripted
    .venv/bin/python scripts/playtest.py --story covenant_of_thrones --strategy adversarial
    .venv/bin/python scripts/playtest.py --story covenant_of_thrones --strategy agent --turns 10
    .venv/bin/python scripts/playtest.py --story covenant_of_thrones --strategy all --parallel 3

Requires the engine server to be running (or use --start-server to auto-start).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env for LLM gateway key
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

from scripts.playtest_analyzers import Issue, analyze_turn, summarize_issues
from scripts.playtest_scenarios import (
    SCENARIOS,
    SCRIPTED_SCENARIOS,
    ADVERSARIAL_SCENARIOS,
    ALL_SCENARIOS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAVES_DIR = Path("app/storage/saves")
REPORTS_DIR = Path("app/storage/playtest_reports")
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8811
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
HEALTH_URL = f"{BASE_URL}/health"
TURN_URL = f"{BASE_URL}/v1/story/turn"
PERSONALIZE_URL = f"{BASE_URL}/v1/story/personalize"

PLAYER_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Echo"]

LLM_GATEWAY_URL = "https://llm-api.amd.com/OnPrem/chat/completions"
LLM_MODEL = "GPT-oss-120B"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_post(url: str, body: dict, timeout: float = 300.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def server_is_running() -> bool:
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

_server_proc: subprocess.Popen | None = None


def start_server() -> subprocess.Popen:
    global _server_proc
    print("  Starting engine server...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--log-level", "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    _server_proc = proc

    # Wait for server
    for _ in range(30):
        if server_is_running():
            print("  Server ready.")
            return proc
        time.sleep(1)

    proc.kill()
    raise RuntimeError("Server failed to start within 30s")


def stop_server():
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


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn_index: int
    user_input: str
    output_text: str
    debug: dict | None = None
    latency_ms: float = 0.0
    short_circuited: bool = False
    issues: list[dict] = field(default_factory=list)


@dataclass
class SessionResult:
    session_id: str
    story_id: str
    strategy: str
    scenario: str
    player_name: str
    turns: list[TurnRecord] = field(default_factory=list)
    all_issues: list[dict] = field(default_factory=list)
    total_duration_ms: float = 0.0


class PlaytestSession:
    """Manages a single playtest session."""

    def __init__(
        self,
        story_id: str,
        strategy: str,
        scenario: str = "",
        player_name: str = "TestPlayer",
    ):
        self.story_id = story_id
        self.strategy = strategy
        self.scenario = scenario
        self.player_name = player_name
        self.session_id = f"playtest_{uuid4().hex[:12]}"
        self.session_dir = SAVES_DIR / self.session_id
        self.turns: list[TurnRecord] = []
        self.checkpoint: dict | None = None
        self.player_character_id = ""

    def setup(self):
        """Create isolated session by copying story checkpoint."""
        source_dir = SAVES_DIR / self.story_id
        if not source_dir.exists():
            raise FileNotFoundError(f"Story '{self.story_id}' not found in {SAVES_DIR}")

        # Copy ckpt_0000 only
        self.session_dir.mkdir(parents=True, exist_ok=True)
        src_ckpt = source_dir / "ckpt_0000.json"
        dst_ckpt = self.session_dir / "ckpt_0000.json"
        shutil.copy2(src_ckpt, dst_ckpt)

        # Update session_id in checkpoint
        with open(dst_ckpt) as f:
            data = json.load(f)
        data["session"]["session_id"] = self.session_id
        with open(dst_ckpt, "w") as f:
            json.dump(data, f)

        # Personalize
        http_post(PERSONALIZE_URL, {
            "session_id": self.session_id,
            "player_name": self.player_name,
        }, timeout=10.0)

        # Load checkpoint for analysis
        self._reload_checkpoint()

    def _reload_checkpoint(self):
        """Load the latest checkpoint for analysis."""
        ckpts = sorted(self.session_dir.glob("ckpt_*.json"))
        if ckpts:
            with open(ckpts[-1]) as f:
                self.checkpoint = json.load(f)
            self.player_character_id = (
                self.checkpoint.get("session", {}).get("player_character_id", "")
            )

    def run_turn(self, action: str) -> TurnRecord:
        """Submit a turn and collect results."""
        t0 = time.monotonic()

        response = http_post(TURN_URL, {
            "session_id": self.session_id,
            "user_input": action,
            "debug": True,
        }, timeout=300.0)

        elapsed = (time.monotonic() - t0) * 1000

        # Reload checkpoint for latest state
        self._reload_checkpoint()

        # Build turn record
        debug = response.get("debug")
        short_circuited = False
        if debug:
            agent_outputs = debug.get("agent_outputs", [])
            short_circuited = len(agent_outputs) == 0 and debug.get("canonical_event", {}).get(
                "world_adjudication", {}
            ).get("feasible") is False

        record = TurnRecord(
            turn_index=response.get("turn_index", 0),
            user_input=action,
            output_text=response.get("output_text", ""),
            debug=debug,
            latency_ms=elapsed,
            short_circuited=short_circuited,
        )

        # Analyze
        turn_data = {
            "turn_index": record.turn_index,
            "user_input": action,
            "output_text": record.output_text,
            "debug": debug,
        }
        issues = analyze_turn(turn_data, self.checkpoint, self.player_character_id)
        record.issues = [asdict(i) for i in issues]

        self.turns.append(record)
        return record

    def teardown(self):
        """Remove session directory."""
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)

    def result(self) -> SessionResult:
        all_issues = []
        for t in self.turns:
            all_issues.extend(t.issues)

        total = sum(t.latency_ms for t in self.turns)

        return SessionResult(
            session_id=self.session_id,
            story_id=self.story_id,
            strategy=self.strategy,
            scenario=self.scenario,
            player_name=self.player_name,
            turns=self.turns,
            all_issues=all_issues,
            total_duration_ms=total,
        )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

async def run_scripted(session: PlaytestSession, scenario_name: str):
    """Run a predefined action sequence."""
    actions = SCENARIOS.get(scenario_name, [])
    if not actions:
        print(f"  Unknown scenario: {scenario_name}")
        return

    for i, action in enumerate(actions):
        record = session.run_turn(action)
        status = "OK" if not record.issues else f"{len(record.issues)} issues"
        sc = " [short-circuit]" if record.short_circuited else ""
        print(f"    Turn {i+1}/{len(actions)}: {status} ({record.latency_ms:.0f}ms){sc}")


async def run_adversarial(session: PlaytestSession, scenario_name: str):
    """Run adversarial probing scenarios."""
    # Same as scripted but with adversarial scenarios
    await run_scripted(session, scenario_name)


# ---------------------------------------------------------------------------
# LLM-driven agent player
# ---------------------------------------------------------------------------

AGENT_PERSONAS = {
    "explorer": {
        "persona": "You are a curious, methodical player who wants to explore the world. You move between locations, examine your surroundings, and talk to people you meet.",
        "goals": [
            "Visit every accessible location",
            "Talk to at least one person in each location",
            "Learn about the world and its history",
        ],
    },
    "social": {
        "persona": "You are a charismatic, outgoing player who loves meeting people. You introduce yourself, ask questions, and try to build relationships.",
        "goals": [
            "Introduce yourself to as many characters as possible",
            "Learn everyone's name and role",
            "Find out what people think about the current situation",
        ],
    },
    "adversarial": {
        "persona": "You are a suspicious player who thinks something is wrong. You probe for secrets, test boundaries, and try to uncover hidden information.",
        "goals": [
            "Ask probing questions about conspiracies and hidden agendas",
            "Try to get characters to reveal secrets",
            "Test what you can and cannot do in this world",
            "Reference rumors about engineered plagues, secret factions, and hidden plots",
        ],
    },
    "passive": {
        "persona": "You are a quiet, observant player. You watch more than you speak, take note of body language, and only act when something catches your attention.",
        "goals": [
            "Observe your surroundings carefully",
            "Watch other characters' behavior without intervening",
            "Only speak when spoken to or when something seems off",
        ],
    },
}


def _llm_call(system_prompt: str, user_prompt: str) -> str:
    """Make a direct call to the LLM gateway."""
    api_key = os.environ.get("LLM_GATEWAY_KEY", "")
    if not api_key:
        raise RuntimeError("LLM_GATEWAY_KEY not set in environment")

    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 200,
    }

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        LLM_GATEWAY_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": api_key,
        },
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())

    choices = result.get("choices") or []
    if not choices:
        return "I look around."
    content = choices[0].get("message", {}).get("content")
    if not content:
        return "I look around."
    return content.strip()


class AgentPlayer:
    """LLM-driven player that decides actions based on persona and story context."""

    def __init__(self, persona_name: str = "explorer"):
        config = AGENT_PERSONAS.get(persona_name, AGENT_PERSONAS["explorer"])
        self.persona = config["persona"]
        self.goals = config["goals"]
        self.persona_name = persona_name
        self.transcript: list[dict] = []

    def decide_action(self, last_output: str, player_name: str, turn_num: int) -> str:
        """Ask the LLM what action to take next."""
        self.transcript.append({"role": "narrator", "text": last_output})

        goals_text = "\n".join(f"- {g}" for g in self.goals)

        # Build transcript context (last 3 exchanges max)
        transcript_lines = []
        recent = self.transcript[-6:]  # last 3 pairs
        for entry in recent:
            if entry["role"] == "player":
                transcript_lines.append(f"You said: {entry['text']}")
            else:
                transcript_lines.append(f"Narrator: {entry['text'][:300]}")
        transcript_text = "\n".join(transcript_lines) if transcript_lines else "(beginning of story)"

        system = (
            f"{self.persona}\n\n"
            f"You are playing as {player_name} in an interactive fiction game. "
            f"Respond with ONLY the action you want to take, written in first person. "
            f"Example: 'I walk to the dining hall and sit down.' "
            f"Do NOT include any commentary, explanation, or out-of-character text. "
            f"Just the action, 1-2 sentences max."
        )

        user = (
            f"## Your Goals\n{goals_text}\n\n"
            f"## Recent Story (turn {turn_num})\n{transcript_text}\n\n"
            f"What do you do next?"
        )

        action = _llm_call(system, user)

        # Clean up — remove quotes, "I would", meta-commentary
        action = action.strip().strip('"').strip("'")
        if action.lower().startswith("action:"):
            action = action[7:].strip()

        self.transcript.append({"role": "player", "text": action})
        return action


async def run_agent(session: PlaytestSession, persona: str, max_turns: int):
    """Run an LLM-driven agent player session."""
    agent = AgentPlayer(persona)

    # Get the opening narrative as first context
    opening = ""
    if session.checkpoint:
        opening = session.checkpoint.get("opening_narrative", "")
        if opening:
            opening = opening.replace("PLAYER_NAME", session.player_name)

    # Fire opening turn to get initial scene
    if opening:
        # Submit the opening turn
        record = session.run_turn("I look around and take in my surroundings.")
        status = "OK" if not record.issues else f"{len(record.issues)} issues"
        sc = " [short-circuit]" if record.short_circuited else ""
        print(f"    Turn 0 (opening): {status} ({record.latency_ms:.0f}ms){sc}")
        last_output = record.output_text
    else:
        last_output = "You find yourself in an unfamiliar place. What do you do?"

    for i in range(max_turns):
        action = agent.decide_action(last_output, session.player_name, i + 1)
        print(f"    Agent [{persona}]: {action[:80]}")

        record = session.run_turn(action)
        last_output = record.output_text

        status = "OK" if not record.issues else f"{len(record.issues)} issues"
        sc = " [short-circuit]" if record.short_circuited else ""
        print(f"    Turn {i+1}/{max_turns}: {status} ({record.latency_ms:.0f}ms){sc}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_session(
    story_id: str,
    strategy: str,
    scenario: str,
    player_name: str,
    keep_sessions: bool = False,
    max_turns: int = 5,
) -> SessionResult:
    """Run a single playtest session."""
    session = PlaytestSession(story_id, strategy, scenario, player_name)

    try:
        session.setup()
        print(f"  Session {session.session_id}: {strategy}/{scenario} as {player_name}")

        if strategy in ("scripted", "adversarial"):
            await run_scripted(session, scenario)
        elif strategy == "agent":
            await run_agent(session, scenario, max_turns)
        else:
            print(f"  Unknown strategy: {strategy}")

        return session.result()
    finally:
        if not keep_sessions:
            session.teardown()


async def run_parallel(
    story_id: str,
    sessions_config: list[dict],
    max_parallel: int = 3,
    keep_sessions: bool = False,
    max_turns: int = 5,
) -> list[SessionResult]:
    """Run multiple sessions, respecting parallelism limit."""
    semaphore = asyncio.Semaphore(max_parallel)
    results: list[SessionResult] = []

    async def run_with_limit(config: dict) -> SessionResult:
        async with semaphore:
            return await run_session(
                story_id=story_id,
                keep_sessions=keep_sessions,
                max_turns=max_turns,
                **config,
            )

    tasks = [run_with_limit(cfg) for cfg in sessions_config]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  Session failed: {r}")
        else:
            final.append(r)

    return final


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(results: list[SessionResult], story_id: str):
    """Write playtest report to disk."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    report_dir = REPORTS_DIR / f"{timestamp}_{story_id}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Collect all issues
    all_issues = []
    for r in results:
        all_issues.extend(r.all_issues)

    summary = summarize_issues([Issue(**i) for i in all_issues])

    # Summary JSON
    summary_data = {
        "timestamp": timestamp,
        "story_id": story_id,
        "sessions": len(results),
        "total_turns": sum(len(r.turns) for r in results),
        "issue_summary": summary,
    }
    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Issues JSON
    with open(report_dir / "issues.json", "w") as f:
        json.dump(all_issues, f, indent=2)

    # Session details
    sessions_dir = report_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    for r in results:
        session_data = {
            "session_id": r.session_id,
            "strategy": r.strategy,
            "scenario": r.scenario,
            "player_name": r.player_name,
            "total_duration_ms": r.total_duration_ms,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "user_input": t.user_input,
                    "output_text": t.output_text[:500],
                    "latency_ms": t.latency_ms,
                    "short_circuited": t.short_circuited,
                    "issue_count": len(t.issues),
                    "issues": t.issues,
                }
                for t in r.turns
            ],
        }
        fname = f"{r.strategy}_{r.scenario}.json"
        with open(sessions_dir / fname, "w") as f:
            json.dump(session_data, f, indent=2)

    # Human-readable report
    lines = [
        f"# Playtest Report: {story_id}",
        f"Date: {timestamp}",
        f"Sessions: {len(results)}",
        f"Total turns: {sum(len(r.turns) for r in results)}",
        "",
        "## Issue Summary",
        f"Total issues: {summary['total']}",
        f"Errors: {summary['errors']}",
        f"Warnings: {summary['warnings']}",
        "",
    ]
    if summary.get("by_category"):
        lines.append("### By Category")
        for cat, count in sorted(summary["by_category"].items()):
            lines.append(f"- {cat}: {count}")
        lines.append("")

    lines.append("## Sessions")
    for r in results:
        issue_count = len(r.all_issues)
        lines.append(
            f"- **{r.strategy}/{r.scenario}** ({r.player_name}): "
            f"{len(r.turns)} turns, {issue_count} issues, "
            f"{r.total_duration_ms:.0f}ms total"
        )

    lines.append("")
    lines.append("## All Issues")
    for issue in all_issues:
        lines.append(
            f"- [{issue['severity'].upper()}] Turn {issue['turn_index']}: "
            f"[{issue['category']}] {issue['description']}"
        )

    with open(report_dir / "report.md", "w") as f:
        f.write("\n".join(lines))

    return report_dir


def print_summary(results: list[SessionResult]):
    """Print a concise summary to stdout."""
    all_issues = []
    for r in results:
        all_issues.extend(r.all_issues)

    summary = summarize_issues([Issue(**i) for i in all_issues])

    total_turns = sum(len(r.turns) for r in results)
    total_time = sum(r.total_duration_ms for r in results)

    print()
    print("=" * 60)
    print(f"  Playtest Complete: {len(results)} sessions, {total_turns} turns")
    print(f"  Total time: {total_time/1000:.1f}s")
    print(f"  Issues: {summary['errors']} errors, {summary['warnings']} warnings")

    if summary.get("by_category"):
        print()
        for cat, count in sorted(summary["by_category"].items()):
            print(f"    {cat}: {count}")

    # Show first few errors
    errors = [i for i in all_issues if i["severity"] == "error"]
    if errors:
        print()
        print("  First errors:")
        for err in errors[:5]:
            desc = err["description"][:80]
            print(f"    Turn {err['turn_index']}: [{err['category']}] {desc}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_session_configs(strategy: str, parallel: int) -> list[dict]:
    """Build session configs based on strategy."""
    configs = []

    if strategy == "scripted":
        scenarios = SCRIPTED_SCENARIOS
        for i, scenario in enumerate(scenarios):
            configs.append({
                "strategy": "scripted",
                "scenario": scenario,
                "player_name": PLAYER_NAMES[i % len(PLAYER_NAMES)],
            })

    elif strategy == "adversarial":
        scenarios = ADVERSARIAL_SCENARIOS
        for i, scenario in enumerate(scenarios):
            configs.append({
                "strategy": "adversarial",
                "scenario": scenario,
                "player_name": PLAYER_NAMES[i % len(PLAYER_NAMES)],
            })

    elif strategy == "agent":
        # One session per persona
        personas = list(AGENT_PERSONAS.keys())
        for i, persona in enumerate(personas):
            configs.append({
                "strategy": "agent",
                "scenario": persona,
                "player_name": PLAYER_NAMES[i % len(PLAYER_NAMES)],
            })

    elif strategy == "all":
        # Scripted + adversarial scenarios
        for i, scenario in enumerate(ALL_SCENARIOS):
            strat = "adversarial" if scenario in ADVERSARIAL_SCENARIOS else "scripted"
            configs.append({
                "strategy": strat,
                "scenario": scenario,
                "player_name": PLAYER_NAMES[i % len(PLAYER_NAMES)],
            })
        # Plus all agent personas
        offset = len(ALL_SCENARIOS)
        for i, persona in enumerate(AGENT_PERSONAS.keys()):
            configs.append({
                "strategy": "agent",
                "scenario": persona,
                "player_name": PLAYER_NAMES[(offset + i) % len(PLAYER_NAMES)],
            })

    else:
        print(f"Unknown strategy: {strategy}")
        sys.exit(1)

    return configs


def main():
    parser = argparse.ArgumentParser(description="Automated playtesting")
    parser.add_argument("--story", required=True, help="Story ID (directory name in saves)")
    parser.add_argument(
        "--strategy", required=True,
        choices=["scripted", "adversarial", "agent", "all"],
        help="Play strategy",
    )
    parser.add_argument("--parallel", type=int, default=1, help="Max parallel sessions")
    parser.add_argument("--turns", type=int, default=5, help="Max turns per agent session")
    parser.add_argument("--scenario", help="Run a specific scenario or agent persona by name")
    parser.add_argument("--keep-sessions", action="store_true", help="Don't clean up session dirs")
    parser.add_argument("--start-server", action="store_true", help="Auto-start engine server")
    args = parser.parse_args()

    # Ensure server is running
    if not server_is_running():
        if args.start_server:
            start_server()
        else:
            print("Engine server is not running. Use --start-server or start it manually.")
            sys.exit(1)

    # Build session configs
    if args.scenario:
        valid_scenarios = set(SCENARIOS.keys()) | set(AGENT_PERSONAS.keys())
        if args.scenario not in valid_scenarios:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Scripted scenarios: {', '.join(SCENARIOS.keys())}")
            print(f"Agent personas: {', '.join(AGENT_PERSONAS.keys())}")
            sys.exit(1)
        strat = args.strategy
        if strat == "all":
            strat = "agent" if args.scenario in AGENT_PERSONAS else "scripted"
        configs = [{
            "strategy": strat,
            "scenario": args.scenario,
            "player_name": PLAYER_NAMES[0],
        }]
    else:
        configs = build_session_configs(args.strategy, args.parallel)

    print(f"\n  Playtesting '{args.story}': {len(configs)} sessions, max {args.parallel} parallel\n")

    # Run
    try:
        results = asyncio.run(run_parallel(
            story_id=args.story,
            sessions_config=configs,
            max_parallel=args.parallel,
            keep_sessions=args.keep_sessions,
            max_turns=args.turns,
        ))

        # Report
        if results:
            report_dir = write_report(results, args.story)
            print_summary(results)
            print(f"\n  Report: {report_dir}\n")
        else:
            print("\n  No results collected.\n")

    finally:
        if args.start_server:
            stop_server()


if __name__ == "__main__":
    main()
