"""Automated output analyzers for playtest turns.

Each checker examines a turn's debug payload and output text,
returning a list of detected issues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Issue:
    """A detected problem in a playtest turn."""
    severity: str  # "error" | "warning" | "info"
    category: str
    description: str
    turn_index: int = 0


# Common action verbs that indicate the narrator made choices for the player
AGENCY_VERBS = [
    r"\byou accept\b",
    r"\byou take\b",
    r"\byou sit\b",
    r"\byou eat\b",
    r"\byou drink\b",
    r"\byou pick up\b",
    r"\byou decide\b",
    r"\byou choose\b",
    r"\byou agree\b",
    r"\byou refuse\b",
    r"\byou nod\b",
    r"\byou shake your head\b",
    r"\byou smile\b",
    r"\byou laugh\b",
]

# Spoiler terms that should never appear in player-facing output
# These are loaded from the checkpoint's hidden_lore/hidden_facts at runtime
DEFAULT_SPOILER_TERMS = [
    "engineered plague",
    "engineered human plague",
    "conspiracy",
    "conspirator",
    "puppet master",
    "true purpose",
    "secret agenda",
    "secretly working",
    "true allegiance",
]


def analyze_turn(
    turn: dict,
    checkpoint: dict | None = None,
    player_character_id: str = "",
) -> list[Issue]:
    """Run all analyzers on a single turn. Returns list of issues."""
    issues: list[Issue] = []
    turn_idx = turn.get("turn_index", 0)

    issues += check_spoiler_leaks(turn, checkpoint, turn_idx)
    issues += check_agency_violations(turn, turn_idx)
    issues += check_player_as_agent(turn, player_character_id, turn_idx)
    issues += check_leakage_flags(turn, turn_idx)
    issues += check_invented_obstacles(turn, turn_idx)
    issues += check_name_leaks(turn, checkpoint, turn_idx)

    return issues


def check_spoiler_leaks(
    turn: dict,
    checkpoint: dict | None,
    turn_index: int,
) -> list[Issue]:
    """Check if hidden lore terms appear in player-facing output."""
    issues = []
    output = turn.get("output_text", "").lower()
    if not output:
        return issues

    # Collect spoiler terms
    terms = list(DEFAULT_SPOILER_TERMS)
    if checkpoint:
        ws = checkpoint.get("world_state", {})
        hidden_lore = ws.get("hidden_lore", "")
        hidden_facts = ws.get("hidden_facts", [])
        # Extract distinctive phrases from hidden lore
        if hidden_lore:
            for line in hidden_lore.split("\n"):
                line = line.strip().strip("-").strip()
                if len(line) > 10:
                    terms.append(line.lower())
        for fact in hidden_facts:
            terms.append(fact.lower())

    for term in terms:
        if term in output:
            issues.append(Issue(
                severity="error",
                category="spoiler_leak",
                description=f"Hidden lore term '{term}' appears in output",
                turn_index=turn_index,
            ))

    return issues


def check_agency_violations(turn: dict, turn_index: int) -> list[Issue]:
    """Check if the narrator made choices for the player."""
    issues = []
    output = turn.get("output_text", "")
    if not output:
        return issues

    user_input = turn.get("user_input", "").lower()

    for pattern in AGENCY_VERBS:
        matches = re.findall(pattern, output, re.IGNORECASE)
        for match in matches:
            # Skip if the player explicitly stated this action
            verb_core = match.lower().replace("you ", "").strip()
            if verb_core in user_input:
                continue
            issues.append(Issue(
                severity="warning",
                category="agency_violation",
                description=f"Narrator may have made a choice for the player: '{match}'",
                turn_index=turn_index,
            ))

    return issues


def check_player_as_agent(
    turn: dict,
    player_character_id: str,
    turn_index: int,
) -> list[Issue]:
    """Check if the player character was included as an agent."""
    issues = []
    debug = turn.get("debug")
    if not debug:
        return issues

    # Check discriminator observers
    disc = debug.get("discriminator", {})
    for obs in disc.get("observers", []):
        if obs.get("character_id") == player_character_id:
            respond = obs.get("should_respond", False)
            severity = "error" if respond else "warning"
            issues.append(Issue(
                severity=severity,
                category="player_as_agent",
                description=(
                    f"Player character '{player_character_id}' in discriminator observers "
                    f"(should_respond={respond})"
                ),
                turn_index=turn_index,
            ))

    # Check agent outputs
    for agent_out in debug.get("agent_outputs", []):
        if agent_out.get("character_id") == player_character_id:
            issues.append(Issue(
                severity="error",
                category="player_as_agent",
                description=f"Player character '{player_character_id}' generated agent output",
                turn_index=turn_index,
            ))

    return issues


def check_leakage_flags(turn: dict, turn_index: int) -> list[Issue]:
    """Report leakage flags from the validator debug data."""
    issues = []
    debug = turn.get("debug")
    if not debug:
        return issues

    for val in debug.get("validations", []):
        if not val.get("passed", True):
            for flag in val.get("flags", []):
                issues.append(Issue(
                    severity="warning",
                    category="knowledge_leak",
                    description=(
                        f"Character {val.get('character_id', '?')}: "
                        f"{flag.get('reason', 'unknown')}"
                    ),
                    turn_index=turn_index,
                ))

    return issues


def check_invented_obstacles(turn: dict, turn_index: int) -> list[Issue]:
    """Flag turns where NP1 declared infeasible for mundane actions."""
    issues = []
    debug = turn.get("debug")
    if not debug:
        return issues

    event = debug.get("canonical_event", {})
    adj = event.get("world_adjudication", {})

    if not adj.get("feasible", True):
        user_input = turn.get("user_input", "").lower()
        # Check if the action was mundane movement
        movement_keywords = [
            "head to", "go to", "walk to", "leave", "enter",
            "move to", "return to", "head back",
        ]
        is_movement = any(kw in user_input for kw in movement_keywords)
        if is_movement:
            issues.append(Issue(
                severity="error",
                category="invented_obstacle",
                description=(
                    f"NP1 declared movement infeasible: "
                    f"'{adj.get('resolved_outcome', '')[:100]}'"
                ),
                turn_index=turn_index,
            ))

    return issues


def check_name_leaks(
    turn: dict,
    checkpoint: dict | None,
    turn_index: int,
) -> list[Issue]:
    """Check if character names are used before formal introduction."""
    issues = []
    if not checkpoint:
        return issues

    output = turn.get("output_text", "")
    if not output:
        return issues

    known_ids = set(checkpoint.get("world_state", {}).get("known_characters", []))
    characters = checkpoint.get("characters", [])
    player_id = checkpoint.get("session", {}).get("player_character_id", "")

    for char in characters:
        char_id = char.get("character_id", "")
        if char_id == player_id:
            continue
        if char_id in known_ids:
            continue

        name = char.get("name", "")
        if not name:
            continue

        # Check if the character's first name appears in output
        first_name = name.split()[0]
        if len(first_name) < 3:
            continue

        if re.search(r'\b' + re.escape(first_name) + r'\b', output):
            issues.append(Issue(
                severity="warning",
                category="name_leak",
                description=(
                    f"Character name '{first_name}' used in output but "
                    f"'{char_id}' not in known_characters"
                ),
                turn_index=turn_index,
            ))

    return issues


def summarize_issues(all_issues: list[Issue]) -> dict:
    """Produce summary statistics from a list of issues."""
    by_severity: dict[str, int] = {}
    by_category: dict[str, int] = {}

    for issue in all_issues:
        by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1
        by_category[issue.category] = by_category.get(issue.category, 0) + 1

    return {
        "total": len(all_issues),
        "by_severity": by_severity,
        "by_category": by_category,
        "errors": by_severity.get("error", 0),
        "warnings": by_severity.get("warning", 0),
    }
