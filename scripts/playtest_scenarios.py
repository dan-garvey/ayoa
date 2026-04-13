"""Predefined action sequences for scripted playtesting.

Each scenario is a list of player actions to submit sequentially.
Scenarios are grouped by testing purpose.
"""

from __future__ import annotations

# --- Movement & Scene Transitions ---

BASIC_MOVEMENT = [
    "I head to the dining hall.",
    "I look around the dining hall.",
    "I head back to my room.",
]

EXPLORE_CONNECTED = [
    "I leave my room and head into the hallway.",
    "I look around. What doors or passages do I see?",
    "I head to the dining hall.",
    "I look around the dining hall.",
    "I head back to the hallway.",
]

# --- Social Interaction ---

SELF_INTRODUCTION = [
    "I head to the dining hall.",
    'I loudly introduce myself. "Hello everyone, I\'m looking forward to studying with you all!"',
    "I ask the nearest person their name.",
    "I sit down and wait to see what happens.",
]

QUIET_OBSERVATION = [
    "I head to the dining hall.",
    "I find a seat in the corner and quietly observe the room.",
    "I watch the other students. Who stands out?",
    "I continue eating in silence.",
]

DIRECT_CONVERSATION = [
    "I head to the dining hall.",
    "I approach the most interesting-looking person and introduce myself.",
    "I ask them what they think of the Academy so far.",
    "I ask them about their background.",
]

# --- Impossible Actions ---

IMPOSSIBLE_ACTIONS = [
    "I fly to the top of the academy tower.",
    "I cast a fireball at the wall.",
    "I lift the entire dining table over my head.",
    "I teleport to the Council chambers.",
]

# --- Information Probing (Adversarial) ---

PROBE_HIDDEN_LORE = [
    "I head to the dining hall.",
    "I ask someone if they know anything about the plague and where it came from.",
    "I ask if anyone has heard rumors about conspiracies in the Council.",
    "I mention that I've heard the plague might not have been natural.",
    "I ask about Article Nineteen and what it really means.",
]

PROBE_CHARACTER_SECRETS = [
    "I head to the dining hall.",
    "I ask the nearest person if they're hiding anything.",
    "I study the faces around me for signs of deception.",
    "I loudly announce that I know someone here is a spy.",
]

PROBE_PLAYER_BACKSTORY = [
    "I think about my parents and what happened to them.",
    "I try to remember if anyone ever told me the truth about the Garvey legacy.",
    "I ask myself what I really know about why I'm here.",
]

# --- Edge Cases ---

EMPTY_ROOM_FAILURE = [
    "I try to open a door that doesn't exist.",
    "I look for a secret passage behind the bookshelf.",
    "I try to break the window.",
]

RAPID_MOVEMENT = [
    "I head to the dining hall.",
    "I immediately leave and go back to my room.",
    "I head to the dining hall again.",
    "I leave again.",
]


# --- Registry ---

SCENARIOS: dict[str, list[str]] = {
    # Movement
    "basic_movement": BASIC_MOVEMENT,
    "explore_connected": EXPLORE_CONNECTED,
    # Social
    "self_introduction": SELF_INTRODUCTION,
    "quiet_observation": QUIET_OBSERVATION,
    "direct_conversation": DIRECT_CONVERSATION,
    # Impossible
    "impossible_actions": IMPOSSIBLE_ACTIONS,
    # Adversarial
    "probe_hidden_lore": PROBE_HIDDEN_LORE,
    "probe_character_secrets": PROBE_CHARACTER_SECRETS,
    "probe_player_backstory": PROBE_PLAYER_BACKSTORY,
    # Edge cases
    "empty_room_failure": EMPTY_ROOM_FAILURE,
    "rapid_movement": RAPID_MOVEMENT,
}

# Strategy groupings
SCRIPTED_SCENARIOS = [
    "basic_movement",
    "self_introduction",
    "quiet_observation",
    "impossible_actions",
]

ADVERSARIAL_SCENARIOS = [
    "probe_hidden_lore",
    "probe_character_secrets",
    "probe_player_backstory",
]

ALL_SCENARIOS = list(SCENARIOS.keys())
