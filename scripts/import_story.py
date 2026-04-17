"""Import a monolithic single-agent IF prompt into engine-consumable structures.

Usage:
    .venv/bin/python scripts/import_story.py story_prompt.txt [--output saves/my_story.json]

Takes a single-agent interactive fiction prompt and decomposes it into:
- WorldState (setting, lore, physics, locations, facts)
- CharacterRecords (one per NPC, with rich text fields)
- Narrative rules (prose discipline, pacing, style)
- SessionConfig

Outputs a CheckpointFile JSON that the engine can load.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from app.llm.client import LLMClient
from app.llm.config import LLMConfig
from app.schemas.characters import (
    CharacterMemory,
    CharacterRecord,
    CharacterStatus,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.narrator import TranscriptEntry
from app.schemas.state import (
    LocationState,
    PhysicsRuleset,
    SessionConfig,
    SessionState,
    StorySetting,
    WorldState,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Extraction prompts ---

WORLD_EXTRACTION_PROMPT = """\
You are a story structure analyst. Given a monolithic interactive fiction prompt, extract the world-building elements into structured JSON.

## Source Prompt
{source_prompt}

## Your Task
Extract the world state from this prompt. Respond with ONLY valid JSON matching this schema:

{{
  "setting": {{
    "genre": "string — e.g. fantasy, sci-fi, historical",
    "era": "string — time period or equivalent",
    "tone": "string — e.g. dark political intrigue, lighthearted adventure",
    "premise": "string — 1-3 sentence premise of the story"
  }},
  "lore": "string — COMMON KNOWLEDGE world lore only: history, factions, political systems, laws, magic systems, religions, key events that are publicly known. Do NOT include hidden conspiracies, secret plots, true causes of mysterious events, or information that characters would need to discover through play. If something is a mystery in the story (e.g. the true cause of a plague), describe it as a mystery — do not reveal the answer here. This should be thorough — multiple paragraphs.",
  "facts": ["string — discrete current-state facts about the world that are publicly known at story start. Do NOT include secret or hidden facts."],
  "physics_ruleset": {{
    "strength_limits": "string — baseline physical capability",
    "magic_enabled": true/false
  }},
  "locations": {{
    "current_scene_id": "string — starting location ID",
    "scene_graph": {{
      "location_id": {{
        "name": "string",
        "description": "string",
        "connected_to": ["location_id"],
        "properties": {{}}
      }}
    }}
  }},
  "narrative_rules": "string — all prose discipline rules, pacing guidelines, style rules, dialogue rules, success/failure metrics for the narrator. Include everything about HOW to narrate, not WHAT to narrate. This should be thorough."
}}"""

CHARACTER_EXTRACTION_PROMPT = """\
You are a character analyst for an interactive fiction engine. Given a monolithic prompt, extract ALL named characters into structured records.

## Source Prompt
{source_prompt}

## Your Task
Extract every named character (NPCs, love interests, rivals, supporting cast, faculty, political figures) into structured records. For each character, capture their full depth.

Respond with ONLY valid JSON — an array of character objects:

[
  {{
    "character_id": "string — lowercase_snake_case unique ID",
    "name": "string — full name",
    "status": "active" or "dormant",
    "location": "string — starting location ID",
    "public_sheet": {{
      "role": "string — their role/title",
      "traits": ["string — personality traits, 3-6"],
      "voice": "string — how they speak, their verbal style",
      "appearance": "string — full physical description",
      "faction": "string — faction/house/group affiliation"
    }},
    "private_state": {{
      "goals": ["string — what they want"],
      "attitudes": {{"character_id_or_user": float_between_neg1_and_1}},
      "secrets": ["string — things they know but hide"],
      "intentions_enabled": true/false
    }},
    "backstory": "string — character history as THEY understand it, from THEIR perspective. Include only what this character knows or believes about themselves and their past. Do NOT include hidden plot information, conspiracy details, or secrets that this character is unaware of.",
    "personality": "string — deep personality description, inner world, private self, what they want and fear",
    "narrative_notes": "string — how to play this character in scenes: what they hide, what pressures them, how they deflect, what kind of interaction changes them, and what narrative role they serve. If the source clearly contains a distinctive tell, include it as situational rather than always-on."
  }}
]

Important:
- Include ALL characters mentioned in the prompt, not just love interests
- Set attitudes between characters where the prompt describes relationships (use character_id as key)
- Set attitude toward the protagonist as "user"
- Capture the FULL depth of each character — personality essays, backstories, narrative notes should be thorough
- For characters described briefly (supporting cast), still create a record with what's available
- Set intentions_enabled=true for characters with hidden agendas or complex inner lives
- In `narrative_notes`, prioritize pressure points, contradictions, social constraints, and dramatic function over surface quirks
- Do NOT manufacture stock mannerisms such as repeated smiles, hair touches, brow lifts, ring touches, or "sparkling eyes" unless the source prompt makes them specifically important
- Do NOT restate appearance details as behavior notes unless the source explicitly ties them to a meaningful, situational tell
- If romance is present, capture the constraints, risks, and what earns trust; do not reduce the character to canned flirtation beats

## Information Isolation — CRITICAL
The source prompt may contain hidden plot details, conspiracies, and secrets. You MUST separate what each character KNOWS from what the READER knows:
- **backstory**: Write ONLY what the character themselves knows or believes. If the protagonist doesn't know a plague was engineered, their backstory says "the plague" not "the engineered plague." If an NPC is secretly a spy, their backstory should read like a normal person's history — the spy role goes in `secrets`.
- **public_sheet**: Contains ONLY what is observable to others (role, appearance, traits, voice, faction).
- **private_state.secrets**: This is where hidden knowledge, true allegiances, conspiracies, and plot-relevant information belongs. A character who is secretly an agent of a conspiracy has that fact HERE, not in their backstory or personality.
- **narrative_notes**: May reference hidden dynamics for the author's benefit, but should not state secrets as if the character is aware of them (unless they are)."""


OPENING_NARRATIVE_PROMPT = """\
You are a narrative writer for an interactive fiction engine. Given a story prompt, write the opening passage that the player reads when they first start the game.

## Source Prompt
{source_prompt}

## Your Task
Write the opening narrative — the first thing the player reads. This sets the scene, establishes who they are and why they're here, and ends at a moment where they can choose what to do.

## Rules
1. Write in **second person present tense**: "You step off the bridge..." not "The player steps..."
2. Use **PLAYER_NAME** as a placeholder for the character's name (it will be replaced later). The name already exists as a placeholder — do NOT ask the user to provide it.
3. Begin immediately in-fiction. Do NOT use preambles like "Before we begin," and do NOT ask setup questions about name, appearance, or backstory.
4. Write **3-5 paragraphs** of polished prose
5. Establish:
   - Who PLAYER_NAME is and why they are arriving at this place
   - What they see, hear, and feel — sensory grounding in the starting location
   - The stakes: what's at risk, what they don't know, what looms ahead
   - A sense of being an outsider entering an unfamiliar world
6. End at a **natural decision point** — PLAYER_NAME has arrived, taken in the scene, and can now choose what to do. Do NOT make any choices for them (don't decide where they sit, what they pick up, who they talk to)
7. Include ONLY **common knowledge**. No conspiracy details, no hidden lore, no secret plots. PLAYER_NAME doesn't know the dark history yet — they're a newcomer
8. Do NOT use the word "you" in the very first word — start with something more evocative
9. Never break the fourth wall. Never mention prompts, players, character creation, or what the reader should supply.

## Output
Return ONLY the prose text. No JSON, no markdown fences, no commentary. Just the narrative passage with paragraph breaks."""


async def extract_world(client: LLMClient, source: str) -> dict:
    """Extract world state and narrative rules from the source prompt."""
    logger.info("Extracting world state...")
    result = await client.complete(
        role="narrator",
        messages=[
            {"role": "user", "content": WORLD_EXTRACTION_PROMPT.format(source_prompt=source)},
        ],
        temperature=0.3,
        max_tokens=8000,
    )
    from app.llm.client import extract_json
    data = json.loads(extract_json(result.content))
    logger.info(f"  Setting: {data.get('setting', {}).get('genre', '?')} / {data.get('setting', {}).get('tone', '?')}")
    logger.info(f"  Locations: {len(data.get('locations', {}).get('scene_graph', {}))} extracted")
    logger.info(f"  Facts: {len(data.get('facts', []))} extracted")
    return data


async def extract_characters(client: LLMClient, source: str) -> list[dict]:
    """Extract all character records from the source prompt."""
    logger.info("Extracting characters...")
    result = await client.complete(
        role="narrator",
        messages=[
            {"role": "user", "content": CHARACTER_EXTRACTION_PROMPT.format(source_prompt=source)},
        ],
        temperature=0.3,
        max_tokens=16000,
    )
    from app.llm.client import extract_json
    raw_json = extract_json(result.content)
    characters = json.loads(raw_json)
    # Unwrap if LLM returned {"characters": [...]} or {"key": [...]}
    if isinstance(characters, dict):
        if "characters" in characters:
            characters = characters["characters"]
        else:
            # Find the first list value
            for v in characters.values():
                if isinstance(v, list):
                    characters = v
                    break
    # Filter to only dict entries (LLM sometimes includes string labels in arrays)
    characters = [c for c in characters if isinstance(c, dict)]
    logger.info(f"  Characters: {len(characters)} extracted")
    for c in characters:
        logger.info(f"    - {c.get('name', '?')} ({c.get('character_id', '?')}) [{c.get('public_sheet', {}).get('role', '?')}]")
    return characters


async def extract_opening(client: LLMClient, source: str) -> str:
    """Generate the opening narrative passage from the source prompt."""
    logger.info("Generating opening narrative...")
    result = await client.complete(
        role="narrator",
        messages=[
            {"role": "user", "content": OPENING_NARRATIVE_PROMPT.format(source_prompt=source)},
        ],
        temperature=0.5,
        max_tokens=4000,
    )
    text = result.content.strip()
    # Strip markdown fences if the LLM wrapped it
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].rstrip()
    logger.info(f"  Opening narrative: {len(text)} chars")
    return text


def build_checkpoint(world_data: dict, characters_data: list[dict], story_id: str, opening_narrative: str = "") -> CheckpointFile:
    """Assemble extracted data into a CheckpointFile."""
    session = SessionState(
        session_id=story_id,
        story_id=story_id,
        turn_index=0,
    )

    setting_data = world_data.get("setting", {})
    setting = StorySetting(
        genre=setting_data.get("genre", ""),
        era=setting_data.get("era", ""),
        tone=setting_data.get("tone", ""),
        premise=setting_data.get("premise", ""),
    )

    locations_data = world_data.get("locations", {})
    locations = LocationState(
        current_scene_id=locations_data.get("current_scene_id", ""),
        scene_graph=locations_data.get("scene_graph", {}),
    )

    physics_data = world_data.get("physics_ruleset", {})
    physics = PhysicsRuleset(
        strength_limits=physics_data.get("strength_limits", "human_baseline"),
        magic_enabled=physics_data.get("magic_enabled", False),
    )

    world_state = WorldState(
        locations=locations,
        facts=world_data.get("facts", []),
        physics_ruleset=physics,
        setting=setting,
        lore=world_data.get("lore", ""),
    )

    config = SessionConfig(
        narrative_rules=world_data.get("narrative_rules", ""),
    )

    # Build character records
    characters = []
    for cd in characters_data:
        ps_data = cd.get("public_sheet", {})
        priv_data = cd.get("private_state", {})

        # Clamp attitudes to [-1, 1]
        attitudes = {}
        for k, v in priv_data.get("attitudes", {}).items():
            try:
                attitudes[k] = max(-1.0, min(1.0, float(v)))
            except (ValueError, TypeError):
                attitudes[k] = 0.0

        cr = CharacterRecord(
            character_id=cd.get("character_id", "unknown"),
            name=cd.get("name", "Unknown"),
            status=CharacterStatus(cd.get("status", "active")),
            location=cd.get("location", ""),
            public_sheet=PublicSheet(
                role=ps_data.get("role", ""),
                traits=ps_data.get("traits", []),
                voice=ps_data.get("voice", ""),
                appearance=ps_data.get("appearance", ""),
                faction=ps_data.get("faction", ""),
            ),
            private_state=PrivateState(
                goals=priv_data.get("goals", []),
                attitudes=attitudes,
                secrets=priv_data.get("secrets", []),
                intentions_enabled=priv_data.get("intentions_enabled", False),
            ),
            backstory=cd.get("backstory", ""),
            personality=cd.get("personality", ""),
            narrative_notes=cd.get("narrative_notes", ""),
        )
        characters.append(cr)

    return CheckpointFile(
        session=session,
        opening_narrative=opening_narrative,
        world_state=world_state,
        characters=characters,
        config=config,
    )


async def import_story(source_path: str, output_path: str, story_id: str) -> CheckpointFile:
    """Main import pipeline: read source, extract via LLM, assemble checkpoint."""
    with open(source_path, "r") as f:
        source = f.read()

    logger.info(f"Source prompt: {len(source)} chars, ~{len(source.split())} words")

    config = LLMConfig.from_env()
    client = LLMClient(config=config)

    try:
        # Run world, character, and opening extraction in parallel
        world_data, characters_data, opening_text = await asyncio.gather(
            extract_world(client, source),
            extract_characters(client, source),
            extract_opening(client, source),
        )

        checkpoint = build_checkpoint(world_data, characters_data, story_id, opening_narrative=opening_text)

        # Write output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(checkpoint.model_dump_json(indent=2))

        logger.info(f"\nCheckpoint written to {output_path}")
        logger.info(f"  Session ID: {checkpoint.session.session_id}")
        logger.info(f"  Characters: {len(checkpoint.characters)}")
        logger.info(f"  Genre: {checkpoint.world_state.setting.genre}")
        logger.info(f"  Lore length: {len(checkpoint.world_state.lore)} chars")
        logger.info(f"  Narrative rules length: {len(checkpoint.config.narrative_rules)} chars")

        return checkpoint
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description="Import a single-agent IF prompt into engine format")
    parser.add_argument("source", help="Path to the source prompt text file")
    parser.add_argument("--output", "-o", default=None, help="Output checkpoint JSON path")
    parser.add_argument("--story-id", default=None, help="Story ID for the checkpoint")
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.source))[0]
    if args.story_id is None:
        args.story_id = base
    if args.output is None:
        args.output = f"app/storage/saves/{args.story_id}/ckpt_0000.json"

    asyncio.run(import_story(args.source, args.output, args.story_id))


if __name__ == "__main__":
    main()
