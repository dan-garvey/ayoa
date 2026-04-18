# Retry Outline Cleanup & NPC Response Fixes

**Date**: 2025-10-24
**Status**: Implemented and Tested

## Two Critical Issues Fixed

### **Issue 1: `retry-outline` Polluted Save Files** ❌

**Problem**: When regenerating a story outline after initial creation failed, the save file retained all the old polluted data from previous attempts.

**Example of Pollution**:
```json
{
  "story_id": "dan_476d1624",
  "outline": {
    "major_characters": ["NEW characters from retry"]
  },
  "agents": {
    // 16 OLD agents from previous attempts still here!
    "agent_f2b8ba44": {"name": "Elara", ...},
    "agent_6482a5e5": {"name": "Lord Varric", ...},
    // ... 14 more obsolete agents
  },
  "turn_history": [
    // 12 turns from failed previous story still here!
  ],
  "storyteller_history": [
    // Old narrative completely unrelated to new outline!
  ]
}
```

**Result**: New story starts with 16 irrelevant agents, conflicting narrative history, and broken character relationships.

### **Issue 2: Storyteller Wouldn't Create NPC Names** ❌

**Problem**: When player interacted with generic NPCs and no agents responded (because they were all "remote"), the Storyteller produced awkward silent responses instead of inventing names and dialogue.

**Example**:
```
>: Hi, what are your names?

NARRATIVE:
The women pause, their eyes flicking to one another...
They observe you silently...
Their silence is not unkind...
```

**No names, no dialogue, just atmospheric descriptions of silence** 😶

## Root Causes

### **Retry-Outline Pollution**

The `retry_outline` command was:
1. Loading entire story state (including all old agents, turns, etc.)
2. Regenerating ONLY the outline
3. Saving everything back (keeping all the pollution)

```python
# core/cli.py:348-367 (OLD)
orchestrator._load_story_state(story_id)  # Loads EVERYTHING
outline = run_async(orchestrator.storyteller.generate_outline(...))
orchestrator.current_outline = outline
orchestrator._save_story_state(story_id)  # Saves EVERYTHING (polluted!)
```

### **Storyteller Silent NPCs**

Two compounding problems:

**1. Restrictive System Prompt** (`core/prompts/storyteller.txt:48`):
```
WHAT YOU MAY NOT DO:
- Add major character actions beyond the moves provided
```

This was intended to prevent the Storyteller from inventing actions for major characters (those with agents), but the Storyteller interpreted this as "don't create any NPCs at all."

**2. Silence-Suggesting Turn Prompt** (`core/roles/storyteller.py:339`):
```python
CHARACTER RESPONSES (for this turn only):
{moves_text if character_moves else "None - characters observe silently"}
                                              ^^^^^^^^^^^^^^^^^^^^^^^^
```

This explicitly told the Storyteller "characters observe silently" when no agents responded!

## Solutions Implemented

### **Fix 1: Clean Retry-Outline**

Modified `core/cli.py:365-376` to wipe all polluted state:

```python
console.print("[bold yellow]Generating outline...[/bold yellow]")

# Regenerate outline
outline = run_async(orchestrator.storyteller.generate_outline(orchestrator.current_config))

# CRITICAL: Clean up polluted state from previous attempts
console.print("[dim]Cleaning up old story data...[/dim]")
orchestrator.current_outline = outline
orchestrator.current_scene = None  # Will be regenerated on start
orchestrator.turn_history = []  # Clear old turns
orchestrator.agent_manager.agents = {}  # Clear old agents
orchestrator.agent_manager.agent_states = {}
orchestrator.storyteller.conversation_history = []  # Clear storyteller memory
orchestrator.storyteller.world_context = None  # Will be regenerated on start

# Save clean state
orchestrator._save_story_state(story_id)
```

**What Gets Wiped:**
- ✓ All agents (will respawn based on new outline)
- ✓ Turn history (fresh start)
- ✓ Current scene (will regenerate opening)
- ✓ Storyteller memory (no conflicting narrative)
- ✓ World context (will regenerate for new story)

**What Stays:**
- ✓ Player character config (user's preferences)
- ✓ Story preferences (genre, tone, etc.)
- ✓ Story ID (for save file continuity)

### **Fix 2: NPC Creation Permissions**

**Updated System Prompt** (`core/prompts/storyteller.txt:13-16`):

```diff
 YOUR RESPONSIBILITIES:
 1. WORLD CONSISTENCY: Enforce established facts and world rules from the world context
 2. NARRATIVE COMPOSITION: Transform player actions and character responses into vivid prose (200-500 words)
 3. TONE MAINTENANCE: Strictly adhere to the story's tone and genre conventions
-4. NPC MANAGEMENT: Control minor NPCs and environmental reactions naturally
+4. NPC MANAGEMENT: Control ALL NPCs and background characters naturally
+   - Minor/background NPCs: Full creative freedom (names, dialogue, actions)
+   - Major characters (with agents): Only add minimal reactions if they didn't respond
 5. PACING: Balance exposition, action, and character moments appropriately
```

**Added Clarification** (`core/prompts/storyteller.txt:56-60`):

```
IMPORTANT DISTINCTION:
- Major characters (those with character agents): Use their provided moves when available
  If they didn't provide moves but player directly addresses them, they can give minimal responses
- Minor/background NPCs: YOU create and control these freely with names, dialogue, and actions
- When in doubt, populate the scene with interesting named NPCs to make the world feel alive
```

**Updated Turn Prompt** (`core/roles/storyteller.py:338-355`):

```python
CHARACTER RESPONSES (for this turn only):
{moves_text if character_moves else "None - major characters didn't respond (they may be remote or observing)"}
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                              Changed from "characters observe silently"

Write 200-500 words of narrative that:
- Describes the player's action and its immediate effects (address player as "you")
- Integrates character responses naturally (preserve exact dialogue!)
- Shows NPC reactions as needed
- If player addresses people and no character responses provided, YOU create appropriate NPC responses with names and dialogue
  (These become minor NPCs under your control)
- Maintains the scene's atmosphere and continuity with previous narrative
- Uses second-person present tense ("You move" not "You moved")
- Ends on a natural pause for the next player input

CRITICAL: If the player asks people for their names and no character responses are provided,
invent interesting named NPCs with appropriate dialogue. Make the world feel populated and alive!
```

## Expected Behavior Now

### **Retry-Outline: Clean Slate**

```bash
$ story retry-outline dan_476d1624

Loading story configuration for dan_476d1624...
Story dan_476d1624 already has an outline.
Premise: A tale of political intrigue...
Generate a new outline? [y/n]: y

Generating outline...
Cleaning up old story data...

Successfully generated outline for dan_476d1624

Premise: A mystery at the magical academy where scholarship students face discrimination
Major Characters:
  - Elara Moonwhisper (ally): A sympathetic elf librarian
  - Lord Thorne (antagonist): A noble who opposes commoner students
  - Kara Vale (rival): A fellow scholarship student competing for top marks

Start your adventure now? [y/n]: y
```

**Save File Before Retry:**
```json
{
  "agents": {/* 16 old agents */},
  "turn_history": [/* 12 old turns */],
  "storyteller_history": [/* old narratives */]
}
```

**Save File After Retry:**
```json
{
  "agents": {},  // CLEAN
  "turn_history": [],  // CLEAN
  "storyteller_history": [],  // CLEAN
  "outline": {/* NEW outline */}
}
```

### **NPC Creation: Populated World**

**Before Fix:**
```
>: Hi, what are your names?

The women pause, their eyes flicking to one another...
They observe you silently...
```

**After Fix:**
```
>: Hi, what are your names?

The silver-haired woman smiles. "I'm Celeste, House Valen scholarship recipient."
A second student with ink-stained fingers nods. "Mara. Guild of Scholars."
The third, her gown bearing a subtle rebel sigil, grins. "Call me Ash."

[DEBUG] 🎭 Dynamically spawned agent: Celeste (ally) [agent_abc123]
[DEBUG] 🎭 Dynamically spawned agent: Mara (neutral) [agent_def456]
[DEBUG] 🎭 Dynamically spawned agent: Ash (ally) [agent_ghi789]
```

**What Happens:**
1. Storyteller invents names and dialogue for the NPCs
2. Dynamic spawning detects the new named characters
3. Future turns: These NPCs now have agents and can respond intelligently!

## Technical Details

### **Major vs Minor NPCs**

The Storyteller now understands this distinction:

**Major Characters (with agents)**:
- Created from story outline during setup
- Have full CharacterAgent with memory and goals
- Director routes information to them
- Storyteller uses their provided moves when available
- If they don't respond (remote), Storyteller can give minimal reactions

**Minor NPCs (Storyteller-created)**:
- Created on the fly by Storyteller when needed
- Initially just names and dialogue in narrative
- Dynamic spawning MAY promote them to agents if significant
- Controlled entirely by Storyteller until they get agents

**Example Flow:**
```
Turn 1: "A merchant approaches you"
  → Storyteller creates unnamed NPC

Turn 2: Player: "What's your name?"
  → Storyteller: "Call me Gareth the Trader"
  → Dynamic spawning detects "Gareth"
  → Creates agent for Gareth

Turn 3: Player: "Gareth, what do you sell?"
  → Gareth now has agent
  → Gareth responds intelligently based on goals/personality
```

### **Safety Mechanisms**

**Major Characters Can't Be Hijacked:**
```
// Major character "Elara" has an agent
Player: "I ask Elara her opinion"

Elara's agent is remote (not in scene)
  ↓
Elara doesn't provide a move
  ↓
Storyteller CAN add: "Elara is not here"
Storyteller CAN add: "You remember Elara mentioning..."
Storyteller CANNOT add: "Elara suddenly appears and says..." (major action)
```

**Minor NPCs Have Creative Freedom:**
```
Player: "I ask the bartender for gossip"

No agent for "bartender"
  ↓
Storyteller creates: "The bartender, a gruff dwarf named Harg, leans in.
'Heard Lord Valen's plotting something,' he whispers."
  ↓
Harg might get an agent if he becomes important
```

## Files Modified

1. **`core/cli.py:365-376`**
   - Clean up polluted state in retry-outline
   - Wipe agents, turn history, scene, storyteller memory

2. **`core/prompts/storyteller.txt:13-16`**
   - Clarified NPC management responsibilities
   - Distinguish major vs minor NPCs

3. **`core/prompts/storyteller.txt:56-60`**
   - Added explicit distinction section
   - Encourage populating world with NPCs

4. **`core/roles/storyteller.py:338-355`**
   - Changed silence prompt to permission prompt
   - Added critical instruction for name requests

## Testing

✅ All 30 existing tests pass
✅ No breaking changes
✅ Retry-outline produces clean saves
✅ Storyteller can create NPCs freely

## Impact

**Before Fixes:**
- ❌ Retry-outline polluted with 16+ old agents
- ❌ NPCs silent and generic
- ❌ World felt empty and lifeless
- ❌ Player interactions fell flat

**After Fixes:**
- ✅ Retry-outline starts completely fresh
- ✅ NPCs have names and personality
- ✅ World feels populated and alive
- ✅ Player interactions get real responses
- ✅ Dynamic spawning can promote NPCs to agents

## Migration Notes

**Existing Stories:**
- Already-polluted saves will remain polluted
- Running `retry-outline` will clean them up
- Next turn after fix: Storyteller will create NPCs normally

**New Stories:**
- Clean from the start
- NPCs populate naturally from turn 1

## Usage Examples

### **Retry-Outline**

```bash
# Story creation failed during outline generation
$ story create
# ... error occurs during outline generation ...

# Retry with same character config
$ story retry-outline alice_abc12345
Loading story configuration for alice_abc12345...
Generating outline...
Cleaning up old story data...

Successfully generated outline for alice_abc12345

Start your adventure now? [y/n]: y
```

### **NPC Interaction**

```bash
$ story continue-story alice_abc12345 --debug

>: I ask the guards their names

[DEBUG] 🎬 Director routing: 0 character(s) evaluated
[DEBUG]    (No major characters in scene)

================================================================================
You approach the two guards flanking the gate. The first, a weathered veteran
with a scar across his cheek, nods. "Captain Vorin," he says gruffly. The
second, younger and nervous, shifts his spear. "Recruit Tam, sir."

You notice Vorin's eyes are sharp, watching every movement.
================================================================================

[DEBUG] 🎭 Dynamically spawned agent: Captain Vorin (neutral) [agent_abc123]
[DEBUG] 🎭 Dynamically spawned agent: Recruit Tam (ally) [agent_def456]

>: Vorin, have you seen anything unusual?

[DEBUG] 🎬 Director routing: 1 character(s) evaluated
[DEBUG]    ✓ Captain Vorin (full) - Present, directly addressed

[DEBUG] 📨 Querying agent: Captain Vorin [agent_abc123] (attention: full)

[DEBUG] 🎬 Director validation: 1 accepted, 0 rejected
[DEBUG]    ✓ Captain Vorin: Consider the question carefully | Says: "Depends what you mean by unusual, stranger."
```

## Future Enhancements

Potential improvements:
- [ ] Track NPC→Agent promotions explicitly
- [ ] Allow Storyteller to "retire" minor NPCs (remove from scene)
- [ ] NPC personality consistency across turns (before agent creation)
- [ ] Option to manually trigger dynamic spawning for specific NPCs
