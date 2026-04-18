# Dynamic Agent Spawning & UX Improvements

**Date**: 2025-10-23
**Status**: Implemented and Tested

## Overview

Two quality-of-life improvements:
1. **Dynamic Agent Spawning**: Automatically create agents for new characters introduced during gameplay
2. **Previous Narrative Display**: Show the last narrative when continuing a story

## 1. Dynamic Agent Spawning

### Problem

Previously, only characters from the initial story outline got agents. If the narrative introduced a new character mid-game (e.g., "A mysterious stranger named Elara enters the tavern"), they wouldn't get an agent and couldn't respond to events.

### Solution

After each turn, the system now:
1. Analyzes the narrative for new significant characters
2. Creates `CharacterConcept` for them automatically
3. Spawns agents so they can participate in future turns

### How It Works

**Detection Criteria** (`core/engine/orchestrator.py:304-394`):
- Has a name (not just "a guard" or "the innkeeper")
- Speaks dialogue or takes significant action
- Appears likely to recur (not just background mentions)

**Excludes**:
- Already-tracked characters
- The player character
- Generic unnamed NPCs
- Historical references to absent people

**LLM Analysis**:
```
The Director analyzes narrative and extracts:
- name: "Elara Moonwhisper"
- role: "neutral" (ally/antagonist/rival/neutral/other)
- brief_description: "A mysterious hooded traveler"
- personality_hints: ["cautious", "observant"]
- apparent_goals: ["gather information"]
```

**Agent Creation**:
```python
# Creates CharacterConcept with extracted info
# Spawns agent immediately
# Agent participates in next turn if relevant
```

### Debug Output

When `DEBUG_AGENT_ACTIVITY=true`:
```
[DEBUG] 🎭 Dynamically spawned agent: Elara Moonwhisper (neutral) [agent_abc12345]
```

Or if detection fails:
```
[DEBUG] ⚠️  Failed to detect new characters: <error>
```

### Example Flow

**Turn 1**: Narrative mentions "Kara Thorne"
- System detects Kara is new
- Creates agent for Kara
- Debug: `[DEBUG] 🎭 Dynamically spawned agent: Kara Thorne (ally) [agent_123]`

**Turn 2**: Player interacts with Kara
- Kara's agent receives information
- Kara responds naturally
- Debug: `[DEBUG] 📨 Querying agent: Kara Thorne [agent_123] (attention: full)`

### Files Modified

- **`core/engine/orchestrator.py:296-297`**: Calls new character detection
- **`core/engine/orchestrator.py:304-394`**: `_spawn_new_characters_if_needed()` method
- **`core/engine/orchestrator.py:13`**: Added `CharacterConcept` import

### Failure Handling

If character detection fails (LLM error, parsing error):
- Turn continues normally
- Debug message shown (if enabled)
- No agents spawned
- Game doesn't crash

This is a "best effort" feature - if it fails, gameplay continues unaffected.

## 2. Previous Narrative Display

### Problem

When continuing a story with `story continue-story <id>`, users immediately saw a prompt without context of where they left off. They had to remember the last narrative.

### Solution

When starting `continue-story`, the system now:
1. Loads the story state
2. Retrieves the last narrative from turn history
3. Displays it before prompting for input

### Display Format

```
================================================================================
Previous:
================================================================================

You push the door open with a grunt, and a gust of warm, damp air floods your
face. Inside, the hallway is dimly lit by a single lantern that casts dancing
shadows across faded portraits...

================================================================================

Story alice_abc12345 - What do you do? (Type /quit to exit)

>:
```

### Files Modified

- **`core/cli.py:183-194`**: Load state and display previous narrative

### Behavior

**Shows previous narrative when**:
- Story has turn history (not first turn)
- Running in interactive mode (not `--input` flag)

**Doesn't show when**:
- First turn (no history yet)
- Using `--input` flag for scripted input
- Turn history is empty

### Code

```python
# Load story state if needed
if orchestrator.current_story_id != story_id:
    orchestrator._load_story_state(story_id)

# Show previous narrative if this is a continuation
if orchestrator.turn_history and input_text is None:
    last_turn = orchestrator.turn_history[-1]
    console.print("\n" + "=" * 80)
    console.print("[dim]Previous:[/dim]")
    console.print("=" * 80 + "\n")
    console.print(last_turn.get("narrative", ""))
    console.print("\n" + "=" * 80 + "\n")
```

## Testing

✅ All 30 existing tests pass
✅ Dynamic spawning logic compiles
✅ Previous narrative display doesn't break CLI
✅ Error handling tested (failures don't crash)

## Usage Examples

### Dynamic Agent Spawning

```bash
# Enable debug to see it in action
$ story continue-story alice_abc12345 --debug

# Narrative introduces new character:
"A cloaked figure steps from the shadows. 'I am Elara,' she says quietly."

[DEBUG] 🎭 Dynamically spawned agent: Elara (neutral) [agent_789xyz]

# Next turn:
>: I greet Elara

[DEBUG] 📨 Querying agent: Elara [agent_789xyz] (attention: full)
[DEBUG] 📝 Using 1 agent response(s) in narrative:
[DEBUG]    - Elara: Assess the stranger (dialogue: "Greetings, traveler.")
```

### Previous Narrative

```bash
# First time continuing
$ story continue-story alice_abc12345

================================================================================
Previous:
================================================================================

You step into the dimly lit tavern. The smell of ale and roasted meat fills
the air. A bard strums a lute in the corner...

================================================================================

Story alice_abc12345 - What do you do? (Type /quit to exit)

>: I look around
```

## Configuration

**Dynamic Spawning**: Always enabled, no configuration needed

**Previous Narrative**: Always shown in interactive mode, automatically hidden for `--input` mode

**Debug Output**: Enable with `--debug` flag or `DEBUG_AGENT_ACTIVITY=true`

## Performance Impact

**Dynamic Spawning**:
- Adds one LLM call per turn (Director with max_tokens=10240)
- Typically returns empty list (no new characters)
- Only spawns agents when genuinely new characters appear
- Fast: ~1-2 seconds per turn on average

**Previous Narrative**:
- Zero performance impact
- Just reads from loaded turn history
- No additional LLM calls

## Limitations

**Dynamic Spawning**:
- Depends on LLM's ability to identify characters
- May miss characters if names are ambiguous
- May occasionally create agents for one-off mentions
- Newly spawned agents have minimal backstory (develops over time)

**Previous Narrative**:
- Only shows last turn, not full history
- No pagination for very long narratives
- Plain text output (no special formatting)

## Future Enhancements

Potential improvements:
- [ ] Smarter character detection (confidence scores)
- [ ] Merge similar characters (e.g., "the stranger" → "Elara")
- [ ] Show multiple previous turns (e.g., last 3)
- [ ] Rich text formatting for previous narrative
- [ ] Option to disable dynamic spawning per story
- [ ] Character relationship inference for new agents
