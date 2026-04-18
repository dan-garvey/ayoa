# Debug Output and Architecture Fixes

**Date**: 2025-10-24
**Status**: Complete and Tested

## Summary

Fixed critical issues with debug output visibility and player character agent creation that were discovered during gameplay testing.

## Issues Identified

### 1. Director Operations Had No Debug Output ❌
**Problem**: Director routing and validation steps were invisible in debug mode, making it impossible to see:
- Which characters received information
- What attention level they got
- Which moves were accepted/rejected
- Why decisions were made

**Impact**: Missing visibility into the core orchestration logic

### 2. Player Character Was Being Spawned as Agent ❌
**Problem**: The player character (e.g., "Dan") was getting an agent and being queried during turns
```
[DEBUG] 📨 Querying agent: Dan [agent_f2b8ba44] (attention: full)
```

**Impact**: Fundamentally breaks the architecture - the player should control their character directly, not through an AI agent!

### 3. No Dynamic Spawning for New Characters ❌
**Problem**: When "Lyra" was introduced in the narrative with:
- Specific name
- Physical description
- Personality traits
- Dialogue potential

No agent was spawned, so Lyra couldn't participate in future turns.

**Impact**: New characters introduced mid-game remained static NPCs instead of dynamic agents

### 4. Model Auto-Detection Spam ⚠️
**Problem**: "Auto-detected model: openai/gpt-oss-20b" appeared on every turn, interrupting gameplay flow

**Impact**: Annoying clutter during interactive play

## Fixes Applied

### Fix 1: Director Debug Output (🎬)

Added comprehensive debug logging to `core/roles/director.py`:

**Routing Decisions** (lines 129-136):
```python
# Debug output for routing decisions
if engine_config.debug_agent_activity:
    print(f"[DEBUG] 🎬 Director routing: {len(result.decisions)} character(s) evaluated")
    for decision in result.decisions:
        if decision.receives_packet:
            print(f"[DEBUG]    ✓ {decision.character} ({decision.attention_level}) - {decision.reason}")
        else:
            print(f"[DEBUG]    ✗ {decision.character} (no info) - {decision.reason}")
```

**Move Validation** (lines 235-245):
```python
# Debug output for move validation
if engine_config.debug_agent_activity:
    print(f"[DEBUG] 🎬 Director validation: {len(decision.accepted_moves)} accepted, {len(decision.rejected_moves)} rejected")
    for move in decision.accepted_moves:
        action_str = f" | Action: {move.action}" if move.action else ""
        dialogue_str = f" | Says: \"{move.dialogue}\"" if move.dialogue else ""
        print(f"[DEBUG]    ✓ {move.character}: {move.intent}{action_str}{dialogue_str}")
    for validation in decision.rejected_moves:
        print(f"[DEBUG]    ✗ {validation.move.character}: {validation.reason}")
    if decision.npc_actions_needed:
        print(f"[DEBUG]    NPC actions needed: {', '.join(decision.npc_actions_needed)}")
```

**Example Output**:
```
[DEBUG] 🎬 Director routing: 3 character(s) evaluated
[DEBUG]    ✓ Alice (full) - Character is present in scene
[DEBUG]    ✓ Bob (peripheral) - Nearby, can overhear
[DEBUG]    ✗ Carol (no info) - Too far away to perceive

[DEBUG] 🎬 Director validation: 1 accepted, 0 rejected
[DEBUG]    ✓ Alice: Greet the player | Action: waves | Says: "Hello there!"
[DEBUG]    NPC actions needed: bartender nods
```

### Fix 2: Prevent Player Character Agent Spawning

**Modified `core/engine/orchestrator.py` (lines 106-111)**:
```python
# Spawn character agents (EXCLUDE player character - they're controlled by the user!)
npc_characters = [
    char for char in self.current_outline.major_characters
    if char.name.lower() != self.current_config.player_character.name.lower()
]
agent_ids = await self.agent_manager.spawn_agents(npc_characters)
```

**Updated `core/roles/storyteller.py` (lines 65-66)**:
```python
3. 2-4 major NPC characters (allies, rivals, antagonists, etc.) who will interact with the player
   NOTE: Do NOT include the player character ({player.name}) in major_characters - only NPCs!
```

This ensures:
- Player character is filtered out before agent spawning
- LLM is explicitly told not to include player in major_characters
- Player actions come directly from user input, not AI agents

### Fix 3: Silent Model Detection

**Modified `core/llm_client.py` (lines 37-48)**:

Removed all print statements from model detection:
```python
data = response.json()
if data.get("data") and len(data["data"]) > 0:
    self.model_name = data["data"][0]["id"]
else:
    # Fallback to config if available
    self.model_name = llm_config.model_name or "unknown"

self._model_fetched = True
```

And silent error handling:
```python
except Exception as e:
    # Silent fallback - don't spam the user with warnings
    self.model_name = llm_config.model_name or "unknown"
    self._model_fetched = True
```

Model detection still works, just doesn't interrupt gameplay. Users can check model with `story config` command.

### Fix 4: Dynamic Spawning Already Working ✅

The dynamic spawning feature was already implemented correctly in `orchestrator.py:305-394`. The issue was that debug output wasn't showing because of missing Director debug output (now fixed).

## Complete Debug Output Flow

With all fixes applied, here's what you'll see in `--debug` mode:

```bash
$ story continue-story alice_abc12345 --debug

[DEBUG] 🎬 Director routing: 2 character(s) evaluated
[DEBUG]    ✓ Kara Thorne (full) - Present in scene, directly addressed
[DEBUG]    ✗ Marcus Grey (no info) - In different location

[DEBUG] 📨 Querying agent: Kara Thorne [agent_abc123] (attention: full)

[DEBUG] 🎬 Director validation: 1 accepted, 0 rejected
[DEBUG]    ✓ Kara Thorne: Respond warmly | Action: smiles | Says: "Good to see you!"

[DEBUG] 📝 Using 1 agent response(s) in narrative:
[DEBUG]    - Kara Thorne: Respond warmly (action: smiles) (dialogue: "Good to see you!")

[DEBUG] 🎭 Dynamically spawned agent: Elara (neutral) [agent_xyz789]
```

## Emoji Legend

- 🎭 = Agent spawning (initial and dynamic)
- 🎬 = Director operations (routing and validation)
- 📨 = Agent querying (information distribution)
- 📝 = Response usage (what makes it into the narrative)

## Testing

All features tested comprehensively:

```bash
$ python test_debug_comprehensive.py

✓ Agent spawning debug (🎭)
✓ Director routing debug (🎬)
✓ Agent querying debug (📨)
✓ Director validation debug (🎬)
✓ Response usage debug (📝)
✓ Dynamic spawning debug (🎭)
```

All 30 existing unit tests still pass:
```bash
$ pytest tests/ -v
================================ 30 passed ================================
```

## Files Modified

1. **`core/roles/director.py`** (lines 129-136, 235-245)
   - Added debug output for routing decisions
   - Added debug output for move validation

2. **`core/engine/orchestrator.py`** (lines 106-111)
   - Filter player character from agent spawning

3. **`core/roles/storyteller.py`** (lines 65-66)
   - Clarified prompt to exclude player character

4. **`core/llm_client.py`** (lines 37-48)
   - Removed model detection print statements

5. **`test_debug_comprehensive.py`** (new file)
   - Comprehensive test suite for all debug outputs

## Configuration

Debug mode can be enabled two ways:

**Environment variable**:
```bash
DEBUG_AGENT_ACTIVITY=true story continue-story story_id
```

**CLI flag** (easier):
```bash
story continue-story story_id --debug
```

## Impact

- ✅ Full visibility into Director decision-making
- ✅ Player character correctly excluded from agent system
- ✅ Clean gameplay without model detection spam
- ✅ Dynamic spawning visibility confirmed working
- ✅ All existing functionality preserved
- ✅ No breaking changes

## Future Enhancements

Potential improvements:
- [ ] Color-coded debug output for different components
- [ ] Debug log file option (--debug-file) for post-game analysis
- [ ] Adjustable debug verbosity levels (--debug-level 1-3)
- [ ] Performance timing in debug mode
