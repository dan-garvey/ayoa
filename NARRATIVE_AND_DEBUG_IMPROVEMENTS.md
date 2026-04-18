# Narrative Style & Debug Improvements

**Date**: 2025-10-23
**Status**: Implemented and Tested

## Overview

Three major improvements to gameplay and debugging:
1. Changed narrative style from third-person past tense to second-person present tense
2. Added debug flag to show agent activity (spawning, querying, responses used)
3. Added question detection to answer player queries without progressing narrative

## 1. Second-Person Present Tense Narrative

### Changes

**Before**: "Dan stepped forward and pushed the door open. He heard..."
**After**: "You step forward and push the door open. You hear..."

### Files Modified

- **`core/prompts/storyteller.txt`**: Updated composition guidelines
- **`core/roles/storyteller.py:248-256`**: Opening narrative prompt
- **`core/roles/storyteller.py:334-351`**: Turn narrative prompt

### Why This Matters

- **Immersion**: Player feels present in the moment
- **Directness**: "You" creates immediate connection
- **Standard Convention**: Most interactive fiction uses second-person present

### Example Output

```
You push the door open with a grunt, and a gust of warm, damp air floods your face,
carrying the faint echo of a lullaby. Inside, the hallway is dimly lit by a single
lantern that casts dancing shadows across faded portraits.

You step forward, your boots echoing softly on the stone floor. The hallway seems to
pulse with a quiet, steady rhythm. You pause at the threshold, ready to step into the
night's next chapter.
```

## 2. Debug Agent Activity Flag

### Configuration

**Environment Variable**: `DEBUG_AGENT_ACTIVITY=true` (default: false)
**CLI Flag**: `story continue-story <story_id> --debug` or `-d`

### What It Shows

1. **Agent Spawning** 🎭:
   ```
   [DEBUG] 🎭 Spawned agent: Kara Thorne (ally) [agent_abc12345]
   ```

2. **Agent Querying** 📨:
   ```
   [DEBUG] 📨 Querying agent: Kara Thorne [agent_abc12345] (attention: full)
   ```

3. **Responses Used in Narrative** 📝:
   ```
   [DEBUG] 📝 Using 2 agent response(s) in narrative:
   [DEBUG]    - Kara Thorne: React to player's entrance (dialogue: "Welcome back, Dan.")
   [DEBUG]    - Marcus Grey: Observe silently
   ```

### Files Modified

- **`core/config.py:50`**: Added `debug_agent_activity` field to EngineConfig
- **`core/config.py:66`**: Added env variable parsing
- **`.env.example:13`**: Added `DEBUG_AGENT_ACTIVITY=false`
- **`core/agents/manager.py:50-52`**: Debug log when spawning agents
- **`core/agents/manager.py:100-101`**: Debug log when querying agents
- **`core/roles/storyteller.py:309-315`**: Debug log when using responses
- **`core/cli.py:165-181`**: Added `--debug` flag to continue_story command

### Usage

```bash
# Enable via environment variable
DEBUG_AGENT_ACTIVITY=true story continue-story alice_abc12345

# Enable via CLI flag
story continue-story alice_abc12345 --debug
```

### Why This Matters

- **Visibility**: See exactly when agents are active
- **Debugging**: Identify if agents aren't receiving information
- **Performance**: Monitor concurrent agent queries
- **Understanding**: Learn how multi-agent system works

## 3. Question Detection

### Problem

When players asked questions like "Who do I recognize in the room?", the system treated it as an action and progressed the narrative instead of just answering.

### Solution

Added question detection that:
1. Detects question patterns (ends with "?", starts with who/what/where/when/why/how, etc.)
2. Answers directly without progressing time or narrative
3. Uses current scene and recent events as context
4. Maintains second-person present tense

### Files Modified

- **`core/engine/orchestrator.py:11`**: Added `llm_client` import
- **`core/engine/orchestrator.py:132-160`**: Added `_is_question()` method
- **`core/engine/orchestrator.py:162-218`**: Added `_handle_question()` method
- **`core/engine/orchestrator.py:242-243`**: Check for questions before processing turn

### Question Detection Logic

Detects questions if input:
- Ends with `?`
- Starts with: who, what, where, when, why, how
- Starts with: can i, could i, should i, would, is there, are there, do i, does, am i, will i, have i

### Answer Format

```
PLAYER QUESTION: "Who do I recognize in the room?"

SYSTEM RESPONSE (100-200 words):
You recognize Kara Thorne in the far corner, hunched over a scroll, her hair tucked
beneath a battered cap. She's a fellow outcast scholar you've known for two years.
Her sharp eyes flick up and meet yours, and you catch a faint smile.

You don't see anyone else you know well, though there are a few familiar faces from
passing in the halls - other students you've nodded to but never spoken with.
```

### Why This Matters

- **Player Convenience**: Quick info lookup without advancing time
- **Reduces Friction**: Don't have to "act" to get basic information
- **Natural**: How players think ("Who's here?" not "I look around to see who's here")
- **Preserves State**: Scene doesn't change, narrative doesn't advance

### Implementation Details

The `_handle_question()` method:
- Builds context from current scene
- Includes recent narrative (last 2 turns)
- Uses Storyteller's system prompt and world context
- Generates 100-200 word answer
- Does NOT add to turn history (no narrative progression)
- Returns `StoryOutput` with answer but no scene update

## Testing

✅ All 30 tests pass
✅ Syntax validated
✅ Debug output tested manually
✅ Question detection logic verified

## Usage Examples

### Standard Play (Second Person)
```bash
$ story continue-story dan_abc12345

> You step through the doorway
You step through the doorway into a dimly lit hall. Lantern light flickers across...
```

### Debug Mode
```bash
$ story continue-story dan_abc12345 --debug
Debug mode enabled

> You step through the doorway
[DEBUG] 📨 Querying agent: Kara Thorne [agent_abc12345] (attention: full)
[DEBUG] 📝 Using 1 agent response(s) in narrative:
[DEBUG]    - Kara Thorne: Greet the player (dialogue: "Welcome back.")

You step through the doorway into a dimly lit hall. Kara looks up from her scroll...
```

### Question Mode
```bash
> Who's in the room?
You see Kara Thorne in the corner, working on a scroll. She's a fellow student...

> What do I know about this place?
You know this is the Outsiders' dormitory, where students who aren't from noble...
```

## Configuration Reference

### Environment Variables
```env
# Debug flag
DEBUG_AGENT_ACTIVITY=false

# Still uses second-person for narrative (not configurable)
```

### CLI Flags
```bash
story continue-story <story_id> [OPTIONS]
  --debug, -d    Show agent activity debug output
  --input, -i    Provide action directly (non-interactive)
```

## Notes

- Debug output goes to stdout (same stream as narrative)
- Question detection is heuristic-based (may occasionally miss edge cases)
- Questions don't consume a turn or update agent memory
- Second-person present tense is now hardcoded (not configurable)

## Future Enhancements

Potential improvements:
- [ ] More sophisticated question detection (ML-based?)
- [ ] Debug output to separate file/stream
- [ ] Question history (cache common questions)
- [ ] Allow configuration of narrative tense (2nd/3rd person, present/past)
