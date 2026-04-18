# Scene Management Fix

**Date**: 2025-10-24
**Status**: Implemented and Tested

## Critical Bug Fixed

### **The Problem: Frozen Scene = No Character Interaction**

During gameplay testing, we discovered that NPCs **never responded** to the player, despite having 16 agents spawned. Here's what was happening:

**Problematic Gameplay:**
```
>: Hi, what are your names?

The women pause, their eyes flicking to one another...
They observe you silently...
Their silence is not unkind...
```

**No names, no dialogue, just atmospheric silence.** 😶

### Root Cause Analysis

**1. Scene State Was Frozen**
```python
# From saves/dan_476d1624.json
{
  "scene": {
    "present_characters": ["Dan"],  // Only player!
    "nearby_characters": []         // Nobody nearby!
  },
  "agents": {
    // 16 agents exist: Elara, Seraphine, Elysia, Lyra, etc.
    // But NONE are in the scene!
  }
}
```

**2. Storyteller Never Updated Scene**
```python
# core/roles/storyteller.py:366
return StoryOutput(
    narrative=narrative,
    visible_moves=character_moves,
    scene_update=None,  # ALWAYS None!
)
```

**3. Orchestrator Checked But Never Updated**
```python
# core/engine/orchestrator.py:295-296
if story_output.scene_update:  # Always False
    self.current_scene = story_output.scene_update
```

### The Cascade of Failures

```
Scene stuck with only [Dan]
    ↓
All agents marked as "remote" by Director
    ↓
All get receives_packet: false
    ↓
NO agents queried
    ↓
NO debug output (nothing to log)
    ↓
NO character responses
    ↓
NPCs remain silent and generic
    ↓
Game feels dead and lifeless
```

## The Solution: Director-Managed Scene Updates

The Director now **actively manages scene state** by analyzing narratives and matching characters to the scene.

### Implementation

**New Method: `Director.update_scene_from_narrative()`**
(`core/roles/director.py:249-339`)

```python
async def update_scene_from_narrative(
    self,
    narrative: str,
    user_action: str,
    current_scene: Scene,
    available_agents: dict[str, "AgentState"],
) -> Scene:
    """
    Analyze narrative and update scene with present/nearby characters.

    This is CRITICAL for character interaction. Without this, all agents
    remain "remote" and never participate in the story.
    """
```

### How It Works

**1. Director Analyzes Narrative**
```python
prompt = f"""Analyze this narrative and determine which characters are present or nearby.

NARRATIVE:
{narrative}

AVAILABLE CHARACTERS:
- Elara (romantic interest): A mysterious elf from the library
- Seraphine (ally): A fellow scholarship student
- Elysia (romantic interest): A noble who sympathizes with commoners
...

IMPORTANT:
- Match narrative descriptions to character names
  (e.g., "a group of female students" might be Elara, Seraphine, Elysia)
- If narrative describes people generically but existing characters fit, include them
- If player approaches a character, that character becomes present
- Be generous - if unsure whether someone is present, include them
"""
```

**2. LLM Returns Updated Scene**
```json
{
  "present_characters": ["Dan", "Elara", "Seraphine", "Elysia"],
  "nearby_characters": ["Lord Varric"],
  "where": "The Collegium's grand courtyard",
  ...
}
```

**3. Debug Output Shows Changes**
```
[DEBUG] 🎬 Scene update:
[DEBUG]    + Present: Elara, Seraphine, Elysia
[DEBUG]    ~ Nearby: Lord Varric
```

**4. Next Turn: Characters Can Respond!**
```
[DEBUG] 🎬 Director routing: 4 character(s) evaluated
[DEBUG]    ✓ Elara (full) - Present in scene, directly addressed
[DEBUG]    ✓ Seraphine (full) - Present in scene
[DEBUG]    ✓ Elysia (full) - Present in scene
[DEBUG]    ✓ Lord Varric (peripheral) - Nearby, observing

[DEBUG] 📨 Querying agent: Elara [agent_f2b8ba44] (attention: full)
[DEBUG] 📨 Querying agent: Seraphine [agent_c0735076] (attention: full)
[DEBUG] 📨 Querying agent: Elysia [agent_d416051e] (attention: full)
```

### Integration Points

**Called After Every Narrative** (`core/engine/orchestrator.py:283-289`):
```python
# 4. Storyteller composes final narrative
story_output = await self.storyteller.compose_narrative(...)

# 5. Update scene based on narrative (CRITICAL!)
self.current_scene = await self.director.update_scene_from_narrative(
    narrative=story_output.narrative,
    user_action=user_input,
    current_scene=self.current_scene,
    available_agents=self.agent_manager.agent_states,
)
```

**Also Called After Opening Scene** (`core/engine/orchestrator.py:123-129`):
```python
# Compose opening narrative
opening_output = await self.storyteller.compose_opening(...)

# Update scene based on opening narrative (add any characters mentioned)
self.current_scene = await self.director.update_scene_from_narrative(
    narrative=opening_output.narrative,
    user_action="Story begins",
    current_scene=self.current_scene,
    available_agents=self.agent_manager.agent_states,
)
```

## Expected Behavior Now

### Opening Scene
```
Dawn spills across the marble plaza... A group of students stand nearby,
including a silver-haired elf and a nervous scholarship student...

[DEBUG] 🎬 Scene update:
[DEBUG]    + Present: Elara, Seraphine
```

### Player Interaction
```
>: Hi, what are your names?

[DEBUG] 🎬 Director routing: 2 character(s) evaluated
[DEBUG]    ✓ Elara (full) - Present, directly addressed
[DEBUG]    ✓ Seraphine (full) - Present, listening

[DEBUG] 📨 Querying agent: Elara [agent_f2b8ba44] (attention: full)
[DEBUG] 📨 Querying agent: Seraphine [agent_c0735076] (attention: full)

[DEBUG] 🎬 Director validation: 2 accepted, 0 rejected
[DEBUG]    ✓ Elara: Introduce myself warmly | Says: "I'm Elara. We met at the library, remember?"
[DEBUG]    ✓ Seraphine: Greet the newcomer | Says: "Seraphine. Fellow scholarship student."

[DEBUG] 📝 Using 2 agent response(s) in narrative:
[DEBUG]    - Elara: Introduce myself warmly (dialogue: "I'm Elara...")
[DEBUG]    - Seraphine: Greet the newcomer (dialogue: "Seraphine...")

NARRATIVE:
The silver-haired elf smiles warmly. "I'm Elara. We met at the library,
remember?" A second student, her gown marking her as another scholarship
recipient, nods. "Seraphine. Fellow scholarship student."
```

## Technical Details

### Scene Update Frequency
- **Every turn** after narrative composition
- **Opening scene** after initial narrative
- **~1 extra LLM call per turn** (Director analyzing narrative)

### Performance Impact
- **Acceptable**: Director already makes 2 calls per turn (routing + validation)
- **3rd call** for scene update is necessary for gameplay to work
- **Alternative considered**: Parsing narrative with regex → Too error-prone
- **Token usage**: ~500 tokens for scene update prompt + response

### Matching Logic

The Director intelligently matches descriptions to agents:

**Generic Description → Specific Characters**
```
Narrative: "a group of female students"
Director: "Likely Elara, Seraphine, Elysia based on roles and presence"
Result: present_characters: ["Dan", "Elara", "Seraphine", "Elysia"]
```

**Approached Character → Present**
```
User action: "I approach Elara"
Director: "Elara is now present"
Result: present_characters: ["Dan", "Elara"]
```

**Mentioned But Not Present → Excluded**
```
Narrative: "You remember the mysterious Lord Varric from yesterday's class"
Director: "Mentioned in memory, not physically present"
Result: present_characters: ["Dan"] (Varric NOT added)
```

## Testing

✅ All 30 existing tests pass
✅ No breaking changes to existing functionality
✅ Scene updates correctly tracked in save files
✅ Debug output shows scene changes

## Migration Notes

**Existing Save Files**: Will auto-correct on next turn
- Old saves have frozen scenes with only player
- First turn after this fix will populate the scene correctly
- No manual intervention needed

**New Stories**: Work perfectly from the start
- Opening scene gets populated immediately
- Characters appear in scene from turn 1

## Files Modified

1. **`core/roles/director.py:249-339`**
   - Added `update_scene_from_narrative()` method
   - Analyzes narrative and matches characters to scene
   - Returns updated Scene object

2. **`core/engine/orchestrator.py:283-289`**
   - Call scene update after every turn's narrative

3. **`core/engine/orchestrator.py:123-129`**
   - Call scene update after opening narrative

## Impact

**Before Fix:**
- ❌ Characters silent and unresponsive
- ❌ No agent interaction despite 16 agents spawned
- ❌ No debug output (nothing happening)
- ❌ Game feels dead

**After Fix:**
- ✅ Characters respond with names and dialogue
- ✅ Agents participate based on scene presence
- ✅ Full debug visibility into character tracking
- ✅ Game feels alive and interactive

## Future Enhancements

Potential improvements:
- [ ] Cache scene analysis to reduce duplicate calls
- [ ] Character entrance/exit events ("Elara enters from the library")
- [ ] Proximity tracking (characters moving closer/farther)
- [ ] Emotional proximity (characters paying attention vs ignoring)
- [ ] Scene transition detection (moving to new location)
