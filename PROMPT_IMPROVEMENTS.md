# System Prompt Improvements

**Date**: 2025-10-23
**Status**: Implemented and Tested

## Overview

Updated all three core system prompts (Storyteller, Director, Character) to be more robust and aligned with the omniscient Storyteller architecture.

## Key Changes

### 1. Storyteller Prompt (`core/prompts/storyteller.txt`)

**Size**: 424 words (was ~150 words)

**Major Additions**:
- ✅ **Omniscient Memory Guidance**: Explains that Storyteller maintains complete memory of all events
- ✅ **World Context Authority**: References comprehensive world context (lore, rules, factions, facts)
- ✅ **Ephemeral Character Moves**: Clarifies that character moves are used once per turn, not stored
- ✅ **Tone & Genre Adherence**: Detailed guidance on maintaining consistent tone (dark/grim, witty, epic, intimate)
- ✅ **Genre Conventions**: Specific guidance for mystery, horror, romance, political intrigue, action
- ✅ **World Authority**: Clear instructions on handling contradictions and conflicts
- ✅ **Composition Guidelines**: Enhanced sensory detail and pacing instructions
- ✅ **Constraints**: Explicit "what you may not do" section

**Why This Matters**:
- Storyteller now understands its omniscient role vs character agents' limited knowledge
- Explicit tone enforcement prevents genre drift (e.g., comedy appearing in grim story)
- World consistency maintained through established facts
- Better handling of character move conflicts

### 2. Director Prompt (`core/prompts/director.txt`)

**Size**: 317 words (was ~100 words)

**Major Additions**:
- ✅ **Physical Factors**: Clear present/nearby/remote distinction
- ✅ **Special Cases**: Eavesdropping, scrying, spies, attention levels
- ✅ **Acceptance/Rejection Criteria**: Explicit checklists with ✓/✗ symbols
- ✅ **Conflict Resolution**: Priority system (faster > attacker > defender > speaker)
- ✅ **Empty Response Handling**: "Silence is valid" - don't reject no_action
- ✅ **NPC Requirements**: Be specific ("Innkeeper places ale" not "someone reacts")
- ✅ **Dramatic Guidance**: Accept creative solutions, note pacing issues

**Why This Matters**:
- Clear routing logic prevents characters knowing things they shouldn't
- Conflict resolution rules create consistent combat/action outcomes
- Validates silence as legitimate choice (was ambiguous before)
- Better NPC integration with specific action requirements

### 3. Character Prompt (`core/prompts/character.txt`)

**Size**: 420 words (was ~130 words)

**Major Additions**:
- ✅ **Memory Limitations**: "You remember last 20 turns clearly" - earlier events vague
- ✅ **Fallible Knowledge**: "Your beliefs may be WRONG" - emphasizes limited perspective
- ✅ **Goal-Driven Behavior**: Explicit section on working toward goals
- ✅ **Decision-Making Process**: 5-step checklist for character responses
- ✅ **Authenticity**: Personality consistency ("coward doesn't suddenly become brave")
- ✅ **When to Stay Silent**: Legitimate reasons for no_action
- ✅ **Speech Style Details**: Vocabulary, sentence structure, mannerisms
- ✅ **JSON Format Examples**: Clear response structure with examples

**Why This Matters**:
- Characters now understand their memory is limited (vs Storyteller's omniscience)
- Goal-driven behavior creates more consistent, purposeful characters
- Authenticity guidance prevents out-of-character behavior
- Silence is legitimized - characters don't have to respond to everything

## Testing Results

✅ All 30 existing tests pass
✅ Prompt loading verified
✅ All key concepts present in prompts

## Before/After Comparison

| Role | Before | After | Change |
|------|--------|-------|--------|
| Storyteller | 150 words | 424 words | +183% |
| Director | 100 words | 317 words | +217% |
| Character | 130 words | 420 words | +223% |

## Implementation Details

**Files Modified**:
- `core/prompts/storyteller.txt`
- `core/prompts/director.txt`
- `core/prompts/character.txt`

**Architecture Alignment**:
- Prompts now reflect the omniscient Storyteller + isolated Character Agent architecture
- Character moves treated as ephemeral (used once, not stored)
- World context referenced as source of truth
- Memory limitations explicit for character agents (20 turns)

## Expected Improvements

1. **Narrative Coherence**: Storyteller's omniscient memory + world context enforcement
2. **Character Consistency**: Goal-driven behavior + personality adherence
3. **Tone Maintenance**: Explicit genre/tone guidelines prevent drift
4. **Information Routing**: Clear present/nearby/remote logic
5. **Conflict Resolution**: Consistent priority system
6. **Silence Handling**: Characters can legitimately choose not to act
7. **World Consistency**: Established facts respected, contradictions handled

## Next Steps

Recommended additional enhancements:
- [ ] Add error recovery guidance (JSON parsing failures, world conflicts)
- [ ] Create tone/genre exemplars for common genres
- [ ] Add combat/action-specific guidance
- [ ] Develop romance/relationship handling guidelines
- [ ] Create "edge case" handling documentation

## Usage Notes

The prompts are automatically loaded when initializing roles:
```python
storyteller = Storyteller()  # Loads core/prompts/storyteller.txt
director = Director()        # Loads core/prompts/director.txt
character_agent = CharacterAgent(...)  # Uses core/prompts/character.txt
```

No code changes required - prompts are loaded at runtime.
