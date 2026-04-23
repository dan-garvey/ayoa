# Design Doc: Multi-Agent Narrative Engine

## 1. Overview

This project is a single-user, chat-based narrative engine for interactive fiction. The engine accepts a user's action or dialogue, adjudicates what happens in the world, determines which characters plausibly perceive the event, gathers character-level responses from those observers, and produces a final narrated reply to the user.

The core loop is:

1. User submits input.
2. Narrator interprets the attempted action and world interaction.
3. Discriminator determines who observes what, and may dynamically create new character agents.
4. Observing character agents generate reactions based only on what they plausibly know.
5. Narrator merges the results into a single final response, maintaining physics, time, and narrative consistency while preserving as much user and character agency as possible.
6. State is checkpointed so the story can be resumed from a save file.

The user-facing experience is simple: they type a choice, and they receive what happens next. Internally, the system is structured, inspectable, and resumable.

## 2. Goals

The system must:

* support multi-turn interactive fiction through a chat interface
* preserve information boundaries so agents only know what they plausibly observe
* support a narrator with strong adjudication authority over world rules, physics, time, and final composition
* support a discriminator that decides observation, response eligibility, and dynamic character generation
* support dynamically generated character agents that are rarely removed and more often placed into dormancy
* persist enough structured state to resume exactly from a checkpoint
* support hidden/private character intentions as an optional feature
* support debug modes that expose intermediate outputs
* support streaming of the final user-facing response, with optional structured debug streaming
* support per-role model selection under a common model gateway URL
* support rich system prompting and optional thinking-capable model APIs

## 3. Non-goals for v1

The system does not need to:

* support multiple simultaneous users
* support external tools such as search, shell, code execution, or file editing
* guarantee deterministic replay
* optimize for large-scale distributed deployment
* expose chain-of-thought to the end user
* implement hard real-time latency guarantees

## 4. Core Design Principles

### 4.1 Knowledge must be local

Character agents must only receive the facts they could plausibly observe, infer, or remember. They must not be given hidden state, omniscient transcript access, or other characters' private intentions unless that knowledge has entered their world model legitimately.

### 4.2 Narrator is the world referee

The narrator is not a passive formatter. The narrator adjudicates attempted actions against the world state and rules. If the user attempts the impossible, the narrator resolves the attempt into a plausible outcome rather than blindly endorsing it.

### 4.3 Character agency should survive contact with structure

The narrator should preserve as much user and character agency as possible while still enforcing world logic. Character responses should feel like they come from distinct actors, not like paraphrases squeezed from one throat.

### 4.4 Internal representations should be structured

Intermediate outputs should be JSON-like and machine-readable. Only the final response to the user should default to plain narrative text.

### 4.5 Saves are sacred

Every turn must be resumable from a checkpoint/save file. Checkpoints must contain enough state to continue the story without hidden dependence on process memory.

## 5. High-Level Architecture

The system consists of the following components:

### 5.1 Chat API Service

The entrypoint for the UI. Accepts user input, loads the current checkpoint, orchestrates the turn pipeline, persists the new checkpoint, and returns the final response.

### 5.2 Narrator

The narrator runs in two phases:

* **Narrator Phase 1: Adjudication**

  * interprets user intent
  * resolves attempted actions against world rules
  * advances time and physical causality
  * emits a canonical event description for downstream components

* **Narrator Phase 2: Final Composition**

  * receives character responses and world deltas
  * merges them into a single coherent reply
  * updates world and transcript-facing narrative state
  * produces the final user-facing text

### 5.3 Discriminator

The discriminator consumes the canonical event and decides:

* which characters perceive the event
* what each observer specifically perceives
* which characters are therefore eligible to respond
* whether any new character agents should be created
* whether any existing agents should be marked dormant or culled

The discriminator is the gatekeeper of perception and therefore of knowledge.

### 5.4 Character Agents

Each character agent represents one world actor. An agent receives:

* its own identity and stable character sheet
* its private memory
* the subset of current events it plausibly observes
* limited scene context
* constraints on what it may reveal

An agent outputs:

* public response fragments such as action, dialogue, expression
* optional private intentions / internal state updates
* memory writes relevant to that character

### 5.5 Character Manager

Responsible for:

* maintaining the registry of character agents
* spawning new agents from discriminator requests
* marking agents active, dormant, or culled
* storing character sheets and memory
* providing the minimal context required by each agent

### 5.6 State Store / Checkpoint Manager

Responsible for:

* loading and saving versioned checkpoints
* maintaining transcript, world state, agent state, visibility history, and configuration
* enabling save/load from a portable file

### 5.7 Prompt Manager

Responsible for:

* storing template prompts for narrator, discriminator, and agents
* injecting role-specific instructions, formatting contracts, and state snippets
* versioning prompts so save files can record which prompt set produced a story turn

### 5.8 Model Client

Responsible for:

* calling the shared LLM gateway
* supporting per-role model names
* optionally passing thinking/reasoning configuration if backend supports it
* handling retries, timeouts, and streaming

## 6. Recommended Turn Lifecycle

Each user turn should follow this sequence.

### Step 1: Load checkpoint

Load the latest checkpoint or the checkpoint referenced by the request.

### Step 2: Narrator Phase 1, adjudicate the attempted action

Input:

* user message
* current world state
* short recent transcript summary
* current scene context
* relevant global rules

Output:

* canonical event(s)
* time advancement
* scene deltas
* normalized action outcome candidates
* structured summary of what actually happened in the world

This output is internal and structured.

### Step 3: Discriminator, determine observation and roster changes

Input:

* canonical event(s)
* character registry
* locations / scene graph
* visibility / audibility / proximity rules
* current world state

Output:

* observation map: which agent observes which facts
* response roster: which agents should generate a response
* spawn list for new agents
* dormancy / cull updates

### Step 4: Spawn any new character agents

For each new agent requested by the discriminator:

* generate a character sheet
* initialize private memory
* assign location and scene affinity
* register as active or dormant as appropriate

### Step 5: Fan out to observing agents

For each agent allowed to respond:

* assemble a private prompt packet
* include only their visible facts and their own memory
* request structured output

These calls should run concurrently.

### Step 6: Narrator Phase 2, compose final output

Input:

* canonical event(s)
* agent public outputs
* optional agent private intention updates
* updated world state candidates

Output:

* final narrative text for the user
* committed world state updates
* transcript entry
* optional narrator summary for future context compression

### Step 7: Persist checkpoint

Write a versioned checkpoint containing all committed state for the end of the turn.

### Step 8: Return response

In normal mode, return only the final user-facing text.
In debug mode, also return structured intermediate artifacts.

## 7. Data Model

## 7.1 Session State

```json
{
  "session_id": "uuid",
  "story_id": "string",
  "turn_index": 42,
  "created_at": "timestamp",
  "updated_at": "timestamp",
  "config": {
    "models": {
      "narrator": "claude-sonnet-4-6",
      "discriminator": "claude-sonnet-4-6",
      "agent_default": "claude-sonnet-4-6"
    },
    "debug": false,
    "stream_mode": "final_only"
  }
}
```

## 7.2 World State

```json
{
  "time": {
    "scene_time": "2026-04-10T15:21:00Z",
    "turn_count": 42
  },
  "locations": {
    "current_scene_id": "estate_courtyard",
    "scene_graph": {}
  },
  "facts": [
    "The courtyard is wet from earlier rain.",
    "The main building is made of stone."
  ],
  "physics_ruleset": {
    "strength_limits": "human_baseline",
    "magic_enabled": false
  },
  "global_flags": {}
}
```

## 7.3 Character Record

```json
{
  "character_id": "guard_17",
  "name": "Captain Vero",
  "status": "active",
  "location": "estate_courtyard",
  "public_sheet": {
    "role": "guard captain",
    "traits": ["disciplined", "dry humor"],
    "voice": "clipped and formal"
  },
  "private_state": {
    "goals": ["maintain order"],
    "attitudes": {
      "user": -0.1
    },
    "intentions_enabled": true
  },
  "memory": {
    "episodic": [],
    "summaries": []
  }
}
```

## 7.4 Canonical Event

Produced by Narrator Phase 1.

```json
{
  "event_id": "evt_0042",
  "user_intent": "lift the building with bare hands",
  "world_adjudication": {
    "attempted_action": "user attempts impossible feat",
    "feasible": false,
    "resolved_outcome": "visible failed strain with no structural movement"
  },
  "scene_delta": {
    "time_advanced_seconds": 6
  },
  "observable_facts": [
    "The user braces against the building.",
    "The building does not move.",
    "The user visibly strains."
  ]
}
```

## 7.5 Discriminator Output

```json
{
  "event_id": "evt_0042",
  "observers": [
    {
      "character_id": "guard_17",
      "observation_level": "direct",
      "facts": [
        "The user braces against the building.",
        "The building does not move.",
        "The user visibly strains."
      ],
      "should_respond": true
    }
  ],
  "spawn": [],
  "dormant": [],
  "cull": []
}
```

## 7.6 Character Agent Output

```json
{
  "character_id": "guard_17",
  "public_response": {
    "actions": ["takes one step closer"],
    "dialogue": ["Need a lever, not a miracle."],
    "expression": "one brow lifts"
  },
  "private_updates": {
    "intentions": ["monitor the user more closely"],
    "attitude_delta": {
      "user": -0.05
    }
  },
  "memory_writes": [
    "Observed the user attempt an impossible physical feat and fail."
  ]
}
```

## 7.7 Narrator Final Output

```json
{
  "final_text": "You plant both palms against the stone and drive upward until your arms shake. Nothing gives. Rainwater slicks beneath your boots. Captain Vero steps closer, one brow raised. \"Need a lever, not a miracle.\""
}
```

The narrator emits `final_text` only. The engine builds the
`TranscriptEntry` from the verbatim player input passed into the
dispatcher (`user`) and the rendered prose (`assistant`); see
`compose_pov_render` in `app/engine/narrator.py`.

## 8. Save File Requirements

A save file must contain enough information to resume from a turn boundary with no dependence on hidden in-memory state.

Minimum required contents:

* session metadata
* full transcript or lossless transcript representation
* structured world state
* character registry
* per-character private memory
* visibility history or sufficient event history to preserve knowledge boundaries
* narrator/discriminator/agent config
* prompt version identifiers
* model selection and any reasoning-related settings used
* latest committed turn index
* latest committed checkpoint timestamp

Recommended save format for v1:

* versioned JSON file per checkpoint

Recommended upgrade path:

* optional append-only event log plus periodic snapshots

For v1, prefer simplicity over infrastructure theater. A versioned JSON save file is enough.

### Save File Schema Sketch

```json
{
  "schema_version": "1.0",
  "session": {},
  "world_state": {},
  "characters": [],
  "transcript": [],
  "visibility_log": [],
  "config": {},
  "prompt_versions": {
    "narrator": "v1",
    "discriminator": "v1",
    "agent": "v1"
  }
}
```

## 9. API Design

## 9.1 External Story API

Recommended endpoint:

```http
POST /v1/story/turn
```

Request:

```json
{
  "session_id": "uuid",
  "checkpoint_id": "optional-string",
  "user_input": "I try the locked door.",
  "stream": true,
  "debug": false,
  "debug_flags": {
    "include_discriminator": false,
    "include_agent_outputs": false,
    "include_internal_state_deltas": false
  }
}
```

Response in normal mode:

```json
{
  "session_id": "uuid",
  "checkpoint_id": "ckpt_0043",
  "turn_index": 43,
  "output_text": "You twist the handle. It doesn't budge..."
}
```

Response in debug mode:

```json
{
  "session_id": "uuid",
  "checkpoint_id": "ckpt_0043",
  "turn_index": 43,
  "output_text": "You twist the handle. It doesn't budge...",
  "debug": {
    "canonical_event": {},
    "discriminator": {},
    "agent_outputs": []
  }
}
```

## 9.2 Streaming Behavior

Default streaming behavior should stream only the final narrator-composed response to the user.

For debug mode, allow optional structured event streaming via SSE or chunked JSON lines. Event types may include:

* `narrator_phase_1_complete`
* `discriminator_complete`
* `agent_response_partial`
* `agent_response_complete`
* `narrator_final_partial`
* `checkpoint_saved`

Debug streaming should be opt-in. The normal UX should remain clean and invisible.

## 9.3 Internal Model Client Contract

The engine calls the Anthropic Messages API using role-specific model names via the `anthropic` SDK.

Example base request shape:

```python
client = anthropic.AsyncAnthropic()
response = await client.messages.create(
    model=model_name,               # e.g. "claude-sonnet-4-6"
    system="<system prompt>",
    messages=[{"role": "user", "content": "<assembled prompt>"}],
    max_tokens=1000,
)
```

Wrap this behind a client abstraction:

```python
class LLMClient:
    def complete(self, role: str, messages: list[dict], response_model=None, max_tokens: int | None = None):
        ...
```

`options` should be reserved for future fields such as reasoning/thinking settings, sampling controls, and structured-output hints.

## 10. Prompting Strategy

## 10.1 Narrator Prompt

The narrator prompt should include:

* world rules and physical constraints
* current scene
* recent transcript summary
* relevant world facts
* explicit instruction to preserve agency while adjudicating plausibility
* strict output schema

The narrator must not simply "yes-and" impossible actions. It must convert user attempts into plausible outcomes.

## 10.2 Discriminator Prompt

The discriminator prompt should include:

* canonical event
* character registry with locations and current relevance
* observation rules
* explicit instruction to prevent hidden-state leakage
* authority to request dynamic character creation, dormancy, or culling
* strict output schema

## 10.3 Character Agent Prompt

Each character prompt should include:

* stable character sheet
* private memory summary
* observed facts only
* explicit instruction not to reference unknown information
* output contract for public response and optional private updates

## 10.4 Prompt Versioning

All prompt templates must be versioned and recorded in the checkpoint. That way, when a story goes strange in chapter forty-seven, the forensic lantern has something to shine on.

## 11. Dynamic Character Generation

Because characters can be generated dynamically and are rarely culled, the system should treat the active roster as elastic.

### 11.1 Spawn Flow

When the discriminator returns a spawn request:

1. invoke a character generation routine
2. create a new character sheet
3. assign initial goals, voice, and location
4. initialize private memory
5. register the agent as active or dormant

### 11.2 Dormancy vs Culling

Prefer dormancy over deletion.

* **Dormant** means retained in the save file, not considered for response unless reactivated
* **Culled** means removed from active consideration and optionally archived

This prevents unnecessary loss of continuity.

### 11.3 Character Genesis Schema

```json
{
  "spawn": [
    {
      "character_id": "stablehand_03",
      "seed": {
        "role": "stablehand",
        "reason_for_presence": "heard shouting from the courtyard",
        "location": "stable_yard"
      }
    }
  ]
}
```

## 12. Hidden Intentions

Private intentions are optional but supported.

If enabled for a character, the agent may emit:

* current short-term intention
* goal shifts
* suspicion / trust / fear deltas
* memory writes not exposed to the user

These fields must not be included in prompts for other characters unless they become externally observable in a later turn.

This gives the system a pressure chamber under the floorboards without letting steam leak through the floor.

## 13. Knowledge Isolation and Leakage Prevention

The implementation must prevent hidden-state leakage by construction, not by hopeful prompt wording alone.

Required safeguards:

* do not pass full transcript to character agents by default
* construct per-agent observation packets
* separate public and private state stores
* record visibility history so future memories only derive from actual perception
* validate agent outputs for references to unknown facts
* drop or repair invalid outputs before final composition

Recommended validation pass:

* extract entities and claims from each agent output
* compare against agent-visible facts and memory
* flag impossible references
* either redact, retry, or let narrator ignore invalid fragments

## 14. Context Management

Context will grow like ivy if left unchecked. The system should separate context into layers.

### 14.1 Global Context

Stable world rules and story setup.

### 14.2 Scene Context

Current location, local actors, recent events, active tensions.

### 14.3 Character Context

Character sheet, private memory summary, last directly relevant events.

### 14.4 Transcript Compression

Older transcript content should be periodically summarized into structured summaries. The raw transcript may still be kept in the save file, but prompts should consume compressed context.

Recommended rule:

* keep recent N turns verbatim
* summarize earlier turns into role-specific memory/state summaries

## 15. Error Handling

The system should fail like a careful stagehand, not like a chandelier.

### 15.1 Model Call Failure

If a model call fails:

* retry with bounded exponential backoff
* if a character agent fails, continue if possible and mark the agent output missing
* if narrator or discriminator fails, abort the turn and do not commit checkpoint

### 15.2 Invalid Structured Output

If any component returns invalid JSON or schema mismatch:

* attempt one repair pass
* if repair fails, retry once with stronger formatting instruction
* if still invalid, fail the turn or degrade gracefully depending on role criticality

### 15.3 Partial Agent Failure

The narrator should be able to compose a final response even if some non-critical agents fail, as long as core world adjudication succeeded.

## 16. Observability and Debuggability

The system should support a clean normal mode and a lantern-under-the-floorboards debug mode.

Must support:

* per-role latency
* model used per call
* prompt version IDs
* token counts if available
* raw structured outputs for narrator, discriminator, agents
* visibility map for a turn
* checkpoint diff summary

Recommended debug artifact per turn:

```json
{
  "turn_index": 43,
  "latency_ms": {
    "narrator_phase_1": 1900,
    "discriminator": 900,
    "agents_total": 3400,
    "narrator_phase_2": 2100
  },
  "models": {
    "narrator": "claude-sonnet-4-6",
    "discriminator": "claude-sonnet-4-6",
    "agent_default": "claude-sonnet-4-6"
  }
}
```

Do not expose chain-of-thought by default. Persist structured outputs, summaries, and decision artifacts instead.

## 17. Concurrency and Performance

Since this is a single-user demo, keep the implementation simple but parallel where it matters.

Recommendations:

* narrator and discriminator run sequentially
* observing character agents run concurrently
* final narrator composition runs after all agent responses or a timeout
* impose a maximum number of responding agents per turn to prevent latency blowups

Suggested v1 defaults:

* max responding agents: 4 to 8
* max spawn per turn: 1 to 3
* agent timeout: bounded and shorter than narrator timeout

If too many characters can observe a scene, use the discriminator to pick the most relevant responders.

## 18. Suggested Repo Structure

```text
project/
  app/
    api/
      story_routes.py
    engine/
      orchestrator.py
      narrator.py
      discriminator.py
      character_manager.py
      checkpoint_manager.py
      prompt_manager.py
      validators.py
      context_builder.py
    llm/
      client.py
      models.py
    schemas/
      requests.py
      responses.py
      state.py
      events.py
      characters.py
    storage/
      saves/
      transcripts/
    prompts/
      narrator_v1.txt
      discriminator_v1.txt
      agent_v1.txt
      character_gen_v1.txt
    tests/
      test_turn_flow.py
      test_visibility.py
      test_checkpoint_resume.py
      test_agent_leakage.py
  ui/
    chat_demo.py
  README.md
```

## 19. Orchestration Pseudocode

```python
def process_turn(request):
    state = checkpoint_manager.load(request.session_id, request.checkpoint_id)

    canonical_event = narrator.phase_1(
        user_input=request.user_input,
        world_state=state.world_state,
        transcript=context_builder.recent_transcript(state),
        scene_context=context_builder.scene_context(state),
    )

    discrim = discriminator.run(
        canonical_event=canonical_event,
        characters=state.characters,
        world_state=state.world_state,
    )

    character_manager.apply_roster_updates(
        state=state,
        spawn=discrim.spawn,
        dormant=discrim.dormant,
        cull=discrim.cull,
    )

    agent_futures = []
    for obs in discrim.observers:
        if obs.should_respond:
            prompt_packet = context_builder.character_packet(
                state=state,
                character_id=obs.character_id,
                observed_facts=obs.facts,
            )
            agent_futures.append(run_character_agent_async(prompt_packet))

    agent_outputs = gather_with_timeouts(agent_futures)

    validated_outputs = validators.filter_agent_outputs(
        agent_outputs=agent_outputs,
        visibility_map=discrim.observers,
        state=state,
    )

    final = narrator.phase_2(
        canonical_event=canonical_event,
        agent_outputs=validated_outputs,
        world_state=state.world_state,
    )

    state = apply_final_updates(state, final, canonical_event, discrim, validated_outputs)

    checkpoint_id = checkpoint_manager.save(state)

    return build_response(
        final_text=final.final_text,
        checkpoint_id=checkpoint_id,
        debug=build_debug_payload(...) if request.debug else None,
        stream=request.stream
    )
```

## 20. Implementation Plan

### Phase 1: Skeleton

* create project structure
* define schemas
* implement model client
* implement save/load manager
* implement basic `/v1/story/turn` route

### Phase 2: Narrator + Discriminator Loop

* implement narrator phase 1
* implement discriminator
* implement narrator phase 2
* wire a single hardcoded character roster

### Phase 3: Dynamic Character System

* implement character manager
* implement character spawning
* implement dormancy/culling mechanics
* add optional private intentions

### Phase 4: Knowledge Isolation

* build visibility map pipeline
* implement per-agent context builder
* add leakage validator tests

### Phase 5: Streaming + Debug

* add final-text streaming
* add opt-in debug payloads
* add structured debug event streaming

### Phase 6: Prompt Refinement + Compression

* prompt versioning
* context summarization
* transcript compression
* better scene-memory shaping

## 21. Acceptance Criteria

The v1 system is acceptable when all of the following are true:

1. A user can submit a turn and receive a coherent narrated response.
2. Impossible actions are plausibly adjudicated rather than blindly accepted.
3. Only characters who plausibly observe an event are allowed to respond.
4. Character agents do not reference hidden information.
5. New character agents can be spawned dynamically by discriminator decision.
6. A full story session can be saved and resumed from a checkpoint file.
7. Normal mode returns only the final response.
8. Debug mode exposes canonical event, discriminator output, and character outputs.
9. The system supports per-role model configuration through the shared gateway.
10. The pipeline continues gracefully when a non-critical character agent fails.

## 22. Recommended Defaults

For v1, I recommend:

* Python backend
* FastAPI for the story API
* Pydantic for schemas
* asyncio for agent fan-out
* JSON save files for checkpoints
* per-turn save-on-success
* one narrator model, one discriminator model, one default character model, all configurable
* strict internal JSON contracts between components
* no raw chain-of-thought persistence

## 23. One Open Choice I Would Preserve

The biggest design seam worth keeping is this:

Should private character intentions be continuously updated every turn, or only when a character actually responds?

My recommendation for v1 is:

* update private intentions only for characters that respond or are directly involved in the event
* allow dormant/background intention drift later if needed

That keeps complexity from blooming into a vine jungle too early.
