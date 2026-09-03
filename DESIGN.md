# Design Doc: Ayoa Narrative Engine

## Repository And Setup

### Dev Environment

* Use `.venv/bin/python` and `.venv/bin/pytest` directly rather than
  sourcing the virtualenv activate script.
* Provider API keys belong in `.env`, never committed. The Anthropic
  client reads `ANTHROPIC_API_KEY`; the OpenAI client reads
  `OPENAI_API_KEY` plus role-specific OpenAI key aliases such as
  `OPEN_AI_ROUTER`. Startup and live-playtest preflights fail fast on
  missing credentials for the configured live roles.
* The LLM client is multi-provider (Anthropic Messages API and OpenAI
  Responses API) with per-role provider/model dispatch. Default models
  are `gpt-5.6-terra` for `event_router` and `narrator`; `gpt-5.6-luna`
  for every character-agent tier and `character_manager`; and `gpt-5-mini`
  for `dnd_combat_manager` and `content_manager`. Per-role
  overrides go through `LLM_PROVIDER_<ROLE>` and `LLM_MODEL_<ROLE>`
  environment variables, or via the `LLM_ROLE_PROVIDERS` /
  `LLM_ROLE_MODELS` JSON env maps. A `provider:model` prefix on a
  model string (for example `anthropic:claude-sonnet-5`) also works.
  Configured OpenAI roles use medium reasoning by default, except Luna
  models use max; explicit global or per-role reasoning overrides win.
  Normal D&D playtests should keep `dnd_combat_manager` on the default
  OpenAI `gpt-5-mini` path; if its preflight fails, configure the
  missing OpenAI role key rather than downgrading the role.

### Code Layout

* `app/engine/` — turn pipeline: orchestrator, event-router dispatch,
  narrator, character agents, character manager, context builders,
  settings, prompt manager, turn-loop contracts.
* `app/schemas/` — Pydantic data models (checkpoint, session state,
  characters, events, agents, requests, responses, conversation, etc.).
* `app/llm/` — multi-provider LLM client wrapper (Anthropic + OpenAI),
  per-role provider/model dispatch, prompt caching, conditional
  compaction, structured output normalization.
* `app/prompts/` — prompt templates (`event_router.txt`, `agent_turn.txt`,
  `agent_perception.txt`, `narrator_phase2.txt`, `character_gen.txt`, `takeover.txt`, plus
  ruleset addons such as `agent_ruleset_dnd5e.txt` and
  `dnd_cat_ii_router.txt`). Prompt history lives in git, not in
  filename suffixes; use `git log <file>` to read the version history.
* `app/bot/` — Discord frontend: slash commands, `EngineBridge`,
  `SessionMap`, embed rendering, `__main__` startup.
* `app/storage/sessions/` — per-session checkpoint directories.
* `app/storage/stories/` — synthetic story seed checkpoints. This directory
  defaults to ignored so local/runtime story drafts do not appear
  accidentally; shipped seed directories must be explicitly allowlisted in
  `.gitignore`.
* `app/storage/story_templates/` — authoring templates and notes for
  coding agents creating synthetic story seeds.
* `scripts/play.py` — interactive terminal REPL frontend, supports
  multi-character play.
* `scripts/image_worker.py` — optional isolated local image-generation worker,
  GPU preflight, weight download, smoke test, and benchmark entrypoint.
* `tests/` — pytest tests.

## 1. Status

This document describes the current v11 architecture. It is no longer a
v1 proposal. The implementation is a Discord-first, checkpointed,
multi-character interactive fiction engine whose core runtime lives in
`app/engine/orchestrator.py`, `app/engine/turn_loop.py`, and
`app/engine/turn_loop_dispatcher.py`.

The old "Narrator Phase 1 + Discriminator + Narrator Phase 2" design has
been collapsed. The current loop is:

1. A human submits an action for a bound character.
2. The event router classifies and canonicalizes the intention.
3. The turn loop broadcasts the canonical event to human render buffers
   and NPC observation inboxes.
4. If the router keeps the beat live, the turn loop projects the router
   output into dispatch targets. Current runtime dispatches one NPC target
   at a time and routes that public result back through the router before
   another same-scene target can act. Human render targets are inferred from
   terminal router events and observing player-bound characters.
   This loops until `event_kind` says to continue no further or the
   engine hits a hard cap.
5. The narrator renders each observing human's POV from visible
   `observable_facts` only.
6. Spawns, dormancy, culls, per-POV narrator history updates, and
   checkpoint saving are applied around the beat.

The system is intentionally prompt-heavy. Prompt files under
`app/prompts/` are versioned source artifacts. Prompt history is tracked
by git, not by filename suffixes.

The key architectural truth is that Ayoa is currently **router-centered**.
The event router is the world arbiter. Character agents propose
character-local behavior; the narrator renders human POV prose; the
orchestrator and turn loop apply router outputs to checkpoint state. Future
mechanics/content extensions should be understood as narrow subflows that
relieve a specific router responsibility, not as replacements for the current
router-centered runtime.

## 2. Goals

For future product direction, see `GOALS.md`. This design document is the
current-runtime handoff and should be treated as authoritative when it
conflicts with aspirational planning notes.

The engine must:

* support multi-turn interactive fiction in Discord
* support multiple human players bound to different characters
* preserve information boundaries by construction
* keep the player's input and physical agency intact
* let NPC agents carry private continuity without leaking it to other roles
* model shared perception, not just shared physical location
* support contested actions without deciding another actor's response early
* persist every committed state change in portable JSON checkpoints
* provide enough logs and checkpoint artifacts to investigate playtest bugs
* keep prompt context small enough for long-running sessions

## 3. Non-goals

The current engine does not attempt to:

* expose a public HTTP story API
* stream final prose to Discord token-by-token
* expose raw chain-of-thought or thinking blocks
* guarantee deterministic replay across LLM calls
* run external tools from inside story agents
* replace the router with an always-on generic rules arbitrator or content
  resolver; D&D combat and content lookup exist as narrow adapter subflows
* maintain a second global player-history surface alongside the canonical
  per-POV narrator conversations

## 4. Design Principles

### 4.1 Facts Are The Contract

The router's `canonical_event.observable_facts` are the authoritative
surface of what happened. The narrator renders those facts. NPCs receive
their visible subset in `pending_observations`. If a detail must be known
by someone, it must exist as an observable fact visible to that character.

### 4.2 Perception Is Contextual

`CharacterRecord.location` is an opaque continuity label, not routing
authority. The router decides who perceives each event from live context:
co-presence, live audio, cameras, radio, telepathy, scrying,
supernatural senses, spies, or any other established channel.

There is no engine-side scene graph and no local/remote observer flag in the
live event model. `broadcast_event()` trusts the router's `observers` list and
then filters by fact-level visibility. In production today, `all_observers`
facts are visible to every listed observer; a remote or mediated character
should only be listed as an observer when the router intends them to receive
the broad facts, or should receive scoped `audience="only"` facts instead.

### 4.3 The Narrator Is A POV Renderer

The narrator is no longer the world adjudicator. It is a per-POV prose
renderer. It receives visible facts, observation levels, relevant
context, the acting player's original input, and the POV's rolling
narrator history. It must not invent actions, dialogue, outcomes,
interiority, or physical business that is not in the visible facts or the
player's input.

### 4.4 The Router Owns Canonicalization

The event router is the adjudication, perception, and beat-pacing
role. It decides:

* Cat I vs Cat II classification
* feasibility
* time advancement
* observable facts and fact-level visibility
* observers and observer routing roles
* beat end state
* spawns, dormancy, and culls

The router also handles router-shaped special entries: `(begin)`, `(arrive)`,
`(query: ...)`, narrator-requested continuation, and Cat II resolution blocks. Directly
supplied and character-agent-authored prose share one proposed actor-submission
framing over the same `EventRouterOutput` schema; neither source pre-commits
fiction.

An `Authoritative Result` is the narrow exception for an outcome already fixed
by a fictional or reviewed rules authority. The router canonicalizes its
surface facts and perception but does not adjudicate it again; runtime attaches
the prevalidated side effects and closes the event without a response frontier.

### 4.5 Agents Author Intentions, Not State

Character agents produce one free-form observable contribution or explicit
silence. The contribution becomes the next intention that the router
canonicalizes. There is no hidden turn summary or parallel intent channel:
continuity comes from the actor's full causal history, durable authored private
state, and later witnessed consequences.

Agents do not write canonical events, mutate roster state, move characters,
or send hidden directives. If an NPC's private plan should affect the world, it
must surface through the agent's public action, a router-selected background
turn, or a later observable fact the router canonicalizes.

This asymmetry is structural, not stylistic. Collapsing it back into a
shared schema field — a `last_intent` mirror on `CharacterRecord`, an
agent's hidden summary piped into the router, narrator, or another
agent's context — recovers the worse single-LLM-pretending-to-be-many
shape that per-actor calls were designed to avoid. Cross-actor signal
must surface through in-fiction events the router canonicalizes (a
courier walks in, a witness sees an action, a note is found), not
through hidden side channels.

### 4.6 Checkpoints Are The Source Of Truth

Every committed turn writes a checkpoint. The runtime can be rebuilt from
checkpoint JSON plus the prompt/code version in git. Process memory,
locks, and API caches are not trusted as durable state.

### 4.7 Rules Adapters Are Modular

Domain-specific rule systems are
modular adapters around the narrative engine, not assumptions baked
into router, narrator, or character-agent behavior. Every change to
the engine's generic surface must survive two questions:

1. Does this preserve the original rules-neutral narrative engine?
2. Does this avoid adding prompt or runtime machinery that is useless
   in non-D&D narrative contexts?

When an adapter needs a hook into the core (a settings flag, a
checkpoint field, a prompt addon, or a ruleset-specific adjudication
path), the hook ships with a non-adapter default that keeps the
narrative engine unchanged. Adapter-specific schema fields default
to empty; adapter-specific prompts only render when the matching
ruleset is active; adapter-specific bot commands no-op or reject
politely outside the matching ruleset; the core router, narrator,
and character-agent prompts stay rules-neutral with adapter
behavior delivered through addons rather than baked into the base
templates. The full current adapter surface lives in §15.

### 4.8 Runtime LLMs Never Process Images

No live model role in Ayoa processes images. Router, narrator,
character-agent, takeover, convenience, and rules-adapter calls receive
text and structured data only. Vision/image-understanding calls are
expensive, hard to cache, and outside the project's runtime scope.

Image-bearing source material is handled at authoring/import time. If a
D&D module includes a scanned page, combat map, illustration, symbol,
handout, or any other image-only information that matters to play, a
coding agent manually inspects that image during import and authors the
resulting checkpoint data, map geometry, room descriptions, asset
metadata, labels, secret notes, or tactical representation. Runtime
models then consume those authored artifacts; they never receive the
source image or a request to infer from it.

Player-facing image display is presentation only. Discord or CLI views may
reveal reviewed imported assets to players. Sessions default to
`presentation_mode="prose"`, which performs no event illustration work. The
opt-in `visual_novel` mode gives each accepted POV render an ordered semantic
page deck and one explicit noncanonical stage transition: `reuse`, `replace`,
or `clear`. Only `reuse` can inherit a prior successful stage. A failed
replacement and `clear` both resolve to the deterministic neutral stage rather
than silently showing stale fiction.

The event-driven visual-novel sidecar projects only router-authored visible
facts, public character metadata, and opaque reviewed-reference handles with
authored applicability hints for equivalent human observers. A dedicated
image-director role may reuse the current plate, select one suitable reviewed
environment plate as the exact stage, request exactly one new 16:9 scene plate
when the supplied pool does not fit, or clear the stage. Reviewed selection is
a preferred fast path, not a closed set. It remains available when the optional
local image worker is unavailable; only the generated fallback requires that
worker. Direction and generation may overlap narration, but the stage
transition becomes eligible only with the accepted, committed POV render. Raw
generated scene images are never delivered independently; the shared
deterministic compositor cover-crops the selected immutable plate and adds the
classic ADV text box. Explicit portrait identity review remains a separate
raw-image workflow.

The same compositor has one optional, rules-neutral foreground layer between
the immutable stage and the ADV text box. A deck section may carry zero, one,
or two already-resolved transparent PNG sprites as an ordered placement list.
Each placement binds opaque identity and variant provenance, the exact source
hash and dimensions, a left/center/right slot, source and target facing,
bottom-center anchor, and integer scale. Two-character sections use distinct
left and right slots; a later page changes pose or expression by starting a new
section against the same stage with another resolved variant. The compositor
never accepts a source path, looks at an image through an LLM, chooses a
character or expression, or generates an asset. It validates immutable bytes,
alpha-composites the placements in order, then draws the dialogue UI last.
Sprite provenance and transforms participate in the content-addressed deck
identity, while CLI and Discord continue to transport only the same
manifest-verified final card bytes.

The narrator may attach zero, one, or two bounded expression cues to an
individual page, choosing only directly present characters from its supplied
safe-name roster. Those cues are semantic requests, not image references. A
private resolver maps them to immutable reviewed variants or already-generated
runtime variants, validates the selected PNG bytes and hash, and supplies only
the resolved opaque provenance and bytes to the compositor. Missing requested
expressions fall back to that character's neutral variant without delaying the
story. The stage director remains solely responsible for the unoccupied
environment layer; it never receives character appearances or references and
cannot paint a second copy of the cast into the background.

Generated sprite packs follow the same resolver contract as reviewed packs.
The runtime first generates and hash-locks one full-body neutral identity on a
saturated magenta screen. It deterministically removes the border-connected
key plus bounded near-exact screen islands enclosed by arms, hair, capes, or
props, using combined red-and-blue dominance and physical unmixing at
partial-alpha edges so red clothing and skin are preserved. It then normalizes
the transparent subject to the shared 1100x1500 frame and baseline and rejects
a candidate if a material amount of visible key color remains.
The remaining seven universal expressions are independent compose requests
that use that immutable neutral PNG as their sole identity reference. Their
prompts require one stable body, outfit, hairstyle, and complete prop set while
allowing a distinct pose and expression. This path does not use image editing
or runtime image understanding; bytes are processed and validated by code, and
failed variants simply remain unavailable.

One-Star's promotion reveal is an adapter-owned projection over this generic
contract. From the disembodied Master's viewpoint, birth-one-star Heroes use a
stable authored masculine or feminine veiled set at one star. Heroes with a
reviewed sprite set reveal that exact set at two stars. Heroes without reviewed
art remain veiled until three stars and begin warming their generated pack one
promotion earlier, whether their character record was authored or generated.
Other viewpoints retain their ordinary textual identity.
While veiled, Master-facing first-look context is generic and cannot disclose
the hidden face or loadout; crossing the reveal threshold reopens the exact
first-look ledger. These rank rules never enter the generic narrator,
compositor, or image-generation schema.

Every materialized director run records exactly one replacement source: a
hash-pinned snapshot of the selected reviewed plate, or the exact set of
diffusion jobs it admitted. Stage readiness, prior-stage context, and final
plate resolution use that run-owned provenance rather than matching only on
event, transaction, or the location pool's current bindings. A run moves
through an explicit `materializing` state until selection or admission is
finalized; only then does it become `succeeded`. A direct reviewed replacement
admits no diffusion job. A generated replacement with no admitted job resolves
to the neutral stage, while a partial admission error fails the run and cancels
its otherwise-unowned jobs. This keeps sibling private POV projections from
observing or waiting on one another's artifacts and preserves exact reuse after
restart or later pool changes.
Admissions are provisional and attempt-scoped before diffusion begins. Every
heartbeat, completion, finalization, failure, cleanup, and active-worker abort
is fenced by the claimed attempt, so an expired worker cannot mutate a reclaimed
run or cancel a sibling's shared job. The durable claim query also holds a later
same-session event position behind any earlier running or materializing
position across process/store handles; lease recovery runs on every queue cycle,
not only when the queue becomes idle. On restart, an unavailable or failed-
preflight process probes the real exclusive queue-owner lease at low cadence.
If another capable process holds it, finalized queued work remains available to
that owner. Once no process holds it, the otherwise-unserviceable linked jobs
are failed atomically so the render barrier reaches a terminal neutral stage
rather than waiting forever; the unavailable process never retains the lease or
starts generation.

Generated illustrations are noncanonical output artifacts and are never
evidence for story state. Imported images retain their manual review and
spoiler/privacy gate. Generated images and frozen reviewed references use
separate runtime provenance with byte, format, dimension, root, and hash
validation. Direct reviewed stages are revalidated when their bytes are
resolved for composition. The image director may receive opaque reviewed-
reference handles
with human-authored, public selection hints so it can choose useful views, but
never receives the image, storage path, hash, dimensions, or runtime-derived
analysis. Image bytes, embeddings, captions derived from images, job
records, and visual output never enter an LLM message, canonical event,
narrator history, or character-agent history. A checkpoint may retain only an
engine-owned identity-reference ID used by the diffusion pipeline; that handle
does not change textual canon.

Authored setting, narrative-rule, and visual context is treated as untrusted
imported text before it reaches any runtime LLM. Control characters, metadata
fields, hashes, asset handles, credential-like filenames, Unix and Windows/UNC
paths, and repository-internal relative paths are removed under the shared
content-privacy policy for narrator, character, and image-director inputs.
Ordinary public HTTP(S) references are preserved rather than mistaken for
filesystem paths.

A generated visual-novel replacement is an unoccupied 16:9 environment plate.
Its subject list is empty, its prompt may not name a roster character, and it
may use only applicable reviewed location references. Character appearances,
identity references, poses, silhouettes, crowds, and body parts are forbidden
from both the director input and the diffusion request. Live binding, file,
byte-count, or hash failure admits no diffusion job, so the generated
replacement resolves through the ordinary neutral-stage failure contract.

When an authored location has reviewed environment plates, a visual-novel
replacement first considers the exact plate whose human-authored applicability
fits the current visible scene. Multiple plates are choices, never a blend. If
none fits, the director leaves the direct-stage handle empty and uses the
ordinary generated replacement path; a location handle may still be an
optional generation guide when its authored use genuinely applies. Location
selection is engine-routed from canonical embodied staging: a directly
observed cast may establish the depicted location for an omit-policy or
remote-interface viewer, while indirect reports, media depictions, future
arrivals, and mediated name mentions stay anchored to the viewer's own
location. Visual-novel projection admits bounded generic visible actions for
this transient location decision; it does not weaken the stricter physical
classifier that persists first-meeting state. A render batch offers only its
final scene's location choices. Higher-tier authored plates may remain validated
but unbound until story progression activates them.

## 5. Runtime Components

### 5.1 Discord Bot And EngineBridge

`app/bot/commands.py` defines the slash-command surface.

Generic narrative engine commands:

* `/session start|resume|list|end`
* `/story list|info|start|characters|delete`
* `/join`, `/begin`, `/leave`, `/character`, `/describe`
* `/act`, `/defer`, `/query`, `/status`, `/settings list|set`
* `/image lock|reroll` — accept or replace a provisional portrait identity
* `/rewind`, `/clear`, `/abort_beat`

D&D adapter commands (active when `ruleset_id == "dnd5e_basic"`; see §15):

* `/attach` — attach a D&D Beyond character snapshot to a player character.
* `/sheet` — display the attached D&D character sheet.
* `/roll` — answer a pending interactive player roll.
* `/xp award` — admin-only D&D experience award to one character or the bound party.
* `/combat begin|status|next|end|damage|heal` — combat lifecycle and HP
  management.

One-Star adapter commands (active when
`ruleset_id == "one_star_ascension"`; see §15):

* `/master status` — account resources, configured discretionary funds,
  facilities, progression, stamina, active mission, and pending-operation state.
* `/master heroes` — owned-Hero roster with core mechanics.
* `/master hero <name|id|#>` — one owned Hero's visible full sheet.
* `/master synthesis <target> from <source>[, <source>...]` — submit a
  synthesis selection through the ordinary canonical turn path.

The bot calls the engine in process through `EngineBridge`; there is no
FastAPI layer in the current runtime. `SessionMap` stores Discord channel
to session mappings, private POV thread mappings, and turn-message refs
used by rewind cleanup.

`EngineBridge` is the frontend boundary for Discord and the CLI REPL. It owns
the live `LLMClient`, `CheckpointManager`, `PromptManager`, and `Orchestrator`;
wraps turns in a per-session frontend lock; runs stale Cat II sweeps before
new `/act` processing; and exposes helper flows for join, begin, arrival,
takeover, leave, settings, query, and rewind. It also owns the optional local
`ImageGenerationCoordinator`, visual-novel event sidecar, and shared
`VisualNovelCardRenderer`. Each finalized event is copied into immutable
observer projections without awaiting presentation work. The sidecar groups
equivalent projections, queues durable image-director runs, and materializes a
replacement direction as either a run-owned reviewed stage snapshot or an
event-provenance diffusion job. Checkpoint save
commits the speculative stage transaction; turn failure or rewind cancels stale
work. GPU inference runs in an isolated subprocess. CLI and Discord both load
the same content-addressed card deck; Discord navigation state lives in
persistent component ids, so previous, next, and transcript controls survive a
bot restart. The frontend lock serializes every checkpoint mutation exposed by
the bridge, including pending-roll continuation and settings changes, in a
fixed bridge-before-orchestrator lock order. Persisted deck manifests are
untrusted restart inputs: supported manifests, card order and metadata, deck
identity, filenames, hashes, PNG format, and 1024x576 dimensions are validated
before a card is served. The deck ID binds both canonical render identity and
the ordered card hashes. Render and reload I/O stays anchored to pinned,
no-follow directory descriptors, with the named root revalidated across the
operation, so an ancestor symlink or directory swap cannot redirect
persistence. A loaded card snapshots its manifest-verified bytes, and CLI and
Discord transport those immutable bytes rather than reopening a mutable path.
Only manifest v2 under the current content-bound renderer identity is accepted,
with exact typed card dimensions, per-card hashes, and an independent generic
source-identifier check. Version 1 and earlier renderer identities predate the
current privacy or integrity contract and are retired rather than migrated or
guessed; unknown versions and renderer identities also fail closed.

### 5.2 Orchestrator

`Orchestrator.process_turn()` is the main turn entry point. The
narrative-engine happy path:

1. loads the latest checkpoint
2. resolves the acting character
3. acquires a per-session lock
4. checks active act slots; rejects, routes as Cat II responder, or proceeds
5. calls `run_beat()`
6. applies character lifecycle changes and spawns
7. preserves each rendered POV in that character's narrator conversation
8. increments the turn index and saves
9. returns a `TurnResponse`

When the D&D adapter is active (`ruleset_id == "dnd5e_basic"`), several
combat-aware branches splice into this flow: a `(defer)` from a
combatant with an open reaction window resolves as a reaction
acknowledgement; `_handle_combat_after_beat` advances initiative state
after the beat closes; router-selected agent turns exclude combatants with
pending reaction prompts so they must answer their reaction window first; and
`_run_automated_combat_turns_locked` drives
NPC combatant turns inline before the player's response returns. The
adapter branches are no-ops outside `dnd5e_basic`. Adapter detail is
collected in §15.

`Orchestrator.resolve_cat_ii()` is the separate pre-turn closeout path for
stale or newly-ready Cat II events. It re-enters the same router resolution
contract, broadcasts the resolved event, renders any resulting POV output, and
saves before the user's new action proceeds.

### 5.3 Turn Loop

`run_beat()` in `app/engine/turn_loop.py` is the v11 state machine. It
supports actor submissions, Cat II responder actions, autonomous cascades,
observation harvest, partial renders for contested attempts, max-event
backstops, and per-human render fan-out. After each closed narrative event,
the narrator judges whether to return control while the selected autonomous
`next_output` is prepared concurrently against isolated checkpoint state. A
render discards that prepared state but leaves the canonical semantic handoff
available: a later player action supersedes it, while `(defer)` resumes it.
`continue` commits its agent memory and router result before the cascade
resumes.

Important state:

* `active_act_slots`: per-beat lock state for initiators, Cat II responders,
  pending D&D player rolls, combat reactions, and D&D combat-start attempts
  blocked behind an already-active initiative ladder
* `open_cat_ii_events`: contested events awaiting responders
* `cat_ii_roll_transactions`: checkpoint-persistent D&D roll plans, pending
  player rolls, completed roll results, and dice ledgers
* `render_buffers`: per-human queues of canonical event ids waiting for
  narrator render
* `canonical_events`: append-only log of closed canonical events

### 5.4 LLMDispatcher

`LLMDispatcher` binds the abstract turn-loop protocol to the live roles:

* `route_intention()` sends every proposed actor submission through the
  `event_router` prompt, whether its text was supplied directly or authored by
  a character agent
* `route_continuation()` asks the router for another grounded event after a
  narrator continuation handoff keeps the visible batch open
* `agent_intend()` calls `CharacterAgent.turn()`
* `harvest_perceptions()` calls `CharacterAgent.perceive()` in parallel
* `narrator_compose()` calls `compose_pov_render()`

It also owns router input shaping: initial roster on the first router call,
the current submitted input, and optional external engine state updates.

The dispatcher snapshots and restores pending engine updates around fresh
router calls so a failed router completion does not silently drain them.

### 5.5 Event Router

The event router prompt and `EventRouterOutput` schema replace both the
old narrator adjudication phase and the old discriminator role. The
router emits one structured object per actor submission or closed repair.

Current router call shapes:

* actor submission (`submitted_actor_id` plus `submission_text`), independent
  of how that text reached the runtime
* Cat II final adjudication (`## Cat II Resolution`)
* narrator-requested continuation (`## Continuation Required`)
* OOC directives such as `(begin)`, `(arrive)`, `(defer)`, and
  `(query: ...)`

The top-level object carries:

* `event_id`
* `decision_rationale`
* `canonical_event`
* Cat I / Cat II fields
* `event_kind`
* `observers`
* `spawn`, `dormant`, `cull`

The nested `canonical_event` deliberately carries only:

* `world_adjudication.feasible`
* `observable_facts`

Outside quoted speech, non-self character references use canonical character
ids. Quoted speech preserves the spoken pronoun and immediately follows it with
a silent bracketed character-id anchor, such as `you [bob]` or `her [alice]`;
one plural pronoun uses one comma-separated anchor such as `you [bob,carol]`.
Character observations retain the player-safe identity anchor; narrator input
deterministically removes it so the spoken grammar reaches prose unchanged.

Legacy audit fields such as `attempted_action` and `resolved_outcome`
are retired. Old checkpoints that still contain them are loaded by
dropping those fields during validation.

### 5.6 Character Agents

Each character has one rolling conversation in
`checkpoint.character_conversations[character_id]`. `agent_turn.txt` handles
foreground, private, and background choices. `agent_perception.txt` is a
separate, narrower exterior-description contract that receives only public
identity and visible loadout, never actor facts.

For a turn, the system message contains only the generic acting, knowledge,
agency, and observable-output contract plus any static ruleset guidance. The
current user packet contains a `<you>` projection with that character's public
identity and sparse second-person actor facts, followed by a `<now>` projection
with witnessed observations, elapsed time, location or immediate circumstance,
and any current rules state. Actor names, actor facts, routing frames, and live
observations never enter the generic system contract.

Each accepted user packet is retained intact in that character's history,
apart from the disposable presentation catalog, and the full per-character
conversation is replayed without provider compaction. The engine parses the
assistant output into:

```json
{
  "character_id": "rashid",
  "public_text": "Rashid sets his glass down. 'Say that again.'",
  "is_silence": false
}
```

The assistant history stores exactly the observable contribution. `<silence/>`
records a chosen observable silence without inventing filler dialogue. Hidden
plans are not synthesized after each response; durable private facts remain in
the character record, while changed relationships and obligations must become
legible through what the actor witnessed and did.

Perception/loadout mode does not accept turn metadata and does not drain
`pending_observations`. It still appends the
perception exchange to the character's rolling history so future agent calls
remember what the character established about their current visual
self-presentation. It is used for observation harvest and private query
enrichment when the engine needs a current visible loadout fragment.

### 5.7 Narrator

`compose_pov_render()` is the only production narrator entry point. It
renders one human POV at a time from render-buffered canonical events.
Events are resolved by id against `checkpoint.canonical_events`, filtered
by fact visibility, and tagged with direct / indirect / inferred
observation level.

Prose-mode narrator output is:

```json
{
  "handoff": "render | continue",
  "handoff_reason": "diagnostic sentence",
  "final_text": "string"
}
```

Visual-novel mode uses a separate typed output:

```json
{
  "handoff": "render | continue",
  "handoff_reason": "diagnostic sentence",
  "pages": [
    {
      "kind": "narration | dialogue",
      "speaker": "required only for dialogue",
      "text": "one semantic ADV page"
    }
  ]
}
```

`handoff_reason` is logged for playtest diagnostics only. It is never forwarded
to the router or treated as canonical event evidence; engine control uses the
typed `handoff` decision.

Narrator composition is faithful semantic compression over the complete
ordered set of facts visible to that POV. Dialogue has the highest preservation
priority, followed by consequential actions, choices, signals, state changes,
outcomes, and causal order. Repeated staging, unchanged ambience, incidental
environmental texture, and inconsequential micro-movements may be fused or
omitted when they add no new information.

A render-buffer reference that cannot be resolved against canonical history is
a loud contract failure, not a skippable stale entry. Likewise, a `render`
handoff requires non-empty prose or at least one semantic page; an empty
draft/deck is permitted only for a discarded `continue` judgment. A forced
render boundary rejects `continue` before image-candidate acceptance, buffer
flush, narrator-history commit, or first-meeting ledger mutation; every POV in
that render batch rolls back together.

Both narrator modes receive optional first-meeting appearance vocabulary in a
separate volatile user-tail block before the submitted attempt and final
authoritative visible result. That block contains only the stable visible
exterior/loadout of a directly observed person. It is not event evidence and
cannot establish presence, a known name, rank, role, faction, relationship,
biography, intent, action, or consequence. The per-viewpoint introduction
ledger advances only when an accepted render directly presents that exterior;
indirect observation and rejected handoffs do not consume it.
Direct-person detection is scoped per subject and predicate: quoted or reported
speech and mediated images, feeds, messages, or recordings cannot consume the
ledger, while a direct speaker, coordinated physical arrival, or bounded
spatial copresence can.

Before a visual-novel deck becomes accessible text or history, every speaker
and page text is checked for exact source ids from the complete checkpoint
roster, including culled records, and for generic underscore-form identifiers.
One rejected deck may be corrected in the same narrator context without
translating the identifier through roster metadata or cosmetic prettification.
That correction must preserve the handoff, page count and kinds, and every
already-safe field exactly; it may change only the unsafe fields. A second
violation or structural rewrite fails loudly, and response assembly and commit
reassert the invariant before exposing output or mutating narrator history or
introduction state.

The engine constructs a transient render record from the real player input and
the deterministic accessible text projection for response assembly. In visual-
novel mode that projection is narration text plus `Speaker: dialogue`, while
the structured pages remain available to the compositor. Durable player
history is the per-character narrator conversation, not a second checkpoint
transcript.

### 5.8 Character Manager

The character manager applies router-directed roster mutations:

* status changes: active, dormant, culled
* LLM-backed spawns from `SpawnRequest`

`culled` is terminal across every ruleset: neither `activate` nor `dormant`
may downgrade a culled record into a reusable character slot.

It also purges live bookkeeping when characters are culled or leave: active
act slots, open Cat II responder state, and render buffers. Movement is not a
roster side effect; arrivals, departures, and transfers must be written as
`observable_facts`.

When one router event requests multiple new characters, the manager first
authors one compact casting plan for the whole requested wave. Every character
branch receives all sibling briefs and the same immutable pre-wave checkpoint,
then the independent authoring calls run concurrently. Results merge in request
order only after every branch succeeds; one failed branch rejects the complete
wave without partially mutating the live roster or its agent histories.

### 5.9 Synthetic Story Seeds

Story authoring now starts from a checkpoint template instead of an LLM
importer. A coding agent fills a synthetic `ckpt_0000.json`, validates it
against `CheckpointFile`, and places it under
`app/storage/stories/<story_id>/ckpt_0000.json`.

Module imports that depend on images follow the same authoring boundary:
the coding agent manually reviews source images at import time and writes
the usable text, structured state, geometry, and asset metadata into the
story/checkpoint artifacts. The runtime never sends source images or
player-visible asset images to an LLM for interpretation.

The authoring template lives at
`app/storage/story_templates/synthetic_checkpoint/ckpt_0000.json`, with
design notes beside it. The template is intentionally outside
`app/storage/stories/` so it does not appear in `/story list`.
New shipped story seeds under `app/storage/stories/` must also be allowlisted
in `.gitignore`; session checkpoints and playtest reports remain ignored
runtime outputs.

The first playable situation is normally composed by the router on `(begin)`
and rendered through the normal narrator path. A seed may optionally provide
`world_state.opening` with router-only authored context and an explicit
`allow_spawns` capability. Missing or false spawn authority preserves the
default that `(begin)` cannot create characters.

Opening dialogue and action use the ordinary router, character-agent, and
narrator contracts. Stories do not carry a second authored-dialogue schema or
post-render opening commit path; `world_state.opening.context` is the one
story-specific opening instruction surface.

### 5.10 Context Builder And Prompt Manager

`context_builder.py` owns shared context formatting and visibility-aware
helper logic. `PromptManager` loads `app/prompts/{name}.txt`, expands
partials, strips HTML comments, splits system/user sections on
`<<<USER>>>`, and renders rolling conversations.

`turn_loop_contracts.py` owns exact prompt-code markers such as
`## Cat II Resolution`, `## Continuation Required`, and the partial-render
marker. Character turns and exterior-perception requests use separate prompt
templates because they have different knowledge and output authority, but the
model does not receive engine routing names such as foreground, background,
agent-turn, or perception.
When code branches on a prompt mode, the marker should live there rather than
as an ad hoc string in the prompt or caller.

Prompt templates are versioned in git. They are not copied into
version-suffixed filenames, and checkpoint JSON does not store prompt
version ids.

### 5.11 LLM Client

`LLMClient` is the provider boundary for live model calls. Callers send
role, messages, sampling settings, and an optional Pydantic response
model; the client selects the configured provider/model for that role
and normalizes the result back into `LLMResponse`.

It supports:

* per-role provider and model selection
* Anthropic Messages and OpenAI Responses adapters
* Pydantic structured output normalized into `response.parsed`
* Anthropic prompt caching and conditional server-side compaction for
  supported models
* per-role Anthropic extended-thinking budgets
* model-aware OpenAI reasoning effort with explicit per-role overrides
* provider-specific retry handling for transient failures

Provider/model selection can be configured with model prefixes such as
`anthropic:claude-sonnet-5`, explicit `role_providers`, or
environment overrides like `LLM_PROVIDER_NARRATOR=openai` and
`LLM_MODEL_NARRATOR=gpt-5.1`. Current defaults are OpenAI `gpt-5.6-terra`
for `event_router` and `narrator`, with OpenAI `gpt-5.6-luna` at max
reasoning for every character-agent tier and `character_manager`.

Active live model roles are `event_router`, `narrator`, `agent`,
`agent_standard`, `agent_convenience`, `dnd_combat_manager`,
`content_manager`, and `character_manager`.

## 6. Turn Lifecycle

### 6.1 Fresh Action

1. A player submits `/act`.
2. `EngineBridge.run_turn()` sweeps stale Cat II pins before processing
   the new action. Any swept events are resolved first and returned as
   `pre_turn_resolutions`.
3. `Orchestrator.process_turn()` resolves actor, then locks the session.
4. `check_act_slot()` either accepts, rejects, or treats the input as a
   Cat II responder intention.
5. `run_beat()` routes the current intention through the event router.
6. The router emits a Cat I or Cat II event.

### 6.2 Cat I

Cat I is self-closing: dialogue, passive action, unambiguous movement,
ordinary observation, OOC directives, and social attempts where the
speech act itself happens.

The turn loop broadcasts the event, then either:

* ends and renders, if `event_kind` is terminal
* delivers public information, if `event_kind=public_fact`
* performs observation harvest, if `event_kind=observation_harvest`
* performs private query harvest, if `event_kind=query_response`
* dispatches the next valid NPC pick and routes that public result back
  through the router
* lets the narrator request another grounded router event when established
  motion or a submitted wait condition remains visibly unresolved
* forces render if `max_events_per_beat` is reached

### 6.3 Cat II

Cat II is contested: violence, contested possession, forced movement
through opposition, consensual physical contact where reciprocation
matters, and similar actions whose outcome depends on another actor's
response.

Character-agent prose remains a proposed actor submission until this step;
it is not pre-committed fiction. The same submitted character id and text use
the same open Cat II schema as directly supplied text. Current bindings are a
downstream scheduling concern: they decide whether a required responder waits
for input or intends autonomously, but do not change the router's adjudication
contract. D&D Cat II packets likewise omit binding and roll-UI policy; those
settings are consulted only when executing an already-authored roll plan.
The D&D router uses `interaction_mode="narrative"` for both generic Cat I and
Cat II. Only initiative start/end remain D&D-specific interaction modes, so an
adapter label cannot contradict or erase the generic responder contract.

Cat II flow:

1. The router emits an attempt-in-progress event.
2. The engine opens an `OpenCatIIEvent`.
3. Bound responders are pinned in `active_act_slots`.
4. Autonomous responders intend concurrently from isolated copies of the same
   post-open checkpoint. Only after all calls succeed does the engine merge
   their agent histories and intentions in required-responder order; failure
   rolls back the whole open/collection transaction.
5. If all responders are present, the router resolves the event inline.
6. If a ruleset adapter is active, final resolution may enter that
   ruleset's router-owned roll planning/finalization subflow. NPC/agent
   rolls execute automatically; player rolls either execute automatically
   or pause for Discord roll UI depending on `player_roll_mode`.
7. If any bound responder or interactive roll is pending, the narrator renders
   partial-mode prose where applicable and the beat pauses.
8. When the pending responder intention or roll arrives, the router receives
   the compact resolution packet and emits the resolved canonical event.
9. The resolution may name any fictional character in semantic `next_output`.
   The runtime yields when that character is bound or dispatches an autonomous
   turn otherwise; it does not invent an initiator follow-up when none was
   requested.

Cat II final resolution is still router-owned. There is no separate rules
arbitrator model role in the live runtime. Roll plans and dice ledgers persist
in checkpoint transactions for rewind/audit but are not appended to
`session_conversation`; future LLM context receives the canonical outcome facts.

### 6.4 Broadcast

`broadcast_event()` appends the router output to `canonical_events` and
fans it out:

* human observers with visible facts get render-buffer entries
* NPC observers with visible facts get their visible facts appended to
  `pending_observations`
* an NPC event actor receives the same visible canonical facts as any other
  NPC observer; their submitted action is already in rolling history, but the
  canonical event may also contain another character's response, environmental
  change, or adjudicated consequence that the actor must not miss

Important current behavior: `broadcast_event()` does not know whether an
observer is local, remote, mediated, or inferred. It passes
`include_all_observers=True` for every listed observer. Therefore broad
`all_observers` facts reach every observer the router lists. If only one
remote character should receive a mediated fact, that fact must be
`audience="only"` and the router should be careful about whether the remote
character belongs in `observers` at all.

`all_observers` does not mean "all characters in the session." It means all
characters in the event's explicit `observers` list. An event with
`observers=[]` is still appended to `canonical_events`, but no human render
buffer or NPC inbox receives even broad `all_observers` facts.

### 6.5 Render

After each eligible closed event, `_end_beat()` offers the accumulated human
render buffers to `narrator_compose()`. A `continue` judgment discards candidate
prose and leaves the buffers open; a `render` judgment flushes them, stores
per-POV narrator history, and returns `per_player_renders`.

The legacy `output_text` mirrors the acting player's render for older
callers.

### 6.6 Roster And World Mutation

After `run_beat()` returns, the orchestrator applies each closed event's:

* `dormant`
* `cull`
* `spawn`

Movement is expressed in `observable_facts`. There is no separate router
movement side-effect.

### 6.7 Router-Selected Agent Turns

The router owns the fictional decision about who should produce the next
in-world output. On any semantic event kind, observers with
`routing_role="next_output"` form an ordered target list. The turn loop resolves
that list against live bindings: the first bound human yields a player turn,
while an autonomous target is prepared in parallel with the narrator's pacing
judgment. The speculative branch uses isolated checkpoint state. If the
narrator renders, the prepared branch and its agent/router memories are
discarded. A substantive next player action supersedes that semantic frontier;
`(defer)` instead resumes the latest still-dispatchable autonomous
`next_output` without routing the deferral as a new fictional action. The
deferral remains the submitted player input for the next narrator judgment, so
its pacing history records that the previous handoff was declined. If the
narrator chooses `continue`, the engine commits the prepared agent turn, strips
presentation metadata, routes the public result back through the router, and
lets the router decide whether another target is still needed.
Additional `next_output` observers are ordered backlog or fallback candidates,
not a simultaneous response group.

The same surface covers foreground responses, private branches, and background
turns. Human-bound characters do not dispatch as agents. Visible facts
accumulate in per-POV render buffers, and the narrator receives a pacing
decision after each eligible event rather than only after target exhaustion.
When no valid next-output target remains, `continue` may request one grounded
router continuation for established motion or a submitted wait condition.
Cat II, rules-adapter resolutions, queries, observation harvests, and safety
caps force an immediate render. The router therefore does not need to know
which character is controlled by a human.

The engine determines the agent frame from visibility, event kind, location
updates, and player bindings, then enforces hard safety filters: no
human-bound characters, no unknown or inactive characters, no pinned Cat II
responders, no active combatants, and no actors blocked by pending D&D reaction
or roll state. `public_fact` targets and targets whose location changes in the
source event dispatch as background turns. The first such background ping gets
a transient local-context block with current location and same-location active
characters; that block is not persisted in the agent's rolling conversation.

Dormancy is explicit story state, not an inference from "has never appeared
on-stage." An unseen autonomous person with `status=active`, an actor record,
and `actor.may_act_offstage=true` may be picked by the router; a dormant
character does not act until a router/spawn/authored state change activates
them.

Independent background threads are a second use of router actor authority, not
another story scheduler. On a fresh accepted player turn, runtime supplies only
the safe active autonomous initiator ids. The router's truthful focal observers,
actor, responders, and depicted characters define who is engaged in the current
event. If any eligible initiator remains semantically outside that event, the
router must emit at least one `background_threads` selection. As foreground
events reveal newly separate threads, the contract remains active; participants
already selected in that player beat are withheld so each independent thread
runs at most once, with four total selections as the safety cap. Each selection
names one actor and the exact autonomous participants available in that separate
thread. Runtime candidate discovery and grouping do not consult
`CharacterRecord.location`; the existing generic router roster may still carry
that field for ordinary movement contracts, but it is not scheduling authority.
No scene map, fairness ledger, or background clock is persisted.

Each concurrently selected set forks one common post-event checkpoint. Its
character agent receives its own rolling history, witnessed inbox, and the router-selected
participant set, then chooses one concrete action. The router canonicalizes one
closed, zero-duration event. Different threads run concurrently with each other
and with foreground narration or next-output preparation. A lasting task may
open the ordinary commitment without a location label, but a background event
cannot move characters, open another response frontier, schedule nested
background work, change lifecycle or rules state, or observe or depict anyone
outside its participants.

Branches may change only the selected actor's record and conversation plus
their appended compact router record. All results are validated against their
selection-time source, then merged by the single live checkpoint writer in
stable source-and-request order, regardless of completion order. A conflict,
invalid result, or failed branch rejects the whole thread set loudly and restores the pre-merge
checkpoint. The resulting canonical facts and actor memory persist for future
turns, but no human render receives a scene with no human observer. This is a
player-beat liveness boundary, not a wall-clock, faction-clock, or periodic
simulation loop.

### 6.8 Begin, Arrive, Defer, And Query

The public commands for non-standard turns are router entries:

* `/begin` runs `(begin)` once after players have joined the lobby.
* Late `/join` runs `(arrive)` after the opening exists.
* `/defer` resumes the latest still-dispatchable autonomous `next_output` when
  a narrator render interrupted that semantic handoff. With no such frontier,
  it runs `(defer)` through the same turn path as `/act`.
* `/query` runs `(query: ...)` through the normal router/narrator path.

For `(begin)`, the ordinary player-characters block is also the authoritative
claim set. It is built from live bindings rather than every claimable roster
record. Authored opening context may branch on those ids. Even when a story
allows opening spawns, the router may emit them only when that context says
new persistent actors are required; existing and human-bound characters must
be placed or activated rather than regenerated.

An opening policy may also attach exact character dialogue after the router's
arrival event has materialized its roster. The router and adapter remain the
only roster authorities: the authored beat validates the just-closed
spawn/activation signals, commits scoped canonical speech, presents exact
pages without a second narrator paraphrase, and stores the speaker's line in
that character's ordinary conversation history. A required-participant set
may make the beat apply to one opening branch while another branch skips it.

`/query` is not a read-only side channel. It can append canonical events,
update router/narrator conversations, and save a new checkpoint. The router is
responsible for producing a private visible fact that the narrator renders for
the querying POV.

## 7. Event And Visibility Model

### 7.1 ObservableFact

Each observable fact is an object:

```json
{
  "text": "Rashid says, to the table: 'Say that again.'",
  "audience": "all_observers",
  "visible_to": []
}
```

For scoped facts:

```json
{
  "text": "A producer whispers in Dante's earpiece: 'A late contestant is on site.'",
  "audience": "only",
  "visible_to": ["dante_royale"]
}
```

Facts are split by audience, not by sentence. If the same exact audience
perceives a full exchange, one packet is usually better than many small
packets.

### 7.2 ObserverEntry

Observers carry event-level perception and routing intent:

```json
{
  "character_id": "dante_royale",
  "observation_level": "d",
  "routing_role": "observe_only"
}
```

Observation levels:

* `d`: direct
* `i`: indirect
* `f`: inferred

`observation_level` says how the observer encountered the event.
Fact-level `audience` / `visible_to` says which facts they receive.

### 7.3 Observer Routing Roles

`routing_role` is the executable routing decision attached to each observer or
enrichment target:

* `observe_only` means the character receives visible facts and no immediate
  output is requested.
* `next_output` means the router wants this character to produce the next live
  output if the narrator keeps the beat open. The runtime yields for a bound
  character, speculatively prepares an eligible autonomous character beside
  the narrator pacing call, and rejects inactive, pinned, disabled, or
  combat-blocked targets.
* `perception_enrichment` means the character is a perception-harvest target
  for `observation_harvest` or `query_response`, not a response actor.

Observer list order is routing order. Multiple `next_output` observers are an
ordered backlog or fallback set; the runtime still dispatches one same-scene
agent output, routes that result back through the router, and then lets the
router decide whether another participant is still live.

D&D extends this enum with `dnd_reaction`, an adapter-owned role for direct
combat observers who should receive a reaction prompt.

### 7.4 Visibility Caveat

The schema has a fact-level helper that can exclude broad `all_observers`
facts (`include_all_observers=False`), but production broadcast and narrator
composition currently pass `include_all_observers=True`. The practical rule is:
the router's observer list is the event boundary. Do not list a character as an
observer unless they are meant to receive the event's broad facts; use scoped
facts for partial private channels.

## 8. Character State

### 8.1 CharacterRecord

Important fields:

```json
{
  "character_id": "dante_royale",
  "name": "Dante Royale",
  "status": "active",
  "location": "great_hall",
  "is_playable": true,
  "public_sheet": {
    "role": "host",
    "appearance": "",
    "faction": "",
    "public_context": ""
  },
  "actor": {
    "may_act_offstage": true,
    "facts": [
      {
        "origin": "lived",
        "text": "You have hosted the winter assembly three times."
      }
    ]
  },
  "pending_observations": [],
  "mechanics": {}
}
```

`is_playable` means "claimable by a human", not "currently
human-controlled." Human control is determined by
`session.character_bindings` and the legacy `session.player_character_id`
fallback.

`actor` is the one private authoring surface for an autonomous fictional
person. It is optional and deliberately sparse: zero facts is valid, and two
characters need not have matching categories or fact counts. Each fact records
authoring provenance (`lived`, `witnessed`, `told`, or `inferred`) for audit,
while its second-person text carries any uncertainty or source boundary the
person experiences. The CharacterAgent sees the fact text but not the
provenance label. Narrator, router, perception, and public roster projections
do not receive actor facts.

`actor.may_act_offstage` is an engine scheduling policy, not a personality
trait or model instruction. A player-authored blank seat or an intentionally
exterior-only walk-on may have `actor: null`.

`mechanics` is the rules-adapter scratch dict. It defaults to `{}` and
is left empty for narrative-only sessions. The D&D 5e adapter reads a
small conventional subset (ability scores, proficiency bonus, skill and
saving-throw proficiencies, AC, HP, conditions, resources, and the raw
imported sheet) when present. See §15 for the adapter surface.

### 8.2 Pending Observations

`pending_observations` is an NPC inbox. It is populated by:

* visible facts from `broadcast_event()`
* private or mediated facts scoped to the character
* arrival/departure/transfer facts when the router writes those as
  `observable_facts`
* a spawn's own location seed when the spawn path needs the new NPC's first
  dispatch to know where they are

The inbox is drained into the agent's next foreground/private/background
agent-turn prompt.

### 8.3 Private Continuity

Mutable continuity lives in each agent's rolling causal conversation. Public
choices, unanswered questions, and consequences carry it. It is not mirrored
onto the character record as `private_updates`, `last_intent`, or directives,
and the runtime does not ask the model to write a hidden recap after each turn.

## 9. Context Management

Context is deliberately one-shot or delta-based where possible.

Router context:

* system prompt contains stable role contract, schema contract, setting,
  world lore/rules, hidden lore, and hidden facts
* initial NPC roster appears only on the first router call
* the initial roster carries only established public identity; private actor
  facts never enter router context
* router-authored context relies on router history
* external engine changes surface once through `pending_engine_state_updates`
* actor inbox entries surface only when building that character's agent turn

Agent context:

* the system message is actor-independent: generic turn or exterior authority
  plus an optional static ruleset addon
* the current user message carries a second-person self packet (`You are ...`;
  `You ...` facts) and the current witnessed and rules frame
* every accepted user packet remains in the complete causal per-character
  history, with only the disposable presentation catalog removed; history is
  replayed without provider compaction
* cache reuse is an optimization, not a reason to distort or summarize input
* an absent or empty actor record means no private life or world knowledge has
  been established; the agent never falls back to global hidden lore
* there is no repeating "Characters Present" block
* local arrivals and exits arrive through the observation inbox

Narrator context:

* each human POV has a separate rolling narrator conversation
* the narrator gets visible facts and observation tags
* agent responses are already folded into those facts

## 10. Dynamic Character And Arrival Flows

### 10.1 Router Spawns

The router may emit `spawn` requests for meaningful recurring
characters. `CharacterManager.spawn_characters()` renders the `character_gen`
prompt to create the record. The spawn request itself is already stored in
compact router history, so no follow-up spawn summary is queued into the next
router input.

This is story-time NPC authoring. The router decides that the world needs a
new actor and provides a `SpawnRequest` with a target `character_id` and seed
facts such as role, location, or objectives. The character manager rejects
duplicate ids, over-cap spawn batches, existing ids, and generation failures
instead of silently dropping requested spawns. It renders the prompt with
setting/lore/rules/location/existing roster context and expects the shared
`AuthoredCharacter` flat schema.

The generated `CharacterRecord` is appended to the roster. Its authoring result
contains public identity, visuals, one sparse `ActorRecord`, and an optional
frontend `router_summary`; the summary is not stored on the record and is
ignored for router-authored NPC spawns. If the router or caller supplied a
location, that location overrides the LLM-authored location. Fresh NPCs also
receive a small pending observation of their own location so their first
dispatch has a concrete self-position.

Stories may optionally author `world_state.knowledge_tiers`. Knowledge grants
remain cumulative through the selected rung: remembered personal depth and
world/plot knowledge from lower rungs are included. The exact selected rung may
also carry `generation_guidance`, a non-cumulative target for sparse actor
facts, public visual specificity, loadout complexity and material finish,
intended visual salience, and story-local presentation guidance.
`character_gen.txt` treats that target as authoritative, including when it
asks for less detail than the generic default. Actor guidance is one open
instruction rather than paired biography/personality quotas, so it cannot
become a mandatory private-dossier checklist.
The target is rendered in the volatile user tail because the selected rung
changes per spawn.

This contract is rules- and genre-neutral: a story can use tiers for social
station, dramatic importance, supernatural maturity, rarity, or another
authored ladder. Empty ladders, and rungs with knowledge but no generation
guidance, preserve the prior character-generation behavior. Intended salience
is authoring direction expressed through the resulting public appearance and
loadout; it is not duplicated as a second character-visual schema or persisted
as a private rank for the image director.

The spawn path renders `character_gen.txt` through the dedicated
`character_manager` model role. `character_gen` is the prompt/template name,
not a second model role.

### 10.2 LLM-Free Custom Player Characters

`EngineBridge.create_player_character_simple()` creates a player-authored
character directly from name, appearance, and at most one optional lived fact.
It leaves location and all other actor material blank, binds the user, and
queues a state-change line telling the router to infer a concrete story role
and immediate on-ramp.

The router must turn that on-ramp into in-fiction observable facts for
the NPCs who would plausibly know. For example, in a production-show
story, a producer/host may receive a quiet roster update that a late
contestant has been added and must be made to work.

### 10.3 Takeover And Replacement

Takeover paths bind a human to an existing or LLM-authored character and
surface the change to the router through the same state-change queue.
When a player leaves, the engine may synthesize a sparse actor record from
that character's own observable rolling contributions before the agent takes
the character back over. It never mines historical user prompt snapshots.

The takeover/custom-PC authoring prompt is a frontend authoring flow, not an
on-stage agent. It uses the `takeover` prompt but currently calls the LLM under
the `event_router` role.

There are four related paths:

* plain takeover: bind a user to an existing roster character; identity and
  `is_playable` are not rewritten.
* LLM-backed custom character: mode `describe` in `takeover.txt` authors a full
  `AuthoredCharacter`, creates a new playable record, binds the user, and
  queues the router summary with a `[player-bound]` tag.
* replacement: mode `suggest` returns candidate NPC slots without mutation;
  mode `replace` grafts the authored character onto the picked id, preserving
  circumstances such as location, status, and pending observations while
  replacing public identity and the actor record and clearing that id's rolling
  conversation.
* LLM-free custom character: `create_player_character_simple()` builds a sparse
  record directly from user-supplied name/appearance/one lived fact, binds it, and
  relies on the `(arrive)` router turn to place the character in fiction.

## 11. Discord Session Behavior

Discord is the primary UX.

* Sessions are mapped to channels.
* Human players are mapped to character ids.
* Private POV delivery uses private threads where possible, falling back
  to DMs.
* `TurnResponse.per_player_renders` is delivered to the relevant
  players.
* `/begin` is the canonical opening command; `/join` before begin binds
  players into the lobby, while late joiners get an `(arrive)` turn.
* Each posted turn message is tracked in `SessionMap`.
* `/rewind` deletes later checkpoints and deletes or hides tracked
  Discord messages for the deleted turns.
* `/clear` removes Discord-side session clutter without deleting the save.
* `/abort_beat` is an admin recovery tool for wedged active slots or open
  Cat II events.

## 12. Request And Response Contract

The live engine contract is Pydantic, not HTTP.

Request:

```json
{
  "session_id": "uuid-or-slug",
  "checkpoint_id": null,
  "user_input": "I ask Dante why the new contestant is here.",
  "acting_character_id": "dan",
  "stream": false
}
```

Response:

```json
{
  "session_id": "uuid-or-slug",
  "checkpoint_id": "ckpt_0043",
  "turn_index": 43,
  "output_text": "You ask Dante...",
  "per_player_renders": {
    "dan": "You ask Dante...",
    "dante_royale": "Dan turns toward Dante..."
  },
  "beat_ended_reason": "cascade_exhausted",
  "pre_turn_resolutions": [],
  "reaction_prompts": {}
}
```

`debug` and `debug_flags` were removed from `TurnRequest`, and
`TurnResponse.debug` was removed. Per-turn diagnostics live in logs and
checkpoint artifacts. The `stream` field remains on the request for
compatibility but is not a live Discord streaming feature.

Visual-novel responses use ordered `VisualNovelRenderSegment` records. Each
segment carries one or more `rendered_event_ids`: ordinary narration keeps one
segment per canonical event, while an explicitly compressed sequence may use
one passage whose ordered ids preserve stage, media, and System-panel
provenance across every event it summarizes.

`reaction_prompts` is the D&D adapter's combat-reaction UI signal:
`character_id → canonical_event_id` for combatants whose reaction
window is open. It is `{}` outside D&D combat. The Discord bot uses
it to render an immediate reaction UI; the orchestrator uses it to
keep those combatants out of router-selected agent turns until reactions
resolve. See §15.

## 13. Checkpoint Schema

Current checkpoints use schema version `6.0`.

Top-level shape:

```json
{
  "schema_version": "6.0",
  "session": {},
  "player_primer": "string",
  "world_state": {},
  "characters": [],
  "session_conversation": [],
  "narrator_conversations": {},
  "character_conversations": {},
  "canonical_events": [],
  "visibility_log": []
}
```

Important notes:

* `canonical_events` stores full `EventRouterOutput` objects.
* `session_conversation` is the router's rolling history.
* `session.config` is the canonical config source for model labels, narrative
  rules, and live settings.
* D&D roll transactions are checkpoint/audit state, not router, narrator, or
  character-agent rolling history.
* `narrator_conversations` are the authoritative rendered history for each
  human POV. `/history` reconstructs turn labels from checkpoint deltas.
* `character_conversations` are per character.
* `world_state.locations` has no runtime topology.
* `CharacterRecord.location` is an opaque continuity label.
* `player_primer` replaces detached preplay opening prose. The router composes
  the opening arrival on `(begin)`; an optional opening-policy character beat
  may add exact scoped dialogue only after that arrival materializes.
* `CheckpointManager` hard-gates schema version. Checkpoints whose
  `schema_version` does not match the current schema fail to load rather than
  being migrated automatically.

## 14. Prompt Strategy

Prompt files are source. Current prompt files include:

Core narrative engine:

* `event_router.txt`
* `agent_turn.txt`
* `agent_perception.txt`
* `narrator_phase2.txt`
* `character_gen.txt`
* `takeover.txt`

D&D 5e adapter (rendered only when `ruleset_id == "dnd5e_basic"`; see §15):

* `agent_ruleset_dnd5e.txt` — system-prompt addon spliced into
  character-agent calls when D&D combat is active.
* `dnd_cat_ii_router.txt` — separate router prompt for D&D-flavored
  Cat II final adjudication.
* `dnd_combat_manager.txt` — per-turn D&D initiative-scene manager prompt.

One-Star Ascension adapter (rendered only when
`ruleset_id == "one_star_ascension"`; see §15):

* `event_router_ruleset_one_star.txt` — router transaction and fictional
  boundary addon.
* `agent_ruleset_one_star.txt` — character-facing mechanics and knowledge
  boundary addon.
* `character_gen_ruleset_one_star.txt` — generated-Hero authoring addon.

Prompt rules:

* use XML-style tags for major sections
* avoid redundant markdown headers immediately inside tags
* avoid implementation details the LLM does not need
* provide input/output contracts, not pipeline internals
* do not tell a model to remember its own history
* do not add tests that freeze approved prompt prose
* do add tests for runtime contracts and forbidden prompt leakage
* keep exact mode markers in `turn_loop_contracts.py` and test rendered
  helper output rather than hand-copying marker strings through callers

## 15. Rules Adapters

The current adapters are D&D 5e (`ruleset_id == "dnd5e_basic"`) and
One-Star Ascension (`ruleset_id == "one_star_ascension"`). Each is opt-in per
session and entirely off by default; the narrative engine
runs unchanged when no adapter is active. Adapter-specific code,
prompts, schema fields, and bot commands all gate on the active
session settings. See §4.7 for the modularity contract.

The adapter follows a thin-kernel split:

This split is fun-first rather than deterministic-engine-first. D&D rules
exist to serve the table experience, so adapter code should compute rules,
resources, geometry, probabilities, and state as accurately as practical, then
feed that structured context to the router or D&D combat resolver. The LLM
remains the flexible adjudicator for action legality, edge-case rulings,
tradeoffs, and what ultimately happens in the fiction. Code should fail loudly
for safety, privacy, missing reviewed content, unsupported automation, or
impossible state; it should not build a parallel deterministic DM that overrides
the router merely because a rule can be calculated.

* LLM router prompts own D&D judgment: action legality, target choice,
  roll planning, situational advantage/disadvantage, special damage
  adjustments visible in the supplied context, combat status, and
  explicit sustained-effect deltas.
* Code owns dice, arithmetic, durable state mutation, and the ledger:
  initiative and roll execution, hit/miss math, damage and healing
  totals, per-component damage arithmetic, resistance/immunity/
  vulnerability adjustment arithmetic, HP/death-save/effect lifecycle
  persistence, checkpoint transactions, and audit lines.
* The narrator owns POV prose from canonical facts. It should render
  what became visible, not recompute D&D mechanics or invent hidden
  dice outcomes.

### 15.1 Settings

Two settings on `SessionSettings` (registered in
`app/engine/settings.py`) toggle adapter behavior:

* `ruleset_id` — default `narrative`. Set to `dnd5e_basic` to enable
  the D&D agent system-prompt addon, D&D combat-mode helpers, and
  ruleset-specific roll/adjudication paths.
* `player_roll_mode` — default `auto`. Controls whether D&D
  player-character dice resolve in code immediately (`auto`) or
  pause for Discord roll UI (`interactive`). NPC and agent rolls are
  always automatic.

The earlier experimental `cat_ii_resolution_mode` setting has been
removed. Ruleset-specific routing belongs under `ruleset_id`; there
should not be a second independent switch for D&D Cat II or combat
adjudication.

### 15.2 Code Surface

* `app/engine/dnd_combat.py` — combat state machine: initiative
  rolling, turn order, reaction windows, 0 HP/down/death-save state,
  and combat lifecycle.
* `app/engine/dnd_spatial.py` — D&D combat tactical-map helpers:
  router-seed normalization, token placement, advisory distance/line-
  of-sight/cover context, spatial deltas, and compact status text.
* `app/engine/dnd_cat_ii.py` — two router-owned resolvers that share
  the D&D roll-planning/finalization machinery and the
  `CatIIRollTransaction` checkpoint shape:
  * `DndCatIIResolver.resolve_cat_ii` is the adapter-flavored Cat II
    final-resolution path. Invoked from
    `LLMDispatcher.route_intention` when a Cat II event is being
    closed and `dnd_cat_ii_router_enabled` is true.
  * `DndCombatResolver.resolve_combat_action` is the per-turn combat
    resolver. Invoked from `run_beat` (and the post-roll continuation
    paths in `Orchestrator`) when the actor is in active D&D combat
    via `LLMDispatcher.route_combat_action`. Combat actions skip the
    generic `route_intention` entirely; the resolver runs PLAN_ROLLS,
    executes/blocks on player rolls, rolls weapon damage from the
    sheet for hits, applies code-owned damage adjustments from sheet
    traits and router-authored situational adjustments, calls
    FINALIZE_OUTCOME, applies HP changes from structured
    `damage_records`, and synthesizes an
    `EventRouterOutput` with `event_kind="ruleset_resolution"`.
    Neither phase is appended to `session_conversation`.
* `app/engine/dnd_character_import.py` — D&D Beyond character sheet
  import. See `DND_CHARACTER_IMPORT.md` and `DND_MODULE_IMPORT.md`.
* `app/engine/mechanics.py` — readers and helpers for the
  `CharacterRecord.mechanics` dict (ability scores, AC, HP,
  conditions, resources, and per-action attack lookups for the
  combat resolver).
* `app/schemas/dnd_cat_ii.py` and
  `app/schemas/dnd_character_snapshot.schema.json` — adapter schemas.

### 15.3 Schema Fields

* `CharacterRecord.mechanics: dict[str, Any]` defaults to `{}`. The
  D&D adapter reads ability scores, proficiencies, AC, HP,
  conditions, resources, and the raw imported sheet when present.
* `TurnResponse.reaction_prompts: dict[str, str]` defaults to `{}`.
  Maps `character_id → canonical_event_id` for combatants whose
  reaction window is open.
* `DndCombatantState.defeat_state` distinguishes `active`, `down`,
  `stable`, `dead`, and ordinary defeated NPCs. Player-controlled
  combatants use death saves at 0 HP; unbound NPCs normally become
  `defeated` at 0 HP. Death-save rolls and counters are checkpoint/UI
  state, not router, narrator, or character-agent history.
* `DndCombatantState.pending_initiating_action` and
  `pending_initiating_event_id` preserve the hostile action that caused
  initiative to begin. The initiating action does not auto-resolve; the
  field reminds the actor, the CLI/Discord combat status, and the
  character-agent prompt on that actor's first initiative turn.
* `DndCombatantState.active_effects` mirrors adapter-owned runtime
  effects for combatants: concentration spells, timed effects,
  save-ends effects, and effect-backed conditions. The persistent
  character-wide copy lives under
  `CharacterRecord.mechanics["dnd5e_runtime"]["active_effects"]` so
  sustained D&D effects can survive combat boundaries without adding
  generic narrative-engine fields.
* `DndCombatState.pending_visible_facts` is a short queue of code-owned
  combat lifecycle facts, currently used for death-save outcomes such as
  regaining consciousness, stabilizing, or dying, and D&D effect lifecycle
  facts such as starts, ends, and updates. The orchestrator flushes these as
  ordinary observable facts after combat advancement, and combat-end paths
  drain the queue into the same visible event before clearing combat.
* `DndCombatState.battle_map` is an optional adapter-owned tactical grid
  for active D&D combat. It stores the single runtime
  `DndBattleMapState`: token coordinates, visible terrain, visible area
  templates, and imported tactical geometry such as spawn anchors,
  keyed-area links, secret features, difficult terrain, and vertical links.
  The router's `battle_map_seed` remains a smaller `DndBattleMapSeed` used
  only to start simple fiction-derived maps. Player/model-facing projections
  strip hidden anchors, keyed refs, reveal triggers, source hashes, and secret
  features before rendering status or combat-manager packets. Battle-map state
  does not change generic `world_state.locations` or `CharacterRecord.location`.
* `DndCombatState.router_observed_facts` stores combat-manager selected
  continuity facts until initiative ends. Each item is only `fact`,
  `salience`, and `reason`; subject ids and event ids stay out of the
  model-facing contract. Combat-end paths queue these as compact
  `pending_engine_state_updates` so the next generic router call sees
  narrative continuity without replaying every combat turn. The same
  combat-end bridge also queues code-owned player lifecycle facts such as
  death or unresolved unconsciousness.
* `SessionState.cat_ii_roll_transactions` carries checkpoint-durable
  D&D roll plans, pending player rolls, completed roll results, and
  dice ledgers. Each transaction is tagged
  `source: Literal["cat_ii", "combat"]` so the same checkpoint shape
  serves Cat II adjudication and combat turns. Combat transactions
  also carry `actor_id`, `intention`, `context` (the LLM packet),
  and `damage_records: list[CatIIRollDamageRecord]` for code-applied
  HP changes. None of this is appended to `session_conversation`.
  Ending combat cancels non-finalized combat transactions and clears
  their `cat_ii_roll` slots.
* `active_act_slots` may contain `combat_blocked` entries. These lock a
  character whose fresh action would start a second D&D combat while the
  session already has active initiative. The entry is not a general actor
  lock: the same player can revise and act normally afterward, which
  abandons the blocked hostile action. `/defer` drops the blocked action
  explicitly.

### 15.4 Prompt Files

* `app/prompts/agent_ruleset_dnd5e.txt` — system-prompt addon spliced
  into character-agent turns when `ruleset_id == "dnd5e_basic"`. Adds
  combat-aware behavior on top of the rules-neutral `agent_turn.txt`.
* `app/prompts/dnd_cat_ii_router.txt` — Cat II-flavored router prompt
  used by `DndCatIIResolver`. Two phases: PLAN_ROLLS emits a
  `RollPlan`; FINALIZE_OUTCOME emits a `RulesAdjudication`.
* `app/prompts/dnd_combat_manager.txt` — per-turn initiative-scene manager
  prompt used by `DndCombatResolver` through the `dnd_combat_manager` LLM
  role. Same two-phase shape and shared `RollPlan` base as the Cat II prompt,
  but FINALIZE_OUTCOME uses `DndCombatManagerAdjudication`, which extends the
  ordinary rules adjudication with lean `router_observed_facts`:
  `fact`, `salience`, and `reason`. The user packet is the combat-state
  snapshot (round, current combatant, all combatants with AC/HP/conditions/
  defeat state/death saves/active effects, optional battle-map/spatial
  advisories, the actor's available actions with
  `id`/`name`/`attack_bonus`/`damage`, and the house rules) rather than a
  Cat II opening context. The manager handles any initiative turn, including
  speech, surrender, rescue, aid, and other non-attack actions.

### 15.5 Orchestrator Hooks

Hooks in `Orchestrator.process_turn` and `run_beat`:

* fresh `/act` from a combatant in active D&D combat routes via
  `LLMDispatcher.route_combat_action` and the separate
  `dnd_combat_manager` role instead of the generic `route_intention`.
  The check is
  `_character_in_dnd_active_combat(ckpt, actor_id)` —
  `ruleset_id == "dnd5e_basic"` AND the actor is listed in
  `session.active_combat.combatants`. Combat-reaction `/act`s gated
  by a `combat_reaction` slot follow the same fork after the slot
  cleanup, so reactions get the same house rules and structured roll
  planning as turns. Ongoing combat turns do not append compact router
  history; only the final combat event is recorded after
  `session.active_combat` clears;
* when the D&D addon router emits `interaction_mode="dnd_combat_start"` from
  a fresh non-combat action, the engine validates participants, rolls
  initiative, creates `session.active_combat`, syncs all combatants into the
  observer list, attaches a validated router-seeded battle map when supplied,
  materializes any `combatant_spawns` into non-playable adapter-owned D&D
  characters, stores the actor's `pending_initiating_action`, and appends only
  player-safe visible facts such as "D&D combat begins." Spawned combatants use
  the router-emitted fallback stat block unless an adapter-local statblock
  override provider supplies corrected values for the `monster_key`. Initiative
  totals and true statblock names stay in combat audit/status surfaces, not in
  narrator or agent facts;
* if `dnd_combat_start` appears while another combat is already active, the
  engine does not open Cat II and does not start a parallel combat. It appends
  a fiction-facing no-effect fact for observers, pins that character with a
  `combat_blocked` act slot for the blocked hostile action, and lets later
  benign or revised `/act`s proceed normally. `/defer` clears the blocked
  action immediately. Engine explanations for the blocked action are frontend
  notices, not observable facts for the narrator or character agents;
* `interaction_mode="dnd_combat_end"` from the generic addon router is
  accepted only from an actor listed in the active combat. Outsider actions in
  other scenes cannot end combat by assertion;
* if the generic router emits Cat II for an actor in active combat
  (prompt-drift safety net), the engine clamps it down to a single
  Cat I-shaped beat with `event_kind="ruleset_cat_ii_suppressed"`
  rather than opening a parallel responder flow against the
  initiative ladder;
* a direct observer of a closed combat beat marked
  `routing_role="dnd_reaction"` can receive a reaction prompt in the
  same beat as the actor instead of waiting for their own turn;
* a `(defer)` from a combatant with an open reaction window resolves
  as a reaction acknowledgement rather than a turn skip;
* `_handle_combat_after_beat` advances initiative state after a beat
  closes. Turn advancement also processes engine-owned D&D effect
  lifecycle work such as recurring save checks, duration countdowns,
  and concentration-loss cleanup;
* `_run_automated_combat_turns_locked` drives NPC combatant turns
  inline before the player's response returns;
* when combat damage newly defeats a non-player combatant with an XP value
  or challenge rating in its D&D mechanics, the combat adapter splits that
  XP among the bound player combatants in the active combat. The payout is
  checkpoint-audited and keyed by combatant id so repeated damage or roll
  finalization cannot duplicate rewards;
* `Orchestrator.submit_cat_ii_roll` and
  `Orchestrator.continue_cat_ii_after_roll` branch on
  `roll_transaction_source(ckpt, event_id) == "combat"` to finalize
  a combat transaction whose triggering event never lived in
  `open_cat_ii_events`. The combat-resolution branch calls
  `_resolve_ready_combat_after_rolls`, which routes through
  `LLMDispatcher.continue_combat_transaction`. If combat has ended or the
  transaction was cancelled before the player submits the roll, the slot is
  cleared and the response is a stale-roll notice rather than a 500;
* combatants with pending `reaction_prompts` are excluded from router-selected
  agent turns until they answer their reaction window.

All hooks are no-ops outside `dnd5e_basic`.

### 15.6 D&D Combat House Rules

Opportunity attacks are automatic. When D&D combat movement provokes
an opportunity attack, the router should signal the opportunity and the
engine should resolve it with code-owned dice for both NPCs and player
characters. Player opportunity attacks do not open a reaction prompt and
do not consume the player's optional reaction resource in Ayoa.

Only reactions that require a meaningful player choice should open
`reaction_prompts` (for example, protective intervention, interruptive
magic, catching someone, or choosing to dive into danger). A player can
answer with `/act` or pass with `/defer`.

Character agents do not need dice results to do their job. The combat
agent addon gives them initiative context, action-economy expectations,
visible facts, and their own combat state; it does not provide initiative
rolls, attack rolls, damage rolls, roll formulas, roll ledgers, or
death-save counters. Dice are exposed to players in UI/status surfaces
because tabletop players expect to see them, and active effect labels are
shown in combat status views for the same reason. These details are retained
in checkpoint audit state for rewind/debugging, but character agents receive
only the combat state they need to choose plausible actions.

Sustained effects are hybrid: the router decides that a D&D spell,
feature, or item starts, ends, or updates an effect and supplies
spell-specific metadata such as conditions, concentration, duration, and
recurring-save timing. If the current action breaks an existing effect,
the router emits an explicit end-effect delta. The engine stores and
advances effects after adjudication: it ends prior concentration when a
new concentration effect starts, checks concentration after damage, ticks
round durations, rolls recurring saves at the declared timing, and keeps
effect-backed conditions synchronized.

Deferred D&D adapter cleanup:

* decide whether the Cat II D&D resolver should eventually move off the
  generic `event_router` role as well, or remain a router-owned contested
  action path
* trim vestigial or diagnostic-only structured fields such as
  `state_deltas`, `PlannedRoll.effect_id`, and effect `source_type`/
  `source_id` fields in a dedicated schema-compatibility pass
* design visibility rules for hidden/private active effects before
  exposing hidden curses, marks, or similar DM-only effects in shared
  tactical surfaces
* decide whether manual `/combat damage` should stay a raw operator
  override or grow typed damage inputs that run through the normal
  resistance/immunity/vulnerability pipeline

### 15.7 One-Star Ascension Adapter

One-Star uses the generic narrative router for fictional judgment and adds one
compact `state_updates` list to that same output. Every update is the same
four-field record: kind, primary target, primary scalar value, and repeated
`key=value` details. Durable operation and state models are not part of the
provider schema. The adapter translates the compact list into private typed
work, validates exact references, arithmetic, clocks, and lifecycle
preconditions, then applies the complete mutation atomically before the
canonical event is broadcast. There is no second resolver or parallel combat
loop.

Durable state stays on existing character records:

* the unique account owner carries configuration and account state under
  `CharacterRecord.mechanics["one_star_account"]`, including per-pool summon
  draw counters and any story-authored external-funds balance;
* each Hero carries HP, XP, stars, stats, equipment, skills, injuries, and
  ownership under `CharacterRecord.mechanics["one_star_hero"]`; exact returned
  gear remains in the account's stored-equipment ledger rather than being
  flattened into an inventory count;
* a story-authored non-Hero may carry only HP and stats under
  `CharacterRecord.mechanics["one_star_combatant"]`; this supplies stable
  conflict authority without granting Hero ownership, progression, summoning,
  promotion, or synthesis semantics;
* `CharacterRecord.status` and `CharacterRecord.location` remain the only
  lifecycle and location authority. The adapter does not duplicate the roster;
* `WorldState.global_flags` carries no One-Star progression or scene state.
  Opening policy, canonical mission events, and the account record already own
  those facts, so a static floor, phase, presence, or lobby flag would be a
  contradictory second ledger;
* story-authored configuration owns costs, rewards, caps, facilities, summon
  pool weights, deterministic progression inputs, progression prerequisites,
  floor scenarios, physical operation requirements, and any fixed cash-to-Gem
  schedule. Engine code contains no One-Star entity names or economy constants.

Opening variants share one `usage="opening_roster"` pool contract rather than
parallel Master/Newcomer schemas. Its ordered slots may be fixed authored
characters, deterministic existing Heroes of an authored grade, or an exact
`bound_player_actor`; the last kind is valid only when that actor is currently
bound. The live player composition selects one complete pool, and the adapter
resolves that ordered roster without a random draw. Its first event contains
only that summon followed by the reviewed first-floor `mission_start`, with no
pending deployment. The adapter materializes every selected Hero directly at
the mission destination and commits acquisition, stamina, and active-mission
state atomically. Dialogue then follows the ordinary routed character path;
there is no branch-specific guide bridge, response-collection round, or exact
post-render briefing path.

The cached router addon receives compact immutable operation authority:
catalogue costs/effects, physical-operation requirements, and standard-pool
costs, star ranges, rates, and usage. Authored opening pools disclose only
usage, required count, and non-identifying slot kind or grade. Their exact
fixed or bound-player character ids and resolved roster identities appear only
in the branch-specific volatile `(begin)` tail. A configured Gem shop
contributes only its immutable starting balance, periodic income schedule, and
fixed pack exchange to that cached authority. Normal routes do not receive a
second live ledger snapshot.
Each accepted compact `state_updates` list is retained in the
same prior-event history as its canonical fiction. State
created independently of a router decision, such as a generated summon sheet,
automatic stamina or external-funds accrual, or deterministic
progression/synthesis result, is projected into that history once. When an
update is rejected, the single repair call receives only the current rows that
prove the conflict—for example the target Hero's current HP or the relevant
balance and configured cost—not the full account or roster, and it preserves
the same opening-identity boundary. Eligible dormant reserves, private RNG
inputs, future
draw results, draw counters, progression formulas, stored-equipment records,
applied-event fingerprints, and private Hero potential are never model context.
A compact pending synthesis preview is an exception: its exact offered, applied,
and wasted XP, returned items, and configured disclosed per-source chance are
current decision context, but never its unrolled outcome. One-Star routes use
`OneStarEventRouterOutput`; continuation routes use the closed subtype. Default
narrative and D&D routes retain their existing schemas and receive no One-Star
prompt or state block.

For a standard summon, the router emits only pool id and count. The adapter
derives the exact weighted birth grades, eligible dormant reserves, fresh stable
ids, and corresponding generic spawn/activation lifecycle without sending any
result or future slate to the model. A successful atomic commit advances the
pool counter, while failed validation and replay cannot reroll or double-
advance. Freshly generated summon identities carry a durable origin marker and
stay on the Luna-backed standard-agent role; activated seeded reserves retain
their authored agent tier. The accepted event hands its exact arrivals directly
to the active configured guide. That owed agent turn and its one exact survival
induction event complete before narrator presentation can close the beat, so a
fast render cannot strand an active unbriefed Hero; ordinary autonomous handoffs
retain narrator-paced speculative behavior. Authored opening pools remain
non-random and do not consume standard draws. Deployment, synthesis, and
promotion are staged:
selection opens a zero-side-effect pending operation, affected Heroes keep
ordinary response ownership, and a synthesis selection also collects the
configured lobby guide's own enforcement intention. A later event may resolve
only after the recorded bodies physically reach the configured gate or chamber.
Tower mission boundaries, pre-existing escape authority, resource underflow,
event-id fingerprints, and exactly-once rewards are hard validation constraints.
Progression is deliberately lightweight deterministic bookkeeping: seed
configuration and recorded XP determine thresholds, levels, stats, and HP, and
synthesis derives its exact ledger consequences from the accepted operation.
Potential is private account data and never enters Master, router, narrator, or
agent projections. The router remains responsible for fictional damage, death,
resistance, reward-worthy action, and character choices; it does not assign
levels, XP totals, stats, HP maxima, or synthesis arithmetic.

Each configured `floor_scenario` is reviewed seed authority for one mission's
destination, premise, immutable completion and failure declarations, counters,
and pressure beats. A mission start copies that authority and the exact selected
party into durable state. Starting positions and later repositioning are
Hero-owned tactics carried by canonical fiction, not a durable Master-controlled
formation variable. These adapter checks constrain mission bookkeeping; the
generic router still judges how the party attempts the scenario and what fiction
follows.

Mission control is asymmetric and derives from bindings rather than a stored
scene selector. If any deployed party Hero is human-bound, floor progression
yields to that player and cannot run hidden autonomous mission turns. Every
Master-started beat during that mission is guarded: the Master may manage the
disjoint lobby, route an active lobby guide's bounded induction, or issue a pure
zero-time watch/status query, but cannot advance or target the floor party.
When a human Hero also owns a live Cat II responder slot, the Master receives
concurrent-turn admission only if every conflicting slot is one of those live
human pins. Each event in the admitted beat is then validated under its actual
actor before commit; the guard rejects party routing, unsafe cross-boundary
movement, mission updates, and unrelated lifecycle or commitment changes. The
frontend lock and canonical event log still serialize accepted turns; this
admission does not create simultaneous checkpoint writers.

If the deployed party is entirely autonomous, an eligible Master management,
watch, or defer turn may continue the existing router-selected agent cascade
until the mission ends, a human handoff appears, no consequential target
remains, or `max_agent_cascades_per_beat` is reached. A targetless result gets
one grounded continuation attempt and then ends unless a newly committed agent
turn creates a new frontier. This is the ordinary turn loop under a smaller
story-configured cap, not a scheduler, scene variable, mission feed, or second
combat loop. At a forced Master-facing boundary, the narrator compresses that
batch into one rapid sequence, eliding repeated combat exchanges and chatter
while preserving consequential actions, injuries, deaths, objective progress,
and the terminal result. Ordinary non-batch narration remains event-aligned.

One-Star installs no private background scheduler, lobby eligibility pass, or
facility-context side channel. The generic router receives the same safe active
autonomous initiator ids used by every story and decides from canonical history
which characters are semantically outside the focal mission event. The adapter
continues to constrain mission observers and state authority, but it neither
chooses a lobby actor nor supplies an activity. There is no private cue, held
floor handoff, location lane, or One-Star resume path.

Independent Hero activity can run at the same time as the foreground narrator
or autonomous mission preparation. The selected character agent receives its
own witnessed history and exact router-selected company; training, cooking,
maintenance, crafting, rest, or another grounded move may emerge from that
context. Longer work may open the ordinary commitment. Reaching toward another
participant stops before inventing that person's reply. The closed event is
pinned to its selection instant and cannot observe or depict the Master, guide,
deployed party, or any other unselected character; mutate the account or
mission; or move anyone through generic location state. Its canonical facts and
character memory persist without entering the Master's active-floor render. A
human-led floor remains untouched.

After a terminal mission result, the active guide may make one concrete prompt for the
Master's next management choice. If the Master defers, that bound handoff goes
back through ordinary router selection and gives exactly one eligible
co-present non-guide Hero a grounded opportunity to act. Ordinary lobby
breathing room is sufficient; this does not resume the guide, rotate a debrief,
force speech, or invent interface motion. Existing autonomous Hero handoffs
still resume their owed output directly when the player defers.

Mission state remains deterministic: the adapter validates counters, outcomes,
survivors, returns, rewards, progression, unlocks, and replay safety. Those
mutations add only their exact System consequences to the same canonical
terminal event. The ordinary narrator presents that committed event once; no
second projector rescans mission history or appends separate report prose or
visual-novel panels. A terminal Hero mutation similarly adds an exact scoped
death notice so live System observers receive the state change without gaining
physical presence or response ownership.

Character-agent, status, and image projections are viewer-scoped. The account
owner sees the local account roster, public XP progress, and exact usable stored
gear; Heroes see their own body at the authored System-detail level; configured
guides receive lobby-management and tutorial state, while characters separately
listed as System observers receive the live canonical mission feed without
becoming physically present or response-eligible before mission end; image
prompts receive only visible current equipment. Narrator input remains
canonical observable facts rather than raw adapter state.

The read-only `/master status`, `/master heroes`, and `/master hero` commands
use the same viewer-scoped projection through `EngineBridge` in both CLI and
Discord. They are available only when the invoking viewpoint is the account
owner; they do not call an LLM, create canonical events, or expose raw mechanics
maps.

`/master synthesis` is deliberately different: it resolves user-facing Hero
references and submits an exact Master interface choice through the same
locked router, Cat II, character-agent, narrator, rollback, and POV-delivery
path as `/act`. The command does not mutate the ledger or declare consent,
movement, coercion, or completion. Selected Heroes decide their responses;
the configured guide decides any enforcement; the router canonicalizes the
resulting physical scene before the adapter may resolve it.

### 15.8 Modularity Contract

Per §4.7, the adapter must not change the narrative engine's generic
behavior. Default settings keep the engine narrative-only; adapter
prompt files only render under the matching `ruleset_id`; adapter
schema fields default to empty; adapter bot commands no-op or return
a clear message outside the matching ruleset; the core router,
narrator, and character-agent prompts stay rules-neutral with
D&D-specific behavior delivered through addons rather than baked
into the base templates.

If a feature seems to require changing the generic engine to support
D&D, surface the design tension in review before implementation. The
correct answer is usually a new adapter hook with a no-op narrative
default, not a rules-flavored core.

## 16. Error Handling

LLM failures:

* transient API errors are retried with exponential backoff and jitter
* permanent schema/prompt errors raise
* a router or narrator failure aborts the turn before checkpoint commit
* empty agent outputs are omitted from the current routed-agent pass
* a failed actor-submission router call aborts before checkpoint commit

Structured output:

* event router and narrator calls use Pydantic response models
* all fields in `EventRouterOutput` are required to keep the structured
  output grammar small
* schema validators assign missing event ids, coerce unknown event kinds
  where safe, enforce Cat II invariants, and enforce fact visibility
  membership

Discord failures:

* turn-message tracking failures are logged but do not fail the engine
  turn
* rewind cleanup falls back from delete to edit/hide when Discord delete
  fails

## 17. Observability

The system exposes:

* per-role logging
* router decision rationale in logs and checkpoint conversation history
* checkpoint snapshots per turn
* canonical event logs
* per-character rolling conversations
* per-POV narrator conversations
* tracked Discord turn-message refs

The router `decision_rationale` field is temporary diagnostic overhead.
When router behavior is stable enough, remove the schema field, prompt
field, and log plumbing together.

## 18. Current Gaps And Maintenance Notes

Known stale or transitional areas:

* some code comments still reference older architecture names or call counts,
  especially around the early v11 turn-loop skeleton and hidden-context
  comments
* `visibility_log` exists but the main v11 flow relies on
  `canonical_events`, render buffers, and NPC inboxes
* `/query` is implemented as a mutating router/narrator turn, not a
  read-only information endpoint
* router-selected background threads provide bounded player-beat liveness, not a
  generalized world clock, faction clock, or periodic simulation
* debug streaming and public HTTP APIs are not implemented
* prompt version ids are not stored in checkpoints; git history is the
  version source

These are acceptable as long as the design doc names them honestly and
tests cover the live contracts rather than preserving dead architecture.

The remaining subsections in this chapter are open architectural
concerns: real, known sharp edges that are not solved. Read the
relevant entry before redesigning router-selected agent routing, the rolling
agent conversations, or anything that times the world (turn counters,
contextual transitions, parallel play).

### 18.1 World time across asynchronous play

The engine carries narrative event timing, not a complete world clock:

* `session.turn_index` — narrative time. Advances on every closed beat
  (both `process_turn` and `resolve_cat_ii` increment it). Player-facing
  narrator history and checkpoint labels are keyed off this counter.
* canonical event timing — `effective_at_s`, `duration_s`, per-character
  clocks, and commitment signals. These let the router place events relative
  to the acting branch but do not constitute a shared monotonic world clock.

In single-player single-scene play these stay close enough to be
indistinguishable. With multiple players acting in multiple scenes
asynchronously (the design target), two players can advance separate branches
without flattening them into one global now. There is still no shared
monotonic world clock for off-stage NPCs, factions, or hazards to reason
about independently of routed events.

This matters because:

* Cross-scene causality (an antagonist in scene A reacts to a player
  victory elsewhere) needs an ordering primitive richer than a single
  local `last_event_at`.
* "Did this happen before or after that?" gets answered differently
  depending on which participant's perspective you ask from.

Adding a global turn-counter gate in this code path requires care:
confirm the gate's semantics are coherent across parallel beats. If
the answer is unclear, surface the problem on a TODO instead of adding
a brittle global counter.

### 18.2 Routed-agent cascade latency on the live action path

Router-selected agent work runs on the live action path. Same-scene agent
turns are sequential: one target produces public output, the router
canonicalizes it, and only then can another target in that scene act with the
updated context. This matches live table pacing better than parallel NPC
fan-out, because later speakers know what earlier speakers just said or did.
The first exception is a set of independent autonomous Cat II responders: each
must react to the same attempt rather than to a sibling's draft, so those
intentions run concurrently on isolated snapshots and merge atomically in
router-required order. The second is router-selected independent background
threads: each exact semantic participant set advances through one bounded
actor-router event, while disjoint threads prepare beside one another and the
foreground before a deterministic atomic merge. Multi-character authoring uses the same
isolation-and-atomic-merge principle. None of these cases changes the sequential
canonical-event order.

This should not be framed as "returning player control" as a special runtime
ontology. A human-bound character is a participant whose immediate output is
rendering visible facts to that player's POV rather than appending an inbox
entry or asking an agent model for an intention. Some live routed-agent work
belongs to scenes with no player currently present; it still needs bounded
liveness and a clear target, not player-control language.

The latency risk is router over-selection. Prompt and schema work should group
characters who share one semantic thread, select the minimum number of threads
needed for liveness, and prefer actors whose newest witnessed pressure can
produce a meaningful move. The engine still enforces hard caps on event count,
agent cascades, and background selections.

### 18.3 Cross-scene observation inbox

`broadcast_event` populates `pending_observations` for every NPC
observer the router lists (excluding the actor and human-bound
characters, who route to render buffers instead). The inbox drains on
the recipient's next foreground/private/background agent-turn call.

Open knob: `pending_observations` has no length cap. A cross-scene
NPC observer who never gets called accumulates inbox entries across
turns. If long-running sessions surface inbox bloat, a per-character
cap on `pending_observations` length is the obvious place to add one.

### 18.4 Rolling agent conversations across prompt-version boundaries

`character_conversations[character_id]` is a rolling list of causal
`ConversationMessage` entries replayed on the actor's next call. Before the
release-candidate migration boundary, old local sessions are not a compatibility
contract: prompt/schema changes may deliberately make their legacy agent
history unsupported. Do not add format reminders, rewrite queues, or shadow
readers merely to preserve experimental parenthetical or JSON-shaped replies.

When release-candidate save compatibility becomes a product requirement, make
the migration boundary explicit and deterministic. Until then, current seeds
and tests use one contract: observable prose or exact silence.

## 19. Engineering Discipline

### 19.1 Vestigial-field destruction policy

A field on a Pydantic schema is a contract: someone is supposed to
write it and someone is supposed to read it. When that contract
breaks, the field becomes a hazard — its value sits on disk in old
saves, it shows up in serialization, it tempts readers into trusting
it, and it accumulates documentation that explains the original
design rather than the live system.

Rules:

1. When you remove the last writer of a field, in the same commit
   remove the field from the schema. Do not leave the field behind
   "for back-compat with old saves." Pydantic v2's default
   `extra='ignore'` silently drops the legacy field on load — no
   deprecation flag is needed.
2. When you remove the last reader of a field, in the same commit
   remove either the field or the writers. A write-only field is
   dead freight on every checkpoint serialization.
3. When you change a field's semantics ("this used to mean X, now it
   means Y"), rename it. Keeping the same name and changing the
   meaning poisons every blame, every reviewer hand-off, and every
   future search.
4. When in doubt, list the field as vestigial in §18 before the
   change ships, so the next contributor knows not to trust it on
   read.

Past failures have included a global location field that was set at
authoring time and never updated, and a `TurnResponse.debug` field with no
orchestrator writer at all. Both wasted reviewer time; one silently
misled a 31-turn playtest summary.

## 20. Future Directions

These are long-horizon hypotheses, not scoped tickets. They describe
questions worth keeping in mind when the relevant subsystem comes up
for change, so opportunities to chip at them can be taken cheaply
rather than chased for their own sake.

### 20.1 Long-term narrative planning beyond the router

Today the router carries the entire load of "what should happen
next." That is fine for short-horizon adjudication (this beat, the
next two turns) but the router has no notion of a multi-act arc, no
concept of "introduce a new character at hour three of the show," and
no mechanism to plant a setup in turn 5 and pay it off in turn 40.
Long-form sessions naturally want fresh faces mid-story, periodic
beats keyed to a story-level cadence, and off-screen plot motion that
advances even when a player camps in one scene.

Possible directions, none chosen:

* a separate "showrunner" LLM that runs every N beats and emits
  high-level intents the router threads into adjudication;
* a persistent story-arc object on the checkpoint with explicit
  tension/pacing/reveal targets the router consults;
* a synthetic story-arc skeleton (acts, reveals, beats-to-trigger)
  that runtime advances against;
* a periodic "casting director" pass that proposes new spawns based
  on roster gaps the LLM identifies.

Whatever shape this takes, keep the router prompt tax-aware: the
router already handles adjudication, perception, and roster decisions
on every turn. Adding long-term planning to its system prompt without
offloading would push it past its current sweet spot. A separate
actor on a longer cadence is the most likely shape.

### 20.2 Spawn discipline beyond MAX_SPAWNS_PER_TURN and prompt rules

Spawn rate is currently constrained only by `MAX_SPAWNS_PER_TURN=3`
and router prompt language preferring canonicalize-with-observer over
spawning for one-shot utility characters. Long-form sessions can
still let the roster bloat past the point where the router prompt
re-summarizes it cheaply on every call.

Open questions:

* should spawned characters carry a TTL — auto-dormant after N beats
  with no participation, recoverable on demand?
* is "named, plot-relevant character" vs "ambient world fixture" a
  distinction worth modeling in the schema, with different cost and
  visibility profiles?
* should the router receive a current-roster-size and recent-spawn-
  rate signal so it self-throttles, instead of relying purely on
  prompt language?

Don't solve preemptively — wait for a session where the roster
genuinely bloats and let that shape the answer.

### 20.3 D&D spatial/grid modeling

D&D combat now has an adapter-owned tactical grid for active combat:
router-seeded map state, participant tokens, visible terrain/areas,
imported runtime geometry, advisory distances, line-of-sight, cover
context, and router-authored spatial deltas. It is intentionally
advisory; code persists and summarizes map state, while the D&D combat
manager still decides action legality and outcome from player-safe map
projections. Future work may add manual map authoring, image rendering,
strict movement/path validation, elevation, hidden tokens, lighting, and
richer area-template geometry.

Image rendering here means player-facing presentation of already reviewed
assets or authored geometry. It must not introduce runtime image analysis
by any LLM; map features, hidden areas, monster placement, and secret
annotations must come from manual import artifacts.

### 20.4 Public information for off-screen saliency

Off-screen characters become objectively salient through
`event_kind="public_fact"`. A public fact is still a normal canonical event:
the router writes public or semi-public `observable_facts`, selects exact
observers who receive them through `pending_observations`, and uses
`routing_role="next_output"` only for recipients who should act on that
information now. `routing_role="observe_only"` delivers the inbox entry without
cascading.

This is the minimal public-information buffer. It has no separate queue,
delay, provenance field, or universal broadcast. The router represents delay,
source, and provenance in the fact text itself, such as criers, courier
reports, rumors, official notices, records, aftermath, or live broadcasts.
Long-running sessions still need care: public facts should name eligible
recipients narrowly so broad public events do not become universal inbox noise.

### 20.5 D&D inventory and economy

Imported character sheets already contain item data, but runtime play
does not yet model carrying, dropping, drawing, buying, selling,
attuning, consuming, or transferring items and currency. A future
inventory layer should stay adapter-owned and should distinguish
equipment that affects mechanics from ordinary narrative possessions.

### 20.6 D&D action economy decomposition

Combat currently treats one player `/act` as one adjudicated turn.
Fuller 5e support needs explicit action, bonus action, movement,
reaction, free interaction, and limited-resource consumption tracking.
Do not add this piecemeal to the generic turn schema; model it as a
D&D combat adapter concern when playtests need finer-grained legality.

## 21. Acceptance Criteria

The current engine is healthy when:

1. A bound player can submit `/act` and receive a coherent POV render.
2. Multiple bound players can receive separate POV renders from the same
   beat.
3. Cat I actions close without stealing another actor's response.
4. Cat II actions render the attempt and wait for required responders.
5. NPC observers receive only facts visible to them.
6. Remote or mediated observers are listed only when a concrete perceptual
   channel makes the event available to them; private partial channels use
   scoped `audience="only"` facts.
7. NPC agents receive no actor-local hidden summary from another agent.
8. The narrator renders from visible observable facts without adding
   unsupported action or attitude.
9. Router-created spawns, dormancy, and culls persist
   to checkpoint state.
10. `/query` answers through the router/narrator path without leaking
    knowledge outside the querying POV.
11. Router-selected private/background turns and independent unnarrated scene
    ticks advance eligible NPCs as router-canonicalized events without
    parallelizing same-scene causality.
12. `/rewind` removes later checkpoints and cleans up tracked Discord
    turn messages.
