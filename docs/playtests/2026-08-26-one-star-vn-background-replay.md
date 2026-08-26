# One-Star VN authored-background replay

Date: 2026-08-26 (America/Chicago)

Session: `one-star-vn-background-replay-20260826`

Repository revision: `229b02cbe68f58d4b14a0a775195fc127dc89bb0`

## Outcome

The authored location-selection work succeeds, but the runtime image path then
throws away most of its value. The director chose the reviewed 1F courtyard for
the ordinary lobby and the reviewed Crack plate for the gate scene. It later
reused the Crack artifact byte-for-byte. No image, path, asset hash, or vision
input reached the director LLM; it selected from authored text hints and opaque
handles.

Both generated derivatives were substantially worse than the reviewed inputs.
FLUX replaced the modest courtyard with an ornate circular arena and hallucinated
two unrelated men despite an explicit empty-scene request. Qwen replaced the
distinctive Crack with a generic dark corridor and two glowing grid rectangles.
The approved backgrounds should therefore be immutable compositor stages, not
references for another runtime background-generation call.

Production also does not yet have the requested character-sprite layer. The card
renderer accepts a stage plus semantic pages and draws the ADV text box, but it
has no foreground character inputs. The two men visible in the first deck came
from the failed background generation, not controlled identities. Iselle and the
speaking Heroes never appear on the later cards.

The narrator tuning moved in the right direction. After the first `/defer`, it
continued through seven ordinary autonomous handoff candidates and rendered at a
real state change. Later it stopped at concrete Master formation choices. All
pages end at complete semantic boundaries. The remaining player-visible roster
bracket is a deterministic contraction-anchor bug, not narrator inventiveness.

The Synthesis Chamber could not be visually tested. A valid synthesis command
failed before mutation with `authoritative result router output attempted to
author fixed side effects`; the three Heroes remained active and no pending
operation was left behind.

## Configuration and coverage

The Master was the only claimed seat. Every embodied character and all three
opening character-generation calls were modelled with Luna. The production
router and narrator stayed on Terra so the playtest remained representative of
the intended narration path.

| Role | Model |
| --- | --- |
| `agent` | `openai:gpt-5.6-luna` |
| `agent_standard` | `openai:gpt-5.6-luna` |
| `agent_convenience` | `openai:gpt-5.6-luna` |
| `character_manager` | `openai:gpt-5.6-luna` |
| `event_router` | `openai:gpt-5.6-terra` |
| `narrator` | `openai:gpt-5.6-terra` |
| `image_director` | `openai:gpt-5-mini` |

The run used the real CLI and EngineBridge, durable director/job store, remote
FLUX compose and Qwen edit pipelines, production stage resolver, and production
ADV compositor. Inputs, in order, were:

1. `/join 2`
2. `/settings presentation_mode visual_novel`
3. `/begin --confirm`
4. `/defer`
5. Read-only `/master status`
6. `/defer`
7. Read-only `/master heroes`
8. `/master synthesis 1 from 2, 3` (failed before mutation)
9. `/defer`
10. Read-only `/master status`, `/status`, and `/history 5`
11. `/quit`

The four accepted decks contain 31 cards. Accepted stage actions were
`replace -> clear -> replace -> reuse`.

## Deck ledger

| Update / event | Director | Cards | Deck id | Stage result |
| --- | --- | ---: | --- | --- |
| 2 / `opening_0001` | replace | 2 | `5df0eca5e9d227db641e980c2fc94380973b3ad7af818452f48a4331754ebacf` | FLUX courtyard derivative `399d9786...` |
| 3 / `opening_0009` | clear | 20 | `58249cf5cc31bf79434f284d37893e1f77b9f3d999ecf86a6356de15c8e08685` | neutral stage `0b2ecf51...` |
| 4 / `opening_0011` | replace | 2 | `4844d294c4b773b3e9f7659e5a6116474d176e0cecb0eade79cfefbf5d94c6bd` | Qwen Crack derivative `59cfc6af...` |
| 5 / `opening_0014` | reuse | 7 | `162b5da71bbe9a231a24c672a78b36747792eeeb99e5b82c8ae7bc2060f6f941` | same `59cfc6af...` bytes |

The first generated job used `osa_loc_1f_courtyard_v1` with FLUX compose and
took 286.909 seconds. The second used `osa_loc_1f_crack_lobby_v1` first in a
Qwen edit and took 97.712 seconds. Both jobs completed successfully; the visual
failures are accepted-model-output failures rather than worker or delivery
failures.

## Feedback by source layer

### P1: reviewed plates should bypass runtime regeneration

The approved courtyard input (`04ac3c7d...`) is a bright, modest plaster-and-
timber 1F space with an open foreground. Its FLUX derivative (`399d9786...`) is
a huge gray circular arcade with frost geometry and two polished male fantasy
characters. This violates both the tier and the director's `No visible people`
instruction.

The approved Crack input (`82b568c7...`) preserves the canonical broad arch,
dark ribbed aperture, segmented white-cyan seam, blue block fragments, daylight,
and plain 1F courtyard. Its Qwen derivative (`59cfc6af...`) removes the ribbing,
fragments, daylight, and courtyard. It resembles the rejected clammy corridor
direction and reduces the Crack to two rectangular light grids.

This is not a location-selection failure. The director chose both correct opaque
handles. The failed step is regenerating an already-reviewed location. Follow-up:
`ayoa-yvja.1`.

### P1: one final event's stage was applied to twenty earlier pages

Turn 3 buffered eight canonical events. The first seven are Iselle, Marta,
Tovin, and Eren speaking in the established lobby. Only final event
`opening_0009` has Eren start toward the accommodation hall. Candidate director
runs for the earlier events were cancelled when the narrator chose `continue`.
The accepted final run chose `clear`, and the entire 20-page render consequently
used the neutral stage.

That result is worse than a simple same-scene clear: pages describing the fading
summon-light, Iselle's welcome, and all three lobby conversations are
retroactively presented after the stage has disappeared. Stage sections need to
align with the events/pages they depict rather than inherit the last event's
decision. Follow-up: `ayoa-yvja.2`.

### P1: production cards have no character-sprite layer

`VisualNovelDeckSection` carries semantic pages and one stage source. The
renderer cover-crops the stage, draws the box, speaker plate, text, counter, and
advance marker; no character identity, pose, expression, placement, or sprite
bytes enter that composition.

This explains why the reviewed-stage decks show only an empty environment under
Tovin and Iselle dialogue. The accepted proof direction—one or two characters
slightly facing one another with bounded expression/pose changes—still needs a
deterministic foreground layer over immutable backgrounds. Follow-up:
`ayoa-yvja.3`.

### P1: valid synthesis fails at the authoritative router boundary

The lobby had an operational level-1 Synthesis Chamber and three active 1-star
Heroes. `/master synthesis 1 from 2, 3` elicited Tovin and Eren's Luna responses,
then `route_authoritative_result` failed with:

> authoritative result router output attempted to author fixed side effects

No new checkpoint was committed for the command. A subsequent status projection
showed all three Heroes active and `Pending management operation: none`, so the
failure rolled back cleanly. It nevertheless blocks the core operation and the
Synthesis Chamber background test. Follow-up: `ayoa-42ag`.

### P2: grouped pronoun anchors fail after a contraction

Iselle's actual Luna output was natural:

> once you're in the Tower, you choose your own movement, targets, tactics, and
> skills.

Router canonicalization correctly anchored the group referent after each pronoun.
`replace_character_ids_for_narrator` stripped the anchors after `you` and `your`
but expects whitespace immediately after a pronoun, so it missed the anchor after
`you're`. The narrator received a safe-name form and page 12 rendered:

> once you're [Marta Vell,Tovin Rusk,Eren Vale] in the Tower

This is an uncovered deterministic projection case from the prior direct-address
fix, not a reason to add more prose to the narrator prompt. `ayoa-fhm6` was
reopened with the exact contraction/group regression.

### P2: `/begin` stops before an already-owed autonomous welcome

The first delivery consists only of one atmospheric sentence and one System
registration notice. `opening_0001` already owed Iselle's response, and the
unseen Master had no meaningful action available, but `/begin` returned control.
The player had to enter a no-op `/defer` before the embodied opening began.

The later ownership-aware narrator candidate path made better judgments. The
initial forced boundary should use the same semantic ownership rule unless the
opening itself presents a real player choice. Follow-up: `ayoa-pt9j`.

## Narrator and character feedback

The current narrator prompt does not need a broad rewrite on the evidence from
this run. It did three things materially better than the previous Luna playtest:

- it chose `continue` seven times during routine onboarding and stopped when Eren
  physically left the hall;
- it later stopped when Iselle presented an actual Master formation choice; and
- all authored pages and compositor overflow pages ended at complete semantic
  boundaries, with no recurrence of the split `her` / `other hand` defect.

There is a tuning tradeoff to review with a human: one `/defer` now produced 20
clicks covering eight events. The prose is coherent and the player truly had no
useful response during most of it, so this is not automatically wrong, but it is
the opposite extreme from the previous one-event interruptions. A card-burden
signal may be more useful in later testing than reinstating a raw character count
or an unconditional event cap.

Luna was sufficient for this functional run. Marta, Tovin, and Eren had distinct
immediate concerns and voices. Iselle's sharper lines and `Cute method!` framing
come from her character agent and story-specific identity, while the narrator
faithfully renders them. Prompt tuning should continue to separate character
voice, router fact shape, narrator presentation, and compositor behavior.

## Performance

- `/begin` to the first card took about 5 minutes 54 seconds; 4 minutes 47
  seconds were the FLUX job itself.
- The eight-event no-generation continuation took about 1 minute 37 seconds.
- The Crack replacement turn took about 2 minutes 17 seconds, including the
  1 minute 38 second Qwen job.
- The final three-event stage-reuse turn took about 47 seconds.
- The failed synthesis probe took roughly 15 seconds and returned a loud error.

Using the approved stage bytes directly would remove both image-generation waits
from these background-only scenes. The remaining model-cascade latency is
separate from image latency.

## Privacy and integrity audit

The production image-director messages were reconstructed from all four accepted
durable projections and outputs. The rendered messages contain authored public
facts, reference selection hints, and opaque ids such as
`osa_loc_1f_crack_lobby_v1`. They contain no image bytes or data URI, asset path,
image filename, reviewed asset SHA-256, or event fingerprint. The generation
worker request resolves the selected handle to hash-validated image bytes only
after the director returns.

No raw character source id appears in any of the 31 card speaker/text fields.
The bracketed safe names are a readability defect, not an id or private-metadata
leak. Deck manifests bind every card hash and the reused Crack stage hash is
identical across updates 4 and 5.

## What worked

- Every observed `character_manager`, `agent_standard`, and
  `agent_convenience` call used `gpt-5.6-luna`.
- The director selected the contextually correct authored courtyard and Crack
  handles from text-only metadata.
- The accepted `reuse` decision produced the exact same stage hash without a
  third generation job.
- The compositor's text box, speaker plate, wrapping, counters, navigation,
  manifest, accessible transcript, and cached-card delivery worked on 31 cards.
- The new narrator handoff context substantially reduced meaningless Master
  interruptions without inventing Master dialogue or in-fiction `/defer` prose.
- Synthesis failed before mutation and left no partial pending operation.

## Preserved artifacts

Raw local evidence (intentionally ignored by Git):

- `app/storage/playtest_reports/one-star-vn-background-replay-20260826/transcript.tty`
- `app/storage/playtest_reports/one-star-vn-background-replay-20260826/transcript.clean.txt`
- `app/storage/playtest_reports/one-star-vn-background-replay-20260826/evidence.json`
- `app/storage/sessions/one-star-vn-background-replay-20260826/ckpt_0000.json`
  through `ckpt_0005.json`
- the four deck folders listed above
- `app/storage/runtime/image_generation/artifacts/39/399d97867006184590ced48f603a36100cd5621cf336463faddb2712b0bb2740.webp`
- `app/storage/runtime/image_generation/artifacts/59/59cfc6af63a67c72b3047277ecdd58d6da4375892297c3aa3857418291071798.webp`
- `app/storage/runtime/image_generation/jobs.sqlite`

The four complete deck folders, clean transcript, evidence export, this report,
and approved-versus-runtime comparison images are copied to
`C:\Users\danim\Pictures\Ayoa\Visual Novel Dialogue Proofs\one-star-vn-background-replay-20260826`.

Durable tracking:

- playtest: `ayoa-xd6g`
- stage/identity umbrella: `ayoa-yvja`
- immutable reviewed stages: `ayoa-yvja.1`
- page/stage alignment: `ayoa-yvja.2`
- sprite compositor: `ayoa-yvja.3`
- contraction anchor regression: `ayoa-fhm6`
- synthesis command: `ayoa-42ag`
- sparse opening handoff: `ayoa-pt9j`
