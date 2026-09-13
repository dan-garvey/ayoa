# Explicit fact visibility and bookkeeping removal

The approved event-consumer audit is implemented except for open commitments.
`OpenCommitment`, all three router commitment directives, and the event state
application function are AST-identical to the previous revision. Commitment
timing, matching, interrupts, and revision prompts remain in place.

## Current contract

- Every fact names its complete `visible_to` recipients. Event observer groups,
  fact `audience`, and narrator/image observation-level fields are retired.
- Partial perception is separate fact text, not a full event plus concealment
  instructions. Every recipient shares that fact's visual scope.
- Existing `visual_subject_ids` controls appearance introductions, harvested
  appearance recipients, sprite rosters, and image-character projection. A name
  mention or audio packet does not grant visual access.
- Canonical event identity, timing, causal scheduling, and media anchors remain.
- Submission provenance/outcomes are transient batch accounting. Narrator jobs
  retain only `event_refs`; their source IDs and highest sequence are derived.
- Router conversation entries reference canonical sequences; canonical prose is
  projected at prompt time, preserving ordering among external context entries.
- Unused `FrontierTurn.created_event_sequence` and checkpoint `visibility_log`
  are removed.

The runtime still resolves a ready batch against one snapshot, commits the
validated batch atomically, and then gives each recipient their facts. Human
POVs have retryable narration; autonomous actors draft intentions from their
own observations. Overlapping next turns remain sequential, while independent
work can draft concurrently. No extra model call or context mode was added.

Two affected privacy consumers were corrected: autonomous contest responders
now read their scoped canonical opening facts, and imported fronts learn only
facts explicitly received by their actor. Neither reader may widen a private
fact to everyone named elsewhere in an event.

## Verification

- Full offline suite: **1,826 passed, 2 skipped**.
- Final checkpoint/schema/prompt-hygiene/visibility/router checks: **70 passed**.
- Changed Python files: Ruff passed. `git diff --check` passed.
- Live router: **12/12 cases, 66/66 checks, no rejected batches**.

Offline regression coverage includes split whisper fan-out to human narration
and NPC observations, checkpoint round-trip, private contest context, front
knowledge, audio-only visual exclusion, scoped appearance harvest, history
projection ordering, transient provenance, and derived narrator references.
Existing handoff/retry tests continue covering human selections, history-sourced
no-event continuation, merged lanes, and independent work.

Live artifacts (local, with exact prompts, raw responses, and usage):

- `app/storage/playtest_reports/router_prompt_targeted_20260913T002938Z.json`
- `app/storage/playtest_reports/router_prompt_targeted_20260913T002938Z.md`
- `app/storage/playtest_reports/router_prompt_targeted_20260913T002938Z.log`

The prior comparison report was
`router_prompt_targeted_20260912T203443Z.json`: 11 cases, 58/59 checks, with an
audio-only visual-subject leak. The current run adds the whisper case and passes
the same mediated-perception check. Both raw output sets were reviewed, not just
their pass counts. This is a targeted run, not a long-session playtest or proof
that stochastic perceptual errors cannot recur; `ayoa-izvs` remains open.

The current run recorded 44,307 total input tokens, including 23,338 cache-read
tokens and 20,969 uncached tokens. The prior 11-case run recorded 41,513 total,
including 23,800 cache-read and 17,713 uncached. Different case counts and cache
warmth mean these totals do not establish a latency or cache-efficiency change.

## Actual whisper output

Rowan and Caelindra received:

> Rowan leans toward Caelindra and whispers, "Meet me in the west garden at midnight. The password is silver-fern."

All six guests received only:

> Rowan leans toward Caelindra and whispers to her; the words are not audible.

Both packets named Rowan and Caelindra as visible subjects. Only Caelindra was
selected for the next contribution. In the separate audio-only pod case, Dan,
Britney, and Maya received the microphone speech with an empty visual-subject
list; Britney was selected to answer.

## Schema boundary

Checkpoint version is now **8.0**. All shipped story seeds and the synthetic
checkpoint template are updated. Existing saved sessions, local imported story
seeds outside source control, and historical playtest artifacts were not
rewritten. Older saves/seeds are not automatically converted. No bot restart or
production session mutation was performed.
