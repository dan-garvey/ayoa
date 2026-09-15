# What we are keeping

The experiments support treating NPC knowledge boundaries as a behavior the
single author must respect. They do not establish that individual character
agents are necessary for coherent storytelling. Sustained single-author trials
produced distinct interests, refusals and consequences, alongside local knowledge,
authorship and continuity errors. They also repeatedly exposed a separate problem:
characters could remain recognizable while the overall conversation felt prepared,
didactic or socially thin. These were not controlled comparisons against the former
engine, nor proof of coherence over an unlimited history.

The active branch banks a small author/editor loop and the best supported prompt
lessons while keeping the remaining literary objective open.

## Evidence archive

All prior prompts, runs, drafts, edits, protocols and assessments are preserved
on [archive/covenant-prompt-trials](https://github.com/dan-garvey/ayoa/tree/archive/covenant-prompt-trials).
The complete pre-cleanup snapshot is
[`d6865cb`](https://github.com/dan-garvey/ayoa/tree/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b).
The links below pin that commit, so future evidence additions cannot change the
historical comparisons. The [full research report](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/PROXY_REPORT.md)
retains the detailed chronology and reviewer disagreements. Original direct
API trials and their reports also remain in that archived experiment directory.

## Findings and their practical consequences

| Finding | What this foundation does | Evidence and limits |
| --- | --- | --- |
| Explicit revision can remove rhetoric that defensive author instructions leave intact. | Give the editor an actual draft; preserve that intermediate output. | [C25](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/reviews/edit25_review.md) found a material game-dialogue improvement and an uneven second result. [C27](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_27/decision.json) sustained the two-call loop across two 15-turn sessions, but did not meet the overall literary bar. |
| A concise editor can remove aphorisms while retaining substantive conversation. | Keep the short editing contract instead of the original 312-word review instructions. | [C42 comparison](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_42/review_response.md): both reviewers found reduced rhetoric in all four short edits. Two drafts and two repeats per condition are a targeted result, not a long-session guarantee. |
| Particular personal engagement can survive the concise cleanup. | Use the selected 40-word editor as the working baseline. | [C43 comparison](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_43/review_response.md): one shared joint success in two augmented repeats; the second was too explanatory. Short-only and long controls also succeeded. The added sentence has no demonstrated repeatable advantage. |
| Revision can remove useful interaction or replace it with a writing workshop. | Review whole exchanges, including what was lost; retain both versions. | [C29 edit diagnosis](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/reviews/round29_edit_diagnosis.md) and [C30 comparison](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_30/decision.json) preserve these failures and mixed preferences. |
| Rich biography remains essential; shortening or changing its presentation is not an established prose remedy. | Keep the restored Covenant biographies and a second rich ensemble in Breakwater. | [C38](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_38/decision.json) improved source fidelity without establishing a literary rescue. The [cast audit](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/reviews/round40_cast_instruction_audit.md) distinguishes genuine psychology from portrayal instructions. |
| Extra history representations and numerical ceilings did not demonstrate a reliable remedy. | Replay published prose directly and impose no visible-word quota. | [C31](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_31/decision.json) found no consistent discarded-draft exposure effect; published-only history is a clean canon contract, not a proven prose intervention. [C41](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_41/decision.json) found shorter dinners still weak and sometimes less interesting. |

The sugar rewrite—“There is no proper amount. There is only enough” becoming
“It could use more sugar”—is a concrete success. That gain is worth keeping
even though it does not settle the quality of the entire scene.

## Current source choices

The reusable author prompt is the unchanged 393-word C43 common prompt. The editor
now uses the user's exact two-sentence review-pass instruction in
`prompts/revision.txt`, targeting habitual aphorism, character voice and specific
dialogue rules. This is the current trial prompt. Its short form is deliberate;
no extra hidden review criteria or character dialogue examples are appended at runtime.

The previous editor used the 40-word C43 augmented instruction. Adding
“Ensure your edits don't break character-specific dialogue rules” did not fix a
Covenant opening where revision flattened Seraphel's verse. The replay retained
a line break while losing the draft's cadence and poetic phrasing. The initial
assessment overstated that result; user review correctly identified the failed
repair. The editor also substantially rewrote the dinner conversation. The
private opening sessions retain their exact requests, drafts and revisions.

The 36-word review-pass replacement also flattened Seraphel's verse in a single
Terra/max replay of the same draft. Most narration remained intact, but the edit
added a "Revised passage:" preamble and moved Thessaly's counting retort ahead of
the line it originally answered. Only the final revision instruction changed
between the three requests. This candidate remains a tested trial, with no
demonstrated improvement on the verse failure.

Sol/max then authored and edited a fresh opening using the same story snapshots
and review prompt. It left the player at the doorway with a clear choice to join
dinner. The edit removed staged banter but shifted the conversation toward routine
housekeeping; Seraphel did not speak. A separate editor-only comparison changed
only the model on the original Terra draft. Sol gave the correction/counting
exchange a coherent question and answer, but again converted Seraphel's verse
into ordinary sentences split over two lines. The verse failure persisted, and
the overall narrative comparison remains mixed. All three Sol calls are preserved
locally alongside the earlier private opening sessions.

Covenant starts from C41's restored source, including Ysolde's early life and
sensory tastes, Caelindra's education and dispersed childhood cohort, Thessaly's
hallway memory, and Rashid's double-booked evening. C42/C43 used an older story
brief for their fixed-draft comparisons; importing it here would lose later
restorations and change the starting circumstances.

Story direction and canon are separated without summarizing the biographies.
Mixed passages retain their psychological and knowledge content; explicit portrayal
guidance moves to direction. The [source audit](https://github.com/dan-garvey/ayoa/blob/caf12e4967c8315a032014b8c08bd224d4cdba39/foundation_20260914/source_migration.json)
records each move. Breakwater's entire C41 brief was imported as byte-identical
canon, accompanied by a short genre/opening direction. It is an experiment-authored
example, not a user-selected replacement for Covenant or a blind held-out story.

The current story files use **the protagonist** in place of Rowan so the player
can choose a name. Covenant's inheritance wording preserves the Garvey family's
property and standing without requiring the chosen surname to be Garvey. Family
history, places and the full biographies remain intact. The default name lives
in `player.json`; session name choices and corrections appear in the user tail.
The pinned playtest snapshots retain their original wording.

Player-owned dialogue, thoughts, feelings, consequential choices and manuscript
contents remain protected. NPCs may pursue their own goals, resist, misunderstand,
enjoy quiet company or show earned warmth. The degree of narrative direction
follows what the player does. New history is welcome when it fits established
events, dates, abilities and disclosures; invention alone is not an error.

## Foundation validation

All 27 offline tests pass, covering context separation, API/proxy request
equivalence, exact draft preservation, publication, restart recovery and failures,
including the real SDK with an in-memory HTTP transport. Ruff and formatting
checks pass. The foundation's direct API transport has not been exercised with
a live call.

The implementation playtests produced eight published passages: an opening plus
three player turns in Covenant and Breakwater, using 16 fresh Terra/max proxies.
Every first draft and edit is preserved. The [audit](https://github.com/dan-garvey/ayoa/blob/caf12e4967c8315a032014b8c08bd224d4cdba39/foundation_20260914/validation.json)
reconstructs every request from frozen sources and published history and verifies
the accepted outputs and complete public input reads. The [evidence index](https://github.com/dan-garvey/ayoa/blob/caf12e4967c8315a032014b8c08bd224d4cdba39/foundation_20260914/README.md)
links both transcripts, all draft/edit pairs, source hashes and quality gates.

The [full review](https://github.com/dan-garvey/ayoa/blob/caf12e4967c8315a032014b8c08bd224d4cdba39/foundation_20260914/review.md)
records a clear local cleanup: “a thin slice is just a small disappointment”
becomes “The first ones were too thick,” while the characters retain their film
preferences. NPC commitments and room reservations survive the player's decision
to leave those activities alone. Covenant honors an explicitly quiet interval.
These are useful gains, alongside regressions: most edits grow longer, a new
stock joke appears, Covenant gains a speaker-attribution error, and Breakwater's
final passage moves the player beyond the requested stopping point.

This short smoke test validates the foundation workflow and exposes literary
problems. It does not establish reliable overall improvement, long-term coherence
or an advantage over the former engine. Proxy usage, latency and hidden context
cannot establish direct API cost, performance or behavioral equivalence. The
broader narrative-quality task remains open.
