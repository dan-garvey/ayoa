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
is the exact 40-word C43 augmented instruction selected by the user. Its short
form is deliberate; no extra hidden review criteria or character dialogue examples
are appended at runtime.

Covenant starts from C41's restored source, including Ysolde's early life and
sensory tastes, Caelindra's education and dispersed childhood cohort, Thessaly's
hallway memory, and Rashid's double-booked evening. C42/C43 used an older story
brief for their fixed-draft comparisons; importing it here would lose later
restorations and change the starting circumstances.

Story direction and canon are separated without summarizing the biographies.
Mixed passages retain their psychological and knowledge content; explicit portrayal
guidance moves to direction. Breakwater's entire C41 brief remains byte-identical
canon, accompanied by a short genre/opening direction. It is an experiment-authored
example, not a user-selected replacement for Covenant or a blind held-out story.

Player-owned dialogue, thoughts, feelings, consequential choices and manuscript
contents remain protected. NPCs may pursue their own goals, resist, misunderstand,
enjoy quiet company or show earned warmth. The degree of narrative direction
follows what the player does. New history is welcome when it fits established
events, dates, abilities and disclosures; invention alone is not an error.

## Foundation validation

The implementation tests context separation, API/proxy request equivalence, exact
draft preservation, publication, restart recovery and failures using offline
clients, including the real SDK with an in-memory HTTP transport. Literary
validation uses the first outputs from fresh Terra/max proxies, an opening plus
three player turns in each story, with each editor seeing only its own draft and
the shared published context. Final evidence and outcomes are recorded after those
runs finish. Proxy usage, latency and hidden context cannot establish direct API
cost, performance or behavioral equivalence.
