# What we are keeping

The experiments support treating NPC knowledge boundaries as a behavior the
single author must respect. They do not establish that individual character
agents are necessary for coherent storytelling. Sustained single-author trials
produced distinct interests, refusals and consequences, alongside local knowledge,
authorship and continuity errors. They also repeatedly exposed a separate problem:
characters could remain recognizable while the overall conversation felt prepared,
didactic or socially thin. These were not controlled comparisons against the former
engine, nor proof of coherence over an unlimited history.

The active branch now publishes one author response directly. Explicit player-requested
regeneration replaces the automatic editor. The prompt lessons and prior editorial
experiments remain preserved while the literary objective stays open.

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

| Finding | Experimental lesson | Evidence and limits |
| --- | --- | --- |
| Explicit revision can remove rhetoric that defensive author instructions leave intact. | Give the editor an actual draft; preserve that intermediate output. | [C25](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/reviews/edit25_review.md) found a material game-dialogue improvement and an uneven second result. [C27](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_27/decision.json) sustained the two-call loop across two 15-turn sessions, but did not meet the overall literary bar. |
| A concise editor can remove aphorisms while retaining substantive conversation. | Short editing instructions can match a longer review on selected examples. | [C42 comparison](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_42/review_response.md): both reviewers found reduced rhetoric in all four short edits. Two drafts and two repeats per condition are a targeted result, not a long-session guarantee. |
| Particular personal engagement can survive the concise cleanup. | The 40-word editor was a working trial, not a demonstrated reliable improvement. | [C43 comparison](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_43/review_response.md): one shared joint success in two augmented repeats; the second was too explanatory. Short-only and long controls also succeeded. The added sentence has no demonstrated repeatable advantage. |
| Revision can remove useful interaction or replace it with a writing workshop. | Review whole exchanges, including what was lost; retain both versions. | [C29 edit diagnosis](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/reviews/round29_edit_diagnosis.md) and [C30 comparison](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_30/decision.json) preserve these failures and mixed preferences. |
| Rich biography remains essential; shortening or changing its presentation is not an established prose remedy. | Keep the restored Covenant biographies and a second rich ensemble in Breakwater. | [C38](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_38/decision.json) improved source fidelity without establishing a literary rescue. The [cast audit](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/reviews/round40_cast_instruction_audit.md) distinguishes genuine psychology from portrayal instructions. |
| Extra history representations and numerical ceilings did not demonstrate a reliable remedy. | Replay published prose directly and impose no visible-word quota. | [C31](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_31/decision.json) found no consistent discarded-draft exposure effect; published-only history is a clean canon contract, not a proven prose intervention. [C41](https://github.com/dan-garvey/ayoa/blob/d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b/experiments/covenant_single_llm/proxy_trials/candidate_41/decision.json) found shorter dinners still weak and sometimes less interesting. |

The sugar rewrite—“There is no proper amount. There is only enough” becoming
“It could use more sugar”—is a concrete success. That gain is worth keeping
even though it does not settle the quality of the entire scene.

## Current source choices

The reusable author prompt keeps the C43 common instructions and adds the user's
sentence: "Avoid excessive aphorism in character dialogue." The automatic editor
and its revision prompt have been removed. After comparing the current Sol playtest,
the user judged the unedited writing better overall and selected direct publication.
This choice retains ordinary situational humor and avoids a mandatory second pass
that can flatten voice or remove interaction. It does not establish that every
individual edit was worse or that the new prompt has passed a fresh literary trial.

The user's opening correction keeps exchanges short enough for the player to
respond. The reusable prompt now applies that handoff when an exchange calls for
the player's response, allowing other passages to end naturally without another
question. This replaces the broader instruction to stop whenever someone addresses
the protagonist, after the closure comparison described below.

Explicit regeneration uses the original player action, the current passage and
the user's new instructions. The new passage replaces the current one atomically.
Later turns contain neither the rejected passage nor regeneration feedback; both
remain saved for evaluation. This also addresses the demonstrated case where an
ordinary chat request to rewrite an opening left superseded events in later context.
Earlier ordinary chat messages are not automatically reinterpreted as regenerations.

Before retirement, the editor used the user's exact 36-word review-pass instruction,
targeting habitual aphorism, character voice and specific dialogue rules. The
following results describe those preserved experiments.

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
checks pass. At that initial validation, the foundation's direct API transport
had not been exercised with a live call.

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

## Reasoning summary capture

Automatic responses now request exposed reasoning summaries. The optional
**Inspect responses** view displays each summary with its corresponding saved
version. Summaries never enter published history, subsequent model input or
transcript downloads; older responses cannot acquire a summary retroactively.

Two isolated Covenant openings verified capture with fresh Codex CLI 0.154.0
processes at maximum reasoning effort: Sol returned two summary entries, Terra
returned one. These were compact planning headings, so capture alone does not
establish their value for diagnosing narrative decisions. The
[preserved requests, responses and browser proofs](https://github.com/dan-garvey/ayoa/tree/4dd10075303a0c7e32fd9138caaa2db6b8bc49ee/reasoning_summaries_20260916)
retain both first results. No private reasoning was inspected, and the user's
existing playtests were preserved. All 84 offline tests pass, including browser
coverage and API summary serialization; the API check used an offline transport.

## Conversational closure and transport comparison

The latest Covenant playtest preserves NPC refusals but repeatedly supplies a
replacement invitation or question when an exchange could end. Fifteen of its
22 published passages end in direct questions. The prompt already protects NPC
independence and quiet; its engagement and turn-taking instructions may nevertheless
encourage a repeated conversational handoff. This is a hypothesis about interacting
instructions and history, not an isolated causal result for any one sentence.

The [prompt audit and live comparison](https://github.com/dan-garvey/ayoa/tree/1ef8fc1a81110cf55dab2c832a7497b5e09e9135/forced_engagement_20260917)
preserve three exact request checkpoints with two Terra/max samples each under
normal Responses message roles, the proxy's tagged text sent directly to Responses,
and the existing fresh Codex proxy. All 18 first calls completed; none was retried
or replaced. The direct calls used the existing request/response functions without
publishing to the original session, whose bytes were verified unchanged.

Explicit new questions or invitations appeared in 3/6 normal API responses,
4/6 tagged API responses, and 5/6 proxy responses. These are descriptive manual
counts, not automatic failure scores or reliable population estimates. Five of six
dinner replays let Ashara pursue practice, then moved the closing invitation to
Dena's card game. One normal API library replay asked about the protagonist's
parents after he thanked Stone and looked back at the book. Both normal API
continuations of the next reading turn avoided another question, though they
still added comments and retained some didactic prose.

The current coding wrapper is therefore unnecessary to reproduce the problem.
All conditions inherit proxy-written history, so this does not rule out earlier
wrapper influence or establish identical serving behavior. The unblinded sample
is small, and message roles and output controls differ. The
[complete passages](https://github.com/dan-garvey/ayoa/blob/1ef8fc1a81110cf55dab2c832a7497b5e09e9135/forced_engagement_20260917/TRANSCRIPTS.md)
and per-response assessments include the counterexamples.

The next candidate should clarify natural conversational endings and chosen quiet
activity while retaining independent NPC initiative and the full biographies.
The [analysis](https://github.com/dan-garvey/ayoa/blob/1ef8fc1a81110cf55dab2c832a7497b5e09e9135/forced_engagement_20260917/ANALYSIS.md)
proposed replacing overlapping direction/handoff wording. It was subsequently
tested and adopted in the coding-agent-only comparison below.

## Tested closure wording and Seraphel correction

The [coding-agent comparison](https://github.com/dan-garvey/ayoa/tree/40ca05c495e316e3edf75d22239d5dd7f8237588/closure_rhyme_20260917)
used 20 fresh Terra/max calls: two samples of the previous and candidate author
prompts at five checkpoints. Both conditions received the same corrected Seraphel
canon and detailed-summary setting, so the author wording was the controlled
difference. Full biographies and original history were preserved. All first
results were retained; no direct API calls or editorial passes were used.

At the dinner-departure, library-thanks and return-to-reading checkpoints, the
previous prompt added a new question or invitation in 6/6 responses; the candidate
did so in 0/6. Both candidate observation samples still advanced independent NPC
plans, and both explicit-engagement samples answered the player's question. The
candidate is now the reusable author prompt. It permits conversations to end and
follows chosen activity while preserving room for NPC initiative and short replies.
Some passages remain too long or contain unnecessary rebukes and didactic lines.
These targeted, unblinded results do not establish consistently strong prose.

Seraphel's story canon now requires rhyming dialogue, including brief replies and
questions, and removes the contradictory permission for unrhymed verse and
preference against couplets. The initial correction produced audible rhyme in
9/10 passages where she spoke, with some unpaired lines and one wholly unrhymed
response. A separate eight-call follow-up tested a stricter requirement for an
end-rhyme partner on every spoken line. It did not consistently improve naturalness
or compliance and renewed demands in both quiet-reading samples, so it was not
adopted. A repeat with the simpler rule also renewed an invitation once, showing
that the closure improvement is not a guarantee. The
[complete responses](https://github.com/dan-garvey/ayoa/blob/40ca05c495e316e3edf75d22239d5dd7f8237588/closure_rhyme_20260917/TRANSCRIPTS.md)
and separate primary/refinement judgments preserve these limitations.

Automatic requests now set `reasoning.summary=detailed`, mapped to
`model_reasoning_summary="detailed"` in the proxy. All 28 live calls returned exposed
summaries, ranging from 110 to 1,158 characters; many still consist of short headings.
The Inspect responses UI displays these existing summary records. Summaries remain
outside published history, subsequent prompts and transcript exports.

The latest Covenant playtest received the promoted author wording and simpler
Seraphel correction through an explicit one-time snapshot update after a full
backup. All 22 passages, player inputs, response versions, raw attempts and
transcript bytes were preserved; other sessions were unchanged. The server was
restarted at the same local/LAN address. Evidence validation checked all 28 exact
requests and outputs. All 78 non-browser and 17 browser tests passed, along with
prompt rendering/hygiene and Ruff checks.
