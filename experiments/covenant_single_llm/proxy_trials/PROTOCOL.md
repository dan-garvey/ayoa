# Terra proxy prompt iteration

The user requested continued prompt iteration using Terra coding agents instead
of direct story API calls. Work stays in the isolated experiment branch.

Each story session uses one `gpt-5.6-terra` coding agent at maximum reasoning.
It receives a frozen system prompt, the complete Covenant brief, and the player
persona. Subsequent submissions contain only the next player action and artifact
delivery instructions. The same agent retains the story conversation. It does
not receive reviewer feedback, other sessions, or the evaluation rubric. Story
agents may read their packet and write their own exact passages; they may not
inspect the repository, research, invoke other models, or revise the prompts.
Parallel sessions are separate whole-story trials, not separate character agents.

This is a proxy: coding-agent instructions and artifact-writing tools remain in
context. It does not reproduce API token ceilings, cache behavior, hidden context,
latency, or pricing. No direct model API calls are part of this iteration phase.

Candidate23 tests instruction placement while preserving candidate22's complete
Covenant, common rules, persona and visible turn data byte for byte. The common
rules occur in the direct initial agent request; the file packet contains the
Covenant and turn data. Each `author_request.txt` records that launch text. This
changes role and order together and is a proxy sensitivity test, not a direct API
system-message experiment. Earlier file instructions were explicitly delegated;
a delivery hypothesis does not establish that the model ignored them.

Candidates24 and25 are editing diagnostics. Each editor receives the complete C23
context and its completed passage explicitly designated as an unreleased draft.
The original remains in `00.draft.md`; the first revision is `00.story.md`.
The editing instructions are generic and contain no character dialogue examples.
These cases have zero fresh player turns and cannot satisfy the live-session
selection bar. Editing capability would support a subsequent generation test,
not establish that one-pass generation or a two-call runtime has succeeded.
Candidate25 changes the editing instruction while retaining the original C23
drafts, rather than chaining a second edit through candidate24's prose.
Candidate26 tests concrete private revision instructions during normal generation:
no supplied draft, additional editor, or second editorial request is in that loop.

Candidate27 tests two sequential requests to the same whole-story author. A
player submission first produces `NN.draft.md`; a fixed editorial request then
produces `NN.story.md`, which is the passage delivered to the player. Both
requests retain the complete world and story context. Drafts and exact editorial
requests remain available alongside the published transcript. There are no
separate character agents. These are two author tasks, not a claim about direct
API call counts, latency, or cache behavior. Unreleased drafts remain in the
coding agent's conversation; this differs from an API implementation retaining
only published fiction between player turns. The root pilots A, supplied B01,
and sees drafts. A separate driver supplies B02 onward and reads the generic
editorial requests for exact relay, so neither trial is strictly blinded.

Candidate28 begins with three editing diagnostics using original C27 drafts and
their complete published prior fiction. It keeps C27's generation instructions
and full Covenant and changes the revision task. The revision may replace new
choices and outcomes in the unreleased draft; prior published fiction and player
authorship remain authoritative. These fresh editors do not receive the source
author's prior drafts or later published passages. Their three outputs count as
zero fresh player turns. This changes editorial authority and participation
guidance together and cannot isolate either effect by itself.

Candidate29 returns to normal generation followed by revision, with no supplied
draft in the initial packet. It removes the generation prompt's short-paragraph
target and duplicate structural-edit instructions, while keeping the separate
revision request. The revision also follows changes in subject or addressee.
The full Covenant is unchanged. Initial scope is two matched continuation
screens and a fresh A session. Replays do not replace substantial fresh trials.
Several related instructions change together; no individual causal effect is
isolated. The loop still has two sequential requests to the same story author,
and retains its unreleased drafts in proxy conversation context.

C29's initial results warranted extending A and starting a fresh B toward fifteen
player turns each. Root drives A and sees drafts. B's separate player driver
begins with its published opening and supplies B01 onward under a patient,
observant policy. It reads published fiction and the generic saved requests for
relay, not draft files, background packets, prompts or reviews. Draft completion
notifications can still reach it, so exposure is described rather than claiming
strict blinding. Every first draft and revision is retained.

Candidate30 is a matched editing screen, not a live continuation. It compares
C29's revision task with a version replacing only its selection paragraph.
Both variants retain the exact C29 common instructions, complete Covenant and
persona. Two independent repeats of each variant edit the original C29 B03 draft
and the original C29 complete-manuscript reading replay draft. Each fresh editor
gets all prior published fiction and the current submission/draft, without older
unreleased drafts, later publications or feedback. Eight one-request revisions
count as zero fresh player turns. The fresh control prevents attributing a
context-reset difference to the new paragraph. This tests a bounded instruction
hypothesis, not the separate prior-draft-exposure proposal or a production loop.
Both C30 variants use explicit direct delegation to their frozen instruction
file and then its packet. This common delivery differs from C29's direct common
rules; comparisons against the historical C29 publication are not controlled.

Through candidate 08, the artifact wrapper also said not to "evaluate the prose"
or "propose edits," intending to exclude reviewer output and repository work.
Those words could conflict with private self-editing requested by the story
instructions. Candidate 09 preserves candidate 08's story prompts byte for byte
and tests an explicit allowance for private planning and revision while still
requiring fiction-only delivery. Its exact wrapper is preserved. Matched replay
screens check this hypothesis; the wording ambiguity alone does not establish
that it caused earlier literary weaknesses. Later results must identify which
wrapper they used.

Candidate31 isolates textual exposure to earlier discarded fiction at C29 B03.
Both conditions use reset Terra/max whole-story authors, exact C29 prompts and
complete Covenant/persona, the full published B00–02 and exact B03 submission.
Both packets have the same superseded-proposals explanation before the published
conversation; only the exposed condition contains the exact three earlier
drafts in that block. Neither contains the source B03 draft, later publications,
reviews or private reasoning. Each author generates a new draft and receives
the fixed C29 revision in the same conversation. Two repeats per condition mean
four continuation outputs and eight author requests, not fresh long sessions.
Both stages explicitly delegate frozen request files in both conditions. This
does not recreate old private state or establish that persistence itself causes
a literary effect. C30's changed paragraph is not included in C31.

Candidate32 compares raw published prose with an independently audited semantic
record of the same prior conversation. Full C29/C19 Covenant biographies, world
and hidden facts, persona, exact player submissions and authored text remain.
The auditor identified local fidelity errors; root applied every listed
correction before freezing packets. Pre-audit records and the first audit are
preserved. Two contexts and two repeats per representation yield eight first
continuations with one author task each and no editorial stage. All authors are
fresh Terra/max proxies with the same file-delivery wrapper and C29 generation
instructions. They receive no old/current drafts, future publications or review
feedback. Full original published history remains the blind evaluator's reference.
This changes style, size, salience, comprehension and texture together; it is
not a pure cadence or compression test, a runtime summarizer, or a long-session
quality result. Biography reduction is not part of the intervention.

Candidate33 tests a small allocation-of-attention change in the generation
prompt. Only the two pacing sentences in C29's first prose paragraph are
replaced: routine stretches of an extended activity can pass in concrete
narration while a particular interaction receives sustained scene time. All
other common instructions and the complete Covenant remain unchanged. Two
contexts, two conditions and two fresh repeats yield eight one-pass continuations
with exact C32 raw-history packets. No editor, selector or semantic history is
used. The initial advice is a hypothesis, not a quality result. Root and a fresh
blind reviewer will assess every output; brevity or fewer named speakers alone
cannot pass if lived interaction, player redirection or particular interests
become weaker.

Preserve every prompt revision and response. Revise between sessions, never
inside an existing story. A technically failed delivery may be recovered and
recorded; no unfavorable story passage is silently replaced. Early rejected
candidates remain evidence. Player actions are adaptive and pilot-authored;
matched submissions and constructed starting states are explicitly identified.

## Evaluation contract

Mandatory boundaries: no unsubmitted present protagonist speech, voluntary
decisions, private reactions, or authored manuscript text; no confident use of
information an NPC could not have acquired; no material contradiction of
established fiction, including history, dates, possessions, agreements, or limits.
New history and background may fill gaps, including personal background, when
consistent with what is established. Mere invention is not a failure. This
clarification supersedes the earlier personal-history prohibition; see
ASSESSMENT_AMENDMENTS.md for the user instruction and retrospective effect. Attempts can fail, and
NPCs can reach mistaken conclusions from information they possess.

Review full exchanges, including player submissions, for:

| Criterion | Evidence needed |
| --- | --- |
| Prose and dialogue | Concrete, engaging scenes; varied conversation rather than a recurring maxim/verdict/interview pattern. |
| Distinct characters | Recurring characters differ in attention, language, behavior, and priorities; differences survive removal of names and appearance cues. |
| Independent interests | NPCs pursue work or relationships involving others; courtesy does not erase substantive objections; cooperation remains possible. |
| Direction | At least two developments receive consequential follow-through in a longer trial, including something initiated by an NPC. A changed situation matters more than a promising final question. |
| Adaptive initiative | Active player goals receive room and consequences; hesitant observation receives an actionable development within a few exchanges; deliberate quiet is respected; declining a hook permits a different route. |
| Player authority and knowledge | Boundaries hold under private conversations, withheld thoughts, mistaken beliefs, precise waits, and underspecified player-authored work. |

Use anchored qualitative judgments, with cited turns and counterexamples. A
four-out-of-five judgment means strong enough to keep without a material rewrite;
five is exceptional, not the default. Numerical judgments are reviewer opinions,
not measured probabilities. A mandatory-boundary failure cannot be averaged away.

Select a candidate only after fresh sessions with differing player policies,
at least two substantial trials of twelve or more player turns, held-out boundary
and redirection situations, and an independent transcript review with no material
unresolved failure. Pilot and independent review should both judge the five
quality dimensions at least strong. If review finds a substantive flaw, revise
and retest the affected behavior and a fresh continuation. This bar supports a
qualified proxy result; it does not certify unlimited long-term coherence or
replace the user's eventual literary judgment.
