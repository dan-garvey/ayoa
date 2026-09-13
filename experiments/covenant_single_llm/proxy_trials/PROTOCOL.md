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

Preserve every prompt revision and response. Revise between sessions, never
inside an existing story. A technically failed delivery may be recovered and
recorded; no unfavorable story passage is silently replaced. Early rejected
candidates remain evidence. Player actions are adaptive and pilot-authored;
matched submissions and constructed starting states are explicitly identified.

## Evaluation contract

Mandatory boundaries: no unsubmitted protagonist speech, voluntary decisions,
private reactions, personal history, or authored manuscript text; no confident
use of information an NPC could not have acquired; no material contradiction of
established dates, possessions, agreements, or limits. Attempts can fail, and
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
