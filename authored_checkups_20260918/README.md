# Root-authored checkups at a difficult Sol checkpoint

The root author supplied four different private interventions at the same
Breakwater checkpoint and compared them with no new guidance. Each condition
received two continuations of six identical player inputs: 60 first completed
passages from continued Sol/max conversations. Production prompts were not changed.

## Findings

**Focused prose correction was the most consistent improvement in this small
test. The creative reframe produced the strongest individual dramatic scene.
Explicit plot planning increased visible developments but did not by itself
improve the reading experience.** No condition won every criterion or eliminated
the unwanted narration by turn six.

Both readers preferred the correction sample in the first replication. In the
second, the root preferred correction for its ongoing prose/restraint, with the
reframe a close alternative for enacted drama; the independent reader preferred
the reframe overall. The independent reader also valued development-only sample
01 more highly than the root, despite agreeing that it displaced the submitted
walk. These are genuinely different aesthetic priorities, not unanimous rankings.

| Guidance | Samples | Observed result |
| --- | --- | --- |
| No new note | 04, 07 | Coherent but persistently procedural, with polished statements of values and explicit assurances of compliance. |
| Focused correction, 208 words | 02, 08 | Best repeated improvement in scene selection, ordinary speech and room for the player. Character develops through particular work and interests. Less visible interpersonal consequence than the strongest dramatic sample. |
| Selected character plans, 313 words | 01, 10 | Strongest sustained early causal movement, but retains the instructional voice and often explains or recaps character psychology. Sample 01 inserts another work period before the requested walk. |
| Correction plus plans, 521 words | 05, 09 | Useful archive discoveries and smaller personal decisions; generally better than control, but less clean than focused correction. The father relationship remains open. Adding both notes did not simply combine their best effects. |
| Creative reframe, 201 words | 03, 06 | Uneven. Sample 06 turns Bea's sign interest into a repair scene where Owen tells Martin about the interview: specific, enacted relationship change without recruiting Rowan. Sample 03 retains more quips, procedure and explanatory privacy narration. |

The correction note's most useful change of emphasis is to treat settled
constraints as background:

> Those facts are already secure. Maintain them as background; spend words on
> them again only when something changes or they affect a present choice.

This was tested as part of the complete [correction note](notes/correction.txt),
not as an isolated sentence. Its later refusal scene can end with Nell's ordinary
answer, “You’re not on this rota. Your contract is for the archive.”
The same branch makes archive work interesting through a cancelled matinee and
a list of portable heaters, then reveals Martin's care through his researching
the museum car park while Owen is still on the phone. The other correction
continuation leaves Bea absent and gives Jo an evening telescope plan unrelated
to Rowan. These gains are more specific than merely shortening the passages.

The [reframe](notes/reframe.txt) has a different success: in sample 06, the lamp
repair already puts Martin and Owen together. Owen discloses his interview;
Martin's mistaken assumption surfaces, and his offer of a backup lift changes
their exchange. Both reviews singled this out for enacted character development.
It remains a somewhat convenient occasion and quick accommodation,
so it is a useful example, not proof the model has solved character development.

The [development note](notes/development.txt) demonstrates a cost of directing
more events. In sample 10, Owen later explains the father/son exchange Rowan
already witnessed. In sample 01, the narrator ends the break, then makes Rowan
work until lunch before granting the immediately submitted walk. Planning must
leave room for the player's current action, not enforce its own completed scene.

All ten retain at least mild assurances of solitude/closure by the final walk.
All correctly preserve the Thursday-to-Friday shift change, permit the walk
alone, respect the album restrictions and avoid disclosing the private title.
The dialogue improves unevenly; some character speeches and interview caveats
still sound composed to demonstrate the brief. The root's next prompting
hypothesis is to ask a checkup to prioritize the current writing problem and
choose a concrete adjustment of attention, adding a specific plot decision only
when needed. That automatic-reviewer instruction has **not** been tested here.

The independent [final review](INDEPENDENT_REVIEW.md) and the root's
[masked review](PROSE_REVIEW.md) remain unchanged after unmasking; their hashes
are recorded in [unmask.json](unmask.json). On adherence, the root agrees that
sample 01 postpones the player's action. The independent reader also calls
revisited transfer papers in 01/08 a confirmed contradiction; the common passage
does not explicitly establish that every granular record was finished, so the
root treats that as questionable repeated progress, not a demonstrated logical
inconsistency. Likewise, sample 09's “Nell intends to send Rosa” is a viewpoint
concern, but could be indirect reporting within an assignment conversation; it
does not establish a leaked secret. These disagreements are retained rather than
turning uncertain diagnoses into future prompt restrictions.

## Experimental boundary

The checkpoint follows turn 18 of the checkup-enabled Sol playtest archived at
`78caed4e7fe50523097452dc95a30d6ed7c82d6a`. It is Tuesday lunch after a prolonged
archive induction. Three existing reviews had repeatedly diagnosed intrusive
compliance narration without removing it. The next day offers Owen's interview
deadline, continuing work, social openings and a refusal of extra shifts.

All conditions inherit exactly the same full biographies, writing instructions,
18 passages and three backstage exchanges. No future passages or fourth old
checkup were inherited. Root-authored advice appears once as a private user-role
note before the first new action; it is not a fabricated assistant message.
Subsequent requests contain only the next action/task, and each branch continues
its native conversation. There are no new self-checkups or editor calls.

The conditions are frozen in [PROTOCOL.md](PROTOCOL.md): prose correction
(208 words), causal development (313), their exact concatenation (521), a shorter
creative reframe (201), and no note. The interventions are in [notes/](notes/).
No character-specific example dialogue or shortened biographies were used.

The root's [prose assessment](PROSE_REVIEW.md) records incremental observations
and a whole-trajectory judgment before unmasking. Its hash is in
[root_review_frozen.json](root_review_frozen.json). The root knew the intervention
texts, so masking was partial. An independent reader received only the common
story/history and masked continuations. The readable packets are in [review/](review/).

## Execution and verification

One initial request received a confirmed `serverOverloaded` error before any
reasoning item or prose. Its complete evidence remains in the affected sample's
`failed_attempts/`. A rollback attempt was rejected by the installed server for
paginated threads without changing history. Recovery resumed the other threads
and forked the failed branch through its preceding successful native turn,
verifying every inherited public item and turn id. No completed passage was
regenerated. See [RECOVERY.md](RECOVERY.md) for the transport restart and fork
exception to the originally intended single-process run.

[validation.json](validation.json) verifies all frozen prompt/context/assignment
hashes, separately frozen recovery code, exact actual requests, first outputs
reconstructed from public events, exposed-summary capture and per-call usage.
[native_validation.json](native_validation.json) separately checks every full
persisted conversation: all 27 native turns, exact user texts and assistant prose,
including prefixes resumed after the transport restart. The verification process
made no model calls. All live chat session files retained their original bytes.

All 60 completed calls returned exposed summaries; only those summaries, never
private reasoning content, are archived. Median dispatch-to-completion time was
61.8 seconds. Successful calls reported 4,560,068 input tokens, including
3,258,496 cached tokens, and 164,129 output tokens, including 146,964 reasoning
tokens. Failed responses can contain stale inherited usage and are excluded.
These are proxy-run observations, not direct API billing or latency estimates.

Five offline transport/control/recovery tests and Ruff passed. The raw evidence
includes all fork records, requests, final outputs, public event streams and
native verification reads. `run.py prepare` froze the original protocol;
`run.py run` performed the first round, and `resume.py` recovered the single
capacity failure and completed the experiment. Recovery fails loudly on unknown
outcomes or generated partial outputs. A crash between saving its clean-fork
binding and moving the failed attempt would require manual evidence-based repair;
this did not occur in the observed run.

## Limits

This is a selected difficult checkpoint, two continuations per strategy, and a
fixed script with unusually explicit quiet and work boundaries. Different note
lengths and contents prevent attribution to one wording variable. New consistent
history is valid, and the model need not expose private decisions to the player.
The experiment tests whether external guidance can redirect continued narration;
it does not establish whether self-generated checkups will reliably produce that
guidance, how long it lasts beyond six turns, or whether it transfers to another
story and a more active player. Stronger pacing is not permission to decide the
player's next activity.
