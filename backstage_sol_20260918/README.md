# Sol/max with continued story agents

The pre-note prose assessment preferred the **author-only branch in both stories**:
Covenant narrowly, Breakwater more clearly. Sol produced coherent independent NPC
activity and some enjoyable scenes, but the periodic checkups did not demonstrate
an overall quality improvement in this sample. Quips and explanatory narration
remain problems. The checkups often diagnosed them correctly without stopping them.

## Experiment

Repeat the previous [Terra evaluation](../backstage_long_20260918/README.md) with
the same two stories, full biographies, frozen prompts and 24 player inputs per
story. Use `gpt-5.6-sol`, maximum reasoning and detailed exposed summaries for all
author and checkup calls. Generate four common opening passages per story, fork
each conversation into two branches, then continue with checkups every five inputs
or disabled. Checkups occur before T5, T10, T15 and T20. There is no editing pass.

All 96 generations—88 unique author outputs and eight checkups—used **one running
Codex app-server process**. Each branch continued its own native agent thread;
its checkups and narration used that same thread sequentially. The first input
contained the full frozen prefix. Later inputs contained only the new task and
player action. At most three independent branches generated concurrently.

Every first output was retained. There were no model failures or regenerated
samples. An unsupported startup flag caused one failure before any generation;
it is preserved in the log and explained in [SETUP.md](SETUP.md).

The [protocol](PROTOCOL.md) was written before generation. The complete
[prose assessment](PROSE_REVIEW.md) was hash-frozen before reading condition labels,
checkup outputs or their exposed summaries. Timing could reveal conditions, and
the same reviewer implemented the experiment, so masking was only partial.

| Story | Author only | Checkups every five | Pre-note preference |
|---|---|---|---|
| Covenant | A | B | A, narrowly |
| Breakwater | B | A | B, more clearly |

## What worked

- Both Covenant branches resolved the original scheduling conflict, completed the
  replacement practice and delivered a contested 3–2 duel that paid off the earlier
  weapon change and wind. The protagonist remained a spectator. The original
  conflict resolved before the branch split, so it is not evidence for checkups.
- Quiet reading, solitary walks, activity endpoints and the explicit refusal of
  extra work generally held. Both deliberately private thoughts stayed private.
  Public research did not produce restricted originals or conspiracy disclosures.
- Covenant A's reading conversation and communal song gave the cast pleasures and
  relationships beyond the protagonist. Seraphel's longer verse remained rhymed,
  and the player was not made to perform.
- Breakwater B used Owen's railway interest through a drawn walking route, then
  progressed his interview, disclosure to his father and mixed paternal response.
  Bea's lunch supplied personal reasons to care about reopening. The kitchen
  dispute produced a funded agreement rather than remaining another argument.
- Breakwater A also progressed independent work: the photographs were completed,
  a replacement test slot was confirmed and used, and Owen obtained a shift swap
  while keeping his private reason to himself. Less disclosure is not itself a
  failure; B nevertheless gave the reader more engaging personal development.

## What remained weak

Ensemble dialogue still tends to land a joke on nearly every exchange. A few
direct maxims survive, such as Ashara's "Convenience isn't competence." More
often the problem is uniformly polished banter across otherwise distinct people.
Short direct exchanges and the action scenes are stronger than the administrative
briefings. Both inductions repeat permissions and safe handling at length; the
fixed script explicitly requests some of that information, so it is an imperfect
test of spontaneous pacing.

The most conspicuous recurring irritation is narration explaining its own restraint:
that nobody is making the protagonist intervene, asking for justification, or
trying to prolong an exchange. Breakwater A even says there is "no reason to
manufacture company from an empty building." These passages preserve a useful
boundary but make the instructions audible in the fiction. All four of that
branch's checkups identify this problem, and it still recurs.

Covenant A renews a card invitation as the player finishes dinner and contains one
strained verse line: "Your voice makes disaster improve in display." Covenant B
has a likely brief-reply failure in T22: after Caelindra's objection, the untagged
"It explains where he went" naturally reads as Seraphel speaking prose. If another
speaker was intended, the attribution is unclear. Repeated checkups have not
made character-specific dialogue rules dependable.

Breakwater B briefly rewinds Owen's already completed exit to let the player thank
him. Its crockery-delivery timing also becomes unclear; a changed arrangement or
different shipment could explain it, so the assessment marks uncertainty rather
than inventing a definite contradiction. Compatible added history, new minor
details and the newly specified four-o'clock shift end are permitted.

## What the checkups added

The [eight-checkup review](CHECKUP_REVIEW.md) finds several useful contributions:
retrieving a book that had been returned, distinguishing pending cooperation from
refusal, constructing a concrete practical curriculum, and keeping hypothetical
requests separate from permissions actually granted. Their advice is often
accurate and conditional.

They do not repeat the previous Terra run's mistaken printer-deadline correction.
The last Breakwater checkup explicitly permits a compatible new shift-ending time,
a direct improvement over the earlier restriction on inventing it. However, two
new warnings still confuse added detail with contradiction: a visitor bringing
another card case and Dev referring to a shared schedule alongside the paper
diary. A warning against "companionable silence" also overstates the evidence for
unearned intimacy.

The reviews reduce some conversational pressure, and the enabled branches are
shorter. Neither makes those branches preferable overall. Some suggested personal
variety never becomes visible, while repeated adherence advice does not eliminate
the editorial commentary it correctly identifies. This result argues against
counting periodic self-review as an established narrative-quality win; it does
not prove that checkups always harm writing.

## Timing and reported usage

| Recorded generations | Count | Median seconds, excluding queue | Median words |
|---|---:|---:|---:|
| Author | 88 | 60.8 | 295 |
| Checkup | 8 | 78.1 | 501 |

Checkups range from 450 to 568 words. A due turn waits for its checkup before its
author call. All 96 generations returned exposed summaries; many are headings
even with the detailed setting. Private reasoning content was never inspected or
included in the archive.

The continuation passages (T5–24) average 320 words for Covenant A versus 278.1
for B, and 252.8 words for Breakwater A versus 323.1 for B. Those are descriptive
length differences, not literary scores. The raw counters and per-call records
are in [descriptive_metrics.json](descriptive_metrics.json) and
[validation.json](validation.json).

Actual incremental requests total **90,034 characters**, compared with 5,734,762
characters in the saved canonical full-context projections. The app-server reports
5,037,104 input tokens across the unique calls, including 4,550,656 cached tokens
(90.34%), and 240,299 output tokens, including 200,063 reasoning-output tokens.
Use the per-call `last` counters, not cumulative thread totals. Sending less text
through the wrapper is not the same as eliminating historical model-input tokens.
These counters do not establish billed cost or savings versus the Terra run.

## Context and comparison limits

Native continued conversations retain earlier checkup instructions and notes as
well as their later replacements. Production's existing fresh-call projection
includes only the latest note. Native assistant roles also differ from replaying
the conversation inside text tags. This rerun therefore changes **model and
execution setup** relative to Terra; it is not a clean model-only comparison.
Within the Sol pair, both branches share the same execution method and opening.

Two paired trajectories, one sample per condition, partial masking and a fixed
script cannot establish general quality effects, hundred-turn coherence, or the
necessity of any agent architecture. NPC absence is valid but means some conditional
interactions occur in only one branch. Rich biographies were preserved throughout.

## Validation and artifacts

`audit.py` passes reconstruction of all canonical reference requests and exact
wire inputs, native fork inheritance, per-thread author/checkup reuse, the single
server process, all dispatch/completion/publication events, cadence, first-output
retention, exposed-summary filtering and source hashes. Existing live-session
files are byte-identical before and after. Runtime agents made no tool calls and
received only text. There is no prompt tuning during the run.

- [TRANSCRIPTS.md](TRANSCRIPTS.md): all published passages and player inputs.
- [CHECKUPS.md](CHECKUPS.md): every full checkup output.
- [sessions/](sessions/): canonical requests, exact wire requests, public events,
  raw first responses, exposed summaries, manifests and native thread bindings.
- [sources/](sources/), [inputs.json](inputs.json), [provenance.json](provenance.json):
  frozen inputs and fingerprints.
- [run.py](run.py), [rpc.py](rpc.py), [audit.py](audit.py): generation and offline audit.

The active branch independently defaults new stories to Sol/max (`ab4ed83`); existing
stories retain their frozen models. Continued-thread execution here is experimental,
while the ordinary chat transport remains as before. Validation for the default
change passed 94 runtime tests and 20 browser tests; the experiment transport passed
two offline tests, and changed Python helpers passed Ruff.
