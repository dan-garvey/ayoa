# Longer periodic-checkup evaluation

Matched Covenant and Breakwater playtests of the periodic private adherence and
development checkup, using the final prompts and runtime from active commit
`661b41c`. All model calls use fresh Terra coding proxies at maximum reasoning
effort with detailed exposed summaries. No direct API calls or editing passes.

Result: mixed narrative benefit, useful real diagnoses, and repeated overcorrection
of consistent new history. The prose review narrowly preferred Covenant with
checkups and Breakwater without them. Production prompts and settings were unchanged.

The [evaluation](review.md) gives the conclusion and concrete comparisons. The
[checkup review](checkup_review.md) examines each private review against canon,
the preceding fiction and the following passages.

## Design and evidence

- [Protocol](PROTOCOL.md): fixed before generation; review criteria and limitations.
- [Player inputs](inputs.json): the identical 24-message script used in each pair.
- [Sources](sources): unchanged author/checkup prompts, direction and full biographies.
- [Provenance](provenance.json): frozen runtime, sources, inputs and pre-run live-session hashes.
- [Condition mapping](conditions.json): randomized A/B assignment, withheld from prose review.
- [Prose observations](prose_review.md): written before opening the mapping or private notes.
- [Review freeze](review_before_unblind.json): timestamp and hash of those observations.
- [Full transcripts](TRANSCRIPTS.md): all shared openings and both continuations.
- [Full checkups](CHECKUPS.md): all eight private notes in order.
- [Validation](validation.json): offline exact-request reconstruction and output checks.
- [Descriptive metrics](descriptive_metrics.json): lengths, question endings, summaries and timings.
- [Sessions](sessions): exact requests, raw first responses, exposed summaries and durable state.
- [Events](events.jsonl) and [progress](progress.json): actual call/publication chronology.

Each story shares its first four generated passages, then branches into checkups
every five messages versus disabled. Each arm reaches message 24; enabled arms
have checkups before responses 5, 10, 15 and 20. That is 88 unique author calls
and eight checkups, not 96 independently generated story passages. Copied prefix
responses remain in each session for complete inspection and are deduplicated in
the audit.

This is two paired trajectories, not enough independent samples to estimate a
general effect size. Subsequent histories diverge. Runtime timing may suggest
condition assignment, so the pre-note prose review is only partially blinded.
New history is allowed when consistent. Automatic word/question counts are
descriptions, not literary scores.

## Reproduction and preservation

`run.py` is the durable runner used for the experiment. It retains first outputs,
limits independent calls to three concurrent processes and refuses automatic
retry after unknown or failed outcomes. Within a due turn, the author waits for
the checkup and receives its new notes. Only the latest notes are included in
later requests; the notes do not become published history.

The frozen runtime is under `runtime/`. The runner received only non-semantic
formatting/lint cleanup after launch; the copied runtime, model input sources and
input script remained hash-frozen. Existing live stories were never submitted to
the runner. Do not rerun completed model calls for a preferable sample.

`audit.py` is offline and reconstructs requests from the stored session snapshots.
It verifies model/effort/summary settings, exact published responses, checkup cadence,
latest-note selection, shared prefixes, retained first attempts and exclusion of
notes from transcripts. It also compares the pre/post live-session hashes.

From this branch root, using the active checkout's Python environment:

```sh
/home/dan/ayoa-worktrees/covenant-single-llm/.venv/bin/python backstage_long_20260918/audit.py
```

The helper `read_prose.py STORY START END --lane A|B|seed` prints only published
prose and player inputs, allowing review before inspecting private guidance.
