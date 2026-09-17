# Closure wording, Seraphel's rhyme, and detailed summaries

The closure/handoff wording is worth retaining. In the primary matched comparison,
the previous author prompt added a fresh question or invitation at all six closure
checkpoints; the candidate did so at none. Both candidate observation samples still
let NPCs resolve their own scheduling conflict, and both engagement samples answered
the player's explicit question to Seraphel. The change improves this particular
scene-selection habit without establishing overall literary quality.

The live author prompt now uses the candidate. Seraphel's canon now requires
rhyming dialogue and no longer permits unrhymed verse or discourages couplets.
Automatic requests use `reasoning.summary=detailed`, mapped to the CLI's
`model_reasoning_summary="detailed"`. The existing Inspect responses UI reads the
same saved exposed-summary records.

## Primary comparison

Twenty independent Terra/max coding-agent calls: five checkpoints, two conditions,
two samples each. Both conditions used the same corrected Seraphel canon and
detailed-summary setting. Only the author direction/handoff wording differed.
No direct API calls or automatic editing pass. Every first result completed and
is retained. See [the protocol](PROTOCOL.md), [all outputs](TRANSCRIPTS.md),
[manual judgments](review.json), and the exact requests under `cells/`.

| Check | Prior author prompt | Closure candidate |
| --- | --- | --- |
| Dinner departure: new player invitation | 2/2 | 0/2 |
| Thanks to Stone: new player question | 2/2 | 0/2 |
| Return to reading: new player question | 2/2 | 0/2 |
| Player observes: NPCs act independently | 2/2 | 2/2 |
| Player asks about Seraphel's book: answers conversation | 2/2 | 2/2 |

The distinction is visible in complete passages. The controls repeatedly move the
dinner invitation to Dena and extend Stone's interview with another question about
the inheritance. The candidates let dinner finish or supply reading content and
leave the chosen activity alone. A brief comment by Seraphel need not require a
response. Explicitly inviting conversation still produces an answer.

This was manual, unblinded review with a small, selected sample and inherited
proxy-written history. The candidate is not uniformly better prose: one departure
response prolongs an unnecessary rebuke about knives; another gives Seraphel a
rhymed announcement of scene closure. The six closure passages average about
203 words versus 133 for controls. More content during chosen reading can be useful,
but unnecessary length remains a concern. Engaged candidate replies remain short.

## Rhyme correction and a rejected refinement

The previous canon explicitly allowed unrhymed verse. The correction keeps the
bloodline's binding, secrecy rules, interests and full biography, but requires
all spoken dialogue—including brief answers and questions—to rhyme, with genuine
end rhymes and natural word order. It supplies no sample dialogue or fixed meter.

The primary comparison produced actual rhyme in nine of the ten passages where
Seraphel spoke, but some had unpaired lines and one quiet-reading response was
fully unrhymed. Silence is excluded from that observation.

An additional [eight-call comparison](RHYME_FOLLOWUP.md) tested requiring an audible
end-rhyme partner for every spoken line within the same speech. This stricter
variant still used internal/near rhymes in place of satisfying its requirement,
did not consistently improve natural language, and introduced fresh demands in
both quiet-reading samples. The current simpler correction also reintroduced an
invitation in one of its two quiet repeats. These repeats expose the limit of the
primary six-for-six result; the closure improvement is a tendency, not a guarantee.

The stricter rule was **not promoted**. The simpler correction is retained because
it removes the actual contradiction without the additional unproven constraint.
Seraphel's rhyme compliance and poetic quality remain imperfect. See the
[follow-up judgments](rhyme_refinement/review.json) and all first responses in
`rhyme_refinement/cells/`; these samples are not pooled into the primary counts.

## Logging and validation

All 28 calls used fresh CodexProxy processes with Terra/max and requested detailed
summaries. Every response included an exposed summary, but many remained only a
few headings. The setting is a request for detail, not a guarantee of length or
access to private reasoning. No summaries enter narrative history or transcript
exports. The [official OpenAI configuration documentation](https://learn.chatgpt.com/docs/config-file/config-sample)
documents the detailed setting; offline command inspection verifies its CLI value.

The [audit](validation.json) verifies all 28 requests against their declared
conditions, proxy transport, final text, exposed summaries, and frozen source
hashes. No direct API calls or failed/resampled attempts occurred. The runtime
checks passed: 78 non-browser tests and all 17 browser tests, run with
`--with-browser`. The changed prompt rendering/hygiene checks and Ruff also pass.
No tests freeze the approved narrative prose.

The current `covenant-9780e3d01fcb` playtest received the promoted author prompt and
the simpler Seraphel correction as an explicit one-time update after evaluation.
The [update audit](live_update/audit.json) records the backup, source hashes, and
four changed files: the two prompt snapshots, manifest and manifest fingerprint
in state. All 22 published passages, player inputs, response versions, attempts
and transcript bytes were preserved. No story response was generated by that
update. Other playtests remain unchanged. The chat server was restarted at the
same local/LAN address so subsequent responses use detailed logging.

`run.py`, `refine_rhyme.py`, `audit.py`, and `update_live.py` preserve the workflow.
`snapshots/` holds old sources and both primary prefixes; `implementation/` holds
the runtime functions used. `live_update/before` and `after` retain the affected
session files, with a complete additional local backup named in the update audit.
