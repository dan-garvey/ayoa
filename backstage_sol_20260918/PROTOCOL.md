# Sol with continued agent conversations

Frozen before the first model generation. Repeat the previous 24-input Covenant
and Breakwater scripts, full canon/biographies, direction and prompts from
`84de4738e8e71c04e11b7a6a60caf661befac590/backstage_long_20260918`. Runtime storage
and canonical request construction use its frozen runtime, with explicit model
`gpt-5.6-sol`, effort `max` and exposed summary `detailed`.

One long-lived Codex app-server process hosts two independent initial story
threads. Each produces four common passages. Fork each completed conversation
at that point into A and B, randomly assigned checkups every five player inputs
versus none, and continue each through input 24. The same thread handles a branch's
checkups and authorship. No new process or conversation is created per response.
Maximum three independent in-flight calls; within a branch all calls are sequential.
Expected total: 88 unique author calls and eight checkups, 96 generations.

The first request supplies the complete frozen prefix, identity and opening
input. Following requests supply only the new player input. A checkup request
adds the unchanged checkup task after that input. The following author request
asks for the next passage and repeats the pending player input. Canonical full
requests remain saved as reference projections; `wire_request.json` and
`wire_request.txt` record exactly what was actually sent. Public outputs and
exposed summaries are retained, never private reasoning content.

Continuing conversations retain earlier checkup tasks and their assistant notes.
They are private advice, not published fiction or character knowledge. This is
an intentional change from the earlier fresh-process/latest-note projection.
Native assistant turns and retained agent context also differ from full histories
replayed inside role tags. Therefore comparison with Terra is a comparison of
model plus execution setup, not a clean isolated model effect. The within-Sol
checkup/no-checkup comparison still has a common execution setup.

No prompt tuning, regenerated samples, editor model, direct API transport,
automatic retry or change to existing live stories. Retain every first result.
Write-ahead wire requests and public event logs allow recovery. A call with an
unknown outcome stops its lane; inspect the saved thread before explicitly
recovering rather than generate again. Preserve all failed attempts if any.

Review all prose before opening checkup notes or the condition mapping where
practical, then freeze that assessment. Runtime timing can suggest conditions;
this is not an independent fully blind review. Assess complete scenes: dialogue,
NPC independence, direction and freedom, genuine contradictions versus allowed
invention, private knowledge, player authorship, specific dialogue rules, checkup
precision and observable carryover. Inspect every checkup against exact sources
and fiction, especially invented printer-deadline faults, stale findings and
overrestrictions on consistent new history. Automatic counts are descriptive aids.

The same fixed-script limitations apply: explicit quiet/non-intervention, possible
repeated dinner endings, conditional interactions sometimes not occurring, and
only two paired trajectories. Do not claim general effect sizes or hundred-turn
coherence. Capture reported usage and timings, without assuming a reduction in
billed tokens from conversation reuse alone.

Protocol reference: [official Codex App Server documentation](https://learn.chatgpt.com/docs/app-server).
Local CLI-generated schemas verify `thread/start`, `thread/fork` and `turn/start`.
