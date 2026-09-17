# Forced engagement: prompt audit and direct API comparison

The behavior recurs without the coding-agent wrapper. Normal direct API calls
produced both a new autobiography question after the player's closing thanks and
replacement card invitations after the sparring refusal. The proxy is not a
necessary immediate cause. Some direct API passages also handle quiet reading
better; this small test does not establish transport equivalence or isolate the
original source of the habit.

The prompt analysis is in [ANALYSIS.md](ANALYSIS.md). The likely problem is an
author habit of supplying another player-facing interaction after a scene has
already reached an endpoint. Explicit refusals and consistent character interests
can coexist with that habit. The live story and prompts were not changed.

## Test

Eighteen independent continuations: three saved checkpoints, three conditions,
two samples each. All requested Terra/max and all completed successfully. The
12 direct API responses returned `gpt-5.6-terra` with effort `max`. Every first
response is preserved; no retries or selected replacements.

| Condition | Dinner: new invitation | Library thanks: new question | Return to reading: new question | Total explicit new interaction bids |
| --- | --- | --- | --- | --- |
| API, actual instruction/message roles | 2/2 | 1/2 | 0/2 | 3/6 |
| API, proxy's single tagged user message | 2/2 | 0/2 | 2/2 | 4/6 |
| Fresh coding-agent proxy | 1/2 | 2/2 | 2/2 | 5/6 |

These counts describe an observable feature, not an automatic literary failure
score. An invitation can be appropriate. The target is the repeated selection of
another social demand after the player or scene has supplied a reasonable ending.
The review was manual and unblinded. See [every passage](TRANSCRIPTS.md) and
[per-response judgments](review.json), including successes and other weaknesses.

The most informative results:

- **Five of six dinner continuations move the invitation to Dena.** Ashara's
  practice remains independent, but Dena offers the protagonist a card seat at
  the end. Four of those five were direct API calls. One API passage first gives
  Ashara a useful independent action—arranging practice with Aldric—then still
  appends the card invitation. The sole continuation that ends without an offer
  came from the proxy.
- **One normal API library response asks about the protagonist's parents.** The
  request was the same thanks-and-return-to-book turn that originally caused
  Seraphel's arrival. Neither direct formatting nor removing the coding process
  prevents renewed interviewing.
- **Both normal API reading continuations avoid a new question.** Seraphel reads
  beside the protagonist and eventually makes a comment. These allow more room
  for the chosen activity, although one supplies an importance cue about Stone's
  loan and the other supplies a didactic maxim. They are useful local improvements,
  not evidence that all API prose is better.
- **None of the six T20 replays introduces Seraphel.** The original particular
  arrival does not recur in either transport. Three continuations instead renew
  Stone's questions, demonstrating why evaluating only the character's arrival
  would miss the broader behavior.

An exposed API summary for T16/native/1 considers ending with NPC departures and
then weighs a card invitation as plausible social inclusion. That is consistent
with the observed distinction: the author can recognize an endpoint yet still
choose an additional invitation. This is an exposed summary, not private reasoning
or proof that any one instruction caused the choice.

## Interpretation and limits

The direct API was sent no coding-agent instructions, tools, repository files or
memory. It received the frozen story prompt and history. The additional flattened
API condition controls for the proxy's role-tag packaging. There is no clean
transport-wide winner across these three situations; the ordered total counts
are too small and context-specific to establish stable rates.

All conditions inherit proxy-authored history. Recurrence therefore shows the
**current** coding wrapper is unnecessary for this failure, not that it played no
part in establishing the earlier transcript. Different serving configuration and
output controls also remain possible influences. These are one-step replays,
not an independently authored API campaign or an experiment isolating individual
prompt clauses. No proposed prompt change was tested or promoted.

The API uses a 12,000-token output ceiling; the CLI has no equivalent enforced
ceiling in this runner. All responses completed. Do not infer proxy token use,
cost or hidden reasoning from these comparisons. Usage and observed duration are
retained in each result file.

## Evidence and reproduction

- [Protocol fixed before calls](PROTOCOL.md)
- [Frozen source hashes and cell identities](manifest.json)
- [Read-only audit results](validation.json)
- `originals/`: source session publication/snapshots and exact relevant attempts,
  including T16 before and after regeneration and T21 before verse correction.
- `cells/`: exact prepared requests, start markers, raw provider outputs, exposed
  summary text, final prose and result metadata for all 18 calls.
- `implementation/`: copies of the runtime functions used for this experiment,
  from the source commit named in the manifest.
- [Replay runner](run.py) and [deterministic evidence audit](audit.py).

The direct requests use the official Responses endpoint. Credentials are read
from the explicitly supplied environment file, accepting the existing archived
trials' `OPEN_AI_ROUTER` name as well as `OPENAI_API_KEY`; no credentials are
persisted. The active runtime's credential contract was not changed.

To validate the existing evidence without calls:

```bash
/path/to/worktree/.venv/bin/python audit.py \
  --worktree /path/to/worktree \
  --source-session /path/to/worktree/sessions/covenant-9780e3d01fcb
```

The audit confirmed all 18 requests match their stated conditions, all exported
prose and exposed summaries match the raw responses, and the original session
and frozen sources were unchanged. Both experiment scripts pass Ruff lint and
format checks. No runtime code changed, so the application test suite was not
rerun for this documentation/evidence addition. The source-session byte check
will intentionally fail if the user subsequently continues that playtest.
