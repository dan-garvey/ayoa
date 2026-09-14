# Standalone foundation evidence

This directory records the implementation smoke playtests for the standalone
foundation on `codex/covenant-single-llm`. The original research remains unchanged
in `experiments/covenant_single_llm/` at archive commit
`d6865cbd6c3b7c9775a57fecb1f8c34adfd1db2b`.

The foundation implementation used for these runs is commit `e937063`.
The [protocol](protocol.md) fixes an opening and three player turns per story,
with a fresh Terra/max author and editor for each passage. Root chooses player
actions from published scenes and reviews the results; this is neither a blind
comparison nor qualification of the broader literary objective.

## Read the results

- [Qualitative review](review.md): concrete improvements, regressions and limits.
- [Covenant published transcript](covenant/session/transcript.md).
- [Breakwater published transcript](breakwater/session/transcript.md).
- [Validation](validation.json): reconstructed requests, response hashes and counts.
- [Source migration](source_migration.json): exact separation of story direction
  and restored canon, without summarizing biographies.
- [Source lock](source_lock.json) and [environment](environment.json): tested
  source hashes and dependency versions.
- [Delivery cleanup](delivery_cleanup.json): three trailing spaces removed from
  Covenant canon after the playtests. Frozen sources remain byte-identical, and
  the verifier checks that the delivered change contains no other differences.

## Inspect a draft and its edit

Files use zero-based turn numbers: `00` is the opening and `01`–`03` are player
turns. For each story, `NN.author.txt` and `NN.editor.txt` are the original files
written by the proxies. `NN.stage.final.txt` preserves the captured first public
final; `NN.stage.capture.json` records model, effort, public read coverage and
the relation between those two versions. No private reasoning is included.

The first two author outputs were accepted from their written files, differing
from the public finals only by a trailing newline. All later outputs were
accepted from their exact captured public finals. [launches.json](launches.json)
records which artifact was accepted for every request id. There were no prose
rerolls or corrective instructions after a draft.

Each archived `session/` contains the frozen prompt/story/player snapshot,
manifest, published state, all exact requests, raw response envelopes and exports.
The editor sees the shared context plus its own draft and the exact 40-word
revision task. Future author requests contain only previously published fiction.

`capture.py` and `advance.py` are local run helpers, with the original machine
paths and public agent-log locations retained for provenance. `verify.py` audits
and copies completed sessions. They are evidence of this run, not a second
runtime; the supported runner lives in the active branch's `narrative/` package.
Proxy token usage and latency are unavailable. No direct API call was made.
