# Capacity-error recovery

The first execution received `serverOverloaded` for sample-05 turn 19 at
20:07:59 UTC, before any reasoning item or final prose. Its full failed request,
response and public error events are retained. The original controller completes
its already queued first-round calls and stops on that failure; no completed
passage is discarded or regenerated.

`resume.py` continues all existing native thread ids. Before retrying a confirmed
pre-output capacity failure, it reads that conversation, verifies its complete
turn-id prefix and that the only additional turn is the failed user submission
with no reasoning or assistant item, then rolls back that failed submission.
Both before/after public records are saved with the failed attempt. It retries
the exact original input once the context is restored. Unknown outcomes, partial
generations, refusals, model changes and narrative failures are not eligible.
Repeated capacity failures are preserved individually and use a bounded retry.

The transport process is restarted for recovery; native conversations continue.
This departs from the intended single process for the entire experiment, while
preserving the requested same-agent continuation instead of rebuilding a fresh
conversation for each response. Prompts, biography/context snapshots, masked
assignments and player inputs remain frozen. Report infrastructure failures
separately from the first completed narrative samples. Failed responses may carry
stale inherited usage; do not count that usage as new generation tokens.
