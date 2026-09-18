# Capacity-error recovery

The first execution received `serverOverloaded` for sample-05 turn 19 at
20:07:59 UTC, before any reasoning item or final prose. Its full failed request,
response and public error events are retained. The original controller completes
its already queued first-round calls and stops on that failure; no completed
passage is discarded or regenerated.

The first recovery tried native `thread/rollback`, which the installed server
rejected for paginated threads. That call made no generation or history change;
its controller, log, and planned recovery description are in `recovery_initial/`.

`resume.py` resumes existing native threads. Before retrying a confirmed
pre-output capacity failure, it reads that conversation, verifies its complete
turn-id prefix and that the only additional turn is the failed user submission
with no reasoning or assistant item, then forks through the immediately preceding
successful turn. All inherited public items and turn ids must match. This creates
a new native thread id only for a failed branch, with its full conversation
preserved; it does not rebuild a fresh prompt. Both before/after public records,
the fork request and previous binding are saved with the failed attempt. It retries
the exact original text and settings after recovering the context. Unknown outcomes, partial
generations, refusals, model changes and narrative failures are not eligible.
Repeated capacity failures are preserved individually and use a bounded retry.

The transport process is restarted for recovery; native conversations continue,
with the failed branch following the recorded fork chain.
This departs from the intended single process for the entire experiment, while
preserving the requested same-agent continuation instead of rebuilding a fresh
conversation for each response. Prompts, biography/context snapshots, masked
assignments and player inputs remain frozen. Report infrastructure failures
separately from the first completed narrative samples. Failed responses may carry
stale inherited usage; do not count that usage as new generation tokens.
