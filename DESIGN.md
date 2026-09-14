# Single-author narrative foundation

The model sees the whole fictional world. Individual NPCs must act only on
plausibly acquired knowledge. These are prompt obligations, not structural
secrecy guarantees. The implementation has one story author, covering world
action, NPC dialogue and narration, and a subsequent prose edit.

## Sources and context

`prompts/author.txt` holds reusable writing instructions. Each story has
`direction.md` for genre, portrayal and opening direction, `canon.md` for rich
character backgrounds, world facts, secrets and initial circumstances, and
`player.json` with a default name and description. Psychological dispositions,
abilities and genuine knowledge restrictions remain canon even when expressed
as constraints. A story may prescribe a protagonist premise; overriding the
player file does not automatically rewrite that premise.

Initialization freezes these files plus `prompts/revision.txt` and model settings.
It makes no model call. The first submitted input requests the opening. Source
and settings hashes are checked on subsequent operations; changed snapshots
require a new session. This is an experimental format without save migration.

The instruction prefix is author instructions, story direction, then canon.
The user-message tail begins with the player description, followed by all
published player/assistant exchanges and the current submission. The author
produces a draft. The editor receives exactly that context, then the draft as
an assistant message and the concise revision task as a user message. Both calls
use the same model and reasoning settings. They are sequential: the editor waits
for the draft. Discarded drafts and revision requests never enter later turns.

## Persistence and transport

`state.json` is the single publication record: completed turns and at most one
pending submission. Each attempt has an exact JSON request, its readable text
projection, metadata, and a raw response or error when available. Drafts remain
in their attempt records. Completing the editor atomically publishes its exact
text and clears the pending submission in one state replacement.

An exclusive POSIX file lock serializes operations on a session, including API
calls. File replacements are flushed and synced before advancing. A response
saved before a process failure is consumed on resume; it is not regenerated.
API attempts have a durable start marker. When a process dies after dispatch
without saving a response, the remote outcome is unknown. Explicit resume may
repeat that call; there is no claim of exactly-once remote execution.

Transport failures, refusals, incomplete responses and empty outputs do not
publish a turn. Resume retries only the failed stage, preserving a successful
draft. A new submission is rejected while one is pending. Proxy acceptance is
bound to a request id and rejects stale or duplicate responses. Text-only proxy
acceptance cannot distinguish a prose refusal from ordinary prose; the operator
must supply the agent's complete final output and inspect it. Empty proxy outputs
are preserved and rejected.

The API transport uses Responses with full text history, `store=false`, disabled
automatic truncation and no SDK retries. It uses only `OPENAI_API_KEY`, optionally
loaded from an explicit environment file. The request/response contract follows
the [official Responses reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
No API credential is needed for initialization, proxy work, exports or tests.

Proxy mode exports the same request as JSON and role-delimited text. A coding
agent reads the text and supplies a final response through `accept`. Tool access,
system wrappers and reasoning controls differ from direct API execution: this
is a narrative-testing proxy, not an assertion of identical model behavior.

## Operating limits

Transcripts contain published fiction and player inputs. Exports are derived
and can be rebuilt after a failure; publication does not depend on their being
present. Usage summaries include both stages and unsuccessful API responses with
reported usage. Prepared proxy requests are not billed API calls; unavailable
token usage and proxy latency are null, not invented estimates.

Complete text history grows with play. Context-limit failures remain explicit;
there is no automatic compaction, background simulation or hidden planning log.
This foundation is a local, single-player terminal workflow on Linux/macOS/WSL.
It retires the former engine, per-character agents, schemas, adapters, media and
UI infrastructure from this branch. The archived implementation and all earlier
trial evidence remain available through docs/findings.md.
