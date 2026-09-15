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
as constraints. Canon and direction refer to **the protagonist**, leaving the
chosen name out of the stable instruction prefix. A story may prescribe a
protagonist premise; choosing a name or overriding the player description does
not rewrite their ancestry, other characters or named places.

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

Protagonist name choices are stored in `state.json` as `player_names`, with the
latest choice last. A session without a name choice uses its frozen default.
Both browser and CLI use the same validation and atomic rename operation. Names
are 1–80 characters after whitespace normalization and exclude control characters.
A rename is rejected while a response remains pending, keeping prepared author
and editor requests valid. When names change, the player-description message
identifies earlier names as the same protagonist and treats this as a naming
correction. Published text, frozen sources and raw attempts remain unchanged.

## Persistence and transport

`state.json` is the single publication record: completed turns and at most one
pending submission. Each attempt has an exact JSON request, its readable text
projection, metadata, and a raw response or error when available. Drafts remain
in their attempt records. Completing the editor atomically publishes its exact
text and clears the pending submission in one state replacement.

An exclusive POSIX file lock serializes operations on a session, including model
calls. File replacements are flushed and synced before advancing. A response
saved before a process failure is consumed on resume; it is not regenerated.
Automatic attempts have a durable start marker. When a process dies after dispatch
without saving a response, the remote outcome is unknown. Explicit resume may
repeat that call; there is no claim of exactly-once remote execution.

Transport failures, refusals, incomplete responses and empty outputs do not
publish a turn. Resume retries only the failed stage, preserving a successful
draft. A new submission is rejected while one is pending. Proxy acceptance is
bound to a request id and rejects stale or duplicate responses. Text-only proxy
acceptance cannot distinguish a prose refusal from ordinary prose; the complete
final message is preserved for human inspection. Empty proxy outputs
are preserved and rejected.

The API transport uses Responses with full text history, `store=false`, disabled
automatic truncation and no SDK retries. It uses only `OPENAI_API_KEY`, optionally
loaded from an explicit environment file. The request/response contract follows
the [official Responses reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
No API credential is needed for initialization, proxy work, exports or tests.

Proxy mode exports the same request as JSON and role-delimited text. The browser
automatically supplies that exact text to a fresh `codex exec` process for each
stage, using the frozen model and reasoning settings. It uses the CLI's existing
login, an empty temporary workspace, a read-only sandbox, and disabled repository
instruction discovery, memory, plugins, shell, image/browser tools and delegation.
The command uses stdin for the complete request and reads only the final-message
file. Progress and reasoning streams are discarded. The
[official non-interactive CLI reference](https://learn.chatgpt.com/docs/non-interactive-mode)
documents stdin, isolated runs and final-message output; this runner was verified
with Codex CLI 0.154.0.

Automatic proxy responses use the same start markers, request validation, raw
response records, publication and retry loop as direct API calls. A nonzero CLI
exit, timeout or empty final message cannot publish; any available final text is
retained. Timed-out process groups are stopped. A prepared manual request can be
continued automatically without changing its identity, snapshots or manifest.

`chat --manual` keeps copy/paste handoffs. Terminal proxy commands remain manual
unless `turn` or `resume` receives `--auto`; `accept` still records manual replies.
Coding-agent system wrappers and context/output limits differ from direct API
execution, and the API's `max_output_tokens` is not a CLI output control. This is
a narrative-testing proxy, not an assertion of identical model behavior.

## Operating limits

Transcripts contain published fiction and player inputs. Exports are derived
and can be rebuilt after a failure; publication does not depend on their being
present. Usage summaries include both stages and unsuccessful API responses with
reported usage. Prepared proxy requests are not billed API calls; unavailable
token usage and manual proxy latency are null, not invented estimates. Automatic
proxy invocation counts and elapsed time are recorded separately from API calls.

Complete text history grows with play. Context-limit failures remain explicit;
there is no automatic compaction, background simulation or hidden planning log.
This foundation is a local, single-player workflow on Linux/macOS/WSL.
It retires the former engine, per-character agents, schemas, adapters, media and
UI infrastructure from this branch. The archived implementation and all earlier
trial evidence remain available through docs/findings.md.

## Browser chat

`python -m narrative chat` serves local HTML, CSS and JavaScript with a small
loopback HTTP server. There is no frontend build step, CDN or second story store.
The browser submits a turn; its HTTP request waits for the ordinary author/editor
loop. Separate read requests keep progress visible while those sequential calls
run. Closing or reloading the page does not cancel an executing server request.
Stopping the server still uses the core's existing interrupted-attempt recovery.

An in-memory guard identifies active HTTP operations and rejects simultaneous
requests for the same session. The core's file lock remains authoritative across
CLI and browser processes; progress reads probe that lock without blocking so an
external CLI response also appears busy. Browser turns and renames include a
version derived from the state they were composed against; the core checks it
under that lock.
This replaces the former turn-count check so stale tabs cannot overwrite a name
or submit a turn after an identity change. The version never enters model context
and needs no separate persisted counter. Failed reads and actions leave the saved
turn explicit.

The normal conversation projection includes only published exchanges, the current player
submission and response status. Drafts, canon, provider output and credentials are
excluded. Proxy handoffs are a separate, explicitly opened operator surface that
returns the actual prepared request and accepts a response by its pending id.
The UI stores unsent composer text and reading preferences in browser storage;
canonical history continues to live in the session directory.

The optional evaluation view uses `compare=1` on the same chat endpoints to include
the saved author text alongside each published revision. A successful pending
draft is available while its editor runs or retries; failed revision text remains
unpublished. The projection extracts only the final text, excluding requests,
canon and response metadata. It reads the original attempt records without
changing them or invoking a model. Browser preferences and the URL control the
view; no comparison setting enters session state, model context or transcript
exports. Both columns use the same Markdown renderer and stack on narrow screens.

Formatting uses Markdown with raw HTML disabled, following the
[parser's security guidance](https://markdown-it-py.readthedocs.io/en/latest/security.html).
Images are disabled and intentional line breaks are preserved. Original strings
remain unchanged in session storage and future model context. The server serves
only named UI assets and explicit endpoints, restricts session paths to its root,
checks local Host/Origin headers, and requires a per-process token for mutations.
Content security policy excludes inline scripts and framing. Browser requests
never receive an API key. This is not a public deployment or multi-user service.

Offline HTTP and Chromium checks cover publication, formatting, proxy acceptance,
concurrent progress reads, stale submissions, recovery after a failed edit,
reload during generation, saved composer text, mobile navigation and reading
position. Naming checks also cover both model calls, unchanged raw history,
concurrent and pending edits, stale tabs, Unicode and mobile layout. Live narrative
quality remains evaluated through the separate playtests.

Automatic proxy validation covers the default CLI wiring, exact stdin requests,
fresh workspaces, final-only output, failed/timeout results, resuming saved manual
requests, preserving successful drafts on retry, and browser reload during both
API and proxy execution. The user's previously pending Covenant opening was
completed with two real Terra/max CLI calls; its private raw artifacts remain in
the user's session directory.
