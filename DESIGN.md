# Single-author narrative foundation

The model sees the whole fictional world. Individual NPCs must act only on
plausibly acquired knowledge. These are prompt obligations, not structural
secrecy guarantees. The implementation has one story author, covering world
action, NPC dialogue and narration. A successful response publishes directly;
there is no automatic editorial call.

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

Initialization freezes these files plus `prompts/regenerate.txt` and model settings.
It makes no model call. The first submitted input requests the opening. Source
and settings hashes are checked on subsequent operations; changed snapshots
require a new session. This is an experimental format without automatic save migration.

The instruction prefix is author instructions, story direction, then canon.
The user-message tail begins with the player description, followed by all
active player/assistant exchanges and the current submission. One author call
returns the published passage. On an explicit regeneration, the same author sees
that history including the current final passage, then the short regeneration
instruction and the player's requested changes in the user tail. Its response
replaces only the final passage, retaining that passage's original player input.
Future calls receive only the replacement; discarded responses and regeneration
instructions remain in saved attempts and never enter subsequent history.

Regeneration instructions apply once. Repeated regeneration sees the current
version and the new instructions without accumulating earlier feedback. The base
writing contract remains the place for enduring prompt preferences.

Protagonist name choices are stored in `state.json` as `player_names`, with the
latest choice last. A session without a name choice uses its frozen default.
Both browser and CLI use the same validation and atomic rename operation. Names
are 1–80 characters after whitespace normalization and exclude control characters.
A rename is rejected while a response remains pending, keeping prepared requests
valid. When names change, the player-description message
identifies earlier names as the same protagonist and treats this as a naming
correction. Published text, frozen sources and raw attempts remain unchanged.

## Persistence and transport

`state.json` is the single publication record: completed turns and at most one
pending submission. Each attempt has an exact JSON request, its readable text
projection, metadata, and a raw response or error when available. Each turn keeps
an ordered list of successful response attempt ids; its final id identifies the
active passage. Completing generation atomically publishes its exact text and
clears the pending submission in one state replacement. Completing regeneration
replaces the last turn's output and appends its attempt id in that same write.
The preceding passage remains published throughout a failed or interrupted attempt.

An exclusive POSIX file lock serializes operations on a session, including model
calls. File replacements are flushed and synced before advancing. A response
saved before a process failure is consumed on resume; it is not regenerated.
Automatic attempts have a durable start marker. When a process dies after dispatch
without saving a response, the remote outcome is unknown. Explicit resume may
repeat that call; there is no claim of exactly-once remote execution.

Transport failures, refusals, incomplete responses and empty outputs do not
publish a turn. Resume retries the failed attempt while preserving the original
submission and any existing passage. A new submission is rejected while one is
pending. Proxy acceptance is
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
response, using the frozen model and reasoning settings. It uses the CLI's existing
login, an empty temporary workspace, a read-only sandbox, and disabled repository
instruction discovery, memory, plugins, shell, image/browser tools and delegation.
The command uses stdin for the complete request, reads the final-message file and
captures completed public reasoning-summary items from the CLI's JSON event stream.
Each attempt retains the exact summary strings separately from final prose; other
events are discarded. Both API and proxy requests opt in with `reasoning.summary=auto`.
The API retains the summary items in its raw response. No summary enters publication,
subsequent model input or transcript exports, and absent summaries are not synthesized.
The [official non-interactive CLI reference](https://learn.chatgpt.com/docs/non-interactive-mode)
documents stdin, isolated runs and final-message output; this runner was verified
with Codex CLI 0.154.0.

Automatic proxy responses use the same start markers, request validation, raw
response records, publication and retry loop as direct API calls. A nonzero CLI
exit, timeout or empty final message cannot publish; any available final text is
retained. Timed-out process groups are stopped. A prepared manual request can be
continued automatically without changing its identity, snapshots or manifest.

`chat --manual` keeps copy/paste handoffs. Terminal proxy commands remain manual
unless `turn`, `regenerate` or `resume` receives `--auto`; `accept` records manual replies.
Coding-agent system wrappers and context/output limits differ from direct API
execution, and the API's `max_output_tokens` is not a CLI output control. This is
a narrative-testing proxy, not an assertion of identical model behavior.

## Operating limits

Transcripts contain published fiction and player inputs. Exports are derived
and can be rebuilt after a failure; publication does not depend on their being
present. Usage summaries include all attempts and unsuccessful API responses with
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
The browser submits a turn or regeneration; its HTTP request waits for one author
call. Separate read requests keep progress visible while it runs. Closing or
reloading the page does not cancel an executing server request.
Stopping the server still uses the core's existing interrupted-attempt recovery.

An in-memory guard identifies active HTTP operations and rejects simultaneous
requests for the same session. The core's file lock remains authoritative across
CLI and browser processes; progress reads probe that lock without blocking so an
external CLI response also appears busy. Browser turns, regenerations and renames include a
version derived from the state they were composed against; the core checks it
under that lock.
This prevents stale tabs from overwriting a name or regenerating a passage that
has already changed. The version never enters model context
and needs no separate persisted counter. A submission UUID lets the browser
recognize its own accepted operation after reload, including when regeneration
leaves the turn count unchanged. It lives in pending state, attempt metadata and
the resulting turn, and never enters model context. A rejected stale submission
cannot clear the composer's unsent text. Failed reads and actions leave the saved
turn explicit.

The normal conversation projection includes only published exchanges, the current player
submission and response status. Replaced passages, canon, provider output and credentials are
excluded. Proxy handoffs are a separate, explicitly opened operator surface that
returns the actual prepared request and accepts a response by its pending id.
The UI stores unsent composer text and reading preferences in browser storage;
canonical history continues to live in the session directory.

The optional **Inspect responses** view uses `compare=1` on the same chat endpoints
to include the previous saved version alongside the active passage and exposed
reasoning summaries from their respective response records. A passage without an
earlier version displays alone with its summary. During regeneration the existing
passage remains visible while the replacement is pending; failed output remains unpublished.
The projection extracts final text and explicitly exposed summary text, excluding
requests, raw reasoning, opaque reasoning data and other response metadata. Summaries
can discuss story secrets, so they appear only in this explicitly opened evaluation
view. Expandable summaries use the same safe Markdown renderer as prose. A missing
summary is identified without inferring whether it was disabled or the model omitted it.
The view reads the original attempt records without
changing them or invoking a model. Browser preferences and the URL control the
view; no comparison setting enters session state, model context or transcript
exports. Both columns use the same Markdown renderer and stack on narrow screens.

Formatting uses Markdown with raw HTML disabled, following the
[parser's security guidance](https://markdown-it-py.readthedocs.io/en/latest/security.html).
Images are disabled and intentional line breaks are preserved. Original strings
remain unchanged in session storage and future model context. The server serves
only named UI assets and explicit endpoints, restricts session paths to its root,
checks configured Host/Origin pairs, and requires a per-process token for mutations.
Localhost HTTP origins are allowed by default. `chat --proxy-origin` explicitly
adds one HTTPS browser origin for a reverse proxy or tunnel. Proxy access requires
`--password-file`; HTTP Basic authentication with username `chat` protects every
page and API route, including localhost. Passwords are read from a private file,
compared in constant time, and excluded from browser projections and model settings.
The application still binds only to loopback, and forwarded headers cannot expand
the allowed origins. Phone browsers
use the HTTPS proxy address, including the secure context needed for submission
UUIDs and clipboard access. Both addresses share the same session files.
Content security policy excludes inline scripts and framing. Browser requests
never receive an API key. This is not a public deployment or multi-user service.

Offline HTTP and Chromium checks cover publication, formatting, proxy acceptance,
concurrent progress reads, stale submissions, recovery after a failed response,
reload during generation, saved composer text, mobile navigation and reading
position. Naming checks also cover generation and regeneration, unchanged raw history,
concurrent and pending operations, stale tabs, Unicode and mobile layout. Live narrative
quality remains evaluated through the separate playtests.

Automatic proxy validation covers the default CLI wiring, exact stdin requests,
fresh workspaces, separate final and summary output, failed/timeout results, resuming
saved manual requests, preserving existing passages on retry, and browser reload during both
API and proxy execution. The user's previously pending Covenant opening was
completed with two real Terra/max CLI calls before the editor was retired; its
private raw artifacts remain in the user's session directory. Current single-call
and regeneration checks run offline against both transports and Chromium.

Format 2 replaces the author/editor fields with one response-history list and
retires the revision snapshot. The runtime accepts only the current format.
Existing local playtests were backed up before their one-time conversion; their
published prose, identities, story canon and raw attempts were preserved. This
does not reinterpret earlier ordinary chat messages as regeneration operations.
