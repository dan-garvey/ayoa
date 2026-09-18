# Narrative foundation

One model authors the world, narration and every supporting character.
Its story response publishes directly. Periodic private checkups review adherence
and plan possible developments before writing the next passage. Characters retain individual interests and
plausible knowledge boundaries through the writing contract and full biographies.

This work continues on **`codex/covenant-single-llm`**. The old engine and raw
research live on **`archive/covenant-prompt-trials`**. Read the
[findings](docs/findings.md) for the improvements, counterexamples and exact evidence.
The automatic editor has been retired following playtest review; sustained literary
quality remains an open research objective.

## Browser chat

Use Python 3.10+ on Linux, macOS or WSL. From this checkout's root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m narrative chat
```

The default chat runs **Sol coding agents automatically**, using your existing
Codex CLI login. Install a current Codex CLI and run `codex login` first if needed;
this integration is verified with version 0.154.0. Each turn gets a fresh author
call at `max` reasoning. Every fifth player message first gets a separate checkup
using the same model and effort. The chat shows when a passage is being written.

Open **http://localhost:8765**. Choose a saved story or select **New story**.
Choose your protagonist’s name before beginning. To rename them between turns,
click the name above the story title. Their background stays the same, and
earlier passages keep their original wording. Future responses receive the new
name and a correction identifying earlier names as the same character.
The conversation shows your turns and published passages, with Markdown emphasis,
headings, quotations, lists, tables, and preserved verse line breaks. Reading
preferences offer light/dark appearance and adjustable text size. New passages
open at their beginning; incoming responses preserve your position when you are
reading earlier text. You can copy a passage or download the published transcript.

For evaluation, enable **Inspect responses** in the header, or open
**http://localhost:8765/?compare=1**. Passages with multiple versions show the previous
one on the left and the current one on the right, with separate copy buttons.
Narrow screens stack the columns. Older playtests retain their draft/edit comparisons.
Expand **Reasoning summary** beneath a response to read its captured summary,
including passages with only one version. New automatic responses request detailed
exposed summaries; their length still depends on what the model returns. Older
responses and models that return none show an unavailable message.
The toggle persists in the browser. Inspection makes no model calls. Summaries stay
in response records; only active passages enter future story history and downloads.

Inspection also shows expandable **Backstage checkup** notes before the passage
they guide, with their exposed reasoning summaries when available. These notes
review adherence and suggest conditional plot and character development. They
remain outside the conversation and transcript download; only the latest notes
are added privately to subsequent model requests. They are advice, not established
events or new knowledge for characters, and may contain story spoilers.

Configure the interval for new sessions with `--checkup-every N` on `chat` or
`init`; use `0` to disable it. The opening counts as one ordinary player message.
Regeneration, retries and renaming do not advance the counter. The checkup runs
before the response to the Nth message, so that response receives its guidance.
Due turns wait for two sequential calls. A failed checkup pauses the response;
**Continue response** retries it. If only the author fails, its completed checkup
is reused. Regeneration does not run another checkup or expose discarded prose
to later checkups.

Write in the multiline composer. **Ctrl+Enter** (or **Cmd+Enter**) sends;
**Enter** inserts a line break. Unsent text and reading preferences survive reloads
in the same browser. Accepted turns are saved to the session files. An unfinished
or failed response offers **Continue response** without resubmitting your turn.

To replace the latest response, write your correction instructions in the composer
and press **Ctrl+Shift+Enter** (or **Cmd+Shift+Enter**), or click **Regenerate**.
The original player action stays in place. The replacement becomes the active
passage; the rejected version and correction instructions are excluded from later
model history and transcript downloads. Instructions apply only to that regeneration.
All versions and exact requests remain saved for evaluation. A failed regeneration
keeps the existing passage until **Continue response** succeeds.

For direct API responses, start the chat with an API key available to the server:

```bash
.venv/bin/python -m narrative chat --transport api --env-file .env
```

For experiments with manually supplied replies, start with `--manual`.
**Response handoff** then lets you copy/download the current request and paste or
load the complete reply. A checkup handoff saves private notes and prepares the
next handoff; accepting the author handoff publishes the passage. The handoff contains
story secrets and stays separate from the conversation. Existing CLI proxy commands also work; their
results appear in the chat automatically.

`--port` changes the local port, and `--sessions DIRECTORY` selects a different
session directory. Nested existing sessions are listed automatically. Model,
reasoning and transport options set defaults for **new** sessions; existing ones
retain their frozen configuration. The defaults are `gpt-5.6-sol` and `max`.
The server binds only to loopback by default; it is a single-user interface.

To use your phone on the same Wi-Fi, start the chat with the computer's private
LAN IPv4 address (replace this example with yours):

```bash
.venv/bin/python -m narrative chat --lan-address 192.168.86.25
```

Open **http://192.168.86.25:8765** on your phone. This uses your home network
directly, with no password or third-party service. The computer must stay on.
Localhost access on the computer still works. Address and action-token checks
remain in place, and turn submission works over plain HTTP.

With WSL's default localhost forwarding, Windows also needs a LAN listener and
firewall rule. Run these once in **PowerShell as Administrator**, using your
Windows LAN address and home subnet:

```powershell
netsh interface portproxy add v4tov4 listenaddress=192.168.86.25 listenport=8765 connectaddress=127.0.0.1 connectport=8765
New-NetFirewallRule -Name AyoaStoryChatLAN8765 -DisplayName 'Ayoa story chat on home Wi-Fi' -Direction Inbound -Action Allow -Protocol TCP -LocalAddress 192.168.86.25 -LocalPort 8765 -RemoteAddress 192.168.86.0/24 -Profile Private
```

The listener forwards to the existing Windows localhost connection to WSL, so
it does not depend on WSL's changing internal IP. The firewall rule allows the
home subnet on the private Windows network profile. See Microsoft's
[WSL networking documentation](https://learn.microsoft.com/en-us/windows/wsl/networking).

## Terminal workflow

The original commands remain available:

```bash
.venv/bin/python -m narrative init --story covenant --session sessions/my-story --player-name 'Avery'
.venv/bin/python -m narrative turn --session sessions/my-story --text 'Begin the story.'
```

Initialization makes no model calls. `breakwater` is a second bundled story:
a contemporary coastal cinema with a rich ensemble.

Omit `--player-name` to use the story’s default. Rename an existing protagonist
after completing any pending response:

```bash
.venv/bin/python -m narrative rename --session sessions/my-story --player-name 'Morgan'
```

Names allow 1–80 characters, including Unicode; extra whitespace is normalized.
Renaming makes no model call and preserves the transcript and saved attempts.

Add `--auto` to `turn`, `regenerate` or `resume` to run the response with Codex:

```bash
.venv/bin/python -m narrative turn --session sessions/my-story --auto --text 'Begin the story.'
.venv/bin/python -m narrative regenerate --session sessions/my-story --auto --text 'Keep the response within the room.'
.venv/bin/python -m narrative resume --session sessions/my-story --auto
```

An already prepared manual request can be resumed automatically, with its saved
context and player input intact. No session conversion is needed. Without
`--auto`, terminal proxy commands retain the manual workflow below.

The turn command returns a request id and paths to exact JSON and readable text
requests. Give the complete text request to a fresh Sol coding agent at maximum
reasoning effort. Save its complete response to a UTF-8 file, then accept it:

```bash
.venv/bin/python -m narrative accept --session sessions/my-story --request-id REQUEST_ID --output-file response.txt
```

Accepting an author response publishes and prints the passage. When a checkup is
due, first accept its notes, then use the returned author request for the passage.
Submit the next player
action with `turn`, or request a replacement with `regenerate`.
Fresh proxies receive all necessary context; do not give them other trials,
assessments, discarded historical drafts or additional writing instructions.

## Direct API execution

Choose API transport when initializing a separate session:

```bash
.venv/bin/python -m narrative init --story covenant --transport api --session sessions/api-story
.venv/bin/python -m narrative turn --session sessions/api-story --env-file .env --text 'Begin the story.'
```

Set `OPENAI_API_KEY` in the environment or the explicitly supplied environment
file. The due checkup and author calls execute automatically. `--model`, `--reasoning`,
`--checkup-every` and `--max-output-tokens` are initialization options; the 12,000-token default is
a per-call output budget including reasoning, not a visible-prose target. Settings
are frozen for the session. The API adapter is tested offline with the real SDK;
the foundation's narrative validation uses coding-agent proxies.
Codex proxies use the CLI's own output limits; the API `max_output_tokens` setting
does not impose a separate limit on those calls.

## Resume and inspect

```bash
.venv/bin/python -m narrative resume --session sessions/my-story
.venv/bin/python -m narrative export --session sessions/my-story
```

Resume reuses saved responses, retries a failed attempt explicitly, or returns
the pending proxy request. Add `--env-file .env` for API sessions when needed.
A failed regeneration never replaces the existing passage or requires resubmitting
the original player action. A new action is rejected while a response remains pending.

Each session contains frozen sources and settings, `state.json`, a directory per
attempt with its exact request and raw response/error, and derived
`transcript.md`/`summary.json` exports. Published turn records identify
all their successful response versions, with the current version last. The transcript
contains only original player inputs and active prose. Completed checkups reference
their saved attempts separately and never appear as published turns. Usage counts
both author and checkup calls and includes unsuccessful
API responses where reported;
unavailable proxy usage is shown as null. Sessions are ignored by Git.
Automatic proxy attempts record CLI exit status and elapsed time separately from
API usage. Final agent messages and exposed reasoning summaries are saved separately.
Other coding-agent events are discarded. These are provider-generated summaries,
not a complete record of internal reasoning.

## Write another story

Create a directory with these three files and pass its name or path to `--story`:

- `canon.md`: world, full biographies, relationships, private knowledge and initial
  circumstances. Refer to the player character as **the protagonist**, using their
  chosen name only in `player.json`. Consistent invented history is welcome;
  contradictory history is not.
- `direction.md`: the story's genre, tone, portrayal and opening instructions.
- `player.json`: `{"name": "...", "description": "..."}` for its default protagonist.

Reusable writing instructions, the checkup task and the brief regeneration instruction live in
`prompts/`. Select another set with `--prompts DIRECTORY` when initializing a new
session. `--player-file` selects another description; it must remain consistent with the story's premise.
`--player-name` overrides its default name. Naming preserves the story’s family
history, places and other characters: Covenant’s Garvey ancestry and estate,
for example, remain part of the premise even when you choose another surname.

[DESIGN.md](DESIGN.md) describes the context and recovery contracts.
[AGENTS.md](AGENTS.md) describes contribution and evidence practices.

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q
.venv/bin/ruff check narrative tests
.venv/bin/ruff format --check narrative tests
```

Browser checks use local fake responses and make no model or external network
calls. Install Chromium once, then include them in the suite:

```bash
.venv/bin/python -m playwright install chromium
.venv/bin/pytest -q --with-browser
```
