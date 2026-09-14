# Narrative foundation

One model authors the world, narration and every supporting character. A second,
sequential call revises its draft with a 40-word instruction. Only the revised
passage becomes published history. Characters retain individual interests and
plausible knowledge boundaries through the writing contract and full biographies.

This work continues on **`codex/covenant-single-llm`**. The old engine and raw
research live on **`archive/covenant-prompt-trials`**. Read the
[findings](docs/findings.md) for the improvements, counterexamples and exact evidence.
The concise editor is a working baseline; sustained literary quality remains an
open research objective.

## Browser chat

Use Python 3.10+ on Linux, macOS or WSL. From this checkout's root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m narrative chat
```

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

Write in the multiline composer. **Ctrl+Enter** (or **Cmd+Enter**) sends;
**Enter** inserts a line break. Unsent text and reading preferences survive reloads
in the same browser. Accepted turns are saved to the session files. An unfinished
or failed response offers **Continue response** without resubmitting your turn.

For automatic responses, start the chat with an API key available to the server:

```bash
.venv/bin/python -m narrative chat --transport api --env-file .env
```

The default is `proxy`: **Response handoff** lets you copy/download the current
request and paste or load the complete reply. Supply the author's draft first,
then the editor's response to publish the passage. The handoff contains story
secrets and stays separate from the conversation. Existing CLI proxy commands
also work; their results appear in the chat automatically.

`--port` changes the local port, and `--sessions DIRECTORY` selects a different
session directory. Nested existing sessions are listed automatically. Model,
reasoning and transport options set defaults for **new** sessions; existing ones
retain their frozen configuration. The defaults remain `gpt-5.6-terra` and `max`.
The server binds only to loopback; it is a local, single-user interface.

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

The turn command returns a request id and paths to exact JSON and readable text
requests. Give the complete text request to a fresh Terra coding agent at maximum
reasoning effort. Save its complete response to a UTF-8 file, then accept it:

```bash
.venv/bin/python -m narrative accept --session sessions/my-story --request-id REQUEST_ID --output-file response.txt
```

Accepting the author's draft returns the editor's request. Give that complete
request to another fresh agent, then accept its response with its new request id.
The published passage is printed. Submit the next player action with `turn`.
Fresh proxies receive all necessary context; do not give them other trials,
assessments, discarded historical drafts or additional writing instructions.

## Direct API execution

Choose API transport when initializing a separate session:

```bash
.venv/bin/python -m narrative init --story covenant --transport api --session sessions/api-story
.venv/bin/python -m narrative turn --session sessions/api-story --env-file .env --text 'Begin the story.'
```

Set `OPENAI_API_KEY` in the environment or the explicitly supplied environment
file. Both sequential calls then execute automatically. `--model`, `--reasoning`
and `--max-output-tokens` are initialization options; the 12,000-token default is
a per-call output budget including reasoning, not a visible-prose target. Settings
are frozen for the session. The API adapter is tested offline with the real SDK;
the foundation's narrative validation uses coding-agent proxies.

## Resume and inspect

```bash
.venv/bin/python -m narrative resume --session sessions/my-story
.venv/bin/python -m narrative export --session sessions/my-story
```

Resume reuses saved responses, retries a failed API stage explicitly, or returns
the pending proxy request. Add `--env-file .env` for API sessions when needed.
An editor failure never publishes the draft or requires resubmitting the player
action. A new action is rejected while a turn remains pending.

Each session contains frozen sources and settings, `state.json`, a directory per
attempt with its exact request and raw response/error, and derived
`transcript.md`/`summary.json` exports. Published turn records identify their
author and editor attempts. The transcript contains only player inputs and
published prose. Usage includes unsuccessful API responses where reported;
unavailable proxy usage is shown as null. Sessions are ignored by Git.

## Write another story

Create a directory with these three files and pass its name or path to `--story`:

- `canon.md`: world, full biographies, relationships, private knowledge and initial
  circumstances. Refer to the player character as **the protagonist**, using their
  chosen name only in `player.json`. Consistent invented history is welcome;
  contradictory history is not.
- `direction.md`: the story's genre, tone, portrayal and opening instructions.
- `player.json`: `{"name": "...", "description": "..."}` for its default protagonist.

Reusable author/editor instructions live in `prompts/`. Select an experimental
pair with `--prompts DIRECTORY` when initializing a new session. `--player-file`
selects another description; it must remain consistent with the story's premise.
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
