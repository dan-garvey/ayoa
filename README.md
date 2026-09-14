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

## Start a story

Use Python 3.10+ on Linux, macOS or WSL. From this checkout's root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m narrative init --story covenant --session sessions/my-story
.venv/bin/python -m narrative turn --session sessions/my-story --text 'Begin the story.'
```

Initialization makes no model calls. The default transport is `proxy`, and the
default model configuration is `gpt-5.6-terra` with `max` reasoning. `breakwater`
is a second bundled story: a contemporary coastal cinema with a rich ensemble.

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
  circumstances. Consistent invented history is welcome; contradictory history is not.
- `direction.md`: the story's genre, tone, portrayal and opening instructions.
- `player.json`: `{"name": "...", "description": "..."}` for its default protagonist.

Reusable author/editor instructions live in `prompts/`. Select an experimental
pair with `--prompts DIRECTORY` when initializing a new session. `--player-file`
selects another description; it must remain consistent with the story's premise.
There is no automatic rewriting of a story when the player file changes.

[DESIGN.md](DESIGN.md) describes the context and recovery contracts.
[AGENTS.md](AGENTS.md) describes contribution and evidence practices.

```bash
.venv/bin/pytest -q
.venv/bin/ruff check narrative tests
.venv/bin/ruff format --check narrative tests
```
