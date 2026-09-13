# Covenant with one storytelling model

An isolated experiment requested by the user, based on commit `780c661`.
The implementation is standalone Python. It imports no Ayoa code and does not
use its router, character agents, narrator, checkpoint schema, rules adapters,
delivery system, or bot. This experiment does not modify the main runtime or
original story seed.

| Model and prompt, max reasoning | 15-turn transcript | Assessment |
| --- | --- | --- |
| Terra, original | [Transcript](runs/first_15_turns/transcript.md) | [Report](REPORT.md) |
| Sol, original | [Transcript](runs/sol_max_15_turns/transcript.md) | [Report and comparison](SOL_MAX_REPORT.md) |
| Terra, chat v2 and style revision | [Transcript](runs/terra_chat_v2_style_15_turns/transcript.md) | [Report and replays](TERRA_STYLE_REPORT.md) |

`system.txt` gives one author control of the world, supporting cast, adjudication,
scene development, and prose. `covenant.txt` is the complete adapted story brief,
including all hidden lore and character motives. Every call receives both and
the complete shared user/assistant prose history. The player still owns Rowan's
choices; character knowledge limits are prompt instructions, with no structural
information isolation or semantic validator. There are no model tools, private
character histories, continuity summaries, director passes, or evaluation calls.

The initial Terra and Sol runs use identical original prompts. The current
`system.txt` and `covenant.txt` incorporate the user's chat-era Covenant and
interactive style guide; the third run tests this revision with Terra. Each run
freezes its own prompt copies, so the original runs retain their exact inputs.
All use max reasoning and an opening plus fifteen adaptive player turns, without
rerolling completed passages. The revised Terra run required one identical
technical retry after turn 15 exhausted the output ceiling entirely on reasoning;
the failure and its usage are preserved. The runner is unchanged. A 12,000-token response ceiling
includes reasoning. The original prompt asks for 150–350 words; the revision asks
for one to five short paragraphs, stopping at unanswered player interactions.
Provider requests are text-only. SDK retries are disabled so every attempted
call has a local request record. Incomplete, empty, or refused output is preserved
without advancing the conversation. Accepted output is never rewritten.

## Run it

Install `requirements.txt` into a Python environment. From this directory:

```bash
python play.py init --run-dir runs/my_session --env-file /path/to/credentials.env
python play.py turn --run-dir runs/my_session --env-file /path/to/credentials.env \
  --text 'I enter the dining hall and introduce myself.'
python play.py export --run-dir runs/my_session
```

For Sol, add `--model gpt-5.6-sol --reasoning max` to the initial command and
choose a fresh run directory. Later turns retain that run's model selection.

The credential file supplies `OPENAI_API_KEY` or the existing `OPEN_AI_ROUTER`
credential. Credentials are never included in saved requests. The initial command
accepts `--model` and `--reasoning`; later turns use the frozen run configuration.
Run directories cannot be initialized twice, and changing their prompt snapshots
causes a failure before a model call. A process lock serializes calls to one run.

The implementation follows the Responses API's
[explicit conversation-history pattern](https://developers.openai.com/api/docs/guides/conversation-state).
It sends the complete textual conversation with `store=false`; provider reasoning
is not a separate persistent memory channel. Existing turns reconstruct a resumed
conversation without additional generation.

## Evidence

`provenance.json` records the initial adaptation. Later runs have their own
provenance files. `sources/` preserves the supplied style guide, relevant chat-v2
excerpts, source hash, and adaptation decisions. The original Windows document
is not modified. The revised prompt also has three recorded same-context replays
under `probes/terra_chat_v2_style/`.

Each run contains frozen prompts, a manifest, exact request/response files under
`attempts/`, accepted turn records, `transcript.md`, and aggregate `summary.json`.
The runner's usage and latency totals cover accepted responses. The revised
Terra run also has `attempt_summary.json`, which includes the incomplete request.
The request logs contain omniscient fictional spoilers. The transcript contains
only the opening/player submissions and delivered prose.

This is a qualitative adaptation test, not a controlled architecture A/B. The
previous Covenant run used Terra at medium effort for router/narrator, Luna for
characters, different role contracts, and incomplete owner-context fragments.
Here the prompt, information access, creative authority, reasoning, output budget,
and state machinery all differ. The coding-agent pilot knows the story brief and
responds adaptively, pursuing a similar social route; it is not a blinded human
playtest. Fifteen turns cannot establish long-term reliability, multiplayer
privacy, or mechanically accurate D&D adjudication.

Offline runner checks:

```bash
pytest -q test_play.py
```

They check one-call behavior, full shared context and restart continuity,
preservation of incomplete output without advancing, frozen-prompt integrity,
and exclusion of implementation details from rendered instructions.
Literary quality is assessed from the actual transcript, not a word-matching test.
