# Execution notes

The first launcher invocation failed before creating story threads or calling a
model: `--ignore-user-config` belongs to `codex exec`, not `codex app-server`.
The failed initialization traceback remains at the beginning of `run.log`.
Removing that unsupported flag allowed a no-generation thread probe to succeed.

The local user configuration was inspected by key without exporting its contents.
It contained model/reasoning, approval/sandbox, trusted-project and UI/feature
settings; no custom instruction, model-provider, MCP, plugin or app configuration
was present. Every story thread explicitly sets Sol, maximum effort, detailed
exposed summaries, never approval and read-only sandboxing. The process disables
tools, web search, skills, plugins, apps, memories, hooks, goals and workspace
discovery, and uses an empty temporary workspace with project-document loading
disabled. Thread creation reports no instruction sources. The normal coding
agent base instructions remain; this is still a coding-agent proxy experiment.

The no-generation bootstrap probe is separate from the single server process
hosting all experiment generations. `events.jsonl` records that process and all
dispatches. The first valid story output is always retained.

Production defaults were changed independently to Sol/max in `ab4ed83` on
`codex/covenant-single-llm`. The live chat was restarted without pending turns.
Existing session files and frozen model settings were preserved. Conversation
reuse is confined to this evaluation; the production transport was not changed.

Offline checks before the run: 94 active runtime tests, 20 browser tests, two
persistent transport tests, and Ruff on changed Python files passed. No live
story call was made solely to test the default setting.
