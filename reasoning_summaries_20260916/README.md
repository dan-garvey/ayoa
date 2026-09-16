# Reasoning summary capture validation

Two isolated Covenant openings verify that the coding-agent proxy can request and
retain exposed reasoning summaries. Both use maximum reasoning effort, one author
call, fresh Codex CLI 0.154.0 processes and `model_reasoning_summary="auto"`.
The complete input is supplied on stdin; completed `reasoning` items from the JSON
event stream are saved separately from the exact final message.

| Model | Completed summary entries | Summary characters | Prose characters |
| --- | ---: | ---: | ---: |
| GPT-5.6 Sol | 2 | 221 | 1,871 |
| GPT-5.6 Terra | 1 | 124 | 794 |

These returned compact planning headings. They are the provider-exposed summaries,
not full internal reasoning or a demonstrated explanation of narrative quality.
No private reasoning was inspected. Each model's first result is preserved;
neither opening was added to the user's existing playtest.

The `covenant-sol/` and `covenant-terra/` directories contain frozen story sources,
manifest, state, exact request packets, raw responses, and derived transcripts.
The raw proxy response's `reasoning_summaries` array preserves individual returned
strings. Viewing them makes no model call and does not modify history.

`validation.json` and `validation-terra.json` record the live outcomes and elapsed
times. `validation-ui.json` records the application checks: 84 offline tests pass,
including browser coverage, and all seven existing sessions were preserved. The
desktop/mobile screenshots show the Sol summary in the new Inspect responses view.
The default reading view, transcript downloads and later model input exclude summaries.
API summary serialization was verified with the real SDK over an offline transport;
no live direct API call was made.
