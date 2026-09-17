# Forced engagement: frozen-prefix transport comparison

Prepared before inspecting any new model output. No live session is modified.

Use Terra (`gpt-5.6-terra`) at maximum reasoning effort, two independent samples
per checkpoint and condition, 18 calls total. Do not select or regenerate the
best result. Retain failed and incomplete calls and do not silently retry them.
All cells use the exact saved author instructions, complete biographies and
published history; only the transport/message packaging changes.

## Checkpoints

- **T16 departure:** exact regeneration request after "Suit yourself." Includes
  the rejected draft and the user's instruction to stop requesting autobiography
  and let the story move forward. Its original replacement ended with Ashara
  inviting the protagonist to watch tonight's practice. This is a regeneration
  test, not an unprompted continuation; preserve both source attempts.
- **T20 library:** the player thanks the professor and looks down at the book.
  The original response introduces Seraphel and asks to sit beside the player.
- **T21 reading:** the player welcomes Seraphel to sit, then returns to reading.
  The original first response interrupts quiet study to ask about copying books.
  Test the ordinary request, before later verse-related regeneration.

## Conditions and actual loops

1. **api_native:** one fresh `Responses.create` call with the original
   `instructions` field and actual user/assistant messages. No tools, editor,
   coding-agent process, prior response id or carried reasoning.
2. **api_flat:** one fresh Responses call with the same complete tagged text
   produced by `render_request`, as a single user message, with no instructions
   field. Controls for the proxy's message packaging without a coding wrapper.
3. **codex_proxy:** one fresh existing `CodexProxy` call with the saved JSON
   request. Its tagged projection is submitted as a coding-agent user task.
   Keep the current built-in wrapper, disabled tools/discovery and empty workspace.

Run the first T20 native API cell to establish credential/model access, then run
all other cells without changing the inputs, at most three concurrent calls.
API calls use the official endpoint, no SDK retries, 300-second timeout,
`store=false`, disabled truncation, 12,000 output-token budget and exposed
summary=`auto`. Proxy uses its existing 900-second timeout. The CLI does not
honor the API token ceiling; no claim of identical inference environments.

## Manual evaluation

Read every complete passage. Record the selected next event, whether an NPC asks
about the player's earlier life, whether it offers/requires another player
interaction, and whether the player's immediate activity can proceed or end.
Distinguish questions from requests/directions: removing a question mark is not
enough. Note continuity, character interests and knowledge, appropriateness of
initiative, and whether quiet narration contains meaningful activity.

An invitation or friendly exchange is not automatically a failure. Evaluate
whether it follows the NPC's current purpose and the player's expressed interest,
and whether it repeats the surrounding pattern of always supplying another
companion, question or role. Do not require hostility or universal passivity.

This small targeted test cannot estimate a stable population rate or prove one
prompt clause caused an effect. Every cell inherits proxy-written history, so
API recurrence shows a current coding wrapper is unnecessary to reproduce the
behavior, not that the wrapper never helped establish that history. A strong
transport difference warrants follow-up; identical model labels do not prove
identical serving configuration. Summaries, if returned, are exposed summaries
only and are not a complete causal explanation.
