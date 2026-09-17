# Why the interaction keeps returning to the protagonist

The current prompt protects the protagonist's decisions more successfully than it
lets other people stop attending to him. Saying no to his offer is only one part
of NPC independence. An NPC can maintain the refusal while the author keeps
finding substitute roles for him in that NPC's evening. The player can technically
decline every offer and still feel that the world is being arranged around him.

This is a scene-selection problem as well as a dialogue problem. Eliminating
aphorisms, reducing the number of questions, or making the refusal harsher would
not by themselves address it.

## What the saved passages actually do

The source is `covenant-9780e3d01fcb`, 22 published passages, frozen in `originals/`.
The source state and all relevant requests are hashed in `manifest.json`.

- T15 gives Ashara a specific reason to reject the offer: she needs a trained
  partner for tomorrow's ranked bout. It already adds an invitation to watch
  tomorrow and ask about lessons afterward.
- T16's first response adds further rebukes, another watching/lessons suggestion,
  and Caelindra's question about the cheating student's assignment. The user's
  regeneration feedback explicitly objects to mining autobiography for engagement.
  The replacement advances dinner to departure, but Ashara then asks whether the
  protagonist is coming to watch **tonight**. The corrective instruction changed
  the kind of handoff without removing the need to supply one.
- T17 accepts the player's choice to study, and Stone's presence is grounded:
  canon places him in this library in the evening. His introductory interaction
  is reasonable. The subsequent sequence repeatedly asks about the protagonist's
  lawyers and prior experiences.
- T20 follows the player's thanks and attention returning to the book with a
  short study interval, then Seraphel's arrival and request to join him.
- T21 follows the player's explicit return to reading with another short interval,
  then Seraphel's question about his former copying work.

Seraphel's arrival is not a knowledge-boundary violation: the rooms connect and
T16 already established that she might seek quiet away from the cards. Ashara's
distinction between a useful partner and a spectator is logically possible.
Those explanations establish plausibility; they do not establish that these were
the best events to choose, especially in combination. The repetitive part is the
author's selection of a fresh approach to the protagonist whenever the previous
exchange could end.

## Relevant prompt pressures

These are hypotheses grounded in the supplied text, not experimentally isolated
effects of individual sentences.

**The handoff rules are more operational than the independence rules.** The
author is told to create an "engaging interactive story," create "people worth
spending time with," allow "at most one question," and end early when someone
addresses the protagonist. A model can satisfy these together by repeatedly
constructing one short social exchange ending in one question. The numerical
ceiling is not an instruction to ask a question, but can become a template in
practice. The early-stop instruction should govern an exchange that needs the
player's reply; it should not motivate manufacturing that exchange.

**Quiet activity can be misclassified as a lack of direction.** "When they watch
or hesitate, let other people's purposes develop into something that happens"
provides a useful response to inactivity. But thanking someone, declining an
invitation, and returning to a book are meaningful choices. Treating them as empty
space to fill creates social interruption instead of following the chosen course.
"Honor quiet" already exists; it is not enough if quiet is interpreted as a few
sentences of atmosphere before the next NPC question.

**The cast has strong dispositions but few immediate, independent activities.**
The full biographies should stay. They explain what would matter to these people.
Some earlier prose remedies also offer convenient social behavior: Ashara is more
likely to show someone a practice grip than explain courage; Seraphel wants a
listener and likes hearing people read aloud. These details can be expressed as
reasons to approach the player every time, rather than interests exercised with
other people or alone. Their presence is a candidate influence, not evidence that
rich backstory itself is harmful.

The Rashid/Dena/Ashara scheduling conflict is the useful counterexample. It gives
people concrete, conflicting commitments, expressly independent of the protagonist.
When the player declines involvement at T13, they do resolve it among themselves.
The failure comes when the response tacks on Caelindra's autobiography question
afterward. The prompt already supplies viable independent action; the author
keeps adding a player-facing invitation after the action has done its work.

**The recent transcript reinforces the pattern.** Fifteen of the 22 active
passages end in a direct question. Future calls receive these published exchanges,
including ordinary OOC messages. Regeneration feedback is correctly excluded,
so the T16 correction does not become an enduring instruction at T20/T21. Its
replacement passage does remain, including Ashara's renewed invitation. A direct
API replay inherits that same history; it cannot establish where the tendency
first arose.

## Coding-agent wrapper and message roles

The live path is one fresh `codex exec` call per passage. The full author/story
prefix and user/assistant history become one tagged string passed as its user
task. Repository instructions, user configuration, tools and memory discovery are
disabled; the current runner does not replace the built-in base instructions.
Official OpenAI documentation separately exposes a
[`model_instructions_file` override](https://learn.chatgpt.com/docs/config-file/config-sample)
for those built-in instructions. The isolation settings are therefore not a claim
that this is a bare narrative API request.

The direct path sends the prefix as `instructions` and the history as real message
roles using [Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
The test also sends the proxy's single tagged string directly to that endpoint.
This third condition distinguishes message packaging from the broader proxy
environment. It still cannot isolate an exact built-in instruction from model
serving configuration. All calls request
[Terra with maximum reasoning effort](https://developers.openai.com/api/docs/models/gpt-5.6-terra).

## What a useful prompt change would target

Retain the rich biographies, player ownership, and room for NPC initiative. Revise
the existing direction and turn-taking rules instead of accumulating another list
of forbidden phrases. A compact candidate, not yet applied or behaviorally tested:

> Follow the activity and degree of interaction the player chooses. People can
> finish conversations and pursue purposes that do not involve the protagonist;
> a natural stopping point is a valid end to a passage. When the player returns
> to an activity, let it proceed, with interruptions arising from the situation
> and other people's purposes rather than a need to offer another interaction.

For the separate turn-taking instruction:

> When an exchange calls for the player's response, leave room for it before
> introducing another demand on their attention.

This should replace overlapping direction/handoff wording, not expand it. It is
a candidate to test, not a claimed fix. No character dialogue examples are needed.

In the existing scenes, satisfactory continuations could simply let Ashara leave
to secure useful practice, give the player something concrete to learn from the
chosen reading, or let Seraphel read beside him. Later NPC initiative remains
welcome when it fits the developing situation. The quality criterion is not fewer
questions in isolation: it is whether the selected event respects both the NPC's
purpose and the player's current interest, while giving the scene actual content.
