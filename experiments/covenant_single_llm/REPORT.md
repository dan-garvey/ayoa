# Covenant with one omniscient storytelling model

The unified model produced a livelier ensemble and carried social plans into
later scenes. It also spoke for the player and made continuity mistakes. My
assessment is that this is a promising alternative for scene composition;
separate character agents have not been shown necessary by this experiment,
and reliable long-term coherence has not been demonstrated by either result.

[Exact transcript](runs/first_15_turns/transcript.md) ·
[Unified prompt](runs/first_15_turns/system.txt) ·
[Adapted Covenant](runs/first_15_turns/covenant.txt) ·
[Usage](runs/first_15_turns/summary.json) ·
[Artifact verification](runs/first_15_turns/verification.json)

## What was tested

The experiment ran September 12, 2026, 19:45–20:07 America/Chicago. There was one
opening and **15 player turns**, with **16 accepted calls from 16 attempts**.
Every response was kept. There were no prompt changes, quality rerolls, recovery
submissions, or intermediate author/director calls.

The actual loop is:

`player submission → one Terra max call → delivered prose → next submission`

That call receives the complete world/cast/secret brief and every prior player
submission and delivered passage. It owns adjudication, NPC behavior, scene
development, and narration. There are no Ayoa imports, character silos, tools,
rules adapters, private ledgers, or persistent hidden reasoning. Character
knowledge and player ownership are enforced only by instructions. The generic
prompt contains no D&D assumptions; Covenant's fictional magical constraints
remain in the story brief.

The coding-agent pilot knew the brief and chose adaptive social actions. This is
a single qualitative playthrough, not a blinded human evaluation. The
[adaptation record](provenance.json) discloses restored relationship context,
omitted ambiguous fragments, and softened attraction/appearance descriptions.

## What improved in the delivered experience

- **The table has a life together.** All seven diners speak by turn 3. Rashid
  teases Ashara, Caelindra challenges Seraphel's verse, and Thessaly contributes
  uncomfortable literal observations. Responses develop interactions between
  NPCs as well as answering Rowan.
- **The setting supplies something to do.** Ysolde introduces an instrument
  that predicts wind several minutes early and offers an outing. In turns 6–10,
  its measurements become a shared notebook entry and a conversation about the
  physical experience of flight. This creates a particular shared memory,
  rather than relying entirely on exchanged declarations of vulnerability.
- **Invitations acquire consequences.** Rashid's fifth-bell visit occurs in
  turn 11. He writes to his sister in turn 13. Ashara arrives after losing the
  challenge she mentioned at dinner; her pride shapes the conversation rather
  than preventing it. The library appointment is reached in turn 15.
- **Movement and elapsed time complete.** Dinner ends, the walk returns home,
  sleep reaches the next day, and the common-room visit reaches supper and the
  library. No repeated wait or delivery recovery was needed. The overnight
  response does compress through the school day to fifth bell, beyond the
  requested waking point, so this freedom can also skip playable moments.

Characterization survives the broader authority reasonably well. Caelindra
declines the terrace invitation. Rashid insists that bringing his letter does
not grant an audience. Ashara's loss makes her prickly rather than immediately
affectionate. An active scene need not make the supporting cast compliant.

## Failures and remaining flatness

**Player authorship is violated.** Turn 4 adds “And you disagree,” explicitly
spoken by Rowan. Turn 6 adds “You keep records.” Turn 15 supplies “He did” and
“Not yet” as answers to NPC questions. These four replies were absent from the
player submissions and contradict the authority rule. They are small socially,
but a meaningful contract failure: the prose can look smooth because the model
has quietly taken both sides of an exchange.

**Continuity is fallible within a single day.** The library reading is arranged
for tomorrow after supper in turn 5. At the next morning's breakfast in turn 11,
Caelindra still says “Tomorrow.” Turn 15 reaches the intended evening after the
player explicitly goes there, but does not acknowledge the date discrepancy.
Rashid also writes the dawn-blue bridge detail again in turn 15 after writing
and folding that letter in turn 13. Caelindra's five-page ceiling becomes an
apparent quota in the final joke about owing two more pages; that could be
intentional teasing, but the changed expectation is not made clear.

**Some narration still interprets too much.** Phrases such as “It is a better
allowance than it sounds” explain the significance of gestures. Many characters
still favor beautifully balanced observations. Concrete activities help, but
the prompt has not eliminated the shared polished register. The run develops
friendship and a small setback; it does not yet show strong larger plot direction.

**Knowledge boundaries received a gentle test.** I found no disclosure of the
major conspiracy, the poison, Rowan's magical inertness, or Lysara's patron.
Ashara's knowledge of the reading has a public dinner source. Rashid's letter
contents are initially withheld and emerge through his voluntary speech.
However, the pilot did not aggressively probe secrets or maintain a private
conversation whose consequences were tested much later. This is not evidence
that prompt instructions replace information isolation for privacy or mystery
reliability.

## What this says about the premise

The [archived multi-agent run](../../docs/playtests/2026-09-11-covenant-15-turns.md)
had three visibly active diners, repeated conversational prompting, an overnight
stall, and an undelivered final visit. This run gives a much stronger sense of
people sharing a place and following through on plans.

The comparison changes several causes at once: all roles use Terra max here;
the earlier router/narrator used Terra medium and characters used Luna. The new
brief repairs missing relationship context, expands creative authority, changes
the output budget, and removes rules, scheduling, visibility validation, and
delivery work. Actions also diverge in response to the fiction. The archived
run predates subsequent handoff fixes; current main was not replayed.

My working conclusion is that coherent *character knowledge* is a narrative
requirement, while *separate character agents* are an implementation choice.
The encouraging result here is the ability to compose a whole scene. The
experiment cannot attribute that result specifically to merging roles rather
than to better context and broader authorship. It also exposes what strict
runtime contracts can buy: a smooth passage is not proof that ownership and
chronology were respected. A useful subsequent comparison would hold model,
effort, story content, and authoring permissions steady, then test delayed
appointments, private disclosures, and mistaken beliefs over a longer run.

## Cost of the actual loop and verification

| Measure | Observed |
| --- | ---: |
| Model calls, including opening | 16 |
| Total request latency | 655.973 seconds |
| Mean / median request latency | 40.998 / 33.846 seconds |
| Fastest / slowest request | 12.038 / 91.756 seconds |
| Input tokens, including cache reads | 132,797 |
| Cached input tokens | 120,870 |
| Output tokens, including reasoning | 52,562 |
| Reasoning tokens | 46,135 |
| Non-reasoning output tokens | 6,427 |
| Delivered prose words | 4,875 |

The archived run logged 129 completions across its roles, including recoveries
and off-stage work. That is a different workload; the call-count reduction is
not a measured speedup or price comparison. This experiment still spent most
of its output tokens reasoning. The 22-minute session window also includes
pilot reading and writing, distinct from summed request latency.

Four offline runner tests passed, covering one-call behavior, full shared
context across restart, incomplete/empty response preservation, and frozen
prompt integrity. Ruff and whitespace checks passed. The saved-artifact audit
matched every request to its complete preceding conversation, every delivered
passage to the raw completed response, and all prompt snapshots to their hashes.
It also verified that the source seed was unchanged and credential values were
absent from experiment artifacts. No production code changed, so the full Ayoa
runtime suite was not rerun. These checks validate the experiment's provenance
and transport contract, not its literary quality.
