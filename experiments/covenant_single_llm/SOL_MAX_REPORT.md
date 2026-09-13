# Covenant: Sol max repeat

Sol produced the more animated ensemble in my reading of these two runs, and
introduced a stronger next story hook. It also took over unsubmitted player
choices and invented a sentence in the player's manuscript. The stronger scene
composition does not resolve the authorship problem found with Terra.

[Exact Sol transcript](runs/sol_max_15_turns/transcript.md) ·
[Terra assessment](REPORT.md) ·
[Sol provenance](runs/sol_max_15_turns/provenance.json) ·
[Artifact audit](runs/sol_max_15_turns/verification.json)

## Method

One opening plus **15 player turns**, September 12, 2026, 21:40–22:07
America/Chicago. All **16 requests completed on their first attempt**. No
quality rerolls, prompt edits, or transport recoveries were needed. Part of
turn 14 was used to dispute a fictional error; it remains in the transcript.

The model changed from `gpt-5.6-terra` to `gpt-5.6-sol`. Max reasoning, the
12,000-token response ceiling, system prompt, Covenant adaptation, player
identity, opening submission, and runner were identical. Both requested and
returned configuration identify Sol with max reasoning.

The loop remains player submission → one full-context model call → delivered
prose. Every request carries the entire shared prose conversation and the full
omniscient brief. There are no character agents, auxiliary author calls, tools,
private ledgers, or Ayoa runtime calls.

The same coding-agent pilot followed the broad dinner/companionship/next-day
route, adapting to Sol's actual scene. The pilot knew the brief and Terra run.
Different actions and generated histories make this an exploratory comparison,
not a paired benchmark with identical inputs or a blinded preference test.

## What happened and what worked

- **Turns 1–4: the table develops its own interactions.** Aldric's attempt to
  move Seraphel's place card becomes a running embarrassment. Rashid and Ashara
  spar over his broken nose and her idea of leisure. All seven diners speak.
  Rashid's imagined afternoon with his sister has a market, cakes, a fountain,
  and a family joke rather than only an abstract statement about obligation.
- **Turns 5–6: conversation becomes a shared activity.** Bastions gives Rashid
  and Caelindra competing offers to make, Thessaly literal promises to consider,
  and Ysolde an opportunity to exploit their distraction. This is a useful
  demonstration of the unified author's scene-building authority, with an
  important player-control failure described below.
- **Turns 7–10: Ysolde accepts company with a limit.** Tonight's invitation
  creates no future obligation. She notices specific behavior from dinner and
  describes flying over volcanic and glacial currents. Her account opens a
  conversation about belonging without turning into immediate romance.
- **Turns 11–13: the social appointment pays off.** Fifth bell brings the
  promised pastries, tea, and story reading on the correct day. Characters react
  differently to the courier's mishap. Caelindra asks whether criticism is
  welcome and returns the pages unmarked, leaving the decision to Rowan.
- **Turns 14–15: the next situation arrives.** A scheduled reception brings
  Rowan to Stone, who recognizes his signet and offers ordinary family memories.
  Windwhisper's approach makes Stone guarded. He proposes a library meeting;
  Lysara then introduces herself in a plausible public setting. Neither this
  meeting nor her entrance required the player to manufacture the hook.

Sol has a strong comic register, but some lines still sound like polished
aphorisms. Much of the group is accommodating, and the familiar reserve/ear
twitch/tail tap mannerisms recur. This remains an early social sample; the
reception opens larger intrigue rather than demonstrating a sustained plot.

## Contract failures

**Turn 6 takes over the game.** The player accepts one round of passage and
conditionally moves one envoy. Sol then narrates Rowan declining two other
offers, misjudging a crossing, and finishing the entire game with two seals.
Those further decisions were not submitted. The clean scene ending partly
comes from removing opportunities for the player to choose.

**Turn 13 invents evidence for its own criticism.** The submitted reading ends
with the keeper raising the sign. Caelindra instead quotes an absent sentence:
“Thus pride charged a dearer toll than any bridgekeeper.” She criticizes that
supposed line for explaining what the ending already conveys. The preceding
response had placed the excess explanation before the ending; now it is
described as afterward. The model has created a defect in Rowan's writing to
give the critique a satisfying shape.

**The repair is responsive but costs a player correction.** In turn 14 Rowan
objects. Sol has Caelindra inspect the page, acknowledge that the sentence is
absent, and apologize. It does not double down. However, the original invention
came from the sole storytelling model; the repair recasts it as Caelindra's
carelessness. That changes her portrayal to accommodate the model's mistake.

A smaller continuity slip describes the already framed, visibly readable letter
as “still unopened” in turn 11. Like Terra, Sol also advances the sleep submission
through classes to fifth bell, beyond its requested morning waking point. The
input mentions the day's obligations, so the intended extent of that compression
is debatable. The fifth-bell appointment itself stays on the correct day.

I found no new quoted replies spoken for Rowan, of the kind seen four times in
Terra. That is narrower than preserving player authorship: the game choices and
manuscript invention still violate it.

## Knowledge and longer-term limits

No major conspiracy secret is disclosed. Stone stays with personal recollection
and a later invitation; Lysara's patronage remains hidden. Other diners do not
pick up Rowan's private printer-shop biography from the terrace conversation.
These are observations from a gentle social route, not adversarial privacy
validation. Stone's new appointment is outside the tested horizon.

For this sample I prefer Sol's ensemble writing and movement into the next scene.
The unresolved issue is precise authority: giving one model room to compose the
whole scene can also let it manufacture player behavior or supposed prior text.
This run does not establish that separate agents are required, or that a single
omniscient model can reliably maintain a long campaign's boundaries.

## Usage and verification

| Measure, opening included | Terra max | Sol max |
| --- | ---: | ---: |
| Accepted calls / attempts | 16 / 16 | 16 / 16 |
| Total request latency | 655.973 s | 819.937 s |
| Mean request latency | 40.998 s | 51.246 s |
| Median request latency | 33.846 s | 51.334 s |
| Input tokens, including cache reads | 132,797 | 135,527 |
| Cached input tokens | 120,870 | 123,215 |
| Output tokens, including reasoning | 52,562 | 36,423 |
| Reasoning tokens | 46,135 | 29,796 |
| Non-reasoning output tokens | 6,427 | 6,627 |
| Delivered prose words | 4,875 | 4,806 |

Sol used fewer reasoning tokens but had longer request latency in this run.
These are observed usage and timing, not billed prices or stable performance
estimates. Session wall time includes the pilot's reading, writing, and pauses.

The artifact audit verified every complete request history, raw response,
accepted passage, usage total, and exported transcript. It confirmed Sol/max
in every request and response, identical prompt/player bytes and opening input,
unchanged runner and Terra artifacts, the unchanged original story seed, and
absence of credential values in experiment artifacts. Runtime code and prompts
did not change, so their offline tests were not rerun. Git whitespace checks
cover the new files while preserving the transcript's Markdown hard breaks.
