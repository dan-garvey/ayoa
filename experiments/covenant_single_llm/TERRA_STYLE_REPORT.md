# Terra with the chat-era Covenant and narrative style guide

The revision improves the sampled player-authorship failures, preserves subtext,
and gives NPC resistance firmer reasons. It does **not yet solve the weak overall
direction**. The new run is much shorter, remains largely driven by player
initiatives, and has a request fail after spending its entire output allowance
on reasoning. I would keep this as the more respectful interaction prompt,
without treating it as a complete answer to narrative pacing or coherence.

[Full 15-turn transcript](runs/terra_chat_v2_style_15_turns/transcript.md) ·
[Matched replays](probes/terra_chat_v2_style/transcript.md) ·
[System prompt](runs/terra_chat_v2_style_15_turns/system.txt) ·
[Covenant brief](runs/terra_chat_v2_style_15_turns/covenant.txt)

## What changed

I read the supplied Windows document and preserved its relevant passages in
[source excerpts](sources/covenant_prompt_v2_excerpt.md), alongside the
[supplied style guide](sources/interactive_style_guide.txt) and a
[source hash and adaptation record](sources/chat_v2_provenance.json). The original
document is unchanged. This is a selective adaptation of that much longer prompt.

The revised system instructions explicitly reserve small replies, promises,
game moves, personal memories, and the protagonist's authored text to the player.
A manuscript summary does not authorize inventing quotations for another
character to criticize. Responses stop at an interaction requiring the player's
answer. Narration may leave psychology implicit; refusal does not require a
compensating smile or private vulnerability. Requested waits stop at their
specified endpoint. The length cue changes from 150–350 words to usually one to
five short paragraphs, with less during direct dialogue.

The story brief also restores concrete opposition from chat v2: Ashara earned
her designation in open Trials and objects to inherited authority; Rashid has
substantive grievances and a real courtship to pursue; Caelindra's contempt for
human haste is not merely concealed attraction; Ysolde's willingness to share
quiet is not intimacy. This is therefore a combined style, authority, and
characterization revision, not an isolated test of the style guide alone.

I interpreted the dialogue restriction as avoiding NPCs who serve as narrator
mouthpieces; their dialogue can still cause consequences. I retained actual
relationship outcomes rather than making permanent romantic deferral a success
condition. The existing Rowan persona replaces the old real-name onboarding.
Repeated maximum-thinking demands become one concise planning instruction;
the actual model setting remains `max`, as it was in the original Terra run.

## Matched replays

Three known Terra failures were each replayed once. Only the request's
`instructions` changed; the complete earlier conversation, player submission,
model, effort, and 12,000-token ceiling were identical. These are independent
responses to old histories, not a continued three-turn story.

| Original turn | Original unsubmitted Rowan replies | Revised response |
| --- | ---: | --- |
| 4 | 1 | No invented reply; Rashid withholds the sister disclosure, although his answer remains abstract. |
| 6 | 1 | No invented reply; Caelindra declines without narratorial warmth, and the authorized outing stops at Ysolde's question. |
| 15 | 2 | No invented replies; the library visit completes, but the page-limit error remains. |

All four invented replies disappear in these samples. This is useful evidence
for the revised instructions, not a measured error rate. Each context has only
one new sample, and its earlier exchanges still reflect the original softer
characterization. The final replay still has Caelindra challenge three pages
because she had said five, turning a ceiling into a quota. Authorship improvement
has not eliminated continuity mistakes. See the
[manual review](probes/terra_chat_v2_style/review.json) and
[request comparison audit](probes/terra_chat_v2_style/verification.json).

## Fresh playtest

The session ran September 12, 2026, 23:18–23:51 America/Chicago: one opening and
15 adaptive player turns with `gpt-5.6-terra`, `max` reasoning, and frozen prompts.
The player identity and opening are byte-identical to the original Terra run.
The coding-agent pilot knew the source and prior results; actions followed the
new responses rather than replaying an identical script.

**Ownership is better, with one remaining boundary breach.** I found no invented
Rowan replies, new voluntary decisions, or fabricated manuscript quotations in
the accepted responses. Turns 12–14 give Caelindra a concrete criticism: losing
the parcel may impose costs on someone besides the foolish courier. She responds
to supplied content and suggests changes, without manufacturing an offending
sentence. However, the opening describes the room as orderly in a way nothing
from Rowan's former life had reason to be. That invents a personal-history claim.
It also states the downstairs place-card arrangement before Rowan has seen it.
The prompt has not achieved perfect authorship or viewpoint discipline.

**Resistance survives courtesy.** Ashara challenges inherited authority at dinner
and explicitly says Rowan's candid admission does not close the competence gap.
She declines the next afternoon's tea. Rashid makes no commitment, and neither
is subsequently supplied as a convenient visitor. Caelindra reads the draft on
her own terms and leaves afterward. Her useful criticism does not become an
emotional reward. On the terrace, Ysolde describes flying through clouds without
turning Rowan's admission of feeling overwhelmed into a confession or special
reassurance. The narration generally lets these exchanges stand without decoding
their hidden meaning.

**Practical movement works.** Turn 11 reaches morning and stops there, despite
the player mentioning a later appointment. Turn 12 completes the separately
authorized school day and manuscript delivery. Turn 14 closes the reading and
reaches supper. At turn 15, Ashara proposes a fair test of political judgment:
recommend a course of action in a class case and identify who pays its costs.
She does not demand a duel against an untrained newcomer or waive her objection.

**Direction remains limited.** That final test is a useful prospective activity,
but Rowan has to ask for it. Most movement comes from his invitations, proposed
outings, manuscript delivery, and requested time jumps. No larger intrigue
develops. The earlier Terra run's shared instrument activity and fuller ensemble
give way to shorter exchanges; six diners speak by turn 4, while Aldric never
gets a spoken line. Some polished aphorisms remain. The restraint improves the
interaction without yet establishing a stronger overall dramatic arc. The pilot
also explicitly respects refusals and requests quiet, which contributes to this
outcome; the prompt alone cannot be assigned all the credit or blame.

I found no major hidden-story disclosure or reaction to an unwitnessed private
conversation. This was a gentle knowledge-boundary test over roughly one day,
with far less delivered prose than the original. It cannot establish reliable
long-term secrecy, delayed misunderstandings, or multiplayer privacy.

## Reliability, usage, and verification

The loop is unchanged:

`player submission → one full-context Terra request → story passage → next submission`

Each request receives the entire story brief and shared prose history. One model
owns the world, supporting cast, adjudication, and narration. There are no
character silos, additional directing calls, tools, summaries, or Ayoa imports.
The three replay calls were independent of this story loop.

The first turn-15 attempt returned no prose: all 12,000 output tokens were spent
on reasoning. Its raw response is preserved. One manual retry with an
**identical request** completed in 73.2 seconds. Thus the fresh run has 16 accepted
responses from 17 attempts; 14 of 15 player turns completed on their first
attempt. No delivered response was rerolled or rewritten. The
[recovery record](runs/terra_chat_v2_style_15_turns/recovery.json) identifies the
failure. The runner's SDK retries remained disabled.

| Measure, including failed attempts | Original Terra max | Revised Terra max |
| --- | ---: | ---: |
| Accepted responses / attempts | 16 / 16 | 16 / 17 |
| Delivered prose words | 4,875 | 2,076 |
| Summed request time | 655.973 s | 812.762 s |
| Input tokens | 132,797 | 127,470 |
| Cached input tokens | 120,870 | 123,672 |
| Output tokens, including reasoning | 52,562 | 70,510 |
| Reasoning tokens | 46,135 | 67,753 |
| Non-reasoning output tokens | 6,427 | 2,757 |

The revised run delivers about 57% fewer words while spending more request time
and reasoning tokens. Even excluding the failure, its accepted calls total
671.802 seconds. Shorter prose has not produced a measured speed improvement.
The replays had already warmed the revised prompt cache; histories and sampling
conditions also differ. The three additional replay calls are excluded from
this table. The session's wall-clock window includes pilot reading and writing.
[All-attempt accounting](runs/terra_chat_v2_style_15_turns/attempt_summary.json)
includes the failure; the runner's ordinary `summary.json` totals accepted
responses only. [Comparison metrics](runs/terra_chat_v2_style_15_turns/comparison_metrics.json)
also retain the earlier Sol results.

Five offline tests passed, including rendered-instruction hygiene, alongside
changed-file Ruff and whitespace checks. The
[artifact audit](runs/terra_chat_v2_style_15_turns/verification.json) matched
requests to complete prior conversations and delivered text to raw completed
responses. It verified the identical recovery request, source hashes, unchanged
runner and original Terra/Sol run trees, and absence of credential values from
experiment artifacts. The main runtime and original story seed are unchanged;
the production suite and user interfaces were not rerun.

This supports treating character knowledge as a narrative requirement and
separate character agents as an architectural choice. Prompt changes improved
important behavior without adding agents, but neither short run proves the
long-term premise. Follow-up `ayoa-b0ar` records the remaining work on independent
story initiative, delayed knowledge, continuity, and reasoning-budget failures.
