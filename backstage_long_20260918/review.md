# Longer evaluation: useful checks, mixed narrative benefit

The periodic checkup is useful as a diagnostic aid, but this run does not show a
reliable improvement in the resulting fiction. Before reading any private notes
or opening the A/B mapping, I slightly preferred Covenant with checkups and
Breakwater without them. Both preferences had substantial counterexamples.
Checkups caught real drift, repeated a mistaken diagnosis, and failed to make
Seraphel's rhyme rule reliable. They also add substantial delay in this coding-proxy
setup. I would keep the mechanism inspectable and experimental, and improve the
precision of its corrections before making it more frequent or claiming it as
the solution to narrative direction.

## What ran

Four 24-turn histories: Covenant and Breakwater, each with checkups every five
messages versus disabled. Each pair shares one generated four-turn opening, then
receives the same fixed player inputs for twenty further turns. The enabled arms
have checkups before T5, T10, T15 and T20, giving four repeated cycles per story.
There were **96 unique calls: 88 author responses and eight checkups**. All first
outputs were retained; no quality resampling, editorial pass or direct API call
occurred. All calls used fresh Terra/max coding proxies and requested detailed
exposed reasoning summaries.

Author/checkup prompts, complete biographies, direction, model and runtime were
frozen at active commit `661b41c`. The control uses the same base author prompt,
including its conditional provision for private guidance, but receives no guidance.
The experiment isolates using the checkup mechanism, not every prompt change
made during its implementation.

I read all prose before the notes and recorded the assessment in
[prose_review.md](prose_review.md), then [froze its hash](review_before_unblind.json)
before opening the mapping. This is only partially blinded: I designed the trial,
and slower completion can suggest which arm makes extra calls. There is no
independent panel or robust population estimate here.

The actual due-turn loop is **player input → checkup → author → published passage**.
The checkup gets the stable author rules, direction and full canon, the active
published history, previous guidance and pending input, followed by its checkup
task. The author waits for those new notes and receives the same story context
with the new guidance. Only the latest notes are carried forward. At ordinary
turns there is only the author call. Notes remain outside published history and
are not NPC knowledge. Independent story runs can execute concurrently; the two
stages within a due turn are sequential.

## What improved, and what did not

| Dimension | Covenant | Breakwater |
| --- | --- | --- |
| Overall reading preference | Slight preference for checkups | Slight preference for no checkups |
| Independent development | Checkup arm gives Dena a direct boundary and a concrete student placement; both arms complete drills/cards and the bout | Control gives Owen a private conversation and later shift cover; checkup arm leaves that thread pending |
| Specific strong scene | Checkup arm's 3–1 duel lets Ashara lose a point and adapt | Checkup arm's lunch with Bea uses her ushering, payroll work and granddaughter's films |
| Persistent weakness | Procedural lectures, cast commentary around Rowan, uneven verse | Repetitive archival instruction and repeated reassurance about the locked album |
| Privacy and access | Both protect the private bird thought and restricted records | Both protect the unwritten title, album consent and bounded work contract |

### Character independence and room for the player

This is the strongest shared success. Ashara is not summoned back merely because
the player conditionally offers to practise. Seraphel can remain absent when the
player would otherwise ask about reading. Owen can be busy. Reading, a garden
walk, cataloguing, departures and a refusal of extra shifts reach their endpoints
without another arbitrary personal demand. Ordinary hospitality or a reciprocal
question about work still occurs; those are not automatically failures.

Both conditions already use the recently improved closure wording. These outcomes
therefore do not demonstrate that periodic checkups caused the improvement. The
script also says explicitly to leave people alone or avoid extra involvement more
often than an ordinary live player might.

The best independent thread in Breakwater occurs **without checkups**. Owen asks
Nell for a private conversation in T6. The player cannot hear its contents. By
Wednesday, Nell is arranging Thursday cover, and Owen later corrects "see you
tomorrow" to Friday. Something changes in their lives without turning the player
into their coordinator or disclosing the interview. The checkup arm repeatedly
lists that dilemma but does not visibly progress it. Its story ends before the
Wednesday-noon reply deadline, so this is an unused opportunity, not a proven
deadline violation.

### Dialogue and scene quality

Covenant's checkup arm has quieter library scenes, useful adviser/placement detail
and a better contest: Ashara concedes a point, adapts, wins 3–1 and receives a
complaint about her heated blade. That is a full activity worth experiencing, not
an invitation for the protagonist to train with her. Its long orientation answer
still expands into comments about Rowan's preparation from several residents.
The control's orientation is worse in that respect, with nearly the whole cast
contributing to one question. The main habit has softened in places, not vanished.

The control's song exchange is more satisfying because we actually hear a stanza
and other residents' different preferences. Its verse also contains padding,
including "all tomorrow's more." The checkup arm gives a shorter recommendation
but then lets Seraphel end with the unrhymed "Not difficult ... Just easy to spoil."
The control also breaks genuine rhyme with voice/voice in T15. After the shared
opening, she speaks in three passages with checkups and five without; each arm
has one clear passage-level rule failure. These small, unequal opportunities do
not establish a reliable difference. Silence is not counted as successful verse.

Breakwater's checkup arm has a good lunch with Bea, whose personal answer comes
from her actual biography and ends when she needs to return to the boatyard.
Elsewhere both arms spend many passages cataloguing. Repeated instructions about
ownership and provenance are partly induced by the fixed player questions. The
checkup arm also repeatedly tells us the cupboard is still locked after the
player has already accepted the restriction. Compliance can become its own
repetitive narrative subject.

Quips remain in both stories. Some belong to the situation; the problem is the
habit of giving every response a neat ending or matching wit with more wit.
There is no defensible claim here that the checkups solve aphoristic dialogue.

## The checkups themselves

The [per-checkup review](checkup_review.md) checks consequential claims, uncertainty,
misses and following behavior. Several diagnoses are valuable:

- Seraphel's weak awhile/mild rhyme in the shared opening.
- Rashid's unsupported claim that the card players had agreed to start without him.
- Dinner departures being replayed, and the reading player silently moving rooms.
- Programme boards being carried into a room where canon already placed them.
- The need to keep private thoughts, loan permissions and student/Council authority distinct.

The earlier smoke test's Friday/Monday error **does not recur**. These reviews
correctly distinguish when a reservation was entered from when the room was used.
That is encouraging evidence for the temporal-verification wording, though not
proof of broad correction accuracy.

The clearest new failure is **rejecting a consistent addition to history**. Canon
says Rosa plans to send the exhibition layout Wednesday. The shared T2 says the
printer wants it by Wednesday. Both can be true. Nevertheless, three successive
checkups call the printer's request unestablished drift and direct the author
back to merely Rosa's plan. The event is already in the published history; this
is not a genuine contradiction. No later passage tests the printer's demand, so
the evidence establishes bad advice, not an observed downstream rewrite of that
fact.

Another note says not to invent an exact shift-ending hour because none was
established. A compatible working day is ordinary gap-filling, which the user
explicitly wants to allow. A Covenant note calls an "injured side" baseless even
though the bout first includes a staff touch to that side; that is at least an
overconfident diagnosis of an ambiguous detail. The review should not convert
"not explicitly explained" into "inconsistent."

The notes also repeatedly renew old material. Correct and incorrect reminders
about the first evening recur after the setting and task have changed. Keeping
only the latest note prevents a literal stack of notes, but the new note can
renew the old mistake. The previous note is in its context; inheritance is a
plausible mechanism, not a proven explanation for each repeat.

Planning can have observable value: Covenant's Dena exchange and explicit
adviser/placement closely follow the notes. Many other suggestions remain lists
of plausible threads. Accurate reminders do not guarantee execution: departures
repeat after a warning, and Seraphel breaks rhyme after four reminders.

## Size, delay and validation

Descriptive figures below cover only T5–24, excluding the shared opening:

| Story / condition | Mean words per passage | Passages ending in a question |
| --- | ---: | ---: |
| Covenant, checkup every 5 | 188.7 | 0/20 |
| Covenant, disabled | 203.9 | 2/20 |
| Breakwater, checkup every 5 | 169.7 | 2/20 |
| Breakwater, disabled | 161.2 | 1/20 |

These are not quality scores. An invitation need not end in a question, and a
good shared activity can deserve a longer passage. The direction of the length
change reverses between stories.

Checkups contained **290–528 words**, median 381. Logged dispatch-to-receive time
was a median **145.5 seconds per checkup**, versus **20.7 seconds per author call**.
Those timings exclude waiting for the trial's three-call concurrency limit;
the raw response duration includes that queue, which is why it differs. A due
turn waits for both stages, so the extra call produces a noticeable pause here.
These are observations of this coding-proxy run, not direct API latency or billing
estimates. All 96 calls returned nonempty exposed summaries; their presence does
not imply access to private reasoning or a complete account of model decisions.

The [offline audit](validation.json) reconstructs every exact request, checks
publication against raw first outputs, verifies four due checkups per enabled
arm, checks the latest guidance and all shared-prefix copies, and reconciles all
96 dispatches/receipts with 88 publications. Runtime, prompt, input and pre-note
review hashes match. Existing live-session files are byte-for-byte unchanged.
Ruff lint and formatting checks pass for the evaluation helpers. No production
code or prompt changed, so the previously completed runtime/browser suite was
not rerun as if this were a new implementation.

## Limits and next step

This is one paired trajectory per story, with stochastic divergence after the
common prefix. It is a much longer test than the smoke check, but not evidence
about hundreds of turns or a general superiority over character-agent systems.
The player script supplies much of the itinerary. It sometimes asks to finish an
already ended dinner or repeats an induction answer. Ashara has already left when
the conditional sparring offer occurs, so no actual sparring refusal is tested.
The runs test adaptive quiet and some direct engagement more strongly than they
test a lost player needing proactive story direction.

I would retain the current evidence and test a narrower correction contract next:
compatible facts introduced in published fiction count as established history;
absence from the original prompt is not a contradiction. Resolved findings should
drop out, leaving room to consider a current independent consequence rather than
continually rehearse old problems. That is a targeted prompt experiment, not a
case for another model layer or a larger planning structure. Rhyme needs its own
continued evaluation because reminding the author has not made it reliable.
These follow-ups are recorded as `ayoa-7gn7` (correction precision) and the existing
`ayoa-3gtc` (natural, reliable rhyme); the completed evaluation is `ayoa-h5s0`.

The production prompt, interval, live sessions and server remain unchanged by
this evaluation.
