# Covenant of Thrones: 15-player-turn live playtest

## Result

The social premise works, but the opening is not yet a dependable player experience.
I made 15 distinct in-character submissions, all eventually accepted, plus four
extra recovery submissions. Five router batches failed. An overnight wait stranded
the player, and the final afternoon action remained unrendered after the autonomous
frontier emptied. This is a completed playtest, not a clean gameplay pass.

The best material was a tentative friendship with Rashid and a restrained connection
with Caelindra. The weakest material was the almost motionless ensemble, abstract
answers to personal questions, sparse environmental grounding, and interrupted
delivery. Fixing the latter does not require making Covenant less guarded or
less politically dangerous.

[Read the exact delivered transcript](2026-09-11-covenant-transcript.md).
[Inputs, accepted checkpoint mapping, and recovery ledger](2026-09-11-covenant-actions.json).
[Measured runtime totals](2026-09-11-covenant-metrics.json).

## Method and limits

- One AI-piloted playthrough, not a panel of human playtesters or a controlled A/B.
- Fresh session: covenant_playtest_20260911t185207z. No existing session was replayed,
  rewound, or edited; the original story seed was unchanged.
- Main revision: 863d1652a737afe28e49082cb7c4a524fbad241c.
- Played the intended player-authored Garvey-heir seat as Rowan Garvey, an adult
  human newcomer. Choices pursued companionship, dinner conversation, exploration,
  and a next-day invitation rather than combat or a preselected mystery solution.
- Used the actual persistent CLI and EngineBridge path, including bindings,
  autonomous workers, per-POV narration, delivery claims, and acknowledgments.
  This was not a Discord transport/mobile UI test.
- Kept the seed's D&D ruleset, automatic rolls, prose presentation, and 12-batch
  autonomous limit. Runtime defaults replaced the seed's obsolete model names:
  Terra router/narrator and Luna character agents. No model overrides were applied.
- The image gateway refused its startup preflight. Prose remained available and
  no visual-novel/image quality claim is made. Runtime model calls were text-only.
- Setup exposed some source metadata to the pilot; subsequent choices were grounded
  in the primer, the player's dossier, and delivered prose. This is not a blinded
  mystery-discovery test. Full hidden-state inspection was reserved for diagnosis.
- Session ran September 11, 2026, 13:52-14:19 America/Chicago. Setup, reading,
  diagnostic pauses, recoveries, and autonomous work are included in that wall time.

## What happened

| Player turn | Choice and result |
| --- | --- |
| 1 | Went from Garvey House to the welcome dinner. Blank output, then a routing error; defer recovered Ashara's greeting. |
| 2-3 | Took the offered seat, used self-deprecating humor, and asked about life outside assessment. Ashara answered and asked about the heir's ink-stained fingers. |
| 4-5 | Asked Rashid about a free afternoon, then invited him and Ashara to the house tomorrow. Both accepted in distinct voices. |
| 6-7 | Left space for the table, then teased its solemnity. Only Ashara returned the toast; Caelindra spoke when directly addressed. |
| 8 | Disclosed discomfort with inheriting a dead family's home. Caelindra's reply required recovery from the second ownership error. |
| 9 | Invited Caelindra for a quiet walk. A visibility error required an exact retry; she then chose to accompany Rowan. |
| 10-12 | Left dinner, explicitly surveyed the grounds, and sat above the storm beneath the Nexus. Caelindra respected the space and offered restrained reassurance. |
| 13 | Returned home and requested sleep until morning. Only going to bed rendered. Eleven autonomous batches later, a late goodnight appeared; sleep remained open with no frontier. |
| 14 | Repeated the overnight wait. Morning arrived immediately. |
| 15 | Prepared the common room and waited for the promised afternoon visit. A source/observer error required an exact retry. Preparation committed, but no final player passage or guest arrival was delivered. |

The opening command is not counted as a player turn. Neither are the two recovery
defers or the exact retries of turns 9 and 15. Turn 14 is counted and identified as
a repeated *fictional wait*, rather than hiding its poor UX as technical recovery.
Checkpoint revision 48 is not “48 player turns.”

## Player-experience feedback

### What worked

**A worthwhile social hook.** Rowan could turn personal vulnerability into an
invitation and choose whom to spend time with. Rashid's answer was the strongest
early line: “Sleep. Then I'd find out whether I know what to do with an afternoon
no one is measuring.” It suggested a person under pressure rather than a quest
dispenser. Ashara's promise to attend without promising to “waste” the hour also
kept her character intact.

**Restraint without automatic rejection.** Caelindra accepted a low-pressure walk,
kept physical space, and did not become instantly adoring. “You need not make a
performance of being unimpressed” was a believable small connection. The player
could invite, disclose, choose a route, and leave without the narrator choosing
consent or romance on their behalf.

**The setting can provide real choices.** The explicit survey at turn 11 revealed
gardens, a lit colonnade, and a terrace above the storm. That immediately gave the
player something concrete to choose and made the location feel fantastical.

### What needs work

**The opening undersells its own premise.** The first passage is only 135 characters
about a coat and signet in a bedroom. The dinner hook is in the separate briefing,
not the opening scene. The dining-room arrival then introduces seven names with
very little physical differentiation. The spectacular Nexus setting does not
become useful until the player explicitly asks for surroundings.

**The player does too much conversational work.** Only Ashara, Rashid, and Caelindra
authored player-observed NPC events: 5, 2, and 7 respectively. Some are gestures,
not dialogue. Ysolde, Seraphel, Thessaly, and Aldric were seated at dinner but never
contributed a player-visible action. Letting the table speak produced another
Ashara toast, not a developing group conversation. This is evidence from one
social route, not a demand for equal airtime or a fixed speaking rotation.

**Personal questions often yield polished abstractions.** Asked for a small thing
she missed from home, Caelindra answered about silence becoming evidence. It fits
her register, but another carefully balanced maxim is less intimate than a
grounded personal detail. These words already occur in her stored character-agent
output and the canonical event; the narrator did not invent the stuffiness.

The current actor facts also have incomplete referents. Ysolde refers to “her
sonnet sequences”; Caelindra says “I have not told her I have read it”; Rashid
mentions “the absence of an answer.” Rashid's established pursuit of Ashara is in
world/narrator material but absent from his own rendered identity. The owner-bounded
context builder is working as written. The generic agent correctly forbids
inventing missing memories and relationships, so repair the authored facts rather
than weaken that boundary. This is a likely contributor, not isolated causal proof.
Follow-up: ayoa-uszv.

**The scene is reactive more often than alive.** Much of each passage repeats the
player's speech and adds one short answer. The router's canonical packets are
usually just submitted speech/body movement; the narrator is not authorized to
manufacture a bustling meal around them. The generic “silent NPCs recede” rule
and compression priorities compound sparse canonical material. Prefer grounded
relationship pressure and observable scene developments over merely asking the
narrator for more adjectives. No conspiracy investigation, combat, or developed
romance was exercised in this run.

## Runtime findings and durable follow-ups

| Priority | Bead | Evidence and consequence |
| --- | --- | --- |
| P1 | ayoa-qm03 | Two NPC replies selected player_garvey for an autonomous character turn. The ownership guard rejected the entire batch, at turns 1 and 8. |
| P1 | ayoa-3k2b | Turn 9's aside was visible only to Caelindra, but the router listed the whole table as observers. Structured validation correctly rejected observers receiving no facts. |
| P1 | ayoa-ryoh | Off-stage work referenced source event 27 in a one-event batch; turn 15 later sourced Rashid from private preparation he had not observed. Both causal-source guards rejected the output. |
| P1 | ayoa-9u8g | The overnight commitment remained open after 11 batches with an empty frontier, below the limit of 12. Turn 14 was required to reach morning. |
| P1 | ayoa-ofsy | Turn 15's narration chose continue, then its continuation exhausted the frontier. A pending narrator job remained with no runnable work and no final passage. |
| P2 | ayoa-grm1 | Ten empty CLI story headings appeared before deferred narration. Discord's earlier placeholder fix did not cover this CLI branch. |
| P2 | ayoa-w29q | Delivery-only bookkeeping changes invalidate the autonomous commit fingerprint. Live identical batches were rerouted; an offline probe confirms the invalidation mechanism. |
| P2 | ayoa-uszv | Incomplete actor facts and missing owner-known relationships merit a reviewed Covenant seed pass. |

These failures are not reasons to remove ownership, privacy, or causal validation.
The rejected fictional drafts were not canonized. Exact rejected outputs and
pre-failure checkpoints are preserved for narrow regression work.

**Chronology is also visible to the player.** Update 33 had already taken Rowan
home to bed, ending at fictional time 1252. Update 44 then rendered Caelindra's
goodnight as current prose, although that event occurs at time 652. This is a
late cross-delivery backfill, not the within-event sorting fixed by fx6w. It is
recorded with the overnight issue; changing decorative wording would not repair
the delivery sequence.

**The final stall is not merely “the visitors are late.”** The final save has one
pending narrator job, zero frontier turns, zero open commitments, and all 21
existing delivery entries acknowledged. The accepted common-room preparation is
still buffered. The final Rashid packet says about 13 hours 19 minutes have passed,
but his contribution answers the previous dinner conversation at fictional time
107, still saying “tomorrow,” while Rowan's current event ends at 48052. No arrival
was established. The CLI status misleadingly says a story update is waiting to
finish rendering; its last visible passage is still waking up from turn 14.

## Call cadence and validation

The actual common dialogue path was: player input -> router -> candidate narrator
(often continue/empty) -> autonomous NPC -> router -> narrator -> outbox delivery.
Independent off-stage agents sometimes prepared beside narration; same-conversation
replies were sequential. For the two-person invitation, Rashid's answer was routed
before Ashara answered, and both were eventually rendered together. This was not
one independent parallel call for each diner.

There were 130 HTTP 200 responses and 129 logged completions: 56 router, 29 narrator,
and 44 agent completions. Logged input totals were 1,112,656 tokens, of which
781,412 were cache reads; output totaled 69,134 tokens. These are logged usage,
not billed totals: structured validation can raise before usage logging. They
include retries and off-stage work. The 44 agent completions produced only
1,619 visible output tokens, with 19,649 reasoning tokens. No claim that changing
reasoning settings would preserve quality has been tested.

For the repeated-call finding, the commit freshness fingerprint hashes the full
checkpoint, while the preparation fingerprint excludes delivery bookkeeping.
Changing only a delivery attempt counter in a copy of real checkpoint 30 changes
the former and leaves the latter identical. This proves a transport-only mutation
can invalidate story work. Attribution of each observed reroute to an ACK is
inferred from the code and timing, not separately instrumented. A fix must preserve
concurrent ACKs, not overwrite them by simply ignoring their fields.

Offline extraction verified all 15 inputs against the raw CLI log, the two exact
retries, accepted checkpoint paths, five rejected batches, delivered prose against
the log, all delivery acknowledgments, the stranded final job, and the unchanged
seed hash. Both local analysis helpers pass Ruff. No production code or prompt
was changed, and the full application test suite was not rerun for this report.

## Preserved artifacts

The raw evidence directory is
app/storage/playtest_reports/covenant_15turn_20260911t185207z/.
It contains the original CLI typescript, a cleaned log, five extracted rejected
batches, exact player inputs, measured totals, a delivered transcript, the seed
snapshot, failure/stall snapshots, final checkpoint 48, and reproducible analysis
and fingerprint-probe helpers. Private/internal model context remains in these
local artifacts rather than the player transcript.

All 49 checkpoint files remain in
app/storage/sessions/covenant_playtest_20260911t185207z/.
The seed SHA-256 before and after was
8ea42185fd3592dd107869d0a06f1d293e38a07c61ff3e09a41c887d41242b23.
The test CLI exited normally. The Discord bot was not restarted or altered.

Recommended order: repair the blocking router/source and narration-liveness
contracts first; then repair the actor-owned relationship facts and retest the
same dinner-to-next-day social route. Preserve the guarded tone while giving the
characters concrete reasons to engage with one another.
