# Terra proxy iteration: current evidence

The single-author approach has produced sustained, coherent social scenes with
different character interests and plausible limits on knowledge. It has not yet
met the complete selection bar across two fresh sessions. The remaining failures
are predominantly literary: recurring polished lessons, stilted conversation,
and, in some routes, helpful characters who leave too much initiative to the player.

The user permits new history that fills gaps consistently. Inventing a relative,
past incident, or an unshown but plausible action is not itself a failure.
Contradicting established events or giving an NPC inaccessible information is a
different issue. The [assessment amendment](proxy_trials/ASSESSMENT_AMENDMENTS.md)
records this distinction and its effect on earlier judgments.

## What is being tested

Each session has one Terra coding agent at maximum reasoning. It receives the
complete Covenant brief, including secrets, frozen storytelling instructions
and the player description. Through candidate22 these are in its initial packet;
candidate23 moves the same common instructions into its direct request while
keeping the full brief and turn data in the packet. A player submission produces
one passage from that same agent, which authors adjudication, every NPC and the
narration together. Candidate27 adds a second, sequential revision request to
that same author before passage delivery. Subsequent turns retain its story
conversation, including the unreleased drafts in that proxy. There is no
router-to-character-to-narrator cascade, separate character memory, private
planning model, or semantic validator in the story loop.

Separate sessions can run in parallel. A player driver chooses submissions in
some sessions; an independent reviewer evaluates completed transcripts afterward.
Neither supplies NPC dialogue or story direction to the author. Exact inputs,
outputs, packets, prompt hashes and reviews are preserved. Unfavorable prose is
not rerolled. Technical artifact recoveries and context-exposure limitations are
recorded where they occur.

This is a coding-agent proxy, with its surrounding instructions and file-writing
tools. It does not reproduce direct API hidden context, response ceilings, cache,
cost or latency. The earlier direct Terra and Sol trials are separate evidence;
their prompts and conditions differ. These results are not a controlled comparison
with Ayoa's existing architecture or a demonstration of unlimited coherence.

## Recent complete trials

Scores below are reviewer opinions in this order: prose/dialogue, character
distinction, independent interests, direction, adaptive initiative. Four means
strong enough to retain without a material rewrite; five means exceptional.
Selection requires both the pilot and independent review to meet four in all
five dimensions in two fresh substantial sessions, with no unresolved material
boundary failure. Supplementary replays and constructed cases cannot replace
those sessions. The full [protocol](proxy_trials/PROTOCOL.md) predates selection.

| Candidate | Fresh player turns A / B | Locked pilot A / B | Independent A / B | Outcome |
| --- | --- | --- | --- | --- |
| 14 | 15 / 22 | 4/4/4/4/4; 3/4/4/4/4 | 4/4/4/4/4; 3/4/4/4/4 | Family anecdotes and letters remain didactic. |
| 16 | 15 / 20 | 4/4/4/4/4; 4/4/4/4/4 | 4/4/4/4/4; 4/3/3/3/3 | B completes an itinerary but lacks sufficient reciprocal initiative. |
| 17 | 18 / 21 | 4/4/4/4/4; 3/4/4/4/4 | 3/3/4/4/4; 4/3/4/4/4 | More reciprocal presence; character voices and dramatization remain weak. |
| 19 | 15 / 15 | 3/3/4/4/4; 3/3/4/4/4 | Not commissioned after pilot rejection | Ordinary letters improve; games and refusals still become lessons. |
| 27 | 15 / 15 | 3/4/4/4/4; 3/4/4/3/4 | 3/4/4/4/4; 3/3/4/3/3 | Less habitual rhetoric; repeated positions and thin reciprocal interaction remain. |

Candidate17's [A transcript](proxy_trials/candidate_17/session_a/transcript.md)
develops Thessaly's game into a shared reading appointment and a book loan. A
separate access campaign reaches an actual hearing, a change in court access and
the first supervised session. Rowan can observe without becoming its spokesman.
The [B transcript](proxy_trials/candidate_17/session_b/transcript.md) follows a
declined petition into family letters, quiet company, a completed walk and a
second visit with Stone. Stone's chess anecdote initially muddles the players;
when asked, he explicitly corrects himself. That repair resolves the conflicting
testimony but does not make the original anecdote well written.

The B player driver saw unrelated completed snippets once through an unfiltered
agent-status response and reports not using them. Its author remained isolated.
This limits any claim that B was a strictly blinded player trial. Root supplied
the final clarification after the driver stopped at20. The [locked pilot](proxy_trials/candidate_17/pilot_review.json)
and [artifact audit](proxy_trials/candidate_17/validation.json) retain further
counterexamples and validation limits.

The [independent review](proxy_trials/reviews/round17_review.md) values B's family
letters more highly than the pilot, while finding A's game and reading too often
summarized. It rates character engagement below the pilot in both sessions:
different interests still share too much of the same restrained, politely witty
register. Both judgments are retained. The reviewer also identifies a localized
source gap when Aldric mentions a letter discussed upstairs with Ashara; no
unshown relay is treated as observed evidence to dismiss that concern.

Candidate19 changes the framed letter to ordinary correspondence about a named
relative's planned visit. Its [family replay](proxy_trials/candidate_19/family_replay/transcript.md)
and fresh letters become more particular. In [A](proxy_trials/candidate_19/session_a/transcript.md),
Ashara fulfills a ten-minute lesson and keeps her refusal of an extension, while
Thessaly completes a game. In [B](proxy_trials/candidate_19/session_b/transcript.md),
a family lookup leads to Rowan's own letter and a sunset invitation reaches an
actual walk and tea. The full trials still fall below the prose bar: game
corrections become maxims, and ordinary refusals or information acquire lessons
about the protagonist. The [pilot decision](proxy_trials/candidate_19/decision.json)
records the rejection and a possible one-year chronology discrepancy without
mistaking all new history for a failure. No independent full review was requested
after this clear pilot rejection.

## Later diagnostic screens

Candidates20 and21 test general dialogue examples, first isolated lines and then
an unrelated complete exchange. Neither produces a clear improvement across
the screens. Candidate20's game also invents present player dialogue during a
delegated game; candidate21 explicitly separates activity delegation from speech
and avoids that particular failure in its replay. Its two fresh sessions are
stopped after five and four player turns because the shared didactic register
remains. They are incomplete trials, not successful long sessions.

The user rejected character-specific speech examples as restrictive and difficult
to transfer between stories; none were implemented. A focused review of official
Terra and GPT-5.6 guidance found general instruction-clarity advice, not a
Terra-specific fiction recipe. The [research note](proxy_trials/reviews/terra_prompting_research.md)
records the sources and their limits.

Candidate22 shortens only the common instructions. Its complete Covenant brief
is byte-identical to candidate19, including all character backgrounds. Three
screens become longer but remain didactic. The user emphasized that rich
backstories are essential; reducing character context is not an acceptable
strategy. Candidate23 keeps the same prompt content and tests direct delivery
of the common rules. Its [game](proxy_trials/candidate_23/game_replay/transcript.md),
[Stone visit](proxy_trials/candidate_23/stone_replay/transcript.md), and
[opening](proxy_trials/candidate_23/session_a/transcript.md) show no clear
literary gain. Role and order change together in this coding-agent proxy; this
does not isolate the effect of an API system-message role.

Candidate24 separately tests editing, preserving C23's original passages as
unreleased drafts. Both revisions largely paraphrase the originals and retain
the troublesome rhetorical observations. Tool-read audits confirm that the
editors received the full character context without truncation. These revisions
are not live player turns and do not qualify a candidate for selection.

Candidate25 explicitly distinguishes events and useful information from removable
rhetorical observations. Its edits remove substantially more commentary while
preserving the game result and the family lookup. The [independent review](proxy_trials/reviews/edit25_review.md)
finds a material dialogue improvement in the game and an uneven improvement in
the Stone passage. It also notes thin anecdotes and administrative conversation.
This warrants testing the operations in normal generation; it does not establish
that a private editing instruction can reproduce a supplied-draft revision.

Candidate26 tests that transfer with the complete Covenant unchanged. Its two
fresh sessions stop after five player turns apiece. A practice bout receives
actual exchanges and a result, and a novel discussion has particular comic
detail. The pilot still finds too much shared, prepared commentary across the
cast, plus a closely read letter reduced to generic description. The
[decision](proxy_trials/candidate_26/decision.json) distinguishes unsolicited
judgments from a game metaphor the player himself introduced. No substantial
session pass or independent full-session score is claimed.

Candidate27 begins a different loop: one whole-story author composes a passage,
then receives a fixed editorial request before delivery. Both stages retain the
full backgrounds. Its two requests are sequential, with originals preserved;
they are not independent character-agent work or evidence of one-request cost.

Both candidate27 fresh sessions now complete fifteen player turns. A reaches an
actual bout, lesson, game and supervised court reopening; B completes music,
manuscript sharing and a morning seminar. The locked pilot rates A3/4/4/4/4 and
B3/4/4/3/4. The [independent B review](proxy_trials/reviews/round27_b_review.md)
rates3/3/4/3/3: habitual maxims have receded, but ordinary exchanges too often
contain only the minimum answer or permission. It credits the music and reading
as real follow-through and finds no definite authorship or knowledge violation.
The [independent A review](proxy_trials/reviews/round27_a_review.md) agrees with
the pilot's3/4/4/4/4: the court dispute reaches a real outcome through NPC action,
but settled positions repeat and some shared activities lack social texture.
The [supplementary review](proxy_trials/reviews/round27_boundary_review.md)
finds the new partial-report and private-reservation case respects the tested
boundaries. The separate game replay leaves its final distance comparison
inadequately explained; the incomplete board proves neither illegality nor a
successful repair. All37 published passages and74 sequential author tasks are
preserved. The [decision](proxy_trials/candidate_27/decision.json) rejects C27.
No candidate is selected. The full Covenant remains byte-identical to candidate19.

Candidate28 begins with three bounded revisions of C27's original drafts. The
new [revision task](proxy_trials/candidate_28/revision_task.txt) allows replacing
unpublished NPC choices while preserving published history and player authorship.
It asks the revision to develop a missing personal response when deleting
rhetoric leaves only an acknowledgment. Complete backgrounds remain unchanged.
These are editing diagnostics with zero fresh player turns; they do not establish
that a live session will improve.
The [independent comparison](proxy_trials/reviews/edit28_review.md) prefers
C28's specific manuscript response and, narrowly, its more social game. It prefers
C27's simpler group reply to C28's unsolicited writing workshop. The game also
has an ambiguous opening move order. The [decision](proxy_trials/candidate_28/decision.json)
retains the bounded gains without selecting C28 or claiming a live improvement.

Candidate29 tests normal generation followed by revision with a simpler common
prompt: no short-paragraph target or duplicate structural-edit instructions.
The separate revision also follows the player's conversational focus. The full
Covenant is unchanged. Its [reading continuation](proxy_trials/candidate_29/reading_replay/transcript.md)
gives Seraphel a particular reader preference and a childhood song. The
[game continuation](proxy_trials/candidate_29/game_replay/transcript.md) completes
play but reverses the fourth moves in narration. An explicit OOC follow-up
receives an admission and a coherent local correction. The
[independent review](proxy_trials/reviews/round29_replays_review.md) finds the
reading strong and the game socially engaging while retaining the initial error
and limits of the repair. Fresh A is still a short ongoing screen. No candidate
is selected from these replays.

Earlier candidates and rejected screens remain under `proxy_trials/`. Through
candidate17 there are47 exported sessions or short cases,304 saved responses and
65,117 words of generated story. These counts include replays and rejected
screens; they are not47 successful independent playtests.

## Implication for the original premise

Plausible character knowledge is an important narrative constraint. These trials
do not support treating an individual agent per character as a demonstrated
necessity for satisfying it. A single author can preserve unequal reports and
private thoughts while carrying NPC business through to consequences. The
constructed reschedule cases are particularly direct evidence: one character
retains the old arrangement, another knows only that it moved, and both acquire
the replacement details only after disclosure.

Structural separation provides a different kind of protection: an agent cannot
use a secret absent from its input. A shared author receives all the secrets and
must refrain from using them inappropriately. A few successful conversations do
not turn that behavioral discipline into a privacy guarantee. Nor does combining
the roles automatically produce good direction: the early proxy trials frequently
stalled in questions, lectures or promises. The useful improvement so far comes
from specific character interests, reciprocal initiative and completing events
within the scope the player requested.

The experiment remains separate from the main runtime. No proxy candidate has
yet replaced the root experiment prompts.
