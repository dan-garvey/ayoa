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
and limits of the repair.

C29's [fresh A](proxy_trials/candidate_29/session_a/transcript.md) now completes
sixteen player turns, including two observed bouts, an aptitude assessment that
remains negative on retest, and an actual half-hour blade lesson. The
[locked pilot](proxy_trials/candidate_29/pilot_session_a.md) rates it 3/4/4/4/4,
with literary concerns and unresolved details of the lesson's footing and card
handling. The [independent review](proxy_trials/reviews/round29_a_review.md)
rates it 4/4/3/4/4: it finds the prose retainable but most NPC purposes less
developed than Ashara's competition. It identifies a definite local continuity
and authorship error when Ashara sees a card being pocketed that the player
already put away before leaving the assessment office. Her knowledge of the
result is independently supported by the player's submitted speech; this is
not a secret-knowledge leak. The court petition has no observed institutional
outcome in this session; no reopening is claimed. The two bout formats can be
interpreted consistently, but their identical spoken announcement and the
seller's inaccurate description of the displayed cards remain confusing.

[B](proxy_trials/candidate_29/session_b/transcript.md) pauses after five player
turns. The [locked pilot](proxy_trials/candidate_29/pilot_screen_b.md) rates the
segment's prose and distinction 3/3; the [independent reviewer](proxy_trials/reviews/round29_b_screen_review.md)
rates both 4, finding the ensemble's wit and quiet progression retainable with
local tightening. Preserve this disagreement. Both assessments credit the
player's chosen quiet, the garden view and independent relationships rather
than demanding an urgent interruption. A [POV addendum](proxy_trials/reviews/round29_b_pov_addendum.md)
finds definite localized narrator overreach in the opening under the explicit
perception rule, without an NPC secret-knowledge breach. These are 27 published
passages across A, B and the replays, produced by 54 sequential author requests
plus one recorded technical path-correction message. B is not a substantial
second trial. The [decision](proxy_trials/candidate_29/decision.json) rejects
C29 as tested while preserving the reviewers' disagreements.

A separate [edit diagnosis](proxy_trials/reviews/round29_edit_diagnosis.md)
finds B03's revision removes a substantive textbook disagreement while preserving
a familiar sequence of jokes. Candidate30 tests one proposed replacement editing
paragraph against the unchanged C29 paragraph. Each fresh editor receives the
complete background, published history, submission and original draft. Two
cases and two repeats per variant yield eight saved revisions and zero fresh
player turns. Both variants use the same explicit file-delegation wrapper;
historical C29 publications are reference material, not the fresh control.
The [locked pilot](proxy_trials/candidate_30/pilot_comparison.md) finds no
consistent dinner improvement: the changed paragraph produces its most and
least preferred dinner responses. It finds all four manuscript responses
retainable. The [independent comparison](proxy_trials/reviews/round30_blind_comparison.md)
receives anonymous outputs and complete published context, without drafts,
prompt variants or pilot ratings. It rates book_b2 at4 and the other dinners
at3; it rates reading_b2 at3 for serial literary appraisals and the other
readings at4. These differ from the locked pilot and remain intact. The
[decision](proxy_trials/candidate_30/decision.json) records an inconsistent local
gain and a possible reading over-correction; the paragraph is not promoted.

Candidate31 separately tests exposure to earlier discarded drafts. Both
conditions start with fresh whole-story Terra/max authors and unchanged C29
prompts, complete backgrounds, published B00–02 and the same B03 submission.
One condition also includes the exact B00–02 discarded fiction, explicitly
marked as superseded. Each author generates a new draft and then receives the
unchanged editorial task. Two repeats per condition now produce four saved
continuations and eight sequential requests. The
[locked pilot](proxy_trials/candidate_31/pilot_comparison.md) rates the revisions
4/3 without old drafts and3/3 with them, with no consistent remedy established.
The second revision without old drafts removes its own draft's unsupported Dena
reference and makes separate endorsements coherent. The first leaves a
four-minute deadline without its removed negotiation. The
[blind comparison](proxy_trials/reviews/round31_blind_comparison.md) rates one
repeat in each condition4 and the other3, retaining exposed_2 where the root
does not. The [decision](proxy_trials/candidate_31/decision.json) records no
reliable old-draft exposure advantage. This tests additional fiction context
under a matched setup, not persistence itself or production API cost.

The [first-pair diagnosis](proxy_trials/reviews/round31_first_pair_diagnosis.md)
recommends testing the representation of published history next. Candidate32
prepares raw-history and semantic-record conditions while keeping the entire
Covenant, persona, player submissions and authored material unchanged. The
records received an [independent fidelity audit](proxy_trials/reviews/round32_history_fidelity.md)
before authors ran. It caught weakened refusal language, an altered access
condition, lost fairness reasoning, and smaller attribution/ownership issues.
The [resolution](proxy_trials/candidate_32/fidelity_resolution.json) records all
corrections, retains the pre-audit records, and states the remaining presentation
losses. All eight one-pass continuations are saved, with no editorial stage.
The [locked pilot](proxy_trials/candidate_32/pilot_comparison.md) rates every
dinner at3; it retains one raw-history reading at4 and rates the other readings
at3. The [blind comparison](proxy_trials/reviews/round32_blind_comparison.md)
also rates all dinners at3. It retains one reading in each representation,
disagreeing with the root about reading_record_1. It identifies a local mismatch
when book_raw_2 treats Aldric's request to review the petition as new despite
his earlier request and Rashid's acceptance. No new material authorship or
secret-knowledge breach is found. The [decision](proxy_trials/candidate_32/decision.json)
does not promote the history record. Public tool records verify all eight
authors read their complete instructions and packets with Terra/max. This is an
isolated context diagnostic, not a runtime summarization feature.

The [scene-selection advice](proxy_trials/reviews/round33_scene_selection_advice.md)
finds that existing guidance and outputs already allow summary and selective
participation. Its remaining hypothesis concerns sustained attention across an
extended activity. Candidate33 changes just two pacing sentences, preserving the
complete Covenant and raw histories. All eight first continuations are saved.
The [locked pilot](proxy_trials/candidate_33/pilot_comparison.md) rates all dinners
at3 and retains only reading_scene_1 at4. It finds the new instruction does not
reliably concentrate attention on a stronger interaction. The
[blind comparison](proxy_trials/reviews/round33_blind_comparison.md) also rates
all dinners at3 and retains one reading in each condition. It additionally flags
Thessaly's table-wobble exchange as violating her truthfulness constraint. The
[root response](proxy_trials/candidate_33/review_response.md) regards that
reference as ambiguous and retains the concern; no change to the binding is
proposed. The [decision](proxy_trials/candidate_33/decision.json) does not promote
the pacing change. One output-path clarification and one exact capture of returned fiction
are recorded in [delivery variations](proxy_trials/candidate_33/delivery_variations.json);
neither requested a new draft. All instructions and packets were read in full,
including the corrected final-line read.

Candidate34 changes only the common prompt's opening sentence to request
naturalistic interactive fiction in a contemporary literary register. The
complete Covenant, remaining rules and raw history are exact. All eight first
outputs are saved. The [locked pilot](proxy_trials/candidate_34/pilot_comparison.md)
and [independent review](proxy_trials/reviews/round34_blind_comparison.md) rate
every dinner at3 and retain both reading controls at4. The independent review
also retains reading_literary_1, which the root regards as too prescriptive.
No definite new material boundary failure is found; local POV and credibility
repairs remain. The [decision](proxy_trials/candidate_34/decision.json) does not
promote the cue. This was a positive style-cue hypothesis, not established model
guidance, and does not establish a broad model limit.

Candidate35 starts two fresh sessions with unchanged C29 generation instructions
and a different [current situation](proxy_trials/candidate_35/current_business.txt).
Rashid has overlapping commitments to cards with Dena and friends and practice
with Ashara. The parties know different parts of the arrangements. All Covenant
material outside current_business remains byte-identical, including the full
character backgrounds and Dena's paragraph. An active root-driven player and a
separate observant player driver follow the same opening and persona. Each
session uses one persistent Terra/max whole-story author with one first passage
per submission. Both [A](proxy_trials/candidate_35/session_a/transcript.md) and
[B](proxy_trials/candidate_35/session_b/transcript.md) complete five player turns,
with twelve first publications totaling 11,209 story words. The
[locked pilot](proxy_trials/candidate_35/pilot_review.md) rates A at 3/3/4/4/4 and
B at 3/3/4/3/4. A follows the scheduling dispute through actual drills and Rashid's
return, while B stays with dinner and cards. The shared dialogue rhythm persists
even among the new friends. A's last game both denies and relies on Dena having
displayed a banner; B's last game supplies an unsupported winning total including
an apparent transfer of Rowan's crossing. These material state concerns prevent
selection independently of the prose scores. The
[independent review](proxy_trials/reviews/round35_review.md) rates A at 3/3/4/3/3
and B at 4/4/4/3/3. It retains B's social writing where the root does not, while
confirming the game-state failures and crediting A's completed practice. The
[response](proxy_trials/candidate_35/review_response.md) preserves these differences;
the [decision](proxy_trials/candidate_35/decision.json) rejects C35 as tested.
An [export correction](proxy_trials/candidate_35/export_correction.json) fixes
only an opening-inclusive turn count and transcript headings, leaving every
player submission, story, and original pilot judgment intact.

The [delivery audit](proxy_trials/candidate_35/publication_delivery_audit.json)
finds the public final answers have the same words as the saved story files after
declared formatting normalization. Nine files differ in quotation typography or
literal line-break escapes; none of the originals is changed. The dialogue and
state defects exist in both surfaces. C36 tests a return-only author, whose exact
final fiction is captured by the root. C35 broadens situation coverage and does
not establish a generic prompt-quality gain.

Candidate36 follows the [delivery advice](proxy_trials/reviews/round36_prompt_advice.md)
with eight matched first outputs: a fresh opening and a quiet dinner continuation,
each with two repeats per delivery condition. All C35 common instructions,
backgrounds and persona remain exact, and the two conditions share identical
context packets. The save condition writes a separate artifact and returns
fiction; the return condition emits final fiction only. The root captures the
exact first public final in both conditions for the primary comparison. This
keeps formatting differences in the authored files from masquerading as better
composition. All eight first outputs are saved, totaling 5,965 story words. The
[locked pilot](proxy_trials/candidate_36/pilot_review.md) rates all four openings
at4 for prose within their limited scope and all four dinners at3. The familiar
reply pattern remains in both conditions, and dinner_save_2 gives Ashara the date
of an earlier promise without a disclosed source. All full packet/task reads,
Terra/max settings and exact public-final captures are verified. The two control
openings match their authored files except for terminal newlines; the two control
dinners differ in formatting but have the same words under declared comparison
normalization. Originals remain unchanged. Blind review is running; there is no
editor, selector or selected prompt. This tests a combined task/delivery obligation
and within-task effects, not accumulated repetition across a persistent
conversation or production API costs.

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
