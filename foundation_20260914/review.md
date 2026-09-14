# Foundation playtest review

This banks a usable author/editor foundation and targeted dialogue improvements.
It does not establish that the 40-word editor reliably improves an entire scene,
or that the broader literary goal has been achieved. The exact C43 author and
augmented editor remain the working baseline; this implementation run makes no
new prompt-selection claim.

## Scope and source fidelity

The [protocol](protocol.md) specifies an opening plus three player turns in each
story, with a fresh Terra/max proxy for every author and editor response. Root
chooses player actions from published scenes and reviews complete pairs. There
are no independent graders, prose rerolls, corrective editor feedback or direct
API calls. This is a short implementation smoke test, not a controlled comparison
with the old engine or a long-session coherence evaluation.

The reusable author is the unchanged 393-word C43 prompt, followed by story
direction and full canon. The editor receives the same context, its current draft
and the exact selected 40-word task. Published edits alone enter future history.
Covenant uses the restored C41 material; Breakwater keeps the complete C41 brief.
The [migration record](source_migration.json) accounts for the separation without
summarizing biographies. All request sources are frozen inside the sessions.

All sixteen first responses were accepted, producing eight published passages.
The [validation audit](validation.json) reconstructs every request from those
frozen sources and previously published history, checks the raw accepted output
and verifies complete public input reads. All sessions are idle and complete.
Twenty-seven offline tests passed, including the real SDK with mocked HTTP,
API/proxy request equivalence, publication, failures and restart recovery.

| Story and turn | Draft words | Published words |
| --- | ---: | ---: |
| Covenant opening | 399 | 447 |
| Covenant 01: join dinner | 681 | 794 |
| Covenant 02: listen through dinner | 856 | 1,092 |
| Covenant 03: quiet evening | 338 | 533 |
| Breakwater opening | 270 | 526 |
| Breakwater 01: film preferences | 791 | 645 |
| Breakwater 02: listen through dinner | 1,144 | 1,409 |
| Breakwater 03: ten minutes drying dishes | 551 | 725 |

## What improved

In [Breakwater turn 01](breakwater/01.author.final.txt), Owen's line “I maintain
that a thin slice is just a small disappointment” becomes “The first ones were
too thick” in the [edit](breakwater/01.editor.final.txt). The exchange retains
the cast's individual film preferences and practical involvement with the cinema.
That is a direct success of the kind sought in the sugar comparison: the line
does useful conversational work without turning into a detachable maxim.

The [Covenant turn 02 edit](covenant/02.editor.final.txt) removes the draft's
primer/tea formulation, the generalized observation about quiet rooms and the
extended pear/history joke. Caelindra's complaint about an abbreviated treaty
history becomes a substantive disagreement with Aldric about the assigned
reading. This adds specific grounds for their positions. The result is still
too extended and includes an attribution error described below.

There are meaningful NPC interests beyond Rowan. Rashid's competing card and
practice commitments emerge at dinner; Ashara presses him to honor the practice
he promised. She reacts after he discloses the other commitment on the page,
rather than knowing it automatically. In Breakwater, Nell upholds Rosa and
Martin's auditorium reservation. Dev contacts Hana and postpones the receiver
test when she cannot come earlier. Rowan is not required to resolve either
conflict, and the cinema work continues after he leaves it until induction.

## What remains weak

Cleanup is inconsistent. The [Breakwater turn 02 edit](breakwater/02.editor.final.txt)
removes Martin's generalization that a board should look as though someone cared
enough to put it there, but introduces Jo's “It’s for anyone who thinks a second
piece of bread is a personality trait.” That stock joke is absent from the
[draft](breakwater/02.author.final.txt). A revision can satisfy the narrow removal
request in one place while creating another irritating line nearby.

Both dinners repeatedly use a correction followed by a counter-correction to
keep every speaker involved. Covenant spends much of its arrival scene on pears,
then extends the same motif into the rest of dinner. Breakwater has more varied
interests, but often voices them in similarly prepared exchanges. The editor can
expand already prolonged scenes: Covenant's dinner moves from 856 to 1,092 words;
Breakwater's moves from 1,144 to 1,409. These are observations about these drafts,
not evidence for imposing an arbitrary word quota.

In Covenant turn 02, Ysolde asks who noticed the north bridge lamp. Later Ashara
says “You asked whether I had seen it,” and Rashid answers “I asked whether anyone
had.” Rashid did not ask that question in the visible exchange. This local speaker
attribution slip is introduced while the editor expands a summarized conversation.
It is a continuity error, distinct from inventing permissible background history.

Breakwater's final edit adds an apparent object-continuity regression: turn 02
says that the pie dish holds only crumbs, but turn 03 introduces a remaining
wedge to take home without establishing another pie or a set-aside portion.
This is a concern about compatibility with the published scene, not a ban on
new everyday details.

New personal recollections, the identity of a former cinema worker, ordinary
school details and the newly supplied seminar time are not treated as failures
merely because they were absent from the initial brief. No contradiction with
established facts was demonstrated for those additions. The user explicitly
allows consistent gap filling.

## Authorship, knowledge and initiative

Covenant's opening draft assigns Rowan a small unsubmitted internal state about
an ancestral name. The edit removes it. Later, the player privately says that he
has not decided whether he likes Rashid, while explicitly neither saying nor
signaling it. The published dinner does not react to that private opinion.
The double booking has an intelligible disclosure path, and the hidden conspiracy
does not become dinner conversation.

Both published dinners reach the requested end of the meal without requiring
another player response. Breakwater accepts Rowan's decision to leave cinema
work until the following day's induction. Covenant's final turn accepts his
refusal of cards and practice, then renders only the delegated unpacking and
list of names heard at dinner. It ends after the requested twenty minutes with
no intrusion, compulsory mystery or invented private judgment. The final edit
nevertheless adds another substantial group farewell, increasing 338 words to
533 instead of allowing the brief departure to remain brief.

Breakwater's final draft and edit both render the authorized ten minutes of
drying dishes, add no new player speech beyond repeating the submitted line,
and keep the interview and other undisclosed matters private. The edit replaces
some snappy corrections with Jo and Owen discussing the café's cupboard space
and their other obligations. It also retains the draft's endpoint error: despite
“Keep the scene here and stop after ten minutes,” the closing sentence moves
Rowan out of the kitchen toward the stairs. Going upstairs was his stated later
intention, but the requested scene stopped before that transition. The short
editor did not repair this scope overshoot.

## What this establishes

The foundation retains the evidence needed to distinguish author behavior from
editor behavior. It can carry rich, separate story sources through a common
context contract, preserve first drafts, publish edits, resume interrupted work
and export readable fiction. The literary evidence supports local rhetorical
cleanup and several instances of NPC independence and flexible initiative.
Persistent overextension, prepared banter and local continuity errors remain
work for the broader prompt-iteration task.
