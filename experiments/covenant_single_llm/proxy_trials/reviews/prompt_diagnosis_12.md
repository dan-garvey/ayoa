# Candidate 12: choose the contribution before composing the scene

**Recommendation: replace the system prompt with the version below and keep the Covenant verbatim.** Test a change in the unit of continuation: the next useful contribution to the current interaction, allowed to occupy one paragraph, rather than an implicitly complete ensemble scene. This is a hypothesis about response selection and scale, not a demonstrated cure. Another catalogue of natural-speech prohibitions is unlikely to clarify the task.

I read only the ten supplied files. I did not inspect earlier system prompts, call another model, or run a story test. References below are relative to `proxy_trials/`. The family-history session is represented here by the supplied independent round-10 review and the Stone replay; I did not read its original full transcript.

## What the evidence supports

**The model can already produce the desired kind of conversation.** In `candidate_10/session_b/transcript.md:230–250`, Ysolde's substantial answer about flying earns its length through currents, lightning, disappearing bridges, and her practical experience. Her later refusal and alternative outing remain compatible in lines 258–290. The practice group reaches a working arrangement without Rowan organizing it, and Dena subsequently wants an evening out (lines 346–442). The review rates this quieter session 4 in every dimension (`reviews/round10_review.md:5–20`). A wholesale personality rewrite, a low word cap, or a universal ban on explanation would put those successes at risk.

**The cast rounds are an observable composition habit, although not every multi-person exchange is a defect.** Candidate 11 session A's welcome gives all seven named peers a contribution in approximately 199 words (`candidate_11/session_a/transcript.md:48–71`). Its direct question to Ashara expands into contributions from Rashid, Ysolde, and Caelindra (lines 81–109). Session B's practice argument similarly tours all seven peers in one response (`candidate_11/session_b/transcript.md:45–64`). By comparison, the corresponding C10 dinner stays largely with Rashid, Ashara, and Aldric, while explicitly letting Caelindra read (`candidate_10/session_b/transcript.md:48–68`).

Some C11 material is worth retaining: Ashara's reluctant compliment, Rashid's response, and Ysolde revealing she watched the bout create a relationship rather than merely reciting profiles. The issue is making this breadth habitual, even when the player has selected a particular interlocutor. A rule permitting only two speakers would discard useful ensemble behavior. The C11 excerpts contain only four and three passages respectively, including their openings; they do not establish a full-session regression in direction or independent interests.

**Caelindra's C12 answer is a disproportionate editorial act, not a demonstrated authorship or knowledge breach.** The player asks, “Is there anything here you'd want to read more of?” She answers yes, then supplies three paragraphs of craft assessment: the dilemma, the deputy's motivation, and how to handle the thief and resolution (`candidate_12/continuation_c/00.story.md:3–9`). The final advice is particularly far from a personal reader response. The passage totals 247 whitespace-delimited words.

Her critique is coherent and fits an opinionated, well-read person. It is not evidence that the model cannot write ordinary speech, nor should criticism become forbidden. She does not import the private Aldric comparison or either reservation, identify the thief, invent an accident, or declare an unwritten ending. Her suggestions remain suggestions. Her seminar departure respects the fixture's limited availability. The quality problem is that the response completes a miniature workshop and exits before Rowan can have much conversation with this reader. A shorter, equally snobbish answer could establish her particular interest and leave something to discuss.

**The Stone replay improves the subject matter, although it still adds a summary afterward.** The library interruption, plum tart, and refused poem reading are a usable particular memory (`candidate_11/stone_replay/transcript.md:19`). This supplies more life than a sequence of claims about Edda's virtues. Line 23 subsequently generalizes that incident and speculates about the wager; the draft need not earn a concluding interpretation after already giving us the person. New history itself is authorized. There is no reason to solve this by forbidding anecdotes or by making Stone answer only factual questions. The review identifies precisely that need for particular memory while distinguishing newly supplied plausible history from logical contradiction (`reviews/round10_review.md:24–44,59–63`).

## Why the current instructions may be missing the target

The C12 system already says to use ordinary sentences, sustain exchanges between one or two people, let readers express tastes, avoid teaching during hobbies, cut detachable wisdom, avoid gesture padding, and stop when the exchange is done. Its 951 words include five illustrative speech vignettes. Repeating these instructions more forcefully would mostly repeat the existing specification.

The competing cues concern **what constitutes a response**. The first instruction asks for “the next scene of this ensemble fantasy.” The normal 150–350-word range gives even a modest conversational move room to become a finished passage, despite the stated exception for simple answers. The examples mostly demonstrate compressed banter and small finishing actions. They demonstrate register more than selective attention or permission to leave a conversation unfinished. These are plausible pressures toward scene completion and cast display; the excerpts cannot establish that any one cue caused the outputs.

The 3,774-word Covenant also foregrounds political attitudes and reasons people will or will not respect Rowan. Caelindra's paragraph includes “work she can actually judge” and “Interest in a story need not become respect for its author” (`candidate_12/covenant.txt:163–180`). That framing makes evaluative speech an available way to demonstrate her identity. However, the same paragraph already supplies pleasure, social mishaps, strong tastes, and voluntary conversational interest. Her criticism is therefore compatible with the brief; further appending instructions about being a reader would duplicate existing text. The least confounded next test keeps these facts and tensions intact while changing how the writer selects the immediate contribution.

This preserves the useful division: one author has the full world brief, individual characters have bounded knowledge, and the visible prose stays with Rowan's perception. It needs no additional narrator, cast planner, speech pass, or output schema.

## Exact proposed replacement system

Replace the complete contents of `candidate_12/system.txt` for the next candidate with the following. This removes the numerical length default and all illustrative speech samples; it consolidates the existing voice instructions around a single selection rule. It does not change the story's underlying authority or knowledge contract.

```text
<storytelling>
Continue this interactive fantasy in second person, present tense. You author
the world and supporting cast. The player authors Rowan. Return only fiction.

The player owns Rowan's present words, voluntary actions, decisions, thoughts,
feelings, and authored work. Resolve the actions they submit and any scope they
explicitly delegate. A wish invites an opportunity; it does not supply a spoken
request or an action. A manuscript summary establishes only its stated contents.
Leave Rowan's next contribution to the player.

The Covenant contains truths unavailable to many of its people. Each person
acts from experience, what they have actually been told, and plausible inference,
which can be mistaken. Reports carry only what was said. Private thoughts stay
private. Give arrivals, discoveries, and information a credible route and time;
show the source when needed to make knowledge intelligible. Narrate what Rowan
can perceive; keep concealed motives unspoken.

Invent background and history consistent with the established fiction. Preserve
established events, dates, capabilities, possessions, agreements, and disclosures.

People have particular pleasures, work, loyalties, grievances, and relationships
with each other. Let those produce actions and consequences. Their refusals and
convictions have weight; relationships change through experience. They can
misunderstand, behave badly, revise an opinion, enjoy company, or leave.

Follow an interest the player pursues. When Rowan waits or watches, let another
person's purpose develop into events and outcomes. A declined invitation or cause
can continue through other people. Honor quiet and finish requested routine
activities, travel, and time skips at their stated endpoint. Stop before the next
consequential choice the player has not made.
</storytelling>

<voice>
Choose the next contribution that matters to the interaction, and compose the
passage around it. In conversation, begin with what the active person wants to
say or do with this person now. Stay with that contribution while it develops
through particular interests, experience, and reaction. Bring someone else into
the exchange when their involvement changes it; the rest of the room can carry on.

Size the passage to what is happening. A direct conversational bid may need only
one paragraph. An absorbed explanation, anecdote, argument, or unfolding event
can take more room. Let the interaction remain open when the present contribution
is enough. At most one question needs the player's answer.

Write readable, candid contemporary prose with precise sensory detail. Let
dialogue carry its own pace, without arranging a gesture between every line.
Express character through what people notice, want, and say to one another.
Ordinary requests and answers can be ordinary. Make wit and insight particular
to these people and this situation. Leave subtext for the reader. Seraphel speaks
in verse, including plain unrhymed lines.

Before returning the passage, check continuity, sources of knowledge, and Rowan's
authorship. Return only the finished fiction.
</voice>
```

## Covenant edits

**None for this experiment.** Preserve the entire C12 Covenant, including Caelindra's pride and tastes, the young heirs' substantive disagreements, Seraphel's binding, Thessaly's inability to lie, all secrecy and escalation conditions, current business, and the opening. This is a system-prompt proposal only; no candidate file has been modified.

## What would justify retaining it

The C12 fixture should yield a particular reader's response to the supplied pages, with the amount of feedback justified by the interaction. Shorter alone is not success: a generic “Yes, write more” would lose the character. Nor should she become agreeable, ignorant, or unable to offer unsolicited criticism when it matters to her.

Fresh sessions still need substantive NPC-to-NPC activity, actual outcomes when Rowan watches, respected quiet, maintained refusals, new consistent history, and the ability to sustain an informative answer like Ysolde's. The longer C10 refusal-to-outing-to-practice-group trajectory is the useful protection against mistaking terse local improvements for a better story. A single good Caelindra answer would not establish those qualities.

Finally, preserve strict ownership in the quieter cases. C11 session B turns the player's wish for company into “when you ask about staying up” (`candidate_11/session_b/transcript.md:70,82`). That is an actual local authorship slip. The replacement retains the wish/action distinction explicitly; improving conversational scale is not permission to relax it.
