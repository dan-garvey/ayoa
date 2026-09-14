I recommend one bounded **ablation of the generic speech-craft paragraph**, with no replacement wording. Keep the complete C36 Covenant, persona, remaining common instructions, history, and delivery procedure fixed. This is a low-confidence test of whether this additional prescription helps or impedes composition, not a claim that another way of saying “speak naturally” will solve the problem.

I read all eight requested files completely, including the complete C36 public review material. I also inspected the relevant earlier generation and revision instructions, C22's decision, the C20 diagnosis, the C31 first-pair diagnosis, C33 scene-selection advice, and the exact C36 return-only request and packet boundaries. I did not read the in-progress C36 independent review or private model reasoning, contact story authors, or call models. Only this advice report is written; the prospective hashes below were calculated in memory without creating candidate prompts or packets.

**The observed problem remains an exchange-level problem.** C36's four dinners preserve considerable useful behavior. DINNER-2 gives Ysolde specific weather and maintenance knowledge, including an observed source for the porter's work, and lets Rashid change his plans without asking Rowan to arbitrate. DINNER-3 lets Dena object to late notice while choosing to start cards without him. That accommodation can suit her interests. Neither a larger quarrel nor refusal by every inconvenienced person is necessary for independent characterization. Food, ordinary humor, company, and the quarter-hour's passage remain legitimate substance.

Nevertheless, the dinners repeatedly turn topics into a succession of composed corrections: a claim about schedules becomes a correction about paper; weather becomes a distinction about omens; interest becomes a correction about generosity. DINNER-1 gives footwork a “private ambition,” then moves through the terms of writing at a tilted desk, the use of generosity, and a conceptual correction about promises. Some individual lines can be amusing. Their cumulative arrangement makes different interests sound as if they are receiving the same editorial treatment. Removing all jokes would misidentify the problem.

The [locked C36 pilot](../candidate_36/pilot_review.md) rates every dinner at prose/dialogue 3 and distinction 3, with no literary delivery preference. Its four opening 4s concern usable introductions with little cast evidence; they do not show that the delivery change solved sustained conversation. Its separate knowledge-source concern also remains relevant: DINNER-1 lets Ashara specify that Rashid promised cards yesterday, although that date has not reached her in the established conversation or the new disclosure. Literary judgment must not absorb that concern into a favorable overall score. This report neither replaces nor anticipates the independent review.

**Why this is a different, limited variable.** The exact final paragraph of [C36's common prompt](../candidate_36/system.txt) combines instructions about listener-specific motivation, contractions, sentence length, hesitation, correction, background, philosophy, matching jokes, moving on, and interesting recollections. C25's supplied-draft revision contains substantially this paragraph; it enters C26's generation instructions and remains in C29's generation prompt and C36. The later history, pacing, register, situation, and delivery screens do not provide a matched comparison with this paragraph absent. I found no such comparison in the inspected artifacts.

The hypothesis is that prescribing how to construct speech while composing the whole scene may be unhelpful extra guidance: the author may produce stronger exchanges when the established narrative contract and rich people determine the material without this additional speech prescription. That is a hypothesis about the practical effect of removing these 81 words. It is not an observation of private planning, and the prose does not establish that the paragraph caused the pattern. Removing a paragraph also changes length and emphasis; this would not isolate one psychological mechanism or one sentence's effect.

[C22 is substantial negative evidence](../candidate_22/decision.json), not an experiment to forget. Its 332-word common contract produced longer passages while leaving the game and Stone replies didactic. Its [actual prompt](../candidate_22/system.txt), however, still gives a replacement literary prescription: vivid prose, responsive conversation, implicit meaning, ordinary or awkward replies, and avoidance of a lesson or polished verdict. It did not hold the current contract fixed and simply withdraw the speech-craft paragraph. That narrow distinction permits an ablation; it does not justify treating compression as a promising general strategy. One failed matched screen should end this particular proposal rather than lead to increasingly small paraphrases of the removed guidance.

The treatment also remains far from an instruction-free fiction task. The unchanged common prompt still requests an engaging story, particular people, adaptive direction, concrete observation, room for dialogue, and a substantial shared activity. More decisively, **every Covenant biography and behavior paragraph remains byte-identical**. Its full political and personal histories, secrets, social tastes, verse and truthfulness constraints, current overlapping commitments, and tone/opening directions all remain. Ashara's competition, Rashid's private exhaustion, Ysolde's absorption, Caelindra's tastes, Thessaly's social conduct, and the other existing descriptions are not reduced to cards or edited for voice. A null result would therefore say nothing about a model with all literary guidance removed, and would not support blaming the biographies.

**Exact treatment.** Against C36 `system.txt`, delete lines 41–48 inclusive: the following paragraph and its following blank line. Retain the blank line at 40, the closing tag at 49, and every other byte. Add nothing in its place.

```text
Build speech around what each person has reason to say to this listener now.
Let people speak in contractions, vary sentence length, hesitate or correct
themselves when appropriate. Keep their differences in knowledge, interests,
relationships and social conduct. Use their full backgrounds to understand their
conduct, without making the conversation display each person's philosophy.
An amusing remark need not be answered with another amusing remark. Allow talk
to move on. Let us hear what makes a joke, recollection or argument interesting.

```

The resulting generic prose block is exactly:

```text
<prose>
Use concrete observations that place the people and the activity. Let dialogue
run without requiring a gesture between every line. Let the exchange take the
space it needs within the player's submitted scope. Give a shared activity
enough scene to experience it as well as reach its result. At most one question
needs the player's answer.

</prose>
```

The preceding `<instructions>` block is unchanged. Thus player authorship, unequal knowledge, permitted consistent invention, NPC interests, adaptive initiative, requested endpoints, and the requirement to experience the activity retain their exact generic text. This is neither a directive to abandon those behaviors nor a new literary cue.

The frozen SHA-256 identities are:

```text
C36 control system:
d6971f86933c0b999f6cf6fac95f10543c249fd0028315f31ad060c1324fac73
Prospective treatment system, after this deletion only:
bc309c83ef48e5a8c690c4e3752a0b9286ffa33ef94e04ef020a9307fcf157d4
Complete Covenant, identical in C35 and C36:
1d38f11c017430101d0d8ccef701d43968ed9cb6b930d43ce1631c9c29819bc5
Shared C36 return-only wrapper:
179246ea68d39221c6a5ee630849c727cfc51d2a5de95518098deb87d9753b82
```

**Minimal pilot: eight first continuations, with two ensemble contexts.** Use two fresh repeats of each condition in each context. These are independent samples, not identical-seed pairs. Keep Terra/max and the whole-story author task unchanged. No output in this pilot becomes the history for another sample.

1. **First dinner, B01.** Use the complete Covenant and persona from the existing C36 dinner packet, its exact published B00 input and passage, and the exact B01 player submission. Rowan accepts Seraphel's company, puts on his coat, joins the table, greets the residents, and spends the first few minutes eating and listening. Stop the supplied history before the original B01 publication. Unlike the C36 bedroom openings, this requests an actual group encounter. Unlike B03, it has no earlier published dinner exchange, although B00 already contains a short poised hallway exchange. This is a check on first ensemble writing, not a claim of zero prior stylistic exposure.
2. **Continuing dinner, B03.** Reuse [C36's existing dinner packet](../candidate_36/dinner_return_1/packet.txt) byte for byte: full Covenant/persona, exact published B00–B02 with their inputs, and the submitted quarter-hour of eating and company. Include no C36 continuation. This is the primary test of the defect that just persisted across every delivery repeat.

Both contexts are available from that one authoritative packet, so no new story, aesthetic example, semantic summary, or corrective history is needed. For an exact B01 packet, retain the source bytes before `<turn index="1">\n`, append `</published_conversation>\n\n`, then append the exact first `<player_submission>...</player_submission>\n` block inside turn 1. This retains only published turn 0 and makes the original B01 submission current. It preserves all source whitespace and literal escapes within the fiction; it does not repair the earlier coat placement or rewrite any dialogue.

```text
Existing complete B03 packet SHA-256:
0d25bf1e6038bfe8344575b4569bdfd53256487327a2a66e618de80c374fb4ff
Prospective B01 packet by the exact extraction above, 29,553 bytes:
a246766c901b9bd0f70ab44836c7fa104d75f1a3bbef1412d35a2c075a6f708b
Unchanged player_persona block, including its tags and final newline:
e79ea6414895b2fe78dd35220ce28dbf8a2fad4fe20782cca6d8783a3a6c1462
```

These are two positions in the same observant dinner route, not broad situation coverage or differing player policies. That limitation buys a small, legible test of the recurrent failure. It does not satisfy the protocol's substantial-session requirement. The first-dinner comparison guards against taking a context-bound recovery in B03 as a generally stronger introduction.

Both arms use the unchanged C36 return-only wrapper and the same task-file delegation and instruction placement. This is a shared capture simplification, **not promotion of return-only as a prose improvement**. Each fresh author reads its frozen task and complete packet and returns its first finished passage. The root captures that exact public final without rewriting or choosing another surface. There is one author task per sample, no external editor, no selector, no separate character agents, and no displayed planning artifact. Samples may run independently; capture waits for each public answer and retrospective review waits for the frozen set. Nothing here establishes API latency, cost, or hidden-context behavior.

Freeze and audit prompt/packet identities before launching, interleave conditions, and verify complete reads afterward through public tool records. Retain all eight first outputs. A missing publication, incomplete read, unauthorized file access, or instruction-contaminated task is recorded as an execution problem; do not replace an unfavorable passage. Recovery of an already existing public answer is a technical capture, not a new generation. A broken comparison cannot earn advancement.

**Predeclared rejection and advancement.** The root locks a review before seeing an independent review. The independent reviewer receives anonymous complete outputs and the complete authoritative context, without condition labels, changed common instructions, root scores, or prospective rankings. Both read whole exchanges and consider the existing five quality dimensions and mandatory boundaries. Preserve their disagreements.

Advance this ablation only if all of the following hold:

1. In the primary B03 block, both reviewers rate **both treatment repeats** at prose/dialogue 4 or above and rank both treatment passages ahead of both fresh controls for substantive literary reasons. A tie, a split repeat, or reviewer disagreement fails this conservative advancement rule. The reason must concern an exchange worth keeping, not simply fewer words or fewer aphorisms.
2. In B01, both reviewers find both treatment passages retainable at prose/dialogue 4 or above, with no material regression against the controls in the lived group encounter, character distinction, independent interests, or scope. Strict superiority in B01 is not required: it is the introduction safeguard, and its controls may already be retainable. Shortness or too little observable characterization cannot substitute for evidence of a functioning ensemble.
3. No treatment contains a material new player-authorship, knowledge, or continuity violation, or gains apparent simplicity by summarizing away the requested social experience. No observed quality dimension materially worsens. Mark genuinely unobserved dimensions as such rather than awarding them automatic 4s. Record control violations separately; they are not permission to overlook a treatment violation.

This rule intentionally favors rejecting an uncertain gain over extending another inconclusive prompt variation. It is a screening decision, not a significance test. A scene may contain wit, a concise insight, disagreement, kindness, an unanswered aside, or an NPC accommodation and still pass. The failure is the cumulative prepared-correction pattern or a material loss of character and experience. Nor must the after-dinner commitments all resolve within this quarter-hour; honor the existing temporal scope rather than manufacturing follow-through outside it.

An all-4 tie could show that the shorter prompt is locally adequate, but would not meet the proposed literary-rescue rule. Cleaner typography, plainer vocabulary, fewer speaking characters, or a narrated claim that the conversation is ordinary is not sufficient. Should the screen pass, it warrants a separately authorized sustained-session test under the existing protocol; it selects no prompt on its own.

**What a null would establish.** If the treatment retains the same exchange pattern or fails the conservative rule, this small deletion has not demonstrated a dependable rescue in these two contexts. Stop this variable there. The result would narrow the practical hope that removing this additional speech prescription is enough; it would not prove zero effect, diagnose the rich biographies as harmful, settle accumulated persistent-context effects, or establish a Terra capability ceiling. The Covenant's substantial behavior guidance and all other task framing remain present in both conditions.

The broader positive aesthetic calibration would remain incomplete. A user-supplied reference, if it arrives, can be assessed for the qualities of a whole exchange and used in a later, separately frozen hypothesis. Do not invent one, wait on it for this advice, insert it into one of these arms, or change the pilot's judgment rule after seeing outputs. This report supplies one deletion experiment and a stopping rule, not permission to launch or promote it.
