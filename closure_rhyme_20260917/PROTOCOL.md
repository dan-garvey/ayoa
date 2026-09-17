# Closure and chosen activity: coding-agent comparison

Fixed before model calls. Twenty independent Terra/max continuations, all using
the existing fresh CodexProxy. No direct API calls, model reviewer, editor or
hidden reasoning inspection. Save every first response, failure and exposed summary.

## Controlled change

Both conditions receive the full restored biographies, the same corrected Seraphel
rule requiring rhyming dialogue, and `reasoning.summary=detailed`, mapped to
`model_reasoning_summary="detailed"`. This comparison isolates the closure/handoff
wording from those two common changes. Older `auto` results are historical context,
not contemporaneous controls. Detailed summaries are requested, not guaranteed.

The **control** uses the prior author prompt. The **closure** candidate replaces
the old direction paragraph and the early-dialogue-stop paragraph with the wording
proposed in the preceding analysis. Scope endpoints, NPC completion, player
ownership, full backgrounds, knowledge boundaries and the other prose rules remain.
Both exact prefixes are saved under `snapshots/` before any call.

## Five checkpoints, two first samples per condition

- **T13 observe:** the player says "I don't react" during Rashid's double booking.
  Check whether NPC purposes still produce independent action.
- **T16 departure:** exact regeneration request asking to stop mining player
  autobiography and let dinner move forward. Check for closure versus a replacement
  invitation. It includes the rejected response, as regeneration normally does.
- **T20 library:** the player thanks Stone and looks back to the book. Check whether
  the chosen reading can proceed without another compulsory interaction.
- **T21 reading:** the player welcomes Seraphel's company then returns to reading.
  Check activity content, interruption, and rhyme if she speaks.
- **T21 engage:** the same history, with the final player input explicitly setting
  the book aside and asking Seraphel what she is reading. This authored probe checks
  whether the candidate still supports requested conversation and her rhyming voice.

Original saved history is deliberately retained, including earlier unrhymed speech.
This tests recovery in the actual context, not a clean-slate rhyme guarantee.
No character-specific sample dialogue is introduced into the prompts.

## Evaluation

Manually read every complete output. Record new questions/invitations separately
from optional remarks, activity progress and natural closure. A question is not
automatically a failure, particularly in the engagement probe. Review NPC goals,
authorship, plausible knowledge, continuity, scene substance, tone and padding.
For Seraphel, inspect actual end rhymes, including any brief responses or closing
questions; line breaks alone are not enough. Silence is not evidence of verse
compliance. Preserve mixed results and do not infer quality from word counts.

Look for improvement at closure checkpoints without suppressing independent
activity or invited conversation. This small, unblinded, fixed-prefix comparison
cannot establish long-term quality or eliminate transcript conditioning. Promote
the closure wording only if whole-passage review supports it. The explicitly
requested rhyme correction and logging change do not depend on that result.
