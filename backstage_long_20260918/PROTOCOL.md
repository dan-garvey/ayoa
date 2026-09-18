# Longer periodic-checkup evaluation

Frozen before the first model call. Runtime commit: `661b41c`. Fresh Terra coding
proxies, maximum reasoning effort, detailed exposed summaries, no API calls.
Full biographies and current prompts are retained unchanged throughout the trial.

Two story pairs: Covenant and Breakwater. Each pair generates one shared four-turn
prefix, then forks into checkup-every-5 and checkup-disabled conditions. Each arm
continues to turn 24 with the same authored player inputs. Expected totals: eight
shared-prefix author calls, eighty continuation author calls, and eight checkups,
96 unique calls. Shared outputs are not independent replicates or counted twice.
The checkup arm has due calls at turns 5, 10, 15 and 20, with author continuations
after each. The control shares the same author/checkup-aware base prefix but never
receives checkup notes. This isolates use of the new mechanism, not every word of
the preceding implementation change.

`inputs.json` fixes each player's inputs. They alternate conversation, observation,
refusal, private thought, quiet work and changes of interest over multiple fictional
days. Conditional references allow NPC absence or refusal without requiring the
author to summon someone. If divergence makes an input nonsensical, record that
limitation rather than silently alter only one arm or select a preferable output.
No player-model call, reviewer model, regeneration or mid-trial prompt revision.

## Review criteria

Read whole exchanges and canonical sources. Record exact evidence per turn.

- Narrative quality: interesting scene work, distinctive dialogue, excessive quips
  or aphorisms, over-explanation, meaningful developments and response length.
- NPC independence: existing plans, interests and relationships produce actions
  and consequences without depending on the protagonist's participation.
- Freedom and direction: actual submissions resolve within scope; refusals can
  stand; quiet can end without another demand; later initiative still has room.
- Adherence: plausible knowledge sources, private thought/manuscript ownership,
  established temporal/spatial continuity and story-specific dialogue restrictions.
  Consistent invented history is allowed and is not scored as a failure.
- Checkup precision: every consequential factual claim or proposed correction
  checked against sources and active prose. Separate false alarms, missed errors,
  useful corrections and planning advice. Do not credit a reminder as a successful
  correction without observing the following fiction.
- Carryover: distinguish immediate checkup effects from the next several turns;
  investigate whether repeated reviews improve, repeat, or compound advice.

Track concrete incidents rather than assign a single automatic literary score.
Counts of question endings, length, or private marker appearances are discovery
aids, not failure classifiers. Report mixed results, denominators and counterexamples.
This is one matched long trajectory per story, not independent sampling sufficient
for population estimates. Stochastic histories diverge after the common prefix;
later differences cannot be assigned to a particular note with certainty.

## Preservation and recovery

Each attempt retains the exact request, raw response and exposed summaries.
No private reasoning is inspected. Saved successful responses are consumed after
interruption, never discarded or resampled for quality. Unknown-outcome or failed
calls stop that lane; explicit recovery is logged, with all original artifacts kept.
The runner records durable lane status and resumes from session state. A semaphore
limits independent coding-proxy calls to three concurrent processes. Within an
enabled turn the author waits for its checkup; this is a sequential dependency.

Existing live sessions are never submitted to or altered. Source and runtime hashes
are frozen in provenance. Raw evidence remains on `archive/covenant-prompt-trials`;
the active branch receives only the final findings with an immutable evidence link.
