# Semantic router handoffs and source resolution

Implemented reviewed points 1–4. Observer pruning (point 5) was explicitly
excluded and remains unchanged.

## Runtime contract

- An active bound human is a valid narrative selection, never an agent-draft
  target. The first human in an ordered causal chain forces visible handoff.
  Ordinary selections remain advisory and do not create Cat-II obligations.
- Matching player actions and deferrals consume their selection. Claims retain
  the selection and change ownership. D&D follow-ups and resolved contributions
  use the same frontier lifecycle.
- The router output field `source_event_index` is retired, without a runtime
  compatibility shim. `causal_group` refers to an input/history group; `null`
  represents an independent character interaction. Merged input aliases resolve
  to their combined canonical result, while a no-event group may use its latest
  established history. Historical lifecycle facts cannot reactivate actors.
- Overlapping selections run in router list order. Blocked heads reserve their
  dependencies; disjoint heads may prepare concurrently. Serial followers rebase
  to their predecessors' committed results. Contested groups remain blocked.
- Scheduling sources order time without granting knowledge. Fact lookup requires
  observer membership; agent cutoffs reflect witnessed fact timing, while human
  cutoffs remain delivery-acknowledgement based.
- Frontier changes wake pending narration even without new events. Restart and
  the autonomous batch limit can flush existing facts without new fiction.
  Failed renders remain explicitly retryable. Merged narration preserves each
  POV's separate visible references.

The obsolete routing-role/background-thread description in DESIGN section 6.7
was replaced with this single frontier contract.

## Verification

The focused cross-path regression file passes all 29 tests. It includes the
actual first Covenant rejection, preserving its fiction and selections while
adapting only the retired source-coordinate field. The saved playtest session
itself was not changed or advanced.

The final runtime full suite passed 1,816 tests with 2 skipped, including pending
narration at the autonomous batch limit. The exact Covenant fixture test was
added after full-suite collection and verified with the complete 29-test focused
file. Changed-file Ruff and `git diff --check` pass.

Live harness: `app/storage/playtest_reports/router_prompt_targeted_20260912T203443Z.json`
and its Markdown/log companions. Model: `gpt-5.6-terra`, configured medium effort.
All 11 responses materialized without a batch-contract exception. Checks:
58/59 passed. These are targeted router calls, not an end-to-end Discord playtest.

Manual raw-output review:

- NPC-to-player pressure selected `dan` in group 0 without deciding his reply
  or opening a contest; Maya's unrelated work used a null group.
- Historical invitation returned a no-event resolution and selected Rashid in
  the existing invitation group, anchored to the later private preparation.
  No private fact was published. Offline tests separately verify that Rashid's
  draft context and observation cutoff exclude that preparation.
- Multi-addressed dialogue selected Ashara first. NPC-to-NPC pressure also
  selected Ashara, not the player. An additional overlapping Britney selection
  is now serialized rather than rejected as simultaneous work.
- Deferral invented no player behavior. The contested opening preserved only
  the attempt; the same-group speech/attempt case merged both inputs. The
  resolution closed once without repeating the opening.
- Arrival placed Britney in a pod and selected independent Dante activity.
  Offstage fan-in kept the dinner and security events in separate causal lanes.
- The remaining failure is the already-tracked `ayoa-izvs`: an audio packet
  available to Britney also carries `visual_subject_ids=["dan"]`. This is not
  fixed by the handoff/source change. Its raw evidence was added to that issue.

Compared manually with the harness's September 1 baseline, dialogue preservation,
NPC-directed pressure, attempted-versus-resolved contests, and bounded arrival
remain intact. The older schema/cases differ, so aggregate score deltas are not
comparable. The baseline audio case passed, but the intermittent defect was
already recorded in `ayoa-izvs` on August 28. Seven current calls show 3,400 cached
input tokens; there is no cache-layout performance claim from this run.

The preceding `20260912T203137Z` report is preserved too. Two of its cases errored
in the harness evaluator, not the router, because callers still used the old
source-index helper. Those callers were corrected before the complete rerun.

## Point 5: the quiet invitation

Exact original evidence:
`app/storage/playtest_reports/covenant_15turn_20260911t185207z/rejected_batch_03.json`.

The sole fact describes Rowan quietly inviting Caelindra for a walk. Its fields
say `audience="only"`, `visible_to=["caelindra_vaeyn"]`, and
`visual_subject_ids=["player_garvey"]`. Nevertheless, the observer list names
Rowan, Caelindra, and six other people at the table.

The validator computes eight observers minus one fact recipient and finds seven
observers with nothing to receive. This is the precise reason for rejection.
`visual_subject_ids` describes who is depicted, not who receives the fact.

Pruning would retain only Caelindra, including dropping Rowan. That avoids this
validation error but produces no player narrator job for Rowan's invitation.
Broadening the recipient list would instead publish private speech. Neither
repair can be assumed from the contradictory output. A coherent output could
give the invitation to Rowan and Caelindra, and include bystanders only if a
separate fact describes something they actually perceive. The existing strict
observer checks and `ayoa-3k2b` remain unchanged/open.
