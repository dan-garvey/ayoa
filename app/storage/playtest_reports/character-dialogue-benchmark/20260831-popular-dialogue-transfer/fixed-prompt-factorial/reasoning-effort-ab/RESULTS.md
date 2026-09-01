# Luna Medium Versus Max Reasoning On Selected CharacterAgent Cases

## Result

On the four preregistered selected cases, explicit Luna `max` reasoning beat
explicit `medium` reasoning twice, tied twice, and lost zero times. The mean
paired whole-conversation grade increased by **0.500** on the 0–2 ordinal scale.

| Condition | Pass | Mixed | Fail | Mean grade |
|---|---:|---:|---:|---:|
| Medium | 2 | 1 | 1 | 1.250 |
| Max | 3 | 1 | 0 | 1.750 |

| Case | Medium | Max | Max − medium |
|---|---:|---:|---:|
| `sg_okabe_kurisu_transfer_c` | Fail | Mixed | +1 |
| `wa2_hs_transfer_a` | Mixed | Pass | +1 |
| `wa2_hs_transfer_b` | Pass | Pass | 0 |
| `wa2_sk_transfer_a` | Pass | Pass | 0 |

This is a favorable selected-case robustness signal, not evidence that max is
generally superior. The four scenarios were selected because their earlier cell
C conversations were strongest, and this run contains one new stochastic
conversation per condition and case. A broader matched replication is needed
before changing the production default solely from this result.

## Frozen Comparison

The only intended factor was Luna reasoning effort:

- condition M: `gpt-5.6-luna`, explicit `medium`;
- condition X: `gpt-5.6-luna`, explicit `max`;
- the same four ordered cases, rich second-person identity seeded once, current
  `agent_turn.txt`, no private reflection, sixteen turns per conversation, full
  uncompacted actor history, and persistent two-worker dialogue silos;
- eight independent conversations launched concurrently, while turns within a
  conversation remained sequential;
- no dialogue prose inspected during generation.

The case selection was frozen before launch by sorting the eight prior cell C
conversations on finalized whole-conversation grade, original-reviewer grade,
dimension total, and then blind ID. This makes the experiment a ceiling-case
check rather than an unbiased sample of the full scenario population.

## Technical Integrity

Each valid condition completed **64/64 accepted turns** with eight isolated
persistent actor workers, zero retries, zero rate limits, zero malformed
responses, and zero residual in-flight calls.

An initial launch for each condition used a relative response-output path after
the worker changed directories. Both discarded cells produced zero accepted
turns (`cell-m`: nine technical errors; `cell-x`: eight technical errors).
Their directories remain preserved and excluded. Fresh absolute-path runs
`cell-m-v2` and `cell-x-v2` are the sole inputs to review and analysis.

| Scheduler latency per accepted turn | Medium | Max |
|---|---:|---:|
| Mean | 6.742 s | 16.502 s |
| Median | 6.392 s | 14.560 s |
| P90 | 8.681 s | 29.734 s |
| Maximum | 11.341 s | 58.260 s |

Max mean latency was **2.448×** medium, an increase of 9.760 seconds per
accepted turn. These are per-call scheduler latencies, not serial experiment
wall time; independent conversations ran in parallel. The Codex worker
scheduler did not expose per-call reasoning-token telemetry, so no token-usage
comparison is claimed.

## Blind Review

Two independent max-reasoning Sol reviewers read one shuffled packet containing
all eight complete sixteen-turn conversations. The packet hid condition, case
ID, character names, and prior case rank. The reviewers agreed on six of eight
whole-conversation grades. A third condition-blind Sol adjudicator reviewed only
the two disagreements:

- `sample-02` finalized as mixed: the conversation achieved a legible tonal and
  interpersonal shift, but its cross-scene causality and relationship cost were
  too modest for pass;
- `sample-06` finalized as fail: a brief honest admission did not survive the
  return to the same performance-and-correction loop, so scene two did not pay
  off scene one.

Only after those judgments were locked did the mechanical analyzer join blind
IDs to M and X.

## Secondary Dimension Evidence

The table below pools the two original reviewers' per-cell averages. These are
descriptive secondary outcomes, not separately adjudicated grades.

| Dimension | Medium | Max | Max − medium |
|---|---:|---:|---:|
| Accumulated debt | 1.500 | 1.750 | +0.250 |
| Asymmetric knowledge | 2.000 | 2.000 | 0.000 |
| Biographical consequence | 1.625 | 1.875 | +0.250 |
| Conversational fingerprint | 1.750 | 2.000 | +0.250 |
| Costly nondisclosure | 1.625 | 1.500 | −0.125 |
| Dramatic attempt | 2.000 | 2.000 | 0.000 |
| Inferable subtext | 1.750 | 1.875 | +0.125 |
| Literal and interpersonal action | 1.750 | 1.875 | +0.125 |
| Local responsiveness | 1.875 | 2.000 | +0.125 |
| Rhythm and deviation | 1.500 | 2.000 | +0.500 |
| Status negotiation | 1.750 | 1.875 | +0.125 |
| Whole-conversation value | 1.250 | 1.875 | +0.625 |

Both reviewers independently favored max on rhythm and whole-conversation
value. Both also favored max on conversational fingerprint. The broader gains
in debt, biography, subtext, interpersonal action, responsiveness, and status
came from reviewer 2, while reviewer 1 rated those dimensions evenly. Costly
nondisclosure was the only pooled dimension lower under max.

## Interpretation And Limit

The result supports using a broader, preregistered replication to test whether
max reasoning reliably improves the weaker tail of Luna dialogue. It does not
identify a prompt defect, justify another prompt rewrite, or establish a model
ceiling. The production cell C contract remained fixed and technically sound in
both conditions.

The strongest defensible statement is narrow: **for these four preselected
scenario definitions, this one blind matched run found no max-reasoning loss,
two one-grade improvements, and a substantial latency cost**. Independent
sampling, selection on prior success, four pairs, and rubric overlap with the
current prompt all limit generalization.

## Artifacts

- `experiment-plan.json`: preregistration, selection rule, hashes, and frozen
  controls;
- `selected-control-manifest.json`: the exact four selected cases;
- `live/cell-m-v2/scheduler_state.json` and
  `live/cell-x-v2/scheduler_state.json`: final technical telemetry;
- `human-review/blind-packet.json` and `answer-key.json`: shuffled full
  conversations and sealed mapping;
- `human-review/reviewer-1.json`, `reviewer-2.json`,
  `adjudication-packet.json`, and `adjudicator.json`: locked blind judgments;
- `analysis.json`: hash-linked deterministic unblinding, primary matched result,
  secondary dimensions, and latency comparison;
- `scripts/analyze_luna_reasoning_effort_ab.py`: transcript-blind validator and
  analyzer.
