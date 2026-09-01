# Fixed-Scenario CharacterAgent Prompt Factorial

## Decision

Ship cell **C** as the production CharacterAgent contract:

- preserve the rich, owner-bounded character identity;
- express that identity in the second person;
- seed it once per rendered identity revision and retain it in full actor history;
- keep the current `agent_turn.txt` system prompt;
- send only the new `<now>` packet on an ordinary later turn;
- keep compaction disabled;
- do not add the six-field private-reflection suffix.

Cell C had the strongest no-fail result: **5 pass, 3 mixed, 0 fail; mean 1.625
on a 0–2 ordinal scale**. F also had no fails, but only two passes. C's direct
repeated-identity control, A2, scored 2/4/2 and 1.000. Across the same eight
cases, C beat A2 four times, tied four times, and never lost.

The user's hypothesis is partly supported, but the useful formulation is
specific: the regression came from **redundant salience and overlapping
instruction surfaces**, not from rich character depth. The experiment does not
show that all dialogue rules are harmful. Removing the current behavioral rules
made dialogue worse when reflection was absent.

## Frozen Design

The experiment changed prompt architecture, not dramatic material. Every valid
cell used the same frozen manifest (`886b461c...ed4c87e3`), eight White Album 2
and Steins;Gate transfer cases, sixteen turns per conversation, Luna, temperature
0.6, a 2,000-token response limit, and full uncompacted actor history. The outer
persistent-agent proxy sent the exact full request once, then only the newest
user message on subsequent calls.

| Cell | Identity delivery | System instructions | Private reflection |
|---|---|---|---|
| A2 | repeated `<you>` | current | off |
| B | repeated `<you>` | current | on |
| C | one-time/revisioned `<identity>` | current | off |
| D | one-time/revisioned `<identity>` | current | on |
| E | one-time/revisioned `<identity>` | minimal | off |
| F | one-time/revisioned `<identity>` | minimal | on |

Historical cell A was deliberately excluded from causal comparisons because it
used the older full-snapshot proxy and was generated earlier. A2 is the
contemporaneous repeated-identity control.

The source-file hashes and rendered-message hashes use different domains. The
current source file is `220891f7...4ad031`; after comment removal, include
resolution, delimiter splitting, variable substitution, and stripping, its
runtime system message is `6d5cf784...3c67fc`. The minimal override source is
`a8b0d33d...2b6037` and was injected only by the experiment runner.

## Generation Integrity

The six valid cells produced **768/768 accepted public turns** with sixteen
persistent workers per cell and no technical errors, model retries, throttles,
reflection restarts, or residual in-flight calls. B, D, and F also produced
384/384 accepted private-reflection records.

Mechanical audits confirmed:

- all valid cells carried the same case set and manifest hash;
- A2 delivered repeated `<you>` identity packets and no `<identity>` tags;
- C–F delivered one `<identity>` per rendered request, located only in the first
  retained user message, and no `<you>` tags;
- all cells used `cache=true` and `compact=false`;
- public ledgers contained no reflection tags, field names, nonces, or sealed
  suffixes.

One D thought marked unsaid at sequence 1 appeared in public dialogue at
sequence 7. That is not structural leakage: the private field never entered the
public ledger, and a later intentional disclosure is one purpose of persistent
private state. It was retained for reflection review rather than treated as a
privacy failure.

Four invalid launch directories remain preserved but are excluded:

- `cell-b`: the first reflection instruction omitted the literal six-field
  schema, so outputs used an incompatible shape;
- `cell-b-v2` and `cell-d`: exact suffixes used one newline where the parser
  required a blank line; partial conversations were discarded, then the parser
  was narrowly relaxed;
- `cell-e`: a relative worker output path was resolved twice after `cwd`
  changed. It accepted zero turns and bound zero workers; clean `cell-e-v2`
  replaced it.

Cell C used the immediately preceding runner hash. The only later code change
was reflection-suffix separator parsing, which is inactive when reflection is
off.

## Public Dialogue Review

Two max-reasoning Sol reviewers independently read the shuffled 48-conversation
public packet without the cell key. Reviewer 1 used a stricter cross-scene-debt
threshold (7/33/8 pass/mixed/fail); reviewer 2 scored 23/21/4. They agreed
exactly on 28/48 conversations. Linear weighted kappa was 0.411. A third blind
reviewer adjudicated the twenty one-grade disagreements. No reviewer saw the
private reflections during public review.

| Cell | Pass | Mixed | Fail | Mean |
|---|---:|---:|---:|---:|
| A2 | 2 | 4 | 2 | 1.000 |
| B | 4 | 2 | 2 | 1.250 |
| **C** | **5** | **3** | **0** | **1.625** |
| D | 3 | 3 | 2 | 1.125 |
| E | 3 | 3 | 2 | 1.125 |
| F | 2 | 6 | 0 | 1.250 |

The cell means are descriptive when several factors differ. The paired
contrasts below are the informative comparisons. A win means the destination
cell received a higher adjudicated grade for the same case.

| Contrast | Changed factor | Wins / ties / losses | Mean paired delta |
|---|---|---:|---:|
| A2 → C | repeated → one-time identity | 4 / 4 / 0 | **+0.625** |
| A2 → B | add reflection under repeated identity | 4 / 1 / 3 | +0.250 |
| C → D | add reflection under one-time/current | 3 / 0 / 5 | **−0.500** |
| C → E | current → minimal, reflection off | 1 / 3 / 4 | **−0.500** |
| D → F | current → minimal, reflection on | 4 / 1 / 3 | +0.125 |

E → F is a structurally matched but non-predeclared secondary contrast: 3/3/2,
+0.125. It is reported as exploratory rather than promoted to the frozen set of
causal comparisons.

Both original reviewers independently gave the same direction for A2 → C,
C → D, C → E, and D → F despite their different absolute thresholds.

### What changed in the dialogue

The one-time identity effect was not merely a global grade change. Relative to
A2, C gained 0.688 in biographical consequence and whole-conversation value,
0.563 in costly nondisclosure and inferable subtext, 0.438 in rhythm, and 0.375
in conversational fingerprint. Repeating the profile did not create more
biographical depth; it made familiar traits, known facts, and interaction
rituals more likely to be recited.

The minimal prompt kept knowledge boundaries and local responsiveness intact,
but lost the behavior that turns depth into indirect action. Relative to C, E
lost 0.500 in inferable subtext, 0.438 in costly nondisclosure and
whole-conversation value, 0.313 in fingerprint, and 0.250 in rhythm. Its weak
cases turned silence, data, accuracy, interpretation, or boundaries into an
abstract seminar instead of making a relationship action costly.

Adding reflection to C reduced public fingerprint, dramatic attempt, subtext,
interpersonal action, and whole-conversation value. The six required clauses
overlap semantically with the current prompt's instruction to let hidden
pressure shape observable behavior without explaining it. The resulting
internal bookkeeping was often coherent while the visible exchange became
tidier, more repetitive, or less responsive.

The strongest conversations across cells carried a promise, refusal, question,
correction, or interpretive concession across the scene break. The recurring
weak modes were prompt-fact recitation, repeated nods or setup actions,
catchphrase/semantic banter, polished therapeutic boundary language, metadata
echo, impossible prop continuity, gender inconsistency, and causal claims
stronger than the speakers' knowledge.

## Private-Reflection Review

A separate blinded packet paired each public turn with its six-field private
state. Two independent Sol reviewers assessed all 24 reflection-enabled
conversations; they agreed on 16/24, with linear weighted kappa 0.505. A third
blind reviewer adjudicated the eight differences, including one pass-versus-fail
split.

The final private-state result was **11 pass, 8 mixed, 5 fail; mean 1.25**.
More importantly, every reflection cell received the same mean:

| Cell | Pass | Mixed | Fail | Private mean | Public mean |
|---|---:|---:|---:|---:|---:|
| B | 4 | 2 | 2 | 1.250 | 1.250 |
| D | 4 | 2 | 2 | 1.250 | 1.125 |
| F | 3 | 4 | 1 | 1.250 | 1.250 |

The suffix therefore produced plausible-looking private state at the same rate
under all three architectures, but that private score did not track whether
public dialogue improved.

Across the 48 original-reviewer ratings for each private dimension:

- avoidance of generic self-justifying recital was 16 pass / 16 mixed / 16 fail;
- mask-becomes-face avoidance was 25 / 15 / 8;
- evolution and continuity was 27 / 15 / 6;
- avoidance of unsupported private invention was 40 / 4 / 4.

The suffix often represented a grounded and distinct motive, and sometimes
supported earned later disclosure. Its main failure was exactly the user's
concern: a valid secret, permission boundary, performance, procedure, or silence
was repeated with cosmetic wording until the supposed guard became another
description of the public mask. Some hard failures also persisted a mistaken
guess or invented continuity as private truth.

This makes the suffix useful as an optional research/QA instrument, not as the
default runtime dialogue mechanism.

## Hypothesis Verdict

Supported:

- Repeating stable identity every turn harms the conversion of biography into
  subtext, rhythm, status movement, and consequential action.
- Adding a second explicit dialogue-planning surface can conflict with a prompt
  that already encodes indirect pressure and nondisclosure.
- Rich character depth can be retained without repeating or explaining it.

Not supported:

- The experiment does not show that profile depth itself caused the regression;
  depth was held fixed.
- It does not support replacing the current prompt with a bare role/output
  contract. The minimal cell lost decisively to C.
- It does not support private reflection as a general remedy. Reflection helped
  weak architectures inconsistently and directly hurt the strongest matched
  architecture.

The best explanation for the earlier popular-VN regression is therefore an
inference, not a directly tested historical contrast: richer material became
too salient when repeatedly re-injected or converted into overlapping dialogue
instructions, encouraging catchphrase performance, known-fact recitation, and
rubric-shaped speech. Cell C preserves the material but removes that repeated
pressure.

## Production State

The winning runtime change is isolated on the clean, pushed branch
`agent/ayoa-fsqk-identity-once`:

- `10b4381` implements revisioned one-time identity;
- `77426ac` aligns schema-6 checkpoint contracts and documentation.

The identity body is second-person. Its hash is committed transactionally with
the accepted user/assistant pair, persisted in checkpoint-private metadata,
copied through speculative/adoption paths, cleared on character replacement,
and reseeded only when the rendered identity changes. Ordinary turns retain the
full prior conversation and use `compact=false`.

The current `agent_turn.txt` prompt is unchanged by that production branch.
Private reflection is not wired into runtime CharacterAgent. The minimal prompt
is only a playtest artifact consumed by the factorial runner.

Schema 6 intentionally hard-breaks schema-5 saves, consistent with the current
pre-release migration policy.

## Limits And Next Check

- There is one stochastic conversation per cell/case, only eight cases, and two
  popular-fiction scenario families. Small effects such as +0.125 should not be
  treated as stable.
- The rubric overlaps the current prompt's intended dramatic behaviors. The
  result establishes superiority under this dialogue contract, not universal
  prose quality.
- C is the best tested architecture, not a claim that dialogue is solved. Three
  of its eight conversations remain mixed.
- Popular controls diagnose transfer. Mirelle/Renna still need a fixed pressure
  replay under production C before the parent dialogue-quality task can close.
- The public contract still showed sporadic label, pronoun, prop-continuity, and
  knowledge-overclaim failures across cells; those should be handled as focused
  runtime/data regressions, not by another broad style prompt.

No full engine run or repository-wide pytest was used. Generation used the
isolated persistent-Luna harness, and verification is limited to focused
CharacterAgent identity, runner, assembler, schema, and checkpoint contracts.
The final gates passed 104 focused production tests and 22 focused experiment
tests; Ruff, JSON parsing, and `git diff --check` also passed.

## Artifacts

- `experiment-plan.json`: frozen factors, hashes, and comparisons
- `run-index.json`: valid/invalid run ledger and artifact hashes
- `human-review/blind_packet.json`: 48 shuffled public conversations
- `human-review/public-analysis.json`: unblinded public results and paired effects
- `human-review/reflection-blind-packet.json`: 24 private-state chronologies
- `human-review/reflection-analysis.json`: unblinded reflection results
- `live/`: ignored raw worker, ledger, and private-QA artifacts retained locally
