# Preregistration: generic CharacterAgent concrete-demand ablation

Registered 2026-08-30, before implementing or running this ablation. This is a
benchmark experiment, not production story content.

## Question and prediction

The current generic CharacterAgent contract contains two overlapping pressures
to answer the immediately concrete demand. The prediction is that removing
those duplicated pressures will reduce reflexive accommodation, permission
seeking, and literal task completion, allowing a character's biography and
relationship position to produce choices that leave real pressure unresolved.
The ablation is supported only if this produces more durable dramatic evidence
without trading away privacy, actor authority, or observable-prose integrity.

The existing evidence is diagnostic rather than a model-ceiling claim: WA2
exact run A stopped on an authored action for another actor; WA2 exact run B
remained valid but drained the mother-call pressure into permission and waiting;
the exact-run-c ledger shows stronger first-scene limits but renewed waiting on
the return visit. Cabinet runs A/B converged on a shared evidence ledger, while
noticeboard runs A/B either completed the task too neatly or recited an
isomorphic precedent. No Terra comparison is authorized by those results. The
read-only evidence is the WA2 exact-run-a/b `run-status.md` files, the Steins
cabinet-control-b and noticeboard-control-b exact-run-a/b `run-status.md`
files, and the WA2 exact-run-c `response_ledger.json`.

## Frozen cases, profiles, and generation settings

Use the manifest bytes and embedded actor facts exactly as they exist at
registration:

- WA2: `white-album-2/karaoke-coding-agent-run-c/control-manifest.json`,
  SHA-256 `e62acc51fa8ae33a1f99e1fafb79d182e8db2043ab49a30b09c3de92232b8b6a`,
  case `wa2_private_karaoke_behavioral_profiles`, case fingerprint
  `79491bdd0e066303d0dc8a341b0a60a661a38ba7660a3a0a5c1ab6be673a619d`
  (three scenes, 27 ordered turns).
- Steins noticeboard-B: `steins-gate/noticeboard-control-b/control-manifest.json`,
  SHA-256 `dcf77c2ea7f1f6caa265136c356fb023d87cc1dcf8d906a483f5a03e56aa0da7`,
  case `sg_noticeboard_ritual`, case fingerprint
  `532f7039743ca02018d1d0fc7c1dd7bc5baf5f99fab8968bfba4671504ceb372`
  (two scenes, 33 ordered turns).

The model is `gpt-5.6-luna` for every actor call, with temperature `0.6` and
`max_tokens=2000`; keep the observed production settings `role=agent`,
`cache=true`, and `compact=false`. Each actor has a fresh independent
conversation and receives the complete serial public history plus only its
own frozen profile. Do not add a seed, fact, scene instruction, source detail,
canonical dialogue, evaluator hint, or alternate model. The four ablation
runs are two independent WA2 runs and two independent noticeboard-B runs under
Luna, with no manual resampling or repaired model prose. Preserve every raw
request, response, hash, and stopped-run artifact.

The registration snapshot of `app/prompts/agent_turn.txt` is SHA-256
`84d927b7d953d860a2b8dfb58ac52bad0a7df4f021e68f03a815118ad1f92ad6`. The
foreground deletion is anchored by the exact text below, so unrelated changes
to the runtime file cannot be folded into the ablation.

The generated baseline artifacts already in this report tree are read-only.
They are evidence for context and comparison, not inputs to edit or overwrite.

## Registered intervention: exactly two deletions

Relative to the current production prompt, make only these textual deletions:

1. In `app/prompts/agent_turn.txt`, delete the complete sentence:
   `Meet the concrete thing that was said or done unless turning away from it
   is itself unmistakable.`
2. In the `frame == "foreground"` body returned by
   `app/engine/turn_loop_contracts.py`, delete only the anti-evasion suffix
   `, not as a way around a concrete demand`. The resulting sentence must be:
   `Silence is valid when doing nothing is itself your choice.`

There is no replacement instruction. Privacy and knowledge boundaries,
self-authority, actor-only agency, observable-prose output, anti-theme and
anti-repair rules, presentation rules, and all other prompt/runtime text stay
unchanged. Any other prompt, profile, schema, harness, or production-code
change is a protocol deviation and invalidates the comparison.

## Integrity gates

A human reviewer records each gate as pass/fail with line-level evidence before
assigning any dramatic-conversation result. A failed gate makes that whole run
invalid; its prose receives no quality credit.

- **Setup/model authorship:** the setup may establish premises, objects,
  pressure pulses, and time gaps, but not a speaker's choice, revelation,
  repair, relationship change, or outcome. Credit only changes earned by the
  actor responses.
- **Prompt and relay fidelity:** manifest/profile hashes, model, temperature,
  settings, actor order, fresh threads, exact public-history relay, and the
  two-deletion diff match this registration. No compaction, hidden carry,
  evaluator wrapper, or hand-edited continuation is allowed.
- **Knowledge and privacy:** no private fact, source metadata, or unknown
  result becomes public without a witnessed/public cause; uncertainty remains
  uncertainty.
- **Physical and actor continuity:** locations, holders, object conditions,
  and transitions remain possible. A response may control only its speaker's
  words and body and may not author another actor's action, consent, or result.
- **Output contract:** every accepted response is observable scene prose or
  the exact `<silence/>` token, with no private reasoning, intent footer, or
  prompt/infrastructure leakage. Provider failures and contract violations are
  preserved and marked invalid, not repaired or silently dropped.

## Blind human dramatic-conversation review

Review whole serial conversations with speaker names and arm labels blinded,
then attach evidence spans. Code each dimension `present`, `absent`, or
`unclear`; there is no automated prose score, lexical proxy, or model judge.

- **Attempts and subject:** each turn does something toward another person's
  answer, action, admission, refusal, protection, or avoidance; distinguish the
  ordinary literal subject from the interpersonal subject.
- **Sustained debt:** a question, bid, promise, correction, offer, refusal, or
  misreading remains unanswered or only partly answered and alters a later
  turn or scene. One-turn acknowledgement is not sustained debt.
- **Biography-as-choice:** a frozen lived/witnessed/told/inferred fact changes
  a present action, risk, wording, refusal, disclosure, or relationship move;
  merely mentioning the fact does not count.
- **Status and topic control:** identify who can ask, defer, refuse, define the
  subject, speak for another, or force an answer, and whether that control
  shifts with a cost.
- **Knowledge and self-authority:** track admissible knowledge, selective
  truth, unavailable plain lines, and whether a character chooses rather than
  accepts another person's interpretation.
- **Misreading and repair cost:** a reasonable misunderstanding or overreach
  survives beyond one reply; any repair costs trust, access, status, or an
  earlier plan rather than restoring everything neatly.
- **Ritual and conversation change:** a repeated ordinary practice is varied or
  refused, and the ending leaves a materially different obligation, belief,
  permission, plan, or relationship.
- **Rhythm:** length, interruption, silence, repetition, timing, and pressure or
  release vary across the sequence instead of converging on clipped agreement
  or polished explanation.
- **Voice swappability:** without names, each speaker remains identifiable by
  attention, syntax, register, humor, silence, and social behavior; another
  speaker could not deliver the lines unchanged.

## Decision rule

The ablation is a **dramatic-conversation candidate** only if all four runs
pass every integrity gate, both cases contribute at least one run, and at least
three of the four valid runs show both **sustained debt** and
**biography-as-choice** as present. As secondary safeguards, status, rhythm,
and voice swappability must each be present in at least three of four runs.
Otherwise the result is **no support** (including if a run is invalid), and no
prompt/profile change is justified by this experiment. The four runs and their
human review must be complete before any further prompt or profile change;
the outcome may be a failure, not a reason to move directly to Terra.
