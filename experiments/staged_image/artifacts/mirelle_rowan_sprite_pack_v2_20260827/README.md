# Mirelle Voss and Rowan Kest pose-expression review pack v2

This is an experimental, unlocked authoring pack. It is not bound to the image
director, runtime, checkpoints, or reviewed-reference registry. No runtime LLM
reads any image in this workflow. Human promotion is still required before any
sprite becomes a production asset.

The pack contains coordinated face-and-body variants for `neutral`, `happy`,
`concerned`, `tense`, `skeptical`, `angry`, `sad`, and `surprised`. The v2 work
keeps Rowan's seven accepted v1 variants, replaces his `sad` variant, corrects
Mirelle's `concerned` and `tense` semantics, and normalizes Mirelle's canonical
double-ended spear across all eight poses.

## Review first

- `contact_sheets/mirelle_rowan_v2_complete_overview.png`
- `contact_sheets/mirelle_voss_complete_sweep_v2.png`
- `contact_sheets/rowan_kest_complete_sweep_v2.png`
- `manual_review.md`
- `prompt_conflict_audit.md`
- `rejection_log.md`
- `REVIEW_INDEX.md`

Every character sheet alternates dark and light backgrounds so matte color,
transparency edges, framing, baseline, identity, pose, expression, and prop
continuity can be judged together. Final experimental sprites are genuine RGBA
files under `sprites/<character>/<label>.png`, all normalized to 1100 x 1500.

## What changed in v2

Mirelle's locked active profile establishes one continuous, double-ended spear.
The primary end has a long central silver blade and two lateral guards; the
opposite end is a small pointed cap with a short tassel. The old pack's prompt
text incorrectly described a single-headed spear. V2 preserves that conflict as
evidence rather than rewriting history.

Repeated prompt-only and broad masked edits did not normalize the weapon. The
successful controlled path was:

1. freeze and hash the locked-reference primary assembly;
2. split rigid silver metal from flexible red cloth;
3. identify the dominant primary end per pose by design and relative footprint,
   never by screen direction;
4. uniformly rotate and scale only the rigid component onto the pose's shaft;
5. preserve pose-natural cloth and exact body/hand/costume pixels with explicit
   cleanup and occlusion masks;
6. hard-fence every edit and require zero changes outside the saved prop mask;
7. derive a clean alpha matte, restore exact opaque subject color, normalize to
   one canvas/baseline, and inspect on alternating dark/light panels.

That deterministic metal-only path supplies seven final Mirelle slots. The
original concerned source hid lower-body pixels behind its oversized head, so
its repair lane was stopped after preserved v6/v7/v8 and one-shot inpaint
failures. One fresh concerned candidate was generated from the locked views and
canonical crop with the complete spear held beside the unobstructed body; it is
the selected eighth slot and received no later model edit.

Tiny Qwen masked seam edits were run once per eligible repaired pose to test
whether model harmonization helped. They did not: each softened the socket or
introduced muddy magenta/burgundy pixels. Deterministic candidates remain
selected, and every model attempt is retained with its exact request, response,
mask, hard composite, metrics, and rejection reason.

Rowan's new `sad` generation is unambiguously sad and retains exactly two
visibly distinct sheathed knives. His other seven selected bases are unchanged
from the original pack. The original 20260826 pack remains byte-identical at
aggregate SHA-256
`6c47390015a9cac61d285ece0bbff59bcda7ea21bb1c83758c013f5932370b33`.

## Artifact map

- `manifest.json` — selected raw, prompt, reference, repair, matte, sprite,
  contact-sheet, Qwen, and hash provenance.
- `REVIEW_INDEX.md` — concise entry point for this pack and the companion
  104-sprite full-cast/default review pack.
- `exact_generation_prompt_ledger.jsonl` — recovered exact built-in prompts and
  hash-matched outputs for the four v2 model generations.
- `prompts/legacy_selected_bases/` — exact prompt files inherited from the
  byte-frozen original pack.
- `generation_raw/` — new v2 built-in outputs, unmodified.
- `grafts/`, `components/`, and `masks/` — deterministic prop repair inputs,
  outputs, and exact edit scopes.
- `component_proofs/` and `mask_proofs/` — readable before/after and mask views.
- `qwen_requests/`, `qwen_raw/`, `qwen_aligned/`, `qwen_composited/`, and
  `qwen_reviews/` — exact one-shot masked-edit evidence.
- `candidates/` — native-size RGBA matte results.
- `sprites/` — normalized 1100 x 1500 RGBA review candidates.
- `rejected/` — preserved generation, graft, matte, and cross-pose failures.
- `inventory.sha256` — hash list used to validate the Windows review copy.
- `build_v2_review_pack.py` — reproducible manifest, overview, prompt recovery,
  image validation, and inventory builder.

The Windows review copy is
`C:\Users\danim\Pictures\Ayoa\MirelleRowanPoseExpressionReviewV2_20260827`.
