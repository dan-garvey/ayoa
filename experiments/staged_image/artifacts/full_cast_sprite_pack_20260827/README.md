# Full-cast VN pose-expression review pack

This is an experimental, unlocked authoring pack. It is not bound to Ayoa's
image director, runtime, checkpoints, or reviewed-reference registry. Runtime
LLMs never read these images. Human promotion remains a separate decision.

The pack contains 104 selected candidates: eight coordinated pose-expression
labels (`neutral`, `happy`, `concerned`, `tense`, `skeptical`, `angry`, `sad`,
and `surprised`) for eleven seeded identities plus masculine and feminine
generic veiled one-star defaults.

## Review first

- `complete_cast_pose_expression_overview.png` — all 13 sweeps in one labeled
  overview; root-reviewed SHA-256
  `39a5ac106bde5f5463ce5311c9bc448b6de0dc02fc21dd00ee8ca80627b62c51`.
- `contact_sheets/<character>_complete_sweep.png` — original sweep sheets with
  alternating dark/light alpha checks.
- `manual_review.md` — the 104-row manual rubric.
- `prompt_conflict_audit.md` — source exclusions, pose semantics, prop lessons,
  and subject-color-safe matte decisions.
- `rejection_log.md` — preserved generation and matte failures.

Full-resolution transparent sprites are under `sprites/<character>/<label>.png`.
Every selected sprite is RGBA 1100 x 1500 on a common baseline.

## Provenance

- `manifest.json` records every selected raw, exact generation prompt and
  reference list, native RGBA candidate, normalized sprite, contact sheet,
  hash, dimensions, alpha statistics, and manual disposition.
- `exact_generation_prompt_ledger.jsonl` reconstructs the actual built-in call
  strings from rollout records and pairs them with their landed raw files.
- `generation_raw/` preserves accepted and rejected opaque chroma intermediates.
- `candidates/` contains native-size alpha results.
- `matte_reports/` records the color-safe processing lanes and invariants.
- `rejected/` and `contact_sheets/rejected/` preserve superseded matte passes.

## Important exclusions

- Wren uses only the top-level `wren_thelantern.png` source. Her locked
  `active_profile.png` was never passed because of the kneepads.
- Soren uses only repaired `identity_base`, `facial_zoom`, `anatomy`, and
  `back_view` references. The prohibited original and locked active profile were
  never passed.

## Processing lesson

One global chroma algorithm is unsafe. Physical magenta unmix removed Wren's
edge fringe but also changed intentional burgundy, purple, copper, red, pink,
and mauve subject colors on other identities. Color-critical opaque characters
therefore use a bounded opaque-hybrid, while translucent Iselle uses a
connected-key treatment that preserves interior colors and wing alpha.

The Windows review copy is
`C:\Users\danim\Pictures\Ayoa\FullCastPoseExpressionReview_20260827`.
