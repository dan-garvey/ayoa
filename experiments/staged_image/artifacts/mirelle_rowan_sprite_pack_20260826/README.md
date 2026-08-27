# Mirelle Voss and Rowan Kest pose-expression review pack

This is an experimental, unlocked authoring pack. It is not bound to the image
director, runtime, checkpoints, or reviewed-reference registry. Human selection
is required before any sprite is called approved or locked.

The pack contains one attempted full-body visual-novel sprite for each core
label—`neutral`, `happy`, `concerned`, `tense`, `skeptical`, `angry`, `sad`, and
`surprised`—for Mirelle Voss and Rowan Kest. Each label changes both facial
expression and body language. Mirelle is consistently oriented toward
screen-left for a right-side VN slot; Rowan is oriented toward screen-right for
a left-side slot.

## Review first

- `contact_sheets/mirelle_voss_complete_sweep.png`
- `contact_sheets/rowan_kest_complete_sweep.png`
- `manual_review.md`
- `prompt_conflict_audit.md`

Each contact-sheet cell is split between dark and light backgrounds so actual
alpha edges are visible. The readable full-resolution candidates are under
`sprites/<character>/<variant>.png`.

## What the pilot established

The built-in image tool produced strong character art but returned RGB files
with checkerboards painted into the pixels when asked for transparency. A
targeted background-extraction edit repeated that failure. All four failed-alpha
files are retained under `rejected/`.

The successful authoring protocol was therefore:

1. manually inspect all four hash-locked views per character;
2. give each view one explicit authority role rather than averaging conflicts;
3. generate each distinct pose-expression in a separate built-in call on a
   uniform magenta chroma background;
4. sample the actual generated border color and derive RGBA with the imagegen
   skill's deterministic chroma-key helper;
5. place native pixels, without resampling, on a common 1100 x 1500 transparent
   canvas with a shared raw-canvas baseline;
6. validate dimensions, alpha range, hashes, coverage, and significant connected
   components, then inspect every image manually.

The result is sixteen genuine RGBA sprites. Every file contains one significant
connected character/weapon component and no environment, text, logo, watermark,
duplicate person, or extra weapon.

## First-pass findings

The pose direction is successful: the emotional silhouettes remain distinct at
contact-sheet scale. Weapon ownership is also substantially better than a group
generation would be. Mirelle keeps one spear throughout; Rowan keeps two knives
throughout, with drawn/sheath state changing by emotion.

The pack is not ready to lock wholesale:

- Mirelle's `concerned` and especially `tense` sprites select the feather clasp,
  shorter upper coat, and repaired stitching from her facial reference more
  strongly than the other variants. Those details are present in a locked view,
  but continuity across the sweep needs a human choice of costume authority.
- Rowan's `sad` sprite has exactly two sheathed weapons, but their exposed pointed
  pommels read unlike his canonical mismatched knife hilts. Regenerate that one
  after the preferred pack direction is selected.
- Mirelle's `tense` face can read as determined or guarded rather than visibly
  anxious. It is pose-distinct but semantically less precise than the other
  seven.

## Artifact map

- `protocol.json` — frozen identity, orientation, weapon, and pose-expression
  specifications.
- `prompts/` — exact prompt text for every generation and pilot correction.
- `generation_raw/` — all sixteen built-in chroma outputs, unmodified.
- `candidates/` — direct alpha-extraction results at native generated dimensions.
- `sprites/` — normalized 1100 x 1500 RGBA review candidates; no art resampling.
- `rejected/` — preserved transparent-output failures.
- `rejection_log.md` — exact failure reasons and iteration outcome.
- `manifest.json` — source, prompt, output, dimension, alpha, component, and hash
  provenance.
- `assemble_review_pack.py` and `build_prompts.py` — reproducible experiment-only
  authoring helpers.

The Windows review copy is
`C:\Users\danim\Pictures\Ayoa\MirelleRowanPoseExpressionReview_20260826`.
Its files are hash-compared against this tracked artifact directory after copy.
