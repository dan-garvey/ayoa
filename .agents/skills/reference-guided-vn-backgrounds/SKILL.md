---
name: reference-guided-vn-backgrounds
description: Curate and generate reusable visual-novel environment plates from local image corpora using coding-time inspection, exact crop provenance, controlled reference-guided generation, and human promotion. Use for tiered lobbies and other character-free VN locations; never use it to add runtime LLM vision or to generate character sprites.
---

# Reference-Guided VN Backgrounds

Build reviewable environment plates whose visual direction comes from curated
source crops rather than free-form runtime prompting. Use the system `imagegen`
skill for the raster-generation calls; this skill adds Ayoa's provenance,
staging, review, and promotion contract.

## Runtime Boundary

All image inspection happens during development by the coding agent. Runtime
router, narrator, character-agent, rules-adapter, and image-director LLM calls
must never receive image bytes or invoke image understanding.

After human approval, runtime selection may receive only an opaque asset handle
and authored text metadata such as location, architectural tier, lighting, and
applicability. Do not expose source paths, crop coordinates, hashes, or
image-derived analysis to runtime models.

## Candidate Workflow

1. Keep the experiment isolated from production bindings, approved originals,
   prompts, and `.env`. Preserve every source file untouched.
2. Screen the complete in-scope source set. Make contact sheets when the corpus
   is larger than can be reviewed reliably one file at a time, then inspect each
   retained candidate at useful resolution.
3. Prefer environment-dominant panels. Reject references dominated by people,
   dialogue balloons, panel borders, furniture clutter, or a mood contrary to
   the requested location. A minor distant figure is acceptable only when the
   generation prompt explicitly excludes people.
4. Create deterministic lossless crops. Record the source path and SHA256,
   half-open pixel rectangle `[left, top, right, bottom)`, crop dimensions and
   SHA256, and why the crop was accepted. Verify the crop pixels against the
   decoded source region.
5. Freeze the environment contract before generation: intended story use,
   architectural tier, camera height, aspect ratio, lighting, staging space,
   required materials, and exclusions. Label every supplied image as a visual
   reference rather than an edit target.
6. Generate one final candidate per independent built-in `image_gen` call.
   References may guide palette, rendering language, architecture, and spatial
   rhythm, but the prompt must forbid copying source characters, dialogue,
   text, logos, panel borders, or exact buildings.
7. Normalize only when needed and record it exactly. Prefer a deterministic
   crop without resampling or repainting. Preserve the raw tool output.
8. Inspect every final at full useful resolution. Check dimensions and ratio,
   broad central/lower staging space for sprites and the dialogue panel,
   requested tier, people or creatures, readable or pseudo-text, UI, logos,
   watermarks, source mimicry, and every prompt-specific exclusion.

## Prompt Conflict Audit

Audit the complete prompt before each generation call. Compare positive scene
instructions and visible reference cues against the constraint and avoid lists;
do not assume a negative prompt will overcome a contradictory positive request.

Common traps include:

- `guild lobby` inviting heraldry, banners, or monumental gates;
- `pavilion` inviting ceremonial symmetry or vaulted grandeur;
- `bright` or `welcoming` inviting decorative fixtures;
- `polished` being interpreted as luxurious architecture.

Resolve each conflict in positive wording before the call. State, for example,
that polish applies to rendering quality while construction remains inexpensive
and restrained. Freeze and hash the audited prompt.

Change one intentional visual variable per comparison. If testing architectural
tier, keep references, rendering language, daylight, viewpoint, framing, and
staging contract fixed. Preserve accepted earlier outputs byte-for-byte and
change only their authored tier metadata.

## Review Pack

Keep accepted references separate from screening and rejected crops. A useful
pack contains:

- untouched-source and accepted-crop provenance;
- contact sheets or indexed screening evidence;
- raw generations and normalized candidates;
- exact prompts, prompt hashes, reference-to-output mapping, and tool mode;
- output dimensions, SHA256 values, and manual acceptance or rejection notes;
- a side-by-side overview when comparing variants or tiers.

Copy or link the pack into a user-named review location when requested, then
verify that review copies hash-match the repository artifacts.

## Promotion

A technically valid image remains a candidate until the user approves it.
Promotion is a separate change that selects the exact file, assigns story-owned
authored metadata and an opaque handle, and updates the location/background
binding. Do not silently replace an approved plate, promote every generated
variant, or make free-form runtime diffusion the only source of continuity for
an authored recurring location.
