# Rejection log: Mirelle Voss and Rowan Kest v2

This is negative provenance for the experimental review pack. A file listed
here is preserved so a failed method or visual decision is not repeated; it is
not production approval. Current positive dispositions are recorded in
`manual_review.md`.

## Generation and semantic rejections

- **Mirelle concerned v2 — pose accepted, generated weapon rejected.**
  `generation_raw/mirelle_voss/concerned_chroma_v2.png` has the intended worried
  reach and concerned expression, but its lower endpoint is an oversized,
  halberd-like primary head and the generation prompt incorrectly says
  “ordinary butt end.” The raw is retained as the semantic/body base only; it
  must not be promoted with its generated weapon unchanged. Exact prompt
  provenance is in
  `prompts/recovered_v2/mirelle_voss/concerned_v2.txt` and
  `exact_generation_prompt_ledger.jsonl`.

- **Mirelle tense v2 — semantic rejection.**
  `generation_raw/mirelle_voss/tense_chroma_v2.png` reads as stern,
  determined, and combat-ready: lowered brows, wide stance, and a two-handed
  intercepting guard. It does not read as anxious threat anticipation. Its
  exact rejected prompt is
  `prompts/recovered_v2/mirelle_voss/tense_v2_rejected.txt`. The semantically
  selected retry is `generation_raw/mirelle_voss/tense_chroma_v3.png`.

- **Rowan sad v1 — generated prop rejection.**
  `../mirelle_rowan_sprite_pack_20260826/generation_raw/rowan_kest/sad_chroma_v1.png`
  has readable sadness, but the two exposed pointed fantasy pommels do not read
  as Rowan's canonical, visibly distinct sheathed knife hilts. The v2
  replacement `generation_raw/rowan_kest/sad_chroma_v2.png` keeps the sad
  face/body read and restores two recognizable sheathed knives.

## Deterministic extraction and graft rejections

- **Concerned reference graft v1 — bad extraction and shaft slivers.**
  `rejected/mirelle_concerned_reference_graft_v1_bad_extraction_and_shaft_slivers/mirelle_concerned_reference_graft_v1.png`
  contains holes in the extracted component plus visible fragments of the old
  shaft/head around the replacement. Its masks, extracted component, metadata,
  and proof remain in the same rejection directory.

- **Concerned reference graft v2 — near-key residue.**
  `rejected/mirelle_concerned_reference_graft_v2_near_key_edge_residue/mirelle_concerned_reference_graft_v2.png`
  improves the extraction but leaves pale/magenta-adjacent edge residue and an
  unfinished seam. Its complete extraction/mask/metadata set remains in the
  same rejection directory.

- **Concerned full-component v6 — cross-pose scale rejection.**
  `rejected/mirelle_concerned_v6_cross_pose_scale/concerned_sprite_v6_rejected.png`
  and
  `rejected/mirelle_concerned_v6_cross_pose_scale/mirelle_voss_complete_sweep_v2_concerned_v6_rejected.png`
  show that the full blade-plus-streamer replacement reads substantially larger
  than Mirelle's weapon and figure treatment in the other seven poses. The
  semantic base remains accepted, but this v6 normalized presentation is not a
  selected Concerned final. The final selection is the unobstructed fresh
  `generation_raw/mirelle_voss/concerned_chroma_v3.png` candidate.

- **Concerned metal-only v7 — detached old metal and fragmented guards.**
  `rejected_provenance/mirelle_voss/concerned_v7_detached_old_metal_and_fragmented_guards/concerned_lower_primary_metal_reference_graft_v7.png`
  uses a cross-pose-compatible physical scale, but detached old silver remains
  near the boots and the downscaled lateral guards read as floating/broken.
  The complete script, masks, proofs, hashes, and corrected disposition are in
  that rejection directory.

- **Concerned metal-only v8 — erased source-hidden lower body.**
  `rejected_provenance/mirelle_voss/concerned_v8_old_head_cleanup_erased_occluded_lower_body/grafts/mirelle_voss/concerned_lower_primary_metal_reference_graft_v8.png`
  passes residual-silver and connected-metal gates, but removing the old
  oversized head exposes large magenta holes where the source image never
  contained the occluded greaves, boots, legs, and coat edges. Its archive
  manifest explicitly withdraws the invalid claim that source-hidden body
  pixels were preserved.

- **Neutral full-component rigid v0 — wrong material coupling and wrong end.**
  `rejected/mirelle_neutral_full_component_rigid_v0_wrong_streamer_coupling/mirelle_neutral_full_component_rigid_v0_close_proof.png`
  rigidly rotates flexible red streamers with the silver metal and repairs the
  opposite lower cap instead of the primary assembly. Exact disposition and
  transform are in
  `rejected/mirelle_neutral_full_component_rigid_v0_wrong_streamer_coupling/rejection_metadata.json`.

- **Happy full-component rigid preview — wrong cloth physics.**
  `rejected/mirelle_happy_full_component_rigid_transform_preview/full_component_rigid_transform_preview.png`
  locks the flexible streamers to the blade's rotation instead of preserving
  pose- and gravity-aware cloth. The preserved disposition is
  `rejected/mirelle_happy_full_component_rigid_transform_preview/rejection.json`.

- **Happy wrong-end v1 — endpoint classification failure.**
  `rejected/mirelle_happy_wrong_end_v1/grafts/mirelle_voss/happy_canonical_spear_metal_repair_proof.png`
  places the canonical primary blade beside the boot on the opposite butt-cap
  endpoint, creating double-primary weapon semantics. The complete archived
  script/mask/proof set and reason are indexed by
  `rejected/mirelle_happy_wrong_end_v1/rejection.json`. The selected correction
  is `grafts/mirelle_voss/happy_upper_primary_metal_reference_graft_v2.png`.

- **Tense rigid-streamer graft v1 — oversized head and cloth collision.**
  `grafts/mirelle_voss/tense_primary_spear_reference_graft_v1.png` rotates the
  full metal-and-streamer component as one rigid object. The head is oversized
  and the streamers flare into Mirelle's face/hair. The close evidence is
  `component_proofs/mirelle_tense_reference_graft_v1.png`; the selected
  metal-only correction is
  `grafts/mirelle_voss/tense_primary_metal_reference_graft_v2.png`.

## Matte rejections

- **Mirelle physical-unmix v1 — opaque subject color damage.**
  `contact_sheets/rejected/mirelle_voss_complete_sweep_physical_unmix_v1.png`
  and `rejected/mirelle_voss_physical_unmix_v1/` preserve the first matte lane.
  Global physical unmix shifts intentional red hair, burgundy clothing, red
  shafts, and tassels toward orange/brown/ochre. The selected opaque-hybrid lane
  keeps its clean alpha and partial-edge RGB but restores raw RGB in opaque and
  high-alpha interior subject pixels; see
  `matte_reports/mirelle_voss_v2_opaque_hybrid.json`.

- **Rowan sad physical-unmix v1 — opaque subject color damage.**
  `rejected/rowan_kest_sad_physical_unmix_v1.png` is retained as the alpha/edge
  source, not selected RGB. Global unmix alters opaque skin, blond hair, green
  scarf, and clothing color. The selected opaque-hybrid result is
  `candidates/rowan_kest/sad.png`, with parameters and hashes in
  `matte_reports/rowan_kest_v2.json`.

## Qwen masked-edit rejections

All Qwen outputs below are preserved experiments. Where a hard composite was
used, pixels outside the saved mask remain exact; rejection is based on the
edited pixels and their boundary. No Qwen output replaced a deterministic
candidate.

### Concerned exploratory calls

- `qwen_composited/mirelle_voss/concerned_spear_masked_v1.png` — the broad edit
  produces an oversized, over-complex halberd-like lower head rather than the
  canonical long central blade, two small guards, and coherent cloth. The raw
  model response is
  `qwen_raw/mirelle_voss/concerned_spear_masked_v1_model_output.png`.

- `qwen_composited/mirelle_voss/concerned_lower_spearhead_masked_v2.png` — the
  narrower retry still leaves the same oversized lower silhouette and does not
  recover the canonical assembly. Exact call parameters and hashes are in
  `qwen_metadata/mirelle_concerned_lower_spearhead_masked_v2.json`.

- `qwen_composited/mirelle_voss/concerned_reference_graft_harmonized_v1.png` —
  harmonization enlarges and restacks the blade/guards and creates excessive
  streamers around the boots, with contaminated mixed-color seam pixels. Exact
  call evidence is in
  `qwen_metadata/mirelle_concerned_reference_graft_harmonize_v1.json`.

- `qwen_composited/mirelle_voss/concerned_reference_graft_harmonized_v2.png` —
  the reduced-mask retry still leaves an over-large head/cloth silhouette and
  muddy blended edges. Exact call evidence is in
  `qwen_metadata/mirelle_concerned_reference_graft_harmonize_v2.json`.

The deterministic Concerned v6 full-component result was also rejected at
cross-pose scale, and the later v7/v8 deterministic repairs also failed visual
silhouette review. None of these Concerned edit artifacts is selected.

- **Concerned lower-body inpaint:**
  `qwen_composited/mirelle_voss/concerned_lower_body_inpaint_v1.png` leaves
  32,566 of 33,668 authorized pixels key-like instead of reconstructing the
  source-hidden greaves, boots, legs, and coat edges. Outside-mask delta is
  zero, and no metal overlay was performed. Exact one-shot disposition and
  hashes are in
  `qwen_reviews/mirelle_concerned_lower_body_inpaint_v1.json`. This failure
  ended the source-pose repair lane; it was not retried.

### One-shot socket-seam calls

- **Neutral:**
  `qwen_composited/mirelle_voss/neutral_socket_seam_qwen_v1.png` introduces
  muddy gray-magenta transition pixels and a stronger hard-mask discontinuity.
  `qwen_reviews/mirelle_neutral_socket_seam_qwen_v1.json` retains the metrics
  and selects `grafts/mirelle_voss/neutral_canonical_metal_repair_v1.png`.

- **Happy:**
  `qwen_composited/mirelle_voss/happy_upper_socket_seam_qwen_v1.png` replaces
  clean key/silver pixels with burgundy and muddy mauve, softening collar
  linework into a pasted tonal patch.
  `qwen_reviews/mirelle_happy_upper_socket_seam_qwen_v1.json` retains the
  metrics and selects
  `grafts/mirelle_voss/happy_upper_primary_metal_reference_graft_v2.png`.

- **Tense:** `qwen_composited/mirelle_voss/tense_socket_seam_qwen_v1.png`
  softens the collar bands and leaves a blocky tonal patch at the hard-mask
  boundary. `qwen_metadata/mirelle_tense_socket_seam_qwen_v1.json` retains the
  original-resolution disposition and selects
  `grafts/mirelle_voss/tense_primary_metal_reference_graft_v2.png`.

- **Skeptical:**
  `qwen_composited/mirelle_voss/skeptical_socket_seam_qwen_v1.png` converts
  clean key and neutral silver into purple, burgundy, and muddy transitions,
  visibly softening the socket.
  `qwen_reviews/mirelle_skeptical_socket_seam_qwen_v1.json` retains the metrics
  and selects
  `grafts/mirelle_voss/skeptical_primary_metal_reference_graft_v1.png`.

- **Angry:** `qwen_composited/mirelle_voss/angry_socket_seam_qwen_v1.png`
  turns clean key/silver into dark burgundy, purple, and muddy gray, softens the
  diagonal connector, and increases boundary discontinuity.
  `qwen_reviews/mirelle_angry_socket_seam_qwen_v1.json` retains the metrics and
  selects `grafts/mirelle_voss/angry_primary_metal_reference_graft_v1.png`.

- **Sad:** `qwen_composited/mirelle_voss/sad_socket_seam_qwen_v1.png`
  introduces a pink-gray vertical strip and muddy mixed colors, worsening the
  tiny patch boundary. `qwen_reviews/mirelle_sad_socket_seam_qwen_v1.json`
  retains the metrics and selects
  `grafts/mirelle_voss/sad_canonical_metal_repair_v1.png`.

- **Surprised:**
  `qwen_composited/mirelle_voss/surprised_socket_seam_qwen_v1.png` introduces
  muddier green-gray socket pixels and raises both mean and median boundary
  jumps. `qwen_metadata/mirelle_surprised_socket_seam_qwen_v1.json` retains the
  original-resolution disposition and selects
  `grafts/mirelle_voss/surprised_primary_metal_reference_graft_v1.png`.

## Selection summary

- Stable Mirelle deterministic selections are
  `grafts/mirelle_voss/neutral_canonical_metal_repair_v1.png`,
  `grafts/mirelle_voss/happy_upper_primary_metal_reference_graft_v2.png`,
  `grafts/mirelle_voss/tense_primary_metal_reference_graft_v2.png`,
  `grafts/mirelle_voss/skeptical_primary_metal_reference_graft_v1.png`,
  `grafts/mirelle_voss/angry_primary_metal_reference_graft_v1.png`,
  `grafts/mirelle_voss/sad_canonical_metal_repair_v1.png`, and
  `grafts/mirelle_voss/surprised_primary_metal_reference_graft_v1.png`.
  Concerned is the separately generated, root-accepted
  `generation_raw/mirelle_voss/concerned_chroma_v3.png`; it keeps the worried
  reach and holds one complete double-ended spear fully off the body silhouette.
  V6, v7, v8, and the lower-body inpaint are all explicitly rejected.

- Rowan's selected sad replacement is
  `generation_raw/rowan_kest/sad_chroma_v2.png` processed to
  `candidates/rowan_kest/sad.png` and `sprites/rowan_kest/sad.png`. His other
  seven v2 variants retain the accepted original-pack bases.

- No rejected Qwen, full-component rigid transform, wrong-end repair,
  physical-unmix RGB result, or superseded semantic generation is selected for
  promotion.
