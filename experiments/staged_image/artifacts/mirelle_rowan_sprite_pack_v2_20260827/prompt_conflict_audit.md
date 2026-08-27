# Mirelle and Rowan v2 prompt-conflict audit

This file supersedes the Mirelle weapon interpretation in the 20260826 pack.
It does not modify that earlier pack: the old files remain byte-identical as
historical evidence of the first experiment.

## Canonical Mirelle spear contract

Full-resolution inspection of the locked `active_profile` establishes one
continuous, deliberately double-ended spear:

- the primary end has one long central silver blade, two small lateral
  wings/guards, and long red streamers tied at its socket;
- the opposite end has a small pointed silver cap and its own short red tassel;
- either end may appear at the top, bottom, left, or right of the frame as the
  weapon rotates with the pose.

The first pack's prose incorrectly resolved the visible second point as an
ordinary butt cap. Its initial angry pilot prompt was more explicit still: it
required one leaf-shaped head and said `no double-ended spear`. The v2
`concerned` prompt similarly said `ordinary butt end`, and the accepted semantic
`tense` retry said `simple butt cap at the opposite end`. Those are prompt facts
that conflict with the actual reference image. They are preserved in the exact
prompt ledger as failed-generation evidence and must not be reused as canonical
guidance.

Seven selected v2 corrections do not ask a model to reinterpret that conflict.
They extract the reference's primary rigid silver assembly once, freeze its
hash, and apply a per-pose uniform scale and rotation to the visually dominant
main end. The original pose-specific red cloth and the opposite small pointed
cap/tassel are retained. Endpoint selection is based on relative blade design
and footprint, never screen direction.

`concerned` is the bounded exception. Its v2 source placed an oversized head
over the legs and coat, so removing that head exposed pixels that never existed
in the source. Deterministic v6/v7/v8 repairs and one lower-body inpaint are
preserved rejections. The selected fresh v3 candidate instead keeps the worried
open reach and holds the complete double-ended spear beside the body. It was
generated once from the four locked views plus the canonical crop and received
no subsequent prop edit.

## Rigid metal versus flexible cloth

The first deterministic graft coupled the reference's blade and long streamers
into one rigid component. On `tense`, rotating that combined component made the
cloth flare into Mirelle's face and hair and enlarged the silhouette. That
candidate is rejected and preserved.

The corrected protocol splits the frozen component by material:

- rigid silver blade, guards, collar, and socket rotate together;
- red streamers remain pose-specific, gravity-aware cloth;
- the extracted reference streamers are used only when a base genuinely lacks
  coherent cloth, as in the approved `concerned` repair.

This removes the instruction-level contradiction between “match the weapon
exactly” and “preserve physically believable pose-dependent cloth.”

## Tense semantics

The first v2 `tense` prompt combined defensive tension with a lowered stance,
hard-drawn brows, split weight, and a two-handed intercepting guard. The result
read stern, determined, and combat-ready—the same semantic collision the retry
was meant to avoid.

The selected `tense_chroma_v3` instead specifies anxious threat anticipation:
slightly widened side-tracking eyes, lifted and pinched brows, raised shoulders,
contracted elbows, weight drawn back, and a close vertical spear support. It is
distinct from `concerned`, which reaches outward to help someone, and `angry`,
which commits forward into confrontation.

## Qwen seam experiments

The Qwen lane is intentionally subordinate to deterministic candidates. Each
approved deterministic pose may receive at most one tiny socket/seam edit. The
mask excludes blade shape, cloth, shaft, character, costume, and the opposite
end; the returned image is aligned and hard-composited so every outside-mask
pixel remains exact.

The completed `neutral`, `happy`, `tense`, `skeptical`, `angry`, `sad`, and
`surprised` seam calls were rejected. Despite prompts asking only for edge
integration, the model introduced muddy key colors, soft collar bands,
pink/gray strips, or stronger hard-mask transitions. Concerned broad/narrow
object edits also failed to normalize the prop, and its later one-shot
lower-body inpaint preserved the magenta void rather than reconstructing hidden
anatomy. These failures are not treated as wording problems to solve through
repeated prompting. The clean deterministic sources remain selected for seven
labels, and fresh concerned v3 is selected without a Qwen pass.

## Reference authority

- `active_profile` is authoritative for outfit, palette, weapon construction,
  and overall silhouette.
- `facial_zoom` is authoritative for face, eyes, hairline, and close identity;
  its cropped garment differences do not override the active profile.
- `anatomy` supplies adult proportions only, never clothing.
- `back_view` supplies rear hair, coat, and carried-weapon construction.

The v2 correction changes only spear pixels and the minimum surrounding key
needed to remove old-head fragments. Face, body, costume, pose, scale, and
framing are preserved from the selected bases.

## Alpha and runtime scope

The built-in generator reliably produced opaque chroma intermediates, not clean
transparent sprites. Prompts therefore ask for one flat magenta plate only;
they do not simultaneously demand transparency. Alpha extraction,
decontamination, normalization, and contact-sheet assembly are deterministic
coding-time operations. No Ayoa runtime LLM reads an image, and no candidate in
this experiment is bound to production.
