# Locked visual reference set

This directory contains the reviewed locked multi-view assets for the 13 seeded visual identities. Each character directory uses the same canonical filenames:

- `anatomy.png`
- `back_view.png`
- `active_profile.png`
- `facial_zoom.png`

Soren also has `identity_base.png`, the repaired Base C-derived locked identity image. The Warden’s four files intentionally contain the same single locked reference because that character was explicitly set to use one reference for every view.

The seed checkpoint hash-pins these locked files with an authored, text-only
selection hint for each view. The image director sees only the opaque handle,
character owner, and hint; image bytes, paths, hashes, and dimensions remain
private to the diffusion runtime. The Warden binds the pre-existing
`warden_of_the_eighth.webp` with those same reviewed bytes because its four
organized files intentionally duplicate one reference. The checkpoint also
keeps each character's approved top-level original as a selectable baseline
anchor. Soren is the sole exception: his rejected top-level original remains
unregistered, and `identity_base.png` is his repaired baseline instead.
