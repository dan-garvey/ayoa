# 1F scene prompt conflict audit

Status: **PASS before generation calls**

Audited prompt hashes:

- corrected `crack_of_space_and_time.txt`: `f0d6ed95a5fa78b11ec31a8a05a4e2c28597bc2f4d9c51e2ebddebb71f5f9e24`
- `synthesis_chamber.txt`: `1f5f682611e36fc202823de37e113e6b6ef5d9db8c03c75cfdf17f23b9b41ce9`

Scope: positive instructions in `prompts/1f/crack_of_space_and_time.txt` and
`prompts/1f/synthesis_chamber.txt`, checked against the approved beginner-tier
visual language, empty VN staging contract, and each scene's avoid list. Input
images are labelled as references rather than edit targets.

## Frozen invariants

- Built-in `image_gen`, one independent call per final asset.
- Existing approved 1F plates and curated PMU crop bytes remain unchanged.
- Polished webtoon rendering language, eye-level exact-16:9 framing, and broad
  central/lower sprite-staging space.
- Clean, maintained, inexpensive 1F architecture with no people, text, UI,
  logos, watermarks, or panel borders.

## Positive-instruction scan

| Prompt | Potential conflict | Resolution before call |
| --- | --- | --- |
| Crack | “Crack of Space and Time” can invite a generic lightning tear in the sky, while the supplied canonical source establishes a broad arch, ribbed dark aperture, and block-glitch seam. | The corrected positive scene explicitly preserves those canonical traits. Only the surrounding lobby remains modest, undecorated 1F construction. |
| Crack | Blue-violet energy can drift into thermal, neon, rainbow, or dark-dungeon color language. | Color is limited to a restrained dark interior plus pearl-white/cold-cyan edge light; the scene stays bright late-morning daylight. |
| Crack | “In front of” can invite a person posed before the aperture. | The asset is repeatedly specified as an empty, character-free environment plate; viewpoint and foreground staging express the relation instead. |
| Synthesis | “Operational chamber” can invite consoles, screens, machinery, laboratory props, or readable interface elements. | The only apparatus is one low architectural floor recess with two plain barrier panels and nonlinguistic light seams. The positive scene explicitly says the rest of the room has no machinery. |
| Synthesis | Clinical unease can invite gore, restraints, torture props, darkness, or dilapidation. | Unease comes only from cleanliness and purpose. Neutral daylight dominates; gore, remains, restraints, horror lighting, decay, and dungeon cues are excluded. |
| Both | “Polished” can be read as luxurious architecture. | Each prompt says rendering quality is polished while construction remains inexpensive, utilitarian, and restrained. |

## Result

No positive instruction requests a person, creature, text, glyph, interface,
logo, watermark, upper-tier monument, ceremonial architecture, luxury, ruin,
grime, damage, or visual clutter. The two scene-specific focal features are
bounded architectural/environmental facts and leave the lower foreground clear.

## Source-reference correction

The first Crack generation completed after the interrupted call but predated the
user-supplied canonical source image. It is preserved as rejected evidence: its
ordinary 1F courtyard passed, but its unsupported star-filled lightning tear
failed the canonical Crack identity. The corrected prompt treats the supplied
image as Image 1 and removes the earlier contradictory prohibition on an
enclosing arch.
