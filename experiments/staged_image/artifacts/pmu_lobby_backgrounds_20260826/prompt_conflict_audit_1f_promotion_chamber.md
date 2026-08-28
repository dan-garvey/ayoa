# 1F Promotion Chamber prompt conflict audit

Status: **PASS before candidate generation**

Scope: the complete positive and exclusion contract in
`prompts/1f/promotion_chamber_candidate_v1.txt`, checked against the approved
beginner-tier 1F material language, the existing fixed Synthesis Chamber, and
the requirement for broad visual-novel sprite staging.

## Frozen reference roles

- The approved 1F courtyard supplies only construction tier, material palette,
  maintenance level, and rendering language.
- The approved 1F Synthesis Chamber supplies only interior scale, camera height,
  and staging discipline. Its cold-blue recessed basin and glass barriers are
  explicitly excluded.
- The approved PMU crop supplies only architectural rhythm and illustration
  quality. It is not an edit target and no exact building may be copied.

## Positive-instruction scan

| Potential conflict | Resolution before the call |
| --- | --- |
| `Promotion Chamber` can invite a shrine, throne room, or luxurious ritual hall. | The positive scene is a small practical facility made from inexpensive plaster, ordinary timber, and gray stone; the avoid contract removes shrine, temple, throne, palace, gold, heraldry, carving, and ceremonial scale. |
| A centered dais and symmetric view can imply an altar or ceremony. | The dais is explicitly a shallow purpose-built mechanism in the rear third, while symmetry is limited to legibility and explicitly not ceremonial. |
| White-gold transformation light can produce gold ornament or holy iconography. | The prompt uses restrained pearly white light with only faint warm cream at its center; gold trim, glyphs, statues, altars, crystals, and religious architecture are excluded. |
| Reusing the Synthesis Chamber as a reference can duplicate its blue basin and barriers. | Its role is limited to scale, construction tier, camera height, and staging. Blue light, recessed transfer basin, glass barriers, tanks, coils, cables, consoles, and laboratory machinery are explicitly excluded. |
| Inlay rings can become readable runes or pseudo-text. | The positive material cue is `subtle pale mineral inlay without letters or symbols`; the constraints separately forbid text, glyphs, and pseudo-writing. |
| `polished` can be interpreted as expensive construction. | The prompt states that polish applies to rendering quality while architecture remains inexpensive and restrained. |

## Result

- No positive instruction requests people, characters, a second apparatus,
  synthesis equipment, luxury, religious meaning, monumental scale, readable
  markings, furniture, or foreground clutter.
- The room remains a clean maintained 1F facility rather than a ruin or a
  high-tier ceremonial chamber.
- The rear apparatus, eye-level framing, and broad empty central/lower floor do
  not compete with one or two full-body sprites or the ADV dialogue panel.
- The candidate is eligible for one independent built-in `image_gen` call.
  It remains an isolated review candidate until explicit human approval.
