# Full-cast pose-expression prompt-conflict audit

This is an experimental authoring audit. It changes no runtime prompt,
checkpoint, reference registry, or production sprite binding.

## Ranked reference roles

For each seeded identity, the four locked views were treated as complementary
authorities rather than four interchangeable pictures:

- active profile: identity silhouette, canonical outfit, palette, and props;
- facial zoom: face, eye color, hairline, and close identity only;
- anatomy: adult or character-specific proportions only, never clothing;
- back view: rear hair, garment layers, and carried-prop construction.

This ranking prevents an anatomy study or cropped facial costume from silently
overriding the canonical outfit. Generated candidates were never reused as
identity references for seeded characters.

## Wren source exclusion

All eight selected Wren calls use only
`app/storage/stories/one_star_ascension_s1/visual-references/wren_thelantern.png`.
The locked `wren_thelantern/active_profile.png` was never passed because its
kneepads conflict with the user-selected design. The exact prompt ledger records
one reference for every selected Wren raw and supplies a reproducible exclusion
check.

## Soren source exclusion

All eight selected Soren calls use only `identity_base`, `facial_zoom`,
`anatomy`, and `back_view` from the repaired locked directory. The prohibited
original/top-level source and prohibited `active_profile` were not passed. The
separate Soren provenance record hashes both allowed and excluded files so the
absence is testable rather than implied.

## One-star veiled defaults

The contract is deliberate and consistent obscuration, not a literal Gaussian
blur request. Masculine and feminine neutral anchors specify a soft smoky dark
veil covering the whole face with no readable features. Their seven variants
use the corresponding neutral output as a same-generic-design reference and
communicate emotion through body language. Prompts forbid glowing eyes, masks,
hoods, censor rectangles, horror traits, props, heraldry, and seeded-character
marks. This avoids the conflict between an unrevealed identity and a facial
expression sweep.

## Pose plus expression

Each label coordinates face, gaze, head angle, shoulders, hands, weight, and
prop relationship. `tense` uses contracted anxious vigilance; `angry` commits
forward; `concerned` reaches outward; `sad` collapses inward; `skeptical` uses an
asymmetric appraisal; and `surprised` recoils. The model is never asked to keep
one rigid pose while merely changing facial features.

Cross-cast review also prevents a different prompt failure: repeating the same
stock palm-up gesture for every skeptical sprite. Wren and Aveline may use that
gesture, while other characters use crossed arms, chin touch, hip-set weight,
mirror appraisal, cuff adjustment, planted-sword posture, or nonhuman limb/crown
language.

## Props and targeted retries

Prompts explicitly count each canonical prop, but counting language is not
sufficient proof. Pilot and full-sheet inspection rejected real geometry
failures—Halcyon's sad hand, Iselle's extra arm, Renna's malformed bow pilots,
and Warden's unattached surprised chain—even when the prose had requested the
correct count. Each retry changed one discriminating defect and preserved its
failed predecessor.

Renna's final eight bows were accepted after full-resolution comparison with
the locked profile; perspective-driven riser/limb differences were not
mislabelled as identity failures. Warden surprised v2 visibly attaches its one
chain, small bead, and large orb to a raised hand.

## Chroma versus intentional subject color

The generation prompt asks for one uniform opaque magenta background. It does
not also demand transparent alpha. Alpha is produced deterministically after
generation.

A global physical-magenta unmix cleaned Wren's edge fringe, but applying it to
every character would conflict with intentional burgundy, purple, copper, red
eyes, pink ties, and translucent mauve wings. Processing is therefore selected
by subject-color risk:

- translucent/color-critical Iselle uses a connected-key method that preserves
  opaque interior RGB and treats wing alpha separately;
- opaque red/purple characters use a bounded opaque-hybrid: clean physical
  alpha and partial-edge RGB, exact raw RGB in opaque/high-alpha interiors;
- every accepted sheet is reviewed on alternating dark and light panels.

Superseded matte passes are preserved. A clean silhouette is not accepted if it
silently changes canonical colors.

## Exact prompt provenance and scope

`exact_generation_prompt_ledger.jsonl` reconstructs the actual built-in call
strings and reference paths from their rollout records, then pairs them with the
immediately landed raw files and hashes both sides. It is evidence of what the
model received, not a rewritten “representative prompt.”

No prompt contains or creates runtime image-reading machinery. The sweep and
its processing scripts remain authoring-time experiments and are not approved
or locked production assets.
