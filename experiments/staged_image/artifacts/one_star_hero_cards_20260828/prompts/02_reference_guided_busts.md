# Built-in image generation prompts: reference-guided portrait candidates

These are the exact prompts submitted to the built-in image-generation tool.
The input order and role for each request are recorded below. Generated output
was used only as an experiment candidate and was subsequently matted with a
frozen BiRefNet mask before deterministic card compositing.

## Renna Holt

Input roles, in order:

1. Existing neutral VN sprite: identity, silhouette, and outfit anchor.
2. Reviewed facial reference: facial identity anchor.
3. Reviewed anatomy reference: build and outfit anchor.

```text
Use case: identity-preserve
Asset type: transparent waist-up hero-card portrait candidate
Primary request: Generate an original front-facing to slight three-quarter waist-up bust of Renna Holt, preserving her reviewed identity and worn practical archer clothing from the inputs. Neutral, alert expression; shoulders relaxed; no action pose.
Input images: Image 1 is the primary identity/outfit anchor and existing neutral VN sprite. Image 2 is a reviewed facial identity reference. Image 3 is a reviewed anatomy/outfit reference. They are references, not edit targets.
Scene/backdrop: genuinely transparent background; isolated single subject.
Subject: the same adult red-haired woman with shoulder-length copper hair, gray high-neck tunic, worn leather bracer and belt details; crop from mid-torso upward; hands and bow may remain outside crop.
Style/medium: polished semi-realistic illustrated visual-novel character art matching the input sprites.
Composition/framing: centered waist-up portrait, eye-level, generous margin above hair and around shoulders, no crop through head.
Lighting/mood: soft neutral studio light, calm and readable at small card size.
Constraints: preserve identity, face shape, hair color/style, age, build, clothing palette and wear; one subject only; actual RGBA transparency; no name, text, stars, logo, emblem, frame, background scene, weapon crossing the face, extra limbs, watermark, or signature.
Avoid: glamour redesign, armor upgrade, modern clothing, smile, exaggerated pose, invented jewelry, identity drift, checkerboard background.
```

## Warden of the Eighth

Input roles, in order:

1. Existing neutral VN sprite: identity, silhouette, and anatomy anchor.
2. Reviewed identity reference: material and full-design anchor.
3. Reviewed anatomy reference: duplicate approved anatomy anchor.

```text
Use case: identity-preserve
Asset type: transparent upper-body hero-card portrait candidate
Primary request: Generate an original front-facing upper-body portrait of the Warden of the Eighth, preserving the exact reviewed nonhuman construct identity from the inputs. It must remain a broad four-limbed black-and-indigo machine with a ribbed oval torso, cracked pale central shell, radial iron crown, single red lens, and hanging chained weight.
Input images: Image 1 is the primary transparent neutral VN sprite and silhouette anchor. Image 2 is the reviewed identity/environment reference. Image 3 is a reviewed anatomy reference. They are references, not edit targets.
Scene/backdrop: genuinely transparent background; isolated single construct.
Subject: the same non-social patterned hazard; crop wide enough to retain its crown, red lens, torso shell, upper jointed limbs, and the chained weight; no humanoid head or face.
Style/medium: polished semi-realistic illustrated visual-novel construct art matching the input sprite.
Composition/framing: centered upper-body portrait, front-facing, symmetrical visual weight, generous margin around the crown and limbs.
Lighting/mood: cool controlled studio light with restrained red lens glow, readable at small card size.
Constraints: preserve nonhuman identity, four-limbed anatomy, materials, colors and proportions; one subject only; actual RGBA transparency; no person, human face, eyes beyond the single lens, mouth, name, text, stars, logo, emblem, frame, background scene, extra limbs, watermark, or signature.
Avoid: humanoid robot redesign, helmeted knight, creature face, speaking expression, friendly mascot styling, checkerboard background.
```

The Warden is included only as a nonhuman silhouette/layout stress test. It is
not represented as a summonable Hero or as a canonical five-member party.

## Halcyon of the Gilded March

Input roles, in order:

1. Existing neutral VN sprite: identity, silhouette, and outfit anchor.
2. Reviewed facial reference: facial identity anchor.
3. Reviewed anatomy reference: build and outfit anchor.

```text
Use case: identity-preserve
Asset type: transparent waist-up hero-card portrait candidate
Primary request: Generate an original front-facing to slight three-quarter waist-up bust of Halcyon of the Gilded March, preserving his reviewed identity, white hair, amber-gold eyes, warm brown skin, white-and-gold armor, blue cloth, and turquoise accents from the inputs. Neutral, self-possessed expression.
Input images: Image 1 is the primary identity/outfit anchor and existing neutral VN sprite. Image 2 is a reviewed facial identity reference. Image 3 is a reviewed anatomy/outfit reference. They are references, not edit targets.
Scene/backdrop: genuinely transparent background; isolated single subject.
Subject: the same adult man, long white hair partly tied in a high bun, elegant white mantle over sculpted gold armor; crop from mid-torso upward; staff and banner may remain outside crop.
Style/medium: polished semi-realistic illustrated visual-novel character art matching the input sprites.
Composition/framing: centered waist-up portrait, eye-level, generous margin above hair and around shoulders, no crop through head.
Lighting/mood: soft neutral studio light with restrained warm highlights, readable at small card size.
Constraints: preserve identity, face shape, skin tone, hair, age, build, armor and clothing palette; one subject only; actual RGBA transparency; no name, text, stars, logo, emblem, frame, background scene, weapon crossing the face, extra limbs, watermark, or signature.
Avoid: pale skin, blond hair, crown, glamour redesign, bulky armor, smile, exaggerated pose, identity drift, checkerboard background.
```

## Veiled feminine default

Input roles, in order:

1. Existing reviewed neutral generic: sole identity-veiling and clothing anchor.

No independent locked portrait exists for this default. Method 2 therefore
uses the same reviewed neutral generic asset; it is not a distinct source.

```text
Use case: identity-preserve
Asset type: transparent waist-up veiled-default hero-card portrait candidate
Primary request: Generate an original waist-up portrait derived from the generic veiled feminine VN sprite while preserving its deliberate identity concealment. The face must remain completely featureless beneath a soft matte charcoal-black veil/shadow, with no eyes, mouth, nose, skin detail, or implied identity.
Input images: Image 1 is the sole locked generic identity-veiling and clothing reference. It is a reference, not an edit target.
Scene/backdrop: genuinely transparent background; isolated single generic figure.
Subject: the same ordinary feminine-presenting generic figure with shoulder-length dark hair, worn charcoal tunic and simple belt; calm neutral stance; crop from mid-torso upward.
Style/medium: polished semi-realistic illustrated visual-novel character art matching Image 1.
Composition/framing: centered waist-up portrait, eye-level, generous margin around hair and shoulders.
Lighting/mood: subdued neutral studio light; facial veil remains opaque and unreadable.
Constraints: preserve exact identity veiling; one subject only; actual RGBA transparency; absolutely no facial features or skin visible on the face; no name, text, stars, logo, emblem, frame, background scene, fantasy class markers, weapon, watermark, or signature.
Avoid: revealing or inventing a face, glowing eyes, mask ornament, hood, glamour redesign, identity cues, checkerboard background.
```

## Veiled masculine default

Input roles, in order:

1. Existing reviewed neutral generic: sole identity-veiling and clothing anchor.

No independent locked portrait exists for this default. Method 2 therefore
uses the same reviewed neutral generic asset; it is not a distinct source.

```text
Use case: identity-preserve
Asset type: transparent waist-up veiled-default hero-card portrait candidate
Primary request: Generate an original waist-up portrait derived from the generic veiled masculine VN sprite while preserving its deliberate identity concealment. The face must remain completely featureless beneath a soft matte charcoal-black veil/shadow, with no eyes, mouth, nose, skin detail, or implied identity.
Input images: Image 1 is the sole locked generic identity-veiling and clothing reference. It is a reference, not an edit target.
Scene/backdrop: genuinely transparent background; isolated single generic figure.
Subject: the same ordinary masculine-presenting generic figure with tousled dark hair, worn charcoal tunic, wrapped forearms and simple belt; calm neutral stance; crop from mid-torso upward.
Style/medium: polished semi-realistic illustrated visual-novel character art matching Image 1.
Composition/framing: centered waist-up portrait, eye-level, generous margin around hair and shoulders.
Lighting/mood: subdued neutral studio light; facial veil remains opaque and unreadable.
Constraints: preserve exact identity veiling; one subject only; actual RGBA transparency; absolutely no facial features or skin visible on the face; no name, text, stars, logo, emblem, frame, background scene, fantasy class markers, weapon, watermark, or signature.
Avoid: revealing or inventing a face, glowing eyes, mask ornament, hood, glamour redesign, identity cues, checkerboard background.
```

## Shared generation limitation

The image-generation tool returned RGB PNGs with a rendered checkerboard for
all five portrait requests even though the prompts requested genuine alpha.
Those raw RGB files are preserved as generation results but are rejected as
final overlays. Each accepted comparison candidate is the same generated RGB
content combined with its recorded BiRefNet mask.
