# Prompt conflict audit

## Reference authority

The four locked views are complementary but not pixel-identical. Asking the
model to preserve all of them without ranking would be contradictory.

- `active_profile` is authoritative for outfit, palette, weapon ownership,
  broad silhouette, and rendering style.
- `facial_zoom` is authoritative for face, eye color, hairline, and expression
  identity. Its cropped costume details are supporting evidence, not automatic
  authority over the active profile.
- `anatomy` is authoritative only for adult body proportions. Its training
  clothes must never replace the canonical outfit.
- `back_view` is authoritative only for rear hair, scarf/coat layers, carried
  gear, and unseen costume construction.

The prompt states those roles directly. It does not ask the model to create a
collage, average four people, or place reference panels in the output.

## Mirelle weapon and costume

Mirelle's authored visual contract owns one long spear. The active profile can
visually suggest a second point at the butt, while the checkpoint describes one
leaf-bladed spear with a tassel beneath its head. The experiment resolves this
as exactly one continuous shaft, one leaf-shaped spearhead, one tassel beneath
that head, and at most an ordinary butt cap—never a second spear or second
functional head.

The facial zoom contains a feather clasp, chest strap, short upper coat, and
visible repaired stitching more strongly than the active profile. Those details
therefore entered `concerned` and `tense` despite the authority ranking. This is
recorded as continuity warning, not rationalized as a new outfit.

## Rowan weapon ownership

Rowan's public visual sheet, default loadout, active profile, and back view all
depict a signature pair of mismatched short knives. The gameplay mechanics entry
names one `Message Blade`, which is a real cross-source tension. For this visual
experiment, the locked visual references and public visual contract are the
authority: every sprite must contain exactly two owned knives, one broad utility
blade and one narrow fighting blade. No engine or checkpoint meaning was changed.

Conversational poses keep both sheathed. `tense` draws one while preserving one
visible sheathed hilt. `angry` draws both and forbids duplicate sheathed hilts.
The first-pass `sad` art keeps the count at two but fails the canonical-hilt
shape review.

## Framing and direction

“Three-quarter” can ambiguously mean a camera angle or a crop. Requiring a
three-quarter crop while also requiring complete boots and a common foot
baseline would conflict. This experiment uses one full figure in a
three-quarter view. The complete head, hands, weapon details, and boots remain
inside a portrait canvas; the lower 25 percent is deliberately safe for a VN
dialogue box to cover. Native generated pixels are placed without resampling on
a common 1100 x 1500 canvas with a shared raw-canvas bottom.

Mirelle consistently faces screen-left for a right-side slot. Rowan consistently
faces screen-right for a left-side slot. Downcast `sad` gazes are allowed without
changing the body orientation.

## Expression versus pose

No prompt asks for a neutral body while separately demanding a strong emotion.
Each core label defines face, head angle, shoulders, hands, weight distribution,
and weapon relationship together. `tense` is defensive; `angry` is forward and
aggressive. `concerned` reaches toward another person; `sad` collapses inward.
`skeptical` is deliberately asymmetric. This avoids eight facial stickers on a
single rigid body.

## Transparency protocol

The pilot prompt requested genuine transparency, but the built-in tool returned
RGB with a checkerboard painted into the image. A single-target alpha-extraction
edit repeated the same failure. The expansion prompts therefore request only a
flat opaque magenta background; they never simultaneously ask for alpha and
chroma. A deterministic authoring-time chroma-key step derives RGBA afterward.
This visual processing is experiment tooling, not runtime LLM image reading.

## Scope boundary

The prompts contain no runtime engine, queue, checkpoint, registry, filesystem,
or provider mechanics. This pack changes no runtime prompt and binds no output
to a character. All candidates remain experimental until a human chooses which
variants, costume details, and corrections to promote.
