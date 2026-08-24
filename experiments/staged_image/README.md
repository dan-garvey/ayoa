# Layered image staging prototype

The default experiment now targets a deliberately modest visual-novel result:

1. FLUX generates one clean widescreen environment plate per seed.
2. Qwen Image Edit generates exactly two reusable three-quarter-body sprite
   anchors from frozen identity references. Their left/right slots and inward
   facing directions are schema invariants, not prompt-only wishes.
3. Each small expression/pose variant edits only its own neutral sprite anchor.
   It never receives the environment, the other character, or a second identity
   image.
4. Native ComfyUI BiRefNet produces a checked RGBA asset for every anchor and
   variant.
5. Pillow reuses the exact same plate, fixed-height stage slots, and deterministic
   sprite shadow for every authored frame. The report artifact is a storyboard;
   it also links every full-resolution frame.

There is no whole-scene model pass in visual-novel mode. Lighting separation is
intentional presentation language rather than a defect to hide with generative
harmonization. The earlier six-tuning/two-holdout cinematic corpus remains at
`corpus.json` for explicit legacy comparisons, but it is no longer the default.

It is not connected to the runtime image director, queue, delivery, reference
promotion, or session state. It only reads already-frozen reference bytes from
the runtime artifact store, validates their hashes and dimensions, and writes
ignored experiment output under `app/storage/runtime/image_prototypes/`.

## Why the visual-novel pivot

The v4 staged control proved that independent components can preserve identity
and subject count, but it accepted 0/8 final cinematic composites. The remaining
failures were the expensive part of the problem: physical collision, variable
scale, prop fidelity, and shared illumination. A visual novel does not require
the model to solve those as one coherent photographed moment.

This is also an established presentation contract rather than a local invention.
Ren'Py keeps a background on its scene layer, places sprites at stable transforms,
and replaces an already displayed character by image tag/attributes when the
expression changes. Its layered-image groups treat expression attributes as
mutually exclusive variants of a stable character asset. See the official
[displaying-images](https://www.renpy.org/doc/html/displaying_images.html) and
[layered-image](https://www.renpy.org/doc/html/layeredimage.html) documentation.

The deployed editor is Qwen-Image-Edit-2511. Qwen's official release describes
2511 as improving character consistency for edits of an input portrait. That is
a better match for narrow anchor-to-expression edits than for repainting the
assembled scene. See the official
[Qwen-Image repository](https://github.com/QwenLM/Qwen-Image#qwen-image-edit-2511-for-image-editing-multiple-image-support-and-improved-consistency).

Latent fusion is therefore out of scope for this target. The simplest useful
unit is `plate + left sprite attribute + right sprite attribute`; if that cannot
pass review, a more complicated attention-coupled model is unlikely to rescue
the asset workflow economically.

## Legacy cinematic baseline

The known bad group image was not just a weak prompt: explicit references for
one character short-circuited the runtime reference resolver, so three newly
generated identities never reached the multi-character model call. Even with
all four references present, a single diffusion pass still had to solve
identity, count, pose, scale, anatomy, long props, occlusion, and lighting at
once. The staged path gives each identity an independent generation budget and
makes layout deterministic before any optional generative blending.

Research systems such as MultiDiffusion, ConsiStory, and MS-Diffusion are real
but do more than splice finished latent tensors. They coordinate regional
denoising, share internal attention/features, or add model-specific
multi-subject attention with layout guidance. Those mechanisms are not exposed
by the deployed FLUX.2/Qwen gateway, and substituting an unrelated Stable
Diffusion stack would make this a model migration rather than a controlled
pipeline comparison. This experiment therefore establishes a measurable
pixel-space baseline first. A future attention-coupled branch should compete
against the same corpus and verdict data rather than replace it on intuition.

Primary references:

- [MultiDiffusion paper](https://arxiv.org/abs/2302.08113)
- [ConsiStory paper](https://arxiv.org/abs/2402.03286)
- [MS-Diffusion implementation](https://github.com/MS-Diffusion/MS-Diffusion)
- [ComfyUI BiRefNet guide](https://docs.comfy.org/tutorials/utility/remove-background-birefnet)

## Failure-informed prompt policy

Agent-authored image prompts are treated as untrusted experimental inputs, not
expert creative direction. Prompt changes need a stated source or hypothesis,
the same four fixed seeds, one targeted protocol change, retained losers, and
manual review before another change. A lucky seed is not evidence that a prompt
works.

The `staged-arrival-20260823-v3` control exposed three concrete failures:

- all four FLUX plates invented three or four background people after the
  environment prompt asked for floor "for four separated figures";
- the authored layout made the supposedly palm-sized Iselle roughly one third
  of an adult's visible height; and
- all four whole-image Qwen diagnostics changed or removed approved identities
  and invented additional people or fairies.

Prompt protocol `official-minimal-v2` follows the resulting constraints:

- FLUX receives a positive, environment-only description beginning with
  `Empty environment plate`; no future subject, prop, count, silhouette, or
  textual negative is allowed into the plate prompt. Black Forest Labs says
  FLUX.2 does not support negative prompts, recommends describing an "empty
  scene" instead of "no people," and advises putting the most important content
  first. See the official [FLUX.2 prompting guide](https://docs.bfl.ai/guides/prompting_guide_flux2).
- Qwen component and harmonization calls use direct labeled input roles and a
  single edit target wherever possible. Qwen's repository warns that edit
  results may be unstable without prompt rewriting, and its own enhancer asks
  for direct, specific instructions plus explicit multi-image roles. This
  prototype uses deterministic authored templates instead of the image-aware
  Qwen-VL prompt rewriter because Ayoa runtime models must not inspect images.
  See the official [Qwen-Image examples](https://github.com/QwenLM/Qwen-Image#qwen-image-edit-for-image-editing-only-support-single-image-input)
  and [prompt-enhancer rules](https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py).
- Count, relative scale, placement, and depth are compositor data rather than
  prompt wishes. GenEval and T2I-CompBench independently identify count,
  position, spatial relationships, and attribute binding as persistent
  compositional failure modes. See [GenEval](https://arxiv.org/abs/2310.11513)
  and [T2I-CompBench](https://arxiv.org/abs/2307.06350).

Qwen-Image-Layered is a relevant future comparison, but not a latent-fusion
drop-in. Its official release decomposes an existing image into editable RGBA
layers and explicitly says text-to-multi-layer generation is limited. It could
compete with BiRefNet as a decomposition/matte track after deployment, using the
same corpus and verdict contract. See
[Qwen-Image-Layered](https://github.com/QwenLM/Qwen-Image-Layered).

The next integration experiment should use a dedicated image-harmonization
method rather than broader edit prose. Harmonization research defines this task
as adjusting a pasted foreground's illumination/color to the surrounding
background while preserving its structure and semantics. HarmonyTransformer is
one published example; the PhD diffusion pipeline is a closer conceptual match
for paste-then-mask/inpaint, but would require a separate model implementation.
See [Image Harmonization With Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_Image_Harmonization_With_Transformer_ICCV_2021_paper.html)
and [Paste, Inpaint and Harmonize via Denoising](https://arxiv.org/abs/2306.07596).
Any such track must be mask-bounded, retain the pixel baseline, and compete on
the existing verdict taxonomy; it must not silently replace a failed image.

## Recorded control comparison

The first two complete batches are retained under the ignored prototype output
root and fully reviewed:

| Run | Wall time | Background count | Component count | Matte gate | Final verdicts |
| --- | ---: | ---: | ---: | ---: | ---: |
| `staged-arrival-20260823-v3` | 25m 49s | 0/4 empty | 12/12 single generated adults plus 4 fixed Iselle assets | 16/16 passed | 0 pass, 12 loser |
| `staged-arrival-20260823-v4` | 16m 03s | 4/4 empty | 12/12 single generated adults plus 4 fixed Iselle assets | 16/16 passed | 0 pass, 8 loser |

v4 eliminated reviewed `identity` and `subject_count` failures. It did not pass
the quality gate: the remaining losers are caused by authored/algorithmic
composition, prop collision, variable box fitting, and foreground illumination.
If the legacy cinematic path is resumed, its next controlled changes are
layout-only sizing/collision work and a dedicated masked harmonization baseline.
Those are no longer default experiment goals.

The visual-novel pivot was then run as
`visual-novel-arrival-20260824-v1`. In 15m 14s it produced 4/4 plates, 8/8
neutral anchors, 8/8 anchor-derived variants, 16/16 valid mattes, and 12/12
frames. Manual storyboard review passed three of four seeds. The remaining seed
failed only `continuity`/`pose`: Maret's neutral anchor faced outward and Edric's
reassuring variant turned his gaze away. There were no reviewed identity,
subject-count, scale, anatomy, prop, matte, lighting, composition, style, or
expression failures. This is a successful direction change, not a perfect
automatic generator: production still needs a reviewed anchor choice, and any
future attempt to improve facing continuity should compare a bounded sprite edit
against these frozen artifacts rather than add broader prompt prose.

## Gateway prerequisites

The prototype adds two isolated endpoints; production endpoint behavior and the
`health.pipelines` contract are unchanged:

- `POST /prototype/matte/birefnet`
- `POST /prototype/edit/qwen/masked`

Download the official `birefnet.safetensors` (SHA-256
`9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154`)
from [Comfy-Org/BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet/blob/main/background_removal/birefnet.safetensors)
and place it at:

```text
/home/nod/.local/share/ayoa-anima-eval/ComfyUI/models/background_removal/birefnet.safetensors
```

The current Comfy pool bind-mounts that tree at `/work`, so every worker sees
the same file after restart. Missing model authority fails the matte stage
loudly; there is no alternate background remover or direct-generation fallback.

## Corpus and batch contract

`visual_novel_corpus.json` is the default. It contains two tuning scenes and one
untouched holdout. Every scene has exactly two subjects, one left-facing-inward
and one right-facing-inward, an all-anchor frame, and named expression/pose
variants used by later frames. `corpus.json` preserves the old six-tuning and
two-holdout cinematic stress suite. `style_packs.json` contains both rendering
languages.

Every run requires exactly four distinct seeds. The tool generates and retains
all requested tracks for all four; it has no winner-selection option. Holdouts
should remain untouched until tuning decisions are frozen.

Validate frozen inputs without calling either model:

```bash
.venv/bin/python scripts/staged_image_prototype.py run \
  --scene vn01_arrival_exchange \
  --dry-run
```

Run the known failure scene:

```bash
.venv/bin/python scripts/staged_image_prototype.py run \
  --scene vn01_arrival_exchange \
  --run-id visual-novel-arrival-v1
```

The default and only eligible visual-novel track is `pixel`. `masked` and
`global` are rejected because they repaint the assembled scene. To reproduce an
old cinematic comparison, explicitly pass
`--corpus experiments/staged_image/corpus.json` and opt into its desired tracks.
Use repeated `--scene` arguments for a subset; omitting them runs the two default
tuning scenes. Holdouts are refused unless `--include-holdout` is explicit.

## Human QA

There is intentionally no runtime vision model and no automatic aesthetic
winner. Append a human verdict for each track:

```bash
.venv/bin/python scripts/staged_image_prototype.py review visual-novel-arrival-v1 \
  --scene vn01_arrival_exchange \
  --seed 26082301 \
  --track pixel \
  --verdict loser \
  --reason continuity \
  --reason expression \
  --note "The expression changed, but the outfit and face also drifted."
```

Valid loser reasons include `identity`, `scale`, `pose`, `anatomy`, `prop`,
`matte`, `lighting`, `composition`, `style`, `expression`, and `continuity`.
A loser requires at least one reason. Reviews are append-only JSONL; the latest
verdict for a scene, seed, and track is the report verdict.

Build the image grid and aggregate JSON:

```bash
.venv/bin/python scripts/staged_image_prototype.py report visual-novel-arrival-v1
```

Each case manifest records the exact scene/style snapshot, source hashes,
ordered references, anchor and variant prompts, derived stage seeds, gateway
model revisions, durations, mask checks, fixed-height placements, exact reused
background hash, frame-to-variant mapping, and output hashes. These fields are
the prototype for eventual stage/attempt/verdict logging if this design is
promoted into the runtime.
