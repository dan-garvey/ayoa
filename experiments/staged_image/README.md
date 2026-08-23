# Staged image composition prototype

This experiment builds a multi-character illustration as auditable parts:

1. FLUX generates a clean environment plate with no subjects.
2. Qwen Image Edit generates one full-body component per character from one or
   two frozen identity references. The environment plate is deliberately not a
   component input: Qwen's multi-image mode is designed to fuse image content,
   while this stage must contain one character only. All independent components
   fan out across four workers. A scene may explicitly declare
   `reference_cutout` when its approved source is already a complete isolated
   full-body asset; this is fixed in the corpus and is never selected after a
   Qwen failure.
3. Native ComfyUI BiRefNet produces a foreground mask for every component.
   Coverage, edge contact, and disconnected-component checks reject bad mattes.
4. Pillow places verified RGBA components at normalized boxes and foot anchors,
   in an explicit depth order, then derives a narrow seam/contact repair mask.
5. The default outputs are the deterministic pixel composite and mask-limited
   Qwen harmonization. The mask-limited output is blended locally after
   inference, proving that pixels outside the saved repair mask are unchanged.
   A whole-image Qwen track remains available only as an opt-in diagnostic; it
   is not an eligible fallback or production candidate.

It is not connected to the runtime image director, queue, delivery, reference
promotion, or session state. It only reads already-frozen reference bytes from
the runtime artifact store, validates their hashes and dimensions, and writes
ignored experiment output under `app/storage/runtime/image_prototypes/`.

## Why this pipeline

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
The next controlled changes should therefore be layout-only sizing/collision
work and a dedicated masked harmonization baseline. Prompt prose is frozen until
those non-prompt causes have been isolated.

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

`corpus.json` contains six tuning scenes and two holdouts. It exercises the
known four-character failure, extreme scale, depth occlusion, long props,
physical contact, interlocked melee, a three-role holdout, and a genre-neutral
holdout. `style_packs.json` defines the current rendering language plus two
predeclared later comparison packs.

Every run requires exactly four distinct seeds. The tool generates and retains
all requested tracks for all four; it has no winner-selection option. Holdouts
should remain untouched until tuning decisions are frozen.

Validate frozen inputs without calling either model:

```bash
.venv/bin/python scripts/staged_image_prototype.py run \
  --scene t01_failed_arrival_hall \
  --dry-run
```

Run the known failure scene:

```bash
.venv/bin/python scripts/staged_image_prototype.py run \
  --scene t01_failed_arrival_hall \
  --run-id staged-arrival-v1
```

The default tracks are `pixel,masked`. Use `--tracks pixel` for a
component-and-compositor diagnostic that skips Qwen final editing while
retaining the same four-seed requirement. Use `--tracks pixel,masked,global`
only when explicitly measuring the unsafe whole-image diagnostic. Use repeated
`--scene` arguments for a subset; omitting them runs the six tuning scenes.
Holdouts are refused unless `--include-holdout` is explicit. After tuning is
frozen, run either holdout by naming it together with that flag. `--style-pack`
can override the scene's current style with any of the three predefined packs
without editing the corpus.

## Human QA

There is intentionally no runtime vision model and no automatic aesthetic
winner. Append a human verdict for each track:

```bash
.venv/bin/python scripts/staged_image_prototype.py review staged-arrival-v1 \
  --scene t01_failed_arrival_hall \
  --seed 26082301 \
  --track masked \
  --verdict loser \
  --reason identity \
  --reason anatomy \
  --note "Maret's face drifted and Sev's knife hand is malformed."
```

Valid loser reasons are `identity`, `subject_count`, `scale`, `pose`,
`anatomy`, `prop`, `matte`, `lighting`, `composition`, `style`,
`text_artifact`, and `other`. A loser requires at least one reason. Reviews are
append-only JSONL; the latest verdict for a scene, seed, and track is the report
verdict.

Build the image grid and aggregate JSON:

```bash
.venv/bin/python scripts/staged_image_prototype.py report staged-arrival-v1
```

Each case manifest records the exact scene/style snapshot, source hashes,
ordered references, rendered prompts and prompt hashes, derived stage seeds,
gateway model revisions, worker/prompt IDs, durations, mask checks, placements,
and output hashes. These fields are the prototype for eventual stage/attempt/
verdict logging if this design is promoted into the runtime.
