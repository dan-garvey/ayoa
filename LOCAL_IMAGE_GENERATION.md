# Local Image Generation

Ayoa can generate output-only illustrations selected by an image-director
sidecar. Every finalized canonical event with a human observer is projected
through that observer's fact visibility; equivalent projections share one
director call and artifact while retaining independent private deliveries.
The director may request zero or more illustrations. There is no engine-owned
cadence or fallback prompt.

Director and diffusion work start at event closure and overlap later routing,
character-agent work, and narration. Narration is only a delivery barrier:
an artifact is released to a POV after the source transaction commits and
prose containing that event has reached that POV. Neither image task blocks
the story loop.

The selected deployment is unadapted
[FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) at pinned
revision `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`, hosted on four MI210
GPUs and reached only through an SSH loopback tunnel. Runtime sampling uses
20 inference steps and guidance 4. The reviewed story-level visual prompt
provides the manhwa treatment; no LoRA or trigger is loaded. The broad
validation sweep found little clean style gain from the private adapter,
substantial drift at high strengths, and about a 3.9x latency penalty from
unfused LoRA execution.

FLUX.2-dev uses the FLUX Dev Non-Commercial License. Keep generated artifacts
private unless a separate rights review authorizes another use. The retired
local Klein Base 4B path remains available as an explicit fallback; it is not
selected silently.

Generated images are AI-created, noncanonical presentation. They are never
sent to the router, narrator, character agents, rules adapters, or any vision
model. Imported maps and handouts still use their separate reviewed-asset
pipeline.

## Prompt authoring

Runtime prompts follow the
[official FLUX.2 prompting guide](https://docs.bfl.ml/guides/prompting_guide_flux2):
put the main subject and action first, then identity anchors, style, setting,
camera/composition, lighting, and secondary detail. The director targets
20-60 words for an ordinary scene direction; longer group or action directions
must earn their extra spatial detail. Diffusion prompts describe the desired
visible result with affirmative language because FLUX.2 has no negative-prompt
channel.

Finished character images are identity inputs, not neutral inspiration boards.
Use them when preserving or deliberately adapting that character. Do not ask a
finished character reference to contribute only abstract silhouette, pose, or
rarity while somehow withholding face, body, and costume—the model may transfer
all of them. For an original character, generate the first concepts from text
and the reviewed story style, select one, and then use that selected original
as the identity reference. Multi-reference scene edits must state what each
input supplies.

### Empirical character-selection findings

The One Star review pass confirmed that fixed-prompt seed variation is a useful
identity-selection method: four seeds using the same 76-80 word prompt exposed
face and silhouette choices without confounding them with prompt changes.
Subject-first text-only prompts produced the strongest original identities.
Explicit adult age, gender presentation, build, hair, face, role prop, palette,
in-world setting, lighting, and an edge-to-edge canvas target were enough for
coherent popular-fantasy characters. Proper names were omitted from diffusion
input because FLUX.2 occasionally printed them as labels.

Direct image references worked when their full identity was actually wanted.
Iselle uses reviewed source art directly, and Wren's review deliberately
compared two text-only originals against two source-identity adaptations. They
did not work as abstract mood boards: asking finished characters to contribute
only silhouette or rarity copied faces, sex presentation, bodies, and costumes
despite contrary prose.

Several visually descriptive terms were taken literally. `borderless frame`
created inset frames, `leaf-bladed` created leaves, and `winged mantle` created
feathered wings. Calling a boss image a `webtoon` also encouraged page layout
and speech bubbles. Prefer concrete material geometry (`two pointed cloth
panels`, `broad steel spearhead`, `artwork filling every edge`) and use
`full-color Korean action-fantasy illustration` when comic-page semantics are
especially risky. Human review still checks for tiny pseudo-signatures at image
edges before any selected asset is bound.

## Remote MI210 service

The diffusion service runs in a private ROCm container on the SSH-config host
alias `mi210`. Docker publishes its port only on the remote host's loopback.
Start the reviewed service, then keep a local tunnel open:

```bash
ssh mi210 \
  'bash /home/nod/.local/share/ayoa-image-server/start.sh'
ssh -N \
  -L 127.0.0.1:8199:127.0.0.1:8199 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  mi210
```

Do not publish port 8188 on a public or LAN interface. The runtime client
rejects non-loopback URLs so prompts and private diffusion references cannot
be sent to an accidentally public endpoint. SSH supplies transport encryption
and host authentication; no key or password belongs in Ayoa configuration.

Verify the tunnel before starting Ayoa:

```bash
python - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8199/health", timeout=30) as r:
    health = json.load(r)
assert health["ok"] is True
assert health["model"] == "black-forest-labs/FLUX.2-dev"
assert health["revision"] == "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
assert health["gpu_count"] == 4
assert health["pipelines"]["compose"]["available"] is True
assert health["pipelines"]["edit"]["available"] is True
print(health)
PY
```

Select the backend:

```bash
AYOA_IMAGE_DIRECTOR_ENABLED=true
AYOA_IMAGE_GENERATION_ENABLED=true
AYOA_IMAGE_WORKER_BACKEND=remote
AYOA_IMAGE_REMOTE_URL=http://127.0.0.1:8199
AYOA_IMAGE_MODEL=black-forest-labs/FLUX.2-dev
AYOA_IMAGE_MODEL_REVISION=26afe3a78bb242c0a8bb181dcc8937bb16e5c66c
AYOA_IMAGE_LORA_PATH=none
AYOA_IMAGE_STYLE_TRIGGER=
AYOA_IMAGE_STEPS=20
AYOA_IMAGE_GUIDANCE=4
AYOA_IMAGE_TIMEOUT_SECONDS=900
```

Remote preflight checks the service model and revision before the durable
queue is claimed. A missing tunnel, mismatched model, malformed response,
wrong seed, wrong dimensions, oversized body, timeout, or remote OOM becomes
a typed job failure; Ayoa never substitutes another model or drops requested
identity/location references.

References remain hash-pinned locally, then their validated bytes are carried
through the encrypted loopback tunnel to diffusion only. No image bytes,
paths, hashes, embeddings, or provider details enter an LLM prompt.

Measured warm no-adapter latency at approximately 512-square area was 19.3
seconds for 20 steps, 26.8 seconds for 28, and 38.3 seconds for 40. The first
request at a new shape may spend roughly another minute compiling ROCm
kernels. Twenty steps showed no consistent visual loss in the completed
612-image review and is the selected default. Unfused dynamic LoRA inference
measured about 3.9x slower and is not selected.

## Local WSL2 fallback prerequisites

Use the current NVIDIA Windows driver with WSL CUDA support. Do not install a
second Linux display driver inside WSL. Confirm the GPU is visible:

```bash
nvidia-smi
```

Plan for an RTX 3090 with 24 GB VRAM, at least 32 GB host RAM, and at least
30 GB free disk space for weights and generated artifacts.

## Install

Keep Torch and diffusion dependencies out of Ayoa's normal `.venv`:

```bash
.venv/bin/python -m venv .venv-image
.venv-image/bin/python -m pip install --upgrade pip
.venv-image/bin/python -m pip install -r requirements-image.txt
.venv-image/bin/python scripts/image_worker.py --preflight
.venv-image/bin/python scripts/image_worker.py --download
```

Use the checked-in environment's Python to create `.venv-image`: its
standalone Python distribution includes development headers needed by Torch's
runtime CUDA helpers. Preflight reports `python_headers_unavailable` instead
of allowing that failure to surface during the first generation.

The default worker revision is
`a3b4f4849157f664bdbc776fd7453c2783562f4d`. Change it only after testing
the replacement revision and its license.

The adapter is intentionally gitignored and must exist at:

```text
app/storage/runtime/image_generation/loras/ayoapmu2-step600.safetensors
```

Its expected SHA-256 is
`3388acd713240f34f9266cf164529ef78a43f13cb2cae7864d1db22638a1fbbc`.
Preflight fails rather than silently generating without the configured adapter.

## Enable the local Klein fallback

With `.venv-image/bin/python` present, `AYOA_IMAGE_GENERATION_ENABLED=auto`
enables the worker automatically. Explicit configuration:

```bash
AYOA_IMAGE_DIRECTOR_ENABLED=true
AYOA_IMAGE_GENERATION_ENABLED=true
AYOA_IMAGE_WORKER_BACKEND=local
AYOA_IMAGE_WORKER_PYTHON=.venv-image/bin/python
AYOA_IMAGE_MODEL=black-forest-labs/FLUX.2-klein-base-4B
AYOA_IMAGE_MODEL_REVISION=a3b4f4849157f664bdbc776fd7453c2783562f4d
AYOA_IMAGE_LORA_PATH=app/storage/runtime/image_generation/loras/ayoapmu2-step600.safetensors
AYOA_IMAGE_LORA_SHA256=3388acd713240f34f9266cf164529ef78a43f13cb2cae7864d1db22638a1fbbc
AYOA_IMAGE_LORA_STRENGTH=0.8
AYOA_IMAGE_STYLE_TRIGGER=ayoapmu2
AYOA_IMAGE_STEPS=50
AYOA_IMAGE_GUIDANCE=4
AYOA_IMAGE_CPU_OFFLOAD=false
```

Optional tuning:

```bash
AYOA_IMAGE_QUEUE_LIMIT=16
AYOA_IMAGE_SESSION_QUEUE_LIMIT=8
AYOA_IMAGE_MAX_REQUESTS=6
AYOA_IMAGE_MAX_SUBJECTS=4
AYOA_IMAGE_MAX_REFERENCES=4
AYOA_IMAGE_MAX_REFERENCE_BYTES=20000000
AYOA_IMAGE_TIMEOUT_SECONDS=300
AYOA_IMAGE_MODEL_CACHE=app/storage/runtime/image_generation/models
```

Set `AYOA_IMAGE_DIRECTOR_ENABLED=shadow` to persist director decisions without
submitting diffusion work. Set it to `false` to disable the sidecar. The
director role uses `AYOA_MODEL_IMAGE_DIRECTOR` when configured. If diffusion
dependencies, CUDA, or the worker are unavailable, text play continues and
director decisions remain durable; the engine never substitutes a generic
image request.

## Smoke test and benchmark

Neither command calls a story LLM:

```bash
.venv-image/bin/python scripts/image_worker.py --smoke --cpu-offload
.venv-image/bin/python scripts/image_worker.py --benchmark --cpu-offload --runs 20
```

Benchmark both CPU-offload and full-GPU modes before changing defaults. Record
cold model load, warm median/max generation time, peak allocated VRAM, host
RAM pressure, and output quality. There is no reliable first-party RTX 3090
latency number to substitute for this local measurement.

The repository default is full-GPU mode. On the target RTX 3090 with Torch
2.13.0+cu129, Base 4B plus the selected adapter measured roughly 36–53 seconds
for 512×512 at 50 steps and about 41 seconds for 768×512. Earlier four-second
measurements applied to the retired distilled four-step model and must not be
used for capacity estimates.

CPU offload remains available for a machine that needs more VRAM headroom.

If FLUX.2 Klein is unstable on the installed CUDA stack, SDXL 1.0 is the
documented fallback because its ControlNet, IP-Adapter, LoRA, and inpainting
ecosystem is mature. A fallback must be configured deliberately; Ayoa never
silently switches models or lowers dimensions after an error.

## Runtime data and recovery

Private jobs, prompts, model cache, temporary files, and content-addressed
WebP artifacts live under:

```text
app/storage/runtime/image_generation/
```

This directory and `.venv-image/` are gitignored. Job storage is created with
private permissions where the filesystem supports them. Logs contain opaque
job IDs, timings, hashes, queue depth, and typed errors—not prompt prose.

On restart, interrupted jobs return to the queue. Rewind and lineage
reconciliation cancel speculative director runs, generation jobs, candidates,
and undelivered destinations whose source event is no longer canonical.
Malformed, animated, wrong-sized, hash-mismatched, or over-8-MB output is
rejected. No content classifier or human approval step is applied to generated
illustrations.

The private image-job store is pre-release schema v5. Opening an older or
intermediate layout retires that disposable queue directly instead of
migrating it. Checkpoint-owned reviewed bindings re-register on session load.

Multiple Ayoa processes share one durable queue. An OS lease elects exactly one
GPU owner and another process takes over automatically if it exits; other
Discord/CLI processes may enqueue and observe jobs through SQLite. Delivery has
its own expiring lease and job-specific receipt. GPU generation and network
delivery use separate workers, so a slow Discord upload cannot idle diffusion.

Successful first individual portraits become provisional identity references.
They are presentation-only and may be used immediately by later diffusion
requests. `/image lock` accepts the candidate; `/image reroll` keeps the old
reference active until its replacement succeeds. Reference files are frozen by
path, byte count, dimensions, MIME type, and SHA-256. A pipeline that cannot
consume requested references fails with `reference_inputs_unsupported`; it
never silently drops them.

## Authored visual references

A story author may select already-reviewed character identity and stable
location/style images before play. This is a diffusion-only path: no source
image, path, hash, dimensions, embedding, or image-derived analysis is sent to
any LLM. The image director sees only opaque reference ids, their visible
character or location applicability, and concise human-authored selection
hints. It chooses an ordered subset for each request; only those frozen files
reach diffusion. Other model roles receive no reference metadata.

1. Manually review the source and its rights, then place a static PNG, JPEG, or
   WebP under the story-private directory:

   ```text
   app/storage/stories/<story_id>/visual-references/
   ```

   One Star Ascension tracks its authored identity files in that directory
   beside the seed. Other stories may still provision reviewed files privately.
   Do not point outside the directory or use a symlink. Animated images are
   rejected. A registry may contain at most 128 files, each at most 20 MB,
   8192 pixels on either edge, and 40 megapixels; declared bytes may total at
   most 256 MB.

2. Compute exact metadata locally. This reads the file with Pillow; it does not
   call an LLM:

   ```bash
   .venv/bin/python - <<'PY'
   import hashlib, json
   from pathlib import Path
   from PIL import Image

   path = Path("app/storage/stories/<story_id>/visual-references/<file>")
   data = path.read_bytes()
   with Image.open(path) as image:
       print(json.dumps({
           "storage_ref": path.name,
           "mime_type": Image.MIME[image.format],
           "width": image.width,
           "height": image.height,
           "byte_count": len(data),
           "sha256": hashlib.sha256(data).hexdigest(),
       }, indent=2))
   PY
   ```

3. Add the result to the seed checkpoint's
   `reviewed_visual_references`. Give it a stable opaque `reference_id`, set
   `purpose` to `identity`, `environment`, or `style`, set `scope` to
   `character` or `location`, set `scope_id` to the owning character id or
   location label, and write a public `selection_hint` describing useful
   framing (for example face close-up, rear view, or environment scale). Set
   `diffusion_authorized: true` only after review. Do not use the
   engine-reserved `imgref_` prefix. Identity purpose requires character scope;
   environment/style require location scope.

4. Select the default character anchor by putting one owned opaque id in the
   existing `characters[].visuals.identity_reference_id`. Every authorized
   identity reference with the same character `scope_id` becomes a selectable
   supporting view. Select location references with
   `location_visual_reference_ids`, whose keys match each reference's
   `scope_id` and whose values are ordered reference-id lists. Do not add a
   second identity field.

5. Validate the seed before play:

   ```bash
   .venv/bin/python - <<'PY'
   from pathlib import Path
   from app.engine.reviewed_visual_references import validate_story_visual_references
   from app.schemas.checkpoint import CheckpointFile

   story = Path("app/storage/stories/<story_id>")
   checkpoint = CheckpointFile.model_validate_json(
       (story / "ckpt_0000.json").read_text()
   )
   validate_story_visual_references(checkpoint, story_dir=story)
   print("reviewed visual references valid")
   PY
   ```

Story start repeats full validation and copies selected inputs into private
content-addressed runtime storage. Session loads recheck that immutable copy.
A missing, changed, wrong-format, wrong-MIME, wrong-sized, or hash-mismatched
selected input is a loud load failure. Checkpoints retain private metadata and
opaque bindings, never asset bytes. Culling suppresses a binding for that
session without deleting shared bytes; rewind restores it. Authored identities
are locked by definition and produce no `/image lock` reminder. Their
`/image reroll` creates a generated provisional replacement while the authored
reference remains available as fallback.

To disable generation while keeping artifacts:

```bash
AYOA_IMAGE_GENERATION_ENABLED=false
```

To reclaim local disk space while Ayoa is stopped, remove generated artifacts
or the model cache under the runtime directory. The next requested model
download recreates the cache.
