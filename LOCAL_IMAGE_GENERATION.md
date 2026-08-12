# Local Image Generation

Ayoa can generate one output-only illustration for the acting player's POV
after each eligible completed beat. Text is delivered first. Generation runs
in an isolated local subprocess, one request at a time, and never changes
canonical events or checkpoint story state.

The default model is
[FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
at a pinned Hugging Face revision. Its weights are Apache-2.0 and may be used
commercially. The model is more precisely described as open-weight unless its
complete training data and training artifacts are also available. Black Forest
Labs documents RTX 3090 support and about 13 GB VRAM with model CPU offload.
The full Diffusers snapshot is approximately 24 GB.

Generated images are AI-created, noncanonical presentation. They are never
sent to the router, narrator, character agents, rules adapters, or any vision
model. Imported maps and handouts still use their separate reviewed-asset
pipeline.

## WSL2 prerequisites

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
`5e67da950fce4a097bc150c22958a05716994cea`. Change it only after testing
the replacement revision and its license.

## Enable

With `.venv-image/bin/python` present, `AYOA_IMAGE_GENERATION_ENABLED=auto`
enables the worker automatically. Explicit configuration:

```bash
AYOA_IMAGE_GENERATION_ENABLED=true
AYOA_IMAGE_WORKER_PYTHON=.venv-image/bin/python
AYOA_IMAGE_MODEL=black-forest-labs/FLUX.2-klein-4B
AYOA_IMAGE_MODEL_REVISION=5e67da950fce4a097bc150c22958a05716994cea
AYOA_IMAGE_CPU_OFFLOAD=false
```

Optional tuning:

```bash
AYOA_IMAGE_QUEUE_LIMIT=16
AYOA_IMAGE_TIMEOUT_SECONDS=300
AYOA_IMAGE_CLI_WAIT_SECONDS=120
AYOA_IMAGE_WIDTH=1024
AYOA_IMAGE_HEIGHT=1024
AYOA_IMAGE_MODEL_CACHE=app/storage/runtime/image_generation/models
```

Per session, `/settings set` (or the CLI settings command) exposes:

- `image_generation_mode`: `actor` or `off`
- `image_generation_every_n_beats`: positive cadence; default `1`

The global capability switch wins over session settings. If dependencies,
CUDA, or the worker are unavailable, text play continues normally and a
single startup message is logged.

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

The repository default is full-GPU mode because the target host was measured
on 2026-08-12 with Torch 2.13.0+cu129 and the pinned model revision:

- CPU offload, 20 runs at 1024×1024: 18.81 s median, 16.89 s minimum,
  25.65 s maximum, and 7.81 GiB peak Torch allocation.
- Full GPU, fixed-seed 1024×1024 smoke: 4.32 s generation, 6.93 s cold load,
  and 17.32 GiB peak Torch allocation.

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

On restart, interrupted jobs return to the queue. Rewind cancels later jobs,
checks the source checkpoint hash before generation and delivery, and removes
already delivered Discord attachments through existing turn-message cleanup.
Malformed, animated, wrong-sized, hash-mismatched, or over-8-MB output is
rejected. No content classifier or human approval step is applied to generated
illustrations.

Multiple Ayoa processes share one durable queue. An OS lease elects exactly one
GPU owner and another process takes over automatically if it exits; other
Discord/CLI processes may enqueue and observe jobs through SQLite. Delivery has
its own expiring lease and job-specific receipt. If a rewind races an upload,
the attachment is removed immediately; failed removals enter a durable cleanup
outbox retried by the bot every minute.

To disable generation while keeping artifacts:

```bash
AYOA_IMAGE_GENERATION_ENABLED=false
```

To reclaim local disk space while Ayoa is stopped, remove generated artifacts
or the model cache under the runtime directory. The next requested model
download recreates the cache.
