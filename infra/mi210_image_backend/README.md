# MI210 image backend

This deployment exposes one loopback-only image gateway while preserving two
different execution shapes:

- FLUX.2-dev uses the existing BF16 service sharded across all four MI210s.
- Qwen Image Edit uses four independent ComfyUI workers, one per MI210.
- ComfyUI-Distributed provides one coordinator UI and supports a different
  prompt and seed on every worker.

The FastAPI gateway serializes model-family transitions. A FLUX request waits
for Comfy jobs to finish, unloads Comfy weights, starts the sharded service,
and proxies the request. A Qwen API request stops FLUX before choosing the
least-busy Comfy worker. Up to four Qwen batch items run concurrently.

## Remote ports

All services listen only on the MI210 host loopback.

| Port | Service |
| --- | --- |
| 8188 | On-demand sharded FLUX.2-dev service |
| 8194 | ComfyUI-Distributed coordinator |
| 8199 | Ayoa FastAPI gateway |
| 8210-8213 | Comfy workers for GPUs 0-3 |

## Deploy

From the Ayoa checkout:

```bash
scp infra/mi210_image_backend/gateway.py \
  mi210:/home/nod/.local/share/ayoa-image-server/gateway.py
scp infra/mi210_image_backend/start_comfy_pool.sh \
  mi210:/home/nod/.local/share/ayoa-anima-eval/scripts/start-comfy-pool.sh
scp infra/mi210_image_backend/ayoa-image-gateway.service \
  mi210:/home/nod/.config/systemd/user/ayoa-image-gateway.service

ssh mi210 \
  'chmod +x /home/nod/.local/share/ayoa-anima-eval/scripts/start-comfy-pool.sh &&
   /home/nod/.local/share/ayoa-anima-eval/scripts/start-comfy-pool.sh &&
   systemctl --user daemon-reload &&
   systemctl --user enable --now ayoa-image-gateway.service'
```

The Comfy containers use `restart=unless-stopped`. The user service starts on
login. Enabling it before login after a host reboot requires an administrator
to enable lingering for the `nod` account.

## Local tunnels

```bash
ssh -N \
  -L 127.0.0.1:8189:127.0.0.1:8194 \
  -L 127.0.0.1:8199:127.0.0.1:8199 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  mi210
```

Open `http://127.0.0.1:8199/comfy` to switch out of FLUX mode and enter the
coordinator UI. Import
`workflows/qwen_image_edit_distributed_api.json`, choose the input image, and
edit `Distributed Value` to set one prompt per worker. `Distributed Seed`
offsets the base seed for each worker.

Do not submit a direct Comfy job while an Ayoa FLUX request is actively
generating. Gateway-originated API requests coordinate correctly, but a
browser connected directly to Comfy cannot participate in the gateway lock.

## API

The gateway preserves Ayoa's existing remote-worker contract:

- `GET /health`
- `POST /generate`
- `POST /img2img`

Additional endpoints:

- `GET /workers`
- `GET /comfy`
- `POST /mode/flux`
- `POST /mode/qwen`
- `POST /generate/flux`
- `POST /img2img/flux`
- `POST /edit/qwen`
- `POST /edit/qwen/batch`

`GET /health` advertises the model-neutral runtime modes under `pipelines`:
`compose` maps to FLUX text/reference generation and accepts up to four
references; `edit` maps to Qwen Image Edit and accepts one to three ordered
references. FLUX `/img2img` remains available for manual experiments but is
not an Ayoa image-director mode.

Configure Ayoa through the local tunnel:

```bash
AYOA_IMAGE_WORKER_BACKEND=remote
AYOA_IMAGE_REMOTE_URL=http://127.0.0.1:8199
AYOA_IMAGE_MODEL=black-forest-labs/FLUX.2-dev
AYOA_IMAGE_MODEL_REVISION=26afe3a78bb242c0a8bb181dcc8937bb16e5c66c
```

The MI210 is CDNA2 and has no native FP8 matrix support. The Qwen FP8-mixed
weights are used for capacity, not as a claim of FP8 acceleration. Do not set
`PYTORCH_ALLOC_CONF=expandable_segments:True` for these workers: on the
current gfx90a/ROCm stack it reports free VRAM but makes allocations fail.
