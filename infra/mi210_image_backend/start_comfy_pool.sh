#!/usr/bin/env bash
set -euo pipefail

ROOT="${AYOA_COMFY_ROOT:-/home/nod/.local/share/ayoa-anima-eval}"
IMAGE="${AYOA_COMFY_IMAGE:-ayoa-anima:comfy}"
MASTER_NAME="${AYOA_COMFY_MASTER_CONTAINER:-ayoa-comfy-master-v2}"
MASTER_PORT="${AYOA_COMFY_MASTER_PORT:-8194}"
WORKER_PREFIX="${AYOA_COMFY_WORKER_PREFIX:-ayoa-comfy-v3-gpu}"
WORKER_PORT_BASE="${AYOA_COMFY_WORKER_PORT_BASE:-8210}"
GPU_COUNT="${AYOA_COMFY_GPU_COUNT:-4}"

if [[ ! -d "$ROOT/ComfyUI/custom_nodes/ComfyUI-Distributed" ]]; then
    echo "ComfyUI-Distributed is not installed under $ROOT/ComfyUI/custom_nodes" >&2
    exit 1
fi

for path in \
    "$ROOT/ComfyUI/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors" \
    "$ROOT/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "$ROOT/ComfyUI/models/vae/qwen_image_vae.safetensors"
do
    if [[ ! -s "$path" ]]; then
        echo "required Qwen model file is missing: $path" >&2
        exit 1
    fi
done

container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

start_existing_or_run() {
    local name="$1"
    shift
    if container_running "$name"; then
        echo "already running container=$name"
        return
    fi
    if docker inspect "$name" >/dev/null 2>&1; then
        docker start "$name" >/dev/null
        echo "started existing container=$name"
        return
    fi
    docker run -d "$@" >/dev/null
    echo "created container=$name"
}

mkdir -p \
    "$ROOT/comfy-user-master-v2" \
    "$ROOT/comfy-temp-master-v2" \
    "$ROOT/logs"

start_existing_or_run "$MASTER_NAME" \
    --name "$MASTER_NAME" \
    --restart unless-stopped \
    --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add 44 \
    --group-add 110 \
    --ipc=host \
    --ulimit memlock=-1 \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=0 \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/hf \
    -v "$ROOT:/work" \
    -v /home/nod/.cache/ayoa-huggingface:/hf \
    -w /work/ComfyUI \
    "$IMAGE" \
    python main.py \
        --listen 127.0.0.1 \
        --port "$MASTER_PORT" \
        --enable-cors-header \
        --disable-auto-launch \
        --preview-method none \
        --highvram \
        --user-directory /work/comfy-user-master-v2 \
        --temp-directory /work/comfy-temp-master-v2 \
        --output-directory /work/ComfyUI/output \
        --database-url sqlite:////work/comfy-user-master-v2/comfyui.db

for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    name="${WORKER_PREFIX}${gpu}"
    port=$((WORKER_PORT_BASE + gpu))
    mkdir -p \
        "$ROOT/comfy-user-v3-gpu${gpu}" \
        "$ROOT/comfy-temp-v3-gpu${gpu}"
    start_existing_or_run "$name" \
        --name "$name" \
        --restart unless-stopped \
        --network host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add 44 \
        --group-add 110 \
        --ipc=host \
        --ulimit memlock=-1 \
        --security-opt seccomp=unconfined \
        -e "HIP_VISIBLE_DEVICES=${gpu}" \
        -e COMFYUI_IS_WORKER=1 \
        -e PYTHONUNBUFFERED=1 \
        -e HF_HOME=/hf \
        -v "$ROOT:/work" \
        -v /home/nod/.cache/ayoa-huggingface:/hf \
        -w /work/ComfyUI \
        "$IMAGE" \
        python main.py \
            --listen 127.0.0.1 \
            --port "$port" \
            --enable-cors-header \
            --disable-auto-launch \
            --preview-method none \
            --highvram \
            --user-directory "/work/comfy-user-v3-gpu${gpu}" \
            --temp-directory "/work/comfy-temp-v3-gpu${gpu}" \
            --output-directory /work/ComfyUI/output \
            --database-url \
            "sqlite:////work/comfy-user-v3-gpu${gpu}/comfyui.db"
done

for ((offset = -1; offset < GPU_COUNT; offset++)); do
    if ((offset == -1)); then
        port="$MASTER_PORT"
    else
        port=$((WORKER_PORT_BASE + offset))
    fi
    ready=0
    for _ in $(seq 1 120); do
        if curl -fsS \
            "http://127.0.0.1:${port}/distributed/local_log" \
            >/dev/null 2>&1
        then
            ready=1
            echo "ready port=$port"
            break
        fi
        sleep 2
    done
    if [[ "$ready" != 1 ]]; then
        echo "Comfy service did not become ready on port $port" >&2
        exit 1
    fi
done

MASTER_PORT="$MASTER_PORT" \
WORKER_PORT_BASE="$WORKER_PORT_BASE" \
GPU_COUNT="$GPU_COUNT" \
python3 - <<'PY'
import json
import os
import urllib.request

master_port = int(os.environ["MASTER_PORT"])
worker_port_base = int(os.environ["WORKER_PORT_BASE"])
gpu_count = int(os.environ["GPU_COUNT"])
base = f"http://127.0.0.1:{master_port}"
workers = [
    {
        "id": f"mi210-gpu{gpu}",
        "name": f"MI210 GPU {gpu}",
        "host": "127.0.0.1",
        "port": worker_port_base + gpu,
        "cuda_device": gpu,
        "enabled": True,
        "extra_args": "",
        "type": "remote",
    }
    for gpu in range(gpu_count)
]
payload = {
    "master": {
        "name": "MI210 Comfy coordinator",
        "host": f"127.0.0.1:{master_port}",
        "port": master_port,
        "cuda_device": 0,
        "extra_args": "",
    },
    "workers": workers,
    "settings": {
        "debug": False,
        "auto_launch_workers": False,
        "stop_workers_on_master_exit": False,
        "master_delegate_only": True,
        "websocket_orchestration": False,
        "worker_probe_concurrency": 8,
        "worker_prep_concurrency": 4,
        "media_sync_concurrency": 4,
        "media_sync_timeout_seconds": 300,
        "worker_timeout_seconds": 1800,
    },
}
request = urllib.request.Request(
    f"{base}/distributed/config",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=30) as response:
    result = json.load(response)
if result.get("status") != "success":
    raise SystemExit(f"could not configure ComfyUI-Distributed: {result}")
print(f"configured {len(workers)} distributed workers")
PY
