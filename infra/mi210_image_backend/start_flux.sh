#!/usr/bin/env bash
set -euo pipefail

ROOT="${AYOA_FLUX_ROOT:-/home/nod/.local/share/ayoa-image-server}"
CONTAINER_NAME="${AYOA_FLUX_CONTAINER:-ayoa-flux2-server}"
IDLE_TIMEOUT_SECONDS="${AYOA_FLUX_IDLE_TIMEOUT_SECONDS:-300}"

if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == "true" ]]; then
    echo "server already running container=$CONTAINER_NAME"
    exit 0
fi
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true

container_id="$(
    docker run -d --rm \
        --name "$CONTAINER_NAME" \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add 44 \
        --group-add 110 \
        --ipc=host \
        --ulimit memlock=-1 \
        --security-opt seccomp=unconfined \
        -p 127.0.0.1:8188:8188 \
        -e HIP_VISIBLE_DEVICES=0,1,2,3 \
        -e PYTHONPATH=/work/container-site-packages:/work/flux2-distributed-inference \
        -e HF_HOME=/models \
        -e HF_HUB_OFFLINE=1 \
        -e HF_ENABLE_PARALLEL_LOADING=true \
        -e HF_PARALLEL_LOADING_WORKERS=32 \
        -e AYOA_REMOTE_MODEL=black-forest-labs/FLUX.2-dev \
        -e AYOA_REMOTE_MODEL_REVISION=26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
        -e AYOA_REMOTE_MODEL_PATH=/models/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
        -e "AYOA_IMAGE_IDLE_TIMEOUT_SECONDS=$IDLE_TIMEOUT_SECONDS" \
        -e AYOA_IMAGE_GPU_ID=0 \
        -v "$ROOT:/work" \
        -v /home/nod/.cache/ayoa-huggingface:/models \
        -w /work \
        rocm/pytorch:latest \
        python -m uvicorn server:app --host 0.0.0.0 --port 8188 --workers 1
)"
printf '%s\n' "$container_id" >"$ROOT/server.container"
echo "started container=$CONTAINER_NAME id=${container_id:0:12}"
