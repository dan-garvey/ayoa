from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from app.engine.image_worker_client import (
    ImageWorkerClient,
    ImageWorkerConfig,
    ImageWorkerError,
)
from app.schemas.image_generation import (
    ImageDeliveryKind,
    ImageGenerationRequest,
    ImageTriggerKind,
)


def _request() -> ImageGenerationRequest:
    return ImageGenerationRequest(
        session_id="protocol",
        checkpoint_id="ckpt_0001",
        checkpoint_sha256="a" * 64,
        turn_index=1,
        actor_character_id="alice",
        trigger_kind=ImageTriggerKind.act,
        prompt="A rain-washed street.",
        prompt_sha256="b" * 64,
        model_id="fake/model",
        model_revision="revision",
        width=256,
        height=256,
        steps=4,
        guidance=1.0,
        seed=7,
        dedupe_key="c" * 64,
        delivery_kind=ImageDeliveryKind.cli,
    )


def _worker_script(tmp_path: Path, *, sleep_seconds: float = 0) -> Path:
    script = tmp_path / "fake_worker.py"
    script.write_text(
        f"""
import argparse
import hashlib
import json
import pathlib
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--worker", action="store_true")
parser.add_argument("--model-id")
parser.add_argument("--revision")
parser.add_argument("--cache-dir")
parser.add_argument("--cpu-offload", action="store_true")
parser.parse_args()

for raw in sys.stdin:
    request = json.loads(raw)
    if request.get("command") == "shutdown":
        break
    time.sleep({sleep_seconds})
    data = b"fake-worker-output"
    pathlib.Path(request["output_path"]).write_bytes(data)
    print(json.dumps({{
        "ok": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "mime_type": "image/webp",
        "width": request["width"],
        "height": request["height"],
        "byte_count": len(data),
        "generation_seconds": {sleep_seconds},
    }}), flush=True)
"""
    )
    return script


def _config(tmp_path: Path, script: Path, *, timeout: float = 2):
    local_model = tmp_path / "local-model"
    local_model.mkdir(exist_ok=True)
    return ImageWorkerConfig(
        enabled=True,
        python_executable=Path(sys.executable),
        worker_script=script,
        model_id=str(local_model),
        model_revision="revision",
        model_cache_dir=tmp_path / "models",
        lock_path=tmp_path / "gpu.lock",
        timeout_seconds=timeout,
        cpu_offload=True,
    )


@pytest.mark.asyncio
async def test_client_sends_bounded_contract_and_reads_one_response(tmp_path):
    script = _worker_script(tmp_path)
    client = ImageWorkerClient(_config(tmp_path, script))
    output = tmp_path / "output.webp"
    try:
        result = await client.generate(_request(), output_path=output)
    finally:
        await client.close()

    assert result.ok is True
    assert result.width == 256
    assert result.height == 256
    assert output.read_bytes() == b"fake-worker-output"


@pytest.mark.asyncio
async def test_client_timeout_terminates_worker_with_typed_error(tmp_path):
    script = _worker_script(tmp_path, sleep_seconds=1)
    client = ImageWorkerClient(_config(tmp_path, script, timeout=0.05))
    try:
        with pytest.raises(ImageWorkerError) as exc_info:
            await client.generate(_request(), output_path=tmp_path / "slow.webp")
    finally:
        await client.close()

    assert exc_info.value.code == "worker_timeout"


@pytest.mark.asyncio
async def test_gpu_lock_rejects_second_process_owner(tmp_path):
    script = _worker_script(tmp_path, sleep_seconds=0.2)
    config = _config(tmp_path, script, timeout=2)
    first = ImageWorkerClient(config)
    second = ImageWorkerClient(config)
    first_task = asyncio.create_task(
        first.generate(_request(), output_path=tmp_path / "first.webp")
    )
    try:
        await asyncio.sleep(0.05)
        with pytest.raises(ImageWorkerError) as exc_info:
            await second.generate(
                _request().model_copy(
                    update={"dedupe_key": "d" * 64, "seed": 8}
                ),
                output_path=tmp_path / "second.webp",
            )
        assert exc_info.value.code == "gpu_in_use"
        await first_task
    finally:
        await first.close()
        await second.close()
