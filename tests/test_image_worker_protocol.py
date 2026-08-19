from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.engine.image_worker_client import (
    DEFAULT_LORA_SHA256,
    DEFAULT_LORA_STRENGTH,
    DEFAULT_LORA_TRIGGER,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_REMOTE_MODEL_ID,
    DEFAULT_REMOTE_MODEL_REVISION,
    DEFAULT_REMOTE_URL,
    ImageWorkerClient,
    ImageWorkerConfig,
    ImageWorkerError,
)
from app.schemas.image_generation import (
    FrozenReferenceInput,
    ImageGenerationRequest,
)
from scripts import image_worker


def _request(
    *,
    reference_inputs: list[FrozenReferenceInput] | None = None,
    generation_mode: str = "compose",
) -> ImageGenerationRequest:
    return ImageGenerationRequest(
        session_id="protocol",
        transaction_id="tx_protocol",
        source_event_id="evt_protocol",
        source_event_fingerprint="a" * 64,
        source_event_sequence=0,
        source_turn_index=1,
        request_ordinal=0,
        kind="establishing",
        generation_mode=generation_mode,
        title="Rain Street",
        subject_character_ids=[],
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
        reference_inputs=reference_inputs or [],
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
parser.add_argument("--reference-root")
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
        reference_root=tmp_path / "references",
        timeout_seconds=timeout,
        cpu_offload=True,
    )


async def _remote_server(
    *,
    model: str = DEFAULT_REMOTE_MODEL_ID,
    revision: str = DEFAULT_REMOTE_MODEL_REVISION,
    generation_status: int = 200,
):
    requests: list[dict[str, object]] = []

    async def handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_line = (await reader.readline()).decode("ascii")
        method, path, _version = request_line.strip().split()
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line == b"\r\n":
                break
            name, value = line.decode("latin-1").split(":", 1)
            headers[name.strip().lower()] = value.strip()
        body = await reader.readexactly(int(headers.get("content-length", "0")))
        if method == "GET" and path == "/health":
            response_body = json.dumps({
                "ok": True,
                "model": model,
                "revision": revision,
                "gpu_count": 4,
                "pipelines": {
                    "compose": {"available": True},
                    "edit": {"available": True},
                },
            }).encode()
            status = 200
            content_type = "application/json"
            extra_headers: dict[str, str] = {}
        else:
            requests.append(json.loads(body))
            status = generation_status
            extra_headers = {}
            if status == 200:
                request = requests[-1]
                image = BytesIO()
                if "image_base64" in request:
                    Image.new("RGB", (128, 384), color=(12, 34, 56)).save(
                        image,
                        format="PNG",
                    )
                else:
                    Image.new(
                        "RGB",
                        (int(request["width"]), int(request["height"])),
                        color=(12, 34, 56),
                    ).save(image, format="WEBP")
                response_body = image.getvalue()
                content_type = (
                    "image/png" if "image_base64" in request else "image/webp"
                )
                extra_headers = {
                    "X-Ayoa-Seed": str(request["seed"]),
                    "X-Ayoa-Generation-Seconds": "1.25",
                }
            else:
                response_body = b'{"detail":"failed"}'
                content_type = "application/json"
        reason = "OK" if status == 200 else "Error"
        response_headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(response_body)),
            "Connection": "close",
            **extra_headers,
        }
        writer.write(
            (
                f"HTTP/1.1 {status} {reason}\r\n"
                + "".join(
                    f"{name}: {value}\r\n"
                    for name, value in response_headers.items()
                )
                + "\r\n"
            ).encode("latin-1")
            + response_body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}", requests


def _remote_config(tmp_path: Path, url: str) -> ImageWorkerConfig:
    return ImageWorkerConfig(
        enabled=True,
        python_executable=tmp_path / "unused-python",
        worker_script=tmp_path / "unused-worker",
        backend="remote",
        remote_url=url,
        model_id=DEFAULT_REMOTE_MODEL_ID,
        model_revision=DEFAULT_REMOTE_MODEL_REVISION,
        reference_root=tmp_path / "references",
        timeout_seconds=2,
    )


def test_environment_defaults_to_selected_base_model_and_lora(
    tmp_path,
    monkeypatch,
):
    for name in (
        "AYOA_IMAGE_MODEL",
        "AYOA_IMAGE_MODEL_REVISION",
        "AYOA_IMAGE_LORA_PATH",
        "AYOA_IMAGE_LORA_SHA256",
        "AYOA_IMAGE_LORA_STRENGTH",
        "AYOA_IMAGE_STYLE_TRIGGER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AYOA_IMAGE_GENERATION_ENABLED", "false")
    monkeypatch.setenv("AYOA_IMAGE_WORKER_BACKEND", "local")

    config = ImageWorkerConfig.from_environment(
        runtime_root=Path("runtime/image_generation"),
        repo_root=tmp_path,
    )

    assert config.model_id == DEFAULT_MODEL_ID
    assert config.model_revision == DEFAULT_MODEL_REVISION
    assert config.lora_path == (
        tmp_path
        / "runtime/image_generation/loras/ayoapmu2-step600.safetensors"
    )
    assert config.lora_sha256 == DEFAULT_LORA_SHA256
    assert config.lora_strength == DEFAULT_LORA_STRENGTH
    assert config.style_trigger == DEFAULT_LORA_TRIGGER
    assert DEFAULT_LORA_SHA256 in config.runtime_revision


def test_remote_environment_uses_dev_base_without_lora(
    tmp_path,
    monkeypatch,
):
    for name in (
        "AYOA_IMAGE_MODEL",
        "AYOA_IMAGE_MODEL_REVISION",
        "AYOA_IMAGE_LORA_PATH",
        "AYOA_IMAGE_LORA_SHA256",
        "AYOA_IMAGE_LORA_STRENGTH",
        "AYOA_IMAGE_STYLE_TRIGGER",
        "AYOA_IMAGE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AYOA_IMAGE_WORKER_BACKEND", "remote")
    monkeypatch.setenv("AYOA_IMAGE_REMOTE_URL", DEFAULT_REMOTE_URL)
    monkeypatch.setenv("AYOA_IMAGE_GENERATION_ENABLED", "true")

    config = ImageWorkerConfig.from_environment(
        runtime_root=tmp_path / "runtime",
        repo_root=tmp_path,
    )

    assert config.backend == "remote"
    assert config.remote_url == DEFAULT_REMOTE_URL
    assert config.model_id == DEFAULT_REMOTE_MODEL_ID
    assert config.model_revision == DEFAULT_REMOTE_MODEL_REVISION
    assert config.lora_path is None
    assert config.lora_strength == 0
    assert config.style_trigger == ""
    assert config.timeout_seconds == 900
    assert config.runtime_revision.endswith("+remote")


def test_remote_environment_rejects_non_loopback_url(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AYOA_IMAGE_WORKER_BACKEND", "remote")
    monkeypatch.setenv("AYOA_IMAGE_REMOTE_URL", "https://images.example.com")

    with pytest.raises(ValueError, match="loopback"):
        ImageWorkerConfig.from_environment(
            runtime_root=tmp_path / "runtime",
            repo_root=tmp_path,
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
async def test_client_sends_hash_validated_frozen_reference(tmp_path):
    script = _worker_script(tmp_path)
    config = _config(tmp_path, script)
    reference_root = config.reference_root
    (reference_root / "artifacts").mkdir(parents=True)
    data = b"reviewed immutable reference"
    reference_path = reference_root / "artifacts" / "identity.webp"
    reference_path.write_bytes(data)
    request = _request(reference_inputs=[
        FrozenReferenceInput(
            reference_id="imgref_test",
            sha256=hashlib.sha256(data).hexdigest(),
            byte_count=len(data),
            relative_path="artifacts/identity.webp",
            allowed_root="artifacts",
            mime_type="image/webp",
            width=32,
            height=32,
        )
    ])
    client = ImageWorkerClient(config)
    try:
        result = await client.generate(
            request,
            output_path=tmp_path / "referenced.webp",
        )
    finally:
        await client.close()
    assert result.ok is True


@pytest.mark.asyncio
async def test_remote_client_preflights_and_sends_reference_bytes(tmp_path):
    server, url, requests = await _remote_server()
    reference_root = tmp_path / "references"
    (reference_root / "artifacts").mkdir(parents=True)
    encoded = BytesIO()
    Image.new("RGB", (32, 24), color=(90, 80, 70)).save(
        encoded,
        format="WEBP",
    )
    data = encoded.getvalue()
    (reference_root / "artifacts/identity.webp").write_bytes(data)
    request = _request(reference_inputs=[
        FrozenReferenceInput(
            reference_id="imgref_remote",
            sha256=hashlib.sha256(data).hexdigest(),
            byte_count=len(data),
            relative_path="artifacts/identity.webp",
            allowed_root="artifacts",
            mime_type="image/webp",
            width=32,
            height=24,
        )
    ])
    client = ImageWorkerClient(_remote_config(tmp_path, url))
    output = tmp_path / "remote.webp"
    try:
        assert await client.preflight() is True
        result = await client.generate(request, output_path=output)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()

    assert result.ok is True
    assert result.generation_seconds == 1.25
    assert output.read_bytes()
    assert requests[0]["prompt"] == request.prompt
    assert requests[0]["seed"] == request.seed
    assert base64.b64decode(requests[0]["reference_images"][0]) == data


@pytest.mark.asyncio
async def test_remote_edit_uses_qwen_contract_and_normalises_webp(tmp_path):
    server, url, requests = await _remote_server()
    reference_root = tmp_path / "references"
    (reference_root / "artifacts").mkdir(parents=True)
    encoded = BytesIO()
    Image.new("RGB", (32, 24), color=(90, 80, 70)).save(
        encoded,
        format="PNG",
    )
    data = encoded.getvalue()
    (reference_root / "artifacts/identity.png").write_bytes(data)
    reference = FrozenReferenceInput(
        reference_id="authored.alice.face",
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        relative_path="artifacts/identity.png",
        allowed_root="artifacts",
        mime_type="image/png",
        width=32,
        height=24,
    )
    request = _request(
        generation_mode="edit",
        reference_inputs=[reference],
    )
    client = ImageWorkerClient(_remote_config(tmp_path, url))
    output = tmp_path / "edited.webp"
    try:
        assert await client.preflight() is True
        result = await client.generate(request, output_path=output)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()

    assert requests[0]["image_base64"] == base64.b64encode(data).decode("ascii")
    assert "reference_images" not in requests[0]
    assert result.mime_type == "image/webp"
    with Image.open(output) as image:
        assert image.format == "WEBP"
        assert image.size == (request.width, request.height)


@pytest.mark.asyncio
async def test_remote_client_rejects_health_model_mismatch(tmp_path):
    server, url, _requests = await _remote_server(model="wrong/model")
    client = ImageWorkerClient(_remote_config(tmp_path, url))
    try:
        assert await client.preflight() is False
        assert client.available is False
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_remote_client_maps_gpu_oom_to_typed_error(tmp_path):
    server, url, _requests = await _remote_server(generation_status=507)
    client = ImageWorkerClient(_remote_config(tmp_path, url))
    try:
        assert await client.preflight() is True
        with pytest.raises(ImageWorkerError) as exc_info:
            await client.generate(
                _request(),
                output_path=tmp_path / "unused.webp",
            )
    finally:
        await client.close()
        server.close()
        await server.wait_closed()
    assert exc_info.value.code == "remote_oom"


@pytest.mark.asyncio
async def test_client_rejects_changed_reference_before_worker_submission(
    tmp_path,
):
    script = _worker_script(tmp_path)
    config = _config(tmp_path, script)
    (config.reference_root / "artifacts").mkdir(parents=True)
    (config.reference_root / "artifacts" / "identity.webp").write_bytes(
        b"changed!"
    )
    request = _request(reference_inputs=[
        FrozenReferenceInput(
            reference_id="imgref_test",
            sha256=hashlib.sha256(b"original").hexdigest(),
            byte_count=len(b"original"),
            relative_path="artifacts/identity.webp",
            allowed_root="artifacts",
            mime_type="image/webp",
            width=32,
            height=32,
        )
    ])
    client = ImageWorkerClient(config)
    with pytest.raises(ImageWorkerError) as exc_info:
        await client.generate(
            request,
            output_path=tmp_path / "invalid.webp",
        )
    assert exc_info.value.code == "reference_hash_mismatch"


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


def test_worker_validates_reference_bytes_format_and_dimensions(tmp_path):
    root = tmp_path / "references"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    output = BytesIO()
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(
        output,
        format="WEBP",
    )
    data = output.getvalue()
    path = artifacts / "identity.webp"
    path.write_bytes(data)

    loaded = image_worker._load_reference_images(
        [
            {
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "byte_count": len(data),
                "mime_type": "image/webp",
                "width": 32,
                "height": 24,
            }
        ],
        reference_root=root,
    )

    assert len(loaded) == 1
    assert loaded[0].size == (32, 24)


def test_worker_fails_loudly_when_pipeline_cannot_consume_references(
    tmp_path,
):
    class FakeTorch:
        class Generator:
            def __init__(self, *, device):
                self.device = device

            def manual_seed(self, seed):
                return self

        @staticmethod
        def inference_mode():
            return contextlib.nullcontext()

    class TextOnlyPipeline:
        def __call__(
            self,
            *,
            prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            generator,
        ):
            raise AssertionError("pipeline must not run")

    with pytest.raises(image_worker.ReferenceInputsUnsupported):
        image_worker._generate(
            FakeTorch(),
            TextOnlyPipeline(),
            prompt="A safe scene.",
            seed=7,
            width=256,
            height=256,
            steps=4,
            guidance=1.0,
            output_path=tmp_path / "unused.webp",
            reference_images=[Image.new("RGB", (8, 8))],
        )
