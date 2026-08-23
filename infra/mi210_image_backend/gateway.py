from __future__ import annotations

import base64
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Iterator

import requests
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field


FLUX_MODEL_ID = os.getenv(
    "AYOA_GATEWAY_FLUX_MODEL",
    "black-forest-labs/FLUX.2-dev",
)
FLUX_MODEL_REVISION = os.getenv(
    "AYOA_GATEWAY_FLUX_REVISION",
    "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
)
QWEN_MODEL_ID = os.getenv(
    "AYOA_GATEWAY_QWEN_MODEL",
    "Qwen/Qwen-Image-Edit-2511",
)
QWEN_MODEL_REVISION = os.getenv(
    "AYOA_GATEWAY_QWEN_REVISION",
    "qwen_image_edit_2511_fp8mixed.safetensors",
)
FLUX_URL = os.getenv(
    "AYOA_GATEWAY_FLUX_URL",
    "http://127.0.0.1:8188",
).rstrip("/")
FLUX_CONTAINER = os.getenv(
    "AYOA_GATEWAY_FLUX_CONTAINER",
    "ayoa-flux2-server",
)
FLUX_START_SCRIPT = os.getenv(
    "AYOA_GATEWAY_FLUX_START_SCRIPT",
    "/home/nod/.local/share/ayoa-image-server/start.sh",
)
COMFY_MASTER_URL = os.getenv(
    "AYOA_GATEWAY_COMFY_MASTER_URL",
    "http://127.0.0.1:8194",
).rstrip("/")
COMFY_PUBLIC_URL = os.getenv(
    "AYOA_GATEWAY_COMFY_PUBLIC_URL",
    "http://127.0.0.1:8189",
).rstrip("/")
COMFY_WORKERS = tuple(
    item.strip().rstrip("/")
    for item in os.getenv(
        "AYOA_GATEWAY_COMFY_WORKERS",
        ",".join(f"http://127.0.0.1:{port}" for port in range(8210, 8214)),
    ).split(",")
    if item.strip()
)
MODE_SWITCH_TIMEOUT_SECONDS = float(
    os.getenv("AYOA_GATEWAY_MODE_SWITCH_TIMEOUT_SECONDS", "900")
)
QWEN_TIMEOUT_SECONDS = int(os.getenv("AYOA_GATEWAY_QWEN_TIMEOUT_SECONDS", "1800"))
MAX_IMAGE_BASE64_LENGTH = int(
    os.getenv("AYOA_GATEWAY_MAX_IMAGE_BASE64_LENGTH", "30000000")
)


app = FastAPI(title="Ayoa MI210 Image Gateway", version="1.0")


class FluxRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    width: int = Field(default=768, ge=256, le=1536)
    height: int = Field(default=1024, ge=256, le=1536)
    steps: int = Field(default=20, ge=1, le=100)
    guidance: float = Field(default=4.0, ge=0, le=30)
    seed: int | None = Field(default=None, ge=0)
    reference_images: list[str] = Field(default_factory=list, max_length=4)


class FluxImg2ImgRequest(FluxRequest):
    init_image: str = Field(min_length=1, max_length=MAX_IMAGE_BASE64_LENGTH)
    strength: float = Field(default=0.55, ge=0, le=1)


class QwenEditRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    image_base64: str = Field(
        min_length=1,
        max_length=MAX_IMAGE_BASE64_LENGTH,
    )
    image2_base64: str = Field(default="", max_length=MAX_IMAGE_BASE64_LENGTH)
    image3_base64: str = Field(default="", max_length=MAX_IMAGE_BASE64_LENGTH)
    seed: int = Field(default=260817801, ge=0)
    steps: int = Field(default=20, ge=1, le=100)
    cfg: float = Field(default=4.0, ge=0, le=30)
    worker: str = ""
    filename_prefix: str = Field(
        default="qwen_edit_gateway",
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    unet_name: str = "qwen_image_edit_2511_fp8mixed.safetensors"
    clip_name: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"


class QwenMaskedEditRequest(QwenEditRequest):
    mask_base64: str = Field(
        min_length=1,
        max_length=MAX_IMAGE_BASE64_LENGTH,
    )
    denoise: float = Field(default=0.45, ge=0, le=1)


class BackgroundMatteRequest(BaseModel):
    image_base64: str = Field(
        min_length=1,
        max_length=MAX_IMAGE_BASE64_LENGTH,
    )
    worker: str = ""
    filename_prefix: str = Field(
        default="background_matte_gateway",
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    model_name: str = Field(
        default="birefnet.safetensors",
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )


class QwenBatchRequest(BaseModel):
    prompts: list[str] = Field(min_length=1, max_length=32)
    image_base64: str = Field(
        min_length=1,
        max_length=MAX_IMAGE_BASE64_LENGTH,
    )
    image2_base64: str = Field(default="", max_length=MAX_IMAGE_BASE64_LENGTH)
    image3_base64: str = Field(default="", max_length=MAX_IMAGE_BASE64_LENGTH)
    seed: int = Field(default=260817801, ge=0)
    steps: int = Field(default=20, ge=1, le=100)
    cfg: float = Field(default=4.0, ge=0, le=30)
    filename_prefix: str = Field(
        default="qwen_edit_batch",
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    unet_name: str = "qwen_image_edit_2511_fp8mixed.safetensors"
    clip_name: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"


def _request_json(method: str, url: str, **kwargs: Any) -> Any:
    response = requests.request(
        method,
        url,
        timeout=kwargs.pop("timeout", 30),
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def _flux_health() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{FLUX_URL}/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except (requests.RequestException, ValueError):
        return None


def _worker_queue(base: str) -> dict[str, Any]:
    try:
        data = _request_json("GET", f"{base}/queue", timeout=5)
        return {
            "base": base,
            "ok": True,
            "running": len(data.get("queue_running") or []),
            "pending": len(data.get("queue_pending") or []),
        }
    except (requests.RequestException, ValueError, TypeError) as exc:
        return {
            "base": base,
            "ok": False,
            "error": type(exc).__name__,
        }


def _all_workers() -> list[dict[str, Any]]:
    return [_worker_queue(worker) for worker in COMFY_WORKERS]


def _wait_for_workers_idle(timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        states = _all_workers()
        if not all(state.get("ok") for state in states):
            raise HTTPException(
                status_code=503,
                detail={"message": "Comfy worker unavailable", "workers": states},
            )
        if all(
            state.get("running", 0) == 0 and state.get("pending", 0) == 0
            for state in states
        ):
            return
        time.sleep(1)
    raise HTTPException(
        status_code=409,
        detail="Comfy workers did not become idle before model switch",
    )


def _free_comfy_models() -> None:
    failures: list[dict[str, str]] = []
    for worker in COMFY_WORKERS:
        try:
            response = requests.post(
                f"{worker}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            failures.append({"worker": worker, "error": type(exc).__name__})
    if failures:
        raise HTTPException(
            status_code=503,
            detail={"message": "Could not unload Comfy models", "failures": failures},
        )


def _stop_flux() -> None:
    if _flux_health() is None:
        return
    try:
        subprocess.run(
            [
                "docker",
                "exec",
                FLUX_CONTAINER,
                "sh",
                "-lc",
                "kill -TERM 1",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if _flux_health() is not None:
            raise HTTPException(
                status_code=503,
                detail=f"Could not stop FLUX service ({type(exc).__name__})",
            ) from exc
        return

    deadline = time.monotonic() + MODE_SWITCH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _flux_health() is None:
            return
        time.sleep(1)
    raise HTTPException(status_code=503, detail="FLUX service did not stop")


def _ensure_flux() -> dict[str, Any]:
    current = _flux_health()
    if current is not None:
        return current

    try:
        subprocess.run(
            ["bash", FLUX_START_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not start FLUX service ({type(exc).__name__})",
        ) from exc

    deadline = time.monotonic() + MODE_SWITCH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        current = _flux_health()
        if (
            current is not None
            and current.get("ok") is True
            and current.get("model") == FLUX_MODEL_ID
            and current.get("revision") == FLUX_MODEL_REVISION
        ):
            return current
        time.sleep(2)
    raise HTTPException(status_code=503, detail="FLUX service did not become ready")


class ModelCoordinator:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._transition_lock = threading.Lock()
        self._active_qwen = 0
        self._active_flux = False
        self._mode = "idle"

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "mode": self._mode,
                "active_qwen": self._active_qwen,
                "active_flux": self._active_flux,
            }

    def _wait_for(
        self,
        predicate: Any,
        *,
        detail: str,
    ) -> None:
        deadline = time.monotonic() + MODE_SWITCH_TIMEOUT_SECONDS
        with self._condition:
            while not predicate():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HTTPException(status_code=409, detail=detail)
                self._condition.wait(timeout=min(remaining, 1))

    @contextmanager
    def qwen_lease(self) -> Iterator[None]:
        with self._transition_lock:
            self._wait_for(
                lambda: not self._active_flux,
                detail="Timed out waiting for active FLUX request",
            )
            _stop_flux()
            with self._condition:
                self._mode = "qwen"
                self._active_qwen += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_qwen -= 1
                self._condition.notify_all()

    @contextmanager
    def flux_lease(self) -> Iterator[None]:
        with self._transition_lock:
            self._wait_for(
                lambda: self._active_qwen == 0 and not self._active_flux,
                detail="Timed out waiting for active Qwen requests",
            )
            _wait_for_workers_idle(MODE_SWITCH_TIMEOUT_SECONDS)
            _free_comfy_models()
            _ensure_flux()
            with self._condition:
                self._mode = "flux"
                self._active_flux = True
        try:
            yield
        finally:
            with self._condition:
                self._active_flux = False
                self._condition.notify_all()


_coordinator = ModelCoordinator()
_worker_reservations = {worker: 0 for worker in COMFY_WORKERS}
_worker_reservation_lock = threading.Lock()


@contextmanager
def _reserve_worker(requested: str = "") -> Iterator[str]:
    states = _all_workers()
    ready = [state for state in states if state.get("ok")]
    if requested:
        if requested not in COMFY_WORKERS:
            raise HTTPException(status_code=400, detail="Unknown worker")
        ready = [state for state in ready if state["base"] == requested]
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={"message": "No Comfy worker available", "workers": states},
        )

    with _worker_reservation_lock:
        ready.sort(
            key=lambda state: (
                state.get("running", 0)
                + state.get("pending", 0)
                + _worker_reservations[state["base"]],
                _worker_reservations[state["base"]],
                state["base"],
            )
        )
        worker = str(ready[0]["base"])
        _worker_reservations[worker] += 1
    try:
        yield worker
    finally:
        with _worker_reservation_lock:
            _worker_reservations[worker] -= 1


def _decode_base64(encoded: str, field_name: str) -> bytes:
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} is not valid base64",
        ) from exc


def _image_upload_metadata(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    raise HTTPException(
        status_code=422,
        detail="Uploaded image must be PNG, JPEG, or WebP",
    )


def _upload_image(base: str, encoded: str, name: str) -> str:
    data = _decode_base64(encoded, name)
    suffix, media_type = _image_upload_metadata(data)
    upload_name = f"{name}{suffix}"
    response = requests.post(
        f"{base}/upload/image",
        files={"image": (upload_name, data, media_type)},
        data={"overwrite": "false"},
        timeout=120,
    )
    try:
        response.raise_for_status()
        payload = response.json()
        return str(payload["name"])
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Comfy image upload failed ({type(exc).__name__})",
        ) from exc


def _qwen_workflow(
    request: QwenEditRequest,
    image1: str,
    image2: str | None,
    image3: str | None,
    request_id: str,
) -> dict[str, Any]:
    positive_inputs: dict[str, Any] = {
        "clip": ["3", 0],
        "vae": ["4", 0],
        "image1": ["5", 0],
        "prompt": request.prompt,
    }
    negative_inputs: dict[str, Any] = {
        "clip": ["3", 0],
        "vae": ["4", 0],
        "image1": ["5", 0],
        "prompt": "",
    }
    if image2:
        positive_inputs["image2"] = ["16", 0]
        negative_inputs["image2"] = ["16", 0]
    if image3:
        positive_inputs["image3"] = ["17", 0]
        negative_inputs["image3"] = ["17", 0]

    workflow: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image1}},
        "2": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": request.unet_name,
                "weight_dtype": "default",
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": request.clip_name,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": request.vae_name},
        },
        "5": {
            "class_type": "FluxKontextImageScale",
            "inputs": {"image": ["1", 0]},
        },
        "6": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["4", 0]},
        },
        "7": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": positive_inputs,
        },
        "8": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": negative_inputs,
        },
        "9": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "conditioning": ["7", 0],
                "reference_latents_method": "index_timestep_zero",
            },
        },
        "10": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "conditioning": ["8", 0],
                "reference_latents_method": "index_timestep_zero",
            },
        },
        "11": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["2", 0], "shift": 3.1},
        },
        "12": {
            "class_type": "CFGNorm",
            "inputs": {
                "model": ["11", 0],
                "strength": 1.0,
                "pre_cfg": False,
            },
        },
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["12", 0],
                "positive": ["9", 0],
                "negative": ["10", 0],
                "latent_image": ["6", 0],
                "seed": request.seed,
                "steps": request.steps,
                "cfg": request.cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "14": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["13", 0], "vae": ["4", 0]},
        },
        "15": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["14", 0],
                "filename_prefix": (f"gateway/{request.filename_prefix}_{request_id}"),
            },
        },
    }
    if image2:
        workflow["16"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image2},
        }
    if image3:
        workflow["17"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image3},
        }
    return workflow


def _masked_qwen_workflow(
    request: QwenMaskedEditRequest,
    image1: str,
    image2: str | None,
    image3: str | None,
    mask: str,
    request_id: str,
) -> dict[str, Any]:
    workflow = _qwen_workflow(
        request,
        image1,
        image2,
        image3,
        request_id,
    )
    workflow.update(
        {
            "18": {"class_type": "LoadImage", "inputs": {"image": mask}},
            "19": {
                "class_type": "FluxKontextImageScale",
                "inputs": {"image": ["18", 0]},
            },
            "20": {
                "class_type": "ImageToMask",
                "inputs": {"image": ["19", 0], "channel": "red"},
            },
            "21": {
                "class_type": "SetLatentNoiseMask",
                "inputs": {"samples": ["6", 0], "mask": ["20", 0]},
            },
        }
    )
    workflow["13"]["inputs"]["latent_image"] = ["21", 0]
    workflow["13"]["inputs"]["denoise"] = request.denoise
    return workflow


def _background_matte_workflow(
    request: BackgroundMatteRequest,
    image: str,
    request_id: str,
) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {"bg_removal_name": request.model_name},
        },
        "3": {
            "class_type": "RemoveBackground",
            "inputs": {
                "bg_removal_model": ["2", 0],
                "image": ["1", 0],
            },
        },
        "4": {
            "class_type": "MaskToImage",
            "inputs": {"mask": ["3", 0]},
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["4", 0],
                "filename_prefix": (f"gateway/{request.filename_prefix}_{request_id}"),
            },
        },
    }


def _queue_and_fetch(
    base: str,
    workflow: dict[str, Any],
) -> tuple[bytes, str, dict[str, Any]]:
    client_id = str(uuid.uuid4())
    try:
        submitted = _request_json(
            "POST",
            f"{base}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=60,
        )
        prompt_id = str(submitted["prompt_id"])
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Comfy submission failed ({type(exc).__name__})",
        ) from exc

    deadline = time.monotonic() + QWEN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            history = _request_json(
                "GET",
                f"{base}/history/{prompt_id}",
                timeout=60,
            )
        except (requests.RequestException, ValueError):
            time.sleep(1)
            continue
        if prompt_id not in history:
            time.sleep(1)
            continue

        entry = history[prompt_id]
        if entry.get("status", {}).get("status_str") == "error":
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Comfy workflow failed",
                    "worker": base,
                    "prompt_id": prompt_id,
                    "status": entry.get("status"),
                },
            )
        images: list[dict[str, Any]] = []
        for node_output in entry.get("outputs", {}).values():
            images.extend(node_output.get("images", []))
        if not images:
            time.sleep(1)
            continue

        image = images[0]
        try:
            response = requests.get(
                f"{base}/view",
                params={
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                },
                timeout=120,
            )
            response.raise_for_status()
        except (requests.RequestException, KeyError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Comfy image fetch failed ({type(exc).__name__})",
            ) from exc
        media_type = response.headers.get(
            "content-type",
            "application/octet-stream",
        ).split(";", 1)[0]
        return (
            response.content,
            media_type,
            {
                "worker": base,
                "prompt_id": prompt_id,
                "image": image,
            },
        )
    raise HTTPException(status_code=504, detail="Qwen workflow timed out")


def _run_qwen(
    request: QwenEditRequest,
) -> tuple[bytes, str, dict[str, Any]]:
    request_id = uuid.uuid4().hex
    with _coordinator.qwen_lease():
        with _reserve_worker(request.worker) as worker:
            image1 = _upload_image(
                worker,
                request.image_base64,
                f"ayoa_{request_id}_image1",
            )
            image2 = (
                _upload_image(
                    worker,
                    request.image2_base64,
                    f"ayoa_{request_id}_image2",
                )
                if request.image2_base64
                else None
            )
            image3 = (
                _upload_image(
                    worker,
                    request.image3_base64,
                    f"ayoa_{request_id}_image3",
                )
                if request.image3_base64
                else None
            )
            workflow = _qwen_workflow(
                request,
                image1,
                image2,
                image3,
                request_id,
            )
            return _queue_and_fetch(worker, workflow)


def _run_masked_qwen(
    request: QwenMaskedEditRequest,
) -> tuple[bytes, str, dict[str, Any]]:
    request_id = uuid.uuid4().hex
    with _coordinator.qwen_lease():
        with _reserve_worker(request.worker) as worker:
            image1 = _upload_image(
                worker,
                request.image_base64,
                f"ayoa_{request_id}_image1",
            )
            image2 = (
                _upload_image(
                    worker,
                    request.image2_base64,
                    f"ayoa_{request_id}_image2",
                )
                if request.image2_base64
                else None
            )
            image3 = (
                _upload_image(
                    worker,
                    request.image3_base64,
                    f"ayoa_{request_id}_image3",
                )
                if request.image3_base64
                else None
            )
            mask = _upload_image(
                worker,
                request.mask_base64,
                f"ayoa_{request_id}_mask",
            )
            workflow = _masked_qwen_workflow(
                request,
                image1,
                image2,
                image3,
                mask,
                request_id,
            )
            return _queue_and_fetch(worker, workflow)


def _run_background_matte(
    request: BackgroundMatteRequest,
) -> tuple[bytes, str, dict[str, Any]]:
    request_id = uuid.uuid4().hex
    with _coordinator.qwen_lease():
        with _reserve_worker(request.worker) as worker:
            image = _upload_image(
                worker,
                request.image_base64,
                f"ayoa_{request_id}_matte_source",
            )
            workflow = _background_matte_workflow(
                request,
                image,
                request_id,
            )
            return _queue_and_fetch(worker, workflow)


def _proxy_flux(path: str, payload: dict[str, Any]) -> Response:
    with _coordinator.flux_lease():
        try:
            response = requests.post(
                f"{FLUX_URL}{path}",
                json=payload,
                timeout=1800,
            )
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"FLUX request failed ({type(exc).__name__})",
            ) from exc
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get(
            "content-type",
            "application/octet-stream",
        ),
        headers={
            key: value
            for key, value in response.headers.items()
            if key.lower().startswith("x-ayoa-")
        },
    )


@app.get("/health")
def health() -> dict[str, Any]:
    flux = _flux_health()
    workers = _all_workers()
    coordinator = _coordinator.snapshot()
    return {
        "ok": all(worker.get("ok") for worker in workers),
        "model": FLUX_MODEL_ID,
        "revision": FLUX_MODEL_REVISION,
        "gpu_count": len(COMFY_WORKERS),
        "model_loaded": flux is not None,
        "pipelines": {
            "compose": {
                "available": True,
                "model": FLUX_MODEL_ID,
                "revision": FLUX_MODEL_REVISION,
                "max_references": 4,
            },
            "edit": {
                "available": all(worker.get("ok") for worker in workers),
                "model": QWEN_MODEL_ID,
                "revision": QWEN_MODEL_REVISION,
                "max_references": 3,
            },
        },
        **coordinator,
        "flux": flux or {"ok": False, "loaded": False},
        "comfy_master": COMFY_MASTER_URL,
        "workers": workers,
    }


@app.get("/workers")
def workers() -> list[dict[str, Any]]:
    states = _all_workers()
    with _worker_reservation_lock:
        for state in states:
            state["reserved"] = _worker_reservations.get(state["base"], 0)
    return states


@app.get("/comfy", response_class=RedirectResponse)
def comfy() -> RedirectResponse:
    with _coordinator.qwen_lease():
        pass
    return RedirectResponse(COMFY_PUBLIC_URL)


@app.post("/mode/qwen")
def use_qwen_mode() -> dict[str, Any]:
    with _coordinator.qwen_lease():
        pass
    return {"ok": True, **_coordinator.snapshot()}


@app.post("/mode/flux")
def use_flux_mode() -> dict[str, Any]:
    with _coordinator.flux_lease():
        pass
    return {"ok": True, **_coordinator.snapshot()}


@app.post("/generate")
def generate(request: FluxRequest) -> Response:
    return _proxy_flux("/generate", request.model_dump())


@app.post("/generate/flux")
def generate_flux(request: FluxRequest) -> Response:
    return generate(request)


@app.post("/img2img")
def img2img(request: FluxImg2ImgRequest) -> Response:
    return _proxy_flux("/img2img", request.model_dump())


@app.post("/img2img/flux")
def img2img_flux(request: FluxImg2ImgRequest) -> Response:
    return img2img(request)


@app.post("/edit/qwen")
def edit_qwen(request: QwenEditRequest) -> Response:
    data, media_type, metadata = _run_qwen(request)
    return Response(
        content=data,
        media_type=media_type,
        headers={
            "X-Ayoa-Worker": metadata["worker"],
            "X-Ayoa-Prompt-Id": metadata["prompt_id"],
            "X-Ayoa-Seed": str(request.seed),
        },
    )


@app.post("/prototype/edit/qwen/masked")
def prototype_edit_qwen_masked(request: QwenMaskedEditRequest) -> Response:
    data, media_type, metadata = _run_masked_qwen(request)
    return Response(
        content=data,
        media_type=media_type,
        headers={
            "X-Ayoa-Worker": metadata["worker"],
            "X-Ayoa-Prompt-Id": metadata["prompt_id"],
            "X-Ayoa-Seed": str(request.seed),
            "X-Ayoa-Prototype": "masked-qwen",
        },
    )


@app.post("/prototype/matte/birefnet")
def prototype_matte_birefnet(request: BackgroundMatteRequest) -> Response:
    data, media_type, metadata = _run_background_matte(request)
    return Response(
        content=data,
        media_type=media_type,
        headers={
            "X-Ayoa-Worker": metadata["worker"],
            "X-Ayoa-Prompt-Id": metadata["prompt_id"],
            "X-Ayoa-Matte-Model": request.model_name,
            "X-Ayoa-Prototype": "background-matte",
        },
    )


@app.post("/edit/qwen/batch")
def edit_qwen_batch(request: QwenBatchRequest) -> dict[str, Any]:
    jobs = [
        QwenEditRequest(
            prompt=prompt,
            image_base64=request.image_base64,
            image2_base64=request.image2_base64,
            image3_base64=request.image3_base64,
            seed=request.seed + index,
            steps=request.steps,
            cfg=request.cfg,
            filename_prefix=f"{request.filename_prefix}_{index:02d}",
            unet_name=request.unet_name,
            clip_name=request.clip_name,
            vae_name=request.vae_name,
        )
        for index, prompt in enumerate(request.prompts)
    ]
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(
        max_workers=min(len(jobs), max(1, len(COMFY_WORKERS)))
    ) as executor:
        futures = {
            executor.submit(_run_qwen, job): index for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            index = futures[future]
            data, media_type, metadata = future.result()
            results[index] = {
                "index": index,
                "seed": jobs[index].seed,
                "media_type": media_type,
                "image_base64": base64.b64encode(data).decode("ascii"),
                **metadata,
            }
    return {"ok": True, "images": results}
