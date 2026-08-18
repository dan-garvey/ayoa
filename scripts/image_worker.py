#!/usr/bin/env python3
"""Isolated FLUX.2 Klein worker and local GPU setup utilities.

The `--worker` mode reserves stdout for one JSON response per JSON request.
All model/runtime logs are redirected to stderr so the host protocol stays
framed even when third-party libraries print during model loading.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import os
import shutil
import statistics
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_MODEL_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
DEFAULT_SMOKE_PROMPT = (
    "A cinematic story illustration of a lone traveler beneath warm lanterns "
    "in a rain-washed old street at dusk, natural composition, no caption."
)


class InvalidWorkerRequest(ValueError):
    pass


class ReferenceInputsUnsupported(RuntimeError):
    pass


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--benchmark", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("app/storage/runtime/image_generation/models"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app/storage/runtime/image_generation/smoke.webp"),
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument("--lora-sha256", default="")
    parser.add_argument("--lora-strength", type=float, default=0.0)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("app/storage/runtime/image_generation"),
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    return parser


def _imports() -> tuple[Any, Any, Any]:
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image

    return torch, Flux2KleinPipeline, Image


def _preflight_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "cache_dir": str(args.cache_dir.resolve()),
    }
    try:
        torch, _pipeline_class, _image_class = _imports()
    except Exception as exc:
        payload["error_code"] = "image_dependencies_unavailable"
        payload["detail"] = type(exc).__name__
        return payload
    python_include = Path(sysconfig.get_paths()["include"])
    payload["python_include"] = str(python_include)
    if not (python_include / "Python.h").is_file():
        payload["error_code"] = "python_headers_unavailable"
        return payload
    try:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        probe = args.cache_dir / ".write-test"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        payload["error_code"] = "cache_not_writable"
        payload["detail"] = type(exc).__name__
        return payload

    payload["torch_version"] = str(torch.__version__)
    payload["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        payload["error_code"] = "cuda_unavailable"
        return payload
    props = torch.cuda.get_device_properties(0)
    total_gib = float(props.total_memory) / (1024**3)
    payload.update(
        {
            "gpu_name": str(props.name),
            "gpu_vram_gib": round(total_gib, 2),
            "cuda_runtime": str(torch.version.cuda or ""),
            "rtx_3090_detected": "3090" in str(props.name),
        }
    )
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        payload["host_ram_gib"] = round((pages * page_size) / (1024**3), 2)
    except (ValueError, OSError, AttributeError):
        payload["host_ram_gib"] = None
    disk = shutil.disk_usage(args.cache_dir)
    payload["disk_free_gib"] = round(disk.free / (1024**3), 2)
    local_model_path = Path(args.model_id).expanduser()
    local_model = local_model_path.exists()
    repo_key = "models--" + args.model_id.replace("/", "--")
    local_snapshot = (
        not local_model
        and (args.cache_dir / repo_key / "snapshots" / args.revision).is_dir()
    )
    local_snapshot_path = (
        args.cache_dir / repo_key / "snapshots" / args.revision
    )
    payload["model_cached"] = local_model or local_snapshot
    if args.lora_path is not None:
        lora_path = args.lora_path.expanduser().resolve()
        payload["lora_path"] = str(lora_path)
        payload["lora_strength"] = args.lora_strength
        try:
            import peft

            payload["peft_version"] = str(peft.__version__)
        except Exception as exc:
            payload["error_code"] = "lora_dependencies_unavailable"
            payload["detail"] = type(exc).__name__
            return payload
        if not 0 <= args.lora_strength <= 2:
            payload["error_code"] = "invalid_lora_strength"
            return payload
        if not lora_path.is_file():
            payload["error_code"] = "lora_unavailable"
            return payload
        try:
            lora_hash = _sha256_path(lora_path)
        except OSError as exc:
            payload["error_code"] = "lora_unavailable"
            payload["detail"] = type(exc).__name__
            return payload
        payload["lora_sha256"] = lora_hash
        expected_lora_hash = args.lora_sha256.strip().lower()
        if expected_lora_hash and lora_hash != expected_lora_hash:
            payload["error_code"] = "lora_hash_mismatch"
            return payload
    if total_gib < 23:
        payload["error_code"] = "insufficient_vram"
        return payload
    required_free_gib = 2 if (local_model or local_snapshot) else 30
    if disk.free < required_free_gib * 1024**3:
        payload["error_code"] = "insufficient_disk"
        return payload
    try:
        from huggingface_hub import HfApi

        if local_model:
            payload["resolved_model_revision"] = str(local_model_path.resolve())
            payload["model_revision_available"] = True
            payload["model_revision_source"] = "local_path"
        elif local_snapshot:
            required = (
                "model_index.json",
                "scheduler",
                "text_encoder",
                "tokenizer",
                "transformer",
                "vae",
            )
            if not all((local_snapshot_path / item).exists() for item in required):
                raise FileNotFoundError("cached model components are incomplete")
            payload["resolved_model_revision"] = str(
                local_snapshot_path.resolve()
            )
            payload["model_revision_available"] = True
            payload["model_revision_source"] = "local_cache"
        else:
            info = HfApi().model_info(args.model_id, revision=args.revision)
            payload["resolved_model_revision"] = str(info.sha or args.revision)
            payload["model_revision_available"] = True
    except Exception as exc:
        payload["error_code"] = "model_revision_unavailable"
        payload["detail"] = type(exc).__name__
        return payload
    payload["ok"] = True
    return payload


def _load_pipeline(args: argparse.Namespace) -> tuple[Any, Any]:
    torch, pipeline_class, _image_class = _imports()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = Path(args.model_id).expanduser()
    repo_key = "models--" + args.model_id.replace("/", "--")
    local_snapshot = (
        args.cache_dir / repo_key / "snapshots" / args.revision
    )
    if local_model_path.exists():
        source = str(local_model_path.resolve())
    elif local_snapshot.is_dir():
        source = str(local_snapshot.resolve())
    else:
        source = args.model_id
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if source == args.model_id:
        kwargs.update(
            revision=args.revision,
            cache_dir=str(args.cache_dir),
        )
    pipeline = pipeline_class.from_pretrained(source, **kwargs)
    if args.lora_path is not None:
        lora_path = args.lora_path.expanduser().resolve()
        if not 0 <= args.lora_strength <= 2:
            raise ValueError("LoRA strength must be between 0 and 2")
        expected_lora_hash = args.lora_sha256.strip().lower()
        if (
            expected_lora_hash
            and _sha256_path(lora_path) != expected_lora_hash
        ):
            raise ValueError("LoRA hash mismatch")
        pipeline.load_lora_weights(
            str(lora_path.parent),
            weight_name=lora_path.name,
            adapter_name="ayoa_default",
        )
        pipeline.set_adapters(
            "ayoa_default",
            adapter_weights=args.lora_strength,
        )
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")
    return torch, pipeline


def _generate(
    torch: Any,
    pipeline: Any,
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    output_path: Path,
    reference_images: list[Any] | None = None,
) -> dict[str, Any]:
    from PIL import Image

    started = time.monotonic()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipeline_arguments: dict[str, Any] = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
    }
    if reference_images:
        try:
            parameters = inspect.signature(pipeline.__call__).parameters
        except (TypeError, ValueError) as exc:
            raise ReferenceInputsUnsupported(
                "configured diffusion pipeline cannot consume references"
            ) from exc
        if "image" not in parameters:
            raise ReferenceInputsUnsupported(
                "configured diffusion pipeline cannot consume references"
            )
        pipeline_arguments["image"] = reference_images
    with torch.inference_mode():
        try:
            generated = pipeline(
                **pipeline_arguments
            ).images[0]
        except TypeError as exc:
            if reference_images and "image" in str(exc).lower():
                raise ReferenceInputsUnsupported(
                    "configured diffusion pipeline rejected references"
                ) from exc
            raise
    clean = Image.new("RGB", generated.size)
    clean.paste(generated.convert("RGB"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    clean.save(partial, format="WEBP", quality=92, method=6)
    os.replace(partial, output_path)
    with Image.open(output_path) as verified:
        if verified.format != "WEBP" or getattr(verified, "n_frames", 1) != 1:
            raise ValueError("worker output must be one static WebP frame")
        if any(
            key in verified.info
            for key in ("exif", "icc_profile", "xmp")
        ):
            raise ValueError("worker output contains image metadata")
        if verified.size != (width, height):
            raise ValueError("worker output dimensions changed during encode")
        verified.verify()
    data = output_path.read_bytes()
    return {
        "ok": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "mime_type": "image/webp",
        "width": clean.width,
        "height": clean.height,
        "byte_count": len(data),
        "generation_seconds": round(time.monotonic() - started, 4),
    }


def _load_reference_images(
    values: Any,
    *,
    reference_root: Path,
) -> list[Any]:
    from PIL import Image

    if not isinstance(values, list) or len(values) > 4:
        raise InvalidWorkerRequest("reference count")
    root = reference_root.resolve()
    allowed = (root / "artifacts").resolve()
    images: list[Any] = []
    total = 0
    for value in values:
        if not isinstance(value, dict):
            raise InvalidWorkerRequest("reference shape")
        path = Path(str(value.get("path") or "")).resolve()
        if path != allowed and allowed not in path.parents:
            raise InvalidWorkerRequest("reference path")
        try:
            expected_bytes = int(value["byte_count"])
            expected_width = int(value["width"])
            expected_height = int(value["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidWorkerRequest("reference metadata") from exc
        if expected_bytes < 1 or expected_bytes > 20_000_000:
            raise InvalidWorkerRequest("reference byte count")
        with path.open("rb") as handle:
            data = handle.read(expected_bytes + 1)
        if len(data) != expected_bytes:
            raise InvalidWorkerRequest("reference byte count mismatch")
        total += len(data)
        if total > 20_000_000:
            raise InvalidWorkerRequest("reference byte limit")
        expected_hash = str(value.get("sha256") or "").lower()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise InvalidWorkerRequest("reference hash mismatch")
        mime_type = str(value.get("mime_type") or "").lower()
        expected_format = {
            "image/jpeg": "JPEG",
            "image/png": "PNG",
            "image/webp": "WEBP",
        }.get(mime_type)
        if expected_format is None:
            raise InvalidWorkerRequest("reference MIME")
        with Image.open(io.BytesIO(data)) as opened:
            if opened.format != expected_format:
                raise InvalidWorkerRequest("reference format mismatch")
            if opened.size != (expected_width, expected_height):
                raise InvalidWorkerRequest("reference dimensions mismatch")
            opened.seek(0)
            images.append(opened.convert("RGB").copy())
    return images


def _error_code(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "outofmemory" in name or "out of memory" in text:
        return "cuda_oom"
    if "cuda" in text and ("driver" in text or "unavailable" in text):
        return "cuda_unavailable"
    if isinstance(exc, (InvalidWorkerRequest, KeyError, json.JSONDecodeError)):
        return "invalid_request"
    if isinstance(exc, ReferenceInputsUnsupported):
        return "reference_inputs_unsupported"
    return "generation_failed"


def _worker(args: argparse.Namespace) -> int:
    protocol_stdout = sys.stdout
    torch = None
    pipeline = None
    for raw in sys.stdin:
        if len(raw) > 64_000:
            _emit(protocol_stdout, {"ok": False, "error_code": "request_too_large"})
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            _emit(protocol_stdout, {"ok": False, "error_code": "invalid_request"})
            continue
        if request.get("command") == "shutdown":
            return 0
        try:
            prompt = str(request["prompt"]).strip()
            if not prompt or len(prompt) > 8_000:
                raise InvalidWorkerRequest("prompt size")
            try:
                width = int(request["width"])
                height = int(request["height"])
                steps = int(request["steps"])
                guidance = float(request["guidance"])
                seed = int(request["seed"])
            except (TypeError, ValueError) as exc:
                raise InvalidWorkerRequest("generation parameter types") from exc
            output_path = Path(str(request["output_path"])).resolve()
            reference_images = _load_reference_images(
                request.get("reference_inputs", []),
                reference_root=args.reference_root,
            )
            if (
                width < 256
                or height < 256
                or width % 16
                or height % 16
                or steps < 1
                or seed < 0
            ):
                raise InvalidWorkerRequest("invalid generation parameters")
            with contextlib.redirect_stdout(sys.stderr):
                if pipeline is None:
                    torch, pipeline = _load_pipeline(args)
                response = _generate(
                    torch,
                    pipeline,
                    prompt=prompt,
                    seed=seed,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=guidance,
                    output_path=output_path,
                    reference_images=reference_images,
                )
        except Exception as exc:
            print(
                f"generation request failed: {_error_code(exc)} "
                f"({type(exc).__name__})",
                file=sys.stderr,
                flush=True,
            )
            response = {"ok": False, "error_code": _error_code(exc)}
        _emit(protocol_stdout, response)
    return 0


def _emit(output: Any, payload: dict[str, Any]) -> None:
    output.write(json.dumps(payload, sort_keys=True) + "\n")
    output.flush()


def _download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            cache_dir=str(args.cache_dir),
        )
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error_code": "download_failed", "detail": type(exc).__name__}
            )
        )
        return 1
    print(json.dumps({"ok": True, "snapshot_path": path}))
    return 0


def _smoke_or_benchmark(args: argparse.Namespace) -> int:
    preflight = _preflight_payload(args)
    if not preflight.get("ok"):
        print(json.dumps(preflight, sort_keys=True))
        return 1
    try:
        load_started = time.monotonic()
        torch, pipeline = _load_pipeline(args)
        load_seconds = time.monotonic() - load_started
        runs = max(1, args.runs if args.benchmark else 1)
        timings: list[float] = []
        result: dict[str, Any] = {}
        for index in range(runs):
            output_path = (
                args.output
                if runs == 1
                else args.output.with_name(f"{args.output.stem}-{index:02d}.webp")
            )
            result = _generate(
                torch,
                pipeline,
                prompt=DEFAULT_SMOKE_PROMPT,
                seed=20260812 + index,
                width=args.width,
                height=args.height,
                steps=args.steps,
                guidance=args.guidance,
                output_path=output_path,
            )
            timings.append(float(result["generation_seconds"]))
        payload = {
            "ok": True,
            "model_load_seconds": round(load_seconds, 4),
            "runs": runs,
            "width": args.width,
            "height": args.height,
            "cpu_offload": bool(args.cpu_offload),
            "seconds_min": min(timings),
            "seconds_median": statistics.median(timings),
            "seconds_max": max(timings),
            "peak_vram_gib": round(
                torch.cuda.max_memory_allocated() / (1024**3),
                3,
            ),
            "last_result": result,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_code": _error_code(exc),
                    "detail": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1


def main() -> int:
    args = _parser().parse_args()
    if args.preflight:
        payload = _preflight_payload(args)
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload.get("ok") else 1
    if args.download:
        return _download(args)
    if args.worker:
        return _worker(args)
    return _smoke_or_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
