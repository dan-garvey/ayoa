from __future__ import annotations

import asyncio
import base64
import contextlib
import fcntl
import hashlib
import ipaddress
import json
import logging
import os
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image, ImageOps

from app.schemas.image_generation import ImageGenerationRequest, ImageWorkerResult
from app.schemas.image_director import ImageGenerationMode


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_MODEL_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
DEFAULT_LORA_FILENAME = "ayoapmu2-step600.safetensors"
DEFAULT_LORA_SHA256 = (
    "3388acd713240f34f9266cf164529ef78a43f13cb2cae7864d1db22638a1fbbc"
)
DEFAULT_LORA_STRENGTH = 0.8
DEFAULT_LORA_TRIGGER = "ayoapmu2"
DEFAULT_REMOTE_MODEL_ID = "black-forest-labs/FLUX.2-dev"
DEFAULT_REMOTE_MODEL_REVISION = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
DEFAULT_REMOTE_EDIT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_REMOTE_EDIT_MODEL_REVISION = "qwen_image_edit_2511_fp8mixed.safetensors"
DEFAULT_REMOTE_URL = "http://127.0.0.1:8199"


def _normalise_remote_url(value: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    parsed = urlsplit(raw)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "AYOA_IMAGE_REMOTE_URL must be an HTTP loopback origin"
        )
    hostname = parsed.hostname
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname.lower() == "localhost"
    if not loopback:
        raise ValueError(
            "AYOA_IMAGE_REMOTE_URL must use loopback; tunnel remote services"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("AYOA_IMAGE_REMOTE_URL has an invalid port") from exc
    if port is None:
        port = 80
    host_for_url = f"[{hostname}]" if ":" in hostname else hostname
    return f"http://{host_for_url}:{port}"


def _remote_modes(payload: dict[str, Any]) -> set[str]:
    pipelines = payload.get("pipelines")
    if not isinstance(pipelines, dict):
        return {"compose"}
    return {
        str(mode)
        for mode, details in pipelines.items()
        if isinstance(details, dict) and details.get("available") is True
    }


def _normalise_remote_output(
    data: bytes,
    *,
    media_type: str,
    width: int,
    height: int,
    allow_resize: bool,
) -> bytes:
    accepted = {"image/webp"}
    if allow_resize:
        accepted.update({"image/png", "image/jpeg"})
    if media_type not in accepted:
        raise ImageWorkerError("remote_protocol_error")
    try:
        with Image.open(BytesIO(data)) as source:
            source.load()
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("animated image")
            if not allow_resize:
                if source.format != "WEBP" or source.size != (width, height):
                    raise ValueError("remote image metadata mismatch")
                return data
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.size != (width, height):
                image = ImageOps.fit(
                    image,
                    (width, height),
                    method=Image.Resampling.LANCZOS,
                )
            output = BytesIO()
            image.save(output, format="WEBP", quality=95, method=6)
            return output.getvalue()
    except ImageWorkerError:
        raise
    except Exception as exc:
        raise ImageWorkerError("remote_protocol_error") from exc


class ImageWorkerError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Image worker failed ({code}).")


@dataclass(frozen=True)
class ImageWorkerConfig:
    enabled: bool
    python_executable: Path
    worker_script: Path
    backend: str = "local"
    remote_url: str = ""
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    model_cache_dir: Path = Path("app/storage/runtime/image_generation/models")
    lock_path: Path = Path("app/storage/runtime/image_generation/gpu.lock")
    reference_root: Path = Path("app/storage/runtime/image_generation")
    timeout_seconds: float = 300.0
    cpu_offload: bool = True
    lora_path: Path | None = None
    lora_sha256: str = ""
    lora_strength: float = 0.0
    style_trigger: str = ""

    def model_id_for(self, generation_mode: ImageGenerationMode) -> str:
        if generation_mode == "edit" and self.backend == "remote":
            return DEFAULT_REMOTE_EDIT_MODEL_ID
        return self.model_id

    def model_revision_for(self, generation_mode: ImageGenerationMode) -> str:
        if generation_mode == "edit" and self.backend == "remote":
            return f"{DEFAULT_REMOTE_EDIT_MODEL_REVISION}+remote"
        return self.runtime_revision

    @property
    def runtime_revision(self) -> str:
        revision = self.model_revision
        if self.lora_path is not None:
            fingerprint = self.lora_sha256 or "unverified"
            revision = (
                f"{revision}+lora:{fingerprint}:{self.lora_strength:.4g}"
            )
        return f"{revision}+remote" if self.backend == "remote" else revision

    @classmethod
    def from_environment(
        cls,
        *,
        runtime_root: str | Path,
        repo_root: str | Path | None = None,
    ) -> "ImageWorkerConfig":
        root = Path(runtime_root)
        repository = Path(repo_root or Path.cwd())
        backend = os.getenv("AYOA_IMAGE_WORKER_BACKEND", "local").strip().lower()
        if backend not in {"local", "remote"}:
            raise ValueError("AYOA_IMAGE_WORKER_BACKEND must be local or remote")
        remote_url = (
            _normalise_remote_url(
                os.getenv("AYOA_IMAGE_REMOTE_URL", DEFAULT_REMOTE_URL)
            )
            if backend == "remote"
            else ""
        )
        python_path = Path(
            os.getenv("AYOA_IMAGE_WORKER_PYTHON", ".venv-image/bin/python")
        )
        worker_script = Path(
            os.getenv("AYOA_IMAGE_WORKER_SCRIPT", "scripts/image_worker.py")
        )
        if not python_path.is_absolute():
            python_path = repository / python_path
        if not worker_script.is_absolute():
            worker_script = repository / worker_script
        raw_enabled = os.getenv("AYOA_IMAGE_GENERATION_ENABLED", "auto").strip().lower()
        if raw_enabled == "auto":
            enabled = bool(remote_url) if backend == "remote" else python_path.is_file()
        elif raw_enabled in {"1", "true", "yes", "on", "enabled"}:
            enabled = True
        elif raw_enabled in {"0", "false", "no", "off", "disabled"}:
            enabled = False
        else:
            raise ValueError(
                "AYOA_IMAGE_GENERATION_ENABLED must be auto, true, or false"
            )
        raw_offload = os.getenv("AYOA_IMAGE_CPU_OFFLOAD", "false").strip().lower()
        cpu_offload = raw_offload not in {"0", "false", "no", "off"}
        default_lora_path = root / "loras" / DEFAULT_LORA_FILENAME
        if not default_lora_path.is_absolute():
            default_lora_path = repository / default_lora_path
        raw_lora_path = os.getenv(
            "AYOA_IMAGE_LORA_PATH",
            "none" if backend == "remote" else str(default_lora_path),
        ).strip()
        if raw_lora_path.lower() in {"", "none", "off", "disabled"}:
            lora_path = None
        else:
            lora_path = Path(raw_lora_path).expanduser()
            if not lora_path.is_absolute():
                lora_path = repository / lora_path
        if backend == "remote" and lora_path is not None:
            raise ValueError(
                "remote image backend does not accept local LoRA configuration"
            )
        lora_strength = float(
            os.getenv(
                "AYOA_IMAGE_LORA_STRENGTH",
                str(DEFAULT_LORA_STRENGTH),
            )
        )
        if not 0 <= lora_strength <= 2:
            raise ValueError("AYOA_IMAGE_LORA_STRENGTH must be between 0 and 2")
        configured_hash = os.getenv(
            "AYOA_IMAGE_LORA_SHA256",
            DEFAULT_LORA_SHA256 if lora_path == default_lora_path else "",
        ).strip().lower()
        style_trigger = (
            os.getenv("AYOA_IMAGE_STYLE_TRIGGER", DEFAULT_LORA_TRIGGER).strip()
            if lora_path is not None
            else ""
        )
        default_model_id = (
            DEFAULT_REMOTE_MODEL_ID if backend == "remote" else DEFAULT_MODEL_ID
        )
        default_model_revision = (
            DEFAULT_REMOTE_MODEL_REVISION
            if backend == "remote"
            else DEFAULT_MODEL_REVISION
        )
        return cls(
            enabled=enabled,
            python_executable=python_path,
            worker_script=worker_script,
            backend=backend,
            remote_url=remote_url,
            model_id=(
                os.getenv("AYOA_IMAGE_MODEL", default_model_id).strip()
                or default_model_id
            ),
            model_revision=(
                os.getenv(
                    "AYOA_IMAGE_MODEL_REVISION",
                    default_model_revision,
                ).strip()
                or default_model_revision
            ),
            model_cache_dir=Path(
                os.getenv("AYOA_IMAGE_MODEL_CACHE", str(root / "models"))
            ),
            lock_path=root / "gpu.lock",
            reference_root=root,
            timeout_seconds=float(
                os.getenv(
                    "AYOA_IMAGE_TIMEOUT_SECONDS",
                    "900" if backend == "remote" else "300",
                )
            ),
            cpu_offload=cpu_offload,
            lora_path=lora_path,
            lora_sha256=configured_hash,
            lora_strength=lora_strength if lora_path is not None else 0.0,
            style_trigger=style_trigger,
        )


class ImageWorkerClient:
    """One supervised local process or loopback-tunneled diffusion service."""

    def __init__(self, config: ImageWorkerConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._remote_writer: asyncio.StreamWriter | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._request_lock = asyncio.Lock()
        self._gpu_lock_handle: Any = None
        self._abort_requested = False
        self._preflight_ok: bool | None = None

    @property
    def available(self) -> bool:
        if not self.config.enabled or self.config.timeout_seconds <= 0:
            return False
        if self.config.backend == "remote":
            return bool(
                self.config.remote_url and self._preflight_ok is not False
            )
        return bool(
            self.config.python_executable.is_file()
            and self.config.worker_script.is_file()
            and self._local_model_available()
            and self._preflight_ok is not False
        )

    @property
    def supported_generation_modes(self) -> tuple[ImageGenerationMode, ...]:
        return ("compose", "edit") if self.config.backend == "remote" else (
            "compose",
        )

    async def preflight(self) -> bool:
        if not self.available:
            self._preflight_ok = False
            return False
        if self.config.backend == "remote":
            return await self._remote_preflight()
        command = [
            str(self.config.python_executable),
            str(self.config.worker_script),
            "--preflight",
            "--model-id",
            self.config.model_id,
            "--revision",
            self.config.model_revision,
            "--cache-dir",
            str(self.config.model_cache_dir),
        ]
        command.extend(self._lora_command_arguments())
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.config.worker_script.parent.parent),
            )
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60,
            )
            payload = json.loads(stdout.decode("utf-8").strip().splitlines()[-1])
            self._preflight_ok = bool(process.returncode == 0 and payload.get("ok"))
            if not self._preflight_ok:
                logger.warning(
                    "local image preflight failed: %s",
                    str(payload.get("error_code") or "unknown"),
                )
        except TimeoutError:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            logger.warning("local image preflight timed out")
            self._preflight_ok = False
        except Exception:
            logger.exception("local image preflight process failed")
            self._preflight_ok = False
        return bool(self._preflight_ok)

    async def generate(
        self,
        request: ImageGenerationRequest,
        *,
        output_path: str | Path,
    ) -> ImageWorkerResult:
        if not self.available:
            raise ImageWorkerError("worker_unavailable")
        async with self._request_lock:
            self._abort_requested = False
            for attempt in range(2):
                try:
                    return await self._generate_once(
                        request,
                        output_path=Path(output_path),
                    )
                except ImageWorkerError as exc:
                    retryable = exc.code in {
                        "worker_exited",
                        "worker_protocol_error",
                        "worker_write_failed",
                        "remote_unavailable",
                        "remote_protocol_error",
                    }
                    await self._stop_process()
                    if self._abort_requested:
                        raise ImageWorkerError("worker_cancelled") from exc
                    if attempt == 0 and retryable:
                        logger.warning(
                            "image worker retrying after %s", exc.code
                        )
                        continue
                    raise
        raise ImageWorkerError("worker_failed")

    async def abort_current(self) -> None:
        """Terminate an in-flight inference without waiting for its lock."""

        self._abort_requested = True
        if self.config.backend == "remote":
            await self._close_remote_writer()
            return
        await self._stop_process()

    async def close(self) -> None:
        async with self._request_lock:
            if self.config.backend == "remote":
                await self._close_remote_writer()
            else:
                await self._stop_process(graceful=True)
            self._release_gpu_lock()

    async def _generate_once(
        self,
        request: ImageGenerationRequest,
        *,
        output_path: Path,
    ) -> ImageWorkerResult:
        if self.config.backend == "remote":
            return await self._generate_remote(
                request,
                output_path=output_path,
            )
        if request.generation_mode != "compose":
            raise ImageWorkerError("generation_mode_unsupported")
        process = await self._ensure_process()
        if process.stdin is None or process.stdout is None:
            raise ImageWorkerError("worker_protocol_error")
        payload = {
            "request_id": request.dedupe_key,
            "prompt": request.prompt,
            "seed": request.seed,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance": request.guidance,
            "output_path": str(output_path),
            "reference_inputs": self._reference_payloads(request),
        }
        try:
            process.stdin.write(
                (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            )
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionError) as exc:
            raise ImageWorkerError("worker_write_failed") from exc

        try:
            raw = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=self.config.timeout_seconds,
            )
        except TimeoutError as exc:
            await self._stop_process()
            raise ImageWorkerError("worker_timeout") from exc
        if not raw:
            raise ImageWorkerError("worker_exited")
        try:
            result = ImageWorkerResult.model_validate_json(raw)
        except Exception as exc:
            raise ImageWorkerError("worker_protocol_error") from exc
        if not result.ok:
            raise ImageWorkerError(result.error_code or "generation_failed")
        return result

    async def _remote_preflight(self) -> bool:
        try:
            status, _headers, body = await asyncio.wait_for(
                self._remote_http_request(
                    method="GET",
                    path="/health",
                    body=b"",
                    max_response_bytes=64_000,
                ),
                timeout=min(self.config.timeout_seconds, 30.0),
            )
            payload = json.loads(body.decode("utf-8"))
            self._preflight_ok = bool(
                status == 200
                and payload.get("ok") is True
                and payload.get("model") == self.config.model_id
                and payload.get("revision") == self.config.model_revision
                and int(payload.get("gpu_count") or 0) >= 1
                and _remote_modes(payload) >= set(
                    self.supported_generation_modes
                )
            )
            if not self._preflight_ok:
                logger.warning(
                    "remote image preflight rejected health contract"
                )
        except Exception:
            logger.warning(
                "remote image preflight failed",
                exc_info=True,
            )
            self._preflight_ok = False
        return bool(self._preflight_ok)

    async def _generate_remote(
        self,
        request: ImageGenerationRequest,
        *,
        output_path: Path,
    ) -> ImageWorkerResult:
        reference_images = [
            base64.b64encode(data).decode("ascii")
            for _metadata, data in self._validated_references(request)
        ]
        if request.generation_mode == "edit":
            if not 1 <= len(reference_images) <= 3:
                raise ImageWorkerError("reference_inputs_required")
            edit_images = reference_images + [""] * (3 - len(reference_images))
            request_path = "/edit/qwen"
            request_payload = {
                "prompt": request.prompt,
                "image_base64": edit_images[0],
                "image2_base64": edit_images[1],
                "image3_base64": edit_images[2],
                "steps": request.steps,
                "cfg": request.guidance,
                "seed": request.seed,
            }
        else:
            request_path = "/generate"
            request_payload = {
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "guidance": request.guidance,
                "seed": request.seed,
                "reference_images": reference_images,
            }
        payload = json.dumps(
            request_payload,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            status, headers, body = await asyncio.wait_for(
                self._remote_http_request(
                    method="POST",
                    path=request_path,
                    body=payload,
                    max_response_bytes=8_000_000,
                ),
                timeout=self.config.timeout_seconds,
            )
        except TimeoutError as exc:
            await self._close_remote_writer()
            raise ImageWorkerError("worker_timeout") from exc
        if status != 200:
            code = {
                422: "invalid_request",
                507: "remote_oom",
            }.get(status, "remote_generation_failed")
            raise ImageWorkerError(code)
        media_type = headers.get("content-type", "").split(";", 1)[0]
        try:
            returned_seed = int(headers["x-ayoa-seed"])
            generation_seconds = float(
                headers.get("x-ayoa-generation-seconds", "0")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ImageWorkerError("remote_protocol_error") from exc
        if returned_seed != request.seed:
            raise ImageWorkerError("remote_seed_mismatch")
        body = _normalise_remote_output(
            body,
            media_type=media_type,
            width=request.width,
            height=request.height,
            allow_resize=request.generation_mode == "edit",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_suffix(output_path.suffix + ".partial")
        partial.write_bytes(body)
        os.replace(partial, output_path)
        return ImageWorkerResult(
            ok=True,
            sha256=hashlib.sha256(body).hexdigest(),
            mime_type="image/webp",
            width=request.width,
            height=request.height,
            byte_count=len(body),
            generation_seconds=max(0.0, generation_seconds),
        )

    async def _remote_http_request(
        self,
        *,
        method: str,
        path: str,
        body: bytes,
        max_response_bytes: int,
    ) -> tuple[int, dict[str, str], bytes]:
        parsed = urlsplit(self.config.remote_url)
        host = parsed.hostname or ""
        port = parsed.port or 80
        host_header = f"[{host}]" if ":" in host else host
        writer: asyncio.StreamWriter | None = None
        try:
            reader, writer = await asyncio.open_connection(host, port)
            self._remote_writer = writer
            headers = [
                f"{method} {path} HTTP/1.1",
                f"Host: {host_header}:{port}",
                "Accept: application/json, image/webp",
                "Connection: close",
                f"Content-Length: {len(body)}",
            ]
            if body:
                headers.append("Content-Type: application/json")
            writer.write(("\r\n".join(headers) + "\r\n\r\n").encode("ascii"))
            if body:
                writer.write(body)
            await writer.drain()
            status_line = await reader.readline()
            if len(status_line) > 8_192:
                raise ImageWorkerError("remote_protocol_error")
            parts = status_line.decode("ascii", errors="strict").split()
            if len(parts) < 2 or not parts[0].startswith("HTTP/"):
                raise ImageWorkerError("remote_protocol_error")
            status = int(parts[1])
            response_headers: dict[str, str] = {}
            header_bytes = 0
            while True:
                line = await reader.readline()
                header_bytes += len(line)
                if header_bytes > 64_000 or not line:
                    raise ImageWorkerError("remote_protocol_error")
                if line == b"\r\n":
                    break
                name, separator, value = line.decode(
                    "latin-1"
                ).partition(":")
                if not separator:
                    raise ImageWorkerError("remote_protocol_error")
                response_headers[name.strip().lower()] = value.strip()
            if "transfer-encoding" in response_headers:
                raise ImageWorkerError("remote_protocol_error")
            try:
                content_length = int(response_headers["content-length"])
            except (KeyError, ValueError) as exc:
                raise ImageWorkerError("remote_protocol_error") from exc
            if content_length < 0 or content_length > max_response_bytes:
                raise ImageWorkerError("remote_response_too_large")
            response_body = await reader.readexactly(content_length)
            return status, response_headers, response_body
        except ImageWorkerError:
            raise
        except (OSError, asyncio.IncompleteReadError) as exc:
            raise ImageWorkerError("remote_unavailable") from exc
        except (UnicodeError, ValueError) as exc:
            raise ImageWorkerError("remote_protocol_error") from exc
        finally:
            if writer is not None:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
            if self._remote_writer is writer:
                self._remote_writer = None

    async def _close_remote_writer(self) -> None:
        writer = self._remote_writer
        self._remote_writer = None
        if writer is None:
            return
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process
        self._acquire_gpu_lock()
        self.config.model_cache_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.config.python_executable),
            str(self.config.worker_script),
            "--worker",
            "--model-id",
            self.config.model_id,
            "--revision",
            self.config.model_revision,
            "--cache-dir",
            str(self.config.model_cache_dir),
            "--reference-root",
            str(self.config.reference_root),
        ]
        command.extend(self._lora_command_arguments())
        if self.config.cpu_offload:
            command.append("--cpu-offload")
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.config.worker_script.parent.parent),
            )
        except OSError as exc:
            self._release_gpu_lock()
            raise ImageWorkerError("worker_start_failed") from exc
        if self._abort_requested:
            await self._stop_process()
            raise ImageWorkerError("worker_cancelled")
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(self._process),
            name="ayoa-image-worker-stderr",
        )
        return self._process

    async def _stop_process(self, *, graceful: bool = False) -> None:
        process = self._process
        self._process = None
        if process is not None and process.returncode is None:
            if graceful and process.stdin is not None:
                try:
                    process.stdin.write(b'{"command":"shutdown"}\n')
                    await process.stdin.drain()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except Exception:
                    process.terminate()
            else:
                process.terminate()
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        task = self._stderr_task
        self._stderr_task = None
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                logger.info("image worker: %s", text[:500])

    def _acquire_gpu_lock(self) -> None:
        if self._gpu_lock_handle is not None:
            return
        self.config.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.config.lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise ImageWorkerError("gpu_in_use") from exc
        self._gpu_lock_handle = handle

    def _release_gpu_lock(self) -> None:
        handle = self._gpu_lock_handle
        self._gpu_lock_handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _local_model_available(self) -> bool:
        if (
            self.config.lora_path is not None
            and not self.config.lora_path.is_file()
        ):
            return False
        model_path = Path(self.config.model_id).expanduser()
        if model_path.exists():
            return True
        repo_key = "models--" + self.config.model_id.replace("/", "--")
        snapshot = (
            self.config.model_cache_dir
            / repo_key
            / "snapshots"
            / self.config.model_revision
        )
        return snapshot.is_dir()

    def _lora_command_arguments(self) -> list[str]:
        if self.config.lora_path is None:
            return []
        arguments = [
            "--lora-path",
            str(self.config.lora_path),
            "--lora-strength",
            str(self.config.lora_strength),
        ]
        if self.config.lora_sha256:
            arguments.extend(
                ["--lora-sha256", self.config.lora_sha256]
            )
        return arguments

    def _reference_payloads(
        self,
        request: ImageGenerationRequest,
    ) -> list[dict[str, object]]:
        return [
            metadata
            for metadata, _data in self._validated_references(request)
        ]

    def _validated_references(
        self,
        request: ImageGenerationRequest,
    ) -> list[tuple[dict[str, object], bytes]]:
        root = self.config.reference_root.resolve()
        allowed = (root / "artifacts").resolve()
        payloads: list[tuple[dict[str, object], bytes]] = []
        total = 0
        for reference in request.reference_inputs:
            if reference.allowed_root != "artifacts":
                raise ImageWorkerError("reference_root_unauthorized")
            path = (root / reference.relative_path).resolve()
            if path != allowed and allowed not in path.parents:
                raise ImageWorkerError("reference_path_unsafe")
            try:
                with path.open("rb") as handle:
                    data = handle.read(reference.byte_count + 1)
            except OSError as exc:
                raise ImageWorkerError("reference_unavailable") from exc
            if len(data) != reference.byte_count:
                raise ImageWorkerError("reference_byte_count_mismatch")
            if hashlib.sha256(data).hexdigest() != reference.sha256:
                raise ImageWorkerError("reference_hash_mismatch")
            if self.config.backend == "remote" and len(data) > 10_000_000:
                raise ImageWorkerError("reference_limits_exceeded")
            total += len(data)
            if len(payloads) >= 4 or total > 20_000_000:
                raise ImageWorkerError("reference_limits_exceeded")
            payloads.append(
                (
                    {
                        "reference_id": reference.reference_id,
                        "sha256": reference.sha256,
                        "mime_type": reference.mime_type,
                        "width": reference.width,
                        "height": reference.height,
                        "byte_count": reference.byte_count,
                        "path": str(path),
                    },
                    data,
                )
            )
        return payloads
