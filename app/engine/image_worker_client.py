from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.schemas.image_generation import ImageGenerationRequest, ImageWorkerResult


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_MODEL_REVISION = "5e67da950fce4a097bc150c22958a05716994cea"


class ImageWorkerError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Local image worker failed ({code}).")


@dataclass(frozen=True)
class ImageWorkerConfig:
    enabled: bool
    python_executable: Path
    worker_script: Path
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    model_cache_dir: Path = Path("app/storage/runtime/image_generation/models")
    lock_path: Path = Path("app/storage/runtime/image_generation/gpu.lock")
    timeout_seconds: float = 300.0
    cpu_offload: bool = True

    @classmethod
    def from_environment(
        cls,
        *,
        runtime_root: str | Path,
        repo_root: str | Path | None = None,
    ) -> "ImageWorkerConfig":
        root = Path(runtime_root)
        repository = Path(repo_root or Path.cwd())
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
            enabled = python_path.is_file()
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
        return cls(
            enabled=enabled,
            python_executable=python_path,
            worker_script=worker_script,
            model_id=(
                os.getenv("AYOA_IMAGE_MODEL", DEFAULT_MODEL_ID).strip()
                or DEFAULT_MODEL_ID
            ),
            model_revision=(
                os.getenv("AYOA_IMAGE_MODEL_REVISION", DEFAULT_MODEL_REVISION).strip()
                or DEFAULT_MODEL_REVISION
            ),
            model_cache_dir=Path(
                os.getenv("AYOA_IMAGE_MODEL_CACHE", str(root / "models"))
            ),
            lock_path=root / "gpu.lock",
            timeout_seconds=float(
                os.getenv("AYOA_IMAGE_TIMEOUT_SECONDS", "300")
            ),
            cpu_offload=cpu_offload,
        )


class ImageWorkerClient:
    """One supervised JSONL subprocess holding the local diffusion pipeline."""

    def __init__(self, config: ImageWorkerConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._request_lock = asyncio.Lock()
        self._gpu_lock_handle: Any = None
        self._abort_requested = False
        self._preflight_ok: bool | None = None

    @property
    def available(self) -> bool:
        return (
            self.config.enabled
            and self.config.python_executable.is_file()
            and self.config.worker_script.is_file()
            and self.config.timeout_seconds > 0
            and self._local_model_available()
            and self._preflight_ok is not False
        )

    async def preflight(self) -> bool:
        if not self.available:
            self._preflight_ok = False
            return False
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
                    }
                    await self._stop_process()
                    if self._abort_requested:
                        raise ImageWorkerError("worker_cancelled") from exc
                    if attempt == 0 and retryable:
                        logger.warning(
                            "local image worker restarting after %s", exc.code
                        )
                        continue
                    raise
        raise ImageWorkerError("worker_failed")

    async def abort_current(self) -> None:
        """Terminate an in-flight inference without waiting for its lock."""

        self._abort_requested = True
        await self._stop_process()

    async def close(self) -> None:
        async with self._request_lock:
            await self._stop_process(graceful=True)
            self._release_gpu_lock()

    async def _generate_once(
        self,
        request: ImageGenerationRequest,
        *,
        output_path: Path,
    ) -> ImageWorkerResult:
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
        ]
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
