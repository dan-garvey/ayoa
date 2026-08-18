from __future__ import annotations

from contextlib import ExitStack

from infra.mi210_image_backend import gateway


def test_image_upload_metadata_accepts_supported_formats():
    assert gateway._image_upload_metadata(b"\x89PNG\r\n\x1a\nrest") == (
        ".png",
        "image/png",
    )
    assert gateway._image_upload_metadata(b"\xff\xd8\xffrest") == (
        ".jpg",
        "image/jpeg",
    )
    assert gateway._image_upload_metadata(b"RIFF1234WEBPrest") == (
        ".webp",
        "image/webp",
    )


def test_worker_reservations_spread_simultaneous_jobs(monkeypatch):
    workers = tuple(f"http://worker-{index}" for index in range(4))
    monkeypatch.setattr(gateway, "COMFY_WORKERS", workers)
    monkeypatch.setattr(
        gateway,
        "_worker_reservations",
        {worker: 0 for worker in workers},
    )
    monkeypatch.setattr(
        gateway,
        "_all_workers",
        lambda: [
            {"base": worker, "ok": True, "running": 0, "pending": 0}
            for worker in workers
        ],
    )

    with ExitStack() as stack:
        selected = [
            stack.enter_context(gateway._reserve_worker())
            for _ in range(4)
        ]
        assert set(selected) == set(workers)


def test_qwen_workflow_uses_unique_output_prefix():
    request = gateway.QwenEditRequest(
        prompt="Preserve the character and change the pose.",
        image_base64="eA==",
        filename_prefix="test",
    )

    workflow = gateway._qwen_workflow(
        request,
        "source.webp",
        None,
        None,
        "request123",
    )

    assert workflow["15"]["inputs"]["filename_prefix"] == (
        "gateway/test_request123"
    )
    assert workflow["7"]["inputs"]["prompt"] == request.prompt
    assert "16" not in workflow
    assert "17" not in workflow


def test_model_coordinator_switches_modes_in_dependency_order(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(gateway, "_stop_flux", lambda: calls.append("stop_flux"))
    monkeypatch.setattr(
        gateway,
        "_wait_for_workers_idle",
        lambda _timeout: calls.append("workers_idle"),
    )
    monkeypatch.setattr(
        gateway,
        "_free_comfy_models",
        lambda: calls.append("free_comfy"),
    )
    monkeypatch.setattr(
        gateway,
        "_ensure_flux",
        lambda: calls.append("start_flux") or {"ok": True},
    )
    coordinator = gateway.ModelCoordinator()

    with coordinator.qwen_lease():
        assert coordinator.snapshot() == {
            "mode": "qwen",
            "active_qwen": 1,
            "active_flux": False,
        }
    with coordinator.flux_lease():
        assert coordinator.snapshot() == {
            "mode": "flux",
            "active_qwen": 0,
            "active_flux": True,
        }

    assert calls == [
        "stop_flux",
        "workers_idle",
        "free_comfy",
        "start_flux",
    ]


def test_health_preserves_ayoa_remote_worker_contract(monkeypatch):
    monkeypatch.setattr(gateway, "_flux_health", lambda: None)
    monkeypatch.setattr(
        gateway,
        "_all_workers",
        lambda: [
            {"base": "http://worker", "ok": True, "running": 0, "pending": 0}
        ],
    )

    payload = gateway.health()

    assert payload["ok"] is True
    assert payload["model"] == gateway.FLUX_MODEL_ID
    assert payload["revision"] == gateway.FLUX_MODEL_REVISION
    assert payload["gpu_count"] == len(gateway.COMFY_WORKERS)
    assert payload["model_loaded"] is False
