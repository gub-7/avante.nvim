"""Tests for the hardware-aware backend selector."""

from __future__ import annotations

from datetime import datetime


def _cpu_only_profile():
    from models.runtime import HardwareProfile

    return HardwareProfile(
        os="Linux 5.15",
        detected_in="container",
        cpu_model="Intel Core i7",
        cpu_cores=4,
        cpu_threads=8,
        ram_bytes=16 * 1024 ** 3,
        gpus=[],
        probe_warnings=["no GPU"],
        captured_at=datetime.utcnow().isoformat(),
    )


def _nvidia_profile():
    from models.runtime import GPUDevice, HardwareProfile

    return HardwareProfile(
        os="Linux 5.15",
        detected_in="container",
        cpu_model="Intel Xeon",
        cpu_cores=8,
        cpu_threads=16,
        ram_bytes=64 * 1024 ** 3,
        gpus=[
            GPUDevice(
                vendor="nvidia",
                name="RTX 3090",
                uuid="GPU-0",
                vram_bytes=24 * 1024 ** 3,
                supports_cuda=True,
            )
        ],
        captured_at=datetime.utcnow().isoformat(),
    )


def test_cpu_only_selects_cpu_accelerator():
    """CPU-only profile should recommend accelerator='cpu'."""
    from runtime.backend_selector import select

    rec = select(_cpu_only_profile())
    assert rec.accelerator == "cpu"


def test_nvidia_selects_cuda():
    """Profile with NVIDIA GPU should recommend accelerator='cuda'."""
    from runtime.backend_selector import select

    rec = select(_nvidia_profile())
    assert rec.accelerator == "cuda"

