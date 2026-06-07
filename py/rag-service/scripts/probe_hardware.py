#!/usr/bin/env python3
"""Host-side hardware probe; mirrors src/runtime/probe.py.

Stdlib-only so it can run without a venv on the user's host machine.
Emits a HardwareProfile-compatible JSON object to stdout which the Lua
plugin can read and POST to /api/v1/runtime/profile.

Usage:
    python3 scripts/probe_hardware.py
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime


def _run(cmd: list[str], timeout: int = 4) -> str:
    """Run a subprocess and return stdout; returns empty string on any error."""
    bin_ = shutil.which(cmd[0])
    if not bin_:
        return ""
    try:
        p = subprocess.run(
            [bin_, *cmd[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def _read(path: str) -> str:
    """Read a file and return its content; returns empty string on any error."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def _cpu() -> tuple[str | None, int, int]:
    """Detect CPU model, physical cores, and logical threads."""
    model = None
    cores = 0
    threads = os.cpu_count() or 0

    # Linux: read /proc/cpuinfo
    info = _read("/proc/cpuinfo")
    if info:
        for line in info.splitlines():
            if line.startswith("model name") and not model:
                model = line.split(":", 1)[1].strip()
            if line.startswith("cpu cores"):
                try:
                    cores = max(cores, int(line.split(":", 1)[1].strip()))
                except ValueError:
                    pass

    # macOS: fall back to sysctl
    if not model and platform.system() == "Darwin":
        model = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or None
        try:
            cores = int(_run(["sysctl", "-n", "hw.physicalcpu"]).strip())
            threads = int(_run(["sysctl", "-n", "hw.logicalcpu"]).strip())
        except ValueError:
            pass

    if not model:
        model = platform.processor() or None

    return model, cores or threads, threads


def _ram() -> int:
    """Return total RAM in bytes."""
    # Linux
    info = _read("/proc/meminfo")
    for line in info.splitlines():
        if line.startswith("MemTotal:"):
            try:
                return int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                pass

    # macOS
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"]).strip()
        try:
            return int(out)
        except ValueError:
            pass

    return 0


def _nvidia() -> list[dict]:
    """Probe NVIDIA GPUs via nvidia-smi."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    gpus: list[dict] = []
    for row in out.splitlines():
        parts = [c.strip() for c in row.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append({
                "vendor": "nvidia",
                "name": parts[0],
                "uuid": parts[1],
                "vram_bytes": int(float(parts[2])) * 1024 * 1024,
                "free_vram_bytes": int(float(parts[3])) * 1024 * 1024,
                "driver": parts[4],
                "compute_capability": parts[5],
                "supports_cuda": True,
                "supports_vulkan": True,
                "supports_rocm": False,
            })
        except ValueError:
            continue
    return gpus


_GFX_RE = re.compile(r"gfx\d+")


def _amd() -> list[dict]:
    """Probe AMD GPUs via rocminfo or rocm-smi."""
    out = _run(["rocminfo"]) or _run(["rocm-smi", "--showproductname"])
    if not out:
        return []
    name = "AMD GPU"
    gfx = None
    for line in out.splitlines():
        m = _GFX_RE.search(line)
        if m:
            gfx = m.group(0)
        if "Marketing Name" in line:
            name = line.split(":", 1)[1].strip() or name
    return [{
        "vendor": "amd",
        "name": name,
        "uuid": None,
        "vram_bytes": None,
        "free_vram_bytes": None,
        "driver": None,
        "compute_capability": None,
        "gfx_target": gfx,
        "supports_cuda": False,
        "supports_rocm": True,
        "supports_vulkan": True,
    }]


def _apple_silicon() -> list[dict]:
    """Detect Apple Silicon unified memory GPU."""
    if platform.system() != "Darwin":
        return []
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    if not (chip.startswith("Apple M") or "Apple" in chip):
        return []
    # Unified memory: report total RAM as VRAM
    ram = _ram()
    return [{
        "vendor": "apple",
        "name": chip,
        "uuid": None,
        "vram_bytes": ram,
        "free_vram_bytes": None,
        "driver": None,
        "compute_capability": None,
        "gfx_target": None,
        "supports_cuda": False,
        "supports_rocm": False,
        "supports_vulkan": False,
    }]


def _vulkan_only() -> list[dict]:
    """Probe Vulkan-capable devices via vulkaninfo as last resort."""
    out = _run(["vulkaninfo", "--summary"])
    devs: list[dict] = []
    for line in out.splitlines():
        if "deviceName" in line:
            devs.append({
                "vendor": "unknown",
                "name": line.split("=", 1)[-1].strip(),
                "uuid": None,
                "vram_bytes": None,
                "free_vram_bytes": None,
                "driver": None,
                "compute_capability": None,
                "gfx_target": None,
                "supports_cuda": False,
                "supports_rocm": False,
                "supports_vulkan": True,
            })
    return devs


def probe() -> dict:
    """Run a best-effort hardware probe and return a HardwareProfile-compatible dict."""
    warnings: list[str] = []
    cpu_model, cores, threads = _cpu()
    ram = _ram()

    gpus = _nvidia()
    if not gpus:
        amd = _amd()
        if amd:
            gpus.extend(amd)
        else:
            warnings.append("no nvidia-smi/rocminfo output")

    if not gpus:
        apple = _apple_silicon()
        if apple:
            gpus.extend(apple)

    if not gpus:
        vk = _vulkan_only()
        if vk:
            gpus.extend(vk)
        else:
            warnings.append("no vulkaninfo output")

    return {
        "os": f"{platform.system()} {platform.release()}",
        "detected_in": "host",
        "cpu_model": cpu_model,
        "cpu_cores": cores,
        "cpu_threads": threads,
        "ram_bytes": ram,
        "gpus": gpus,
        "probe_warnings": warnings,
        "captured_at": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))

