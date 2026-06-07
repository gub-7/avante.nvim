"""In-container hardware probe; best-effort detection of CPUs, RAM, and GPUs."""
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime

from models.runtime import GPUDevice, HardwareProfile


def _run(cmd: list[str], timeout: int = 4) -> str:
    """Run a subprocess command and return stdout; returns empty string on any error."""
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
        return open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""


def _cpu() -> tuple[str | None, int, int]:
    """Detect CPU model, physical cores, and logical threads."""
    model = None
    cores = 0
    threads = os.cpu_count() or 0
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
    if not model:
        model = platform.processor() or None
    return model, cores or threads, threads


def _ram() -> int:
    """Return total RAM in bytes by reading /proc/meminfo."""
    info = _read("/proc/meminfo")
    for line in info.splitlines():
        if line.startswith("MemTotal:"):
            kb = int(line.split()[1])
            return kb * 1024
    return 0


def _nvidia() -> list[GPUDevice]:
    """Probe NVIDIA GPUs via nvidia-smi."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    gpus: list[GPUDevice] = []
    for row in out.splitlines():
        parts = [c.strip() for c in row.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(GPUDevice(
                vendor="nvidia",
                name=parts[0],
                uuid=parts[1],
                vram_bytes=int(float(parts[2])) * 1024 * 1024,
                free_vram_bytes=int(float(parts[3])) * 1024 * 1024,
                driver=parts[4],
                compute_capability=parts[5],
                supports_cuda=True,
                supports_vulkan=True,
            ))
        except ValueError:
            continue
    return gpus


_GFX_RE = re.compile(r"gfx\d+")


def _amd() -> list[GPUDevice]:
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
    return [GPUDevice(
        vendor="amd",
        name=name,
        gfx_target=gfx,
        supports_rocm=True,
        supports_vulkan=True,
    )]


def _vulkan_only() -> list[GPUDevice]:
    """Probe Vulkan-capable devices via vulkaninfo as last resort."""
    out = _run(["vulkaninfo", "--summary"])
    devs: list[GPUDevice] = []
    for line in out.splitlines():
        if "deviceName" in line:
            devs.append(GPUDevice(
                vendor="unknown",
                name=line.split("=", 1)[-1].strip(),
                supports_vulkan=True,
            ))
    return devs


def probe(source: str = "container") -> HardwareProfile:
    """Run a best-effort hardware probe and return a HardwareProfile."""
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
        vk = _vulkan_only()
        if vk:
            gpus.extend(vk)
        else:
            warnings.append("no vulkaninfo output")

    return HardwareProfile(
        os=f"{platform.system()} {platform.release()}",
        detected_in=source,
        cpu_model=cpu_model,
        cpu_cores=cores,
        cpu_threads=threads,
        ram_bytes=ram,
        gpus=gpus,
        probe_warnings=warnings,
        captured_at=datetime.utcnow().isoformat(),
    )

