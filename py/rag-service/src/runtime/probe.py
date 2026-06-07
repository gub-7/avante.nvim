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


_GFX_RE = re.compile(r"gfx\d+\w*")
_AGENT_STAR_RE = re.compile(r"^\*+$")
_AGENT_HDR_RE = re.compile(r"^Agent\s+\d+", re.IGNORECASE)
_SMI_VRAM_TOTAL_RE = re.compile(
    r"GPU\[(\d+)\].*?VRAM\s+Total\s+Memory\s*\(B\)\s*:\s*(\d+)", re.IGNORECASE
)
_SMI_VRAM_USED_RE = re.compile(
    r"GPU\[(\d+)\].*?VRAM\s+Total\s+Used\s+Memory\s*\(B\)\s*:\s*(\d+)", re.IGNORECASE
)
_SMI_USE_RE = re.compile(r"GPU\[(\d+)\].*?GPU\s+use\s*\(%\)\s*:\s*(\d+)", re.IGNORECASE)


def _parse_rocminfo(out: str) -> list[dict]:
    """Parse rocminfo stdout into a list of per-GPU info dicts.

    Each dict has keys: name, gfx, vram_bytes.
    Only agents that declare ``Device Type: GPU`` (or contain a gfx target)
    are included — this filters out CPU agents reported by rocminfo.
    """
    agents: list[dict] = []
    current: dict | None = None
    in_global_heap = False

    for raw_line in out.splitlines():
        line = raw_line.strip()

        # Separator line between agents (e.g. "********")
        if _AGENT_STAR_RE.match(line):
            if current is not None and (current["is_gpu"] or current["gfx"]):
                agents.append(current)
            current = None
            in_global_heap = False
            continue

        if _AGENT_HDR_RE.match(line):
            current = {"name": "AMD GPU", "gfx": None, "vram_bytes": None, "is_gpu": False}
            in_global_heap = False
            continue

        if current is None:
            continue

        # GPU type flag
        if "Device Type:" in line and "GPU" in line.upper():
            current["is_gpu"] = True

        # gfx target (e.g. "  Name:   gfx906" or "  ISA Info:  gfx906")
        m = _GFX_RE.search(line)
        if m:
            current["gfx"] = m.group(0)

        # Human-readable GPU name
        if "Marketing Name:" in line:
            val = line.split(":", 1)[1].strip()
            if val:
                current["name"] = val

        # VRAM size: only count the first GLOBAL heap (device memory)
        if "Heap Type:" in line:
            in_global_heap = "GLOBAL" in line.upper()

        if in_global_heap and "Size(in bytes):" in line:
            try:
                raw = line.split(":", 1)[1].strip().split()[0]
                current["vram_bytes"] = int(raw)
                in_global_heap = False  # take only the first GLOBAL entry
            except (ValueError, IndexError):
                pass

    # Flush final agent
    if current is not None and (current["is_gpu"] or current["gfx"]):
        agents.append(current)

    return agents


def _rocm_smi_vram() -> tuple[dict[int, int], dict[int, int]]:
    """Return (total_bytes_by_idx, free_bytes_by_idx) from rocm-smi --showmeminfo vram.

    Falls back to empty dicts if rocm-smi is absent or fails.
    """
    out = _run(["rocm-smi", "--showmeminfo", "vram"])
    total: dict[int, int] = {}
    used: dict[int, int] = {}
    for line in out.splitlines():
        mt = _SMI_VRAM_TOTAL_RE.search(line)
        if mt:
            total[int(mt.group(1))] = int(mt.group(2))
        mu = _SMI_VRAM_USED_RE.search(line)
        if mu:
            used[int(mu.group(1))] = int(mu.group(2))
    free = {idx: total[idx] - used.get(idx, 0) for idx in total}
    return total, free


def _amd() -> list[GPUDevice]:
    """Probe AMD GPUs via rocminfo (with per-GPU VRAM from rocm-smi).

    Multiple GPUs are properly enumerated.  Falls back to a minimal
    single-device entry when rocminfo is unavailable.
    """
    rocminfo_out = _run(["rocminfo"])
    if rocminfo_out:
        agent_dicts = _parse_rocminfo(rocminfo_out)
        if agent_dicts:
            vram_total, vram_free = _rocm_smi_vram()
            gpus: list[GPUDevice] = []
            for i, ag in enumerate(agent_dicts):
                # Prefer rocm-smi VRAM data (more reliable); fall back to
                # rocminfo heap size if rocm-smi is absent.
                tb = vram_total.get(i) or ag["vram_bytes"]
                fb = vram_free.get(i)
                if fb is None and tb is not None:
                    fb = tb  # assume fully free if used data absent
                gpus.append(GPUDevice(
                    vendor="amd",
                    name=ag["name"],
                    gfx_target=ag["gfx"],
                    vram_bytes=tb,
                    free_vram_bytes=fb,
                    supports_rocm=True,
                    supports_vulkan=True,
                ))
            return gpus

    # Fallback: rocm-smi --showproductname gives at least device names
    smi_out = _run(["rocm-smi", "--showproductname"])
    if not smi_out:
        return []
    # Attempt to pair product names with VRAM data
    names: list[str] = []
    for line in smi_out.splitlines():
        if "Card series:" in line or "Card model:" in line or "GPU[" in line:
            val = line.split(":", 1)[-1].strip()
            if val:
                names.append(val)
    if not names:
        names = ["AMD GPU"]
    vram_total, vram_free = _rocm_smi_vram()
    return [
        GPUDevice(
            vendor="amd",
            name=names[i] if i < len(names) else f"AMD GPU {i}",
            vram_bytes=vram_total.get(i),
            free_vram_bytes=vram_free.get(i),
            supports_rocm=True,
            supports_vulkan=True,
        )
        for i in range(max(len(names), len(vram_total) or 1))
    ]


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



# ---------------------------------------------------------------------------
# GPU availability helper — cached per process
# ---------------------------------------------------------------------------

_GPU_AVAILABLE: bool | None = None


def gpu_available() -> bool:
    """Return True if at least one NVIDIA GPU is detected on this host.

    The result is cached after the first call so repeated invocations do
    not re-run nvidia-smi.  The cache is intentionally process-scoped;
    hardware topology does not change at runtime.

    Only NVIDIA GPUs are considered here because the Milvus
    GPU_CAGRA index requires CUDA support.  AMD/Vulkan-only devices
    do not qualify.
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        gpus = _nvidia()
        _GPU_AVAILABLE = len(gpus) > 0
    return _GPU_AVAILABLE
