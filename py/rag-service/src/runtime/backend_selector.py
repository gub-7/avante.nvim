"""Selects the optimal LLM backend based on detected hardware and performance mode."""
from models.runtime import BackendRecommendation, HardwareProfile


def select(
    profile: HardwareProfile,
    performance_mode: str = "balanced",
    prefer_vendor: str = "auto",
) -> BackendRecommendation:
    """Return an advisory BackendRecommendation for the given hardware profile.

    The recommendation is purely advisory — the service never spawns LLM backends.
    """
    nvidia = [g for g in profile.gpus if g.vendor == "nvidia"]
    amd = [g for g in profile.gpus if g.vendor == "amd"]

    if (prefer_vendor in {"auto", "nvidia"}) and nvidia:
        if len(nvidia) > 1 and performance_mode != "quiet":
            return BackendRecommendation(
                backend="vllm",
                accelerator="cuda",
                reason="multi-GPU NVIDIA; vLLM/SGLang for tensor parallel",
                env={
                    "CUDA_VISIBLE_DEVICES": ",".join(
                        g.uuid or str(i) for i, g in enumerate(nvidia)
                    )
                },
                risk="medium",
            )
        return BackendRecommendation(
            backend="ollama",
            accelerator="cuda",
            reason="single NVIDIA GPU; Ollama/llama.cpp ideal for dev",
            env={"CUDA_VISIBLE_DEVICES": nvidia[0].uuid or "0"},
            risk="low",
        )

    if (prefer_vendor in {"auto", "amd"}) and amd:
        primary = amd[0]
        if primary.supports_rocm:
            return BackendRecommendation(
                backend="llama.cpp",
                accelerator="rocm",
                reason="AMD ROCm available",
                env={"ROCR_VISIBLE_DEVICES": "0"},
                risk="low",
            )
        if primary.supports_vulkan:
            return BackendRecommendation(
                backend="llama.cpp",
                accelerator="vulkan",
                reason="AMD without ROCm; falling back to Vulkan",
                env={},
                risk="medium",
            )

    return BackendRecommendation(
        backend="llama.cpp",
        accelerator="cpu",
        reason="no GPU detected; use small quantized model",
        env={},
        risk="medium",
    )

