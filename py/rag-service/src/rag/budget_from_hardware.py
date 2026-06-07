"""Derives a HardwareAwareRagBudget from the current hardware profile."""
from models.runtime import HardwareAwareRagBudget, HardwareProfile


def compute_budget(profile: HardwareProfile) -> HardwareAwareRagBudget:
    """Map hardware capability to RAG context budget limits.

    Tiers:
    - CPU-only or no GPU: small context to preserve TTFT.
    - ≥ 24 GB VRAM:  large context is safe.
    - ≥ 12 GB VRAM:  medium context.
    - ≥  8 GB VRAM:  moderate context; better retrieval is preferred.
    - < 8 GB VRAM:   keep context tight.

    Returns:
        HardwareAwareRagBudget with limits appropriate for the detected hardware.
    """
    if not profile.gpus:
        return HardwareAwareRagBudget(
            max_retrieved_tokens=6000,
            max_spans=6,
            max_tool_log_tokens=800,
            max_agentic_retrieval_steps=2,
            reason="CPU-only: keep context small to preserve TTFT",
        )

    best = max(profile.gpus, key=lambda g: g.vram_bytes or 0)
    vram_gb = (best.vram_bytes or 0) / (1024**3)

    if vram_gb >= 24:
        return HardwareAwareRagBudget(
            max_retrieved_tokens=32000,
            max_spans=20,
            max_tool_log_tokens=2000,
            max_agentic_retrieval_steps=6,
            reason=f"{vram_gb:.0f}GB VRAM: large context safe",
        )
    if vram_gb >= 12:
        return HardwareAwareRagBudget(
            max_retrieved_tokens=18000,
            max_spans=14,
            max_tool_log_tokens=1500,
            max_agentic_retrieval_steps=4,
            reason=f"{vram_gb:.0f}GB VRAM: medium context",
        )
    if vram_gb >= 8:
        return HardwareAwareRagBudget(
            max_retrieved_tokens=10000,
            max_spans=10,
            max_tool_log_tokens=1000,
            max_agentic_retrieval_steps=3,
            reason=f"{vram_gb:.0f}GB VRAM: prefer better retrieval over longer context",
        )
    return HardwareAwareRagBudget(
        max_retrieved_tokens=6000,
        max_spans=6,
        max_tool_log_tokens=800,
        max_agentic_retrieval_steps=2,
        reason="low VRAM: keep context tight",
    )

