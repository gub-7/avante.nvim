"""Generates llama.cpp-compatible GPU offload plans based on hardware and model fit."""
from models.runtime import HardwareProfile, ModelRuntimePlan, OffloadPlan


def plan_offload(model_plan: ModelRuntimePlan, profile: HardwareProfile) -> OffloadPlan:
    """Return an OffloadPlan describing how many layers to place on GPU(s).

    Strategies:
    - No GPU or model doesn't fit: keep everything on CPU (gpu_layers=0).
    - Single GPU and model fits: offload all layers to GPU 0.
    - Multiple GPUs and model fits: split layers across all GPUs weighted by VRAM size.
    """
    if not profile.gpus or not model_plan.fits_in_vram:
        return OffloadPlan(
            gpu_layers=0,
            context_size=model_plan.context_tokens,
            batch_size=model_plan.batch_size,
            split_mode="none",
        )

    if len(profile.gpus) == 1:
        return OffloadPlan(
            gpu_layers="all",
            main_gpu=0,
            context_size=model_plan.context_tokens,
            batch_size=model_plan.batch_size,
            split_mode="none",
        )

    # Multi-GPU: split proportionally by VRAM
    sizes = [g.vram_bytes or 0 for g in profile.gpus]
    total = sum(sizes) or 1
    split = [round(s / total, 3) for s in sizes]
    main = max(range(len(sizes)), key=lambda i: sizes[i])
    return OffloadPlan(
        gpu_layers="all",
        main_gpu=main,
        tensor_split=split,
        split_mode="layer",
        context_size=model_plan.context_tokens,
        batch_size=model_plan.batch_size,
    )

