"""Multi-GPU deployment strategy planner."""
from models.runtime import HardwareProfile


def strategy(
    profile: HardwareProfile,
    model_fits_one: bool,
    throughput_oriented: bool,
) -> tuple[str, str]:
    """Return a (strategy_name, rationale) pair for the given hardware and model fit.

    Strategies:
    - single_gpu: only one GPU present or no GPU.
    - data_parallel: model fits on each GPU and throughput is priority.
    - tensor_parallel: multiple equal-sized GPUs, model too large for one.
    - layer_split: mismatched GPUs; route more layers to the largest.
    """
    gs = profile.gpus
    if not gs:
        return "single_gpu", "no GPU"
    if len(gs) == 1:
        return "single_gpu", "one GPU"
    if model_fits_one and throughput_oriented:
        return "data_parallel", "many requests; replicate per GPU"
    if not model_fits_one:
        sizes = sorted((g.vram_bytes or 0) for g in gs)
        if sizes[-1] >= 2 * (sizes[0] or 1):
            return "layer_split", "mismatched GPUs; use largest as main"
        return "tensor_parallel", "equal GPUs; tensor parallel"
    return "single_gpu", "default"

