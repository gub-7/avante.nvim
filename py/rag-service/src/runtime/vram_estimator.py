"""VRAM estimation and model runtime planning based on quantisation and context size."""
from models.runtime import HardwareProfile, ModelRuntimePlan

# Bits-per-parameter per quantisation scheme
QUANT_BPP: dict[str, float] = {
    "q4_0": 4.5,
    "q4_k_m": 4.8,
    "q5_k_m": 5.6,
    "q6_k": 6.6,
    "q8_0": 8.5,
    "fp16": 16.0,
    "bf16": 16.0,
}


def estimate(
    model_params_b: float,
    quant: str,
    ctx_tokens: int,
    layers: int = 32,
    hidden: int = 4096,
    kv_heads: int = 32,
    batch: int = 1,
    concurrent: int = 1,
) -> int:
    """Return the estimated VRAM requirement in bytes.

    Args:
        model_params_b: Model parameter count in billions (e.g. 7.0 for a 7B model).
        quant: Quantisation scheme string (must be a key in QUANT_BPP, else 5.0 bpp).
        ctx_tokens: Context window size in tokens.
        layers: Number of transformer layers.
        hidden: Hidden state size.
        kv_heads: Number of KV attention heads.
        batch: Batch size.
        concurrent: Expected concurrent requests.

    Returns:
        Total estimated VRAM in bytes (model weights + KV cache + overhead + safety margin).
    """
    bpp = QUANT_BPP.get(quant, 5.0)
    model_bytes = int(model_params_b * 1e9 * bpp / 8)
    # KV cache: 2 * layers * hidden * ctx * 2 bytes (fp16) * batch * concurrent
    kv_bytes = 2 * layers * hidden * ctx_tokens * 2 * batch * concurrent
    overhead = 600 * 1024 * 1024        # 600 MB fixed overhead
    safety = int(1.5 * 1024**3)         # 1.5 GB safety margin
    return model_bytes + kv_bytes + overhead + safety


def plan(
    model_name: str,
    params_b: float,
    quant: str,
    ctx_tokens: int,
    profile: HardwareProfile,
    batch: int = 1,
    concurrent: int = 1,
) -> ModelRuntimePlan:
    """Build a ModelRuntimePlan for the given model and hardware profile."""
    bpp = QUANT_BPP.get(quant, 5.0)
    model_bytes = int(params_b * 1e9 * bpp / 8)
    required = estimate(params_b, quant, ctx_tokens, batch=batch, concurrent=concurrent)
    largest_vram = (
        max((g.vram_bytes or 0) for g in profile.gpus) if profile.gpus else 0
    )
    fits = required <= largest_vram
    rec = (
        "full GPU offload"
        if fits
        else (
            "reduce RAG context first; then batch; then smaller model/quant; "
            "CPU offload only if allowed"
        )
    )
    return ModelRuntimePlan(
        model_name=model_name,
        quantization=quant,
        model_bytes=model_bytes,
        context_tokens=ctx_tokens,
        batch_size=batch,
        expected_concurrent_requests=concurrent,
        kv_cache_bytes_estimate=required - model_bytes,
        required_vram_bytes=required,
        fits_in_vram=fits,
        recommendation=rec,
    )

