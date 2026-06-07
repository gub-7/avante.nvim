"""Tests for the VRAM estimation logic."""

from __future__ import annotations


def test_70b_fp16_does_not_fit_in_24gb():
    """A 70B FP16 model should require more than 24 GB of VRAM."""
    from runtime.vram_estimator import estimate

    bytes_24gb = 24 * 1024 ** 3
    required = estimate(model_params_b=70.0, quant="fp16", ctx_tokens=4096)
    assert required > bytes_24gb, (
        f"Expected 70B fp16 to exceed 24 GB, but estimate was {required / 1e9:.1f} GB"
    )


def test_small_model_lower_vram():
    """A smaller model should need less VRAM than a larger one (same quant)."""
    from runtime.vram_estimator import estimate

    small = estimate(model_params_b=7.0, quant="q4_0", ctx_tokens=2048)
    large = estimate(model_params_b=70.0, quant="q4_0", ctx_tokens=2048)
    assert small < large


def test_estimate_positive():
    """Estimate must always return a positive number."""
    from runtime.vram_estimator import estimate

    assert estimate(model_params_b=3.0, quant="q4_0", ctx_tokens=1024) > 0

