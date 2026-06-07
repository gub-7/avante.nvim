"""High-level composition router that produces a backend recommendation."""
from models.runtime import BackendRecommendation
from runtime.backend_selector import select
from runtime.hardware_profile import get_or_probe
from runtime.performance_modes import load_mode


def recommend(prefer_vendor: str = "auto") -> BackendRecommendation:
    """Return an advisory BackendRecommendation using the current hardware profile."""
    profile = get_or_probe()
    mode = load_mode()
    return select(profile, performance_mode=mode, prefer_vendor=prefer_vendor)

