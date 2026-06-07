"""DB-backed hardware profile store; merges container and host probes."""
import hashlib
import json

from libs.db import get_db_connection
from models.runtime import HardwareProfile


def save_profile(profile: HardwareProfile, source: str = "container") -> None:
    """Persist a hardware profile to the single-row hardware_profile table."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO hardware_profile(id, profile_json, source) VALUES (1, ?, ?)",
            (profile.model_dump_json(), source),
        )
        conn.commit()


def load_profile() -> HardwareProfile | None:
    """Load the cached hardware profile from the DB; returns None if absent."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT profile_json, source FROM hardware_profile WHERE id = 1"
        ).fetchone()
    if not row:
        return None
    p = HardwareProfile.model_validate_json(row["profile_json"])
    p.detected_in = row["source"]
    return p


def hw_hash(profile: HardwareProfile) -> str:
    """Return a short stable hash of the hardware fingerprint (cpu+ram+gpus)."""
    payload = {
        "cpu": profile.cpu_model,
        "ram": profile.ram_bytes,
        "gpus": [(g.vendor, g.name, g.vram_bytes) for g in profile.gpus],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


def get_or_probe() -> HardwareProfile:
    """Return the cached profile; fall back to an in-container probe and cache the result."""
    p = load_profile()
    if p:
        return p
    from runtime.probe import probe  # deferred to avoid circular imports at module load
    p = probe("container")
    save_profile(p, "container")
    return p

