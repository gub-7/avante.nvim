"""Tests for the hardware probe module."""

from __future__ import annotations


def test_probe_returns_profile():
    """probe() must return a HardwareProfile with positive cpu_threads and ram."""
    from runtime.probe import probe

    profile = probe(source="test")
    assert profile.cpu_threads > 0, f"Expected cpu_threads > 0, got {profile.cpu_threads}"
    assert profile.ram_bytes > 0, f"Expected ram_bytes > 0, got {profile.ram_bytes}"


def test_probe_no_exception():
    """probe() must not raise any exception even when GPU tools are absent."""
    from runtime.probe import probe

    try:
        probe(source="test")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"probe() raised unexpectedly: {exc}")


import pytest  # noqa: E402  (import after test functions to satisfy linters)

