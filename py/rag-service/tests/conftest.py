"""Shared pytest fixtures for the RAG service test suite."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Redirect all data I/O to a fresh temp directory for each test.

    Reloads ``libs.configs`` and ``libs.db`` so that module-level path
    constants pick up the new ``DATA_DIR`` value, then initialises the
    SQLite schema from scratch.

    Yields:
        The temporary ``Path`` object used as ``DATA_DIR``.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Re-import to pick up new BASE_DATA_DIR
    import libs.configs
    import libs.db

    importlib.reload(libs.configs)
    importlib.reload(libs.db)
    libs.db.init_db()
    yield tmp_path


@pytest.fixture
def fake_repo(tmp_path):
    """Create a minimal fake git repository for indexing tests.

    The repository has a ``.git`` directory and two Python source files
    inside ``src/``:

    - ``src/main.py`` — defines ``foo()`` and ``class Bar``.
    - ``src/test_main.py`` — a trivial pytest test.

    Returns:
        The root ``Path`` of the fake repository (same as ``tmp_path``).
    """
    (tmp_path / ".git").mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "def foo():\n    return 1\n\nclass Bar:\n    def baz(self):\n        return 2\n"
    )
    (src / "test_main.py").write_text("def test_foo():\n    assert foo() == 1\n")
    return tmp_path

