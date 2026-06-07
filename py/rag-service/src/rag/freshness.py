"""
Freshness signals for the RAG reranker.

Computes two sets of file URIs from a project root:

* **stale_uris** — files under generated or legacy directories that should
  be demoted in ranking (e.g. ``node_modules/``, ``vendor/``, ``legacy/``).
* **recent_uris** — files with uncommitted or recently staged changes that
  should be promoted (detected via ``git status --porcelain``).

Phase 7 implementation.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from libs.logger import logger

if TYPE_CHECKING:
    from pathlib import Path
from libs.utils import path_to_uri

# ---------------------------------------------------------------------------
# Path markers
# ---------------------------------------------------------------------------

# Number of leading characters occupied by git status XY codes + space
_GIT_STATUS_PREFIX_LEN: int = 3

#: Directories whose contents are always generated or vendored — demote.
GENERATED_MARKERS: tuple[str, ...] = (
    "/node_modules/",
    "/vendor/",
    "/dist/",
    "/build/",
    "/target/",
    "/.venv/",
    "/.tox/",
    "/__pycache__/",
    "/.git/",
    "/.mypy_cache/",
    "/.pytest_cache/",
    "/.ruff_cache/",
)

#: Legacy/deprecated directories whose contents should be demoted.
STALE_MARKERS: tuple[str, ...] = (
    "/legacy/",
    "/deprecated/",
    "/old/",
    "/archive/",
)

_GIT: str | None = shutil.which("git")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command in *cwd* and return stdout, or empty string on failure."""
    if not _GIT:
        return ""
    try:
        result = subprocess.run(
            [_GIT, "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("git command failed in %s: %s", cwd, exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_freshness(base_path: Path) -> tuple[set[str], set[str]]:
    """
    Compute freshness signal sets for the project at *base_path*.

    Inspects ``git status --porcelain`` to identify recently modified files
    (promoted) and walks the directory tree to identify generated/legacy
    files (demoted).

    Args:
        base_path: Absolute path to the project root directory.

    Returns:
        A tuple ``(stale_uris, recent_uris)`` where each element is a set of
        ``file://`` URI strings.  The ``stale`` set should be passed to the
        reranker as-is, the ``recent`` set provides a boost.

    """
    stale: set[str] = set()
    recent: set[str] = set()

    # --- Recent edits via git status ---
    porcelain = _git(["status", "--porcelain"], base_path)
    for line in porcelain.splitlines():
        # git status --porcelain lines: "XY filename" (first 3 chars are status+space)
        if len(line) > _GIT_STATUS_PREFIX_LEN:
            rel = line[_GIT_STATUS_PREFIX_LEN:].strip()
            # Handle rename: "old_name -> new_name"
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[-1]
            full = (base_path / rel).resolve()
            if full.is_file():
                recent.add(path_to_uri(full))

    # --- Stale / generated files via path-component matching ---
    try:
        for p in base_path.rglob("*"):
            if not p.is_file():
                continue
            s = str(p).replace("\\", "/")
            if any(marker in s for marker in GENERATED_MARKERS) or any(
                marker in s for marker in STALE_MARKERS
            ):
                stale.add(path_to_uri(p))
    except (OSError, PermissionError) as exc:
        logger.debug("freshness walk failed under %s: %s", base_path, exc)

    return stale, recent

