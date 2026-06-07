"""Project profile builder and cache.

A :class:`ProjectProfile` captures high-level metadata about the project
being served (detected stack, package manager, common commands, etc.).  It
is built lazily on the first retrieval call and cached in the
``project_profiles`` SQLite table so that subsequent calls pay no I/O cost
unless one of the :data:`TRIGGER_FILES` has changed.

Typical usage::

    profile = get_or_build("file:///home/user/myproject")
    if profile:
        print(profile.stack)  # e.g. ["python", "node"]
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from pydantic import BaseModel

from libs.db import get_db_connection
from libs.utils import uri_to_path


class ProjectProfile(BaseModel):
    """High-level metadata inferred from well-known project files."""

    project_name: str
    stack: list[str] = []
    package_manager: str | None = None
    test_commands: list[str] = []
    build_commands: list[str] = []
    lint_commands: list[str] = []
    important_paths: list[str] = []
    generated_paths: list[str] = []
    conventions: list[str] = []
    test_patterns: list[str] = []
    updated_at: str


#: Files whose content is hashed to decide whether the profile is stale.
TRIGGER_FILES: tuple[str, ...] = (
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
    "flake.nix",
    "shell.nix",
    "Dockerfile",
    "Makefile",
    "README.md",
)


def _input_hash(base: object) -> str:  # base: Path
    """Compute a deterministic hash of all present trigger-file contents.

    Only the first 64 KB of each file is included to keep the hash fast.
    """
    h = hashlib.sha256()
    for name in TRIGGER_FILES:
        p = base / name  # type: ignore[operator]
        if p.exists():  # type: ignore[union-attr]
            h.update(name.encode())
            h.update(p.read_bytes()[:64_000])  # type: ignore[union-attr]
    return h.hexdigest()


def get_or_build(resource_uri: str) -> ProjectProfile | None:
    """Return a cached :class:`ProjectProfile` or build and cache a new one.

    The profile is considered fresh as long as none of the
    :data:`TRIGGER_FILES` have changed (detected via a SHA-256 hash of
    their contents).

    Args:
        resource_uri: The ``file://`` URI of the project root.

    Returns:
        A :class:`ProjectProfile`, or ``None`` if the directory does not
        exist.
    """
    base = uri_to_path(resource_uri)
    if not base.exists():
        return None

    new_hash = _input_hash(base)

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT profile_json, profile_hash FROM project_profiles WHERE resource_uri = ?",
            (resource_uri,),
        ).fetchone()

        if row and row["profile_hash"] == new_hash:
            return ProjectProfile.model_validate_json(row["profile_json"])

        profile = _build(base)
        conn.execute(
            """INSERT OR REPLACE INTO project_profiles
               (resource_uri, profile_json, profile_hash)
               VALUES (?, ?, ?)""",
            (resource_uri, profile.model_dump_json(), new_hash),
        )
        conn.commit()
        return profile


def _build(base: object) -> ProjectProfile:  # base: Path
    """Inspect well-known project files and return a :class:`ProjectProfile`.

    Detection rules (all additive, a project can match multiple stacks):

    * ``package.json``   → stack ``node``, npm scripts extracted.
    * ``pyproject.toml`` → stack ``python``, pytest assumed.
    * ``Cargo.toml``     → stack ``rust``, cargo commands assumed.
    * ``go.mod``         → stack ``go``, go test assumed.

    Args:
        base: Resolved :class:`~pathlib.Path` of the project root.

    Returns:
        A freshly constructed :class:`ProjectProfile`.
    """
    stack: list[str] = []
    pm: str | None = None
    tests: list[str] = []
    builds: list[str] = []
    lint: list[str] = []

    pkg_json = base / "package.json"  # type: ignore[operator]
    if pkg_json.exists():  # type: ignore[union-attr]
        stack.append("node")
        pm = "npm"
        try:
            pkg = json.loads(pkg_json.read_text())  # type: ignore[union-attr]
            scripts: dict = pkg.get("scripts") or {}
            if "test" in scripts:
                tests.append("npm test")
            if "build" in scripts:
                builds.append("npm run build")
            if "lint" in scripts:
                lint.append("npm run lint")
        except (OSError, json.JSONDecodeError):
            pass

    if (base / "pyproject.toml").exists():  # type: ignore[operator, union-attr]
        stack.append("python")
        pm = pm or "pip"
        tests.append("pytest")

    if (base / "Cargo.toml").exists():  # type: ignore[operator, union-attr]
        stack.append("rust")
        pm = pm or "cargo"
        tests.append("cargo test")
        builds.append("cargo build")
        lint.append("cargo clippy")

    if (base / "go.mod").exists():  # type: ignore[operator, union-attr]
        stack.append("go")
        tests.append("go test ./...")

    return ProjectProfile(
        project_name=base.name,  # type: ignore[union-attr]
        stack=stack,
        package_manager=pm,
        test_commands=tests,
        build_commands=builds,
        lint_commands=lint,
        important_paths=["src", "lib"],
        generated_paths=["dist", "build", "target", "node_modules", ".venv"],
        conventions=[],
        test_patterns=[],
        updated_at=datetime.utcnow().isoformat(),
    )

