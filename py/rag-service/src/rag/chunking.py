"""Document chunking, splitting, and directory-scanning utilities."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pathspec
from libs.logger import logger
from libs.utils import get_node_uri, inject_uri_to_node, is_path_node, uri_to_path
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.schema import Document
from tree_sitter_language_pack import SupportedLanguage, get_parser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.95
MAX_SAMPLE_SIZE = 100

code_ext_map: dict[str, SupportedLanguage] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".vue": "vue",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "cpp",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".scala": "scala",
    ".kt": "kotlin",
    ".swift": "swift",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
    ".t": "perl",
    ".pm6": "perl",
    ".m": "perl",
}

required_exts = [
    ".txt",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".rst",
    ".json",
    ".ini",
    ".conf",
    ".toml",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".xml",
    ".yaml",
    ".yml",
    ".css",
    ".scss",
    ".less",
    ".sass",
    ".styl",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".rb",
    ".java",
    ".go",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".vue",
    ".py",
    ".php",
    ".c",
    ".cpp",
    ".h",
    ".rs",
    ".swift",
    ".kt",
    ".lua",
    ".perl",
    ".pl",
    ".pm",
    ".t",
    ".pm6",
    ".m",
]

binary_extensions = [
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".tiff",
    ".exr",
    ".hdr",
    ".svg",
    ".psd",
    ".ai",
    ".eps",
    # Audio/Video
    ".mp3",
    ".wav",
    ".mp4",
    ".avi",
    ".mov",
    ".webm",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".flv",
    ".mkv",
    ".wmv",
    # Documents
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".iso",
    ".dmg",
    ".pkg",
    ".deb",
    ".rpm",
    ".msi",
    ".apk",
    ".xz",
    ".bz2",
    # Compiled
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".class",
    ".pyc",
    ".o",
    ".obj",
    ".lib",
    ".a",
    ".out",
    ".app",
    ".jar",
    # Fonts
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".eot",
    # Other binary
    ".bin",
    ".dat",
    ".db",
    ".sqlite",
    ".DS_Store",
]


# ---------------------------------------------------------------------------
# Text validation helpers
# ---------------------------------------------------------------------------


def is_valid_text(text: str) -> bool:
    """Check if the text is valid and readable."""
    if not text:
        logger.debug("Text content is empty")
        return False

    # Check if the text mainly contains printable characters
    printable_ratio = sum(1 for c in text if c.isprintable() or c in "\n\r\t") / len(text)
    if printable_ratio <= SIMILARITY_THRESHOLD:
        logger.debug("Printable character ratio too low: %.2f%%", printable_ratio * 100)
        # Output a small sample for analysis
        sample = text[:MAX_SAMPLE_SIZE] if len(text) > MAX_SAMPLE_SIZE else text
        logger.debug("Text sample: %r", sample)
    return printable_ratio > SIMILARITY_THRESHOLD


def clean_text(text: str) -> str:
    """Clean text content by removing non-printable characters."""
    return "".join(char for char in text if char.isprintable() or char in "\n\r\t")


# ---------------------------------------------------------------------------
# Gitignore / git-crypt helpers
# ---------------------------------------------------------------------------


def get_gitignore_files(directory: Path) -> list[str]:
    """Get patterns from .gitignore file."""
    patterns = []

    # Always include .git/ if it exists
    if (directory / ".git").is_dir():
        patterns.append(".git/")

    # Check for .gitignore
    gitignore_path = directory / ".gitignore"
    if gitignore_path.exists():
        with gitignore_path.open("r", encoding="utf-8") as f:
            patterns.extend(f.readlines())

    return patterns


def get_gitcrypt_files(directory: Path) -> list[str]:
    """Get patterns of git-crypt encrypted files using git command."""
    git_crypt_patterns: list[str] = []
    git_executable = shutil.which("git")

    if not git_executable:
        logger.warning("git command not found, git-crypt files will not be excluded")
        return git_crypt_patterns

    try:
        # Find git root directory
        git_root_cmd = subprocess.run(
            [git_executable, "-C", str(directory), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )

        if git_root_cmd.returncode != 0:
            logger.warning(
                "Not a git repository or git command failed: %s",
                git_root_cmd.stderr.strip(),
            )
            return git_crypt_patterns

        git_root = Path(git_root_cmd.stdout.strip())

        # Get relative path from git root to our directory
        rel_path = directory.relative_to(git_root) if directory != git_root else Path()

        # Execute git commands separately and pipe the results
        git_ls_files = subprocess.run(
            [git_executable, "-C", str(git_root), "ls-files", "-z"],
            capture_output=True,
            text=False,
            check=False,
        )

        if git_ls_files.returncode != 0:
            return git_crypt_patterns

        # Use Python to process the output instead of xargs, grep, and cut
        git_check_attr = subprocess.run(
            [
                git_executable,
                "-C",
                str(git_root),
                "check-attr",
                "filter",
                "--stdin",
                "-z",
            ],
            input=git_ls_files.stdout,
            capture_output=True,
            text=False,
            check=False,
        )

        if git_check_attr.returncode != 0:
            return git_crypt_patterns

        # Process the output in Python to find git-crypt files
        output = git_check_attr.stdout.decode("utf-8")
        lines = output.split("\0")

        for i in range(0, len(lines) - 2, 3):
            if i + 2 < len(lines) and lines[i + 2] == "git-crypt":
                file_path = lines[i]
                # Only include files that are in our directory or subdirectories
                file_path_obj = Path(file_path)
                if str(rel_path) == "." or file_path_obj.is_relative_to(rel_path):
                    git_crypt_patterns.append(file_path)

        # Log if git-crypt patterns were found
        if git_crypt_patterns:
            logger.debug("Excluding git-crypt encrypted files: %s", git_crypt_patterns)
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Error getting git-crypt files: %s", str(e))

    return git_crypt_patterns


def get_pathspec(directory: Path) -> pathspec.PathSpec | None:
    """Get pathspec for the directory."""
    # Collect patterns from both sources
    patterns = get_gitignore_files(directory)
    patterns.extend(get_gitcrypt_files(directory))

    return pathspec.GitIgnoreSpec.from_lines(patterns)


def scan_directory(directory: Path) -> list[str]:
    """Scan directory and return a list of matched files."""
    spec = get_pathspec(directory)

    matched_files = []

    for root, _, files in os.walk(directory):
        file_paths = [str(Path(root) / file) for file in files]
        for file in file_paths:
            file_ext = Path(file).suffix.lower()
            if file_ext in binary_extensions:
                logger.debug("Skipping binary file: %s", file)
                continue

            if spec and spec.match_file(os.path.relpath(file, directory)):
                logger.debug("Ignoring file: %s", file)
            else:
                matched_files.append(file)

    return matched_files


# ---------------------------------------------------------------------------
# Structural chunking constants
# ---------------------------------------------------------------------------

# Maximum characters before a config file falls back to byte-window splits
MAX_CHUNK_CHARS = 1500

# ATX heading pattern (matches lines like "# Title", "## Section", etc.)
ATX_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)

# Extensions treated as config files (whole-file or byte-window chunks)
CONFIG_EXTS: frozenset[str] = frozenset({".toml", ".yaml", ".yml", ".json", ".ini", ".conf"})

# Extensions treated as markdown documents (section-split on ATX headings)
MARKDOWN_EXTS: frozenset[str] = frozenset({".md", ".markdown"})


# ---------------------------------------------------------------------------
# Document splitting
# ---------------------------------------------------------------------------


def _line_bounds_of_chunk(full_text: str, chunk_text: str, prev_end: int) -> tuple[int, int]:
    """Find 1-indexed start/end lines of chunk_text inside full_text.

    Search starts at the line corresponding to *prev_end* to handle
    overlapping CodeSplitter chunks correctly.
    """
    if not chunk_text:
        return prev_end, prev_end
    full_lines = full_text.splitlines()
    chunk_lines = chunk_text.splitlines()
    if not chunk_lines:
        return prev_end, prev_end
    head = chunk_lines[0]
    start_search = max(0, prev_end - 1)
    for i in range(start_search, len(full_lines)):
        if full_lines[i] == head:
            return i + 1, min(len(full_lines), i + len(chunk_lines))
    # Fallback: assume the chunk begins right after the previous one
    return prev_end, prev_end + len(chunk_lines)


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into typed chunks with structural metadata.

    Each output chunk carries ``start_line``, ``end_line``, ``text_hash``,
    ``chunk_kind``, and ``symbols`` metadata regardless of the file type.

    Dispatch logic:
    - Code files (``code_ext_map``): CodeSplitter → ``chunk_kind="code"``
    - Markdown files (``MARKDOWN_EXTS``): ATX-heading split → ``chunk_kind="section"``
    - Config files (``CONFIG_EXTS``): whole-file or byte-window → ``chunk_kind="config"``
    - Everything else: passed through unchanged (with ``orig_doc_id`` set).
    """
    processed_documents: list[Document] = []
    for doc in documents:
        uri = get_node_uri(doc)
        if not uri:
            continue
        if not is_path_node(doc):
            processed_documents.append(doc)
            continue
        file_path = uri_to_path(uri)
        file_ext = file_path.suffix.lower()

        # ------------------------------------------------------------------
        # Branch 1: Code files — CodeSplitter with line/hash metadata
        # ------------------------------------------------------------------
        if file_ext in code_ext_map:
            language = code_ext_map.get(file_ext, "python")
            parser = get_parser(language)
            code_splitter = CodeSplitter(
                language=language,
                chunk_lines=80,
                chunk_lines_overlap=15,
                max_chars=1500,
                parser=parser,
            )
            try:
                t = doc.get_content()
                texts = code_splitter.split_text(t)
            except ValueError as e:
                logger.error(
                    "Error splitting document: %s, so skipping split, error: %s",
                    doc.doc_id,
                    str(e),
                )
                processed_documents.append(doc)
                continue

            prev_end = 0
            for i, text in enumerate(texts):
                start_line, end_line = _line_bounds_of_chunk(t, text, prev_end)
                prev_end = end_line
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                new_doc = Document(
                    text=text,
                    doc_id=f"{doc.doc_id}__part_{i}",
                    metadata={
                        **doc.metadata,
                        "chunk_number": i,
                        "total_chunks": len(texts),
                        "language": code_splitter.language,
                        "orig_doc_id": doc.doc_id,
                        # Phase 3 structural metadata
                        "chunk_kind": "code",   # refined to function/class/test in Phase 4
                        "symbols": [],           # populated in Phase 4
                        "start_line": start_line,
                        "end_line": end_line,
                        "text_hash": text_hash,
                    },
                )
                inject_uri_to_node(new_doc)
                processed_documents.append(new_doc)

        # ------------------------------------------------------------------
        # Branch 2: Markdown files — split on ATX headings into sections
        # ------------------------------------------------------------------
        elif file_ext in MARKDOWN_EXTS:
            t = doc.get_content()
            lines = t.splitlines()
            # Collect sections: each starts at an ATX heading line
            sections: list[tuple[int, list[str]]] = []
            current_start = 0
            current_lines: list[str] = []
            for line_idx, line in enumerate(lines):
                if ATX_HEADING_RE.match(line) and current_lines:
                    # Flush the accumulated section before this heading
                    sections.append((current_start, current_lines))
                    current_start = line_idx
                    current_lines = [line]
                else:
                    current_lines.append(line)
            if current_lines:
                sections.append((current_start, current_lines))

            if not sections:
                # No headings found — pass through as a single chunk
                doc.metadata["orig_doc_id"] = doc.doc_id
                processed_documents.append(doc)
                continue

            total_sections = len(sections)
            for i, (start_line_0, sect_lines) in enumerate(sections):
                text = "\n".join(sect_lines)
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                new_doc = Document(
                    text=text,
                    doc_id=f"{doc.doc_id}__part_{i}",
                    metadata={
                        **doc.metadata,
                        "chunk_number": i,
                        "total_chunks": total_sections,
                        "orig_doc_id": doc.doc_id,
                        # Phase 3 structural metadata
                        "chunk_kind": "section",
                        "symbols": [],
                        "start_line": start_line_0 + 1,  # convert to 1-indexed
                        "end_line": start_line_0 + len(sect_lines),
                        "text_hash": text_hash,
                    },
                )
                inject_uri_to_node(new_doc)
                processed_documents.append(new_doc)

        # ------------------------------------------------------------------
        # Branch 3: Config files — whole-file chunk or byte-window fallback
        # ------------------------------------------------------------------
        elif file_ext in CONFIG_EXTS:
            t = doc.get_content()
            if len(t) <= MAX_CHUNK_CHARS:
                # Emit the entire file as a single chunk
                all_lines = t.splitlines()
                text_hash = hashlib.sha256(t.encode()).hexdigest()
                new_doc = Document(
                    text=t,
                    doc_id=f"{doc.doc_id}__part_0",
                    metadata={
                        **doc.metadata,
                        "chunk_number": 0,
                        "total_chunks": 1,
                        "orig_doc_id": doc.doc_id,
                        # Phase 3 structural metadata
                        "chunk_kind": "config",
                        "symbols": [],
                        "start_line": 1,
                        "end_line": len(all_lines) if all_lines else 1,
                        "text_hash": text_hash,
                    },
                )
                inject_uri_to_node(new_doc)
                processed_documents.append(new_doc)
            else:
                # Byte-window fallback: accumulate lines until ~MAX_CHUNK_CHARS
                all_lines = t.splitlines()
                chunks: list[tuple[int, list[str]]] = []
                current_chunk_lines: list[str] = []
                current_len = 0
                chunk_start_line = 0
                for line_idx, line in enumerate(all_lines):
                    line_len = len(line) + 1  # +1 for the newline character
                    if current_len + line_len > MAX_CHUNK_CHARS and current_chunk_lines:
                        chunks.append((chunk_start_line, current_chunk_lines))
                        current_chunk_lines = [line]
                        current_len = line_len
                        chunk_start_line = line_idx
                    else:
                        current_chunk_lines.append(line)
                        current_len += line_len
                if current_chunk_lines:
                    chunks.append((chunk_start_line, current_chunk_lines))

                total_chunks = len(chunks)
                for i, (start_line_0, chunk_lines) in enumerate(chunks):
                    text = "\n".join(chunk_lines)
                    text_hash = hashlib.sha256(text.encode()).hexdigest()
                    new_doc = Document(
                        text=text,
                        doc_id=f"{doc.doc_id}__part_{i}",
                        metadata={
                            **doc.metadata,
                            "chunk_number": i,
                            "total_chunks": total_chunks,
                            "orig_doc_id": doc.doc_id,
                            # Phase 3 structural metadata
                            "chunk_kind": "config",
                            "symbols": [],
                            "start_line": start_line_0 + 1,  # convert to 1-indexed
                            "end_line": start_line_0 + len(chunk_lines),
                            "text_hash": text_hash,
                        },
                    )
                    inject_uri_to_node(new_doc)
                    processed_documents.append(new_doc)

        # ------------------------------------------------------------------
        # Branch 4: Everything else — pass through unchanged
        # ------------------------------------------------------------------
        else:
            doc.metadata["orig_doc_id"] = doc.doc_id
            processed_documents.append(doc)

    return processed_documents

