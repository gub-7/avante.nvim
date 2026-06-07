"""RAG engine: index lifecycle, document processing, and resource indexing."""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from libs.logger import logger
from libs.utils import inject_uri_to_node, path_to_uri
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from models.resource import Resource
from rag.chunking import (
    clean_text,
    is_valid_text,
    required_exts,
    scan_directory,
    split_documents,
)
from rag.semantic_search import init_semantic_search
from services.indexing_history import indexing_history_service
from services.resource import resource_service

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext
    from watchdog.observers.api import BaseObserver

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BATCH_PROCESSING_DELAY = 1  # seconds
MAX_WORKERS = multiprocessing.cpu_count()
BATCH_SIZE = 40  # documents per batch

# ---------------------------------------------------------------------------
# Module-level mutable state
# ---------------------------------------------------------------------------

index_lock = threading.Lock()
watched_resources: dict[str, BaseObserver] = {}
file_last_modified: dict[Path, float] = {}

# These are set by init_engine() and consumed by API handlers via get_index()
_index: VectorStoreIndex | None = None
_vector_store: ChromaVectorStore | None = None
_storage_context: StorageContext | None = None


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_index() -> VectorStoreIndex:
    """Return the active VectorStoreIndex (raises if engine not initialised)."""
    if _index is None:
        msg = "RAG engine has not been initialised; call init_engine() first."
        raise RuntimeError(msg)
    return _index


# ---------------------------------------------------------------------------
# Engine initialisation
# ---------------------------------------------------------------------------


def init_engine() -> None:
    """Initialise Chroma + LlamaIndex; must be called once from the lifespan leader."""
    global _index, _vector_store, _storage_context  # noqa: PLW0603

    logger.info("Initialising RAG engine …")
    _index, _vector_store, _storage_context = init_semantic_search()
    logger.info("RAG engine ready.")


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------


def process_document_batch(documents: list[Document]) -> bool:  # noqa: C901, PLR0912, PLR0915
    """Process a batch of documents for embedding.

    Returns True when all documents succeeded, False when any failed.
    """
    try:
        valid_documents: list[Document] = []
        invalid_documents: list[str] = []

        for doc in documents:
            doc_id = doc.doc_id

            # Skip already-indexed documents
            status_records = indexing_history_service.get_indexing_status(doc=doc)
            if status_records and status_records[0].status == "completed":
                logger.debug(
                    "Document with same hash already processed, skipping: %s",
                    doc.doc_id,
                )
                continue

            logger.debug("Processing document: %s", doc.doc_id)
            try:
                content = doc.get_content()

                # Decode bytes if needed
                if isinstance(content, bytes):
                    try:
                        content = content.decode("utf-8", errors="replace")
                    except (UnicodeDecodeError, OSError) as e:
                        error_msg = f"Unable to decode document content: {doc_id}, error: {e!s}"
                        logger.warning(error_msg)
                        indexing_history_service.update_indexing_status(doc, "failed", error_message=error_msg)
                        invalid_documents.append(doc_id)
                        continue

                content = str(content)

                if not is_valid_text(content):
                    error_msg = f"Invalid document content: {doc_id}"
                    logger.warning(error_msg)
                    indexing_history_service.update_indexing_status(doc, "failed", error_message=error_msg)
                    invalid_documents.append(doc_id)
                    continue

                cleaned_content = clean_text(content)
                metadata = getattr(doc, "metadata", {}).copy()

                new_doc = Document(
                    text=cleaned_content,
                    doc_id=doc_id,
                    metadata=metadata,
                )
                inject_uri_to_node(new_doc)
                valid_documents.append(new_doc)
                indexing_history_service.update_indexing_status(doc, "indexing")

            except OSError as e:
                error_msg = f"Document processing failed: {doc_id}, error: {e!s}"
                logger.exception(error_msg)
                indexing_history_service.update_indexing_status(doc, "failed", error_message=error_msg)
                invalid_documents.append(doc_id)

        try:
            if valid_documents:
                with index_lock:
                    get_index().refresh_ref_docs(valid_documents)

            for doc in valid_documents:
                indexing_history_service.update_indexing_status(
                    doc,
                    "completed",
                    metadata=doc.metadata,
                )

            return not invalid_documents

        except OSError as e:
            error_msg = f"Batch indexing failed: {e!s}"
            logger.exception(error_msg)
            for doc in valid_documents:
                indexing_history_service.update_indexing_status(doc, "failed", error_message=error_msg)
            return False

    except OSError as e:
        error_msg = f"Batch processing failed: {e!s}"
        logger.exception(error_msg)
        for doc in documents:
            indexing_history_service.update_indexing_status(doc, "failed", error_message=error_msg)
        return False


# ---------------------------------------------------------------------------
# Single-file incremental update (called by FileSystemHandler)
# ---------------------------------------------------------------------------


def update_index_for_file(directory: Path, abs_file_path: Path) -> None:
    """Update the index for a single file."""
    logger.debug("Starting to index file: %s", abs_file_path)

    if not abs_file_path.is_file():
        logger.debug("File does not exist or is not a file, skipping: %s", abs_file_path)
        return

    rel_file_path = abs_file_path.relative_to(directory)

    from rag.chunking import get_pathspec  # noqa: PLC0415

    spec = get_pathspec(directory)
    if spec and spec.match_file(rel_file_path):
        logger.debug("File is ignored, skipping: %s", abs_file_path)
        return

    resource = resource_service.get_resource(path_to_uri(directory))
    if not resource:
        logger.error("Resource not found for directory: %s", directory)
        return

    resource_service.update_resource_indexing_status(resource.uri, "indexing", "")

    documents = SimpleDirectoryReader(
        input_files=[abs_file_path],
        filename_as_id=True,
        required_exts=required_exts,
    ).load_data()

    logger.debug("Updating index: %s", abs_file_path)
    processed_documents = split_documents(documents)
    success = process_document_batch(processed_documents)

    # Extract and store symbols for this file
    from rag.symbol_index import replace_symbols_for_file  # noqa: PLC0415

    try:
        text = abs_file_path.read_text(encoding="utf-8", errors="ignore")
        replace_symbols_for_file(path_to_uri(abs_file_path), resource.uri, text)
    except OSError:
        pass

    if success:
        resource_service.update_resource_indexing_status(resource.uri, "indexed", "")
        logger.debug("File indexing completed: %s", abs_file_path)
    else:
        resource_service.update_resource_indexing_status(resource.uri, "failed", "unknown error")
        logger.error("File indexing failed: %s", abs_file_path)


# ---------------------------------------------------------------------------
# Async full-resource indexing
# ---------------------------------------------------------------------------


async def index_local_resource_async(resource: Resource) -> None:
    """Asynchronously index a local directory resource."""
    resource_service.update_resource_indexing_status(resource.uri, "indexing", "")
    from libs.utils import uri_to_path  # noqa: PLC0415

    directory_path = uri_to_path(resource.uri)
    try:
        logger.info("Loading directory content: %s", directory_path)

        source_files = scan_directory(directory_path)
        documents = SimpleDirectoryReader(
            input_files=source_files,
            filename_as_id=True,
            required_exts=required_exts,
        ).load_data()

        processed_documents = split_documents(documents)

        logger.info("Found %d documents", len(processed_documents))
        logger.debug("Document list: %s", [doc.doc_id for doc in processed_documents])

        total_documents = len(processed_documents)
        batches = [processed_documents[i : i + BATCH_SIZE] for i in range(0, total_documents, BATCH_SIZE)]
        logger.info("Splitting documents into %d batches for processing", len(batches))

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = await loop.run_in_executor(
                executor,
                lambda: list(executor.map(process_document_batch, batches)),
            )

        # Extract symbols for every source file (best-effort; errors are non-fatal)
        # scan_directory() returns strings, so we wrap each entry in Path.
        from rag.symbol_index import replace_symbols_for_file  # noqa: PLC0415

        for file_path_str in source_files:
            try:
                file_path = Path(file_path_str)
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                replace_symbols_for_file(path_to_uri(file_path), resource.uri, text)
            except OSError:
                pass

        if all(results):
            logger.info("Directory %s indexing completed", directory_path)
            resource_service.update_resource_indexing_status(resource.uri, "indexed", "")
        else:
            failed_batches = len([r for r in results if not r])
            error_msg = f"Some batches failed processing ({failed_batches}/{len(batches)})"
            resource_service.update_resource_indexing_status(resource.uri, "indexed", error_msg)
            logger.error(error_msg)

    except OSError as e:
        error_msg = f"Directory indexing failed: {directory_path}"
        resource_service.update_resource_indexing_status(resource.uri, "failed", error_msg)
        logger.exception(error_msg)
        raise e  # noqa: TRY201


async def index_remote_resource_async(resource: Resource) -> None:
    """Asynchronously index a remote (HTTPS) resource."""
    from rag.remote_fetch import fetch_markdown, markdown_to_links  # noqa: PLC0415

    resource_service.update_resource_indexing_status(resource.uri, "indexing", "")
    url = resource.uri
    try:
        logger.debug("Loading resource content: %s", url)

        markdown = fetch_markdown(url)

        link_md_pairs = [(url, markdown)]

        links = markdown_to_links(url, markdown)

        logger.debug("Found %d sub links", len(links))
        logger.debug("Link list: %s", links)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            mds: list[str] = await loop.run_in_executor(
                executor,
                lambda: list(executor.map(fetch_markdown, links)),
            )

        link_md_pairs.extend(zip(links, mds, strict=True))  # pyright: ignore

        documents = [Document(text=markdown, doc_id=link) for link, markdown in link_md_pairs]

        logger.debug("Found %d documents", len(documents))
        logger.debug("Document list: %s", [doc.doc_id for doc in documents])

        total_documents = len(documents)
        batches = [documents[i : i + BATCH_SIZE] for i in range(0, total_documents, BATCH_SIZE)]
        logger.debug("Splitting documents into %d batches for processing", len(batches))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = await loop.run_in_executor(
                executor,
                lambda: list(executor.map(process_document_batch, batches)),
            )

        if all(results):
            logger.debug("Resource %s indexing completed", url)
            resource_service.update_resource_indexing_status(resource.uri, "indexed", "")
        else:
            failed_batches = len([r for r in results if not r])
            error_msg = f"Some batches failed processing ({failed_batches}/{len(batches)})"
            logger.error(error_msg)
            resource_service.update_resource_indexing_status(resource.uri, "indexed", error_msg)

    except OSError as e:
        error_msg = f"Resource indexing failed: {url}"
        logger.exception(error_msg)
        resource_service.update_resource_indexing_status(resource.uri, "failed", error_msg)
        raise e  # noqa: TRY201

