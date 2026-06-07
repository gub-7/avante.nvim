"""Retrieval endpoint: POST /api/v1/retrieve."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from libs.logger import logger
from libs.utils import get_node_uri, is_local_uri, is_path_node, uri_to_path
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from pydantic import BaseModel, Field
from rag.chunking import clean_text, is_valid_text
from rag.engine import get_index

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore, QueryBundle

router = APIRouter(prefix="/api/v1", tags=["retrieve"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SourceDocument(BaseModel):
    """Model for source document information."""

    uri: str = Field(..., description="URI of the source")
    content: str = Field(..., description="Content snippet from the document")
    score: float | None = Field(None, description="Relevance score of the document")


class RetrieveRequest(BaseModel):
    """Request model for information retrieval."""

    query: str = Field(
        ...,
        description="The query text to search for in the indexed documents",
    )
    base_uri: str = Field(..., description="The base URI to search in")
    top_k: int | None = Field(5, description="Number of top results to return", ge=1, le=20)


class RetrieveResponse(BaseModel):
    """Response model for information retrieval."""

    response: str = Field(..., description="Generated response to the query")
    sources: list[SourceDocument] = Field(..., description="List of source documents used")


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve information from indexed documents",
    description=(
        "Performs a semantic search over all indexed documents and returns relevant information. "
        "The response includes both the answer and the source documents used to generate it."
    ),
    responses={
        200: {"description": "Successfully retrieved information"},
        500: {"description": "Internal server error during retrieval"},
    },
)
async def retrieve(request: RetrieveRequest):  # noqa: ANN201, C901, PLR0915
    """Semantic retrieval handler."""
    if is_local_uri(request.base_uri):
        directory = uri_to_path(request.base_uri)
        if not directory.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {request.base_uri}")

    logger.info(
        "Received retrieval request: %s for base uri: %s",
        request.query,
        request.base_uri,
    )

    cached_file_contents: dict = {}

    # -----------------------------------------------------------------------
    # Directory-scoped filter
    # -----------------------------------------------------------------------
    def filter_documents(node: NodeWithScore) -> bool:
        uri = get_node_uri(node.node)
        if not uri:
            return False
        if is_path_node(node.node):
            file_path = uri_to_path(uri)
            file_path = file_path.resolve()
            directory = uri_to_path(request.base_uri).resolve()
            try:
                file_path.relative_to(directory)
                if not file_path.exists():
                    logger.warning("File not found: %s", file_path)
                    return False
                content = cached_file_contents.get(file_path)
                if content is None:
                    with file_path.open("r", encoding="utf-8") as f:
                        content = f.read()
                        cached_file_contents[file_path] = content
                if node.node.get_content() not in content:
                    logger.warning("File content does not match: %s", file_path)
                    return False
                return True
            except ValueError:
                return False
        if uri == request.base_uri:
            return True
        base_uri = request.base_uri
        if not base_uri.endswith(os.path.sep):
            base_uri += os.path.sep
        return uri.startswith(base_uri)

    # -----------------------------------------------------------------------
    # Post-processor that uses the filter closure above
    # -----------------------------------------------------------------------
    class ResourceFilterPostProcessor(MetadataReplacementPostProcessor):
        """Post-processor for filtering nodes based on directory."""

        def __init__(self: ResourceFilterPostProcessor) -> None:
            """Initialise the post-processor."""
            super().__init__(target_metadata_key="filtered")

        def postprocess_nodes(
            self: ResourceFilterPostProcessor,
            nodes: list[NodeWithScore],
            query_bundle: QueryBundle | None = None,  # noqa: ARG002, pyright: ignore
            query_str: str | None = None,  # noqa: ARG002, pyright: ignore
        ) -> list[NodeWithScore]:
            """Filter nodes based on directory path."""
            return [node for node in nodes if filter_documents(node)]

    # -----------------------------------------------------------------------
    # Execute query
    # -----------------------------------------------------------------------
    query_engine = get_index().as_query_engine(
        node_postprocessors=[ResourceFilterPostProcessor()],
    )

    logger.info("Executing retrieval query")
    response = query_engine.query(request.query)

    if not response.source_nodes:
        raise HTTPException(
            status_code=404,
            detail=f"No relevant documents found in uri: {request.base_uri}",
        )

    # -----------------------------------------------------------------------
    # Build sources list
    # -----------------------------------------------------------------------
    sources = []
    for node in response.source_nodes[: request.top_k]:
        try:
            content = node.node.get_content()
            uri = get_node_uri(node.node)

            if isinstance(content, bytes):
                try:
                    content = content.decode("utf-8", errors="replace")
                except UnicodeDecodeError as e:
                    logger.warning(
                        "Unable to decode document content: %s, error: %s",
                        uri,
                        str(e),
                    )
                    continue

            if is_valid_text(str(content)):
                cleaned_content = clean_text(str(content))
                sources.append(
                    {
                        "uri": uri,
                        "content": cleaned_content,
                        "score": float(node.score) if node.score is not None else None,
                    }
                )
            else:
                logger.warning("Skipping invalid document content: %s", uri)

        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("Error processing source document", exc_info=True)
            continue

    logger.info("Retrieval completed, found %d relevant documents", len(sources))

    response_text = str(response)
    response_text = "".join(char for char in response_text if char.isprintable() or char in "\n\r\t")

    return {
        "response": response_text,
        "sources": sources,
    }

