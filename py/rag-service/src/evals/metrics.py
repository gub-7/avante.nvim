"""Retrieval evaluation metrics.

Implements the core IR metrics used to score RAG retrieval quality:
recall@k, precision@k, and mean reciprocal rank (MRR).
"""

from __future__ import annotations


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """Fraction of expected files present in the top-k retrieved results.

    Args:
        retrieved: Ordered list of retrieved file paths/URIs.
        expected: Set of expected file paths/URIs.
        k: Cut-off rank.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if not expected:
        return 0.0
    top = retrieved[:k]
    return sum(1 for f in expected if f in top) / len(expected)


def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """Fraction of top-k retrieved results that are relevant.

    Args:
        retrieved: Ordered list of retrieved file paths/URIs.
        expected: Set of expected file paths/URIs.
        k: Cut-off rank.

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k == 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for f in top if f in expected) / k


def mrr(retrieved: list[str], expected: set[str]) -> float:
    """Mean reciprocal rank of the first relevant result.

    Args:
        retrieved: Ordered list of retrieved file paths/URIs.
        expected: Set of expected file paths/URIs.

    Returns:
        MRR score in [0.0, 1.0]; 0 if no relevant result found.
    """
    for i, f in enumerate(retrieved, 1):
        if f in expected:
            return 1.0 / i
    return 0.0

