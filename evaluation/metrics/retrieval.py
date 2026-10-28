"""Information Retrieval metrics: Recall@k, Precision@k, MRR, HitRate@k, nDCG@k."""
from __future__ import annotations

import math
from typing import Sequence


def _normalize_id(chunk_id: str) -> str:
    """Normalize chunk or section ID for matching (strip whitespace, lowercase)."""
    return chunk_id.strip().lower()


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Calculate Recall@k: fraction of gold relevant documents found in top-k."""
    if not gold_ids or k <= 0:
        return 0.0
    gold_set = {_normalize_id(g) for g in gold_ids}
    retrieved_top_k = {_normalize_id(r) for r in retrieved_ids[:k]}
    hits = len(retrieved_top_k & gold_set)
    return hits / len(gold_set)


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Calculate Precision@k: fraction of top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0
    if not retrieved_ids or not gold_ids:
        return 0.0
    gold_set = {_normalize_id(g) for g in gold_ids}
    top_k_list = [_normalize_id(r) for r in retrieved_ids[:k]]
    hits = sum(1 for r in top_k_list if r in gold_set)
    return hits / k


def mean_reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Calculate Reciprocal Rank (RR): reciprocal of the 1-based rank of the first relevant document."""
    if not gold_ids or not retrieved_ids:
        return 0.0
    gold_set = {_normalize_id(g) for g in gold_ids}
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if _normalize_id(doc_id) in gold_set:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Calculate HitRate@k: 1.0 if at least one gold item appears in top-k, else 0.0."""
    if not gold_ids or not retrieved_ids or k <= 0:
        return 0.0
    gold_set = {_normalize_id(g) for g in gold_ids}
    for doc_id in retrieved_ids[:k]:
        if _normalize_id(doc_id) in gold_set:
            return 1.0
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k (binary relevance)."""
    if not gold_ids or not retrieved_ids or k <= 0:
        return 0.0
    gold_set = {_normalize_id(g) for g in gold_ids}

    # DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if _normalize_id(doc_id) in gold_set:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG@k
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

