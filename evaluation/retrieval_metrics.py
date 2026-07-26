import math
from typing import Any


def hit_rate_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int = 5,
) -> float:
    """
    Computes Hit Rate@K.
    Returns 1.0 if at least one expected source ID appears in top K retrieved IDs.
    For unanswerable queries (expected_ids = []), returns 1.0.
    """
    if not expected_ids:
        return 1.0

    top_k = retrieved_ids[:k] if retrieved_ids else []
    expected_set = set(expected_ids)

    for item in top_k:
        if item in expected_set:
            return 1.0
    return 0.0


def precision_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int = 5,
) -> float:
    """
    Computes Precision@K.
    Returns fraction of top K retrieved IDs that are relevant (in expected_ids).
    For unanswerable queries (expected_ids = []), returns 1.0.
    """
    if not expected_ids:
        return 1.0

    top_k = retrieved_ids[:k] if retrieved_ids else []
    if not top_k:
        return 0.0

    expected_set = set(expected_ids)
    hits = sum(1 for item in top_k if item in expected_set)
    return hits / float(len(top_k))


def recall_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int = 5,
) -> float:
    """
    Computes Recall@K.
    Returns fraction of expected source IDs present in top K retrieved IDs.
    For unanswerable queries (expected_ids = []), returns 1.0.
    """
    if not expected_ids:
        return 1.0

    top_k = retrieved_ids[:k] if retrieved_ids else []
    expected_set = set(expected_ids)
    hits = len(set(top_k) & expected_set)

    return hits / float(len(expected_set))


def reciprocal_rank(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> float:
    """
    Computes Reciprocal Rank (RR).
    Returns 1 / rank of the first relevant retrieved ID (1-indexed).
    For unanswerable queries (expected_ids = []), returns 1.0.
    If no expected IDs retrieved, returns 0.0.
    """
    if not expected_ids:
        return 1.0

    expected_set = set(expected_ids)
    for rank, item in enumerate(retrieved_ids or [], start=1):
        if item in expected_set:
            return 1.0 / float(rank)

    return 0.0


def mean_reciprocal_rank(rr_scores: list[float]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) across a list of RR scores.
    Returns average RR score, or 0.0 if empty list.
    """
    if not rr_scores:
        return 0.0
    return float(sum(rr_scores)) / float(len(rr_scores))


def ndcg_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int = 5,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance.
    For unanswerable queries (expected_ids = []), returns 1.0.
    """
    if not expected_ids:
        return 1.0

    top_k = retrieved_ids[:k] if retrieved_ids else []
    expected_set = set(expected_ids)

    # DCG calculation
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        rel = 1.0 if item in expected_set else 0.0
        dcg += rel / math.log2(i + 1.0)

    # Ideal DCG calculation (IDCG)
    idcg = 0.0
    ideal_hits = min(k, len(expected_set))
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1.0)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def metadata_filter_success_rate(
    applied_filters_list: list[dict[str, Any]],
    expected_filters_list: list[dict[str, Any]],
) -> float:
    """
    Computes Metadata-Filter Success Rate.
    Returns fraction of query cases where applied metadata filters match expected filters.
    """
    if not expected_filters_list:
        return 1.0

    total = len(expected_filters_list)
    successes = 0

    for i in range(total):
        applied = applied_filters_list[i] if i < len(applied_filters_list) and isinstance(applied_filters_list[i], dict) else {}
        expected = expected_filters_list[i] if isinstance(expected_filters_list[i], dict) else {}

        # Check if all key-value pairs in expected filter are present in applied filter
        match = True
        for k, v in expected.items():
            if v is not None and applied.get(k) != v:
                match = False
                break
        if match:
            successes += 1

    return successes / float(total)


def exact_version_success_rate(
    retrieved_versions_list: list[str | None],
    target_versions_list: list[str | None],
) -> float:
    """
    Computes Exact-Version Match Success Rate.
    Returns fraction of queries where retrieved chunk version matches target version (or version is 'all'/neutral).
    """
    if not target_versions_list:
        return 1.0

    total = len(target_versions_list)
    matches = 0

    for i in range(total):
        target = target_versions_list[i]
        retrieved = retrieved_versions_list[i] if i < len(retrieved_versions_list) else None

        if not target or target.lower() == "all" or target == "none":
            matches += 1
        elif retrieved is not None and (str(retrieved).strip().lower() == str(target).strip().lower() or str(retrieved).strip().lower() == "all"):
            matches += 1

    return matches / float(total)


def average_retrieval_latency(latencies: list[float]) -> float:
    """
    Computes Average Retrieval Latency in seconds across query runs.
    Returns 0.0 if empty list.
    """
    if not latencies:
        return 0.0
    return float(sum(latencies)) / float(len(latencies))
