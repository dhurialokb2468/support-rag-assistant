import math
from evaluation.retrieval_metrics import (
    average_retrieval_latency,
    exact_version_success_rate,
    hit_rate_at_k,
    mean_reciprocal_rank,
    metadata_filter_success_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_rate_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc2", "doc5"]

    assert hit_rate_at_k(retrieved, expected, k=2) == 1.0
    assert hit_rate_at_k(retrieved, expected, k=1) == 0.0

    # Unanswerable query case (expected_ids = [])
    assert hit_rate_at_k(retrieved, [], k=5) == 1.0


def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    expected = ["doc1", "doc3"]

    # Top 2: doc1, doc2 -> 1 hit out of 2 = 0.5
    assert precision_at_k(retrieved, expected, k=2) == 0.5

    # Top 4: doc1, doc2, doc3, doc4 -> 2 hits out of 4 = 0.5
    assert precision_at_k(retrieved, expected, k=4) == 0.5

    # Unanswerable query case
    assert precision_at_k(retrieved, [], k=5) == 1.0


def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc1", "doc3", "doc5"]

    # Top 3 retrieves doc1, doc3 -> 2 out of 3 expected = 2/3
    assert math.isclose(recall_at_k(retrieved, expected, k=3), 2.0 / 3.0)

    # Unanswerable query case
    assert recall_at_k(retrieved, [], k=5) == 1.0


def test_reciprocal_rank():
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc2"]

    # First hit is doc2 at rank 2 -> RR = 0.5
    assert reciprocal_rank(retrieved, expected) == 0.5

    # No hit -> RR = 0.0
    assert reciprocal_rank(retrieved, ["doc99"]) == 0.0

    # Unanswerable query case
    assert reciprocal_rank(retrieved, []) == 1.0


def test_mean_reciprocal_rank():
    rr_list = [1.0, 0.5, 0.0, 0.5]
    assert mean_reciprocal_rank(rr_list) == 0.5
    assert mean_reciprocal_rank([]) == 0.0


def test_ndcg_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc1", "doc2"]

    # Perfect ranking doc1, doc2 -> NDCG = 1.0
    assert ndcg_at_k(retrieved, expected, k=3) == 1.0

    # Reversed ranking
    rev_retrieved = ["doc99", "doc1"]
    assert ndcg_at_k(rev_retrieved, expected, k=2) < 1.0
    assert ndcg_at_k(rev_retrieved, expected, k=2) > 0.0

    # Unanswerable query case
    assert ndcg_at_k(retrieved, [], k=5) == 1.0


def test_metadata_filter_success_rate():
    applied = [{"version": "3.2", "category": "reporting"}, {"version": "3.1"}]
    expected = [{"version": "3.2"}, {"version": "3.1"}]

    assert metadata_filter_success_rate(applied, expected) == 1.0

    mismatched = [{"version": "2.0"}, {"version": "3.1"}]
    assert metadata_filter_success_rate(mismatched, expected) == 0.5


def test_exact_version_success_rate():
    retrieved = ["3.2", "all", "3.1"]
    target = ["3.2", "3.2", "3.1"]

    assert exact_version_success_rate(retrieved, target) == 1.0

    mismatched = ["2.0", "3.1", "3.1"]
    assert exact_version_success_rate(mismatched, target) == 1.0 / 3.0


def test_average_retrieval_latency():
    latencies = [0.10, 0.20, 0.30]
    assert math.isclose(average_retrieval_latency(latencies), 0.20)
    assert average_retrieval_latency([]) == 0.0
