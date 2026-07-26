from datetime import datetime, timezone
from app.scoring import (
    calculate_authority_score,
    calculate_freshness_score,
    calculate_version_score,
    normalize_reranker_scores,
    score_candidates,
)


def test_authority_influence():
    # Two candidates with identical relevance and freshness, but different authority
    candidates = [
        {
            "chunk_id": "c_low_auth",
            "raw_reranker_score": 2.0,
            "text": "Low authority guide",
            "metadata": {"authority_score": 0.3, "updated_at": "2026-05-01"},
        },
        {
            "chunk_id": "c_high_auth",
            "raw_reranker_score": 2.0,
            "text": "High authority guide",
            "metadata": {"authority_score": 0.9, "updated_at": "2026-05-01"},
        },
    ]

    scored = score_candidates("InsightFlow query", candidates)

    assert scored[0]["chunk_id"] == "c_high_auth"
    assert scored[0]["authority_score"] == 0.9
    assert scored[1]["authority_score"] == 0.3
    assert scored[0]["final_score"] > scored[1]["final_score"]


def test_freshness_influence():
    ref_date = datetime(2026, 7, 25, tzinfo=timezone.utc)

    # Two candidates with identical relevance and authority, but different dates
    candidates = [
        {
            "chunk_id": "c_old",
            "raw_reranker_score": 3.0,
            "text": "Older article",
            "metadata": {"authority_score": 0.5, "updated_at": "2024-01-01"},
        },
        {
            "chunk_id": "c_new",
            "raw_reranker_score": 3.0,
            "text": "Newer article",
            "metadata": {"authority_score": 0.5, "updated_at": "2026-07-20"},
        },
    ]

    scored = score_candidates("InsightFlow query", candidates, reference_date=ref_date)

    assert scored[0]["chunk_id"] == "c_new"
    assert scored[0]["freshness_score"] > scored[1]["freshness_score"]
    assert scored[0]["final_score"] > scored[1]["final_score"]


def test_exact_version_preference():
    # Query specifies version 3.2
    query = "InsightFlow 3.2 export issue"

    candidates = [
        {
            "chunk_id": "c_ver_2_0",
            "raw_reranker_score": 3.0,
            "text": "InsightFlow 2.0 guide",
            "metadata": {"version": "2.0", "authority_score": 0.5, "updated_at": "2026-07-20"},  # Newer date
        },
        {
            "chunk_id": "c_ver_3_2",
            "raw_reranker_score": 3.0,
            "text": "InsightFlow 3.2 guide",
            "metadata": {"version": "3.2", "authority_score": 0.5, "updated_at": "2025-01-01"},  # Older date
        },
        {
            "chunk_id": "c_ver_all",
            "raw_reranker_score": 3.0,
            "text": "InsightFlow general guide",
            "metadata": {"version": "all", "authority_score": 0.5, "updated_at": "2025-01-01"},  # Neutral doc remains eligible
        },
    ]

    scored = score_candidates(query, candidates)

    # Exact version match (3.2) must outrank mismatched version (2.0) despite older date
    assert scored[0]["chunk_id"] == "c_ver_3_2"
    assert scored[0]["version_score"] > scored[1]["version_score"]


def test_old_but_authoritative_source():
    ref_date = datetime(2026, 7, 25, tzinfo=timezone.utc)

    candidates = [
        {
            "chunk_id": "c_new_low_auth",
            "raw_reranker_score": 3.0,
            "text": "New community forum post",
            "metadata": {"authority_score": 0.1, "updated_at": "2026-07-24"},
        },
        {
            "chunk_id": "c_old_high_auth",
            "raw_reranker_score": 3.0,
            "text": "Official administrative manual",
            "metadata": {"authority_score": 1.0, "updated_at": "2025-01-01"},
        },
    ]

    scored = score_candidates("InsightFlow query", candidates, reference_date=ref_date)

    assert scored[0]["chunk_id"] == "c_old_high_auth"
    assert scored[0]["final_score"] > scored[1]["final_score"]


def test_new_irrelevant_source_not_outranking():
    ref_date = datetime(2026, 7, 25, tzinfo=timezone.utc)

    candidates = [
        {
            "chunk_id": "c_new_irrelevant",
            "raw_reranker_score": 0.5,  # Low relevance
            "text": "Brand new post about unrelated feature",
            "metadata": {"authority_score": 0.5, "updated_at": "2026-07-25"},  # Very new
        },
        {
            "chunk_id": "c_relevant_evidence",
            "raw_reranker_score": 5.0,  # High relevance
            "text": "Exact matching solution for error code",
            "metadata": {"authority_score": 0.5, "updated_at": "2025-06-01"},  # Older
        },
    ]

    scored = score_candidates("InsightFlow error EXP-3204", candidates, reference_date=ref_date)

    assert scored[0]["chunk_id"] == "c_relevant_evidence"
    assert scored[0]["final_score"] > scored[1]["final_score"]


def test_weights_normalization():
    # Pass weights that do not sum to 1 (0.8 + 0.4 + 0.4 = 1.6)
    candidates = [
        {
            "chunk_id": "c1",
            "raw_reranker_score": 3.0,
            "text": "Test doc",
            "metadata": {"authority_score": 0.8, "updated_at": "2026-01-01"},
        }
    ]

    scored = score_candidates(
        "query",
        candidates,
        relevance_weight=0.8,
        authority_weight=0.4,
        freshness_weight=0.4,
    )

    assert len(scored) == 1
    # Check that individual score components are inspectable
    item = scored[0]
    assert "normalized_reranker_score" in item
    assert "authority_score" in item
    assert "freshness_score" in item
    assert "version_score" in item
    assert "final_score" in item
