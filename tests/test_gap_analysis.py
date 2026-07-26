from unittest.mock import MagicMock

from app.gap_analysis import (
    analyze_documentation_gaps,
    identify_gap_candidates,
    is_gap_candidate,
)


def test_is_gap_candidate_criteria():
    # 1. Abstained
    cand_abst, reason = is_gap_candidate({"abstained": True})
    assert cand_abst is True
    assert "Abstained" in reason

    # 2. Low confidence
    cand_low_conf, reason = is_gap_candidate({"confidence_score": 0.40})
    assert cand_low_conf is True
    assert "Low Confidence" in reason

    # 3. Weak source
    cand_weak_src, reason = is_gap_candidate({"retrieved_sources": [{"final_score": 0.35}]})
    assert cand_weak_src is True
    assert "Weak Source" in reason

    # 4. User unhelpful feedback
    cand_unhelp, reason = is_gap_candidate({"feedback": [{"helpful": 0, "issue_type": "incorrect answer"}]})
    assert cand_unhelp is True
    assert "Unhelpful" in reason

    # 5. Normal interaction
    normal, _ = is_gap_candidate({
        "confidence_score": 0.90,
        "retrieved_sources": [{"final_score": 0.85}],
        "feedback": [{"helpful": 1}],
    })
    assert normal is False


def test_identify_gap_candidates_recurring():
    records = [
        {"question": "How to export CSV?"},
        {"question": "How to export CSV?"},
        {"question": "Salesforce sync failed", "confidence_score": 0.95, "retrieved_sources": [{"final_score": 0.90}]},
    ]

    cands = identify_gap_candidates(records)
    assert len(cands) == 2
    assert cands[0]["question"] == "How to export CSV?"
    assert cands[0]["gap_reason"] == "Recurring Query"


def test_analyze_documentation_gaps_too_few_records():
    records = [
        {"question": "Query 1", "abstained": True},
    ]

    res = analyze_documentation_gaps(records, min_records=3)
    assert res["status"] == "too_few_records"
    assert res["total_candidates"] == 1
    assert len(res["clusters"]) == 0


def test_analyze_documentation_gaps_clustering():
    records = [
        {
            "question": "How to fix CSV export error EXP-3204?",
            "abstained": True,
            "category": "reporting",
            "version": "3.2",
            "confidence_score": 0.20,
        },
        {
            "question": "CSV export fails with error code EXP-3204 on version 3.2",
            "abstained": True,
            "category": "reporting",
            "version": "3.2",
            "confidence_score": 0.25,
        },
        {
            "question": "Why is Salesforce integration sync delayed in 3.2?",
            "confidence_score": 0.30,
            "category": "integrations",
            "version": "3.2",
        },
        {
            "question": "Salesforce synchronization issue in version 3.2",
            "confidence_score": 0.35,
            "category": "integrations",
            "version": "3.2",
        },
        {
            "question": "Password reset link not arriving in inbox",
            "confidence_score": 0.15,
            "category": "authentication",
            "version": "3.1",
        },
        {
            "question": "Password recovery email fails to deliver",
            "confidence_score": 0.20,
            "category": "authentication",
            "version": "3.1",
        },
    ]

    # Mock embedding service to avoid loading sentence transformer model in fast test
    mock_embedder = MagicMock()
    # Simple dummy 4D embeddings for 6 records (2 reporting, 2 integrations, 2 auth)
    mock_embedder.embed_documents.return_value = (
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
        ],
        0.01,
    )

    report = analyze_documentation_gaps(records, min_records=3, num_clusters=3, embedding_service=mock_embedder)

    assert report["status"] == "success"
    assert report["total_candidates"] == 6
    assert report["clusters_count"] == 3

    clusters = report["clusters"]
    first_cluster = clusters[0]

    assert "cluster_id" in first_cluster
    assert "question_count" in first_cluster
    assert "representative_question" in first_cluster
    assert "example_questions" in first_cluster
    assert "recommended_documentation_action" in first_cluster
    assert len(first_cluster["example_questions"]) > 0
