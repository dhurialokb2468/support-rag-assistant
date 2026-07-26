from app.confidence import compute_deterministic_confidence
from app.models import SupportAnswer


def create_sample_support_answer(
    conflicts_detected: bool = False,
    escalation_required: bool = False,
):
    return SupportAnswer(
        answer="InsightFlow 3.2 error EXP-3204 resolved by updating network credentials.",
        likely_cause="Network credential error.",
        resolution_steps=["Verify network credentials."],
        citations=["S1", "S2"],
        confidence="high",
        confidence_score=0.9,
        escalation_required=escalation_required,
        escalation_reason=None,
        conflicts_detected=conflicts_detected,
    )


def test_strong_evidence():
    ans = create_sample_support_answer()
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "final_score": 0.95,
            "authority_score": 0.9,
            "metadata": {"source": "guide_v3.2.md", "version": "3.2"},
        },
        {
            "chunk_id": "c2",
            "document_id": "doc2",
            "final_score": 0.70,
            "authority_score": 0.8,
            "metadata": {"source": "release_notes.md", "version": "3.2"},
        },
        {
            "chunk_id": "c3",
            "document_id": "doc3",
            "final_score": 0.60,
            "authority_score": 0.8,
            "metadata": {"source": "kb_article.md", "version": "3.2"},
        },
    ]

    score, level, breakdown = compute_deterministic_confidence(
        support_answer=ans,
        retrieved_chunks=chunks,
        citation_validation_passed=True,
        query_version="3.2",
    )

    assert score >= 0.75
    assert level == "high"
    assert breakdown["top_relevance"] == 0.95
    assert breakdown["num_independent_sources"] == 3
    assert breakdown["exact_version_match"] is True
    assert breakdown["citation_validation_passed"] is True


def test_weak_retrieval():
    ans = create_sample_support_answer()
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "final_score": 0.30,  # Weak relevance score
            "authority_score": 0.4,
            "metadata": {"source": "doc1.md", "version": "3.2"},
        },
    ]

    score, level, breakdown = compute_deterministic_confidence(
        support_answer=ans,
        retrieved_chunks=chunks,
        citation_validation_passed=True,
        query_version="3.2",
    )

    assert score < 0.50
    assert level == "low"
    assert breakdown["top_relevance"] == 0.30


def test_invalid_citations():
    ans = create_sample_support_answer()
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "final_score": 0.90,
            "authority_score": 0.8,
            "metadata": {"source": "doc1.md", "version": "3.2"},
        },
    ]

    score_valid, level_valid, _ = compute_deterministic_confidence(
        support_answer=ans,
        retrieved_chunks=chunks,
        citation_validation_passed=True,
        query_version="3.2",
    )

    score_invalid, level_invalid, breakdown_invalid = compute_deterministic_confidence(
        support_answer=ans,
        retrieved_chunks=chunks,
        citation_validation_passed=False,  # Failed citation validation
        query_version="3.2",
    )

    assert score_invalid < score_valid
    assert breakdown_invalid["citation_score"] == 0.0


def test_version_mismatch():
    ans = create_sample_support_answer()
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "final_score": 0.85,
            "authority_score": 0.8,
            "metadata": {"source": "guide_v2.0.md", "version": "2.0"},  # Mismatched version 2.0
        },
    ]

    _, _, breakdown_mismatch = compute_deterministic_confidence(
        support_answer=ans,
        retrieved_chunks=chunks,
        citation_validation_passed=True,
        query_version="3.2",  # Query asked for 3.2
    )

    assert breakdown_mismatch["exact_version_match"] is False
    assert breakdown_mismatch["version_score"] == 0.0


def test_conflicting_sources():
    ans_conflict = create_sample_support_answer(conflicts_detected=True)
    ans_no_conflict = create_sample_support_answer(conflicts_detected=False)

    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "final_score": 0.90,
            "authority_score": 0.8,
            "metadata": {"source": "doc1.md", "version": "3.2"},
        },
    ]

    score_conflict, _, breakdown_conflict = compute_deterministic_confidence(
        support_answer=ans_conflict,
        retrieved_chunks=chunks,
    )

    score_clean, _, _ = compute_deterministic_confidence(
        support_answer=ans_no_conflict,
        retrieved_chunks=chunks,
    )

    assert breakdown_conflict["conflict_penalty"] == 0.20
    assert score_conflict == max(0.0, round(score_clean - 0.20, 4))


def test_multiple_independent_sources():
    ans = create_sample_support_answer()

    single_source_chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "final_score": 0.80,
            "authority_score": 0.8,
            "metadata": {"source": "doc1.md", "version": "3.2"},
        },
        {
            "chunk_id": "c2",
            "document_id": "doc1",  # Same source document doc1.md
            "final_score": 0.80,
            "authority_score": 0.8,
            "metadata": {"source": "doc1.md", "version": "3.2"},
        },
    ]

    multi_source_chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "final_score": 0.80,
            "authority_score": 0.8,
            "metadata": {"source": "doc1.md", "version": "3.2"},
        },
        {
            "chunk_id": "c2",
            "document_id": "doc2",  # Distinct independent source document doc2.md
            "final_score": 0.80,
            "authority_score": 0.8,
            "metadata": {"source": "doc2.md", "version": "3.2"},
        },
        {
            "chunk_id": "c3",
            "document_id": "doc3",  # Distinct independent source document doc3.md
            "final_score": 0.80,
            "authority_score": 0.8,
            "metadata": {"source": "doc3.md", "version": "3.2"},
        },
    ]

    score_single, _, breakdown_single = compute_deterministic_confidence(ans, single_source_chunks)
    score_multi, _, breakdown_multi = compute_deterministic_confidence(ans, multi_source_chunks)

    assert breakdown_single["num_independent_sources"] == 1
    assert breakdown_multi["num_independent_sources"] == 3
    assert breakdown_multi["sources_score"] > breakdown_single["sources_score"]
    assert score_multi > score_single
