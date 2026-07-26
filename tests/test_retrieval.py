from unittest.mock import MagicMock

from app.models import QueryMetadata
from app.retriever import MetadataAwareRetriever


def test_exact_version_filtering():
    mock_vector_store = MagicMock()
    mock_vector_store.semantic_search.return_value = [
        {"chunk_id": "c1", "text": "v3.2 doc", "metadata": {"version": "3.2"}, "semantic_score": 0.9}
    ] * 3

    retriever = MetadataAwareRetriever(vector_store=mock_vector_store, min_results=3)
    results, trace = retriever.retrieve("InsightFlow version 3.2 bug")

    assert trace.filters_requested.get("version") == "3.2"
    assert trace.strict_succeeded is True
    assert trace.fallback_used is False

    mock_vector_store.semantic_search.assert_called_once()
    _, kwargs = mock_vector_store.semantic_search.call_args
    assert "version" in kwargs["where"]


def test_version_neutral_documents_eligible():
    mock_vector_store = MagicMock()
    mock_vector_store.semantic_search.return_value = [
        {"chunk_id": "c1", "text": "v3.2 specific", "metadata": {"version": "3.2"}, "semantic_score": 0.95},
        {"chunk_id": "c2", "text": "General doc all", "metadata": {"version": "all"}, "semantic_score": 0.90},
        {"chunk_id": "c3", "text": "General doc empty", "metadata": {"version": ""}, "semantic_score": 0.85},
    ]

    retriever = MetadataAwareRetriever(vector_store=mock_vector_store, min_results=3)
    results, trace = retriever.retrieve("InsightFlow 3.2 guide")

    assert len(results) == 3
    versions_returned = [r["metadata"]["version"] for r in results]
    assert "3.2" in versions_returned
    assert "all" in versions_returned
    assert "" in versions_returned


def test_category_filtering():
    mock_vector_store = MagicMock()
    mock_vector_store.semantic_search.return_value = [
        {"chunk_id": "c1", "text": "CSV report guide", "metadata": {"category": "reporting"}, "semantic_score": 0.88}
    ] * 3

    retriever = MetadataAwareRetriever(vector_store=mock_vector_store, min_results=3)
    results, trace = retriever.retrieve("How to export CSV report?")

    assert trace.filters_requested.get("category") == "reporting"
    assert trace.strict_succeeded is True
    assert len(results) == 3


def test_relaxed_fallback():
    mock_vector_store = MagicMock()

    # Strict search returns empty list, relaxed search returns 3 results
    mock_vector_store.semantic_search.side_effect = [
        [],  # Strict attempt
        [   # Relaxed attempt 1 (drop category)
            {"chunk_id": "fallback_1", "text": "Fallback doc 1", "metadata": {}, "semantic_score": 0.75},
            {"chunk_id": "fallback_2", "text": "Fallback doc 2", "metadata": {}, "semantic_score": 0.70},
            {"chunk_id": "fallback_3", "text": "Fallback doc 3", "metadata": {}, "semantic_score": 0.65},
        ]
    ]

    retriever = MetadataAwareRetriever(vector_store=mock_vector_store, min_results=3)
    results, trace = retriever.retrieve("InsightFlow 3.2 CSV export fails", min_results=3)

    assert trace.strict_succeeded is False
    assert trace.fallback_used is True
    assert trace.strict_result_count == 0
    assert trace.relaxed_result_count == 3
    assert len(results) == 3


def test_queries_without_filters():
    mock_vector_store = MagicMock()
    mock_vector_store.semantic_search.return_value = [
        {"chunk_id": "c1", "text": "General info", "metadata": {}, "semantic_score": 0.80}
    ]

    retriever = MetadataAwareRetriever(vector_store=mock_vector_store, min_results=1)
    results, trace = retriever.retrieve("How do I fix a general problem?")

    assert trace.filters_requested == {}
    assert trace.filters_applied == {}
    assert trace.strict_succeeded is True
    assert trace.fallback_used is False
    mock_vector_store.semantic_search.assert_called_once_with(
        "How do I fix a general problem?", top_k=5, where=None
    )
