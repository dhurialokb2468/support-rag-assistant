from unittest.mock import MagicMock

from app.retriever import HybridRetriever, compute_rrf_score


def test_rrf_calculation():
    ranks = [1, 3]
    rrf_k = 60
    expected = (1.0 / (60 + 1)) + (1.0 / (60 + 3))
    assert abs(compute_rrf_score(ranks, rrf_k=rrf_k) - expected) < 1e-7


def test_duplicate_candidate_merging_and_both_methods():
    mock_vector_store = MagicMock()
    mock_bm25_store = MagicMock()

    mock_vector_store.semantic_search.return_value = [
        {"chunk_id": "c1", "text": "Common doc", "metadata": {"title": "Doc 1"}, "semantic_score": 0.95},
        {"chunk_id": "c2", "text": "Semantic only doc", "metadata": {"title": "Doc 2"}, "semantic_score": 0.85},
    ]
    mock_bm25_store.search.return_value = [
        {"chunk_id": "c3", "text": "Keyword only doc", "metadata": {"title": "Doc 3"}, "keyword_score": 0.90},
        {"chunk_id": "c1", "text": "Common doc", "metadata": {"title": "Doc 1"}, "keyword_score": 0.70},
    ]

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        bm25_store=mock_bm25_store,
        rrf_k=60,
        enable_multi_query=False,
    )

    results, trace = retriever.retrieve("test query", top_k=5, mode="hybrid")

    assert len(results) == 3
    chunk_ids = [r["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))

    c1_item = next(r for r in results if r["chunk_id"] == "c1")
    assert "semantic" in c1_item["retrieval_methods"]
    assert "keyword" in c1_item["retrieval_methods"]
    assert c1_item["semantic_rank"] == 1
    assert c1_item["keyword_rank"] == 2
    assert abs(c1_item["fused_score"] - ((1.0 / 61.0) + (1.0 / 62.0))) < 1e-7

    assert results[0]["chunk_id"] == "c1"


def test_semantic_only_chunk():
    mock_vector_store = MagicMock()
    mock_bm25_store = MagicMock()

    mock_vector_store.semantic_search.return_value = [
        {"chunk_id": "c_sem", "text": "Semantic doc", "metadata": {"title": "Sem"}, "semantic_score": 0.90}
    ]
    mock_bm25_store.search.return_value = []

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        bm25_store=mock_bm25_store,
        rrf_k=60,
        enable_multi_query=False,
    )
    results, _ = retriever.retrieve("query", mode="hybrid")

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c_sem"
    assert results[0]["retrieval_methods"] == ["semantic"]
    assert results[0]["semantic_rank"] == 1
    assert results[0]["keyword_rank"] is None


def test_keyword_only_chunk():
    mock_vector_store = MagicMock()
    mock_bm25_store = MagicMock()

    mock_vector_store.semantic_search.return_value = []
    mock_bm25_store.search.return_value = [
        {"chunk_id": "c_kw", "text": "Keyword doc", "metadata": {"title": "KW"}, "keyword_score": 0.88}
    ]

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        bm25_store=mock_bm25_store,
        rrf_k=60,
        enable_multi_query=False,
    )
    results, _ = retriever.retrieve("query", mode="hybrid")

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c_kw"
    assert results[0]["retrieval_methods"] == ["keyword"]
    assert results[0]["keyword_rank"] == 1
    assert results[0]["semantic_rank"] is None


def test_multi_query_retrieval_fusion_and_query_recording():
    mock_generator = MagicMock()
    mock_generator.generate_json.return_value = ({
        "queries": [
            "Original Query",
            "Symptom Query",
            "Documentation Query"
        ]
    }, 0.1)

    mock_vector_store = MagicMock()
    mock_bm25_store = MagicMock()

    # Define return values per query call
    def semantic_side_effect(query, top_k, where=None):
        if query == "Original Query":
            return [{"chunk_id": "c1", "text": "Doc 1", "metadata": {"title": "T1"}, "semantic_score": 0.9}]
        elif query == "Symptom Query":
            return [{"chunk_id": "c2", "text": "Doc 2", "metadata": {"title": "T2"}, "semantic_score": 0.85}]
        elif query == "Documentation Query":
            return [{"chunk_id": "c1", "text": "Doc 1", "metadata": {"title": "T1"}, "semantic_score": 0.88}]
        return []

    def bm25_side_effect(query, top_k, where=None):
        if query == "Symptom Query":
            return [{"chunk_id": "c3", "text": "Doc 3", "metadata": {"title": "T3"}, "keyword_score": 0.80}]
        return []

    mock_vector_store.semantic_search.side_effect = semantic_side_effect
    mock_bm25_store.search.side_effect = bm25_side_effect

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        bm25_store=mock_bm25_store,
        generator=mock_generator,
        rrf_k=60,
        enable_multi_query=True,
    )

    results, trace = retriever.retrieve("Original Query", mode="hybrid")

    assert trace.generated_queries == ["Original Query", "Symptom Query", "Documentation Query"]
    assert len(results) == 3

    # c1 was retrieved by Original Query (semantic rank 1) and Documentation Query (semantic rank 1)
    c1 = next(r for r in results if r["chunk_id"] == "c1")
    assert "Original Query" in c1["queries"]
    assert "Documentation Query" in c1["queries"]
    assert len(c1["queries"]) == 2
    # RRF score for c1 has two rank 1 occurrences: 1/61 + 1/61
    expected_c1_score = (1.0 / 61.0) + (1.0 / 61.0)
    assert abs(c1["fused_score"] - expected_c1_score) < 1e-7

    # c2 was retrieved by Symptom Query (semantic rank 1)
    c2 = next(r for r in results if r["chunk_id"] == "c2")
    assert c2["queries"] == ["Symptom Query"]

    # c3 was retrieved by Symptom Query (keyword rank 1)
    c3 = next(r for r in results if r["chunk_id"] == "c3")
    assert c3["queries"] == ["Symptom Query"]
    assert c3["retrieval_methods"] == ["keyword"]


def test_enable_disable_multi_query_config():
    mock_generator = MagicMock()
    mock_generator.generate_json.return_value = ({
        "queries": ["Orig Query", "Variation 1", "Variation 2"]
    }, 0.1)

    mock_vector_store = MagicMock()
    mock_bm25_store = MagicMock()
    mock_vector_store.semantic_search.return_value = []
    mock_bm25_store.search.return_value = []

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        bm25_store=mock_bm25_store,
        generator=mock_generator,
        enable_multi_query=False,
    )

    # With enable_multi_query=False
    _, trace_disabled = retriever.retrieve("Orig Query", enable_multi_query=False)
    assert trace_disabled.generated_queries == ["Orig Query"]
    mock_generator.generate_json.assert_not_called()

    # Overriding with enable_multi_query=True
    _, trace_enabled = retriever.retrieve("Orig Query", enable_multi_query=True)
    assert trace_enabled.generated_queries == ["Orig Query", "Variation 1", "Variation 2"]
    mock_generator.generate_json.assert_called_once()

