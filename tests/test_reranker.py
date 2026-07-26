from unittest.mock import MagicMock
from app.reranker import RerankerService, sigmoid
from app.retriever import HybridRetriever


def test_sigmoid_math():
    assert abs(sigmoid(0.0) - 0.5) < 1e-6


def test_reranker_service_with_mock():
    mock_cross_encoder = MagicMock()
    # Return raw scores: candidate 1 gets -1.0, candidate 2 gets 4.0
    mock_cross_encoder.predict.return_value = [-1.0, 4.0]

    reranker = RerankerService()
    reranker.model = mock_cross_encoder

    candidates = [
        {"chunk_id": "c1", "text": "Doc 1 text", "semantic_score": 0.90},
        {"chunk_id": "c2", "text": "Doc 2 text", "semantic_score": 0.80},
    ]

    reranked, elapsed = reranker.rerank("test query", candidates, top_k=2)

    assert len(reranked) == 2
    assert elapsed >= 0.0

    # Candidate c2 got score 4.0 so it ranks #1
    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[0]["raw_reranker_score"] == 4.0
    assert reranked[0]["reranker_rank"] == 1
    assert abs(reranked[0]["reranker_score"] - sigmoid(4.0)) < 1e-6

    # Candidate c1 got score -1.0 so it ranks #2
    assert reranked[1]["chunk_id"] == "c1"
    assert reranked[1]["raw_reranker_score"] == -1.0
    assert reranked[1]["reranker_rank"] == 2


def test_hybrid_retriever_reranked_mode():
    mock_vector_store = MagicMock()
    mock_bm25_store = MagicMock()
    mock_reranker = MagicMock()

    mock_vector_store.semantic_search.return_value = [
        {"chunk_id": "c1", "text": "Text 1", "metadata": {"title": "T1"}, "semantic_score": 0.9},
        {"chunk_id": "c2", "text": "Text 2", "metadata": {"title": "T2"}, "semantic_score": 0.8},
    ]
    mock_bm25_store.search.return_value = []

    mock_reranker.rerank.return_value = (
        [
            {
                "chunk_id": "c2",
                "text": "Text 2",
                "metadata": {"title": "T2"},
                "semantic_score": 0.8,
                "raw_reranker_score": 3.5,
                "reranker_score": sigmoid(3.5),
                "reranker_rank": 1,
            }
        ],
        0.025,
    )

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        bm25_store=mock_bm25_store,
        reranker_service=mock_reranker,
    )

    results, trace = retriever.retrieve("query", top_k=1, mode="reranked")

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["reranker_score"] > 0.5
    assert trace.rerank_latency == 0.025
    mock_reranker.rerank.assert_called_once()
