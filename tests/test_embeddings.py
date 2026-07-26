from unittest.mock import MagicMock, patch

import numpy as np

from app.embeddings import EmbeddingService


@patch("app.embeddings.SentenceTransformer")
def test_embedding_service_shape_and_norm(mock_st_cls):
    mock_model = MagicMock()
    # Mock sentence transformer encoding 2 documents to 384-dim vectors
    dummy_vecs = np.random.randn(2, 384)
    # Normalize vectors
    dummy_vecs = dummy_vecs / np.linalg.norm(dummy_vecs, axis=1, keepdims=True)
    mock_model.encode.return_value = dummy_vecs
    mock_st_cls.return_value = mock_model

    service = EmbeddingService()
    vectors, elapsed = service.embed_documents(["doc 1 text", "doc 2 text"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384
    assert elapsed >= 0.0

    # Test single query embedding
    mock_model.encode.return_value = dummy_vecs[0]
    q_vec, q_elapsed = service.embed_query("EXP-3204 export error")

    assert len(q_vec) == 384
    assert q_elapsed >= 0.0


def test_embedding_service_empty_input():
    with patch("app.embeddings.SentenceTransformer"):
        service = EmbeddingService()
        vecs, elapsed = service.embed_documents([])
        assert vecs == []
        assert elapsed == 0.0
