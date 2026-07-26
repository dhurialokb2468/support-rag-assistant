from unittest.mock import MagicMock, patch
import chromadb

from app.bm25_store import BM25Store
from app.chunking import chunk_documents
from app.feedback import FeedbackDB
from app.loaders import load_all_documents
from app.models import SupportAnswer
from app.pipeline import BasicRAGPipeline
from app.retriever import HybridRetriever
from app.vector_store import VectorStore


def test_end_to_end_pipeline_integration(tmp_path):
    chroma_dir = str(tmp_path / "chroma_e2e")
    bm25_file = tmp_path / "bm25_e2e.pkl"
    sqlite_db = tmp_path / "feedback_e2e.db"

    # 1. Initialize temporary Chroma VectorStore
    with patch("app.vector_store.settings") as mock_settings:
        mock_settings.chroma_path = chroma_dir
        mock_settings.chroma_collection = "e2e_test_collection"

        vector_store = VectorStore()
        vector_store.reset()

        # Load and chunk documents
        docs = load_all_documents()
        assert len(docs) > 0

        chunks = chunk_documents(docs, strategy="parent-child")
        assert len(chunks) > 0

        vector_store.add_chunks(chunks)
        assert vector_store.count() == len(chunks)

        # 2. Initialize temporary BM25Store
        bm25_store = BM25Store(catalog_path=bm25_file)
        bm25_store.index_chunks(chunks)
        bm25_store.save()

        # 3. Instantiate HybridRetriever
        retriever = HybridRetriever(vector_store=vector_store, bm25_store=bm25_store)

        # 4. Instantiate Pipeline with Mocked Ollama Generator
        pipeline = BasicRAGPipeline(vector_store=vector_store, retriever=retriever)

        # Mock Ollama generator response to return structured JSON answer
        mock_support_ans = SupportAnswer(
            answer="To resolve error EXP-3204, clear the report configuration cache and recreate export settings.",
            likely_cause="Cached export settings referencing deprecated schema.",
            resolution_steps=["Clear report configuration cache", "Recreate export settings"],
            citations=["S1"],
            confidence="high",
            confidence_score=0.95,
            escalation_required=False,
        )

        with patch.object(pipeline.generator, "generate_support_answer", return_value=(mock_support_ans, {})):
            question = "Why does EXP-3204 occur in version 3.2?"
            result = pipeline.answer(question)

            # Assert structured response attributes
            assert result["question"] == question
            assert "answer" in result
            assert result["answer"]["confidence"] in ("high", "medium", "low")
            assert len(result["sources"]) > 0
            assert "pipeline_trace" in result

            # 5. Persist interaction in temporary FeedbackDB
            fb_db = FeedbackDB(db_path=sqlite_db)
            i_id = fb_db.save_interaction(
                question=question,
                rewritten_query=result["trace"].get("rewritten_query"),
                answer_data=result["answer"],
                confidence_score=result["answer"]["confidence_score"],
                total_latency=result["pipeline_trace"]["total_latency"],
                abstained=result["trace"]["abstention_triggered"],
                escalated=result["answer"]["escalation_required"],
                retrieved_sources=result["sources"],
                pipeline_trace=result["pipeline_trace"],
            )

            assert i_id is not None
            saved_interaction = fb_db.get_interaction(i_id)
            assert saved_interaction["question"] == question
            assert len(saved_interaction["retrieved_sources"]) > 0

            # Attach feedback
            f_id = fb_db.save_feedback(interaction_id=i_id, helpful=True)
            assert f_id is not None
