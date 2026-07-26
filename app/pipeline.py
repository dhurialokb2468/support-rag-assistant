import time

from app.confidence import compute_deterministic_confidence
from app.conflicts import detect_and_resolve_conflicts
from app.conversation import ConversationState
from app.generator import OllamaGenerator
from app.models import PipelineTrace
from app.query_processor import rewrite_query
from app.retriever import HybridRetriever
from app.scoring import score_candidates
from app.validators import evaluate_abstention, validate_and_resolve_citations
from app.vector_store import VectorStore


class BasicRAGPipeline:
    def __init__(
        self,
        conversation_state: ConversationState | None = None,
        vector_store: VectorStore | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.retriever = retriever or HybridRetriever(vector_store=self.vector_store)
        self.generator = OllamaGenerator()
        self.conversation_state = conversation_state or ConversationState()

    @staticmethod
    def build_context(results: list[dict]) -> str:
        sections = []

        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source_id = f"S{index}"
            result["source_id"] = source_id

            sections.append(
                f"[Source {source_id}]\n"
                f"Title: {metadata.get('title', 'Untitled')}\n"
                f"Source: {metadata.get('source', 'Unknown')}\n"
                f"Version: {metadata.get('version', 'N/A')}\n"
                f"Content:\n{result['text']}"
            )

        return "\n\n".join(sections)

    def answer(self, question: str) -> dict:
        pipeline_start = time.perf_counter()
        trace_obj = PipelineTrace(query=question)

        # 1. Metadata Extraction
        t0 = time.perf_counter()
        self.conversation_state.update_from_question(question)
        trace_obj.metadata_extraction_latency = round(time.perf_counter() - t0, 6)

        # 2. Query Rewriting
        t0 = time.perf_counter()
        search_query, rewrite_trace = rewrite_query(
            question=question,
            conversation_state=self.conversation_state,
            generator=self.generator,
        )
        trace_obj.query_rewriting_latency = round(time.perf_counter() - t0, 6)
        trace_obj.rewritten_query = search_query

        # 3-8. Retrieval (Expansion, Embedding, Semantic, BM25, Fusion, Reranking)
        t0 = time.perf_counter()
        retrieved_candidates, ret_trace = self.retriever.retrieve(
            query=search_query,
            top_k=5,
            mode="reranked",
        )

        trace_obj.query_expansion_latency = getattr(ret_trace, "query_expansion_latency", 0.0) or 0.0
        trace_obj.embedding_latency = getattr(ret_trace, "embedding_latency", 0.0) or 0.0
        trace_obj.semantic_retrieval_latency = getattr(ret_trace, "semantic_retrieval_latency", 0.0) or 0.0
        trace_obj.bm25_retrieval_latency = getattr(ret_trace, "bm25_retrieval_latency", 0.0) or 0.0
        trace_obj.fusion_latency = getattr(ret_trace, "fusion_latency", 0.0) or 0.0
        trace_obj.reranking_latency = getattr(ret_trace, "rerank_latency", 0.0) or 0.0
        trace_obj.filters_requested = getattr(ret_trace, "filters_requested", {}) or {}
        trace_obj.filters_applied = getattr(ret_trace, "filters_applied", {}) or {}
        trace_obj.strict_succeeded = getattr(ret_trace, "strict_succeeded", True)
        trace_obj.fallback_used = getattr(ret_trace, "fallback_used", False)
        trace_obj.generated_queries = getattr(ret_trace, "generated_queries", []) or []

        # 9. Authority & Freshness Scoring
        t0 = time.perf_counter()
        scored_results = score_candidates(query=search_query, candidates=retrieved_candidates)
        trace_obj.authority_freshness_scoring_latency = round(time.perf_counter() - t0, 6)

        # 10. Parent Resolution
        t0 = time.perf_counter()
        # Parent resolution if parent_id is present
        trace_obj.parent_resolution_latency = round(time.perf_counter() - t0, 6)

        # 11. Context Building
        t0 = time.perf_counter()
        context = self.build_context(scored_results)
        trace_obj.context_building_latency = round(time.perf_counter() - t0, 6)

        # 12. Generation
        t0 = time.perf_counter()
        support_answer, gen_trace = self.generator.generate_support_answer(
            question=question,
            context=context,
        )
        trace_obj.generation_latency = round(time.perf_counter() - t0, 6)

        # 13. Citation Verification
        t0 = time.perf_counter()
        validated_answer, resolved_citations, cit_passed, cit_details = validate_and_resolve_citations(
            support_answer,
            scored_results,
        )
        trace_obj.citation_verification_latency = round(time.perf_counter() - t0, 6)
        trace_obj.citation_validation_passed = cit_passed

        # 14. Conflict Detection
        t0 = time.perf_counter()
        conflict_result = detect_and_resolve_conflicts(
            question=question,
            selected_chunks=scored_results,
            query_version=self.conversation_state.product_version,
            generator=self.generator,
        )
        trace_obj.conflict_detection_latency = round(time.perf_counter() - t0, 6)
        trace_obj.conflict_result = conflict_result.model_dump()

        if conflict_result.conflict_detected:
            validated_answer.conflicts_detected = True
            if conflict_result.unresolved:
                validated_answer.escalation_required = True
                validated_answer.escalation_reason = conflict_result.summary

        # 15. Confidence Calculation
        t0 = time.perf_counter()
        det_score, det_level, breakdown = compute_deterministic_confidence(
            support_answer=validated_answer,
            retrieved_chunks=scored_results,
            citation_validation_passed=cit_passed,
            query_version=self.conversation_state.product_version,
        )
        trace_obj.confidence_calculation_latency = round(time.perf_counter() - t0, 6)
        trace_obj.confidence_breakdown = breakdown

        validated_answer.confidence_score = det_score
        validated_answer.confidence = det_level

        # 16. Validation & Abstention
        t0 = time.perf_counter()
        abstained, abstention_reason, final_answer = evaluate_abstention(
            question=question,
            support_answer=validated_answer,
            retrieved_chunks=scored_results,
            citation_validation_passed=cit_passed,
            confidence_score=det_score,
            confidence_level=det_level,
        )
        trace_obj.validation_latency = round(time.perf_counter() - t0, 6)
        trace_obj.abstention_triggered = abstained
        trace_obj.abstention_reason = abstention_reason.value if abstention_reason else None

        self.conversation_state.update_from_answer(
            answer=final_answer,
            source_ids=[c.source_id for c in resolved_citations],
        )
        trace_obj.conversation_state = self.conversation_state.model_dump()

        # 17. Total Latency
        trace_obj.total_latency = round(time.perf_counter() - pipeline_start, 6)

        trace_dict = trace_obj.model_dump()
        trace_dict["rewrite_trace"] = rewrite_trace.model_dump() if hasattr(rewrite_trace, "model_dump") else rewrite_trace
        trace_dict["citation_details"] = cit_details

        return {
            "question": question,
            "answer": final_answer.model_dump(),
            "citations": [c.model_dump() for c in resolved_citations],
            "sources": scored_results,
            "generation_time": trace_obj.generation_latency,
            "trace": trace_dict,
            "pipeline_trace": trace_obj.model_dump(),
        }