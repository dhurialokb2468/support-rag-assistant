from typing import Any, Literal

from app.bm25_store import BM25Store
from app.config import settings
from app.logger import get_logger
from app.models import QueryMetadata, RetrievalTrace
from app.query_processor import QueryProcessor, generate_search_queries
from app.reranker import RerankerService
from app.vector_store import VectorStore

logger = get_logger("retriever")


def compute_rrf_score(ranks: list[int], rrf_k: int = 60) -> float:
    """Computes RRF score: sum(1 / (rrf_k + rank)) for each 1-based rank."""
    return sum(1.0 / (rrf_k + rank) for rank in ranks)


class MetadataAwareRetriever:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        query_processor: QueryProcessor | None = None,
        min_results: int = 3,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.query_processor = query_processor or QueryProcessor()
        self.min_results = min_results

    def build_where_clause(
        self,
        metadata: QueryMetadata,
        include_category: bool = True,
        include_version: bool = True,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        clauses: list[dict[str, Any]] = []
        applied: dict[str, Any] = {}

        if include_version and metadata.version:
            clauses.append(
                {"version": {"$in": [metadata.version, "all", ""]}}
            )
            applied["version"] = metadata.version

        if include_category and metadata.category:
            clauses.append({"category": metadata.category})
            applied["category"] = metadata.category

        if not clauses:
            return None, applied
        elif len(clauses) == 1:
            return clauses[0], applied
        else:
            return {"$and": clauses}, applied

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_results: int | None = None,
        metadata_filter: QueryMetadata | None = None,
    ) -> tuple[list[dict[str, Any]], RetrievalTrace]:
        target_min = min_results if min_results is not None else self.min_results
        metadata = metadata_filter or self.query_processor.process(query)

        filters_requested: dict[str, Any] = {}
        if metadata.version:
            filters_requested["version"] = metadata.version
        if metadata.category:
            filters_requested["category"] = metadata.category

        strict_where, filters_applied = self.build_where_clause(
            metadata, include_category=True, include_version=True
        )

        strict_results = self.vector_store.semantic_search(
            query, top_k=top_k, where=strict_where
        )

        strict_count = len(strict_results)
        has_filters = bool(filters_requested)

        strict_succeeded = (
            strict_count >= target_min if has_filters else True
        )

        logger.debug(f"Strict search count: {strict_count} (filters_applied: {filters_applied})")

        if strict_succeeded or not has_filters:
            trace = RetrievalTrace(
                query=query,
                filters_requested=filters_requested,
                filters_applied=filters_applied,
                strict_result_count=strict_count,
                relaxed_result_count=strict_count,
                strict_succeeded=True,
                fallback_used=False,
            )
            return strict_results, trace

        # Fallback / Relaxed retrieval
        logger.warning(
            f"Strict metadata filter returned only {strict_count} results (minimum required: {target_min}). "
            f"Triggering fallback retrieval..."
        )
        fallback_results: list[dict[str, Any]] = []
        relaxed_applied: dict[str, Any] = {}

        if metadata.version and metadata.category:
            relaxed_where, relaxed_applied = self.build_where_clause(
                metadata, include_category=False, include_version=True
            )
            fallback_results = self.vector_store.semantic_search(
                query, top_k=top_k, where=relaxed_where
            )

        if len(fallback_results) < target_min:
            relaxed_applied = {}
            fallback_results = self.vector_store.semantic_search(
                query, top_k=top_k, where=None
            )

        combined_results: list[dict[str, Any]] = list(strict_results)
        seen_ids = {r["chunk_id"] for r in strict_results}

        for item in fallback_results:
            if item["chunk_id"] not in seen_ids:
                seen_ids.add(item["chunk_id"])
                combined_results.append(item)

        final_results = combined_results[:top_k]

        trace = RetrievalTrace(
            query=query,
            filters_requested=filters_requested,
            filters_applied=relaxed_applied,
            strict_result_count=strict_count,
            relaxed_result_count=len(final_results),
            strict_succeeded=False,
            fallback_used=True,
        )

        return final_results, trace


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        metadata_retriever: MetadataAwareRetriever | None = None,
        query_processor: QueryProcessor | None = None,
        reranker_service: RerankerService | None = None,
        generator: Any | None = None,
        rrf_k: int = settings.rrf_k,
        enable_multi_query: bool = settings.enable_multi_query,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        if not self.bm25_store.bm25:
            self.bm25_store.load()
        self.metadata_retriever = metadata_retriever or MetadataAwareRetriever(
            vector_store=self.vector_store,
            query_processor=query_processor,
        )
        self.query_processor = query_processor or QueryProcessor()
        self.reranker_service = reranker_service or RerankerService()
        self.generator = generator
        self.rrf_k = rrf_k
        self.enable_multi_query = enable_multi_query

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["semantic", "keyword", "hybrid", "reranked"] = "hybrid",
        top_k_semantic: int = settings.top_k_semantic,
        top_k_keyword: int = settings.top_k_keyword,
        top_k_rerank: int = settings.top_k_rerank,
        where: dict[str, Any] | None = None,
        enable_multi_query: bool | None = None,
        generator: Any | None = None,
    ) -> tuple[list[dict[str, Any]], RetrievalTrace]:
        target_mode = "hybrid" if mode == "reranked" else mode

        use_multi_query = (
            enable_multi_query
            if enable_multi_query is not None
            else self.enable_multi_query
        )
        gen = generator if generator is not None else self.generator

        if use_multi_query:
            generated_queries = generate_search_queries(query, generator=gen)
        else:
            generated_queries = [query]

        trace = RetrievalTrace(query=query, generated_queries=generated_queries)

        combined: dict[str, dict[str, Any]] = {}

        for q in generated_queries:
            q_semantic_results: list[dict[str, Any]] = []
            q_trace: RetrievalTrace | None = None

            if target_mode in ("semantic", "hybrid"):
                if where is not None:
                    q_semantic_results = self.vector_store.semantic_search(
                        q, top_k=top_k_semantic, where=where
                    )
                    if not trace.filters_applied:
                        trace.filters_applied = where
                else:
                    q_semantic_results, q_trace = self.metadata_retriever.retrieve(
                        q, top_k=top_k_semantic
                    )
                    if not trace.filters_applied and q_trace.filters_applied:
                        trace.filters_applied = q_trace.filters_applied
                        trace.filters_requested = q_trace.filters_requested

                for rank, item in enumerate(q_semantic_results, start=1):
                    cid = item["chunk_id"]
                    if cid not in combined:
                        combined[cid] = {
                            "chunk_id": cid,
                            "text": item["text"],
                            "metadata": item["metadata"],
                            "semantic_score": item.get("semantic_score"),
                            "semantic_rank": rank,
                            "keyword_score": None,
                            "keyword_rank": None,
                            "fused_score": None,
                            "retrieval_methods": ["semantic"],
                            "queries": [q],
                            "_all_ranks": [rank],
                        }
                    else:
                        cand = combined[cid]
                        cand["_all_ranks"].append(rank)
                        if "semantic" not in cand["retrieval_methods"]:
                            cand["retrieval_methods"].append("semantic")
                        if q not in cand["queries"]:
                            cand["queries"].append(q)
                        if cand["semantic_rank"] is None or rank < cand["semantic_rank"]:
                            cand["semantic_rank"] = rank
                        if (
                            cand["semantic_score"] is None
                            or (item.get("semantic_score") is not None and item["semantic_score"] > cand["semantic_score"])
                        ):
                            cand["semantic_score"] = item.get("semantic_score")

            q_keyword_results: list[dict[str, Any]] = []
            if target_mode in ("keyword", "hybrid"):
                kw_where = (
                    where
                    if where is not None
                    else (q_trace.filters_applied if q_trace else trace.filters_applied)
                )
                q_keyword_results = self.bm25_store.search(
                    q, top_k=top_k_keyword, where=kw_where
                )
                if not q_keyword_results and kw_where:
                    q_keyword_results = self.bm25_store.search(
                        q, top_k=top_k_keyword, where=None
                    )

                for rank, item in enumerate(q_keyword_results, start=1):
                    cid = item["chunk_id"]
                    if cid not in combined:
                        combined[cid] = {
                            "chunk_id": cid,
                            "text": item["text"],
                            "metadata": item["metadata"],
                            "semantic_score": None,
                            "semantic_rank": None,
                            "keyword_score": item.get("keyword_score"),
                            "keyword_rank": rank,
                            "fused_score": None,
                            "retrieval_methods": ["keyword"],
                            "queries": [q],
                            "_all_ranks": [rank],
                        }
                    else:
                        cand = combined[cid]
                        cand["_all_ranks"].append(rank)
                        if "keyword" not in cand["retrieval_methods"]:
                            cand["retrieval_methods"].append("keyword")
                        if q not in cand["queries"]:
                            cand["queries"].append(q)
                        if cand["keyword_rank"] is None or rank < cand["keyword_rank"]:
                            cand["keyword_rank"] = rank
                        if (
                            cand["keyword_score"] is None
                            or (item.get("keyword_score") is not None and item["keyword_score"] > cand["keyword_score"])
                        ):
                            cand["keyword_score"] = item.get("keyword_score")

        if target_mode == "semantic":
            candidates = list(combined.values())
            for cand in candidates:
                ranks = cand.pop("_all_ranks")
                cand["fused_score"] = compute_rrf_score(ranks, rrf_k=self.rrf_k)
            candidates.sort(
                key=lambda x: (
                    x["semantic_rank"] if x["semantic_rank"] is not None else 999999,
                    -x["fused_score"],
                )
            )
            return candidates[:top_k], trace

        if target_mode == "keyword":
            candidates = list(combined.values())
            for cand in candidates:
                ranks = cand.pop("_all_ranks")
                cand["fused_score"] = compute_rrf_score(ranks, rrf_k=self.rrf_k)
            candidates.sort(
                key=lambda x: (
                    x["keyword_rank"] if x["keyword_rank"] is not None else 999999,
                    -x["fused_score"],
                )
            )
            return candidates[:top_k], trace

        # Hybrid or Reranked mode: compute RRF across all ranks (methods + queries)
        candidates = list(combined.values())
        for cand in candidates:
            ranks = cand.pop("_all_ranks")
            cand["fused_score"] = compute_rrf_score(ranks, rrf_k=self.rrf_k)

        candidates.sort(key=lambda x: x["fused_score"], reverse=True)

        if mode == "reranked":
            rerank_k = top_k_rerank if top_k_rerank else top_k
            reranked_results, latency = self.reranker_service.rerank(
                query, candidates, top_k=rerank_k
            )
            trace.rerank_latency = latency
            return reranked_results, trace

        return candidates[:top_k], trace
