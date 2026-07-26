import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.context_builder import resolve_children_to_parents
from app.conversation import ConversationState
from app.query_processor import rewrite_query
from app.retriever import HybridRetriever
from app.scoring import score_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightFlow Support RAG Search CLI")
    parser.add_argument("query", help="User search query or question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--mode",
        choices=["semantic", "keyword", "hybrid", "reranked", "full"],
        default="full",
        help="Retrieval mode: semantic, keyword, hybrid, reranked, or full (reranked + authority/freshness scoring + parent resolution)",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Prior conversation context for query rewriting demo",
    )
    args = parser.parse_args()

    conv_state = ConversationState()
    if args.context:
        conv_state.update_from_question(args.context)
        conv_state.update_from_answer("Prior answer context.")

    query_to_search, rewrite_trace = rewrite_query(
        question=args.query,
        conversation_state=conv_state,
    )

    print("=" * 80)
    print("QUERY REWRITE TRACE:")
    print(f"Original Question: {rewrite_trace.original_question}")
    print(f"Rewritten Query:   {rewrite_trace.rewritten_query}")
    print(f"Rewrite Used:      {rewrite_trace.rewrite_used}")
    print(f"Rewrite Latency:   {rewrite_trace.rewrite_latency:.4f}s")

    retriever = HybridRetriever()
    effective_mode = "reranked" if args.mode == "full" else args.mode

    results, trace = retriever.retrieve(
        query_to_search,
        top_k=args.top_k,
        mode=effective_mode,
    )
    trace.rewrite_trace = rewrite_trace

    if args.mode == "full":
        scored_results = score_candidates(query=query_to_search, candidates=results)
        results = resolve_children_to_parents(scored_results)

    print("=" * 80)
    print(f"SEARCH MODE: {args.mode.upper()}")
    print(f"Generated Search Queries: {trace.generated_queries}")
    print(f"Metadata Filters Requested: {trace.filters_requested}")
    print(f"Metadata Filters Applied:   {trace.filters_applied}")
    print(f"Strict Retrieval Succeeded: {trace.strict_succeeded}")
    print(f"Relaxed Fallback Used:     {trace.fallback_used}")
    if trace.rerank_latency is not None:
        print(f"Rerank Latency:            {trace.rerank_latency:.4f}s")

    print("=" * 80)
    print(f"RETRIEVED CANDIDATES (Count: {len(results)}):")

    for index, item in enumerate(results, start=1):
        meta = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        print("=" * 80)
        print(f"Rank {index}: {meta.get('title', 'Untitled')}")
        print(f"Source Path: {meta.get('source', 'Unknown')}")
        print(f"Doc Type:    {meta.get('document_type', 'N/A')}")
        print(f"Version:     {meta.get('version', 'N/A')}")
        print(f"Category:    {meta.get('category', 'N/A')}")
        print(f"Updated At:  {meta.get('updated_at', 'N/A')}")
        print(f"Reviewed:    {meta.get('reviewed', False)}")
        methods = ", ".join(item.get("retrieval_methods", []))
        print(f"Retrieval Methods: [{methods}]")

        # Every score display
        sem_score = item.get("semantic_score")
        kw_score = item.get("keyword_score")
        fused_score = item.get("fused_score")
        reranker_score = item.get("reranker_score")
        raw_reranker = item.get("raw_reranker_score")
        auth_score = item.get("authority_score")
        fresh_score = item.get("freshness_score")
        ver_score = item.get("version_score")
        final_score = item.get("final_score")

        print("Scores Breakdown:")
        if sem_score is not None:
            print(f"  - Semantic Score:  {sem_score:.4f} (Rank: {item.get('semantic_rank')})")
        if kw_score is not None:
            print(f"  - Keyword Score:   {kw_score:.4f} (Rank: {item.get('keyword_rank')})")
        if fused_score is not None:
            print(f"  - Fused RRF Score: {fused_score:.6f}")
        if reranker_score is not None:
            print(f"  - Reranker Score:  {reranker_score:.4f} (Raw: {raw_reranker:.4f})")
        if auth_score is not None:
            print(f"  - Authority Score: {auth_score:.4f}")
        if fresh_score is not None:
            print(f"  - Freshness Score: {fresh_score:.4f}")
        if ver_score is not None:
            print(f"  - Version Score:   {ver_score:.4f}")
        if final_score is not None:
            print(f"  - FINAL COMPOSITE SCORE: {final_score:.4f}")

        if item.get("parent_id") or item.get("is_parent"):
            print("Parent Chunk Info:")
            print(f"  - Parent ID: {item.get('parent_id') or item.get('chunk_id')}")
            if item.get("parent_text"):
                print(f"  - Parent Text Snippet: {item['parent_text'][:150]}...")
            if item.get("triggered_children"):
                print(f"  - Triggered Children Chunks: {item.get('triggered_children')}")

        print("-" * 80)
        snippet = item.get("text", "")[:350].replace("\n", " ")
        print(f"Text Snippet:\n{snippet}...")


if __name__ == "__main__":
    main()