import argparse
import csv
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.context_builder import resolve_children_to_parents
from app.conversation import ConversationState
from app.models import QueryMetadata
from app.query_processor import rewrite_query
from app.retriever import HybridRetriever
from app.scoring import score_candidates
from evaluation.retrieval_metrics import (
    average_retrieval_latency,
    exact_version_success_rate,
    hit_rate_at_k,
    mean_reciprocal_rank,
    metadata_filter_success_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


CONFIGURATIONS = [
    "Semantic only",
    "BM25 only",
    "Hybrid",
    "Hybrid + metadata",
    "Hybrid + reranking",
    "Hybrid + reranking + query rewriting",
    "Full retrieval pipeline",
]


def run_single_eval(
    item: dict,
    config_name: str,
    retriever: HybridRetriever,
    top_k: int = 5,
) -> dict:
    started = time.perf_counter()

    question = item["question"]
    context_text = item.get("conversation_context")
    expected_source_ids = item.get("expected_source_ids", [])
    target_version = item.get("version")
    target_category = item.get("category")

    conv_state = None
    if context_text:
        conv_state = ConversationState()
        conv_state.update_from_question(context_text)
        conv_state.update_from_answer("Context prior turn.")

    # Determine rewriting
    query_to_use = question
    rewrite_used = False
    if config_name in ("Hybrid + reranking + query rewriting", "Full retrieval pipeline"):
        if context_text:
            query_to_use, rw_trace = rewrite_query(question, conversation_state=conv_state)
            rewrite_used = rw_trace.rewrite_used
        else:
            query_to_use = question

    # Determine metadata filtering
    where_filter = None
    requested_filters = {}
    if config_name in ("Hybrid + metadata", "Hybrid + reranking", "Hybrid + reranking + query rewriting", "Full retrieval pipeline"):
        meta = QueryMetadata(version=target_version, category=target_category)
        if target_version:
            requested_filters["version"] = target_version
        if target_category:
            requested_filters["category"] = target_category

    # Map configuration to retrieval mode
    mode = "hybrid"
    if config_name == "Semantic only":
        mode = "semantic"
    elif config_name == "BM25 only":
        mode = "keyword"
    elif config_name in ("Hybrid", "Hybrid + metadata"):
        mode = "hybrid"
    elif config_name in ("Hybrid + reranking", "Hybrid + reranking + query rewriting"):
        mode = "reranked"
    elif config_name == "Full retrieval pipeline":
        mode = "full"

    effective_mode = "reranked" if mode == "full" else mode

    raw_candidates, trace = retriever.retrieve(
        query=query_to_use,
        top_k=top_k,
        mode=effective_mode,
        enable_multi_query=False if config_name in ("Semantic only", "BM25 only") else None,
    )

    final_candidates = raw_candidates
    if config_name == "Full retrieval pipeline":
        scored = score_candidates(query=query_to_use, candidates=raw_candidates)
        final_candidates = resolve_children_to_parents(scored)

    latency = time.perf_counter() - started

    # Extract IDs & Scores
    retrieved_source_ids = []
    retrieved_versions = []
    scores_list = []

    for rank_idx, cand in enumerate(final_candidates, start=1):
        meta_d = cand.get("metadata", {}) if isinstance(cand.get("metadata"), dict) else {}
        sid = meta_d.get("source") or cand.get("document_id") or cand.get("chunk_id")
        # Extract basename or doc ID
        clean_sid = Path(str(sid)).stem if sid else cand.get("chunk_id")
        retrieved_source_ids.append(clean_sid)
        retrieved_versions.append(meta_d.get("version"))

        scores_list.append({
            "rank": rank_idx,
            "chunk_id": cand.get("chunk_id"),
            "source_id": clean_sid,
            "semantic_score": cand.get("semantic_score"),
            "keyword_score": cand.get("keyword_score"),
            "fused_score": cand.get("fused_score"),
            "reranker_score": cand.get("reranker_score"),
            "authority_score": cand.get("authority_score"),
            "freshness_score": cand.get("freshness_score"),
            "version_score": cand.get("version_score"),
            "final_score": cand.get("final_score"),
        })

    # Metrics
    hr = hit_rate_at_k(retrieved_source_ids, expected_source_ids, k=top_k)
    prec = precision_at_k(retrieved_source_ids, expected_source_ids, k=top_k)
    rec = recall_at_k(retrieved_source_ids, expected_source_ids, k=top_k)
    rr = reciprocal_rank(retrieved_source_ids, expected_source_ids)
    ndcg = ndcg_at_k(retrieved_source_ids, expected_source_ids, k=top_k)

    return {
        "question_id": item["id"],
        "configuration": config_name,
        "question": question,
        "rewritten_query": query_to_use,
        "test_type": item.get("test_type"),
        "difficulty": item.get("difficulty"),
        "expected_source_ids": expected_source_ids,
        "retrieved_source_ids": retrieved_source_ids,
        "scores": scores_list,
        "requested_filters": requested_filters,
        "applied_filters": trace.filters_applied,
        "fallback_used": trace.fallback_used,
        "latency_seconds": latency,
        "hit_rate_5": hr,
        "precision_5": prec,
        "recall_5": rec,
        "reciprocal_rank": rr,
        "ndcg_5": ndcg,
        "retrieved_versions": retrieved_versions,
        "target_version": target_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightFlow RAG Retrieval Benchmark Runner")
    parser.add_argument(
        "--questions",
        default="evaluation/questions.json",
        help="Path to questions evaluation JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="storage/evaluation",
        help="Output storage directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K retrieved items for evaluation",
    )

    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"Error: Questions file {questions_path} does not exist.")
        sys.exit(1)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STARTING RETRIEVAL BENCHMARK EVALUATION ({len(questions_data)} Questions x {len(CONFIGURATIONS)} Configs)")
    print("=" * 80)

    retriever = HybridRetriever()

    detailed_results = []
    summary_by_config = {}

    for config_name in CONFIGURATIONS:
        print(f"\nRunning Configuration [{CONFIGURATIONS.index(config_name)+1}/7]: [{config_name}]...")
        config_evals = []

        for q_idx, item in enumerate(questions_data, start=1):
            res = run_single_eval(item, config_name, retriever, top_k=args.top_k)
            detailed_results.append(res)
            config_evals.append(res)

            if q_idx % 15 == 0 or q_idx == len(questions_data):
                print(f"  - Progress: {q_idx}/{len(questions_data)} queries completed.")

        # Aggregate metrics for config
        hrs = [r["hit_rate_5"] for r in config_evals]
        precs = [r["precision_5"] for r in config_evals]
        recs = [r["recall_5"] for r in config_evals]
        rrs = [r["reciprocal_rank"] for r in config_evals]
        ndcgs = [r["ndcg_5"] for r in config_evals]
        lats = [r["latency_seconds"] for r in config_evals]
        fallbacks = [1.0 if r["fallback_used"] else 0.0 for r in config_evals]

        applied_filters_list = [r["applied_filters"] for r in config_evals]
        expected_filters_list = [r["requested_filters"] for r in config_evals]
        filter_success = metadata_filter_success_rate(applied_filters_list, expected_filters_list)

        ret_ver_list = [r["retrieved_versions"][0] if r["retrieved_versions"] else None for r in config_evals]
        tgt_ver_list = [r["target_version"] for r in config_evals]
        ver_success = exact_version_success_rate(ret_ver_list, tgt_ver_list)

        summary_by_config[config_name] = {
            "configuration": config_name,
            "hit_rate_5": sum(hrs) / len(hrs),
            "precision_5": sum(precs) / len(precs),
            "recall_5": sum(recs) / len(recs),
            "mrr": mean_reciprocal_rank(rrs),
            "ndcg_5": sum(ndcgs) / len(ndcgs),
            "filter_success_rate": filter_success,
            "version_success_rate": ver_success,
            "fallback_rate": sum(fallbacks) / len(fallbacks),
            "avg_latency_s": average_retrieval_latency(lats),
        }

    # Save detailed JSON
    detailed_json_path = out_dir / "retrieval_eval_detailed.json"
    with open(detailed_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, default=str)

    # Save detailed CSV
    detailed_csv_path = out_dir / "retrieval_eval_detailed.csv"
    fieldnames = [
        "question_id", "configuration", "question", "rewritten_query", "test_type",
        "difficulty", "expected_source_ids", "retrieved_source_ids", "fallback_used",
        "latency_seconds", "hit_rate_5", "precision_5", "recall_5", "reciprocal_rank", "ndcg_5"
    ]
    with open(detailed_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in detailed_results:
            row_copy = dict(row)
            row_copy["expected_source_ids"] = ";".join(row_copy["expected_source_ids"])
            row_copy["retrieved_source_ids"] = ";".join(row_copy["retrieved_source_ids"])
            writer.writerow(row_copy)

    # Save summary CSV
    summary_csv_path = out_dir / "retrieval_eval_summary.csv"
    summary_fieldnames = [
        "configuration", "hit_rate_5", "precision_5", "recall_5", "mrr", "ndcg_5",
        "filter_success_rate", "version_success_rate", "fallback_rate", "avg_latency_s"
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for cfg in CONFIGURATIONS:
            writer.writerow(summary_by_config[cfg])

    print("\n" + "=" * 105)
    print("RETRIEVAL EVALUATION BENCHMARK SUMMARY COMPARISON TABLE")
    print("=" * 105)
    print(f"{'Configuration':<38} | {'HitRate@5':<9} | {'Prec@5':<7} | {'Recall@5':<8} | {'MRR':<6} | {'NDCG@5':<7} | {'Latency':<8}")
    print("-" * 105)
    for cfg in CONFIGURATIONS:
        s = summary_by_config[cfg]
        print(
            f"{s['configuration']:<38} | "
            f"{s['hit_rate_5']:<9.4f} | "
            f"{s['precision_5']:<7.4f} | "
            f"{s['recall_5']:<8.4f} | "
            f"{s['mrr']:<6.4f} | "
            f"{s['ndcg_5']:<7.4f} | "
            f"{s['avg_latency_s']:<8.4f}s"
        )
    print("=" * 105)
    print(f"\nArtifacts saved successfully to {out_dir}:")
    print(f"  - {detailed_json_path}")
    print(f"  - {detailed_csv_path}")
    print(f"  - {summary_csv_path}")


if __name__ == "__main__":
    main()
