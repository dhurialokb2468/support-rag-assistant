import argparse
import csv
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bm25_store import BM25Store
from app.chunking import chunk_documents
from app.context_builder import resolve_children_to_parents
from app.loaders import load_all_documents
from app.retriever import HybridRetriever
from app.scoring import score_candidates
from app.vector_store import VectorStore
from evaluation.retrieval_metrics import (
    average_retrieval_latency,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

CHUNKING_CONFIGS = [
    {
        "name": "fixed 300",
        "strategy": "fixed",
        "chunk_size": 300,
        "overlap": 50,
    },
    {
        "name": "fixed 500",
        "strategy": "fixed",
        "chunk_size": 500,
        "overlap": 80,
    },
    {
        "name": "fixed 800",
        "strategy": "fixed",
        "chunk_size": 800,
        "overlap": 120,
    },
    {
        "name": "fixed 1200",
        "strategy": "fixed",
        "chunk_size": 1200,
        "overlap": 150,
    },
    {
        "name": "section-aware",
        "strategy": "section",
        "chunk_size": 800,
        "overlap": 120,
    },
    {
        "name": "parent-child",
        "strategy": "parent-child",
        "parent_size": 2000,
        "child_size": 500,
        "child_overlap": 80,
    },
]


def run_experiment_for_config(
    cfg: dict,
    documents: list,
    questions_data: list,
    top_k: int = 5,
) -> tuple[dict, list[dict]]:
    cfg_name = cfg["name"]
    strategy = cfg["strategy"]

    ingest_start = time.perf_counter()

    # 1. Chunk documents according to configuration
    if strategy == "parent-child":
        chunks = chunk_documents(
            documents,
            strategy=strategy,
            parent_size=cfg.get("parent_size", 2000),
            child_size=cfg.get("child_size", 500),
            child_overlap=cfg.get("child_overlap", 80),
        )
    else:
        chunks = chunk_documents(
            documents,
            strategy=strategy,
            chunk_size=cfg.get("chunk_size", 800),
            overlap=cfg.get("overlap", 120),
        )

    # 2. Rebuild index (reset VectorStore and BM25Store)
    vector_store = VectorStore()
    vector_store.reset()
    vector_store.add_chunks(chunks)

    bm25_store = BM25Store()
    bm25_store.index_chunks(chunks)
    bm25_store.save()

    ingestion_time = time.perf_counter() - ingest_start
    chunk_count = len(chunks)

    # 3. Instantiate fresh Retriever with newly created stores
    retriever = HybridRetriever(vector_store=vector_store, bm25_store=bm25_store)

    query_evals = []
    latencies = []

    for item in questions_data:
        q_start = time.perf_counter()
        question = item["question"]
        expected_sids = item.get("expected_source_ids", [])

        # Retrieve candidates
        raw_candidates, trace = retriever.retrieve(
            query=question,
            top_k=top_k,
            mode="hybrid",
        )

        final_candidates = raw_candidates
        if strategy == "parent-child":
            scored = score_candidates(query=question, candidates=raw_candidates)
            final_candidates = resolve_children_to_parents(scored)

        q_lat = time.perf_counter() - q_start
        latencies.append(q_lat)

        # Extract retrieved source IDs (clean doc IDs / file basenames)
        retrieved_sids = []
        for cand in final_candidates:
            meta = cand.get("metadata", {}) if isinstance(cand.get("metadata"), dict) else {}
            sid = meta.get("source") or cand.get("document_id") or cand.get("chunk_id")
            clean_sid = Path(str(sid)).stem if sid else cand.get("chunk_id")
            retrieved_sids.append(clean_sid)

        # Calculate metrics for query
        hr = hit_rate_at_k(retrieved_sids, expected_sids, k=top_k)
        prec = precision_at_k(retrieved_sids, expected_sids, k=top_k)
        rec = recall_at_k(retrieved_sids, expected_sids, k=top_k)
        rr = reciprocal_rank(retrieved_sids, expected_sids)
        ndcg = ndcg_at_k(retrieved_sids, expected_sids, k=top_k)

        query_evals.append({
            "config_name": cfg_name,
            "question_id": item["id"],
            "question": question,
            "expected_source_ids": expected_sids,
            "retrieved_source_ids": retrieved_sids,
            "latency_seconds": q_lat,
            "hit_rate_5": hr,
            "precision_5": prec,
            "recall_5": rec,
            "reciprocal_rank": rr,
            "ndcg_5": ndcg,
        })

    # Summary metrics for config
    hrs = [q["hit_rate_5"] for q in query_evals]
    precs = [q["precision_5"] for q in query_evals]
    recs = [q["recall_5"] for q in query_evals]
    rrs = [q["reciprocal_rank"] for q in query_evals]
    ndcgs = [q["ndcg_5"] for q in query_evals]

    summary = {
        "config_name": cfg_name,
        "strategy": strategy,
        "number_of_chunks": chunk_count,
        "ingestion_time_s": round(ingestion_time, 4),
        "hit_rate_5": round(sum(hrs) / len(hrs), 4),
        "precision_5": round(sum(precs) / len(precs), 4),
        "recall_5": round(sum(recs) / len(recs), 4),
        "mrr": round(mean_reciprocal_rank(rrs), 4),
        "ndcg_5": round(sum(ndcgs) / len(ndcgs), 4),
        "avg_retrieval_latency_s": round(average_retrieval_latency(latencies), 4),
    }

    return summary, query_evals


def restore_primary_index(documents: list) -> None:
    """Restores primary parent-child index to leave vector store in default state."""
    chunks = chunk_documents(documents, strategy="parent-child")
    vector_store = VectorStore()
    vector_store.reset()
    vector_store.add_chunks(chunks)

    bm25_store = BM25Store()
    bm25_store.index_chunks(chunks)
    bm25_store.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightFlow RAG Chunking Experiment Runner")
    parser.add_argument(
        "--questions",
        default="evaluation/questions.json",
        help="Path to questions evaluation JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="storage/evaluation/chunking",
        help="Output storage directory for chunking experiment results",
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
    print(f"STARTING CHUNKING STRATEGY EXPERIMENT ({len(CHUNKING_CONFIGS)} Configurations x {len(questions_data)} Questions)")
    print("=" * 80)

    documents = load_all_documents()
    all_summaries = []
    all_query_details = []

    for idx, cfg in enumerate(CHUNKING_CONFIGS, start=1):
        cfg_name = cfg["name"]
        print(f"\n[{idx}/{len(CHUNKING_CONFIGS)}] Running Configuration: [{cfg_name}]...")
        summary, query_details = run_experiment_for_config(
            cfg, documents, questions_data, top_k=args.top_k
        )
        all_summaries.append(summary)
        all_query_details.extend(query_details)
        print(f"  -> Chunks: {summary['number_of_chunks']} | Ingestion Time: {summary['ingestion_time_s']}s | HitRate@5: {summary['hit_rate_5']:.4f} | MRR: {summary['mrr']:.4f}")

    # Restore primary index
    print("\nRestoring primary parent-child index...")
    restore_primary_index(documents)

    # Save detailed JSON
    detailed_json_path = out_dir / "chunking_experiment_detailed.json"
    with open(detailed_json_path, "w", encoding="utf-8") as f:
        json.dump({"summaries": all_summaries, "query_details": all_query_details}, f, indent=2)

    # Save summary CSV
    summary_csv_path = out_dir / "chunking_experiment_summary.csv"
    summary_fieldnames = [
        "config_name", "strategy", "number_of_chunks", "ingestion_time_s",
        "hit_rate_5", "precision_5", "recall_5", "mrr", "ndcg_5", "avg_retrieval_latency_s"
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for sum_row in all_summaries:
            writer.writerow(sum_row)

    print("\n" + "=" * 105)
    print("CHUNKING STRATEGY EXPERIMENT COMPARISON TABLE")
    print("=" * 105)
    print(f"{'Chunking Strategy':<18} | {'Chunks':<7} | {'Ingest Time':<11} | {'HitRate@5':<9} | {'Prec@5':<7} | {'Recall@5':<8} | {'MRR':<6} | {'NDCG@5':<7} | {'Latency':<8}")
    print("-" * 105)
    for s in all_summaries:
        print(
            f"{s['config_name']:<18} | "
            f"{s['number_of_chunks']:<7} | "
            f"{s['ingestion_time_s']:<11.4f} | "
            f"{s['hit_rate_5']:<9.4f} | "
            f"{s['precision_5']:<7.4f} | "
            f"{s['recall_5']:<8.4f} | "
            f"{s['mrr']:<6.4f} | "
            f"{s['ndcg_5']:<7.4f} | "
            f"{s['avg_retrieval_latency_s']:<8.4f}s"
        )
    print("=" * 105)
    print(f"\nArtifacts saved successfully to {out_dir}:")
    print(f"  - {detailed_json_path}")
    print(f"  - {summary_csv_path}")


if __name__ == "__main__":
    main()
