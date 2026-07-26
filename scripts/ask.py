import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import BasicRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightFlow Support RAG Q&A Assistant CLI")
    parser.add_argument("question", help="User support question")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display detailed pipeline execution trace and 17-stage latency breakdown in JSON format",
    )
    args = parser.parse_args()

    pipeline = BasicRAGPipeline()
    result = pipeline.answer(args.question)
    ans = result.get("answer", {})

    print("=" * 80)
    print("INSIGHTFLOW SUPPORT ASSISTANT ANSWER")
    print("=" * 80)

    print("\n--- ANSWER ---")
    print(ans.get("answer", "No answer text available."))

    if ans.get("likely_cause"):
        print("\n--- LIKELY CAUSE ---")
        print(ans["likely_cause"])

    steps = ans.get("resolution_steps", [])
    if steps:
        print("\n--- RESOLUTION STEPS ---")
        for idx, step in enumerate(steps, start=1):
            print(f"{idx}. {step}")

    print("\n" + "=" * 80)
    print("CONFIDENCE & EVALUATION")
    print("=" * 80)

    conf_level = ans.get("confidence", "low").upper()
    conf_score = ans.get("confidence_score", 0.0)
    print(f"Confidence Level: {conf_level} (Score: {conf_score:.4f})")

    esc_req = ans.get("escalation_required", False)
    esc_reason = ans.get("escalation_reason") or "N/A"
    print(f"Escalation Required: {esc_req}")
    if esc_req:
        print(f"Escalation Reason:   {esc_reason}")

    conflicts_detected = ans.get("conflicts_detected", False)
    print(f"Conflict Detected:   {conflicts_detected}")
    conflict_info = result.get("trace", {}).get("conflict_result", {})
    if conflicts_detected and conflict_info:
        print(f"Conflict Summary:    {conflict_info.get('summary')}")
        if conflict_info.get("preference_reason"):
            print(f"Preference Reason:   {conflict_info.get('preference_reason')}")
        if conflict_info.get("unresolved"):
            print("WARNING: Conflict remains UNRESOLVED between equally authoritative sources.")

    citations = result.get("citations", [])
    print("\n" + "=" * 80)
    print(f"RESOLVED CITATIONS (Count: {len(citations)})")
    print("=" * 80)

    if citations:
        for c in citations:
            print(f"[{c.get('source_id')}] {c.get('title')} ({c.get('source')})")
            quoted = c.get("quoted_text", "").replace("\n", " ")
            print(f"    Quoted Passage: \"{quoted[:150]}...\"")
    else:
        print("No citations provided or required for this response.")

    if args.debug:
        trace = result.get("trace", {})
        print("\n" + "=" * 80)
        print("PIPELINE LATENCY BREAKDOWN (17 STAGES)")
        print("=" * 80)
        print(f"{'Pipeline Stage':<38} | {'Latency (seconds)':<18}")
        print("-" * 60)
        timing_keys = [
            "metadata_extraction_latency",
            "query_rewriting_latency",
            "query_expansion_latency",
            "embedding_latency",
            "semantic_retrieval_latency",
            "bm25_retrieval_latency",
            "fusion_latency",
            "reranking_latency",
            "authority_freshness_scoring_latency",
            "parent_resolution_latency",
            "context_building_latency",
            "conflict_detection_latency",
            "generation_latency",
            "validation_latency",
            "citation_verification_latency",
            "confidence_calculation_latency",
            "total_latency",
        ]
        for k in timing_keys:
            val = trace.get(k, 0.0) or 0.0
            stage_name = k.replace("_latency", "").replace("_", " ").title()
            print(f"{stage_name:<38} | {val:<18.6f}s")
        print("=" * 80)
        print("\nFULL PIPELINE DEBUG TRACE JSON:")
        print(json.dumps(trace, indent=2, default=str))


if __name__ == "__main__":
    main()