import argparse
import csv
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.conversation import ConversationState
from app.pipeline import BasicRAGPipeline
from evaluation.answer_metrics import (
    evaluate_deterministic_answer_quality,
    judge_answer_quality,
)


def run_single_answer_eval(
    item: dict,
    pipeline: BasicRAGPipeline,
    use_judge: bool = False,
) -> dict:
    started = time.perf_counter()

    question = item["question"]
    context_text = item.get("conversation_context")
    expected_points = item.get("expected_answer_points", [])

    # Initialize conversation state if context is provided
    if context_text:
        pipeline.conversation_state.clear()
        pipeline.conversation_state.update_from_question(context_text)
        pipeline.conversation_state.update_from_answer("Prior context turn.")
    else:
        pipeline.conversation_state.clear()

    # 1. Run full pipeline
    pipeline_result = pipeline.answer(question)
    latency = time.perf_counter() - started

    # 2. Extract structured answer data
    ans_data = pipeline_result.get("answer", {})
    ans_text = ans_data.get("answer", "") if isinstance(ans_data, dict) else str(ans_data)

    # 3. Compute deterministic metrics
    det_metrics = evaluate_deterministic_answer_quality(pipeline_result, item)

    # 4. Optional local LLM judging
    judge_metrics = None
    if use_judge:
        formatted_context = pipeline.build_context(pipeline_result.get("sources", []))
        judge_metrics = judge_answer_quality(
            question=question,
            retrieved_context=formatted_context,
            generated_answer=ans_data,
            expected_answer_points=expected_points,
            generator=pipeline.generator,
        )

    res = {
        "question_id": item["id"],
        "question": question,
        "test_type": item.get("test_type"),
        "difficulty": item.get("difficulty"),
        "answerable": item.get("answerable"),
        "expected_abstention": item.get("expected_abstention"),
        "expected_escalation": item.get("expected_escalation"),
        "generated_answer": ans_text,
        "likely_cause": ans_data.get("likely_cause"),
        "resolution_steps": ans_data.get("resolution_steps", []),
        "confidence": ans_data.get("confidence"),
        "confidence_score": ans_data.get("confidence_score"),
        "escalation_required": ans_data.get("escalation_required"),
        "escalation_reason": ans_data.get("escalation_reason"),
        "conflicts_detected": ans_data.get("conflicts_detected"),
        "citations": [c.get("source_id") for c in pipeline_result.get("citations", [])],
        "latency_seconds": latency,
        "pipeline_trace": pipeline_result.get("pipeline_trace"),
        "deterministic_metrics": det_metrics,
        "judge_metrics": judge_metrics,
    }

    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightFlow Support RAG Answer Evaluation Runner")
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
        "--use-judge",
        action="store_true",
        help="Enable optional local LLM judging (0-2 rubric)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of evaluation cases for fast runs",
    )

    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"Error: Questions file {questions_path} does not exist.")
        sys.exit(1)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    if args.max_cases and args.max_cases > 0:
        questions_data = questions_data[:args.max_cases]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STARTING ANSWER EVALUATION BENCHMARK ({len(questions_data)} Test Cases)")
    print(f"Local LLM Judging: {'ENABLED' if args.use_judge else 'DISABLED'}")
    print("=" * 80)

    pipeline = BasicRAGPipeline()
    results = []

    for q_idx, item in enumerate(questions_data, start=1):
        print(f"[{q_idx}/{len(questions_data)}] Evaluating: {item['id']} ({item['test_type']})...")
        res = run_single_answer_eval(item, pipeline, use_judge=args.use_judge)
        results.append(res)

    # Compute Aggregate Metrics
    total = len(results)
    schema_valid_rate = sum(1 for r in results if r["deterministic_metrics"]["schema_validity"]) / total
    cit_valid_rate = sum(1 for r in results if r["deterministic_metrics"]["citation_id_validity"]) / total
    cit_presence_rate = sum(1 for r in results if r["deterministic_metrics"]["citation_presence"]) / total
    mean_coverage = sum(r["deterministic_metrics"]["answer_point_coverage"] for r in results) / total
    correct_abst_rate = sum(1 for r in results if r["deterministic_metrics"]["correct_abstention"]) / total
    correct_esc_rate = sum(1 for r in results if r["deterministic_metrics"]["correct_escalation"]) / total
    conflict_disc_rate = sum(1 for r in results if r["deterministic_metrics"]["conflict_disclosure"]) / total
    non_empty_steps_rate = sum(1 for r in results if r["deterministic_metrics"]["non_empty_resolution_steps_when_expected"]) / total
    no_steps_abst_rate = sum(1 for r in results if r["deterministic_metrics"]["no_resolution_steps_when_abstaining"]) / total
    avg_latency = sum(r["latency_seconds"] for r in results) / total

    # Save detailed JSON
    detailed_json_path = out_dir / "answer_eval_detailed.json"
    with open(detailed_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Save detailed CSV
    detailed_csv_path = out_dir / "answer_eval_detailed.csv"
    csv_fieldnames = [
        "question_id", "question", "test_type", "difficulty", "answerable",
        "expected_abstention", "expected_escalation", "confidence", "confidence_score",
        "escalation_required", "conflicts_detected", "citations_count", "latency_seconds",
        "schema_validity", "citation_id_validity", "answer_point_coverage", "correct_abstention",
        "correct_escalation", "conflict_disclosure"
    ]
    with open(detailed_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {
                "question_id": r["question_id"],
                "question": r["question"],
                "test_type": r["test_type"],
                "difficulty": r["difficulty"],
                "answerable": r["answerable"],
                "expected_abstention": r["expected_abstention"],
                "expected_escalation": r["expected_escalation"],
                "confidence": r["confidence"],
                "confidence_score": r["confidence_score"],
                "escalation_required": r["escalation_required"],
                "conflicts_detected": r["conflicts_detected"],
                "citations_count": len(r["citations"]),
                "latency_seconds": round(r["latency_seconds"], 4),
                "schema_validity": r["deterministic_metrics"]["schema_validity"],
                "citation_id_validity": r["deterministic_metrics"]["citation_id_validity"],
                "answer_point_coverage": round(r["deterministic_metrics"]["answer_point_coverage"], 4),
                "correct_abstention": r["deterministic_metrics"]["correct_abstention"],
                "correct_escalation": r["deterministic_metrics"]["correct_escalation"],
                "conflict_disclosure": r["deterministic_metrics"]["conflict_disclosure"],
            }
            writer.writerow(row)

    # Summary Metrics Dict
    summary_dict = {
        "total_test_cases": total,
        "schema_validity_rate": schema_valid_rate,
        "citation_id_validity_rate": cit_valid_rate,
        "citation_presence_rate": cit_presence_rate,
        "mean_answer_point_coverage": mean_coverage,
        "correct_abstention_rate": correct_abst_rate,
        "correct_escalation_rate": correct_esc_rate,
        "conflict_disclosure_rate": conflict_disc_rate,
        "non_empty_resolution_steps_rate": non_empty_steps_rate,
        "no_steps_when_abstaining_rate": no_steps_abst_rate,
        "avg_latency_seconds": avg_latency,
    }

    if args.use_judge:
        j_faiths = [r["judge_metrics"]["faithfulness"] for r in results if r.get("judge_metrics")]
        j_rel = [r["judge_metrics"]["answer_relevance"] for r in results if r.get("judge_metrics")]
        j_cit = [r["judge_metrics"]["citation_support"] for r in results if r.get("judge_metrics")]
        j_conf = [r["judge_metrics"]["conflict_handling"] for r in results if r.get("judge_metrics")]

        summary_dict["judge_mean_faithfulness"] = sum(j_faiths) / max(1, len(j_faiths))
        summary_dict["judge_mean_answer_relevance"] = sum(j_rel) / max(1, len(j_rel))
        summary_dict["judge_mean_citation_support"] = sum(j_cit) / max(1, len(j_cit))
        summary_dict["judge_mean_conflict_handling"] = sum(j_conf) / max(1, len(j_conf))

    # Save summary CSV
    summary_csv_path = out_dir / "answer_eval_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_dict.keys()))
        writer.writeheader()
        writer.writerow(summary_dict)

    # Print Formatted Summary Metrics Table
    print("\n" + "=" * 80)
    print("END-TO-END ANSWER EVALUATION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Evaluation Cases:                  {total}")
    print(f"Pydantic Schema Validity Rate:           {schema_valid_rate * 100:.2f}%")
    print(f"Citation ID Validity Rate:               {cit_valid_rate * 100:.2f}%")
    print(f"Citation Presence Rate:                  {cit_presence_rate * 100:.2f}%")
    print(f"Mean Answer-Point Coverage:              {mean_coverage * 100:.2f}%")
    print(f"Correct Abstention Accuracy:             {correct_abst_rate * 100:.2f}%")
    print(f"Correct Escalation Accuracy:             {correct_esc_rate * 100:.2f}%")
    print(f"Conflict Disclosure Accuracy:            {conflict_disc_rate * 100:.2f}%")
    print(f"Non-Empty Resolution Steps Rate:         {non_empty_steps_rate * 100:.2f}%")
    print(f"No Steps When Abstaining Rate:           {no_steps_abst_rate * 100:.2f}%")

    if args.use_judge:
        print("-" * 80)
        print("LOCAL LLM JUDGE METRICS (0-2 Rubric):")
        print(f"  - Mean Faithfulness:                    {summary_dict.get('judge_mean_faithfulness', 0):.2f} / 2.0")
        print(f"  - Mean Answer Relevance:                {summary_dict.get('judge_mean_answer_relevance', 0):.2f} / 2.0")
        print(f"  - Mean Citation Support:                {summary_dict.get('judge_mean_citation_support', 0):.2f} / 2.0")
        print(f"  - Mean Conflict Handling:               {summary_dict.get('judge_mean_conflict_handling', 0):.2f} / 2.0")

    print("-" * 80)
    print(f"Average Pipeline Latency:                {avg_latency:.2f}s")
    print("=" * 80)
    print(f"\nArtifacts saved successfully to {out_dir}:")
    print(f"  - {detailed_json_path}")
    print(f"  - {detailed_csv_path}")
    print(f"  - {summary_csv_path}")


if __name__ == "__main__":
    main()
