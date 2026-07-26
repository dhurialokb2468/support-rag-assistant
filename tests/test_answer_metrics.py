from unittest.mock import MagicMock

from evaluation.answer_metrics import (
    eval_answer_point_coverage,
    eval_citation_id_validity,
    eval_citation_presence,
    eval_conflict_disclosure,
    eval_correct_abstention,
    eval_correct_escalation,
    eval_pydantic_schema_validity,
    eval_resolution_steps_structure,
    evaluate_deterministic_answer_quality,
    judge_answer_quality,
)


def create_sample_pipeline_result(
    answer="Clear report configuration cache and recreate export settings.",
    escalation_required=False,
    conflicts_detected=False,
    abstention_triggered=False,
):
    return {
        "question": "How do I fix EXP-3204?",
        "answer": {
            "answer": answer,
            "likely_cause": "Deprecated report schema.",
            "resolution_steps": ["Clear report configuration cache", "Recreate export settings"],
            "citations": ["S1"],
            "confidence": "high",
            "confidence_score": 0.88,
            "escalation_required": escalation_required,
            "escalation_reason": "Escalation requested" if escalation_required else None,
            "conflicts_detected": conflicts_detected,
        },
        "citations": [
            {
                "source_id": "S1",
                "title": "Release Notes 3.2",
                "source": "version_3_2.md",
                "quoted_text": "Clear cache.",
            }
        ],
        "sources": [
            {
                "source_id": "S1",
                "chunk_id": "c1",
                "text": "Clear report configuration cache.",
                "metadata": {"title": "Release Notes 3.2", "source": "version_3_2.md"},
            }
        ],
        "trace": {
            "citation_validation_passed": True,
            "abstention_triggered": abstention_triggered,
            "conflict_result": {"conflict_detected": conflicts_detected},
        },
    }


def test_eval_pydantic_schema_validity():
    valid_res = create_sample_pipeline_result()
    assert eval_pydantic_schema_validity(valid_res) is True

    invalid_res = {"answer": {"invalid_field": True}}
    assert eval_pydantic_schema_validity(invalid_res) is False


def test_eval_citation_id_validity():
    valid_res = create_sample_pipeline_result()
    assert eval_citation_id_validity(valid_res) is True

    invalid_res = create_sample_pipeline_result()
    invalid_res["trace"]["citation_validation_passed"] = False
    assert eval_citation_id_validity(invalid_res) is False


def test_eval_citation_presence():
    valid_res = create_sample_pipeline_result()
    assert eval_citation_presence(valid_res, expected_answerable=True) is True

    abstained_res = create_sample_pipeline_result(answer="I could not find sufficient evidence.", escalation_required=True, abstention_triggered=True)
    abstained_res["answer"]["citations"] = []
    abstained_res["citations"] = []
    assert eval_citation_presence(abstained_res, expected_answerable=False) is True


def test_eval_answer_point_coverage():
    ans_text = "To resolve EXP-3204, clear the report configuration cache and recreate export settings."
    expected_points = ["Clear report configuration cache", "Recreate export settings"]

    assert eval_answer_point_coverage(ans_text, expected_points) == 1.0

    partial_text = "You should clear the report configuration cache."
    assert eval_answer_point_coverage(partial_text, expected_points) == 0.5


def test_eval_correct_abstention():
    abstained_res = create_sample_pipeline_result(abstention_triggered=True)
    assert eval_correct_abstention(abstained_res, expected_abstention=True) is True

    normal_res = create_sample_pipeline_result(abstention_triggered=False)
    assert eval_correct_abstention(normal_res, expected_abstention=False) is True


def test_eval_correct_escalation():
    esc_res = create_sample_pipeline_result(escalation_required=True)
    assert eval_correct_escalation(esc_res, expected_escalation=True) is True

    normal_res = create_sample_pipeline_result(escalation_required=False)
    assert eval_correct_escalation(normal_res, expected_escalation=False) is True


def test_eval_conflict_disclosure():
    conflict_res = create_sample_pipeline_result(conflicts_detected=True)
    assert eval_conflict_disclosure(conflict_res, is_conflict_case=True) is True


def test_eval_resolution_steps_structure():
    normal_res = create_sample_pipeline_result()
    non_empty, no_steps_abst = eval_resolution_steps_structure(normal_res, expected_answerable=True)
    assert non_empty is True
    assert no_steps_abst is True

    abstained_res = create_sample_pipeline_result(escalation_required=True, abstention_triggered=True)
    abstained_res["answer"]["resolution_steps"] = []
    non_empty, no_steps_abst = eval_resolution_steps_structure(abstained_res, expected_answerable=False)
    assert no_steps_abst is True


def test_evaluate_deterministic_answer_quality():
    pipeline_res = create_sample_pipeline_result()
    question_case = {
        "answerable": True,
        "expected_abstention": False,
        "expected_escalation": False,
        "expected_answer_points": ["Clear report configuration cache", "Recreate export settings"],
    }

    metrics = evaluate_deterministic_answer_quality(pipeline_res, question_case)

    assert metrics["schema_validity"] is True
    assert metrics["citation_id_validity"] is True
    assert metrics["citation_presence"] is True
    assert metrics["answer_point_coverage"] == 1.0
    assert metrics["correct_abstention"] is True
    assert metrics["correct_escalation"] is True


def test_judge_answer_quality_mock():
    mock_gen = MagicMock()
    mock_gen.generate.return_value = (
        '{"faithfulness": 2, "answer_relevance": 2, "citation_support": 2, "conflict_handling": 2, "reasoning": "Fully supported and accurate."}',
        0.15,
    )

    judge_res = judge_answer_quality(
        question="How to fix EXP-3204?",
        retrieved_context="Clear cache.",
        generated_answer="Clear report configuration cache.",
        expected_answer_points=["Clear cache"],
        generator=mock_gen,
    )

    assert judge_res["faithfulness"] == 2
    assert judge_res["answer_relevance"] == 2
    assert judge_res["citation_support"] == 2
    assert judge_res["conflict_handling"] == 2
    assert "Fully supported" in judge_res["reasoning"]
