from app.models import AbstentionReason, SupportAnswer
from app.validators import ABSTENTION_ANSWER_TEXT, evaluate_abstention


def create_sample_answer(
    conflicts_detected: bool = False,
    escalation_required: bool = False,
):
    return SupportAnswer(
        answer="InsightFlow 3.2 CSV export error EXP-3204 is resolved by verifying network configuration.",
        likely_cause="Network configuration error.",
        resolution_steps=["Verify network configuration."],
        citations=["S1"],
        confidence="high",
        confidence_score=0.88,
        escalation_required=escalation_required,
        escalation_reason=None,
        conflicts_detected=conflicts_detected,
    )


def create_sample_chunks():
    return [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "text": "InsightFlow 3.2 CSV export error EXP-3204 occurs when network permissions are invalid.",
            "final_score": 0.90,
            "metadata": {"title": "Guide 3.2", "source": "guide_v3.2.md", "version": "3.2"},
        }
    ]


def test_answerable_question():
    ans = create_sample_answer()
    chunks = create_sample_chunks()

    abstained, reason, final_ans = evaluate_abstention(
        question="How do I fix error EXP-3204 in InsightFlow 3.2?",
        support_answer=ans,
        retrieved_chunks=chunks,
        citation_validation_passed=True,
        confidence_score=0.88,
        confidence_level="high",
    )

    assert abstained is False
    assert reason is None
    assert final_ans.answer.startswith("InsightFlow 3.2 CSV export error")
    assert final_ans.escalation_required is False


def test_unanswerable_future_plan_question():
    ans = create_sample_answer()
    chunks = create_sample_chunks()  # Chunks have guide 3.2, no roadmap/future plans

    abstained, reason, final_ans = evaluate_abstention(
        question="What are the planned features for InsightFlow version 4.0 future release next year?",
        support_answer=ans,
        retrieved_chunks=chunks,
        citation_validation_passed=True,
        confidence_score=0.85,
        confidence_level="high",
    )

    assert abstained is True
    assert reason == AbstentionReason.UNDOCUMENTED_FUTURE_PLANS
    assert final_ans.answer == ABSTENTION_ANSWER_TEXT
    assert final_ans.escalation_required is True
    assert "undocumented_future_plans" in (final_ans.escalation_reason or "")


def test_low_relevance_question():
    ans = create_sample_answer()
    weak_chunks = [
        {
            "chunk_id": "c_weak",
            "document_id": "doc_weak",
            "text": "Generic irrelevant text.",
            "final_score": 0.15,  # Very low relevance
            "metadata": {"title": "Doc", "source": "doc.md"},
        }
    ]

    abstained, reason, final_ans = evaluate_abstention(
        question="Unrelated question about astrophysics?",
        support_answer=ans,
        retrieved_chunks=weak_chunks,
        citation_validation_passed=True,
        confidence_score=0.25,
        confidence_level="low",
    )

    assert abstained is True
    assert reason == AbstentionReason.LOW_RELEVANCE
    assert final_ans.answer == ABSTENTION_ANSWER_TEXT
    assert final_ans.escalation_required is True


def test_invalid_citation_question():
    ans = create_sample_answer()
    chunks = create_sample_chunks()

    abstained, reason, final_ans = evaluate_abstention(
        question="How do I fix error EXP-3204?",
        support_answer=ans,
        retrieved_chunks=chunks,
        citation_validation_passed=False,  # Invalid citation
        confidence_score=0.40,
        confidence_level="low",
    )

    assert abstained is True
    assert reason == AbstentionReason.NO_VALID_CITATIONS
    assert final_ans.answer == ABSTENTION_ANSWER_TEXT
    assert final_ans.escalation_required is True


def test_unresolved_conflict_question():
    ans_conflict = create_sample_answer(conflicts_detected=True)
    chunks = create_sample_chunks()

    abstained, reason, final_ans = evaluate_abstention(
        question="How to configure export?",
        support_answer=ans_conflict,
        retrieved_chunks=chunks,
        citation_validation_passed=True,
        confidence_score=0.60,
        confidence_level="medium",
    )

    assert abstained is True
    assert reason == AbstentionReason.UNRESOLVED_SOURCE_CONFLICT
    assert final_ans.answer == ABSTENTION_ANSWER_TEXT
    assert final_ans.escalation_required is True


def test_policy_question():
    ans = create_sample_answer()
    chunks = create_sample_chunks()  # Chunks contain tech guide, no policy/pricing

    abstained, reason, final_ans = evaluate_abstention(
        question="What is the refund and discount rate compensation policy for billing disputes?",
        support_answer=ans,
        retrieved_chunks=chunks,
        citation_validation_passed=True,
        confidence_score=0.85,
        confidence_level="high",
    )

    assert abstained is True
    assert reason == AbstentionReason.POLICY_LEGAL_BILLING_DECISION
    assert final_ans.answer == ABSTENTION_ANSWER_TEXT
    assert final_ans.escalation_required is True
