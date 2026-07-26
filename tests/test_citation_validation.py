from app.models import Citation, SupportAnswer
from app.validators import (
    normalize_citation_id,
    validate_and_resolve_citations,
    verify_citation_support,
)


def create_sample_selected_chunks():
    return [
        {
            "source_id": "S1",
            "document_id": "doc1",
            "text": "InsightFlow version 3.2 CSV export error EXP-3204 occurs when network permissions are missing.",
            "metadata": {
                "title": "InsightFlow 3.2 Guide",
                "source": "guide_v3.2.md",
                "version": "3.2",
            },
        },
        {
            "source_id": "S2",
            "document_id": "doc2",
            "text": "SSO authentication tokens expire after 24 hours of inactivity.",
            "metadata": {
                "title": "Authentication Manual",
                "source": "auth_manual.md",
                "version": "all",
            },
        },
    ]


def test_valid_citations():
    ans = SupportAnswer(
        answer="Error EXP-3204 is caused by missing network permissions.",
        likely_cause="Missing network permissions.",
        resolution_steps=["Verify network configuration."],
        citations=["S1", "[S2]"],
        confidence="high",
        confidence_score=0.9,
        escalation_required=False,
        escalation_reason=None,
        conflicts_detected=False,
    )

    chunks = create_sample_selected_chunks()

    updated_ans, citations, val_passed, details = validate_and_resolve_citations(ans, chunks)

    assert val_passed is True
    assert len(citations) == 2
    assert isinstance(citations[0], Citation)
    assert citations[0].source_id == "S1"
    assert citations[0].title == "InsightFlow 3.2 Guide"
    assert citations[1].source_id == "S2"
    assert updated_ans.confidence == "high"


def test_unknown_citations():
    ans = SupportAnswer(
        answer="Error EXP-3204 is caused by network issues.",
        likely_cause="Network issue.",
        resolution_steps=["Fix network."],
        citations=["S1", "S99"],  # S99 is unknown
        confidence="high",
        confidence_score=0.9,
        escalation_required=False,
        escalation_reason=None,
        conflicts_detected=False,
    )

    chunks = create_sample_selected_chunks()

    updated_ans, citations, val_passed, details = validate_and_resolve_citations(ans, chunks)

    assert val_passed is False
    assert "S99" in details["unknown_citations"]
    assert len(citations) == 1
    assert citations[0].source_id == "S1"
    # Confidence should be lowered from high to medium
    assert updated_ans.confidence == "medium"
    assert updated_ans.confidence_score == 0.7


def test_missing_citations():
    # Factual non-abstained answer with no citations provided
    ans = SupportAnswer(
        answer="CSV export failure EXP-3204 requires updating token permissions.",
        likely_cause="Token permission mismatch.",
        resolution_steps=["Update tokens."],
        citations=[],  # Missing
        confidence="high",
        confidence_score=0.9,
        escalation_required=False,
        escalation_reason=None,
        conflicts_detected=False,
    )

    chunks = create_sample_selected_chunks()

    updated_ans, citations, val_passed, details = validate_and_resolve_citations(ans, chunks)

    assert val_passed is False
    assert len(citations) == 0
    assert updated_ans.escalation_required is True
    assert "without any valid citations" in (updated_ans.escalation_reason or "")


def test_duplicated_citations():
    ans = SupportAnswer(
        answer="SSO session tokens expire after 24 hours.",
        likely_cause="Token expiration.",
        resolution_steps=["Relogin to renew token."],
        citations=["S2", "[S2]", "s2"],  # Duplicated variations
        confidence="medium",
        confidence_score=0.8,
        escalation_required=False,
        escalation_reason=None,
        conflicts_detected=False,
    )

    chunks = create_sample_selected_chunks()

    updated_ans, citations, val_passed, details = validate_and_resolve_citations(ans, chunks)

    assert val_passed is True
    assert len(citations) == 1
    assert citations[0].source_id == "S2"
    assert updated_ans.citations == ["S2"]


def test_unsupported_core_claim():
    claim = "InsightFlow automatically encrypts database backups using RSA-4096 keys."
    passage = "InsightFlow version 3.2 CSV export error EXP-3204 occurs when network permissions are missing."

    result = verify_citation_support(claim, passage)

    assert result == "unsupported"

    # Test supported claim
    supported_passage = "InsightFlow automatically encrypts database backups using RSA-4096 keys for enterprise security."
    supported_result = verify_citation_support(claim, supported_passage)

    assert supported_result == "supported"
