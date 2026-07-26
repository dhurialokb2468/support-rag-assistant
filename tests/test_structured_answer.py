from unittest.mock import MagicMock

from app.generator import OllamaGenerator
from app.models import SupportAnswer
from app.validators import (
    clean_markdown_fences,
    create_fallback_support_answer,
    parse_json_string,
    validate_support_answer,
)


def test_valid_json():
    raw_json = """
{
  "answer": "Error EXP-3204 is caused by network configuration issues.",
  "likely_cause": "Network connector permission failure.",
  "resolution_steps": ["Check network settings.", "Re-verify connector tokens."],
  "citations": ["S1"],
  "confidence": "high",
  "confidence_score": 0.95,
  "escalation_required": false,
  "escalation_reason": null,
  "conflicts_detected": false
}
"""
    ans, err = validate_support_answer(raw_json)

    assert err is None
    assert isinstance(ans, SupportAnswer)
    assert ans.answer.startswith("Error EXP-3204")
    assert ans.confidence == "high"
    assert ans.confidence_score == 0.95
    assert ans.escalation_required is False
    assert ans.citations == ["S1"]


def test_fenced_json():
    fenced = """```json
{
  "answer": "SSO authentication tokens expire after 24 hours.",
  "likely_cause": "Expired SSO token.",
  "resolution_steps": ["Renew SSO session."],
  "citations": ["S2"],
  "confidence": "medium",
  "confidence_score": 0.75,
  "escalation_required": false,
  "escalation_reason": null,
  "conflicts_detected": false
}
```"""

    cleaned = clean_markdown_fences(fenced)
    parsed, parse_err = parse_json_string(fenced)
    ans, val_err = validate_support_answer(fenced)

    assert parse_err is None
    assert val_err is None
    assert isinstance(ans, SupportAnswer)
    assert ans.citations == ["S2"]
    assert ans.confidence == "medium"


def test_malformed_json_and_repair():
    generator = OllamaGenerator()

    # Mock generator to return malformed JSON first, then valid JSON on repair
    malformed_first = "{ 'answer': 'Malformed json missing quotes... }"
    valid_second = """
{
  "answer": "Repaired valid support answer.",
  "likely_cause": "Initial format error.",
  "resolution_steps": ["Check JSON parser."],
  "citations": ["S1"],
  "confidence": "medium",
  "confidence_score": 0.8,
  "escalation_required": false,
  "escalation_reason": null,
  "conflicts_detected": false
}
"""
    generator.generate = MagicMock(side_effect=[
        (malformed_first, 0.1),
        (valid_second, 0.1),
    ])

    ans, trace = generator.generate_support_answer("Query?", "Context S1")

    assert generator.generate.call_count == 2
    assert trace["repaired"] is True
    assert trace["fallback_used"] is False
    assert ans.answer == "Repaired valid support answer."


def test_repair_success():
    generator = OllamaGenerator()

    # Mock generator to return missing required Pydantic field first, then valid JSON on repair
    missing_confidence_first = """
{
  "answer": "Answer snippet",
  "likely_cause": "Cause",
  "resolution_steps": ["Step 1"],
  "citations": ["S1"]
}
"""
    repaired_second = """
{
  "answer": "Answer snippet",
  "likely_cause": "Cause",
  "resolution_steps": ["Step 1"],
  "citations": ["S1"],
  "confidence": "high",
  "confidence_score": 0.9,
  "escalation_required": false,
  "escalation_reason": null,
  "conflicts_detected": false
}
"""
    generator.generate = MagicMock(side_effect=[
        (missing_confidence_first, 0.1),
        (repaired_second, 0.1),
    ])

    ans, trace = generator.generate_support_answer("Query?", "Context S1")

    assert generator.generate.call_count == 2
    assert trace["repaired"] is True
    assert trace["fallback_used"] is False
    assert ans.confidence == "high"


def test_fallback_behavior():
    generator = OllamaGenerator()

    # Mock generator to return completely invalid output twice
    invalid_output = "Invalid non-json text response."

    generator.generate = MagicMock(side_effect=[
        (invalid_output, 0.1),
        (invalid_output, 0.1),
    ])

    ans, trace = generator.generate_support_answer("Query?", "Context S1")

    assert generator.generate.call_count == 2
    assert trace["fallback_used"] is True
    assert ans.escalation_required is True
    assert ans.confidence == "low"
    assert ans.confidence_score == 0.0
    assert "Validation failure" in (ans.escalation_reason or "")
    assert trace["initial_error"] is not None
