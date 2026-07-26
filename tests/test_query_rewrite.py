from unittest.mock import MagicMock

from app.conversation import ConversationState
from app.query_processor import needs_query_rewrite, rewrite_query


def test_needs_query_rewrite_triggers():
    # Pronoun detection
    assert needs_query_rewrite("Why does it fail?") is True
    assert needs_query_rewrite("How to fix this issue?") is True

    # Short question
    assert needs_query_rewrite("CSV export") is True

    # Context dependency
    conv = ConversationState()
    conv.add_turn("How to use InsightFlow version 3.2?", "Here is the guide...")

    # Question lacks version and product, but conversation context has it
    assert needs_query_rewrite("Why does export fail?", conv) is True


def test_needs_query_rewrite_no_trigger():
    # Complete, clear question with product and details
    q = "Why does CSV export fail with EXP-3204 in InsightFlow 3.2?"
    assert needs_query_rewrite(q) is False


def test_rewrite_query_skipped_when_not_needed():
    mock_generator = MagicMock()
    q = "Why does CSV export fail with EXP-3204 in InsightFlow 3.2?"

    res, trace = rewrite_query(q, generator=mock_generator)

    assert res == q
    assert trace.rewrite_used is False
    assert trace.rewrite_latency == 0.0
    mock_generator.generate.assert_not_called()


def test_rewrite_query_with_mocked_generator():
    mock_generator = MagicMock()
    mock_generator.generate.return_value = ("InsightFlow 3.2 CSV export error EXP-3204", 0.12)

    conv = ConversationState()
    conv.add_turn("InsightFlow 3.2 report export guide", "See documentation...")

    q = "Why does it fail?"

    res, trace = rewrite_query(q, conversation_state=conv, generator=mock_generator)

    assert res == "InsightFlow 3.2 CSV export error EXP-3204"
    assert trace.rewrite_used is True
    assert trace.original_question == q
    assert trace.rewritten_query == "InsightFlow 3.2 CSV export error EXP-3204"
    assert trace.rewrite_latency == 0.12
    mock_generator.generate.assert_called_once()


def test_rewrite_query_fallback_on_exception():
    mock_generator = MagicMock()
    mock_generator.generate.side_effect = RuntimeError("Generator unavailable")

    q = "Why does it fail?"

    res, trace = rewrite_query(q, generator=mock_generator)

    assert res == q
    assert trace.rewrite_used is False
    assert trace.original_question == q
    assert trace.rewritten_query == q
