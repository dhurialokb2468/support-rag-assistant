from app.conversation import ConversationState


def test_version_carried_into_followup():
    state = ConversationState()

    # Turn 1: User mentions version 3.2
    state.update_from_question("How do I export CSV reports in InsightFlow version 3.2?")
    state.update_from_answer("Navigate to Export -> CSV.", source_ids=["S1"])

    assert state.product == "InsightFlow"
    assert state.product_version == "3.2"

    # Turn 2: Follow-up question missing explicit version
    state.update_from_question("What if the CSV button is disabled?")

    # Version 3.2 remains carried over in state
    assert state.product_version == "3.2"
    ctx = state.build_rewrite_context()
    assert "Version: 3.2" in ctx
    assert "User: What if the CSV button is disabled?" in ctx


def test_error_code_carried_into_followup():
    state = ConversationState()

    # Turn 1: User reports error code EXP-3204
    state.update_from_question("I am seeing error EXP-3204 on export.")
    state.update_from_answer("Check network credentials.", source_ids=["S1"])

    assert "EXP-3204" in state.error_codes

    # Turn 2: Follow-up question
    state.update_from_question("How do I fix this permission issue?")

    assert "EXP-3204" in state.error_codes
    ctx = state.build_rewrite_context()
    assert "Error Codes: EXP-3204" in ctx


def test_user_correction_replaces_older_state():
    state = ConversationState()

    # Initial turn: User mentions version 2.0
    state.update_from_question("How to configure SSO in InsightFlow 2.0?")
    assert state.product_version == "2.0"

    # Correction turn: User explicitly states they are actually on 3.5
    state.update_from_question("Actually, we upgraded and we are on version 3.5 now.")

    # State must update product_version to 3.5, replacing 2.0
    assert state.product_version == "3.5"
    ctx = state.build_rewrite_context()
    assert "Version: 3.5" in ctx


def test_clear_conversation():
    state = ConversationState()

    state.update_from_question("InsightFlow 3.2 error EXP-3204")
    state.update_from_answer("Fix credentials.", source_ids=["S1", "S2"])

    assert state.product == "InsightFlow"
    assert state.product_version == "3.2"
    assert "EXP-3204" in state.error_codes
    assert len(state.recent_questions) == 1
    assert len(state.recent_answers) == 1
    assert len(state.previous_source_ids) == 2

    # Clear conversation state
    state.clear()

    assert state.product is None
    assert state.product_version is None
    assert state.issue_category is None
    assert len(state.error_codes) == 0
    assert len(state.recent_questions) == 0
    assert len(state.recent_answers) == 0
    assert len(state.previous_source_ids) == 0
    assert state.is_empty() is True
