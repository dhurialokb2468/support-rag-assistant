from app.feedback import FEEDBACK_REASONS, FeedbackDB


def test_init_db(tmp_path):
    db_file = tmp_path / "test_feedback.db"
    db = FeedbackDB(db_path=db_file)

    with db.get_connection() as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]
        assert "interactions" in tables
        assert "retrieved_sources" in tables
        assert "feedback" in tables


def test_save_interaction(tmp_path):
    db_file = tmp_path / "test_feedback.db"
    db = FeedbackDB(db_path=db_file)

    answer_data = {
        "answer": "Clear report configuration cache.",
        "likely_cause": "Deprecated schema.",
        "resolution_steps": ["Clear cache", "Recreate settings"],
        "confidence": "high",
        "confidence_score": 0.92,
        "escalation_required": False,
    }

    retrieved_sources = [
        {"source_id": "version_3_2", "final_score": 0.89},
        {"source_id": "export_failures", "final_score": 0.82},
    ]

    i_id = db.save_interaction(
        question="Why does EXP-3204 happen?",
        rewritten_query="EXP-3204 CSV export failure version 3.2",
        answer_data=answer_data,
        confidence_score=0.92,
        total_latency=0.45,
        abstained=False,
        escalated=False,
        retrieved_sources=retrieved_sources,
    )

    saved = db.get_interaction(i_id)
    assert saved is not None
    assert saved["interaction_id"] == i_id
    assert saved["question"] == "Why does EXP-3204 happen?"
    assert saved["confidence_score"] == 0.92
    assert len(saved["retrieved_sources"]) == 2
    assert saved["retrieved_sources"][0]["source_id"] == "version_3_2"


def test_save_helpful_feedback(tmp_path):
    db_file = tmp_path / "test_feedback.db"
    db = FeedbackDB(db_path=db_file)

    i_id = db.save_interaction(
        question="How do I reset password?",
        rewritten_query=None,
        answer_data={"answer": "Check email delivery."},
        confidence_score=0.75,
        total_latency=0.30,
        abstained=False,
        escalated=False,
    )

    f_id = db.save_feedback(
        interaction_id=i_id,
        helpful=True,
    )

    saved = db.get_interaction(i_id)
    assert len(saved["feedback"]) == 1
    fb = saved["feedback"][0]
    assert fb["feedback_id"] == f_id
    assert fb["helpful"] == 1
    assert fb["issue_type"] is None


def test_save_not_helpful_feedback_with_reasons(tmp_path):
    db_file = tmp_path / "test_feedback.db"
    db = FeedbackDB(db_path=db_file)

    i_id = db.save_interaction(
        question="Salesforce sync delay",
        rewritten_query=None,
        answer_data={"answer": "Reauthorize connection."},
        confidence_score=0.60,
        total_latency=0.50,
        abstained=False,
        escalated=False,
    )

    # Test all 7 required feedback reasons
    for reason in FEEDBACK_REASONS:
        f_id = db.save_feedback(
            interaction_id=i_id,
            helpful=False,
            issue_type=reason,
            comment=f"Details for {reason}",
        )
        assert f_id is not None

    saved = db.get_interaction(i_id)
    assert len(saved["feedback"]) == 7

    all_fb = db.get_all_feedback()
    assert len(all_fb) == 7
    issues = [f["issue_type"] for f in all_fb]
    for reason in FEEDBACK_REASONS:
        assert reason in issues
