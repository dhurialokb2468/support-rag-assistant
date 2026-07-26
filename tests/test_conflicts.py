from app.conflicts import detect_and_resolve_conflicts
from app.models import ConflictResult


def test_no_conflict():
    chunks = [
        {
            "source_id": "S1",
            "document_id": "doc1",
            "text": "InsightFlow 3.2 supports CSV export feature.",
            "metadata": {"title": "Guide 3.2", "document_type": "product_documentation", "version": "3.2"},
        },
        {
            "source_id": "S2",
            "document_id": "doc2",
            "text": "CSV export allows downloading reports in UTF-8 format.",
            "metadata": {"title": "Release Notes", "document_type": "release_notes", "version": "3.2"},
        },
    ]

    res = detect_and_resolve_conflicts("CSV export question", chunks)

    assert isinstance(res, ConflictResult)
    assert res.conflict_detected is False
    assert res.unresolved is False


def test_old_ticket_versus_new_release_note():
    # Passages recommend opposing actions (enable vs disable)
    chunks = [
        {
            "source_id": "S1",
            "document_id": "ticket_101",
            "text": "For CSV export, disable SSL verification setting.",
            "metadata": {
                "title": "Old Ticket 101",
                "document_type": "support_ticket",
                "reviewed": False,
                "authority_score": 0.4,
                "updated_at": "2024-01-01",
                "version": "3.2",
            },
        },
        {
            "source_id": "S2",
            "document_id": "release_note_32",
            "text": "For CSV export, enable SSL verification setting for enterprise security.",
            "metadata": {
                "title": "Release Notes 3.2",
                "document_type": "release_notes",
                "reviewed": True,
                "authority_score": 0.9,
                "updated_at": "2026-05-01",
                "version": "3.2",
            },
        },
    ]

    res = detect_and_resolve_conflicts("How to configure SSL for CSV export?", chunks)

    assert res.conflict_detected is True
    assert "S1" in res.conflicting_source_ids
    assert "S2" in res.conflicting_source_ids
    assert res.preferred_source_id == "S2"
    assert res.unresolved is False
    assert "release_notes" in (res.preference_reason or "")


def test_two_equally_authoritative_conflicting_sources():
    # Opposing instructions with identical metadata and authority
    chunks = [
        {
            "source_id": "S1",
            "document_id": "doc_a",
            "text": "Set authentication parameter enable=true for SSO integration.",
            "metadata": {
                "title": "Doc A",
                "document_type": "product_documentation",
                "reviewed": True,
                "authority_score": 0.8,
                "updated_at": "2026-01-01",
                "version": "all",
            },
        },
        {
            "source_id": "S2",
            "document_id": "doc_b",
            "text": "Set authentication parameter enable=false for SSO integration.",
            "metadata": {
                "title": "Doc B",
                "document_type": "product_documentation",
                "reviewed": True,
                "authority_score": 0.8,
                "updated_at": "2026-01-01",
                "version": "all",
            },
        },
    ]

    res = detect_and_resolve_conflicts("SSO setting?", chunks)

    assert res.conflict_detected is True
    assert res.preferred_source_id is None
    assert res.unresolved is True


def test_conflict_resolved_by_version_match():
    chunks = [
        {
            "source_id": "S1",
            "document_id": "doc_v2",
            "text": "In version 2.0, CSV export setting is disabled by default.",
            "metadata": {
                "title": "Guide 2.0",
                "document_type": "product_documentation",
                "version": "2.0",
                "authority_score": 0.8,
            },
        },
        {
            "source_id": "S2",
            "document_id": "doc_v32",
            "text": "In version 3.2, CSV export setting is enabled by default.",
            "metadata": {
                "title": "Guide 3.2",
                "document_type": "product_documentation",
                "version": "3.2",
                "authority_score": 0.8,
            },
        },
    ]

    res = detect_and_resolve_conflicts(
        "CSV export default in 3.2",
        chunks,
        query_version="3.2",
    )

    assert res.conflict_detected is True
    assert res.preferred_source_id == "S2"
    assert res.unresolved is False
    assert "3.2" in (res.preference_reason or "")


def test_unresolved_conflict():
    chunks = [
        {
            "source_id": "S1",
            "document_id": "doc1",
            "text": "SSO timeout value is required to be 3600 seconds.",
            "metadata": {"title": "Doc 1", "document_type": "product_documentation", "authority_score": 0.5},
        },
        {
            "source_id": "S2",
            "document_id": "doc2",
            "text": "SSO timeout value is optional and set automatically.",
            "metadata": {"title": "Doc 2", "document_type": "product_documentation", "authority_score": 0.5},
        },
    ]

    res = detect_and_resolve_conflicts("SSO timeout setting?", chunks)

    assert res.conflict_detected is True
    assert res.unresolved is True
    assert res.preferred_source_id is None
