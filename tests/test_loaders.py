import csv
from pathlib import Path
import pytest

from app.loaders import (
    create_document_id,
    load_markdown_file,
    load_support_tickets,
    load_all_documents,
)


def test_create_document_id():
    doc_id = create_document_id("test_source")
    assert isinstance(doc_id, str)
    assert len(doc_id) == 16


def test_load_markdown_file(tmp_path: Path):
    md_content = """---
title: Test Title
document_type: faq
product: InsightFlow
version: "1.0"
category: general
updated_at: "2026-01-01"
authority_score: 0.8
reviewed: true
---
This is the body content of the markdown file.
"""
    file_path = tmp_path / "test.md"
    file_path.write_text(md_content, encoding="utf-8")

    doc = load_markdown_file(file_path)
    assert doc.metadata.title == "Test Title"
    assert doc.metadata.document_type == "faq"
    assert doc.metadata.authority_score == 0.8
    assert doc.metadata.reviewed is True
    assert doc.text == "This is the body content of the markdown file."


def test_load_markdown_file_invalid_header(tmp_path: Path):
    file_path = tmp_path / "invalid.md"
    file_path.write_text("No frontmatter header", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing metadata header"):
        load_markdown_file(file_path)


def test_load_support_tickets(tmp_path: Path):
    csv_path = tmp_path / "tickets.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticket_id", "question", "resolution", "product_version",
            "category", "created_at", "authority_score", "reviewed", "priority", "status"
        ])
        writer.writerow([
            "TCK-101", "How to reset pass?", "Go to settings", "2.0",
            "Auth", "2026-02-01", "0.9", "true", "High", "Closed"
        ])

    tickets = load_support_tickets(csv_path)
    assert len(tickets) == 1
    assert tickets[0].metadata.title == "Support Ticket TCK-101"
    assert "Customer question:\nHow to reset pass?" in tickets[0].text
    assert tickets[0].metadata.extra["ticket_id"] == "TCK-101"
