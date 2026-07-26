import pytest
from app.models import DocumentMetadata, SourceDocument
from app.chunking import (
    chunk_document_fixed,
    chunk_document_section_aware,
    chunk_documents,
    split_markdown_sections,
)


@pytest.fixture
def sample_document():
    metadata = DocumentMetadata(
        title="Sample Document",
        source="sample.md",
        document_type="product_docs",
    )
    text = (
        "# Introduction\n"
        "This is the introduction section of the test document.\n\n"
        "## Setup\n"
        "Here are the setup instructions for InsightFlow system."
    )
    return SourceDocument(
        document_id="doc123",
        text=text,
        metadata=metadata,
    )


def test_split_markdown_sections(sample_document):
    sections = split_markdown_sections(sample_document.text)
    assert len(sections) == 2
    assert sections[0][0] == "# Introduction"
    assert "introduction section" in sections[0][1]
    assert sections[1][0] == "## Setup"
    assert "setup instructions" in sections[1][1]


def test_chunk_document_fixed(sample_document):
    chunks = chunk_document_fixed(sample_document, chunk_size=50, overlap=10)
    assert len(chunks) > 0
    assert chunks[0].document_id == "doc123"
    assert chunks[0].chunk_index == 0


def test_chunk_document_invalid_overlap(sample_document):
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_document_fixed(sample_document, chunk_size=50, overlap=50)


def test_chunk_document_section_aware(sample_document):
    chunks = chunk_document_section_aware(sample_document, chunk_size=200, overlap=20)
    assert len(chunks) == 2
    assert chunks[0].metadata.extra.get("section_heading") == "# Introduction"
    assert chunks[1].metadata.extra.get("section_heading") == "## Setup"


def test_chunk_documents_strategy(sample_document):
    fixed_chunks = chunk_documents([sample_document], strategy="fixed")
    section_chunks = chunk_documents([sample_document], strategy="section")
    assert len(fixed_chunks) > 0
    assert len(section_chunks) > 0

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_documents([sample_document], strategy="invalid_strategy")
