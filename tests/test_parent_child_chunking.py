from app.chunking import chunk_document_parent_child, chunk_documents
from app.context_builder import build_context, resolve_children_to_parents
from app.models import DocumentMetadata, SourceDocument


def create_sample_doc():
    markdown_text = """# Section 1: Overview
InsightFlow version 3.2 is an enterprise analytics platform.
It provides real-time data processing, custom report exports, and SSO authentication integration.

# Section 2: CSV Export Troubleshooting
When encountering error code EXP-3204 during CSV export, check network permission settings.
Ensure that Salesforce connector credentials have been properly configured and token expiration is set appropriately.
"""
    return SourceDocument(
        document_id="doc_sample_1",
        text=markdown_text,
        metadata=DocumentMetadata(
            title="InsightFlow 3.2 Guide",
            source="guide_v3.2.md",
            document_type="product_documentation",
            product="InsightFlow",
            version="3.2",
            category="reporting",
        ),
    )


def test_parent_creation():
    doc = create_sample_doc()
    parents, children = chunk_document_parent_child(
        doc,
        parent_size=2000,
        child_size=150,
        child_overlap=30,
    )

    assert len(parents) == 2
    assert parents[0].document_id == "doc_sample_1"
    assert parents[0].chunk_id.startswith("parent_")
    assert "Overview" in parents[0].text
    assert "EXP-3204" in parents[1].text


def test_child_parent_relationship():
    doc = create_sample_doc()
    parents, children = chunk_document_parent_child(
        doc,
        parent_size=2000,
        child_size=150,
        child_overlap=30,
    )

    assert len(children) > 0

    for child in children:
        assert child.parent_id is not None
        assert child.parent_id.startswith("parent_")
        assert child.parent_text is not None
        assert len(child.text) <= 180  # Around child_size

    # Verify every parent lists its child IDs
    parent_map = {p.chunk_id: p for p in parents}
    for child in children:
        p_obj = parent_map[child.parent_id]
        assert child.chunk_id in p_obj.child_ids


def test_parent_deduplication():
    # Simulate two retrieved child chunks originating from the SAME parent chunk
    parent_id = "parent_doc1_sec1"
    parent_text = "Full section text containing all details about CSV export troubleshooting and EXP-3204 resolution."

    child1 = {
        "chunk_id": "child_1",
        "parent_id": parent_id,
        "parent_text": parent_text,
        "document_id": "doc1",
        "text": "child 1 snippet about EXP-3204",
        "metadata": {"title": "Doc 1", "source": "doc1.md", "parent_id": parent_id, "parent_text": parent_text},
        "semantic_score": 0.85,
    }
    child2 = {
        "chunk_id": "child_2",
        "parent_id": parent_id,
        "parent_text": parent_text,
        "document_id": "doc1",
        "text": "child 2 snippet about CSV export",
        "metadata": {"title": "Doc 1", "source": "doc1.md", "parent_id": parent_id, "parent_text": parent_text},
        "semantic_score": 0.92,  # Higher score
    }

    resolved = resolve_children_to_parents([child1, child2])

    # Should resolve to 1 parent chunk
    assert len(resolved) == 1
    p = resolved[0]

    assert p["chunk_id"] == parent_id
    assert p["text"] == parent_text
    # Preserves best child score (0.92)
    assert p["semantic_score"] == 0.92
    # Records triggered children
    assert "child_1" in p["triggered_children"]
    assert "child_2" in p["triggered_children"]


def test_retrieval_child_context_parent():
    doc = create_sample_doc()
    chunks = chunk_documents([doc], strategy="parent-child", parent_size=2000, child_size=150)

    # Filtered retrieval mock selecting 2 children from same parent section
    retrieved_children = [c for c in chunks if "EXP-3204" in c.text or "Salesforce" in c.text]
    assert len(retrieved_children) >= 1

    res = build_context(retrieved_children, max_context_characters=8000)

    assert len(res.selected_chunks) == 1  # 1 parent selected
    assert "EXP-3204" in res.formatted_context
    assert "CSV Export Troubleshooting" in res.formatted_context
