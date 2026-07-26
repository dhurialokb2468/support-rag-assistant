from app.context_builder import ContextBuilder, build_context


def test_duplicate_removal_exact_id_and_normalized_text():
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "text": "InsightFlow 3.2 supports CSV export feature.",
            "metadata": {"title": "Doc 1", "source": "doc1.md", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c1",  # Exact chunk ID duplicate
            "document_id": "doc1",
            "text": "InsightFlow 3.2 supports CSV export feature.",
            "metadata": {"title": "Doc 1", "source": "doc1.md", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c2",
            "document_id": "doc2",
            "text": "insightflow 3.2 supports csv export feature.",  # Exact normalized text duplicate
            "metadata": {"title": "Doc 2", "source": "doc2.md", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c3",
            "document_id": "doc3",
            "text": "Authentication configuration instructions for SSO token setup.",
            "metadata": {"title": "Doc 3", "source": "doc3.md", "document_type": "product_documentation"},
        },
    ]

    res = build_context(chunks, max_chunks_per_source=5, max_context_characters=5000)

    assert len(res.selected_chunks) == 2
    assert len(res.excluded_duplicates) == 2

    selected_ids = [c["chunk_id"] for c in res.selected_chunks]
    assert selected_ids == ["c1", "c3"]


def test_near_duplicate_removal():
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "text": "To configure Salesforce integration, navigate to Settings > Integrations > Salesforce.",
            "metadata": {"title": "Doc 1", "source": "doc1.md", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c2",
            "document_id": "doc2",
            "text": "To configure Salesforce integration, please navigate to Settings > Integrations > Salesforce.",  # Near duplicate
            "metadata": {"title": "Doc 2", "source": "doc2.md", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c3",
            "document_id": "doc3",
            "text": "Release notes for version 3.2 detail new webhook rate limit features.",
            "metadata": {"title": "Doc 3", "source": "doc3.md", "document_type": "release_notes"},
        },
    ]

    builder = ContextBuilder(near_duplicate_threshold=0.85)
    res = builder.build_context(chunks)

    assert len(res.selected_chunks) == 2
    assert len(res.excluded_duplicates) == 1
    assert res.excluded_duplicates[0]["chunk_id"] == "c2"


def test_source_limits():
    chunks = [
        {
            "chunk_id": "c1_1",
            "document_id": "doc_shared",
            "text": "First chunk from shared document.",
            "metadata": {"title": "Shared Doc", "source": "doc_shared", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c1_2",
            "document_id": "doc_shared",
            "text": "Second chunk from shared document.",
            "metadata": {"title": "Shared Doc", "source": "doc_shared", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c1_3",  # 3rd chunk from same doc -> exceeds max_chunks_per_source=2
            "document_id": "doc_shared",
            "text": "Third chunk from shared document.",
            "metadata": {"title": "Shared Doc", "source": "doc_shared", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c2_1",
            "document_id": "doc_other",
            "text": "Chunk from another document.",
            "metadata": {"title": "Other Doc", "source": "doc_other", "document_type": "known_issues"},
        },
    ]

    res = build_context(chunks, max_chunks_per_source=2, max_context_characters=5000)

    assert len(res.selected_chunks) == 3
    assert len(res.excluded_source_limit) == 1
    assert res.excluded_source_limit[0]["chunk_id"] == "c1_3"


def test_context_budget():
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "text": "Short text for first chunk.",
            "metadata": {"title": "Doc 1", "source": "doc1.md", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c2",
            "document_id": "doc2",
            "text": "A very long text chunk that will exceed the context character budget when small budget is set.",
            "metadata": {"title": "Doc 2", "source": "doc2.md", "document_type": "release_notes"},
        },
    ]

    # Set tight budget (e.g. 200 chars)
    res = build_context(chunks, max_context_characters=200)

    assert len(res.selected_chunks) == 1
    assert res.selected_chunks[0]["chunk_id"] == "c1"
    assert len(res.excluded_budget_overflow) == 1
    assert res.excluded_budget_overflow[0]["chunk_id"] == "c2"


def test_stable_source_numbering():
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "text": "First source text content.",
            "metadata": {"title": "Guide Title", "source": "guide.md", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c2",
            "document_id": "doc2",
            "text": "Second source text content.",
            "metadata": {"title": "Release Notes Title", "source": "rn.md", "document_type": "release_notes"},
        },
        {
            "chunk_id": "c3",
            "document_id": "doc3",
            "text": "Third source text content.",
            "metadata": {"title": "Ticket Title", "source": "ticket.md", "document_type": "support_ticket"},
        },
    ]

    res = build_context(chunks)

    assert len(res.selected_chunks) == 3
    assert res.selected_chunks[0]["source_id"] == "S1"
    assert res.selected_chunks[1]["source_id"] == "S2"
    assert res.selected_chunks[2]["source_id"] == "S3"

    assert "[Source S1]" in res.formatted_context
    assert "[Source S2]" in res.formatted_context
    assert "[Source S3]" in res.formatted_context


def test_source_diversity_ordering():
    chunks = [
        {
            "chunk_id": "c1_doc",
            "document_id": "doc1",
            "text": "First doc chunk.",
            "metadata": {"title": "Doc 1", "source": "doc1.md", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c2_doc",
            "document_id": "doc1",
            "text": "Second doc chunk.",
            "metadata": {"title": "Doc 1", "source": "doc1.md", "document_type": "product_documentation"},
        },
        {
            "chunk_id": "c3_rn",
            "document_id": "doc2",
            "text": "Release notes chunk.",
            "metadata": {"title": "Doc 2", "source": "doc2.md", "document_type": "release_notes"},
        },
        {
            "chunk_id": "c4_bug",
            "document_id": "doc3",
            "text": "Known issue bug chunk.",
            "metadata": {"title": "Doc 3", "source": "doc3.md", "document_type": "known_issues"},
        },
    ]

    res = build_context(chunks)

    # Top candidate from each distinct preferred doc type selected first
    selected_ids = [c["chunk_id"] for c in res.selected_chunks]
    assert selected_ids[0] == "c1_doc"
    assert selected_ids[1] == "c3_rn"
    assert selected_ids[2] == "c4_bug"
    assert selected_ids[3] == "c2_doc"
