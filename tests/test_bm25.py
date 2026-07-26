from pathlib import Path
from app.bm25_store import BM25Store, tokenize
from app.models import Chunk, DocumentMetadata


def test_tokenize_preserves_technical_tokens():
    text = "Error EXP-3204 in SalesforceSyncV2 with version 3.2 and user-management config report.export.schema."
    tokens = tokenize(text)

    assert "exp-3204" in tokens
    assert "EXP-3204" in tokens
    assert "salesforcesyncv2" in tokens
    assert "SalesforceSyncV2" in tokens
    assert "3.2" in tokens
    assert "user-management" in tokens
    assert "report.export.schema" in tokens


def test_exact_error_code_ranking():
    doc_meta = DocumentMetadata(
        title="Test Doc",
        source="test.md",
        document_type="known_issues",
    )

    chunk1 = Chunk(
        chunk_id="c1",
        document_id="d1",
        text="General documentation about exports and reports in InsightFlow.",
        metadata=doc_meta,
        chunk_index=0,
    )
    chunk2 = Chunk(
        chunk_id="c2",
        document_id="d2",
        text="Known issue EXP-3204 causing CSV export failure after version 3.2 upgrade.",
        metadata=doc_meta,
        chunk_index=0,
    )
    chunk3 = Chunk(
        chunk_id="c3",
        document_id="d3",
        text="Unrelated ticket regarding Salesforce integration AUTH-101.",
        metadata=doc_meta,
        chunk_index=0,
    )

    bm25_store = BM25Store()
    bm25_store.index_chunks([chunk1, chunk2, chunk3])

    results = bm25_store.search("EXP-3204", top_k=3)

    assert len(results) >= 1
    assert results[0]["chunk_id"] == "c2"
    assert "EXP-3204" in results[0]["text"]
    assert results[0]["keyword_score"] > 0
    assert results[0]["bm25_score"] > 0


def test_bm25_persistence(tmp_path: Path):
    doc_meta = DocumentMetadata(
        title="Persistent Doc",
        source="pers.md",
        document_type="faq",
    )
    chunk = Chunk(
        chunk_id="c_pers",
        document_id="d_pers",
        text="Testing BM25 catalog serialization with error code SYNC-2201.",
        metadata=doc_meta,
        chunk_index=0,
    )

    catalog_file = tmp_path / "chunks_catalog.json"
    bm25_store = BM25Store(catalog_path=catalog_file)
    bm25_store.index_chunks([chunk])
    bm25_store.save()

    assert catalog_file.exists()

    loaded_store = BM25Store(catalog_path=catalog_file)
    loaded = loaded_store.load()
    assert loaded is True
    assert loaded_store.count() == 1

    results = loaded_store.search("SYNC-2201")
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c_pers"
