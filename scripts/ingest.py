import argparse

from collections import Counter
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bm25_store import BM25Store
from app.chunking import chunk_documents
from app.config import settings
from app.loaders import load_all_documents
from app.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightFlow Support RAG Ingestion CLI")

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing vector store before adding chunks",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Chunk size for fixed chunking strategy",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Chunk overlap size",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "section", "parent-child"],
        default="parent-child",
        help="Chunking strategy: fixed, section, or parent-child",
    )

    args = parser.parse_args()

    started = time.perf_counter()

    documents = load_all_documents()

    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strategy=args.strategy,
    )

    store = VectorStore()

    if args.reset:
        store.reset()

    store.add_chunks(chunks)

    bm25_store = BM25Store()
    bm25_store.index_chunks(chunks)
    bm25_store.save()

    doc_type_counter = Counter()
    category_counter = Counter()
    version_counter = Counter()

    for chunk in chunks:
        meta = chunk.metadata if hasattr(chunk, "metadata") else chunk.get("metadata")
        if meta:
            dt = getattr(meta, "document_type", None) or (meta.get("document_type") if isinstance(meta, dict) else "Unknown")
            cat = getattr(meta, "category", None) or (meta.get("category") if isinstance(meta, dict) else "N/A") or "N/A"
            ver = getattr(meta, "version", None) or (meta.get("version") if isinstance(meta, dict) else "N/A") or "N/A"

            doc_type_counter[dt] += 1
            category_counter[cat] += 1
            version_counter[ver] += 1

    elapsed = time.perf_counter() - started

    print("=" * 80)
    print("INSIGHTFLOW RAG INGESTION COMPLETE")
    print("=" * 80)
    print(f"Strategy:               {args.strategy}")
    print(f"Documents loaded:       {len(documents)}")
    print(f"Total Chunks created:   {len(chunks)}")
    print(f"Vector Chunks indexed:  {store.count()}")
    print(f"BM25 Chunks indexed:    {bm25_store.count()}")
    print(f"Processing Time:        {elapsed:.2f} seconds")

    print("\n" + "=" * 80)
    print("INGESTION METADATA BREAKDOWN")
    print("=" * 80)

    print("\nCounts by Document Type:")
    for dt, count in doc_type_counter.most_common():
        print(f"  - {dt}: {count} chunks")

    print("\nCounts by Category:")
    for cat, count in category_counter.most_common():
        print(f"  - {cat}: {count} chunks")

    print("\nCounts by Version:")
    for ver, count in version_counter.most_common():
        print(f"  - {ver}: {count} chunks")

    print("=" * 80)


if __name__ == "__main__":
    main()