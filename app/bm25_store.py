import json
from pathlib import Path
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import settings
from app.models import Chunk


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9_.-]+", text)
    tokens: list[str] = []

    for raw in raw_tokens:
        clean = raw.strip(".-_")
        if not clean:
            continue

        tokens.append(clean.lower())
        if clean != clean.lower():
            tokens.append(clean)

        parts = re.split(r"[-._]", clean)
        if len(parts) > 1:
            for part in parts:
                if part:
                    tokens.append(part.lower())

    return tokens


class BM25Store:
    def __init__(self, catalog_path: str | Path | None = None) -> None:
        self.catalog_path = Path(catalog_path or settings.bm25_catalog_path)
        self.chunks: list[Chunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def index_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def save(self, path: str | Path | None = None) -> None:
        target_path = Path(path) if path else self.catalog_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        data = [chunk.model_dump() for chunk in self.chunks]
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path | None = None) -> bool:
        target_path = Path(path) if path else self.catalog_path
        if not target_path.exists():
            return False

        try:
            with target_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            chunks = [Chunk.model_validate(item) for item in data]
            self.index_chunks(chunks)
            return True
        except Exception:
            return False

    def count(self) -> int:
        return len(self.chunks)

    def _matches_where(self, chunk: Chunk, where: dict[str, Any] | None) -> bool:
        if not where:
            return True

        metadata = chunk.metadata
        chunk_dict = {
            "document_id": chunk.document_id,
            "title": metadata.title,
            "source": metadata.source,
            "document_type": metadata.document_type,
            "product": metadata.product,
            "version": metadata.version or "",
            "category": metadata.category or "",
        }

        if "$and" in where:
            return all(self._matches_where(chunk, clause) for clause in where["$and"])

        for key, value in where.items():
            if key == "$and":
                continue
            val_in_chunk = chunk_dict.get(key, "")
            if isinstance(value, dict) and "$in" in value:
                allowed = value["$in"]
                if val_in_chunk not in allowed:
                    return False
            elif val_in_chunk != value:
                return False

        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)
        raw_scores = self.bm25.get_scores(query_tokens)

        # Identify candidate indices with overlapping tokens
        candidate_indices = []
        for index, doc_tokens in enumerate(self.tokenized_corpus):
            if query_token_set.intersection(doc_tokens):
                chunk = self.chunks[index]
                if self._matches_where(chunk, where):
                    candidate_indices.append(index)

        if not candidate_indices:
            return []

        candidate_scores = [raw_scores[i] for i in candidate_indices]
        min_s = min(candidate_scores)
        max_s = max(candidate_scores)
        score_range = max_s - min_s

        candidates = []
        for index in candidate_indices:
            chunk = self.chunks[index]
            raw_score = float(raw_scores[index])

            if score_range > 0:
                norm_score = (raw_score - min_s) / score_range
            else:
                norm_score = 1.0

            metadata_dict = {
                "document_id": chunk.document_id,
                "source": chunk.metadata.source,
                "title": chunk.metadata.title,
                "document_type": chunk.metadata.document_type,
                "product": chunk.metadata.product,
                "version": chunk.metadata.version or "",
                "category": chunk.metadata.category or "",
                "updated_at": chunk.metadata.updated_at or "",
                "authority_score": chunk.metadata.authority_score,
                "reviewed": chunk.metadata.reviewed,
                "chunk_index": chunk.chunk_index,
                "parent_id": chunk.parent_id or "",
            }

            candidates.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": metadata_dict,
                    "keyword_score": float(norm_score),
                    "bm25_score": raw_score,
                }
            )

        candidates.sort(key=lambda x: x["bm25_score"], reverse=True)
        return candidates[:top_k]

