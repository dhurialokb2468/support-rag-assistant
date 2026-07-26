import chromadb

from app.config import settings
from app.embeddings import EmbeddingService
from app.logger import get_logger
from app.models import Chunk

logger = get_logger("vector_store")


class VectorStore:
    def __init__(self) -> None:
        logger.info(f"Initializing Chroma PersistentClient at '{settings.chroma_path}'...")
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(name=settings.chroma_collection)
        self.embedding_service = EmbeddingService()
        logger.info(f"Chroma collection '{settings.chroma_collection}' active (Current count: {self.collection.count()}).")

    def reset(self) -> None:
        logger.warning(f"Resetting Chroma collection '{settings.chroma_collection}'...")
        try:
            item_ids = self.collection.get().get("ids", [])
            if item_ids:
                self.collection.delete(ids=item_ids)
        except Exception:
            try:
                self.client.delete_collection(settings.chroma_collection)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(name=settings.chroma_collection)
        logger.info(f"Chroma collection reset complete (Count: {self.collection.count()}).")

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def metadata_for_chroma(chunk: Chunk) -> dict:
        metadata = chunk.metadata

        return {
            "document_id": chunk.document_id,
            "source": metadata.source,
            "title": metadata.title,
            "document_type": metadata.document_type,
            "product": metadata.product,
            "version": metadata.version or "",
            "category": metadata.category or "",
            "updated_at": metadata.updated_at or "",
            "authority_score": metadata.authority_score,
            "reviewed": metadata.reviewed,
            "chunk_index": chunk.chunk_index,
            "parent_id": getattr(chunk, "parent_id", None) or metadata.extra.get("parent_id", ""),
            "parent_text": getattr(chunk, "parent_text", None) or metadata.extra.get("parent_text", ""),
        }

    def add_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 32,
    ) -> None:
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings, _ = self.embedding_service.embed_documents(texts)

            self.collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[self.metadata_for_chroma(chunk) for chunk in batch],
            )
        logger.info(f"Successfully indexed {len(chunks)} chunks in Chroma (Total count: {self.collection.count()}).")

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        logger.debug(f"Semantic search query (len={len(query)}), top_k={top_k}, where={where}")
        query_embedding, _ = self.embedding_service.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results.get("ids") and len(results["ids"]) > 0:
            for index, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][index]
                similarity = 1.0 - float(distance)

                output.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][index],
                    "metadata": results["metadatas"][0][index],
                    "semantic_score": similarity,
                })

        logger.debug(f"Semantic search returned {len(output)} candidate chunks.")
        return output