import time

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.logger import get_logger

logger = get_logger("embeddings")


class EmbeddingService:
    def __init__(self) -> None:
        logger.info(f"Loading SentenceTransformer model: '{settings.embedding_model}'...")
        self.model = SentenceTransformer(settings.embedding_model)
        logger.info("SentenceTransformer model loaded successfully.")

    def embed_documents(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], float]:
        if not texts:
            return [], 0.0

        start = time.perf_counter()
        logger.debug(f"Embedding {len(texts)} document texts...")

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        elapsed = time.perf_counter() - start
        logger.debug(f"Embedded {len(texts)} texts in {elapsed:.4f}s.")

        return vectors.tolist(), elapsed

    def embed_query(
        self,
        query: str,
    ) -> tuple[list[float], float]:
        start = time.perf_counter()
        logger.debug(f"Embedding query (len={len(query)}): '{query[:30]}...'")

        vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        elapsed = time.perf_counter() - start
        logger.debug(f"Query embedded in {elapsed:.4f}s.")

        return vector.tolist(), elapsed