from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )
    ollama_model: str = os.getenv(
        "OLLAMA_MODEL",
        "qwen2.5:7b",
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    chroma_path: str = os.getenv(
        "CHROMA_PATH",
        "storage/chroma",
    )
    chroma_collection: str = os.getenv(
        "CHROMA_COLLECTION",
        "support_knowledge",
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k_semantic: int = int(os.getenv("TOP_K_SEMANTIC", "12"))
    top_k_keyword: int = int(os.getenv("TOP_K_KEYWORD", "12"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "5"))
    min_confidence_score: float = float(
        os.getenv("MIN_CONFIDENCE_SCORE", "0.45")
    )
    bm25_catalog_path: str = os.getenv(
        "BM25_CATALOG_PATH",
        "storage/chunks_catalog.json",
    )
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    enable_multi_query: bool = os.getenv(
        "ENABLE_MULTI_QUERY", "true"
    ).lower() in ("true", "1", "yes")
    max_search_queries: int = int(os.getenv("MAX_SEARCH_QUERIES", "3"))
    max_context_characters: int = int(
        os.getenv("MAX_CONTEXT_CHARACTERS", "8000")
    )
    max_chunks_per_source: int = int(
        os.getenv("MAX_CHUNKS_PER_SOURCE", "2")
    )
    near_duplicate_threshold: float = float(
        os.getenv("NEAR_DUPLICATE_THRESHOLD", "0.85")
    )
    relevance_weight: float = float(os.getenv("RELEVANCE_WEIGHT", "0.65"))
    authority_weight: float = float(os.getenv("AUTHORITY_WEIGHT", "0.20"))
    freshness_weight: float = float(os.getenv("FRESHNESS_WEIGHT", "0.15"))
    version_boost: float = float(os.getenv("VERSION_BOOST", "0.10"))
    parent_chunk_size: int = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
    child_chunk_size: int = int(os.getenv("CHILD_CHUNK_SIZE", "500"))
    child_chunk_overlap: int = int(os.getenv("CHILD_CHUNK_OVERLAP", "80"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    storage_dir: str = os.getenv("STORAGE_DIR", "storage")


settings = Settings()
