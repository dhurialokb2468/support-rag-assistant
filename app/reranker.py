import math
import time
from typing import Any

from sentence_transformers import CrossEncoder

from app.config import settings
from app.logger import get_logger
from app.scoring import score_candidates

logger = get_logger("reranker")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class RerankerService:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self.model: CrossEncoder | None = None

    def get_model(self) -> CrossEncoder:
        if self.model is None:
            logger.info(f"Loading CrossEncoder reranker model: '{self.model_name}'...")
            self.model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder reranker model loaded successfully.")
        return self.model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = settings.top_k_rerank,
    ) -> tuple[list[dict[str, Any]], float]:
        if not candidates:
            logger.debug("Rerank requested with 0 candidates. Returning empty list.")
            return [], 0.0

        started = time.perf_counter()
        logger.info(f"Reranking {len(candidates)} candidates for query (len={len(query)})...")
        pairs = [(query, c["text"]) for c in candidates]
        model = self.get_model()
        scores = model.predict(pairs)
        elapsed = time.perf_counter() - started

        reranked_candidates: list[dict[str, Any]] = []
        for candidate, raw_score in zip(candidates, scores):
            cand_copy = dict(candidate)
            score_val = float(raw_score)
            cand_copy["raw_reranker_score"] = score_val
            cand_copy["reranker_score"] = sigmoid(score_val)
            reranked_candidates.append(cand_copy)

        scored_results = score_candidates(query, reranked_candidates)
        top_candidates = scored_results[:top_k]
        top_score = top_candidates[0].get("final_score", 0.0) if top_candidates else 0.0

        logger.info(f"Reranking completed in {elapsed:.4f}s. Top candidate final_score={top_score:.4f}.")
        return top_candidates, elapsed
