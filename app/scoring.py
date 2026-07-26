import math
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.query_processor import QueryProcessor


def normalize_reranker_scores(candidates: list[dict[str, Any]]) -> None:
    """Normalizes raw reranker scores across candidate list to [0, 1] range."""
    if not candidates:
        return

    raw_scores: list[float] = []
    for c in candidates:
        val = c.get("raw_reranker_score")
        if val is not None:
            raw_scores.append(float(val))

    if not raw_scores:
        for c in candidates:
            c["normalized_reranker_score"] = c.get("reranker_score", 0.5)
        return

    min_s = min(raw_scores)
    max_s = max(raw_scores)

    for c in candidates:
        raw = c.get("raw_reranker_score")
        if raw is None:
            c["normalized_reranker_score"] = 0.5
        elif math.isclose(max_s, min_s):
            c["normalized_reranker_score"] = 1.0 / (1.0 + math.exp(-float(raw)))
        else:
            c["normalized_reranker_score"] = (float(raw) - min_s) / (max_s - min_s)


def calculate_freshness_score(
    updated_at: str | None,
    reference_date: datetime | None = None,
) -> float:
    """Calculates freshness score in range [0.0, 1.0] from updated_at timestamp."""
    if not updated_at or not isinstance(updated_at, str) or not updated_at.strip():
        return 0.5

    try:
        clean_date_str = updated_at.strip()
        if "T" in clean_date_str:
            dt = datetime.fromisoformat(clean_date_str.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(clean_date_str[:10], "%Y-%m-%d")

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        ref = reference_date or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)

        age_days = (ref - dt).days
        if age_days < 0:
            return 1.0

        freshness = 1.0 / (1.0 + (age_days / 180.0))
        return max(0.0, min(1.0, freshness))
    except Exception:
        return 0.5


def calculate_authority_score(metadata: dict[str, Any]) -> float:
    """Extracts and clamps authority score from document metadata."""
    if not isinstance(metadata, dict):
        return 0.5

    auth = metadata.get("authority_score", 0.5)
    try:
        val = float(auth)
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return 0.5


def calculate_version_score(
    candidate_version: str | None,
    query_version: str | None,
    version_boost: float = settings.version_boost,
) -> float:
    """Calculates version boost score for exact version match vs mismatch."""
    if not query_version:
        return 0.0

    cand_ver = str(candidate_version).strip().lower() if candidate_version is not None else ""
    q_ver = str(query_version).strip().lower()

    if cand_ver == q_ver:
        return version_boost
    elif cand_ver in ("all", "", "none"):
        return 0.0
    else:
        return -0.05


def score_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    query_processor: QueryProcessor | None = None,
    relevance_weight: float = settings.relevance_weight,
    authority_weight: float = settings.authority_weight,
    freshness_weight: float = settings.freshness_weight,
    version_boost: float = settings.version_boost,
    reference_date: datetime | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    # 1. Validate that weights sum to 1 (normalize if they do not)
    total_w = relevance_weight + authority_weight + freshness_weight
    if not math.isclose(total_w, 1.0, abs_tol=1e-5) and total_w > 0:
        relevance_weight /= total_w
        authority_weight /= total_w
        freshness_weight /= total_w

    # 2. Normalize reranker scores
    normalize_reranker_scores(candidates)

    # 3. Extract query version
    qp = query_processor or QueryProcessor()
    q_meta = qp.process(query)
    query_version = q_meta.version

    scored_candidates: list[dict[str, Any]] = []
    for cand in candidates:
        cand_copy = dict(cand)
        meta = cand_copy.get("metadata", {})

        rel_score = cand_copy.get("normalized_reranker_score", 0.5)
        auth_score = calculate_authority_score(meta)

        updated_at = meta.get("updated_at") if isinstance(meta, dict) else None
        fresh_score = calculate_freshness_score(updated_at, reference_date=reference_date)

        cand_version = meta.get("version") if isinstance(meta, dict) else None
        ver_score = calculate_version_score(cand_version, query_version, version_boost=version_boost)

        final_score = (
            relevance_weight * rel_score
            + authority_weight * auth_score
            + freshness_weight * fresh_score
            + ver_score
        )

        cand_copy["normalized_reranker_score"] = rel_score
        if cand_copy.get("reranker_score") is None and cand_copy.get("raw_reranker_score") is not None:
            cand_copy["reranker_score"] = 1.0 / (1.0 + math.exp(-float(cand_copy["raw_reranker_score"])))
        elif cand_copy.get("reranker_score") is None:
            cand_copy["reranker_score"] = rel_score

        cand_copy["authority_score"] = auth_score
        cand_copy["freshness_score"] = fresh_score
        cand_copy["version_score"] = ver_score
        cand_copy["final_score"] = final_score

        scored_candidates.append(cand_copy)

    # 4. Sort final candidates by final score descending
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    for rank, item in enumerate(scored_candidates, start=1):
        item["reranker_rank"] = rank

    return scored_candidates
