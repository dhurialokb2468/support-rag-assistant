from typing import Any

from app.models import SupportAnswer


def compute_deterministic_confidence(
    support_answer: SupportAnswer,
    retrieved_chunks: list[dict[str, Any]],
    citation_validation_passed: bool = True,
    relaxed_retrieval_used: bool = False,
    query_version: str | None = None,
    top_relevance_weight: float = 0.30,
    score_separation_weight: float = 0.15,
    sources_count_weight: float = 0.15,
    version_match_weight: float = 0.10,
    authority_weight: float = 0.15,
    citation_validation_weight: float = 0.15,
    relaxed_retrieval_penalty: float = 0.15,
    conflict_penalty: float = 0.20,
) -> tuple[float, str, dict[str, Any]]:
    """
    Computes deterministic confidence score in [0.0, 1.0] and maps to 'high'|'medium'|'low'.
    Returns (final_score, confidence_level, breakdown_dict).
    """
    # 1. Top relevance score & score separation
    scores: list[float] = []
    for c in retrieved_chunks:
        val = c.get("final_score")
        if val is None:
            val = c.get("normalized_reranker_score")
        if val is None:
            val = c.get("reranker_score")
        if val is None:
            val = c.get("semantic_score", 0.5)
        scores.append(float(val))

    scores.sort(reverse=True)
    top_relevance = max(0.0, min(1.0, scores[0])) if scores else 0.0

    if len(scores) >= 2:
        sep = max(0.0, scores[0] - scores[1])
        score_separation = min(1.0, sep / 0.3)
    else:
        score_separation = 0.0

    # 2. Number of independent supporting sources & authority
    distinct_sources: set[str] = set()
    authorities: list[float] = []
    has_exact_version = False

    for c in retrieved_chunks:
        meta = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else {}
        src_id = meta.get("source") or c.get("document_id")
        if src_id:
            distinct_sources.add(str(src_id))

        auth = c.get("authority_score")
        if auth is None:
            auth = meta.get("authority_score", 0.5)
        authorities.append(float(auth))

        ver = meta.get("version")
        if query_version and ver:
            ver_clean = str(ver).strip().lower()
            qver_clean = str(query_version).strip().lower()
            if ver_clean in (qver_clean, "all", ""):
                has_exact_version = True

    num_sources = len(distinct_sources)
    sources_score = min(1.0, num_sources / 3.0)

    # 3. Exact version match
    if not query_version:
        version_score = 1.0
    else:
        version_score = 1.0 if has_exact_version else 0.0

    # 4. Average authority
    avg_authority = (sum(authorities) / len(authorities)) if authorities else 0.5
    avg_authority = max(0.0, min(1.0, avg_authority))

    # 5. Citation validation score
    cit_score = 1.0 if citation_validation_passed else 0.0

    # Weighted sum
    weighted_sum = (
        top_relevance_weight * top_relevance
        + score_separation_weight * score_separation
        + sources_count_weight * sources_score
        + version_match_weight * version_score
        + authority_weight * avg_authority
        + citation_validation_weight * cit_score
    )

    # Apply penalties
    p_relaxed = relaxed_retrieval_penalty if relaxed_retrieval_used else 0.0
    p_conflict = conflict_penalty if support_answer.conflicts_detected else 0.0

    final_score = weighted_sum - p_relaxed - p_conflict
    final_score = max(0.0, min(1.0, round(final_score, 4)))

    # Map score to confidence level
    if final_score >= 0.75:
        level = "high"
    elif final_score >= 0.50:
        level = "medium"
    else:
        level = "low"

    breakdown = {
        "top_relevance": top_relevance,
        "score_separation": score_separation,
        "num_independent_sources": num_sources,
        "sources_score": sources_score,
        "exact_version_match": version_score == 1.0,
        "version_score": version_score,
        "average_authority": avg_authority,
        "citation_validation_passed": citation_validation_passed,
        "citation_score": cit_score,
        "relaxed_retrieval_penalty": p_relaxed,
        "conflict_penalty": p_conflict,
        "weighted_sum": round(weighted_sum, 4),
        "final_confidence_score": final_score,
        "confidence_level": level,
    }

    return final_score, level, breakdown
