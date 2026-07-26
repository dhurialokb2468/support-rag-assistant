from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from app.embeddings import EmbeddingService


def is_gap_candidate(interaction: dict[str, Any]) -> tuple[bool, str]:
    """
    Determines whether an interaction represents a documentation-gap candidate.
    Criteria:
    1. User marked answer unhelpful (helpful == 0)
    2. Similar question recurring
    3. System abstained / Escalated
    4. Confidence was low (< 0.50 or confidence level "low")
    5. No strong source found (top score < 0.60 when retrieved sources present)
    """
    ans_data = interaction.get("answer_json", {}) if isinstance(interaction.get("answer_json"), dict) else {}

    # 1. User marked unhelpful
    feedback_list = interaction.get("feedback", [])
    for fb in feedback_list:
        if isinstance(fb, dict) and fb.get("helpful") == 0:
            return True, f"Unhelpful Feedback ({fb.get('issue_type') or 'negative'})"

    # 2. Recurring query flag
    if interaction.get("is_recurring"):
        return True, "Recurring Query"

    # 3. Abstained / Escalated
    if interaction.get("abstained") or ans_data.get("escalation_required"):
        return True, "Abstained / Escalated"

    # 4. Low confidence
    conf_score = float(interaction.get("confidence_score", ans_data.get("confidence_score", 1.0) or 1.0))
    conf_level = str(ans_data.get("confidence", "")).lower()
    if conf_score < 0.50 or conf_level == "low":
        return True, f"Low Confidence ({conf_score:.2f})"

    # 5. No strong source found
    retrieved_sources = interaction.get("retrieved_sources")
    if retrieved_sources is not None:
        top_score = float(retrieved_sources[0].get("final_score", 0.0)) if retrieved_sources else 0.0
        if not retrieved_sources or top_score < 0.60:
            return True, f"Weak Source Relevance ({top_score:.2f})"

    return False, "Sufficient Documentation"


def identify_gap_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filters records and annotates documentation-gap candidates."""
    candidates = []
    question_counts = Counter(r.get("question", "").strip().lower() for r in records if r.get("question"))

    for r in records:
        r_copy = dict(r)
        q_norm = r_copy.get("question", "").strip().lower()
        if question_counts.get(q_norm, 0) > 1:
            r_copy["is_recurring"] = True

        is_cand, reason = is_gap_candidate(r_copy)
        if is_cand:
            r_copy["gap_reason"] = reason
            candidates.append(r_copy)

    return candidates


def generate_recommended_action(
    category: str | None,
    representative_question: str,
    common_reason: str | None,
) -> str:
    """Generates an actionable documentation recommendation statement."""
    cat_str = f"in {category.upper()}" if category else "in Knowledge Base"
    if "Abstained" in str(common_reason):
        return f"Publish new official guide {cat_str} to resolve unanswerable query: '{representative_question}'."
    elif "Low Confidence" in str(common_reason):
        return f"Expand existing documentation details {cat_str} for: '{representative_question}'."
    elif "Weak Source" in str(common_reason):
        return f"Add explicit terminology & error codes {cat_str} for: '{representative_question}'."
    else:
        return f"Update & review documentation {cat_str} addressing: '{representative_question}'."


def analyze_documentation_gaps(
    records: list[dict[str, Any]],
    min_records: int = 3,
    num_clusters: int | None = None,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, Any]:
    """
    Identifies gap candidates and clusters them using AgglomerativeClustering or KMeans.
    Returns structured cluster report or status 'too_few_records' if candidates < min_records.
    """
    candidates = identify_gap_candidates(records)
    total_candidates = len(candidates)

    if total_candidates < min_records:
        return {
            "status": "too_few_records",
            "message": f"Candidate count ({total_candidates}) is less than minimum required ({min_records}).",
            "total_candidates": total_candidates,
            "clusters": [],
        }

    # Embed candidate questions using EmbeddingService
    embedder = embedding_service or EmbeddingService()
    texts = [c.get("question", "") for c in candidates]
    vectors_list, _ = embedder.embed_documents(texts)
    embeddings = np.array(vectors_list)

    # Determine number of clusters
    k = num_clusters or max(2, min(5, total_candidates // 2))

    # Cluster using KMeans if candidates >= 6 else AgglomerativeClustering
    if total_candidates >= 6:
        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings)
    else:
        k_agg = min(k, total_candidates)
        clusterer = AgglomerativeClustering(n_clusters=k_agg)
        labels = clusterer.fit_predict(embeddings)

    clusters_output = []

    for cluster_idx in range(len(set(labels))):
        member_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_idx]
        if not member_indices:
            continue

        member_candidates = [candidates[i] for i in member_indices]
        member_vectors = embeddings[member_indices]

        # Calculate centroid and find representative question (closest to centroid)
        centroid = np.mean(member_vectors, axis=0)
        distances = np.linalg.norm(member_vectors - centroid, axis=1)
        rep_idx = member_indices[np.argmin(distances)]
        rep_question = candidates[rep_idx].get("question", "")

        # Compute metadata aggregates
        categories = [
            c.get("category") or c.get("answer_json", {}).get("category")
            for c in member_candidates
            if c.get("category") or (isinstance(c.get("answer_json"), dict) and c.get("answer_json", {}).get("category"))
        ]
        main_cat = Counter(categories).most_common(1)[0][0] if categories else None

        versions = [
            c.get("version") or c.get("target_version")
            for c in member_candidates
            if c.get("version") or c.get("target_version")
        ]
        common_vers = [v for v, _ in Counter(versions).most_common(3) if v]

        conf_scores = [
            float(c.get("confidence_score", c.get("answer_json", {}).get("confidence_score", 0.0) or 0.0))
            for c in member_candidates
        ]
        avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0

        reasons = [c.get("gap_reason", "Low Confidence") for c in member_candidates]
        common_reason = Counter(reasons).most_common(1)[0][0] if reasons else None

        esc_reasons = [
            c.get("answer_json", {}).get("escalation_reason")
            for c in member_candidates
            if isinstance(c.get("answer_json"), dict) and c.get("answer_json", {}).get("escalation_reason")
        ]
        common_esc_reason = Counter(esc_reasons).most_common(1)[0][0] if esc_reasons else None

        example_questions = [c.get("question", "") for c in member_candidates[:4]]

        rec_action = generate_recommended_action(main_cat, rep_question, common_reason)

        clusters_output.append({
            "cluster_id": f"gap_{cluster_idx + 1:02d}",
            "question_count": len(member_candidates),
            "representative_question": rep_question,
            "example_questions": example_questions,
            "main_category": main_cat,
            "common_versions": common_vers,
            "average_confidence": round(avg_conf, 4),
            "common_escalation_reason": common_esc_reason or common_reason,
            "recommended_documentation_action": rec_action,
        })

    return {
        "status": "success",
        "total_candidates": total_candidates,
        "clusters_count": len(clusters_output),
        "clusters": clusters_output,
    }
