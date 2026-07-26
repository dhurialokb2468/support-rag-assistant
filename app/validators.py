import json
import re
from typing import Any

from pydantic import ValidationError

from app.config import settings
from app.logger import get_logger
from app.models import AbstentionReason, Citation, SupportAnswer

logger = get_logger("validators")

ABSTENTION_ANSWER_TEXT = "I could not find sufficient evidence in the available InsightFlow documentation to answer this reliably."


def clean_markdown_fences(text: str) -> str:
    """Removes Markdown code fences (e.g. ```json ... ```) and leading/trailing whitespace."""
    cleaned = text.strip()
    if "```" in cleaned:
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def parse_json_string(text: str) -> tuple[Any | None, str | None]:
    """Cleans code fences and parses JSON string. Returns (parsed_obj, error_message)."""
    cleaned = clean_markdown_fences(text)
    try:
        data = json.loads(cleaned)
        return data, None
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg} at line {exc.lineno} column {exc.colno}"


def validate_support_answer(data_or_text: Any) -> tuple[SupportAnswer | None, str | None]:
    """Validates raw dict or JSON string against SupportAnswer Pydantic model. Returns (SupportAnswer, error_message)."""
    if isinstance(data_or_text, str):
        data, err = parse_json_string(data_or_text)
        if err:
            return None, err
    else:
        data = data_or_text

    if not isinstance(data, dict):
        return None, f"Expected JSON object/dict, got {type(data).__name__}"

    try:
        answer_obj = SupportAnswer.model_validate(data)
        return answer_obj, None
    except ValidationError as exc:
        return None, str(exc)


def create_fallback_support_answer(error_message: str, question: str = "") -> SupportAnswer:
    """Creates a safe fallback SupportAnswer requiring escalation and preserving the error message."""
    return SupportAnswer(
        answer="Insufficient clear information or valid format available to answer the query reliably. The request has been escalated to a support specialist.",
        likely_cause=None,
        resolution_steps=[],
        citations=[],
        confidence="low",
        confidence_score=0.0,
        escalation_required=True,
        escalation_reason=f"Validation failure: {error_message}",
        conflicts_detected=False,
    )


def normalize_citation_id(raw_id: str) -> str:
    """Normalizes raw citation string (e.g. '[S1]', 'S1 ', ' s1 ') into canonical 'S1' format."""
    clean = re.sub(r"[\[\]\s]", "", str(raw_id)).upper()
    return clean


def validate_and_resolve_citations(
    support_answer: SupportAnswer,
    selected_chunks: list[dict[str, Any]],
) -> tuple[SupportAnswer, list[Citation], bool, dict[str, Any]]:
    """
    Verifies every cited source ID exists in selected_chunks.
    Deduplicates citations, removes unknown source IDs, resolves valid citations into Citation objects,
    adjusts confidence / escalation status if citations are invalid or missing for factual answers.
    Returns (updated_support_answer, resolved_citations, citation_validation_passed, details_dict).
    """
    sources_by_id: dict[str, dict[str, Any]] = {}
    for idx, chunk in enumerate(selected_chunks, start=1):
        sid = chunk.get("source_id") or f"S{idx}"
        sources_by_id[sid.upper()] = chunk

    raw_citations = support_answer.citations or []
    normalized_citations: list[str] = []
    seen_cits: set[str] = set()

    for c in raw_citations:
        norm = normalize_citation_id(c)
        if norm and norm not in seen_cits:
            seen_cits.add(norm)
            normalized_citations.append(norm)

    valid_citations: list[str] = []
    unknown_citations: list[str] = []

    for c in normalized_citations:
        if c in sources_by_id:
            valid_citations.append(c)
        else:
            unknown_citations.append(c)

    resolved_citations: list[Citation] = []
    for sid in valid_citations:
        src = sources_by_id[sid]
        meta = src.get("metadata", {}) if isinstance(src.get("metadata"), dict) else {}
        title = meta.get("title") or "Untitled Document"
        source_path = meta.get("source") or src.get("document_id") or "Unknown"
        text_snippet = src.get("text", "")

        resolved_citations.append(
            Citation(
                source_id=sid,
                title=title,
                source=source_path,
                quoted_text=text_snippet[:300],
            )
        )

    is_abstained = support_answer.escalation_required or "insufficient information" in support_answer.answer.lower()
    validation_passed = True
    reasons: list[str] = []

    if unknown_citations:
        validation_passed = False
        reasons.append(f"Unknown citation IDs present: {unknown_citations}")

    if not is_abstained and not valid_citations:
        validation_passed = False
        reasons.append("Factual non-abstained answer provided without any valid citations")

    updated_answer_dict = support_answer.model_dump()
    updated_answer_dict["citations"] = valid_citations

    if not validation_passed:
        if updated_answer_dict["confidence"] == "high":
            updated_answer_dict["confidence"] = "medium"
            updated_answer_dict["confidence_score"] = max(0.0, float(updated_answer_dict["confidence_score"]) - 0.20)
        elif updated_answer_dict["confidence"] == "medium":
            updated_answer_dict["confidence"] = "low"
            updated_answer_dict["confidence_score"] = max(0.0, float(updated_answer_dict["confidence_score"]) - 0.30)

        if not is_abstained and not valid_citations:
            updated_answer_dict["escalation_required"] = True
            esc_reason = updated_answer_dict.get("escalation_reason") or ""
            new_reason = "; ".join(reasons)
            updated_answer_dict["escalation_reason"] = f"{esc_reason}; {new_reason}".strip("; ")

    updated_support_answer = SupportAnswer.model_validate(updated_answer_dict)

    details = {
        "valid_citations": valid_citations,
        "unknown_citations": unknown_citations,
        "validation_passed": validation_passed,
        "reasons": reasons,
    }

    return updated_support_answer, resolved_citations, validation_passed, details


def verify_citation_support(
    claim: str,
    passage: str,
    generator: Any = None,
) -> str:
    """
    Second-pass verifier checking whether a cited passage supports a claim.
    Returns 'supported', 'partially_supported', or 'unsupported'. Does not introduce new facts.
    """
    if not claim or not claim.strip() or not passage or not passage.strip():
        return "unsupported"

    if generator is not None and hasattr(generator, "generate"):
        prompt = f"""
You are a strict factual verification assistant.

Evaluate whether the Passage supports the Claim.

Rules:
1. Do NOT use outside knowledge or introduce new facts.
2. Return ONLY one of these three exact words:
   - "supported" if the Passage fully supports the Claim.
   - "partially_supported" if the Passage partially supports the Claim.
   - "unsupported" if the Passage does not support or contradicts the Claim.

Claim:
{claim}

Passage:
{passage}
"""
        try:
            resp, _ = generator.generate(prompt, temperature=0.0)
            clean_resp = resp.strip().lower()
            if "partially_supported" in clean_resp or "partially" in clean_resp:
                return "partially_supported"
            elif "supported" in clean_resp and "unsupported" not in clean_resp:
                return "supported"
            elif "unsupported" in clean_resp:
                return "unsupported"
        except Exception:
            pass

    # Deterministic fallback evaluation (token overlap check)
    claim_norm = re.sub(r"\s+", " ", claim.strip().lower())
    passage_norm = re.sub(r"\s+", " ", passage.strip().lower())

    if claim_norm == passage_norm:
        return "supported"

    claim_tokens = set(re.findall(r"\w+", claim_norm))
    passage_tokens = set(re.findall(r"\w+", passage_norm))

    if not claim_tokens or not passage_tokens:
        return "unsupported"

    intersection = claim_tokens & passage_tokens
    overlap_ratio = len(intersection) / len(claim_tokens)

    if overlap_ratio >= 0.5:
        return "supported"
    elif overlap_ratio >= 0.2:
        return "partially_supported"
    else:
        return "unsupported"


def evaluate_abstention(
    question: str,
    support_answer: SupportAnswer,
    retrieved_chunks: list[dict[str, Any]],
    citation_validation_passed: bool,
    confidence_score: float,
    confidence_level: str,
    min_confidence_threshold: float = settings.min_confidence_score,
) -> tuple[bool, AbstentionReason | None, SupportAnswer]:
    """
    Evaluates whether the system should abstain across 7 criteria.
    Returns (should_abstain, abstention_reason_enum, updated_support_answer).
    """
    q_norm = question.strip().lower()
    scores = [
        float(c.get("final_score") or c.get("normalized_reranker_score") or c.get("reranker_score") or c.get("semantic_score", 0.0))
        for c in retrieved_chunks
    ]
    top_score = max(scores) if scores else 0.0

    reason: AbstentionReason | None = None

    # Criterion 4: Undocumented future plans
    future_keywords = ["future plan", "roadmap", "future release", "version 4.0", "next year", "planned feature", "when will", "upcoming feature"]
    if any(k in q_norm for k in future_keywords):
        has_doc_plan = any("roadmap" in c.get("text", "").lower() or "future" in c.get("text", "").lower() for c in retrieved_chunks)
        if not has_doc_plan:
            reason = AbstentionReason.UNDOCUMENTED_FUTURE_PLANS

    # Criterion 6: Policy, legal, security, billing, or compensation decisions
    policy_keywords = ["pricing", "refund", "billing dispute", "legal liability", "gdpr fine", "compensation policy", "discount rate", "legal action"]
    if reason is None and any(k in q_norm for k in policy_keywords):
        has_policy_doc = any("policy" in c.get("text", "").lower() or "terms" in c.get("text", "").lower() for c in retrieved_chunks)
        if not has_policy_doc:
            reason = AbstentionReason.POLICY_LEGAL_BILLING_DECISION

    # Criterion 5: Unresolved source conflicts
    if reason is None and support_answer.conflicts_detected:
        reason = AbstentionReason.UNRESOLVED_SOURCE_CONFLICT

    # Criterion 1: Low relevance threshold
    if reason is None and (not retrieved_chunks or top_score < 0.35):
        reason = AbstentionReason.LOW_RELEVANCE

    # Criterion 3: Outside knowledge base
    if reason is None and (top_score < 0.40 and not any("insightflow" in c.get("text", "").lower() for c in retrieved_chunks)):
        reason = AbstentionReason.OUTSIDE_KNOWLEDGE_BASE

    # Criterion 2: No valid citations for non-abstained factual answer
    is_already_abstained = support_answer.escalation_required or ABSTENTION_ANSWER_TEXT in support_answer.answer or "insufficient information" in support_answer.answer.lower()
    if reason is None and not citation_validation_passed and not is_already_abstained:
        reason = AbstentionReason.NO_VALID_CITATIONS

    # Criterion 7: Confidence below threshold
    if reason is None and (confidence_score < min_confidence_threshold or confidence_level == "low" or is_already_abstained):
        reason = AbstentionReason.CONFIDENCE_BELOW_THRESHOLD

    # Check if abstention was already requested or if any criterion matched
    if reason is not None or is_already_abstained:
        final_reason = reason or AbstentionReason.CONFIDENCE_BELOW_THRESHOLD
        logger.warning(f"Abstention triggered (Reason: '{final_reason.value}') for query: '{question[:40]}...'")
        abstained_ans = SupportAnswer(
            answer=ABSTENTION_ANSWER_TEXT,
            likely_cause=None,
            resolution_steps=[],
            citations=support_answer.citations,
            confidence="low",
            confidence_score=min(confidence_score, 0.40),
            escalation_required=True,
            escalation_reason=f"Abstention triggered: {final_reason.value}",
            conflicts_detected=support_answer.conflicts_detected,
        )
        return True, final_reason, abstained_ans

    return False, None, support_answer
