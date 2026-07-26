from datetime import datetime, timezone
import json
import re
from typing import Any

from app.logger import get_logger
from app.models import ConflictResult

logger = get_logger("conflicts")


def doc_type_priority(doc_type: str) -> int:
    dt = doc_type.lower().strip()
    if any(k in dt for k in ["release_notes", "release"]):
        return 4
    elif any(k in dt for k in ["product_documentation", "documentation", "guide", "manual"]):
        return 3
    elif any(k in dt for k in ["known_issues", "known_issue", "bug"]):
        return 2
    elif any(k in dt for k in ["support_ticket", "ticket"]):
        return 1
    return 0


def parse_date(date_str: str | None) -> datetime | None:
    if not date_str or not isinstance(date_str, str) or not date_str.strip():
        return None
    try:
        clean = date_str.strip()
        if "T" in clean:
            dt = datetime.fromisoformat(clean.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(clean[:10], "%Y-%m-%d")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def detect_conflict_between_passages(
    text1: str,
    text2: str,
    generator: Any = None,
) -> tuple[bool, str]:
    """Detects if two passages contain materially conflicting instructions or claims."""
    t1_norm = text1.lower()
    t2_norm = text2.lower()

    contradiction = False
    reason = ""

    if ("enable" in t1_norm and "disable" in t2_norm) or ("disable" in t1_norm and "enable" in t2_norm):
        contradiction = True
        reason = "Passages recommend opposing actions (enable vs. disable)."
    elif ("true" in t1_norm and "false" in t2_norm) or ("false" in t1_norm and "true" in t2_norm):
        contradiction = True
        reason = "Passages contain contradictory boolean parameter settings."
    elif ("supported" in t1_norm and ("unsupported" in t2_norm or "deprecated" in t2_norm)) or (("unsupported" in t1_norm or "deprecated" in t1_norm) and "supported" in t2_norm):
        contradiction = True
        reason = "Passages conflict on feature support status (supported vs. unsupported/deprecated)."
    elif ("required" in t1_norm and "optional" in t2_norm) or ("optional" in t1_norm and "required" in t2_norm):
        contradiction = True
        reason = "Passages conflict on parameter requirement (required vs. optional)."

    if not contradiction and generator is not None and hasattr(generator, "generate"):
        prompt = f"""
Compare these two passages from product documentation or support sources.

Determine if they give materially conflicting or incompatible instructions or claims.

Passage 1:
{text1}

Passage 2:
{text2}

Return ONLY a JSON object:
{{
  "conflict": true or false,
  "reason": "<short summary of conflict if present, else empty>"
}}
"""
        try:
            resp, _ = generator.generate(prompt, temperature=0.0)
            cleaned = resp.strip()
            if "```" in cleaned:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
                if match:
                    cleaned = match.group(1).strip()
            data = json.loads(cleaned)
            if data.get("conflict"):
                return True, data.get("reason", "Constrained LLM verification detected material conflict.")
        except Exception as exc:
            logger.debug(f"LLM conflict verification skipped/failed: {exc}")

    return contradiction, reason


def resolve_conflict_priority(
    chunk1: dict[str, Any],
    chunk2: dict[str, Any],
    query_version: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    meta1 = chunk1.get("metadata", {}) if isinstance(chunk1.get("metadata"), dict) else {}
    meta2 = chunk2.get("metadata", {}) if isinstance(chunk2.get("metadata"), dict) else {}

    sid1 = chunk1.get("source_id") or chunk1.get("chunk_id", "Source 1")
    sid2 = chunk2.get("source_id") or chunk2.get("chunk_id", "Source 2")

    # Rule 1: Exact version match
    if query_version:
        qv = str(query_version).strip().lower()
        v1 = str(meta1.get("version", "")).strip().lower()
        v2 = str(meta2.get("version", "")).strip().lower()

        match1 = (v1 == qv)
        match2 = (v2 == qv)

        if match1 and not match2:
            return chunk1, f"Source {sid1} matches exact target version {query_version}, whereas {sid2} applies to version {meta2.get('version')}."
        if match2 and not match1:
            return chunk2, f"Source {sid2} matches exact target version {query_version}, whereas {sid1} applies to version {meta1.get('version')}."

    # Rule 2: Document type priority
    dt1_score = doc_type_priority(meta1.get("document_type", ""))
    dt2_score = doc_type_priority(meta2.get("document_type", ""))

    if dt1_score > dt2_score:
        return chunk1, f"Official {meta1.get('document_type')} ({sid1}) supersedes {meta2.get('document_type')} ({sid2})."
    if dt2_score > dt1_score:
        return chunk2, f"Official {meta2.get('document_type')} ({sid2}) supersedes {meta1.get('document_type')} ({sid1})."

    # Rule 3: Reviewed status
    rev1 = bool(meta1.get("reviewed", False))
    rev2 = bool(meta2.get("reviewed", False))

    if rev1 and not rev2:
        return chunk1, f"Reviewed/approved source {sid1} supersedes unreviewed source {sid2}."
    if rev2 and not rev1:
        return chunk2, f"Reviewed/approved source {sid2} supersedes unreviewed source {sid1}."

    # Rule 4: Authority score
    auth1 = float(chunk1.get("authority_score") or meta1.get("authority_score", 0.5))
    auth2 = float(chunk2.get("authority_score") or meta2.get("authority_score", 0.5))

    if auth1 > auth2 + 0.05:
        return chunk1, f"Source {sid1} has higher authority score ({auth1:.2f}) than {sid2} ({auth2:.2f})."
    if auth2 > auth1 + 0.05:
        return chunk2, f"Source {sid2} has higher authority score ({auth2:.2f}) than {sid1} ({auth1:.2f})."

    # Rule 5: Freshness
    d1 = parse_date(meta1.get("updated_at"))
    d2 = parse_date(meta2.get("updated_at"))

    if d1 and d2:
        if d1 > d2:
            return chunk1, f"Source {sid1} is newer ({meta1.get('updated_at')}) than {sid2} ({meta2.get('updated_at')})."
        elif d2 > d1:
            return chunk2, f"Source {sid2} is newer ({meta2.get('updated_at')}) than {sid1} ({meta1.get('updated_at')})."

    return None, f"Sources {sid1} and {sid2} have equal authority, document type, and freshness; conflict remains unresolved."


def detect_and_resolve_conflicts(
    question: str,
    selected_chunks: list[dict[str, Any]],
    query_version: str | None = None,
    generator: Any = None,
) -> ConflictResult:
    if not selected_chunks or len(selected_chunks) < 2:
        return ConflictResult(conflict_detected=False)

    for i in range(len(selected_chunks)):
        for j in range(i + 1, len(selected_chunks)):
            c1 = selected_chunks[i]
            c2 = selected_chunks[j]

            sid1 = c1.get("source_id") or c1.get("chunk_id", f"S{i+1}")
            sid2 = c2.get("source_id") or c2.get("chunk_id", f"S{j+1}")

            is_conflict, conflict_reason = detect_conflict_between_passages(
                c1.get("text", ""),
                c2.get("text", ""),
                generator=generator,
            )

            if is_conflict:
                logger.warning(f"Material conflict detected between sources '{sid1}' and '{sid2}': {conflict_reason}")
                preferred_chunk, resolution_reason = resolve_conflict_priority(c1, c2, query_version=query_version)

                if preferred_chunk is not None:
                    pref_id = preferred_chunk.get("source_id") or preferred_chunk.get("chunk_id")
                    summary_text = f"Conflict detected between {sid1} and {sid2}: {conflict_reason} Resolved in favor of {pref_id}."
                    logger.info(f"Conflict resolved in favor of '{pref_id}': {resolution_reason}")
                    return ConflictResult(
                        conflict_detected=True,
                        conflicting_source_ids=[sid1, sid2],
                        summary=summary_text,
                        preferred_source_id=pref_id,
                        preference_reason=resolution_reason,
                        unresolved=False,
                    )
                else:
                    summary_text = f"Unresolved material conflict detected between {sid1} and {sid2}: {conflict_reason}"
                    logger.warning(f"UNRESOLVED CONFLICT between '{sid1}' and '{sid2}': {resolution_reason}")
                    return ConflictResult(
                        conflict_detected=True,
                        conflicting_source_ids=[sid1, sid2],
                        summary=summary_text,
                        preferred_source_id=None,
                        preference_reason=resolution_reason,
                        unresolved=True,
                    )

    return ConflictResult(conflict_detected=False)
