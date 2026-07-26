import re
from typing import Any

from app.config import settings
from app.models import ContextBuildResult


def normalize_text(text: str) -> str:
    """Normalize text by stripping, lowercasing, and collapsing whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def compute_text_similarity(text1: str, text2: str) -> float:
    """Computes token Jaccard similarity ratio between two normalized texts."""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    if norm1 == norm2:
        return 1.0

    tokens1 = set(re.findall(r"\w+", norm1))
    tokens2 = set(re.findall(r"\w+", norm2))

    if not tokens1 or not tokens2:
        return 0.0

    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def categorize_doc_type(doc_type: str) -> str:
    dt = doc_type.lower().strip()
    if any(k in dt for k in ["product_documentation", "documentation", "guide", "manual", "doc"]):
        return "product_documentation"
    elif any(k in dt for k in ["release_notes", "release", "changelog"]):
        return "release_notes"
    elif any(k in dt for k in ["known_issues", "known_issue", "bug", "issue"]):
        return "known_issues"
    elif any(k in dt for k in ["support_ticket", "ticket", "reviewed_ticket"]):
        return "support_tickets"
    return "other"


def extract_item_fields(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        d = item.model_dump()
        if "chunk" in d and isinstance(d["chunk"], dict):
            c_dict = d["chunk"]
            cid = c_dict.get("chunk_id", "")
            text = c_dict.get("text", "")
            meta = c_dict.get("metadata", {})
            doc_id = c_dict.get("document_id") or meta.get("source") or meta.get("title") or "unknown_doc"
        else:
            cid = d.get("chunk_id", "")
            text = d.get("text", "")
            meta = d.get("metadata", {})
            doc_id = d.get("document_id") or meta.get("source") or meta.get("title") or "unknown_doc"
        return {"chunk_id": cid, "document_id": doc_id, "text": text, "metadata": meta, "original": item}

    elif isinstance(item, dict):
        cid = item.get("chunk_id", "")
        text = item.get("text", "")
        meta = item.get("metadata", {})
        doc_id = item.get("document_id") or meta.get("source") or meta.get("title") or "unknown_doc"
        return {"chunk_id": cid, "document_id": doc_id, "text": text, "metadata": meta, "original": item}

    else:
        cid = getattr(item, "chunk_id", str(item))
        text = getattr(item, "text", str(item))
        meta = getattr(item, "metadata", {})
        doc_id = meta.get("source") if isinstance(meta, dict) else "unknown_doc"
        return {"chunk_id": cid, "document_id": doc_id, "text": text, "metadata": meta, "original": item}


def _resolve_children_to_parents(chunks: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not chunks:
        return [], []

    parent_groups: dict[str, dict[str, Any]] = {}
    ordered_parent_keys: list[str] = []
    standalone_duplicates: list[dict[str, Any]] = []

    for item in chunks:
        fields = extract_item_fields(item)
        cid = fields["chunk_id"]
        meta = fields["metadata"] if isinstance(fields["metadata"], dict) else {}

        pid = meta.get("parent_id") or getattr(item, "parent_id", None)
        ptext = meta.get("parent_text") or getattr(item, "parent_text", None)

        if pid and ptext:
            key = str(pid)
            text_to_use = str(ptext)
            is_child = True
        else:
            key = cid
            text_to_use = fields["text"]
            is_child = False

        if key not in parent_groups:
            parent_groups[key] = {
                "chunk_id": key,
                "document_id": fields["document_id"],
                "text": text_to_use,
                "metadata": meta,
                "semantic_score": fields.get("original", {}).get("semantic_score") if isinstance(fields.get("original"), dict) else getattr(item, "semantic_score", None),
                "keyword_score": fields.get("original", {}).get("keyword_score") if isinstance(fields.get("original"), dict) else getattr(item, "keyword_score", None),
                "fused_score": fields.get("original", {}).get("fused_score") if isinstance(fields.get("original"), dict) else getattr(item, "fused_score", None),
                "final_score": fields.get("original", {}).get("final_score") if isinstance(fields.get("original"), dict) else getattr(item, "final_score", None),
                "triggered_children": [cid] if is_child else [],
                "original": fields["original"],
            }
            ordered_parent_keys.append(key)
        else:
            if is_child:
                group = parent_groups[key]
                if cid not in group["triggered_children"]:
                    group["triggered_children"].append(cid)

                for score_key in ["semantic_score", "keyword_score", "fused_score", "final_score"]:
                    item_score = fields.get("original", {}).get(score_key) if isinstance(fields.get("original"), dict) else getattr(item, score_key, None)
                    if item_score is not None:
                        if group.get(score_key) is None or item_score > group[score_key]:
                            group[score_key] = item_score
            else:
                standalone_duplicates.append(fields["original"])

    return [parent_groups[k] for k in ordered_parent_keys], standalone_duplicates


def resolve_children_to_parents(chunks: list[Any]) -> list[dict[str, Any]]:
    """Resolves child chunks to their parent chunks, merging multiple children of the same parent."""
    resolved, _ = _resolve_children_to_parents(chunks)
    return resolved


def format_source_block(source_id: str, fields: dict[str, Any]) -> str:
    meta = fields.get("metadata", {})
    title = meta.get("title", "Untitled Document") if isinstance(meta, dict) else "Untitled Document"
    source = meta.get("source", fields.get("document_id", "Unknown")) if isinstance(meta, dict) else "Unknown"
    doc_type = meta.get("document_type", "General") if isinstance(meta, dict) else "General"
    version = meta.get("version", "N/A") if isinstance(meta, dict) else "N/A"
    text = fields.get("text", "")

    return (
        f"[Source {source_id}]\n"
        f"Title: {title}\n"
        f"Source: {source}\n"
        f"Type: {doc_type}\n"
        f"Version: {version if version else 'N/A'}\n"
        f"Content:\n{text}"
    )


class ContextBuilder:
    def __init__(
        self,
        max_chunks_per_source: int = settings.max_chunks_per_source,
        max_context_characters: int = settings.max_context_characters,
        near_duplicate_threshold: float = settings.near_duplicate_threshold,
    ) -> None:
        self.max_chunks_per_source = max_chunks_per_source
        self.max_context_characters = max_context_characters
        self.near_duplicate_threshold = near_duplicate_threshold

    def build_context(self, chunks: list[Any]) -> ContextBuildResult:
        if not chunks:
            return ContextBuildResult()

        resolved_chunks, initial_duplicates = _resolve_children_to_parents(chunks)
        parsed_candidates = [extract_item_fields(c) for c in resolved_chunks]

        # Step 1: Deduplication (exact chunk ID, exact normalized text, near-duplicate similarity)
        non_duplicate_candidates: list[dict[str, Any]] = []
        excluded_duplicates: list[dict[str, Any]] = list(initial_duplicates)

        seen_chunk_ids: set[str] = set()
        seen_normalized_texts: set[str] = set()
        accepted_normalized_texts: list[str] = []

        for cand in parsed_candidates:
            cid = cand["chunk_id"]
            text = cand["text"]
            norm_text = normalize_text(text)

            # Check 1: Exact chunk ID
            if cid and cid in seen_chunk_ids:
                excluded_duplicates.append(cand["original"])
                continue

            # Check 2: Exact normalized text
            if norm_text in seen_normalized_texts:
                excluded_duplicates.append(cand["original"])
                continue

            # Check 3: Near-duplicate detection
            is_near_dup = False
            for prev_norm in accepted_normalized_texts:
                if compute_text_similarity(norm_text, prev_norm) >= self.near_duplicate_threshold:
                    is_near_dup = True
                    break

            if is_near_dup:
                excluded_duplicates.append(cand["original"])
                continue

            # Mark accepted non-duplicate
            if cid:
                seen_chunk_ids.add(cid)
            seen_normalized_texts.add(norm_text)
            accepted_normalized_texts.append(norm_text)
            non_duplicate_candidates.append(cand)

        # Step 2: Source diversity ordering while preserving rank order where possible
        preferred_categories = ["product_documentation", "release_notes", "known_issues", "support_tickets"]

        ordered_candidates: list[dict[str, Any]] = []
        added_indices: set[int] = set()

        # Pass A: Pick top candidate from each unrepresented preferred category
        for cat in preferred_categories:
            for idx, cand in enumerate(non_duplicate_candidates):
                doc_type = cand["metadata"].get("document_type", "") if isinstance(cand["metadata"], dict) else ""
                if categorize_doc_type(doc_type) == cat:
                    if idx not in added_indices:
                        ordered_candidates.append(cand)
                        added_indices.add(idx)
                    break

        # Pass B: Append remaining candidates in original rank order
        for idx, cand in enumerate(non_duplicate_candidates):
            if idx not in added_indices:
                ordered_candidates.append(cand)
                added_indices.add(idx)

        # Step 3: Enforce source document limits and character budget
        selected_chunks: list[dict[str, Any]] = []
        excluded_source_limit: list[dict[str, Any]] = []
        excluded_budget_overflow: list[dict[str, Any]] = []

        source_counts: dict[str, int] = {}
        formatted_blocks: list[str] = []
        current_char_count: int = 0

        for cand in ordered_candidates:
            doc_id = cand["document_id"]
            current_count = source_counts.get(doc_id, 0)

            # Check source limit
            if current_count >= self.max_chunks_per_source:
                excluded_source_limit.append(cand["original"])
                continue

            # Assign stable source ID (S1, S2, S3...)
            source_id = f"S{len(selected_chunks) + 1}"
            formatted_block = format_source_block(source_id, cand)

            # Check character budget
            projected_chars = current_char_count + len(formatted_block) + (2 if formatted_blocks else 0)
            if projected_chars > self.max_context_characters:
                excluded_budget_overflow.append(cand["original"])
                continue

            # Accept candidate
            source_counts[doc_id] = current_count + 1
            current_char_count = projected_chars
            formatted_blocks.append(formatted_block)

            selected_item = cand["original"]
            if isinstance(selected_item, dict):
                selected_item_copy = dict(selected_item)
                selected_item_copy["source_id"] = source_id
                selected_chunks.append(selected_item_copy)
            else:
                selected_chunks.append(selected_item)

        formatted_context = "\n\n".join(formatted_blocks)

        return ContextBuildResult(
            selected_chunks=selected_chunks,
            excluded_duplicates=excluded_duplicates,
            excluded_budget_overflow=excluded_budget_overflow,
            excluded_source_limit=excluded_source_limit,
            formatted_context=formatted_context,
        )


def build_context(
    chunks: list[Any],
    max_chunks_per_source: int = settings.max_chunks_per_source,
    max_context_characters: int = settings.max_context_characters,
    near_duplicate_threshold: float = settings.near_duplicate_threshold,
) -> ContextBuildResult:
    builder = ContextBuilder(
        max_chunks_per_source=max_chunks_per_source,
        max_context_characters=max_context_characters,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    return builder.build_context(chunks)
