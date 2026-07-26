import hashlib
import re

from app.config import settings
from app.models import ChildChunk, Chunk, ParentChunk, SourceDocument


def create_chunk_id(document_id: str, index: int, text: str) -> str:
    raw = f"{document_id}:{index}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def split_at_word_boundary(text: str, end: int) -> int:
    if end >= len(text):
        return len(text)

    space = text.rfind(" ", 0, end)

    if space == -1:
        return end

    return space


def chunk_document_fixed(
    document: SourceDocument,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document.text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = split_at_word_boundary(text, proposed_end)

        if end <= start:
            end = proposed_end

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=create_chunk_id(
                        document.document_id,
                        index,
                        chunk_text,
                    ),
                    document_id=document.document_id,
                    text=chunk_text,
                    metadata=document.metadata,
                    chunk_index=index,
                )
            )

        if end >= len(text):
            break

        start = max(0, end - overlap)
        index += 1

    return chunks


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    pattern = r"(?m)^(#{1,6}\s+.+)$"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return [("", text)]

    sections = []

    for index, match in enumerate(matches):
        heading = match.group(1)
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        body = text[start:end].strip()
        sections.append((heading, body))

    return sections


def chunk_document_section_aware(
    document: SourceDocument,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    chunks = []
    global_index = 0

    for heading, body in split_markdown_sections(document.text):
        section_text = f"{heading}\n\n{body}".strip()

        temporary_document = SourceDocument(
            document_id=document.document_id,
            text=section_text,
            metadata=document.metadata.model_copy(
                update={
                    "extra": {
                        **document.metadata.extra,
                        "section_heading": heading,
                    }
                }
            ),
        )

        section_chunks = chunk_document_fixed(
            temporary_document,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in section_chunks:
            chunk.chunk_index = global_index
            chunk.chunk_id = create_chunk_id(
                document.document_id,
                global_index,
                chunk.text,
            )
            chunks.append(chunk)
            global_index += 1

    return chunks


def chunk_document_parent_child(
    document: SourceDocument,
    parent_size: int = settings.parent_chunk_size,
    child_size: int = settings.child_chunk_size,
    child_overlap: int = settings.child_chunk_overlap,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    sections = split_markdown_sections(document.text)

    parent_chunks: list[ParentChunk] = []
    child_chunks: list[ChildChunk] = []
    parent_index = 0
    global_child_index = 0

    for heading, body in sections:
        section_text = f"{heading}\n\n{body}".strip() if heading else body.strip()
        if not section_text:
            continue

        temp_doc = SourceDocument(
            document_id=document.document_id,
            text=section_text,
            metadata=document.metadata.model_copy(
                update={
                    "extra": {
                        **document.metadata.extra,
                        "section_heading": heading,
                    }
                }
            ),
        )

        parent_fixed_chunks = chunk_document_fixed(
            temp_doc,
            chunk_size=parent_size,
            overlap=min(200, max(0, parent_size - 1)),
        )

        for p_chunk in parent_fixed_chunks:
            p_id = f"parent_{create_chunk_id(document.document_id, parent_index, p_chunk.text)}"

            parent_obj = ParentChunk(
                chunk_id=p_id,
                document_id=document.document_id,
                text=p_chunk.text,
                metadata=p_chunk.metadata,
                chunk_index=parent_index,
                parent_id=p_id,
                parent_text=p_chunk.text,
                child_ids=[],
            )

            parent_as_doc = SourceDocument(
                document_id=document.document_id,
                text=p_chunk.text,
                metadata=p_chunk.metadata.model_copy(
                    update={
                        "extra": {
                            **p_chunk.metadata.extra,
                            "parent_id": p_id,
                            "parent_text": p_chunk.text,
                        }
                    }
                ),
            )

            raw_children = chunk_document_fixed(
                parent_as_doc,
                chunk_size=child_size,
                overlap=child_overlap,
            )

            for child in raw_children:
                c_id = f"child_{create_chunk_id(document.document_id, global_child_index, child.text)}"
                child_obj = ChildChunk(
                    chunk_id=c_id,
                    document_id=document.document_id,
                    text=child.text,
                    metadata=child.metadata,
                    chunk_index=global_child_index,
                    parent_id=p_id,
                    parent_text=p_chunk.text,
                )
                parent_obj.child_ids.append(c_id)
                child_chunks.append(child_obj)
                global_child_index += 1

            parent_chunks.append(parent_obj)
            parent_index += 1

    return parent_chunks, child_chunks


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int = 800,
    overlap: int = 120,
    strategy: str = "section",
    parent_size: int = settings.parent_chunk_size,
    child_size: int = settings.child_chunk_size,
    child_overlap: int = settings.child_chunk_overlap,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        if strategy == "fixed":
            document_chunks = chunk_document_fixed(
                document,
                chunk_size,
                overlap,
            )
        elif strategy == "section":
            document_chunks = chunk_document_section_aware(
                document,
                chunk_size,
                overlap,
            )
        elif strategy == "parent-child":
            _, child_chunks = chunk_document_parent_child(
                document,
                parent_size=parent_size,
                child_size=child_size,
                child_overlap=child_overlap,
            )
            document_chunks = child_chunks
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        chunks.extend(document_chunks)

    return chunks