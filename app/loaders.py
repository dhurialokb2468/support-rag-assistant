import csv
import hashlib
from pathlib import Path

import yaml

from app.models import DocumentMetadata, SourceDocument


def create_document_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def load_markdown_file(path: Path) -> SourceDocument:
    content = path.read_text(encoding="utf-8")

    if not content.startswith("---"):
        raise ValueError(f"Missing metadata header: {path}")

    parts = content.split("---", 2)

    if len(parts) < 3:
        raise ValueError(f"Invalid metadata header: {path}")

    metadata_raw = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()

    metadata = DocumentMetadata(
        title=metadata_raw["title"],
        source=str(path),
        document_type=metadata_raw["document_type"],
        product=metadata_raw.get("product", "InsightFlow"),
        version=str(metadata_raw.get("version", "")) or None,
        category=metadata_raw.get("category"),
        updated_at=metadata_raw.get("updated_at"),
        authority_score=float(
            metadata_raw.get("authority_score", 0.5)
        ),
        reviewed=bool(metadata_raw.get("reviewed", False)),
    )

    return SourceDocument(
        document_id=create_document_id(str(path)),
        text=body,
        metadata=metadata,
    )


def load_markdown_documents(data_root: str) -> list[SourceDocument]:
    documents = []

    folders = [
        "product_docs",
        "faqs",
        "release_notes",
        "known_issues",
    ]

    for folder in folders:
        folder_path = Path(data_root) / folder

        if not folder_path.exists():
            continue

        for path in folder_path.rglob("*.md"):
            documents.append(load_markdown_file(path))

    return documents


def load_support_tickets(path: Path) -> list[SourceDocument]:
    documents = []

    if not path.exists():
        return documents

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            text = (
                f"Customer question:\n{row['question']}\n\n"
                f"Resolution:\n{row['resolution']}"
            )

            metadata = DocumentMetadata(
                title=f"Support Ticket {row['ticket_id']}",
                source=str(path),
                document_type="support_ticket",
                product="InsightFlow",
                version=row.get("product_version"),
                category=row.get("category"),
                updated_at=row.get("created_at"),
                authority_score=float(
                    row.get("authority_score", 0.5)
                ),
                reviewed=row.get("reviewed", "").lower() == "true",
                extra={
                    "ticket_id": row["ticket_id"],
                    "priority": row.get("priority"),
                    "status": row.get("status"),
                },
            )

            documents.append(
                SourceDocument(
                    document_id=create_document_id(
                        f"{path}:{row['ticket_id']}"
                    ),
                    text=text,
                    metadata=metadata,
                )
            )

    return documents


def load_all_documents(data_root: str = "data") -> list[SourceDocument]:
    documents = load_markdown_documents(data_root)

    ticket_path = (
        Path(data_root)
        / "tickets"
        / "support_tickets.csv"
    )

    documents.extend(load_support_tickets(ticket_path))

    return documents