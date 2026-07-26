from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid

from app.config import settings
from app.logger import get_logger

logger = get_logger("feedback")

FEEDBACK_REASONS = [
    "incorrect answer",
    "missing information",
    "wrong source",
    "outdated information",
    "too vague",
    "should have escalated",
    "irrelevant retrieval",
]


class FeedbackDB:
    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            storage_dir_str = getattr(settings, "storage_dir", "storage")
            storage_dir = Path(storage_dir_str)
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(storage_dir / "feedback.db")
        else:
            self.db_path = str(db_path)

        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initializes database tables automatically with parameterized SQL schema."""
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    question TEXT NOT NULL,
                    rewritten_query TEXT,
                    answer_json TEXT NOT NULL,
                    confidence_score REAL,
                    total_latency REAL,
                    abstained INTEGER NOT NULL DEFAULT 0,
                    escalated INTEGER NOT NULL DEFAULT 0,
                    pipeline_trace_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieved_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    final_score REAL,
                    FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    interaction_id TEXT NOT NULL,
                    helpful INTEGER NOT NULL,
                    issue_type TEXT,
                    comment TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id) ON DELETE CASCADE
                );
                """
            )
            conn.commit()

    def save_interaction(
        self,
        question: str,
        rewritten_query: str | None,
        answer_data: dict[str, Any],
        confidence_score: float,
        total_latency: float,
        abstained: bool,
        escalated: bool,
        retrieved_sources: list[dict[str, Any]] | None = None,
        pipeline_trace: dict[str, Any] | None = None,
        interaction_id: str | None = None,
    ) -> str:
        """Stores a completed assistant interaction, its trace, and its retrieved sources."""
        i_id = interaction_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        answer_json_str = json.dumps(answer_data, default=str)
        trace_json_str = json.dumps(pipeline_trace, default=str) if pipeline_trace else None

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO interactions (
                    interaction_id, timestamp, question, rewritten_query,
                    answer_json, confidence_score, total_latency, abstained, escalated, pipeline_trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    i_id,
                    now_iso,
                    question,
                    rewritten_query,
                    answer_json_str,
                    float(confidence_score),
                    float(total_latency),
                    1 if abstained else 0,
                    1 if escalated else 0,
                    trace_json_str,
                ),
            )

            if retrieved_sources:
                for idx, src in enumerate(retrieved_sources, start=1):
                    meta = src.get("metadata", {}) if isinstance(src.get("metadata"), dict) else {}
                    sid = (
                        src.get("source_id")
                        or meta.get("source")
                        or src.get("document_id")
                        or src.get("chunk_id")
                        or f"S{idx}"
                    )
                    clean_sid = Path(str(sid)).stem if sid else str(sid)
                    score = float(src.get("final_score", src.get("reranker_score", 0.0) or 0.0))

                    conn.execute(
                        """
                        INSERT INTO retrieved_sources (
                            interaction_id, source_id, rank, final_score
                        ) VALUES (?, ?, ?, ?);
                        """,
                        (i_id, clean_sid, idx, score),
                    )

            conn.commit()

        return i_id

    def save_feedback(
        self,
        interaction_id: str,
        helpful: bool,
        issue_type: str | None = None,
        comment: str | None = None,
        feedback_id: str | None = None,
    ) -> str:
        """Stores user helpful or not-helpful feedback connected to an interaction_id."""
        f_id = feedback_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        if issue_type and issue_type not in FEEDBACK_REASONS:
            pass  # Allowed custom issue type

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    feedback_id, interaction_id, helpful, issue_type, comment, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    f_id,
                    interaction_id,
                    1 if helpful else 0,
                    issue_type,
                    comment,
                    now_iso,
                ),
            )
            conn.commit()

        return f_id

    def get_interaction(self, interaction_id: str) -> dict[str, Any] | None:
        """Fetches interaction details by interaction_id."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM interactions WHERE interaction_id = ?;", (interaction_id,)
            ).fetchone()

            if not row:
                return None

            sources_rows = conn.execute(
                "SELECT source_id, rank, final_score FROM retrieved_sources WHERE interaction_id = ? ORDER BY rank ASC;",
                (interaction_id,),
            ).fetchall()

            feedback_rows = conn.execute(
                "SELECT feedback_id, helpful, issue_type, comment, timestamp FROM feedback WHERE interaction_id = ? ORDER BY timestamp DESC;",
                (interaction_id,),
            ).fetchall()

            res = dict(row)
            res["answer_json"] = json.loads(res["answer_json"])
            res["retrieved_sources"] = [dict(r) for r in sources_rows]
            res["feedback"] = [dict(r) for r in feedback_rows]
            return res

    def get_all_feedback(self) -> list[dict[str, Any]]:
        """Returns all feedback joined with interaction details."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT f.feedback_id, f.interaction_id, f.helpful, f.issue_type, f.comment, f.timestamp AS feedback_timestamp,
                       i.question, i.confidence_score, i.abstained, i.escalated
                FROM feedback f
                JOIN interactions i ON f.interaction_id = i.interaction_id
                ORDER BY f.timestamp DESC;
                """
            ).fetchall()
            return [dict(r) for r in rows]
