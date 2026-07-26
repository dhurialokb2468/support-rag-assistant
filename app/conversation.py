from typing import Any

from pydantic import BaseModel, Field


class ConversationState(BaseModel):
    product: str | None = None
    product_version: str | None = None
    issue_category: str | None = None
    error_codes: list[str] = Field(default_factory=list)
    customer_environment: dict[str, Any] = Field(default_factory=dict)
    recent_questions: list[str] = Field(default_factory=list)
    recent_answers: list[str] = Field(default_factory=list)
    previous_source_ids: list[str] = Field(default_factory=list)
    max_history: int = 5

    def add_turn(self, question: str, answer: Any) -> None:
        """Helper to add a question-answer turn."""
        self.update_from_question(question)
        self.update_from_answer(answer)

    def update_from_question(self, question: str) -> None:
        """Updates state and metadata from the user's question. Explicit user inputs override prior state."""
        if not question or not question.strip():
            return

        from app.query_processor import QueryProcessor

        qp = QueryProcessor()
        meta = qp.process(question)

        # Explicit user inputs override prior state
        if meta.product:
            self.product = meta.product
        if meta.version:
            self.product_version = meta.version
        if meta.category:
            self.issue_category = meta.category

        for code in meta.error_codes:
            if code not in self.error_codes:
                self.error_codes.append(code)

        clean_q = question.strip()
        self.recent_questions.append(clean_q)
        if len(self.recent_questions) > self.max_history:
            self.recent_questions = self.recent_questions[-self.max_history:]

    def update_from_answer(
        self,
        answer: Any,
        source_ids: list[str] | None = None,
    ) -> None:
        """Appends recent answer and updates previous source IDs without treating model answers as factual metadata."""
        ans_str = ""
        if isinstance(answer, dict):
            ans_str = str(answer.get("answer", ""))
        elif hasattr(answer, "answer"):
            ans_str = str(answer.answer)
        else:
            ans_str = str(answer)

        if ans_str.strip():
            self.recent_answers.append(ans_str.strip())
            if len(self.recent_answers) > self.max_history:
                self.recent_answers = self.recent_answers[-self.max_history:]

        if source_ids:
            for sid in source_ids:
                if sid not in self.previous_source_ids:
                    self.previous_source_ids.append(sid)

    def build_rewrite_context(self) -> str:
        """Formats active conversation metadata state and recent turns into structured rewrite context."""
        parts = []

        active_meta = []
        if self.product:
            active_meta.append(f"Product: {self.product}")
        if self.product_version:
            active_meta.append(f"Version: {self.product_version}")
        if self.issue_category:
            active_meta.append(f"Category: {self.issue_category}")
        if self.error_codes:
            active_meta.append(f"Error Codes: {', '.join(self.error_codes)}")

        if active_meta:
            parts.append("Active Conversation State:\n" + "\n".join(active_meta))

        turns = []
        min_turns = min(len(self.recent_questions), len(self.recent_answers))
        for i in range(min_turns):
            turns.append(f"User: {self.recent_questions[i]}\nAssistant: {self.recent_answers[i]}")
        if len(self.recent_questions) > min_turns:
            turns.append(f"User: {self.recent_questions[-1]}")

        if turns:
            parts.append("Recent Conversation Turns:\n" + "\n---\n".join(turns))

        return "\n\n".join(parts)

    def get_context_text(self, max_messages: int = 6) -> str:
        return self.build_rewrite_context()

    def is_empty(self) -> bool:
        return len(self.recent_questions) == 0

    def clear(self) -> None:
        """Clears all conversation state, history, and metadata."""
        self.product = None
        self.product_version = None
        self.issue_category = None
        self.error_codes.clear()
        self.customer_environment.clear()
        self.recent_questions.clear()
        self.recent_answers.clear()
        self.previous_source_ids.clear()
