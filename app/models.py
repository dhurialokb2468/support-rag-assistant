from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.conversation import ConversationState


class AbstentionReason(str, Enum):
    LOW_RELEVANCE = "low_relevance"
    NO_VALID_CITATIONS = "no_valid_citations"
    OUTSIDE_KNOWLEDGE_BASE = "outside_knowledge_base"
    UNDOCUMENTED_FUTURE_PLANS = "undocumented_future_plans"
    UNRESOLVED_SOURCE_CONFLICT = "unresolved_source_conflict"
    POLICY_LEGAL_BILLING_DECISION = "policy_legal_billing_decision"
    CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"


class DocumentMetadata(BaseModel):
    title: str
    source: str
    document_type: str
    product: str = "InsightFlow"
    version: str | None = None
    category: str | None = None
    updated_at: str | None = None
    authority_score: float = 0.5
    reviewed: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class SourceDocument(BaseModel):
    document_id: str
    text: str
    metadata: DocumentMetadata


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: DocumentMetadata
    chunk_index: int
    parent_id: str | None = None
    parent_text: str | None = None


class ParentChunk(Chunk):
    child_ids: list[str] = Field(default_factory=list)


class ChildChunk(Chunk):
    parent_id: str
    parent_text: str | None = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    raw_reranker_score: float | None = None
    reranker_score: float | None = None
    reranker_rank: int | None = None
    authority_score: float | None = None
    freshness_score: float | None = None
    version_score: float | None = None
    normalized_reranker_score: float | None = None
    final_score: float | None = None
    retrieval_methods: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    source_id: str
    title: str
    source: str
    quoted_text: str


class SupportAnswer(BaseModel):
    answer: str
    likely_cause: str | None = None
    resolution_steps: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    confidence_score: float
    escalation_required: bool
    escalation_reason: str | None = None
    conflicts_detected: bool = False


class QueryMetadata(BaseModel):
    product: str | None = None
    version: str | None = None
    category: str | None = None
    error_codes: list[str] = Field(default_factory=list)
    extracted_terms: list[str] = Field(default_factory=list)


class QueryRewriteTrace(BaseModel):
    original_question: str
    rewritten_query: str
    rewrite_used: bool = False
    rewrite_latency: float = 0.0


class RetrievalTrace(BaseModel):
    query: str
    generated_queries: list[str] = Field(default_factory=list)
    filters_requested: dict[str, Any] = Field(default_factory=dict)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    strict_result_count: int = 0
    relaxed_result_count: int = 0
    strict_succeeded: bool = False
    fallback_used: bool = False
    rerank_latency: float | None = None
    rewrite_trace: QueryRewriteTrace | None = None


class ContextBuildResult(BaseModel):
    selected_chunks: list[dict[str, Any]] = Field(default_factory=list)
    excluded_duplicates: list[dict[str, Any]] = Field(default_factory=list)
    excluded_budget_overflow: list[dict[str, Any]] = Field(default_factory=list)
    excluded_source_limit: list[dict[str, Any]] = Field(default_factory=list)
    formatted_context: str = ""


class ConflictResult(BaseModel):
    conflict_detected: bool = False
    conflicting_source_ids: list[str] = Field(default_factory=list)
    summary: str | None = None
    preferred_source_id: str | None = None
    preference_reason: str | None = None
    unresolved: bool = False


class PipelineTrace(BaseModel):
    metadata_extraction_latency: float = 0.0
    query_rewriting_latency: float = 0.0
    query_expansion_latency: float = 0.0
    embedding_latency: float = 0.0
    semantic_retrieval_latency: float = 0.0
    bm25_retrieval_latency: float = 0.0
    fusion_latency: float = 0.0
    reranking_latency: float = 0.0
    authority_freshness_scoring_latency: float = 0.0
    parent_resolution_latency: float = 0.0
    context_building_latency: float = 0.0
    conflict_detection_latency: float = 0.0
    generation_latency: float = 0.0
    validation_latency: float = 0.0
    citation_verification_latency: float = 0.0
    confidence_calculation_latency: float = 0.0
    total_latency: float = 0.0

    query: str = ""
    rewritten_query: str = ""
    generated_queries: list[str] = Field(default_factory=list)
    filters_requested: dict[str, Any] = Field(default_factory=dict)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    strict_succeeded: bool = True
    fallback_used: bool = False
    citation_validation_passed: bool = True
    abstention_triggered: bool = False
    abstention_reason: str | None = None
    conflict_result: dict[str, Any] = Field(default_factory=dict)
    confidence_breakdown: dict[str, Any] = Field(default_factory=dict)
    conversation_state: dict[str, Any] = Field(default_factory=dict)
