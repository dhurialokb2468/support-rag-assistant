import time
import pandas as pd
import streamlit as st

from app.context_builder import ContextBuilder, resolve_children_to_parents
from app.conversation import ConversationState
from app.query_processor import QueryProcessor, generate_search_queries, rewrite_query
from app.retriever import HybridRetriever
from app.scoring import score_candidates

st.set_page_config(
    page_title="Inspect Retrieval - InsightFlow RAG",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Retrieval & Context Inspection Studio")
st.caption("Deep-dive inspection of Query Processing, Multi-Query Fusion, Scoring & Context Building")

# Input Controls
query_input = st.text_input(
    "Enter Question / Query to Inspect:",
    value="How do I fix EXP-3204 CSV export in version 3.2?",
)

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    mode_input = st.selectbox(
        "Retrieval Mode:",
        options=["full", "reranked", "hybrid", "semantic", "keyword"],
        index=0,
    )
with col_ctrl2:
    top_k_input = st.slider("Top K Candidates:", min_value=1, max_value=20, value=5)
with col_ctrl3:
    use_context = st.checkbox("Include Active Session Conversation Context", value=False)

if st.button("🚀 Inspect Retrieval & Scoring", type="primary"):
    started = time.perf_counter()

    conv_state = None
    if use_context and "conversation_state" in st.session_state:
        conv_state = st.session_state["conversation_state"]

    # 1. Query Processing & Rewrite
    rewritten_query, rewrite_trace = rewrite_query(
        question=query_input,
        conversation_state=conv_state,
    )

    qp = QueryProcessor()
    q_meta = qp.process(rewritten_query)

    # 2. Multi-Query Generation
    gen_queries = generate_search_queries(rewritten_query)

    # 3. Hybrid Retrieval
    retriever = HybridRetriever()
    effective_mode = "reranked" if mode_input == "full" else mode_input
    raw_results, trace = retriever.retrieve(
        query=rewritten_query,
        top_k=top_k_input,
        mode=effective_mode,
    )

    # 4. Scoring & Parent Resolution
    scored_results = score_candidates(query=rewritten_query, candidates=raw_results)
    resolved_results = resolve_children_to_parents(scored_results)

    # 5. Context Builder & Exclusions
    cb = ContextBuilder()
    context_build_res = cb.build_context(resolved_results)

    total_latency = time.perf_counter() - started

    # --- SECTION 1: QUERY & METADATA TRACE ---
    st.divider()
    st.subheader("1. 📝 Query Processing & Metadata Trace")

    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.markdown(f"**Original Question:** `{rewrite_trace.original_question}`")
        st.markdown(f"**Rewritten Query:** `{rewrite_trace.rewritten_query}`")
        st.markdown(f"**Rewrite Used:** `{rewrite_trace.rewrite_used}` | **Rewrite Latency:** `{rewrite_trace.rewrite_latency:.4f}s`")

    with q_col2:
        st.markdown("**Extracted Query Metadata:**")
        st.json(q_meta.model_dump())

    st.markdown("**Generated Multi-Query Variations:**")
    for g_idx, gq in enumerate(gen_queries, start=1):
        st.markdown(f"{g_idx}. `{gq}`")

    # --- SECTION 2: FILTERING & LATENCY TRACE ---
    st.divider()
    st.subheader("2. 🎯 Filtering & Latency Trace")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.markdown(f"**Filters Requested:**\n`{trace.filters_requested}`")
    with f_col2:
        st.markdown(f"**Filters Applied:**\n`{trace.filters_applied}`")
    with f_col3:
        if trace.strict_succeeded:
            st.success("🟢 Strict Filtering Succeeded")
        else:
            st.warning("🟠 Strict Filtering Failed")
    with f_col4:
        if trace.fallback_used:
            st.info("ℹ️ Relaxed Fallback Used")
        else:
            st.success("🟢 No Fallback Needed")

    st.caption(
        f"⏱️ Total Latency: {total_latency:.2f}s | Rerank Latency: {trace.rerank_latency if trace.rerank_latency else 0.0:.4f}s"
    )

    with st.expander("⏱️ Detailed 17-Stage Pipeline Latency Breakdown"):
        pipe_res = pipeline.answer(user_question)
        pipe_trace = pipe_res.get("pipeline_trace", {})
        timing_data = []
        timing_keys = [
            "metadata_extraction_latency", "query_rewriting_latency", "query_expansion_latency",
            "embedding_latency", "semantic_retrieval_latency", "bm25_retrieval_latency",
            "fusion_latency", "reranking_latency", "authority_freshness_scoring_latency",
            "parent_resolution_latency", "context_building_latency", "conflict_detection_latency",
            "generation_latency", "validation_latency", "citation_verification_latency",
            "confidence_calculation_latency", "total_latency"
        ]
        for k in timing_keys:
            val = pipe_trace.get(k, 0.0) or 0.0
            stage_name = k.replace("_latency", "").replace("_", " ").title()
            timing_data.append({"Pipeline Stage": stage_name, "Latency (seconds)": f"{val:.6f}s"})
        st.dataframe(pd.DataFrame(timing_data), use_container_width=True)

    # --- SECTION 3: DATAFRAME CANDIDATES SCORE COMPARISON ---
    st.divider()
    st.subheader("3. 📊 Candidates Score Comparison DataFrame")

    df_rows = []
    for r_idx, c in enumerate(scored_results, start=1):
        meta = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else {}
        df_rows.append({
            "Rank": r_idx,
            "Chunk ID": c.get("chunk_id"),
            "Title": meta.get("title", "Untitled"),
            "Doc Type": meta.get("document_type"),
            "Version": meta.get("version"),
            "Category": meta.get("category"),
            "Semantic Rank": c.get("semantic_rank"),
            "Semantic Score": round(c["semantic_score"], 4) if c.get("semantic_score") is not None else None,
            "Keyword Rank": c.get("keyword_rank"),
            "Keyword Score": round(c["keyword_score"], 4) if c.get("keyword_score") is not None else None,
            "Fused RRF Score": round(c["fused_score"], 6) if c.get("fused_score") is not None else None,
            "Reranker Score": round(c["reranker_score"], 4) if c.get("reranker_score") is not None else None,
            "Authority Score": round(c["authority_score"], 4) if c.get("authority_score") is not None else None,
            "Freshness Score": round(c["freshness_score"], 4) if c.get("freshness_score") is not None else None,
            "Version Score": round(c["version_score"], 4) if c.get("version_score") is not None else None,
            "Final Score": round(c["final_score"], 4) if c.get("final_score") is not None else None,
            "Parent ID": c.get("parent_id"),
        })

    df = pd.DataFrame(df_rows)
    st.dataframe(df, use_container_width=True)

    # --- SECTION 4: CONTEXT BUILDING & EXCLUSIONS ---
    st.divider()
    st.subheader("4. 🏗️ Context Building & Exclusions Breakdown")

    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    with ex_col1:
        st.metric(label="Selected Context Chunks", value=len(context_build_res.selected_chunks))
    with ex_col2:
        st.metric(label="Duplicate Exclusions", value=len(context_build_res.excluded_duplicates))
    with ex_col3:
        st.metric(label="Budget Overflow Exclusions", value=len(context_build_res.excluded_budget_overflow))
    with ex_col4:
        st.metric(label="Source Limit Exclusions", value=len(context_build_res.excluded_source_limit))

    with st.expander("📄 Formatted Context Output"):
        st.text(context_build_res.formatted_context)

    # --- SECTION 5: PASSAGE VIEWER ---
    st.divider()
    st.subheader("5. 📖 Candidate Passage Viewer")

    selected_chunk_id = st.selectbox(
        "Select Candidate Chunk to View Passage:",
        options=[c.get("chunk_id") for c in scored_results],
    )

    selected_cand = next((c for c in scored_results if c.get("chunk_id") == selected_chunk_id), None)

    if selected_cand:
        p_col1, p_col2 = st.columns([2, 1])
        with p_col1:
            st.markdown(f"### Passage Text (`{selected_cand.get('chunk_id')}`)")
            st.info(selected_cand.get("text", "No text"))
            if selected_cand.get("parent_text"):
                st.markdown("### Parent Chunk Full Text")
                st.success(selected_cand.get("parent_text"))

        with p_col2:
            st.markdown("### Candidate Metadata & Scores")
            st.json(selected_cand)
