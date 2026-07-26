import time
import streamlit as st

from app.conversation import ConversationState
from app.feedback import FEEDBACK_REASONS, FeedbackDB
from app.pipeline import BasicRAGPipeline

st.set_page_config(
    page_title="Ask Assistant - InsightFlow Support RAG",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Ask InsightFlow Support Assistant")
st.caption("Interactive AI Technical Support with Deterministic Verification & Conflict Resolution")

# Initialize Feedback DB
feedback_db = FeedbackDB()

# Initialize Session State
if "conversation_state" not in st.session_state:
    st.session_state["conversation_state"] = ConversationState()

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = BasicRAGPipeline(
        conversation_state=st.session_state["conversation_state"]
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "feedback_submitted" not in st.session_state:
    st.session_state["feedback_submitted"] = {}

pipeline: BasicRAGPipeline = st.session_state["pipeline"]
conv_state: ConversationState = st.session_state["conversation_state"]

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Active Conversation State")

    if conv_state.product or conv_state.product_version or conv_state.issue_category:
        st.write(f"**Product:** `{conv_state.product or 'InsightFlow'}`")
        st.write(f"**Version:** `{conv_state.product_version or 'N/A'}`")
        st.write(f"**Category:** `{conv_state.issue_category or 'N/A'}`")
        if conv_state.error_codes:
            st.write(f"**Error Codes:** `{', '.join(conv_state.error_codes)}`")
    else:
        st.info("No active metadata state extracted yet.")

    st.divider()
    if st.button("🧹 Clear Conversation History", type="secondary", use_container_width=True):
        conv_state.clear()
        st.session_state["messages"].clear()
        st.session_state["feedback_submitted"].clear()
        st.rerun()

# Display Chat History
for idx, msg in enumerate(st.session_state["messages"]):
    role = msg["role"]
    with st.chat_message(role):
        if role == "user":
            st.markdown(msg["content"])
        else:
            ans_data = msg.get("answer_data", {})
            trace_data = msg.get("trace_data", {})
            i_id = msg.get("interaction_id")

            # 1. Main Answer Text
            st.markdown(ans_data.get("answer", "No answer generated."))

            # 2. Likely Cause
            if ans_data.get("likely_cause"):
                st.markdown(f"**Likely Cause:** {ans_data['likely_cause']}")

            # 3. Resolution Steps
            steps = ans_data.get("resolution_steps", [])
            if steps:
                st.markdown("**Resolution Steps:**")
                for s_idx, step in enumerate(steps, start=1):
                    st.markdown(f"{s_idx}. {step}")

            # 4. Confidence Badge & Score
            conf_level = ans_data.get("confidence", "low").lower()
            conf_score = ans_data.get("confidence_score", 0.0)

            if conf_level == "high":
                st.success(f"🟢 **Confidence Level:** HIGH ({conf_score:.2f} / 1.0)")
            elif conf_level == "medium":
                st.warning(f"🟠 **Confidence Level:** MEDIUM ({conf_score:.2f} / 1.0)")
            else:
                st.error(f"🔴 **Confidence Level:** LOW ({conf_score:.2f} / 1.0) - Human Support Verification Required")

            # 5. Escalation Notice (Do not hide!)
            if ans_data.get("escalation_required"):
                st.error(
                    f"⚠️ **HUMAN ESCALATION REQUIRED**\n\n"
                    f"**Reason:** {ans_data.get('escalation_reason') or 'Uncertainty or policy decision threshold reached.'}"
                )

            # 6. Conflict Warning (Do not hide!)
            if ans_data.get("conflicts_detected"):
                conflict_info = trace_data.get("conflict_result", {})
                st.warning(
                    f"⚡ **MATERIAL CONFLICT DETECTED ACROSS SOURCES**\n\n"
                    f"**Summary:** {conflict_info.get('summary', 'Conflicting documentation detected.')}\n\n"
                    f"**Preference Reason:** {conflict_info.get('preference_reason', 'N/A')}"
                )
                if conflict_info.get("unresolved"):
                    st.error("🚨 **UNRESOLVED CONFLICT:** Equivocal documentation requires human supervisor review.")

            # 7. Fallback Retrieval Notice (Do not hide!)
            if trace_data.get("fallback_used"):
                st.info("ℹ️ **Fallback Retrieval Used:** Metadata filters were relaxed to expand candidates.")

            # 8. Expandable Citations
            citations = msg.get("citations", [])
            with st.expander(f"📚 Resolved Citations ({len(citations)})"):
                if citations:
                    for c in citations:
                        st.markdown(f"**[{c.get('source_id')}] {c.get('title')}** (`{c.get('source')}`)")
                        st.caption(f"Quoted Passage: \"{c.get('quoted_text')}\"")
                else:
                    st.write("No explicit citations resolved.")

            # 9. Expandable Evidence & Candidates
            sources = msg.get("sources", [])
            with st.expander(f"🔍 Evidence & Retrieved Chunks ({len(sources)})"):
                for s_idx, src in enumerate(sources, start=1):
                    meta = src.get("metadata", {})
                    st.markdown(f"**{s_idx}. {meta.get('title', 'Untitled')}** ({meta.get('source', 'Unknown')})")
                    st.caption(
                        f"Doc Type: {meta.get('document_type')} | Version: {meta.get('version')} | Category: {meta.get('category')} | Score: {src.get('final_score', 0.0):.4f}"
                    )
                    st.text(src.get("text", "")[:300] + "...")

            # 10. Latency Breakdown
            gen_time = msg.get("generation_time", 0.0)
            rewrite_lat = trace_data.get("rewrite_trace", {}).get("rewrite_latency", 0.0)
            rerank_lat = trace_data.get("rerank_latency", 0.0)
            st.caption(
                f"⏱️ Total Latency: {gen_time:.2f}s | Query Rewrite: {rewrite_lat:.4f}s | Reranker: {rerank_lat if rerank_lat else 0.0:.4f}s"
            )

            # 11. SQLite Feedback Controls (Helpful / Not Helpful)
            if i_id:
                already_fb = st.session_state["feedback_submitted"].get(i_id)
                if already_fb:
                    st.caption(f"✓ Feedback recorded ({already_fb})")
                else:
                    fb_col1, fb_col2, _ = st.columns([1.2, 1.5, 7.3])
                    with fb_col1:
                        if st.button("👍 Helpful", key=f"up_{idx}"):
                            feedback_db.save_feedback(interaction_id=i_id, helpful=True)
                            st.session_state["feedback_submitted"][i_id] = "Helpful"
                            st.toast("Thank you for your feedback! Saved to SQLite. 👍", icon="✅")
                            st.rerun()

                    with fb_col2:
                        with st.popover("👎 Not Helpful"):
                            st.markdown("### Report Feedback")
                            reason = st.selectbox(
                                "Select Issue Reason:",
                                options=FEEDBACK_REASONS,
                                key=f"reason_{idx}",
                            )
                            comment = st.text_input(
                                "Optional Details / Comment:",
                                key=f"comment_{idx}",
                            )
                            if st.button("Submit Feedback", key=f"sub_{idx}"):
                                feedback_db.save_feedback(
                                    interaction_id=i_id,
                                    helpful=False,
                                    issue_type=reason,
                                    comment=comment if comment.strip() else None,
                                )
                                st.session_state["feedback_submitted"][i_id] = f"Not Helpful ({reason})"
                                st.toast("Feedback recorded in SQLite database! 👎", icon="📝")
                                st.rerun()

# User Input Box
user_prompt = st.chat_input("Ask a technical question about InsightFlow...")

if user_prompt:
    # Append User Message
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documentation, retrieving sources & resolving conflicts..."):
            started = time.perf_counter()
            try:
                pipeline_result = pipeline.answer(user_prompt)
                elapsed = time.perf_counter() - started
            except Exception as exc:
                elapsed = time.perf_counter() - started
                from app.logger import get_logger
                get_logger("streamlit").error(f"Error processing question: {exc}", exc_info=True)
                st.error("⚠️ An unexpected error occurred while processing your request. Our support team has been notified. Please try again or rephrase your question.")
                st.stop()

        ans_data = pipeline_result.get("answer", {})
        citations_data = pipeline_result.get("citations", [])
        sources_data = pipeline_result.get("sources", [])
        trace_data = pipeline_result.get("trace", {})

        # Store interaction in SQLite feedback database
        rewritten_q = trace_data.get("rewrite_trace", {}).get("rewritten_query")
        conf_score = ans_data.get("confidence_score", 0.0)
        abstained = bool(trace_data.get("abstention_triggered"))
        escalated = bool(ans_data.get("escalation_required"))

        interaction_id = feedback_db.save_interaction(
            question=user_prompt,
            rewritten_query=rewritten_q,
            answer_data=ans_data,
            confidence_score=conf_score,
            total_latency=elapsed,
            abstained=abstained,
            escalated=escalated,
            retrieved_sources=sources_data,
            pipeline_trace=pipeline_result.get("pipeline_trace"),
        )

        # Render Assistant Turn
        st.markdown(ans_data.get("answer", "No answer generated."))

        if ans_data.get("likely_cause"):
            st.markdown(f"**Likely Cause:** {ans_data['likely_cause']}")

        steps = ans_data.get("resolution_steps", [])
        if steps:
            st.markdown("**Resolution Steps:**")
            for s_idx, step in enumerate(steps, start=1):
                st.markdown(f"{s_idx}. {step}")

        # Confidence Badge & Score
        conf_level = ans_data.get("confidence", "low").lower()
        conf_score = ans_data.get("confidence_score", 0.0)

        if conf_level == "high":
            st.success(f"🟢 **Confidence Level:** HIGH ({conf_score:.2f} / 1.0)")
        elif conf_level == "medium":
            st.warning(f"🟠 **Confidence Level:** MEDIUM ({conf_score:.2f} / 1.0)")
        else:
            st.error(f"🔴 **Confidence Level:** LOW ({conf_score:.2f} / 1.0) - Human Support Verification Required")

        # Escalation Notice
        if ans_data.get("escalation_required"):
            st.error(
                f"⚠️ **HUMAN ESCALATION REQUIRED**\n\n"
                f"**Reason:** {ans_data.get('escalation_reason') or 'Uncertainty or policy decision threshold reached.'}"
            )

        # Conflict Warning
        if ans_data.get("conflicts_detected"):
            conflict_info = trace_data.get("conflict_result", {})
            st.warning(
                f"⚡ **MATERIAL CONFLICT DETECTED ACROSS SOURCES**\n\n"
                f"**Summary:** {conflict_info.get('summary', 'Conflicting documentation detected.')}\n\n"
                f"**Preference Reason:** {conflict_info.get('preference_reason', 'N/A')}"
            )
            if conflict_info.get("unresolved"):
                st.error("🚨 **UNRESOLVED CONFLICT:** Equivocal documentation requires human supervisor review.")

        # Fallback Retrieval Notice
        if trace_data.get("fallback_used"):
            st.info("ℹ️ **Fallback Retrieval Used:** Metadata filters were relaxed to expand candidates.")

        # Expandable Citations
        with st.expander(f"📚 Resolved Citations ({len(citations_data)})"):
            if citations_data:
                for c in citations_data:
                    st.markdown(f"**[{c.get('source_id')}] {c.get('title')}** (`{c.get('source')}`)")
                    st.caption(f"Quoted Passage: \"{c.get('quoted_text')}\"")
            else:
                st.write("No explicit citations resolved.")

        # Expandable Evidence & Candidates
        with st.expander(f"🔍 Evidence & Retrieved Chunks ({len(sources_data)})"):
            for s_idx, src in enumerate(sources_data, start=1):
                meta = src.get("metadata", {})
                st.markdown(f"**{s_idx}. {meta.get('title', 'Untitled')}** ({meta.get('source', 'Unknown')})")
                st.caption(
                    f"Doc Type: {meta.get('document_type')} | Version: {meta.get('version')} | Category: {meta.get('category')} | Score: {src.get('final_score', 0.0):.4f}"
                )
                st.text(src.get("text", "")[:300] + "...")

        # Latency Breakdown
        rewrite_lat = trace_data.get("rewrite_trace", {}).get("rewrite_latency", 0.0)
        rerank_lat = trace_data.get("rerank_latency", 0.0)
        st.caption(
            f"⏱️ Total Latency: {elapsed:.2f}s | Query Rewrite: {rewrite_lat:.4f}s | Reranker: {rerank_lat if rerank_lat else 0.0:.4f}s"
        )

        # Append Assistant Response to Chat History
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": ans_data.get("answer", ""),
                "answer_data": ans_data,
                "citations": citations_data,
                "sources": sources_data,
                "trace_data": trace_data,
                "generation_time": elapsed,
                "interaction_id": interaction_id,
            }
        )

        st.rerun()
