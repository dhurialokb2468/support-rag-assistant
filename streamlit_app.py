import streamlit as st

from app.bm25_store import BM25Store
from app.config import settings
from app.generator import OllamaGenerator
from app.vector_store import VectorStore

st.set_page_config(
    page_title="InsightFlow Support RAG Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 InsightFlow Support RAG Assistant")
st.caption("Enterprise-grade Agentic Retrieval-Augmented Generation Platform for Support Operations")

st.markdown(
    """
### Project Overview
The **InsightFlow Support RAG Assistant** provides deterministic, highly reliable support answers for technical products.
It combines **Multi-Query Expansion**, **Hybrid Retrieval (Vector + BM25 RRF)**, **Cross-Encoder Reranking**, **Parent-Child Chunking**,
**Deterministic Confidence Scoring**, **Conflict Resolution**, and **Explicit Abstention / Human Escalation**.
"""
)

st.divider()
st.subheader("📊 System Health & Configuration Dashboard")

# Check status of components
generator = OllamaGenerator()
ollama_active = generator.is_available()

vector_store = VectorStore()
vector_count = vector_store.count()

bm25_store = BM25Store()
if not bm25_store.bm25:
    bm25_store.load()
bm25_count = bm25_store.count()

col1, col2, col3, col4 = st.columns(4)

with col1:
    if ollama_active:
        st.success("🟢 **Ollama Service**\nConnected & Online")
    else:
        st.warning("🟠 **Ollama Service**\nOffline (Using Fallback)")

with col2:
    st.metric(label="Vector Chunks Indexed", value=vector_count)

with col3:
    st.metric(label="BM25 Chunks Indexed", value=bm25_count)

with col4:
    total_docs = len(vector_store.get_all_documents()) if hasattr(vector_store, "get_all_documents") else vector_count
    st.metric(label="System Status", value="Healthy ⚡")

st.divider()
st.subheader("⚙️ Configured Models & Parameters")

m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.info(f"**LLM Model**\n\n`{settings.ollama_model}`\nHost: `{settings.ollama_base_url}`")

with m_col2:
    st.info(f"**Embedding Model**\n\n`{settings.embedding_model}`\nDimensions: 768")

with m_col3:
    st.info(f"**Cross-Encoder Reranker**\n\n`{settings.reranker_model}`\nMode: Reranked Hybrid")

st.divider()
st.markdown(
    """
👈 **Use the Sidebar Navigation** to select **Ask Assistant** and ask technical support questions!
"""
)
