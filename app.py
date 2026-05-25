"""
DocChat — RAG-Powered Document Q&A System
==========================================
Stack: Streamlit · LangChain · ChromaDB · sentence-transformers · Groq (Llama 3.3)

Run:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

load_dotenv()

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — RAG Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📄 DocChat")
    st.caption("RAG-Powered Document Q&A")
    st.divider()

    api_key = st.text_input(
        "🔑 OpenRouter API Key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="Get your key at openrouter.ai/keys",
    )

    st.divider()

    st.markdown("### 📂 Upload a PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        **Architecture**
        | Step | Tool |
        |------|------|
        | PDF parsing | `pypdf` |
        | Chunking | `RecursiveCharacterTextSplitter` |
        | Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
        | Vector store | **ChromaDB** (cosine similarity) |
        | LLM | **Llama 3.3 70B** via OpenRouter |
        """
    )

# ── Require API key ────────────────────────────────────────────────────────────
if not api_key:
    st.warning("⚠️ Enter your **OpenRouter API key** in the sidebar to start.")
    st.info("Get your key at [openrouter.ai/keys](https://openrouter.ai/keys)")
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline: RAGPipeline | None = None
if "messages" not in st.session_state:
    st.session_state.messages: list = []
if "active_doc" not in st.session_state:
    st.session_state.active_doc: str | None = None

# ── Process uploaded PDF ────────────────────────────────────────────────────────
if uploaded_file and st.session_state.active_doc != uploaded_file.name:
    with st.spinner(f"⏳ Processing **{uploaded_file.name}** — parsing → chunking → embedding…"):
        # Save to a temp file (PyPDFLoader needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            chroma_dir = tempfile.mkdtemp(prefix="docchat_chroma_")
            pipeline = RAGPipeline(
                api_key=api_key,
                persist_directory=chroma_dir,
            )
            n_chunks = pipeline.process_pdf(tmp_path)

            # Persist in session
            st.session_state.pipeline = pipeline
            st.session_state.active_doc = uploaded_file.name
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"✅ **{uploaded_file.name}** indexed into **{n_chunks} chunks** in ChromaDB.\n\n"
                        "Embeddings generated locally with `sentence-transformers/all-MiniLM-L6-v2`.\n\n"
                        "Ask me anything about the document!"
                    ),
                }
            ]
            st.success(f"Ready — {n_chunks} chunks stored in ChromaDB.")

        except Exception as exc:
            st.error(f"Error processing PDF: {exc}")
            st.session_state.pipeline = None

        finally:
            os.unlink(tmp_path)

# ── Chat interface ──────────────────────────────────────────────────────────────
st.markdown("## 💬 Document Q&A")

if not st.session_state.pipeline:
    st.info("👈 Upload a PDF from the sidebar to initialize the RAG pipeline.")
    st.stop()

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📚 Source passages — {len(msg['citations'])} chunks retrieved"):
                for cite in msg["citations"]:
                    st.markdown(cite)
                    st.divider()

# New user question
if prompt := st.chat_input("Ask a question about the document…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages → generating answer with Groq Llama 3.3…"):
            try:
                answer, citations = st.session_state.pipeline.query(prompt)
                st.markdown(answer)

                if citations:
                    with st.expander(
                        f"📚 Source passages — {len(citations)} chunks retrieved from ChromaDB"
                    ):
                        for cite in citations:
                            st.markdown(cite)
                            st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "citations": citations}
                )

            except Exception as exc:
                err = f"❌ Error: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
