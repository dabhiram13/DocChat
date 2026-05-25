"""
DocChat — RAG Pipeline
======================
Stack (as described on resume):
  • PDF parsing    : pypdf (via LangChain Document format)
  • Chunking       : RecursiveCharacterTextSplitter
  • Embeddings     : OpenAI text-embedding-3-small via OpenRouter
  • Vector store   : ChromaDB (cosine similarity, in-memory)
  • LLM            : OpenRouter API — Llama 3.3 70B (OpenAI-compatible)
"""

from __future__ import annotations

from typing import List, Tuple

import pypdf
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class RAGPipeline:
    """End-to-end RAG pipeline for PDF question-answering."""

    # OpenAI-compatible embedding model via OpenRouter
    EMBED_MODEL = "openai/text-embedding-3-small"

    # Llama 3.3 70B via OpenRouter
    LLM_MODEL = "meta-llama/llama-3.3-70b-instruct"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str) -> None:
        # ── OpenAI embeddings via OpenRouter ───────────────────────────────────
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
            model=self.EMBED_MODEL,
        )

        # ── LLM via OpenRouter (OpenAI-compatible API) ─────────────────────────
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
            model=self.LLM_MODEL,
            temperature=0.1,
            max_tokens=1024,
        )

        self.vectorstore: Chroma | None = None
        self.retriever = None

    # ── Step 1: Ingest PDF ─────────────────────────────────────────────────────
    def process_pdf(self, pdf_path: str) -> int:
        """
        Parse a PDF, chunk the text, embed each chunk, and store vectors
        in ChromaDB (in-memory, ephemeral — recreated per session).

        Returns:
            Number of chunks indexed.
        """
        # Parse PDF — preserve page number metadata for citations
        documents: list[Document] = []
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"page": page_num, "source": pdf_path},
                        )
                    )

        # Chunk with overlap so context isn't cut off at boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        # Embed each chunk and index in ChromaDB (ephemeral, no disk persist)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        # Similarity retriever — return top-4 chunks per query
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        return len(chunks)

    # ── Step 2: Query ──────────────────────────────────────────────────────────
    def query(self, question: str) -> Tuple[str, List[str]]:
        """
        RAG retrieval-then-generation:
          1. Embed the question using text-embedding-3-small via OpenRouter.
          2. Retrieve top-4 most similar chunks from ChromaDB (cosine similarity).
          3. Pass chunks as grounding context to Llama 3.3 via OpenRouter.

        Returns:
            (answer_text, citations_list)
        """
        if self.retriever is None:
            raise ValueError("No document loaded. Please upload a PDF first.")

        # ── Retrieval ──────────────────────────────────────────────────────────
        source_docs = self.retriever.invoke(question)
        context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

        # ── Generation ─────────────────────────────────────────────────────────
        system_prompt = (
            "You are a helpful assistant that answers questions strictly based on "
            "the provided document excerpts. "
            "If the answer is not found in the excerpts, say: "
            "\"I don't find that information in the document.\" "
            "Do not make up information. Be concise and accurate. "
            "Reference the source passage when helpful."
        )
        user_prompt = (
            f"Document excerpts (retrieved via semantic search from ChromaDB):\n\n{context}"
            f"\n\nQuestion: {question}"
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        # ── Citations — include page numbers ───────────────────────────────────
        citations: List[str] = []
        for doc in source_docs:
            page_num = int(doc.metadata.get("page", 0)) + 1  # 0-indexed → 1-indexed
            excerpt = doc.page_content[:300]
            if len(doc.page_content) > 300:
                excerpt += "…"
            citations.append(f"**Page {page_num}:** {excerpt}")

        return response.content, citations
