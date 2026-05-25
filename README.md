# 📄 DocChat — RAG-Powered Document Q&A System

Ask natural-language questions about any PDF and get answers **grounded in the document** with source citations back to the exact page.

## Tech Stack

| Layer | Tool |
|---|---|
| **Frontend** | Streamlit |
| **PDF Parsing** | pypdf |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local) |
| **Vector Store** | ChromaDB (cosine similarity) |
| **LLM** | Llama 3.3 70B via OpenRouter API |

## Architecture

```
PDF Upload
    │
    ▼
pypdf → extract text per page
    │
    ▼
RecursiveCharacterTextSplitter
(chunk_size=1000, overlap=200)
    │
    ▼
sentence-transformers (local embeddings, CPU)
    │
    ▼
ChromaDB ──── vector store (persisted)
    │
    ▼ (on each question)
cosine similarity retrieval → top-4 chunks
    │
    ▼
Llama 3.3 70B (OpenRouter) → grounded answer + page citations
```

## Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/your-username/DocChat.git
cd DocChat
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

```bash
cp .env.example .env
# Open .env and paste your OpenRouter API key
```

Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys).

### 4. Run

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**.

## Usage

1. Enter your OpenRouter API key in the sidebar (or set `OPENROUTER_API_KEY` in `.env`)
2. Upload a PDF document
3. Wait ~5 seconds for parsing, chunking, and embedding
4. Ask any question about the document
5. Expand **Source passages** under each answer to see the exact chunks retrieved from ChromaDB

## Project Structure

```
DocChat/
├── app.py              # Streamlit UI — upload, chat, citations
├── rag_pipeline.py     # RAG logic — parse → chunk → embed → retrieve → generate
├── requirements.txt    # Python dependencies
├── .env.example        # API key template (copy to .env)
└── .gitignore
```

## Key Design Decisions

- **Local embeddings** — sentence-transformers runs on CPU, no API cost, no data sent to a third party for embedding
- **Chunk overlap** — 200-character overlap prevents context being cut off at chunk boundaries
- **Page-level citations** — each answer shows the page number and exact excerpt it was pulled from, so answers are verifiable
- **ChromaDB** — persists vectors between queries in the same session; can be extended to multi-document or persistent storage
