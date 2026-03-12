# Handwriting Context-Aware Note

An AI-powered note search system that **detects handwritten annotations** (underlines, circles, ink drawings, etc.) in PDF documents and answers questions like "I can't remember what I underlined when studying AI".

## Architecture

```
Browser (Vue 3)
   │  /api/* proxy
   ▼
Nginx (Docker) / Vite Dev Server
   │
   ▼
FastAPI Backend
   ├─ PostgreSQL  ─ document & chunk metadata
   ├─ Qdrant      ─ vector embedding search
   └─ OpenAI      ─ embeddings + LLM (gpt-4o-mini)
```


## Project Structure

```
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── main.py              app entry point (FastAPI + lifespan)
│   ├── config.py            environment config (Pydantic Settings)
│   ├── agents/              LangGraph RAG pipeline
│   │   ├── graph.py         node connection graph
│   │   ├── query_rewrite.py DSPy query rewriting
│   │   ├── retriever.py     Qdrant hybrid search
│   │   ├── reranker.py      result reranking
│   │   └── synthesis.py     GPT response generation
│   ├── api/                 HTTP routers
│   ├── models/              SQLAlchemy ORM + Pydantic schemas
│   └── services/            DB, Qdrant, OCR, embedding, chunking
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    └── src/
        ├── components/      ChatPanel, PdfViewer, DocumentSidebar
        └── services/        Axios API client
```


### Processing Pipeline

1. **PDF Upload** → extract text blocks with PyMuPDF + detect annotations/drawings
2. **Spatial Chunking** → tag text near handwritten markings as annotated chunks
3. **Embedding** → store in Qdrant via OpenAI text-embedding-3-small
4. **Query** → rewrite query with DSPy → vector search (MMR) → generate answer with GPT-4o-mini

---

## Quick Start (Docker Compose) ✅ Recommended

### Prerequisites
- Docker Desktop or Docker Engine + Compose v2
- Your OpenAI API Key

### 1. Configure Environment

```bash
cp backend/.env.example backend/.env
# Open backend/.env and set your OPENAI_API_KEY
```

### 2. Build & Run

```bash
docker compose up --build
```

> The first build may take several minutes due to Python package downloads (easyocr, torch, etc.).

Services started:

| Service | Port | Description |
|---------|------|-------------|
| postgres | 5432 | PostgreSQL 16 (database auto-created) |
| qdrant | 6333 / 6334 | Qdrant vector store |
| backend | 8000 | FastAPI server |
| frontend | 5173 | Vue 3 app (Nginx) |

### 3. Open the App

Navigate to [http://localhost:5173](http://localhost:5173) in your browser.

### Stop & Clean Up

```bash
docker compose down        # stop containers
docker compose down -v     # stop containers + delete volumes (DB data)
```

---

## Manual Setup (without Docker)

### Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### Backend

```bash
cd backend
cp .env.example .env          # fill in OPENAI_API_KEY
uv sync                       # install dependencies
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173 (Vite proxies /api → :8000)
```
